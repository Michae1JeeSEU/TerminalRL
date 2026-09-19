from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
)
runner = importlib.import_module("terminal_marl.training.audit_v4_2_p2_gate_learnability_v2")


def _row(
    index: int,
    gate: int,
    *,
    seed: int = 783001,
    split: str = "train",
    interface: str = "o2m",
) -> dict:
    phase = {"o2m": "ring_outer", "m2i": "ring_middle", "i2f": "ring_inner"}[interface]
    observation = np.zeros(88, dtype=np.float32)
    observation[index % 8] = 1.0
    return {
        "decision_event_id": f"audit-v2-event-{split}-{index}",
        "seed": seed,
        "observation_v2_1": observation.tolist(),
        "teacher_gate": gate,
        "actor_mask": True,
        "support_split": split,
        "scenario": "flow6" if index % 2 else "flow8",
        "interface": interface,
        "phase": phase,
    }


def _pass_metrics() -> dict:
    metric = {
        "macro_F1": 0.80,
        "WAIT_recall": 0.70,
        "RELEASE_recall": 0.80,
        "brier": 0.15,
        "binary_nll": 0.50,
    }
    metric["interface"] = {
        name: {"macro_F1": 0.70} for name in runner.INTERFACES
    }
    return metric


def test_protocol_is_byte_bound_and_uses_v5_and_fresh_seeds() -> None:
    value = runner.protocol()
    assert value["protocol"] == "V4_2_P2_GATE_LEARNABILITY_AUDIT_V2"
    assert tuple(value["training"]["seeds"]) == runner.PROBE_SEEDS
    assert value["bindings"]["support_manifest_sha256"] == runner.SUPPORT_MANIFEST_SHA
    assert value["bindings"]["support_dev_sha256"] == runner.SUPPORT_DEV_SHA
    assert value["support_requirements"]["dev_rows"] == 2415


def test_v5_status_validation_accepts_exact_shape_and_rejects_authorization() -> None:
    manifest = {
        "stage": "V4_2_P2_GATE_SUPPORT_COLLECTION_V5_DEV_SUPPLEMENT",
        "SUPPORT_COLLECTION": "PASS",
        "SUPPORT_FROZEN": True,
        "LEARNABILITY_AUDIT_ALLOWED": True,
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
        "future_qualification_markers_mutated": False,
        "formal_episode_plan_complete": True,
        "formal_marker_created": True,
        "training_called": False,
        "optimizer_called": False,
        "student_authoritative": True,
        "teacher_label_only": True,
        "teacher_replaced_behavior": False,
        "metrics": {"train": {"rows": 3786}, "dev": {"rows": 2415}},
        "gates": {
            "train": {key: True for key in {
                "cardinality", "teacher_class_minimums", "teacher_class_proportions",
                "interface_minimums", "interface_class_minimums", "schema_88d_finite",
                "decision_event_id_unique", "seed_coverage",
                "all_planned_episodes_execution_complete",
                "all_episode_shield_actor_fraction_within_inherited_limit",
                "all_episode_gates", "cross_split_seed_disjoint",
                "cross_split_event_disjoint", "all",
            }},
            "dev": {key: True for key in {
                "cardinality", "teacher_class_minimums", "teacher_class_proportions",
                "interface_minimums", "interface_class_minimums", "schema_88d_finite",
                "decision_event_id_unique", "seed_coverage",
                "all_planned_episodes_execution_complete",
                "all_episode_shield_actor_fraction_within_inherited_limit",
                "all_episode_gates", "cross_split_seed_disjoint",
                "cross_split_event_disjoint", "all",
            }},
        },
    }
    runner._validate_support_status(manifest)
    manifest["LEARNABILITY_AUDIT_ALLOWED"] = False
    with pytest.raises(runner.AuditInputError, match="LEARNABILITY_AUDIT_ALLOWED"):
        runner._validate_support_status(manifest)


def test_future_qualification_seed_is_rejected() -> None:
    row = _row(0, 0, seed=runner.FUTURE_QUALIFICATION_SEEDS[0])
    with pytest.raises(runner.AuditInputError, match="future qualification"):
        runner._validate_rows([row], split="train")


def test_calibrated_candidate_adapter_updates_only_adapter_and_preserves_proposal_and_critic() -> None:
    rows = [
        _row(index, index % 2, interface=runner.INTERFACES[index % 3])
        for index in range(12)
    ]
    model = TemperatureCalibratedGateResidualAdapterActorCentralCriticV42(88)
    before_model = copy.deepcopy(model)
    before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    training = runner.train_gate_probe(
        model, rows, seed=runner.PROBE_SEEDS[0], trainable_prefixes=("gate_adapter",),
        epochs=2, batch_size=4,
    )
    changed = runner._state_changed(before_state, model.state_dict())
    changed_parameters = runner._parameter_changed(before_model, model)
    assert training["trainable_parameter_names"]
    assert changed
    assert set(changed_parameters) <= {name for name in changed if name.startswith("gate_adapter")}
    invariance = runner.proposal_logits_invariance(before_model, model, {"train": rows})
    assert invariance["train"]["bitwise_equal"]
    assert invariance["invariant"]
    assert runner._component_state_invariance(
        before_state, model.state_dict(), "critic_"
    )["bitwise_equal"]
    assert runner._state_invariance(
        before_state, model.state_dict(), excluded_prefixes=("gate_adapter",)
    )["bitwise_equal"]


def test_thresholds_are_inclusive_and_require_all_three_fresh_seeds() -> None:
    metric = _pass_metrics()
    assert runner.threshold_checks(metric)["all"]
    metric["interface"]["i2f"]["macro_F1"] = runner.THRESHOLDS["interface_macro_F1_min"] - 1e-9
    assert not runner.threshold_checks(metric)["all"]
    results = {str(seed): {"evaluation": {"dev": _pass_metrics()}}
               for seed in runner.PROBE_SEEDS}
    assert runner.fresh_probe_thresholds(results)["pass"]
    results.pop(str(runner.PROBE_SEEDS[-1]))
    assert not runner.fresh_probe_thresholds(results)["pass"]


def test_run_writes_diagnostic_manifest_without_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    support_train = tmp_path / "train.jsonl"
    support_dev = tmp_path / "dev.jsonl"
    support_train.write_text("{}\n", encoding="utf-8")
    support_dev.write_text("{}\n", encoding="utf-8")
    support_rows = {
        "train": [_row(0, 0), _row(1, 1)],
        "dev": [_row(2, 0, split="dev", seed=785001), _row(3, 1, split="dev", seed=785002)],
    }
    support_manifest = tmp_path / "support.json"
    support_manifest.write_text("{}", encoding="utf-8")
    candidate_path = tmp_path / "candidate.pt"
    candidate_path.write_bytes(b"candidate")
    d1_dataset = tmp_path / "d1.jsonl"
    d1_dataset.write_text("{}\n", encoding="utf-8")
    d1_manifest = tmp_path / "d1.json"
    d1_manifest.write_text("{}", encoding="utf-8")

    candidate_model = TemperatureCalibratedGateResidualAdapterActorCentralCriticV42(88)
    candidate = type("Policy", (), {"model": candidate_model})()

    monkeypatch.setattr(runner, "protocol", lambda: {"protocol": "mock"})
    monkeypatch.setattr(
        runner, "validate_support_manifest",
        lambda path: ({"stage": "V5"}, support_rows,
                       {"train": support_train, "dev": support_dev}),
    )
    monkeypatch.setattr(
        runner, "validate_d1_anchor",
        lambda dataset, manifest: ({"aggregate_rows": 2}, support_rows["train"]),
    )
    monkeypatch.setattr(
        runner, "_load_candidate",
        lambda path: (candidate, {"stage": "calibration", "architecture_id": runner.CANDIDATE_ARCHITECTURE_ID,
                                  "temperature": runner.CANDIDATE_TEMPERATURE,
                                  "qualification_only": True, "checkpoint_promotable": False}),
    )
    monkeypatch.setattr(runner, "_sha256", lambda path: "mock-sha")
    monkeypatch.setattr(runner, "train_gate_probe", lambda model, rows, **kwargs: {
        "seed": kwargs["seed"], "epochs": 40, "final_epoch": 40,
        "trainable_parameter_names": [], "frozen_parameter_names": [],
        "loss_final_epoch": 0.1, "loss_first_epoch": 0.2,
    })
    monkeypatch.setattr(runner, "_save_probe_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "evaluate_model", lambda model, datasets: {
        key: _pass_metrics() for key in datasets
    })
    monkeypatch.setattr(runner, "_probe_result", lambda model, train_rows, datasets, training, **kwargs: {
        "evaluation": {key: _pass_metrics() for key in datasets},
        "informative": True,
    })
    result = runner.run(
        tmp_path / "audit",
        support_manifest=support_manifest,
        candidate_checkpoint=candidate_path,
        d1_dataset=d1_dataset,
        d1_manifest=d1_manifest,
    )
    assert result["audit_pass"] is True
    assert result["GATE_REPAIR_ALLOWED"] is False
    assert result["PPO_ALLOWED"] is False
    assert result["checkpoint_promotable"] is False
    manifest = json.loads((tmp_path / "audit" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["environment_called"] is False
    assert manifest["teacher_called"] is False
