from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from terminal_marl.algo.joint_ppo_v4_2 import HierarchicalActorCentralCriticV42
from terminal_marl.training import audit_v4_2_p2_gate_learnability_v1 as runner


def _row(index: int, gate: int, *, seed: int = 778000, interface: str = "o2m") -> dict:
    phase = {"o2m": "ring_outer", "m2i": "ring_middle", "i2f": "ring_inner"}[interface]
    observation = np.zeros(88, dtype=np.float32)
    observation[index % 8] = 1.0
    return {
        "decision_event_id": f"audit-event-{index}",
        "seed": seed,
        "observation_v2_1": observation.tolist(),
        "teacher_gate": gate,
        "actor_mask": True,
        "support_split": "train",
        "scenario": "flow6" if index % 2 else "flow8",
        "interface": interface,
        "phase": phase,
    }


def test_gate_metrics_reports_confusion_calibration_and_breakdowns() -> None:
    rows = [_row(0, 1, interface="o2m"), _row(1, 0, interface="m2i"),
            _row(2, 1, interface="i2f"), _row(3, 0, interface="o2m")]
    result = runner.gate_metrics(rows, np.asarray([0.9, 0.1, 0.2, 0.8]))
    assert result["confusion"]["Teacher_WAIT_predicted_RELEASE"] == 1
    assert result["confusion"]["Teacher_RELEASE_predicted_WAIT"] == 1
    assert result["false_WAIT"] == result["false_RELEASE"] == 1
    assert result["WAIT_recall"] == result["RELEASE_recall"] == 0.5
    assert result["WAIT_precision"] == result["RELEASE_precision"] == 0.5
    assert np.isfinite(result["brier"])
    assert np.isfinite(result["binary_nll"])
    assert set(result["interface"]) == {"o2m", "m2i", "i2f"}
    assert result["probabilities_by_teacher_label"]["WAIT"]["n"] == 2


def test_threshold_boundary_is_inclusive() -> None:
    metrics = {
        "macro_F1": runner.THRESHOLDS["dev_macro_F1_min"],
        "WAIT_recall": runner.THRESHOLDS["dev_WAIT_recall_min"],
        "RELEASE_recall": runner.THRESHOLDS["dev_RELEASE_recall_min"],
        "brier": runner.THRESHOLDS["dev_brier_max"],
        "binary_nll": runner.THRESHOLDS["dev_binary_nll_max"],
        "interface": {name: {"macro_F1": runner.THRESHOLDS["interface_macro_F1_min"]}
                      for name in runner.INTERFACES},
    }
    assert runner.threshold_checks(metrics)["all"]
    metrics["interface"]["i2f"]["macro_F1"] = runner.THRESHOLDS["interface_macro_F1_min"] - 1e-9
    assert not runner.threshold_checks(metrics)["all"]


def test_support_row_validation_rejects_future_seed_and_duplicate_events() -> None:
    future = _row(0, 0, seed=runner.FUTURE_QUALIFICATION_SEEDS[0])
    with pytest.raises(runner.AuditInputError, match="future qualification"):
        runner._validate_rows([future], split="train")

    first = _row(0, 0)
    duplicate = {**_row(1, 1), "decision_event_id": first["decision_event_id"]}
    with pytest.raises(runner.AuditInputError, match="not unique"):
        runner._validate_rows([first, duplicate], split="train")


def test_frozen_encoder_probe_changes_only_gate_head_and_preserves_proposal_logits() -> None:
    rows = [_row(index, index % 2, interface=runner.INTERFACES[index % 3]) for index in range(12)]
    model = HierarchicalActorCentralCriticV42(88)
    before = copy.deepcopy(model)
    before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    training = runner.train_gate_probe(model, rows, seed=778101, trainable_prefixes=("gate_head",), epochs=2,
                                       batch_size=4)
    changed = runner._state_changed(before_state, model.state_dict())
    assert training["trainable_parameter_names"] == ["gate_head.weight", "gate_head.bias"]
    assert changed and set(changed) <= {"gate_head.weight", "gate_head.bias"}
    invariance = runner.proposal_logits_invariance(before, model, {"train": rows})
    assert invariance["train"]["bitwise_equal"]
    assert invariance["train"]["max_abs_delta"] == 0.0
    assert invariance["invariant"]
