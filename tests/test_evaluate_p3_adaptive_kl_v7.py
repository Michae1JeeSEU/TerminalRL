from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator


def _fake_record(*, label: str, scenario: str, seed: int, failed: bool = False) -> dict:
    rows = [] if failed else [{
        "uid": 1,
        "base_priority": 3,
        "initial_soc": 0.7,
        "landing_soc": 0.6,
        "wait_seconds": 4.0,
        "request_to_touchdown_seconds": 5.0,
    }]
    return {
        "label": label,
        "scenario": scenario,
        "seed": seed,
        "status": "failed" if failed else "completed",
        "requests": 1,
        "completed": 0 if failed else 1,
        "normal": 0 if failed else 1,
        "emergency": 0,
        "failed_energy": 0,
        # Mirrors V5's failed/default record: the terminal outcome counters
        # are zero even though requests is nonzero.
        "unresolved": 0,
        "other_failures": {},
        "summary": {"terminal": not failed},
        "action_tape": "tape.json",
        "lifecycle": {
            "replay_exact": not failed,
            "rows": rows,
            "by_priority": {"3": {"count": len(rows)}},
            "low_soc": {"mean_wait_seconds": None, "count": 0},
            "priority_weighted_wait_seconds": None if failed else 4.0,
        } if not failed else None,
        "gates": {"safety": True},
        "hard_gate_pass": not failed,
        "weights_unchanged": True,
        "teacher_enabled": False,
        "PPO_ALLOWED": False,
        "label_rows": 0,
        "reward_telemetry": {"available": True, "components_consistent": True},
        "_steps": [],
    }


def test_protocol_plan_has_16_fixed_cells_and_stress_bindings() -> None:
    protocol = evaluator._protocol()
    plan = evaluator._evaluation_plan()
    assert len(plan) == 16
    assert [item["seed"] for item in plan[:12]] == list(range(880101, 880113))
    assert [item["scenario"] for item in plan[:12]] == list(evaluator.IN_DISTRIBUTION_SCENARIOS)
    assert [(item["scenario"], item["seed"]) for item in plan[-4:]] == [
        ("dense30", 880201), ("dense30", 880202),
        ("random_dense24", 880301), ("random_dense24", 880302),
    ]
    configs = evaluator._build_configs(protocol)
    assert configs["dense30"].warm_start_uavs == 30
    assert configs["dense30"].max_simulation_seconds == 1800.0
    assert configs["random_dense24"].warm_start_layout_randomized is True


def test_formal_manifest_requires_completed_240_and_declared_checkpoint_sha(tmp_path: Path) -> None:
    formal_root = tmp_path / "formal"
    for run_index in range(3):
        run_dir = formal_root / f"run{run_index}"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "final.pt"
        checkpoint.write_bytes(f"checkpoint-{run_index}".encode())
        manifest = {
            "status": "COMPLETED",
            "processed_episodes": 240,
            "source_checkpoint_sha256": evaluator.SOURCE_CHECKPOINT_SHA256,
            "protocol_sha256": "protocol-sha",
            "final_checkpoint": str(checkpoint),
            "final_checkpoint_sha256": evaluator._sha256(checkpoint),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bindings = evaluator._validate_formal_manifests(formal_root, protocol_sha256="protocol-sha")
    assert [item["run_index"] for item in bindings] == [0, 1, 2]

    broken = json.loads((formal_root / "run1" / "manifest.json").read_text(encoding="utf-8"))
    broken["final_checkpoint_sha256"] = "wrong"
    (formal_root / "run1" / "manifest.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(evaluator.EvaluationInputError, match="SHA mismatch"):
        evaluator._validate_formal_manifests(formal_root, protocol_sha256="protocol-sha")


def test_load_eval_policy_rejects_checkpoint_sha_before_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = evaluator._build_configs(evaluator._protocol())["dense24"]
    monkeypatch.setattr(evaluator, "load_checkpoint", lambda *args, **kwargs: pytest.fail("loader called"))
    with pytest.raises(evaluator.EvaluationInputError, match="checkpoint SHA mismatch"):
        evaluator._load_eval_policy(
            checkpoint,
            config=config,
            expected_checkpoint_sha256="wrong",
            protocol_sha256="protocol-sha",
            final_checkpoint=False,
            learning_rate=3e-5,
        )


def test_real_smoke_uses_two_steps_and_serializes_tape(tmp_path: Path) -> None:
    protocol = evaluator._protocol()
    config = evaluator._build_configs(protocol)["dense24"]
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"source")
    result = evaluator._smoke(
        output=tmp_path / "smoke",
        policy=policy,
        checkpoint=checkpoint,
        config=config,
        protocol=protocol,
        runtime_protocol=evaluator.v6._runtime_protocol(protocol),
    )
    assert result["status"] == "SMOKE_COMPLETED"
    assert result["checks"]["exact_step_count"] is True
    assert result["checks"]["action_tape_present"] is True
    assert (tmp_path / "smoke" / "manifest.json").is_file()


def test_failed_default_record_keeps_unobserved_requests_in_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        total_requested_uavs=1,
        max_simulation_seconds=2.0,
        dt=1.0,
        teacher_enabled=False,
    )
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"source")
    monkeypatch.setattr(
        evaluator.admission,
        "_record_rollout",
        lambda **kwargs: _fake_record(
            label=kwargs["label"], scenario=kwargs["scenario"],
            seed=kwargs["seed"], failed=True,
        ),
    )
    monkeypatch.setattr(evaluator.v6, "_dev_pass", lambda record: False)
    record = evaluator._episode_record(
        policy=SimpleNamespace(),
        label="source_03c0",
        checkpoint=checkpoint,
        config=config,
        protocol={},
        runtime_protocol={},
        scenario="flow6",
        seed=1,
        output=tmp_path / "rollout",
        max_steps=2,
        require_full_resolution=False,
    )
    assert record["raw_outcome_counts"] == {
        "normal": 0,
        "emergency": 0,
        "failed_energy": 0,
        "other_failures": 0,
        "unresolved": 0,
    }
    assert record["remaining_requests_not_observed"] == 1
    assert record["unknown_outcomes"] == 1
    assert record["unresolved_includes_unknown"] is True
    assert record["unresolved"] == 1
    assert record["outcome_accounting_complete"] is False


def test_smoke_branch_is_source_only_and_skips_formal_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        total_requested_uavs=1,
        max_simulation_seconds=2.0,
        dt=1.0,
        warm_start_layout_randomized=False,
    )
    protocol = {"optimizer": {"learning_rate": 3e-4}}
    source_policy = SimpleNamespace(optimizer=SimpleNamespace(state={}))
    source_item = {
        "label": "source_03c0",
        "checkpoint": tmp_path / "source.pt",
        "checkpoint_sha256": "source-sha",
        "policy": source_policy,
        "model_weight_digest": "weights",
    }
    source_item["checkpoint"].write_bytes(b"source")
    monkeypatch.setattr(evaluator, "_protocol", lambda: protocol)
    monkeypatch.setattr(evaluator, "_sha256", lambda path: "protocol-sha")
    monkeypatch.setattr(evaluator, "_build_configs", lambda _: {"dense24": config})
    monkeypatch.setattr(
        evaluator, "_evaluation_plan",
        lambda: pytest.fail("source-only smoke must not consume evaluation seeds"),
    )
    monkeypatch.setattr(evaluator.v6, "_runtime_protocol", lambda _: {})
    monkeypatch.setattr(
        evaluator, "_load_source_binding",
        lambda **kwargs: (source_item, {
            "protocol_sha256": "protocol-sha",
            "source_checkpoint_sha256": "source-sha",
        }, {"source_checkpoint": str(source_item["checkpoint"])}),
    )
    monkeypatch.setattr(
        evaluator, "_load_bindings",
        lambda **kwargs: pytest.fail("formal manifests must not be read for smoke"),
    )
    monkeypatch.setattr(
        evaluator, "_smoke",
        lambda **kwargs: {
            "stage": f"{evaluator.STAGE}_SMOKE",
            "status": "SMOKE_COMPLETED",
            "checks": {"exact_step_count": True},
        },
    )
    result = evaluator.run(
        formal_root=tmp_path / "missing-formal",
        output=tmp_path / "evaluation",
        smoke_output=tmp_path / "smoke",
    )
    assert result["status"] == "SMOKE_COMPLETED"
    assert result["output_created"] is False
    assert result["arms"][0]["label"] == "source_03c0"
    assert not (tmp_path / "evaluation").exists()


def test_run_accounts_partial_arm_failure_and_keeps_other_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        total_requested_uavs=1,
        max_simulation_seconds=2.0,
        dt=1.0,
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        runtime_proposal_shortlist_k=3,
        teacher_enabled=False,
        scheduler_mode="execution",
        high_level_enabled=False,
        reconcile_late_merge_slots_enabled=False,
        warm_start_layout_randomized=False,
    )
    configs = {name: config for name in evaluator.EVALUATION_SCENARIOS}
    protocol = {"reward": {"low_soc_urgency_threshold": 0.65}}
    monkeypatch.setattr(evaluator, "_protocol", lambda: protocol)
    monkeypatch.setattr(evaluator, "_build_configs", lambda _: configs)
    monkeypatch.setattr(evaluator, "_evaluation_plan", lambda: [
        {"scenario": "flow6", "seed": 1},
        {"scenario": "dense30", "seed": 2},
    ])
    monkeypatch.setattr(evaluator, "_protocol_digest", lambda _: "protocol-digest")
    monkeypatch.setattr(evaluator, "_sha256", lambda path: "protocol-sha" if str(path).endswith(".yaml") else "checkpoint-sha")
    monkeypatch.setattr(evaluator.v6, "_runtime_protocol", lambda _: {})
    policies = []
    loaded = []
    for label in evaluator.ARM_LABELS:
        policy = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {}), optimizer=SimpleNamespace(state={}))
        policies.append(policy)
        loaded.append({
            "label": label,
            "checkpoint": Path(f"{label}.pt"),
            "checkpoint_sha256": "checkpoint-sha",
            "policy": policy,
            "payload": {},
            "model_weight_digest": "weights",
        })
    monkeypatch.setattr(
        evaluator,
        "_load_bindings",
        lambda **kwargs: (loaded, {"protocol_sha256": "protocol-sha", "source_checkpoint_sha256": evaluator.SOURCE_CHECKPOINT_SHA256}, {"final_manifests": []}),
    )
    monkeypatch.setattr(evaluator.v6, "_dev_pass", lambda record: record.get("status") == "completed")
    calls = []

    def fake_record(**kwargs):
        calls.append(kwargs["label"])
        failed = kwargs["label"] == "final_run1" and kwargs["seed"] == 2
        return _fake_record(label=kwargs["label"], scenario=kwargs["scenario"], seed=kwargs["seed"], failed=failed)

    monkeypatch.setattr(evaluator.admission, "_record_rollout", fake_record)
    result = evaluator.run(
        formal_root=tmp_path / "formal",
        output=tmp_path / "evaluation",
    )
    assert result["status"] == "COMPLETED"
    assert result["episodes_expected"] == 64
    assert result["episodes_processed"] == 8
    assert len(calls) == 8
    assert result["arms"]["final_run1"]["counts"]["unresolved"] == 1
    assert result["arms"]["final_run1"]["counts"]["remaining_requests_not_observed"] == 1
    assert result["arms"]["final_run2"]["episodes_processed"] == 2
    assert result["P3_QUALIFIED"] is False
    assert (tmp_path / "evaluation" / "pair_summary.json").is_file()
    assert (tmp_path / "evaluation" / "binding_manifest.json").is_file()
