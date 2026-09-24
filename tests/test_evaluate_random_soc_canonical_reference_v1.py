from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from terminal_marl.training import evaluate_random_soc_canonical_reference_v1 as evaluator


@dataclass
class _Config:
    total_requested_uavs: int = 24
    max_simulation_seconds: float = 1800.0
    dt: float = 1.0
    teacher_enabled: bool = False
    reconcile_late_merge_slots_enabled: bool = False
    scheduler_mode: str = "execution"
    release_arbitration_mode: str = "downstream_priority"


def _patch_common(monkeypatch):
    config = _Config()
    monkeypatch.setattr(evaluator, "_load_yaml_protocol", lambda: {})
    monkeypatch.setattr(evaluator, "_build_config", lambda scenario, protocol: config)
    monkeypatch.setattr(
        evaluator,
        "_config_snapshot",
        lambda value: {
            "total_requested_uavs": value.total_requested_uavs,
            "max_simulation_seconds": value.max_simulation_seconds,
            "release_arbitration_mode": value.release_arbitration_mode,
        },
    )
    return config


def _record(
    label, *, status="completed", passed=False, safety=False, performance=False, ppo=False
):
    gates = {key: True for key in evaluator._STOP_ON_FALSE_GATES}
    gates["no_failed_outcomes"] = not performance
    return {
        "label": label,
        "status": status,
        "PASS": passed,
        "safety_invariant_error": safety,
        "summary": {"safety": {"collisions": int(safety)}},
        "gates": gates,
        "weights_unchanged": True,
        "teacher_enabled": False,
        "label_rows": 0,
        "PPO_ALLOWED": ppo,
        "lifecycle": {"replay_exact": True},
    }


def test_smoke_uses_distinct_fixed_loaders_and_common_config(tmp_path, monkeypatch):
    config = _patch_common(monkeypatch)
    loader_calls = []
    arm_calls = []

    def canonical_loader(path, loaded_config):
        loader_calls.append(("canonical", path, loaded_config))
        return object(), {"checkpoint_identity": "canonical_initial", "canonical_initial": True}

    def candidate_loader(path, loaded_config, *, expected_sha256):
        loader_calls.append(("candidate", path, loaded_config, expected_sha256))
        return object(), {"stage": "RANDOM_SOC_GATE_REPAIR_V1"}

    monkeypatch.setattr(evaluator.diagnostic, "load_canonical_initial_checkpoint", canonical_loader)
    monkeypatch.setattr(evaluator.repair, "load_repaired_checkpoint", candidate_loader)

    def fake_smoke_arm(**kwargs):
        policy, metadata = kwargs["checkpoint_loader"](kwargs["checkpoint"], kwargs["config"])
        arm_calls.append((kwargs["label"], kwargs["config"], policy, metadata))
        return _record(kwargs["label"]), False, None

    monkeypatch.setattr(evaluator, "_smoke_arm", fake_smoke_arm)
    result = evaluator.run(
        canonical_checkpoint=tmp_path / "canonical.pt",
        candidate_checkpoint=tmp_path / "candidate.pt",
        output=tmp_path / "smoke",
        seed=evaluator.SMOKE_SEED,
        smoke=True,
    )

    assert result["status"] == "SMOKE_COMPLETED"
    assert [item[0] for item in arm_calls] == ["canonical_initial", "repair_candidate"]
    assert arm_calls[0][1] is config
    assert arm_calls[1][1] is config
    assert [item[0] for item in loader_calls] == ["canonical", "candidate"]
    assert loader_calls[0][2] is config
    assert loader_calls[1][2] is config
    assert loader_calls[1][3] == evaluator.CANDIDATE_CHECKPOINT_SHA256
    assert result["canonical_checkpoint_sha256"] == evaluator.CANONICAL_INITIAL_CHECKPOINT_SHA256
    assert result["candidate_checkpoint_sha256"] == evaluator.CANDIDATE_CHECKPOINT_SHA256
    assert result["actual_config_snapshots"]["canonical_initial"] == result["actual_config_snapshots"]["repair_candidate"]
    assert result["config_diff"] == {}
    assert result["horizon_steps"] == 1800
    assert result["smoke_max_steps"] == 2
    assert result["selection"]["selection_report_only"] is True
    saved = json.loads((tmp_path / "smoke" / "pair_summary.json").read_text(encoding="utf-8"))
    assert saved["checkpoints"]["canonical_initial"]["loader"] == (
        "diagnostic.load_canonical_initial_checkpoint"
    )
    json.dumps(saved, allow_nan=False)
    with pytest.raises(FileExistsError, match="fresh"):
        evaluator.run(
            canonical_checkpoint=tmp_path / "canonical.pt",
            candidate_checkpoint=tmp_path / "candidate.pt",
            output=tmp_path / "smoke",
            seed=evaluator.SMOKE_SEED,
            smoke=True,
        )


def test_full_source_failure_stops_candidate_arm(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    calls = []
    monkeypatch.setattr(evaluator, "_aggregate", lambda record: (None, None))
    monkeypatch.setattr(evaluator, "_dev_selection_gates", lambda initial, candidate: {"report": False})

    def fake_full_arm(**kwargs):
        calls.append(kwargs["label"])
        return _record(kwargs["label"], status="failed", safety=True), True, "canonical failure"

    monkeypatch.setattr(evaluator, "_evaluate_arm", fake_full_arm)
    result = evaluator.run(
        canonical_checkpoint=tmp_path / "canonical.pt",
        candidate_checkpoint=tmp_path / "candidate.pt",
        output=tmp_path / "failed",
        seed=evaluator.FULL_SEEDS[0],
    )

    assert calls == ["canonical_initial"]
    assert result["status"] == "FAILED"
    assert result["records"][1]["status"] == "not_run"
    assert result["records"][1]["loader_identity"] == "repaired_candidate"
    assert result["failure"]["safety_invariant_error"] is True


def test_full_candidate_performance_failure_is_recorded_and_pair_continues(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    calls = []
    monkeypatch.setattr(evaluator, "_aggregate", lambda record: (None, None))
    monkeypatch.setattr(evaluator, "_dev_selection_gates", lambda initial, candidate: {"report": False})

    def fake_full_arm(**kwargs):
        calls.append(kwargs["label"])
        return _record(kwargs["label"], passed=False, performance=True), False, None

    monkeypatch.setattr(evaluator, "_evaluate_arm", fake_full_arm)
    result = evaluator.run(
        canonical_checkpoint=tmp_path / "canonical.pt",
        candidate_checkpoint=tmp_path / "candidate.pt",
        output=tmp_path / "performance",
        seed=evaluator.FULL_SEEDS[1],
    )

    assert calls == ["canonical_initial", "repair_candidate"]
    assert result["status"] == "COMPLETED"
    assert [row["status"] for row in result["records"]] == ["completed", "completed"]
    assert all(row["PASS"] is False for row in result["records"])
    assert result["selection"]["PASS"] is False
    assert result["selection"]["selection_report_only"] is True


def test_full_nonzero_safety_counter_stops_but_outcome_failure_continues(
    tmp_path, monkeypatch
):
    _patch_common(monkeypatch)
    monkeypatch.setattr(evaluator, "_aggregate", lambda record: (None, None))
    monkeypatch.setattr(evaluator, "_dev_selection_gates", lambda initial, candidate: {"report": False})
    safety_calls = []

    def safety_arm(**kwargs):
        safety_calls.append(kwargs["label"])
        return _record(kwargs["label"], safety=True), False, None

    monkeypatch.setattr(evaluator, "_evaluate_arm", safety_arm)
    result = evaluator.run(
        canonical_checkpoint=tmp_path / "canonical.pt",
        candidate_checkpoint=tmp_path / "candidate.pt",
        output=tmp_path / "safety",
        seed=evaluator.FULL_SEEDS[0],
    )
    assert safety_calls == ["canonical_initial"]
    assert result["records"][0]["integrity_failure"] is True
    assert result["records"][1]["status"] == "not_run"

    performance_calls = []

    def performance_arm(**kwargs):
        performance_calls.append(kwargs["label"])
        return _record(kwargs["label"], performance=True), False, None

    monkeypatch.setattr(evaluator, "_evaluate_arm", performance_arm)
    result = evaluator.run(
        canonical_checkpoint=tmp_path / "canonical.pt",
        candidate_checkpoint=tmp_path / "candidate.pt",
        output=tmp_path / "performance_only",
        seed=evaluator.FULL_SEEDS[0],
    )
    assert performance_calls == ["canonical_initial", "repair_candidate"]
    assert result["status"] == "COMPLETED"


def test_full_true_ppo_flag_is_not_masked_by_pair_metadata(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(evaluator, "_aggregate", lambda record: (None, None))
    monkeypatch.setattr(evaluator, "_dev_selection_gates", lambda initial, candidate: {"report": False})
    calls = []

    def ppo_arm(**kwargs):
        calls.append(kwargs["label"])
        return _record(kwargs["label"], ppo=True), False, None

    monkeypatch.setattr(evaluator, "_evaluate_arm", ppo_arm)
    result = evaluator.run(
        canonical_checkpoint=tmp_path / "canonical.pt",
        candidate_checkpoint=tmp_path / "candidate.pt",
        output=tmp_path / "ppo",
        seed=evaluator.FULL_SEEDS[0],
    )
    assert calls == ["canonical_initial"]
    assert result["records"][0]["PPO_ALLOWED"] is True
    assert result["records"][1]["status"] == "not_run"
