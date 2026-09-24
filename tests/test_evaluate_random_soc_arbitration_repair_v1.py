from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.training import evaluate_random_soc_arbitration_repair_v1 as evaluator
from terminal_marl.training import evaluate_random_soc_gate_repair_v1 as base


@dataclass
class _Config:
    total_requested_uavs: int = 24
    max_simulation_seconds: float = 1800.0
    dt: float = 1.0
    teacher_enabled: bool = False
    reconcile_late_merge_slots_enabled: bool = False
    scheduler_mode: str = "execution"
    release_arbitration_mode: str = "legacy_insertion"
    vehicle_profile: str = "research_evtol"


def _patch_configs(monkeypatch):
    monkeypatch.setattr(evaluator, "_load_yaml_protocol", lambda: {})
    monkeypatch.setattr(evaluator, "_build_config", lambda scenario, protocol: _Config())


def _fake_record(label, config, *, status="completed", passed=True):
    return {
        "label": label,
        "status": status,
        "PASS": passed,
        "completed": 24,
        "requests": 24,
        "unresolved": 0,
        "weights_unchanged": True,
        "label_rows": 0,
        "PPO_ALLOWED": False,
        "teacher_enabled": config.teacher_enabled,
        "safety_invariant_error": status == "failed",
        "lifecycle": {"replay_exact": True, "rows": []},
        "gates": {"all": passed},
    }


def _patch_full_run(monkeypatch, *, stop_first=False, performance_fail=False):
    _patch_configs(monkeypatch)
    calls = []

    def fake_arm(**kwargs):
        calls.append(kwargs)
        label = kwargs["label"]
        if label == "baseline" and stop_first:
            return (
                _fake_record(label, kwargs["config"], status="failed", passed=False),
                True,
                "first arm stopped",
            )
        passed = not (performance_fail and label == "baseline")
        return _fake_record(label, kwargs["config"], passed=passed), False, None

    monkeypatch.setattr(evaluator, "_evaluate_arm", fake_arm)
    monkeypatch.setattr(
        evaluator,
        "_aggregate",
        lambda record: ({
            "completion_fraction": 1.0,
            "priority_weighted_wait_seconds": 2.0,
            "low_soc_mean_wait_seconds": 2.0,
        }, None),
    )
    monkeypatch.setattr(
        evaluator,
        "_dev_selection_gates",
        lambda initial, candidate: {"report_only": True},
    )
    return calls


def test_pair_config_snapshot_has_only_arbitration_diff(monkeypatch):
    _patch_configs(monkeypatch)
    baseline, soc_priority, _, _, diff = evaluator._paired_configs({})
    assert baseline.release_arbitration_mode == "downstream_priority"
    assert soc_priority.release_arbitration_mode == "downstream_soc_priority"
    assert diff == {
        "release_arbitration_mode": {
            "baseline": "downstream_priority",
            "soc_priority": "downstream_soc_priority",
        }
    }


def test_both_full_arms_use_repaired_loader_seam_and_same_sha(tmp_path, monkeypatch):
    calls = _patch_full_run(monkeypatch)
    result = evaluator.run(
        checkpoint=tmp_path / "candidate.pt",
        checkpoint_sha256=evaluator.CANDIDATE_CHECKPOINT_SHA256,
        output=tmp_path / "pair",
        seed=847101,
    )
    assert [call["label"] for call in calls] == ["baseline", "soc_priority"]
    assert all(call["candidate_sha256"] == evaluator.CANDIDATE_CHECKPOINT_SHA256 for call in calls)
    assert [call["config"].release_arbitration_mode for call in calls] == [
        "downstream_priority", "downstream_soc_priority",
    ]
    assert result["development_only"] is True
    assert result["P3_QUALIFIED"] is False
    assert result["PPO_ALLOWED"] is False
    assert result["P4_ALLOWED"] is False
    assert result["promotable"] is False
    assert result["development_protocol"].replace("\\", "/").endswith(
        "docs/evidence/v4_2/random_soc_arbitration_repair_v1_20260925.md"
    )
    assert all(row["development_only"] is True for row in result["records"])
    assert result["selection"]["selection_report_only"] is True


def test_smoke_uses_two_steps_without_changing_actual_config(tmp_path, monkeypatch):
    _patch_configs(monkeypatch)
    loaders = []
    rollouts = []

    class Step:
        teacher_used = False

        def __init__(self):
            self.world = np.zeros((1, 88), dtype=np.float32)

    def fake_loader(path, config, *, expected_sha256):
        loaders.append((path, config, expected_sha256))
        return SimpleNamespace(), {"stage": "repair"}

    def fake_rollout(policy, config, **kwargs):
        rollouts.append((config, kwargs))
        output = kwargs["output"]
        output.mkdir(parents=True)
        tape = output / "action_tape.json"
        tape.write_text(json.dumps({"schema_version": "test", "action_tape": [{}, {}]}), encoding="utf-8")
        return [Step(), Step()], {
            "steps": 2,
            "completed": 0,
            "outcomes": {},
            "action_tape": str(tape),
            "label_rows": 0,
            "PPO_ALLOWED": False,
            "safety": {"SafetyInvariantError": 0},
            "safety_error": None,
        }

    monkeypatch.setattr(evaluator, "load_repaired_checkpoint", fake_loader)
    monkeypatch.setattr(evaluator, "_policy_weight_digest", lambda policy: "stable")
    monkeypatch.setattr(evaluator.student, "rollout", fake_rollout)
    result = evaluator.run(
        checkpoint=tmp_path / "candidate.pt",
        checkpoint_sha256=evaluator.CANDIDATE_CHECKPOINT_SHA256,
        output=tmp_path / "smoke",
        seed=847000,
        smoke=True,
    )
    assert result["status"] == "SMOKE_COMPLETED"
    assert len(loaders) == 2
    assert len(rollouts) == 2
    assert [kwargs["max_steps"] for _, kwargs in rollouts] == [2, 2]
    assert all(config.max_simulation_seconds == 1800.0 for config, _ in rollouts)
    assert all(config.teacher_enabled is False for config, _ in rollouts)
    assert all(row["weights_unchanged"] is True for row in result["records"])
    assert all(row["PASS"] is False for row in result["records"])
    json.dumps(evaluator._jsonable(result), allow_nan=False)


def test_smoke_arm_uses_custom_checkpoint_loader(tmp_path, monkeypatch):
    config = _Config()
    loader_calls = []
    rollout_calls = []

    class Step:
        teacher_used = False

        def __init__(self):
            self.world = np.zeros((1, 88), dtype=np.float32)

    def custom_loader(path, loaded_config):
        loader_calls.append((path, loaded_config))
        return SimpleNamespace(), {"checkpoint_identity": "canonical_initial"}

    def fake_rollout(policy, loaded_config, **kwargs):
        rollout_calls.append((loaded_config, kwargs))
        kwargs["output"].mkdir(parents=True)
        tape = kwargs["output"] / "action_tape.json"
        tape.write_text(json.dumps({"action_tape": [{}, {}]}), encoding="utf-8")
        return [Step(), Step()], {
            "completed": 0,
            "outcomes": {},
            "action_tape": str(tape),
            "label_rows": 0,
            "PPO_ALLOWED": False,
            "safety": {"SafetyInvariantError": 0},
            "safety_error": None,
        }

    monkeypatch.setattr(evaluator, "_policy_weight_digest", lambda policy: "stable")
    monkeypatch.setattr(evaluator.student, "rollout", fake_rollout)
    record, stopped, reason = evaluator._smoke_arm(
        label="canonical_initial",
        checkpoint=tmp_path / "canonical.pt",
        candidate_sha256="0" * 64,
        config=config,
        seed=847000,
        output=tmp_path / "arm",
        checkpoint_loader=custom_loader,
    )

    assert stopped is False
    assert reason is None
    assert loader_calls == [(tmp_path / "canonical.pt", config)]
    assert rollout_calls[0][0] is config
    assert rollout_calls[0][1]["max_steps"] == 2
    assert config.max_simulation_seconds == 1800.0
    assert record["loader"] == {"checkpoint_identity": "canonical_initial"}


def test_smoke_nonzero_safety_counter_stops_second_arm(tmp_path, monkeypatch):
    _patch_configs(monkeypatch)

    class Step:
        teacher_used = False

        def __init__(self):
            self.world = np.zeros((1, 88), dtype=np.float32)

    def fake_loader(path, config, *, expected_sha256):
        return SimpleNamespace(), {"stage": "repair"}

    def fake_rollout(policy, config, **kwargs):
        output = kwargs["output"]
        output.mkdir(parents=True)
        tape = output / "action_tape.json"
        tape.write_text(json.dumps({"action_tape": [{}, {}]}), encoding="utf-8")
        return [Step(), Step()], {
            "steps": 2,
            "completed": 0,
            "outcomes": {},
            "action_tape": str(tape),
            "label_rows": 0,
            "PPO_ALLOWED": False,
            "safety": {"collisions": 1, "SafetyInvariantError": 0},
            "safety_error": None,
        }

    monkeypatch.setattr(evaluator, "load_repaired_checkpoint", fake_loader)
    monkeypatch.setattr(evaluator, "_policy_weight_digest", lambda policy: "stable")
    monkeypatch.setattr(evaluator.student, "rollout", fake_rollout)
    result = evaluator.run(
        checkpoint=tmp_path / "candidate.pt",
        checkpoint_sha256=evaluator.CANDIDATE_CHECKPOINT_SHA256,
        output=tmp_path / "smoke_safety",
        seed=847000,
        smoke=True,
    )
    assert result["status"] == "SMOKE_FAILED"
    assert result["records"][0]["safety_invariant_error"] is True
    assert result["records"][1]["status"] == "not_run"


@pytest.mark.parametrize("seed,sha", [(847001, evaluator.CANDIDATE_CHECKPOINT_SHA256), (847101, "0" * 64)])
def test_invalid_seed_or_hash_is_rejected_before_protocol_or_reset(tmp_path, monkeypatch, seed, sha):
    monkeypatch.setattr(evaluator, "_load_yaml_protocol", lambda: pytest.fail("must reject before protocol load"))
    with pytest.raises(ValueError):
        evaluator.run(
            checkpoint=tmp_path / "candidate.pt",
            checkpoint_sha256=sha,
            output=tmp_path / "invalid",
            seed=seed,
            smoke=(seed == 847001),
        )
    assert not (tmp_path / "invalid").exists()


def test_safety_error_stops_second_arm(tmp_path, monkeypatch):
    calls = _patch_full_run(monkeypatch, stop_first=True)
    result = evaluator.run(
        checkpoint=tmp_path / "candidate.pt",
        checkpoint_sha256=evaluator.CANDIDATE_CHECKPOINT_SHA256,
        output=tmp_path / "safety",
        seed=847101,
    )
    assert [call["label"] for call in calls] == ["baseline"]
    assert result["stopped_after_baseline"] is True
    assert result["records"][1]["status"] == "not_run"


def test_performance_failure_still_runs_second_arm(tmp_path, monkeypatch):
    calls = _patch_full_run(monkeypatch, performance_fail=True)
    result = evaluator.run(
        checkpoint=tmp_path / "candidate.pt",
        checkpoint_sha256=evaluator.CANDIDATE_CHECKPOINT_SHA256,
        output=tmp_path / "performance",
        seed=847102,
    )
    assert [call["label"] for call in calls] == ["baseline", "soc_priority"]
    assert result["records"][0]["PASS"] is False
    assert result["records"][1]["status"] == "completed"
    assert result["stopped_after_baseline"] is False


def test_summary_is_json_serializable_and_output_is_fresh(tmp_path, monkeypatch):
    _patch_full_run(monkeypatch)
    output = tmp_path / "pair"
    result = evaluator.run(
        checkpoint=tmp_path / "candidate.pt",
        checkpoint_sha256=evaluator.CANDIDATE_CHECKPOINT_SHA256,
        output=output,
        seed=847101,
    )
    assert (output / "pair_summary.json").is_file()
    json.dumps(evaluator._jsonable(result), allow_nan=False)
    with pytest.raises(FileExistsError, match="fresh"):
        evaluator.run(
            checkpoint=tmp_path / "candidate.pt",
            checkpoint_sha256=evaluator.CANDIDATE_CHECKPOINT_SHA256,
            output=output,
            seed=847101,
        )
