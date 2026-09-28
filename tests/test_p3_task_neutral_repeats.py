from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from terminal_marl.training import evaluate_p3_task_neutral as neutral
from terminal_marl.training import evaluate_p3_task_neutral_repeats as repeats


@dataclasses.dataclass
class _Config:
    task_priority_enabled: bool = True
    low_level_priority_time_weight: float = 1.0
    label: str = "dense24"


def _configs():
    original = _Config()
    return {"dense24": original}, {"dense24": neutral._neutralize_config(original)}, {}


def _record(seed: int, policy_seed: int, *, failed_energy: int = 0) -> dict:
    requests = 24
    rows = [
        {
            "arrival_id": index,
            "request_time": float(index),
            "touchdown_time": float(index + 1),
            "wait_seconds": 1.0,
            "base_priority": 1,
            "initial_soc": 0.8,
            "mission_type": "standard",
            "outcome": "failed_energy" if failed_energy and index == requests - 1 else "completed",
        }
        for index in range(requests)
    ]
    lifecycle = {
        "scope": "all_requested_terminal_rows",
        "replay_exact": True,
        "all_requests_observed": True,
        "ledger_alignment": True,
        "requested_count": requests,
        "row_count": requests,
        "missing_request_count": 0,
        "outcomes": {"completed": requests - failed_energy, "failed_energy": failed_energy},
        "rows": rows,
    }
    return {
        "scenario": "dense24",
        "seed": seed,
        "policy_seed": policy_seed,
        "status": "completed",
        "requests": requests,
        "lifecycle": lifecycle,
        "lifecycle_valid": True,
        "outcome_counts": {
            "requests": requests,
            "normal": requests - failed_energy,
            "emergency": 0,
            "failed_energy": failed_energy,
            "other": 0,
            "unresolved": 0,
        },
        "weights_unchanged": True,
        "safety_nonzero": False,
        "teacher_called": False,
        "training_or_update_called": False,
        "summary": {
            "reward_total": float(requests),
            "reward_component_totals": {"total": float(requests)},
            "reward_components_consistent": True,
            "safety": {"collisions": 0},
        },
    }


def _patch_runtime(monkeypatch, tmp_path: Path, *, failure_at: int | None = None):
    before = tmp_path / "before.pt"
    after = tmp_path / "after.pt"
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    monkeypatch.setattr(repeats, "BEFORE_CHECKPOINT", before)
    monkeypatch.setattr(repeats, "AFTER_CHECKPOINT", after)
    monkeypatch.setattr(repeats, "BEFORE_CHECKPOINT_SHA256", "before-sha")
    monkeypatch.setattr(repeats, "AFTER_CHECKPOINT_SHA256", "after-sha")
    monkeypatch.setattr(repeats.neutral_eval, "_build_neutral_inputs", _configs)
    monkeypatch.setattr(
        repeats.neutral_eval,
        "_checkpoint_from_report",
        lambda checkpoint, expected_sha: (Path(checkpoint), None),
    )
    calls = {"load": 0, "runtime": 0, "updates": 0}

    class _Policy:
        def update(self, *args, **kwargs):
            calls["updates"] += 1
            raise AssertionError("repeat evaluation must not update policy")

    def fake_load(*args, **kwargs):
        calls["load"] += 1
        return _Policy(), {}

    monkeypatch.setattr(repeats.rank_eval, "_load_policy", fake_load)
    monkeypatch.setattr(repeats.rank_eval, "_model_digest", lambda policy: "digest")

    def fake_runtime(**kwargs):
        calls["runtime"] += 1
        if failure_at is not None and calls["runtime"] == failure_at:
            return {
                **_record(kwargs["seed"], kwargs["policy_seed"]),
                "status": "failed",
                "error": {"type": "SyntheticFailure", "message": "synthetic failure"},
            }
        return _record(
            kwargs["seed"], kwargs["policy_seed"],
            failed_energy=1 if calls["runtime"] == 1 else 0,
        )

    monkeypatch.setattr(repeats.rank_eval, "_runtime_record", fake_runtime)
    return calls


def test_repeat_matrix_has_two_arms_two_cells_and_four_policy_seeds():
    matrix = repeats.repeat_matrix()
    keys = {
        (item["arm"], item["env_seed"], item["policy_seed"])
        for item in matrix
    }
    assert len(matrix) == 16
    assert len(keys) == 16
    assert {item["arm"] for item in matrix} == {"before", "after"}
    assert {item["env_seed"] for item in matrix} == {873103, 873106}
    assert {item["policy_seed"] for item in matrix} == set(repeats.POLICY_SEEDS)


def test_check_only_loads_both_weights_without_runtime_or_update(monkeypatch, tmp_path):
    calls = _patch_runtime(monkeypatch, tmp_path)
    result = repeats.run(output_dir=tmp_path / "check", check_only=True)
    assert result["status"] == "CHECK_ONLY_OK"
    assert result["check_only"] is True
    assert result["full_evaluation"] is False
    assert result["technical_ok"] is None
    assert calls["load"] == 2
    assert calls["runtime"] == 0
    assert calls["updates"] == 0
    progress = json.loads((tmp_path / "check" / "progress.json").read_text(encoding="utf-8"))
    assert progress["full_evaluation"] is False


def test_full_matrix_reloads_each_arm_seed_and_keeps_performance_failure(monkeypatch, tmp_path):
    calls = _patch_runtime(monkeypatch, tmp_path)
    result = repeats.run(output_dir=tmp_path / "full")
    assert result["status"] == "COMPLETED"
    assert result["records_processed"] == 16
    assert result["technical_ok"] is True
    assert calls["runtime"] == 16
    assert calls["load"] == 18  # two binding loads plus one fresh load per rollout
    assert calls["updates"] == 0
    assert result["arms"]["before"]["failed"] == 1
    assert len(result["paired_differences"]) == 8
    assert len(list((tmp_path / "full" / "records" / "before").glob("*.json"))) == 8
    assert len(list((tmp_path / "full" / "records" / "after").glob("*.json"))) == 8
    progress = json.loads((tmp_path / "full" / "progress.json").read_text(encoding="utf-8"))
    assert progress["last_record"] == {
        "arm": "after", "scenario": "dense24", "env_seed": 873106,
        "policy_seed": repeats.POLICY_SEEDS[-1],
    }


def test_technical_failure_stops_matrix_and_preserves_failed_record(monkeypatch, tmp_path):
    calls = _patch_runtime(monkeypatch, tmp_path, failure_at=1)
    result = repeats.run(output_dir=tmp_path / "failed")
    assert result["status"] == "FAILED"
    assert result["records_processed"] == 1
    assert calls["runtime"] == 1
    assert calls["updates"] == 0
    assert "runtime status is 'failed'" in result["stop_reason"]
    record = json.loads(next((tmp_path / "failed" / "records").rglob("*.json")).read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"]["message"] == "synthetic failure"
    assert record["policy_seed"] == repeats.POLICY_SEEDS[0]


def test_cli_help_is_available(capsys):
    with pytest.raises(SystemExit) as exc:
        repeats.main(["--help"])
    assert exc.value.code == 0
    assert "--output-dir" in capsys.readouterr().out
