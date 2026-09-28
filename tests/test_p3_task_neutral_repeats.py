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
    max_simulation_seconds: float = 1800.0
    dt: float = 1.0


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


def _write_reuse_root(tmp_path: Path, *, mutate: tuple[str, object] | None = None) -> Path:
    root = tmp_path / "old_full"
    source_dir = root / "records" / "before"
    neutral_config = neutral._config_snapshot(neutral._neutralize_config(_Config()))
    for item in repeats.repeat_matrix():
        if item["arm"] != "before":
            continue
        record = _record(int(item["env_seed"]), int(item["policy_seed"]))
        record.update({
            "arm": "before",
            "env_seed": int(item["env_seed"]),
            "checkpoint_sha256": repeats.BEFORE_CHECKPOINT_SHA256,
            "checkpoint_sha256_expected": repeats.BEFORE_CHECKPOINT_SHA256,
            "config": neutral_config,
            "task_priority_enabled": False,
            "low_level_priority_time_weight": 0.0,
            "technical_ok": True,
            "model_unchanged": True,
            "PPO_ALLOWED": False,
        })
        record["summary"].update({
            "seed": int(item["env_seed"]),
            "policy_seed": int(item["policy_seed"]),
        })
        if mutate and (int(item["env_seed"]), int(item["policy_seed"])) == (873103, 874301):
            record[mutate[0]] = mutate[1]
        path = source_dir / (
            f"dense24_env{item['env_seed']}_policy{item['policy_seed']}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
    return root


def _write_joint_team_record_root(tmp_path: Path, *, mutate: tuple[str, object] | None = None) -> Path:
    root = tmp_path / "joint_team_records"
    neutral_config = neutral._config_snapshot(neutral._neutralize_config(_Config()))
    for item in repeats.joint_team_matrix():
        if not item["reused"]:
            continue
        record = _record(int(item["env_seed"]), int(item["policy_seed"]))
        record.update({
            "arm": str(item["arm"]),
            "env_seed": int(item["env_seed"]),
            "checkpoint_sha256": (
                repeats.BEFORE_CHECKPOINT_SHA256
                if item["arm"] == "before" else repeats.JOINT_TEAM_AFTER_CHECKPOINT_SHA256
            ),
            "config": neutral_config,
            "task_priority_enabled": False,
            "low_level_priority_time_weight": 0.0,
            "technical_ok": True,
            "model_unchanged": True,
            "PPO_ALLOWED": False,
        })
        record["summary"].update({
            "seed": int(item["env_seed"]),
            "policy_seed": int(item["policy_seed"]),
        })
        if mutate and (str(item["arm"]), int(item["env_seed"])) == ("before", 875101):
            record[mutate[0]] = mutate[1]
        path = root / str(item["arm"]) / f"dense24_{item['env_seed']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
    return root


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


def _patch_joint_team_runtime(monkeypatch, tmp_path: Path, *, failure_at: int | None = None):
    calls = _patch_runtime(monkeypatch, tmp_path, failure_at=failure_at)
    joint_after = tmp_path / "joint_batch.pt"
    joint_after.write_bytes(b"joint-after")
    monkeypatch.setattr(repeats, "JOINT_TEAM_AFTER_CHECKPOINT", joint_after)
    monkeypatch.setattr(repeats, "JOINT_TEAM_AFTER_CHECKPOINT_SHA256", "joint-after-sha")
    monkeypatch.setattr(
        repeats.pilot_eval, "_validate_pilot_payload",
        lambda payload, config: {"phase_config_match": True},
    )
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


def test_joint_team_matrix_has_three_policy_seeds_per_four_environment_cells():
    matrix = repeats.joint_team_matrix()
    keys = {(item["arm"], item["env_seed"], item["policy_seed"]) for item in matrix}
    assert len(matrix) == len(keys) == 24
    assert {item["arm"] for item in matrix} == {"before", "after"}
    assert {item["env_seed"] for item in matrix} == set(repeats.JOINT_TEAM_ENV_SEEDS)
    assert all(
        {item["policy_seed"] for item in matrix if item["env_seed"] == env_seed}
        == set(repeats.JOINT_TEAM_POLICY_SEEDS[env_seed])
        for env_seed in repeats.JOINT_TEAM_ENV_SEEDS
    )
    assert sum(bool(item["reused"]) for item in matrix) == 8
    assert sum(not bool(item["reused"]) for item in matrix) == 16
    assert all(
        sum(1 for item in matrix if item["arm"] == arm and item["env_seed"] == env_seed) == 3
        for arm in repeats.ARMS for env_seed in repeats.JOINT_TEAM_ENV_SEEDS
    )


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


def test_joint_team_check_only_validates_24_matrix_8_reuse_16_new_and_zero_runtime(monkeypatch, tmp_path):
    calls = _patch_joint_team_runtime(monkeypatch, tmp_path)
    record_root = _write_joint_team_record_root(tmp_path)
    result = repeats.run(
        output_dir=tmp_path / "joint_check", check_only=True,
        joint_team_cohort=True, joint_team_record_root=record_root,
    )
    assert result["status"] == "CHECK_ONLY_OK"
    assert result["records_expected"] == 24
    assert result["records_processed"] == 0
    assert result["planned_reused_records"] == 8
    assert result["planned_new_rollouts"] == 16
    assert result["binding"]["after_checkpoint_sha256"] == "joint-after-sha"
    assert result["cohort"]["requests_per_arm"] == 288
    assert result["cohort"]["horizon_steps_per_rollout"] == 1800
    assert result["cohort"]["max_new_rollout_steps"] == 16 * repeats.JOINT_TEAM_HORIZON_STEPS
    assert result["binding"]["horizon_steps_per_rollout"] == 1800
    assert result["checks"]["matrix_unique"] is True
    assert result["checks"]["reused_records_validated"] is True
    assert result["checks"]["runtime_zero"] is True
    assert result["checks"]["runtime_rollouts"] == 0
    assert calls["runtime"] == 0
    assert calls["updates"] == 0


def test_joint_team_cohort_reuses_eight_records_and_runs_sixteen_new_rollouts(monkeypatch, tmp_path):
    calls = _patch_joint_team_runtime(monkeypatch, tmp_path)
    record_root = _write_joint_team_record_root(tmp_path)
    result = repeats.run(
        output_dir=tmp_path / "joint_full",
        joint_team_cohort=True, joint_team_record_root=record_root,
    )
    assert result["status"] == "COMPLETED"
    assert result["records_expected"] == result["records_processed"] == 24
    assert result["records_reused"] == 8
    assert result["new_rollouts"] == 16
    assert result["arms"]["before"]["records"] == 12
    assert result["arms"]["after"]["records"] == 12
    assert result["arms"]["before"]["requests"] == 288
    assert result["arms"]["after"]["requests"] == 288
    assert result["new_only_arms"]["before"]["records"] == 8
    assert result["new_only_arms"]["after"]["records"] == 8
    assert len(result["paired_differences"]) == 12
    assert len(result["new_only_paired_differences"]) == 8
    assert result["new_only_records_processed"] == 16
    assert calls["runtime"] == 16
    assert calls["updates"] == 0
    copied = list((tmp_path / "joint_full" / "records").glob("*/dense24_env*_policy*.json"))
    reused_copies = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in copied
        if "source_provenance" in json.loads(path.read_text(encoding="utf-8"))
    ]
    assert len(reused_copies) == 8
    assert all(row["source_provenance"]["kind"] == "reused_joint_team_record" for row in reused_copies)


@pytest.mark.parametrize("field,value", [("seed", 999999), ("checkpoint_sha256", "wrong-sha")])
def test_joint_team_reuse_rejects_identity_or_checkpoint(monkeypatch, tmp_path, field, value):
    calls = _patch_joint_team_runtime(monkeypatch, tmp_path)
    record_root = _write_joint_team_record_root(tmp_path, mutate=(field, value))
    result = repeats.run(
        output_dir=tmp_path / f"joint_bad_{field}", check_only=True,
        joint_team_cohort=True, joint_team_record_root=record_root,
    )
    assert result["status"] == "FAILED"
    assert result["records_processed"] == 0
    assert calls["runtime"] == 0
    assert result["stop_reason"]


def test_joint_team_cohort_rejects_legacy_after_or_reuse_arguments(tmp_path):
    with pytest.raises(repeats.TaskNeutralRepeatsError, match="fixes the after checkpoint"):
        repeats.run(
            output_dir=tmp_path / "joint_custom_after",
            joint_team_cohort=True,
            after_checkpoint=tmp_path / "other.pt",
            after_checkpoint_sha256="other-sha",
        )
    with pytest.raises(repeats.TaskNeutralRepeatsError, match="does not accept --reuse-before-root"):
        repeats.run(
            output_dir=tmp_path / "joint_legacy_reuse",
            joint_team_cohort=True,
            reuse_before_root=tmp_path / "old",
        )


def test_full_matrix_reloads_each_arm_seed_and_keeps_performance_failure(monkeypatch, tmp_path):
    calls = _patch_runtime(monkeypatch, tmp_path)
    result = repeats.run(output_dir=tmp_path / "full")
    assert result["status"] == "COMPLETED"
    assert result["records_processed"] == 16
    assert result["technical_ok"] is True
    assert calls["runtime"] == 16
    assert calls["load"] == 18  # two binding loads plus one fresh load per rollout
    assert calls["updates"] == 0
    assert result["records_reused"] == 0
    assert result["new_rollouts"] == 16
    assert result["arms"]["before"]["failed"] == 1
    assert len(result["paired_differences"]) == 8
    assert len(list((tmp_path / "full" / "records" / "before").glob("*.json"))) == 8
    assert len(list((tmp_path / "full" / "records" / "after").glob("*.json"))) == 8
    progress = json.loads((tmp_path / "full" / "progress.json").read_text(encoding="utf-8"))
    assert progress["last_record"] == {
        "arm": "after", "scenario": "dense24", "env_seed": 873106,
        "policy_seed": repeats.POLICY_SEEDS[-1],
    }


def test_reuse_root_copies_before_records_and_runs_after_only(monkeypatch, tmp_path):
    calls = _patch_runtime(monkeypatch, tmp_path)
    reuse_root = _write_reuse_root(tmp_path)
    source_path = reuse_root / "records" / "before" / "dense24_env873103_policy874301.json"
    source_before = source_path.read_bytes()
    result = repeats.run(output_dir=tmp_path / "reuse", reuse_before_root=reuse_root)
    assert result["status"] == "COMPLETED"
    assert result["records_processed"] == 16
    assert result["records_reused"] == 8
    assert result["new_rollouts"] == 8
    assert result["planned_reused_records"] == 8
    assert result["planned_new_rollouts"] == 8
    assert calls["runtime"] == 8
    assert calls["load"] == 10
    assert source_path.read_bytes() == source_before
    copied = json.loads(next((tmp_path / "reuse" / "records" / "before").glob("*.json")).read_text())
    assert copied["record_path"].startswith(str(tmp_path / "reuse"))
    assert copied["source_provenance"]["kind"] == "reused_before_record"
    assert len(result["reused_source_paths"]) == 8


@pytest.mark.parametrize("field,value", [
    ("policy_seed", 999),
    ("checkpoint_sha256", "wrong-sha"),
    ("config", {"wrong": True}),
])
def test_reuse_preflight_rejects_identity_sha_or_config(monkeypatch, tmp_path, field, value):
    reuse_root = _write_reuse_root(tmp_path, mutate=(field, value))
    calls = _patch_runtime(monkeypatch, tmp_path)
    result = repeats.run(output_dir=tmp_path / f"bad_{field}", reuse_before_root=reuse_root, check_only=True)
    assert result["status"] == "FAILED"
    assert result["records_processed"] == 0
    assert calls["runtime"] == 0
    assert result["stop_reason"]


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


def test_custom_after_requires_checkpoint_and_sha_pair(tmp_path, capsys):
    result = repeats.main([
        "--output-dir", str(tmp_path / "pair"),
        "--after-checkpoint", str(tmp_path / "custom.pt"),
    ])
    assert result == 2
    assert "supplied together" in capsys.readouterr().out


def test_custom_after_check_validates_pilot_payload_without_env(monkeypatch, tmp_path):
    calls = _patch_runtime(monkeypatch, tmp_path)
    custom = tmp_path / "custom.pt"
    custom.write_bytes(b"custom")
    checked = []
    monkeypatch.setattr(
        repeats.pilot_eval, "_validate_pilot_payload",
        lambda payload, config: checked.append((payload, config)) or {"phase_config_match": True},
    )
    result = repeats.run(
        output_dir=tmp_path / "custom_check", check_only=True,
        after_checkpoint=custom, after_checkpoint_sha256="custom-sha",
    )
    assert result["status"] == "CHECK_ONLY_OK"
    assert result["records_processed"] == 0
    assert len(checked) == 1
    assert result["binding"]["arms"]["after"]["declared_checkpoint"] == str(custom)
    assert result["binding"]["arms"]["after"]["expected_checkpoint_sha256"] == "custom-sha"
    assert calls["runtime"] == 0
