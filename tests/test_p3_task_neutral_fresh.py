from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.training import evaluate_p3_task_neutral_fresh as fresh


def _configs():
    config = SimpleNamespace(
        task_priority_enabled=False,
        low_level_priority_time_weight=0.0,
        total_requested_uavs=24,
    )
    return (
        {"flow6": config, "flow12": config, "dense24": config},
        {"flow6": config, "flow12": config, "dense24": config},
        {},
    )


def _metrics(record):
    failed = int(record.get("fake_failed_energy", 0))
    return {
        "requests": 24,
        "normal": 24 - failed,
        "emergency": 0,
        "failed_energy": failed,
        "failed": failed,
        "other": 0,
        "unresolved": 0,
        "mean_wait": 10.0,
        "p95_wait": 20.0,
        "max_wait": 30.0,
        "lowSOC_wait": 11.0,
        "priority_wait": 10.0,
        "reward": {
            "total": 25.0 if record.get("arm") == "after" else 24.0,
            "components": {},
        },
        "safety": {},
        "safety_nonzero": bool(record.get("fake_safety", False)),
        "weights_unchanged": True,
        "technical_ok": True,
        "lifecycle_valid": True,
    }


def _aggregate(records, *, smoke=False):
    del smoke
    failed = sum(int(row.get("fake_failed_energy", 0)) for row in records)
    return {
        "records": len(records),
        "requests": 24 * len(records),
        "normal": 24 * len(records) - failed,
        "emergency": 0,
        "failed": failed,
        "failed_energy": failed,
        "unresolved": 0,
        "mean_wait": 10.0,
        "p95_wait": 20.0,
        "max_wait": 30.0,
        "lowSOC_wait": 11.0,
        "reward_total": 24.0 * len(records),
        "technical_ok": True,
        "full_request_complete": True,
    }


def _install_fake_runtime(monkeypatch, calls, *, fatal_seed=None, failed_seed=None):
    monkeypatch.setattr(fresh.neutral_eval, "_build_neutral_inputs", _configs)
    monkeypatch.setattr(
        fresh,
        "_load_profile",
        lambda arm, config: (
            SimpleNamespace(digest=f"{arm}-digest"),
            {
                "arm": arm,
                "resolved_checkpoint": f"{arm}.pt",
                "model_digest": f"{arm}-digest",
                "checkpoint_sha256": f"{arm}-sha",
                "expected_checkpoint_sha256": f"{arm}-sha",
                "pilot_payload_check": {"phase_config_match": True},
            },
        ),
    )
    monkeypatch.setattr(fresh.rank_eval, "_model_digest", lambda policy: policy.digest)
    monkeypatch.setattr(fresh.neutral_eval, "_record_metrics", lambda record, smoke=False: _metrics(record))
    monkeypatch.setattr(fresh.neutral_eval, "_aggregate_arm", _aggregate)
    monkeypatch.setattr(
        fresh.neutral_eval,
        "_record_fatal_reason",
        lambda record, smoke=False: record.get("fake_fatal"),
    )

    def fake_runtime(**kwargs):
        seed = int(kwargs["seed"])
        arm = "after" if "after" in str(kwargs["output"]).replace("\\", "/") else "before"
        calls.append((arm, seed, int(kwargs["policy_seed"])))
        result = {
            "arm": arm,
            "scenario": kwargs["scenario"],
            "seed": seed,
            "policy_seed": int(kwargs["policy_seed"]),
            "weights_unchanged": True,
            "lifecycle_valid": True,
            "safety_nonzero": False,
            "summary": {},
            "outcome_counts": {"requests": 24, "normal": 24, "emergency": 0,
                                "failed_energy": 0, "other": 0, "unresolved": 0},
        }
        if failed_seed == seed:
            result["fake_failed_energy"] = 1
        if fatal_seed == seed:
            result["fake_fatal"] = "runtime safety field indicates a violation"
        return result

    monkeypatch.setattr(fresh.rank_eval, "_runtime_record", fake_runtime)


def test_fixed_matrix_passes_distinct_policy_seed_to_all_eight_calls(monkeypatch, tmp_path: Path) -> None:
    calls = []
    _install_fake_runtime(monkeypatch, calls)
    result = fresh.run(output_dir=tmp_path / "full")

    assert result["status"] == "COMPLETED"
    assert result["records_processed"] == 8
    assert calls == [
        (arm, seed, fresh.POLICY_SEEDS[seed])
        for arm in fresh.ARMS
        for seed in (874601, 874602, 874603, 874604)
    ]
    assert result["arms"]["before"]["summary"]["requests"] == 96
    assert result["arms"]["after"]["summary"]["requests"] == 96
    assert len(list((tmp_path / "full" / "records" / "before").glob("*.json"))) == 4
    assert len(list((tmp_path / "full" / "records" / "after").glob("*.json"))) == 4
    assert len(result["paired_cells"]) == 4
    assert result["paired_cells"][0]["delta_after_minus_before"]["reward_total"] == pytest.approx(1.0)


def test_check_only_does_zero_runtime_rollouts_and_plans_full_matrix(monkeypatch, tmp_path: Path) -> None:
    calls = []
    _install_fake_runtime(monkeypatch, calls)
    monkeypatch.setattr(fresh.rank_eval, "_runtime_record", lambda **kwargs: pytest.fail("runtime called"))
    result = fresh.run(output_dir=tmp_path / "check", check_only=True)

    assert result["status"] == "CHECK_ONLY_OK"
    assert result["check_only"] is True
    assert result["records_processed"] == 0
    assert result["binding"]["matrix_expected"] == 8
    assert result["checks"]["check_only_zero_runtime"] is True
    assert calls == []
    progress = json.loads((tmp_path / "check" / "progress.json").read_text(encoding="utf-8"))
    assert progress["runtime_rollout_calls"] == 0


def test_failed_energy_is_retained_without_performance_stop(monkeypatch, tmp_path: Path) -> None:
    calls = []
    _install_fake_runtime(monkeypatch, calls, failed_seed=874602)
    result = fresh.run(output_dir=tmp_path / "failed_energy")

    assert result["status"] == "COMPLETED"
    assert result["records_processed"] == 8
    assert result["arms"]["before"]["summary"]["failed"] == 1
    saved = json.loads(
        (tmp_path / "failed_energy" / "records" / "before" / "dense24_874602.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["fake_failed_energy"] == 1


def test_fresh_aggregation_does_not_call_registered_six_split(monkeypatch) -> None:
    monkeypatch.setattr(
        fresh.rank_eval,
        "_candidate_summaries",
        lambda *args, **kwargs: pytest.fail("registered six-cell split helper called"),
    )
    monkeypatch.setattr(fresh.neutral_eval, "_record_metrics", lambda record, smoke=False: _metrics(record))
    records = [
        {"seed": seed, "scenario": "dense24", "fake_failed_energy": 0}
        for seed in (874601, 874602, 874603, 874604)
    ]
    summary = fresh.neutral_eval._aggregate_arm(records, smoke=False)
    assert summary["requests"] == 96


def test_safety_failure_stops_after_saving_failing_record(monkeypatch, tmp_path: Path) -> None:
    calls = []
    _install_fake_runtime(monkeypatch, calls, fatal_seed=874603)
    result = fresh.run(output_dir=tmp_path / "safety_stop")

    assert result["status"] == "FAILED"
    assert result["records_processed"] == 3
    assert result["stop_reason"].startswith("before/dense24/874603:")
    assert (tmp_path / "safety_stop" / "records" / "before" / "dense24_874603.json").is_file()
    assert not (tmp_path / "safety_stop" / "records" / "after").exists()


def test_after_profile_validates_fixed_pilot_payload(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "pilot.pt"
    checkpoint.write_bytes(b"pilot")
    config = _configs()[1]["dense24"]
    checked = []
    monkeypatch.setattr(fresh.neutral_eval, "_synced_path", lambda path: checkpoint)
    monkeypatch.setattr(fresh, "_sha256", lambda path: fresh.AFTER_CHECKPOINT_SHA256)
    monkeypatch.setattr(
        fresh.rank_eval,
        "_load_policy",
        lambda path, config: (SimpleNamespace(digest="after-digest"), {"stage": "pilot"}),
    )
    monkeypatch.setattr(fresh.rank_eval, "_model_digest", lambda policy: policy.digest)
    monkeypatch.setattr(
        fresh.pilot,
        "_validate_pilot_payload",
        lambda payload, cfg: checked.append((payload, cfg)) or {"phase_config_match": True},
    )

    _policy, profile = fresh._load_profile("after", config)
    assert profile["checkpoint_sha256"] == fresh.AFTER_CHECKPOINT_SHA256
    assert profile["pilot_payload_check"]["phase_config_match"] is True
    assert checked and checked[0][0]["stage"] == "pilot"


def test_cli_help_has_only_output_and_check_only(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        fresh.main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--output-dir" in output
    assert "--check-only" in output
