from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.core.uav import MissionType
from terminal_marl.training import evaluate_p3_task_neutral as runner


@dataclasses.dataclass
class _Config:
    task_priority_enabled: bool = True
    low_level_priority_time_weight: float = 1.0
    unrelated: int = 7


def test_neutral_config_changes_only_registered_fields() -> None:
    original = _Config()
    neutral = runner._neutralize_config(original)
    assert original.task_priority_enabled is True
    assert original.low_level_priority_time_weight == 1.0
    assert neutral.task_priority_enabled is False
    assert neutral.low_level_priority_time_weight == 0.0
    assert neutral.unrelated == original.unrelated
    assert runner._config_delta(original, neutral) == {
        "task_priority_enabled": {"old": True, "new": False},
        "low_level_priority_time_weight": {"old": 1.0, "new": 0.0},
    }


class _Request:
    def __init__(self, index: int, *, neutral: bool) -> None:
        self.arrival_id = index
        self.request_time = float(index) * 2.0
        self.gate_id = index % 4
        self.mission_type = (MissionType.CARGO, MissionType.STANDARD, MissionType.EMERGENCY)[index % 3]
        self.base_priority = 1 if neutral else (5 if index % 2 else 3)
        self.energy_capacity = 1500.0
        self.initial_soc = 0.70 + index / 100.0


class _ArrivalProcess:
    def __init__(self, config: _Config, seed: int | None = None) -> None:
        del seed
        self.config = config
        self.index = 0
        self.next_time = 0.0

    def exhausted(self) -> bool:
        return self.index >= 6

    def requests_until(self, now: float):
        del now
        if self.exhausted():
            return []
        request = _Request(self.index, neutral=not self.config.task_priority_enabled)
        self.index += 1
        self.next_time = float(self.index) * 2.0
        return [request]


def test_arrival_preflight_preserves_physical_fields_and_neutralizes_base(monkeypatch) -> None:
    monkeypatch.setattr(runner, "ArrivalProcess", _ArrivalProcess)
    result = runner.validate_arrival_preflight(
        _Config(), runner._neutralize_config(_Config()), seed=873101, expected_requests=6,
    )
    assert result["physical_equal"] is True
    assert result["new_base_all_one"] is True
    assert [row["request_time"] for row in result["old"]] == [row["request_time"] for row in result["new"]]
    assert [row["gate_id"] for row in result["old"]] == [row["gate_id"] for row in result["new"]]
    assert [row["mission_type"] for row in result["old"]] == [row["mission_type"] for row in result["new"]]
    assert [row["service_seconds"] for row in result["old"]] == [row["service_seconds"] for row in result["new"]]
    assert all(row["base_priority"] == 1 for row in result["new"])


def _record(seed: int, scenario: str, waits: list[float], *, failed_energy: int = 0,
            smoke: bool = False) -> dict:
    requests = len(waits)
    rows = []
    for index, wait in enumerate(waits):
        failed = failed_energy and index == len(waits) - 1
        rows.append({
            "arrival_id": index,
            "request_time": float(index),
            "touchdown_time": float(index) + wait,
            "wait_seconds": wait,
            "base_priority": 1,
            "initial_soc": 0.60 if index == 0 else 0.80,
            "mission_type": "standard",
            "outcome": "failed_energy" if failed else "completed",
        })
    lifecycle = {
        "scope": "smoke_prefix" if smoke else "all_requested_terminal_rows",
        "replay_exact": not smoke,
        "all_requests_observed": not smoke,
        "ledger_alignment": not smoke,
        "requested_count": requests,
        "row_count": requests,
        "missing_request_count": 0 if not smoke else requests,
        "outcomes": {"completed": requests - failed_energy, "failed_energy": failed_energy},
        "rows": rows,
    }
    return {
        "seed": seed,
        "scenario": scenario,
        "requests": requests,
        "status": "smoke_prefix" if smoke else "completed",
        "lifecycle": lifecycle,
        "lifecycle_valid": not smoke,
        "outcome_counts": {
            "requests": requests,
            "normal": requests - failed_energy,
            "emergency": 0,
            "failed_energy": failed_energy,
            "other": 0,
            "unresolved": 0,
        },
        "normal": requests - failed_energy,
        "emergency": 0,
        "failed_energy": failed_energy,
        "unresolved": 0,
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


def test_pairing_rejects_duplicate_or_missing_cells() -> None:
    cells = runner._evaluation_cells()
    source = [_record(int(cell["seed"]), str(cell["scenario"]), [1.0]) for cell in cells]
    candidate = [_record(int(cell["seed"]), str(cell["scenario"]), [1.0]) for cell in cells]
    assert len(runner.pair_records(source, candidate)) == 6
    with pytest.raises(runner.TaskNeutralEvaluationError, match="duplicate"):
        runner.pair_records(source + [source[0]], candidate)
    with pytest.raises(runner.TaskNeutralEvaluationError, match="exactly"):
        runner.pair_records(source[:-1], candidate)


def test_aggregate_uses_all_lifecycle_rows_and_keeps_failure_denominator(monkeypatch) -> None:
    monkeypatch.setattr(
        runner.rank_eval,
        "_candidate_summaries",
        lambda records, smoke: {"train": {}, "validation": {}, "overall": {"records": len(records)}},
    )
    cells = runner._evaluation_cells()
    records = []
    waits = [[0.0, 10.0], [100.0], [5.0], [5.0], [5.0], [5.0]]
    for cell, values in zip(cells, waits):
        records.append(_record(int(cell["seed"]), str(cell["scenario"]), values,
                               failed_energy=int(cell["seed"]) == 873102))
    summary = runner.aggregate_arm(records)
    all_waits = [0.0, 10.0, 100.0, 5.0, 5.0, 5.0, 5.0]
    assert summary["requests"] == len(all_waits)
    assert summary["failed"] == 1
    assert summary["mean_wait"] == pytest.approx(sum(all_waits) / len(all_waits))
    assert summary["p95_wait"] == pytest.approx(73.0)
    assert summary["max_wait"] == 100.0
    assert summary["lowSOC_wait"] == pytest.approx(20.0)
    assert summary["reward"]["total"] == pytest.approx(float(len(all_waits)))


def test_complete_failed_energy_is_performance_failure_but_not_technical_fatal() -> None:
    record = _record(873101, "flow6", [10.0, 20.0], failed_energy=1)
    assert runner._record_fatal_reason(record, smoke=False) is None
    assert runner._record_metrics(record)["failed"] == 1
    record["safety_nonzero"] = True
    assert runner._record_fatal_reason(record, smoke=False) is not None


def test_check_only_writes_terminal_report_without_runtime_rollout(monkeypatch, tmp_path: Path) -> None:
    config = _Config()
    configs = {name: config for name in ("flow6", "flow12", "dense24")}
    neutral = {name: runner._neutralize_config(config) for name in configs}
    monkeypatch.setattr(runner, "_build_neutral_inputs", lambda: (configs, neutral, {}))
    monkeypatch.setattr(runner, "preflight_arrival_process", lambda *args: [{"physical_equal": True, "new_base_all_one": True}])
    monkeypatch.setattr(runner, "_weight_preflight", lambda *args, **kwargs: {"checkpoint_sha256": "ok"})
    monkeypatch.setattr(runner.rank_eval, "_runtime_record", lambda **kwargs: pytest.fail("rollout called in check-only"))
    result = runner.run(output_dir=tmp_path / "check", check_only=True)
    assert result["status"] == "CHECK_ONLY_OK"
    assert result["check_only"] is True
    assert result["full_evaluation"] is False
    assert result["technical_ok"] is None
    assert (tmp_path / "check" / "progress.json").is_file()
    report = json.loads((tmp_path / "check" / "report.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "check" / "progress.json").read_text(encoding="utf-8"))
    assert progress["check_only"] is True
    assert progress["full_evaluation"] is False
    assert report["candidate_retained"] is None
    assert report["retention_decision"] == "pending_primary_review"


def test_full_mock_run_writes_one_paired_record_per_arm_cell(monkeypatch, tmp_path: Path) -> None:
    config = _Config()
    configs = {name: config for name in ("flow6", "flow12", "dense24")}
    neutral = {name: runner._neutralize_config(config) for name in configs}
    monkeypatch.setattr(runner, "_build_neutral_inputs", lambda: (configs, neutral, {}))
    cells = runner._evaluation_cells()
    monkeypatch.setattr(
        runner,
        "preflight_arrival_process",
        lambda configs, neutral, selected: [
            {"seed": int(cell["seed"]), "scenario": str(cell["scenario"]),
             "physical_equal": True, "new_base_all_one": True}
            for cell in selected
        ],
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(runner, "_checkpoint_from_report", lambda *args: (checkpoint, None))
    monkeypatch.setattr(runner, "_weight_preflight", lambda *args, **kwargs: {
        "checkpoint_sha256": "fixed", "model_digest": "digest", "optimizer_state_empty": True,
    })
    monkeypatch.setattr(runner.rank_eval, "_load_policy", lambda *args: (SimpleNamespace(), {}))

    # The real entry point passes arm through the output path, so keep a
    # deterministic source/candidate counter for this mocked runtime.
    calls = {"source": 0, "candidate": 0}

    def fake_runtime_with_arm(**kwargs):
        arm = "candidate" if "candidate" in str(kwargs["output"]) else "source"
        calls[arm] += 1
        requests = runner.EXPECTED_REQUESTS[kwargs["scenario"]]
        return _record(kwargs["seed"], kwargs["scenario"], [float(i) for i in range(requests)],
                       failed_energy=int(arm == "candidate" and kwargs["seed"] == 873103))

    monkeypatch.setattr(runner.rank_eval, "_runtime_record", fake_runtime_with_arm)
    result = runner.run(output_dir=tmp_path / "full")
    assert result["status"] == "COMPLETED"
    assert result["records_processed"] == 12
    assert result["arms"]["source"]["summary"]["requests"] == runner.EXPECTED_TOTAL_REQUESTS
    assert result["arms"]["candidate"]["summary"]["failed"] == 1
    assert calls == {"source": 6, "candidate": 6}
    assert len(list((tmp_path / "full" / "records" / "source").glob("*.json"))) == 6
    assert len(list((tmp_path / "full" / "records" / "candidate").glob("*.json"))) == 6
    assert json.loads((tmp_path / "full" / "report.json").read_text(encoding="utf-8"))["candidate_retained"] is None


def test_preflight_failure_leaves_failed_report_and_progress(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "_build_neutral_inputs", lambda: (_ for _ in ()).throw(
        runner.TaskNeutralEvaluationError("preflight boom")
    ))
    result = runner.run(output_dir=tmp_path / "failed", check_only=True)
    assert result["status"] == "FAILED"
    report = json.loads((tmp_path / "failed" / "report.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "failed" / "progress.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAILED"
    assert "preflight boom" in report["stop_reason"]
    assert progress["status"] == "FAILED"


def test_cli_help_is_available(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        runner.main(["--help"])
    assert exc.value.code == 0
    assert "--output-dir" in capsys.readouterr().out
