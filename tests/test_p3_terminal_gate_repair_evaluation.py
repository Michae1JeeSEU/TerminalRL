from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from terminal_marl.training import evaluate_p3_terminal_gate_repair as runner


def _fit_report_fixture(tmp_path, monkeypatch, *, stage: str) -> tuple:
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "SOURCE_CHECKPOINT_SHA256", source_sha)
    report = {
        "stage": stage,
        "status": "COMPLETED",
        "source_checkpoint_sha256": source_sha,
        "checkpoint_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "metadata": {"training_scope": "gate_adapter_only"},
        "invariants": {
            "trainable_parameter_names": ["gate_adapter.0.weight"],
            "non_adapter_state_bitwise_unchanged": True,
            "proposal_logits_bitwise_unchanged": True,
            "checkpoint_roundtrip_model_exact": True,
            "adapter_state_changed": True,
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, candidate, source


@pytest.mark.parametrize("stage", sorted(runner.SUPPORTED_STAGES))
def test_fit_report_accepts_only_registered_gate_repair_stages(tmp_path, monkeypatch, stage) -> None:
    report_path, candidate, _ = _fit_report_fixture(tmp_path, monkeypatch, stage=stage)
    report, _report_sha = runner._load_fit_report(report_path, candidate)
    assert report["stage"] == stage


def test_fit_report_rejects_unregistered_stage(tmp_path, monkeypatch) -> None:
    report_path, candidate, _ = _fit_report_fixture(
        tmp_path, monkeypatch, stage="P3_TERMINAL_GATE_REPAIR_V3"
    )
    with pytest.raises(runner.TerminalGateRepairEvaluationError, match="stage"):
        runner._load_fit_report(report_path, candidate)


def test_check_only_binding_and_progress_use_validated_stage(tmp_path, monkeypatch) -> None:
    stage = "P3_TERMINAL_GATE_REPAIR_V2"
    report_path, candidate, source = _fit_report_fixture(tmp_path, monkeypatch, stage=stage)
    monkeypatch.setattr(runner, "SOURCE_CHECKPOINT", source)
    monkeypatch.setattr(runner, "load_baseline_records", lambda: [])
    monkeypatch.setattr(runner, "_validate_baseline_records", lambda records: None)
    monkeypatch.setattr(runner, "_baseline_summaries", lambda records: {"overall": {}})
    config = SimpleNamespace()
    monkeypatch.setattr(
        runner.collector,
        "_load_source_inputs",
        lambda root: ({}, {"flow6": config, "flow12": config, "dense24": config}, [], tmp_path / "source-manifest.json"),
    )
    monkeypatch.setattr(runner, "_load_policy", lambda checkpoint, config: (object(), {}))
    monkeypatch.setattr(
        runner,
        "_validate_freeze",
        lambda source_policy, candidate_policy: {"candidate_model_digest": "candidate-digest"},
    )

    result = runner.run(
        checkpoint=candidate,
        fit_report=report_path,
        output_dir=tmp_path / "check-only-output",
        check_only=True,
    )
    assert result["stage"] == stage

    progress_dir = tmp_path / "progress"
    runner._progress(progress_dir, stage=stage, status="RUNNING", processed=0, expected=6)
    assert json.loads((progress_dir / "progress.json").read_text(encoding="utf-8"))["stage"] == stage


def test_main_uses_result_stage(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        runner,
        "run",
        lambda **kwargs: {
            "stage": "P3_TERMINAL_GATE_REPAIR_V2",
            "status": "CHECK_ONLY_OK",
            "candidate_retained": False,
            "records_processed": 0,
        },
    )
    assert runner.main(["--checkpoint", "candidate.pt", "--fit-report", "report.json", "--output-dir", "out"]) == 0
    assert json.loads(capsys.readouterr().out)["stage"] == "P3_TERMINAL_GATE_REPAIR_V2"


def _record(cell: dict, *, normal: int | None = None, failed: int = 0,
            unresolved: int = 0, priority: float = 10.0,
            low_soc: float | None = 8.0, low_count: int = 1,
            safety: bool = False) -> dict:
    requests = runner.rank_eval.EXPECTED_REQUESTS[cell["scenario"]]
    if normal is None:
        normal = requests - failed - unresolved
    low_observed = low_count if low_soc is not None else 0
    return {
        "seed": int(cell["seed"]),
        "scenario": str(cell["scenario"]),
        "status": "completed",
        "lifecycle_valid": True,
        "outcome_counts": {
            "requests": requests,
            "normal": normal,
            "emergency": 0,
            "failed_energy": failed,
            "other": 0,
            "unresolved": unresolved,
        },
        "lifecycle": {
            "scope": "all_requested_terminal_rows",
            "replay_exact": True,
            "all_requests_observed": True,
            "missing_request_count": 0,
            "requested_count": requests,
            "priority_weighted_wait_seconds": priority,
            "low_soc": {
                "count": low_count,
                "observed_wait_count": low_observed,
                "mean_wait_seconds": low_soc,
            },
            "rows": [],
        },
        "safety_nonzero": safety,
        "teacher_called": False,
        "training_or_update_called": False,
    }


def _records(*, candidate_overrides: dict[int, dict] | None = None) -> list[dict]:
    overrides = candidate_overrides or {}
    return [
        _record(cell, **overrides.get(int(cell["seed"]), {}))
        for cell in runner._evaluation_cells()
    ]


def _summary(*, priority: float = 100.0, low_soc: float | None = 80.0) -> dict:
    return {"overall": {
        "priority_weighted_wait_seconds": priority,
        "low_soc_wait_seconds": low_soc,
    }}


def test_pair_records_requires_one_unique_record_per_scenario_seed() -> None:
    baseline = _records()
    candidate = _records()
    pairs = runner._pair_records(baseline, candidate)
    assert [(cell["seed"], cell["scenario"]) for cell, _, _ in pairs] == [
        (cell["seed"], cell["scenario"]) for cell in runner._evaluation_cells()
    ]

    with pytest.raises(runner.TerminalGateRepairEvaluationError, match="duplicate"):
        runner._pair_records(baseline + [baseline[0]], candidate)


def test_normal_decline_rejects_candidate() -> None:
    baseline = _records()
    candidate = _records(candidate_overrides={873103: {"normal": 23, "failed": 1}})
    result = runner._compare_records(
        baseline,
        candidate,
        _summary(priority=100.0, low_soc=80.0),
        _summary(priority=90.0, low_soc=70.0),
    )
    dense = next(item for item in result["cells"] if item["seed"] == 873103)
    assert dense["checks"]["normal"]["passed"] is False
    assert result["candidate_retained"] is False


def test_wait_tradeoff_rejects_candidate() -> None:
    records = _records()
    result = runner._compare_records(
        records,
        records,
        _summary(priority=100.0, low_soc=80.0),
        _summary(priority=90.0, low_soc=81.0),
    )
    assert result["aggregate_wait"]["priority_wait"]["passed"] is True
    assert result["aggregate_wait"]["low_soc_wait"]["passed"] is False
    assert result["candidate_retained"] is False


def test_low_soc_missing_is_reported_as_na_without_zero_fill() -> None:
    baseline = _records(candidate_overrides={873101: {"low_soc": None, "low_count": 0}})
    candidate = _records(candidate_overrides={873101: {"low_soc": None, "low_count": 0}})
    result = runner._compare_records(
        baseline,
        candidate,
        _summary(priority=100.0, low_soc=None),
        _summary(priority=90.0, low_soc=None),
    )
    cell = next(item for item in result["cells"] if item["seed"] == 873101)
    assert cell["checks"]["low_soc_wait"]["status"] == "N/A"
    assert cell["checks"]["low_soc_wait"]["candidate"] is None
    assert result["aggregate_wait"]["low_soc_wait"]["status"] == "N/A"


class _FakeModel:
    def __init__(self, *, adapter_delta: bool = False, encoder_delta: bool = False) -> None:
        self.training = False
        self._state = {
            "gate_adapter.0.weight": torch.tensor([1.0 + float(adapter_delta)]),
            "proposal_head.weight": torch.tensor([2.0]),
            "actor_encoder.weight": torch.tensor([3.0 + float(encoder_delta)]),
            "gate_temperature": torch.tensor(2.5),
        }

    def state_dict(self):
        return self._state


def test_freeze_allows_only_gate_adapter_change() -> None:
    source = SimpleNamespace(model=_FakeModel(), optimizer=SimpleNamespace(state={}))
    candidate = SimpleNamespace(model=_FakeModel(adapter_delta=True), optimizer=SimpleNamespace(state={}))
    result = runner._validate_freeze(source, candidate)
    assert result["gate_adapter_parameter_changed"] is True
    assert result["nonadapter_parameter_bitwise_unchanged"] is True
    assert result["proposal_parameter_bitwise_unchanged"] is True

    bad = SimpleNamespace(model=_FakeModel(adapter_delta=True, encoder_delta=True), optimizer=SimpleNamespace(state={}))
    with pytest.raises(runner.TerminalGateRepairEvaluationError, match="non-adapter"):
        runner._validate_freeze(source, bad)
