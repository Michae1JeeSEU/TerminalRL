from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from terminal_marl.training import evaluate_p3_terminal_gate_repair as runner


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
