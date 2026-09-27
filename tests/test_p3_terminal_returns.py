from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from terminal_marl.training.audit_p3_terminal_returns import (
    ReturnAccumulator,
    _comparison_row,
    _augment_new_admission_phases,
)


def _step(reward: float, *, progress: float | None = None):
    value = float(reward if progress is None else progress)
    components = {
        "time": 0.0,
        "progress": value,
        "release": 0.0,
        "touchdown": 0.0,
        "failure": 0.0,
        "teacher": 0.0,
        "shield": 0.0,
        "urgency": 0.0,
        "emergency": 0.0,
        "total": float(reward),
    }
    return {"0": float(reward)}, {"reward_components": {"0": components}}


def test_discount_starts_at_gamma_zero_and_components_match() -> None:
    accumulator = ReturnAccumulator(gamma=.99)
    rewards0, info0 = _step(2.0)
    rewards1, info1 = _step(3.0)
    accumulator.add_step(rewards0, info0, pre_step_phase={"0": "ring_outer"}, step_index=0)
    accumulator.add_step(rewards1, info1, pre_step_phase={"0": "ring_inner"}, step_index=1)

    result = accumulator.summary()
    assert result["raw_return_discounted"] == pytest.approx(2.0 + .99 * 3.0)
    assert result["components_discounted"]["progress"] == pytest.approx(2.0 + .99 * 3.0)
    assert result["progress_by_pre_step_phase"]["ring_outer"]["discounted"] == pytest.approx(2.0)
    assert result["progress_by_pre_step_phase"]["ring_inner"]["discounted"] == pytest.approx(.99 * 3.0)
    assert result["raw_components_consistent"] is True


def test_missing_component_is_not_filled_with_zero() -> None:
    accumulator = ReturnAccumulator()
    rewards, info = _step(1.0)
    del info["reward_components"]["0"]["emergency"]
    with pytest.raises(ValueError, match="component emergency missing"):
        accumulator.add_step(rewards, info, pre_step_phase={"0": "ring_outer"}, step_index=0)


def test_return_summary_is_json_serializable() -> None:
    accumulator = ReturnAccumulator()
    rewards, info = _step(1.25)
    accumulator.add_step(rewards, info, pre_step_phase={"0": "ring_outer"}, step_index=0)
    encoded = json.dumps(accumulator.summary(), allow_nan=False)
    assert '"raw_return_discounted"' in encoded


def test_new_admission_phase_is_reported_without_dropping_progress() -> None:
    accumulator = ReturnAccumulator()
    rewards, info = _step(1.5, progress=1.5)
    phases = _augment_new_admission_phases(
        SimpleNamespace(active={0: object()}, service_uavs={}, completed_uavs=[]),
        {},
        rewards,
    )
    accumulator.add_step(
        rewards,
        info,
        pre_step_phase=phases,
        step_index=0,
    )

    result = accumulator.summary()
    assert result["progress_by_pre_step_phase"]["newly_admitted_this_step"]["undiscounted"] == pytest.approx(1.5)
    assert result["raw_return_undiscounted"] == pytest.approx(1.5)


def _comparison_record(arm: str, scenario: str, seed: int, raw: float, wait: float, low_soc_wait):
    return {
        "arm": arm,
        "scenario": scenario,
        "seed": seed,
        "returns": {
            "raw_return_undiscounted": raw,
            "raw_return_discounted": raw / 10.0,
        },
        "original_summary": {"reward_total": raw},
        "outcomes": {"normal": 1, "emergency": 0, "failed_energy": 0, "unresolved": 0},
        "waits": {
            "mean_wait_seconds": wait,
            "low_soc_mean_wait_seconds": low_soc_wait,
            "priority_weighted_wait_seconds": wait + 1.0,
        },
    }


def test_comparison_row_matches_arm_scenario_seed_and_preserves_low_soc_null() -> None:
    rows = []
    for scenario, seed, raw, wait, low_soc in (
        ("flow6", 1, 10.0, 2.0, None),
        ("flow12", 2, 20.0, 8.0, 7.0),
    ):
        for arm, offset in (("source_03c0", 0.0), ("final_run0", 1.0), ("final_run1", 2.0), ("final_run2", 3.0)):
            rows.append(_comparison_record(arm, scenario, seed, raw + offset, wait + offset, low_soc))

    flow6 = _comparison_row(rows, scenario="flow6", seed=1)
    flow12 = _comparison_row(rows, scenario="flow12", seed=2)
    assert flow6["final_run2"]["reconstructed_raw_return"] == pytest.approx(13.0)
    assert flow6["final_run2"]["low_soc_mean_wait_seconds"] is None
    assert flow12["final_run2"]["reconstructed_raw_return"] == pytest.approx(23.0)
    assert flow12["final_run2"]["low_soc_mean_wait_seconds"] == pytest.approx(7.0)


def test_comparison_row_rejects_missing_or_duplicate_identity() -> None:
    source = _comparison_record("source_03c0", "flow6", 1, 1.0, 1.0, None)
    rows = [source]
    for arm in ("final_run0", "final_run1"):
        rows.append(_comparison_record(arm, "flow6", 1, 1.0, 1.0, None))
    with pytest.raises(ValueError, match="comparison identity must have one record"):
        _comparison_row(rows, scenario="flow6", seed=1)

    rows.append(_comparison_record("final_run2", "flow6", 1, 1.0, 1.0, None))
    rows.append(_comparison_record("source_03c0", "flow6", 1, 2.0, 2.0, None))
    with pytest.raises(ValueError, match="comparison identity must have one record"):
        _comparison_row(rows, scenario="flow6", seed=1)
