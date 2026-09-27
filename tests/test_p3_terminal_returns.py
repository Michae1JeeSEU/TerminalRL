from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from terminal_marl.training.audit_p3_terminal_returns import (
    ReturnAccumulator,
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
