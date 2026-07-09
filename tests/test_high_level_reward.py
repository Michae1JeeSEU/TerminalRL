"""Tests for interval-level terminal-airspace reward diagnostics."""

import numpy as np

from terminal_marl.rewards.high_level_reward import HighLevelReward


def snapshot(**overrides) -> dict[str, float]:
    values = {
        "time": 0.0,
        "landed_count": 0.0,
        "final_queue_time": 0.0,
        "conflict_count": 0.0,
        "safety_override_count": 0.0,
        "ring_occupancy_0": 0.5,
        "ring_occupancy_1": 0.5,
    }
    values.update(overrides)
    return values


def test_landing_and_throughput_produce_positive_reward() -> None:
    model = HighLevelReward()
    reward, diagnostics = model.compute(
        snapshot(),
        snapshot(time=10.0, landed_count=2.0),
        np.ones(8),
        np.ones(8),
    )
    assert reward > 0.0
    assert diagnostics["landed_delta"] == 2.0
    assert diagnostics["throughput"] == 0.2


def test_conflict_waiting_override_and_congestion_are_penalized() -> None:
    model = HighLevelReward()
    reward, diagnostics = model.compute(
        snapshot(),
        snapshot(
            time=10.0,
            final_queue_time=20.0,
            conflict_count=2.0,
            safety_override_count=3.0,
            ring_occupancy_0=1.5,
            ring_occupancy_1=0.2,
        ),
        np.ones(8),
        np.ones(8),
    )
    assert reward < 0.0
    assert diagnostics["component_conflict"] < 0.0
    assert diagnostics["component_waiting"] < 0.0
    assert diagnostics["congestion"] > 0.0


def test_airspace_change_and_change_smoothness_are_reported() -> None:
    model = HighLevelReward()
    _, first = model.compute(
        snapshot(),
        snapshot(time=5.0),
        np.ones(8),
        np.full(8, 1.1),
    )
    _, second = model.compute(
        snapshot(time=5.0),
        snapshot(time=10.0),
        np.full(8, 1.1),
        np.full(8, 1.2),
    )
    assert first["airspace_change_norm"] > 0.0
    assert first["smoothness_norm"] > 0.0
    assert second["smoothness_norm"] < first["smoothness_norm"]
    assert "total_reward" in second
