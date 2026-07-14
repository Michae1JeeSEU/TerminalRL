import numpy as np

from terminal_marl.rewards.high_level_reward import HighLevelReward


def snapshot(time: float, safety: float) -> dict[str, float]:
    return {
        "time": time,
        "landed_count": 0.0,
        "final_queue_time": 0.0,
        "conflict_count": 0.0,
        "safety_override_count": safety,
        "ring_occupancy_0": 0.0,
        "ring_occupancy_1": 0.0,
    }


def test_safety_override_penalty_uses_interval_delta_not_cumulative_count() -> None:
    reward_model = HighLevelReward()
    old_airspace = np.ones(8)
    new_airspace = np.ones(8)
    reward_1, diagnostics_1 = reward_model.compute(
        snapshot(0.0, 100.0),
        snapshot(10.0, 101.0),
        old_airspace,
        new_airspace,
    )
    reward_2, diagnostics_2 = reward_model.compute(
        snapshot(10.0, 500.0),
        snapshot(20.0, 501.0),
        old_airspace,
        new_airspace,
    )
    assert diagnostics_1["safety_override_delta"] == 1.0
    assert diagnostics_2["safety_override_delta"] == 1.0
    assert reward_1 == reward_2

