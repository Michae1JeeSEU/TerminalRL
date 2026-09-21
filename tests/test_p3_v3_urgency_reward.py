import math

import pytest

from terminal_marl.core.config import SimConfig
from terminal_marl.core.rewards import (
    LowLevelReward,
    low_level_urgency_penalty,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _penalty(**overrides):
    values = {
        "base_time_penalty": 0.01,
        "dt": 1.0,
        "base_priority": 3,
        "soc": 0.50,
        "priority_weight": 0.7,
        "soc_weight": 0.8,
        "soc_urgency_threshold": 0.65,
        "soc_reserve": 0.25,
    }
    values.update(overrides)
    return low_level_urgency_penalty(**values)


def test_default_urgency_weights_preserve_historical_reward_total():
    historical = LowLevelReward(
        time=-0.01,
        progress=0.2,
        release=-0.05,
        touchdown=25.0,
    )
    assert SimConfig().low_level_priority_time_weight == 0.0
    assert SimConfig().low_level_soc_urgency_time_weight == 0.0
    assert _penalty(priority_weight=0.0, soc_weight=0.0) == 0.0
    assert historical.total == sum(historical.as_dict()[name] for name in (
        "time", "progress", "release", "touchdown", "failure", "teacher", "shield", "urgency"
    ))
    assert historical.total == LowLevelReward(
        time=historical.time,
        progress=historical.progress,
        release=historical.release,
        touchdown=historical.touchdown,
        failure=historical.failure,
        teacher=historical.teacher,
        shield=historical.shield,
    ).total


def test_priority_urgency_is_monotone_for_135():
    values = [
        _penalty(base_priority=priority, soc=0.65, priority_weight=1.0, soc_weight=0.0)
        for priority in (1, 3, 5)
    ]
    assert values[0] == 0.0
    assert values[0] > values[1] > values[2]


def test_soc_urgency_is_monotone_below_threshold_and_zero_above():
    low_to_high = [
        _penalty(base_priority=1, soc=soc, priority_weight=0.0, soc_weight=1.0)
        for soc in (0.25, 0.40, 0.64)
    ]
    assert low_to_high[0] < low_to_high[1] < low_to_high[2] < 0.0
    assert _penalty(base_priority=1, soc=0.65, priority_weight=0.0, soc_weight=1.0) == 0.0
    assert _penalty(base_priority=1, soc=0.90, priority_weight=0.0, soc_weight=1.0) == 0.0


def test_priority_and_soc_urgency_terms_are_additive():
    priority = _penalty(soc=0.65, soc_weight=0.0)
    soc = _penalty(base_priority=1, priority_weight=0.0)
    both = _penalty()
    assert math.isclose(both, priority + soc, rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.parametrize(
    "field,value",
    [
        ("low_level_priority_time_weight", -1.0),
        ("low_level_priority_time_weight", float("nan")),
        ("low_level_soc_urgency_time_weight", float("inf")),
        ("low_level_soc_urgency_threshold", 0.25),
        ("low_level_soc_urgency_threshold", 1.01),
        ("low_level_soc_urgency_threshold", float("nan")),
    ],
)
def test_config_rejects_invalid_urgency_parameters(field, value):
    with pytest.raises(ValueError):
        SimConfig(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"soc": float("nan")},
        {"priority_weight": float("inf")},
        {"soc_urgency_threshold": 0.25},
        {"soc_reserve": 0.65, "soc_urgency_threshold": 0.65},
    ],
)
def test_urgency_function_fails_closed_for_nonfinite_or_invalid_threshold(overrides):
    with pytest.raises(ValueError):
        _penalty(**overrides)


def test_environment_reward_components_include_urgency_and_sum_to_total():
    config = SimConfig(
        total_requested_uavs=1,
        fixed_arrival_schedule=(0.0,),
        max_simulation_seconds=20.0,
        low_level_priority_time_weight=0.6,
        low_level_soc_urgency_time_weight=0.8,
        low_level_soc_urgency_threshold=0.65,
    )
    env = TerminalParallelEnv(config)
    env.reset(7)
    env.step({})  # admit the scheduled request; newly active UAVs are not retroactively rewarded
    assert env.active
    before = {
        uid: (uav.base_priority, uav.soc)
        for uid, uav in env.active.items()
    }
    _, rewards, _, _, info = env.step({})
    assert rewards
    for uid, components in info["reward_components"].items():
        expected_urgency = _penalty(
            base_time_penalty=config.low_level_time_penalty_per_second,
            dt=config.dt,
            base_priority=before[int(uid)][0],
            soc=before[int(uid)][1],
            priority_weight=config.low_level_priority_time_weight,
            soc_weight=config.low_level_soc_urgency_time_weight,
            soc_urgency_threshold=config.low_level_soc_urgency_threshold,
            soc_reserve=config.soc_reserve,
        )
        component_sum = sum(
            value for key, value in components.items() if key != "total"
        )
        assert components["urgency"] == pytest.approx(expected_urgency)
        assert components["total"] == pytest.approx(component_sum)
        assert rewards[uid] == pytest.approx(components["total"])
