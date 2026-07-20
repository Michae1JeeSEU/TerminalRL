import numpy as np

from terminal_marl import ExperimentMode, SimConfig, TerminalParallelEnv
from terminal_marl.policies import DynamicRuleHighLevelPolicy


def test_congestion_prefers_throughput_actions_over_ring_expansion() -> None:
    env = TerminalParallelEnv(
        SimConfig(safe_distance=1.0),
        experiment_mode=ExperimentMode.DYNAMIC_RULE,
    )
    env.reset(seed=1)
    outer = env.airspace.outer_ring
    env.vtac.queues[outer.index].extend([item.uid for item in env.aircraft])
    action = DynamicRuleHighLevelPolicy().act(env)
    assert action.shape == (8,)
    assert action[4] > 0
    assert action[6] < 0
    assert abs(action[0]) <= 0.1
    old = env.airspace.outer_ring.radius
    old_speed = env.airspace.outer_ring.speed
    env.apply_high_level_action(action)
    assert env.airspace.outer_ring.radius <= old + 3.0
    assert env.airspace.outer_ring.speed > old_speed


def test_policy_actions_are_bounded() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.DYNAMIC_RULE)
    env.reset(seed=2)
    action = DynamicRuleHighLevelPolicy().act(env)
    assert np.all((-1 <= action) & (action <= 1))


def test_dynamic_rule_no_arrival_smoke_reaches_landing_pipeline() -> None:
    config = SimConfig(
        initial_aircraft_count=6,
        formation_rows=3,
        formation_cols=2,
        max_active_uavs=6,
        arrival_rate=0.0,
    )
    env = TerminalParallelEnv(config, experiment_mode=ExperimentMode.DYNAMIC_RULE)
    env.reset(seed=0)
    policy = DynamicRuleHighLevelPolicy()
    low_level_actions = {agent: np.zeros(4, dtype=np.float32) for agent in env.agents}

    for _ in range(600):
        if env.should_update_high_level():
            env.apply_high_level_action(policy.act(env))
        _, _, _, _, _ = env.step(low_level_actions)
        low_level_actions = {
            agent: np.zeros(4, dtype=np.float32) for agent in env.agents
        }
        if env.metrics()["landed_count"] >= 5:
            break

    metrics = env.metrics()
    assert metrics["landed_count"] >= 5.0
    assert metrics["queue_0"] + metrics["queue_1"] + metrics["queue_2"] < 6
    assert policy.last_info["normalized_conflict_rate"] >= 0.0
