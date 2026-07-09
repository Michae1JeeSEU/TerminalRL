import numpy as np

from terminal_marl import ExperimentMode, SimConfig, TerminalParallelEnv
from terminal_marl.policies import DynamicRuleHighLevelPolicy


def test_congestion_expands_outer_ring_and_speed() -> None:
    env = TerminalParallelEnv(
        SimConfig(safe_distance=1.0),
        experiment_mode=ExperimentMode.DYNAMIC_RULE,
    )
    env.reset(seed=1)
    outer = env.airspace.outer_ring
    env.vtac.queues[outer.index].extend([item.uid for item in env.aircraft[:5]])
    action = DynamicRuleHighLevelPolicy().act(env)
    assert action.shape == (8,)
    assert action[0] > 0 and action[4] > 0
    old = env.airspace.outer_ring.radius
    env.apply_high_level_action(action)
    assert env.airspace.outer_ring.radius > old


def test_policy_actions_are_bounded() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.DYNAMIC_RULE)
    env.reset(seed=2)
    action = DynamicRuleHighLevelPolicy().act(env)
    assert np.all((-1 <= action) & (action <= 1))
