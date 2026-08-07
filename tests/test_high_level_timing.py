"""Tests for the explicit high-level HRL update clock."""

import numpy as np

from terminal_marl import ExperimentMode, SimConfig, TerminalParallelEnv


def make_env(interval: int = 3) -> TerminalParallelEnv:
    config = SimConfig(high_level_interval_steps=interval)
    env = TerminalParallelEnv(config, experiment_mode=ExperimentMode.DYNAMIC_HRL)
    env.reset(seed=2)
    return env


def zero_actions(env: TerminalParallelEnv) -> dict[str, np.ndarray]:
    return {
        agent: np.zeros(4, dtype=np.float32)
        for agent in env.agents
    }


def test_should_update_only_at_high_level_boundaries() -> None:
    env = make_env(interval=3)
    assert env.should_update_high_level()
    env.step(zero_actions(env))
    assert not env.should_update_high_level()
    env.step(zero_actions(env))
    assert not env.should_update_high_level()
    env.step(zero_actions(env))
    assert env.should_update_high_level()


def test_action_between_boundaries_is_cached_then_applied() -> None:
    env = make_env(interval=3)
    env.step(zero_actions(env))
    old_state = env.airspace.state_vector()
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    info = env.apply_high_level_action(action)
    assert not info["applied"]
    assert info["cached"]
    assert np.array_equal(env.airspace.state_vector(), old_state)

    env.step(zero_actions(env))
    env.step(zero_actions(env))
    env.step(zero_actions(env))

    assert env.last_high_level_info["applied"]
    assert env.airspace.outer_ring.radius > old_state[0]
    assert env.pending_high_level_action is None


def test_high_level_observation_is_finite_and_stable() -> None:
    env = make_env(interval=4)
    observation = env.get_high_level_observation()

    assert observation.shape == (44,)
    assert np.all(np.isfinite(observation))
    assert np.all((-1.0 <= observation[:8]) & (observation[:8] <= 1.0))


def test_fixed_mode_never_opens_high_level_clock() -> None:
    env = TerminalParallelEnv(
        SimConfig(high_level_interval_steps=2),
        experiment_mode=ExperimentMode.FIXED_RULE,
    )
    env.reset(seed=3)
    state = env.airspace.state_vector()

    assert not env.should_update_high_level()
    info = env.apply_high_level_action(np.ones(8))
    assert info["applied"]
    assert "fixed_airspace_override" in info["violations"]
    assert np.array_equal(env.airspace.state_vector(), state)
