"""Tests for the trainable high-level Gym-style wrapper."""

import numpy as np
import pytest

from terminal_marl import ExperimentMode, SimConfig, TerminalParallelEnv
from terminal_marl.envs.high_level_wrapper import HighLevelHRLWrapper


def make_wrapper(interval: int = 3) -> HighLevelHRLWrapper:
    env = TerminalParallelEnv(
        SimConfig(high_level_interval_steps=interval, max_steps=20),
        experiment_mode=ExperimentMode.DYNAMIC_HRL,
    )
    return HighLevelHRLWrapper(env, low_level_policy="rule-only", seed=1)


def test_wrapper_exposes_gym_style_spaces_and_reset() -> None:
    wrapper = make_wrapper()
    observation, info = wrapper.reset(seed=2)
    assert wrapper.action_space.shape == (8,)
    assert wrapper.observation_space.shape == observation.shape
    assert observation.shape == (14,)
    assert info["experiment_mode"] == "dynamic_hrl"


def test_one_high_level_step_executes_complete_low_level_interval() -> None:
    wrapper = make_wrapper(interval=4)
    wrapper.reset(seed=3)
    _obs, reward, terminated, truncated, info = wrapper.step(np.zeros(8))
    assert wrapper.env.step_count == 4
    assert info["low_level_steps"] == 4
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert "reward_diagnostics" in info


def test_wrapper_supports_random_low_level_placeholder() -> None:
    env = TerminalParallelEnv(
        SimConfig(high_level_interval_steps=2),
        experiment_mode=ExperimentMode.DYNAMIC_HRL,
    )
    wrapper = HighLevelHRLWrapper(env, low_level_policy="random-policy", seed=4)
    wrapper.reset(seed=4)
    result = wrapper.step(np.ones(8))
    assert len(result) == 5


def test_wrapper_rejects_fixed_mode() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.FIXED_RULE)
    with pytest.raises(ValueError, match="dynamic HRL"):
        HighLevelHRLWrapper(env)
