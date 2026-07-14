"""Repeated low-level HOLDs should not over-count high-level safety overrides."""

import numpy as np

from terminal_marl.config import load_experiment_config
from terminal_marl.envs.high_level_wrapper import HighLevelHRLWrapper
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def test_safety_override_count_is_deduplicated_with_raw_count_preserved() -> None:
    config = load_experiment_config("dynamic_hrl")
    config.initial_aircraft_count = 2
    config.formation_rows = 1
    config.formation_cols = 2
    config.col_spacing = 1.0
    config.arrival_rate = 0.0
    config.max_active_uavs = 2
    wrapper = HighLevelHRLWrapper(TerminalParallelEnv(config), seed=0)
    wrapper.reset(seed=0)

    _obs, _reward, _terminated, _truncated, info = wrapper.step(
        np.zeros(wrapper.action_space.shape, dtype=np.float32)
    )
    diagnostics = info["reward_diagnostics"]

    assert diagnostics["raw_safety_override_delta"] >= diagnostics["safety_override_delta"]
    assert diagnostics["total_raw_safety_override_count"] >= diagnostics["total_safety_override_count"]
