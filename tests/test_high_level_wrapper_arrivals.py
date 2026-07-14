import numpy as np

from terminal_marl.config import load_experiment_config
from terminal_marl.envs.high_level_wrapper import HighLevelHRLWrapper
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.experiments import apply_demand_profile


def test_dynamic_hrl_medium_has_traffic_within_high_level_steps() -> None:
    config = apply_demand_profile(load_experiment_config("dynamic_hrl"), "medium")
    wrapper = HighLevelHRLWrapper(TerminalParallelEnv(config), low_level_policy="rule-only")
    wrapper.reset(seed=0)
    total_spawned_count = len(wrapper.env.aircraft)
    active_count_max = sum(not aircraft.finished for aircraft in wrapper.env.aircraft)
    for _ in range(8):
        _obs, _reward, terminated, truncated, info = wrapper.step(
            np.zeros(8, dtype=np.float32)
        )
        diagnostics = info["reward_diagnostics"]
        total_spawned_count = max(total_spawned_count, diagnostics["total_spawned_count"])
        active_count_max = max(active_count_max, diagnostics["active_count_max"])
        if terminated or truncated:
            break
    assert total_spawned_count > 0 or active_count_max > 0

