"""Rule-only landing-pipeline regression for a single UAV."""

import numpy as np

from terminal_marl.config import load_experiment_config
from terminal_marl.experiments.ablation_runner import apply_demand_profile
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.core.uav import FlightPhase


def test_rule_only_single_uav_reaches_finished_phase() -> None:
    config = apply_demand_profile(load_experiment_config("dynamic_hrl"), "medium")
    config.initial_aircraft_count = 1
    config.formation_rows = 1
    config.formation_cols = 1
    config.arrival_rate = 0.0
    config.max_active_uavs = 1
    env = TerminalParallelEnv(config)
    env.reset(seed=0)

    for _ in range(260):
        env.step(
            {
                agent: np.zeros(env.action_spec(agent).shape, dtype=np.float32)
                for agent in env.agents
            }
        )
        if env.aircraft[0].finished:
            break

    assert env.aircraft[0].phase == FlightPhase.FINISHED
    assert env.metrics()["total_landed_count"] == 1.0
    assert env.safety_override_count == 0
