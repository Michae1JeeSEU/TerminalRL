"""Rule-only multi-UAV phase progression should not deadlock on the outer ring."""

import numpy as np

from terminal_marl.config import load_experiment_config
from terminal_marl.experiments.ablation_runner import apply_demand_profile
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.core.uav import FlightPhase


def test_rule_only_six_uavs_progress_through_inner_final_and_land() -> None:
    config = apply_demand_profile(load_experiment_config("dynamic_hrl"), "medium")
    config.arrival_rate = 0.0
    config.max_active_uavs = config.initial_aircraft_count
    env = TerminalParallelEnv(config)
    env.reset(seed=0)

    saw_inner = False
    saw_final = False
    for _ in range(650):
        env.step(
            {
                agent: np.zeros(env.action_spec(agent).shape, dtype=np.float32)
                for agent in env.agents
            }
        )
        saw_inner = saw_inner or any(
            aircraft.assigned_ring == env.vtac.inner_ring_index for aircraft in env.aircraft
        )
        saw_final = saw_final or any(
            aircraft.phase == FlightPhase.FINAL for aircraft in env.aircraft
        )
        if all(aircraft.finished for aircraft in env.aircraft):
            break

    assert saw_inner
    assert saw_final
    assert env.metrics()["total_landed_count"] == float(config.initial_aircraft_count)
