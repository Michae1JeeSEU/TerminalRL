"""Regression test for safe outer-to-inner release once VTAC eligibility holds."""

import numpy as np

from terminal_marl.config import load_experiment_config
from terminal_marl.experiments.ablation_runner import apply_demand_profile
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _single_uav_env() -> TerminalParallelEnv:
    config = apply_demand_profile(load_experiment_config("dynamic_hrl"), "medium")
    config.initial_aircraft_count = 1
    config.formation_rows = 1
    config.formation_cols = 1
    config.arrival_rate = 0.0
    config.max_active_uavs = 1
    return TerminalParallelEnv(config)


def test_outer_to_inner_is_approved_after_vtac_eligibility() -> None:
    env = _single_uav_env()
    env.reset(seed=0)

    approved = None
    for _ in range(90):
        env.step(
            {
                agent: np.zeros(env.action_spec(agent).shape, dtype=np.float32)
                for agent in env.agents
            }
        )
        approval = env.last_release_approval.get(1)
        if approval and approval.get("target_stage") == "outer_to_inner":
            if approval["approved"]:
                approved = approval
                break

    assert approved is not None
    kinematics = approved["diagnostics"]["trajectory"]["diagnostics"]["kinematic_detail"]
    assert kinematics["max_speed_observed"] <= kinematics["speed_max_allowed"]
    assert (
        kinematics["max_acceleration_observed"]
        <= kinematics["acceleration_max_allowed"]
    )
    assert not approved["violations"]


def test_pre_eligibility_hold_is_not_counted_as_safety_override() -> None:
    env = _single_uav_env()
    env.reset(seed=0)

    for _ in range(20):
        env.step(
            {
                agent: np.zeros(env.action_spec(agent).shape, dtype=np.float32)
                for agent in env.agents
            }
        )

    approval = env.last_release_approval[1]
    assert approval["reason"] == "vtac_not_eligible"
    assert env.safety_override_count == 0
    assert env.raw_safety_override_count == 0
