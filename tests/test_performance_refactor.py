from __future__ import annotations

import math

import numpy as np
import pytest

from terminal_marl.core.config import SimConfig
from terminal_marl.core.observations import build_low_observation_v2
from terminal_marl.core.topology_neighbors import TopologyCache
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.trajectory_time_parameterization import GeometricPath, TrajectoryConstraints, TrajectoryTimeParameterizer
from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def test_trajectory_plan_hash_is_cached_and_trajectory_arrays_are_read_only():
    plan = TrajectoryPlan("cached", 1, "test", [
        TrajectoryPoint(0.0, np.zeros(3)), TrajectoryPoint(1.0, np.ones(3)),
    ], 0.0, 1.0)
    assert plan.trajectory_hash == plan.metadata["trajectory_hash"]
    assert plan.verify_trajectory_integrity()
    with pytest.raises(ValueError):
        plan.points[0].position[0] = 1.0
    with pytest.raises(AttributeError):
        plan.points.append(TrajectoryPoint(2.0, np.ones(3)))
    with pytest.raises(AttributeError):
        plan.points = ()


def test_fast_duration_estimate_is_close_to_canonical_parameterization():
    parameterizer = TrajectoryTimeParameterizer(VehicleDynamics(RESEARCH_EVTOL))
    path = GeometricPath.cubic_bezier(np.zeros(3), np.asarray((80.0, 0.0, 20.0)),
                                     np.asarray((160.0, 80.0, 50.0)), np.asarray((220.0, 80.0, 60.0)))
    constraints = TrajectoryConstraints(end_horizontal_speed=12.0)
    estimate = parameterizer.estimate_duration(path, initial_velocity=np.zeros(3), target_horizontal_speed=18.0, constraints=constraints)
    canonical = parameterizer.parameterize(path, start_time=0.0, initial_velocity=np.zeros(3), target_horizontal_speed=18.0,
                                            constraints=constraints)[-1].t
    assert math.isfinite(estimate)
    assert abs(estimate - canonical) / canonical < 0.05


def test_topology_cache_matches_reference_v2_observation():
    config = SimConfig(reset_mode="warm_start_rings", warm_start_uavs=10, fixed_arrival_schedule=(0.0,) * 10,
                       total_requested_uavs=10, max_active_uavs=10)
    env = TerminalParallelEnv(config)
    env.reset(seed=5)
    cache = TopologyCache(env.airspace, env.active.values(), now=env.now, cpa_horizon=config.shield_horizon_seconds)
    for uav in env.active.values():
        assert np.allclose(build_low_observation_v2(uav, env), build_low_observation_v2(uav, env, topology_cache=cache))
