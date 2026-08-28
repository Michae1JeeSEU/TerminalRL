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
from terminal_marl.core.runtime_traffic_safety import PredictiveSafetyShield
from terminal_marl.core.merge_scheduler import LowLevelIntent


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


@pytest.mark.parametrize("sampling_dt", (.2, .5, 1.0))
def test_safety_sampling_witness_keeps_shield_trigger(sampling_dt: float):
    config = SimConfig(experiment_mode="fixed_low_student", reset_mode="warm_start_rings", warm_start_uavs=1,
                       fixed_arrival_schedule=(0.0,), total_requested_uavs=1, safety_sampling_dt=sampling_dt)
    env = TerminalParallelEnv(config)
    env.reset(7)
    holder = next(iter(env.active.values()))
    ring = env.airspace.actual_ring(holder.ring_role)
    holder.position = np.asarray((ring.radius, 0.0, ring.height)); holder.ring_progress = 0.0
    shield = PredictiveSafetyShield(config)
    endpoint = shield._holder_position(holder, env.airspace, 10.0, 0.0)
    plan = TrajectoryPlan("sampling-witness", 999, "transfer", [
        TrajectoryPoint(0.0, np.asarray((ring.radius * 2.0, 0.0, ring.height))),
        TrajectoryPoint(10.0, endpoint),
    ], 0.0, 10.0, metadata={"committed": True})
    env.active[999] = type("CommittedTraffic", (), {"uid": 999, "ring_role": None, "committed_plan": plan})()
    assert shield.assess(env).trigger_reason == "predicted_swept_separation"
    shield.protect(env, {holder.uid: LowLevelIntent(0.0, 0.0, 0.0)})
    assert shield.telemetry["shield_fail_closed"] + shield.telemetry["shield_interventions"] >= 1


@pytest.mark.parametrize("samples", (81, 41, 33))
def test_bezier_sampling_convergence_preserves_feasible_duration(samples: int):
    parameterizer = TrajectoryTimeParameterizer(VehicleDynamics(RESEARCH_EVTOL))
    path = GeometricPath.cubic_bezier(np.zeros(3), np.asarray((80., 0., 20.)),
                                     np.asarray((160., 80., 50.)), np.asarray((220., 80., 60.)), samples=samples)
    plan = parameterizer.parameterize(path, start_time=0., initial_velocity=np.zeros(3), target_horizontal_speed=18.)
    assert not VehicleDynamics(RESEARCH_EVTOL).validate_plan(plan)
    assert plan[-1].t > 0.0


def test_bezier_41_sample_production_choice_matches_81_sample_reference():
    parameterizer = TrajectoryTimeParameterizer(VehicleDynamics(RESEARCH_EVTOL))
    durations = []
    for samples in (81, 41):
        path = GeometricPath.cubic_bezier(np.zeros(3), np.asarray((80., 0., 20.)),
                                         np.asarray((160., 80., 50.)), np.asarray((220., 80., 60.)), samples=samples)
        durations.append(parameterizer.parameterize(path, start_time=0., initial_velocity=np.zeros(3), target_horizontal_speed=18.)[-1].t)
    assert abs(durations[1] - durations[0]) / durations[0] < .01


@pytest.mark.parametrize("samples", (101, 51, 41))
def test_final_sampling_convergence_preserves_feasible_duration(samples: int):
    parameterizer = TrajectoryTimeParameterizer(VehicleDynamics(RESEARCH_EVTOL))
    path = GeometricPath.cubic_bezier(np.asarray((200., 0., 40.)), np.asarray((170., 90., 35.)),
                                     np.asarray((60., 70., 8.)), np.zeros(3), samples=samples)
    plan = parameterizer.parameterize(path, start_time=0., initial_velocity=np.zeros(3), target_horizontal_speed=10.,
                                      constraints=TrajectoryConstraints(end_horizontal_speed=0.0))
    assert not VehicleDynamics(RESEARCH_EVTOL).validate_plan(plan)
    assert plan[-1].t > 0.0


def test_final_51_sample_production_choice_matches_101_sample_reference():
    parameterizer = TrajectoryTimeParameterizer(VehicleDynamics(RESEARCH_EVTOL))
    durations = []
    for samples in (101, 51):
        path = GeometricPath.cubic_bezier(np.asarray((200., 0., 40.)), np.asarray((170., 90., 35.)),
                                         np.asarray((60., 70., 8.)), np.zeros(3), samples=samples)
        durations.append(parameterizer.parameterize(path, start_time=0., initial_velocity=np.zeros(3), target_horizontal_speed=10.,
                                                     constraints=TrajectoryConstraints(end_horizontal_speed=0.0))[-1].t)
    assert abs(durations[1] - durations[0]) / durations[0] < .01
