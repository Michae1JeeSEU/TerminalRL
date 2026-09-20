from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest

from terminal_marl.core.config import SimConfig
from terminal_marl.core.observations import build_low_observation_v2
from terminal_marl.core.topology_neighbors import TopologyCache
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.trajectory_time_parameterization import GeometricPath, TrajectoryConstraints, TrajectoryTimeParameterizer
from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from terminal_marl.core.runtime_traffic_safety import PredictiveSafetyShield, SafetyInvariantError
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.core.forecast_world import ForecastWorld
from terminal_marl.core.config import RingRole


def test_p0_qualified_runtime_defaults_are_explicit():
    config = SimConfig()
    assert config.physics_dt == pytest.approx(0.2)
    assert config.safety_sampling_dt == pytest.approx(1.0)
    assert config.trajectory_bezier_samples == 41
    assert config.trajectory_final_samples == 51


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
    try:
        shield.protect(env, {holder.uid: LowLevelIntent(0.0, 0.0, 0.0)})
    except SafetyInvariantError:
        # A deliberately unrecoverable witness is valid only when the
        # fail-closed Safety Closure contract blocks physics.
        pass
    assert shield.telemetry["shield_fail_closed"] + shield.telemetry["shield_interventions"] >= 1


@pytest.mark.parametrize("residual", (-2.0, 0.0, 2.0))
def test_joint_position_tracks_match_reference_plan_sampling(residual: float):
    config = SimConfig(
        experiment_mode="fixed_low_student",
        reset_mode="warm_start_rings",
        warm_start_uavs=2,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0, 0.0),
        total_requested_uavs=2,
    )
    env = TerminalParallelEnv(config)
    env.reset(23)
    shield = PredictiveSafetyShield(config)
    uid = min(env.active)
    holder = env.active[uid]
    end = env.now + config.shield_horizon_seconds

    samples, tracks = shield._joint_position_tracks(
        env, {uid: residual}, {uid},
    )
    reference = shield._repair_plan(holder, env.airspace, env.now, end, residual)
    expected = np.asarray([reference.sample(float(time)).position for time in samples])

    assert np.allclose(tracks[uid], expected, atol=1e-12, rtol=0.0)


def test_joint_position_pair_assessment_matches_existing_safety_arbiter():
    config = SimConfig(
        experiment_mode="fixed_low_student",
        reset_mode="warm_start_rings",
        warm_start_uavs=2,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0, 0.0),
        total_requested_uavs=2,
    )
    env = TerminalParallelEnv(config)
    env.reset(29)
    shield = PredictiveSafetyShield(config)
    pair = tuple(sorted(env.active))
    end = env.now + config.shield_horizon_seconds
    residuals = {pair[0]: -1.25, pair[1]: 1.5}
    reference_tracks = {
        uid: shield._repair_plan(env.active[uid], env.airspace, env.now, end, residuals[uid])
        for uid in pair
    }
    reference = env.safety.verify_predicted_tracks(
        reference_tracks, start=env.now, end=end,
    )
    reference_minimum = env.safety.last_predicted_track_audit["minimum_separation"]

    actual = shield._joint_assessment(env, residuals, pairs=(pair,))

    assert actual.approved is reference.approved
    assert actual.unsafe_pairs == (() if reference.approved else (pair,))
    assert actual.minimum_separation == pytest.approx(reference_minimum, abs=1e-12)


def test_vectorized_sample_feasibility_matches_authoritative_plan_validation():
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    feasible_times = np.asarray((0.0, 1.0, 2.0))
    feasible_positions = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    feasible_velocities = np.asarray(((1.0, 0.0, 0.0),) * 3)
    feasible_points = tuple(
        TrajectoryPoint(time, position, velocity, np.zeros(3, dtype=float))
        for time, position, velocity in zip(feasible_times, feasible_positions, feasible_velocities)
    )
    assert not dynamics.validate_plan(feasible_points)
    assert dynamics.samples_are_feasible(
        feasible_times, feasible_positions, feasible_velocities,
    )

    rng = np.random.default_rng(20260920)
    for _ in range(50):
        count = int(rng.integers(2, 30))
        times = np.cumsum(rng.uniform(0.05, 2.0, size=count))
        positions = np.cumsum(rng.normal(0.0, 5.0, size=(count, 3)), axis=0)
        velocities = rng.normal(0.0, 18.0, size=(count, 3))
        accelerations = np.vstack((
            np.zeros(3, dtype=float),
            np.diff(velocities, axis=0) / np.diff(times)[:, None],
        ))
        yaw_rates = rng.normal(0.0, 0.5, size=count)
        points = tuple(
            TrajectoryPoint(time, position, velocity, acceleration, 0.0, yaw_rate)
            for time, position, velocity, acceleration, yaw_rate in
            zip(times, positions, velocities, accelerations, yaw_rates)
        )

        expected = not dynamics.validate_plan(points)
        actual = dynamics.samples_are_feasible(times, positions, velocities, yaw_rates)

        assert actual is expected


@pytest.mark.parametrize("duration", (0.2, 1.13, 8.0))
def test_ring_hold_fast_path_matches_ring_motion_reference(duration: float):
    config = SimConfig()
    env = TerminalParallelEnv(config)
    env.reset(20260920)
    generator = env.generator
    ring = env.airspace.actual_ring(RingRole.OUTER)
    theta = 0.37
    speed = 11.25
    target_speed = 14.5
    start = 3.75

    actual = generator.ring_hold_points(ring, theta, speed, target_speed, start, duration)
    samples = max(2, int(math.ceil(duration / config.physics_dt)) + 1)
    step = duration / (samples - 1)
    tangent = np.asarray((-math.sin(theta), math.cos(theta), 0.0))
    radial = np.asarray((-math.cos(theta), -math.sin(theta), 0.0))
    expected = [TrajectoryPoint(
        start,
        generator._ring_position(ring, theta),
        tangent * speed,
        radial * (speed * speed / ring.radius),
    )]
    for index in range(1, samples):
        motion = generator.dynamics.ring_motion_step(ring, theta, speed, target_speed, step)
        theta, speed = motion.theta, motion.speed
        expected.append(TrajectoryPoint(
            start + index * step, motion.position, motion.velocity, motion.acceleration,
        ))

    for fast, reference in zip(actual, expected):
        assert fast.t == reference.t
        assert np.array_equal(fast.position, reference.position)
        assert np.array_equal(fast.velocity, reference.velocity)
        assert np.array_equal(fast.acceleration, reference.acceleration)


def test_l1_holder_committed_witness_checks_the_legal_speed_residual_envelope(monkeypatch):
    """Regression for L1-S seed 2003's holder--committed hard witness."""
    config = SimConfig(experiment_mode="fixed_teacher_heavy", scheduler_mode="teacher_full", teacher_enabled=True,
                       reset_mode="warm_start_rings", warm_start_uavs=1, fixed_arrival_schedule=(0.0,),
                       total_requested_uavs=1, shield_horizon_seconds=30.0)
    env = TerminalParallelEnv(config); env.reset(2003)
    holder = next(iter(env.active.values()))
    ring = env.airspace.actual_ring(holder.ring_role)
    holder.position = np.asarray((ring.radius, 0.0, ring.height)); holder.ring_progress = 0.0; holder.speed = ring.nominal_speed
    times = np.asarray((0.0, 30.0))
    nominal = env.safety._holder_positions(holder, WorldSnapshot(env.now, env.active, env.airspace), times, 0.0)
    tangent = np.asarray((0.0, 1.0, 0.0))
    # A transfer that is barely safe against nominal guidance becomes unsafe
    # when the holder legally requests +2 m/s for the 30 s shield horizon.
    plan = TrajectoryPlan("l1-s-holder-committed", 999, "transfer", [
        TrajectoryPoint(0.0, nominal[0] + tangent * 85.0),
        TrajectoryPoint(30.0, nominal[1] + tangent * 85.0),
    ], 0.0, 30.0, metadata={"trajectory_schema_version": config.trajectory_schema_version,
                              "vehicle_profile_id": config.vehicle_profile_id, "target_phase": "ring_middle"})
    monkeypatch.setattr(env.safety.dynamics, "validate_plan", lambda points: [])
    verdict = env.safety.filter_plan(plan, {}, WorldSnapshot(env.now, env.active, env.airspace))
    assert not verdict.approved
    assert verdict.reason in {"predicted_holder_speed_envelope", "predicted_holder_speed_envelope_swept",
                              "predicted_holder_reachable_tube"}


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


def test_selected_trajectory_densities_meet_tracking_tolerance_and_next_lower_do_not():
    from terminal_marl.training.performance_qualification import trajectory_sampling_convergence
    report = trajectory_sampling_convergence()
    bezier = {row["samples"]: row for row in report["bezier"]}
    final = {row["samples"]: row for row in report["final"]}
    assert bezier[41]["dense_reference_tracking_error"] < 0.05
    assert bezier[33]["dense_reference_tracking_error"] > 0.05
    assert final[51]["dense_reference_tracking_error"] < 0.05
    assert final[41]["dense_reference_tracking_error"] > 0.05
    assert all(row["dynamics_feasible"] and row["safety_filter_approved"] for rows in report.values() for row in rows)


def test_forecast_world_matches_single_aircraft_reference_replay_and_release():
    """Freeze the former clone-step semantics without using it in production."""
    config = SimConfig(
        experiment_mode="fixed_teacher_heavy", scheduler_mode="teacher_full", teacher_enabled=True,
        reset_mode="warm_start_rings", warm_start_uavs=1,
        fixed_arrival_schedule=(0.0,), total_requested_uavs=1,
    )
    env = TerminalParallelEnv(config)
    env.reset(19)
    subject = next(iter(env.active.values()))
    requested = LowLevelIntent(1.0, 0.0, 0.0)
    candidate = next(row for row in env.teacher_scheduler.candidates(requested) if row.delay_seconds == 5.0)

    reference = deepcopy(env)
    for _ in range(int(round(candidate.delay_seconds / config.dt))):
        reference.step({str(subject.uid): np.asarray((0.0, candidate.intent.speed_residual, 0.0), dtype=np.float32)})

    world = ForecastWorld(env)
    world.advance(subject.uid, candidate.intent.speed_residual, candidate.delay_seconds)
    projected = world.active[subject.uid]
    replayed = reference.active[subject.uid]
    assert world.now == pytest.approx(reference.now)
    assert projected.phase is replayed.phase
    assert projected.ring_role is replayed.ring_role
    assert projected.speed == pytest.approx(replayed.speed, abs=1e-10)
    assert projected.ring_progress == pytest.approx(replayed.ring_progress, abs=1e-10)
    assert np.allclose(projected.position, replayed.position, atol=1e-10)
    assert np.allclose(projected.velocity, replayed.velocity, atol=1e-10)

    proposal, plan, reason = world.candidate_plan(subject.uid, candidate, RingRole.MIDDLE, lz_id=None)
    replayed.safety_retry_after = min(replayed.safety_retry_after, reference.now)
    decision = reference._release_legacy(replayed, candidate.intent)
    assert (reason is None and plan is not None and proposal is not None) == (decision == "approved")
    if decision == "approved":
        reservation = reference.scheduler.reservations[replayed.merge_reservation_id]
        assert proposal.merge_time == pytest.approx(reservation.merge_time, abs=1e-10)
        assert proposal.merge_angle == pytest.approx(reservation.merge_angle, abs=1e-10)
