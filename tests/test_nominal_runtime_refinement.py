"""Commit/runtime curve certificate agreement at the 489s emergency witness."""
from dataclasses import replace
import math

import numpy as np
import pytest

from terminal_marl.configs.p3_v25_course import build_config
from terminal_marl.core.config import RingRole
from terminal_marl.core.runtime_traffic_safety import PredictiveSafetyShield
from terminal_marl.core.uav import ArrivalRequest, FlightPhase, MissionType, QueuedArrivalState, UAVState
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def witness():
    env = TerminalParallelEnv(build_config('dense24_mixed'))
    env.active.clear()
    env.now = 489.
    for uid, role, theta, speed in (
            (2, RingRole.INNER, 3.1843984263186393, 16.),
            (31, RingRole.MIDDLE, 4.2685356587065275, 20.)):
        ring = env.airspace.actual_ring(role)
        request = ArrivalRequest(uid, 0., 0, MissionType.STANDARD, 1, 1500., .8)
        queued = QueuedArrivalState(request, 1200.)
        u = UAVState.from_request(uid, request, queued,
            np.array([ring.radius*math.cos(theta), ring.radius*math.sin(theta), ring.height]), env.now)
        u.ring_role = role
        u.phase = FlightPhase.RING_INNER if role == RingRole.INNER else FlightPhase.RING_MIDDLE
        u.ring_progress = theta
        u.speed = speed
        u.velocity = np.array([-speed*math.sin(theta), speed*math.cos(theta), 0.])
        u.acceleration = np.array([-speed**2*math.cos(theta)/ring.radius,
                                   -speed**2*math.sin(theta)/ring.radius, 0.])
        u.heading = math.atan2(u.velocity[1], u.velocity[0])
        env.active[uid] = u
    u = env.active[31]
    plan = env.generator.emergency_landing(u, env.lz_manager.zones[5], env.airspace, env.now)
    u.committed_plan = plan
    u.ring_role = None
    u.phase = FlightPhase.EMERGENCY_LANDING
    return env


def test_accepted_emergency_has_same_continuous_certificate_at_runtime():
    env = witness()
    plan = env.active[31].committed_plan
    assert env.safety.filter_plan(plan, {}, WorldSnapshot(env.now, env.active, env.airspace)).approved
    shield = env.runtime_traffic
    ts, tracks = shield._joint_position_tracks(env, {}, {2, 31})
    coarse, bound = shield._assess_position_pair(tracks[2], tracks[31], times=ts,
        left_info=shield._joint_track_info[2], right_info=shield._joint_track_info[31])
    assert not coarse and 79.98 < bound < 80.
    refined = shield._joint_assessment(env, {}, pairs=((2, 31),))
    assert refined.approved and 80.35 < refined.minimum_separation < 80.38
    # A dense diagnostic is not the certificate: it confirms the bound lies
    # below the actual minimum for the same generated flight and holder.
    dense = np.arange(489., 519.0001, .005)
    holder = env.safety._holder_positions(env.active[2],
        WorldSnapshot(env.now, env.active, env.airspace), dense, 0.)
    gap = np.linalg.norm(np.asarray(holder)-plan.sample_positions(dense), axis=1).min()
    assert refined.minimum_separation < gap < 80.4


def test_refinement_retains_real_conflict_and_operational_threshold():
    env = witness()
    # Move the holder onto the candidate's current position. Physics cannot
    # proceed from an already unsafe pair, whatever sampling is requested.
    env.active[2].position = env.active[31].position.copy()
    verdict = env.runtime_traffic._joint_assessment(env, {}, pairs=((2, 31),))
    assert not verdict.approved and verdict.unsafe_pairs == ((2, 31),)
    assert verdict.minimum_separation == 0.


def test_refinement_does_not_certify_unknown_trajectory():
    env = witness()
    env.active[31].committed_plan.metadata['trajectory_schema_version'] = 'unknown'
    verdict = env.runtime_traffic._joint_assessment(env, {}, pairs=((2, 31),))
    assert not verdict.approved


def test_direct_physics_grid_matches_finite_acceleration_and_short_last_step():
    env = witness()
    env.active[2].speed = 12.
    shield = env.runtime_traffic
    ts, tracks = shield._joint_position_tracks(env, {2: -.4}, {2}, horizon=3.13, sample_dt=.2)
    expected = env.safety._holder_positions(env.active[2],
        WorldSnapshot(env.now, env.active, env.airspace), ts, -.4)
    np.testing.assert_allclose(tracks[2], expected, atol=1e-10, rtol=0)
    assert ts[-1] == pytest.approx(492.13)
    assert shield._joint_track_info[2] == {'kind': 'holder_proxy', 'resampled': False}


def test_cache_reuses_final_refined_verdict_and_only_failed_pair(monkeypatch):
    env = witness()
    shield = env.runtime_traffic
    shield._joint_pair_verdict_cache = {}
    shield._joint_pair_snapshot_refs = {}
    calls = []
    original = shield._joint_position_tracks
    def counted(*args, **kwargs):
        calls.append(kwargs.get('sample_dt'))
        return original(*args, **kwargs)
    monkeypatch.setattr(shield, '_joint_position_tracks', counted)
    first = shield._joint_assessment(env, {}, pairs=((2, 31),))
    second = shield._joint_assessment(env, {}, pairs=((2, 31),))
    assert first == second and first.approved
    assert calls == [None, .2]


def test_old_contract_does_not_silently_change_certificate():
    env = witness()
    shield = PredictiveSafetyShield(replace(env.config, nominal_holder_separation_mode='speed_tube_legacy'))
    assert not shield._joint_assessment(env, {}, pairs=((2, 31),)).approved


def test_refinement_still_rejects_between_sample_curve_excursion():
    # Point-only acceptance would miss this legal-acceleration inward bulge.
    from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
    env = witness()
    for uid, gap, velocity in ((2, 0., 0.), (31, 80.005, .15)):
        plan = TrajectoryPlan(str(uid), uid, 'transfer', (
            TrajectoryPoint(489., np.array([0., gap, 40.]), np.array([0., -velocity, 0.]), np.zeros(3)),
            TrajectoryPoint(490., np.array([0., gap, 40.]), np.array([0., velocity, 0.]), np.zeros(3)),
        ), 489., 490., metadata={'trajectory_schema_version': 'executable_trajectory_v3', 'target_phase': 'transfer'})
        env.active[uid].committed_plan = plan
    verdict = env.runtime_traffic._joint_assessment(env, {}, pairs=((2, 31),), horizon=1.)
    assert not verdict.approved and verdict.minimum_separation < 80.
