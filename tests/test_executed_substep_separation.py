"""Separation is measured on executed substeps, not a policy-tick shortcut."""
from dataclasses import replace
from types import SimpleNamespace
import copy
import math

import numpy as np
import pytest

from terminal_marl.configs.p3_v25_course import build_config
from terminal_marl.core.runtime_separation import RuntimeSeparationMonitor
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def observe(paths):
    states = {uid: SimpleNamespace(position=np.asarray(rows[-1][1], float)) for uid, rows in paths.items()}
    previous = {uid: np.asarray(rows[0][1], float) for uid, rows in paths.items()}
    monitor = RuntimeSeparationMonitor(build_config('dense24_fixed'))
    encounters = monitor.observe(states, previous, executed_tracks=paths)
    return monitor, encounters


def test_circle_shortcut_is_not_an_executed_buffer_entry():
    times = np.linspace(0., 1., 6)
    phase = 2*np.arcsin(80.02/400.)
    paths = {uid: [(t, np.array([200*np.cos(.08*t+offset), 200*np.sin(.08*t+offset), 40.]))
                   for t in times] for uid, offset in ((1, 0.), (2, phase))}
    states = {uid: SimpleNamespace(position=rows[-1][1]) for uid, rows in paths.items()}
    previous = {uid: rows[0][1] for uid, rows in paths.items()}
    old = RuntimeSeparationMonitor(build_config('dense24_fixed'))
    old.observe(states, previous)
    assert old.telemetry['operational_buffer_entries'] == 1
    monitor, encounters = observe(paths)
    assert not encounters and monitor.telemetry['operational_buffer_entries'] == 0
    assert 80. < monitor.telemetry['minimum_swept_separation'] < 80.02


def test_substep_collision_is_not_hidden_by_safe_tick_endpoints():
    paths = {1: [(0., [0.,0.,40.]), (.5,[0.,0.,40.]), (1.,[0.,0.,40.])],
             2: [(0., [100.,0.,40.]), (.5,[0.,0.,40.]), (1.,[100.,0.,40.])]}
    monitor, encounters = observe(paths)
    assert monitor.telemetry['collisions'] == 1
    assert monitor.telemetry['minimum_actual_separation'] == 0.
    assert encounters[0].swept_fraction == pytest.approx(.5)


def test_crossing_between_substep_endpoints_is_still_detected():
    paths = {1: [(0., [0.,0.,40.]), (.2,[0.,0.,40.])],
             2: [(0., [100.,0.,40.]), (.2,[-100.,0.,40.])]}
    monitor, encounters = observe(paths)
    assert monitor.telemetry['minimum_actual_separation'] == 100.
    assert monitor.telemetry['minimum_swept_separation'] == 0.
    assert encounters[0].collision_entry and encounters[0].swept_fraction == pytest.approx(.1)


def test_touchdown_track_only_protects_shared_airborne_interval():
    paths = {1: [(0., [0.,0.,0.]), (.3,[0.,0.,0.])],
             2: [(0., [200.,0.,0.]), (.2,[160.,0.,0.]), (1.,[0.,0.,0.])]}
    monitor, encounters = observe(paths)
    assert not encounters and monitor.telemetry['minimum_swept_separation'] == pytest.approx(140.)
    # A conflict before touchdown is still counted, despite the first plane
    # no longer belonging to env.active at the end of the policy tick.
    paths[2] = [(0., [100.,0.,0.]), (.2,[0.,0.,0.]), (1.,[200.,0.,0.])]
    monitor, encounters = observe(paths)
    assert monitor.telemetry['collisions'] == 1 and encounters


def test_invalid_or_incomplete_actual_track_cannot_silently_fallback():
    state = {1: SimpleNamespace(position=np.array([1.,0.,0.]))}
    previous = {1: np.zeros(3)}
    monitor = RuntimeSeparationMonitor(build_config('dense24_fixed'))
    with pytest.raises(KeyError):
        monitor.observe(state, previous, executed_tracks={})
    with pytest.raises(ValueError, match='endpoint differs'):
        monitor.observe(state, previous, executed_tracks={1:[(0.,[0.,0.,0.]),(1.,[2.,0.,0.])]})
    with pytest.raises(ValueError, match='instantaneous'):
        monitor.observe(state, previous, executed_tracks={1:[(0.,[0.,0.,0.]),(0.,[1.,0.,0.])]})


def test_v3_step_supplies_actual_physics_knots_not_prediction(monkeypatch):
    config = replace(build_config('dense24_mixed'), total_requested_uavs=3,
                     warm_start_uavs=3, fixed_arrival_schedule=(0.,)*3)
    env = TerminalParallelEnv(config, seed=792)
    before = {uid: u.position.copy() for uid, u in env.active.items()}
    original = env.separation_monitor.observe
    recorded = {}
    def capture(states, previous, *, executed_tracks=None):
        recorded.update(executed_tracks)
        assert all(np.array_equal(previous[uid], before[uid]) for uid in states)
        return original(states, previous, executed_tracks=executed_tracks)
    monkeypatch.setattr(env.separation_monitor, 'observe', capture)
    monkeypatch.setattr(env, '_next_state_prediction', lambda intents: {uid:np.zeros(3) for uid in env.active})
    env.step({str(uid):np.zeros(3) for uid in env.active})
    assert set(recorded) == set(before)
    for uid, samples in recorded.items():
        assert len(samples) == 6
        np.testing.assert_allclose([t for t, _ in samples], np.arange(6)*.2, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(samples[-1][1], env.active[uid].position)
        assert not np.array_equal(samples[-1][1], np.zeros(3))


def test_executor_records_handoff_remainder_without_changing_motion():
    from tests.test_nominal_runtime_refinement import witness
    from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
    from terminal_marl.core.uav import FlightPhase
    env = witness(); u = env.active[2]
    u.phase = FlightPhase.TRANSFER_M2I
    ring = env.airspace.actual_ring(u.ring_role)
    end = env.now+.37
    theta = u.ring_progress + u.speed*.37/ring.radius
    point = TrajectoryPoint(end, np.array([ring.radius*math.cos(theta),ring.radius*math.sin(theta),ring.height]),
        np.array([-u.speed*math.sin(theta),u.speed*math.cos(theta),0.]),
        np.array([-u.speed**2*math.cos(theta)/ring.radius,-u.speed**2*math.sin(theta)/ring.radius,0.]))
    plan = TrajectoryPlan('handoff', u.uid, 'transfer', (
        TrajectoryPoint(env.now,u.position.copy(),u.velocity.copy(),u.acceleration.copy()),point),
        env.now,end,metadata={'trajectory_schema_version':'executable_trajectory_v3','target_phase':'ring_inner'})
    u.committed_plan=plan
    env.safety.checked_trajectory_hashes[plan.plan_id]=plan.trajectory_hash
    reference=copy.deepcopy(env)
    samples=[(env.now,u.position.copy())]
    actual=env._advance_plan(u,position_samples=samples)
    expected=reference._advance_plan(reference.active[2])
    assert actual==expected and u.committed_plan is None
    np.testing.assert_array_equal(u.position,reference.active[2].position)
    np.testing.assert_allclose([t-env.now for t,_ in samples],[0.,.2,.37,.57,.77,.97,1.],atol=1e-12,rtol=0)


def test_executor_touchdown_record_ends_at_actual_plan_endpoint():
    from tests.test_nominal_runtime_refinement import witness
    env=witness();u=env.active[31];plan=u.committed_plan
    env.now=plan.end_time-.13
    point=plan.sample(env.now)
    u.position=point.position.copy();u.velocity=point.velocity.copy()
    u.assigned_lz=5
    assert env.lz_manager.reserve(5,u.uid,env.now).approved
    env.safety.checked_trajectory_hashes[plan.plan_id]=plan.trajectory_hash
    samples=[(env.now,u.position.copy())]
    _,touchdown=env._advance_plan(u,position_samples=samples)
    assert touchdown and u.uid not in env.active
    assert samples[-1][0]==plan.end_time < env.now+env.config.dt
    np.testing.assert_array_equal(samples[-1][1],u.position)
