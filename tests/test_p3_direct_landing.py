"""Direct landing and near-zero ring-segment regression contracts."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.configs.p3_integrated_repair import build_config
from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.trajectory_generator import TrajectoryGenerator, TrajectoryPlan


def fixture(role=RingRole.INNER):
    config = build_config(requests=1, warm_start=0, high_soc=True)
    airspace = DynamicAirspace(config)
    ring = airspace.actual_ring(role)
    speed = ring.nominal_speed
    uav = SimpleNamespace(uid=1, position=np.array([ring.radius, 0., ring.height]),
        velocity=np.array([0., speed, 0.]), acceleration=np.array([-speed**2/ring.radius, 0., 0.]),
        speed=speed, ring_progress=0., ring_role=role, committed_plan=None)
    pad = SimpleNamespace(lz_id=0, position=np.array([0., -40., 0.]))
    return config, airspace, TrajectoryGenerator(config), uav, pad


def final(generator, airspace, uav, pad, now=600., delay=0., residual=0.):
    reservation = SimpleNamespace(reservation_id='test', merge_time=now+delay)
    return generator.final_approach(uav, pad, reservation, now,
        speed_residual_mps=residual, airspace=airspace)


def assert_boundary_and_touchdown(generator, uav, pad, plan):
    assert generator.validate_kinematics(plan) == []
    for name in ('position', 'velocity', 'acceleration'):
        np.testing.assert_allclose(getattr(plan.points[0], name), getattr(uav, name), atol=1e-9)
    np.testing.assert_allclose(plan.points[-1].position, pad.position, atol=1e-8)
    np.testing.assert_allclose(plan.points[-1].velocity, 0., atol=1e-8)
    np.testing.assert_allclose(plan.points[-1].acceleration, 0., atol=1e-8)
    for left, right in zip(plan.polynomial_segments, plan.polynomial_segments[1:]):
        for a, b in zip(left.sample(left.end_time), right.sample(right.start_time)):
            np.testing.assert_allclose(a, b, atol=1e-6)


def test_direct_entry_preserves_live_motion_and_has_no_braking_prerequisite():
    _, airspace, generator, uav, pad = fixture()
    plan = final(generator, airspace, uav, pad)
    assert_boundary_and_touchdown(generator, uav, pad, plan)
    assert plan.metadata['v3_landing_path_mode'] == 'direct_c2'
    assert plan.metadata['required_braking_duration_seconds'] == 0.
    assert plan.metadata['accepted_reservation_departure'] == 600.
    assert plan.metadata['actual_entry_speed_mps'] == 16.
    staging = plan.points[1]
    np.testing.assert_allclose(staging.position, [*pad.position[:2], uav.position[2]])
    np.testing.assert_allclose(staging.velocity, 0.)
    np.testing.assert_allclose(staging.acceleration, 0.)
    # The first segment already leaves the circle while moving continuously.
    assert np.linalg.norm(plan.sample_position(610.)[:2]) < np.linalg.norm(uav.position[:2])-1.


def test_delayed_direct_departure_holds_nominal_speed_until_reserved_slot():
    _, airspace, generator, uav, pad = fixture()
    plan = final(generator, airspace, uav, pad, delay=6.)
    assert_boundary_and_touchdown(generator, uav, pad, plan)
    assert plan.metadata['accepted_reservation_departure'] == 606.
    for t in (600., 602., 604., 606.):
        point = plan.sample(t)
        assert np.linalg.norm(point.velocity[:2]) == pytest.approx(uav.speed, abs=1e-6)
        assert np.linalg.norm(point.position[:2]) == pytest.approx(200., abs=1e-6)


def test_direct_speed_proposal_changes_duration_with_physical_limits_retained():
    _, airspace, generator, uav, pad = fixture()
    plans = [final(generator, airspace, uav, pad, residual=r) for r in (-2., 0., 2.)]
    assert len({p.trajectory_hash for p in plans}) == 3
    requested = [p.metadata['requested_entry_duration_seconds'] for p in plans]
    assert requested[0] > requested[1] > requested[2]
    for plan in plans:
        assert_boundary_and_touchdown(generator, uav, pad, plan)
        assert plan.metadata['actual_entry_duration_seconds'] >= plan.metadata['requested_entry_duration_seconds']


@pytest.mark.parametrize('role', list(RingRole))
def test_emergency_uses_same_continuous_entry_with_safe_vertical_leg(role):
    _, airspace, generator, uav, pad = fixture(role)
    plan = generator.emergency_landing(uav, pad, airspace, 850.)
    assert_boundary_and_touchdown(generator, uav, pad, plan)
    assert plan.metadata['v3_landing_path_mode'] == 'direct_c2'
    assert plan.metadata['scheduled_departure'] == 850.


def test_direct_mode_is_explicit_and_cannot_be_used_as_v2():
    assert SimConfig().v3_landing_path_mode == 'brake_then_curve'
    with pytest.raises(ValueError, match='requires V3'):
        replace(build_config(requests=1, warm_start=0), trajectory_schema_version='executable_trajectory_v2')


@pytest.mark.parametrize('offset', [-4e-13, 0., 4e-13])
def test_near_equal_ramp_end_does_not_create_ill_conditioned_segment(offset):
    _, airspace, generator, uav, _ = fixture()
    ring = airspace.actual_ring(uav.ring_role)
    ramp = generator.dynamics.ring_transition_duration(16., 10., ring.radius)
    points = generator._v3_ring_hold_points_for_ramp(ring, 0., 16., 10., 600., ramp+offset,
        initial_acceleration=uav.acceleration, ramp_duration=ramp)
    plan = TrajectoryPlan('ramp-regression', 1, 'ring_hold', tuple(points), points[0].t, points[-1].t,
        metadata={'trajectory_schema_version':'executable_trajectory_v3'})
    assert min(s.duration for s in plan.polynomial_segments) > .01
    assert generator.validate_kinematics(plan) == []


def test_six_recorded_transfer_boundaries_no_longer_have_segment_mismatch():
    from terminal_marl.scripts.diagnose_p3_inner_exit import probe_transfer_boundaries
    rows = probe_transfer_boundaries()
    assert len(rows) == 6
    for row in rows:
        assert row['dynamics_errors'] == []
        assert row['endpoint_differences'] == []
        assert row['minimum_segment_duration'] > .01
