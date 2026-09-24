"""Unit checks for the canonical finite-time flow trajectory candidates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.flow_trajectories import FlowTrajectoryGenerator
from terminal_marl.core.landing_zone import LandingZoneManager
from terminal_marl.core.trajectory_generator import TrajectoryGenerator
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    UAVState,
)


def _components(role: RingRole = RingRole.OUTER):
    config = SimConfig()
    airspace = DynamicAirspace(config)
    base = TrajectoryGenerator(config)
    flow = FlowTrajectoryGenerator(config, base)
    request = ArrivalRequest(1, 0.0, 0, MissionType.STANDARD, 1, config.energy_capacity, 1.0)
    queued = QueuedArrivalState(request, request.energy_capacity)
    uav = UAVState.from_request(1, request, queued, np.zeros(3), 0.0)
    uav.ring_role = role
    uav.phase = {
        RingRole.OUTER: FlightPhase.RING_OUTER,
        RingRole.MIDDLE: FlightPhase.RING_MIDDLE,
        RingRole.INNER: FlightPhase.RING_INNER,
    }[role]
    ring = airspace.actual_ring(role)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    uav.acceleration = np.asarray((-(ring.nominal_speed**2) / ring.radius, 0.0, 0.0), dtype=float)
    uav.ring_progress = 0.0
    return config, airspace, base, flow, uav


@pytest.mark.parametrize(
    ("source", "target", "duration", "expected_success"),
    [
        (RingRole.OUTER, RingRole.MIDDLE, 40.0, False),
        (RingRole.OUTER, RingRole.MIDDLE, 50.0, True),
        (RingRole.MIDDLE, RingRole.INNER, 40.0, False),
        (RingRole.MIDDLE, RingRole.INNER, 50.0, True),
    ],
)
def test_ring_transfer_is_finite_forward_and_dynamics_valid(source, target, duration, expected_success):
    _, airspace, _, flow, uav = _components(source)
    target_ring = airspace.actual_ring(target)
    if not expected_success:
        with pytest.raises(RuntimeError, match="dynamics validation"):
            flow.ring_transfer(uav, target, airspace, 12.0, duration)
        return
    plan = flow.ring_transfer(uav, target, airspace, 12.0, duration)

    assert plan.mode == "ring_transfer"
    assert plan.reservation_ids == ()
    assert plan.start_time == pytest.approx(12.0)
    assert plan.end_time == pytest.approx(12.0 + duration)
    assert plan.metadata["flow_geometry"] == "quintic_annular_v1"
    assert plan.metadata["scheduled_departure"] == pytest.approx(12.0)
    assert plan.metadata["merge_time"] == pytest.approx(plan.end_time)
    assert plan.points[0].t == pytest.approx(12.0)
    np.testing.assert_allclose(plan.points[0].position, uav.position, atol=1e-12)
    np.testing.assert_allclose(plan.points[0].velocity, uav.velocity, atol=1e-12)
    endpoint_theta = math.atan2(plan.points[-1].position[1], plan.points[-1].position[0])
    endpoint_tangent = np.asarray((-math.sin(endpoint_theta), math.cos(endpoint_theta), 0.0))
    assert np.linalg.norm(plan.points[-1].velocity[:2]) == pytest.approx(target_ring.nominal_speed)
    assert np.dot(plan.points[-1].velocity, endpoint_tangent) == pytest.approx(target_ring.nominal_speed)
    assert np.linalg.norm(plan.points[-1].position[:2]) == pytest.approx(target_ring.radius, abs=1e-8)
    assert plan.points[-1].position[2] == pytest.approx(target_ring.height, abs=1e-8)
    assert not flow.dynamics.validate_plan(plan.points)

    xy = np.asarray([point.position[:2] for point in plan.points])
    radii = np.linalg.norm(xy, axis=1)
    assert np.all(np.diff(radii) <= 1e-7)
    cross = xy[:-1, 0] * xy[1:, 1] - xy[:-1, 1] * xy[1:, 0]
    assert np.all(cross >= -1e-7)
    assert np.all(np.diff([point.t for point in plan.points]) <= 0.21)

    positions = np.asarray([point.position for point in plan.points])
    times = np.asarray([point.t for point in plan.points])
    declared_velocity = np.asarray([point.velocity for point in plan.points])
    centered_velocity = (positions[2:] - positions[:-2]) / (times[2:, None] - times[:-2, None])
    assert np.max(np.linalg.norm(centered_velocity - declared_velocity[1:-1], axis=1)) < 0.02
    # The first nonzero planar displacement must follow the positive tangent,
    # which catches an accidental inward/outward basis reversal.
    first_step = positions[1, :2] - positions[0, :2]
    assert positions[0, 0] * first_step[1] - positions[0, 1] * first_step[0] > 0.0


def test_ring_transfer_zero_bias_matches_default_sampled_profile():
    _, airspace, _, flow, uav = _components(RingRole.MIDDLE)
    target_ring = airspace.actual_ring(RingRole.INNER)

    default = _point_signature(flow._ring_transfer_points(uav, target_ring, 84.0))
    explicit_zero = _point_signature(
        flow._ring_transfer_points(
            uav,
            target_ring,
            84.0,
            tangential_speed_bias_mps=0.0,
        )
    )
    _assert_point_signatures_equal(default, explicit_zero)

    plan = flow.ring_transfer(
        uav,
        RingRole.INNER,
        airspace,
        12.0,
        84.0,
        tangential_speed_bias_mps=0.0,
    )
    assert plan.metadata["flow_tangential_speed_bias_mps"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("duration", "bias"),
    ((60.0, 0.0), (72.0, -4.0), (84.0, -8.0)),
)
def test_ring_transfer_preview_matches_real_plan_endpoint(duration, bias):
    _, airspace, _, flow, uav = _components(RingRole.MIDDLE)
    context = flow._ring_transfer_context(
        uav,
        RingRole.INNER,
        airspace,
        12.0,
        duration,
        bias,
    )
    (
        prepared_duration,
        prepared_bias,
        source,
        target,
        target_ring,
        start,
    ) = context
    prepared = flow._prepare_ring_transfer(
        uav, target_ring, prepared_duration, prepared_bias,
    )
    preview = flow._ring_transfer_preview(
        uav,
        source,
        target,
        start,
        prepared_duration,
        prepared_bias,
        prepared,
    )
    real = flow.ring_transfer(
        uav,
        RingRole.INNER,
        airspace,
        12.0,
        duration,
        tangential_speed_bias_mps=bias,
    )

    assert preview.end_time == pytest.approx(real.end_time)
    assert preview.metadata["target_phase"] == real.metadata["target_phase"]
    assert not hasattr(preview, "trajectory_hash")
    np.testing.assert_array_equal(
        preview.points[-1].position, real.points[-1].position,
    )
    np.testing.assert_array_equal(
        preview.points[-1].velocity, real.points[-1].velocity,
    )
    np.testing.assert_array_equal(
        preview.sample_position(preview.end_time),
        real.sample_position(real.end_time),
    )


def test_prepared_ring_transfer_materializes_same_points_hash_and_dynamics():
    _, airspace, _, flow, uav = _components(RingRole.MIDDLE)
    context = flow._ring_transfer_context(
        uav, RingRole.INNER, airspace, 12.0, 84.0, -4.0,
    )
    duration, bias, source, target, target_ring, start = context
    prepared = flow._prepare_ring_transfer(uav, target_ring, duration, bias)
    materialized = flow._materialize_ring_transfer(
        uav, source, target, start, duration, bias, prepared,
    )
    public = flow.ring_transfer(
        uav,
        RingRole.INNER,
        airspace,
        12.0,
        84.0,
        tangential_speed_bias_mps=-4.0,
    )

    _assert_point_signatures_equal(
        _point_signature(public.points),
        _point_signature(materialized.points),
    )
    assert materialized.trajectory_hash == public.trajectory_hash
    assert flow.dynamics.validate_plan(materialized.points) == []


@pytest.mark.parametrize("bias", [-4.0, -8.0])
def test_negative_ring_transfer_bias_has_valid_duration_and_signed_angle_shift(bias):
    _, airspace, _, flow, uav = _components(RingRole.MIDDLE)
    target_ring = airspace.actual_ring(RingRole.INNER)
    valid = []

    for duration in (72.0, 84.0, 96.0):
        try:
            plan = flow.ring_transfer(
                uav,
                RingRole.INNER,
                airspace,
                12.0,
                duration,
                tangential_speed_bias_mps=bias,
            )
        except RuntimeError as exc:
            assert "dynamics validation" in str(exc)
            continue

        valid.append(duration)
        assert not flow.dynamics.validate_plan(plan.points)
        assert plan.metadata["flow_tangential_speed_bias_mps"] == pytest.approx(bias)
        np.testing.assert_allclose(plan.points[0].position, uav.position, atol=1e-12)
        np.testing.assert_allclose(plan.points[0].velocity, uav.velocity, atol=1e-12)
        np.testing.assert_allclose(plan.points[0].acceleration, uav.acceleration, atol=1e-12)
        np.testing.assert_allclose(
            plan.points[-1].position[:2],
            target_ring.radius * plan.points[-1].position[:2] /
            np.linalg.norm(plan.points[-1].position[:2]),
            atol=1e-8,
        )
        assert np.linalg.norm(plan.points[-1].position[:2]) == pytest.approx(
            target_ring.radius, abs=1e-8,
        )
        assert plan.points[-1].position[2] == pytest.approx(target_ring.height, abs=1e-8)
        assert np.linalg.norm(plan.points[-1].velocity[:2]) == pytest.approx(
            target_ring.nominal_speed,
        )
        assert not np.allclose(plan.points[-1].acceleration, 0.0)

        baseline = flow._ring_transfer_points(uav, target_ring, duration, 0.0)
        baseline_angles = np.unwrap(
            np.arctan2(
                [point.position[1] for point in baseline],
                [point.position[0] for point in baseline],
            )
        )
        biased_angles = np.unwrap(
            np.arctan2(
                [point.position[1] for point in plan.points],
                [point.position[0] for point in plan.points],
            )
        )
        assert (biased_angles[-1] - biased_angles[0]) - (
            baseline_angles[-1] - baseline_angles[0]
        ) < -1e-4

    assert valid, f"no dynamics-valid duration for bias {bias}"


@pytest.mark.parametrize("bias", [math.nan, math.inf, -math.inf])
def test_nonfinite_tangential_speed_bias_is_rejected(bias):
    _, airspace, _, flow, uav = _components(RingRole.MIDDLE)
    with pytest.raises(ValueError, match="tangential_speed_bias_mps"):
        flow.ring_transfer(
            uav,
            RingRole.INNER,
            airspace,
            12.0,
            84.0,
            tangential_speed_bias_mps=bias,
        )


@pytest.mark.parametrize("duration", [25.0, 35.0, 45.0])
@pytest.mark.parametrize("emergency", [False, True])
def test_final_approach_has_vertical_fixed_xy_leg(duration, emergency):
    _, airspace, _, flow, uav = _components(RingRole.INNER)
    lz = LandingZoneManager().zones[0]
    with pytest.raises(RuntimeError, match="dynamics validation"):
        flow.final_approach(uav, lz, airspace, 20.0, duration, emergency=emergency)


@pytest.mark.parametrize("emergency", [False, True])
def test_final_approach_success_keeps_vertical_fixed_xy_leg(emergency):
    _, airspace, _, flow, uav = _components(RingRole.INNER)
    lz = LandingZoneManager().zones[0]
    plan = flow.final_approach(uav, lz, airspace, 20.0, 50.0, emergency=emergency)

    assert plan.mode == ("emergency_landing" if emergency else "final_approach")
    assert plan.metadata["target_phase"] == ("emergency_landing" if emergency else "landing")
    assert plan.metadata["scheduled_departure"] == pytest.approx(20.0)
    assert plan.reservation_ids == ()
    assert not flow.dynamics.validate_plan(plan.points)
    np.testing.assert_allclose(plan.points[0].position, uav.position, atol=1e-12)
    np.testing.assert_allclose(plan.points[0].velocity, uav.velocity, atol=1e-12)
    np.testing.assert_allclose(plan.points[-1].position, lz.position, atol=1e-8)

    staging = np.asarray(plan.metadata["staging_point"], dtype=float)
    first = next(
        index for index, point in enumerate(plan.points)
        if point.t == pytest.approx(plan.start_time + 50.0)
    )
    assert np.allclose(plan.points[first].position, staging, atol=1e-8)
    vertical = np.asarray([point.position for point in plan.points[first:]])
    assert np.allclose(vertical[:, :2], lz.position[:2], atol=2e-4)
    assert np.all(np.diff(vertical[:, 2]) <= 1e-8)


@pytest.mark.parametrize("duration", [0.0, -1.0, math.inf, math.nan])
def test_invalid_duration_is_rejected(duration):
    _, airspace, _, flow, uav = _components(RingRole.OUTER)
    with pytest.raises(ValueError, match="duration"):
        flow.ring_transfer(uav, RingRole.MIDDLE, airspace, 0.0, duration)


def test_missing_ring_and_committed_plan_are_rejected():
    _, airspace, base, flow, uav = _components(RingRole.OUTER)
    uav.ring_role = None
    with pytest.raises(ValueError, match="ring"):
        flow.ring_transfer(uav, RingRole.MIDDLE, airspace, 0.0, 40.0)
    _, airspace, base, flow, uav = _components(RingRole.OUTER)
    uav.committed_plan = object()
    with pytest.raises(ValueError, match="uncommitted"):
        flow.final_approach(uav, LandingZoneManager().zones[0], airspace, 0.0, 35.0)


def test_normal_final_approach_is_inner_ring_only():
    _, airspace, _, flow, uav = _components(RingRole.OUTER)
    with pytest.raises(ValueError, match="inner-ring"):
        flow.final_approach(uav, LandingZoneManager().zones[0], airspace, 0.0, 60.0)


def _point_signature(points):
    return [
        (
            float(point.t),
            np.asarray(point.position, dtype=float).copy(),
            np.asarray(point.velocity, dtype=float).copy(),
            np.asarray(point.acceleration, dtype=float).copy(),
        )
        for point in points
    ]


def _assert_point_signatures_equal(expected, actual):
    assert len(expected) == len(actual)
    for expected_row, actual_row in zip(expected, actual):
        assert actual_row[0] == pytest.approx(expected_row[0], abs=0.0)
        np.testing.assert_array_equal(actual_row[1], expected_row[1])
        np.testing.assert_array_equal(actual_row[2], expected_row[2])
        np.testing.assert_array_equal(actual_row[3], expected_row[3])


def test_flow_sample_prescreen_does_not_change_valid_helper_output(monkeypatch):
    _, airspace, _, flow, uav = _components(RingRole.OUTER)
    target_ring = airspace.actual_ring(RingRole.MIDDLE)
    expected_ring = _point_signature(flow._ring_transfer_points(uav, target_ring, 50.0))

    _, airspace_disabled, _, flow_disabled, uav_disabled = _components(RingRole.OUTER)
    target_ring_disabled = airspace_disabled.actual_ring(RingRole.MIDDLE)
    monkeypatch.setattr(
        flow_disabled.dynamics,
        "samples_are_feasible",
        lambda *args, **kwargs: True,
    )
    actual_ring = _point_signature(
        flow_disabled._ring_transfer_points(uav_disabled, target_ring_disabled, 50.0)
    )
    _assert_point_signatures_equal(expected_ring, actual_ring)

    _, airspace, _, flow, uav = _components(RingRole.INNER)
    lz = LandingZoneManager().zones[0]
    staging = np.asarray((lz.position[0], lz.position[1], uav.position[2]), dtype=float)
    expected_departure = _point_signature(flow._inward_departure_points(uav, staging, 50.0))

    _, _, _, flow_disabled, uav_disabled = _components(RingRole.INNER)
    monkeypatch.setattr(
        flow_disabled.dynamics,
        "samples_are_feasible",
        lambda *args, **kwargs: True,
    )
    actual_departure = _point_signature(
        flow_disabled._inward_departure_points(uav_disabled, staging, 50.0)
    )
    _assert_point_signatures_equal(expected_departure, actual_departure)


def test_flow_sample_prescreen_remains_fail_closed(monkeypatch):
    _, airspace, _, flow, uav = _components(RingRole.OUTER)
    monkeypatch.setattr(
        flow.dynamics,
        "samples_are_feasible",
        lambda *args, **kwargs: False,
    )
    with pytest.raises(RuntimeError, match="flow ring transfer failed dynamics validation"):
        flow._ring_transfer_points(uav, airspace.actual_ring(RingRole.MIDDLE), 50.0)

    _, _, _, flow, uav = _components(RingRole.INNER)
    lz = LandingZoneManager().zones[0]
    staging = np.asarray((lz.position[0], lz.position[1], uav.position[2]), dtype=float)
    monkeypatch.setattr(
        flow.dynamics,
        "samples_are_feasible",
        lambda *args, **kwargs: False,
    )
    with pytest.raises(RuntimeError, match="flow final inward departure failed dynamics validation"):
        flow._inward_departure_points(uav, staging, 50.0)
