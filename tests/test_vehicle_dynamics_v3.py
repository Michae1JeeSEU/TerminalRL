"""Focused validation tests for the executable V3 trajectory contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from terminal_marl.core.polynomial_trajectory import QuinticSegment
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL


V3 = {"trajectory_schema_version": "executable_trajectory_v3"}


def _two_point_plan(*, duration: float, position: np.ndarray, velocity: np.ndarray | None = None,
                    acceleration: np.ndarray | None = None) -> TrajectoryPlan:
    zero = np.zeros(3, dtype=float)
    initial_velocity = zero if velocity is None else np.asarray(velocity, dtype=float)
    initial_acceleration = zero if acceleration is None else np.asarray(acceleration, dtype=float)
    points = (
        TrajectoryPoint(0.0, zero, initial_velocity, initial_acceleration),
        TrajectoryPoint(duration, np.asarray(position, dtype=float), initial_velocity, zero),
    )
    return TrajectoryPlan("v3-dynamics", 0, "test", points, 0.0, duration, metadata=dict(V3))


def test_v3_rejects_interior_acceleration_peak_that_node_validator_misses() -> None:
    plan = _two_point_plan(duration=1.0, position=np.asarray((1.0, 0.0, 0.0)))
    dynamics = VehicleDynamics(RESEARCH_EVTOL)

    assert dynamics.validate_plan(plan.points) == []
    assert "horizontal_acceleration_limit" in dynamics.validate_trajectory(plan)


def test_v3_accepts_valid_quintic_and_checks_initial_acceleration_boundary() -> None:
    plan = _two_point_plan(
        duration=4.0,
        position=np.asarray((2.0, 0.0, 0.0)),
        velocity=np.asarray((0.5, 0.0, 0.0)),
        acceleration=np.asarray((0.2, 0.0, 0.0)),
    )
    dynamics = VehicleDynamics(RESEARCH_EVTOL)

    assert dynamics.validate_trajectory(plan) == []
    np.testing.assert_allclose(plan.polynomial_segments[0].sample(0.0)[2], plan.points[0].acceleration, atol=1e-12)


def test_v3_rejects_segment_boundary_mismatch() -> None:
    points = (
        TrajectoryPoint(0.0, np.zeros(3), np.zeros(3), np.zeros(3)),
        TrajectoryPoint(1.0, np.asarray((1.0, 0.0, 0.0)), np.zeros(3), np.zeros(3)),
    )
    plan = SimpleNamespace(
        metadata=dict(V3),
        points=points,
        start_time=0.0,
        end_time=1.0,
        polynomial_segments=(QuinticSegment(0.0, 1.0, np.zeros((6, 3))),),
    )

    errors = VehicleDynamics(RESEARCH_EVTOL).validate_trajectory(plan)
    assert "trajectory_segment_mismatch" in errors


def test_v2_validation_delegates_exactly_to_existing_point_validator(monkeypatch) -> None:
    plan = TrajectoryPlan(
        "v2-dynamics",
        0,
        "test",
        (
            TrajectoryPoint(0.0, np.zeros(3), np.zeros(3), np.zeros(3)),
            TrajectoryPoint(1.0, np.ones(3), np.zeros(3), np.zeros(3)),
        ),
        0.0,
        1.0,
    )
    dynamics = VehicleDynamics(RESEARCH_EVTOL)

    def sentinel(points):
        assert points is plan.points
        return ["sentinel"]

    monkeypatch.setattr(dynamics, "validate_plan", sentinel)
    assert dynamics.validate_trajectory(plan) == ["sentinel"]


def test_v3_rejects_nonfinite_bounds_and_unknown_schema_closed() -> None:
    class NonfiniteBoundsSegment:
        start_time = 0.0
        end_time = 1.0

        def bounds(self):
            return {
                "max_horizontal_speed": np.inf,
                "max_climb_rate": 0.0,
                "max_descent_rate": 0.0,
                "max_horizontal_accel": 0.0,
                "max_vertical_accel_up": 0.0,
                "max_vertical_accel_down": 0.0,
            }

        def sample(self, _time):
            zero = np.zeros(3)
            return zero, zero, zero

    points = (
        TrajectoryPoint(0.0, np.zeros(3), np.zeros(3), np.zeros(3)),
        TrajectoryPoint(1.0, np.zeros(3), np.zeros(3), np.zeros(3)),
    )
    nonfinite = SimpleNamespace(
        metadata=dict(V3),
        points=points,
        start_time=0.0,
        end_time=1.0,
        polynomial_segments=(NonfiniteBoundsSegment(),),
    )
    unknown = SimpleNamespace(metadata={"trajectory_schema_version": "future"}, points=points)
    dynamics = VehicleDynamics(RESEARCH_EVTOL)

    assert "trajectory_bounds_nonfinite" in dynamics.validate_trajectory(nonfinite)
    assert dynamics.validate_trajectory(unknown) == ["trajectory_schema_mismatch"]


def test_v3_checks_yaw_rate_limits_and_finite_yaw_fields() -> None:
    points = (
        TrajectoryPoint(0.0, np.zeros(3), np.zeros(3), np.zeros(3), yaw=0.0, yaw_rate=1.0),
        TrajectoryPoint(1.0, np.zeros(3), np.zeros(3), np.zeros(3), yaw=0.0, yaw_rate=1.0),
    )
    plan = TrajectoryPlan("yaw-rate", 0, "test", points, 0.0, 1.0, metadata=dict(V3))
    assert "yaw_rate_limit" in VehicleDynamics(RESEARCH_EVTOL).validate_trajectory(plan)

    malformed_points = (
        TrajectoryPoint(0.0, np.zeros(3), np.zeros(3), np.zeros(3), yaw=np.nan),
        TrajectoryPoint(1.0, np.zeros(3), np.zeros(3), np.zeros(3)),
    )
    malformed = SimpleNamespace(
        metadata=dict(V3),
        points=malformed_points,
        start_time=0.0,
        end_time=1.0,
        polynomial_segments=(QuinticSegment(0.0, 1.0, np.zeros((6, 3))),),
    )
    assert VehicleDynamics(RESEARCH_EVTOL).validate_trajectory(malformed) == ["invalid_time_parameterization"]


def test_v3_rejects_analytic_below_ground_overshoot_between_nodes() -> None:
    coefficients = np.zeros((6, 3), dtype=float)
    coefficients[0, 2] = 0.5
    coefficients[1, 2] = -3.0
    coefficients[2, 2] = 3.0
    points = (
        TrajectoryPoint(0.0, np.asarray((0.0, 0.0, 0.5)), np.asarray((0.0, 0.0, -1.0)), np.asarray((0.0, 0.0, 2.0 / 3.0))),
        TrajectoryPoint(3.0, np.asarray((0.0, 0.0, 0.5)), np.asarray((0.0, 0.0, 1.0)), np.asarray((0.0, 0.0, 2.0 / 3.0))),
    )
    plan = SimpleNamespace(
        metadata=dict(V3),
        points=points,
            start_time=0.0,
            end_time=3.0,
            polynomial_segments=(QuinticSegment(0.0, 3.0, coefficients),),
    )

    assert "altitude_limit" in VehicleDynamics(RESEARCH_EVTOL).validate_trajectory(plan)


def test_continuous_curve_limits_preserve_v2_default_and_bound_braking_step() -> None:
    radius = 240.0
    current = 16.0
    target = 14.0
    dt = 0.2
    default = VehicleDynamics(RESEARCH_EVTOL)
    explicit_v2 = VehicleDynamics(RESEARCH_EVTOL, continuous_curve_limits=False)
    v3 = VehicleDynamics(RESEARCH_EVTOL, continuous_curve_limits=True)

    default_speed = default.limited_tangential_speed(current, target, radius, dt)
    assert explicit_v2.limited_tangential_speed(current, target, radius, dt) == default_speed
    v3_speed = v3.limited_tangential_speed(current, target, radius, dt)
    assert v3_speed > default_speed
    tangential = (v3_speed - current) / dt
    centripetal = max(current, v3_speed) ** 2 / radius
    assert np.hypot(tangential, centripetal) <= RESEARCH_EVTOL.max_lateral_accel + 1e-9
