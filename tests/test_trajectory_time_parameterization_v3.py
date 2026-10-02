from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.trajectory_generator import TrajectoryPlan
from terminal_marl.core.trajectory_time_parameterization import (
    GeometricPath,
    TrajectoryConstraints,
    TrajectoryTimeParameterizer,
)
from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL


def _parameterizer(*, schema_version: str = "executable_trajectory_v3") -> TrajectoryTimeParameterizer:
    return TrajectoryTimeParameterizer(VehicleDynamics(RESEARCH_EVTOL), schema_version=schema_version)


def test_v3_straight_zero_to_zero_has_exact_boundaries_and_feasible_segments() -> None:
    path = GeometricPath.line(np.zeros(3), np.asarray((60.0, 0.0, 0.0)), samples=9)
    points = _parameterizer().parameterize(
        path,
        start_time=3.0,
        initial_velocity=np.zeros(3),
        initial_acceleration=np.zeros(3),
        target_horizontal_speed=12.0,
        constraints=TrajectoryConstraints(end_horizontal_speed=0.0),
    )

    assert points[0].t == pytest.approx(3.0)
    assert points[-1].t > points[0].t
    np.testing.assert_allclose(points[0].position, path.points[0])
    np.testing.assert_allclose(points[0].velocity, np.zeros(3))
    np.testing.assert_allclose(points[0].acceleration, np.zeros(3))
    np.testing.assert_allclose(points[-1].position, path.points[-1])
    np.testing.assert_allclose(points[-1].velocity, np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(points[-1].acceleration, np.zeros(3), atol=1e-8)

    plan = TrajectoryPlan(
        "v3-straight",
        1,
        "test",
        points,
        points[0].t,
        points[-1].t,
        metadata={"trajectory_schema_version": "executable_trajectory_v3"},
    )
    assert all(segment.bounds()["max_horizontal_accel"] <= RESEARCH_EVTOL.max_horizontal_accel + 1e-3
               for segment in plan.polynomial_segments)


def test_v3_curved_path_uses_curvature_and_end_direction_constraints() -> None:
    path = GeometricPath.cubic_bezier(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((25.0, 0.0, 0.0)),
        np.asarray((55.0, 20.0, 0.0)),
        np.asarray((85.0, 25.0, 0.0)),
        samples=25,
    )
    points = _parameterizer().parameterize(
        path,
        start_time=0.0,
        initial_velocity=np.zeros(3),
        target_horizontal_speed=10.0,
        constraints=TrajectoryConstraints(
            end_horizontal_speed=4.0,
            end_direction=np.asarray((30.0, 5.0, 0.0)),
            end_curvature_vector=np.zeros(3),
        ),
    )

    assert points[-1].velocity[0] > 0.0
    assert points[-1].velocity[1] > 0.0
    np.testing.assert_allclose(points[-1].acceleration, np.zeros(3), atol=1e-8)
    plan = TrajectoryPlan(
        "v3-curved",
        2,
        "test",
        points,
        points[0].t,
        points[-1].t,
        metadata={"trajectory_schema_version": "executable_trajectory_v3"},
    )
    assert all(
        segment.bounds()["max_horizontal_speed"] <= RESEARCH_EVTOL.max_horizontal_speed + 1e-3
        for segment in plan.polynomial_segments
    )


def test_v3_preserves_nonzero_boundary_state_and_analytic_derivative() -> None:
    path = GeometricPath.line(np.asarray((10.0, 0.0, 3.0)), np.asarray((80.0, 0.0, 3.0)), samples=7)
    initial_velocity = np.asarray((3.0, 0.0, 0.0))
    initial_acceleration = np.asarray((0.2, 0.0, 0.0))
    points = _parameterizer().parameterize(
        path,
        start_time=5.0,
        initial_velocity=initial_velocity,
        initial_acceleration=initial_acceleration,
        target_horizontal_speed=8.0,
        constraints=TrajectoryConstraints(end_horizontal_speed=3.0),
    )
    plan = TrajectoryPlan(
        "v3-boundary",
        3,
        "test",
        points,
        points[0].t,
        points[-1].t,
        metadata={"trajectory_schema_version": "executable_trajectory_v3"},
    )

    start = plan.sample(points[0].t)
    np.testing.assert_allclose(start.position, path.points[0], atol=1e-10)
    np.testing.assert_allclose(start.velocity, initial_velocity, atol=1e-10)
    np.testing.assert_allclose(start.acceleration, initial_acceleration, atol=1e-10)
    moment = (points[1].t + points[2].t) / 2.0
    epsilon = 1e-5
    before = plan.sample(moment - epsilon)
    after = plan.sample(moment + epsilon)
    current = plan.sample(moment)
    np.testing.assert_allclose((after.position - before.position) / (2.0 * epsilon), current.velocity, atol=1e-6)
    np.testing.assert_allclose((after.velocity - before.velocity) / (2.0 * epsilon), current.acceleration, atol=1e-5)


def test_v3_rejects_short_stop_when_backward_start_cap_is_infeasible() -> None:
    path = GeometricPath.line(np.zeros(3), np.asarray((0.05, 0.0, 0.0)), samples=2)
    with pytest.raises(RuntimeError, match="dynamically_infeasible_boundary_state"):
        _parameterizer().parameterize(
            path,
            start_time=0.0,
            initial_velocity=np.asarray((10.0, 0.0, 0.0)),
            target_horizontal_speed=10.0,
            constraints=TrajectoryConstraints(end_horizontal_speed=0.0),
        )


def test_v3_rejects_unqualified_tangent_yaw_request() -> None:
    path = GeometricPath.line(np.zeros(3), np.asarray((20.0, 0.0, 0.0)), samples=3)
    with pytest.raises(ValueError, match="V3 tangent_yaw is unsupported"):
        _parameterizer().parameterize(
            path,
            start_time=0.0,
            initial_velocity=np.zeros(3),
            target_horizontal_speed=5.0,
            constraints=TrajectoryConstraints(tangent_yaw=True),
        )


def test_v3_rejects_path_outside_vehicle_altitude_envelope() -> None:
    path = GeometricPath.line(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((20.0, 0.0, RESEARCH_EVTOL.max_operating_altitude_agl_m + 10.0)),
        samples=5,
    )
    with pytest.raises(RuntimeError, match="dynamically_infeasible_boundary_state"):
        _parameterizer().parameterize(
            path,
            start_time=0.0,
            initial_velocity=np.zeros(3),
            target_horizontal_speed=5.0,
            constraints=TrajectoryConstraints(end_horizontal_speed=0.0),
        )


def test_v2_default_and_explicit_schema_keep_legacy_results_identical() -> None:
    path = GeometricPath.cubic_bezier(
        np.zeros(3),
        np.asarray((30.0, 0.0, 5.0)),
        np.asarray((55.0, 15.0, 10.0)),
        np.asarray((85.0, 20.0, 15.0)),
        samples=17,
    )
    constraints = TrajectoryConstraints(end_horizontal_speed=4.0)
    arguments = dict(
        start_time=2.0,
        initial_velocity=np.zeros(3),
        target_horizontal_speed=10.0,
        constraints=constraints,
    )
    default = TrajectoryTimeParameterizer(VehicleDynamics(RESEARCH_EVTOL))
    explicit = _parameterizer(schema_version="executable_trajectory_v2")
    left = default.parameterize(path, **arguments)
    right = explicit.parameterize(path, **arguments)
    assert len(left) == len(right)
    for first, second in zip(left, right):
        assert first.t == pytest.approx(second.t, abs=1e-12)
        np.testing.assert_array_equal(first.position, second.position)
        np.testing.assert_array_equal(first.velocity, second.velocity)
        np.testing.assert_array_equal(first.acceleration, second.acceleration)
    assert default.estimate_duration(path, initial_velocity=np.zeros(3), target_horizontal_speed=10.0,
                                     constraints=constraints) == pytest.approx(
                                         explicit.estimate_duration(path, initial_velocity=np.zeros(3),
                                                                     target_horizontal_speed=10.0,
                                                                     constraints=constraints), abs=1e-12,
                                     )
