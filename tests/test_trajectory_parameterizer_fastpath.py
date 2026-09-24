from __future__ import annotations

import math

import numpy as np
import pytest

from terminal_marl.core.trajectory_time_parameterization import (
    GeometricPath,
    TrajectoryConstraints,
    TrajectoryTimeParameterizer,
)
from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL


def _reference_parameterize(
    parameterizer: TrajectoryTimeParameterizer,
    path: GeometricPath,
    *,
    start_time: float,
    initial_velocity: np.ndarray,
    target_horizontal_speed: float,
    constraints: TrajectoryConstraints,
):
    """Test-local copy of the pre-fast-path scalar computations."""
    from terminal_marl.core.trajectory_generator import TrajectoryPoint

    points = path.points
    tangents = parameterizer._tangents(points)
    segments = path.segment_lengths
    if np.any(segments <= 1e-9):
        raise ValueError("invalid_time_parameterization")
    curvature = parameterizer._curvature_xy(points)
    horizontal_factor = np.linalg.norm(tangents[:, :2], axis=1)
    vertical_factor = np.abs(tangents[:, 2])
    profile = parameterizer.dynamics.profile
    limits = np.full(len(points), np.inf, dtype=float)
    limits = np.minimum(limits, profile.max_horizontal_speed / np.maximum(horizontal_factor, 1e-9))
    limits = np.minimum(limits, profile.max_climb_rate / np.maximum(vertical_factor, 1e-9))
    limits = np.minimum(limits, profile.max_descent_rate / np.maximum(vertical_factor, 1e-9))
    for index, value in enumerate(curvature):
        horizontal_limit = math.sqrt(.8 * profile.max_lateral_accel / max(value, 1e-12))
        if constraints.tangent_yaw:
            horizontal_limit = min(
                horizontal_limit,
                parameterizer.dynamics.max_feasible_speed_for_curvature(value, tangent_yaw=True),
            )
        limits[index] = min(limits[index], horizontal_limit / max(horizontal_factor[index], 1e-9))
    requested_path_speed = max(0.1, target_horizontal_speed / max(horizontal_factor.mean(), 1e-9))
    limits = np.minimum(limits, requested_path_speed)
    speeds = limits.copy()
    start_speed = float(np.linalg.norm(np.asarray(initial_velocity, dtype=float)))
    speeds[0] = min(speeds[0], start_speed)
    for index, distance in enumerate(segments):
        speeds[index + 1] = min(
            speeds[index + 1],
            math.sqrt(max(0.0, speeds[index] ** 2 + 2.0 * (.57 * profile.max_horizontal_accel) * distance)),
        )
    if constraints.end_horizontal_speed is not None:
        speeds[-1] = min(speeds[-1], constraints.end_horizontal_speed / max(horizontal_factor[-1], 1e-9))
    for index in range(len(segments) - 1, -1, -1):
        speeds[index] = min(
            speeds[index],
            math.sqrt(max(0.0, speeds[index + 1] ** 2 + 2.0 * (.57 * profile.max_horizontal_decel) * segments[index])),
        )
    yaw = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0])) if constraints.tangent_yaw else np.full(len(points), np.nan)
    for _ in range(24):
        times = [float(start_time)]
        for distance, first, second in zip(segments, speeds, speeds[1:]):
            times.append(times[-1] + 2.0 * float(distance) / max(first + second, 1e-6))
        velocities = tangents * speeds[:, None]
        velocities[0] = np.asarray(initial_velocity, dtype=float)
        accelerations = np.zeros_like(velocities)
        for index in range(1, len(points)):
            accelerations[index] = (velocities[index] - velocities[index - 1]) / (times[index] - times[index - 1])
        if len(points) > 1:
            accelerations[0] = accelerations[1]
        yaw_rates = np.zeros(len(points))
        if constraints.tangent_yaw:
            yaw_rates[1:] = np.diff(yaw) / np.diff(times)
            yaw_rates[0] = yaw_rates[1]
        if parameterizer.dynamics.samples_are_feasible(
            np.asarray(times, dtype=float), points, velocities,
            None if not constraints.tangent_yaw else yaw_rates,
        ):
            return [
                TrajectoryPoint(
                    float(time), point, velocity, acceleration,
                    None if not constraints.tangent_yaw else float(angle),
                    None if not constraints.tangent_yaw else float(rate),
                )
                for time, point, velocity, acceleration, angle, rate in
                zip(times, points, velocities, accelerations, yaw, yaw_rates)
            ]
        speeds *= .88
    raise RuntimeError("could not time-parameterize a dynamically feasible trajectory")


@pytest.fixture(params=(False, True), ids=("position_yaw", "tangent_yaw"))
def tangent_yaw(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


@pytest.fixture(
    params=(
        GeometricPath.line(np.zeros(3), np.asarray((150.0, 0.0, 60.0)), samples=41),
        GeometricPath.cubic_bezier(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((80.0, 0.0, 20.0)),
            np.asarray((160.0, 80.0, 50.0)),
            np.asarray((220.0, 80.0, 60.0)),
            samples=51,
        ),
    ),
    ids=("line", "bezier"),
)
def feasible_path(request: pytest.FixtureRequest) -> GeometricPath:
    return request.param


def test_parameterize_fastpath_matches_original_scalar_reference(
    feasible_path: GeometricPath,
    tangent_yaw: bool,
):
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    parameterizer = TrajectoryTimeParameterizer(dynamics)
    constraints = TrajectoryConstraints(end_horizontal_speed=8.0, tangent_yaw=tangent_yaw)
    initial_velocity = np.zeros(3, dtype=float)

    actual = parameterizer.parameterize(
        feasible_path,
        start_time=2.0,
        initial_velocity=initial_velocity,
        target_horizontal_speed=14.0,
        constraints=constraints,
    )
    expected = _reference_parameterize(
        parameterizer,
        feasible_path,
        start_time=2.0,
        initial_velocity=initial_velocity,
        target_horizontal_speed=14.0,
        constraints=constraints,
    )

    assert len(actual) == len(expected)
    for actual_point, expected_point in zip(actual, expected):
        assert actual_point.t == pytest.approx(expected_point.t, rel=1e-12, abs=1e-12)
        assert np.array_equal(actual_point.position, expected_point.position)
        assert np.allclose(actual_point.velocity, expected_point.velocity, rtol=1e-12, atol=1e-12)
        assert np.allclose(actual_point.acceleration, expected_point.acceleration, rtol=1e-12, atol=1e-12)
        assert actual_point.yaw == pytest.approx(expected_point.yaw, rel=1e-12, abs=1e-12)
        assert actual_point.yaw_rate == pytest.approx(expected_point.yaw_rate, rel=1e-12, abs=1e-12)


def test_fastpath_output_passes_authoritative_dynamics_validation(feasible_path: GeometricPath, tangent_yaw: bool):
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    points = TrajectoryTimeParameterizer(dynamics).parameterize(
        feasible_path,
        start_time=0.0,
        initial_velocity=np.zeros(3, dtype=float),
        target_horizontal_speed=14.0,
        constraints=TrajectoryConstraints(end_horizontal_speed=8.0, tangent_yaw=tangent_yaw),
    )
    assert not dynamics.validate_plan(points)
    times = np.asarray([point.t for point in points], dtype=float)
    positions = np.asarray([point.position for point in points], dtype=float)
    velocities = np.asarray([point.velocity for point in points], dtype=float)
    yaw_rates = None if not tangent_yaw else np.asarray([point.yaw_rate for point in points], dtype=float)
    assert dynamics.samples_are_feasible(times, positions, velocities, yaw_rates)
