from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.trajectory_time_parameterization import (
    GeometricPath,
    TrajectoryConstraints,
    TrajectoryTimeParameterizer,
)
from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL


def _scalar_curvature_xy(points: np.ndarray) -> np.ndarray:
    values = np.zeros(len(points), dtype=float)
    for index in range(1, len(points) - 1):
        left, middle, right = points[index - 1:index + 2, :2]
        a = np.linalg.norm(middle - left)
        b = np.linalg.norm(right - middle)
        c = np.linalg.norm(right - left)
        first, second = middle - left, right - left
        area2 = abs(first[0] * second[1] - first[1] * second[0])
        values[index] = 0.0 if min(a, b, c) <= 1e-12 else 2.0 * area2 / (a * b * c)
    values[0] = values[1]
    values[-1] = values[-2]
    return values


@pytest.mark.parametrize(
    "points",
    (
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        np.column_stack((np.linspace(0.0, 80.0, 17), np.zeros(17), np.linspace(0.0, 12.0, 17))),
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0 + 5e-13, 0.0, 0.0), (2.0, 1.0, 0.0))),
        GeometricPath.cubic_bezier(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((35.0, 60.0, 10.0)),
            np.asarray((100.0, -20.0, 24.0)),
            np.asarray((150.0, 40.0, 30.0)),
            samples=81,
        ).points,
    ),
)
def test_vectorized_curvature_matches_scalar_reference(points: np.ndarray):
    actual = TrajectoryTimeParameterizer._curvature_xy(points)
    expected = _scalar_curvature_xy(points)
    assert np.array_equal(actual, expected)
    assert actual[0] == actual[1]
    assert actual[-1] == actual[-2]


def test_vectorized_curvature_matches_scalar_reference_on_random_paths():
    rng = np.random.default_rng(20260924)
    for _ in range(200):
        count = int(rng.integers(2, 90))
        points = np.cumsum(rng.normal(0.0, 10.0, size=(count, 3)), axis=0)
        if count >= 3 and rng.random() < 0.5:
            index = int(rng.integers(1, count - 1))
            points[index] = points[index - 1] + rng.normal(0.0, 5e-13, size=3)
        actual = TrajectoryTimeParameterizer._curvature_xy(points)
        expected = _scalar_curvature_xy(points)
        assert np.array_equal(actual, expected)


class _ScalarReferenceParameterizer(TrajectoryTimeParameterizer):
    @staticmethod
    def _curvature_xy(points: np.ndarray) -> np.ndarray:
        return _scalar_curvature_xy(points)


@pytest.mark.parametrize(
    "path",
    (
        GeometricPath.cubic_bezier(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((80.0, 0.0, 20.0)),
            np.asarray((160.0, 80.0, 50.0)),
            np.asarray((220.0, 80.0, 60.0)),
            samples=81,
        ),
        GeometricPath.cubic_bezier(
            np.asarray((200.0, 0.0, 40.0)),
            np.asarray((170.0, 90.0, 35.0)),
            np.asarray((60.0, 70.0, 8.0)),
            np.zeros(3),
            samples=51,
        ),
    ),
)
def test_parameterizer_outputs_match_scalar_curvature_reference(path: GeometricPath):
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    vectorized = TrajectoryTimeParameterizer(dynamics)
    scalar = _ScalarReferenceParameterizer(dynamics)
    constraints = TrajectoryConstraints(end_horizontal_speed=12.0, tangent_yaw=True)
    initial_velocity = np.asarray((0.0, 0.0, 0.0))

    assert vectorized.entry_speed_limit(path) == pytest.approx(scalar.entry_speed_limit(path), rel=1e-13, abs=1e-13)
    assert vectorized.estimate_duration(
        path, initial_velocity=initial_velocity, target_horizontal_speed=18.0, constraints=constraints,
    ) == pytest.approx(scalar.estimate_duration(
        path, initial_velocity=initial_velocity, target_horizontal_speed=18.0, constraints=constraints,
    ), rel=1e-13, abs=1e-13)

    actual = vectorized.parameterize(
        path, start_time=2.0, initial_velocity=initial_velocity,
        target_horizontal_speed=18.0, constraints=constraints,
    )
    expected = scalar.parameterize(
        path, start_time=2.0, initial_velocity=initial_velocity,
        target_horizontal_speed=18.0, constraints=constraints,
    )
    assert len(actual) == len(expected)
    for actual_point, expected_point in zip(actual, expected):
        assert actual_point.t == pytest.approx(expected_point.t, rel=1e-13, abs=1e-13)
        assert np.array_equal(actual_point.position, expected_point.position)
        assert np.allclose(actual_point.velocity, expected_point.velocity, rtol=1e-13, atol=1e-13)
        assert np.allclose(actual_point.acceleration, expected_point.acceleration, rtol=1e-13, atol=1e-13)
        assert actual_point.yaw == pytest.approx(expected_point.yaw, rel=1e-13, abs=1e-13)
        assert actual_point.yaw_rate == pytest.approx(expected_point.yaw_rate, rel=1e-13, abs=1e-13)
