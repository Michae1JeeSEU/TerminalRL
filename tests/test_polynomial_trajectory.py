"""Physics-focused tests for analytic quintic trajectory segments."""

from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.polynomial_trajectory import QuinticSegment


def _boundary_segment() -> QuinticSegment:
    return QuinticSegment.from_boundary(
        2.0,
        5.0,
        (1.0, -2.0, 3.0),
        (2.0, 0.5, -1.0),
        (0.3, -0.2, 0.4),
        (8.0, 4.0, 0.0),
        (-1.0, 2.0, 0.5),
        (-0.1, 0.6, -0.2),
    )


def test_boundary_position_velocity_acceleration_and_read_only_copy() -> None:
    coefficients = np.zeros((6, 3), dtype=float)
    segment = QuinticSegment(0.0, 1.0, coefficients)
    coefficients[0, 0] = 99.0
    assert segment.coefficients[0, 0] == 0.0
    assert segment.coefficients.flags.writeable is False
    with pytest.raises(ValueError):
        segment.coefficients[0, 0] = 1.0

    segment = _boundary_segment()
    start = segment.sample(segment.start_time)
    end = segment.sample(segment.end_time)
    np.testing.assert_allclose(start[0], (1.0, -2.0, 3.0), atol=1e-12)
    np.testing.assert_allclose(start[1], (2.0, 0.5, -1.0), atol=1e-12)
    np.testing.assert_allclose(start[2], (0.3, -0.2, 0.4), atol=1e-12)
    np.testing.assert_allclose(end[0], (8.0, 4.0, 0.0), atol=1e-10)
    np.testing.assert_allclose(end[1], (-1.0, 2.0, 0.5), atol=1e-10)
    np.testing.assert_allclose(end[2], (-0.1, 0.6, -0.2), atol=1e-9)


def test_analytic_derivatives_match_finite_differences() -> None:
    segment = _boundary_segment()
    time = 3.17
    epsilon = 1.0e-5
    position_minus, velocity_minus, _ = segment.sample(time - epsilon)
    position, velocity, acceleration = segment.sample(time)
    position_plus, velocity_plus, _ = segment.sample(time + epsilon)

    np.testing.assert_allclose(
        (position_plus - position_minus) / (2.0 * epsilon), velocity, rtol=1e-8, atol=1e-8,
    )
    np.testing.assert_allclose(
        (velocity_plus - velocity_minus) / (2.0 * epsilon), acceleration, rtol=1e-7, atol=1e-7,
    )
    assert np.all(np.isfinite(position))


def test_scalar_and_batch_position_sampling_share_clamped_semantics() -> None:
    segment = _boundary_segment()
    times = np.asarray((0.0, 2.0, 2.75, 5.0, 7.0), dtype=float)
    batch = segment.sample_positions(times)
    assert batch.shape == (5, 3)
    for index, time in enumerate(times):
        np.testing.assert_allclose(batch[index], segment.sample(float(time))[0], atol=1e-12)
    np.testing.assert_allclose(segment.sample_positions(3.0), segment.sample(3.0)[0], atol=1e-12)
    np.testing.assert_allclose(batch[0], segment.sample(segment.start_time)[0], atol=1e-12)
    np.testing.assert_allclose(batch[-1], segment.sample(segment.end_time)[0], atol=1e-12)


def test_adjacent_segments_are_c2_at_shared_boundary() -> None:
    first = QuinticSegment.from_boundary(
        0.0, 2.0, (0.0, 0.0, 1.0), (1.0, 2.0, 0.0), (0.1, 0.2, -0.1),
        (2.0, 3.0, 0.5), (0.0, 1.0, -0.2), (-0.2, 0.1, 0.3),
    )
    boundary = first.sample(first.end_time)
    second = QuinticSegment.from_boundary(
        2.0, 5.0, boundary[0], boundary[1], boundary[2],
        (1.0, 6.0, 0.0), (-1.0, 0.0, 0.3), (0.0, -0.2, -0.1),
    )
    for left, right in zip(boundary, second.sample(second.start_time)):
        np.testing.assert_allclose(left, right, atol=1e-10)


def test_analytic_bounds_match_dense_samples() -> None:
    segment = QuinticSegment.from_boundary(
        0.0,
        4.0,
        (0.0, 0.0, 0.0),
        (3.0, -1.0, 2.0),
        (0.5, 0.2, -0.3),
        (20.0, 5.0, 3.0),
        (1.0, 2.0, -1.0),
        (-0.2, 0.3, 0.1),
    )
    times = np.linspace(segment.start_time, segment.end_time, 100001)
    samples = [segment.sample(float(time)) for time in times]
    velocities = np.asarray([sample[1] for sample in samples])
    accelerations = np.asarray([sample[2] for sample in samples])
    dense = {
        "max_horizontal_speed": float(np.max(np.linalg.norm(velocities[:, :2], axis=1))),
        "max_horizontal_accel": float(np.max(np.linalg.norm(accelerations[:, :2], axis=1))),
        "max_climb_rate": float(max(0.0, np.max(velocities[:, 2]))),
        "max_descent_rate": float(max(0.0, -np.min(velocities[:, 2]))),
        "max_vertical_accel_up": float(max(0.0, np.max(accelerations[:, 2]))),
        "max_vertical_accel_down": float(max(0.0, -np.min(accelerations[:, 2]))),
    }
    bounds = segment.bounds()
    for key, value in dense.items():
        assert bounds[key] >= value - 1.0e-9
        assert bounds[key] == pytest.approx(value, rel=2.0e-6, abs=2.0e-6)


def test_internal_speed_and_acceleration_peaks_are_included() -> None:
    segment = QuinticSegment.from_boundary(
        0.0,
        1.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 2.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    bounds = segment.bounds()
    assert bounds["max_horizontal_speed"] > 1.0
    assert bounds["max_climb_rate"] > 1.0
    assert bounds["max_horizontal_accel"] > 1.0
    assert bounds["max_vertical_accel_up"] > 1.0
    assert bounds["max_vertical_accel_down"] > 1.0


def test_altitude_bounds_include_interior_vertical_position_extrema() -> None:
    segment = QuinticSegment.from_boundary(
        0.0, 1.0,
        (0.0, 0.0, 0.0), (0.0, 0.0, 4.0), (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0), (0.0, 0.0, -4.0), (0.0, 0.0, 0.0),
    )
    bounds = segment.bounds()
    assert bounds["min_altitude"] == pytest.approx(0.0, abs=1e-10)
    assert bounds["max_altitude"] > 1.0


def test_invalid_times_shapes_and_nonfinite_values_are_rejected() -> None:
    zero_vectors = np.zeros(3, dtype=float)
    with pytest.raises(ValueError):
        QuinticSegment.from_boundary(1.0, 1.0, zero_vectors, zero_vectors, zero_vectors, zero_vectors, zero_vectors, zero_vectors)
    with pytest.raises(ValueError):
        QuinticSegment.from_boundary(np.nan, 1.0, zero_vectors, zero_vectors, zero_vectors, zero_vectors, zero_vectors, zero_vectors)
    with pytest.raises(ValueError):
        QuinticSegment(0.0, 1.0, np.zeros((5, 3), dtype=float))
    with pytest.raises(ValueError):
        QuinticSegment(0.0, 1.0, np.full((6, 3), np.inf))

    segment = _boundary_segment()
    with pytest.raises(ValueError):
        segment.sample(np.nan)
    with pytest.raises(ValueError):
        segment.sample_positions((0.0, np.nan))
