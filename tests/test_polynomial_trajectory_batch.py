"""Regression tests for batched analytic quintic bounds priming."""

from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.polynomial_trajectory import (
    QuinticSegment,
    prime_quintic_bounds,
)


def _segments() -> list[QuinticSegment]:
    return [
        QuinticSegment.from_boundary(
            0.0,
            1.0,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 1.5),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        QuinticSegment.from_boundary(
            2.0,
            4.5,
            (1.0, -2.0, 0.5),
            (2.0, 0.5, -1.0),
            (0.3, -0.2, 0.4),
            (8.0, 4.0, 0.0),
            (-1.0, 2.0, 0.5),
            (-0.1, 0.6, -0.2),
        ),
        QuinticSegment(
            7.0,
            12.0,
            np.zeros((6, 3), dtype=float),
        ),
        QuinticSegment.from_boundary(
            13.0,
            17.0,
            (0.0, 0.0, 0.0),
            (3.0, -1.0, 2.0),
            (0.5, 0.2, -0.3),
            (20.0, 5.0, 3.0),
            (1.0, 2.0, -1.0),
            (-0.2, 0.3, 0.1),
        ),
    ]


def _fresh(segment: QuinticSegment) -> QuinticSegment:
    return QuinticSegment(
        segment.start_time,
        segment.end_time,
        np.array(segment.coefficients, dtype=float, copy=True),
    )


def test_batch_matches_unprimed_scalar_reference_for_mixed_degrees() -> None:
    reference_segments = _segments()
    reference = [segment.bounds() for segment in reference_segments]
    primed_segments = [_fresh(segment) for segment in reference_segments]

    result = prime_quintic_bounds(primed_segments)

    assert len(result) == len(reference)
    assert all(segment._bounds_cache is not None for segment in primed_segments)
    for expected, actual in zip(reference, result):
        assert set(actual) == set(expected)
        for key in expected:
            assert actual[key] == pytest.approx(expected[key], rel=0.0, abs=1.0e-9)
    # The zero segment exercises scalar/constant root polynomials in the same
    # call as higher-degree horizontal and vertical extrema polynomials.
    assert result[2]["max_horizontal_speed"] == 0.0
    assert result[2]["min_altitude"] == 0.0


def test_batch_detects_interior_speed_acceleration_and_altitude_extrema() -> None:
    segment = QuinticSegment.from_boundary(
        0.0,
        1.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 4.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -4.0),
        (0.0, 0.0, 0.0),
    )
    peak_segment = QuinticSegment.from_boundary(
        2.0,
        3.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 2.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    primed = _fresh(segment)
    primed_peak = _fresh(peak_segment)
    reference = segment.bounds()
    peak_reference = peak_segment.bounds()
    actual, peak_actual = prime_quintic_bounds([primed, primed_peak])

    for key, value in reference.items():
        assert actual[key] == pytest.approx(value, rel=0.0, abs=1.0e-9)
    assert actual["max_climb_rate"] > 1.0
    assert actual["max_descent_rate"] > 1.0
    assert peak_actual["max_vertical_accel_up"] > 1.0
    assert peak_actual["max_vertical_accel_down"] > 1.0
    assert actual["max_altitude"] > 1.0
    for key, value in peak_reference.items():
        assert peak_actual[key] == pytest.approx(value, rel=0.0, abs=1.0e-9)


def test_preexisting_cache_is_reused_and_public_results_are_copies() -> None:
    segments = _segments()
    cached = segments[0].bounds()
    cache_before = segments[0]._bounds_cache
    assert cache_before is not None

    result = prime_quintic_bounds(segments)

    assert result[0] == cached
    assert segments[0]._bounds_cache == cache_before
    result[0]["max_horizontal_speed"] = -123.0
    assert segments[0].bounds()["max_horizontal_speed"] >= 0.0


def test_empty_batch_and_generator_input_preserve_order() -> None:
    assert prime_quintic_bounds(()) == ()
    segments = _segments()[:2]
    result = prime_quintic_bounds(segment for segment in segments)
    assert len(result) == 2
    for segment, actual in zip(segments, result):
        assert actual == segment.bounds()


def test_invalid_batch_input_and_nonfinite_construction_fail_closed() -> None:
    with pytest.raises(TypeError, match="QuinticSegment"):
        prime_quintic_bounds([object()])

    with pytest.raises(ValueError, match="finite"):
        QuinticSegment(
            0.0,
            1.0,
            np.full((6, 3), np.inf, dtype=float),
        )
