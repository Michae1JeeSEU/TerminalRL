"""Exact numeric regressions for reducing quintic bounds Python overhead."""
from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core import polynomial_trajectory as polynomial


def _original_trim(coefficients):
    values = np.asarray(list(coefficients), dtype=float).reshape(-1)
    if values.size == 0:
        return np.zeros(1)
    tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(values))))
    last = values.size - 1
    while last > 0 and abs(values[last]) <= tolerance:
        last -= 1
    trimmed = np.array(values[:last + 1], copy=True)
    return trimmed if np.any(np.abs(trimmed) > tolerance) else np.zeros(1)


def _original_dot(left, right, axes):
    result = np.zeros(1)
    for axis in axes:
        product = np.polynomial.polynomial.polymul(left[:, axis], right[:, axis])
        if result.size < product.size:
            result = np.pad(result, (0, product.size - result.size))
        result[:product.size] += product
    return _original_trim(result)


@pytest.mark.parametrize('values', [
    [], [0.0, -0.0], [1e-12, 0.0], [1.0, 1e-10],
    [1.0, np.nextafter(1e-10, np.inf)],
    [1.0, np.nextafter(1e-10, 0.0)],
    [1e8, 0.01, 0.0], [1e8, np.nextafter(0.01, np.inf)],
    [np.nan, 1.0], [1.0, np.nan], [np.inf, 0.0],
])
def test_trim_preserves_threshold_and_nonfinite_rules_without_aliasing(values):
    array = np.array(values)
    before = array.copy()
    expected = _original_trim(values)
    for data in (array, list(values), iter(values)):
        actual = polynomial._trim_polynomial(data)
        np.testing.assert_array_equal(actual, expected, strict=True)
        assert not np.shares_memory(actual, array)
    np.testing.assert_array_equal(array, before)


@pytest.mark.parametrize('degrees', [(5, 4), (4, 3), (1, 1)])
def test_dot_matches_polymul_exactly_for_cancellation_and_trailing_zeros(degrees):
    rng = np.random.default_rng(73101)
    for i in range(80):
        left = rng.normal(size=(degrees[0], 3)) * (1e-12 if i % 5 == 0 else 1e4)
        right = rng.normal(size=(degrees[1], 3))
        if i % 3 == 0:
            left[-1] = 0.0
        if i % 4 == 0:
            right[-1] = -0.0
        if i % 7 == 0:
            left[:, 1], right[:, 1] = left[:, 0], -right[:, 0]
        for axes in ((0, 1), (2,), ()):
            expected = _original_dot(left, right, axes)
            actual = polynomial._dot_polynomial(left, right, axes)
            np.testing.assert_array_equal(actual, expected, strict=True)


def test_point_clamping_keeps_endpoints_signed_zero_and_last_kept_deduplication():
    # Consecutive differences alone are insufficient: compare to last kept point.
    points = [-1e-10, -0.0, 0.0, 0.75e-10, 1.5e-10, 1.0, 1.0+1e-10]
    for data in (points, [-0.0, 0.0], [np.nan, 0.2, np.nan, 1.2]):
        clipped = sorted(float(np.clip(x, 0.0, 1.0)) for x in data)
        expected = []
        for value in clipped:
            if not expected or abs(value - expected[-1]) > 1e-10:
                expected.append(value)
        actual = polynomial._unique_unit_points(iter(data))
        np.testing.assert_array_equal(actual, np.asarray(expected))
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_batch_bounds_match_original_helpers_including_near_zero_segments(monkeypatch):
    rng = np.random.default_rng(73102)
    coefficients = [rng.normal(size=(6, 3)) * scale for scale in (0.0, 1e-12, 1.0, 1e4)]
    coefficients += [rng.normal(size=(6, 3)) for _ in range(24)]
    coefficients[4][3:] = 0.0
    segments = [polynomial.QuinticSegment(0.0, 2.0, c) for c in coefficients]
    actual = polynomial.prime_quintic_bounds(segments)
    with monkeypatch.context() as patch:
        patch.setattr(polynomial, '_trim_polynomial', _original_trim)
        patch.setattr(polynomial, '_dot_polynomial', _original_dot)
        expected = polynomial.prime_quintic_bounds(
            polynomial.QuinticSegment(0.0, 2.0, c) for c in coefficients)
    assert actual == expected


def test_root_acceptance_near_endpoints_is_unchanged():
    roots = [-2e-10, -0.5e-10, 0.0, 0.5, 1.0, 1+0.5e-10, 1+2e-10]
    coefficients = [np.array([-root, 1.0]) for root in roots]
    expected = [[], [0.0], [0.0], [0.5], [1.0], [1.0], []]
    assert polynomial._batch_real_roots_in_unit_interval(coefficients) == expected
    assert [polynomial._real_roots_in_unit_interval(c) for c in coefficients] == expected
