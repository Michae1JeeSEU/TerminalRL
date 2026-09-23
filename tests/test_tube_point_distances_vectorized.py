"""Parity and throughput checks for the reachable-tube point distance kernel."""

from __future__ import annotations

import math
import time

import numpy as np

from terminal_marl.core.config import RingRole, RingSpec
from terminal_marl.core.safety_filter import SafetyFilter


def _scalar_reference(points: np.ndarray, ring: RingSpec, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    values = []
    for point, lo, hi in zip(np.asarray(points, dtype=float), lower, upper):
        radius = float(np.linalg.norm(point[:2]))
        angle = float(np.arctan2(point[1], point[0]))
        candidates = [lo, hi]
        for turns in range(-2, 3):
            shifted = angle + turns * 2.0 * math.pi
            if lo <= shifted <= hi:
                candidates.append(shifted)
        values.append(min(
            math.sqrt(max(0.0, ring.radius**2 + radius**2
                          - 2.0 * ring.radius * radius * math.cos(candidate - angle))
                        + (ring.height - point[2])**2)
            for candidate in candidates
        ))
    return np.asarray(values, dtype=float)


def _random_case(seed: int, count: int) -> tuple[np.ndarray, RingSpec, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ring = RingSpec(RingRole.MIDDLE, radius=400.0, height=70.0, nominal_speed=20.0)
    points = rng.normal(size=(count, 3))
    points[:, :2] *= 800.0
    points[:, 2] = rng.uniform(-20.0, 140.0, count)
    centers = rng.uniform(-8.0 * math.pi, 8.0 * math.pi, count)
    spans = rng.uniform(0.0, 7.0 * math.pi, count)
    lower = centers - spans / 2.0
    upper = centers + spans / 2.0
    return points, ring, lower, upper


def test_vectorized_tube_distances_match_random_scalar_reference():
    points, ring, lower, upper = _random_case(20260924, 5000)
    actual = SafetyFilter._tube_point_distances(points, ring, lower, upper)
    expected = _scalar_reference(points, ring, lower, upper)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_vectorized_tube_distances_match_wrapped_multiturn_and_zero_length_intervals():
    ring = RingSpec(RingRole.INNER, radius=200.0, height=40.0, nominal_speed=16.0)
    angles = np.asarray((math.pi - 1e-12, -math.pi + 1e-12, 0.4, -2.1, 1.7, 0.0))
    radii = np.asarray((200.0, 200.0, 180.0, 520.0, 200.0, 0.0))
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles),
                              np.asarray((40.0, 40.0, 10.0, 90.0, 40.0, 40.0))))
    lower = np.asarray((math.pi - 0.2, -math.pi - 0.2, 0.4, -4.0 * math.pi, 1.7, 5.0))
    upper = np.asarray((math.pi + 0.2, -math.pi + 0.2, 0.4, 4.0 * math.pi, 1.7, 5.0))
    expected = _scalar_reference(points, ring, lower, upper)
    actual = SafetyFilter._tube_point_distances(points, ring, lower, upper)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_vectorized_tube_distances_preserve_zip_truncation():
    points, ring, lower, upper = _random_case(7, 8)
    actual = SafetyFilter._tube_point_distances(points, ring, lower[:5], upper[:5])
    expected = _scalar_reference(points, ring, lower[:5], upper[:5])
    assert actual.shape == (5,)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_vectorized_tube_distances_threshold_near_cases():
    ring = RingSpec(RingRole.OUTER, radius=600.0, height=100.0, nominal_speed=22.0)
    safe_distance = 80.0
    offsets = np.asarray((-1e-10, 0.0, 1e-10, -1e-8, 1e-8))
    points = np.column_stack((np.full(len(offsets), ring.radius) + safe_distance + offsets,
                              np.zeros(len(offsets)), np.full(len(offsets), ring.height)))
    lower = np.zeros(len(offsets))
    upper = np.zeros(len(offsets))
    actual = SafetyFilter._tube_point_distances(points, ring, lower, upper)
    expected = _scalar_reference(points, ring, lower, upper)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(actual < safe_distance, expected < safe_distance)


def test_vectorized_tube_distances_microbenchmark():
    """Print representative scalar/vector timings while checking parity."""
    for count in (300, 1500):
        points, ring, lower, upper = _random_case(1000 + count, count)
        expected = _scalar_reference(points, ring, lower, upper)
        scalar_start = time.perf_counter()
        for _ in range(3):
            scalar = _scalar_reference(points, ring, lower, upper)
        scalar_seconds = (time.perf_counter() - scalar_start) / 3.0
        vector_start = time.perf_counter()
        for _ in range(12):
            actual = SafetyFilter._tube_point_distances(points, ring, lower, upper)
        vector_seconds = (time.perf_counter() - vector_start) / 12.0
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        print(f"tube_point_distances n={count}: scalar={scalar_seconds:.6e}s vector={vector_seconds:.6e}s")
        assert scalar.shape == actual.shape
