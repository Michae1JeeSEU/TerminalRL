"""Geometry-derived capacity model for circular terminal holding rings."""

from __future__ import annotations

import math


def geometric_ring_capacity(
    radius: float,
    required_separation: float,
    *,
    epsilon: float = 1e-9,
    max_capacity: int | None = None,
) -> int:
    """Return the largest evenly spaced ring occupancy satisfying separation.

    For ``N`` UAVs placed uniformly on a circle of radius ``R``, adjacent chord
    spacing is ``2 R sin(pi / N)``.  The returned capacity is the largest
    integer ``N >= 1`` whose adjacent chord is at least ``required_separation``.
    """

    radius = float(radius)
    required = float(required_separation)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if not math.isfinite(required) or required <= 0.0:
        raise ValueError("required_separation must be positive and finite")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")

    if required > 2.0 * radius + epsilon:
        return 1

    ratio = min(1.0, max(0.0, required / (2.0 * radius)))
    angle = math.asin(ratio)
    if angle <= epsilon:
        capacity = 1
    else:
        capacity = max(1, int(math.floor(math.pi / angle + epsilon)))
    if max_capacity is not None:
        capacity = min(capacity, int(max_capacity))
    return int(capacity)


def adjacent_chord_spacing(radius: float, occupancy: int) -> float:
    """Adjacent spacing for ``occupancy`` uniformly spaced UAVs on a ring."""

    occupancy = int(occupancy)
    radius = float(radius)
    if occupancy <= 1:
        return math.inf
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    return float(2.0 * radius * math.sin(math.pi / occupancy))


def minimum_radius_for_occupancy(
    occupancy: int,
    required_separation: float,
    *,
    epsilon: float = 1e-9,
) -> float:
    """Minimum radius that can safely hold ``occupancy`` evenly spaced UAVs."""

    occupancy = int(occupancy)
    required = float(required_separation)
    if occupancy <= 1:
        return 0.0
    if not math.isfinite(required) or required <= 0.0:
        raise ValueError("required_separation must be positive and finite")
    denom = 2.0 * math.sin(math.pi / occupancy)
    if denom <= epsilon:
        raise ValueError("occupancy is too large for stable capacity inversion")
    return float(required / denom)
