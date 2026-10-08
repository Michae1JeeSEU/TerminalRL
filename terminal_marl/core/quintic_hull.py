"""Sufficient continuous quintic bounds, with analytic fallback at callers.

A polynomial lies in the convex hull of its Bernstein control points. The
largest control-vector norm therefore bounds the derivative norm everywhere,
including between samples. These conservative bounds never replace or populate
QuinticSegment's analytic extrema cache. Ambiguous or nonfinite bounds cannot
certify a segment; callers retain the original analytic check in that case.
"""
from __future__ import annotations

import math
import numpy as np

from .polynomial_trajectory import QuinticSegment


def _matrix(degree):
    return np.asarray([
        [math.comb(k, i) / math.comb(degree, i) if i <= k else 0.
         for i in range(degree + 1)] for k in range(degree + 1)
    ])


_VELOCITY_MATRIX = _matrix(4)
_ACCELERATION_MATRIX = _matrix(3)
_POSITION_MATRIX = _matrix(5)


def quintic_hull_bounds(segments):
    """Return conservative bounds without changing a segment or its cache."""
    segments = tuple(segments)
    if not segments:
        return ()
    if any(not isinstance(segment, QuinticSegment) for segment in segments):
        raise TypeError('segments must contain only QuinticSegment objects')
    coefficients = np.stack([s.coefficients for s in segments])
    duration = np.asarray([s.duration for s in segments])[:, None, None]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        velocity = coefficients[:, 1:] * np.arange(1, 6)[None, :, None] / duration
        acceleration = coefficients[:, 2:] * np.asarray([2., 6., 12., 20.])[None, :, None] / duration**2
        values = []
        for power, matrix in ((coefficients, _POSITION_MATRIX),
                              (velocity, _VELOCITY_MATRIX),
                              (acceleration, _ACCELERATION_MATRIX)):
            control = np.einsum('ki,nid->nkd', matrix, power)
            # Outward margin scales with the un-cancelled power coefficients,
            # deliberately preferring exact fallback near a limit.
            margin = 1e-8 * (1. + np.linalg.norm(power, axis=-1).sum(axis=-1))
            values.append((control, margin))
        (p, pm), (v, vm), (a, am) = values
        columns = dict(
            max_horizontal_speed=np.linalg.norm(v[:, :, :2], axis=-1).max(axis=-1) + vm,
            max_horizontal_accel=np.linalg.norm(a[:, :, :2], axis=-1).max(axis=-1) + am,
            max_climb_rate=np.maximum(0., v[:, :, 2].max(axis=-1)) + vm,
            max_descent_rate=np.maximum(0., -v[:, :, 2].min(axis=-1)) + vm,
            max_vertical_accel_up=np.maximum(0., a[:, :, 2].max(axis=-1)) + am,
            max_vertical_accel_down=np.maximum(0., -a[:, :, 2].min(axis=-1)) + am,
            min_altitude=p[:, :, 2].min(axis=-1) - pm,
            max_altitude=p[:, :, 2].max(axis=-1) + pm,
        )
    return tuple({key: float(value[i]) for key, value in columns.items()}
                 for i in range(len(segments)))


def horizontal_hull_certifies(segments, speed_limit, acceleration_limit):
    """Sufficient horizontal acceptance only; false means use analytic bounds."""
    segments = tuple(segments)
    if not segments:
        return False
    coefficients = np.stack([s.coefficients for s in segments])
    durations = np.asarray([s.duration for s in segments])[:, None, None]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        velocity = coefficients[:, 1:, :2] * np.arange(1, 6)[None, :, None] / durations
        acceleration = coefficients[:, 2:, :2] * np.asarray([2., 6., 12., 20.])[None, :, None] / durations**2
        for values, matrix, limit in ((velocity, _VELOCITY_MATRIX, speed_limit),
                                     (acceleration, _ACCELERATION_MATRIX, acceleration_limit)):
            controls = np.einsum('ki,nid->nkd', matrix, values)
            bound = np.linalg.norm(controls, axis=-1).max(axis=-1)
            margin = 1e-8 * (1. + np.linalg.norm(values, axis=-1).sum(axis=-1))
            if not np.isfinite(bound + margin).all() or not np.all(bound + margin <= limit):
                return False
    return True
