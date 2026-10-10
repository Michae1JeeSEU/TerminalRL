"""Immutable analytic quintic trajectory segments.

Coefficients use normalized time ``u`` in ``[0, 1]`` and are stored in
ascending polynomial order as a ``(6, 3)`` array.  This module intentionally
does not import the executable ``TrajectoryPoint`` type so it can be used by
trajectory construction and diagnostics without introducing an import cycle.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


_ROOT_TOLERANCE = 1.0e-10
_IMAG_TOLERANCE = 1.0e-8

# Search candidates repeatedly reconstruct identical hold/transfer segments.
# Bounds depend only on the exact coefficients and duration, not on absolute
# time or a vehicle's limits. Never round a key or reuse a safety verdict here.
_BOUNDS_REUSE_LIMIT = 8192
_BOUNDS_REUSE_CACHE: OrderedDict[tuple, tuple[tuple[str, float], ...]] = OrderedDict()


def _bounds_reuse_key(segment):
    values = segment.coefficients
    return (segment.duration, values.shape, values.dtype.str, values.tobytes())


def _finite_scalar(value: Any, name: str) -> float:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if array.ndim != 0:
        raise ValueError(f"{name} must be a finite scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _vector3(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (3,) and finite values") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape (3,) and finite values")
    return np.array(array, dtype=float, copy=True)


def _coefficients(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("coefficients must have shape (6, 3) and finite values") from exc
    if array.shape != (6, 3) or not np.all(np.isfinite(array)):
        raise ValueError("coefficients must have shape (6, 3) and finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _derivative_coefficients(coefficients: np.ndarray, order: int) -> np.ndarray:
    result = np.asarray(coefficients, dtype=float)
    for _ in range(order):
        if result.shape[0] <= 1:
            return np.zeros((1, result.shape[1]), dtype=float)
        result = result[1:] * np.arange(1, result.shape[0], dtype=float)[:, None]
    return result


def _evaluate_polynomial(coefficients: np.ndarray, u: Any) -> np.ndarray:
    values = np.asarray(u, dtype=float)
    result = np.zeros(values.shape + (coefficients.shape[1],), dtype=float)
    for coefficient in coefficients[::-1]:
        result = result * values[..., None] + coefficient
    return result


def _trim_polynomial(coefficients: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(coefficients), dtype=float).reshape(-1)
    if values.size == 0:
        return np.zeros(1, dtype=float)
    scale = max(1.0, float(np.max(np.abs(values))))
    tolerance = _ROOT_TOLERANCE * scale
    last = values.size - 1
    while last > 0 and abs(values[last]) <= tolerance:
        last -= 1
    trimmed = np.array(values[: last + 1], dtype=float, copy=True)
    if not np.any(np.abs(trimmed) > tolerance):
        return np.zeros(1, dtype=float)
    return trimmed


def _real_roots_in_unit_interval(coefficients: Iterable[float]) -> list[float]:
    trimmed = _trim_polynomial(coefficients)
    if trimmed.size <= 1 or not np.any(np.abs(trimmed) > 0.0):
        return []
    roots = np.polynomial.polynomial.polyroots(trimmed)
    result: list[float] = []
    for root in roots:
        complex_root = complex(root)
        if abs(complex_root.imag) > _IMAG_TOLERANCE:
            continue
        real_root = float(complex_root.real)
        if -_ROOT_TOLERANCE <= real_root <= 1.0 + _ROOT_TOLERANCE:
            result.append(float(np.clip(real_root, 0.0, 1.0)))
    return result


def _unique_unit_points(points: Iterable[float]) -> np.ndarray:
    values = sorted(float(np.clip(point, 0.0, 1.0)) for point in points)
    unique: list[float] = []
    for value in values:
        if not unique or abs(value - unique[-1]) > _ROOT_TOLERANCE:
            unique.append(value)
    return np.asarray(unique, dtype=float)


def _dot_polynomial(left: np.ndarray, right: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    result = np.zeros(1, dtype=float)
    for axis in axes:
        product = np.polynomial.polynomial.polymul(left[:, axis], right[:, axis])
        if result.size < product.size:
            result = np.pad(result, (0, product.size - result.size))
        result[: product.size] += product
    return _trim_polynomial(result)


def _scalar_extrema(values: np.ndarray, roots: Iterable[float], scale: float = 1.0) -> tuple[float, float]:
    points = _unique_unit_points((0.0, 1.0, *roots))
    evaluated = _evaluate_polynomial(values[:, None], points)[:, 0] / scale
    return float(np.max(evaluated)), float(np.min(evaluated))


@dataclass(frozen=True)
class QuinticSegment:
    """One immutable three-axis quintic segment in normalized time."""

    start_time: float
    end_time: float
    coefficients: np.ndarray
    _bounds_cache: tuple[tuple[str, float], ...] | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        start = _finite_scalar(self.start_time, "start_time")
        end = _finite_scalar(self.end_time, "end_time")
        if end <= start:
            raise ValueError("end_time must be greater than start_time")
        coefficients = _coefficients(self.coefficients)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "_bounds_cache", None)

    @classmethod
    def from_boundary(
        cls,
        t0: float,
        t1: float,
        p0: Any,
        v0: Any,
        a0: Any,
        p1: Any,
        v1: Any,
        a1: Any,
    ) -> "QuinticSegment":
        start = _finite_scalar(t0, "t0")
        end = _finite_scalar(t1, "t1")
        if end <= start:
            raise ValueError("t1 must be greater than t0")
        position0 = _vector3(p0, "p0")
        velocity0 = _vector3(v0, "v0")
        acceleration0 = _vector3(a0, "a0")
        position1 = _vector3(p1, "p1")
        velocity1 = _vector3(v1, "v1")
        acceleration1 = _vector3(a1, "a1")
        duration = end - start

        c0 = position0
        c1 = duration * velocity0
        c2 = 0.5 * duration**2 * acceleration0
        delta_position = position1 - c0 - c1 - c2
        delta_velocity = duration * velocity1 - c1 - 2.0 * c2
        delta_acceleration = duration**2 * acceleration1 - 2.0 * c2
        c3 = 10.0 * delta_position - 4.0 * delta_velocity + 0.5 * delta_acceleration
        c4 = -15.0 * delta_position + 7.0 * delta_velocity - delta_acceleration
        c5 = 6.0 * delta_position - 3.0 * delta_velocity + 0.5 * delta_acceleration
        coefficients = np.stack((c0, c1, c2, c3, c4, c5), axis=0)
        return cls(start, end, coefficients)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def _normalized_time(self, time: Any) -> float:
        value = _finite_scalar(time, "time")
        return float(np.clip((value - self.start_time) / self.duration, 0.0, 1.0))

    def sample(self, time: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return position, velocity and acceleration at a clamped time."""
        u = self._normalized_time(time)
        position = _evaluate_polynomial(self.coefficients, u)
        velocity = _evaluate_polynomial(_derivative_coefficients(self.coefficients, 1), u) / self.duration
        acceleration = _evaluate_polynomial(_derivative_coefficients(self.coefficients, 2), u) / self.duration**2
        return tuple(np.array(value, dtype=float, copy=True) for value in (position, velocity, acceleration))

    def sample_positions(self, times: Any) -> np.ndarray:
        """Return clamped positions for scalar or array-like times."""
        values = np.asarray(times, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("times must contain only finite values")
        normalized = np.clip((values - self.start_time) / self.duration, 0.0, 1.0)
        positions = _evaluate_polynomial(self.coefficients, normalized)
        return np.array(positions, dtype=float, copy=True)

    def bounds(self) -> dict[str, float]:
        """Return analytic speed and acceleration envelope bounds for this segment."""
        if self._bounds_cache is not None:
            return {key: float(value) for key, value in self._bounds_cache}
        duration = self.duration
        velocity_u = _derivative_coefficients(self.coefficients, 1)
        acceleration_u = _derivative_coefficients(self.coefficients, 2)
        jerk_u = _derivative_coefficients(self.coefficients, 3)

        horizontal_speed_roots = _real_roots_in_unit_interval(
            _dot_polynomial(velocity_u, acceleration_u, (0, 1))
        )
        horizontal_acceleration_roots = _real_roots_in_unit_interval(
            _dot_polynomial(acceleration_u, jerk_u, (0, 1))
        )
        vertical_speed_roots = _real_roots_in_unit_interval(acceleration_u[:, 2])
        vertical_acceleration_roots = _real_roots_in_unit_interval(jerk_u[:, 2])
        altitude_roots = _real_roots_in_unit_interval(velocity_u[:, 2])

        horizontal_speed_points = _unique_unit_points((0.0, 1.0, *horizontal_speed_roots))
        horizontal_acceleration_points = _unique_unit_points((0.0, 1.0, *horizontal_acceleration_roots))
        vertical_speed_points = _unique_unit_points((0.0, 1.0, *vertical_speed_roots))
        vertical_acceleration_points = _unique_unit_points((0.0, 1.0, *vertical_acceleration_roots))
        altitude_points = _unique_unit_points((0.0, 1.0, *altitude_roots))

        horizontal_velocity = _evaluate_polynomial(velocity_u, horizontal_speed_points)[:, :2] / duration
        horizontal_acceleration = _evaluate_polynomial(
            acceleration_u, horizontal_acceleration_points
        )[:, :2] / duration**2
        vertical_velocity = _evaluate_polynomial(velocity_u, vertical_speed_points)[:, 2] / duration
        vertical_acceleration = _evaluate_polynomial(
            acceleration_u, vertical_acceleration_points
        )[:, 2] / duration**2
        altitude = _evaluate_polynomial(
            self.coefficients[:, 2:3], altitude_points
        )[:, 0]

        bounds = {
            "max_horizontal_speed": float(np.max(np.linalg.norm(horizontal_velocity, axis=1))),
            "max_horizontal_accel": float(np.max(np.linalg.norm(horizontal_acceleration, axis=1))),
            "max_climb_rate": float(max(0.0, np.max(vertical_velocity))),
            "max_descent_rate": float(max(0.0, -np.min(vertical_velocity))),
            "max_vertical_accel_up": float(max(0.0, np.max(vertical_acceleration))),
            "max_vertical_accel_down": float(max(0.0, -np.min(vertical_acceleration))),
            "min_altitude": float(np.min(altitude)),
            "max_altitude": float(np.max(altitude)),
        }
        object.__setattr__(self, "_bounds_cache", tuple((key, float(value)) for key, value in bounds.items()))
        return dict(bounds)


def _batch_real_roots_in_unit_interval(
    coefficients: Iterable[Iterable[float]],
) -> list[list[float]]:
    """Return the scalar root-filtering result for a batch of polynomials.

    The trimming and root acceptance rules intentionally mirror
    :func:`_real_roots_in_unit_interval`.  Polynomials are grouped only by
    their exact trimmed degree.  Coefficients are not rounded or used as a
    cache key; each group receives its own companion matrix and eigensolve.
    """
    trimmed = [_trim_polynomial(values) for values in coefficients]
    result: list[list[float]] = [[] for _ in trimmed]
    groups: dict[int, list[int]] = {}
    for index, values in enumerate(trimmed):
        if values.size <= 1 or not np.any(np.abs(values) > 0.0):
            continue
        groups.setdefault(values.size - 1, []).append(index)

    for degree, indices in groups.items():
        # The installed scalar ``polyroots`` source calls
        # ``m = polycompanion(c)`` directly.  ``polycompanion`` itself builds
        # the power-series companion matrix with its subdiagonal and final
        # column in the same ascending-coefficient convention, so the batch
        # eigensolve uses that exact matrix orientation.
        if degree == 1:
            roots = np.asarray(
                [[-trimmed[index][0] / trimmed[index][1]] for index in indices],
                dtype=float,
            )
        else:
            matrices = np.stack(
                [np.polynomial.polynomial.polycompanion(trimmed[index]) for index in indices],
                axis=0,
            )
            roots = np.linalg.eigvals(matrices)
        if roots.shape != (len(indices), degree):
            raise RuntimeError("batched companion eigensolve returned an invalid shape")
        for row, index in enumerate(indices):
            accepted: list[float] = []
            for root in roots[row]:
                complex_root = complex(root)
                if abs(complex_root.imag) > _IMAG_TOLERANCE:
                    continue
                real_root = float(complex_root.real)
                if -_ROOT_TOLERANCE <= real_root <= 1.0 + _ROOT_TOLERANCE:
                    accepted.append(float(np.clip(real_root, 0.0, 1.0)))
            result[index] = accepted
    return result


def _batch_points(roots: Iterable[Iterable[float]]) -> np.ndarray:
    """Build a padded candidate-point matrix with scalar point semantics."""
    points = [_unique_unit_points((0.0, 1.0, *values)) for values in roots]
    if not points:
        return np.empty((0, 0), dtype=float)
    width = max(values.size for values in points)
    result = np.zeros((len(points), width), dtype=float)
    for index, values in enumerate(points):
        result[index, : values.size] = values
    return result


def _evaluate_polynomial_batch(coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate one vector polynomial per batch row using Horner's rule."""
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 3 or points.ndim != 2 or values.shape[0] != points.shape[0]:
        raise ValueError("batched vector polynomial shapes are incompatible")
    result = np.zeros((values.shape[0], points.shape[1], values.shape[2]), dtype=float)
    for coefficient_index in range(values.shape[1] - 1, -1, -1):
        coefficient = values[:, coefficient_index, :]
        result = result * points[:, :, None] + coefficient[:, None, :]
    return result


def _evaluate_scalar_polynomial_batch(coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate one scalar polynomial per batch row using Horner's rule."""
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 2 or points.ndim != 2 or values.shape[0] != points.shape[0]:
        raise ValueError("batched scalar polynomial shapes are incompatible")
    result = np.zeros((values.shape[0], points.shape[1]), dtype=float)
    for coefficient_index in range(values.shape[1] - 1, -1, -1):
        coefficient = values[:, coefficient_index]
        result = result * points + coefficient[:, None]
    return result


def prime_quintic_bounds(segments: Iterable[QuinticSegment]) -> tuple[dict[str, float], ...]:
    """Prime analytic bounds caches for a batch of quintic segments.

    The scalar :meth:`QuinticSegment.bounds` implementation remains the
    reference path.  This helper applies the same coefficient trimming,
    companion-matrix root filtering, candidate endpoints, and Horner
    evaluation in batches, then stores the resulting immutable cache tuple on
    each uncached segment.  A tuple of fresh public dictionaries is returned
    in input order; repeated or already-cached segments retain their scalar
    cache semantics.
    """
    batch = tuple(segments)
    if not batch:
        return ()
    if any(not isinstance(segment, QuinticSegment) for segment in batch):
        raise TypeError("segments must contain only QuinticSegment objects")

    for segment in batch:
        # Custom subclasses may define additional semantics; retain their
        # existing calculation path rather than sharing the base-type cache.
        if type(segment) is QuinticSegment and segment._bounds_cache is None:
            key = _bounds_reuse_key(segment)
            cached = _BOUNDS_REUSE_CACHE.get(key)
            if cached is not None:
                _BOUNDS_REUSE_CACHE.move_to_end(key)
                object.__setattr__(segment, "_bounds_cache", cached)

    uncached_indices = [
        index for index, segment in enumerate(batch) if segment._bounds_cache is None
    ]
    if not uncached_indices:
        return tuple(
            {key: float(value) for key, value in segment._bounds_cache or ()}
            for segment in batch
        )

    uncached = [batch[index] for index in uncached_indices]
    durations = np.asarray([segment.duration for segment in uncached], dtype=float)
    coefficients = np.stack([segment.coefficients for segment in uncached], axis=0)
    velocity_u = np.stack(
        [_derivative_coefficients(segment.coefficients, 1) for segment in uncached], axis=0
    )
    acceleration_u = np.stack(
        [_derivative_coefficients(segment.coefficients, 2) for segment in uncached], axis=0
    )
    jerk_u = np.stack(
        [_derivative_coefficients(segment.coefficients, 3) for segment in uncached], axis=0
    )

    horizontal_speed_coefficients = [
        _dot_polynomial(velocity_u[index], acceleration_u[index], (0, 1))
        for index in range(len(uncached))
    ]
    horizontal_acceleration_coefficients = [
        _dot_polynomial(acceleration_u[index], jerk_u[index], (0, 1))
        for index in range(len(uncached))
    ]
    vertical_speed_coefficients = [acceleration_u[index, :, 2] for index in range(len(uncached))]
    vertical_acceleration_coefficients = [jerk_u[index, :, 2] for index in range(len(uncached))]
    altitude_root_coefficients = [velocity_u[index, :, 2] for index in range(len(uncached))]
    altitude_coefficients = [coefficients[index, :, 2] for index in range(len(uncached))]

    horizontal_speed_roots = _batch_real_roots_in_unit_interval(horizontal_speed_coefficients)
    horizontal_acceleration_roots = _batch_real_roots_in_unit_interval(
        horizontal_acceleration_coefficients
    )
    vertical_speed_roots = _batch_real_roots_in_unit_interval(vertical_speed_coefficients)
    vertical_acceleration_roots = _batch_real_roots_in_unit_interval(
        vertical_acceleration_coefficients
    )
    altitude_roots = _batch_real_roots_in_unit_interval(altitude_root_coefficients)

    horizontal_speed_points = _batch_points(horizontal_speed_roots)
    horizontal_acceleration_points = _batch_points(horizontal_acceleration_roots)
    vertical_speed_points = _batch_points(vertical_speed_roots)
    vertical_acceleration_points = _batch_points(vertical_acceleration_roots)
    altitude_points = _batch_points(altitude_roots)

    horizontal_velocity = _evaluate_polynomial_batch(
        velocity_u, horizontal_speed_points
    )[:, :, :2] / durations[:, None, None]
    horizontal_acceleration = _evaluate_polynomial_batch(
        acceleration_u, horizontal_acceleration_points
    )[:, :, :2] / durations[:, None, None] ** 2
    vertical_velocity = _evaluate_polynomial_batch(
        velocity_u, vertical_speed_points
    )[:, :, 2] / durations[:, None]
    vertical_acceleration = _evaluate_polynomial_batch(
        acceleration_u, vertical_acceleration_points
    )[:, :, 2] / durations[:, None] ** 2
    altitude = _evaluate_scalar_polynomial_batch(coefficients[:, :, 2], altitude_points)

    scalar_bounds: list[dict[str, float]] = []
    for index in range(len(uncached)):
        bounds = {
            "max_horizontal_speed": float(
                np.max(np.linalg.norm(horizontal_velocity[index], axis=1))
            ),
            "max_horizontal_accel": float(
                np.max(np.linalg.norm(horizontal_acceleration[index], axis=1))
            ),
            "max_climb_rate": float(max(0.0, np.max(vertical_velocity[index]))),
            "max_descent_rate": float(max(0.0, -np.min(vertical_velocity[index]))),
            "max_vertical_accel_up": float(
                max(0.0, np.max(vertical_acceleration[index]))
            ),
            "max_vertical_accel_down": float(
                max(0.0, -np.min(vertical_acceleration[index]))
            ),
            "min_altitude": float(np.min(altitude[index])),
            "max_altitude": float(np.max(altitude[index])),
        }
        if not all(np.isfinite(value) for value in bounds.values()):
            raise ValueError("batched quintic bounds must be finite")
        scalar_bounds.append(bounds)

    for segment, bounds in zip(uncached, scalar_bounds):
        immutable_bounds = tuple((key, float(value)) for key, value in bounds.items())
        object.__setattr__(
            segment,
            "_bounds_cache",
            immutable_bounds,
        )
        if type(segment) is QuinticSegment:
            key = _bounds_reuse_key(segment)
            _BOUNDS_REUSE_CACHE[key] = immutable_bounds
            _BOUNDS_REUSE_CACHE.move_to_end(key)
            while len(_BOUNDS_REUSE_CACHE) > _BOUNDS_REUSE_LIMIT:
                _BOUNDS_REUSE_CACHE.popitem(last=False)

    result: list[dict[str, float]] = []
    for segment in batch:
        if segment._bounds_cache is None:
            # This branch is unreachable after the cache writes above, but
            # retaining the guard makes an unexpected cache mutation fail
            # closed rather than fabricating a public result.
            raise RuntimeError("quintic bounds cache was not populated")
        result.append({key: float(value) for key, value in segment._bounds_cache})
    return tuple(result)
