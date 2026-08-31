"""Geometry-to-executable trajectory conversion under one vehicle profile."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .vehicle_dynamics import VehicleDynamics


@dataclass(frozen=True)
class GeometricPath:
    points: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.points, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("GeometricPath requires at least two finite 3-D points")
        object.__setattr__(self, "points", values)

    @property
    def segment_lengths(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.points, axis=0), axis=1)

    @property
    def arc_length(self) -> float:
        return float(np.sum(self.segment_lengths))

    @staticmethod
    def line(start: np.ndarray, end: np.ndarray, samples: int = 17) -> "GeometricPath":
        return GeometricPath(np.linspace(np.asarray(start, dtype=float), np.asarray(end, dtype=float), samples))

    @staticmethod
    def cubic_bezier(p0, p1, p2, p3, samples: int = 81) -> "GeometricPath":
        s = np.linspace(0.0, 1.0, samples)[:, None]
        values = (1 - s) ** 3 * np.asarray(p0) + 3 * (1 - s) ** 2 * s * np.asarray(p1) + 3 * (1 - s) * s ** 2 * np.asarray(p2) + s ** 3 * np.asarray(p3)
        return GeometricPath(values)


@dataclass(frozen=True)
class TrajectoryConstraints:
    end_horizontal_speed: float | None = None
    tangent_yaw: bool = False


class TrajectoryTimeParameterizer:
    """Forward/backward speed profiles with curvature and vertical limits."""

    def __init__(self, dynamics: VehicleDynamics) -> None:
        self.dynamics = dynamics

    @staticmethod
    def _tangents(points: np.ndarray) -> np.ndarray:
        derivative = np.gradient(points, axis=0)
        lengths = np.linalg.norm(derivative, axis=1, keepdims=True)
        return derivative / np.maximum(lengths, 1e-12)

    @staticmethod
    def _curvature_xy(points: np.ndarray) -> np.ndarray:
        values = np.zeros(len(points), dtype=float)
        for index in range(1, len(points) - 1):
            left, middle, right = points[index - 1:index + 2, :2]
            a, b, c = np.linalg.norm(middle - left), np.linalg.norm(right - middle), np.linalg.norm(right - left)
            first, second = middle - left, right - left
            area2 = abs(first[0] * second[1] - first[1] * second[0])
            values[index] = 0.0 if min(a, b, c) <= 1e-12 else 2.0 * area2 / (a * b * c)
        values[0] = values[1]; values[-1] = values[-2]
        return values

    def entry_speed_limit(self, path: GeometricPath) -> float:
        """Maximum dynamically admissible scalar speed at the path entrance.

        This is a local geometric bound used when a ring holder leaves along
        a tangent-continuous curve.  It uses the same curvature and vertical
        margins as the canonical forward/backward parameterizer; it does not
        change the vehicle envelope or approve a trajectory by itself.
        """
        tangents = self._tangents(path.points)
        curvature = self._curvature_xy(path.points)
        horizontal_factor = max(float(np.linalg.norm(tangents[0, :2])), 1e-9)
        vertical_factor = max(abs(float(tangents[0, 2])), 1e-9)
        profile = self.dynamics.profile
        curvature_limit = math.sqrt(.8 * profile.max_lateral_accel / max(float(curvature[0]), 1e-12))
        return float(min(
            profile.max_horizontal_speed / horizontal_factor,
            profile.max_climb_rate / vertical_factor,
            profile.max_descent_rate / vertical_factor,
            curvature_limit / horizontal_factor,
        ))

    def parameterize(self, path: GeometricPath, *, start_time: float, initial_velocity: np.ndarray,
                     target_horizontal_speed: float, constraints: TrajectoryConstraints | None = None):
        """Return rich ``TrajectoryPoint`` objects; import avoids a cycle."""
        from .trajectory_generator import TrajectoryPoint

        constraints = constraints or TrajectoryConstraints()
        points = path.points
        tangents = self._tangents(points)
        segments = path.segment_lengths
        if np.any(segments <= 1e-9):
            raise ValueError("invalid_time_parameterization")
        curvature = self._curvature_xy(points)
        horizontal_factor = np.linalg.norm(tangents[:, :2], axis=1)
        vertical_factor = np.abs(tangents[:, 2])
        profile = self.dynamics.profile
        limits = np.full(len(points), np.inf, dtype=float)
        limits = np.minimum(limits, profile.max_horizontal_speed / np.maximum(horizontal_factor, 1e-9))
        limits = np.minimum(limits, profile.max_climb_rate / np.maximum(vertical_factor, 1e-9))
        limits = np.minimum(limits, profile.max_descent_rate / np.maximum(vertical_factor, 1e-9))
        for index, value in enumerate(curvature):
            # Leave a tangential-acceleration margin; curvature consumes the
            # remainder of the same finite horizontal acceleration envelope.
            horizontal_limit = math.sqrt(.8 * self.dynamics.profile.max_lateral_accel / max(value, 1e-12))
            if constraints.tangent_yaw:
                horizontal_limit = min(horizontal_limit, self.dynamics.max_feasible_speed_for_curvature(value, tangent_yaw=True))
            limits[index] = min(limits[index], horizontal_limit / max(horizontal_factor[index], 1e-9))
        requested_path_speed = max(0.1, target_horizontal_speed / max(horizontal_factor.mean(), 1e-9))
        limits = np.minimum(limits, requested_path_speed)
        speeds = limits.copy()
        start_speed = float(np.linalg.norm(np.asarray(initial_velocity, dtype=float)))
        speeds[0] = min(speeds[0], start_speed)
        for index, distance in enumerate(segments):
            speeds[index + 1] = min(speeds[index + 1], math.sqrt(max(0.0, speeds[index] ** 2 + 2.0 * (.57 * profile.max_horizontal_accel) * distance)))
        if constraints.end_horizontal_speed is not None:
            speeds[-1] = min(speeds[-1], constraints.end_horizontal_speed / max(horizontal_factor[-1], 1e-9))
        for index in range(len(segments) - 1, -1, -1):
            speeds[index] = min(speeds[index], math.sqrt(max(0.0, speeds[index + 1] ** 2 + 2.0 * (.57 * profile.max_horizontal_decel) * segments[index])))
        yaw = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0])) if constraints.tangent_yaw else np.full(len(points), np.nan)
        # Curvature estimates and forward/backward scalar passes are useful
        # upper bounds, but a discrete 3-D polyline can still exceed the
        # vector acceleration norm at a sharp sampled tangent.  The final
        # authority is the same dynamics validator used by SafetyFilter:
        # uniformly reduce the profile until its executable samples pass.
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
            result = [TrajectoryPoint(float(time), point, velocity, acceleration,
                                      None if not constraints.tangent_yaw else float(angle),
                                      None if not constraints.tangent_yaw else float(rate))
                      for time, point, velocity, acceleration, angle, rate in zip(times, points, velocities, accelerations, yaw, yaw_rates)]
            if not self.dynamics.validate_plan(result):
                return result
            speeds *= .88
        raise RuntimeError("could not time-parameterize a dynamically feasible trajectory")

    def estimate_duration(self, path: GeometricPath, *, initial_velocity: np.ndarray, target_horizontal_speed: float,
                          constraints: TrajectoryConstraints | None = None) -> float:
        """Fast conservative duration estimate for candidate scheduling.

        This deliberately does not allocate trajectory points or call the
        expensive discrete-plan validator.  The canonical ``parameterize``
        path remains mandatory when a candidate is actually committed.
        """
        constraints = constraints or TrajectoryConstraints()
        points = path.points
        tangents = self._tangents(points)
        segments = path.segment_lengths
        if np.any(segments <= 1e-9):
            raise ValueError("invalid_time_parameterization")
        profile = self.dynamics.profile
        horizontal_factor = np.linalg.norm(tangents[:, :2], axis=1)
        vertical_factor = np.abs(tangents[:, 2])
        curvature = self._curvature_xy(points)
        limits = np.full(len(points), max(0.1, float(target_horizontal_speed)), dtype=float)
        limits = np.minimum(limits, profile.max_horizontal_speed / np.maximum(horizontal_factor, 1e-9))
        limits = np.minimum(limits, profile.max_climb_rate / np.maximum(vertical_factor, 1e-9))
        limits = np.minimum(limits, profile.max_descent_rate / np.maximum(vertical_factor, 1e-9))
        limits = np.minimum(limits, np.sqrt(.8 * profile.max_lateral_accel / np.maximum(curvature, 1e-12)) /
                            np.maximum(horizontal_factor, 1e-9))
        speeds = limits.copy()
        speeds[0] = min(speeds[0], float(np.linalg.norm(np.asarray(initial_velocity, dtype=float))))
        for index, distance in enumerate(segments):
            speeds[index + 1] = min(speeds[index + 1], math.sqrt(max(0.0, speeds[index] ** 2 + 2.0 * (.57 * profile.max_horizontal_accel) * distance)))
        if constraints.end_horizontal_speed is not None:
            speeds[-1] = min(speeds[-1], constraints.end_horizontal_speed / max(horizontal_factor[-1], 1e-9))
        for index in range(len(segments) - 1, -1, -1):
            speeds[index] = min(speeds[index], math.sqrt(max(0.0, speeds[index + 1] ** 2 + 2.0 * (.57 * profile.max_horizontal_decel) * segments[index])))
        return float(np.sum(2.0 * segments / np.maximum(speeds[:-1] + speeds[1:], 1e-6)))
