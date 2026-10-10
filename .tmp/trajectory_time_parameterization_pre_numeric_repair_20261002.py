"""Geometry-to-executable trajectory conversion under one vehicle profile."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .vehicle_dynamics import VehicleDynamics
from .polynomial_trajectory import prime_quintic_bounds


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
    end_direction: np.ndarray | None = None
    end_curvature_vector: np.ndarray | None = None


class TrajectoryTimeParameterizer:
    """Forward/backward speed profiles with curvature and vertical limits."""

    def __init__(self, dynamics: VehicleDynamics, *, schema_version: str = "executable_trajectory_v2") -> None:
        if schema_version not in {"executable_trajectory_v2", "executable_trajectory_v3"}:
            raise ValueError(f"unsupported trajectory schema version {schema_version!r}")
        self.dynamics = dynamics
        self.schema_version = schema_version

    @staticmethod
    def _tangents(points: np.ndarray) -> np.ndarray:
        derivative = np.gradient(points, axis=0)
        lengths = np.linalg.norm(derivative, axis=1, keepdims=True)
        return derivative / np.maximum(lengths, 1e-12)

    @staticmethod
    def _curvature_xy(points: np.ndarray) -> np.ndarray:
        values = np.zeros(len(points), dtype=float)
        if len(points) > 2:
            xy = np.asarray(points[:, :2], dtype=float)
            first = xy[1:-1] - xy[:-2]
            second = xy[2:] - xy[:-2]
            middle_to_right = xy[2:] - xy[1:-1]
            a = np.linalg.norm(first, axis=1)
            b = np.linalg.norm(middle_to_right, axis=1)
            c = np.linalg.norm(second, axis=1)
            valid = np.minimum(np.minimum(a, b), c) > 1e-12
            interior = np.zeros(len(points) - 2, dtype=float)
            cross = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
            interior[valid] = 2.0 * cross[valid] / (a[valid] * b[valid] * c[valid])
            values[1:-1] = interior
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
        curvature_reserve = .25 if self.schema_version == "executable_trajectory_v3" else .8
        curvature_limit = math.sqrt(curvature_reserve * profile.max_lateral_accel / max(float(curvature[0]), 1e-12))
        return float(min(
            profile.max_horizontal_speed / horizontal_factor,
            profile.max_climb_rate / vertical_factor,
            profile.max_descent_rate / vertical_factor,
            curvature_limit / horizontal_factor,
        ))

    @staticmethod
    def _v3_vector(value: np.ndarray | None, name: str, *, default_zero: bool = False) -> np.ndarray:
        if value is None:
            if default_zero:
                return np.zeros(3, dtype=float)
            raise ValueError(f"{name} is required")
        result = np.asarray(value, dtype=float)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must have finite shape (3,)")
        return np.array(result, dtype=float, copy=True)

    def _v3_speed_caps(
        self,
        tangents: np.ndarray,
        curvature: np.ndarray,
        target_horizontal_speed: float,
    ) -> np.ndarray:
        """Return scalar path-speed caps from the existing vehicle bounds."""
        target = float(target_horizontal_speed)
        if not np.isfinite(target):
            raise ValueError("target_horizontal_speed must be finite")
        profile = self.dynamics.profile
        horizontal_factor = np.linalg.norm(tangents[:, :2], axis=1)
        vertical_factor = np.abs(tangents[:, 2])
        caps = np.full(len(tangents), np.inf, dtype=float)
        caps = np.minimum(
            caps,
            profile.max_horizontal_speed / np.maximum(horizontal_factor, 1e-9),
        )
        climb_caps = profile.max_climb_rate / np.maximum(vertical_factor, 1e-9)
        descent_caps = profile.max_descent_rate / np.maximum(vertical_factor, 1e-9)
        caps = np.minimum(caps, np.where(tangents[:, 2] >= 0.0, climb_caps, descent_caps))
        curvature_caps = np.sqrt(
            .8 * profile.max_lateral_accel / np.maximum(curvature, 1e-12)
        ) / np.maximum(horizontal_factor, 1e-9)
        caps = np.minimum(caps, curvature_caps)
        requested_path_speed = max(0.1, target / max(float(np.mean(horizontal_factor)), 1e-9))
        caps = np.minimum(caps, requested_path_speed)
        return caps

    def _v3_direction_speed_cap(self, direction: np.ndarray, curvature: float) -> float:
        profile = self.dynamics.profile
        horizontal_factor = max(float(np.linalg.norm(direction[:2])), 1e-9)
        vertical_factor = abs(float(direction[2]))
        cap = profile.max_horizontal_speed / horizontal_factor
        if vertical_factor > 1e-9:
            cap = min(
                cap,
                (profile.max_climb_rate if direction[2] >= 0.0 else profile.max_descent_rate)
                / vertical_factor,
            )
        curvature_cap = math.sqrt(.8 * profile.max_lateral_accel / max(float(curvature), 1e-12))
        return float(min(cap, curvature_cap / horizontal_factor))

    def _v3_speed_profile(
        self,
        caps: np.ndarray,
        segments: np.ndarray,
        start_speed: float,
        end_speed_cap: float | None,
        acceleration_reserve: float,
    ) -> np.ndarray | None:
        """Run bounded forward/backward scalar planning for one V3 trial."""
        profile = self.dynamics.profile
        tolerance = 1e-3
        acceleration = float(acceleration_reserve) * profile.max_horizontal_accel
        deceleration = float(acceleration_reserve) * profile.max_horizontal_decel
        speeds = np.asarray(caps, dtype=float).copy()
        if not np.all(np.isfinite(speeds)) or np.any(speeds < 0.0):
            return None
        if start_speed > speeds[0] + tolerance:
            return None
        speeds[0] = start_speed
        for index, distance in enumerate(segments):
            reachable = math.sqrt(max(0.0, speeds[index] ** 2 + 2.0 * acceleration * float(distance)))
            speeds[index + 1] = min(speeds[index + 1], reachable)
        if end_speed_cap is not None:
            speeds[-1] = min(speeds[-1], max(0.0, float(end_speed_cap)))
        for index in range(len(segments) - 1, -1, -1):
            reachable = math.sqrt(max(0.0, speeds[index + 1] ** 2 + 2.0 * deceleration * float(segments[index])))
            if index == 0 and reachable + tolerance < start_speed:
                return None
            speeds[index] = min(speeds[index], reachable)
        if speeds[0] + tolerance < start_speed:
            return None
        speeds[0] = start_speed
        return speeds

    def _v3_segments_feasible(self, segments) -> bool:
        """Check analytic segment bounds and a small finite-state sanity set."""
        segments = tuple(segments)
        try:
            prime_quintic_bounds(segments)
        except (TypeError, ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return False
        profile = self.dynamics.profile
        tolerance = 1e-3
        required_bounds = (
            "max_horizontal_speed",
            "max_horizontal_accel",
            "max_climb_rate",
            "max_descent_rate",
            "max_vertical_accel_up",
            "max_vertical_accel_down",
            "min_altitude",
            "max_altitude",
        )
        horizontal_accel_limit = min(
            float(profile.max_horizontal_accel),
            float(profile.max_horizontal_decel),
            float(profile.max_lateral_accel),
        )
        for segment in segments:
            bounds = segment.bounds()
            if any(name not in bounds for name in required_bounds):
                return False
            if any(not np.isfinite(float(bounds[name])) for name in required_bounds):
                return False
            if (
                bounds["max_horizontal_speed"] > profile.max_horizontal_speed + tolerance
                or bounds["max_climb_rate"] > profile.max_climb_rate + tolerance
                or bounds["max_descent_rate"] > profile.max_descent_rate + tolerance
                or bounds["max_horizontal_accel"] > horizontal_accel_limit + tolerance
                or bounds["max_vertical_accel_up"] > profile.max_vertical_accel_up + tolerance
                or bounds["max_vertical_accel_down"] > profile.max_vertical_accel_down + tolerance
                or bounds["min_altitude"] < -tolerance
                or bounds["max_altitude"]
                > float(profile.max_operating_altitude_agl_m) + tolerance
            ):
                return False
            sample_times = (
                float(segment.start_time),
                (float(segment.start_time) + float(segment.end_time)) / 2.0,
                float(segment.end_time),
            )
            for sample_time in sample_times:
                position, velocity, acceleration = segment.sample(float(sample_time))
                if not (
                    np.all(np.isfinite(position))
                    and np.all(np.isfinite(velocity))
                    and np.all(np.isfinite(acceleration))
                ):
                    return False
                if (
                    np.linalg.norm(velocity[:2]) > profile.max_horizontal_speed + tolerance
                    or velocity[2] > profile.max_climb_rate + tolerance
                    or -velocity[2] > profile.max_descent_rate + tolerance
                    or np.linalg.norm(acceleration[:2]) > horizontal_accel_limit + tolerance
                    or acceleration[2] > profile.max_vertical_accel_up + tolerance
                    or -acceleration[2] > profile.max_vertical_accel_down + tolerance
                    or position[2] < -tolerance
                    or position[2] > float(profile.max_operating_altitude_agl_m) + tolerance
                ):
                    return False
        return True

    def _v3_candidate(
        self,
        path: GeometricPath,
        *,
        start_time: float,
        initial_velocity: np.ndarray,
        initial_acceleration: np.ndarray,
        target_horizontal_speed: float,
        constraints: TrajectoryConstraints,
        caps: np.ndarray,
        acceleration_reserve: float,
        speed_cap_scale: float,
    ):
        from .polynomial_trajectory import QuinticSegment
        from .trajectory_generator import TrajectoryPoint

        points = path.points
        segments = path.segment_lengths
        tangents = self._tangents(points)
        curvature = self._curvature_xy(points)
        start_speed = float(np.linalg.norm(initial_velocity))
        scaled_caps = np.asarray(caps, dtype=float) * float(speed_cap_scale)
        end_direction = None
        if constraints.end_direction is not None:
            end_direction = self._v3_vector(constraints.end_direction, "end_direction")
            direction_norm = float(np.linalg.norm(end_direction))
            if direction_norm <= 1e-12:
                raise ValueError("end_direction must be nonzero")
            end_direction /= direction_norm
            scaled_caps[-1] = min(
                scaled_caps[-1], self._v3_direction_speed_cap(end_direction, curvature[-1])
            )
        end_speed_cap = None
        if constraints.end_horizontal_speed is not None:
            end_horizontal_speed = float(constraints.end_horizontal_speed)
            if not np.isfinite(end_horizontal_speed) or end_horizontal_speed < 0.0:
                raise ValueError("end_horizontal_speed must be finite and nonnegative")
            end_horizontal_factor = (
                float(np.linalg.norm(end_direction[:2])) if end_direction is not None
                else float(np.linalg.norm(tangents[-1, :2]))
            )
            end_speed_cap = end_horizontal_speed / max(end_horizontal_factor, 1e-9)
        speeds = self._v3_speed_profile(
            scaled_caps, segments, start_speed, end_speed_cap, acceleration_reserve,
        )
        if speeds is None:
            return None
        durations = 2.0 * segments / np.maximum(speeds[:-1] + speeds[1:], 1e-6)
        times = float(start_time) + np.concatenate(([0.0], np.cumsum(durations)))
        velocities = tangents * speeds[:, None]
        velocities[0] = initial_velocity
        if end_direction is not None:
            velocities[-1] = end_direction * speeds[-1]
        if len(velocities) > 2:
            accelerations = np.gradient(velocities, times, axis=0)
        else:
            accelerations = np.zeros_like(velocities)
        accelerations[0] = initial_acceleration
        if speeds[-1] <= 1e-9:
            accelerations[-1] = 0.0
        elif constraints.end_curvature_vector is not None:
            end_curvature = self._v3_vector(constraints.end_curvature_vector, "end_curvature_vector")
            accelerations[-1] = end_curvature * speeds[-1] ** 2
        else:
            accelerations[-1] = (velocities[-1] - velocities[-2]) / max(durations[-1], 1e-9)
        polynomial_segments = tuple(
            QuinticSegment.from_boundary(
                times[index], times[index + 1], points[index], velocities[index], accelerations[index],
                points[index + 1], velocities[index + 1], accelerations[index + 1],
            )
            for index in range(len(points) - 1)
        )
        if not self._v3_segments_feasible(polynomial_segments):
            return None
        first_state = polynomial_segments[0].sample(times[0])
        last_state = polynomial_segments[-1].sample(times[-1])
        if not (
            np.allclose(first_state[0], points[0], atol=1e-9, rtol=0.0)
            and np.allclose(first_state[1], initial_velocity, atol=1e-9, rtol=0.0)
            and np.allclose(first_state[2], initial_acceleration, atol=1e-9, rtol=0.0)
            and np.allclose(last_state[0], points[-1], atol=1e-9, rtol=0.0)
            and np.allclose(last_state[1], velocities[-1], atol=1e-9, rtol=0.0)
            and np.allclose(last_state[2], accelerations[-1], atol=1e-9, rtol=0.0)
        ):
            return None
        return [
            TrajectoryPoint(float(time), point, velocity, acceleration)
            for time, point, velocity, acceleration in zip(times, points, velocities, accelerations)
        ]

    def _parameterize_v3(
        self,
        path: GeometricPath,
        *,
        start_time: float,
        initial_velocity: np.ndarray,
        initial_acceleration: np.ndarray | None,
        target_horizontal_speed: float,
        constraints: TrajectoryConstraints | None,
    ):
        points = path.points
        segments = path.segment_lengths
        if np.any(segments <= 1e-9):
            raise ValueError("invalid_time_parameterization")
        initial = self._v3_vector(initial_velocity, "initial_velocity")
        initial_acceleration_value = self._v3_vector(
            initial_acceleration, "initial_acceleration", default_zero=True,
        )
        constraints = constraints or TrajectoryConstraints()
        if constraints.tangent_yaw:
            raise ValueError(
                "V3 tangent_yaw is unsupported; V3 yaw parameterization is not qualified"
            )
        tangents = self._tangents(points)
        curvature = self._curvature_xy(points)
        caps = self._v3_speed_caps(tangents, curvature, target_horizontal_speed)
        for acceleration_reserve in (.4, .25, .15, .08):
            for speed_cap_scale in (1.0, .8, .6, .4):
                candidate = self._v3_candidate(
                    path,
                    start_time=float(start_time),
                    initial_velocity=initial,
                    initial_acceleration=initial_acceleration_value,
                    target_horizontal_speed=float(target_horizontal_speed),
                    constraints=constraints,
                    caps=caps,
                    acceleration_reserve=acceleration_reserve,
                    speed_cap_scale=speed_cap_scale,
                )
                if candidate is not None:
                    return candidate
        raise RuntimeError(
            "dynamically_infeasible_boundary_state: no bounded V3 candidate satisfies "
            "the boundary state and vehicle dynamics"
        )

    def parameterize(self, path: GeometricPath, *, start_time: float, initial_velocity: np.ndarray,
                     target_horizontal_speed: float, constraints: TrajectoryConstraints | None = None,
                     initial_acceleration: np.ndarray | None = None):
        """Return rich ``TrajectoryPoint`` objects; import avoids a cycle."""
        if self.schema_version == "executable_trajectory_v3":
            return self._parameterize_v3(
                path,
                start_time=start_time,
                initial_velocity=initial_velocity,
                initial_acceleration=initial_acceleration,
                target_horizontal_speed=target_horizontal_speed,
                constraints=constraints,
            )
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
        # Leave a tangential-acceleration margin; curvature consumes the
        # remainder of the same finite horizontal acceleration envelope.  The
        # tangent-yaw branch still calls the dynamics helper per sample because
        # it adds the yaw-rate bound.  The ordinary branch has only the scalar
        # curvature envelope and can be evaluated in one NumPy pass.
        if constraints.tangent_yaw:
            for index, value in enumerate(curvature):
                horizontal_limit = math.sqrt(.8 * profile.max_lateral_accel / max(value, 1e-12))
                horizontal_limit = min(horizontal_limit, self.dynamics.max_feasible_speed_for_curvature(value, tangent_yaw=True))
                limits[index] = min(limits[index], horizontal_limit / max(horizontal_factor[index], 1e-9))
        else:
            horizontal_limits = np.sqrt(.8 * profile.max_lateral_accel / np.maximum(curvature, 1e-12))
            limits = np.minimum(limits, horizontal_limits / np.maximum(horizontal_factor, 1e-9))
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
            times_array = np.asarray(times, dtype=float)
            velocities = tangents * speeds[:, None]
            velocities[0] = np.asarray(initial_velocity, dtype=float)
            yaw_rates = np.zeros(len(points))
            if constraints.tangent_yaw:
                yaw_rates[1:] = np.diff(yaw) / np.diff(times_array)
                yaw_rates[0] = yaw_rates[1]
            if self.dynamics.samples_are_feasible(
                times_array, points, velocities,
                None if not constraints.tangent_yaw else yaw_rates,
            ):
                # Acceleration is part of the returned executable point but
                # is not needed by the feasibility predicate.  Construct it
                # only for the first passing refinement.
                accelerations = np.empty_like(velocities)
                accelerations[1:] = np.diff(velocities, axis=0) / np.diff(times_array)[:, None]
                accelerations[0] = accelerations[1]
                return [TrajectoryPoint(float(time), point, velocity, acceleration,
                                        None if not constraints.tangent_yaw else float(angle),
                                        None if not constraints.tangent_yaw else float(rate))
                        for time, point, velocity, acceleration, angle, rate in
                        zip(times, points, velocities, accelerations, yaw, yaw_rates)]
            speeds *= .88
        raise RuntimeError("could not time-parameterize a dynamically feasible trajectory")

    def estimate_duration(self, path: GeometricPath, *, initial_velocity: np.ndarray, target_horizontal_speed: float,
                          constraints: TrajectoryConstraints | None = None,
                          initial_acceleration: np.ndarray | None = None) -> float:
        """Fast conservative duration estimate for candidate scheduling.

        This deliberately does not allocate trajectory points or call the
        expensive discrete-plan validator.  The canonical ``parameterize``
        path remains mandatory when a candidate is actually committed.
        """
        if self.schema_version == "executable_trajectory_v3":
            points = self.parameterize(
                path,
                start_time=0.0,
                initial_velocity=initial_velocity,
                initial_acceleration=initial_acceleration,
                target_horizontal_speed=target_horizontal_speed,
                constraints=constraints,
            )
            return float(points[-1].t - points[0].t)
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
