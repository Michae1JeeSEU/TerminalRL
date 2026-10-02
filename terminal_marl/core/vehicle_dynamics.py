"""Single authoritative performance-constrained point-mass dynamics model."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .vehicle_profiles import VehiclePerformanceProfile


@dataclass(frozen=True)
class DynamicsState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float = 0.0
    yaw_rate: float = 0.0

    @property
    def horizontal_speed(self) -> float:
        return float(np.linalg.norm(self.velocity[:2]))

    @property
    def vertical_speed(self) -> float:
        return float(self.velocity[2])


@dataclass(frozen=True)
class RingMotion:
    """One finite-acceleration step on a circular terminal holding ring."""

    theta: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    speed: float


class VehicleDynamics:
    """Validation and integration shared by planning, safety and execution."""

    def __init__(self, profile: VehiclePerformanceProfile, *, continuous_curve_limits: bool = False) -> None:
        self.profile = profile
        self.continuous_curve_limits = bool(continuous_curve_limits)

    def max_feasible_speed_for_curvature(self, curvature: float, *, tangent_yaw: bool = False) -> float:
        if curvature <= 1e-12:
            return self.profile.max_horizontal_speed
        lateral = math.sqrt(self.profile.max_lateral_accel / curvature)
        yaw = math.inf if not tangent_yaw else math.radians(self.profile.max_yaw_rate_deg_s) / curvature
        return min(self.profile.max_horizontal_speed, lateral, yaw)

    def ring_speed_limit(self, radius: float, tangential_acceleration: float = 0.0) -> float:
        if radius <= 0.0:
            raise ValueError("ring radius must be positive")
        remaining = self.profile.max_lateral_accel ** 2 - tangential_acceleration ** 2
        # sqrt(a_t^2 + (v^2/r)^2) <= a_xy_max
        # therefore v <= sqrt(r * sqrt(a_xy_max^2 - a_t^2)).
        return min(self.profile.max_horizontal_speed, math.sqrt(radius * math.sqrt(max(0.0, remaining))))

    def limited_tangential_speed(self, current_speed: float, target_speed: float, radius: float, dt: float) -> float:
        """Return the next safe ring speed without a production solver loop.

        The continuous acceleration envelope reserves centripetal load before
        applying a bounded tangential update.  A short discrete check guards
        the chord-versus-tangent difference of the actual executor; it is a
        correction only, never the previous 48-iteration default path.
        """
        if dt <= 0.0:
            raise ValueError("dynamics dt must be positive")
        current = min(max(float(current_speed), 0.0), self.ring_speed_limit(radius))
        target = min(max(float(target_speed), 0.0), self.ring_speed_limit(radius))
        if abs(target - current) <= 1e-12:
            return current
        direction = 1.0 if target >= current else -1.0
        nominal_limit = self.profile.max_horizontal_accel if direction > 0.0 else self.profile.max_horizontal_decel
        preliminary = min(target, current + nominal_limit * dt) if direction > 0.0 else max(target, current - nominal_limit * dt)
        # The old bound used ``max(current, preliminary)`` for the radial
        # term.  At the inner-ring speed limit that makes the centripetal
        # term consume the complete acceleration budget even when the
        # finite-step velocity change is safe, so a braking/acceleration
        # request is silently turned into a hold.  Start with the full
        # target-directed longitudinal interval and let the exact finite-step
        # predicate below select its feasible boundary.
        candidate = preliminary

        def feasible(speed: float) -> bool:
            tangential_acceleration = (speed - current) / dt
            radial_speed = max(current, speed) if self.continuous_curve_limits else speed
            centripetal = radial_speed * radial_speed / radius
            if math.hypot(tangential_acceleration, centripetal) > self.profile.max_lateral_accel + 1e-9:
                return False
            # Check the exact discrete velocity-vector change that the
            # executor will use.  The continuous combined formula is its
            # limiting case, while this catches the small chord/tangent
            # discrepancy at finite physics timesteps.
            angle = .5 * (current + speed) * dt / radius
            delta_x = -math.sin(angle) * speed
            delta_y = math.cos(angle) * speed - current
            return math.hypot(delta_x, delta_y) / dt <= self.profile.max_lateral_accel + 1e-9
        if feasible(candidate):
            return candidate
        # Only a finite-step boundary needs refinement.  A bounded search
        # preserves a feasible endpoint without reintroducing the expensive
        # trajectory-level solver loop used by the old implementation.
        if direction > 0.0:
            lo, hi = current, candidate
            for _ in range(8):
                middle = .5 * (lo + hi)
                if feasible(middle):
                    lo = middle
                else:
                    hi = middle
            return lo
        lo, hi = candidate, current
        for _ in range(8):
            middle = .5 * (lo + hi)
            if feasible(middle):
                hi = middle
            else:
                lo = middle
        return hi

    def ring_motion_step(self, ring, theta: float, current_speed: float, target_speed: float, dt: float) -> RingMotion:
        """Integrate a bounded holder step shared by execution and prediction.

        The tangential update is constrained jointly with the centripetal
        term, so ring speed commands cannot create a physically impossible
        instantaneous acceleration.  ``ring`` is deliberately duck-typed to
        keep this core model independent of the airspace implementation.
        """
        if dt <= 0.0:
            raise ValueError("dynamics dt must be positive")
        radius = float(ring.radius)
        speed0 = max(0.0, float(current_speed))
        speed1 = self.limited_tangential_speed(speed0, float(target_speed), radius, dt)
        theta1 = float(theta) + .5 * (speed0 + speed1) * dt / radius
        tangent = np.asarray((-math.sin(theta1), math.cos(theta1), 0.0), dtype=float)
        radial = np.asarray((-math.cos(theta1), -math.sin(theta1), 0.0), dtype=float)
        tangential_acceleration = (speed1 - speed0) / dt
        return RingMotion(
            theta=theta1,
            position=np.asarray((radius * math.cos(theta1), radius * math.sin(theta1), float(ring.height)), dtype=float),
            velocity=tangent * speed1,
            acceleration=tangent * tangential_acceleration + radial * (speed1 * speed1 / radius),
            speed=speed1,
        )

    def validate_state(self, state: DynamicsState) -> list[str]:
        errors: list[str] = []
        tolerance = 1e-3  # interpolation / floating-point validation tolerance
        horizontal_speed = state.horizontal_speed
        if horizontal_speed > self.profile.max_horizontal_speed + tolerance:
            errors.append("speed_limit")
        if state.vertical_speed > self.profile.max_climb_rate + tolerance:
            errors.append("climb_rate_limit")
        if -state.vertical_speed > self.profile.max_descent_rate + tolerance:
            errors.append("descent_rate_limit")
        horizontal_acceleration = float(np.linalg.norm(state.acceleration[:2]))
        if horizontal_acceleration > self.profile.max_horizontal_accel + tolerance:
            errors.append("horizontal_acceleration_limit")
        if state.acceleration[2] > self.profile.max_vertical_accel_up + tolerance:
            errors.append("vertical_acceleration_limit")
        if -state.acceleration[2] > self.profile.max_vertical_accel_down + tolerance:
            errors.append("vertical_acceleration_limit")
        if abs(state.yaw_rate) > math.radians(self.profile.max_yaw_rate_deg_s) + tolerance:
            errors.append("yaw_rate_limit")
        return errors

    def validate_transition(self, previous, current) -> list[str]:
        dt = float(current.t - previous.t)
        if not np.isfinite(dt) or dt <= 0.0:
            return ["invalid_time_parameterization"]
        velocity = np.asarray(current.velocity, dtype=float)
        acceleration = (velocity - np.asarray(previous.velocity, dtype=float)) / dt
        state = DynamicsState(np.asarray(current.position, dtype=float), velocity, acceleration,
                              0.0 if current.yaw is None else current.yaw,
                              0.0 if current.yaw_rate is None else current.yaw_rate)
        errors = self.validate_state(state)
        # A serialized executable plan carries velocity explicitly.  Also
        # constrain its sampled displacement so a stale/forged zero-velocity
        # record cannot bypass the profile limit.
        segment_velocity = (np.asarray(current.position, dtype=float) - np.asarray(previous.position, dtype=float)) / dt
        segment_state = DynamicsState(np.asarray(current.position, dtype=float), segment_velocity, acceleration)
        for error in self.validate_state(segment_state):
            if error not in errors:
                errors.append(error)
        return errors

    def samples_are_feasible(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        yaw_rates: np.ndarray | None = None,
    ) -> bool:
        """Vectorized feasibility predicate for in-memory planner samples.

        The time parameterizer previously allocated immutable trajectory
        objects for every rejected speed-profile refinement and then routed
        them through ``validate_plan`` one transition at a time.  Candidate
        search only needs the same pass/fail result at that stage.  Keep the
        authoritative, diagnostic ``validate_plan`` API unchanged and apply
        its exact numeric inequalities directly to the planner arrays here.
        """
        times = np.asarray(times, dtype=float)
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        if len(times) < 2:
            return False
        dt = np.diff(times)
        if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
            return False
        current_velocity = velocities[1:]
        segment_velocity = np.diff(positions, axis=0) / dt[:, None]
        acceleration = np.diff(velocities, axis=0) / dt[:, None]
        tolerance = 1e-3
        profile = self.profile

        def velocity_limits_exceeded(values: np.ndarray) -> bool:
            return bool(
                np.any(np.linalg.norm(values[:, :2], axis=1) > profile.max_horizontal_speed + tolerance)
                or np.any(values[:, 2] > profile.max_climb_rate + tolerance)
                or np.any(-values[:, 2] > profile.max_descent_rate + tolerance)
            )

        if velocity_limits_exceeded(current_velocity) or velocity_limits_exceeded(segment_velocity):
            return False
        if (
            np.any(np.linalg.norm(acceleration[:, :2], axis=1) > profile.max_horizontal_accel + tolerance)
            or np.any(acceleration[:, 2] > profile.max_vertical_accel_up + tolerance)
            or np.any(-acceleration[:, 2] > profile.max_vertical_accel_down + tolerance)
        ):
            return False
        if yaw_rates is not None:
            values = np.asarray(yaw_rates, dtype=float)[1:]
            if np.any(np.abs(values) > math.radians(profile.max_yaw_rate_deg_s) + tolerance):
                return False
        return True

    def validate_plan(self, points: Iterable) -> list[str]:
        rows = list(points)
        if len(rows) < 2:
            return ["invalid_time_parameterization"]

        # Normal trajectory points already carry the complete sampled arrays
        # needed by the vectorized predicate.  Use it as the cheap pass path;
        # an infeasible result falls through to the scalar validator so the
        # public diagnostic error order remains authoritative.
        try:
            times = np.asarray([row.t for row in rows], dtype=float)
            positions = np.asarray([row.position for row in rows], dtype=float)
            velocities = np.asarray([row.velocity for row in rows], dtype=float)
            yaw_rates = np.asarray(
                [0.0 if row.yaw_rate is None else row.yaw_rate for row in rows],
                dtype=float,
            )
            if self.samples_are_feasible(times, positions, velocities, yaw_rates):
                return []
        except (AttributeError, IndexError, TypeError, ValueError):
            # Preserve the existing scalar diagnostic fallback for malformed
            # or otherwise non-standard point iterables.
            pass

        errors: list[str] = []
        for previous, current in zip(rows, rows[1:]):
            errors.extend(self.validate_transition(previous, current))
        return list(dict.fromkeys(errors))

    def validate_trajectory(self, plan) -> list[str]:
        """Validate the executable representation selected by ``plan``.

        V2 plans retain the exact point validator and its diagnostic order.
        V3 plans additionally certify each analytic polynomial segment, because
        node values alone cannot expose an interior speed or acceleration peak.
        The method is deliberately fail-closed for malformed V3 metadata,
        segment data, bounds, or boundary samples.
        """
        metadata = getattr(plan, "metadata", None)
        if not isinstance(metadata, dict):
            return ["trajectory_schema_mismatch"]
        schema = metadata.get("trajectory_schema_version", "executable_trajectory_v2")
        if schema == "executable_trajectory_v2":
            return self.validate_plan(plan.points)
        if schema != "executable_trajectory_v3":
            return ["trajectory_schema_mismatch"]

        tolerance = 1e-3  # shared interpolation / floating-point tolerance
        errors: list[str] = []

        def add(error: str) -> None:
            if error not in errors:
                errors.append(error)

        try:
            points = tuple(plan.points)
            segments = tuple(plan.polynomial_segments)
        except (AttributeError, TypeError, ValueError):
            return ["invalid_time_parameterization"]
        if len(points) < 2 or not segments or len(segments) != len(points) - 1:
            return ["invalid_time_parameterization"]

        try:
            point_times = np.asarray([point.t for point in points], dtype=float)
            positions = np.asarray([point.position for point in points], dtype=float)
            velocities = np.asarray([point.velocity for point in points], dtype=float)
            accelerations = np.asarray([point.acceleration for point in points], dtype=float)
            plan_start = float(plan.start_time)
            plan_end = float(plan.end_time)
        except (AttributeError, TypeError, ValueError):
            return ["invalid_time_parameterization"]

        if (
            point_times.ndim != 1
            or positions.shape != (len(points), 3)
            or velocities.shape != (len(points), 3)
            or accelerations.shape != (len(points), 3)
            or not np.all(np.isfinite(point_times))
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(velocities))
            or not np.all(np.isfinite(accelerations))
            or not np.isfinite(plan_start)
            or not np.isfinite(plan_end)
            or plan_end <= plan_start
            or np.any(np.diff(point_times) <= 0.0)
            or abs(plan_start - point_times[0]) > tolerance
            or abs(plan_end - point_times[-1]) > tolerance
        ):
            return ["invalid_time_parameterization"]

        required_bounds = (
            "max_horizontal_speed",
            "max_climb_rate",
            "max_descent_rate",
            "max_horizontal_accel",
            "max_vertical_accel_up",
            "max_vertical_accel_down",
            "min_altitude",
            "max_altitude",
        )
        try:
            horizontal_acceleration_limit = min(
                float(self.profile.max_horizontal_accel),
                float(self.profile.max_horizontal_decel),
                float(self.profile.max_lateral_accel),
            )
            limits = {
                "max_horizontal_speed": float(self.profile.max_horizontal_speed),
                "max_climb_rate": float(self.profile.max_climb_rate),
                "max_descent_rate": float(self.profile.max_descent_rate),
                "max_horizontal_accel": horizontal_acceleration_limit,
                "max_vertical_accel_up": float(self.profile.max_vertical_accel_up),
                "max_vertical_accel_down": float(self.profile.max_vertical_accel_down),
                "min_altitude": 0.0,
                "max_altitude": float(self.profile.max_operating_altitude_agl_m),
            }
        except (AttributeError, TypeError, ValueError):
            return ["invalid_time_parameterization"]
        if not all(np.isfinite(value) and value >= 0.0 for value in limits.values()):
            return ["invalid_time_parameterization"]

        yaw_rate_limit = math.radians(float(self.profile.max_yaw_rate_deg_s))
        for point in points:
            for field_name in ("yaw", "yaw_rate"):
                value = getattr(point, field_name, None)
                if value is None:
                    continue
                try:
                    scalar = float(value)
                except (TypeError, ValueError, OverflowError):
                    return ["invalid_time_parameterization"]
                if not np.isfinite(scalar):
                    return ["invalid_time_parameterization"]
                if field_name == "yaw_rate" and abs(scalar) > yaw_rate_limit + tolerance:
                    add("yaw_rate_limit")

        boundary_states: list[tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        for index, segment in enumerate(segments):
            try:
                start_time = float(segment.start_time)
                end_time = float(segment.end_time)
                bounds = segment.bounds()
            except (AttributeError, TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
                add("invalid_time_parameterization")
                continue

            if (
                not np.isfinite(start_time)
                or not np.isfinite(end_time)
                or end_time <= start_time
                or abs(start_time - point_times[index]) > tolerance
                or abs(end_time - point_times[index + 1]) > tolerance
            ):
                add("invalid_time_parameterization")
                add("trajectory_segment_mismatch")

            if not isinstance(bounds, dict):
                add("trajectory_bounds_nonfinite")
            else:
                for key in required_bounds:
                    try:
                        value = float(bounds[key])
                    except (KeyError, TypeError, ValueError):
                        add("trajectory_bounds_nonfinite")
                        continue
                    if not np.isfinite(value):
                        add("trajectory_bounds_nonfinite")
                        continue
                    if key == "min_altitude":
                        if value < -tolerance:
                            add("altitude_limit")
                        continue
                    if key == "max_altitude":
                        if value > limits[key] + tolerance:
                            add("altitude_limit")
                        continue
                    if value < 0.0:
                        add("trajectory_bounds_nonfinite")
                        continue
                    if value > limits[key] + tolerance:
                        if key == "max_horizontal_speed":
                            add("speed_limit")
                        elif key == "max_climb_rate":
                            add("climb_rate_limit")
                        elif key == "max_descent_rate":
                            add("descent_rate_limit")
                        elif key == "max_horizontal_accel":
                            add("horizontal_acceleration_limit")
                        else:
                            add("vertical_acceleration_limit")

            try:
                start_state = tuple(np.asarray(value, dtype=float) for value in segment.sample(start_time))
                end_state = tuple(np.asarray(value, dtype=float) for value in segment.sample(end_time))
            except (AttributeError, TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
                add("trajectory_segment_mismatch")
                continue
            if len(start_state) != 3 or len(end_state) != 3:
                add("trajectory_segment_mismatch")
                continue
            if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (*start_state, *end_state)):
                add("trajectory_segment_mismatch")
                continue
            boundary_states.append((start_state, end_state))
            for state, expected in (
                (start_state, (positions[index], velocities[index], accelerations[index])),
                (end_state, (positions[index + 1], velocities[index + 1], accelerations[index + 1])),
            ):
                if any(not np.allclose(actual, target, rtol=0.0, atol=tolerance) for actual, target in zip(state, expected)):
                    add("trajectory_segment_mismatch")

        for previous, current in zip(boundary_states, boundary_states[1:]):
            if any(
                not np.allclose(left, right, rtol=0.0, atol=tolerance)
                for left, right in zip(previous[1], current[0])
            ):
                add("trajectory_continuity")
        return errors
