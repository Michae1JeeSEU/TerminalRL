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

    def __init__(self, profile: VehiclePerformanceProfile) -> None:
        self.profile = profile

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
        if dt <= 0.0:
            raise ValueError("dynamics dt must be positive")
        target = min(max(target_speed, 0.0), self.ring_speed_limit(radius))
        lower = max(0.0, current_speed - self.profile.max_horizontal_decel * dt)
        upper = min(target, current_speed + self.profile.max_horizontal_accel * dt) if target >= current_speed else target
        direction = 1.0 if target >= current_speed else -1.0
        def feasible(speed: float) -> bool:
            # Check the exact discrete velocity-vector change that the
            # executor will use.  The continuous combined formula is its
            # limiting case, while this catches the small chord/tangent
            # discrepancy at finite physics timesteps.
            angle = .5 * (current_speed + speed) * dt / radius
            previous = np.asarray((0.0, current_speed), dtype=float)
            current = np.asarray((-math.sin(angle) * speed, math.cos(angle) * speed), dtype=float)
            return float(np.linalg.norm(current - previous)) / dt <= self.profile.max_lateral_accel + 1e-9
        if direction < 0.0:
            if feasible(lower):
                return lower
            lo, hi = lower, current_speed
            for _ in range(48):
                mid = .5 * (lo + hi)
                if feasible(mid): hi = mid
                else: lo = mid
            return hi
        if feasible(upper):
            return upper
        lo, hi = lower, upper
        for _ in range(48):
            mid = .5 * (lo + hi)
            if feasible(mid): lo = mid
            else: hi = mid
        return lo

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

    def validate_plan(self, points: Iterable) -> list[str]:
        rows = list(points)
        if len(rows) < 2:
            return ["invalid_time_parameterization"]
        errors: list[str] = []
        for previous, current in zip(rows, rows[1:]):
            errors.extend(self.validate_transition(previous, current))
        return list(dict.fromkeys(errors))
