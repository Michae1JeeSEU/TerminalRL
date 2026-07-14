"""Hardware-agnostic executable trajectory generation for terminal intentions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class TrajectoryPoint:
    """A timed position and yaw waypoint."""

    x: float
    y: float
    z: float
    yaw: float
    t: float


@dataclass
class TrajectoryPlan:
    """A complete trajectory intent suitable for simulation or dry-run use."""

    uav_id: str
    mode: str
    points: list[TrajectoryPoint]
    metadata: dict[str, Any] = field(default_factory=dict)


class TrajectoryGenerator:
    """Convert terminal-stage intentions into continuous waypoint plans."""

    def __init__(
        self,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ring_direction: int = 1,
        max_speed: float = math.inf,
        max_acceleration: float = math.inf,
        height_min: float = -math.inf,
        height_max: float = math.inf,
        ring_tolerance: float = 1e-6,
    ) -> None:
        if ring_direction not in (-1, 1):
            raise ValueError("ring_direction must be -1 or 1")
        if max_speed <= 0.0 or max_acceleration <= 0.0:
            raise ValueError("speed and acceleration limits must be positive")
        if height_min > height_max:
            raise ValueError("height_min must not exceed height_max")
        if ring_tolerance < 0.0:
            raise ValueError("ring_tolerance must be non-negative")

        center_array = np.asarray(center, dtype=np.float64)
        if center_array.shape != (3,) or not np.all(np.isfinite(center_array)):
            raise ValueError("center must contain three finite coordinates")

        self.center = center_array
        self.ring_direction = ring_direction
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.height_min = float(height_min)
        self.height_max = float(height_max)
        self.ring_tolerance = float(ring_tolerance)

    def ring_hold(
        self,
        uav_state: object,
        ring: object,
        duration: float,
        dt: float,
    ) -> TrajectoryPlan:
        """Generate a constant-radius, constant-height holding orbit."""

        times = self._sample_times(duration, dt)
        position = self._position(uav_state)
        radius = float(self._value(ring, "radius"))
        height = float(self._value(ring, "height"))
        speed = float(self._value(ring, "speed", default=self._value(uav_state, "speed", default=0.0)))
        if radius <= 0.0 or speed < 0.0:
            raise ValueError("ring radius must be positive and ring speed non-negative")

        relative = position[:2] - self.center[:2]
        angle0 = (
            math.atan2(float(relative[1]), float(relative[0]))
            if np.linalg.norm(relative) > 1e-12
            else float(self._value(uav_state, "heading", default=0.0)) - self.ring_direction * math.pi / 2.0
        )
        angular_speed = speed / radius
        points = []
        for t in times:
            angle = angle0 + self.ring_direction * angular_speed * float(t)
            yaw = self._wrap_angle(angle + self.ring_direction * math.pi / 2.0)
            points.append(
                TrajectoryPoint(
                    x=float(self.center[0] + radius * math.cos(angle)),
                    y=float(self.center[1] + radius * math.sin(angle)),
                    z=height,
                    yaw=yaw,
                    t=float(t),
                )
            )

        plan = TrajectoryPlan(
            uav_id=self._uav_id(uav_state),
            mode="ring_hold",
            points=points,
            metadata={
                "duration": float(duration),
                "dt": float(dt),
                "ring_index": self._value(ring, "index", default=None),
                "radius": radius,
                "height": height,
                "speed": speed,
            },
        )
        return self._finalize(plan)

    def ring_transfer(
        self,
        uav_state: object,
        from_ring: object,
        to_ring: object,
        duration: float,
        dt: float,
    ) -> TrajectoryPlan:
        """Generate a smooth radial and vertical transfer between two rings."""

        times = self._sample_times(duration, dt)
        position = self._position(uav_state)
        relative = position[:2] - self.center[:2]
        angle0 = math.atan2(float(relative[1]), float(relative[0]))

        nominal_radius0 = float(self._value(from_ring, "radius"))
        radius1 = float(self._value(to_ring, "radius"))
        actual_radius0 = float(np.linalg.norm(relative))
        radius0 = actual_radius0 if actual_radius0 > 1e-9 else nominal_radius0
        height0 = float(position[2])
        nominal_height0 = float(self._value(from_ring, "height"))
        height1 = float(self._value(to_ring, "height"))
        speed0 = float(self._value(from_ring, "speed", default=self._value(uav_state, "speed", default=0.0)))
        speed1 = float(self._value(to_ring, "speed", default=speed0))
        if radius0 <= 0.0 or radius1 <= 0.0 or min(speed0, speed1) < 0.0:
            raise ValueError("ring radii must be positive and speeds non-negative")

        points = []
        for t in times:
            fraction = float(t / duration)
            blend = self._smoothstep(fraction)
            radius = radius0 + (radius1 - radius0) * blend
            height = height0 + (height1 - height0) * blend
            average_angular_speed = 0.5 * (speed0 / radius0 + speed1 / radius1)
            angle = angle0 + self.ring_direction * average_angular_speed * float(t)
            yaw = self._wrap_angle(angle + self.ring_direction * math.pi / 2.0)
            points.append(
                TrajectoryPoint(
                    x=float(self.center[0] + radius * math.cos(angle)),
                    y=float(self.center[1] + radius * math.sin(angle)),
                    z=float(height),
                    yaw=yaw,
                    t=float(t),
                )
            )

        plan = TrajectoryPlan(
            uav_id=self._uav_id(uav_state),
            mode="ring_transfer",
            points=points,
            metadata={
                "duration": float(duration),
                "dt": float(dt),
                "from_ring": self._value(from_ring, "index", default=None),
                "to_ring": self._value(to_ring, "index", default=None),
                "start_radius": radius0,
                "target_radius": radius1,
                "nominal_source_radius": nominal_radius0,
                "start_height": height0,
                "target_height": height1,
                "nominal_source_height": nominal_height0,
            },
        )
        return self._finalize(plan)

    def final_approach(
        self,
        uav_state: object,
        landing_pad: object,
        duration: float,
        dt: float,
    ) -> TrajectoryPlan:
        """Generate a smooth path from the current state to an approach point."""

        target = self._pad_position(landing_pad, approach=True)
        return self._point_to_point_plan(
            uav_state,
            target,
            mode="final_approach",
            duration=duration,
            dt=dt,
        )

    def landing(
        self,
        uav_state: object,
        landing_pad: object,
        duration: float,
        dt: float,
    ) -> TrajectoryPlan:
        """Generate a smooth touchdown path to the landing-pad position."""

        target = self._pad_position(landing_pad, approach=False)
        return self._point_to_point_plan(
            uav_state,
            target,
            mode="landing",
            duration=duration,
            dt=dt,
        )

    def emergency_hover(
        self,
        uav_state: object,
        duration: float,
        dt: float,
    ) -> TrajectoryPlan:
        """Generate a stationary hover plan at the current pose."""

        times = self._sample_times(duration, dt)
        position = self._position(uav_state)
        yaw = float(self._value(uav_state, "heading", "yaw", default=0.0))
        points = [
            TrajectoryPoint(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
                yaw=self._wrap_angle(yaw),
                t=float(t),
            )
            for t in times
        ]
        plan = TrajectoryPlan(
            uav_id=self._uav_id(uav_state),
            mode="emergency_hover",
            points=points,
            metadata={"duration": float(duration), "dt": float(dt)},
        )
        return self._finalize(plan)

    def validate_plan(self, plan: TrajectoryPlan) -> list[str]:
        """Return explainable time, kinematic, height, and geometry violations."""

        violations: list[str] = []
        if not plan.points:
            return ["empty_trajectory"]

        values = np.asarray(
            [[point.x, point.y, point.z, point.yaw, point.t] for point in plan.points],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            violations.append("non_finite_waypoint")

        times = values[:, 4]
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            violations.append("non_monotonic_time")
        if np.any(values[:, 2] < self.height_min) or np.any(values[:, 2] > self.height_max):
            violations.append("height_bounds")

        if len(values) > 1:
            intervals = np.diff(times)
            if np.all(intervals > 0.0):
                velocities = np.diff(values[:, :3], axis=0) / intervals[:, None]
                speeds = np.linalg.norm(velocities, axis=1)
                if np.any(speeds > self.max_speed + 1e-9):
                    violations.append("speed_limit")
                if len(velocities) > 1:
                    acceleration_dt = 0.5 * (intervals[:-1] + intervals[1:])
                    accelerations = np.diff(velocities, axis=0) / acceleration_dt[:, None]
                    if np.any(np.linalg.norm(accelerations, axis=1) > self.max_acceleration + 1e-9):
                        violations.append("acceleration_limit")

        if plan.mode == "ring_hold" and "radius" in plan.metadata:
            radii = np.linalg.norm(values[:, :2] - self.center[:2], axis=1)
            if not np.allclose(
                radii,
                float(plan.metadata["radius"]),
                atol=self.ring_tolerance,
                rtol=0.0,
            ):
                violations.append("ring_radius")
            if not np.allclose(
                values[:, 2],
                float(plan.metadata["height"]),
                atol=self.ring_tolerance,
                rtol=0.0,
            ):
                violations.append("ring_height")
        return violations

    def _point_to_point_plan(
        self,
        uav_state: object,
        target: np.ndarray,
        *,
        mode: str,
        duration: float,
        dt: float,
    ) -> TrajectoryPlan:
        times = self._sample_times(duration, dt)
        start = self._position(uav_state)
        delta = target - start
        horizontal_yaw = (
            math.atan2(float(delta[1]), float(delta[0]))
            if np.linalg.norm(delta[:2]) > 1e-12
            else float(self._value(uav_state, "heading", "yaw", default=0.0))
        )
        points = []
        for t in times:
            blend = self._smoothstep(float(t / duration))
            position = start + delta * blend
            points.append(
                TrajectoryPoint(
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                    yaw=self._wrap_angle(horizontal_yaw),
                    t=float(t),
                )
            )

        plan = TrajectoryPlan(
            uav_id=self._uav_id(uav_state),
            mode=mode,
            points=points,
            metadata={
                "duration": float(duration),
                "dt": float(dt),
                "target": target.tolist(),
            },
        )
        return self._finalize(plan)

    def _finalize(self, plan: TrajectoryPlan) -> TrajectoryPlan:
        plan.metadata["violations"] = self.validate_plan(plan)
        plan.metadata["executable"] = not plan.metadata["violations"]
        return plan

    @staticmethod
    def _sample_times(duration: float, dt: float) -> np.ndarray:
        if not (math.isfinite(duration) and math.isfinite(dt)):
            raise ValueError("duration and dt must be finite")
        if duration <= 0.0 or dt <= 0.0:
            raise ValueError("duration and dt must be positive")
        interval_count = max(1, int(math.ceil(duration / dt)))
        return np.linspace(0.0, duration, interval_count + 1, dtype=np.float64)

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _value(source: object, *names: str, default: Any = ...) -> Any:
        for name in names:
            if isinstance(source, Mapping) and name in source:
                return source[name]
            if hasattr(source, name):
                return getattr(source, name)
        if default is not ...:
            return default
        raise ValueError(f"missing required field: {' or '.join(names)}")

    def _position(self, state: object) -> np.ndarray:
        raw = self._value(state, "position", default=None)
        if raw is None:
            raw = [
                self._value(state, "x"),
                self._value(state, "y"),
                self._value(state, "z"),
            ]
        position = np.asarray(raw, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position must contain three finite coordinates")
        return position

    def _pad_position(self, landing_pad: object, *, approach: bool) -> np.ndarray:
        field_names = ("approach_position", "position") if approach else ("touchdown_position", "position")
        raw = self._value(landing_pad, *field_names, default=None)
        if raw is None:
            raw = [
                self._value(landing_pad, "x"),
                self._value(landing_pad, "y"),
                self._value(landing_pad, "z"),
            ]
        target = np.asarray(raw, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("landing pad position must contain three finite coordinates")
        if approach:
            approach_height = self._value(landing_pad, "approach_height", default=None)
            if approach_height is not None:
                target = target.copy()
                target[2] = float(approach_height)
        return target

    def _uav_id(self, state: object) -> str:
        identifier = self._value(state, "uav_id", "uid", "id", default="unknown")
        return str(identifier)
