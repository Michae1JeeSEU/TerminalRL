"""Authoritative geometric-path and executable-trajectory generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_right
import hashlib
import json
import math
import uuid

import numpy as np

from .config import RingRole
from .polynomial_trajectory import QuinticSegment, prime_quintic_bounds
from .trajectory_time_parameterization import (
    GeometricPath, TrajectoryConstraints, TrajectoryInfeasibleError, TrajectoryTimeParameterizer,
)
from .vehicle_dynamics import VehicleDynamics


@dataclass(frozen=True)
class TrajectoryPoint:
    t: float
    position: np.ndarray
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    yaw: float | None = None
    yaw_rate: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("position", self.position),
            ("velocity", np.zeros(3, dtype=float) if self.velocity is None else self.velocity),
            ("acceleration", np.zeros(3, dtype=float) if self.acceleration is None else self.acceleration),
        ):
            array = np.asarray(value, dtype=float).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)


@dataclass(frozen=True)
class TrajectoryPlan:
    plan_id: str
    uav_id: int
    mode: str
    points: tuple[TrajectoryPoint, ...]
    start_time: float
    end_time: float
    reservation_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict = field(default_factory=dict)
    _trajectory_hash: str = field(init=False, repr=False)
    _sample_node_times: np.ndarray = field(init=False, repr=False, compare=False)
    _sample_node_positions: np.ndarray = field(init=False, repr=False, compare=False)
    _trajectory_schema_version: str = field(init=False, repr=False, compare=False)
    _polynomial_segments: tuple[QuinticSegment, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "reservation_ids", tuple(self.reservation_ids))
        schema = self.metadata.get("trajectory_schema_version", "executable_trajectory_v2")
        if schema not in {"executable_trajectory_v2", "executable_trajectory_v3"}:
            raise ValueError(f"unsupported trajectory schema version {schema!r}")
        self.metadata.setdefault("trajectory_schema_version", schema)
        if len(self.points) < 2 or self.end_time <= self.start_time:
            raise ValueError("trajectory plan requires at least two increasing-time points")
        if any(right.t <= left.t for left, right in zip(self.points, self.points[1:])):
            raise ValueError("trajectory points must have strictly increasing timestamps")
        # There is one permitted research-vehicle profile in the main
        # simulator.  Retain that explicit identity even for small test and
        # diagnostic plans constructed directly rather than by the generator.
        self.metadata.setdefault("vehicle_profile_id", "research_evtol")
        self.metadata.setdefault("vehicle_profile_version", "eh216s_research_v2")
        object.__setattr__(self, "_trajectory_schema_version", schema)
        if schema == "executable_trajectory_v3":
            try:
                plan_start = float(self.start_time)
                plan_end = float(self.end_time)
            except (TypeError, ValueError) as exc:
                raise ValueError("V3 plan times must be finite scalars") from exc
            if not np.isfinite(plan_start) or not np.isfinite(plan_end):
                raise ValueError("V3 plan times must be finite scalars")
            if plan_start != float(self.points[0].t) or plan_end != float(self.points[-1].t):
                raise ValueError("V3 plan times must match first and last point times")
            for index, point in enumerate(self.points):
                try:
                    point_time = float(point.t)
                    position = np.asarray(point.position, dtype=float)
                    velocity = np.asarray(point.velocity, dtype=float)
                    acceleration = np.asarray(point.acceleration, dtype=float)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError(f"V3 point {index} is malformed") from exc
                if not np.isfinite(point_time):
                    raise ValueError(f"V3 point {index} time must be finite")
                for name, value in (("position", position), ("velocity", velocity), ("acceleration", acceleration)):
                    if value.shape != (3,) or not np.all(np.isfinite(value)):
                        raise ValueError(f"V3 point {index} {name} must have finite shape (3,)")
                for name in ("yaw", "yaw_rate"):
                    value = getattr(point, name, None)
                    if value is not None and (not np.isscalar(value) or not np.isfinite(float(value))):
                        raise ValueError(f"V3 point {index} {name} must be finite when present")
            object.__setattr__(self, "start_time", plan_start)
            object.__setattr__(self, "end_time", plan_end)
            segments = tuple(
                QuinticSegment.from_boundary(
                    left.t,
                    right.t,
                    left.position,
                    left.velocity,
                    left.acceleration,
                    right.position,
                    right.velocity,
                    right.acceleration,
                )
                for left, right in zip(self.points, self.points[1:])
            )
        else:
            segments = ()
        object.__setattr__(self, "_polynomial_segments", segments)
        sample_node_times = np.asarray([point.t for point in self.points], dtype=float)
        sample_node_times.setflags(write=False)
        sample_node_positions = np.asarray([point.position for point in self.points], dtype=float)
        sample_node_positions.setflags(write=False)
        object.__setattr__(self, "_sample_node_times", sample_node_times)
        object.__setattr__(self, "_sample_node_positions", sample_node_positions)
        payload = self._integrity_payload()
        object.__setattr__(self, "_trajectory_hash", hashlib.sha256(json.dumps(payload, separators=(",", ":"), default=float).encode("utf-8")).hexdigest())
        self.metadata.setdefault("trajectory_hash", self._trajectory_hash)

    def _integrity_payload(self):
        if self._trajectory_schema_version == "executable_trajectory_v3":
            return {
                "trajectory_schema_version": self._trajectory_schema_version,
                "segments": [
                    {
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "coefficients": np.asarray(segment.coefficients, dtype=float).tolist(),
                    }
                    for segment in self._polynomial_segments
                ],
                "yaw_nodes": [
                    (float(point.t), None if point.yaw is None else float(point.yaw),
                     None if point.yaw_rate is None else float(point.yaw_rate))
                    for point in self.points
                ],
            }
        return [(round(point.t, 8), np.round(point.position, 8).tolist(), np.round(point.velocity, 8).tolist(),
                 np.round(point.acceleration, 8).tolist(), point.yaw, point.yaw_rate) for point in self.points]

    @property
    def trajectory_hash(self) -> str:
        return self._trajectory_hash

    @property
    def polynomial_segments(self) -> tuple[QuinticSegment, ...]:
        return self._polynomial_segments

    def verify_trajectory_integrity(self) -> bool:
        """Expensive opt-in integrity check for qualification/debug runs."""
        payload = self._integrity_payload()
        actual = hashlib.sha256(json.dumps(payload, separators=(",", ":"), default=float).encode("utf-8")).hexdigest()
        return actual == self._trajectory_hash

    def with_reservation_ids(self, reservation_ids: list[str] | tuple[str, ...]) -> "TrajectoryPlan":
        """Attach an external reservation without changing trajectory bytes."""
        return TrajectoryPlan(self.plan_id, self.uav_id, self.mode, self.points, self.start_time, self.end_time,
                              tuple(reservation_ids), dict(self.metadata))

    def sample(self, now: float) -> TrajectoryPoint:
        if self._trajectory_schema_version == "executable_trajectory_v3":
            value = float(now)
            if not np.isfinite(value):
                raise ValueError("V3 sample time must be finite")
            if value <= self.start_time:
                index = 0
                sample_time = self.start_time
                fraction = 0.0
            elif value >= self.end_time:
                index = len(self._polynomial_segments) - 1
                sample_time = self.end_time
                fraction = 1.0
            else:
                right_index = bisect_right(self.points, value, key=lambda point: point.t)
                index = right_index - 1
                left = self.points[index]
                right = self.points[right_index]
                sample_time = value
                fraction = (value - left.t) / (right.t - left.t)
            position, velocity, acceleration = self._polynomial_segments[index].sample(sample_time)
            left = self.points[index]
            if fraction >= 1.0 and index + 1 < len(self.points):
                right = self.points[index + 1]
            elif index + 1 < len(self.points):
                right = self.points[index + 1]
            else:
                right = left
            yaw = None if left.yaw is None or right.yaw is None else left.yaw + fraction * (right.yaw - left.yaw)
            yaw_rate = None if left.yaw_rate is None or right.yaw_rate is None else left.yaw_rate + fraction * (right.yaw_rate - left.yaw_rate)
            return TrajectoryPoint(sample_time, position, velocity, acceleration, yaw, yaw_rate)
        if now <= self.start_time:
            return self.points[0]
        if now >= self.end_time:
            return self.points[-1]
        right_index = bisect_right(self.points, now, key=lambda point: point.t)
        left, right = self.points[right_index - 1], self.points[right_index]
        fraction = (now - left.t) / (right.t - left.t)
        yaw = None if left.yaw is None or right.yaw is None else left.yaw + fraction * (right.yaw - left.yaw)
        yaw_rate = None if left.yaw_rate is None or right.yaw_rate is None else left.yaw_rate + fraction * (right.yaw_rate - left.yaw_rate)
        return TrajectoryPoint(now, left.position + fraction * (right.position - left.position),
                               left.velocity + fraction * (right.velocity - left.velocity),
                               left.acceleration + fraction * (right.acceleration - left.acceleration), yaw, yaw_rate)

    def sample_position(self, now: float) -> np.ndarray:
        """Interpolate only position for read-only collision calculations.

        Safety review samples trajectories many thousands of times and does
        not need velocity, acceleration, yaw, or a copied TrajectoryPoint.
        This preserves the exact piecewise-linear position semantics of
        :meth:`sample` while avoiding those unrelated allocations.
        """
        if self._trajectory_schema_version == "executable_trajectory_v3":
            value = float(now)
            if not np.isfinite(value):
                raise ValueError("V3 sample time must be finite")
            if value <= self.start_time:
                index = 0
                sample_time = self.start_time
            elif value >= self.end_time:
                index = len(self._polynomial_segments) - 1
                sample_time = self.end_time
            else:
                index = bisect_right(self.points, value, key=lambda point: point.t) - 1
                sample_time = value
            return self._polynomial_segments[index].sample(sample_time)[0]
        if now <= self.start_time:
            return self.points[0].position
        if now >= self.end_time:
            return self.points[-1].position
        right_index = bisect_right(self.points, now, key=lambda point: point.t)
        left, right = self.points[right_index - 1], self.points[right_index]
        fraction = (now - left.t) / (right.t - left.t)
        return left.position + fraction * (right.position - left.position)

    def sample_positions(self, times: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        """Interpolate positions for an ordered or unordered batch of times.

        The endpoint handling and piecewise-linear interpolation formula are
        the same as :meth:`sample_position`; only the node lookup and the
        arithmetic are performed in one NumPy pass.  A scalar input returns a
        three-element position, while an array input appends a final position
        axis to its shape.
        """
        values = np.asarray(times, dtype=float)
        if self._trajectory_schema_version == "executable_trajectory_v3":
            if not np.all(np.isfinite(values)):
                raise ValueError("V3 sample times must be finite")
            flat_values = values.reshape(-1)
            result = np.empty((flat_values.size, 3), dtype=float)
            before = flat_values <= self.start_time
            after = flat_values >= self.end_time
            interior = ~(before | after)
            if np.any(before):
                result[before] = self._polynomial_segments[0].sample_positions(self.start_time)
            if np.any(after):
                result[after] = self._polynomial_segments[-1].sample_positions(self.end_time)
            if np.any(interior):
                interior_indices = np.flatnonzero(interior)
                interior_values = flat_values[interior]
                segment_indices = np.searchsorted(self._sample_node_times, interior_values, side="right") - 1
                for index in np.unique(segment_indices):
                    selected = segment_indices == index
                    result[interior_indices[selected]] = self._polynomial_segments[int(index)].sample_positions(
                        interior_values[selected]
                    )
            return result.reshape(values.shape + (3,))
        flat_values = values.reshape(-1)
        result = np.empty((flat_values.size, 3), dtype=float)
        before = flat_values <= self.start_time
        after = flat_values >= self.end_time
        interior = ~(before | after)
        if np.any(before):
            result[before] = self._sample_node_positions[0]
        if np.any(after):
            result[after] = self._sample_node_positions[-1]
        if np.any(interior):
            interior_values = flat_values[interior]
            right_indices = np.searchsorted(self._sample_node_times, interior_values, side="right")
            left_indices = right_indices - 1
            fractions = ((interior_values - self._sample_node_times[left_indices]) /
                         (self._sample_node_times[right_indices] - self._sample_node_times[left_indices]))
            left_positions = self._sample_node_positions[left_indices]
            right_positions = self._sample_node_positions[right_indices]
            result[interior] = left_positions + fractions[:, None] * (right_positions - left_positions)
        return result.reshape(values.shape + (3,))


class TrajectoryGenerator:
    def __init__(self, config) -> None:
        self.config = config
        self.dynamics = VehicleDynamics(
            config.vehicle_profile,
            continuous_curve_limits=(config.trajectory_schema_version == "executable_trajectory_v3"),
        )
        self.parameterizer = TrajectoryTimeParameterizer(
            self.dynamics,
            schema_version=config.trajectory_schema_version,
        )

    def _schema_metadata(self, metadata: dict | None = None) -> dict:
        details = dict(metadata or {})
        if self.config.trajectory_schema_version == "executable_trajectory_v3":
            details["trajectory_schema_version"] = "executable_trajectory_v3"
        return details

    def _plan(self, uav, mode: str, path: GeometricPath, start: float, target_speed: float, *, reservation=None,
              metadata: dict | None = None, end_speed: float | None = None) -> TrajectoryPlan:
        points = self.parameterizer.parameterize(path, start_time=start, initial_velocity=uav.velocity,
                                                 target_horizontal_speed=target_speed,
                                                 constraints=TrajectoryConstraints(end_horizontal_speed=end_speed, tangent_yaw=False))
        details = self._schema_metadata({**(metadata or {}), "vehicle_profile_id": self.config.vehicle_profile_id,
                                         "vehicle_profile_version": self.config.vehicle_profile.profile_version})
        return TrajectoryPlan(str(uuid.uuid4()), uav.uid, mode, points, start, points[-1].t,
                              () if reservation is None else (reservation.reservation_id,), details)

    def _ring_position(self, ring, theta: float) -> np.ndarray:
        return np.asarray((ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height), dtype=float)

    @staticmethod
    def _v3_finite_vector(value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise RuntimeError(f"{name}_invalid")
        return np.array(vector, dtype=float, copy=True)

    def _v3_ring_transition_parameters(
        self,
        ring,
        theta: float,
        current_speed: float,
        target_speed: float,
        *,
        initial_acceleration: np.ndarray | None,
    ) -> tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Return the bounded analytic holder transition parameters.

        The returned values are ``(ease, ramp, v_after_ease, alpha0, p0,
        v0, a0)``.  This helper is only used by the opt-in V3 executable
        representation; the legacy finite-step holder below is unchanged.
        """
        try:
            radius = float(ring.radius)
            height = float(ring.height)
            phase = float(theta)
            current = float(current_speed)
            target = float(target_speed)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("ring_hold_state_invalid") from exc
        if (not all(math.isfinite(value) for value in (radius, height, phase, current, target))
                or radius <= 0.0 or current < -1e-9 or target < -1e-9):
            raise RuntimeError("ring_hold_state_invalid")
        current = max(0.0, current)
        target = max(0.0, target)
        speed_limit = float(self.dynamics.ring_speed_limit(radius))
        if current > speed_limit + 1e-6:
            raise RuntimeError("ring_hold_initial_speed_exceeds_limit")
        if target > speed_limit + 1e-6:
            raise RuntimeError("ring_hold_target_speed_exceeds_limit")

        tangent = np.asarray((-math.sin(phase), math.cos(phase), 0.0), dtype=float)
        radial = np.asarray((-math.cos(phase), -math.sin(phase), 0.0), dtype=float)
        p0 = self._ring_position(ring, phase)
        v0 = tangent * current
        physical_radial = radial * (current * current / radius)
        if initial_acceleration is None:
            a0 = physical_radial.copy()
        else:
            a0 = self._v3_finite_vector(initial_acceleration, "ring_hold_initial_acceleration")
            radial_component = float(np.dot(a0, radial))
            tangential_component = float(np.dot(a0, tangent))
            if abs(float(a0[2])) > 1e-7 or abs(radial_component - current * current / radius) > 1e-5:
                raise RuntimeError("ring_hold_initial_acceleration_inconsistent")
            # Keep the serialized live vector at the boundary while deriving
            # the analytic scalar transition from its tangential component.
            a0 = physical_radial + tangential_component * tangent

        alpha0 = float(np.dot(a0, tangent))
        ease = 0.4 if abs(alpha0) > 1e-8 else 0.0
        v_after_ease = current + .5 * alpha0 * ease
        if v_after_ease < -1e-7 or v_after_ease > speed_limit + 1e-6:
            raise RuntimeError("ring_hold_ease_state_invalid")
        v_after_ease = max(0.0, v_after_ease)
        delta = target - v_after_ease
        limiting_acceleration = .97 * min(
            float(self.dynamics.profile.max_horizontal_accel),
            float(self.dynamics.profile.max_horizontal_decel),
            float(self.dynamics.profile.max_lateral_accel),
        )
        peak_speed = max(current, v_after_ease, target)
        radial_load = peak_speed * peak_speed / radius
        available_squared = limiting_acceleration ** 2 - radial_load ** 2
        if abs(delta) > 1e-8 and available_squared <= 1e-12:
            raise TrajectoryInfeasibleError("ring_hold_tangential_budget_unavailable")
        available = math.sqrt(max(0.0, available_squared))
        ramp = 0.0 if abs(delta) <= 1e-8 else 1.5 * abs(delta) / available
        return ease, ramp, v_after_ease, alpha0, p0, v0, a0

    def _v3_ring_hold_points_for_ramp(
        self,
        ring,
        theta: float,
        current_speed: float,
        target_speed: float,
        start: float,
        duration: float,
        *,
        initial_acceleration: np.ndarray | None,
        ramp_duration: float,
    ) -> list[TrajectoryPoint]:
        ease, _minimum_ramp, v_after_ease, alpha0, p0, _v0, a0 = self._v3_ring_transition_parameters(
            ring, theta, current_speed, target_speed,
            initial_acceleration=initial_acceleration,
        )
        target = max(0.0, float(target_speed))
        duration = float(duration)
        ramp_duration = float(ramp_duration)
        ramp_end = ease + ramp_duration
        if duration <= 0.0 or duration + 1e-9 < ramp_end:
            raise RuntimeError("ring_hold_duration_shorter_than_required")
        if ramp_end > duration:
            # Required duration is computed independently and can differ from
            # the caller's subtraction by a few ulps.  Keep the requested
            # endpoint time authoritative instead of emitting a node beyond
            # it, while retaining the same bounded ramp law.
            ramp_end = duration

        boundaries = [0.0, duration]
        if ease > 1e-9:
            boundaries.append(ease)
        if ramp_duration > 1e-9:
            boundaries.append(ramp_end)
        boundaries = sorted(set(float(value) for value in boundaries))
        times_relative = [boundaries[0]]
        physics_dt = float(self.config.physics_dt)
        if not math.isfinite(physics_dt) or physics_dt <= 0.0:
            raise RuntimeError("ring_hold_physics_dt_invalid")
        for left, right in zip(boundaries, boundaries[1:]):
            count = max(1, int(math.ceil((right - left) / physics_dt)))
            for index in range(1, count + 1):
                times_relative.append(left + (right - left) * index / count)

        radius = float(ring.radius)
        height = float(ring.height)
        phase = float(theta)
        positions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        accelerations: list[np.ndarray] = []
        for index, relative_time in enumerate(times_relative):
            t = float(relative_time)
            if ease > 1e-9 and t <= ease + 1e-10:
                speed = float(current_speed) + alpha0 * (t - t * t / (2.0 * ease))
                alpha = alpha0 * (1.0 - t / ease)
                angle_delta = float(current_speed) * t + alpha0 * (t * t / 2.0 - t ** 3 / (6.0 * ease))
            elif ramp_duration > 1e-9 and t <= ramp_end + 1e-10:
                tau = t - ease
                s = min(1.0, max(0.0, tau / ramp_duration))
                delta = target - v_after_ease
                speed = v_after_ease + delta * (3.0 * s * s - 2.0 * s ** 3)
                alpha = delta * (6.0 * s - 6.0 * s * s) / ramp_duration
                angle_delta = (
                    float(current_speed) * ease + (alpha0 * ease * ease / 3.0)
                    + v_after_ease * tau + delta * ramp_duration * (s ** 3 - .5 * s ** 4)
                )
            else:
                speed = target
                alpha = 0.0
                angle_delta = (
                    float(current_speed) * ease + (alpha0 * ease * ease / 3.0)
                    + v_after_ease * ramp_duration
                    + .5 * (target - v_after_ease) * ramp_duration
                    + target * (t - ramp_end)
                )
            angle = phase + angle_delta / radius
            tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0), dtype=float)
            radial = np.asarray((-math.cos(angle), -math.sin(angle), 0.0), dtype=float)
            positions.append(np.asarray((radius * math.cos(angle), radius * math.sin(angle), height), dtype=float))
            velocities.append(tangent * speed)
            accelerations.append(tangent * alpha + radial * (speed * speed / radius))
        if initial_acceleration is not None:
            accelerations[0] = self._v3_finite_vector(initial_acceleration, "ring_hold_initial_acceleration")
        else:
            accelerations[0] = a0
        return [
            TrajectoryPoint(float(start + relative_time), position, velocity, acceleration)
            for relative_time, position, velocity, acceleration in zip(
                times_relative, positions, velocities, accelerations
            )
        ]

    def _v3_ring_hold_points(
        self,
        ring,
        theta: float,
        current_speed: float,
        target_speed: float,
        start: float,
        duration: float,
        *,
        initial_acceleration: np.ndarray | None = None,
    ) -> list[TrajectoryPoint]:
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise RuntimeError("ring_hold_duration_invalid")
        ease, minimum_ramp, _va, _alpha, _p0, _v0, _a0 = self._v3_ring_transition_parameters(
            ring, theta, current_speed, target_speed,
            initial_acceleration=initial_acceleration,
        )
        required = ease + minimum_ramp
        if duration + 1e-9 < required:
            raise RuntimeError("ring_hold_duration_shorter_than_required")
        # A bounded set of longer ramps reduces interpolation overshoot while
        # preserving the requested endpoint time and the exact first state.
        candidates = [minimum_ramp]
        room = max(0.0, duration - ease)
        for fraction in (.20, .40, .60, .80, 1.0):
            candidates.append(min(room, minimum_ramp * (1.0 + fraction)))
        last_error: Exception | None = None
        profile = self.dynamics.profile
        for ramp in candidates:
            if ease + ramp > duration + 1e-9:
                continue
            points = self._v3_ring_hold_points_for_ramp(
                ring, theta, current_speed, target_speed, start, duration,
                initial_acceleration=initial_acceleration, ramp_duration=ramp,
            )
            try:
                segments = tuple(
                    QuinticSegment.from_boundary(
                        left.t, right.t, left.position, left.velocity, left.acceleration,
                        right.position, right.velocity, right.acceleration,
                    )
                    for left, right in zip(points, points[1:])
                )
                prime_quintic_bounds(segments)
                if all(
                    segment.bounds()["max_horizontal_speed"] <= profile.max_horizontal_speed + 1e-3
                    and segment.bounds()["max_horizontal_accel"] <= profile.max_horizontal_accel + 1e-3
                    for segment in segments
                ):
                    return points
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                last_error = exc
        if last_error is not None:
            raise RuntimeError("ring_hold_interpolation_infeasible") from last_error
        raise RuntimeError("ring_hold_interpolation_infeasible")

    def ring_hold_points(self, ring, theta: float, current_speed: float, target_speed: float, start: float,
                         duration: float, *, initial_acceleration: np.ndarray | None = None) -> list[TrajectoryPoint]:
        """Exact finite-acceleration circular holder samples for all consumers."""
        if self.config.trajectory_schema_version == "executable_trajectory_v3":
            return self._v3_ring_hold_points(
                ring, theta, current_speed, target_speed, start, duration,
                initial_acceleration=initial_acceleration,
            )
        samples = max(2, int(math.ceil(duration / self.config.physics_dt)) + 1)
        step = duration / (samples - 1)
        points: list[TrajectoryPoint] = []
        speed = max(0.0, current_speed)
        angle = theta
        radius = float(ring.radius)
        height = float(ring.height)
        for index in range(samples):
            if index == 0:
                tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0))
                radial = np.asarray((-math.cos(angle), -math.sin(angle), 0.0))
                points.append(TrajectoryPoint(start, self._ring_position(ring, angle), tangent * speed,
                                              radial * (speed * speed / radius)))
                continue
            previous_speed = speed
            speed = self.dynamics.limited_tangential_speed(previous_speed, target_speed, radius, step)
            angle += .5 * (previous_speed + speed) * step / radius
            tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0), dtype=float)
            radial = np.asarray((-math.cos(angle), -math.sin(angle), 0.0), dtype=float)
            position = np.asarray((radius * math.cos(angle), radius * math.sin(angle), height), dtype=float)
            acceleration = tangent * ((speed - previous_speed) / step) + radial * (speed * speed / radius)
            points.append(TrajectoryPoint(start + index * step, position, tangent * speed, acceleration))
        return points

    def _ring_hold_endpoint(self, ring, theta: float, current_speed: float, target_speed: float,
                            duration: float, *, initial_acceleration: np.ndarray | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact endpoint state of :meth:`ring_hold_points` without allocating its prefix."""
        if self.config.trajectory_schema_version == "executable_trajectory_v3":
            if duration <= 1e-12:
                phase = float(theta)
                speed = max(0.0, float(current_speed))
                tangent = np.asarray((-math.sin(phase), math.cos(phase), 0.0), dtype=float)
                return self._ring_position(ring, phase), tangent * speed
            point = self._v3_ring_hold_points(
                ring, theta, current_speed, target_speed, 0.0, duration,
                initial_acceleration=initial_acceleration,
            )[-1]
            return point.position.copy(), point.velocity.copy()
        samples = max(2, int(math.ceil(duration / self.config.physics_dt)) + 1)
        step = duration / (samples - 1)
        speed = max(0.0, current_speed)
        angle = theta
        radius = float(ring.radius)
        for _ in range(1, samples):
            previous_speed = speed
            speed = self.dynamics.limited_tangential_speed(previous_speed, target_speed, radius, step)
            angle += .5 * (previous_speed + speed) * step / radius
        tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0), dtype=float)
        return self._ring_position(ring, angle), tangent * speed

    def _v3_uav_acceleration(self, uav) -> np.ndarray:
        return self._v3_finite_vector(
            getattr(uav, "acceleration", np.zeros(3, dtype=float)),
            "initial_acceleration",
        )

    def _inbound_v3(self, uav, gate, airspace, now: float) -> TrajectoryPlan:
        ring = airspace.actual_ring(RingRole.OUTER)
        angle = float(gate.angle)
        if not math.isfinite(angle):
            raise RuntimeError("inbound_gate_angle_invalid")
        target = self._ring_position(ring, angle)
        tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0), dtype=float)
        radial = np.asarray((math.cos(angle), math.sin(angle), 0.0), dtype=float)
        distance = float(np.linalg.norm(target - np.asarray(uav.position, dtype=float)))
        control = min(.35 * distance, 160.0)
        path = GeometricPath.cubic_bezier(
            uav.position,
            np.asarray(uav.position, dtype=float) + control * self._unit_or(tangent, uav.velocity),
            target - control * tangent,
            target,
            samples=self.config.trajectory_bezier_samples,
        )
        initial_velocity = self._v3_finite_vector(uav.velocity, "initial_velocity")
        initial_acceleration = self._v3_uav_acceleration(uav)
        points = self.parameterizer.parameterize(
            path,
            start_time=float(now),
            initial_velocity=initial_velocity,
            initial_acceleration=initial_acceleration,
            target_horizontal_speed=float(ring.nominal_speed),
            constraints=TrajectoryConstraints(
                end_horizontal_speed=float(ring.nominal_speed),
                tangent_yaw=False,
                end_direction=tangent,
                end_curvature_vector=-radial / float(ring.radius),
            ),
        )
        details = self._schema_metadata({
            "target_phase": "ring_outer",
            "end_direction": tangent.tolist(),
            "vehicle_profile_id": self.config.vehicle_profile_id,
            "vehicle_profile_version": self.config.vehicle_profile.profile_version,
        })
        return TrajectoryPlan(
            str(uuid.uuid4()), uav.uid, "inbound", points, points[0].t, points[-1].t,
            metadata=details,
        )

    def inbound(self, uav, gate, airspace, now: float) -> TrajectoryPlan:
        if self.config.trajectory_schema_version == "executable_trajectory_v3":
            return self._inbound_v3(uav, gate, airspace, now)
        ring = airspace.actual_ring(RingRole.OUTER)
        angle = gate.angle
        target = self._ring_position(ring, angle)
        tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0))
        distance = float(np.linalg.norm(target - uav.position))
        control = min(.35 * distance, 160.0)
        path = GeometricPath.cubic_bezier(uav.position, uav.position + control * self._unit_or(tangent, uav.velocity),
                                          target - control * tangent, target, samples=self.config.trajectory_bezier_samples)
        return self._plan(uav, "inbound", path, now, ring.nominal_speed,
                          metadata={"target_phase": "ring_outer"}, end_speed=ring.nominal_speed)

    @staticmethod
    def _unit_or(fallback: np.ndarray, value: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        return fallback if norm <= 1e-9 else np.asarray(value, dtype=float) / norm

    def _transfer_path(self, p0, p3, source_tangent, target_tangent):
        """Keep admissible legacy curves; repair only reverse/overshoot geometry.

        Endpoint tangency alone does not prevent a cubic from crossing the
        inner disk and returning against circulation. The fallback progresses
        counterclockwise with smooth, monotone radius/height interpolation.
        Timing and the existing safety chain still decide reachability.
        """
        length = float(np.linalg.norm(p3 - p0))
        control = min(.42 * length, 160.0)
        path = GeometricPath.cubic_bezier(p0, p0 + control * source_tangent,
            p3 - control * target_tangent, p3, samples=self.config.trajectory_bezier_samples)
        xy = path.points[:, :2]
        radii = np.linalg.norm(xy, axis=1)
        r0, r1 = float(np.linalg.norm(p0[:2])), float(np.linalg.norm(p3[:2]))
        cross = xy[:-1, 0] * xy[1:, 1] - xy[:-1, 1] * xy[1:, 0]
        # The executor linearly interpolates nodes. Even an admissible node
        # radius can hide a chord cutting inside the target ring. Check the
        # exact closest point of every segment against the existing 5 cm
        # discretization allowance before retaining a legacy curve.
        delta = np.diff(xy, axis=0)
        fraction = np.clip(-np.sum(xy[:-1] * delta, axis=1) /
                           np.maximum(np.sum(delta * delta, axis=1), 1e-20), 0., 1.)
        segment_min_radius = np.linalg.norm(xy[:-1] + fraction[:, None] * delta, axis=1).min()
        # Permit only sub-millimetre node noise, not operational excursions.
        if segment_min_radius >= min(r0, r1) - .05 and cross.min() >= -1e-8 and radii.min() >= min(r0, r1) - 1e-4 and radii.max() <= max(r0, r1) + 1e-4:
            return path, False
        theta0 = math.atan2(p0[1], p0[0])
        angle = (math.atan2(p3[1], p3[0]) - theta0) % (2 * math.pi)
        # Coincident bearings need forward travel, not a radial departure
        # with zero tangent. An unreachable full circuit will be rejected by
        # the existing merge-time check rather than shortcut through the disk.
        if angle < 1e-6:
            angle = 2 * math.pi
        samples = max(81, self.config.trajectory_bezier_samples, int(math.ceil(angle / .02)) + 1)
        s = np.linspace(0., 1., samples)
        blend = s ** 3 * (10. - 15. * s + 6. * s ** 2)
        radius = r0 + (r1 - r0) * blend
        theta = theta0 + angle * s
        values = np.column_stack((radius * np.cos(theta), radius * np.sin(theta),
                                  p0[2] + (p3[2] - p0[2]) * blend))
        values[0], values[-1] = p0, p3
        return GeometricPath(values), True

    def _ring_transfer_v3(
        self,
        uav,
        source: RingRole,
        target: RingRole,
        reservation,
        airspace,
        now: float,
        speed_residual_mps: float = 0.0,
    ) -> TrajectoryPlan:
        current = self._v3_finite_vector(uav.position, "transfer_initial_position")
        initial_velocity = self._v3_finite_vector(uav.velocity, "initial_velocity")
        initial_acceleration = self._v3_uav_acceleration(uav)
        source_ring = airspace.actual_ring(source)
        target_ring = airspace.actual_ring(target)
        try:
            moment_now = float(now)
            merge_time = float(reservation.merge_time)
            residual = float(speed_residual_mps)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("transfer_timing_inputs_invalid") from exc
        if not all(math.isfinite(value) for value in (moment_now, merge_time, residual)):
            raise RuntimeError("transfer_timing_inputs_nonfinite")
        theta0 = float(
            uav.ring_progress
            if uav.ring_progress is not None
            else math.atan2(float(current[1]), float(current[0]))
        )
        target_theta = float(reservation.merge_angle or 0.0)
        source_target_speed = float(np.clip(
            source_ring.nominal_speed + airspace.speed_profile_bias + residual,
            self.config.speed_min, self.config.max_speed,
        ))
        target_speed = float(np.clip(
            target_ring.nominal_speed + airspace.speed_profile_bias + residual,
            self.config.speed_min, self.config.max_speed,
        ))
        target_position = self._ring_position(target_ring, target_theta)
        target_tangent = np.asarray((-math.sin(target_theta), math.cos(target_theta), 0.0), dtype=float)
        target_radial = np.asarray((math.cos(target_theta), math.sin(target_theta), 0.0), dtype=float)
        live_speed = float(np.linalg.norm(initial_velocity[:2]))
        if abs(float(initial_velocity[2])) > 1e-6 or not math.isfinite(live_speed):
            raise RuntimeError("transfer_initial_velocity_invalid")

        def state_at(moment: float, entry_cap: float):
            delay = float(moment) - moment_now
            if delay > 1e-9:
                hold_target = min(live_speed, max(0.0, float(entry_cap)))
                hold = self.ring_hold_points(
                    source_ring, theta0, live_speed, hold_target, moment_now, delay,
                    initial_acceleration=initial_acceleration,
                )
                return hold[:-1], hold[-1]
            return [], TrajectoryPoint(moment_now, current, initial_velocity, initial_acceleration)

        def build(moment: float, entry_cap: float):
            prefix, start_state = state_at(moment, entry_cap)
            p0 = np.asarray(start_state.position, dtype=float)
            v0 = np.asarray(start_state.velocity, dtype=float)
            a0 = np.asarray(start_state.acceleration, dtype=float)
            angle = math.atan2(float(p0[1]), float(p0[0]))
            source_tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0), dtype=float)
            path, repaired = self._transfer_path(p0, target_position, source_tangent, target_tangent)
            transfer_points = self.parameterizer.parameterize(
                path,
                start_time=float(moment),
                initial_velocity=v0,
                initial_acceleration=a0,
                target_horizontal_speed=max(float(np.linalg.norm(v0[:2])), target_speed),
                constraints=TrajectoryConstraints(
                    end_horizontal_speed=target_speed,
                    tangent_yaw=False,
                    end_direction=target_tangent,
                    end_curvature_vector=-target_radial / float(target_ring.radius),
                ),
            )
            return prefix, transfer_points, source_tangent, v0, repaired

        initial_tangent = self._unit_or(
            np.asarray((-math.sin(theta0), math.cos(theta0), 0.0)), initial_velocity,
        )
        initial_path, _initial_repaired = self._transfer_path(
            current, target_position, initial_tangent, target_tangent,
        )
        entry_cap = min(source_target_speed, self.parameterizer.entry_speed_limit(initial_path))
        chosen = None
        dynamic_unreachable = False
        for _ in range(5):
            required = self._v3_required_ring_braking_duration(
                source_ring, theta0, live_speed, min(live_speed, entry_cap),
                initial_acceleration=initial_acceleration,
            )
            earliest = moment_now + required
            try:
                _prefix, earliest_state = state_at(earliest, entry_cap)
                earliest_angle = math.atan2(float(earliest_state.position[1]), float(earliest_state.position[0]))
                earliest_tangent = np.asarray((-math.sin(earliest_angle), math.cos(earliest_angle), 0.0), dtype=float)
                earliest_path, _ = self._transfer_path(
                    earliest_state.position, target_position, earliest_tangent, target_tangent,
                )
                estimated = self.parameterizer.estimate_duration(
                    earliest_path,
                    initial_velocity=earliest_state.velocity,
                    initial_acceleration=earliest_state.acceleration,
                    target_horizontal_speed=max(float(np.linalg.norm(earliest_state.velocity[:2])), target_speed),
                    constraints=TrajectoryConstraints(
                        end_horizontal_speed=target_speed,
                        tangent_yaw=False,
                        end_direction=target_tangent,
                        end_curvature_vector=-target_radial / float(target_ring.radius),
                    ),
                )
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
                entry_cap *= .8
                continue
            slot_start = max(earliest, merge_time - float(estimated))
            # Evaluate the entire bounded departure window before selecting a
            # candidate.  An earliest physically feasible transfer can be
            # valid yet erase the requested merge-time intention.
            candidates = [slot_start, earliest]
            if merge_time >= earliest:
                candidates.extend(float(value) for value in np.linspace(earliest, merge_time, 17))
            candidates = list(dict.fromkeys(round(float(value), 12) for value in candidates))
            feasible_results = []
            for moment in candidates:
                try:
                    result = build(float(moment), entry_cap)
                except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
                    continue
                feasible_results.append((float(moment), result))
            if feasible_results:
                chosen = min(
                    feasible_results,
                    key=lambda item: (
                        abs(float(item[1][1][-1].t) - merge_time),
                        0 if float(item[1][1][-1].t) <= merge_time + 1e-6 else 1,
                    ),
                )
                dynamic_unreachable = chosen[1][1][-1].t > merge_time + 1e-6
            if chosen is not None:
                break
            entry_cap *= .8
        if chosen is None:
            raise RuntimeError("could not time-parameterize a dynamically feasible transfer")

        departure, result = chosen
        prefix, transfer_points, source_tangent, entry_velocity, geometry_repaired = result
        details = self._schema_metadata({
            "target_phase": "ring_middle" if target is RingRole.MIDDLE else "ring_inner",
            "transfer_geometry": "forward_annular_v1" if geometry_repaired else "legacy_cubic",
            "source_tangent": source_tangent.tolist(),
            "target_tangent": target_tangent.tolist(),
            "speed_residual_mps": residual,
            "scheduled_departure": departure,
            "accepted_reservation_departure": departure,
            "transfer_entry_speed_mps": float(np.linalg.norm(entry_velocity[:2])),
            "requested_entry_speed_mps": source_target_speed,
            "merge_time": merge_time,
            "vehicle_profile_id": self.config.vehicle_profile_id,
            "vehicle_profile_version": self.config.vehicle_profile.profile_version,
        })
        all_points = (*prefix, *transfer_points)
        plan = TrajectoryPlan(
            str(uuid.uuid4()), uav.uid, "ring_transfer", all_points,
            moment_now if prefix else departure,
            transfer_points[-1].t,
            (reservation.reservation_id,), details,
        )
        if dynamic_unreachable or plan.end_time > merge_time + 1e-6:
            plan.metadata["dynamic_unreachable_merge_time"] = True
        return plan

    def ring_transfer(self, uav, source: RingRole, target: RingRole, reservation, airspace, now: float,
                      speed_residual_mps: float = 0.0) -> TrajectoryPlan:
        if self.config.trajectory_schema_version == "executable_trajectory_v3":
            return self._ring_transfer_v3(
                uav, source, target, reservation, airspace, now, speed_residual_mps,
            )
        current = np.asarray(uav.position, dtype=float)
        source_ring, target_ring = airspace.actual_ring(source), airspace.actual_ring(target)
        theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(current[1], current[0])
        target_theta = reservation.merge_angle or 0.0
        source_target_speed = float(np.clip(source_ring.nominal_speed + airspace.speed_profile_bias + speed_residual_mps,
                                            self.config.speed_min, self.config.max_speed))
        # The Bezier leaves tangent to the source ring, so a normal holder can
        # carry its finite tangential velocity into the transfer.  Braking
        # every aircraft to zero on the occupied source ring lets its follower
        # close the protected headway and makes every dense-ring release
        # permanently infeasible.  Any required source-speed adjustment is
        # still integrated explicitly by ``ring_hold_points``.
        transfer_entry_speed = source_target_speed
        target_speed = float(np.clip(target_ring.nominal_speed + airspace.speed_profile_bias + speed_residual_mps,
                                     self.config.speed_min, self.config.max_speed))
        departure = now
        initial_theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(current[1], current[0])
        p0 = current
        p3 = self._ring_position(target_ring, target_theta)

        def departure_state(moment: float, requested_entry_speed: float):
            angle = float(initial_theta)
            speed = max(0.0, float(uav.speed))
            remaining = max(0.0, moment - now)
            while remaining > 1e-9:
                step = min(self.config.physics_dt, remaining)
                next_speed = self.dynamics.limited_tangential_speed(
                    speed, requested_entry_speed, source_ring.radius, step,
                )
                angle += .5 * (speed + next_speed) * step / source_ring.radius
                speed, remaining = next_speed, remaining - step
            tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0))
            return angle, speed, self._ring_position(source_ring, angle), tangent * speed

        # Couple the scheduled departure, finite source-ring speed response,
        # and curve-local entry limit.  Four bounded refinements are enough
        # because both the merge slot and the cubic geometry are fixed.
        for _ in range(4):
            theta, actual_entry_speed, p0, entry_velocity = departure_state(departure, transfer_entry_speed)
            length = float(np.linalg.norm(p3 - p0))
            source_tangent = np.asarray((-math.sin(theta), math.cos(theta), 0.0))
            target_tangent = np.asarray((-math.sin(target_theta), math.cos(target_theta), 0.0))
            tangent_length = min(.42 * length, 160.0)
            transfer_path = GeometricPath.cubic_bezier(p0, p0 + tangent_length * source_tangent,
                                                        p3 - tangent_length * target_tangent, p3,
                                                        samples=self.config.trajectory_bezier_samples)
            transfer_entry_speed = min(source_target_speed, self.parameterizer.entry_speed_limit(transfer_path))
            theta, actual_entry_speed, p0, entry_velocity = departure_state(departure, transfer_entry_speed)
            duration = self.parameterizer.estimate_duration(transfer_path, initial_velocity=entry_velocity,
                                                             target_horizontal_speed=max(actual_entry_speed, target_speed),
                                                             constraints=TrajectoryConstraints(end_horizontal_speed=target_speed, tangent_yaw=False))
            braking_time = max(0.0, uav.speed - transfer_entry_speed) / self.dynamics.profile.max_horizontal_decel
            departure = max(now + braking_time, reservation.merge_time - duration)
        points: list[TrajectoryPoint] = []
        transfer_uav = uav
        if departure > now + 1e-9:
            held = self.ring_hold_points(source_ring, initial_theta, uav.speed, transfer_entry_speed, now, departure - now)
            points = held[:-1]
            p0 = held[-1].position
            theta = math.atan2(p0[1], p0[0])
            source_tangent = np.asarray((-math.sin(theta), math.cos(theta), 0.0))
            length = float(np.linalg.norm(p3 - p0)); tangent_length = min(.42 * length, 160.0)
            transfer_path = GeometricPath.cubic_bezier(p0, p0 + tangent_length * source_tangent,
                                                        p3 - tangent_length * target_tangent, p3,
                                                        samples=self.config.trajectory_bezier_samples)
            transfer_uav = type("State", (), {"uid": uav.uid, "velocity": held[-1].velocity})()
        transfer_path, geometry_repaired = self._transfer_path(p0, p3, source_tangent, target_tangent)

        def executable_transfer(moment):
            if moment > now + 1e-9:
                start_position, velocity = self._ring_hold_endpoint(
                    source_ring, initial_theta, uav.speed, transfer_entry_speed, moment - now,
                )
            else:
                start_position = current
                velocity = uav.velocity
            angle = math.atan2(start_position[1], start_position[0])
            tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0))
            path, repaired = self._transfer_path(start_position, p3, tangent, target_tangent)
            result = self.parameterizer.parameterize(path, start_time=moment,
                initial_velocity=velocity,
                target_horizontal_speed=max(float(np.linalg.norm(velocity)), target_speed),
                constraints=TrajectoryConstraints(end_horizontal_speed=target_speed, tangent_yaw=False))
            return result, tangent, velocity, repaired

        if geometry_repaired:
            # The forward curve changes its feasible entry speed and departure.
            # Search only this reservation's existing window;
            # do not change slots, speed bounds, or safety acceptance criteria.
            transfer_entry_speed = min(source_target_speed, self.parameterizer.entry_speed_limit(transfer_path))
            earliest = now + max(0., uav.speed-transfer_entry_speed) / self.dynamics.profile.max_horizontal_decel
            chosen = None
            candidates = [max(earliest, departure), *np.linspace(earliest, max(earliest, reservation.merge_time), 17)]
            for moment in candidates:
                try:
                    candidate = executable_transfer(float(moment))
                except RuntimeError:
                    continue
                if chosen is None:
                    chosen = (float(moment), candidate)
                if candidate[0][-1].t <= reservation.merge_time + 1e-6:
                    chosen = (float(moment), candidate)
                    break
            if chosen is None:
                raise RuntimeError("could not time-parameterize a dynamically feasible transfer")
            departure, (transfer_points, source_tangent, velocity, geometry_repaired) = chosen
            if departure > now + 1e-9:
                held = self.ring_hold_points(source_ring, initial_theta, uav.speed,
                                             transfer_entry_speed, now, departure - now)
                points = held[:-1]
            else:
                points = []
            transfer_uav = type("State", (), {"uid": uav.uid, "velocity": velocity})()
        else:
            transfer_points = self.parameterizer.parameterize(transfer_path, start_time=departure,
                initial_velocity=transfer_uav.velocity,
                target_horizontal_speed=max(float(np.linalg.norm(transfer_uav.velocity)), target_speed),
                constraints=TrajectoryConstraints(end_horizontal_speed=target_speed, tangent_yaw=False))
        details = {"target_phase": "ring_middle" if target is RingRole.MIDDLE else "ring_inner",
                   "transfer_geometry": "forward_annular_v1" if geometry_repaired else "legacy_cubic",
                   "source_tangent": source_tangent, "target_tangent": target_tangent,
                   "speed_residual_mps": float(speed_residual_mps), "scheduled_departure": departure,
                   "transfer_entry_speed_mps": float(np.linalg.norm(transfer_uav.velocity)),
                   "merge_time": reservation.merge_time, "vehicle_profile_id": self.config.vehicle_profile_id,
                   "vehicle_profile_version": self.config.vehicle_profile.profile_version}
        all_points = (*points, *transfer_points)
        transfer = TrajectoryPlan(str(uuid.uuid4()), uav.uid, "ring_transfer", all_points, now if points else departure,
                                  transfer_points[-1].t, (reservation.reservation_id,), details)
        if transfer.end_time > reservation.merge_time + 1e-6:
            transfer.metadata["dynamic_unreachable_merge_time"] = True
        return transfer

    def _v3_required_ring_braking_duration(
        self,
        ring,
        theta: float,
        current_speed: float,
        target_speed: float,
        *,
        initial_acceleration: np.ndarray | None = None,
    ) -> float:
        current = float(current_speed)
        target = float(target_speed)
        if not math.isfinite(current) or not math.isfinite(target) or current < -1e-9 or target < -1e-9:
            raise RuntimeError("final_approach_initial_speed_invalid")
        if current <= target + 1e-8:
            return 0.0
        ease, ramp, _va, _alpha, _p0, _v0, _a0 = self._v3_ring_transition_parameters(
            ring, theta, current, target,
            initial_acceleration=initial_acceleration,
        )
        return float(ease + ramp)

    def _required_ring_braking_duration(
        self,
        ring,
        theta: float,
        current_speed: float,
        target_speed: float,
    ) -> float:
        """Return a bounded discrete duration for one-way ring braking.

        The endpoint is evaluated through the same ``ring_hold_points``
        integration grid used by the executable prefix.  A bounded bracket
        and binary search avoid replacing the live velocity with a target
        value at the departure instant.
        """
        if self.config.trajectory_schema_version == "executable_trajectory_v3":
            return self._v3_required_ring_braking_duration(
                ring, theta, current_speed, target_speed,
            )
        current = float(current_speed)
        target = float(target_speed)
        if not math.isfinite(current) or not math.isfinite(target):
            raise RuntimeError("final_approach_initial_speed_nonfinite")
        if current < -1e-9 or target < -1e-9:
            raise RuntimeError("final_approach_initial_speed_invalid")
        current = max(0.0, current)
        target = max(0.0, target)
        if current <= target + 1e-8:
            return 0.0
        physics_dt = float(self.config.physics_dt)
        max_decel = float(self.dynamics.profile.max_horizontal_decel)
        if not math.isfinite(physics_dt) or physics_dt <= 0.0 or max_decel <= 0.0:
            raise RuntimeError("final_approach_braking_bounds_invalid")

        def endpoint_speed(duration: float) -> float:
            _position, velocity = self._ring_hold_endpoint(
                ring, theta, current, target, float(duration),
            )
            speed = float(np.linalg.norm(np.asarray(velocity, dtype=float)[:2]))
            if not math.isfinite(speed):
                raise RuntimeError("final_approach_braking_endpoint_nonfinite")
            return speed

        # The upper bracket is deliberately bounded.  The normal profile
        # reaches the target well before this cap; failure is explicit when
        # the shared finite-step dynamics cannot make progress.
        upper = max(physics_dt, (current - target) / max_decel)
        for _ in range(10):
            if endpoint_speed(upper) <= target + 1e-7:
                break
            upper *= 2.0
        else:
            raise RuntimeError("final_approach_braking_target_unreachable")
        if upper > 120.0:
            raise RuntimeError("final_approach_braking_duration_unbounded")

        lower = 0.0
        for _ in range(32):
            middle = 0.5 * (lower + upper)
            if middle <= 1e-12:
                lower = middle
                continue
            if endpoint_speed(middle) <= target + 1e-7:
                upper = middle
            else:
                lower = middle
        if endpoint_speed(upper) > target + 1e-6:
            raise RuntimeError("final_approach_braking_target_unreachable")
        return float(upper)

    @staticmethod
    def _replace_ring_hold_initial_state(hold_points: list[TrajectoryPoint], uav, now: float) -> None:
        """Preserve the serialized live state at the start of a new-mode hold."""
        position = np.asarray(uav.position, dtype=float)
        velocity = np.asarray(uav.velocity, dtype=float)
        acceleration = np.asarray(getattr(uav, "acceleration", np.zeros(3)), dtype=float)
        if (position.shape != (3,) or velocity.shape != (3,) or acceleration.shape != (3,)
                or not np.isfinite(position).all()
                or not np.isfinite(velocity).all()
                or not np.isfinite(acceleration).all()):
            raise RuntimeError("final_approach_initial_state_invalid")
        if not hold_points:
            raise RuntimeError("final_approach_braking_prefix_empty")
        first = hold_points[0]
        hold_points[0] = TrajectoryPoint(
            float(now), position, velocity, acceleration,
            first.yaw, first.yaw_rate,
        )

    def _continuous_inward_landing_points(
        self,
        uav,
        lz,
        airspace,
        now: float,
        *,
        scheduled_delay: float = 0.0,
        target_horizontal_speed: float = 10.0,
        holding_target_speed: float | None = None,
    ) -> tuple[tuple[TrajectoryPoint, ...], float, float, np.ndarray, float, bool]:
        """Build the shared tangent departure, staging and vertical-leg route.

        The emergency route historically used this exact geometry.  Keeping
        it in one helper lets the opt-in normal final approach reuse the
        safety-tested corridor without changing the established emergency
        points or metadata.  A positive scheduled delay is represented by a
        ring hold at the current speed; the final departure starts exactly at
        the hold endpoint.
        """
        p0 = np.asarray(uav.position, dtype=float)
        initial_velocity = np.asarray(uav.velocity, dtype=float)
        high = min(self.config.height_max, self.dynamics.profile.max_operating_altitude_agl_m)
        cruise_height = high if uav.ring_role is not RingRole.INNER else float(p0[2])
        staging = np.asarray((lz.position[0], lz.position[1], cruise_height), dtype=float)

        prefix: list[TrajectoryPoint] = []
        departure_start = float(now)
        departure_theta = (
            uav.ring_progress
            if uav.ring_progress is not None
            else math.atan2(p0[1], p0[0])
        )
        if scheduled_delay > 1e-9:
            ring = airspace.actual_ring(uav.ring_role)
            current_ring_speed = max(0.0, float(uav.speed))
            hold_target_speed = current_ring_speed if holding_target_speed is None else min(
                current_ring_speed, max(0.0, float(holding_target_speed))
            )
            hold_points = self.ring_hold_points(
                ring, departure_theta, current_ring_speed, hold_target_speed,
                now, float(scheduled_delay),
            )
            if holding_target_speed is not None:
                self._replace_ring_hold_initial_state(hold_points, uav, now)
            prefix = hold_points[:-1]
            start_state = hold_points[-1]
            p0, initial_velocity = start_state.position, start_state.velocity
            departure_start = float(start_state.t)
            departure_theta = math.atan2(p0[1], p0[0])

        # Carry the holder's actual tangential velocity into one smooth,
        # inward departure.  Braking to zero while circling on the source ring
        # leaves a stationary holder in the safety prediction tube and can
        # make every subsequent emergency candidate unreachable.  The first
        # Bezier control point is tangent to the source ring; the second one
        # approaches the staging point horizontally so its terminal tangent is
        # horizontal and the vertical pad leg can start from zero velocity.
        source_tangent = self._unit_or(np.asarray((
            -math.sin(departure_theta), math.cos(departure_theta), 0.0,
        )), initial_velocity)
        displacement = staging - p0
        displacement_xy = np.asarray((displacement[0], displacement[1], 0.0), dtype=float)
        approach = self._unit_or(source_tangent, displacement_xy)
        distance = float(np.linalg.norm(displacement))
        tangent_control = min(.70 * distance, 240.0)
        approach_control = min(.25 * distance, 120.0)
        p1 = p0 + tangent_control * source_tangent
        p2 = staging - approach_control * approach
        p2[2] = staging[2]
        departure_path = GeometricPath.cubic_bezier(
            p0, p1, p2, staging, samples=7
        )
        departure = self.parameterizer.parameterize(
            departure_path,
            start_time=departure_start,
            initial_velocity=initial_velocity,
            target_horizontal_speed=float(target_horizontal_speed),
            constraints=TrajectoryConstraints(end_horizontal_speed=0.0, tangent_yaw=False),
        )

        pad = np.asarray(lz.position, dtype=float)
        points = list(prefix)
        points.extend(departure[:-1])
        leg_path = GeometricPath.line(staging, pad, samples=17)
        leg = self.parameterizer.parameterize(
            leg_path,
            start_time=departure[-1].t,
            initial_velocity=departure[-1].velocity,
            target_horizontal_speed=5.0,
            constraints=TrajectoryConstraints(end_horizontal_speed=0.0, tangent_yaw=False),
        )
        points.extend(leg)
        moment = float(leg[-1].t)
        return tuple(points), departure_start, moment, staging, cruise_height, bool(prefix)

    def _v3_continuous_inward_landing_points(
        self,
        uav,
        lz,
        airspace,
        now: float,
        *,
        scheduled_delay: float = 0.0,
        target_horizontal_speed: float = 10.0,
        requested_departure: float | None = None,
        holding_target_speed: float | None = None,
    ) -> dict:
        """Build the continuous corridor with an analytic V3 ring boundary."""
        if airspace is None or uav.ring_role is None:
            raise RuntimeError("v3_continuous_landing_requires_ring_context")
        p_live = self._v3_finite_vector(uav.position, "final_approach_initial_position")
        v_live = self._v3_finite_vector(uav.velocity, "initial_velocity")
        a_live = self._v3_uav_acceleration(uav)
        current_speed = float(np.linalg.norm(v_live[:2]))
        if abs(float(v_live[2])) > 1e-6 or not math.isfinite(current_speed):
            raise RuntimeError("final_approach_initial_velocity_not_horizontal")
        stored_speed = float(uav.speed)
        if not math.isfinite(stored_speed) or stored_speed < -1e-9:
            raise RuntimeError("final_approach_initial_speed_invalid")
        if abs(current_speed - max(0.0, stored_speed)) > 1e-5:
            raise RuntimeError("final_approach_initial_speed_mismatch")
        current_speed = max(0.0, stored_speed)
        ring = airspace.actual_ring(uav.ring_role)
        if current_speed > self.dynamics.ring_speed_limit(float(ring.radius)) + 1e-6:
            raise RuntimeError("final_approach_initial_speed_exceeds_ring_limit")
        theta = float(
            uav.ring_progress
            if uav.ring_progress is not None
            else math.atan2(float(p_live[1]), float(p_live[0]))
        )
        if not math.isfinite(theta):
            raise RuntimeError("final_approach_initial_ring_phase_invalid")
        try:
            candidate_delay = max(0.0, float(scheduled_delay))
            requested_delay = (0.0 if requested_departure is None
                               else max(0.0, float(requested_departure) - float(now)))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("final_approach_timing_inputs_invalid") from exc
        if not all(math.isfinite(value) for value in (candidate_delay, requested_delay)):
            raise RuntimeError("final_approach_timing_inputs_nonfinite")
        base_delay = max(candidate_delay, requested_delay)

        high = min(self.config.height_max, self.dynamics.profile.max_operating_altitude_agl_m)
        cruise_height = high if uav.ring_role is not RingRole.INNER else float(p_live[2])
        staging = np.asarray((lz.position[0], lz.position[1], cruise_height), dtype=float)
        pad = np.asarray(lz.position, dtype=float)

        def geometry(p0: np.ndarray, entry_velocity: np.ndarray, entry_acceleration: np.ndarray):
            departure_theta = math.atan2(float(p0[1]), float(p0[0]))
            source_tangent = self._unit_or(
                np.asarray((-math.sin(departure_theta), math.cos(departure_theta), 0.0)),
                entry_velocity,
            )
            displacement = staging - p0
            displacement_xy = np.asarray((displacement[0], displacement[1], 0.0), dtype=float)
            approach = self._unit_or(source_tangent, displacement_xy)
            distance = float(np.linalg.norm(displacement))
            tangent_control = min(.70 * distance, 240.0)
            approach_control = min(.25 * distance, 120.0)
            p1 = p0 + tangent_control * source_tangent
            p2 = staging - approach_control * approach
            p2[2] = staging[2]
            departure_path = GeometricPath.cubic_bezier(p0, p1, p2, staging, samples=7)
            return departure_path, departure_theta

        path0, _ = geometry(p_live, v_live, a_live)
        entry_cap = min(float(target_horizontal_speed), self.parameterizer.entry_speed_limit(path0))
        if holding_target_speed is not None:
            entry_cap = min(entry_cap, max(0.0, float(holding_target_speed)))
        if not math.isfinite(entry_cap) or entry_cap < 0.0:
            raise RuntimeError("final_approach_entry_speed_nonfinite")

        for _ in range(7):
            hold_target = min(current_speed, entry_cap)
            required_prefix = self._v3_required_ring_braking_duration(
                ring, theta, current_speed, hold_target,
                initial_acceleration=a_live,
            )
            actual_delay = max(base_delay, required_prefix)
            departure_start = float(now) + actual_delay
            prefix: list[TrajectoryPoint] = []
            if actual_delay > 1e-9:
                hold_points = self.ring_hold_points(
                    ring, theta, current_speed, hold_target, float(now), actual_delay,
                    initial_acceleration=a_live,
                )
                prefix = hold_points[:-1]
                start_state = hold_points[-1]
                p0 = np.asarray(start_state.position, dtype=float)
                entry_velocity = np.asarray(start_state.velocity, dtype=float)
                entry_acceleration = np.asarray(start_state.acceleration, dtype=float)
            else:
                p0, entry_velocity, entry_acceleration = p_live, v_live, a_live
            departure_path, _departure_theta = geometry(p0, entry_velocity, entry_acceleration)
            try:
                departure = self.parameterizer.parameterize(
                    departure_path,
                    start_time=departure_start,
                    initial_velocity=entry_velocity,
                    initial_acceleration=entry_acceleration,
                    target_horizontal_speed=float(target_horizontal_speed),
                    constraints=TrajectoryConstraints(
                        end_horizontal_speed=0.0,
                        tangent_yaw=False,
                        end_direction=np.asarray((
                            (staging[0] - p0[0]), (staging[1] - p0[1]), 0.0,
                        ), dtype=float),
                        end_curvature_vector=np.zeros(3, dtype=float),
                    ),
                )
                if not np.allclose(departure[-1].velocity, np.zeros(3), atol=1e-6):
                    raise RuntimeError("v3_staging_boundary_velocity_nonzero")
                leg_path = GeometricPath.line(staging, pad, samples=17)
                leg = self.parameterizer.parameterize(
                    leg_path,
                    start_time=departure[-1].t,
                    initial_velocity=departure[-1].velocity,
                    initial_acceleration=departure[-1].acceleration,
                    target_horizontal_speed=5.0,
                    constraints=TrajectoryConstraints(
                        end_horizontal_speed=0.0,
                        tangent_yaw=False,
                        end_direction=np.asarray((0.0, 0.0, -1.0)),
                        end_curvature_vector=np.zeros(3, dtype=float),
                    ),
                )
                points = [*prefix, *departure[:-1], *leg]
                if len(points) < 2:
                    raise RuntimeError("v3_continuous_landing_points_missing")
                actual_entry_speed = float(np.linalg.norm(entry_velocity[:2]))
                return {
                    "points": tuple(points),
                    "departure_start": departure_start,
                    "moment": float(leg[-1].t),
                    "staging": staging,
                    "cruise_height": cruise_height,
                    "has_prefix": bool(prefix),
                    "entry_speed": actual_entry_speed,
                    "entry_cap": float(entry_cap),
                    "required_prefix": float(required_prefix),
                    "actual_delay": float(actual_delay),
                }
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
                entry_cap *= .8
                if entry_cap < 0.1:
                    break
        raise TrajectoryInfeasibleError("could not time-parameterize a dynamically feasible continuous landing")

    def _final_approach_v3(
        self,
        uav,
        lz,
        reservation,
        now: float,
        *,
        scheduled_delay: float,
        speed_residual_mps: float,
        airspace,
    ) -> TrajectoryPlan:
        if uav.ring_role is not RingRole.INNER or airspace is None:
            raise RuntimeError("V3 final approach requires inner-ring airspace context")
        if self.config.final_approach_geometry_mode != "continuous_inward_v1":
            return self._final_approach_v3_brake(
                uav, lz, reservation, now,
                scheduled_delay=scheduled_delay,
                speed_residual_mps=speed_residual_mps,
                airspace=airspace,
            )
        residual = float(speed_residual_mps)
        requested_target = 10.0 + residual
        mode = self.config.final_approach_execution_mode
        if mode == "legacy":
            # The historical continuous mapping does not bind the proposal's
            # residual or reservation timing to executable final geometry.
            accepted_target = 10.0
            speed_clamp_reason = "legacy_fixed_target"
        else:
            accepted_target = float(np.clip(requested_target, 0.1, self.config.max_speed))
            speed_clamp_reason = (
                "minimum_target_speed" if requested_target < 0.1 else
                "maximum_target_speed" if requested_target > self.config.max_speed else "none"
            )
        reservation_departure = (float(reservation.merge_time)
                                 if mode == "speed_timing_v1" else float(now))
        candidate_delay = max(0.0, float(scheduled_delay))
        result = self._v3_continuous_inward_landing_points(
            uav, lz, airspace, now,
            scheduled_delay=candidate_delay,
            target_horizontal_speed=accepted_target,
            requested_departure=reservation_departure if mode == "speed_timing_v1" else None,
        )
        actual_departure = float(result["departure_start"])
        timing_constraints = {
            "now_floor": float(now),
            "candidate_delay_departure": float(now) + candidate_delay,
            "finite_ring_braking_departure": float(now) + float(result["required_prefix"]),
        }
        if mode == "speed_timing_v1":
            timing_constraints["requested_reservation_departure"] = reservation_departure
        timing_clamped = mode == "speed_timing_v1" and actual_departure > reservation_departure + 1e-9
        timing_reasons = []
        if timing_clamped:
            for name, bound in timing_constraints.items():
                if name != "requested_reservation_departure" and bound > reservation_departure + 1e-9 and abs(bound - actual_departure) <= 1e-7:
                    timing_reasons.append(name)
        details = self._schema_metadata({
            "target_phase": "landing",
            "scheduled_departure": actual_departure,
            "scheduled_delay": candidate_delay,
            "speed_residual_mps": residual,
            "final_approach_geometry": "continuous_inward_v1",
            "final_approach_execution_mode": mode,
            "lz_id": lz.lz_id,
            "corridor_height": result["cruise_height"],
            "source_ring_role": uav.ring_role.value,
            "staging_point": result["staging"].tolist(),
            "requested_target_speed_mps": requested_target,
            "accepted_target_speed_mps": accepted_target,
            "requested_reservation_departure": reservation_departure,
            "accepted_reservation_departure": actual_departure,
            "actual_entry_speed_mps": result["entry_speed"],
            "speed_clamp_reason": speed_clamp_reason,
            "timing_clamped": timing_clamped,
            "timing_clamp_reason": ",".join(timing_reasons) if timing_reasons else "none",
            "timing_constraints": timing_constraints,
            "candidate_delay_seconds": candidate_delay,
            "required_braking_duration_seconds": result["required_prefix"],
            "effective_delay_seconds": result["actual_delay"],
            "vehicle_profile_id": self.config.vehicle_profile_id,
            "vehicle_profile_version": self.config.vehicle_profile.profile_version,
        })
        plan = TrajectoryPlan(
            str(uuid.uuid4()), uav.uid, "final_approach", result["points"],
            float(now) if result["has_prefix"] else actual_departure,
            result["moment"], (reservation.reservation_id,), details,
        )
        if self.validate_kinematics(plan):
            raise RuntimeError("v3_final_approach_kinematics_invalid")
        return plan

    def _final_approach_v3_brake(
        self,
        uav,
        lz,
        reservation,
        now: float,
        *,
        scheduled_delay: float,
        speed_residual_mps: float,
        airspace,
    ) -> TrajectoryPlan:
        p_live = self._v3_finite_vector(uav.position, "final_approach_initial_position")
        v_live = self._v3_finite_vector(uav.velocity, "initial_velocity")
        a_live = self._v3_uav_acceleration(uav)
        endpoint = np.asarray(lz.position, dtype=float)
        start = float(now) + max(0.0, float(scheduled_delay))
        prefix: list[TrajectoryPoint] = []
        p0, v0, a0 = p_live, v_live, a_live
        final_points = None
        target_cap = min(10.0, self.config.max_speed)
        required = 0.0
        actual_entry_speed = float(np.linalg.norm(v_live[:2]))
        for _ in range(7):
            prefix = []
            start = float(now) + max(0.0, float(scheduled_delay))
            p0, v0, a0 = p_live, v_live, a_live
            if uav.ring_role is RingRole.INNER and airspace is not None:
                ring = airspace.actual_ring(RingRole.INNER)
                theta = float(uav.ring_progress if uav.ring_progress is not None else math.atan2(p_live[1], p_live[0]))
                current_speed = float(np.linalg.norm(v_live[:2]))
                direction0 = self._unit_or(np.asarray((0.0, -1.0, 0.0)), v_live)
                distance0 = float(np.linalg.norm(endpoint - p0))
                path0 = GeometricPath.cubic_bezier(
                    p0,
                    p0 + min(.30 * distance0, 120.0) * direction0,
                    endpoint - min(.25 * distance0, 80.0) * np.asarray((direction0[0], direction0[1], 0.0)),
                    endpoint,
                    samples=self.config.trajectory_final_samples,
                )
                entry_limit = self.parameterizer.entry_speed_limit(path0)
                target = min(current_speed, target_cap, entry_limit)
                required = self._v3_required_ring_braking_duration(
                    ring, theta, current_speed, target, initial_acceleration=a_live,
                )
                start = float(now) + max(0.0, float(scheduled_delay), required)
                if start > float(now) + 1e-9:
                    hold = self.ring_hold_points(
                        ring, theta, current_speed, target, float(now), start - float(now),
                        initial_acceleration=a_live,
                    )
                    prefix = hold[:-1]
                    p0, v0, a0 = hold[-1].position, hold[-1].velocity, hold[-1].acceleration
            direction = self._unit_or(np.asarray((0.0, -1.0, 0.0)), v0)
            distance = float(np.linalg.norm(endpoint - p0))
            p1 = p0 + min(.30 * distance, 120.0) * direction
            p2 = endpoint - min(.25 * distance, 80.0) * np.asarray((direction[0], direction[1], 0.0))
            path = GeometricPath.cubic_bezier(p0, p1, p2, endpoint, samples=self.config.trajectory_final_samples)
            try:
                final_points = self.parameterizer.parameterize(
                    path,
                    start_time=start,
                    initial_velocity=v0,
                    initial_acceleration=a0,
                    target_horizontal_speed=min(10.0, self.config.max_speed),
                    constraints=TrajectoryConstraints(end_horizontal_speed=0.0, tangent_yaw=False),
                )
                actual_entry_speed = float(np.linalg.norm(v0[:2]))
                break
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
                target_cap *= .8
                if target_cap < 0.1:
                    break
        if final_points is None:
            raise RuntimeError("could not time-parameterize a dynamically feasible brake-bezier approach")
        requested_target = 10.0 + float(speed_residual_mps)
        details = self._schema_metadata({
            "target_phase": "landing",
            "scheduled_departure": start,
            "scheduled_delay": max(0.0, float(scheduled_delay)),
            "speed_residual_mps": float(speed_residual_mps),
            "final_approach_execution_mode": self.config.final_approach_execution_mode,
            "requested_target_speed_mps": requested_target,
            "accepted_target_speed_mps": 10.0,
            "requested_reservation_departure": float(now),
            "accepted_reservation_departure": float(start),
            "actual_entry_speed_mps": actual_entry_speed,
            "speed_clamp_reason": "legacy_fixed_target",
            "timing_clamped": float(start) > float(now) + max(0.0, float(scheduled_delay)) + 1e-9,
            "timing_clamp_reason": "finite_ring_braking" if required > max(0.0, float(scheduled_delay)) + 1e-9 else "none",
            "timing_constraints": {
                "now_floor": float(now),
                "candidate_delay_departure": float(now) + max(0.0, float(scheduled_delay)),
                "finite_ring_braking_departure": float(now) + float(required),
            },
            "required_braking_duration_seconds": float(required),
            "vehicle_profile_id": self.config.vehicle_profile_id,
            "vehicle_profile_version": self.config.vehicle_profile.profile_version,
        })
        plan = TrajectoryPlan(
            str(uuid.uuid4()), uav.uid, "final_approach", (*prefix, *final_points),
            float(now) if prefix else start, final_points[-1].t,
            (reservation.reservation_id,), details,
        )
        if self.validate_kinematics(plan):
            raise RuntimeError("v3_final_approach_kinematics_invalid")
        return plan

    def _final_approach_speed_timing_v1(
        self,
        uav,
        lz,
        reservation,
        now: float,
        *,
        scheduled_delay: float,
        speed_residual_mps: float,
        airspace,
    ) -> TrajectoryPlan:
        """Build the opt-in continuous final approach with executable timing."""
        if uav.ring_role is not RingRole.INNER or airspace is None:
            raise RuntimeError(
                "speed_timing_v1 final approach requires inner-ring airspace context"
            )
        if self.config.final_approach_geometry_mode != "continuous_inward_v1":
            raise RuntimeError(
                "speed_timing_v1 final approach requires continuous_inward_v1 geometry"
            )
        try:
            moment_now = float(now)
            residual = float(speed_residual_mps)
            requested_departure = float(reservation.merge_time)
            candidate_delay = float(scheduled_delay)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("final_approach_timing_inputs_invalid") from exc
        if not all(math.isfinite(value) for value in (
            moment_now, residual, requested_departure, candidate_delay,
        )):
            raise RuntimeError("final_approach_timing_inputs_nonfinite")
        candidate_delay = max(0.0, candidate_delay)

        requested_target = 10.0 + residual
        if not math.isfinite(requested_target):
            raise RuntimeError("final_approach_target_speed_nonfinite")
        accepted_target = float(np.clip(requested_target, 0.1, self.config.max_speed))
        if requested_target < 0.1:
            speed_clamp_reason = "minimum_target_speed"
        elif requested_target > self.config.max_speed:
            speed_clamp_reason = "maximum_target_speed"
        else:
            speed_clamp_reason = "none"

        position = np.asarray(uav.position, dtype=float)
        velocity = np.asarray(uav.velocity, dtype=float)
        acceleration = np.asarray(getattr(uav, "acceleration", np.zeros(3)), dtype=float)
        if (position.shape != (3,) or velocity.shape != (3,) or acceleration.shape != (3,)
                or not np.isfinite(position).all()
                or not np.isfinite(velocity).all()
                or not np.isfinite(acceleration).all()):
            raise RuntimeError("final_approach_initial_state_invalid")
        if abs(float(velocity[2])) > 1e-6:
            raise RuntimeError("final_approach_initial_velocity_not_horizontal")
        current_speed = float(np.linalg.norm(velocity[:2]))
        stored_speed = float(uav.speed)
        if not math.isfinite(stored_speed) or stored_speed < -1e-9:
            raise RuntimeError("final_approach_initial_speed_invalid")
        if abs(current_speed - max(0.0, stored_speed)) > 1e-5:
            raise RuntimeError("final_approach_initial_speed_mismatch")
        current_speed = max(0.0, stored_speed)
        ring = airspace.actual_ring(RingRole.INNER)
        if current_speed > self.dynamics.ring_speed_limit(ring.radius) + 1e-6:
            raise RuntimeError("final_approach_initial_speed_exceeds_ring_limit")
        theta = (
            float(uav.ring_progress)
            if uav.ring_progress is not None
            else math.atan2(float(position[1]), float(position[0]))
        )
        if not math.isfinite(theta):
            raise RuntimeError("final_approach_initial_ring_phase_invalid")

        required_braking = self._required_ring_braking_duration(
            ring, theta, current_speed, accepted_target,
        )
        requested_delay = max(0.0, requested_departure - moment_now)
        actual_delay = max(candidate_delay, requested_delay, required_braking)
        actual_departure = moment_now + actual_delay
        timing_constraints = {
            "now_floor": moment_now,
            "candidate_delay_departure": moment_now + candidate_delay,
            "requested_reservation_departure": requested_departure,
            "finite_ring_braking_departure": moment_now + required_braking,
        }
        timing_clamped = actual_departure > requested_departure + 1e-9
        timing_reasons: list[str] = []
        if timing_clamped:
            for reason, bound in timing_constraints.items():
                if reason == "requested_reservation_departure":
                    continue
                if (bound > requested_departure + 1e-9
                        and abs(actual_departure - bound) <= 1e-7):
                    timing_reasons.append(reason)
        timing_clamp_reason = ",".join(timing_reasons) if timing_reasons else "none"

        # Holding at a lower current speed is intentional: a proposal whose
        # accepted target is above the live speed does not force an initial
        # acceleration on the ring.  The curve parameterizer may accelerate
        # later under its own finite dynamics limits.
        holding_target = min(current_speed, accepted_target)
        points, departure_start, moment, staging, cruise_height, has_prefix = (
            self._continuous_inward_landing_points(
                uav,
                lz,
                airspace,
                moment_now,
                scheduled_delay=actual_delay,
                target_horizontal_speed=accepted_target,
                holding_target_speed=holding_target,
            )
        )
        # The authoritative executable plan must begin with the complete
        # serialized live state, including acceleration, even when no ring
        # prefix was needed.  The parameterizer owns subsequent derivative
        # samples; this first node is the handoff boundary.
        if not points:
            raise RuntimeError("final_approach_initial_state_missing")
        initial_point = points[0]
        points = (
            TrajectoryPoint(
                initial_point.t,
                position,
                velocity,
                acceleration,
                initial_point.yaw,
                initial_point.yaw_rate,
            ),
            *points[1:],
        )
        if abs(float(departure_start) - actual_departure) > 1e-7:
            raise RuntimeError("final_approach_departure_timing_mismatch")
        entry_point = next(
            (point for point in points if abs(float(point.t) - float(departure_start)) <= 1e-7),
            None,
        )
        if entry_point is None:
            raise RuntimeError("final_approach_entry_state_missing")
        actual_entry_speed = float(np.linalg.norm(np.asarray(entry_point.velocity, dtype=float)[:2]))
        if not math.isfinite(actual_entry_speed):
            raise RuntimeError("final_approach_entry_speed_nonfinite")
        if current_speed > accepted_target + 1e-7 and actual_entry_speed > accepted_target + 1e-5:
            raise RuntimeError("final_approach_braking_prefix_target_missed")

        details = {
            "target_phase": "landing",
            "scheduled_departure": float(departure_start),
            "scheduled_delay": candidate_delay,
            "speed_residual_mps": residual,
            "final_approach_geometry": "continuous_inward_v1",
            "final_approach_execution_mode": "speed_timing_v1",
            "lz_id": lz.lz_id,
            "corridor_height": cruise_height,
            "source_ring_role": uav.ring_role.value,
            "staging_point": staging.tolist(),
            "requested_target_speed_mps": requested_target,
            "accepted_target_speed_mps": accepted_target,
            "requested_reservation_departure": requested_departure,
            "accepted_reservation_departure": float(actual_departure),
            "actual_entry_speed_mps": actual_entry_speed,
            "speed_clamp_reason": speed_clamp_reason,
            "timing_clamped": timing_clamped,
            "timing_clamp_reason": timing_clamp_reason,
            "timing_constraints": timing_constraints,
            "candidate_delay_seconds": candidate_delay,
            "required_braking_duration_seconds": required_braking,
            "effective_delay_seconds": actual_delay,
            "vehicle_profile_id": self.config.vehicle_profile_id,
            "vehicle_profile_version": self.config.vehicle_profile.profile_version,
        }
        return TrajectoryPlan(
            str(uuid.uuid4()),
            uav.uid,
            "final_approach",
            points,
            moment_now if has_prefix else departure_start,
            moment,
            (reservation.reservation_id,),
            details,
        )

    def final_approach(self, uav, lz, reservation, now: float, *, scheduled_delay: float = 0.0,
                       speed_residual_mps: float = 0.0, airspace=None) -> TrajectoryPlan:
        if self.config.trajectory_schema_version == "executable_trajectory_v3":
            return self._final_approach_v3(
                uav, lz, reservation, now,
                scheduled_delay=scheduled_delay,
                speed_residual_mps=speed_residual_mps,
                airspace=airspace,
            )
        if self.config.final_approach_execution_mode == "speed_timing_v1":
            return self._final_approach_speed_timing_v1(
                uav,
                lz,
                reservation,
                now,
                scheduled_delay=scheduled_delay,
                speed_residual_mps=speed_residual_mps,
                airspace=airspace,
            )
        if (self.config.final_approach_geometry_mode == "continuous_inward_v1"
                and uav.ring_role is RingRole.INNER and airspace is not None):
            points, departure_start, moment, staging, cruise_height, has_prefix = (
                self._continuous_inward_landing_points(
                    uav, lz, airspace, now, scheduled_delay=max(0.0, scheduled_delay),
                )
            )
            details = {
                "target_phase": "landing",
                "scheduled_departure": departure_start,
                "scheduled_delay": max(0.0, scheduled_delay),
                "speed_residual_mps": float(speed_residual_mps),
                "final_approach_geometry": "continuous_inward_v1",
                "lz_id": lz.lz_id,
                "corridor_height": cruise_height,
                "source_ring_role": uav.ring_role.value,
                "staging_point": staging.tolist(),
                "vehicle_profile_id": self.config.vehicle_profile_id,
                "vehicle_profile_version": self.config.vehicle_profile.profile_version,
            }
            return TrajectoryPlan(
                str(uuid.uuid4()), uav.uid, "final_approach", points,
                now if has_prefix else departure_start, moment,
                (reservation.reservation_id,), details,
            )

        # A final approach cannot turn directly from a circulating 16 m/s
        # holder into an arbitrary descent curve.  Allocate an explicit,
        # finite-deceleration inner-ring prefix before the final geometry.
        braking_delay = (uav.speed / self.dynamics.profile.max_horizontal_decel
                          if uav.ring_role is RingRole.INNER and airspace is not None else 0.0)
        start = now + max(0.0, scheduled_delay, braking_delay)
        p0 = np.asarray(uav.position, dtype=float)
        initial_velocity = np.asarray(uav.velocity, dtype=float)
        prefix: list[TrajectoryPoint] = []
        if start > now + 1e-9 and uav.ring_role is RingRole.INNER and airspace is not None:
            ring = airspace.actual_ring(RingRole.INNER)
            theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(p0[1], p0[0])
            hold_points = self.ring_hold_points(ring, theta, uav.speed, 0.0, now, start - now)
            prefix = hold_points[:-1]
            start_state = hold_points[-1]
            p0, initial_velocity = start_state.position, start_state.velocity
        # A cubic Bezier with matched ingress tangent removes the historical
        # polyline corner while the parameterizer supplies finite acceleration.
        endpoint = np.asarray(lz.position, dtype=float)
        horizontal = endpoint - p0; horizontal[2] = 0.0
        direction = self._unit_or(np.asarray((0.0, -1.0, 0.0)), initial_velocity)
        distance = float(np.linalg.norm(endpoint - p0))
        p1 = p0 + min(.30 * distance, 120.0) * direction
        p2 = endpoint - min(.25 * distance, 80.0) * np.asarray((direction[0], direction[1], 0.0))
        path = GeometricPath.cubic_bezier(p0, p1, p2, endpoint, samples=self.config.trajectory_final_samples)
        proxy = type("State", (), {"uid": uav.uid, "velocity": initial_velocity})()
        final_points = self.parameterizer.parameterize(path, start_time=start, initial_velocity=proxy.velocity,
                                                       target_horizontal_speed=min(10.0, self.config.max_speed),
                                                       constraints=TrajectoryConstraints(end_horizontal_speed=0.0, tangent_yaw=False))
        details = {"target_phase": "landing", "scheduled_departure": start,
                   "scheduled_delay": max(0.0, scheduled_delay), "speed_residual_mps": float(speed_residual_mps),
                   "vehicle_profile_id": self.config.vehicle_profile_id,
                   "vehicle_profile_version": self.config.vehicle_profile.profile_version}
        return TrajectoryPlan(str(uuid.uuid4()), uav.uid, "final_approach", (*prefix, *final_points), now if prefix else start,
                              final_points[-1].t, (reservation.reservation_id,), details)

    def emergency_landing(self, uav, lz, airspace, now: float) -> TrajectoryPlan:
        """Fixed over-ring exit, inward corridor and vertical pad descent.

        The route is a candidate only. The caller must reserve its pad and
        certify the whole timed trajectory with SafetyFilter before execution.
        """
        if uav.ring_role is None or uav.committed_plan is not None:
            raise ValueError("emergency corridor requires an uncommitted ring holder")
        if self.config.trajectory_schema_version == "executable_trajectory_v3":
            result = self._v3_continuous_inward_landing_points(
                uav, lz, airspace, now,
                scheduled_delay=0.0,
                target_horizontal_speed=max(0.1, min(float(uav.speed), self.config.max_speed)),
                holding_target_speed=max(0.1, min(float(uav.speed), self.config.max_speed)),
            )
            details = self._schema_metadata({
                "target_phase": "emergency_landing",
                "lz_id": lz.lz_id,
                "corridor_height": result["cruise_height"],
                "source_ring_role": uav.ring_role.value,
                "staging_point": result["staging"].tolist(),
                "scheduled_departure": result["departure_start"],
                "accepted_reservation_departure": result["departure_start"],
                "actual_entry_speed_mps": result["entry_speed"],
                "vehicle_profile_id": self.config.vehicle_profile_id,
                "vehicle_profile_version": self.config.vehicle_profile.profile_version,
            })
            plan = TrajectoryPlan(
                str(uuid.uuid4()), uav.uid, "emergency_landing", result["points"],
                float(now), result["moment"], metadata=details,
            )
            if self.validate_kinematics(plan):
                raise RuntimeError("v3_emergency_landing_kinematics_invalid")
            return plan
        points, _, moment, staging, cruise_height, _ = self._continuous_inward_landing_points(
            uav, lz, airspace, now,
        )
        return TrajectoryPlan(str(uuid.uuid4()), uav.uid, "emergency_landing", tuple(points), now, moment,
                              metadata={"target_phase": "emergency_landing", "lz_id": lz.lz_id,
                                        "corridor_height": cruise_height,
                                        "source_ring_role": uav.ring_role.value,
                                        "staging_point": staging.tolist(),
                                        "vehicle_profile_id": self.config.vehicle_profile_id,
                                        "vehicle_profile_version": self.config.vehicle_profile.profile_version})

    def validate_kinematics(self, plan: TrajectoryPlan) -> list[str]:
        if plan.metadata.get("trajectory_schema_version") == "executable_trajectory_v3":
            errors = self.dynamics.validate_trajectory(plan)
        else:
            errors = self.dynamics.validate_plan(plan.points)
        if plan.metadata.get("dynamic_unreachable_merge_time"):
            errors.append("dynamic_unreachable_merge_time")
        return errors

    def estimate_minimum_duration(self, path: GeometricPath, uav, target_speed: float) -> float:
        return self.parameterizer.estimate_duration(path, initial_velocity=uav.velocity, target_horizontal_speed=target_speed)
