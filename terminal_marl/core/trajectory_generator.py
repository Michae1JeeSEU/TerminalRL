"""Authoritative geometric-path and executable-trajectory generation."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import uuid

import numpy as np

from .config import RingRole
from .trajectory_time_parameterization import GeometricPath, TrajectoryConstraints, TrajectoryTimeParameterizer
from .vehicle_dynamics import VehicleDynamics


@dataclass
class TrajectoryPoint:
    t: float
    position: np.ndarray
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    yaw: float | None = None
    yaw_rate: float | None = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.zeros(3, dtype=float) if self.velocity is None else np.asarray(self.velocity, dtype=float)
        self.acceleration = np.zeros(3, dtype=float) if self.acceleration is None else np.asarray(self.acceleration, dtype=float)


@dataclass
class TrajectoryPlan:
    plan_id: str
    uav_id: int
    mode: str
    points: list[TrajectoryPoint]
    start_time: float
    end_time: float
    reservation_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.points) < 2 or self.end_time <= self.start_time:
            raise ValueError("trajectory plan requires at least two increasing-time points")
        if any(right.t <= left.t for left, right in zip(self.points, self.points[1:])):
            raise ValueError("trajectory points must have strictly increasing timestamps")
        self.metadata.setdefault("trajectory_schema_version", "executable_trajectory_v2")
        # There is one permitted research-vehicle profile in the main
        # simulator.  Retain that explicit identity even for small test and
        # diagnostic plans constructed directly rather than by the generator.
        self.metadata.setdefault("vehicle_profile_id", "research_evtol")
        self.metadata.setdefault("vehicle_profile_version", "eh216s_research_v1")
        self.metadata.setdefault("trajectory_hash", self.trajectory_hash)

    @property
    def trajectory_hash(self) -> str:
        payload = [(round(point.t, 8), np.round(point.position, 8).tolist(), np.round(point.velocity, 8).tolist(),
                    np.round(point.acceleration, 8).tolist(), point.yaw, point.yaw_rate) for point in self.points]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":"), default=float).encode("utf-8")).hexdigest()

    def sample(self, now: float) -> TrajectoryPoint:
        if now <= self.start_time:
            return self.points[0]
        if now >= self.end_time:
            return self.points[-1]
        for left, right in zip(self.points, self.points[1:]):
            if left.t <= now <= right.t:
                fraction = (now - left.t) / (right.t - left.t)
                yaw = None if left.yaw is None or right.yaw is None else left.yaw + fraction * (right.yaw - left.yaw)
                yaw_rate = None if left.yaw_rate is None or right.yaw_rate is None else left.yaw_rate + fraction * (right.yaw_rate - left.yaw_rate)
                return TrajectoryPoint(now, left.position + fraction * (right.position - left.position),
                                       left.velocity + fraction * (right.velocity - left.velocity),
                                       left.acceleration + fraction * (right.acceleration - left.acceleration), yaw, yaw_rate)
        return self.points[-1]


class TrajectoryGenerator:
    def __init__(self, config) -> None:
        self.config = config
        self.dynamics = VehicleDynamics(config.vehicle_profile)
        self.parameterizer = TrajectoryTimeParameterizer(self.dynamics)

    def _plan(self, uav, mode: str, path: GeometricPath, start: float, target_speed: float, *, reservation=None,
              metadata: dict | None = None, end_speed: float | None = None) -> TrajectoryPlan:
        points = self.parameterizer.parameterize(path, start_time=start, initial_velocity=uav.velocity,
                                                 target_horizontal_speed=target_speed,
                                                 constraints=TrajectoryConstraints(end_horizontal_speed=end_speed, tangent_yaw=False))
        plan = TrajectoryPlan(str(uuid.uuid4()), uav.uid, mode, points, start, points[-1].t,
                              [] if reservation is None else [reservation.reservation_id], metadata or {})
        plan.metadata.update({"vehicle_profile_id": self.config.vehicle_profile_id,
                              "vehicle_profile_version": self.config.vehicle_profile.profile_version,
                              "trajectory_hash": plan.trajectory_hash})
        return plan

    def _ring_position(self, ring, theta: float) -> np.ndarray:
        return np.asarray((ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height), dtype=float)

    def ring_hold_points(self, ring, theta: float, current_speed: float, target_speed: float, start: float, duration: float) -> list[TrajectoryPoint]:
        """Exact finite-acceleration circular holder samples for all consumers."""
        samples = max(2, int(math.ceil(duration / self.config.physics_dt)) + 1)
        step = duration / (samples - 1)
        points: list[TrajectoryPoint] = []
        speed = max(0.0, current_speed)
        angle = theta
        for index in range(samples):
            if index == 0:
                tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0))
                radial = np.asarray((-math.cos(angle), -math.sin(angle), 0.0))
                points.append(TrajectoryPoint(start, self._ring_position(ring, angle), tangent * speed,
                                              radial * (speed * speed / ring.radius)))
                continue
            motion = self.dynamics.ring_motion_step(ring, angle, speed, target_speed, step)
            angle, speed = motion.theta, motion.speed
            points.append(TrajectoryPoint(start + index * step, motion.position, motion.velocity, motion.acceleration))
        return points

    def inbound(self, uav, gate, airspace, now: float) -> TrajectoryPlan:
        ring = airspace.actual_ring(RingRole.OUTER)
        angle = gate.angle
        target = self._ring_position(ring, angle)
        tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0))
        distance = float(np.linalg.norm(target - uav.position))
        control = min(.35 * distance, 160.0)
        path = GeometricPath.cubic_bezier(uav.position, uav.position + control * self._unit_or(tangent, uav.velocity),
                                          target - control * tangent, target)
        return self._plan(uav, "inbound", path, now, ring.nominal_speed,
                          metadata={"target_phase": "ring_outer"}, end_speed=ring.nominal_speed)

    @staticmethod
    def _unit_or(fallback: np.ndarray, value: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        return fallback if norm <= 1e-9 else np.asarray(value, dtype=float) / norm

    def ring_transfer(self, uav, source: RingRole, target: RingRole, reservation, airspace, now: float,
                      speed_residual_mps: float = 0.0) -> TrajectoryPlan:
        current = np.asarray(uav.position, dtype=float)
        source_ring, target_ring = airspace.actual_ring(source), airspace.actual_ring(target)
        theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(current[1], current[0])
        target_theta = reservation.merge_angle or 0.0
        source_target_speed = float(np.clip(source_ring.nominal_speed + airspace.speed_profile_bias + speed_residual_mps,
                                            self.config.speed_min, self.config.max_speed))
        # A curved transfer cannot inherit an arbitrary high ring velocity at
        # its first Bezier sample.  Deceleration remains an explicit source
        # ring arc, never an instantaneous change at departure.
        transfer_entry_speed = 0.0
        target_speed = float(np.clip(target_ring.nominal_speed + airspace.speed_profile_bias + speed_residual_mps,
                                     self.config.speed_min, self.config.max_speed))
        departure = now
        p0 = current
        p3 = self._ring_position(target_ring, target_theta)
        for _ in range(3):
            length = float(np.linalg.norm(p3 - p0))
            source_tangent = np.asarray((-math.sin(theta), math.cos(theta), 0.0))
            target_tangent = np.asarray((-math.sin(target_theta), math.cos(target_theta), 0.0))
            tangent_length = min(.42 * length, 160.0)
            transfer_path = GeometricPath.cubic_bezier(p0, p0 + tangent_length * source_tangent,
                                                        p3 - tangent_length * target_tangent, p3)
            # The transfer curve begins only after the explicit source-ring
            # braking prefix, so its duration must be evaluated from that
            # zero-speed hand-off rather than the current holder velocity.
            duration = self.parameterizer.estimate_duration(transfer_path, initial_velocity=np.zeros(3),
                                                             target_horizontal_speed=target_speed,
                                                             constraints=TrajectoryConstraints(end_horizontal_speed=target_speed, tangent_yaw=False))
            braking_time = max(0.0, uav.speed - transfer_entry_speed) / self.dynamics.profile.max_horizontal_decel
            departure = max(now + braking_time, reservation.merge_time - duration)
            theta = (uav.ring_progress if uav.ring_progress is not None else math.atan2(current[1], current[0])) + source_target_speed * (departure - now) / source_ring.radius
            p0 = self._ring_position(source_ring, theta)
        points: list[TrajectoryPoint] = []
        transfer_uav = uav
        if departure > now + 1e-9:
            initial_theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(current[1], current[0])
            held = self.ring_hold_points(source_ring, initial_theta, uav.speed, transfer_entry_speed, now, departure - now)
            points = held[:-1]
            p0 = held[-1].position
            theta = math.atan2(p0[1], p0[0])
            source_tangent = np.asarray((-math.sin(theta), math.cos(theta), 0.0))
            length = float(np.linalg.norm(p3 - p0)); tangent_length = min(.42 * length, 160.0)
            transfer_path = GeometricPath.cubic_bezier(p0, p0 + tangent_length * source_tangent,
                                                        p3 - tangent_length * target_tangent, p3)
            transfer_uav = type("State", (), {"uid": uav.uid, "velocity": held[-1].velocity})()
        transfer = self._plan(transfer_uav, "ring_transfer", transfer_path, departure, target_speed, reservation=reservation,
                              metadata={"target_phase": "ring_middle" if target is RingRole.MIDDLE else "ring_inner",
                                        "source_tangent": source_tangent, "target_tangent": target_tangent,
                                        "speed_residual_mps": float(speed_residual_mps), "scheduled_departure": departure,
                                        "merge_time": reservation.merge_time}, end_speed=target_speed)
        if points:
            # Preserve exactly the trajectory used in planning; hand-off only
            # joins at the same sampled start point/time.
            transfer.points = [*points, *transfer.points]
            transfer.start_time = now
            transfer.metadata["trajectory_hash"] = transfer.trajectory_hash
        if transfer.end_time > reservation.merge_time + 1e-6:
            transfer.metadata["dynamic_unreachable_merge_time"] = True
        return transfer

    def final_approach(self, uav, lz, reservation, now: float, *, scheduled_delay: float = 0.0,
                       speed_residual_mps: float = 0.0, airspace=None) -> TrajectoryPlan:
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
        path = GeometricPath.cubic_bezier(p0, p1, p2, endpoint, samples=101)
        proxy = type("State", (), {"uid": uav.uid, "velocity": initial_velocity})()
        plan = self._plan(proxy, "final_approach", path, start, min(10.0, self.config.max_speed), reservation=reservation,
                          metadata={"target_phase": "landing", "scheduled_departure": start,
                                    "scheduled_delay": max(0.0, scheduled_delay), "speed_residual_mps": float(speed_residual_mps)},
                          end_speed=0.0)
        if prefix:
            plan.points = [*prefix, *plan.points]
            plan.start_time = now
            plan.metadata["trajectory_hash"] = plan.trajectory_hash
        return plan

    def validate_kinematics(self, plan: TrajectoryPlan) -> list[str]:
        errors = self.dynamics.validate_plan(plan.points)
        if plan.metadata.get("dynamic_unreachable_merge_time"):
            errors.append("dynamic_unreachable_merge_time")
        return errors

    def estimate_minimum_duration(self, path: GeometricPath, uav, target_speed: float) -> float:
        return self.parameterizer.estimate_duration(path, initial_velocity=uav.velocity, target_horizontal_speed=target_speed)
