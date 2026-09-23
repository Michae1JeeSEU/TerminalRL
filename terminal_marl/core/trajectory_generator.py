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
from .trajectory_time_parameterization import GeometricPath, TrajectoryConstraints, TrajectoryTimeParameterizer
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "reservation_ids", tuple(self.reservation_ids))
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
        payload = [(round(point.t, 8), np.round(point.position, 8).tolist(), np.round(point.velocity, 8).tolist(),
                    np.round(point.acceleration, 8).tolist(), point.yaw, point.yaw_rate) for point in self.points]
        object.__setattr__(self, "_trajectory_hash", hashlib.sha256(json.dumps(payload, separators=(",", ":"), default=float).encode("utf-8")).hexdigest())
        self.metadata.setdefault("trajectory_hash", self._trajectory_hash)

    @property
    def trajectory_hash(self) -> str:
        return self._trajectory_hash

    def verify_trajectory_integrity(self) -> bool:
        """Expensive opt-in integrity check for qualification/debug runs."""
        payload = [(round(point.t, 8), np.round(point.position, 8).tolist(), np.round(point.velocity, 8).tolist(),
                    np.round(point.acceleration, 8).tolist(), point.yaw, point.yaw_rate) for point in self.points]
        actual = hashlib.sha256(json.dumps(payload, separators=(",", ":"), default=float).encode("utf-8")).hexdigest()
        return actual == self._trajectory_hash

    def with_reservation_ids(self, reservation_ids: list[str] | tuple[str, ...]) -> "TrajectoryPlan":
        """Attach an external reservation without changing trajectory bytes."""
        return TrajectoryPlan(self.plan_id, self.uav_id, self.mode, self.points, self.start_time, self.end_time,
                              tuple(reservation_ids), dict(self.metadata))

    def sample(self, now: float) -> TrajectoryPoint:
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
        if now <= self.start_time:
            return self.points[0].position
        if now >= self.end_time:
            return self.points[-1].position
        right_index = bisect_right(self.points, now, key=lambda point: point.t)
        left, right = self.points[right_index - 1], self.points[right_index]
        fraction = (now - left.t) / (right.t - left.t)
        return left.position + fraction * (right.position - left.position)


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
        details = {**(metadata or {}), "vehicle_profile_id": self.config.vehicle_profile_id,
                   "vehicle_profile_version": self.config.vehicle_profile.profile_version}
        return TrajectoryPlan(str(uuid.uuid4()), uav.uid, mode, points, start, points[-1].t,
                              () if reservation is None else (reservation.reservation_id,), details)

    def _ring_position(self, ring, theta: float) -> np.ndarray:
        return np.asarray((ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height), dtype=float)

    def ring_hold_points(self, ring, theta: float, current_speed: float, target_speed: float, start: float, duration: float) -> list[TrajectoryPoint]:
        """Exact finite-acceleration circular holder samples for all consumers."""
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

    def inbound(self, uav, gate, airspace, now: float) -> TrajectoryPlan:
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

    def ring_transfer(self, uav, source: RingRole, target: RingRole, reservation, airspace, now: float,
                      speed_residual_mps: float = 0.0) -> TrajectoryPlan:
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
            held = self.ring_hold_points(source_ring, initial_theta, uav.speed,
                                         transfer_entry_speed, now, moment - now) if moment > now + 1e-9 else []
            start_position = held[-1].position if held else current
            velocity = held[-1].velocity if held else uav.velocity
            angle = math.atan2(start_position[1], start_position[0])
            tangent = np.asarray((-math.sin(angle), math.cos(angle), 0.0))
            path, repaired = self._transfer_path(start_position, p3, tangent, target_tangent)
            result = self.parameterizer.parameterize(path, start_time=moment,
                initial_velocity=velocity,
                target_horizontal_speed=max(float(np.linalg.norm(velocity)), target_speed),
                constraints=TrajectoryConstraints(end_horizontal_speed=target_speed, tangent_yaw=False))
            return held[:-1], result, tangent, velocity, repaired

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
                if candidate[1][-1].t <= reservation.merge_time + 1e-6:
                    chosen = (float(moment), candidate)
                    break
            if chosen is None:
                raise RuntimeError("could not time-parameterize a dynamically feasible transfer")
            departure, (points, transfer_points, source_tangent, velocity, geometry_repaired) = chosen
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
        p0 = np.asarray(uav.position, dtype=float)
        initial_velocity = np.asarray(uav.velocity, dtype=float)
        high = min(self.config.height_max, self.dynamics.profile.max_operating_altitude_agl_m)
        cruise_height = high if uav.ring_role is not RingRole.INNER else float(p0[2])
        staging = np.asarray((lz.position[0], lz.position[1], cruise_height), dtype=float)

        # Carry the holder's actual tangential velocity into one smooth,
        # inward departure.  Braking to zero while circling on the source ring
        # leaves a stationary holder in the safety prediction tube and can
        # make every subsequent emergency candidate unreachable.  The first
        # Bezier control point is tangent to the source ring; the second one
        # approaches the staging point horizontally so its terminal tangent is
        # horizontal and the vertical pad leg can start from zero velocity.
        source_tangent = self._unit_or(np.asarray((-math.sin(
            uav.ring_progress if uav.ring_progress is not None else math.atan2(p0[1], p0[0])),
            math.cos(uav.ring_progress if uav.ring_progress is not None else math.atan2(p0[1], p0[0])),
            0.0)), initial_velocity)
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
            start_time=now,
            initial_velocity=initial_velocity,
            target_horizontal_speed=10.0,
            constraints=TrajectoryConstraints(end_horizontal_speed=0.0, tangent_yaw=False),
        )

        # Keep the existing final vertical leg, now starting at the staging
        # point reached by the continuous departure curve.
        pad = np.asarray(lz.position, dtype=float)
        points = list(departure[:-1])
        leg_path = GeometricPath.line(staging, pad, samples=17)
        leg = self.parameterizer.parameterize(
            leg_path,
            start_time=departure[-1].t,
            initial_velocity=departure[-1].velocity,
            target_horizontal_speed=5.0,
            constraints=TrajectoryConstraints(end_horizontal_speed=0.0, tangent_yaw=False),
        )
        points.extend(leg)
        moment = leg[-1].t
        return TrajectoryPlan(str(uuid.uuid4()), uav.uid, "emergency_landing", tuple(points), now, moment,
                              metadata={"target_phase": "emergency_landing", "lz_id": lz.lz_id,
                                        "corridor_height": cruise_height,
                                        "source_ring_role": uav.ring_role.value,
                                        "staging_point": staging.tolist(),
                                        "vehicle_profile_id": self.config.vehicle_profile_id,
                                        "vehicle_profile_version": self.config.vehicle_profile.profile_version})

    def validate_kinematics(self, plan: TrajectoryPlan) -> list[str]:
        errors = self.dynamics.validate_plan(plan.points)
        if plan.metadata.get("dynamic_unreachable_merge_time"):
            errors.append("dynamic_unreachable_merge_time")
        return errors

    def estimate_minimum_duration(self, path: GeometricPath, uav, target_speed: float) -> float:
        return self.parameterizer.estimate_duration(path, initial_velocity=uav.velocity, target_horizontal_speed=target_speed)
