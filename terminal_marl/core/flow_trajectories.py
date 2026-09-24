"""Canonical, finite-time flow trajectories used by the admission layer.

The legacy trajectory generator contains several geometry-specific candidate
builders.  This module provides a small deterministic builder for flow
transfers and inward approaches.  It only generates a candidate; callers must
still pass the returned plan through the normal safety and energy checks before
binding it to an aircraft.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import numpy as np

from .config import RingRole
from .trajectory_generator import TrajectoryPlan, TrajectoryPoint
from .trajectory_time_parameterization import GeometricPath, TrajectoryConstraints


@dataclass(frozen=True)
class FlowTrajectoryPreview:
    """Read-only endpoint view used before a full plan is materialised.

    A preview deliberately has no plan id, reservation ids, or trajectory hash.
    It contains only the endpoint data needed by merge/headway calculations.
    """

    end_time: float
    metadata: dict
    points: tuple[TrajectoryPoint, ...]

    def sample_position(self, now: float) -> np.ndarray:
        """Return the endpoint position for the endpoint sample."""
        if not math.isfinite(float(now)):
            raise ValueError("preview sample time must be finite")
        if abs(float(now) - float(self.end_time)) > 1e-8:
            raise ValueError("flow trajectory preview only supports endpoint sampling")
        return self.points[-1].position


class FlowTrajectoryGenerator:
    """Build finite-time, derivative-continuous flow trajectory candidates."""

    def __init__(self, config, generator) -> None:
        self.config = config
        self.generator = generator
        self.dynamics = generator.dynamics
        self.parameterizer = generator.parameterizer

    def _validate_inputs(self, uav, duration: float) -> float:
        try:
            value = float(duration)
        except (TypeError, ValueError) as exc:
            raise ValueError("duration must be a positive finite number") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("duration must be a positive finite number")
        if getattr(uav, "ring_role", None) is None:
            raise ValueError("flow trajectory requires a ring-assigned UAV")
        if getattr(uav, "committed_plan", None) is not None:
            raise ValueError("flow trajectory requires an uncommitted UAV")
        return value

    def _time_grid(self, duration: float) -> np.ndarray:
        step = float(self.config.physics_dt)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("physics_dt must be a positive finite number")
        count = max(1, int(math.ceil(duration / step)))
        # linspace makes the final point exact while retaining a maximum
        # interval no larger than the configured physics step.
        return np.linspace(0.0, duration, count + 1, dtype=float)

    @staticmethod
    def _shift_points(points: tuple[TrajectoryPoint, ...], offset: float) -> tuple[TrajectoryPoint, ...]:
        if abs(float(offset)) <= 1e-12:
            return points
        return tuple(
            TrajectoryPoint(point.t + offset, point.position, point.velocity,
                            point.acceleration, point.yaw, point.yaw_rate)
            for point in points
        )

    @staticmethod
    def _smoothstep(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        dq = 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
        ddq = 60.0 * s - 180.0 * s**2 + 120.0 * s**3
        return q, dq, ddq

    @staticmethod
    def _validate_tangential_speed_bias(value: float) -> float:
        try:
            bias = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("tangential_speed_bias_mps must be finite") from exc
        if not math.isfinite(bias):
            raise ValueError("tangential_speed_bias_mps must be finite")
        return bias

    @staticmethod
    def _unit_tangent(theta: float) -> np.ndarray:
        return np.asarray((-math.sin(theta), math.cos(theta), 0.0), dtype=float)

    def _prescreen_samples(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        context: str,
    ) -> None:
        """Reject an obviously infeasible sample array before point allocation.

        The public builder still runs the authoritative ``validate_plan`` after
        it constructs a candidate plan.  This predicate is only a cheap
        candidate-side filter for the two analytic flow helpers.
        """
        if not self.dynamics.samples_are_feasible(times, positions, velocities):
            raise RuntimeError(
                f"flow {context} failed dynamics validation: ['samples_are_feasible']"
            )

    def _prepare_ring_transfer(
        self,
        uav,
        target_ring,
        duration: float,
        tangential_speed_bias_mps: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sampled annular arrays before allocating trajectory points."""
        tangential_speed_bias_mps = self._validate_tangential_speed_bias(
            tangential_speed_bias_mps,
        )
        times = self._time_grid(duration)
        local = times / duration
        q, dq_ds, ddq_ds = self._smoothstep(local)
        bias_profile = 16.0 * local**2 * (1.0 - local)**2
        bias_profile_derivative = 32.0 * local * (1.0 - local) * (1.0 - 2.0 * local)

        current = np.asarray(uav.position, dtype=float)
        current_velocity = np.asarray(uav.velocity, dtype=float)
        current_acceleration = np.asarray(
            getattr(uav, "acceleration", np.zeros(3, dtype=float)), dtype=float
        )
        if current.shape != (3,) or current_velocity.shape != (3,) or current_acceleration.shape != (3,):
            raise ValueError("UAV position, velocity, and acceleration must be 3-D vectors")
        if not (np.isfinite(current).all() and np.isfinite(current_velocity).all()
                and np.isfinite(current_acceleration).all()):
            raise ValueError("UAV state must be finite")

        r0 = float(np.linalg.norm(current[:2]))
        if r0 <= 1e-9 or float(target_ring.radius) <= 1e-9:
            raise ValueError("ring transfer requires positive source and target radii")
        r1 = float(target_ring.radius)
        z0 = float(current[2])
        z1 = float(target_ring.height)
        v0 = float(np.linalg.norm(current_velocity[:2]))
        v1 = float(target_ring.nominal_speed)
        if not all(math.isfinite(value) for value in (r0, r1, z0, z1, v0, v1)):
            raise ValueError("ring geometry and velocity must be finite")

        r_delta = r1 - r0
        z_delta = z1 - z0
        v_delta = v1 - v0
        r_dot = r_delta * dq_ds / duration
        r_ddot = r_delta * ddq_ds / duration**2
        z_dot = z_delta * dq_ds / duration
        z_ddot = z_delta * ddq_ds / duration**2
        speed = v0 + v_delta * q + tangential_speed_bias_mps * bias_profile
        speed_dot = (
            v_delta * dq_ds / duration
            + tangential_speed_bias_mps * bias_profile_derivative / duration
        )
        radius = r0 + r_delta * q
        radius_dot = r_dot
        omega = speed / radius
        theta0 = math.atan2(float(current[1]), float(current[0]))
        theta = np.empty_like(times)
        theta[0] = theta0
        if len(times) > 1:
            theta[1:] = theta0 + np.cumsum(
                0.5 * (omega[:-1] + omega[1:]) * np.diff(times)
            )
        theta_dot = omega
        theta_ddot = speed_dot / radius - speed * radius_dot / radius**2

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        # ``e_r`` is the outward radial basis.  The signed radial derivative
        # is negative for an inward transfer; using an inward basis here would
        # reverse that motion and invalidate the analytic acceleration.
        radial = np.column_stack((cos_theta, sin_theta, np.zeros_like(theta)))
        tangent = np.column_stack((-sin_theta, cos_theta, np.zeros_like(theta)))
        positions = np.column_stack((radius * cos_theta, radius * sin_theta, z0 + z_delta * q))
        velocities = radial * radius_dot[:, None] + tangent * speed[:, None]
        velocities[:, 2] = z_dot
        accelerations = (
            radial * (r_ddot - speed**2 / radius)[:, None]
            + tangent * (speed_dot + speed * radius_dot / radius)[:, None]
        )
        accelerations[:, 2] = z_ddot

        # Preserve the live source state exactly.  The smooth profile starts
        # with the same radial/tangent derivatives for a normal ring holder,
        # while retaining the actual serialized state avoids a hidden snap.
        positions[0] = current
        velocities[0] = current_velocity
        accelerations[0] = current_acceleration
        # The final point is explicitly on the destination ring, including its
        # nominal tangential velocity.  theta[-1] is the integrated forward
        # bearing, so the transfer never reverses its circulation direction.
        positions[-1] = np.asarray((r1 * cos_theta[-1], r1 * sin_theta[-1], z1), dtype=float)
        velocities[-1] = self._unit_tangent(float(theta[-1])) * v1
        accelerations[-1] = (
            np.asarray((cos_theta[-1], sin_theta[-1], 0.0), dtype=float) * (-v1**2 / r1)
        )
        self._prescreen_samples(times, positions, velocities, "ring transfer")
        return times, positions, velocities, accelerations

    @staticmethod
    def _materialize_points(
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        offset: float = 0.0,
    ) -> tuple[TrajectoryPoint, ...]:
        """Convert prepared arrays to canonical immutable trajectory points."""
        return tuple(
            TrajectoryPoint(float(t) + float(offset), position, velocity, acceleration)
            for t, position, velocity, acceleration in zip(
                times, positions, velocities, accelerations
            )
        )

    def _ring_transfer_points(
        self,
        uav,
        target_ring,
        duration: float,
        tangential_speed_bias_mps: float = 0.0,
    ) -> tuple[TrajectoryPoint, ...]:
        """Compatibility wrapper returning the historical relative-time points."""
        prepared = self._prepare_ring_transfer(
            uav, target_ring, duration, tangential_speed_bias_mps,
        )
        return self._materialize_points(*prepared)

    def _ring_transfer_context(
        self,
        uav,
        target,
        airspace,
        now: float,
        duration: float,
        tangential_speed_bias_mps: float = 0.0,
    ):
        """Validate a transfer request and return its canonical build context."""
        duration = self._validate_inputs(uav, duration)
        tangential_speed_bias_mps = self._validate_tangential_speed_bias(
            tangential_speed_bias_mps,
        )
        try:
            source = (
                uav.ring_role
                if isinstance(uav.ring_role, RingRole)
                else RingRole(uav.ring_role)
            )
            target_role = target if isinstance(target, RingRole) else RingRole(target)
        except (TypeError, ValueError) as exc:
            raise ValueError("source and target must be valid ring roles") from exc
        if source is target_role:
            raise ValueError("ring transfer requires distinct source and target rings")
        if (source, target_role) not in {
            (RingRole.OUTER, RingRole.MIDDLE),
            (RingRole.MIDDLE, RingRole.INNER),
        }:
            raise ValueError("flow ring transfer requires an adjacent inward ring pair")
        try:
            # Resolve both rings here to retain the public helper's validation
            # semantics even though only the target geometry is sampled.
            airspace.actual_ring(source)
            target_ring = airspace.actual_ring(target_role)
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError("airspace must expose both source and target rings") from exc
        start = float(now)
        if not math.isfinite(start):
            raise ValueError("now must be finite")
        return duration, tangential_speed_bias_mps, source, target_role, target_ring, start

    def _ring_transfer_metadata(
        self,
        source: RingRole,
        target_role: RingRole,
        start: float,
        tangential_speed_bias_mps: float,
    ) -> dict:
        return {
            "target_phase": "ring_middle" if target_role is RingRole.MIDDLE else "ring_inner",
            "source_ring": source.value,
            "target_ring": target_role.value,
            "source_ring_role": source.value,
            "target_ring_role": target_role.value,
            "flow_geometry": "quintic_annular_v1",
            "scheduled_departure": start,
            "merge_time": start,
            "flow_tangential_speed_bias_mps": tangential_speed_bias_mps,
            "vehicle_profile_id": self.config.vehicle_profile_id,
            "vehicle_profile_version": self.config.vehicle_profile.profile_version,
        }

    def _materialize_ring_transfer(
        self,
        uav,
        source: RingRole,
        target_role: RingRole,
        start: float,
        duration: float,
        tangential_speed_bias_mps: float,
        prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> TrajectoryPlan:
        """Materialise a prepared transfer into the canonical executable plan."""
        times, positions, velocities, accelerations = prepared
        points = self._materialize_points(
            times, positions, velocities, accelerations, offset=start,
        )
        details = self._ring_transfer_metadata(
            source, target_role, start, tangential_speed_bias_mps,
        )
        end = start + duration
        details["merge_time"] = end
        return TrajectoryPlan(
            str(uuid.uuid4()), uav.uid, "ring_transfer", points, start, end,
            (), details,
        )

    def _ring_transfer_preview(
        self,
        uav,
        source: RingRole,
        target_role: RingRole,
        start: float,
        duration: float,
        tangential_speed_bias_mps: float,
        prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> FlowTrajectoryPreview:
        """Create an endpoint-only view without a plan id or trajectory hash."""
        _times, positions, velocities, accelerations = prepared
        end = float(start) + float(duration)
        endpoint = TrajectoryPoint(
            end, positions[-1], velocities[-1], accelerations[-1],
        )
        details = self._ring_transfer_metadata(
            source, target_role, float(start), tangential_speed_bias_mps,
        )
        details["merge_time"] = end
        return FlowTrajectoryPreview(end, details, (endpoint,))

    def ring_transfer(
        self,
        uav,
        target,
        airspace,
        now: float,
        duration: float,
        tangential_speed_bias_mps: float = 0.0,
    ) -> TrajectoryPlan:
        """Create a finite-time source-ring to target-ring flow candidate."""
        (
            duration,
            tangential_speed_bias_mps,
            source,
            target_role,
            target_ring,
            start,
        ) = self._ring_transfer_context(
            uav,
            target,
            airspace,
            now,
            duration,
            tangential_speed_bias_mps,
        )
        prepared = self._prepare_ring_transfer(
            uav,
            target_ring,
            duration,
            tangential_speed_bias_mps,
        )
        plan = self._materialize_ring_transfer(
            uav,
            source,
            target_role,
            start,
            duration,
            tangential_speed_bias_mps,
            prepared,
        )
        errors = self.dynamics.validate_plan(plan.points)
        if errors:
            raise RuntimeError(f"flow ring transfer failed dynamics validation: {errors}")
        return plan

    @staticmethod
    def _quintic_coefficients(
        start_position: np.ndarray,
        start_velocity: np.ndarray,
        start_acceleration: np.ndarray,
        end_position: np.ndarray,
        duration: float,
    ) -> np.ndarray:
        """Return normalized-time quintic coefficients, shape ``(6, 3)``."""
        c0 = np.asarray(start_position, dtype=float)
        c1 = float(duration) * np.asarray(start_velocity, dtype=float)
        c2 = 0.5 * float(duration) ** 2 * np.asarray(start_acceleration, dtype=float)
        rhs = np.vstack(
            (
                np.asarray(end_position, dtype=float) - c0 - c1 - c2,
                -c1 - 2.0 * c2,
                -2.0 * c2,
            )
        )
        matrix = np.asarray(
            ((1.0, 1.0, 1.0), (3.0, 4.0, 5.0), (6.0, 12.0, 20.0)),
            dtype=float,
        )
        c3, c4, c5 = np.linalg.solve(matrix, rhs)
        return np.vstack((c0, c1, c2, c3, c4, c5))

    def _inward_departure_points(self, uav, staging: np.ndarray, duration: float) -> tuple[TrajectoryPoint, ...]:
        times = self._time_grid(duration)
        tau = times / duration
        current = np.asarray(uav.position, dtype=float)
        velocity = np.asarray(uav.velocity, dtype=float)
        acceleration = np.asarray(getattr(uav, "acceleration", np.zeros(3)), dtype=float)
        coefficients = self._quintic_coefficients(current, velocity, acceleration, staging, duration)
        powers = np.column_stack((
            np.ones_like(tau), tau, tau**2, tau**3, tau**4, tau**5,
        ))
        positions = powers @ coefficients
        first = np.column_stack((
            np.zeros_like(tau),
            np.ones_like(tau),
            2.0 * tau,
            3.0 * tau**2,
            4.0 * tau**3,
            5.0 * tau**4,
        )) @ coefficients / duration
        second = np.column_stack((
            np.zeros_like(tau),
            np.zeros_like(tau),
            2.0 * np.ones_like(tau),
            6.0 * tau,
            12.0 * tau**2,
            20.0 * tau**3,
        )) @ coefficients / duration**2
        positions[0] = current
        first[0] = velocity
        second[0] = acceleration
        positions[-1] = staging
        first[-1] = 0.0
        second[-1] = 0.0
        self._prescreen_samples(times, positions, first, "final inward departure")
        return tuple(
            TrajectoryPoint(float(t), p, v, a)
            for t, p, v, a in zip(times, positions, first, second)
        )

    def final_approach(self, uav, lz, airspace, now: float, duration: float, emergency: bool = False) -> TrajectoryPlan:
        """Create a smooth inward departure followed by a vertical pad leg."""
        duration = self._validate_inputs(uav, duration)
        start = float(now)
        if not math.isfinite(start):
            raise ValueError("now must be finite")
        try:
            source = uav.ring_role if isinstance(uav.ring_role, RingRole) else RingRole(uav.ring_role)
            source_ring = airspace.actual_ring(source)
            pad = np.asarray(lz.position, dtype=float)
            lz_id = lz.lz_id
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("final approach requires a ring, airspace, and landing zone") from exc
        if not emergency and source is not RingRole.INNER:
            raise ValueError("normal final approach requires an inner-ring UAV")
        if pad.shape != (3,) or not np.isfinite(pad).all():
            raise ValueError("landing-zone position must be a finite 3-D vector")
        profile_height = min(float(self.config.height_max), float(self.dynamics.profile.max_operating_altitude_agl_m))
        cruise_height = float(uav.position[2]) if source is RingRole.INNER else profile_height
        staging = np.asarray((pad[0], pad[1], cruise_height), dtype=float)
        departure = self._shift_points(
            self._inward_departure_points(uav, staging, duration), start,
        )

        leg_path = GeometricPath.line(staging, pad, samples=17)
        leg = self.parameterizer.parameterize(
            leg_path,
            start_time=start + duration,
            initial_velocity=departure[-1].velocity,
            target_horizontal_speed=5.0,
            constraints=TrajectoryConstraints(end_horizontal_speed=0.0, tangent_yaw=False),
        )
        points = tuple(departure[:-1]) + tuple(leg)
        end = float(leg[-1].t)
        target_phase = "emergency_landing" if emergency else "landing"
        mode = "emergency_landing" if emergency else "final_approach"
        details = {
            "target_phase": target_phase,
            "source_ring": source.value,
            "source_ring_role": source.value,
            "flow_geometry": "quintic_inward_v1",
            "scheduled_departure": start,
            "staging_point": staging.tolist(),
            "corridor_height": cruise_height,
            "lz_id": lz_id,
            "vehicle_profile_id": self.config.vehicle_profile_id,
            "vehicle_profile_version": self.config.vehicle_profile.profile_version,
        }
        plan = TrajectoryPlan(str(uuid.uuid4()), uav.uid, mode, points, start, end, (), details)
        errors = self.dynamics.validate_plan(plan.points)
        if errors:
            raise RuntimeError(f"flow final approach failed dynamics validation: {errors}")
        return plan
