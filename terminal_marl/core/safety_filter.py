"""Resource/intent then canonical-plan safety verification."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .runtime_separation import swept_minimum_distance
from .vehicle_dynamics import VehicleDynamics
@dataclass(frozen=True)
class IntentSafetyResult: approved:bool;reason:str=""
@dataclass(frozen=True)
class TrajectorySafetyResult:
    approved:bool;reason:str="";checked_plan_id:str|None=None;checked_trajectory_hash:str|None=None
class SafetyFilter:
    def __init__(self,config):
        self.config=config
        self._v3_enabled = config.trajectory_schema_version == "executable_trajectory_v3"
        self.dynamics=VehicleDynamics(config.vehicle_profile, continuous_curve_limits=self._v3_enabled)
        self.checked_plan_ids=[];self.checked_trajectory_hashes={}
        self._holder_track_cache={}
        self._holder_tube_cache={}
        self._committed_track_cache={}
        # Candidate verifications are auditable.  ``None`` is deliberately
        # distinct from a safe infinite separation so callers cannot turn a
        # missing prediction into a passing safety metric.
        self.last_predicted_track_audit={"minimum_separation": None, "checked_pairs": 0, "reason": "not_run"}

    @property
    def _horizontal_acceleration_bound(self) -> float:
        return min(
            float(self.config.vehicle_profile.max_horizontal_accel),
            float(self.config.vehicle_profile.max_horizontal_decel),
            float(self.config.vehicle_profile.max_lateral_accel),
        ) + 1.0e-3

    @property
    def _flight_acceleration_bound(self) -> float:
        profile = self.config.vehicle_profile
        vertical_bound = max(
            float(profile.max_vertical_accel_up),
            float(profile.max_vertical_accel_down),
        ) + 1.0e-3
        return math.sqrt(self._horizontal_acceleration_bound ** 2 + vertical_bound ** 2)

    @property
    def _flight_speed_bound(self) -> float:
        profile = self.config.vehicle_profile
        return math.sqrt((float(profile.max_horizontal_speed) + 1.0e-3) ** 2 + max(
            float(profile.max_climb_rate),
            float(profile.max_descent_rate),
        ) ** 2 + 2.0e-3 * max(float(profile.max_climb_rate), float(profile.max_descent_rate)) + 1.0e-6)

    @property
    def _holder_speed_bound(self) -> float:
        return float(self.config.vehicle_profile.max_horizontal_speed) + 1.0e-3

    @property
    def _holder_acceleration_bound(self) -> float:
        return self._horizontal_acceleration_bound

    def _track_descriptor(self, track):
        """Return the bounded source kind used by the V3 separation margin."""
        if not self._v3_enabled:
            return "legacy"
        metadata = getattr(track, "metadata", {})
        if not isinstance(metadata, dict):
            return None
        if metadata.get("runtime_executor_equivalent", False):
            return {"kind": "holder_proxy", "resampled": True}
        if metadata.get("trajectory_schema_version") == "executable_trajectory_v3":
            try:
                if bool(track.verify_trajectory_integrity()):
                    return "flight_v3"
            except (AttributeError, TypeError, ValueError):
                return None
            return None
        if metadata.get("trajectory_schema_version", "executable_trajectory_v2") == "executable_trajectory_v2":
            return "legacy" if not self._v3_enabled else None
        return None

    def _track_error_budget(self, descriptor: str | None, dt: float) -> float | None:
        if not self._v3_enabled or descriptor == "legacy":
            return 0.0
        if descriptor == "flight_v3":
            return self._flight_acceleration_bound * float(dt) ** 2 / 8.0
        if descriptor == "holder_proxy" or (
                isinstance(descriptor, dict) and descriptor.get("kind") == "holder_proxy"):
            resampled = isinstance(descriptor, dict) and bool(descriptor.get("resampled", False))
            source_error = float(self.config.physics_dt) ** 2 if resampled else 0.0
            return self._holder_acceleration_bound * (float(dt) ** 2 + source_error) / 8.0
        if isinstance(descriptor, dict) and descriptor.get("kind") == "flight_v3":
            return self._flight_acceleration_bound * float(dt) ** 2 / 8.0
        return None

    def _pair_curve_error(self, left_descriptor: str | None, right_descriptor: str | None, dt: float) -> float | None:
        left = self._track_error_budget(left_descriptor, dt)
        right = self._track_error_budget(right_descriptor, dt)
        if left is None or right is None:
            return None
        return left + right

    def _track_point_error(self, descriptor) -> float | None:
        if not self._v3_enabled or descriptor == "legacy":
            return 0.0
        if descriptor == "holder_proxy":
            return 0.0
        if isinstance(descriptor, dict) and descriptor.get("kind") == "holder_proxy":
            if descriptor.get("resampled", False):
                return self._holder_acceleration_bound * float(self.config.physics_dt) ** 2 / 8.0
            return 0.0
        if descriptor == "flight_v3" or (isinstance(descriptor, dict) and descriptor.get("kind") == "flight_v3"):
            return 0.0
        return None

    def _track_distances_with_margin(self, left, right, times, left_descriptor, right_descriptor):
        point, swept = self._track_distances(left, right)
        if not self._v3_enabled:
            return point, swept
        left_point_error = self._track_point_error(left_descriptor)
        right_point_error = self._track_point_error(right_descriptor)
        if left_point_error is None or right_point_error is None:
            point = np.full_like(point, -np.inf)
        else:
            point = point - left_point_error - right_point_error
        if len(swept) == 0:
            return point, swept
        values = np.asarray(times, dtype=float)
        if len(values) != len(left) or len(values) != len(right):
            return point, np.full_like(swept, -np.inf)
        corrections = []
        for dt in np.diff(values):
            correction = self._pair_curve_error(left_descriptor, right_descriptor, float(dt))
            corrections.append(np.inf if correction is None else correction)
        return point, np.maximum(0.0, swept - np.asarray(corrections, dtype=float))

    def _tube_distances_with_margin(self, points, ring, lower, upper, times, candidate_descriptor):
        point = self._tube_point_distances(points, ring, lower, upper)
        if not self._v3_enabled or len(point) < 2:
            return point
        values = np.asarray(times, dtype=float)
        if len(values) != len(point):
            return np.full(len(point), -np.inf, dtype=float)
        interval_distances = np.minimum(point[:-1], point[1:])
        corrections = []
        for dt in np.diff(values):
            correction = self._pair_curve_error(candidate_descriptor, "holder_proxy", float(dt))
            if correction is None:
                corrections.append(np.inf)
            else:
                speed_margin = (self._flight_speed_bound + self._holder_speed_bound) * float(dt) / 2.0
                corrections.append(correction + speed_margin)
        return np.concatenate((point, interval_distances - np.asarray(corrections, dtype=float)))
    def begin_prediction_step(self):
        """Invalidate reusable holder tracks after an authoritative state tick."""
        self._holder_track_cache.clear()
        self._holder_tube_cache.clear()
        self._committed_track_cache.clear()
    def filter_intent(self,uav,intent,*,interface_frozen=False,lz_available=True,final_available=True):
        if interface_frozen:return IntentSafetyResult(False,"geometry_reconfiguring")
        if not lz_available:return IntentSafetyResult(False,"lz_unavailable")
        if not final_available:return IntentSafetyResult(False,"final_unavailable")
        if uav.soc<=0:return IntentSafetyResult(False,"energy_depleted")
        return IntentSafetyResult(True)
    def _holding_prediction(self, uav, snapshot, time: float) -> np.ndarray:
        """Predict an uncommitted holder on its current actual ring reference."""
        if uav.ring_role is None:
            return np.asarray(uav.position, dtype=float)
        ring = snapshot.airspace.actual_ring(uav.ring_role)
        theta = uav.ring_progress
        if theta is None:
            theta = float(np.arctan2(uav.position[1], uav.position[0]))
        target = float(np.clip(ring.nominal_speed + snapshot.airspace.speed_profile_bias, self.config.speed_min, self.config.max_speed))
        return self._ring_prediction(ring, theta, uav.speed, target, max(0.0, time - snapshot.now))

    def _post_plan_prediction(self, other_plan, snapshot, time: float) -> np.ndarray:
        endpoint = other_plan.points[-1].position
        target = {"ring_outer": "outer", "ring_middle": "middle", "ring_inner": "inner"}.get(other_plan.metadata.get("target_phase"))
        if target is None:
            return endpoint
        from .config import RingRole
        role = RingRole(target)
        ring = snapshot.airspace.actual_ring(role)
        theta = float(np.arctan2(endpoint[1], endpoint[0]))
        end = other_plan.points[-1]
        target_speed = float(np.clip(ring.nominal_speed + snapshot.airspace.speed_profile_bias, self.config.speed_min, self.config.max_speed))
        return self._ring_prediction(ring, theta, float(np.linalg.norm(end.velocity[:2])), target_speed,
                                     max(0.0, time - other_plan.end_time))

    def _ring_prediction(self, ring, theta: float, current_speed: float, target_speed: float, duration: float) -> np.ndarray:
        """Use the authoritative finite-acceleration holder integrator."""
        remaining = float(duration); angle = float(theta); speed = max(0.0, float(current_speed))
        while remaining > 1e-9:
            step = min(self.config.physics_dt, remaining)
            motion = self.dynamics.ring_motion_step(ring, angle, speed, target_speed, step)
            angle, speed, remaining = motion.theta, motion.speed, remaining - step
        return np.asarray((ring.radius * np.cos(angle), ring.radius * np.sin(angle), ring.height), dtype=float)

    def _ring_prediction_positions(self, ring, theta: float, current_speed: float, target_speed: float,
                                   origin: float, times: np.ndarray) -> list[np.ndarray]:
        """Predict an ordered holder track once instead of reintegrating per query.

        ``filter_plan`` and its swept checks request monotonically increasing
        safety samples.  Replaying every query from the snapshot time made a
        300-second, 0.2-second horizon quadratic in physics integrations.
        This routine preserves the exact same ``ring_motion_step`` dynamics
        while advancing each interval only once.
        """
        if len(times) == 0:
            return []
        if np.any(np.diff(times) < -1e-9):
            raise ValueError("holder prediction times must be ordered")
        angle, speed, previous = float(theta), max(0.0, float(current_speed)), float(origin)
        positions: list[np.ndarray] = []
        for value in times:
            moment = float(value)
            remaining = max(0.0, moment - previous)
            while remaining > 1e-9:
                step = min(self.config.physics_dt, remaining)
                next_speed = self.dynamics.limited_tangential_speed(speed, target_speed, ring.radius, step)
                angle += .5 * (speed + next_speed) * step / ring.radius
                speed, remaining = next_speed, remaining - step
            positions.append(np.asarray((ring.radius * np.cos(angle), ring.radius * np.sin(angle), ring.height), dtype=float))
            previous = max(previous, moment)
        return positions

    def _post_plan_positions(self, plan, snapshot, times: np.ndarray) -> list[np.ndarray]:
        endpoint = plan.points[-1]
        target = {"ring_outer": "outer", "ring_middle": "middle", "ring_inner": "inner"}.get(plan.metadata.get("target_phase"))
        if target is None:
            return [endpoint.position.copy() for _ in times]
        from .config import RingRole
        ring = snapshot.airspace.actual_ring(RingRole(target))
        theta = float(np.arctan2(endpoint.position[1], endpoint.position[0]))
        target_speed = float(np.clip(ring.nominal_speed + snapshot.airspace.speed_profile_bias, self.config.speed_min, self.config.max_speed))
        return self._ring_prediction_positions(ring, theta, float(np.linalg.norm(endpoint.velocity[:2])),
                                               target_speed, plan.end_time, times)

    def _committed_positions(self, plan, snapshot, times: np.ndarray) -> np.ndarray:
        """Canonical committed-plan plus post-handoff samples for one horizon."""
        values = np.asarray(times, dtype=float)
        key = (plan.plan_id, plan.trajectory_hash, plan.metadata.get("target_phase"), tuple(np.round(values, 8)))
        cached = self._committed_track_cache.get(key)
        if cached is not None:
            return cached
        values = np.asarray(times, dtype=float)
        result = np.empty((len(values), 3), dtype=float)
        pre_mask = values <= plan.end_time + 1e-9
        if np.any(pre_mask):
            result[pre_mask] = plan.sample_positions(values[pre_mask])
        post_indices = np.flatnonzero(~pre_mask)
        if len(post_indices):
            post_times = values[post_indices]
            result[post_indices] = self._post_plan_positions(plan, snapshot, post_times)
        result.setflags(write=False)
        self._committed_track_cache[key] = result
        return result

    def _holder_positions(self, uav, snapshot, times: np.ndarray, speed_residual_mps: float = 0.0) -> list[np.ndarray]:
        if uav.ring_role is None:
            return [np.asarray(uav.position, dtype=float) for _ in times]
        ring = snapshot.airspace.actual_ring(uav.ring_role)
        theta = uav.ring_progress
        if theta is None:
            theta = float(np.arctan2(uav.position[1], uav.position[0]))
        target = float(np.clip(ring.nominal_speed + snapshot.airspace.speed_profile_bias + speed_residual_mps,
                               self.config.speed_min, self.config.max_speed))
        # Candidate plans in one policy step share the same state and 0.2-s
        # safety grid.  Extend one authoritative finite-response track rather
        # than re-integrating it for every candidate verification.
        key=(uav.uid, round(float(snapshot.now),8), round(float(theta),8), round(float(uav.speed),8),
             round(float(ring.radius),8), round(float(ring.height),8), round(target,8))
        track=self._holder_track_cache.get(key)
        if track is None:
            track=[
                [np.asarray(uav.position,dtype=float).copy()],
                [float(theta)],
                [max(0.0,float(uav.speed))],
            ]
            self._holder_track_cache[key]=track
        positions, angles, speeds = track
        needed=int(np.floor((float(times[-1])-snapshot.now)/self.config.physics_dt + 1e-9))
        while len(positions)-1 < needed:
            angle, speed = angles[-1], speeds[-1]
            motion = self.dynamics.ring_motion_step(ring, angle, speed, target, self.config.physics_dt)
            positions.append(np.asarray(motion.position, dtype=float))
            angles.append(motion.theta)
            speeds.append(motion.speed)
        offsets=(times-snapshot.now)/self.config.physics_dt
        indices=np.floor(offsets+1e-9).astype(int)
        result=[]
        for moment,index,offset in zip(times,indices,offsets):
            fraction=float(offset-index)*self.config.physics_dt
            if fraction <= 1e-8:
                result.append(np.asarray(positions[index],dtype=float))
            else:
                motion = self.dynamics.ring_motion_step(ring, angles[index], speeds[index], target, fraction)
                result.append(np.asarray(motion.position, dtype=float))
        return result

    def holder_reachable_tube(self, uav, snapshot, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Angular reachable tube for every legal future holder residual sequence.

        The control can switch once per policy tick, but propagating the two
        extremal targets at every physics tick is a conservative superset of
        those piecewise-constant legal sequences.  It therefore replaces the
        unsafe three-trajectory approximation (nominal/+max/-max).
        """
        if uav.ring_role is None:
            position = np.asarray(uav.position, dtype=float)
            return np.repeat(position[None, :], len(times), axis=0), np.repeat(position[None, :], len(times), axis=0)
        if len(times) == 0 or np.any(np.diff(times) < -1e-9):
            raise ValueError("reachable-tube times must be non-empty and ordered")
        ring = snapshot.airspace.actual_ring(uav.ring_role)
        theta = uav.ring_progress if uav.ring_progress is not None else float(np.arctan2(uav.position[1], uav.position[0]))
        key=(uav.uid,round(float(snapshot.now),8),round(float(theta),8),round(float(uav.speed),8),
             round(float(ring.radius),8),round(float(ring.height),8))
        target_low = float(np.clip(ring.nominal_speed + snapshot.airspace.speed_profile_bias - self.config.max_speed_residual_mps,
                                   self.config.speed_min, self.config.max_speed))
        target_high = float(np.clip(ring.nominal_speed + snapshot.airspace.speed_profile_bias + self.config.max_speed_residual_mps,
                                    self.config.speed_min, self.config.max_speed))
        # Cache one authoritative physics-dt envelope per immutable state
        # tick, then extend it only when a later candidate needs a longer
        # horizon.  Candidate plans commonly differ in merge timing but share
        # exactly this holder state; a whole-grid cache key would otherwise
        # replay the same tube dozens of times during teacher search.
        cached=self._holder_tube_cache.get(key)
        if cached is None:
            cached={"angles_low": [float(theta)], "speeds_low": [max(0.0, float(uav.speed))],
                    "angles_high": [float(theta)], "speeds_high": [max(0.0, float(uav.speed))]}
            self._holder_tube_cache[key]=cached
        max_offset=max(0.0, float(np.max(times)) - float(snapshot.now))
        needed=int(np.floor(max_offset/self.config.physics_dt + 1e-9))
        while len(cached["angles_low"])-1 < needed:
            low = self.dynamics.ring_motion_step(ring, cached["angles_low"][-1], cached["speeds_low"][-1],
                                                  target_low, self.config.physics_dt)
            high = self.dynamics.ring_motion_step(ring, cached["angles_high"][-1], cached["speeds_high"][-1],
                                                   target_high, self.config.physics_dt)
            cached["angles_low"].append(low.theta); cached["speeds_low"].append(low.speed)
            cached["angles_high"].append(high.theta); cached["speeds_high"].append(high.speed)
        # Gather all grid-aligned samples in one NumPy pass.  Fractional
        # offsets still use the authoritative finite-step integrator so the
        # result remains numerically identical to the scalar lookup path.
        values = np.asarray(times, dtype=float)
        offsets = np.maximum(0.0, values - float(snapshot.now))
        indices = np.floor(offsets / self.config.physics_dt + 1e-9).astype(int)
        fractions = offsets - indices * self.config.physics_dt
        grid_aligned = fractions <= 1e-8

        lower = np.empty(len(values), dtype=float)
        upper = np.empty(len(values), dtype=float)
        angles_low = np.asarray(cached["angles_low"], dtype=float)
        angles_high = np.asarray(cached["angles_high"], dtype=float)
        lower[grid_aligned] = angles_low[indices[grid_aligned]]
        upper[grid_aligned] = angles_high[indices[grid_aligned]]

        for position in np.flatnonzero(~grid_aligned):
            index = int(indices[position])
            fraction = float(fractions[position])
            low = self.dynamics.ring_motion_step(
                ring, angles_low[index], cached["speeds_low"][index], target_low, fraction
            )
            high = self.dynamics.ring_motion_step(
                ring, angles_high[index], cached["speeds_high"][index], target_high, fraction
            )
            lower[position] = low.theta
            upper[position] = high.theta
        return lower, upper

    @staticmethod
    def _tube_point_distances(points: np.ndarray, ring, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Exact minimum distance from each point to its circular reachable arc."""
        # Keep the historical zip semantics for malformed caller inputs while
        # evaluating all points and all five legal turns in one NumPy pass.
        values = np.asarray(points, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        count = min(len(values), len(lower), len(upper))
        if count == 0:
            return np.empty(0, dtype=float)

        values = values[:count]
        lower = lower[:count]
        upper = upper[:count]
        radius = np.linalg.norm(values[:, :2], axis=1)
        angle = np.arctan2(values[:, 1], values[:, 0])
        radial_vertical = (float(ring.height) - values[:, 2]) ** 2

        # The two interval endpoints are always legal candidates.  The
        # closest circular point is the point's polar angle when one of its
        # five explicitly supported unwrapped copies lies in [lower, upper].
        endpoint_angles = np.stack((lower, upper), axis=1)
        endpoint_delta = endpoint_angles - angle[:, None]
        endpoint_squared = (
            float(ring.radius) ** 2
            + radius[:, None] ** 2
            - 2.0 * float(ring.radius) * radius[:, None] * np.cos(endpoint_delta)
        )

        turns = np.arange(-2, 3, dtype=float)
        shifted = angle[:, None] + turns[None, :] * (2.0 * math.pi)
        inside = (lower[:, None] <= shifted) & (shifted <= upper[:, None])
        interior_delta = shifted - angle[:, None]
        interior_squared = (
            float(ring.radius) ** 2
            + radius[:, None] ** 2
            - 2.0 * float(ring.radius) * radius[:, None] * np.cos(interior_delta)
        )
        interior_squared = np.where(inside, interior_squared, np.inf)
        minimum_squared = np.minimum(np.min(endpoint_squared, axis=1), np.min(interior_squared, axis=1))
        return np.sqrt(np.maximum(0.0, minimum_squared) + radial_vertical)

    @staticmethod
    def _track_distances(left, right) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized equivalent of point and pairwise swept checks."""
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        point = np.linalg.norm(left - right, axis=1)
        if len(left) < 2:
            return point, np.empty(0, dtype=float)
        relative_start = left[:-1] - right[:-1]
        relative_delta = (left[1:] - left[:-1]) - (right[1:] - right[:-1])
        denominator = np.einsum("ij,ij->i", relative_delta, relative_delta)
        projection = np.zeros_like(denominator)
        moving = denominator > 1e-12
        projection[moving] = -np.einsum(
            "ij,ij->i", relative_start[moving], relative_delta[moving]
        ) / denominator[moving]
        projection = np.clip(projection, 0.0, 1.0)
        closest = relative_start + projection[:, None] * relative_delta
        return point, np.linalg.norm(closest, axis=1)

    def _holder_broad_phase_safe(self, uav, snapshot, candidate_positions: np.ndarray) -> bool:
        """Conservatively reject an obviously-safe full-circle holder pair."""
        if self._v3_enabled:
            return False
        if uav.ring_role is None or len(candidate_positions) == 0:
            return False
        ring = snapshot.airspace.actual_ring(uav.ring_role)
        xy = candidate_positions[:, :2]
        radii = np.linalg.norm(xy, axis=1)
        radial_min = float(np.min(radii))
        if len(xy) > 1:
            start = xy[:-1]
            delta = xy[1:] - start
            denominator = np.einsum("ij,ij->i", delta, delta)
            projection = np.zeros_like(denominator)
            moving = denominator > 1e-12
            projection[moving] = -np.einsum("ij,ij->i", start[moving], delta[moving]) / denominator[moving]
            closest = start + np.clip(projection, 0.0, 1.0)[:, None] * delta
            radial_min = min(radial_min, float(np.min(np.linalg.norm(closest, axis=1))))
        radial_max = float(np.max(radii))
        radial_gap = max(radial_min - ring.radius, ring.radius - radial_max, 0.0)
        z_min = float(np.min(candidate_positions[:, 2]))
        z_max = float(np.max(candidate_positions[:, 2]))
        vertical_gap = max(z_min - ring.height, ring.height - z_max, 0.0)
        return math.hypot(radial_gap, vertical_gap) >= self.config.safe_distance

    def filter_plan(self,plan,committed_plans,snapshot):
        if plan.plan_id!=getattr(plan,"plan_id",None):return TrajectorySafetyResult(False,"invalid_plan")
        if plan.metadata.get("trajectory_schema_version") != self.config.trajectory_schema_version:
            return TrajectorySafetyResult(False,"trajectory_schema_mismatch",plan.plan_id)
        if plan.metadata.get("vehicle_profile_id") != self.config.vehicle_profile_id:
            return TrajectorySafetyResult(False,"vehicle_profile_mismatch",plan.plan_id)
        if self._v3_enabled:
            try:
                if not plan.verify_trajectory_integrity():
                    return TrajectorySafetyResult(False,"trajectory_integrity_mismatch",plan.plan_id)
            except (AttributeError, TypeError, ValueError):
                return TrajectorySafetyResult(False,"trajectory_integrity_mismatch",plan.plan_id)
        dynamics_errors = self.dynamics.validate_trajectory(plan)
        if dynamics_errors:
            return TrajectorySafetyResult(False,f"dynamics_{dynamics_errors[0]}",plan.plan_id,plan.trajectory_hash)
        if self._v3_enabled:
            try:
                max_altitude = max(float(segment.bounds()["max_altitude"]) for segment in plan.polynomial_segments)
            except (AttributeError, KeyError, TypeError, ValueError, FloatingPointError):
                return TrajectorySafetyResult(False, "dynamics_altitude_limit", plan.plan_id, plan.trajectory_hash)
            if max_altitude > float(self.config.height_max) + 1.0e-3:
                return TrajectorySafetyResult(False, "dynamics_altitude_limit", plan.plan_id, plan.trajectory_hash)
        if plan.metadata.get("dynamic_unreachable_merge_time"):
            return TrajectorySafetyResult(False,"dynamics_unreachable_merge_time",plan.plan_id,plan.trajectory_hash)
        # Both committed-trajectory and holder checks use this exact candidate
        # horizon.  It is immutable, so sampling it once avoids repeating the
        # same canonical interpolation for every protected aircraft.
        samples=np.arange(plan.start_time,plan.end_time+1e-9,self.config.safety_sampling_dt)
        if samples[-1] < plan.end_time: samples=np.append(samples,plan.end_time)
        candidate_positions = np.asarray(plan.sample_positions(samples), dtype=float)
        tube_times = None
        tube_positions = None
        committed_tracks = {}
        for other_id,other in committed_plans.items():
            if other_id==plan.uav_id:continue
            # The other plan may finish before this candidate.  Use the same
            # post-plan ring-holder prediction for the remaining candidate
            # horizon rather than silently dropping it at ``other.end_time``.
            committed_positions = np.asarray(self._committed_positions(other, snapshot, samples), dtype=float)
            committed_tracks[other_id] = committed_positions
            other_descriptor = self._track_descriptor(other)
            if self._v3_enabled and other_descriptor is None:
                return TrajectorySafetyResult(False,"trajectory_uncertified",plan.plan_id)
            point_distances, swept_distances = self._track_distances_with_margin(
                candidate_positions, committed_positions, samples,
                self._track_descriptor(plan), other_descriptor,
            )
            if np.any(point_distances < self.config.safe_distance):
                return TrajectorySafetyResult(False,"predicted_separation",plan.plan_id)
            if np.any(swept_distances < self.config.safe_distance):
                return TrajectorySafetyResult(False,"predicted_swept_separation",plan.plan_id)
            # A candidate that hands off to a ring must remain safe for the
            # full continuation of both committed plans plus the finite
            # response horizon against every committed track.  Capping this
            # at ``candidate.end + horizon`` misses the symmetric ordering in
            # which the *other* immutable transfer ends later and only then
            # hands off into the candidate's ring trajectory.  That exact
            # gap later appeared as an uncontrollable committed--committed
            # continuation witness at runtime.
            if plan.metadata.get("target_phase") in {"ring_outer", "ring_middle", "ring_inner"}:
                post_start=plan.end_time
                post_end=max(plan.end_time, other.end_time) + self.config.shield_horizon_seconds
                post_samples=np.arange(post_start,post_end+1e-9,self.config.safety_sampling_dt)
                if post_samples[-1] < post_end: post_samples=np.append(post_samples,post_end)
                candidate_post_positions = np.asarray(self._post_plan_positions(plan, snapshot, post_samples), dtype=float)
                other_positions = np.asarray(self._committed_positions(other, snapshot, post_samples), dtype=float)
                point_distances, swept_distances = self._track_distances_with_margin(
                    candidate_post_positions, other_positions, post_samples,
                    "holder_proxy", other_descriptor,
                )
                if np.any(point_distances < self.config.safe_distance):
                    return TrajectorySafetyResult(False,"predicted_post_plan_holder_separation",plan.plan_id)
                if np.any(swept_distances < self.config.safe_distance):
                    return TrajectorySafetyResult(False,"predicted_post_plan_holder_swept_separation",plan.plan_id)
        # A holder has no committed transfer plan, but is nevertheless an
        # active aircraft.  Omitting it lets a new transfer cut through the
        # occupied target ring; this was the source of the historical A5
        # fixed-rule collision reports.  Use the same actual-ring guidance
        # reference used by execution for a conservative prediction.
        for other_id, other_uav in snapshot.active.items():
            if other_id == plan.uav_id:
                continue
            other_plan = other_uav.committed_plan
            if other_id in committed_tracks:
                # Standard point/swept checks already ran above.  Retain only
                # the deliberately larger executor hand-off margin.
                if plan.metadata.get("target_phase") in {"ring_outer", "ring_middle", "ring_inner"}:
                    endpoint = samples >= plan.end_time - 1e-9
                    if np.any(endpoint):
                        distances = np.linalg.norm(
                            candidate_positions[endpoint] - committed_tracks[other_id][endpoint], axis=1
                        )
                        if np.any(distances < self.config.safe_distance + self.config.max_speed * self.config.dt):
                            return TrajectorySafetyResult(False, "predicted_holder_separation", plan.plan_id)
                continue
            target_role_name = {"ring_outer": "outer", "ring_middle": "middle", "ring_inner": "inner"}.get(
                plan.metadata.get("target_phase")
            )
            same_target_ring = target_role_name is not None and getattr(other_uav.ring_role, "value", None) == target_role_name
            if other_plan is None and not same_target_ring and self._holder_broad_phase_safe(other_uav, snapshot, candidate_positions):
                continue
            # Check the full legal speed-control reachable tube at physics
            # resolution.  A commit must be safe for any bounded future
            # holder residual sequence, not only three constant trajectories.
            if other_plan is None:
                if tube_times is None:
                    tube_times = np.arange(plan.start_time, plan.end_time + 1e-9, self.config.physics_dt)
                    if tube_times[-1] < plan.end_time:
                        tube_times = np.append(tube_times, plan.end_time)
                    tube_positions = np.asarray(plan.sample_positions(tube_times), dtype=float)
                lower, upper = self.holder_reachable_tube(other_uav, snapshot, tube_times)
                ring = snapshot.airspace.actual_ring(other_uav.ring_role)
                if np.any(self._tube_distances_with_margin(
                        tube_positions, ring, lower, upper, tube_times,
                        self._track_descriptor(plan),
                    ) < self.config.safe_distance):
                    return TrajectorySafetyResult(False, "predicted_holder_reachable_tube", plan.plan_id)
                if abs(float(self.config.max_speed_residual_mps)) <= 1e-12:
                    # The nominal fixed-rule contract has no residual-speed
                    # envelope to certify.  Keep the reachable-tube proof,
                    # then run the exact nominal holder track through the
                    # same candidate samples so point and linearly swept
                    # separation use the runtime certificate semantics.
                    nominal_positions = np.asarray(
                        self._holder_positions(other_uav, snapshot, samples, 0.0),
                        dtype=float,
                    )
                    point_distances, swept_distances = self._track_distances_with_margin(
                        candidate_positions, nominal_positions, samples,
                        self._track_descriptor(plan), "holder_proxy",
                    )
                    if np.any(point_distances < self.config.safe_distance):
                        return TrajectorySafetyResult(
                            False, "nominal_holder_separation", plan.plan_id,
                        )
                    if np.any(swept_distances < self.config.safe_distance):
                        return TrajectorySafetyResult(
                            False, "nominal_holder_swept_separation", plan.plan_id,
                        )
                if same_target_ring:
                    post_end = plan.end_time + self.config.shield_horizon_seconds
                    post_times = np.arange(plan.end_time, post_end + 1e-9, self.config.physics_dt)
                    if (abs(float(self.config.max_speed_residual_mps)) <= 1e-12
                            and post_times[-1] < post_end):
                        post_times = np.append(post_times, post_end)
                    post_positions = np.asarray(self._post_plan_positions(plan, snapshot, post_times), dtype=float)
                    lower, upper = self.holder_reachable_tube(other_uav, snapshot, post_times)
                    if np.any(self._tube_distances_with_margin(
                            post_positions, ring, lower, upper, post_times,
                            "holder_proxy",
                        ) < self.config.safe_distance):
                        return TrajectorySafetyResult(False, "predicted_post_plan_holder_reachable_tube", plan.plan_id)
                    if abs(float(self.config.max_speed_residual_mps)) <= 1e-12:
                        nominal_post_times = np.arange(
                            plan.end_time,
                            post_end + 1e-9,
                            self.config.safety_sampling_dt,
                        )
                        if nominal_post_times[-1] < post_end:
                            nominal_post_times = np.append(nominal_post_times, post_end)
                        nominal_post_candidate = np.asarray(
                            self._post_plan_positions(plan, snapshot, nominal_post_times),
                            dtype=float,
                        )
                        nominal_post_positions = np.asarray(
                            self._holder_positions(other_uav, snapshot, nominal_post_times, 0.0),
                            dtype=float,
                        )
                        point_distances, swept_distances = self._track_distances_with_margin(
                            nominal_post_candidate, nominal_post_positions, nominal_post_times,
                            "holder_proxy", "holder_proxy",
                        )
                        if np.any(point_distances < self.config.safe_distance):
                            return TrajectorySafetyResult(
                                False, "nominal_post_plan_holder_separation", plan.plan_id,
                            )
                        if np.any(swept_distances < self.config.safe_distance):
                            return TrajectorySafetyResult(
                                False, "nominal_post_plan_holder_swept_separation", plan.plan_id,
                            )
                continue
            # A plan ending inside the candidate's horizon becomes a holder at
            # its endpoint.  The dense canonical plan samples provide a cheap
            # conservative hand-off check without multiplying every rejected
            # rule action by a one-second cross-product.
            required = np.full(len(samples), self.config.safe_distance, dtype=float)
            if plan.metadata.get("target_phase") in {"ring_outer", "ring_middle", "ring_inner"}:
                required[samples >= plan.end_time - 1e-9] += self.config.max_speed * self.config.dt
            # A committed candidate must remain safe even if any current
            # holder legally applies its full bounded speed residual on the
            # next policy tick.  Previously this check used only nominal
            # holder speed, then the runtime Shield discovered the mismatch
            # after the immutable transfer had already begun.
            for residual in (0.0,):
                other_positions = self._committed_positions(other_plan, snapshot, samples)
                other_positions = np.asarray(other_positions, dtype=float)
                other_descriptor = self._track_descriptor(other_plan)
                if self._v3_enabled and other_descriptor is None:
                    return TrajectorySafetyResult(False, "trajectory_uncertified", plan.plan_id)
                point_distances, swept_distances = self._track_distances_with_margin(
                    candidate_positions, other_positions, samples,
                    self._track_descriptor(plan), other_descriptor,
                )
                if np.any(point_distances < required):
                    return TrajectorySafetyResult(False, "predicted_holder_speed_envelope", plan.plan_id)
                if np.any(swept_distances < self.config.safe_distance):
                    return TrajectorySafetyResult(False, "predicted_holder_speed_envelope_swept", plan.plan_id)
            if other_plan is None and same_target_ring:
                # An inbound/transfer endpoint can be outside the hard sphere
                # yet still close rapidly while its low hand-off speed
                # converges to ring nominal speed.  Verify that finite-speed
                # transient over the same shield horizon before admitting it.
                post_end = plan.end_time + self.config.shield_horizon_seconds
                post_samples = np.arange(plan.end_time, post_end + 1e-9, self.config.safety_sampling_dt)
                candidate_post = np.asarray(self._post_plan_positions(plan, snapshot, post_samples), dtype=float)
                for residual in (0.0, self.config.max_speed_residual_mps, -self.config.max_speed_residual_mps):
                    other_post = np.asarray(self._holder_positions(other_uav, snapshot, post_samples, residual), dtype=float)
                    point_distances, swept_distances = self._track_distances_with_margin(
                        candidate_post, other_post, post_samples,
                        "holder_proxy", "holder_proxy",
                    )
                    if np.any(point_distances < self.config.safe_distance):
                        return TrajectorySafetyResult(False, "predicted_post_plan_holder_speed_envelope", plan.plan_id)
                    if np.any(swept_distances < self.config.safe_distance):
                        return TrajectorySafetyResult(False, "predicted_post_plan_holder_speed_envelope_swept", plan.plan_id)
        trajectory_hash = plan.trajectory_hash
        self.checked_plan_ids.append(plan.plan_id);self.checked_trajectory_hashes[plan.plan_id] = trajectory_hash
        return TrajectorySafetyResult(True,checked_plan_id=plan.plan_id,checked_trajectory_hash=trajectory_hash)

    def verify_predicted_tracks(self, tracks, *, start: float, end: float):
        """Final arbiter for geometry repairs using canonical predicted plans."""
        ordered = list(tracks.items())
        minimum=float("inf"); checked_pairs=0
        def finish(approved, reason, plan_id=None):
            self.last_predicted_track_audit={"minimum_separation": None if not np.isfinite(minimum) else float(minimum),
                                             "checked_pairs": checked_pairs, "reason": reason}
            return TrajectorySafetyResult(approved, reason, plan_id)
        descriptors = {}
        # Committed trajectories have already passed ``filter_plan``.  New
        # holder repair tracks are sampled from bounded ring-speed guidance;
        # validate only those kinematics here rather than reinterpreting a
        # previously checked transfer through a one-second approximation.
        for uid, plan in ordered:
            descriptor = self._track_descriptor(plan)
            descriptors[uid] = descriptor
            if self._v3_enabled and descriptor is None:
                return finish(False, "trajectory_uncertified", plan.plan_id)
            if plan.metadata.get("committed", False):
                if self._v3_enabled and plan.metadata.get("trajectory_schema_version") == "executable_trajectory_v3":
                    try:
                        if not plan.verify_trajectory_integrity():
                            return finish(False, "trajectory_integrity_mismatch", plan.plan_id)
                    except (AttributeError, TypeError, ValueError):
                        return finish(False, "trajectory_integrity_mismatch", plan.plan_id)
                    checked_hash = self.checked_trajectory_hashes.get(plan.plan_id)
                    if checked_hash != plan.trajectory_hash:
                        if self.dynamics.validate_trajectory(plan):
                            return finish(False, "repair_dynamics", plan.plan_id)
                continue
            if plan.metadata.get("runtime_executor_equivalent", False):
                # Runtime holder tracks are not arbitrary plans.  They were
                # generated by the same bounded ring-motion integrator as
                # physics execution, so re-validating their sampled chord
                # with free-trajectory semantics can reject a trajectory the
                # executor will actually fly.  Separation is still verified
                # below with the canonical sampled track.
                continue
            if self.dynamics.validate_trajectory(plan):
                return finish(False, "repair_dynamics", plan.plan_id)
        if len(ordered) < 2:
            return finish(True, "approved", "reconfiguration_tracks")
        samples = np.arange(start, end + 1e-9, self.config.safety_sampling_dt)
        if samples[-1] < end:
            samples = np.append(samples, end)
        # Candidate reconfiguration checks can contain many holders over a
        # long protected horizon.  Sample each canonical plan once, then
        # retain the original (ordered) pairwise semantics in vector form.
        # This is deliberately not an approximation: both the point and
        # swept tests below use precisely the samples that the scalar loop
        # used previously.
        positions = np.asarray([
            [plan.sample_position(float(time)) for time in samples]
            for _, plan in ordered
        ], dtype=float)
        active_ends = np.asarray([
            plan.metadata.get("active_end", plan.end_time) for _, plan in ordered
        ], dtype=float)
        upper = np.triu_indices(len(ordered), k=1)

        for sample_index, time in enumerate(samples):
            active = time <= active_ends
            eligible = active[upper[0]] & active[upper[1]]
            if not np.any(eligible):
                continue
            deltas = positions[:, sample_index, None, :] - positions[None, :, sample_index, :]
            distances = np.linalg.norm(deltas[upper], axis=1)
            if self._v3_enabled:
                adjusted = distances.copy()
                for pair_index, (left_index, right_index) in enumerate(zip(upper[0], upper[1])):
                    left_error = self._track_point_error(descriptors[ordered[int(left_index)][0]])
                    right_error = self._track_point_error(descriptors[ordered[int(right_index)][0]])
                    adjusted[pair_index] = (
                        -np.inf if left_error is None or right_error is None
                        else distances[pair_index] - left_error - right_error
                    )
                distances = adjusted
            checked_pairs += int(np.count_nonzero(eligible))
            current_minimum = float(np.min(distances[eligible]))
            minimum = min(minimum, current_minimum)
            bad = np.flatnonzero(eligible & (distances < self.config.safe_distance))
            if len(bad):
                index = int(bad[0]); left_index, right_index = int(upper[0][index]), int(upper[1][index])
                uid, plan = ordered[left_index]; other_uid, _ = ordered[right_index]
                return finish(False, f"repair_separation:{uid}:{other_uid}", plan.plan_id)

        for sample_index, left in enumerate(samples[:-1]):
            active = left < active_ends
            eligible = active[upper[0]] & active[upper[1]]
            if not np.any(eligible):
                continue
            start_positions = positions[:, sample_index]
            velocities = positions[:, sample_index + 1] - start_positions
            relative_start = start_positions[:, None, :] - start_positions[None, :, :]
            relative_velocity = velocities[:, None, :] - velocities[None, :, :]
            velocity_squared = np.einsum("ijk,ijk->ij", relative_velocity, relative_velocity)
            projection = np.divide(
                -np.einsum("ijk,ijk->ij", relative_start, relative_velocity), velocity_squared,
                out=np.zeros_like(velocity_squared), where=velocity_squared > 1e-12,
            )
            projection = np.clip(projection, 0., 1.)
            swept = relative_start + projection[..., None] * relative_velocity
            distances = np.linalg.norm(swept[upper], axis=1)
            checked_pairs += int(np.count_nonzero(eligible))
            if not self._v3_enabled:
                current_minimum = float(np.min(distances[eligible]))
                minimum = min(minimum, current_minimum)
                bad = np.flatnonzero(eligible & (distances < self.config.safe_distance))
                if len(bad):
                    index = int(bad[0]); left_index, right_index = int(upper[0][index]), int(upper[1][index])
                    uid, plan = ordered[left_index]; other_uid, _ = ordered[right_index]
                    return finish(False, f"repair_swept_separation:{uid}:{other_uid}", plan.plan_id)
                continue
            adjusted = distances.copy()
            for pair_index, (left_index, right_index) in enumerate(zip(upper[0], upper[1])):
                correction = self._pair_curve_error(
                    descriptors[ordered[int(left_index)][0]],
                    descriptors[ordered[int(right_index)][0]],
                    float(samples[sample_index + 1] - samples[sample_index]),
                )
                adjusted[pair_index] = -np.inf if correction is None else max(0.0, adjusted[pair_index] - correction)
            current_minimum = float(np.min(adjusted[eligible]))
            minimum = min(minimum, current_minimum)
            bad = np.flatnonzero(eligible & (adjusted < self.config.safe_distance))
            if len(bad):
                index = int(bad[0]); left_index, right_index = int(upper[0][index]), int(upper[1][index])
                uid, plan = ordered[left_index]; other_uid, _ = ordered[right_index]
                return finish(False, f"repair_swept_separation:{uid}:{other_uid}", plan.plan_id)
        return finish(True, "approved", "reconfiguration_tracks")
