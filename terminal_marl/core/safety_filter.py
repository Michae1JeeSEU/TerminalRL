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
        self.config=config;self.dynamics=VehicleDynamics(config.vehicle_profile);self.checked_plan_ids=[];self.checked_trajectory_hashes={}
        self._holder_track_cache={}
        # Candidate verifications are auditable.  ``None`` is deliberately
        # distinct from a safe infinite separation so callers cannot turn a
        # missing prediction into a passing safety metric.
        self.last_predicted_track_audit={"minimum_separation": None, "checked_pairs": 0, "reason": "not_run"}
    def begin_prediction_step(self):
        """Invalidate reusable holder tracks after an authoritative state tick."""
        self._holder_track_cache.clear()
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
        endpoint = other_plan.sample(other_plan.end_time).position
        target = {"ring_outer": "outer", "ring_middle": "middle", "ring_inner": "inner"}.get(other_plan.metadata.get("target_phase"))
        if target is None:
            return endpoint
        from .config import RingRole
        role = RingRole(target)
        ring = snapshot.airspace.actual_ring(role)
        theta = float(np.arctan2(endpoint[1], endpoint[0]))
        end = other_plan.sample(other_plan.end_time)
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
        endpoint = plan.sample(plan.end_time)
        target = {"ring_outer": "outer", "ring_middle": "middle", "ring_inner": "inner"}.get(plan.metadata.get("target_phase"))
        if target is None:
            return [endpoint.position.copy() for _ in times]
        from .config import RingRole
        ring = snapshot.airspace.actual_ring(RingRole(target))
        theta = float(np.arctan2(endpoint.position[1], endpoint.position[0]))
        target_speed = float(np.clip(ring.nominal_speed + snapshot.airspace.speed_profile_bias, self.config.speed_min, self.config.max_speed))
        return self._ring_prediction_positions(ring, theta, float(np.linalg.norm(endpoint.velocity[:2])),
                                               target_speed, plan.end_time, times)

    def _committed_positions(self, plan, snapshot, times: np.ndarray) -> list[np.ndarray]:
        """Canonical committed-plan plus post-handoff samples for one horizon."""
        positions: list[np.ndarray | None] = [None] * len(times)
        post_indices = []
        for index, value in enumerate(times):
            if value <= plan.end_time + 1e-9:
                positions[index] = plan.sample(float(value)).position
            else:
                post_indices.append(index)
        if post_indices:
            post_times = np.asarray([times[index] for index in post_indices], dtype=float)
            for index, position in zip(post_indices, self._post_plan_positions(plan, snapshot, post_times)):
                positions[index] = position
        return [np.asarray(position, dtype=float) for position in positions]

    def _holder_positions(self, uav, snapshot, times: np.ndarray) -> list[np.ndarray]:
        if uav.ring_role is None:
            return [np.asarray(uav.position, dtype=float) for _ in times]
        ring = snapshot.airspace.actual_ring(uav.ring_role)
        theta = uav.ring_progress
        if theta is None:
            theta = float(np.arctan2(uav.position[1], uav.position[0]))
        target = float(np.clip(ring.nominal_speed + snapshot.airspace.speed_profile_bias, self.config.speed_min, self.config.max_speed))
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
            next_speed=self.dynamics.limited_tangential_speed(speed,target,ring.radius,self.config.physics_dt)
            next_angle=angle+.5*(speed+next_speed)*self.config.physics_dt/ring.radius
            positions.append(np.asarray((ring.radius*math.cos(next_angle),ring.radius*math.sin(next_angle),ring.height),dtype=float))
            angles.append(next_angle)
            speeds.append(next_speed)
        offsets=(times-snapshot.now)/self.config.physics_dt
        indices=np.floor(offsets+1e-9).astype(int)
        result=[]
        for moment,index,offset in zip(times,indices,offsets):
            fraction=float(offset-index)*self.config.physics_dt
            if fraction <= 1e-8:
                result.append(np.asarray(positions[index],dtype=float))
            else:
                next_speed=self.dynamics.limited_tangential_speed(speeds[index],target,ring.radius,fraction)
                next_angle=angles[index]+.5*(speeds[index]+next_speed)*fraction/ring.radius
                result.append(np.asarray((ring.radius*math.cos(next_angle),ring.radius*math.sin(next_angle),ring.height),dtype=float))
        return result

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
        dynamics_errors = self.dynamics.validate_plan(plan.points)
        if dynamics_errors:
            return TrajectorySafetyResult(False,f"dynamics_{dynamics_errors[0]}",plan.plan_id,plan.trajectory_hash)
        if plan.metadata.get("dynamic_unreachable_merge_time"):
            return TrajectorySafetyResult(False,"dynamics_unreachable_merge_time",plan.plan_id,plan.trajectory_hash)
        # Both committed-trajectory and holder checks use this exact candidate
        # horizon.  It is immutable, so sampling it once avoids repeating the
        # same canonical interpolation for every protected aircraft.
        samples=np.arange(plan.start_time,plan.end_time+1e-9,self.config.safety_sampling_dt)
        if samples[-1] < plan.end_time: samples=np.append(samples,plan.end_time)
        candidate_positions = np.asarray([plan.sample(float(time)).position for time in samples], dtype=float)
        committed_tracks = {}
        for other_id,other in committed_plans.items():
            if other_id==plan.uav_id:continue
            # The other plan may finish before this candidate.  Use the same
            # post-plan ring-holder prediction for the remaining candidate
            # horizon rather than silently dropping it at ``other.end_time``.
            committed_positions = np.asarray(self._committed_positions(other, snapshot, samples), dtype=float)
            committed_tracks[other_id] = committed_positions
            point_distances, swept_distances = self._track_distances(candidate_positions, committed_positions)
            if np.any(point_distances < self.config.safe_distance):
                return TrajectorySafetyResult(False,"predicted_separation",plan.plan_id)
            if np.any(swept_distances < self.config.safe_distance):
                return TrajectorySafetyResult(False,"predicted_swept_separation",plan.plan_id)
            # A candidate that hands off to a ring must remain safe as the
            # canonical post-plan holder while an already committed transfer
            # continues.  Checking only the overlap of the two plan objects
            # misses this future encounter (e.g. INBOUND -> OUTER holder vs
            # a previously committed O2M transfer).
            if plan.metadata.get("target_phase") in {"ring_outer", "ring_middle", "ring_inner"} and other.end_time > plan.end_time:
                post_start=max(plan.end_time, other.start_time)
                post_end=other.end_time
                post_samples=np.arange(post_start,post_end+1e-9,self.config.safety_sampling_dt)
                if post_samples[-1] < post_end: post_samples=np.append(post_samples,post_end)
                candidate_post_positions = np.asarray(self._post_plan_positions(plan, snapshot, post_samples), dtype=float)
                other_positions = np.asarray([other.sample(float(time)).position for time in post_samples], dtype=float)
                point_distances, swept_distances = self._track_distances(candidate_post_positions, other_positions)
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
            # A plan ending inside the candidate's horizon becomes a holder at
            # its endpoint.  The dense canonical plan samples provide a cheap
            # conservative hand-off check without multiplying every rejected
            # rule action by a one-second cross-product.
            other_positions = (self._committed_positions(other_plan, snapshot, samples)
                               if other_plan is not None else self._holder_positions(other_uav, snapshot, samples))
            other_positions = np.asarray(other_positions, dtype=float)
            required = np.full(len(samples), self.config.safe_distance, dtype=float)
            if plan.metadata.get("target_phase") in {"ring_outer", "ring_middle", "ring_inner"}:
                required[samples >= plan.end_time - 1e-9] += self.config.max_speed * self.config.dt
            point_distances, swept_distances = self._track_distances(candidate_positions, other_positions)
            if np.any(point_distances < required):
                return TrajectorySafetyResult(False, "predicted_holder_separation", plan.plan_id)
            if np.any(swept_distances < self.config.safe_distance):
                return TrajectorySafetyResult(False, "predicted_holder_swept_separation", plan.plan_id)
            if other_plan is None and same_target_ring:
                # An inbound/transfer endpoint can be outside the hard sphere
                # yet still close rapidly while its low hand-off speed
                # converges to ring nominal speed.  Verify that finite-speed
                # transient over the same shield horizon before admitting it.
                post_end = plan.end_time + self.config.shield_horizon_seconds
                post_samples = np.arange(plan.end_time, post_end + 1e-9, self.config.safety_sampling_dt)
                candidate_post = np.asarray(self._post_plan_positions(plan, snapshot, post_samples), dtype=float)
                other_post = np.asarray(self._holder_positions(other_uav, snapshot, post_samples), dtype=float)
                point_distances, swept_distances = self._track_distances(candidate_post, other_post)
                if np.any(point_distances < self.config.safe_distance):
                    return TrajectorySafetyResult(False, "predicted_post_plan_holder_separation", plan.plan_id)
                if np.any(swept_distances < self.config.safe_distance):
                    return TrajectorySafetyResult(False, "predicted_post_plan_holder_swept_separation", plan.plan_id)
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
        # Committed trajectories have already passed ``filter_plan``.  New
        # holder repair tracks are sampled from bounded ring-speed guidance;
        # validate only those kinematics here rather than reinterpreting a
        # previously checked transfer through a one-second approximation.
        for _, plan in ordered:
            if plan.metadata.get("committed", False):
                continue
            if self.dynamics.validate_plan(plan.points):
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
            [plan.sample(float(time)).position for time in samples]
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
            current_minimum = float(np.min(distances[eligible]))
            minimum = min(minimum, current_minimum)
            bad = np.flatnonzero(eligible & (distances < self.config.safe_distance))
            if len(bad):
                index = int(bad[0]); left_index, right_index = int(upper[0][index]), int(upper[1][index])
                uid, plan = ordered[left_index]; other_uid, _ = ordered[right_index]
                return finish(False, f"repair_swept_separation:{uid}:{other_uid}", plan.plan_id)
        return finish(True, "approved", "reconfiguration_tracks")
