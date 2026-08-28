"""Resource/intent then canonical-plan safety verification."""
from __future__ import annotations
from dataclasses import dataclass
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
        # Candidate verifications are auditable.  ``None`` is deliberately
        # distinct from a safe infinite separation so callers cannot turn a
        # missing prediction into a passing safety metric.
        self.last_predicted_track_audit={"minimum_separation": None, "checked_pairs": 0, "reason": "not_run"}
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
                motion = self.dynamics.ring_motion_step(ring, angle, speed, target_speed, step)
                angle, speed, remaining = motion.theta, motion.speed, remaining - step
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
        return self._ring_prediction_positions(ring, theta, uav.speed, target, snapshot.now, times)

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
        candidate_positions = [plan.sample(float(time)).position for time in samples]
        for other_id,other in committed_plans.items():
            if other_id==plan.uav_id:continue
            # The other plan may finish before this candidate.  Use the same
            # post-plan ring-holder prediction for the remaining candidate
            # horizon rather than silently dropping it at ``other.end_time``.
            committed_positions = self._committed_positions(other, snapshot, samples)
            for candidate_position, committed_position in zip(candidate_positions, committed_positions):
                if np.linalg.norm(candidate_position-committed_position) < self.config.safe_distance:
                    return TrajectorySafetyResult(False,"predicted_separation",plan.plan_id)
            for index in range(len(samples) - 1):
                distance,_=swept_minimum_distance(candidate_positions[index], candidate_positions[index + 1],
                                                   committed_positions[index], committed_positions[index + 1])
                if distance<self.config.safe_distance:return TrajectorySafetyResult(False,"predicted_swept_separation",plan.plan_id)
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
                candidate_post_positions = self._post_plan_positions(plan, snapshot, post_samples)
                other_positions = [other.sample(float(time)).position for time in post_samples]
                for candidate_position, other_position in zip(candidate_post_positions, other_positions):
                    if np.linalg.norm(candidate_position-other_position) < self.config.safe_distance:
                        return TrajectorySafetyResult(False,"predicted_post_plan_holder_separation",plan.plan_id)
                for index in range(len(post_samples) - 1):
                    distance,_=swept_minimum_distance(
                        candidate_post_positions[index], candidate_post_positions[index + 1],
                        other_positions[index], other_positions[index + 1],
                    )
                    if distance < self.config.safe_distance:
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
            # A plan ending inside the candidate's horizon becomes a holder at
            # its endpoint.  The dense canonical plan samples provide a cheap
            # conservative hand-off check without multiplying every rejected
            # rule action by a one-second cross-product.
            other_positions = (self._committed_positions(other_plan, snapshot, samples)
                               if other_plan is not None else self._holder_positions(other_uav, snapshot, samples))
            for index, (time, candidate_position, other_position) in enumerate(zip(samples, candidate_positions, other_positions)):
                # A plan that ends on a ring crosses an executor hand-off on
                # the next tick.  Reserve one bounded holder-motion step at
                # that boundary so a checked endpoint cannot immediately
                # become an unverified same-ring encounter.
                required = self.config.safe_distance
                if time >= plan.end_time - 1e-9 and plan.metadata.get("target_phase") in {"ring_outer", "ring_middle", "ring_inner"}:
                    required += self.config.max_speed * self.config.dt
                if np.linalg.norm(candidate_position-other_position) < required:
                    return TrajectorySafetyResult(False, "predicted_holder_separation", plan.plan_id)
            for index in range(len(samples) - 1):
                distance,_=swept_minimum_distance(candidate_positions[index], candidate_positions[index + 1],
                                                   other_positions[index], other_positions[index + 1])
                if distance < self.config.safe_distance:
                    return TrajectorySafetyResult(False, "predicted_holder_swept_separation", plan.plan_id)
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
