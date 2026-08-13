"""Resource/intent then canonical-plan safety verification."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .runtime_separation import swept_minimum_distance
@dataclass(frozen=True)
class IntentSafetyResult: approved:bool;reason:str=""
@dataclass(frozen=True)
class TrajectorySafetyResult: approved:bool;reason:str="";checked_plan_id:str|None=None
class SafetyFilter:
    def __init__(self,config):
        self.config=config;self.checked_plan_ids=[]
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
        speed = float(np.clip(ring.nominal_speed + snapshot.airspace.speed_profile_bias, self.config.speed_min, self.config.speed_max))
        angle = theta + speed * max(0.0, time - snapshot.now) / ring.radius
        return np.asarray((ring.radius * np.cos(angle), ring.radius * np.sin(angle), ring.height))

    def _post_plan_prediction(self, other_plan, snapshot, time: float) -> np.ndarray:
        endpoint = other_plan.sample(other_plan.end_time).position
        target = {"ring_outer": "outer", "ring_middle": "middle", "ring_inner": "inner"}.get(other_plan.metadata.get("target_phase"))
        if target is None:
            return endpoint
        from .config import RingRole
        role = RingRole(target)
        ring = snapshot.airspace.actual_ring(role)
        theta = float(np.arctan2(endpoint[1], endpoint[0]))
        speed = float(np.clip(ring.nominal_speed + snapshot.airspace.speed_profile_bias, self.config.speed_min, self.config.speed_max))
        angle = theta + speed * max(0.0, time - other_plan.end_time) / ring.radius
        return np.asarray((ring.radius * np.cos(angle), ring.radius * np.sin(angle), ring.height))

    def filter_plan(self,plan,committed_plans,snapshot):
        if plan.plan_id!=getattr(plan,"plan_id",None):return TrajectorySafetyResult(False,"invalid_plan")
        for a,b in zip(plan.points,plan.points[1:]):
            dt=b.t-a.t
            velocity=(b.position-a.position)/dt
            if np.linalg.norm(velocity)>self.config.max_speed+1e-9:return TrajectorySafetyResult(False,"speed_limit",plan.plan_id)
            vertical=velocity[2]
            if vertical>self.config.max_climb_rate+1e-9 or -vertical>self.config.max_descent_rate+1e-9:return TrajectorySafetyResult(False,"vertical_rate_limit",plan.plan_id)
        for other_id,other in committed_plans.items():
            if other_id==plan.uav_id:continue
            # The other plan may finish before this candidate.  Use the same
            # post-plan ring-holder prediction for the remaining candidate
            # horizon rather than silently dropping it at ``other.end_time``.
            start=plan.start_time; end=plan.end_time
            samples=np.arange(start,end+1e-9,1.)
            if samples[-1] < end: samples=np.append(samples,end)
            def committed_position(time):
                return other.sample(float(time)).position if time <= other.end_time else self._post_plan_prediction(other,snapshot,float(time))
            for time in samples:
                if np.linalg.norm(plan.sample(float(time)).position-committed_position(float(time))) < self.config.safe_distance:
                    return TrajectorySafetyResult(False,"predicted_separation",plan.plan_id)
            for left,right in zip(samples,samples[1:]):
                distance,_=swept_minimum_distance(plan.sample(left).position,plan.sample(right).position,committed_position(left),committed_position(right))
                if distance<self.config.safe_distance:return TrajectorySafetyResult(False,"predicted_swept_separation",plan.plan_id)
            # A candidate that hands off to a ring must remain safe as the
            # canonical post-plan holder while an already committed transfer
            # continues.  Checking only the overlap of the two plan objects
            # misses this future encounter (e.g. INBOUND -> OUTER holder vs
            # a previously committed O2M transfer).
            if plan.metadata.get("target_phase") in {"ring_outer", "ring_middle", "ring_inner"} and other.end_time > plan.end_time:
                post_start=max(plan.end_time, other.start_time)
                post_end=other.end_time
                post_samples=np.arange(post_start,post_end+1e-9,1.)
                if post_samples[-1] < post_end: post_samples=np.append(post_samples,post_end)
                for time in post_samples:
                    if np.linalg.norm(self._post_plan_prediction(plan,snapshot,float(time))-other.sample(float(time)).position) < self.config.safe_distance:
                        return TrajectorySafetyResult(False,"predicted_post_plan_holder_separation",plan.plan_id)
                for left,right in zip(post_samples,post_samples[1:]):
                    distance,_=swept_minimum_distance(
                        self._post_plan_prediction(plan,snapshot,float(left)), self._post_plan_prediction(plan,snapshot,float(right)),
                        other.sample(float(left)).position, other.sample(float(right)).position,
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
            samples=np.arange(plan.start_time, plan.end_time + 1e-9, 1.0)
            if samples[-1] < plan.end_time: samples=np.append(samples,plan.end_time)
            def position(time):
                if other_plan is not None and time <= other_plan.end_time: return other_plan.sample(float(time)).position
                if other_plan is not None: return self._post_plan_prediction(other_plan, snapshot, float(time))
                return self._holding_prediction(other_uav, snapshot, float(time))
            for time in samples:
                # A plan that ends on a ring crosses an executor hand-off on
                # the next tick.  Reserve one bounded holder-motion step at
                # that boundary so a checked endpoint cannot immediately
                # become an unverified same-ring encounter.
                required = self.config.safe_distance
                if time >= plan.end_time - 1e-9 and plan.metadata.get("target_phase") in {"ring_outer", "ring_middle", "ring_inner"}:
                    required += self.config.max_speed * self.config.dt
                if np.linalg.norm(plan.sample(float(time)).position-position(float(time))) < required:
                    return TrajectorySafetyResult(False, "predicted_holder_separation", plan.plan_id)
            for left,right in zip(samples,samples[1:]):
                distance,_=swept_minimum_distance(plan.sample(left).position,plan.sample(right).position,position(left),position(right))
                if distance < self.config.safe_distance:
                    return TrajectorySafetyResult(False, "predicted_holder_swept_separation", plan.plan_id)
        self.checked_plan_ids.append(plan.plan_id);return TrajectorySafetyResult(True,checked_plan_id=plan.plan_id)

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
            for a, b in zip(plan.points, plan.points[1:]):
                dt = b.t - a.t
                if dt <= 0 or np.linalg.norm((b.position-a.position) / dt) > self.config.max_speed + 1e-9:
                    return finish(False, "repair_kinematics", plan.plan_id)
        samples = np.arange(start, end + 1e-9, 1.0)
        if samples[-1] < end:
            samples = np.append(samples, end)
        for time in samples:
            for index, (uid, plan) in enumerate(ordered):
                for other_uid, other in ordered[index+1:]:
                    if time > plan.metadata.get("active_end", plan.end_time) or time > other.metadata.get("active_end", other.end_time):
                        continue
                    distance=float(np.linalg.norm(plan.sample(float(time)).position-other.sample(float(time)).position))
                    minimum=min(minimum, distance); checked_pairs += 1
                    if distance < self.config.safe_distance:
                        return finish(False, f"repair_separation:{uid}:{other_uid}", plan.plan_id)
        for left, right in zip(samples, samples[1:]):
            for index, (uid, plan) in enumerate(ordered):
                for other_uid, other in ordered[index+1:]:
                    if left >= plan.metadata.get("active_end", plan.end_time) or left >= other.metadata.get("active_end", other.end_time):
                        continue
                    distance, _ = swept_minimum_distance(
                        plan.sample(float(left)).position, plan.sample(float(right)).position,
                        other.sample(float(left)).position, other.sample(float(right)).position,
                    )
                    minimum=min(minimum, float(distance)); checked_pairs += 1
                    if distance < self.config.safe_distance:
                        return finish(False, f"repair_swept_separation:{uid}:{other_uid}", plan.plan_id)
        return finish(True, "approved", "reconfiguration_tracks")
