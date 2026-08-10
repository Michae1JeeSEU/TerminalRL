"""Resource/intent then canonical-plan safety verification."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class IntentSafetyResult: approved:bool;reason:str=""
@dataclass(frozen=True)
class TrajectorySafetyResult: approved:bool;reason:str="";checked_plan_id:str|None=None
class SafetyFilter:
    def __init__(self,config):self.config=config;self.checked_plan_ids=[]
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
            start=max(plan.start_time,other.start_time);end=min(plan.end_time,other.end_time)
            if end<start:continue
            for t in np.arange(start,end+1e-9,1.):
                if np.linalg.norm(plan.sample(t).position-other.sample(t).position)<self.config.safe_distance:return TrajectorySafetyResult(False,"predicted_separation",plan.plan_id)
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
            for time in np.arange(plan.start_time, plan.end_time + 1e-9, 1.0):
                candidate_position = plan.sample(float(time)).position
                if other_plan is not None and time <= other_plan.end_time:
                    other_position = other_plan.sample(float(time)).position
                elif other_plan is not None:
                    other_position = self._post_plan_prediction(other_plan, snapshot, float(time))
                else:
                    other_position = self._holding_prediction(other_uav, snapshot, float(time))
                if np.linalg.norm(candidate_position - other_position) < self.config.safe_distance:
                    return TrajectorySafetyResult(False, "predicted_holder_separation", plan.plan_id)
        self.checked_plan_ids.append(plan.plan_id);return TrajectorySafetyResult(True,checked_plan_id=plan.plan_id)
