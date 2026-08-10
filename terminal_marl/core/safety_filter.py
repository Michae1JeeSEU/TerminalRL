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
        self.checked_plan_ids.append(plan.plan_id);return TrajectorySafetyResult(True,checked_plan_id=plan.plan_id)
