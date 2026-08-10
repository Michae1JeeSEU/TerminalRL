"""Time/gap based merge reservations; no fixed release angles."""
from __future__ import annotations
from dataclasses import dataclass
import math, uuid
from .config import RingRole
from .uav import UAVState

@dataclass(frozen=True)
class LowLevelIntent:
    release_request:float; speed_residual:float; merge_time_residual:float
    @classmethod
    def from_array(cls,raw):
        values=[float(v) for v in raw]
        if len(values)!=3 or any(not math.isfinite(v) for v in values):raise ValueError("low-level intent must be finite shape (3,)")
        return cls(*(max(-1.,min(1.,v)) for v in values))
@dataclass
class MergeReservation:
    reservation_id:str; uav_id:int; source_stage:str; target_stage:str; merge_time:float; merge_angle:float|None; predecessor_id:int|None; successor_id:int|None; status:str="committed"; expires_at:float=0.
@dataclass(frozen=True)
class MergeProposal: target:RingRole|str; merge_time:float; merge_angle:float|None

class MergeScheduler:
    def __init__(self,config):self.config=config;self.reservations={};self.last_release={"o2m":-math.inf,"m2i":-math.inf,"i2f":-math.inf}
    def _key(self,target):return {RingRole.MIDDLE:"o2m",RingRole.INNER:"m2i","final":"i2f"}[target]
    def baseline_time(self,uav,target,snapshot):
        # Nominal merge preparation time.  A bounded residual modifies the
        # scheduler's reservation subject to the hard earliest headway.
        earliest=max(snapshot.now,snapshot.airspace.headways[self._key(target)]+self.last_release[self._key(target)])
        return earliest + 6.0
    def propose(self,uav,intent,snapshot,target):
        base=self.baseline_time(uav,target,snapshot); hard_earliest=max(snapshot.now,snapshot.airspace.headways[self._key(target)]+self.last_release[self._key(target)])
        t=max(hard_earliest, base+intent.merge_time_residual*self.config.max_merge_time_residual_seconds); angle=None
        if isinstance(target,RingRole):
            ring=snapshot.airspace.actual_ring(target); angle=self._gap_center(target,t,snapshot,ring.nominal_speed)
        return MergeProposal(target,t,angle)
    def _gap_center(self,target,time,snapshot,speed):
        states=[u for u in snapshot.active.values() if u.ring_role is target]
        if not states:return 0.0
        angles=sorted(((u.ring_progress or 0.)+speed*time/snapshot.airspace.actual_ring(target).radius)%(2*math.pi) for u in states)
        gaps=[((angles[(i+1)%len(angles)]-a)%(2*math.pi),a) for i,a in enumerate(angles)]
        gap,start=max(gaps); return (start+gap/2)%(2*math.pi)
    def reserve(self,uav,proposal):
        key=self._key(proposal.target)
        if proposal.merge_time<self.last_release[key]+self.config.__dict__[f"headway_{key}"]-1e-9: raise ValueError("headway reservation conflict")
        rid=str(uuid.uuid4()); r=MergeReservation(rid,uav.uid,uav.phase.value,str(getattr(proposal.target,'value',proposal.target)),proposal.merge_time,proposal.merge_angle,None,None,expires_at=proposal.merge_time+60.)
        self.reservations[rid]=r;self.last_release[key]=proposal.merge_time;return r
    def cancel(self,reservation_id,reason=""): self.reservations.pop(reservation_id,None)
