"""Three-ring actual/target geometry with event-gated reconfiguration."""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from .config import SimConfig, RingRole
from .ring_capacity import geometric_ring_capacity, minimum_radius_for_occupancy

@dataclass
class RingState: role:RingRole; radius:float; height:float; nominal_speed:float; derived_capacity:int=0
@dataclass
class GeometryTransition: start_time:float; end_time:float; start:dict; target:dict; affected_interfaces:set[str]
@dataclass(frozen=True)
class AirspaceUpdateResult: accepted:bool; structural_applied:bool; violations:tuple[str,...]; affected_interfaces:frozenset[str]

class DynamicAirspace:
    ACTION_SIZE=10
    def __init__(self, config:SimConfig):
        self.config=config; self._actual={role:RingState(role,s.radius,s.height,s.nominal_speed) for role,s in config.rings.items()}; self._target={role:replace(s) for role,s in self._actual.items()}
        self.speed_profile_bias=0.; self.headways={"o2m":config.headway_o2m,"m2i":config.headway_m2i,"i2f":config.headway_i2f}; self.transition=None; self.last_geometry_update=-float("inf"); self._occupancy={r:0 for r in RingRole}; self._committed={r:0 for r in RingRole}; self._update_capacity()
    def _update_capacity(self):
        for r in RingRole:self._actual[r].derived_capacity=geometric_ring_capacity(self._actual[r].radius,self.config.safe_distance)
    def actual_ring(self,role):return self._actual[role]
    def target_ring(self,role):return self._target[role]
    def capacity(self,role):return self._actual[role].derived_capacity
    def occupancy(self,role):return self._occupancy[role]
    def set_occupancy(self,values,committed_incoming=None):
        self._occupancy={r:int(values.get(r,0)) for r in RingRole}; self._committed={r:int((committed_incoming or {}).get(r,0)) for r in RingRole}
    def state_vector(self): return np.array([self._actual[r].radius for r in RingRole]+[self._actual[r].height for r in RingRole]+[self.speed_profile_bias,self.headways['o2m'],self.headways['m2i'],self.headways['i2f']],dtype=float)
    def normalized_state_vector(self):
        c=self.config; lo=np.array([c.radius_min]*3+[c.height_min]*3+[-2,c.headway_min]*3); hi=np.array([c.radius_max]*3+[c.height_max]*3+[2,c.headway_max]*3); return (2*(self.state_vector()-lo)/(hi-lo)-1).astype(np.float32)
    def affected_interfaces(self): return set() if self.transition is None else set(self.transition.affected_interfaces)
    def step_transition(self,now):
        if self.transition is None:return
        tr=self.transition; p=min(1.,max(0.,(now-tr.start_time)/(tr.end_time-tr.start_time)))
        for r in RingRole:
            a,b=tr.start[r],tr.target[r]; self._actual[r].radius=a.radius+(b.radius-a.radius)*p; self._actual[r].height=a.height+(b.height-a.height)*p
        self._update_capacity()
        if p>=1.: self.transition=None
    def request_update(self,action,*,now,occupancies=None,committed_incoming=None,geometry_gate=True):
        raw=np.asarray(action,dtype=float)
        if raw.shape!=(10,) or not np.all(np.isfinite(raw)):raise ValueError("high-level action must be finite shape (10,)")
        self.set_occupancy(occupancies or self._occupancy,committed_incoming); x=np.clip(raw,-1,1); c=self.config
        structural=bool(geometry_gate and now-self.last_geometry_update>=c.geometry_min_dwell_seconds)
        candidate={r:replace(self._target[r]) for r in RingRole}; violations=[]
        if structural:
            for i,r in enumerate(RingRole):
                candidate[r].radius=np.clip(candidate[r].radius+x[i]*c.max_radius_change_per_update,c.radius_min,c.radius_max); candidate[r].height=np.clip(candidate[r].height+x[i+3]*c.max_height_change_per_update,c.height_min,c.height_max)
            # project ordered outer > middle > inner and occupancy feasibility
            for outer,inner in ((RingRole.OUTER,RingRole.MIDDLE),(RingRole.MIDDLE,RingRole.INNER)):
                candidate[outer].radius=max(candidate[outer].radius,candidate[inner].radius+c.min_ring_gap); candidate[outer].height=max(candidate[outer].height,candidate[inner].height+c.min_height_gap)
            for r in RingRole:
                minimum=minimum_radius_for_occupancy(self._occupancy[r]+self._committed[r],c.safe_distance)
                if candidate[r].radius<minimum: candidate[r].radius=minimum; violations.append("occupancy_radius_projection")
            affected={"o2m","m2i","i2f"}; self.transition=GeometryTransition(now,now+c.geometry_transition_seconds,{r:replace(v) for r,v in self._actual.items()},candidate,affected); self._target=candidate; self.last_geometry_update=now
        else: violations.append("geometry_gate")
        self.speed_profile_bias=float(np.clip(self.speed_profile_bias+x[6]*c.max_speed_profile_change_per_update,-2.,2.))
        for key,i in (("o2m",7),("m2i",8),("i2f",9)): self.headways[key]=float(np.clip(self.headways[key]+x[i]*c.max_headway_change_per_update,c.headway_min,c.headway_max))
        return AirspaceUpdateResult(True,structural,tuple(violations),frozenset(self.affected_interfaces()))
