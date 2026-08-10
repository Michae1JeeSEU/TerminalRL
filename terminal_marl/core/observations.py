"""Named, fixed-width A5 actor observations and variable-N set batching."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import RingRole
from .uav import FlightPhase

@dataclass(frozen=True)
class FeatureSpec: name:str; size:int
@dataclass(frozen=True)
class ObservationSchema:
    features:tuple[FeatureSpec,...]
    @property
    def dimension(self):return sum(x.size for x in self.features)
    @property
    def slices(self):
        at=0;result={}
        for f in self.features:result[f.name]=slice(at,at+f.size);at+=f.size
        return result

A5_OBSERVATION_SCHEMA=ObservationSchema((
    FeatureSpec("phase",8), FeatureSpec("ring_role",3), FeatureSpec("ring_progress",2),
    FeatureSpec("speed",1), FeatureSpec("soc",1), FeatureSpec("wait",1), FeatureSpec("priority",1),
    FeatureSpec("mission",4), FeatureSpec("gate",4), FeatureSpec("lz",7),
    FeatureSpec("ring_occupancy",3), FeatureSpec("headways",3), FeatureSpec("resources",3),
))

def _one(value, values):return [float(value==x) for x in values]
def build_a5_observation(uav, env)->np.ndarray:
    phase_values=(FlightPhase.INBOUND,FlightPhase.RING_OUTER,FlightPhase.TRANSFER_O2M,FlightPhase.RING_MIDDLE,FlightPhase.TRANSFER_M2I,FlightPhase.RING_INNER,FlightPhase.FINAL,FlightPhase.LANDING)
    role=_one(uav.ring_role,tuple(RingRole));theta=uav.ring_progress or 0.
    mission=_one(uav.mission_type,tuple(type(uav.mission_type)));gate=_one(uav.assigned_gate,range(4));lz=[float(uav.assigned_lz==i) for i in range(6)]+[float(uav.assigned_lz is None)]
    c=env.config;occupancy=[env.airspace.occupancy(r)/max(1,env.airspace.capacity(r)) for r in RingRole]
    values=[*_one(uav.phase,phase_values),*role,np.sin(theta),np.cos(theta),uav.speed/c.max_speed,uav.soc,min(1.,(uav.external_wait_time+uav.internal_wait_time)/c.priority_wait_reference_seconds),min(2.,uav.dynamic_priority/5.),*mission,*gate,*lz,*occupancy,env.airspace.headways["o2m"]/c.headway_max,env.airspace.headways["m2i"]/c.headway_max,env.airspace.headways["i2f"]/c.headway_max,sum(z.status.value=="free" for z in env.lz_manager.zones)/6.,len(env.admission.queue.items)/c.max_active_uavs,float(any(u.committed_plan for u in env.active.values()))]
    vector=np.asarray(values,dtype=np.float32)
    if vector.shape!=(A5_OBSERVATION_SCHEMA.dimension,):raise RuntimeError("observation schema mismatch")
    return vector

def pad_agent_sets(agent_sets:list[np.ndarray])->tuple[np.ndarray,np.ndarray]:
    """Return B×N×D observations and B×N validity mask for the centralized critic."""
    if not agent_sets:raise ValueError("agent_sets cannot be empty")
    d=agent_sets[0].shape[-1];n=max(len(x) for x in agent_sets);batch=np.zeros((len(agent_sets),n,d),np.float32);mask=np.zeros((len(agent_sets),n),np.bool_)
    for i,x in enumerate(agent_sets):
        if x.ndim!=2 or x.shape[1]!=d:raise ValueError("inconsistent agent set dimensions")
        batch[i,:len(x)]=x;mask[i,:len(x)]=True
    return batch,mask
