"""Named, versioned low-level observations and variable-N set batching."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from .config import RingRole
from .uav import FlightPhase
from .topology_neighbors import NeighborSlot, TopologyCache, TopologyNeighborSelector

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
    @property
    def feature_hash(self) -> str:
        payload="|".join(f"{item.name}:{item.size}" for item in self.features)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

A5_OBSERVATION_SCHEMA=ObservationSchema((
    FeatureSpec("phase",8), FeatureSpec("ring_role",3), FeatureSpec("ring_progress",2),
    FeatureSpec("speed",1), FeatureSpec("soc",1), FeatureSpec("wait",1), FeatureSpec("priority",1),
    FeatureSpec("mission",4), FeatureSpec("gate",4), FeatureSpec("lz",7),
    FeatureSpec("ring_occupancy",3), FeatureSpec("headways",3), FeatureSpec("resources",3),
))

# V1 stays available only for reproducing the historical A5 baseline.  V2 is
# deliberately explicit: no anonymous dimension is permitted in a checkpoint
# or a paper result.
LOW_OBS_V1_LEGACY = A5_OBSERVATION_SCHEMA
LOW_OBS_V2 = ObservationSchema((
    FeatureSpec("phase", 8), FeatureSpec("ring_role", 3), FeatureSpec("ring_progress", 2),
    FeatureSpec("speed", 1), FeatureSpec("soc", 1), FeatureSpec("wait", 1),
    FeatureSpec("priority", 1), FeatureSpec("release_eligible", 1),
    FeatureSpec("mission", 4), FeatureSpec("gate", 4), FeatureSpec("assigned_lz_present", 1),
    FeatureSpec("assigned_lz_usable", 1), FeatureSpec("final_resource_available", 1), FeatureSpec("free_lz_fraction", 1),
    FeatureSpec("same_ring_leader", 9), FeatureSpec("same_ring_follower", 9),
    FeatureSpec("target_ring_predecessor", 9), FeatureSpec("target_ring_successor", 9),
    FeatureSpec("nearest_predicted_conflict", 9),
    FeatureSpec("ring_occupancy", 3), FeatureSpec("headways", 3), FeatureSpec("external_queue_ratio", 1),
    FeatureSpec("final_queue_pressure", 1), FeatureSpec("global_free_lz_fraction", 1),
    FeatureSpec("interface_frozen", 1), FeatureSpec("interface_headway_slack", 1), FeatureSpec("reconfiguration_active_pending", 1),
))

# V2.1 preserves the V2 prefix byte-for-byte and appends precisely one online
# state variable.  Rejected release execution sets ``safety_retry_after`` to
# ``now + max(5 s, interface_headway)``; DynamicAirspace bounds every headway
# by the frozen formal V4.2 ``headway_max`` (30 s).  The matching config value
# is checked at construction, so a V2.1 deployment cannot silently normalize
# a retry interval outside its declared range.
LOW_OBS_V2_1 = ObservationSchema(LOW_OBS_V2.features + (
    FeatureSpec("safety_retry_remaining_norm", 1),
))

if LOW_OBS_V2.dimension != 87:  # import-time guard against accidental schema drift
    raise RuntimeError("phaseA_low_obs_v2_topology_87d must be exactly 87D")
if LOW_OBS_V2_1.dimension != 88:
    raise RuntimeError("phaseA_low_obs_v2_1_retry_state must be exactly 88D")

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


def _slot_vector(slot: NeighborSlot, config) -> list[float]:
    if not slot.present:
        return [0.0] * 9
    distance_scale = max(config.safe_distance * 4.0, 1.0)
    return [
        1.0, float(np.clip(slot.path_gap / distance_scale, -1.0, 1.0)),
        float(np.clip(slot.euclidean_distance / distance_scale, 0.0, 1.0)),
        float(np.clip(slot.relative_speed / config.max_speed, -1.0, 1.0)),
        float(np.clip(slot.closing_rate / config.max_speed, 0.0, 1.0)),
        float(np.clip(slot.d_cpa / distance_scale, 0.0, 1.0)),
        float(np.clip(slot.t_cpa / config.shield_horizon_seconds, 0.0, 1.0)),
        float(np.clip(slot.priority_delta / config.priority_feature_scale, -1.0, 1.0)),
        float(slot.reserved_or_committed),
    ]


def _release_eligible(uav, env) -> bool:
    if uav.committed_plan is not None or uav.soc <= 0.0:
        return False
    interface = {RingRole.OUTER: "o2m", RingRole.MIDDLE: "m2i", RingRole.INNER: "i2f"}.get(uav.ring_role)
    return interface is not None and interface not in env.airspace.affected_interfaces()


def build_low_observation_v2(uav, env, *, topology_cache: TopologyCache | None = None) -> np.ndarray:
    """Build the canonical 87D online topology observation without leakage."""
    phase_values=(FlightPhase.INBOUND,FlightPhase.RING_OUTER,FlightPhase.TRANSFER_O2M,FlightPhase.RING_MIDDLE,FlightPhase.TRANSFER_M2I,FlightPhase.RING_INNER,FlightPhase.FINAL,FlightPhase.LANDING)
    config=env.config; theta=uav.ring_progress or 0.0
    free_zones=sum(zone.status.value == "free" for zone in env.lz_manager.zones)
    assigned_usable = bool(uav.assigned_lz is not None and env.lz_manager.zones[uav.assigned_lz].status.value == "free")
    final_available = float(sum(item.phase is FlightPhase.FINAL for item in env.active.values()) < config.final_queue_capacity)
    if topology_cache is None:
        selector=TopologyNeighborSelector(env.airspace, now=env.now, cpa_horizon=config.shield_horizon_seconds)
        active=env.active.values()
        slots=(selector.same_ring_leader(uav, active), selector.same_ring_follower(uav, active),
               selector.target_ring_predecessor(uav, active), selector.target_ring_successor(uav, active),
               selector.nearest_predicted_conflict(uav, active))
    else:
        slots = topology_cache.slots_for(uav.uid)
    interface={RingRole.OUTER:"o2m", RingRole.MIDDLE:"m2i", RingRole.INNER:"i2f"}.get(uav.ring_role)
    last_release = -np.inf if interface is None else env.scheduler.last_executed_release[interface]
    headway = 0.0 if interface is None else env.airspace.headways[interface]
    headway_slack = 1.0 if not np.isfinite(last_release) else float(np.clip((env.now - last_release) / max(headway, 1e-9), 0.0, 1.0))
    values=[
        *_one(uav.phase, phase_values), *_one(uav.ring_role, tuple(RingRole)), np.sin(theta), np.cos(theta),
        float(np.clip(uav.speed/config.max_speed, 0.0, 1.0)), uav.soc,
        min(1.0,(uav.external_wait_time+uav.internal_wait_time)/config.priority_wait_reference_seconds),
        float(np.clip(uav.dynamic_priority/config.priority_feature_scale, 0.0, 1.0)), float(_release_eligible(uav, env)),
        *_one(uav.mission_type, tuple(type(uav.mission_type))), *_one(uav.assigned_gate, range(config.gate_count)),
        float(uav.assigned_lz is not None), float(assigned_usable), final_available, free_zones/max(1,config.num_landing_zones),
    ]
    for slot in slots:
        values.extend(_slot_vector(slot, config))
    values.extend([
        *[env.airspace.occupancy(role)/max(1,env.airspace.capacity(role)) for role in RingRole],
        env.airspace.headways["o2m"]/config.headway_max, env.airspace.headways["m2i"]/config.headway_max, env.airspace.headways["i2f"]/config.headway_max,
        len(env.admission.queue.items)/max(1,config.max_active_uavs),
        sum(item.phase is FlightPhase.FINAL for item in env.active.values())/max(1,config.final_queue_capacity),
        free_zones/max(1,config.num_landing_zones), float(interface in env.airspace.affected_interfaces()), headway_slack,
        float(env.airspace.transition is not None),
    ])
    vector=np.asarray(values,dtype=np.float32)
    if vector.shape != (LOW_OBS_V2.dimension,):
        raise RuntimeError(f"V2 observation schema mismatch: {vector.shape}")
    return vector


def safety_retry_remaining_norm(uav, env) -> float:
    """Current-state-only retry mask appended by Observation V2.1.

    This is deliberately separate from ``release_eligible``: its zero/nonzero
    semantics expose the Teacher's retry predicate without changing any V2
    field or using a future trajectory, reservation, or policy action.
    """
    retry_remaining = max(0.0, float(uav.safety_retry_after - env.now))
    horizon = float(env.config.retry_norm_horizon_seconds)
    if horizon <= 0.0:
        raise ValueError("retry_norm_horizon_seconds must be positive")
    return float(np.clip(retry_remaining / horizon, 0.0, 1.0))


def build_low_observation_v2_1(uav, env, *, topology_cache: TopologyCache | None = None) -> np.ndarray:
    """Return V2's unchanged 87D prefix plus normalized retry remaining."""
    prefix = build_low_observation_v2(uav, env, topology_cache=topology_cache)
    vector = np.concatenate((prefix, np.asarray([safety_retry_remaining_norm(uav, env)], dtype=np.float32)))
    if vector.shape != (LOW_OBS_V2_1.dimension,):
        raise RuntimeError(f"V2.1 observation schema mismatch: {vector.shape}")
    return vector


def observation_schema_for_version(version: str) -> ObservationSchema:
    if version == "phaseA_low_obs_v1_41d":
        return LOW_OBS_V1_LEGACY
    if version == "phaseA_low_obs_v2_topology_87d":
        return LOW_OBS_V2
    if version == "phaseA_low_obs_v2_1_retry_state":
        return LOW_OBS_V2_1
    raise ValueError(f"unsupported low-level observation schema: {version}")


def build_low_observation(uav, env, *, topology_cache: TopologyCache | None = None) -> np.ndarray:
    if env.config.observation_schema_version == "phaseA_low_obs_v1_41d":
        return build_a5_observation(uav, env)
    if env.config.observation_schema_version == "phaseA_low_obs_v2_topology_87d":
        return build_low_observation_v2(uav, env, topology_cache=topology_cache)
    if env.config.observation_schema_version == "phaseA_low_obs_v2_1_retry_state":
        return build_low_observation_v2_1(uav, env, topology_cache=topology_cache)
    raise ValueError(f"unsupported low-level observation schema: {env.config.observation_schema_version}")

def pad_agent_sets(agent_sets:list[np.ndarray])->tuple[np.ndarray,np.ndarray]:
    """Return B×N×D observations and B×N validity mask for the centralized critic."""
    if not agent_sets:raise ValueError("agent_sets cannot be empty")
    d=agent_sets[0].shape[-1];n=max(len(x) for x in agent_sets);batch=np.zeros((len(agent_sets),n,d),np.float32);mask=np.zeros((len(agent_sets),n),np.bool_)
    for i,x in enumerate(agent_sets):
        if x.ndim!=2 or x.shape[1]!=d:raise ValueError("inconsistent agent set dimensions")
        batch[i,:len(x)]=x;mask[i,:len(x)]=True
    return batch,mask
