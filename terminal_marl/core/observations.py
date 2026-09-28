"""Named, versioned low-level observations and variable-N set batching."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
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

# V2.2 keeps the qualified V2.1 prefix intact and exposes only scheduler state
# that already exists at the actor's pre-action instant.  It does not scan the
# 25-way proposal catalogue and does not expose a future trajectory or safety
# certificate.  The final two fields describe deterministic same-step
# execution competition under TerminalParallelEnv's active-UAV iteration
# order; they are state facts, not predictions of another policy's action.
LOW_OBS_V2_2 = ObservationSchema(LOW_OBS_V2_1.features + (
    FeatureSpec("interface_active_reservations_norm", 1),
    FeatureSpec("interface_future_reservations_norm", 1),
    FeatureSpec("interface_committed_transfers_norm", 1),
    FeatureSpec("interface_reservation_block_remaining_norm", 1),
    FeatureSpec("same_step_interface_contenders_norm", 1),
    FeatureSpec("same_step_interface_predecessors_norm", 1),
))

# V2.3 keeps the qualified 88D width while naming every field by its source.
# It is intentionally a separate schema: the values at several positions no
# longer mean the V2.1 global resource fields, so an old checkpoint must not be
# loaded into this actor by position alone.
LOW_OBS_V2_3 = ObservationSchema((
    FeatureSpec("self_phase", 8), FeatureSpec("self_ring_role", 3), FeatureSpec("self_ring_progress", 2),
    FeatureSpec("self_speed", 1), FeatureSpec("self_soc", 1), FeatureSpec("self_wait", 1),
    FeatureSpec("self_priority", 1), FeatureSpec("self_release_eligible", 1),
    FeatureSpec("self_mission_type", 4), FeatureSpec("self_assigned_gate", 4),
    FeatureSpec("self_assigned_lz_present", 1), FeatureSpec("center_urgent_unassigned_count_norm", 1),
    FeatureSpec("center_final_resource_available", 1), FeatureSpec("center_free_lz_fraction", 1),
    FeatureSpec("neighbor_same_ring_leader", 9), FeatureSpec("neighbor_same_ring_follower", 9),
    FeatureSpec("neighbor_target_ring_predecessor", 9), FeatureSpec("neighbor_target_ring_successor", 9),
    FeatureSpec("neighbor_nearest_predicted_conflict", 9),
    FeatureSpec("center_ring_occupancy", 3), FeatureSpec("center_headways", 3),
    FeatureSpec("center_external_queue_ratio", 1), FeatureSpec("center_final_queue_pressure", 1),
    FeatureSpec("center_earliest_compatible_lz_release_wait_norm", 1),
    FeatureSpec("center_interface_frozen", 1), FeatureSpec("center_interface_headway_slack", 1),
    FeatureSpec("center_reconfiguration_active_pending", 1),
    FeatureSpec("self_safety_retry_remaining_norm", 1),
))


@dataclass(frozen=True)
class CenterLandingZoneBroadcast:
    """Per-pad state published by the vertiport center at one observation tick."""

    lz_id: int
    status: str
    compatible_missions: tuple[str, ...]
    eta_seconds: float


@dataclass(frozen=True)
class CenterBroadcast:
    """Aggregate and resource state available from the vertiport center."""

    now: float
    ring_occupancy: tuple[float, ...]
    headways: tuple[float, ...]
    external_queue_ratio: float
    final_queue_pressure: float
    free_lz_fraction: float
    final_resource_available: bool
    affected_interfaces: tuple[str, ...]
    interface_headway_slack: tuple[float, ...]
    reconfiguration_active_pending: bool
    urgent_unassigned_uids: tuple[int, ...]
    landing_zones: tuple[CenterLandingZoneBroadcast, ...]

if LOW_OBS_V2.dimension != 87:  # import-time guard against accidental schema drift
    raise RuntimeError("phaseA_low_obs_v2_topology_87d must be exactly 87D")
if LOW_OBS_V2_1.dimension != 88:
    raise RuntimeError("phaseA_low_obs_v2_1_retry_state must be exactly 88D")
if LOW_OBS_V2_2.dimension != 94:
    raise RuntimeError("phaseA_low_obs_v2_2_scheduler_context_94d must be exactly 94D")
if LOW_OBS_V2_3.dimension != 88:
    raise RuntimeError("phaseA_low_obs_v2_3_source_bound_88d must be exactly 88D")

def _one(value, values):return [float(value==x) for x in values]

def _mission_observation(uav, config) -> list[float]:
    """Encode mission identity, or the neutral curriculum token."""
    if not config.task_priority_enabled:
        return [1.0, 0.0, 0.0, 0.0]
    return _one(uav.mission_type, tuple(type(uav.mission_type)))

def _encoded_phase(phase):
    # Preserve the qualified 8-way observation schema. Emergency landing is
    # an environment-owned committed final trajectory, so expose it through
    # the existing FINAL slot instead of an all-zero unknown phase.
    return FlightPhase.FINAL if phase is FlightPhase.EMERGENCY_LANDING else phase
def build_a5_observation(uav, env)->np.ndarray:
    phase_values=(FlightPhase.INBOUND,FlightPhase.RING_OUTER,FlightPhase.TRANSFER_O2M,FlightPhase.RING_MIDDLE,FlightPhase.TRANSFER_M2I,FlightPhase.RING_INNER,FlightPhase.FINAL,FlightPhase.LANDING)
    role=_one(uav.ring_role,tuple(RingRole));theta=uav.ring_progress or 0.
    mission=_mission_observation(uav, env.config);gate=_one(uav.assigned_gate,range(4));lz=[float(uav.assigned_lz==i) for i in range(6)]+[float(uav.assigned_lz is None)]
    c=env.config;occupancy=[env.airspace.occupancy(r)/max(1,env.airspace.capacity(r)) for r in RingRole]
    values=[*_one(_encoded_phase(uav.phase),phase_values),*role,np.sin(theta),np.cos(theta),uav.speed/c.max_speed,uav.soc,min(1.,(uav.external_wait_time+uav.internal_wait_time)/c.priority_wait_reference_seconds),min(2.,uav.dynamic_priority/5.),*mission,*gate,*lz,*occupancy,env.airspace.headways["o2m"]/c.headway_max,env.airspace.headways["m2i"]/c.headway_max,env.airspace.headways["i2f"]/c.headway_max,sum(z.status.value=="free" for z in env.lz_manager.zones)/6.,len(env.admission.queue.items)/c.max_active_uavs,float(any(u.committed_plan for u in env.active.values()))]
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


def _source_bound_interface(uav) -> str | None:
    return {RingRole.OUTER: "o2m", RingRole.MIDDLE: "m2i", RingRole.INNER: "i2f"}.get(uav.ring_role)


def _source_bound_release_eligible(uav, center: CenterBroadcast) -> bool:
    if uav.committed_plan is not None or uav.soc <= 0.0:
        return False
    interface = _source_bound_interface(uav)
    return interface is not None and interface not in center.affected_interfaces


def _center_wait_norm(eta_seconds: float, config) -> float:
    """Normalize a center-reported pad ETA; unknown ETA is unavailable=1."""
    if not math.isfinite(float(eta_seconds)):
        return 1.0
    wait = max(0.0, float(eta_seconds))
    headway_max = max(float(config.headway_max), 1e-9)
    return float(np.clip(wait / (wait + headway_max), 0.0, 1.0))


def build_center_broadcast(env) -> CenterBroadcast:
    """Build one center publication from the current environment state.

    This is the sole env/global-state boundary for V2.3.  The V2.3 actor
    builder consumes only the resulting immutable broadcast and its filtered
    topology slots.
    """
    config = env.config
    interfaces = ("o2m", "m2i", "i2f")
    ring_occupancy = tuple(
        float(env.airspace.occupancy(role) / max(1, env.airspace.capacity(role)))
        for role in RingRole
    )
    headways = tuple(float(env.airspace.headways[key]) for key in interfaces)
    interface_slack = []
    for key in interfaces:
        last_release = env.scheduler.last_executed_release[key]
        headway = env.airspace.headways[key]
        if not np.isfinite(last_release):
            interface_slack.append(1.0)
        else:
            interface_slack.append(float(np.clip((env.now - last_release) / max(headway, 1e-9), 0.0, 1.0)))

    urgent_unassigned_uids = tuple(sorted(
        int(item.uid) for item in env.active.values()
        if item.assigned_lz is None
        and (item.emergency_landing or item.soc <= config.emergency_trigger_soc)
    ))
    landing_zones = []
    for zone in env.lz_manager.zones:
        status = zone.status.value
        if status == "free":
            eta_seconds = 0.0
        elif status == "occupied":
            eta_seconds = max(0.0, float(zone.service_remaining))
        else:
            # A committed reservation carries a center-known conservative
            # release timestamp.  A plain reservation has no ETA yet and is
            # therefore broadcast as unavailable.
            eta_seconds = (
                math.inf if zone.expected_release_at is None
                else max(0.0, float(zone.expected_release_at) - float(env.now))
            )
        landing_zones.append(CenterLandingZoneBroadcast(
            lz_id=int(zone.lz_id), status=status,
            compatible_missions=tuple(sorted(mission.value for mission in zone.compatible_missions)),
            eta_seconds=eta_seconds,
        ))

    free_lz = sum(zone.status.value == "free" for zone in env.lz_manager.zones)
    final_count = sum(item.phase is FlightPhase.FINAL for item in env.active.values())
    return CenterBroadcast(
        now=float(env.now), ring_occupancy=ring_occupancy, headways=headways,
        external_queue_ratio=float(len(env.admission.queue.items) / max(1, config.max_active_uavs)),
        final_queue_pressure=float(final_count / max(1, config.final_queue_capacity)),
        free_lz_fraction=float(free_lz / max(1, config.num_landing_zones)),
        final_resource_available=bool(final_count < config.final_queue_capacity),
        affected_interfaces=tuple(sorted(env.airspace.affected_interfaces())),
        interface_headway_slack=tuple(interface_slack),
        reconfiguration_active_pending=bool(env.airspace.transition is not None),
        urgent_unassigned_uids=urgent_unassigned_uids,
        landing_zones=tuple(landing_zones),
    )


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
        *_one(_encoded_phase(uav.phase), phase_values), *_one(uav.ring_role, tuple(RingRole)), np.sin(theta), np.cos(theta),
        float(np.clip(uav.speed/config.max_speed, 0.0, 1.0)), uav.soc,
        min(1.0,(uav.external_wait_time+uav.internal_wait_time)/config.priority_wait_reference_seconds),
        float(np.clip(uav.dynamic_priority/config.priority_feature_scale, 0.0, 1.0)), float(_release_eligible(uav, env)),
        *_mission_observation(uav, config), *_one(uav.assigned_gate, range(config.gate_count)),
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


def build_low_observation_v2_3(
    uav, config, center_broadcast: CenterBroadcast,
    slots: tuple[NeighborSlot, ...],
) -> np.ndarray:
    """Build source-bound V2.3 from self state, center broadcast, and slots.

    This function deliberately has no environment or active-UAV access.  The
    caller must provide exactly the five slots after the communication-range
    filter has been applied.
    """
    if len(slots) != 5:
        raise ValueError("V2.3 requires five filtered topology slots")
    phase_values = (
        FlightPhase.INBOUND, FlightPhase.RING_OUTER, FlightPhase.TRANSFER_O2M,
        FlightPhase.RING_MIDDLE, FlightPhase.TRANSFER_M2I, FlightPhase.RING_INNER,
        FlightPhase.FINAL, FlightPhase.LANDING,
    )
    theta = uav.ring_progress or 0.0
    interface = _source_bound_interface(uav)
    interface_index = {"o2m": 0, "m2i": 1, "i2f": 2}.get(interface)
    mission_name = getattr(uav.mission_type, "value", str(uav.mission_type))
    compatible_etas = [
        zone.eta_seconds for zone in center_broadcast.landing_zones
        if mission_name in zone.compatible_missions
    ]
    earliest_eta = min(compatible_etas, default=math.inf)
    urgent_count = sum(uid != int(uav.uid) for uid in center_broadcast.urgent_unassigned_uids)
    values = [
        *_one(_encoded_phase(uav.phase), phase_values), *_one(uav.ring_role, tuple(RingRole)),
        np.sin(theta), np.cos(theta),
        float(np.clip(uav.speed / config.max_speed, 0.0, 1.0)),
        float(np.clip(uav.soc, 0.0, 1.0)),
        min(1.0, (uav.external_wait_time + uav.internal_wait_time) / config.priority_wait_reference_seconds),
        float(np.clip(uav.dynamic_priority / config.priority_feature_scale, 0.0, 1.0)),
        float(_source_bound_release_eligible(uav, center_broadcast)),
        *_mission_observation(uav, config), *_one(uav.assigned_gate, range(config.gate_count)),
        float(uav.assigned_lz is not None),
        min(1.0, urgent_count / max(1, config.max_active_uavs)),
        float(center_broadcast.final_resource_available), center_broadcast.free_lz_fraction,
    ]
    for slot in slots:
        values.extend(_slot_vector(slot, config))
    values.extend([
        *center_broadcast.ring_occupancy,
        *(headway / config.headway_max for headway in center_broadcast.headways),
        center_broadcast.external_queue_ratio, center_broadcast.final_queue_pressure,
        _center_wait_norm(earliest_eta, config),
        float(interface in center_broadcast.affected_interfaces),
        1.0 if interface_index is None else center_broadcast.interface_headway_slack[interface_index],
        float(center_broadcast.reconfiguration_active_pending),
        float(np.clip(
            max(0.0, float(uav.safety_retry_after) - center_broadcast.now)
            / max(float(config.retry_norm_horizon_seconds), 1e-9), 0.0, 1.0
        )),
    ])
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (LOW_OBS_V2_3.dimension,):
        raise RuntimeError(f"V2.3 observation schema mismatch: {vector.shape}")
    return vector


def _interface_context(uav) -> tuple[str, str, str] | None:
    """Return interface, reservation target, and plan target-phase keys."""
    return {
        RingRole.OUTER: ("o2m", RingRole.MIDDLE.value, "ring_middle"),
        RingRole.MIDDLE: ("m2i", RingRole.INNER.value, "ring_inner"),
        RingRole.INNER: ("i2f", "final", "landing"),
    }.get(uav.ring_role)


def _v2_2_scheduler_context(uav, env) -> np.ndarray:
    """Six bounded, pre-action scheduler/competition state features.

    Counts are normalized by declared capacity, while the potentially
    unbounded reservation horizon uses a monotone tanh scale.  All values are
    recomputable from the current environment state and deterministic
    executor order; no requested action or Teacher result is consulted.
    """
    context = _interface_context(uav)
    if context is None:
        return np.zeros(6, dtype=np.float32)
    interface, target_stage, target_phase = context
    reservations = tuple(
        item for item in env.scheduler.reservations.values()
        if item.target_stage == target_stage
    )
    future_uids = set(env.future_release_reservations)
    future_reservations = tuple(
        item for item in reservations
        if item.status == "future" and item.uav_id in future_uids
    )
    committed_transfers = sum(
        item.committed_plan is not None
        and item.committed_plan.metadata.get("target_phase") == target_phase
        for item in env.active.values()
    )
    last_reserved = env.scheduler.last_release[interface]
    block_remaining = 0.0 if not np.isfinite(last_reserved) else max(
        0.0, float(last_reserved + env.airspace.headways[interface] - env.now)
    )
    block_norm = float(np.tanh(block_remaining / max(env.config.headway_max, 1e-9)))

    affected = set(env.airspace.affected_interfaces())
    def is_contender(item) -> bool:
        return (
            item.ring_role is uav.ring_role
            and item.committed_plan is None
            and item.uid not in future_uids
            and item.soc > 0.0
            and env.now >= item.safety_retry_after
            and interface not in affected
        )

    ordered = tuple(env.active.values())
    contenders = tuple(item for item in ordered if is_contender(item))
    position = next((index for index, item in enumerate(ordered) if item.uid == uav.uid), len(ordered))
    predecessors = sum(is_contender(item) for item in ordered[:position])
    ring_capacity = max(1, env.airspace.capacity(uav.ring_role))
    system_capacity = max(1, env.config.max_active_uavs)
    return np.asarray([
        min(1.0, len(reservations) / system_capacity),
        min(1.0, len(future_reservations) / system_capacity),
        min(1.0, committed_transfers / system_capacity),
        block_norm,
        min(1.0, len(contenders) / ring_capacity),
        min(1.0, predecessors / ring_capacity),
    ], dtype=np.float32)


def build_low_observation_v2_2(uav, env, *, topology_cache: TopologyCache | None = None) -> np.ndarray:
    """Return V2.1's unchanged 88D prefix plus scheduler context."""
    prefix = build_low_observation_v2_1(uav, env, topology_cache=topology_cache)
    vector = np.concatenate((prefix, _v2_2_scheduler_context(uav, env)))
    if vector.shape != (LOW_OBS_V2_2.dimension,):
        raise RuntimeError(f"V2.2 observation schema mismatch: {vector.shape}")
    return vector


def observation_schema_for_version(version: str) -> ObservationSchema:
    if version == "phaseA_low_obs_v1_41d":
        return LOW_OBS_V1_LEGACY
    if version == "phaseA_low_obs_v2_topology_87d":
        return LOW_OBS_V2
    if version == "phaseA_low_obs_v2_1_retry_state":
        return LOW_OBS_V2_1
    if version == "phaseA_low_obs_v2_2_scheduler_context_94d":
        return LOW_OBS_V2_2
    if version == "phaseA_low_obs_v2_3_source_bound_88d":
        return LOW_OBS_V2_3
    raise ValueError(f"unsupported low-level observation schema: {version}")


def build_low_observation(
    uav, env, *, topology_cache: TopologyCache | None = None,
    center_broadcast: CenterBroadcast | None = None,
) -> np.ndarray:
    if env.config.observation_schema_version == "phaseA_low_obs_v1_41d":
        return build_a5_observation(uav, env)
    if env.config.observation_schema_version == "phaseA_low_obs_v2_topology_87d":
        return build_low_observation_v2(uav, env, topology_cache=topology_cache)
    if env.config.observation_schema_version == "phaseA_low_obs_v2_1_retry_state":
        return build_low_observation_v2_1(uav, env, topology_cache=topology_cache)
    if env.config.observation_schema_version == "phaseA_low_obs_v2_2_scheduler_context_94d":
        return build_low_observation_v2_2(uav, env, topology_cache=topology_cache)
    if env.config.observation_schema_version == "phaseA_low_obs_v2_3_source_bound_88d":
        if topology_cache is None or center_broadcast is None:
            raise ValueError("V2.3 requires a filtered TopologyCache and CenterBroadcast")
        return build_low_observation_v2_3(
            uav, env.config, center_broadcast, topology_cache.slots_for(uav.uid),
        )
    raise ValueError(f"unsupported low-level observation schema: {env.config.observation_schema_version}")

def pad_agent_sets(agent_sets:list[np.ndarray])->tuple[np.ndarray,np.ndarray]:
    """Return B×N×D observations and B×N validity mask for the centralized critic."""
    if not agent_sets:raise ValueError("agent_sets cannot be empty")
    d=agent_sets[0].shape[-1];n=max(len(x) for x in agent_sets);batch=np.zeros((len(agent_sets),n,d),np.float32);mask=np.zeros((len(agent_sets),n),np.bool_)
    for i,x in enumerate(agent_sets):
        if x.ndim!=2 or x.shape[1]!=d:raise ValueError("inconsistent agent set dimensions")
        batch[i,:len(x)]=x;mask[i,:len(x)]=True
    return batch,mask
