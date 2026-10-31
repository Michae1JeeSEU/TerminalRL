"""Future-time normal planning without running a speculative world branch.

Only normal planning uses this interface. Emergency/admission retain the existing
priority response semantics; their measured CPU service postpones normal work.
"""
from dataclasses import asdict, dataclass, replace
from bisect import bisect_right
from hashlib import sha256
from itertools import combinations
import math
import time

from .config import freeze, thaw, service_duration
from .energy import evaluate_power
from .models import ValidationResult
from .safety import validate_separation
from .trajectory import TrajectoryPlan, make_stable_plan
from .world import (AIRBORNE, HOLD, _planning_context,
                    evaluate_planning_snapshot)


EFFECTIVE_STATE_VERSION = "planning_effective_state_v1"
EMERGENCY_TIMING_MODE = "existing_priority_response_cpu_charged"
DEPENDENCY_PROTOCOL_VERSION = "affected_traffic_revalidation_v1"
SEARCH_HANDOFF_VERSION = "search_handoff_reserve_v1"
LOCAL_TRAFFIC_POLICY = "action_local_recovery_v1"
EPS = 1.e-8


@dataclass(frozen=True)
class PlanningTiming:
    snapshot_time: float
    effective_time: float
    ready_deadline: float

    def __post_init__(self):
        if (not all(math.isfinite(x) for x in asdict(self).values())
                or not self.snapshot_time < self.ready_deadline < self.effective_time):
            raise ValueError("snapshot_time < ready_deadline < effective_time required")

    @classmethod
    def from_lead(cls, snapshot_time, lead_seconds=30., preparation_margin=1.):
        return cls(snapshot_time, snapshot_time+lead_seconds,
                   snapshot_time+lead_seconds-preparation_margin)


class ProjectionRejected(ValueError):
    pass


def _capacity(config):
    return config.energy.equivalent_voltage_V*config.energy.reference_capacity_Ah*3600.


def _motion(snapshot, original, target):
    """Known commitment transitions and energy portions; no policy or integrator."""
    a, cursor = original, snapshot.time
    portions = []
    fixed = tuple(p for p in (snapshot.active.get(a.id), snapshot.soft.get(a.id)) if p is not None)
    active = soft = None
    if a.status not in AIRBORNE:
        if a.status == "LANDED_SERVICE" and a.service_end <= target+EPS:
            a = replace(a, status="CLEARED", version=a.version+1)
        return a, active, soft, portions
    base = (TrajectoryPlan.from_dict(thaw(snapshot.trajectories[a.trajectory_hash]), snapshot.config)
            if a.trajectory_hash in snapshot.trajectories else None)
    for k, p in enumerate(fixed):
        if p.start_time > cursor:
            end = min(target, p.start_time)
            if end > cursor:
                hold = base if base is not None and base.kind == "STABLE" else make_stable_plan(
                    a.id, a.ring, a.slot, cursor, max(cursor+1., end),
                    source_version=a.version, config=snapshot.config)
                portions.append((hold, cursor, end))
                s = hold.state(end)
                a = replace(a, p=s.p, v=s.v, a=s.a, jerk=s.jerk)
            cursor = end
            if target < p.start_time:
                return a, None, p, portions
        end = min(target, p.end_time)
        if end > cursor:
            portions.append((p, cursor, end))
        s = p.state(end)
        a = replace(a, p=s.p, v=s.v, a=s.a, jerk=s.jerk,
                    trajectory_hash=p.trajectory_hash)
        cursor = end
        if target < p.end_time:
            status = ("TRANSFER_O_M" if p.source_ring == "outer" else "TRANSFER_M_I") if p.kind == "TRANSFER" else p.kind
            if p.pad_id is not None and target >= p.segments[0].end_time:
                status = "VERTICAL_DESCENT"
            a = replace(a, status=status)
            return a, p, fixed[k+1] if k+1 < len(fixed) else None, portions
        if p.pad_id is not None:
            end_service = p.end_time+service_duration(snapshot.config, a.service_class)
            a = replace(a, status="LANDED_SERVICE", ring=None, slot=None, pad_id=p.pad_id,
                        touchdown_time=p.end_time, service_end=end_service,
                        version=a.version+1, battery_power_w=0.)
            if end_service <= target+EPS:
                a = replace(a, status="CLEARED", version=a.version+1)
            return a, None, None, portions
        base = make_stable_plan(a.id, p.target_ring, p.target_slot, p.end_time,
            p.end_time+1000., source_version=a.version+1, config=snapshot.config)
        a = replace(a, status=HOLD[p.target_ring], ring=p.target_ring, slot=p.target_slot,
                    version=a.version+1, trajectory_hash=base.trajectory_hash)
    if target > cursor:
        if base is None:
            raise ProjectionRejected("MISSING_APPROVED_CONTINUATION:"+a.id)
        portions.append((base, cursor, target))
        s = base.state(target)
        a = replace(a, p=s.p, v=s.v, a=s.a, jerk=s.jerk)
    return a, active, soft, portions


def project_visible_snapshot(snapshot, t_effective, *, timing=None, fine=False):
    """Project revealed commitments only; never reveal arrivals or invoke advance.

    SOC is a lower bound and peak use an upper bound including numerical error.
    Unknown/nonfinite energy or exhausted bounds explicitly reject projection.
    """
    if not math.isfinite(t_effective) or t_effective < snapshot.time:
        raise ProjectionRejected("INVALID_EFFECTIVE_TIME")
    if timing is not None and abs(t_effective-timing.effective_time) > EPS:
        raise ProjectionRejected("TIMING_MISMATCH")
    context = _planning_context(snapshot)
    aircraft, active, soft = {}, {}, {}
    trajectories = {h: thaw(v) for h, v in snapshot.trajectories.items()}
    for aid, original in snapshot.aircraft.items():
        if fine and original.status in AIRBORNE:
            context._refine_executed_energy(aid)
        source = context.aircraft[aid]
        a, current, future, portions = _motion(snapshot, source, t_effective)
        estimates = []
        for p, start, end in portions:
            q = context._energy(p, start, end, fine=True if fine else None)
            if not q.valid or not all(math.isfinite(x) for x in
                    (q.joules, q.soc_used, q.peak_seconds, q.error_j, q.peak_error_s)):
                raise ProjectionRejected("ENERGY_UNKNOWN:"+aid)
            estimates.append(q)
            trajectories[p.trajectory_hash] = p.to_dict()
            context._record_energy_charge(aid, p, start, end, q)
        history = context._energy_state()["executed"].get(aid, ())
        error_j = math.fsum(r["error_j"] for r in history)
        peak_error = math.fsum(r["peak_error_s"] for r in history)
        # Stable portions have no recorded numerical uncertainty.
        soc = source.soc-math.fsum(q.soc_used for q in estimates)-error_j/_capacity(snapshot.config)
        peak = source.peak_seconds+math.fsum(q.peak_seconds for q in estimates)+peak_error
        if source.status in AIRBORNE and (soc <= 0 or peak > 120.):
            raise ProjectionRejected("PROJECTED_ENERGY_INSUFFICIENT:"+aid)
        a = replace(a, soc=soc, peak_seconds=peak,
                    energy_j=source.energy_j+math.fsum(q.joules for q in estimates),
                    battery_power_w=(evaluate_power(a.v,a.a,snapshot.config).battery_w
                                     if a.status in AIRBORNE else 0.))
        aircraft[aid] = a
        if current is not None: active[aid] = current
        if future is not None: soft[aid] = future
        if a.status in HOLD.values() and a.trajectory_hash not in trajectories:
            # The exact stable plan generated at a known completed endpoint.
            completed = [p for p in (snapshot.active.get(aid),snapshot.soft.get(aid))
                         if p is not None and p.end_time <= t_effective]
            if completed:
                p = completed[-1]
                hold = make_stable_plan(aid,p.target_ring,p.target_slot,p.end_time,p.end_time+1000.,
                                        source_version=a.version,config=snapshot.config)
                trajectories[hold.trajectory_hash] = hold.to_dict()
    live = {p.trajectory_hash for p in (*active.values(), *soft.values())}
    dependencies = {h: v for h,v in snapshot.dependencies.items() if h in live}
    fields = (EFFECTIVE_STATE_VERSION,t_effective,aircraft,active,soft,snapshot.slots,
              snapshot.pads,dependencies,snapshot.source_clear,context._energy_state(),timing)
    version = sha256(repr(fields).encode()).hexdigest()
    return replace(snapshot,time=t_effective,version=version,aircraft=freeze(aircraft),
        active=freeze(active),soft=freeze(soft),dependencies=freeze(dependencies),
        trajectories=freeze(trajectories),energy_ledger=freeze(context._energy_state()),
        planning_timing=timing,effective_state_version=EFFECTIVE_STATE_VERSION)


def _emergency_key(a):
    e = a.emergency
    return (e.eligible,e.declared,e.pending,e.charged,e.state,e.declaration_time,
            e.declaration_ring,e.start_time,e.start_ring)


def _physical_key(a):
    return (a.status,a.ring,a.slot,a.pad_id,a.version,a.trajectory_hash,
            a.touchdown_time,a.service_end,a.service_class)


def _reservations(rows, resources, effective):
    # Kind/hash may naturally become occupied after a known maneuver settles.
    # Release support identity is checked separately via active/soft/dependencies.
    return tuple(sorted((repr(r.resource),r.aircraft_id,max(r.start,effective),r.end)
        for r in rows if r.resource in resources and r.end > effective))


def _commitment_table(source, original, effective):
    """Precompute natural transition signatures during measured preparation."""
    cuts = {source.time,effective}
    for p in (source.active.get(original.id),source.soft.get(original.id)):
        if p is not None:
            cuts.update((p.start_time,p.end_time))
            if p.pad_id is not None:
                cuts.update((p.segments[0].end_time,
                             p.end_time+service_duration(source.config,original.service_class)))
    if original.service_end is not None: cuts.add(original.service_end)
    rows = []
    for t in sorted(t for t in cuts if source.time<=t<=effective):
        a,active,soft,_ = _motion(source,original,t)
        rows.append((t,_physical_key(a),active,soft))
    return tuple(rows)


def _dependency_table(source, effective):
    """Known active/soft release graphs at their approved transition times."""
    approved=tuple((*source.active.values(),*source.soft.values()))
    cuts={source.time,effective}
    cuts.update(t for p in approved for t in (p.start_time,p.end_time) if source.time<=t<=effective)
    context=_planning_context(source)
    rows=[]
    for t in sorted(cuts):
        active,soft={},{}
        for aid,a in source.aircraft.items():
            _,current,future,_=_motion(source,a,t)
            if current is not None: active[aid]=current
            if future is not None: soft[aid]=future
        context.active,context.soft=active,soft
        completed={p.trajectory_hash for p in approved if p.end_time<=t}
        graph={}
        for p in (*active.values(), *soft.values()):
            if source.active.get(p.aircraft_id)==p or source.soft.get(p.aircraft_id)==p:
                if p.trajectory_hash not in source.dependencies: continue
                supports=source.dependencies[p.trajectory_hash]
            else:
                supports=(context._release_supports(p.target_ring,p.target_slot,p.aircraft_id,context.slots,soft)
                          if p.target_ring is not None else ())
            graph[p.trajectory_hash]=tuple(sorted(set(supports)-completed))
        rows.append((t,graph,tuple(sorted(completed))))
    return tuple(rows)


def _direct_dependencies(source, plans, starts, resources, pad_resources, graph_table, target):
    hashes={p.trajectory_hash for p in plans}
    hashes.update(h for _,deps in starts.values() for h in deps)
    ids={p.aircraft_id for p in plans}
    for r in (*source.slots,*source.pads):
        used=resources if isinstance(r.resource,tuple) else pad_resources
        if r.resource in used and r.end>target:
            ids.add(r.aircraft_id)
            if r.plan_hash: hashes.add(r.plan_hash)
    approved=tuple((*source.active.values(),*source.soft.values()))
    while True:
        old=(set(ids),set(hashes))
        for p in approved:
            if p.aircraft_id in ids or p.trajectory_hash in hashes:
                ids.add(p.aircraft_id); hashes.add(p.trajectory_hash)
        for _,graph,_ in graph_table:
            for h,values in graph.items():
                if h in hashes or hashes.intersection(values):
                    hashes.add(h); hashes.update(values)
        if old==(ids,hashes): break
    return tuple(sorted(ids)),tuple(sorted(hashes))


def _traffic_signature(traffic):
    return (traffic.base.trajectory_hash, tuple(p.trajectory_hash for p in traffic.plans),
            traffic.start_time, traffic.end_time)


def _pair_geometry(context, soft, proposed_ids):
    traffic=[context._traffic(a,soft) for a in context.aircraft.values()
             if a.status in AIRBORNE or a.status=="LANDED_SERVICE"]
    horizon=max([context.time,*(p.end_time for p in context.active.values()),
                 *(p.end_time for p in soft.values())])+2*math.pi/.035
    rows={}
    for x,y in combinations(sorted(traffic,key=lambda x:x.aircraft_id),2):
        if x.aircraft_id not in proposed_ids and y.aircraft_id not in proposed_ids: continue
        rows[(x.aircraft_id,y.aircraft_id)]=(_traffic_signature(x),_traffic_signature(y),context.time,min(horizon,x.end_time,y.end_time),x,y)
    return horizon,rows


def _external_delta(world, packet, baseline=None):
    baseline=packet if baseline is None else baseline
    source=baseline["traffic_source"]; tables=baseline["traffic_tables"]
    direct=set(packet["direct_ids"]); affected=set()
    for aid,table in tables.items():
        actual=world.aircraft.get(aid)
        if actual is None:
            return ValidationResult(False,"REJECTED","EXTERNAL_TRAFFIC_MISSING",dict(aircraft_id=aid))
        # Known non-live removal is conservative for all retained certificates.
        if actual.status not in AIRBORNE and actual.status!="LANDED_SERVICE": continue
        _,key,active,soft=table[bisect_right(tuple(r[0] for r in table),world.time)-1]
        if (_physical_key(actual)!=key or world.active.get(aid)!=active or world.soft.get(aid)!=soft
                or _emergency_key(actual)!=_emergency_key(source.aircraft[aid])):
            affected.add(aid)
        if abs(world.time-packet["timing"].effective_time)<=EPS:
            expected=baseline["traffic_projected"].aircraft.get(aid)
            if expected is None or any(math.dist(getattr(actual,k),getattr(expected,k))>tol
                    for k,tol in (("p",1e-6),("v",1e-7),("a",1e-7))):
                affected.add(aid)
    affected.update(a.id for a in world.aircraft.values() if a.id not in direct and
                    a.id not in tables and (a.status in AIRBORNE or a.status=="LANDED_SERVICE"))
    return (ValidationResult(False,"REJECTED","TRAFFIC_REVALIDATION_REQUIRED",dict(affected_aircraft_ids=sorted(affected)))
            if affected else ValidationResult(True,"GUARDED"))


def prepare_submission(world, plans, timing, *, traffic_policy="legacy", pair_certificates=None, selection_binding=None):
    """Issue a world-owned packet after full certification and transaction prep.

    The public publish API accepts only an issued token, never caller-supplied
    prepared calendars or a caller-constructed SubmissionEvaluation.
    """
    started = time.perf_counter()
    if traffic_policy not in ("legacy", LOCAL_TRAFFIC_POLICY):
        raise ValueError("unknown traffic policy")
    plans = tuple(plans)
    if (not isinstance(timing,PlanningTiming) or world.time > timing.ready_deadline
            or any(p.start_time<timing.effective_time-EPS for p in plans)
            or any(p.kind not in ("TRANSFER","FINAL_NORMAL") for p in plans)):
        return ValidationResult(False,"REJECTED","INVALID_LATENCY_PREFIX"), None
    source = world.build_planning_snapshot()
    source = replace(source,aircraft=freeze(dict(source.aircraft)),active=freeze(dict(source.active)),
                     soft=freeze(dict(source.soft)),requests=freeze(dict(source.requests)))
    try:
        projected = project_visible_snapshot(source,timing.effective_time,timing=timing,fine=True)
    except ProjectionRejected as exc:
        return ValidationResult(False,"REJECTED",str(exc)), None
    context = _planning_context(projected)
    if not plans:
        return ValidationResult(False,"REJECTED","NO_EXECUTABLE_PREFIX"), None
    if any(p.aircraft_id in projected.active for p in plans):
        return ValidationResult(False,"REJECTED","EXISTING_EFFECTIVE_COMMITMENT"), None
    pair_counts = dict(pairs_checked=0, reused_pairs=0)
    prepared = context._prepare_submission(plans, pair_certificates=pair_certificates, pair_counts=pair_counts)
    if isinstance(prepared,ValidationResult): return prepared, None
    if (context.aircraft != dict(projected.aircraft)
            or freeze(context._energy_state()["executed"]) != projected.energy_ledger["executed"]):
        return ValidationResult(False,"REJECTED","UNIFORM_REFINEMENT_REQUIRED"), None
    soft, slots, pads, clears, redirects, fine_plans = prepared
    ids = {p.aircraft_id for p in plans}
    # Normal prefixes cannot secretly preempt another entity's commitment.
    if redirects or any(soft.get(aid)!=p for aid,p in projected.soft.items() if aid not in ids):
        return ValidationResult(False,"REJECTED","NONLOCAL_LATENCY_TRANSACTION"), None
    starts = {}
    for p in plans:
        q = context._energy(p,p.start_time,p.end_time,fine=True)
        a = projected.aircraft[p.aircraft_id]
        portions = []
        if p.start_time > timing.effective_time:
            hold = make_stable_plan(a.id,a.ring,a.slot,timing.effective_time,p.start_time,
                                    source_version=a.version,config=world.config)
            portions.append(context._energy(hold,timing.effective_time,p.start_time,fine=True))
        holding = context._sum_energy(portions)
        normal_exception = (p.kind=="FINAL_NORMAL" and a.ring=="inner" and
            not a.emergency.declared and not a.emergency.pending and not context._rescue_v6())
        if (a.emergency.declared or ((a.emergency.eligible or a.emergency.pending) and not normal_exception)):
            return ValidationResult(False,"REJECTED","EMERGENCY_DISPOSITION_REQUIRED"), None
        if (not holding.valid or not q.valid
                or a.soc <= holding.soc_used+holding.error_j/_capacity(world.config)+q.soc_used+q.error_j/_capacity(world.config)
                or a.soc < holding.soc_used+holding.error_j/_capacity(world.config)+q.soc_used+q.error_j/_capacity(world.config)+(.25 if p.kind=="FINAL_NORMAL" else 0.)
                or a.peak_seconds+holding.peak_seconds+holding.peak_error_s+q.peak_seconds+q.peak_error_s > 120.):
            return ValidationResult(False,"REJECTED","PREPARED_ENERGY_BOUND"), None
        fine_plans[p.trajectory_hash] = True
        deps = (context._release_supports(p.target_ring,p.target_slot,p.aircraft_id,slots,soft)
                if p.target_ring is not None else set())
        starts[p.aircraft_id] = (q,tuple(sorted(deps)))
    resources = {r.resource for r in (*source.slots,*slots.reservations) if r.aircraft_id in ids}
    pad_resources = {r.resource for r in (*source.pads,*pads.reservations) if r.aircraft_id in ids}
    registry = world.__dict__.setdefault("_planning_latency_packets",{})
    token = sha256(repr((timing,plans,source.version,len(registry),id(prepared))).encode()).hexdigest()
    graph_table=_dependency_table(source,timing.effective_time)
    direct_ids,direct_hashes=_direct_dependencies(source,plans,starts,resources,pad_resources,graph_table,timing.effective_time)
    horizon,pairs=_pair_geometry(context,soft,ids)
    registry[token] = dict(timing=timing,source=source,projected=projected,plans=plans,
        slots=tuple(r for r in slots.reservations if r.aircraft_id in ids),
        pads=tuple(r for r in pads.reservations if r.aircraft_id in ids),
        resources=tuple(sorted(resources,key=repr)),pad_resources=tuple(sorted(pad_resources,key=repr)),
        resource_key=_reservations(source.slots,resources,timing.effective_time),
        pad_key=_reservations(source.pads,pad_resources,timing.effective_time),
        clears={p.trajectory_hash:clears[p.trajectory_hash] for p in plans},
        fine_plans={p.trajectory_hash:True for p in plans},starts=starts,
        commitment_tables={aid:_commitment_table(source,a,timing.effective_time)
            for aid,a in source.aircraft.items() if a.status in AIRBORNE or a.status=="LANDED_SERVICE"},
        dependency_table=graph_table,direct_ids=direct_ids,direct_hashes=direct_hashes,
        dependency_protocol=DEPENDENCY_PROTOCOL_VERSION,pair_horizon=horizon,
        pair_certificates={key:row[:4] for key,row in pairs.items()},
        traffic_source=source,traffic_projected=projected,
        traffic_tables={aid:_commitment_table(source,a,timing.effective_time) for aid,a in source.aircraft.items()
                        if aid not in direct_ids and (a.status in AIRBORNE or a.status=="LANDED_SERVICE")},
        revalidation=None,revalidation_sequence=0,
        ready_at=None, traffic_policy=traffic_policy, certification_checks=pair_counts)
    if traffic_policy == LOCAL_TRAFFIC_POLICY:
        from .local_traffic_recovery import initialize_packet
        initialize_packet(registry[token])
    registry[token]["not_ready_before"] = world.time+time.perf_counter()-started
    from .future_commitment import delivery_units
    registry[token]["delivery_units"] = delivery_units(plans,timing,projected)
    registry[token]["selection_binding"] = selection_binding
    for unit in registry[token]["delivery_units"]:
        world._event("reservation_prepared",unit["aircraft_id"],**{k:v for k,v in unit.items() if k!="aircraft_id"},
                     selection_binding=selection_binding)
    return ValidationResult(True,"PREPARED","",dict(token=token)), token


def _direct_guard(world, packet):
    source, target = packet["source"], packet["timing"].effective_time
    if world.failed or world.time > target+EPS or world.config.config_hash != source.config.config_hash:
        return ValidationResult(False,"REJECTED","LATENCY_PACKET_EXPIRED")
    if (getattr(world,"rescue_rule_profile","legacy_v5") !=
            (source.emergency_monitor or {}).get("rescue_rule_profile","legacy_v5")):
        return ValidationResult(False,"REJECTED","EMERGENCY_POLICY_CHANGED")
    for aid, table in packet["commitment_tables"].items():
        if aid not in packet["direct_ids"]: continue
        original = source.aircraft[aid]
        row = table[bisect_right(tuple(r[0] for r in table),world.time)-1]
        _, expected_key, active, soft = row
        actual = world.aircraft.get(aid)
        if actual is None or _physical_key(actual)!=expected_key:
            return ValidationResult(False,"REJECTED","TRAFFIC_COMMITMENT_CHANGED",dict(aircraft_id=aid))
        if (_emergency_key(actual)!=_emergency_key(original)
                or world.active.get(aid)!=active or world.soft.get(aid)!=soft):
            return ValidationResult(False,"REJECTED","DEPENDENCY_CHANGED",dict(aircraft_id=aid))
        capacity = _capacity(world.config)
        if abs((actual.soc+actual.energy_j/capacity)-(original.soc+original.energy_j/capacity)) > 1e-9:
            return ValidationResult(False,"REJECTED","ENERGY_BASIS_CHANGED",dict(aircraft_id=aid))
        if abs(world.time-target)<=EPS:
            bound = packet["projected"].aircraft[aid]
            if any(math.dist(getattr(actual,k),getattr(bound,k)) > tol
                   for k,tol in (("p",1e-6),("v",1e-7),("a",1e-7))):
                return ValidationResult(False,"REJECTED","CONTINUATION_STATE_CHANGED",dict(aircraft_id=aid))
            if actual.soc+1e-12 < bound.soc or actual.peak_seconds > bound.peak_seconds+1e-10:
                return ValidationResult(False,"REJECTED","ENERGY_BOUND_CHANGED",dict(aircraft_id=aid))
    if (_reservations(world.slots.reservations,packet["resources"],target)!=packet["resource_key"]
            or _reservations(world.pads.reservations,packet["pad_resources"],target)!=packet["pad_key"]):
        return ValidationResult(False,"REJECTED","RESOURCE_DEPENDENCY_CHANGED",dict(
            slot_before=packet["resource_key"],slot_after=_reservations(world.slots.reservations,packet["resources"],target),
            pad_before=packet["pad_key"],pad_after=_reservations(world.pads.reservations,packet["pad_resources"],target)))
    graph_table=packet["dependency_table"]
    _,expected_graph,completed=graph_table[bisect_right(tuple(r[0] for r in graph_table),world.time)-1]
    direct_hashes=set(packet["direct_hashes"])
    def relevant(graph):
        normalized={h:tuple(sorted(set(values)-set(completed))) for h,values in graph.items()}
        return {h:values for h,values in normalized.items() if h in direct_hashes or direct_hashes.intersection(values)}
    actual_graph=relevant(world.dependencies); expected_graph=relevant(expected_graph)
    if expected_graph != actual_graph:
        return ValidationResult(False,"REJECTED","RELEASE_DEPENDENCY_CHANGED")
    for p in (*world.active.values(),*world.soft.values()):
        if p.trajectory_hash not in direct_hashes: continue
        if source.source_clear.get(p.trajectory_hash) != world._source_clear.get(p.trajectory_hash):
            return ValidationResult(False,"REJECTED","RELEASE_TIME_CHANGED")
    for p in packet["plans"]:
        if world._emergency_priority_wait(p) is not None:
            return ValidationResult(False,"REJECTED","EMERGENCY_PRIORITY_CHANGED")
    return ValidationResult(True,"GUARDED")


def guard_submission(world, token):
    packet=world.__dict__.get("_planning_latency_packets",{}).get(token)
    if packet is None: return ValidationResult(False,"REJECTED","UNISSUED_LATENCY_PACKET")
    if packet.get("traffic_policy") == LOCAL_TRAFFIC_POLICY:
        from .local_traffic_recovery import guard_submission as local_guard
        return local_guard(world, packet)
    direct=_direct_guard(world,packet)
    return _external_delta(world,packet) if direct.accepted else direct


def revalidate_submission(world, token, *, max_work_s=None, provider=None):
    """Measured affected-pair certification; retain original own plans/energy/delta."""
    started=time.perf_counter()
    if max_work_s is not None and (not math.isfinite(max_work_s) or max_work_s<0):
        return ValidationResult(False,"UNKNOWN","REVALIDATION_WORK_DEADLINE")
    work_deadline=math.inf if max_work_s is None else started+max_work_s
    packet=world.__dict__.get("_planning_latency_packets",{}).get(token)
    if packet is None: return ValidationResult(False,"REJECTED","UNISSUED_LATENCY_PACKET")
    if packet.get("traffic_policy") == LOCAL_TRAFFIC_POLICY:
        from .local_traffic_recovery import revalidate_submission as local_revalidate
        return local_revalidate(world, token, max_work_s=max_work_s,provider=provider)
    packet["ready_at"]=None
    packet["revalidation"]=None
    direct=_direct_guard(world,packet)
    if not direct.accepted: return direct
    timing=packet["timing"]
    if world.time>timing.ready_deadline: return ValidationResult(False,"REJECTED","READY_DEADLINE_MISSED")
    delta=_external_delta(world,packet)
    if delta.accepted: return ValidationResult(True,"UNCHANGED",details=dict(affected_aircraft_ids=[]))
    if delta.reason!="TRAFFIC_REVALIDATION_REQUIRED": return delta
    checked=[]; reused=0; extended=0
    def expired():
        return ValidationResult(False,"UNKNOWN","REVALIDATION_WORK_DEADLINE",dict(
            affected_aircraft_ids=delta.details["affected_aircraft_ids"],pairs_checked=len(checked),
            reused_pairs=reused,full_fallback=False,horizon_extension_pairs=extended,
            minimum_work_s=time.perf_counter()-started))
    if time.perf_counter()>=work_deadline: return expired()
    current=world.build_planning_snapshot()
    current=replace(current,aircraft=freeze(dict(current.aircraft)),active=freeze(dict(current.active)),
                    soft=freeze(dict(current.soft)),requests=freeze(dict(current.requests)))
    foreign={aid:a for aid,a in current.aircraft.items() if aid not in packet["direct_ids"] and
             (a.status in AIRBORNE or a.status=="LANDED_SERVICE")}
    source=replace(current,aircraft=freeze(foreign),active=freeze({aid:p for aid,p in current.active.items() if aid in foreign}),
                   soft=freeze({aid:p for aid,p in current.soft.items() if aid in foreign}))
    try: projected=project_visible_snapshot(source,timing.effective_time,timing=timing,fine=True)
    except ProjectionRejected as exc: return ValidationResult(False,"REJECTED",str(exc),delta.details)
    if time.perf_counter()>=work_deadline: return expired()
    original=packet["projected"]
    merged=replace(original,aircraft=freeze({**{aid:a for aid,a in original.aircraft.items() if aid in packet["direct_ids"]},**dict(projected.aircraft)}),
        active=freeze({**{aid:p for aid,p in original.active.items() if aid in packet["direct_ids"]},**dict(projected.active)}),
        soft=freeze({**{aid:p for aid,p in original.soft.items() if aid in packet["direct_ids"]},**dict(projected.soft)}),
        slots=current.slots,pads=current.pads,trajectories=freeze({**dict(original.trajectories),**dict(projected.trajectories)}))
    context=_planning_context(merged); staged_soft=dict(merged.soft)
    staged_soft.update({p.aircraft_id:p for p in packet["plans"]})
    horizon,pairs=_pair_geometry(context,staged_soft,{p.aircraft_id for p in packet["plans"]})
    certificates={}
    for key,row in pairs.items():
        if time.perf_counter()>=work_deadline: return expired()
        sig_x,sig_y,start,end,x,y=row
        old=packet["pair_certificates"].get(key)
        if old is not None and old[:3]==row[:3] and end<=old[3]:
            certificates[key]=old; reused+=1; continue
        if old is not None and end>old[3]: extended+=1
        if end>start:
            result=validate_separation(x,y,start,end,config=world.config)
            checked.append(key)
            if time.perf_counter()>=work_deadline: return expired()
            if not result.accepted or result.status!="SAFE":
                return ValidationResult(False,"REJECTED",result.reason or result.status,
                    dict(affected_aircraft_ids=delta.details["affected_aircraft_ids"],pairs_checked=len(checked),
                         reused_pairs=reused,full_fallback=False,validation_status=result.status,pair=key))
        certificates[key]=(sig_x,sig_y,start,end)
    if time.perf_counter()>=work_deadline: return expired()
    packet["revalidation_sequence"]+=1; attempt=packet["revalidation_sequence"]
    packet["revalidation"]=dict(attempt=attempt,traffic_source=source,traffic_projected=projected,
        traffic_tables={aid:_commitment_table(source,a,timing.effective_time) for aid,a in foreign.items()},
        pair_certificates=certificates,pair_horizon=horizon,not_ready_before=world.time+time.perf_counter()-started)
    return ValidationResult(True,"REVALIDATING",details=dict(attempt=attempt,affected_aircraft_ids=delta.details["affected_aircraft_ids"],
        pairs_checked=len(checked),reused_pairs=reused,full_fallback=False,horizon_extension_pairs=extended,
        minimum_work_s=time.perf_counter()-started))


def complete_revalidation(world, token, attempt, *, promote=False):
    packet=world.__dict__.get("_planning_latency_packets",{}).get(token)
    if packet is None: return ValidationResult(False,"REJECTED","UNISSUED_LATENCY_PACKET")
    if packet.get("traffic_policy") == LOCAL_TRAFFIC_POLICY:
        from .local_traffic_recovery import complete_revalidation as local_complete
        return local_complete(world, token, attempt, promote=promote)
    candidate=packet.get("revalidation")
    if candidate is None or candidate["attempt"]!=attempt:
        return ValidationResult(False,"REJECTED","UNISSUED_REVALIDATION")
    if world.time<candidate["not_ready_before"]:
        return ValidationResult(False,"WAIT","REVALIDATION_NOT_COMPLETE",dict(due=candidate["not_ready_before"]))
    if world.time>packet["timing"].ready_deadline:
        return ValidationResult(False,"REJECTED","READY_DEADLINE_MISSED")
    direct=_direct_guard(world,packet)
    if not direct.accepted: return direct
    delta=_external_delta(world,packet,candidate)
    if not delta.accepted: return delta
    if promote:
        for key in ("traffic_source","traffic_projected","traffic_tables","pair_certificates","pair_horizon"):
            packet[key]=candidate[key]
        packet["revalidation"]=None; packet["not_ready_before"]=world.time
    return ValidationResult(True,"REVALIDATED")


def mark_submission_ready(world, token, *, handoff_delay=0.):
    """World-issued readiness receipt, required independently of wrapper state."""
    started = time.perf_counter()
    if not math.isfinite(handoff_delay) or handoff_delay < 0:
        return ValidationResult(False,"REJECTED","INVALID_HANDOFF_DELAY")
    packet = world.__dict__.get("_planning_latency_packets",{}).get(token)
    if packet is None: return ValidationResult(False,"REJECTED","UNISSUED_LATENCY_PACKET")
    if world.time < packet["not_ready_before"]:
        return ValidationResult(False,"REJECTED","PREPARATION_NOT_COMPLETE")
    if world.time > packet["timing"].ready_deadline+EPS:
        return ValidationResult(False,"REJECTED","READY_DEADLINE_MISSED")
    result = guard_submission(world,token)
    ready_at = world.time+handoff_delay+time.perf_counter()-started
    if ready_at > packet["timing"].ready_deadline+EPS:
        return ValidationResult(False,"REJECTED","READY_DEADLINE_MISSED")
    if result.accepted:
        packet["ready_at"] = ready_at
        return ValidationResult(True,"GUARDED","",dict(ready_at=ready_at))
    return result


def publish_submission(world, token):
    """Light guard plus atomic delta/activation; no trajectory/energy integration."""
    registry = world.__dict__.get("_planning_latency_packets",{})
    packet = registry.get(token)
    if packet is None: return ValidationResult(False,"REJECTED","UNISSUED_LATENCY_PACKET")
    if abs(world.time-packet["timing"].effective_time)>EPS:
        return ValidationResult(False,"REJECTED","NOT_EFFECTIVE_TIME")
    if packet["ready_at"] is None or packet["ready_at"] > packet["timing"].ready_deadline+EPS:
        return ValidationResult(False,"REJECTED","NO_TIMELY_READY_RECEIPT")
    if world.time < packet["ready_at"]:
        return ValidationResult(False,"REJECTED","PREPARATION_NOT_COMPLETE")
    guarded = guard_submission(world,token)
    if not guarded.accepted:
        from .future_commitment import discard_pending
        discard_pending(world,token,guarded.reason)
        return guarded
    registry.pop(token,None)
    plans = packet["plans"]
    ids = {p.aircraft_id for p in plans}
    # Preserve all unrelated current transactions and past occupancy intervals.
    slots, pads = world.slots.clone(), world.pads.clone()
    for calendar,name in ((slots,"slots"),(pads,"pads")):
        calendar._entries = [r for r in calendar.reservations if r.aircraft_id not in ids or r.end<=world.time]
        calendar._entries.extend(r for r in packet[name] if r.end>world.time)
        calendar._entries.sort(key=lambda r:(str(r.resource),r.start,r.aircraft_id,r.kind))
    world.slots, world.pads = slots, pads
    world._source_clear.update(packet["clears"])
    world._energy_state()["fine_plans"].update(packet["fine_plans"])
    for p in plans:
        a = world.aircraft[p.aircraft_id]
        q, deps = packet["starts"][a.id]
        world.trajectories[p.trajectory_hash] = p.to_dict()
        old = world.soft.pop(a.id,None)
        if old is not None and old.trajectory_hash!=p.trajectory_hash:
            world.dependencies.pop(old.trajectory_hash,None)
        world.dependencies[p.trajectory_hash] = deps
        if p.start_time > world.time+EPS:
            world.soft[a.id] = p
            world._event("soft_plan_reserved",a.id,trajectory_hash=p.trajectory_hash,
                start_time=p.start_time,end_time=p.end_time,planning_timing=asdict(packet["timing"]))
            continue
        world.active[a.id] = p
        world.slots.mark_committed(p.trajectory_hash)
        world.pads.mark_committed(p.trajectory_hash)
        world.dependencies[p.trajectory_hash] = deps
        e = a.emergency
        if world._rescue_v6():
            e = replace(e,normal_path_soc=None,normal_path_wait_seconds=0.,
                        normal_estimate_epoch=None,normal_estimate_source=None)
        if p.kind=="FINAL_NORMAL" and a.ring=="inner" and not e.declared:
            e = replace(e,eligible=False,pending=False,state="NONE")
        status = ("TRANSFER_O_M" if p.source_ring=="outer" else "TRANSFER_M_I") if p.kind=="TRANSFER" else p.kind
        touchdown_soc = a.soc-q.soc_used if p.kind=="FINAL_NORMAL" else None
        world.aircraft[a.id] = replace(a,status=status,emergency=e,trajectory_hash=p.trajectory_hash,
                                      normal_final_touchdown_soc=touchdown_soc)
        world._event("plan_committed",a.id,trajectory_hash=p.trajectory_hash,
                     start_time=p.start_time,end_time=p.end_time,planning_timing=asdict(packet["timing"]))
        details = (dict(start_soc=a.soc,required_soc=.25+q.soc_used,predicted_touchdown_soc=touchdown_soc)
                   if p.kind=="FINAL_NORMAL" else {})
        world._event("maneuver_started",a.id,trajectory_hash=p.trajectory_hash,
                     dependencies=list(deps),maneuver_kind=p.kind,**details)
    from .future_commitment import register_published
    units = register_published(world,packet,token)
    world._record(force=True)
    return ValidationResult(True,"ACCEPTED","",dict(submitted=len(plans),
        reservation_ids=[u["reservation_id"] for u in units],delivery_units=units,
        delivery_selection_binding=packet.get('selection_binding'),
        actual_delivery_selection_binding=packet.get('selection_binding'),
        waiting_for_reservation_start=any(p.start_time>world.time+EPS for p in plans)))


class _FuturePort:
    """Read-only future state plus a deferred proposal sink."""
    def __init__(self, snapshot, source, planning_wall_deadline=None, *, search_wall_deadline=None,
                 planning_wall_origin=None, search_handoff_reserve_seconds=0.):
        self._snapshot, self._source = snapshot, source
        self.planning_wall_deadline = planning_wall_deadline
        self.search_wall_deadline = search_wall_deadline
        self.planning_wall_origin = planning_wall_origin
        self.search_handoff_reserve_seconds = search_handoff_reserve_seconds
        self.search_handoff = None
        self.proposal, self.evidence = None, []

    @property
    def time(self): return self._snapshot.time
    @property
    def config(self): return self._snapshot.config
    @property
    def aircraft(self): return self._snapshot.aircraft
    @property
    def active(self): return self._snapshot.active
    @property
    def soft(self): return self._snapshot.soft
    @property
    def requests(self): return self._snapshot.requests
    def build_planning_snapshot(self): return self._snapshot
    def snapshot(self): return self._snapshot
    def evaluate_submission(self,plans,*,cancel=(),snapshot=None):
        return evaluate_planning_snapshot(snapshot or self._snapshot,plans,cancel=cancel)
    def refine_planning_boundary(self):
        self._snapshot = project_visible_snapshot(self._source,self.time,
            timing=self._snapshot.planning_timing,fine=True)
        return self._snapshot
    def commit_validated_prefix(self,result,snapshot_version=None):
        if (result.snapshot_version != self._snapshot.version or not result.accepted
                or result.refinement_required or result.cancel):
            return ValidationResult(False,"REJECTED","INVALID_DEFERRED_PROPOSAL")
        self.proposal = tuple(result.plans)
        return ValidationResult(False,"DEFERRED","AWAITING_EFFECTIVE_TIME")
    def report_normal_evidence(self,rows,snapshot_version):
        self.evidence.extend(dict(row) for row in rows)
        return ValidationResult(False,"DEFERRED","AWAITING_ACTUAL_COMMIT")


class LatencyAwareScheduler:
    """One normal task with coalesced triggers and executor-owned delivery events."""
    def __init__(self, callback, *, lead_seconds=30., preparation_margin=1.,
                 search_handoff_reserve_seconds=0., clock=time.perf_counter, restart_policy="legacy",
                 traffic_policy="auto"):
        PlanningTiming.from_lead(0.,lead_seconds,preparation_margin)
        if not math.isfinite(search_handoff_reserve_seconds) or search_handoff_reserve_seconds<0:
            raise ValueError("invalid search handoff reserve")
        if restart_policy not in ("legacy", "coalesced_restart_v2"):
            raise ValueError("unknown normal restart policy")
        self.restart_policy = restart_policy
        self._restart_requested = None
        self._last_normal_start_time = None
        self._last_restart_time = None
        self.callback = callback
        if traffic_policy == "auto":
            traffic_policy = (LOCAL_TRAFFIC_POLICY if getattr(callback, "evaluation_profile", None)
                              == "service_opportunity_v1" else "legacy")
        if traffic_policy not in ("legacy", LOCAL_TRAFFIC_POLICY):
            raise ValueError("unknown traffic policy")
        self.traffic_policy = traffic_policy
        self.lead_seconds, self.preparation_margin = lead_seconds, preparation_margin
        self.search_handoff_reserve_seconds = search_handoff_reserve_seconds
        self.clock = clock
        self.pending = None
        self.dirty = False
        self.records, self.online_services = [], []
        self._idle_service = (None,0.)
        self._service_until = 0.
        self._handoff_service = 0.

    @property
    def planning_profile(self): return getattr(self.callback,"planning_profile","progress_tail_v7")

    @property
    def delivery_policy(self): return "absolute_first_action_reservation_v1"

    @property
    def next_event_time(self):
        if self.pending is None: return math.inf
        return self.pending["due"]

    def online_service(self, now, wall_s, cpu_s, kind):
        if wall_s < 0 or not math.isfinite(wall_s): raise ValueError("invalid online service")
        self.online_services.append(dict(sequence=len(self.online_services),time=now,kind=kind,
            wall_s=wall_s,cpu_s=cpu_s,emergency_timing_mode=EMERGENCY_TIMING_MODE))
        if self.pending is not None:
            self.pending["online_service_s"] += wall_s
            if self.pending["stage"] in ("RESULT","PREPARING","READY_SERVICE","REVALIDATING"):
                if self.pending["due"] <= now+EPS:
                    # Work has completed before this priority handoff. Charge the
                    # handoff once to the next stage, rather than re-serving a
                    # completion event forever at successively later timestamps.
                    self._handoff_service += wall_s
                else:
                    self.pending["due"] += wall_s
            elif self.pending["stage"] in ("PREPARED", "REPAIR_START"):
                self._service_until=max(now,self._service_until)+wall_s
            elif self.pending["stage"] == "REPAIRING":
                self._service_until=max(now,self._service_until)+wall_s
                delivery = self.pending["repair_delivery_due"]
                if delivery <= now+EPS: self._handoff_service += wall_s
                else:
                    delivery += wall_s
                    self.pending["repair_delivery_due"] = delivery
                    self.pending["due"] = min(delivery,self.pending["timing"].effective_time)
        else:
            old_time, value = self._idle_service
            self._idle_service = (now,wall_s+(value if old_time==now else 0.))
            self._service_until = max(now,self._service_until)+wall_s

    def __call__(self, observation, port=None):
        port = observation if port is None else port
        if self.pending is not None:
            self._observe_selective(port)
            if not self.dirty:
                self.dirty = True
                port.planning_latency_event("normal_trigger_coalesced")
            return
        if self.restart_policy == "coalesced_restart_v2" and self._last_normal_start_time == port.time:
            return
        self._last_normal_start_time = port.time
        self._restart_requested = None
        timing = PlanningTiming.from_lead(port.time,self.lead_seconds,self.preparation_margin)
        self.dirty = False
        actual_started=time.perf_counter()
        planning_wall_deadline=actual_started+(timing.ready_deadline-timing.snapshot_time)-max(0.,self._service_until-port.time)
        search_wall_deadline=(planning_wall_deadline-self.search_handoff_reserve_seconds
                              if self.search_handoff_reserve_seconds else None)
        started, started_cpu = self.clock(), time.process_time()
        proposal, evidence, reason, future = None, [], None, None
        try:
            source = port.build_planning_snapshot()
            if getattr(self.callback,"selective_method",None) is not None:
                if getattr(self.callback.selective_history,"progress_policy","legacy")=="maneuver_started_v2":
                    self.callback.observe_execution_snapshot(source,port.time,events=self._normal_start_events(port))
                else: self.callback.observe_execution_snapshot(source,port.time)
            future = _FuturePort(project_visible_snapshot(source,timing.effective_time,timing=timing),source,planning_wall_deadline,
                search_wall_deadline=search_wall_deadline,planning_wall_origin=actual_started,
                search_handoff_reserve_seconds=self.search_handoff_reserve_seconds)
            self.callback(future)
            proposal, evidence = future.proposal, future.evidence
            if proposal is None: reason = "NO_CERTIFIED_PREFIX"
            elif not proposal: reason = "NO_EXECUTABLE_PREFIX"
        except ProjectionRejected as exc:
            reason = str(exc)
        duration = self.clock()-started
        if duration < 0 or not math.isfinite(duration): raise ValueError("invalid normal duration")
        service = max(0.,self._service_until-port.time)
        self._service_until = port.time
        self._idle_service = (None,0.)
        record = dict(interface=EFFECTIVE_STATE_VERSION,emergency_timing_mode=EMERGENCY_TIMING_MODE,
            delivery_policy="absolute_first_action_reservation_v1",
            traffic_policy=self.traffic_policy,
            dependency_protocol=DEPENDENCY_PROTOCOL_VERSION,revalidation_work_s=0.,revalidation_work_cpu_s=0.,
            revalidation_count=0,revalidations=[],dependency_guard_work_s=0.,dependency_guard_work_cpu_s=0.,
            **asdict(timing),normal_work_s=duration,normal_work_cpu_s=time.process_time()-started_cpu,
            prepare_work_s=None,online_service_s=service,
            status="COMPUTING",submitted=0,commit_status="DEFERRED")
        final_result=getattr(self.callback,"last_result",None)
        if final_result is not None:
            record["selection_binding"]=dict(objective=list(final_result.objective),
                selected_signatures=[c.signature for c in final_result.selected],
                first_action_hashes=[c.plans[0].trajectory_hash for c in final_result.selected if c.plans],
                prefix_hashes=[p.trajectory_hash for p in final_result.prefix],
                certified=final_result.certified)
        if getattr(self.callback,"selective_method",None) is not None:
            record["proposed_trajectory_hashes"]=[p.trajectory_hash for p in proposal or ()]
        if self.search_handoff_reserve_seconds:
            record["search_handoff"] = (future.search_handoff if future is not None else None)
        self.records.append(record)
        self.pending = dict(timing=timing,stage="RESULT",due=port.time+duration+service,
            proposal=proposal,evidence=evidence,original_evidence=list(evidence),reason=reason,token=None,online_service_s=service,
            record_index=len(self.records)-1)
        port.planning_latency_event("normal_task_started",**record)

    def _normal_start_events(self, port):
        read = getattr(port,"normal_maneuver_start_events",None)
        history = getattr(self.callback,"selective_history",None)
        return read(getattr(history,"event_cursor",0)) if callable(read) else ()

    def restart_coalesced(self, port):
        """Executor calls after public boundary work, never from _finish."""
        if (self.restart_policy != "coalesced_restart_v2" or self.pending is not None
                or self._restart_requested is None or not self.dirty
                or self._last_restart_time == port.time or self._last_normal_start_time == port.time):
            return False
        previous = self._restart_requested
        previous_index = previous["record_index"]
        if previous_index is None and not previous.get("external_predecessor"):
            raise ValueError("external restart predecessor provenance required")
        self._restart_requested = None
        self._last_restart_time = port.time
        self(port)
        self.records[-1].update(restart_reason=previous["reason"],previous_finished_time=previous["finished_time"],
            previous_record_index=previous["record_index"],finish_to_next_start_s=port.time-previous["finished_time"])
        if previous_index is not None:
            self.records[previous_index].update(next_started_time=port.time,
                finish_to_next_start_s=port.time-previous["finished_time"],restart_reason=previous["reason"])
        else:
            self.records[-1]["external_predecessor"] = previous["external_predecessor"]
        port.planning_latency_event("normal_coalesced_restart",**previous,next_started_time=port.time)
        return True

    def _observe_selective(self, port, delivery_timing=None):
        if getattr(self.callback,"selective_method",None) is None: return
        started, cpu=time.perf_counter(),time.process_time()
        snapshot=port.build_planning_snapshot()
        if getattr(self.callback.selective_history,"progress_policy","legacy")=="maneuver_started_v2":
            self.callback.observe_execution_snapshot(snapshot,port.time,events=self._normal_start_events(port))
        else: self.callback.observe_execution_snapshot(snapshot,port.time)
        if delivery_timing is not None: self.callback.record_evaluation_delivery(delivery_timing,snapshot,port.time)
        wall,used_cpu=time.perf_counter()-started,time.process_time()-cpu
        self.online_service(port.time,wall,used_cpu,"selective_history")
        if self.pending is not None:
            record=self.records[self.pending["record_index"]]
            record["selective_history_work_s"]=record.get("selective_history_work_s",0.)+wall
            record["selective_history_work_cpu_s"]=record.get("selective_history_work_cpu_s",0.)+used_cpu

    def _finish(self, port, status, reason="", result=None):
        p = self.pending
        record = self.records[p["record_index"]]
        if self.traffic_policy == LOCAL_TRAFFIC_POLICY and p["token"]:
            events = list(port.latency_submission_recovery_log(p["token"]))
            if events: record["local_recovery_events"] = events
        if p["token"]: port.discard_latency_submission(p["token"],reason or status)
        record.update(status=status,reason=reason,finished_time=port.time,
            online_service_s=p["online_service_s"],commit_status=getattr(result,"status","REJECTED"),
            submitted=(len(p["proposal"]) if result is not None and result.accepted else 0),
            dirty=self.dirty)
        if result is not None and result.accepted:
            record.update({key:result.details[key] for key in
                ('reservation_ids','delivery_units','waiting_for_reservation_start','delivery_selection_binding',
                 'actual_delivery_selection_binding') if key in result.details})
        port.planning_latency_event("normal_task_finished",**record)
        acknowledge = getattr(self.callback,"record_latency_ack",None)
        self._observe_selective(port)
        if callable(acknowledge):
            acknowledge(p["timing"],record,tuple(p["proposal"] or ()))
        self.pending = None
        if self.restart_policy == "coalesced_restart_v2" and self.dirty:
            self._restart_requested = dict(record_index=p["record_index"],finished_time=port.time,
                reason="COALESCED_AFTER_"+status)
        if self._handoff_service:
            # A rejection can end normal work while this priority dispatch's
            # service still occupies the virtual CPU. Do not give it away.
            self._service_until = max(port.time,self._service_until)+self._handoff_service
        self._handoff_service = 0.

    def _revalidation_budget(self, port, *, extra_delay=0.):
        queued=max(0.,self._service_until-port.time)+self._handoff_service+extra_delay
        started=port.time+queued
        return dict(dispatch_time=port.time,queued_service_s=queued,planned_started_time=started,
            available_work_s=max(0.,self.pending["timing"].ready_deadline-port.time-queued))

    def _sync_local_proposal(self, port):
        """Acknowledgements and evidence follow the world-issued active token."""
        plans = port.latency_submission_plans(self.pending["token"])
        if plans is None: raise ValueError("issued local latency packet missing")
        self.pending["proposal"] = plans
        ids = {plan.aircraft_id for plan in plans}
        self.pending["evidence"] = [row for row in self.pending.get("original_evidence",self.pending["evidence"])
                                    if row.get("aircraft_id") in ids]
        self.records[self.pending["record_index"]]["retained_trajectory_hashes"] = [
            plan.trajectory_hash for plan in plans]

    def _begin_revalidation(self, port, *, extra_delay=0.):
        p=self.pending; record=self.records[p["record_index"]]
        budget=self._revalidation_budget(port,extra_delay=extra_delay)
        record["last_revalidation_budget"]=budget
        if port.time>p["timing"].ready_deadline:
            self._handoff_service+=extra_delay
            self._finish(port,"REJECTED","TRAFFIC_CHANGED_AFTER_DEADLINE")
            return
        previous_ready=record.get("ready_time")
        record["ready_time"]=None
        started,started_cpu=self.clock(),time.process_time()
        provider=getattr(self.callback,'provider',None)
        evaluation=port.revalidate_latency_submission(p["token"],max_work_s=budget["available_work_s"],
            **({'provider':provider} if provider is not None else {}))
        duration=self.clock()-started; cpu=time.process_time()-started_cpu
        if duration<0 or not math.isfinite(duration): raise ValueError("invalid revalidation duration")
        duration=max(duration,evaluation.details.get("minimum_work_s",0.))
        delay=budget["queued_service_s"]
        self._service_until=port.time; self._handoff_service=0.
        row=dict(**budget,started_time=port.time+delay,due=port.time+delay+duration,
            affected_aircraft_ids=list(evaluation.details.get("affected_aircraft_ids",())),
            pairs_checked=evaluation.details.get("pairs_checked",0),reused_pairs=evaluation.details.get("reused_pairs",0),
            full_fallback=evaluation.details.get("full_fallback",False),
            horizon_extension_pairs=evaluation.details.get("horizon_extension_pairs",0),
            reason=evaluation.reason or ("HORIZON_EXTENDED" if evaluation.details.get("horizon_extension_pairs") else "AFFECTED_TRAFFIC"),
            status="COMPUTING",wall_s=duration,cpu_s=cpu)
        if previous_ready is not None: row["previous_ready_time"]=previous_ready
        record["revalidation_count"]+=1; record["revalidation_work_s"]+=duration
        record["revalidation_work_cpu_s"]+=cpu; record["revalidations"].append(row)
        record.update(status="REVALIDATING",commit_status="DEFERRED")
        p.update(stage="REVALIDATING",due=row["due"],revalidation_attempt=evaluation.details.get("attempt"),
            revalidation_accepted=evaluation.accepted,revalidation_reason=evaluation.reason,
            repair_requested=bool(evaluation.details.get("affected_action_hashes")))
        if self.traffic_policy == LOCAL_TRAFFIC_POLICY:
            row["local_recovery"] = dict(evaluation.details)
        port.planning_latency_event("normal_revalidation_started",**row)

    def inspect_dependencies(self, port):
        if self.pending is None or self.pending["stage"] not in ("PREPARED", "REPAIR_START", "REPAIRING"): return
        record=self.records[self.pending["record_index"]]
        started,started_cpu=self.clock(),time.process_time()
        result=port.latency_packet_guard(self.pending["token"])
        duration=self.clock()-started; cpu=time.process_time()-started_cpu
        record["dependency_guard_work_s"]+=duration; record["dependency_guard_work_cpu_s"]+=cpu
        self.online_service(port.time,duration,cpu,"dependency_guard")
        if self.traffic_policy == LOCAL_TRAFFIC_POLICY and result.details.get("affected_aircraft_ids"):
            record.setdefault("guard_local_recovery", []).append(dict(result.details,
                observed_time=port.time, accepted=result.accepted, reason=result.reason, wall_s=duration))
        if result.reason=="TRAFFIC_REVALIDATION_REQUIRED": self._begin_revalidation(port)
        elif not result.accepted: self._finish(port,"REJECTED",result.reason)

    def planning_event(self, port):
        if self.pending is None: return
        p = self.pending
        timing = p["timing"]
        if p["due"] > port.time+EPS: return
        self._observe_selective(port,timing if p["stage"]=="RESULT" else None)
        if p["stage"] in ("RESULT","PREPARING","READY_SERVICE","REVALIDATING") and (
                port.time > timing.ready_deadline+EPS or p["due"] > timing.ready_deadline+EPS):
            if p["stage"]=="REVALIDATING":
                self.records[p["record_index"]]["revalidations"][-1].update(completed_time=port.time,status="LATE",reason="READY_DEADLINE_MISSED")
            elif p.get("completion_revalidation_index") is not None:
                self.records[p["record_index"]]["revalidations"][p["completion_revalidation_index"]].update(
                    completed_time=port.time,status="LATE",reason="READY_DEADLINE_MISSED")
            self._finish(port,"REJECTED","READY_DEADLINE_MISSED")
            return
        record = self.records[p["record_index"]]
        if p["stage"] == "RESULT":
            record["result_time"] = port.time+self._handoff_service
            if p["reason"]:
                self._finish(port,"NO_EXECUTABLE_PREFIX" if p["reason"]=="NO_EXECUTABLE_PREFIX" else "REJECTED",p["reason"])
                return
            started, started_cpu = self.clock(), time.process_time()
            options=({'traffic_policy':self.traffic_policy} if self.traffic_policy!='legacy' else {})
            if record.get('selection_binding') is not None: options['selection_binding']=record['selection_binding']
            evaluation, token = port.prepare_latency_submission(p["proposal"],timing,**options)
            duration = self.clock()-started
            if duration < 0 or not math.isfinite(duration): raise ValueError("invalid preparation duration")
            preparation_start = port.time+self._handoff_service
            p.update(stage="PREPARING",due=preparation_start+duration,
                     token=token,reason=None if evaluation.accepted else evaluation.reason)
            self._handoff_service = 0.
            record.update(prepare_work_s=duration,prepare_work_cpu_s=time.process_time()-started_cpu,
                          preparation_started_time=preparation_start,
                          preparation_snapshot_time=port.time,
                          status="PREPARING")
            port.planning_latency_event("normal_preparation_started",**record)
        elif p["stage"] == "PREPARING":
            if p["reason"]:
                self._finish(port,"REJECTED",p["reason"])
                return
            started, started_cpu = self.clock(), time.process_time()
            guarded = port.latency_packet_guard(p["token"]) if p["token"] is not None else None
            guard_work = self.clock()-started
            if guard_work < 0 or not math.isfinite(guard_work): raise ValueError("invalid ready guard duration")
            record["ready_guard_work_s"] = guard_work
            record["prepare_work_s"] += guard_work
            record["prepare_work_cpu_s"] += time.process_time()-started_cpu
            if (self.traffic_policy == LOCAL_TRAFFIC_POLICY and guarded is not None
                    and guarded.details.get("affected_aircraft_ids")):
                record.setdefault("guard_local_recovery", []).append(dict(guarded.details,
                    observed_time=port.time,accepted=guarded.accepted,reason=guarded.reason,wall_s=guard_work))
            if guarded is not None and guarded.reason=="TRAFFIC_REVALIDATION_REQUIRED":
                self._begin_revalidation(port,extra_delay=guard_work)
                return
            ready = port.time+self._handoff_service+guard_work
            self._handoff_service = 0.
            p.update(stage="READY_SERVICE",due=ready,
                     reason=guarded.reason if guarded is not None and not guarded.accepted and guarded.reason!="TRAFFIC_REVALIDATION_REQUIRED" else None)
            record["ready_time"] = ready
        elif p["stage"] == "READY_SERVICE":
            completed_index=p.pop("completion_revalidation_index",None)
            completed_row=None if completed_index is None else record["revalidations"][completed_index]
            if completed_row is not None:
                completed_row.update(completed_time=port.time,status=p.pop("completion_revalidation_status"))
                if completed_row["status"]=="SUPERSEDED":
                    self._begin_revalidation(port)
                    return
            if p["reason"]:
                self._finish(port,"REJECTED",p["reason"])
                return
            bookkeeping, bookkeeping_cpu = time.perf_counter(), time.process_time()
            handoff = self._handoff_service
            guard=(port.complete_latency_revalidation(p["token"],p["revalidation_attempt"],promote=True)
                   if completed_row is not None and p["revalidation_attempt"] is not None else None)
            if guard is None or guard.accepted:
                guard = port.mark_latency_submission_ready(p["token"],handoff+time.perf_counter()-bookkeeping)
            registration = time.perf_counter()-bookkeeping
            registration_cpu = time.process_time()-bookkeeping_cpu
            record["ready_registration_bookkeeping_s"] = record.get("ready_registration_bookkeeping_s",0.)+registration
            record["ready_registration_bookkeeping_cpu_s"] = record.get("ready_registration_bookkeeping_cpu_s",0.)+registration_cpu
            if not guard.accepted:
                if completed_row is not None:
                    completed_row.update(status="SUPERSEDED" if guard.reason=="TRAFFIC_REVALIDATION_REQUIRED" else "REJECTED",reason=guard.reason)
                if guard.reason=="TRAFFIC_REVALIDATION_REQUIRED":
                    self._handoff_service=handoff
                    self._begin_revalidation(port,extra_delay=registration)
                    return
                # _finish carries the already-recorded handoff into the CPU
                # queue; only the new registration work is recorded again.
                self._finish(port,"REJECTED",guard.reason)
                self.online_service(port.time,registration,registration_cpu,"ready_registration")
            else:
                self._handoff_service = 0.
                p.update(stage="PREPARED",due=timing.effective_time)
                record.update(status="PREPARED",commit_status="DEFERRED",ready_time=guard.details["ready_at"])
                if self.traffic_policy == LOCAL_TRAFFIC_POLICY:
                    self._sync_local_proposal(port)
                    if p.pop("repair_requested", False):
                        record["standby_ready_time"] = guard.details["ready_at"]
                        record["standby_trajectory_hashes"] = [plan.trajectory_hash for plan in p["proposal"]]
                        p.update(stage="REPAIR_START",due=guard.details["ready_at"])
                self.online_service(port.time,registration,registration_cpu,"ready_registration")
                self._service_until=max(self._service_until,guard.details["ready_at"])
                port.planning_latency_event("normal_packet_prepared",**record)
        elif p["stage"]=="REVALIDATING":
            row=record["revalidations"][-1]
            started,started_cpu=self.clock(),time.process_time()
            if p["revalidation_accepted"]:
                result=(port.complete_latency_revalidation(p["token"],p["revalidation_attempt"])
                        if p["revalidation_attempt"] is not None else port.latency_packet_guard(p["token"]))
            else: result=ValidationResult(False,"REJECTED",p["revalidation_reason"])
            guard_work=self.clock()-started; guard_cpu=time.process_time()-started_cpu
            if guard_work<0 or not math.isfinite(guard_work): raise ValueError("invalid revalidation guard duration")
            record["revalidation_work_s"]+=guard_work; record["revalidation_work_cpu_s"]+=guard_cpu
            row["wall_s"]+=guard_work; row["cpu_s"]+=guard_cpu
            if result.status=="WAIT":
                p["due"]=max(result.details["due"],port.time+guard_work+self._handoff_service)
                row["due"]=p["due"]; self._handoff_service=0.; return
            row.update(geometry_completed_time=port.time,status="GUARDING")
            if result.reason: row["reason"]=result.reason
            due=port.time+guard_work+self._handoff_service
            row["guard_delivery_due"]=due
            p.update(stage="READY_SERVICE",due=due,completion_revalidation_index=len(record["revalidations"])-1,
                completion_revalidation_status="CERTIFIED" if result.accepted else
                    "SUPERSEDED" if result.reason=="TRAFFIC_REVALIDATION_REQUIRED" else "REJECTED",
                reason=None if result.accepted else result.reason)
            self._handoff_service=0.
        elif p["stage"] == "REPAIR_START":
            budget = self._revalidation_budget(port)
            if budget["available_work_s"] <= 0:
                p.update(stage="PREPARED", due=timing.effective_time)
                record["recovery_delivery"] = "standby"
                return
            started, cpu_started = self.clock(), time.process_time()
            result = port.repair_latency_submission(p["token"],getattr(self.callback,"provider",None),
                max_work_s=budget["available_work_s"])
            duration = max(self.clock()-started,result.details.get("minimum_work_s",0.))
            cpu = time.process_time()-cpu_started
            if duration < 0 or not math.isfinite(duration): raise ValueError("invalid repair duration")
            self.online_service(port.time,duration,cpu,"local_traffic_repair")
            due = port.time+budget["queued_service_s"]+duration
            self._service_until = max(self._service_until,due)
            self._handoff_service = 0.
            record.update(repair_work_s=duration,repair_work_cpu_s=cpu,repair=dict(result.details),
                repair_due=due,recovery_delivery=result.details.get("delivery","standby"))
            p.update(stage="REPAIRING",due=min(due,timing.effective_time),
                repair_delivery_due=due,
                repair_late=due>timing.ready_deadline,repair_result_accepted=result.accepted,
                repair_result_reason=result.reason)
        elif p["stage"] == "REPAIRING":
            started, cpu_started = time.perf_counter(), time.process_time()
            # Late repair delivery keeps the previously registered standby.
            # complete_repair rejects its candidate receipt without extending
            # the deadline or changing that original standby receipt.
            result = port.complete_latency_repair(p["token"],self._handoff_service)
            duration, cpu = time.perf_counter()-started,time.process_time()-cpu_started
            self._handoff_service = 0.
            if not result.accepted:
                if result.reason == "TRAFFIC_REVALIDATION_REQUIRED":
                    self._begin_revalidation(port,extra_delay=duration)
                else: self._finish(port,"REJECTED",result.reason)
                return
            self._sync_local_proposal(port)
            record.update(recovery_delivery="repaired" if result.status=="REPAIRED" else "standby",
                ready_time=result.details["ready_at"])
            p.update(stage="PREPARED",due=timing.effective_time)
            self.online_service(port.time,duration,cpu,"local_repair_promotion")
        elif p["stage"] == "PREPARED":
            bookkeeping, bookkeeping_cpu = time.perf_counter(), time.process_time()
            if self.traffic_policy == LOCAL_TRAFFIC_POLICY:
                record["local_recovery_events"] = list(port.latency_submission_recovery_log(p["token"]))
            result = port.publish_latency_submission(p["token"])
            publication_wall, publication_cpu = time.perf_counter()-bookkeeping,time.process_time()-bookkeeping_cpu
            record.update(publication_bookkeeping_wall_s=publication_wall,
                publication_bookkeeping_cpu_s=publication_cpu,
                publication_bookkeeping_in_delivery_delay=False)
            evidence_wall = evidence_cpu = None
            if result.accepted and p["evidence"]:
                started, started_cpu = time.perf_counter(), time.process_time()
                report = port.report_normal_evidence(p["evidence"],port.build_planning_snapshot().version)
                evidence_wall,evidence_cpu = time.perf_counter()-started,time.process_time()-started_cpu
                record["normal_evidence_status"] = report.status
            record.update(normal_evidence_wall_s=0. if evidence_wall is None else evidence_wall,
                          normal_evidence_cpu_s=0. if evidence_cpu is None else evidence_cpu)
            self._finish(port,"COMMITTED" if result.accepted else "REJECTED",result.reason,result)
            # Atomic event bookkeeping is measured, then occupies the subsequent
            # CPU service timeline. It is never backdated as pre-commit latency.
            self.online_service(port.time,publication_wall,publication_cpu,"publication_bookkeeping")
            if evidence_wall is not None:
                self.online_service(port.time,evidence_wall,evidence_cpu,"normal_evidence")

    def state_dict(self):
        return dict(interface=EFFECTIVE_STATE_VERSION,emergency_timing_mode=EMERGENCY_TIMING_MODE,
            delivery_policy=self.delivery_policy,
            dependency_protocol=DEPENDENCY_PROTOCOL_VERSION,
            lead_seconds=self.lead_seconds,preparation_margin=self.preparation_margin,
            search_handoff_reserve_seconds=self.search_handoff_reserve_seconds,
            callback=self.callback.state_dict(),pending=self.pending,dirty=self.dirty,
            records=self.records,online_services=self.online_services,
            idle_service=self._idle_service,handoff_service=self._handoff_service,
            service_until=self._service_until,restart_policy=self.restart_policy,traffic_policy=self.traffic_policy,
            restart_requested=self._restart_requested,last_normal_start_time=self._last_normal_start_time,
            last_restart_time=self._last_restart_time)

    def load_state_dict(self, state):
        if state.get('delivery_policy',self.delivery_policy)!=self.delivery_policy:
            raise ValueError('planning latency delivery policy binding mismatch')
        if (state["interface"]!=EFFECTIVE_STATE_VERSION
                or state.get("dependency_protocol")!=DEPENDENCY_PROTOCOL_VERSION
                or state["emergency_timing_mode"]!=EMERGENCY_TIMING_MODE
                or state["lead_seconds"]!=self.lead_seconds
                or state.get("search_handoff_reserve_seconds",0.)!=self.search_handoff_reserve_seconds
                or state["preparation_margin"]!=self.preparation_margin
                or state.get("restart_policy","legacy")!=self.restart_policy):
            raise ValueError("planning latency interface binding mismatch")
        if state.get("traffic_policy","legacy") != self.traffic_policy:
            raise ValueError("planning latency traffic policy binding mismatch")
        self.callback.load_state_dict(state["callback"])
        self.pending,self.dirty = state["pending"],state["dirty"]
        self.records,self.online_services = state["records"],state["online_services"]
        self._idle_service,self._handoff_service = state["idle_service"],state["handoff_service"]
        self._service_until = state["service_until"]
        self._restart_requested = state.get("restart_requested")
        self._last_normal_start_time = state.get("last_normal_start_time")
        self._last_restart_time = state.get("last_restart_time")
