"""World-owned certificates for one absolute-time normal first action.

The executor uses the existing soft/calendar machinery.  These records bind
that machinery to the issued delivery transaction and its joint continuation
certificate; they are not another scheduler or a translated prediction chain.
"""
from dataclasses import replace
from hashlib import sha256
import math
import time

from .config import freeze
from .models import ValidationResult


def reservation_identity(plan, effective_time):
    return sha256(repr(("normal_first_action_v1",effective_time,plan.trajectory_hash,plan)).encode()).hexdigest()


def delivery_units(plans, timing, snapshot=None):
    return tuple(dict(reservation_id=reservation_identity(p,timing.effective_time),
        aircraft_id=p.aircraft_id,trajectory_hash=p.trajectory_hash,
        effective_time=timing.effective_time,wait_start=timing.effective_time,
        wait_end=p.start_time,hold_trajectory_hash=(None if snapshot is None else
            snapshot.aircraft[p.aircraft_id].trajectory_hash),
        start_time=p.start_time,end_time=p.end_time,
        certificate_interval=(timing.effective_time,p.end_time),
        state="PENDING_DELIVERY") for p in plans)


def discard_pending(world, token, reason, *, superseded=False):
    packet=world.__dict__.get('_planning_latency_packets',{}).pop(token,None)
    if packet is None: return None
    for unit in packet.get('delivery_units',()):
        world._event('reservation_pending_superseded' if superseded else 'reservation_cancelled',
            unit['aircraft_id'],reservation_id=unit['reservation_id'],trajectory_hash=unit['trajectory_hash'],
            start_time=unit['start_time'],end_time=unit['end_time'],
            state='SUPERSEDED_PENDING' if superseded else 'CANCELLED',previous_state='PENDING_DELIVERY',
            reason=reason,published=False)
    return packet


def _snapshot(world):
    s=world.build_planning_snapshot()
    # Stage2Codec supports frozen Sections, not MappingProxyType. Keep the
    # authoritative certificate detached from the public snapshot wrappers.
    return replace(s,aircraft=freeze(dict(s.aircraft)),active=freeze(dict(s.active)),
                   soft=freeze(dict(s.soft)),requests=freeze(dict(s.requests)))


def register_published(world, packet, token):
    records=world.__dict__.setdefault("_future_reservations",{})
    units=packet.get("delivery_units") or delivery_units(packet["plans"],packet["timing"],packet["projected"])
    for p,unit in zip(packet["plans"],units):
        replaced=[]
        for old in records.values():
            if old["aircraft_id"]==p.aircraft_id and old["state"]=="RESERVED" and old["reservation_id"]!=unit['reservation_id']:
                replaced.append(old['reservation_id'])
                old['replacement_reservation_id']=unit['reservation_id']
                transition(world,old,"CANCELLED","atomic_replacement")
        record=dict(unit,packet_token=token,plan=p,certificate_valid=True,
            config_hash=world.config.config_hash,
            lead_seconds=packet['timing'].effective_time-packet['timing'].snapshot_time,
            dependency_hashes=packet["starts"][p.aircraft_id][1],
            pair_certificates=dict(packet["pair_certificates"]),
            certificate_horizon=packet["pair_horizon"],
            certificate_snapshot=None,
            traffic_signature=None,replaced_reservation_ids=replaced,
            selection_binding=packet.get("selection_binding"),
            state="RESERVED" if p.start_time>world.time+1.e-8 else "EXECUTING")
        records[unit["reservation_id"]]=record
        world._event("reservation_published",p.aircraft_id,reservation_id=unit["reservation_id"],
            trajectory_hash=p.trajectory_hash,effective_time=unit["effective_time"],
            start_time=p.start_time,end_time=p.end_time,state=record["state"],
            certificate_interval=record["certificate_interval"],replaced_reservation_ids=replaced,
            selection_binding=record.get('selection_binding'))
        if record['state']=='EXECUTING':
            transition(world,record,'EXECUTING')
    source=_snapshot(world)
    for unit in units:
        records[unit["reservation_id"]]["certificate_snapshot"]=source
        records[unit["reservation_id"]]["traffic_signature"]=traffic_signature(world)
    return units


def record_for_plan(world, plan, *, include_terminal=False):
    matching=[r for r in world.__dict__.get("_future_reservations",{}).values()
        if r["trajectory_hash"]==plan.trajectory_hash and r["aircraft_id"]==plan.aircraft_id]
    live=next((r for r in reversed(matching) if r['state'] in ('RESERVED','EXECUTING')),None)
    return live if live is not None else (matching[-1] if include_terminal and matching else None)


def transition(world, record, state, reason=""):
    record.update(state=state,last_transition_time=world.time)
    if state in ('CANCELLED','COMPLETED'):
        record.update(certificate_valid=False,certificate_snapshot=None,pair_certificates={},traffic_signature=None,
            terminal_reason=reason)
    world._event({"EXECUTING":"reservation_started","COMPLETED":"reservation_completed",
        "CANCELLED":"reservation_cancelled"}[state],record["aircraft_id"],
        reservation_id=record["reservation_id"],trajectory_hash=record["trajectory_hash"],
        start_time=record["start_time"],end_time=record["end_time"],reason=reason,state=state,
        replacement_reservation_id=record.get('replacement_reservation_id'))


def guard_start(world, plan, record, *, at_start=True):
    """Reuse known continuations; certify only changed space-time interactions.

    Waiting and action energy were certified jointly at delivery. The regular
    executor checks remaining action energy again immediately after this guard.
    Every affected geometry is checked against the entire retained soft set.
    """
    from .planning_latency import _motion, _physical_key, _emergency_key
    from .world import AIRBORNE, _planning_context
    from .local_traffic_recovery import envelope_safe
    from .safety import validate_separation
    started=time.perf_counter(); cpu=time.process_time()
    details=dict(reservation_id=record["reservation_id"],pairs_checked=0,reused_pairs=0,envelope_proofs=0)
    reason=""
    if (not record.get("certificate_valid") or record["state"]!="RESERVED"
            or record["plan"]!=plan or record["reservation_id"]!=reservation_identity(plan,record["effective_time"])
            or record["config_hash"]!=world.config.config_hash):
        reason="RESERVATION_CERTIFICATE_INVALID"
    if at_start and abs(world.time-plan.start_time)>1.e-8: reason="RESERVATION_START_TIME_MISMATCH"
    if reason:
        details.update(wall_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu)
        return ValidationResult(False,"REJECTED",reason,details)
    source=record["certificate_snapshot"]
    context=_planning_context(world.build_planning_snapshot())
    own=world.aircraft[plan.aircraft_id]
    expected,_,_,_=_motion(source,source.aircraft[own.id],world.time)
    # _motion regards an exact start as active; the executor has not performed
    # that transition yet. Compare the certified source identity separately.
    if (own.version!=plan.source_version or own.ring!=plan.source_ring or own.slot!=plan.source_slot
            or own.id in world.active or _emergency_key(own)!=_emergency_key(source.aircraft[own.id])):
        reason="RESERVATION_SOURCE_CHANGED"
    if any(math.dist(getattr(own,k),getattr(expected,k))>tol
        for k,tol in (("p",1.e-6),("v",1.e-7),("a",1.e-7))):
        reason="RESERVATION_CONTINUATION_CHANGED"
    first=context._traffic(own)
    if not reason:
        from .resources import ResourceConflict
        try: context._calendar_stage(dict(world.soft),())
        except (ResourceConflict,ValueError,KeyError) as exc:
            reason='RESERVATION_RESOURCE_CERTIFICATE_INVALID'
            details['resource_error']=str(exc)
    if not reason:
        from .planning_latency import _capacity
        current=world.build_planning_snapshot()
        _,_,_,portions=_motion(current,own,plan.end_time)
        estimates=[context._energy(p,s,e,fine=True) for p,s,e in portions]
        energy=context._sum_energy(estimates)
        lower=own.soc-energy.soc_used-energy.error_j/_capacity(world.config)
        upper=own.peak_seconds+energy.peak_seconds+energy.peak_error_s
        details['remaining_energy']=dict(soc_lower=lower,peak_upper=upper,
            interval=(world.time,plan.end_time),precision='fine')
        if not energy.valid or lower<=0 or upper>120. or (plan.kind=='FINAL_NORMAL' and lower<.25):
            reason='RESERVATION_ENERGY_BOUND_CHANGED'
    if not reason:
        for aid,other in sorted(world.aircraft.items()):
            if aid==own.id or other.status not in AIRBORNE|{"LANDED_SERVICE"}: continue
            previous=source.aircraft.get(aid)
            known=False
            if previous is not None:
                projected,active,soft,_=_motion(source,previous,world.time)
                known=(_physical_key(other)==_physical_key(projected)
                    and world.active.get(aid)==active and world.soft.get(aid)==soft
                    and _emergency_key(other)==_emergency_key(previous))
            second=context._traffic(other)
            horizon=max([record['certificate_horizon'],
                *(p.end_time+2*math.pi/.035 for p in (*world.active.values(),*world.soft.values()))])
            end=min(horizon,first.end_time,second.end_time)
            if end<=world.time: continue
            old=record['pair_certificates'].get(tuple(sorted((own.id,aid))))
            if known and old is not None and old[2]<=world.time and end<=old[3]:
                details["reused_pairs"]+=1; continue
            if envelope_safe(first,second,world.time,end,world.config):
                details["envelope_proofs"]+=1; continue
            result=validate_separation(first,second,world.time,end,config=world.config)
            details["pairs_checked"]+=1
            if not result.accepted:
                reason="RESERVATION_TRAFFIC_CERTIFICATE_INVALID"
                details.update(other_aircraft_id=aid,validation=dict(result.details)); break
    # A release identity is protected while either a future appointment or an
    # executing consumer uses it. Re-check the current resource transaction.
    if not reason:
        supports=world._release_supports(plan.target_ring,plan.target_slot,own.id,world.slots,world.soft) if plan.target_ring else set()
        if not set(supports).issubset(set(record["dependency_hashes"])):
            reason="RESERVATION_RELEASE_DEPENDENCY_CHANGED"
        witness=world._future_dependency_witness(set(record["dependency_hashes"]),world.soft,world.slots)
        if witness is not None:
            reason="FUTURE_EMERGENCY_DISPOSITION";details["witness"]=witness
    details.update(wall_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu)
    world._event("reservation_start_guard" if at_start else "reservation_traffic_guard",own.id,trajectory_hash=plan.trajectory_hash,
        accepted=not reason,reason=reason,**details)
    if not reason and not at_start:
        from .planning_latency import _pair_geometry
        horizon,pairs=_pair_geometry(context,context.soft,{own.id})
        record.update(certificate_snapshot=_snapshot(world),traffic_signature=traffic_signature(world),
            pair_certificates={key:row[:4] for key,row in pairs.items()},certificate_horizon=horizon)
    return ValidationResult(not reason,"GUARDED" if not reason else "REJECTED",reason,details)


def cancel_invalid(world, plan, reason):
    """Certify the entire release-dependent fallback before changing anything."""
    from .world import _planning_context, EmergencyDependencyConflict
    records=world.__dict__.get("_future_reservations",{})
    removed={plan.trajectory_hash}
    while True:
        extra={h for h,deps in world.dependencies.items() if removed.intersection(deps)}
        if extra.issubset(removed): break
        removed.update(extra)
    if any(p.trajectory_hash in removed for p in world.active.values()):
        raise EmergencyDependencyConflict(plan.aircraft_id,"RESERVATION_FALLBACK_ACTIVE_DEPENDENT",dict(reason=reason))
    ids={aid for aid,p in world.soft.items() if p.trajectory_hash in removed}
    context=_planning_context(world.build_planning_snapshot())
    context.dependencies={h:d for h,d in context.dependencies.items() if h not in removed}
    prepared=context._prepare_submission((),cancel=tuple(sorted(ids)))
    if isinstance(prepared,ValidationResult):
        raise EmergencyDependencyConflict(plan.aircraft_id,"RESERVATION_FALLBACK_UNCERTIFIED",
            dict(reason=reason,fallback_reason=prepared.reason,details=dict(prepared.details)))
    from .local_traffic_recovery import _fallback_energy
    from .planning_latency import PlanningTiming
    fallback=replace(world.build_planning_snapshot(),aircraft=freeze(dict(context.aircraft)),
        energy_ledger=freeze(context._energy_state()),slots=prepared[1].reservations,
        pads=prepared[2].reservations,dependencies=freeze(dict(context.dependencies)))
    # The original lead is preserved as the finite next-adjustment horizon.
    own_record=record_for_plan(world,plan,include_terminal=True)
    lead=(20. if own_record is None else own_record.get('lead_seconds',20.))
    fallback=replace(fallback,soft=freeze(dict(prepared[0])),planning_timing=
        PlanningTiming(world.time-lead,world.time,world.time-min(1.,lead/2.)))
    energy,rows=_fallback_energy(fallback,ids,deadline=math.inf)
    if not energy.accepted:
        raise EmergencyDependencyConflict(plan.aircraft_id,"RESERVATION_FALLBACK_UNCERTIFIED",
            dict(reason=reason,fallback_reason=energy.reason,fallback_energy=rows))
    # No invalid original action is executed when a hold fallback is unknown.
    world.dependencies={h:d for h,d in world.dependencies.items() if h not in removed}
    world._commit_submission((),prepared,cancellation_reason=reason)
    for record in records.values():
        if record["trajectory_hash"] in removed and record["state"]=="RESERVED":
            transition(world,record,"CANCELLED",reason)
    return ValidationResult(True,"CANCELLED",reason,dict(aircraft_ids=sorted(ids)))


def traffic_signature(world):
    from .planning_latency import _physical_key, _emergency_key, _capacity
    return (world.config.config_hash,tuple((aid,_physical_key(a),_emergency_key(a),
        round(a.soc+a.energy_j/_capacity(world.config),9))
        for aid,a in sorted(world.aircraft.items())),tuple(sorted(world.active.items())),
        tuple(sorted(world.soft.items())),tuple(sorted(world.dependencies.items())),
        tuple(sorted(world._source_clear.items())),world.slots.reservations,world.pads.reservations)


def maintain_reservations(world, latency=None):
    """Cheap delta gate before physical advance; no full check per physics tick."""
    records=[r for r in world.__dict__.get('_future_reservations',{}).values() if r['state']=='RESERVED']
    if not records: return
    signature=traffic_signature(world)
    for record in records:
        if record.get('traffic_signature')==signature: continue
        plan=record['plan']
        if world.soft.get(plan.aircraft_id)!=plan: continue
        started=time.perf_counter();cpu=time.process_time()
        result=guard_start(world,plan,record,at_start=False)
        if not result.accepted: cancel_invalid(world,plan,result.reason)
        if latency is not None:
            latency.online_service(world.time,time.perf_counter()-started,time.process_time()-cpu,
                'reservation_traffic_revalidation')
