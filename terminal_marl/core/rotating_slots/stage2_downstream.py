"""Deterministic downstream service approximation, never an executor resource.

The virtual pad and queue order are frozen per dispatch. Real selected finals
replace their virtual job, so every other job sees their actual pad occupancy.
Integer decijoules and deciseconds are shared by Python and CP-SAT; original-unit
diagnostics are retained separately. No trajectory generation occurs here.
"""
from dataclasses import dataclass
import math

from .config import service_duration
from .energy import stable_power
from .rescue_rules import normal_energy_lower_bound

TICKS_PER_SECOND = 10
ENERGY_UNIT_J = .1
UNKNOWN_RISK_J = 36000  # 1 Wh of explicit UNKNOWN penalty, in decijoules
DESCENT_SECONDS = 13.5
QUEUE_ASSUMPTION = "fixed_virtual_pad_and_slack_release_id_order_earliest_calendar_gap"
LEGACY_EVALUATION_PROFILE = "legacy_window_tail_v8"
STITCHED_EVALUATION_PROFILE = "stitched_continuation_v1"
EVALUATION_PROFILES = (LEGACY_EVALUATION_PROFILE, STITCHED_EVALUATION_PROFILE)


def validate_evaluation_profile(profile):
    if profile not in EVALUATION_PROFILES: raise ValueError("unknown evaluation profile")
    return profile


def ticks(value): return int(round(value * TICKS_PER_SECOND))
def joules(value): return int(round(value / ENERGY_UNIT_J))


def earliest_gap(calendar, ready, duration):
    start = ready
    for lo, hi in sorted(calendar):
        if start + duration <= lo: break
        if start < hi and lo < start + duration: start = hi
    return start


@dataclass(frozen=True)
class Demand:
    aircraft_id: str
    full: bool
    ready: int
    duration: int
    base_energy: int
    hold_per_tick: int
    reserve_energy: int
    unknown: int
    failure: int
    landed: int
    touchdown: int
    ring: str | None
    diagnostics: object
    real_intervals: tuple = ()


def describe_column(snapshot, column, references, context, *, evaluation_profile=LEGACY_EVALUATION_PROFILE):
    """Use existing quadrature and cached motion reference, never simulate."""
    validate_evaluation_profile(evaluation_profile)
    if evaluation_profile == STITCHED_EVALUATION_PROFILE:
        return _describe_stitched(snapshot,column,references,context)
    from .stage2_planning import HORIZON, _node, remaining_normal_time_lower
    a = snapshot.aircraft[column.aircraft_id]
    node = _node(snapshot, a)
    plans = context.all_plans(a.id, column.plans)
    full = bool(plans and plans[-1].pad_id is not None)
    terminal = column.node or node
    ring = terminal.ring
    horizon = snapshot.time + HORIZON
    # qh was already integrated while materializing the column. Beyond-window
    # chain portions are disjoint, including complete chains crossing H.
    window = column.downstream.get("window_j",column.objective[4])
    beyond = context.energy_portions(tuple((p, max(horizon, p.start_time), p.end_time)
        for p in plans if p.end_time > horizon))[0]
    # Stable gaps after H are not trajectory portions.
    cursor = snapshot.time
    prior_ring = a.ring
    holds = {r: 0. for r in ("outer", "middle", "inner")}
    tail_known_holds = {r: 0. for r in ("outer", "middle", "inner")}
    for p in plans:
        if prior_ring is not None and p.start_time > cursor:
            holds[prior_ring] += max(0., min(horizon, p.start_time) - cursor)
            tail_known_holds[prior_ring] += max(0., p.start_time-max(cursor,horizon))
            beyond += stable_power(prior_ring, snapshot.config).battery_w * max(0., p.start_time-max(cursor,horizon))
        cursor, prior_ring = p.end_time, p.target_ring
    if not full and prior_ring is not None:
        holds[prior_ring] += max(0., horizon-cursor)
    status, remaining_j, remaining_s = "KNOWN", 0., 0.
    if not full:
        anchor = max(horizon, terminal.time)
        exact_key = (snapshot.config.config_hash, ring, terminal.slot, anchor, terminal.version, a.id)
        exact = references.cache.get(exact_key)
        candidates = [exact] if exact and exact.status == "KNOWN" else []
        candidates += [r for key, r in references.cache.items()
                       if len(key) >= 6 and key[-1] == a.id and r.status == "KNOWN" and r not in candidates]
        status = "UNKNOWN"
        for ref in candidates:
            # Match the actual remaining ring; discard earlier transfer legs.
            index = next((k for k,p in enumerate(ref.plans) if p.source_ring == ring), None)
            if index is None: continue
            suffix = ref.plans[index:]
            values = []
            for p in suffix:
                cached=references.motion_energy_index.get((snapshot.config.config_hash,p.trajectory_hash))
                if cached is None: break
                n,q = cached
                values.append(max(0., q[0]-stable_power(n.ring,snapshot.config).battery_w*max(0.,p.start_time-n.time)))
            if len(values) != len(suffix): continue
            remaining_j = math.fsum(values)
            remaining_s = math.fsum(p.end_time-p.start_time for p in suffix)
            status = "KNOWN" if ref is exact else "ESTIMATED"
            break
        if status == "UNKNOWN":
            capacity = snapshot.config.energy.equivalent_voltage_V*snapshot.config.energy.reference_capacity_Ah*3600.
            remaining_j = max(1.,normal_energy_lower_bound(snapshot.config,ring)*capacity)
            remaining_s = remaining_normal_time_lower(snapshot,ring)
        ready = ticks(anchor + max(0., remaining_s-DESCENT_SECONDS))
        touchdown = ready + ticks(DESCENT_SECONDS)
        hold_rate = joules(stable_power(ring,snapshot.config).battery_w/TICKS_PER_SECOND)
    else:
        ready = ticks(plans[-1].end_time-DESCENT_SECONDS)
        touchdown = ticks(plans[-1].end_time)
        hold_rate = 0
    capacity = snapshot.config.energy.equivalent_voltage_V*snapshot.config.energy.reference_capacity_Ah*3600.
    real = tuple((r.resource[1], ticks(r.start), ticks(r.end), r.aircraft_id)
                 for r in column.resources if r.resource[0] == "pad")
    diagnostics = dict(reference_status=status, reference_estimate_kind=("normal_energy_and_time_lower_bound_proxy" if status=="UNKNOWN" else "cached_motion_suffix"),
        energy_estimate_bound=("remaining_motion_lower_bound_proxy" if status=="UNKNOWN" else "no_bound_claim" if status=="ESTIMATED" else "actual_cached_quadrature"),
        reference_unknown_flag=status=="UNKNOWN",uncertainty_risk_penalty_wh=1. if status!="KNOWN" else 0.,
        window_j=window, tail_motion_j=beyond+remaining_j, window_wait_seconds_by_ring=holds,
        tail_known_wait_seconds_by_ring=tail_known_holds,
        terminal_wait_ring=ring, estimated_touchdown=touchdown/TICKS_PER_SECOND,
        candidate_timing=("IMMEDIATE" if not column.plans or column.plans[0].start_time<=node.time+1.e-8 else "DELAYED"),
        delay_reason=("resource_release_or_virtual_downstream_window" if column.plans and column.plans[0].start_time>node.time+1.e-8 else "current_epoch"),
        joint_queue_assumption=QUEUE_ASSUMPTION)
    return Demand(a.id,full,ready,ticks(DESCENT_SECONDS+service_duration(snapshot.config,a.service_class)),
        joules(window+beyond+remaining_j), hold_rate, joules((a.soc-.25)*capacity), int(status!="KNOWN"),
        int(column.objective[0]), int(column.objective[1]),touchdown,ring,diagnostics,real)


def _motion_suffix(snapshot, aircraft_id, terminal, anchor, references):
    """Read cached motion evidence, never search or translate a trajectory."""
    from .stage2_planning import remaining_normal_time_lower
    ring, slot, version = terminal.ring, terminal.slot, terminal.version
    exact_key=(snapshot.config.config_hash,ring,slot,anchor,version,aircraft_id)
    cache=getattr(references,"cache",{}); index=getattr(references,"motion_energy_index",{})
    exact=cache.get(exact_key)
    candidates=([exact] if exact is not None else [])+[
        ref for key,ref in cache.items() if len(key)>=6 and key[0]==snapshot.config.config_hash
        and key[-1]==aircraft_id and ref is not exact]
    for ref in candidates:
        if ref.status!="KNOWN" or not ref.plans or ref.plans[-1].pad_id is None: continue
        start=next((i for i,p in enumerate(ref.plans)
            if (p.source_ring,p.source_slot,p.source_version)==(ring,slot,version)),None)
        if start is None: continue
        suffix=ref.plans[start:]; motion=[]; current=(ring,slot,version); previous=anchor; exact_epochs=ref is exact
        valid=True
        for p in suffix:
            cached=index.get((snapshot.config.config_hash,p.trajectory_hash))
            if (p.source_ring,p.source_slot,p.source_version)!=current or cached is None:
                valid=False; break
            n,q=cached
            if (n.ring,n.slot,n.version)!=current or not all(math.isfinite(x) for x in q):
                valid=False; break
            hold=stable_power(n.ring,snapshot.config).battery_w*max(0.,p.start_time-n.time)
            motion.append(max(0.,q[0]-hold))
            exact_epochs=exact_epochs and abs(p.start_time-previous)<=1.e-8
            previous=p.end_time
            if p.target_ring is not None: current=(p.target_ring,p.target_slot,p.source_version+1)
        if not valid: continue
        return ("KNOWN" if exact_epochs else "ESTIMATED",math.fsum(motion),
            math.fsum(p.end_time-p.start_time for p in suffix),"cached_motion_suffix",suffix)
    capacity=snapshot.config.energy.equivalent_voltage_V*snapshot.config.energy.reference_capacity_Ah*3600.
    return "UNKNOWN",max(1.,normal_energy_lower_bound(snapshot.config,ring)*capacity),remaining_normal_time_lower(snapshot,ring),"normal_energy_and_time_lower_bound_proxy",()


def _describe_stitched(snapshot,column,references,context):
    from .stage2_planning import HORIZON, _node
    from .trajectory import make_stable_plan
    a=snapshot.aircraft[column.aircraft_id]
    plans=context.all_plans(a.id,column.plans)
    full=bool(plans and plans[-1].pad_id is not None) or a.status=="LANDED_SERVICE"
    portions=[]; cursor=snapshot.time; ring,slot=a.ring,a.slot; explicit_holds={r:0. for r in ("outer","middle","inner")}
    for p in plans:
        if p.end_time<=cursor: continue
        if p.start_time>cursor and ring is not None:
            hold=make_stable_plan(a.id,ring,slot,cursor,p.start_time,config=snapshot.config)
            portions.append((hold,cursor,p.start_time)); explicit_holds[ring]+=p.start_time-cursor
        portions.append((p,max(cursor,p.start_time),p.end_time))
        cursor=p.end_time; ring,slot=p.target_ring,p.target_slot
    q=context.energy_portions(tuple(portions)) if portions else (0.,0.,0.)
    if not all(math.isfinite(x) for x in q): raise ValueError("stitched explicit energy unknown")
    explicit=q[0]; explicit_end=cursor; peak_failure=int(a.peak_seconds+q[2]>120.)
    timing=getattr(snapshot,"planning_timing",None)
    lead=timing.effective_time-timing.snapshot_time if timing is not None else 20.
    anchor=explicit_end; adjustment_s=adjustment_j=motion_j=motion_s=0.; status="KNOWN"; source="explicit_final_or_ground"
    occupancy_unknown=False; suffix=()
    if not full:
        terminal=column.node or _node(snapshot,a)
        ring=terminal.ring
        anchor=max(explicit_end,snapshot.time+lead)
        adjustment_s=anchor-explicit_end
        adjustment_j=stable_power(ring,snapshot.config).battery_w*adjustment_s
        status,motion_j,motion_s,source,suffix=_motion_suffix(snapshot,a.id,terminal,anchor,references)
        for p in suffix:
            resource=(p.target_ring,p.target_slot) if p.target_ring is not None else p.pad_id
            reservations=snapshot.slots if p.target_ring is not None else snapshot.pads
            if any(r.resource==resource and r.aircraft_id!=a.id and not math.isfinite(r.end) for r in reservations):
                occupancy_unknown=True
        # Indefinite real pad occupancy cannot be represented as a finite queue
        # window; preserve uncertainty instead of silently certifying release.
        occupancy_unknown=occupancy_unknown or any(r.aircraft_id!=a.id and not math.isfinite(r.end) for r in snapshot.pads)
        if occupancy_unknown: status="UNKNOWN"
        ready=ticks(anchor+max(0.,motion_s-DESCENT_SECONDS)); touchdown=ready+ticks(DESCENT_SECONDS)
        hold_rate=joules(stable_power(ring,snapshot.config).battery_w/TICKS_PER_SECOND)
    else:
        touchdown=ticks(plans[-1].end_time if plans else a.touchdown_time if a.touchdown_time is not None else snapshot.time)
        ready=touchdown-ticks(DESCENT_SECONDS); hold_rate=0; ring=None
    capacity=snapshot.config.energy.equivalent_voltage_V*snapshot.config.energy.reference_capacity_Ah*3600.
    base_energy=joules(explicit+adjustment_j+motion_j)
    real=tuple((r.resource[1],ticks(r.start),ticks(r.end),r.aircraft_id) for r in column.resources
        if isinstance(r.resource,tuple) and r.resource[0]=="pad")
    diagnostics=dict(evaluation_profile=STITCHED_EVALUATION_PROFILE,reference_status=status,
        reference_estimate_kind=source,reference_unknown_flag=status=="UNKNOWN",uncertainty_risk_penalty_wh=1. if status!="KNOWN" else 0.,
        energy_estimate_bound="remaining_motion_lower_bound_proxy" if source.endswith("lower_bound_proxy") else "no_bound_claim" if status!="KNOWN" else "actual_cached_quadrature",
        window_j=explicit,tail_motion_j=adjustment_j+motion_j,window_wait_seconds_by_ring=explicit_holds,
        tail_known_wait_seconds_by_ring={},terminal_wait_ring=ring,estimated_touchdown=touchdown/TICKS_PER_SECOND,
        candidate_timing="IMMEDIATE" if not column.plans or abs(column.plans[0].start_time-snapshot.time)<=1.e-8 else "DELAYED",
        delay_reason="resource_release_or_virtual_downstream_window" if column.plans and column.plans[0].start_time>snapshot.time+1.e-8 else "current_epoch",
        joint_queue_assumption=QUEUE_ASSUMPTION,explicit_prefix_j=explicit,explicit_prefix_peak_seconds=q[2],
        explicit_prefix_peak_failure=peak_failure,available_energy_threshold=joules(max(0.,a.soc)*capacity),
        estimated_adjustment_wait_j=adjustment_j,estimated_adjustment_wait_s=adjustment_s,
        estimated_suffix_motion_j=motion_j,estimated_suffix_motion_s=motion_s,
        estimated_queue_wait_j=0.,estimated_queue_wait_s=0.,
        timeline_start=snapshot.time,timeline_explicit_end=explicit_end,timeline_anchor=anchor,
        timeline_unqueued_end=touchdown/TICKS_PER_SECOND,timeline_end=touchdown/TICKS_PER_SECOND,
        lead_seconds=lead,lead_source="planning_timing" if timing is not None else "nominal20_estimate",
        suffix_peak_status="UNKNOWN_NOT_A_SAFETY_CERTIFICATE" if not full else "EXPLICIT_ONLY",
        resource_occupancy_unknown=occupancy_unknown,executed_reservations_modified=False,
        scoring_estimate_not_safety_certificate=True,
        base_energy_quantization_j=base_energy*ENERGY_UNIT_J-(explicit+adjustment_j+motion_j))
    landed=-int(full and touchdown<=ticks(snapshot.time+HORIZON))
    return Demand(a.id,full,ready,ticks(DESCENT_SECONDS+service_duration(snapshot.config,a.service_class)),
        base_energy,hold_rate,joules((a.soc-.25)*capacity),int(status!="KNOWN"),peak_failure,landed,touchdown,ring,diagnostics,real)


class JointDownstream:
    """Shared service calendars and explicit optional CP queue intervals."""
    def __init__(self, demands, fixed_calendars, order, *, now=0, pad_count=6, fixed_demands=(), assignment=None, frame_id=None,
                 evaluation_profile=LEGACY_EVALUATION_PROFILE):
        self.evaluation_profile=validate_evaluation_profile(evaluation_profile)
        self.demands = demands
        self.fixed_demands = {d.aircraft_id:d for d in fixed_demands}
        if any(d.diagnostics.get("evaluation_profile",LEGACY_EVALUATION_PROFILE)!=self.evaluation_profile
                for d in (*self.demands.values(),*self.fixed_demands.values())):
            raise ValueError("downstream demand evaluation profile mismatch")
        self.order = tuple(order)
        self.now = now
        self.frame_id = frame_id
        self.calendars = {p:[] for p in range(pad_count)}
        self.fixed_identity = set()
        for p,lo,hi,aid in fixed_calendars:
            identity=(p,lo,hi,aid)
            if identity not in self.fixed_identity:
                self.calendars[p].append((lo,hi)); self.fixed_identity.add(identity)
        # One assignment based on the baseline reference readiness, then frozen.
        provisional = {p:list(rows) for p,rows in self.calendars.items()}
        self.assignment = dict(assignment or {})
        for aid in self.order:
            options = [d for key,d in demands.items() if d.aircraft_id==aid and not d.full]
            d = self.fixed_demands.get(aid) or (options[0] if options else None)
            if d is None: continue
            if assignment is not None:
                if not d.full and aid not in self.assignment: raise ValueError("frozen frame missing virtual pad")
                continue
            start,pad=min((earliest_gap(provisional[p],d.ready,d.duration),p) for p in provisional)
            self.assignment[aid]=pad
            provisional[pad].append((start,start+d.duration))

    def evaluate(self, selected):
        chosen = {c.aircraft_id:self.demands[c.signature] for c in selected}
        chosen.update(self.fixed_demands)
        calendar={p:list(rows) for p,rows in self.calendars.items()}
        for d in chosen.values():
            for p,lo,hi,aid in d.real_intervals:
                if (p,lo,hi,aid) not in self.fixed_identity: calendar[p].append((lo,hi))
        previous={}; rows={}; total=[0,0,0,0,0]
        for aid in self.order:
            d=chosen.get(aid)
            if d is None: continue
            wait=0; td=d.touchdown
            if not d.full:
                pad=self.assignment[aid]
                start=earliest_gap(calendar[pad],max(d.ready,previous.get(pad,d.ready)),d.duration)
                wait=start-d.ready; td=start+ticks(DESCENT_SECONDS)
                calendar[pad].append((start,start+d.duration)); previous[pad]=start+d.duration
            energy=d.base_energy+d.hold_per_tick*wait
            risk=max(0,energy-d.reserve_energy)+d.unknown*UNKNOWN_RISK_J
            failure=(max(int(d.diagnostics["explicit_prefix_peak_failure"]),int(energy>=d.diagnostics["available_energy_threshold"]))
                if self.evaluation_profile==STITCHED_EVALUATION_PROFILE else d.failure)
            for k,value in enumerate((failure,d.landed,risk,energy,td-self.now)): total[k]+=value
            diag=dict(d.diagnostics,estimated_touchdown=td/TICKS_PER_SECOND,
                tail_wait_seconds=wait/TICKS_PER_SECOND,tail_wait_j=d.hold_per_tick*wait*ENERGY_UNIT_J,
                total_j=energy*ENERGY_UNIT_J,normal_reserve_shortfall_j=max(0,energy-d.reserve_energy)*ENERGY_UNIT_J,unknown_risk_flag=bool(d.unknown),
                virtual_pad=None if d.full else self.assignment[aid])
            diag["uncertainty_risk_flag"]=bool(d.unknown)
            diag["tail_j"]=diag.get("tail_motion_j",0.)+diag["tail_wait_j"]
            wait_by_ring=dict(diag.get("window_wait_seconds_by_ring",{}))
            for ring,seconds in diag.get("tail_known_wait_seconds_by_ring",{}).items():
                wait_by_ring[ring]=wait_by_ring.get(ring,0.)+seconds
            if d.ring is not None: wait_by_ring[d.ring]=wait_by_ring.get(d.ring,0.)+wait/TICKS_PER_SECOND
            diag["estimated_wait_seconds_by_ring"]=wait_by_ring
            if self.evaluation_profile==STITCHED_EVALUATION_PROFILE:
                diag.update(estimated_queue_wait_s=wait/TICKS_PER_SECOND,estimated_queue_wait_j=diag["tail_wait_j"],
                    timeline_queue_start=d.ready/TICKS_PER_SECOND,timeline_queue_end=(d.ready+wait)/TICKS_PER_SECOND,
                    timeline_end=td/TICKS_PER_SECOND,estimated_failure=failure)
                if d.ring is not None:
                    diag["estimated_wait_seconds_by_ring"][d.ring]=diag["estimated_wait_seconds_by_ring"].get(d.ring,0.)+diag["estimated_adjustment_wait_s"]
                diag["component_sum_j"]=sum(diag[name] for name in ("explicit_prefix_j","estimated_adjustment_wait_j","estimated_suffix_motion_j","estimated_queue_wait_j"))
                diag["timeline_segments"]=[dict(kind="explicit_prefix",start=diag["timeline_start"],end=diag["timeline_explicit_end"],energy_j=diag["explicit_prefix_j"])]
                if not d.full:
                    ready=d.ready/TICKS_PER_SECOND; td_s=td/TICKS_PER_SECOND; seconds=diag["estimated_suffix_motion_s"]
                    before=max(0.,seconds-DESCENT_SECONDS); motion=diag["estimated_suffix_motion_j"]
                    pre_j=motion*before/seconds if seconds else 0.
                    diag["timeline_segments"] += [
                        dict(kind="estimated_adjustment_wait",start=diag["timeline_explicit_end"],end=diag["timeline_anchor"],energy_j=diag["estimated_adjustment_wait_j"]),
                        dict(kind="estimated_suffix_before_descent",start=diag["timeline_anchor"],end=ready,energy_j=pre_j),
                        dict(kind="estimated_queue_wait",start=ready,end=(d.ready+wait)/TICKS_PER_SECOND,energy_j=diag["estimated_queue_wait_j"]),
                        dict(kind="estimated_suffix_descent",start=(d.ready+wait)/TICKS_PER_SECOND,end=td_s,energy_j=motion-pre_j)]
            rows[aid]=diag
        return tuple(total),rows

    def key(self,selected):
        f,l,r,e,t=self.evaluate(selected)[0]
        return f,l,r*ENERGY_UNIT_J/3600.,e*ENERGY_UNIT_J,t/TICKS_PER_SECOND

    def cp_expressions(self,model,variables,lookup):
        """Column-dependent readiness, energy and wait rate share one queue."""
        all_demands=(*self.demands.values(),*self.fixed_demands.values())
        # Every job can be placed after all fixed/selected real intervals;
        # this finite tail bound covers that schedule without a physical gate.
        max_duration={aid:max((d.duration for d in all_demands if d.aircraft_id==aid),default=0) for aid in self.order}
        max_time=max([self.now]+[d.ready+d.duration for d in all_demands]
                     +[hi for rows in self.calendars.values() for _,hi in rows]
                     +[hi for d in all_demands for _,_,hi,_ in d.real_intervals])+sum(max_duration.values())+ticks(1.)
        intervals={p:[] for p in self.calendars}; terms=[[] for _ in range(5)]; virtual={}
        for p,rows in self.calendars.items():
            for n,(lo,hi) in enumerate(rows):
                intervals[p].append(model.NewIntervalVar(lo,hi-lo,hi,f"fixed_pad_{p}_{n}"))
        fixed_seen=set(self.fixed_identity)
        for d in self.fixed_demands.values() if self.frame_id is not None else ():
            for p,lo,hi,aid in d.real_intervals:
                identity=(p,lo,hi,aid)
                if identity not in fixed_seen:
                    intervals[p].append(model.NewIntervalVar(lo,hi-lo,hi,f"background_real_{aid}_{len(intervals[p])}"))
                    fixed_seen.add(identity)
        for key,c in lookup.items():
            d=self.demands[key]; v=variables[key]
            for p,lo,hi,aid in d.real_intervals:
                if (p,lo,hi,aid) not in self.fixed_identity:
                    intervals[p].append(model.NewOptionalIntervalVar(lo,hi-lo,hi,v,f"real_{aid}_{len(intervals[p])}"))
        for aid in self.order:
            options=[(key,d) for key,d in self.demands.items() if d.aircraft_id==aid and key in variables]
            fixed=self.fixed_demands.get(aid)
            if fixed: options=[(None,fixed)]
            active=[]; starts=[]; ends=[]
            for index,(key,d) in enumerate(options):
                v=variables[key] if key is not None else model.NewConstant(1)
                wait=0; td=d.touchdown*v
                if not d.full:
                    start=model.NewIntVar(0,max_time,f"virtual_start_{aid}_{index}")
                    end=model.NewIntVar(0,max_time,f"virtual_end_{aid}_{index}")
                    wait=model.NewIntVar(0,max_time,f"virtual_wait_{aid}_{index}")
                    model.Add(start>=d.ready).OnlyEnforceIf(v)
                    model.Add(wait==start-d.ready).OnlyEnforceIf(v)
                    model.Add(wait==0).OnlyEnforceIf(v.Not())
                    # Optional interval's absent start/end are canonical zero.
                    model.Add(start==0).OnlyEnforceIf(v.Not()); model.Add(end==0).OnlyEnforceIf(v.Not())
                    intervals[self.assignment[aid]].append(model.NewOptionalIntervalVar(start,d.duration,end,v,f"virtual_{aid}_{index}"))
                    active.append(v); starts.append(start); ends.append(end)
                    td=start+ticks(DESCENT_SECONDS)*v
                energy=d.base_energy*v+d.hold_per_tick*wait
                deficit=model.NewIntVar(0,max(0,d.base_energy-d.reserve_energy)+d.hold_per_tick*max_time,f"risk_{aid}_{index}")
                model.AddMaxEquality(deficit,[0,energy-d.reserve_energy*v])
                failure=d.failure*v
                if self.evaluation_profile==STITCHED_EVALUATION_PROFILE:
                    threshold=d.diagnostics["available_energy_threshold"]
                    depleted=model.NewBoolVar(f"energy_depleted_{aid}_{index}")
                    model.Add(energy>=threshold).OnlyEnforceIf([v,depleted])
                    model.Add(energy<=threshold-1).OnlyEnforceIf([v,depleted.Not()])
                    model.Add(depleted==0).OnlyEnforceIf(v.Not())
                    failure=v if d.diagnostics["explicit_prefix_peak_failure"] else depleted
                for k,term in enumerate((failure,d.landed*v,deficit+d.unknown*UNKNOWN_RISK_J*v,energy,td-self.now*v)): terms[k].append(term)
            if active:
                present=model.NewBoolVar(f"virtual_present_{aid}"); model.Add(present==sum(active))
                start=model.NewIntVar(0,max_time,f"chosen_virtual_start_{aid}"); model.Add(start==sum(starts))
                end=model.NewIntVar(0,max_time,f"chosen_virtual_end_{aid}"); model.Add(end==sum(ends))
                virtual[aid]=(present,start,end)
        # Pairwise constraints preserve order even if an intervening job is real.
        for p in intervals:
            jobs=[virtual[aid] for aid in self.order if self.assignment.get(aid)==p and aid in virtual]
            for i,(present,start,end) in enumerate(jobs):
                for later,lstart,lend in jobs[i+1:]: model.Add(end<=lstart).OnlyEnforceIf([present,later])
            model.AddNoOverlap(intervals[p])
        return tuple(sum(row) for row in terms)
