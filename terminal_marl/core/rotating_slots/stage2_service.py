"""Opt-in nominal completion and reachable service-opportunity scoring.

The queue is an estimate, never a reservation or a safety certificate. One
calibration per config uses three feasible canonical primitives and real energy
quadrature; background aircraft reuse those scalar quantities. Uncertainty is
reported separately from the predicted reserve deficit. All time arithmetic in
the queue, including its CP encoding, uses the existing decisecond grid.
"""
from dataclasses import replace
import math
import time

from .config import service_duration
from .energy import integrate_energy, stable_power
from .stage2_downstream import (Demand, SERVICE_EVALUATION_PROFILE, ticks, joules,
    earliest_gap, TICKS_PER_SECOND, ENERGY_UNIT_J, DESCENT_SECONDS)

_NOMINAL = {}
QUEUE_ASSUMPTION = "global_id_order_reachable_pad_earliest_gap_pad_index_tie"


def service_nominal_completion(config, ring):
    """Return (joules, seconds, metadata), estimated from canonical primitives.

    This is a point estimate calibrated with kinematically accepted trajectories,
    not a lower bound or evidence that a particular continuation is feasible.
    Calibration cost occurs synchronously inside the planning callback.
    """
    key=config.config_hash
    if key not in _NOMINAL:
        from .trajectory import generate_transfer, generate_final
        clock=time.perf_counter(); rows={}
        for source in ("outer","middle","inner"):
            result=(generate_final("nominal",source,0,config.pads.count-1,0.,config=config)
                if source=="inner" else generate_transfer("nominal",source,0,0,0.,config=config))
            if result.plan is None:
                raise ValueError("nominal service calibration primitive not found: "+source)
            plan=result.plan
            q=integrate_energy(plan,plan.start_time,plan.end_time,config=config)
            if not q.valid or not math.isfinite(q.joules):
                raise ValueError("nominal service calibration energy unknown: "+source)
            rows[source]=(q.joules,plan.end_time-plan.start_time,
                dict(source_ring=source,target_ring=plan.target_ring,target_slot=plan.target_slot,
                    pad_id=plan.pad_id,duration_s=plan.end_time-plan.start_time,
                    energy_j=q.joules,trajectory_hash=plan.trajectory_hash))
        elapsed=time.perf_counter()-clock
        if len(_NOMINAL)>=16: _NOMINAL.pop(next(iter(_NOMINAL)))
        _NOMINAL[key]=(rows,elapsed)
    rows,elapsed=_NOMINAL[key]
    sources={"outer":("outer","middle","inner"),"middle":("middle","inner"),"inner":("inner",)}[ring]
    return (math.fsum(rows[r][0] for r in sources),math.fsum(rows[r][1] for r in sources),
        dict(model="canonical_feasible_primitive_quadrature_v1",config_hash=key,
             calibration_wall_s=elapsed,primitives=tuple(rows[r][2] for r in sources),
             assumption="representative_motion_no_phase_or_resource_feasibility_claim"))


def describe_service_column(snapshot,column,references,context):
    from .stage2_planning import HORIZON, _node
    from .trajectory import make_stable_plan
    a=snapshot.aircraft[column.aircraft_id]; plans=context.all_plans(a.id,column.plans)
    full=bool(plans and plans[-1].pad_id is not None) or a.status=="LANDED_SERVICE"
    portions=[]; cursor=snapshot.time; ring,slot=a.ring,a.slot
    holds={r:0. for r in ("outer","middle","inner")}
    for p in plans:
        if p.end_time<=cursor: continue
        if p.start_time>cursor and ring is not None:
            hold=make_stable_plan(a.id,ring,slot,cursor,p.start_time,config=snapshot.config)
            portions.append((hold,cursor,p.start_time)); holds[ring]+=p.start_time-cursor
        portions.append((p,max(cursor,p.start_time),p.end_time))
        cursor=p.end_time; ring,slot=p.target_ring,p.target_slot
    q=context.energy_portions(tuple(portions)) if portions else (0.,0.,0.)
    if not all(math.isfinite(x) for x in q): raise ValueError("service explicit prefix energy unknown")
    timing=getattr(snapshot,"planning_timing",None)
    lead=timing.effective_time-timing.snapshot_time if timing is not None else 20.
    adjustment=motion_j=motion_s=0.; anchor=cursor; status="KNOWN"; source="explicit_final_or_ground"; calibration={}
    pads=tuple(range(snapshot.config.pads.count)); blocked=[]
    for pad in pads:
        if any((r.resource if isinstance(r.resource,int) else r.resource[1])==pad and r.aircraft_id!=a.id
               and not math.isfinite(r.end) for r in snapshot.pads): blocked.append(pad)
    reachable=tuple(p for p in pads if p not in blocked)
    blocked_chain=False
    if not full:
        terminal=column.node or _node(snapshot,a); ring=terminal.ring
        # A pending action ends the current commitment. KEEP waits only until
        # the same next adjustment epoch, never an artificial horizon hold.
        anchor=max(cursor,snapshot.time+lead); adjustment=anchor-cursor
        motion_j,motion_s,calibration=service_nominal_completion(snapshot.config,ring)
        status="ESTIMATED"; source="nominal_canonical_primitive_quadrature"
        # Stable slot reservations can be released by later joint decisions;
        # an occupied ring is not a proof of unreachable normal continuation.
        ready=ticks(anchor+max(0.,motion_s-DESCENT_SECONDS)); touchdown=ready+ticks(DESCENT_SECONDS)
        hold_rate=joules(stable_power(ring,snapshot.config).battery_w/TICKS_PER_SECOND)
    else:
        touchdown=ticks(plans[-1].end_time if plans else a.touchdown_time if a.touchdown_time is not None else snapshot.time)
        ready=touchdown-ticks(DESCENT_SECONDS); hold_rate=0; ring=None
    adjustment_j=0. if ring is None else stable_power(ring,snapshot.config).battery_w*adjustment
    capacity=snapshot.config.energy.equivalent_voltage_V*snapshot.config.energy.reference_capacity_Ah*3600.
    base=joules(q[0]+adjustment_j+motion_j)
    real=tuple((r.resource[1],ticks(r.start),ticks(r.end),r.aircraft_id) for r in column.resources
        if isinstance(r.resource,tuple) and r.resource[0]=="pad" and math.isfinite(r.end))
    peak=int(a.peak_seconds+q[2]>120.)
    diag=dict(evaluation_profile=SERVICE_EVALUATION_PROFILE,reference_status=status,
        reference_estimate_kind=source,reference_unknown_flag=status!="KNOWN",energy_estimate_bound="no_bound_claim",
        uncertainty_risk_penalty_wh=0.,continuation_uncertainty=status!="KNOWN",
        reachable_pads=reachable,unavailable_forecast_pads=tuple(blocked),known_blocked_pads=(),known_blocked_chain=blocked_chain,
        resource_occupancy_unknown=bool(blocked),
        reachability_status="EXPLICIT" if full else "NOMINAL_UNKNOWN_NOT_CERTIFIED",
        nominal_calibration=calibration,service_baseline_ready=ready-ticks(adjustment),
        service_opportunity_disabled=a.status=="LANDED_SERVICE",queue_pad_feasibility="NOMINAL_UNKNOWN_NOT_CERTIFIED" if not full else "EXPLICIT",
        explicit_prefix_j=q[0],explicit_prefix_peak_seconds=q[2],explicit_prefix_peak_failure=peak,
        available_energy_threshold=joules(max(0.,a.soc)*capacity),estimated_adjustment_wait_s=adjustment,
        estimated_adjustment_wait_j=adjustment_j,estimated_suffix_motion_j=motion_j,estimated_suffix_motion_s=motion_s,
        estimated_queue_wait_s=0.,estimated_queue_wait_j=0.,window_j=q[0],tail_motion_j=adjustment_j+motion_j,
        window_wait_seconds_by_ring=holds,tail_known_wait_seconds_by_ring={},terminal_wait_ring=ring,
        timeline_start=snapshot.time,timeline_explicit_end=cursor,timeline_anchor=anchor,
        timeline_unqueued_end=touchdown/TICKS_PER_SECOND,timeline_end=touchdown/TICKS_PER_SECOND,
        estimated_touchdown=touchdown/TICKS_PER_SECOND,lead_seconds=lead,
        lead_source="planning_timing" if timing is not None else "nominal20_estimate",
        joint_queue_assumption=QUEUE_ASSUMPTION,executed_reservations_modified=False,
        scoring_estimate_not_safety_certificate=True,suffix_peak_status="UNKNOWN_NOT_A_SAFETY_CERTIFICATE" if not full else "EXPLICIT_ONLY",
        base_energy_quantization_j=base*ENERGY_UNIT_J-(q[0]+adjustment_j+motion_j))
    return Demand(a.id,full,ready,ticks(DESCENT_SECONDS+service_duration(snapshot.config,a.service_class)),
        base,hold_rate,joules((a.soc-.25)*capacity),int(status!="KNOWN"),peak,
        -int(full and touchdown<=ticks(snapshot.time+HORIZON)),touchdown,ring,diag,real)


class ServiceOpportunityDownstream:
    """Identical deterministic greedy service allocation in Python and CP."""
    def __init__(self,demands,fixed_calendars,order,*,now=0,pad_count=6,fixed_demands=(),assignment=None,
                 frame_id=None,evaluation_profile=SERVICE_EVALUATION_PROFILE,baseline_demands=None):
        self.evaluation_profile=evaluation_profile; self.demands=demands
        self.fixed_demands={d.aircraft_id:d for d in fixed_demands}; self.order=tuple(order)
        self.now=now; self.horizon=now+ticks(240.); self.frame_id=frame_id
        self.calendars={p:[] for p in range(pad_count)}; self.fixed_identity=set()
        for p,lo,hi,aid in fixed_calendars:
            row=(p,lo,hi,aid)
            if row not in self.fixed_identity: self.calendars[p].append((lo,hi)); self.fixed_identity.add(row)
        for d in (*demands.values(),*self.fixed_demands.values()):
            if d.diagnostics.get("evaluation_profile")!=SERVICE_EVALUATION_PROFILE: raise ValueError("downstream demand evaluation profile mismatch")
        # assignment remains available for frame compatibility; never controls
        # this profile's dynamic pad choice.
        self.assignment={}
        baseline=list(baseline_demands) if baseline_demands is not None else list(self.fixed_demands.values())
        if baseline_demands is None:
            seen={d.aircraft_id for d in baseline}
            for d in demands.values():
                if d.aircraft_id not in seen: baseline.append(d); seen.add(d.aircraft_id)
        immediate={d.aircraft_id:replace(d,ready=int(d.diagnostics.get("service_baseline_ready",d.ready))) for d in baseline}
        self.baseline_demands=tuple(baseline)
        self.baseline_starts=self._allocate(immediate)[0]

    def _reachable(self,d):
        return tuple(p for p in d.diagnostics.get("reachable_pads",tuple(self.calendars)) if p in self.calendars)

    def _allocate(self,chosen):
        calendar={p:list(rows) for p,rows in self.calendars.items()}; seen=set(self.fixed_identity)
        for d in chosen.values():
            for p,lo,hi,aid in d.real_intervals:
                if (p,lo,hi,aid) not in seen: calendar[p].append((lo,hi)); seen.add((p,lo,hi,aid))
        previous={p:0 for p in calendar}; starts={}; pads={}
        for aid in self.order:
            d=chosen.get(aid)
            if d is None: continue
            if d.full: starts[aid]=max(self.now,d.ready); pads[aid]=None; continue
            reachable=self._reachable(d)
            if not reachable: starts[aid]=None; pads[aid]=None; continue
            start,pad=min((earliest_gap(calendar[p],max(d.ready,previous[p]),d.duration),p) for p in reachable)
            starts[aid]=start; pads[aid]=pad; previous[pad]=start+d.duration
        return starts,pads

    def _eligible_baseline(self,aid):
        start=self.baseline_starts.get(aid)
        baseline=next((d for d in self.baseline_demands if d.aircraft_id==aid),None)
        return start is not None and start<self.horizon and not (baseline and baseline.diagnostics.get("service_opportunity_disabled",False))

    def evaluate(self,selected):
        chosen={c.aircraft_id:self.demands[c.signature] for c in selected}; chosen.update(self.fixed_demands)
        starts,pads=self._allocate(chosen); total=[0]*6; rows={}
        for aid in self.order:
            d=chosen.get(aid)
            if d is None: continue
            start=starts[aid]; wait=0 if d.full or start is None else start-d.ready
            td=d.touchdown if d.full or start is None else start+ticks(DESCENT_SECONDS)
            energy=d.base_energy+d.hold_per_tick*wait; risk=max(0,energy-d.reserve_energy)
            failure=max(d.failure,int(energy>=d.diagnostics["available_energy_threshold"]))
            g=(max(0,min(self.horizon,self.horizon if start is None else start)-self.baseline_starts[aid])
                if self._eligible_baseline(aid) else 0)
            for k,value in enumerate((failure,d.landed,risk,g,energy,td-self.now)): total[k]+=value
            diag=dict(d.diagnostics,estimated_touchdown=td/TICKS_PER_SECOND,
                estimated_queue_wait_s=wait/TICKS_PER_SECOND,estimated_queue_wait_j=d.hold_per_tick*wait*ENERGY_UNIT_J,
                tail_wait_seconds=wait/TICKS_PER_SECOND,tail_wait_j=d.hold_per_tick*wait*ENERGY_UNIT_J,
                total_j=energy*ENERGY_UNIT_J,normal_reserve_shortfall_j=risk*ENERGY_UNIT_J,
                unknown_risk_flag=False,uncertainty_risk_flag=bool(d.unknown),virtual_pad=pads[aid],
                service_start=None if start is None else start/TICKS_PER_SECOND,
                service_baseline_start=None if self.baseline_starts.get(aid) is None else self.baseline_starts[aid]/TICKS_PER_SECOND,
                service_opportunity_delay_s=g/TICKS_PER_SECOND,service_opportunity_eligible=self._eligible_baseline(aid),
                service_opportunity_mask=(("NO_FORECAST_WINDOW_UNKNOWN_OCCUPANCY" if d.diagnostics.get("resource_occupancy_unknown") else "PROVEN_BLOCKED")
                    if start is None else "NO_WITHIN_WINDOW_BASELINE" if not self._eligible_baseline(aid) else "ELIGIBLE_NOMINAL"),
                timeline_queue_start=d.diagnostics.get("timeline_anchor",d.ready/TICKS_PER_SECOND),
                timeline_queue_end=d.diagnostics.get("timeline_anchor",d.ready/TICKS_PER_SECOND)+wait/TICKS_PER_SECOND,
                timeline_end=td/TICKS_PER_SECOND,estimated_failure=failure)
            waits=dict(diag.get("window_wait_seconds_by_ring",{}))
            if d.ring is not None: waits[d.ring]=waits.get(d.ring,0.)+diag.get("estimated_adjustment_wait_s",0.)+wait/TICKS_PER_SECOND
            diag["estimated_wait_seconds_by_ring"]=waits
            diag["component_sum_j"]=sum(diag.get(k,0.) for k in ("explicit_prefix_j","estimated_adjustment_wait_j","estimated_suffix_motion_j","estimated_queue_wait_j"))
            diag["tail_j"]=diag.get("tail_motion_j",0.)+diag["tail_wait_j"]
            if "timeline_start" in diag:
                diag["timeline_segments"]=[dict(kind="explicit_prefix",start=diag["timeline_start"],
                    end=diag["timeline_explicit_end"],energy_j=diag["explicit_prefix_j"])]
                if not d.full:
                    diag["timeline_segments"] += [
                        dict(kind="estimated_adjustment_wait",start=diag["timeline_explicit_end"],end=diag["timeline_anchor"],energy_j=diag["estimated_adjustment_wait_j"]),
                        dict(kind="estimated_queue_wait_at_terminal_ring",start=diag["timeline_anchor"],end=diag["timeline_queue_end"],energy_j=diag["estimated_queue_wait_j"]),
                        dict(kind="estimated_suffix_motion",start=diag["timeline_queue_end"],end=td/TICKS_PER_SECOND,energy_j=diag["estimated_suffix_motion_j"])]
            rows[aid]=diag
        return tuple(total),rows

    def key(self,selected):
        f,l,r,g,e,t=self.evaluate(selected)[0]
        return f,l,r*ENERGY_UNIT_J/3600.,g/TICKS_PER_SECOND,e*ENERGY_UNIT_J,t/TICKS_PER_SECOND

    def cp_expressions(self,model,variables,lookup):
        all_demands=(*self.demands.values(),*self.fixed_demands.values())
        max_time=max([self.now,self.horizon]+[d.ready+d.duration for d in all_demands]
            +[hi for rows in self.calendars.values() for _,hi in rows]
            +[hi for d in all_demands for _,_,hi,_ in d.real_intervals])+sum(
                max((d.duration for d in all_demands if d.aircraft_id==aid),default=0) for aid in self.order)+ticks(1.)
        # Every real interval is scanned in chronological start order. The
        # optional-overlap jump is exactly earliest_gap, including absent jobs.
        real={p:[(lo,hi,None) for lo,hi in rows] for p,rows in self.calendars.items()}
        seen=set(self.fixed_identity)
        for d in self.fixed_demands.values():
            for p,lo,hi,aid in d.real_intervals:
                if (p,lo,hi,aid) not in seen: real[p].append((lo,hi,None)); seen.add((p,lo,hi,aid))
        for key in lookup:
            d=self.demands[key]
            for p,lo,hi,aid in d.real_intervals:
                if (p,lo,hi,aid) not in self.fixed_identity: real[p].append((lo,hi,variables[key]))
        real={p:sorted(rows,key=lambda row:(row[0],row[1])) for p,rows in real.items()}
        previous={p:0 for p in real}; terms=[[] for _ in range(6)]; counter=0
        def integer(label,upper=max_time):
            nonlocal counter
            counter+=1; return model.NewIntVar(0,upper,f"service_{label}_{counter}")
        def boolean(label):
            nonlocal counter
            counter+=1; return model.NewBoolVar(f"service_{label}_{counter}")
        for aid in self.order:
            options=[(key,d,variables[key]) for key,d in self.demands.items() if d.aircraft_id==aid and key in variables]
            if aid in self.fixed_demands: options=[(None,self.fixed_demands[aid],model.NewConstant(1))]
            pad_ends={p:[] for p in real}
            for key,d,v in options:
                reachable=self._reachable(d); wait=0; td=d.touchdown*v; start=None
                if not d.full and reachable:
                    candidates={}
                    for p in reachable:
                        current=integer("ready"); model.AddMaxEquality(current,[d.ready,previous[p]])
                        for lo,hi,present in real[p]:
                            before=boolean("before"); after=boolean("after"); overlap=boolean("overlap")
                            model.Add(current<hi).OnlyEnforceIf(before); model.Add(current>=hi).OnlyEnforceIf(before.Not())
                            model.Add(current+d.duration>lo).OnlyEnforceIf(after); model.Add(current+d.duration<=lo).OnlyEnforceIf(after.Not())
                            conditions=[before,after]+([] if present is None else [present])
                            model.AddBoolAnd(conditions).OnlyEnforceIf(overlap)
                            model.AddBoolOr([c.Not() for c in conditions]).OnlyEnforceIf(overlap.Not())
                            following=integer("gap")
                            model.Add(following==hi).OnlyEnforceIf(overlap); model.Add(following==current).OnlyEnforceIf(overlap.Not())
                            current=following
                        candidates[p]=current
                    count=len(real); best=integer("best",max_time*max(1,count)+count)
                    model.AddMinEquality(best,[s*count+p for p,s in candidates.items()])
                    picks=[]; start=integer("start")
                    for p,s in candidates.items():
                        pick=boolean("pad"); picks.append(pick)
                        model.Add(best==s*count+p).OnlyEnforceIf(pick)
                        model.Add(start==s).OnlyEnforceIf(pick)
                        end=integer("end"); model.Add(end==s+d.duration).OnlyEnforceIf(pick)
                        model.Add(end==0).OnlyEnforceIf(pick.Not()); pad_ends[p].append(end)
                    model.Add(sum(picks)==v); model.Add(start==0).OnlyEnforceIf(v.Not())
                    wait=integer("wait"); model.Add(wait==start-d.ready).OnlyEnforceIf(v); model.Add(wait==0).OnlyEnforceIf(v.Not())
                    td=start+ticks(DESCENT_SECONDS)*v
                energy=d.base_energy*v+d.hold_per_tick*wait
                risk=integer("risk",max(0,d.base_energy-d.reserve_energy)+d.hold_per_tick*max_time)
                model.AddMaxEquality(risk,[0,energy-d.reserve_energy*v])
                depleted=boolean("depleted"); threshold=d.diagnostics["available_energy_threshold"]
                model.Add(energy>=threshold).OnlyEnforceIf([v,depleted]); model.Add(energy<threshold).OnlyEnforceIf([v,depleted.Not()])
                model.Add(depleted==0).OnlyEnforceIf(v.Not()); failure=v if d.failure else depleted
                g=0
                if self._eligible_baseline(aid):
                    clipped=integer("clipped",self.horizon)
                    if d.full: model.Add(clipped==min(self.horizon,max(self.now,d.ready))*v)
                    elif not reachable: model.Add(clipped==self.horizon*v)
                    else: model.AddMinEquality(clipped,[self.horizon*v,start])
                    g=integer("delay",self.horizon)
                    model.AddMaxEquality(g,[0,clipped-self.baseline_starts[aid]*v])
                for k,term in enumerate((failure,d.landed*v,risk,g,energy,td-self.now*v)): terms[k].append(term)
            for p,ends in pad_ends.items():
                if ends:
                    updated=integer("previous"); model.AddMaxEquality(updated,[previous[p],*ends]); previous[p]=updated
        return tuple(sum(row) for row in terms)
