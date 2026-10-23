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


def describe_column(snapshot, column, references, context):
    """Use existing quadrature and cached motion reference, never simulate."""
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


class JointDownstream:
    """Shared service calendars and explicit optional CP queue intervals."""
    def __init__(self, demands, fixed_calendars, order, *, now=0, pad_count=6, fixed_demands=()):
        self.demands = demands
        self.fixed_demands = {d.aircraft_id:d for d in fixed_demands}
        self.order = tuple(order)
        self.now = now
        self.calendars = {p:[] for p in range(pad_count)}
        self.fixed_identity = set()
        for p,lo,hi,aid in fixed_calendars:
            identity=(p,lo,hi,aid)
            if identity not in self.fixed_identity:
                self.calendars[p].append((lo,hi)); self.fixed_identity.add(identity)
        # One assignment based on the baseline reference readiness, then frozen.
        provisional = {p:list(rows) for p,rows in self.calendars.items()}
        self.assignment = {}
        for aid in self.order:
            options = [d for key,d in demands.items() if d.aircraft_id==aid and not d.full]
            d = self.fixed_demands.get(aid) or (options[0] if options else None)
            if d is None: continue
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
            for k,value in enumerate((d.failure,d.landed,risk,energy,td-self.now)): total[k]+=value
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
        for key,c in lookup.items():
            d=self.demands[key]; v=variables[key]
            for p,lo,hi,aid in d.real_intervals:
                if (p,lo,hi,aid) not in self.fixed_identity:
                    intervals[p].append(model.NewOptionalIntervalVar(lo,hi-lo,hi,v,f"real_{aid}_{len(intervals[p])}"))
        for aid in self.order:
            options=[(key,d) for key,d in self.demands.items() if d.aircraft_id==aid]
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
                for k,term in enumerate((d.failure*v,d.landed*v,deficit+d.unknown*UNKNOWN_RISK_J*v,energy,td-self.now*v)): terms[k].append(term)
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
