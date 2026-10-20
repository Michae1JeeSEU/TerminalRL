"""Bounded, read-only multistep planning for FT-RS-S2-1.0.

The executor owns commitment.  Columns describe predictions, never commands for
their complete suffix; only a freshly certified first maneuver can be committed.
OR-Tools is an optional import, but is mandatory when selecting the CP method.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import Counter
from itertools import combinations
from types import MappingProxyType
import math
import time
import heapq
import json

from .energy import integrate_energy, stable_power, practical_numeric_stats
from .geometry import ring_config
from .models import ValidationResult
from .resources import Reservation
from .safety import validate_separation
from .trajectory import generate_transfer, generate_final, make_stable_plan, make_ground_plan

HORIZON = 240.0
OFFSETS = (0., 5., 10., 15., 20., 30.)
SERVICE_SECONDS = {"short": 60., "standard": 120., "long": 240.}
PHASE_SHARES = (.3, .8, .3, .4, .2)


@dataclass
class PlanningBudget:
    soft_seconds: float = 5.
    watchdog_seconds: float = 30.
    candidate_seconds: float = 3.
    max_per_aircraft: int = 64
    max_attempts: int = 2048
    wall_start: float = field(default_factory=time.perf_counter)
    cpu_start: float = field(default_factory=time.process_time)
    attempts: Counter = field(default_factory=Counter)
    timings: dict = field(default_factory=dict)
    truncations: set = field(default_factory=set)
    _timing_stack: list = field(default_factory=list, repr=False)

    @property
    def elapsed(self): return time.perf_counter() - self.wall_start

    def available(self, aircraft_id=None):
        if self.elapsed >= min(self.candidate_seconds,self.soft_seconds):
            self.truncations.add("CANDIDATE_WORK_CUTOFF"); return False
        if sum(self.attempts.values()) >= self.max_attempts:
            self.truncations.add("GLOBAL_ATTEMPT_LIMIT"); return False
        if aircraft_id is not None and self.attempts[aircraft_id] >= self.max_per_aircraft:
            self.truncations.add("AIRCRAFT_ATTEMPT_LIMIT"); return False
        return True

    def watchdog_ok(self):
        if self.elapsed < self.watchdog_seconds: return True
        self.truncations.add("WATCHDOG"); return False

    def search_available(self):
        if self.elapsed<self.soft_seconds: return True
        self.truncations.add("SOFT_BUDGET"); return False

    @contextmanager
    def measure(self, name):
        w, c = time.perf_counter(), time.process_time()
        child = [0., 0.]
        self._timing_stack.append(child)
        try: yield
        finally:
            elapsed = (time.perf_counter()-w, time.process_time()-c)
            self._timing_stack.pop()
            if self._timing_stack:
                self._timing_stack[-1][0] += elapsed[0]
                self._timing_stack[-1][1] += elapsed[1]
            old = self.timings.get(name, (0., 0.))
            self.timings[name] = (old[0]+elapsed[0]-child[0], old[1]+elapsed[1]-child[1])


@dataclass(frozen=True)
class NormalReference:
    status: str
    plans: tuple = ()
    soc_required: float | None = None
    peak_required: float | None = None
    duration: float | None = None
    energy_j: float | None = None
    identity: tuple = ()
    source: str = "cold_bounded_normal"


@dataclass(frozen=True)
class PlanningNode:
    ring: str
    slot: int
    time: float
    soc: float
    peak: float
    version: int
    error_j: float = 0.
    peak_error_s: float = 0.


@dataclass(frozen=True)
class PredictionColumn:
    aircraft_id: str
    kind: str
    plans: tuple
    resources: tuple
    objective: tuple
    node: PlanningNode | None = None
    reference: NormalReference | None = None
    emergency_prediction: str = "NONE"
    energy_j: float = 0.
    peak_seconds: float = 0.

    @property
    def signature(self):
        return (self.aircraft_id, self.kind, tuple(p.trajectory_hash for p in self.plans))

    @property
    def first_target(self):
        if not self.plans: return None
        p = self.plans[0]
        return (p.target_ring, p.target_slot, p.pad_id)


@dataclass(frozen=True)
class CandidateCatalog:
    columns: object
    slacks: object
    attempts: object
    incomplete: tuple
    cache_hits: int = 0


@dataclass(frozen=True)
class PlanningResult:
    selected: tuple
    prefix: tuple
    evaluation: object
    status: str
    objective: tuple
    solver_records: tuple = ()
    refinement_rounds: int = 0
    certified: bool = False
    details: object = field(default_factory=dict)


def build_planning_snapshot(port):
    return port.build_planning_snapshot()


def _sum_key(columns):
    return tuple(math.fsum(c.objective[k] for c in columns) for k in range(5))


def solver_coefficients(column):
    """Quantize objective coefficients only: seconds/0.1, joules/3600."""
    f, l, u, tau, joules = column.objective
    return int(f), int(l), int(u), int(round(tau*10)), int(round(joules/3600))


def _node(snapshot, aircraft):
    a = aircraft
    history=snapshot.energy_ledger.get("executed",{}).get(a.id,())
    err=math.fsum(row["error_j"] for row in history)
    perr=math.fsum(row["peak_error_s"] for row in history)
    fixed=tuple(p for p in (snapshot.active.get(a.id),snapshot.soft.get(a.id)) if p is not None)
    if fixed:
        if fixed[-1].target_ring is None: return None
        portions=[]; cursor=snapshot.time; ring,slot=a.ring,a.slot
        for p in fixed:
            if p.start_time>cursor and ring is not None:
                hold=make_stable_plan(a.id,ring,slot,cursor,p.start_time,config=snapshot.config)
                portions.append((hold,cursor,p.start_time))
            portions.append((p,max(cursor,p.start_time),p.end_time))
            cursor=p.end_time; ring,slot=p.target_ring,p.target_slot
        q=_energy_details(portions,snapshot.config)
        return PlanningNode(ring,slot,cursor,a.soc-q[1],a.peak_seconds+q[2],a.version+len(fixed),err+q[3],perr+q[4])
    if a.ring is None: return None
    return PlanningNode(a.ring,a.slot,snapshot.time,a.soc,a.peak_seconds,a.version,err,perr)


def _energy_portions(portions, config, *, fine=False):
    return _energy_details(portions,config,fine=fine)[:3]


def _energy_details(portions, config, *, fine=False):
    values = [integrate_energy(p, s, e, config=config, fine=fine) for p, s, e in portions if e > s]
    if any(not q.valid for q in values): return (math.inf,)*5
    return tuple(math.fsum(getattr(q, attr) for q in values) for attr in ("joules", "soc_used", "peak_seconds","error_j","peak_error_s"))


def _move_energy(node, plan, config):
    hold = make_stable_plan(plan.aircraft_id, node.ring, node.slot, node.time,
                            max(node.time+1.e-9, plan.start_time), config=config)
    portions = ((hold, node.time, plan.start_time), (plan, plan.start_time, plan.end_time))
    q = _energy_details(portions, config)
    reserve = .25 if plan.pad_id is not None else 0.
    # Same whole-plan precision policy as executor; this never refines history.
    width=(node.error_j+q[3])/(config.energy.equivalent_voltage_V*config.energy.reference_capacity_Ah*3600)+1.e-5
    peak_width=node.peak_error_s+q[4]+.05
    if abs(node.soc-q[1])<=width or (reserve and abs(node.soc-q[1]-reserve)<=width) or abs(node.peak+q[2]-120.)<=peak_width:
        q=_energy_details(portions,config,fine=True)
    return q


def _decision_aircraft(snapshot):
    return tuple(a for a in snapshot.aircraft.values()
                 if (a.status.startswith("HOLD_") or a.status in ("INBOUND", "TRANSFER_O_M", "TRANSFER_M_I"))
                 and not (a.emergency.eligible or a.emergency.declared or a.emergency.pending)
                 and _node(snapshot, a) is not None)


def energy_slack(snapshot, a, reference):
    node = _node(snapshot, a)
    if reference.status != "KNOWN": return (1, 0., a.release_time, a.id)
    power = stable_power(node.ring, snapshot.config)
    rate = power.current_a/(snapshot.config.energy.reference_capacity_Ah*3600.)
    peak_rate = float(power.shaft_w > snapshot.config.vehicle.rotor_count*snapshot.config.energy.reference_motor_continuous_power_kW_each*1000)
    es = node.soc-.25-reference.soc_required
    ps = 120.-node.peak-reference.peak_required
    slack = min(es/rate, ps/peak_rate if peak_rate else math.inf)
    return (0 if slack < 0 else 2, slack, a.release_time, a.id)


class ReferenceProvider:
    """Nonrecursive deterministic single-aircraft normal chain construction."""
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.templates = {}

    def generate(self,snapshot,a,node,target,start,duration,budget):
        key=(snapshot.config.config_hash,a.id,node.ring,node.slot,node.version,target,start,duration)
        if key in self.templates:
            self.hits+=1
            return True,self.templates[key]
        if not budget.available(a.id): return False,None
        # A transfer duration can construct two unwrapped angular branches.
        # Reserve their worst-case count, then charge the directly observed
        # number of constructed branch templates, including cached certificates.
        cost=1 if node.ring=="inner" else 2
        if budget.attempts[a.id]+cost>budget.max_per_aircraft:
            budget.truncations.add("AIRCRAFT_ATTEMPT_LIMIT"); return False,None
        if sum(budget.attempts.values())+cost>budget.max_attempts:
            budget.truncations.add("GLOBAL_ATTEMPT_LIMIT"); return False,None
        before=practical_numeric_stats().get("transfer_feasibility_calls",0)
        p=_generate(snapshot,a.id,node,target,start,duration)
        used=1 if node.ring=="inner" else practical_numeric_stats().get("transfer_feasibility_calls",0)-before
        budget.attempts[a.id]+=used
        self.templates[key]=p
        if len(self.templates)>4096: self.templates.pop(next(iter(self.templates)))
        return True,p

    def build_steps(self,snapshot,aircraft,node,budget):
        """Yield at every primitive attempt so callers can round-robin aircraft."""
        key=(snapshot.config.config_hash,node.ring,node.slot,node.time,node.version,aircraft.id)
        if key in self.cache:
            self.hits+=1; return self.cache[key]
        current,plans=node,[]
        for _ in range({"outer":3,"middle":2,"inner":1}[node.ring]):
            found=[]
            targets=range(6 if current.ring=="inner" else ring_config({"outer":"middle","middle":"inner"}[current.ring],snapshot.config).slot_count)
            for duration in range(20,301 if current.ring=="inner" else 181,5):
                for target in targets:
                    ok,p=self.generate(snapshot,aircraft,current,target,current.time,duration,budget)
                    if not ok: return NormalReference("UNKNOWN",identity=key)
                    yield None
                    if p is not None:
                        q=_move_energy(current,p,snapshot.config)
                        if math.isfinite(q[0]): found.append((p.end_time,q[0],target,p))
                if found: break
            if not found: return NormalReference("UNKNOWN",identity=key)
            p=min(found,key=lambda row:row[:3])[3]; plans.append(p)
            if p.pad_id is not None: break
            q=_move_energy(current,p,snapshot.config)
            current=PlanningNode(p.target_ring,p.target_slot,p.end_time,current.soc-q[1],current.peak+q[2],current.version+1,current.error_j+q[3],current.peak_error_s+q[4])
        r=self._reference(snapshot,node,tuple(plans),"cold_bounded_normal") if plans and plans[-1].pad_id is not None else NormalReference("UNKNOWN",identity=key)
        if r.status=="KNOWN": self.cache[key]=r
        return r

    def get(self, snapshot, aircraft, node=None, budget=None, complete_columns=()):
        node = node or _node(snapshot, aircraft)
        full = [c for c in complete_columns if c.plans and c.plans[-1].pad_id is not None]
        if full:
            c = min(full, key=lambda c: c.energy_j)
            q = self._reference(snapshot, node, c.plans, "catalog_minimum")
            return q
        key = (snapshot.config.config_hash, node.ring, node.slot, node.time, node.version,aircraft.id)
        if key in self.cache: self.hits += 1; return self.cache[key]
        gen=self.build_steps(snapshot,aircraft,node,budget or PlanningBudget())
        while True:
            try: next(gen)
            except StopIteration as stop: return stop.value

    @staticmethod
    def _reference(snapshot, node, plans, source):
        current, energy, soc, peak = node, 0., 0., 0.
        for p in plans:
            q = _move_energy(current, p, snapshot.config)
            energy += q[0]; soc += q[1]; peak += q[2]
            if p.target_ring is not None:
                current=PlanningNode(p.target_ring,p.target_slot,p.end_time,current.soc-q[1],current.peak+q[2],current.version+1,current.error_j+q[3],current.peak_error_s+q[4])
        return NormalReference("KNOWN", plans, soc, peak, plans[-1].end_time-node.time, energy,
                               tuple(p.trajectory_hash for p in plans), source)


def _generate(snapshot, aircraft_id, node, target, start, duration):
    if node.ring == "inner":
        r = generate_final(aircraft_id, node.ring, node.slot, target, start,
                           source_version=node.version, config=snapshot.config, durations=(duration,))
    else:
        r = generate_transfer(aircraft_id, node.ring, node.slot, target, start,
                              source_version=node.version, config=snapshot.config, durations=(duration,))
    return r.plan if r.accepted else None


class EvaluationContext:
    """Private trajectory/calendar checks, with no permanent-hold prefilter."""
    def __init__(self, snapshot, prefix_evaluator=None, budget=None):
        self.snapshot = snapshot
        self.prefix_evaluator = prefix_evaluator
        self.protection = dict(snapshot.source_clear)
        self.pairs = {}
        self.decisions = {a.id for a in _decision_aircraft(snapshot)}
        self.budget=budget

    def _time_ok(self): return self.budget is None or self.budget.watchdog_ok()

    def protect_times(self, p):
        if p.trajectory_hash in self.protection: return self.protection[p.trajectory_hash]
        horizon = p.end_time+2*math.pi/.035
        clear, capture = p.start_time, p.start_time
        if p.source_ring is not None:
            base = make_stable_plan("source", p.source_ring, p.source_slot, p.start_time, horizon, config=self.snapshot.config)
            lo, hi = p.start_time, p.end_time
            if not validate_separation(p, base, hi, horizon, self.snapshot.config).accepted: return None, None
            while hi-lo > .01:
                mid = (lo+hi)/2
                if validate_separation(p, base, mid, horizon, self.snapshot.config).accepted: hi = mid
                else: lo = mid
            clear = hi
        if p.target_ring is not None:
            base = make_stable_plan("target", p.target_ring, p.target_slot, p.start_time, horizon, config=self.snapshot.config)
            lo, hi = p.start_time, p.end_time
            while hi-lo > .01:
                mid = (lo+hi)/2
                if validate_separation(p, base, p.start_time, mid, self.snapshot.config).accepted: lo = mid
                else: hi = mid
            capture = lo
        self.protection[p.trajectory_hash] = (clear, capture)
        return clear, capture

    def all_plans(self, aircraft_id, plans):
        fixed = tuple(p for p in (self.snapshot.active.get(aircraft_id), self.snapshot.soft.get(aircraft_id)) if p is not None)
        return tuple(sorted({p.trajectory_hash:p for p in (*fixed, *plans)}.values(), key=lambda p:p.start_time))

    def resources(self, a, plans):
        plans = self.all_plans(a.id, plans)
        rows = []
        if a.ring is not None:
            end = self.protect_times(plans[0])[0] if plans else math.inf
            if end is None: return None
            if end > self.snapshot.time:
                rows.append(Reservation((a.ring,a.slot), a.id, self.snapshot.time, end))
        for k, p in enumerate(plans):
            clear, capture = self.protect_times(p)
            if clear is None: return None
            if p.target_ring is not None:
                end = self.protect_times(plans[k+1])[0] if k+1 < len(plans) else math.inf
                rows.append(Reservation((p.target_ring,p.target_slot), a.id, capture, end))
            if p.pad_id is not None:
                rows.append(Reservation(("pad",p.pad_id), a.id, p.end_time-13.5, p.end_time+SERVICE_SECONDS[a.service_class]))
        if a.status == "LANDED_SERVICE":
            rows.append(Reservation(("pad",a.pad_id), a.id, self.snapshot.time,a.service_end))
        return tuple(rows)

    def traffic(self, column):
        from .world import _Traffic
        a = self.snapshot.aircraft[column.aircraft_id]
        plans = self.all_plans(a.id, column.plans)
        base = self.snapshot.active.get(a.id)
        if base is None:
            base = (make_stable_plan(a.id,a.ring,a.slot,self.snapshot.time,self.snapshot.time+901,config=self.snapshot.config)
                    if a.ring is not None else make_ground_plan(a.id,a.pad_id,self.snapshot.time,a.service_end or self.snapshot.time+901,config=self.snapshot.config))
        final = plans[-1] if plans else base
        end = final.end_time+SERVICE_SECONDS[a.service_class] if final.pad_id is not None else math.inf
        if a.status == "LANDED_SERVICE": end = a.service_end
        return _Traffic(a.id,base,plans,self.snapshot.time,end)

    @staticmethod
    def resource_conflict(a, b):
        return any(x.resource == y.resource and x.start < y.end and y.start < x.end for x in a.resources for y in b.resources)

    def check_pair(self, a, b):
        if not self._time_ok(): return ValidationResult(False,"UNKNOWN","WATCHDOG")
        key = tuple(sorted((a.signature,b.signature)))
        if key in self.pairs: return self.pairs[key]
        if self.resource_conflict(a,b):
            result = ValidationResult(False,"CONFLICT","RESOURCE_CONFLICT")
        else:
            ta, tb = self.traffic(a), self.traffic(b)
            # Include every maneuver/service endpoint, then one full common
            # rotation of retained stable tails.  Ground entities end at service.
            endpoints = [self.snapshot.time+HORIZON]
            for t in (ta,tb):
                endpoints.extend(p.end_time for p in t.plans)
                if math.isfinite(t.end_time): endpoints.append(t.end_time)
            end = max(endpoints)+2*math.pi/.035
            end = min(end,ta.end_time,tb.end_time)
            result = (validate_separation(ta,tb,self.snapshot.time,end,self.snapshot.config)
                      if end > self.snapshot.time else ValidationResult(True,"SAFE"))
        self.pairs[key] = result
        return result

    def check_single(self, column):
        # Other decision aircraft are deliberately absent here. Joint checks
        # determine their departure or continued holding.
        for a in self.snapshot.aircraft.values():
            if not self._time_ok(): return ValidationResult(False,"UNKNOWN","WATCHDOG")
            if a.id == column.aircraft_id or a.id in self.decisions: continue
            if a.status not in ("LANDED_SERVICE","FINAL_NORMAL","FINAL_EMERGENCY","VERTICAL_DESCENT") and not a.status.startswith(("HOLD_","TRANSFER_","INBOUND")): continue
            background = PredictionColumn(a.id,"BACKGROUND",(),self.resources(a,()) or (), (0,0,0,0,0))
            r = self.check_pair(column, background)
            if not r.accepted: return r
        return ValidationResult(True,"SAFE")

    def check_batch(self, columns):
        for c in columns:
            if not self._time_ok(): return ValidationResult(False,"UNKNOWN","WATCHDOG"),()
            r = self.check_single(c)
            if not r.accepted: return r, (c,)
        for a,b in combinations(columns,2):
            r = self.check_pair(a,b)
            if not r.accepted: return r, (a,b)
        return ValidationResult(True,"SAFE"), ()

    def prefix(self, columns):
        out = []
        for c in columns:
            if c.kind in ("KEEP","WAIT") or not c.plans: continue
            p = c.plans[0]
            if p.start_time <= self.snapshot.time+5.+1.e-8 and p.aircraft_id not in self.snapshot.soft:
                out.append(p)
        return tuple(out)

    def check_commit_prefix(self, columns):
        if not self._time_ok(): return ValidationResult(False,"UNKNOWN","WATCHDOG")
        prefix = self.prefix(columns)
        if self.prefix_evaluator is not None: return self.prefix_evaluator(prefix)
        from .world import evaluate_planning_snapshot
        return evaluate_planning_snapshot(self.snapshot,prefix)


class CandidateProvider:
    def __init__(self, reference_provider=None):
        self.references = reference_provider or ReferenceProvider()
        self.previous = {}
        self.aligned_starts=[]
        self._aligned_by_node={}

    @staticmethod
    def _emergency_prediction(snapshot,a,node):
        from .world import emergency_threshold
        if emergency_threshold(node.soc,None,0.): return "ELIGIBLE"
        monitor=getattr(snapshot,"emergency_monitor",None) or {}
        hint=monitor.get("imminent",{}).get(a.id)
        known_imminent=bool(hint and node.version<=hint["version"] and node.ring==hint["ring"] and node.slot==hint["slot"] and node.time>=hint["time"])
        if monitor.get("enabled",False) and not a.emergency.reference_missing and not known_imminent and node.soc>monitor.get("soc_threshold",.35):
            return "SOC_GATE_NO_REFERENCE_QUERY"
        # Reference values are bound to their actual source epoch. They cannot
        # silently be reused at an unrelated predicted phase.
        if node.time==snapshot.time and a.emergency.q_e is not None:
            qhold=stable_power(node.ring,snapshot.config).current_a*10/(snapshot.config.energy.reference_capacity_Ah*3600)
            return "ELIGIBLE" if emergency_threshold(node.soc,a.emergency.q_e,qhold) else "KNOWN_CLEAR"
        return "UNKNOWN"

    def _column(self, snapshot, a, plans, kind, context, reference=None, emergency="NONE",budget=None):
        node = _node(snapshot,a)
        current = node
        energy, peak = 0.,0.
        for p in plans:
            q = _move_energy(current,p,snapshot.config)
            energy += q[0]; peak += q[2]
            if p.target_ring is not None:
                current=PlanningNode(p.target_ring,p.target_slot,p.end_time,current.soc-q[1],current.peak+q[2],current.version+1,current.error_j+q[3],current.peak_error_s+q[4])
        allplans = context.all_plans(a.id,plans)
        portions = []
        cursor = snapshot.time
        ring,slot = a.ring,a.slot
        for p in allplans:
            if p.start_time > cursor and ring is not None:
                hold = make_stable_plan(a.id,ring,slot,cursor,p.start_time,config=snapshot.config)
                portions.append((hold,cursor,min(snapshot.time+HORIZON,p.start_time)))
            portions.append((p,max(cursor,p.start_time),min(snapshot.time+HORIZON,p.end_time)))
            cursor = p.end_time; ring,slot = p.target_ring,p.target_slot
        landed = bool(allplans and allplans[-1].pad_id is not None)
        if not landed and cursor < snapshot.time+HORIZON:
            hold = make_stable_plan(a.id,ring,slot,cursor,snapshot.time+HORIZON,config=snapshot.config)
            portions.append((hold,cursor,snapshot.time+HORIZON))
        qh = _energy_portions(portions,snapshot.config)
        soc_h = a.soc-qh[1]; peak_h = a.peak_seconds+qh[2]
        f = int(soc_h <= 0 or peak_h > 120.)
        l = int(landed and allplans[-1].end_time <= snapshot.time+HORIZON)
        if landed:
            u=0; tau=min(900.,allplans[-1].end_time-snapshot.time)
        else:
            tailtime=max(snapshot.time+HORIZON,current.time)
            tailkey=(snapshot.config.config_hash,current.ring,current.slot,tailtime,current.version,a.id)
            ref=self.references.cache.get(tailkey,NormalReference("UNKNOWN",identity=tailkey))
            if ref.status!="KNOWN" and budget is not None and budget.available(a.id):
                tailnode=PlanningNode(current.ring,current.slot,tailtime,min(soc_h,current.soc),max(peak_h,current.peak),current.version,current.error_j,current.peak_error_s)
                ref=self.references.get(snapshot,a,node=tailnode,budget=budget)
            reference=ref
            known = ref is not None and ref.status == "KNOWN"
            future_soc=min(soc_h,current.soc);future_peak=max(peak_h,current.peak)
            urgent = not known or future_soc-.25-ref.soc_required < 0 or 120.-future_peak < ref.peak_required
            u = ({"inner":1,"middle":2,"outer":3}[ring] if known else 3) if urgent else 0
            tau = min(900.,HORIZON+ref.duration) if known else 901.
        rows = context.resources(a,plans)
        if rows is None: return None
        return PredictionColumn(a.id,kind,tuple(plans),rows,(f,-l,u,tau,qh[0]),current,reference,emergency,energy,peak)

    def _paths(self,snapshot,a,node,budget,context,prefix=()):
        """Partial chains ordered by arrival estimate, energy and signature."""
        frontier=[]; deferred=[]; serial=0;root_counts=Counter()
        def key(n,chain):
            r=ring_config(n.ring,snapshot.config)
            lower=max((r.radius_m-100)/25,(r.height_agl_m-15)/2)+13.5
            refkey=(snapshot.config.config_hash,n.ring,n.slot,n.time,n.version,a.id)
            ref=self.references.cache.get(refkey)
            eta=n.time+(ref.duration if ref and ref.status=="KNOWN" else lower)
            energy=self.references._reference(snapshot,node,chain,"ordering").energy_j if chain else 0.
            signature=tuple((p.target_ring or "pad",p.target_slot if p.target_slot is not None else p.pad_id,p.start_time,p.end_time) for p in chain)
            return eta,energy,signature
        heapq.heappush(frontier,(key(node,prefix),serial,node,prefix,None))
        while (frontier or deferred) and budget.available(a.id):
            if not frontier:
                heapq.heappush(frontier,heapq.heappop(deferred))
            _,_,current,chain,cursor=heapq.heappop(frontier)
            if chain and root_counts[(chain[0].target_ring,chain[0].target_slot,chain[0].pad_id)]>=2: continue
            if chain and (current.time>=snapshot.time+HORIZON or len(chain)>=3):
                yield ("COLUMN",chain,"UNKNOWN"); continue
            state="NONE"
            if chain:
                state=self._emergency_prediction(snapshot,a,current)
                if current.ring!="inner" and state in ("ELIGIBLE","UNKNOWN"):
                    root_counts[(chain[0].target_ring,chain[0].target_slot,chain[0].pad_id)]+=1
                    yield ("COLUMN",chain,state); continue
            immediate_only=bool(chain and current.ring=="inner" and state in ("ELIGIBLE","UNKNOWN"))
            cursor=cursor or self._successors(snapshot,a,current,budget,root_counts if not chain else None,immediate_only)
            emitted=False;exhausted=False
            while budget.available(a.id):
                try: p=next(cursor)
                except StopIteration: exhausted=True;break
                yield ("ATTEMPT",None,None)
                if p is None: continue
                q=_move_energy(current,p,snapshot.config)
                if current.soc-q[1] < (.25 if p.pad_id is not None else 0.) or current.peak+q[2]>120: continue
                plans=(*chain,p)
                serial+=1
                if p.pad_id is not None:
                    root_counts[(plans[0].target_ring,plans[0].target_slot,plans[0].pad_id)]+=1
                    emitted=True
                    predicted=self._emergency_prediction(snapshot,a,current)
                    yield ("COLUMN",plans,predicted+"_IMMEDIATE_NORMAL_EXCEPTION" if immediate_only else predicted)
                else:
                    child=PlanningNode(p.target_ring,p.target_slot,p.end_time,current.soc-q[1],current.peak+q[2],current.version+1,current.error_j+q[3],current.peak_error_s+q[4])
                    heapq.heappush(frontier,(key(child,plans),serial,child,plans,None))
                serial+=1
                # A cursor is unconstructed work, not a partial chain. Resume it
                # after the materialized frontier finishes, in parent-key order.
                heapq.heappush(deferred,(key(current,chain),serial,current,chain,cursor))
                break
            if immediate_only and not emitted and (exhausted or not budget.available(a.id)):
                yield ("COLUMN",chain,state)

    def _successors(self,snapshot,a,node,budget,root_counts=None,immediate_only=False):
        targets=range(6 if node.ring=="inner" else ring_config({"outer":"middle","middle":"inner"}[node.ring],snapshot.config).slot_count)
        rows=snapshot.pads if node.ring=="inner" else snapshot.slots
        next_ring=None if node.ring=="inner" else {"outer":"middle","middle":"inner"}[node.ring]
        opportunities=sorted({(r.end,str(r.resource),r.resource) for r in rows
                              if math.isfinite(r.end) and r.end>node.time and
                              (next_ring is None or (isinstance(r.resource,tuple) and r.resource[0]==next_ring))})[:2]
        aligned_used=self._aligned_by_node.setdefault((a.id,node.ring,node.slot,node.time,node.version),set())
        for offset in OFFSETS:
            if immediate_only and offset!=0.: break
            for target in targets:
                first=(None,None,target) if node.ring=="inner" else ({"outer":"middle","middle":"inner"}[node.ring],target,None)
                if root_counts is not None and root_counts[first]>=2: continue
                # Finite release opportunities; pad alignment uses descent start.
                resource=target if node.ring=="inner" else ({"outer":"middle","middle":"inner"}[node.ring],target)
                aligned=[end for end,_,r in opportunities if r==resource]
                feasible_durations=Counter()
                for duration in range(20,301 if node.ring=="inner" else 181,5):
                    if root_counts is not None and root_counts[first]>=2: break
                    if feasible_durations.get(node.time+offset,0)>=2 and not aligned: break
                    starts={node.time+offset}
                    for end in aligned:
                        start=end-duration if node.ring=="inner" else end
                        if immediate_only and abs(start-node.time)>1.e-8: continue
                        if start>=node.time and (start in aligned_used or len(aligned_used)<2): starts.add(start)
                    for start in sorted(starts):
                        if root_counts is not None and root_counts[first]>=2: break
                        if feasible_durations[start]>=2: continue
                        if start > snapshot.time+900: continue
                        ok,p=self.references.generate(snapshot,a,node,target,start,duration,budget)
                        if not ok: return
                        if p is None or p.end_time>snapshot.time+900:
                            yield None; continue
                        if start!=node.time+offset and start not in aligned_used:
                            aligned_used.add(start)
                            self.aligned_starts.append((a.id,node.ring,node.slot,node.time,start))
                        feasible_durations[start]+=1
                        yield p

    def get_columns(self,snapshot,budget=None,context=None):
        budget=budget or PlanningBudget(); context=context or EvaluationContext(snapshot,budget=budget)
        self.aligned_starts=[]
        self._aligned_by_node={}
        start_hits=self.references.hits
        aircraft=_decision_aircraft(snapshot)
        refs={}; slacks={}; columns={}; incomplete=[]
        # Round-robin candidate construction precedes any cold reference work.
        # UNKNOWN priorities are deterministic until a complete chain is present.
        for a in aircraft:
            node=_node(snapshot,a)
            key=(snapshot.config.config_hash,node.ring,node.slot,node.time,node.version,a.id)
            refs[a.id]=self.references.cache.get(key,NormalReference("UNKNOWN"))
            slacks[a.id]=energy_slack(snapshot,a,refs[a.id])
            columns[a.id]=[self._column(snapshot,a,(),"KEEP",context,refs[a.id])]
            columns[a.id].append(self._column(snapshot,a,(),"WAIT",context,refs[a.id]))
            previous=self.previous.get(a.id,())
            if previous and previous[0].start_time>=node.time and previous[0].source_version==node.version:
                c=self._column(snapshot,a,previous,"MOVE",context,refs[a.id])
                if c is not None and context.check_single(c).accepted: columns[a.id].append(c)
        ordered=sorted(aircraft,key=lambda a:slacks[a.id])
        generators={a.id:self._paths(snapshot,a,_node(snapshot,a),budget,context) for a in ordered}
        counts={a.id:Counter(c.first_target for c in columns[a.id] if c.kind=="MOVE") for a in ordered}
        while generators and budget.available():
            for a in ordered:
                gen=generators.get(a.id)
                if gen is None: continue
                try: tag,plans,emergency=next(gen)
                except StopIteration: generators.pop(a.id,None); continue
                if tag!="COLUMN": continue
                first=(plans[0].target_ring,plans[0].target_slot,plans[0].pad_id)
                if counts[a.id][first]>=2: continue
                ref=(ReferenceProvider._reference(snapshot,_node(snapshot,a),plans,"catalog_minimum")
                     if plans[-1].pad_id is not None else refs[a.id])
                c=self._column(snapshot,a,plans,"MOVE",context,ref,emergency,budget=budget)
                if c is not None and c.signature not in {x.signature for x in columns[a.id]} and context.check_single(c).accepted:
                    columns[a.id].append(c); counts[a.id][first]+=1
                    if ref.status=="KNOWN" and (refs[a.id].status!="KNOWN" or ref.energy_j<refs[a.id].energy_j): refs[a.id]=ref
                if len(columns[a.id])>=10: generators.pop(a.id,None); budget.truncations.add("MOVING_COLUMN_LIMIT")
        for a in ordered:
            if refs[a.id].status!="KNOWN":
                refs[a.id]=self.references.get(snapshot,a,budget=budget)
            if refs[a.id].status=="KNOWN":
                n=_node(snapshot,a)
                self.references.cache[(snapshot.config.config_hash,n.ring,n.slot,n.time,n.version,a.id)]=refs[a.id]
                if len(self.references.cache)>4096: self.references.cache.pop(next(iter(self.references.cache)))
            slacks[a.id]=energy_slack(snapshot,a,refs[a.id])
            columns[a.id][:2]=[self._column(snapshot,a,(),k,context,refs[a.id],budget=budget) for k in ("KEEP","WAIT")]
            if len(columns[a.id])<3 or a.id in generators: incomplete.append(a.id)
        return CandidateCatalog(MappingProxyType({k:tuple(v) for k,v in columns.items()}),MappingProxyType(slacks),
                                MappingProxyType(dict(budget.attempts)),tuple(incomplete),self.references.hits-start_hits)


class EnergySlackScheduler:
    method="energy_slack"
    def plan(self,snapshot,candidates,context,budget):
        chosen={i:cols[0] for i,cols in candidates.columns.items()}
        for i in sorted(chosen,key=lambda i:candidates.slacks[i]):
            if not budget.search_available(): break
            for c in sorted(candidates.columns[i],key=lambda c:(c.objective,c.signature)):
                trial=dict(chosen); trial[i]=c
                r,_=context.check_batch(tuple(trial.values()))
                if r.accepted: chosen=trial; break
        selected=tuple(chosen.values())
        with budget.measure("selected_validation"):
            r,_=context.check_batch(selected)
            e=context.check_commit_prefix(selected) if r.accepted and budget.watchdog_ok() else None
        if e is not None and e.accepted:
            return PlanningResult(selected,context.prefix(selected),e,"FEASIBLE",_sum_key(selected),certified=True)
        # One local repair: rejected predictive release cannot authorize execution.
        with budget.measure("fallback"):
            keep=tuple(cols[0] for cols in candidates.columns.values())
            e=context.check_commit_prefix(keep) if budget.watchdog_ok() else None
        return PlanningResult(keep,(),e,"PREFIX_NOT_EXECUTABLE",_sum_key(keep),certified=bool(e and e.accepted))


class RollingHorizonCPSATScheduler:
    method="rh_cp_sat"
    def plan(self,snapshot,candidates,context,budget):
        try: from ortools.sat.python import cp_model
        except ImportError as exc:
            raise RuntimeError("RH-CP-SAT requires OR-Tools; no replacement solver is used") from exc
        # Own timed construction, on this method's snapshot and catalog.
        with budget.measure("hint_construction"):
            incumbent=EnergySlackScheduler().plan(snapshot,candidates,context,budget)
        if not incumbent.certified: incumbent=None
        exclusions=[]; nogoods=[]; records=[]; solve_used=0.; all_optimal=True;final_solver_key=None
        ids=tuple(candidates.columns)
        with budget.measure("resource_conflict_build"):
            for i,j in combinations(ids,2):
                if not budget.search_available() or not budget.watchdog_ok(): break
                for c in candidates.columns[i]:
                    if not budget.search_available() or not budget.watchdog_ok(): break
                    for d in candidates.columns[j]:
                        if context.resource_conflict(c,d): exclusions.append((c.signature,d.signature))
        for refinement in range(3):
            if not budget.search_available() or solve_used>=2.: break
            with budget.measure("model_build"):
                model=cp_model.CpModel(); variables={}; lookup={}
                for i in ids:
                    vs=[]
                    for n,c in enumerate(candidates.columns[i]):
                        v=model.NewBoolVar(f"x_{i}_{n}"); variables[c.signature]=v; lookup[c.signature]=c; vs.append(v)
                    model.AddExactlyOne(vs)
                for a,b in exclusions: model.Add(variables[a]+variables[b]<=1)
                for keys in nogoods: model.Add(sum(variables[k] for k in keys)<=len(keys)-1)
                for c in lookup.values():
                    if not budget.watchdog_ok(): break
                    r=context.check_single(c)
                    if not r.accepted: model.Add(variables[c.signature]==0)
                if incumbent:
                    chosen={c.signature for c in incumbent.selected}
                    for key,v in variables.items(): model.AddHint(v,int(key in chosen))
            invalid=False; carry=0.
            for phase,share in enumerate(PHASE_SHARES):
                if not budget.search_available() or solve_used>=2.: all_optimal=False; break
                limit=min(share+carry,2.-solve_used,max(0.,budget.soft_seconds-budget.elapsed))
                if limit<=0: all_optimal=False; break
                expression=sum(solver_coefficients(c)[phase]*variables[key] for key,c in lookup.items())
                model.Minimize(expression)
                solver=cp_model.CpSolver(); solver.parameters.num_search_workers=1; solver.parameters.random_seed=71
                solver.parameters.max_time_in_seconds=limit
                with budget.measure("cp_solve"):
                    started=time.perf_counter(); status=solver.Solve(model); used=time.perf_counter()-started
                solve_used+=used; carry=max(0.,limit-used)
                rec={"round":refinement+1,"phase":phase,"status":solver.StatusName(status),"wall_s":used,
                     "bound":solver.BestObjectiveBound(),"branches":solver.NumBranches(),"conflicts":solver.NumConflicts(),"objective":None}
                records.append(rec)
                if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE): all_optimal=False; break
                selected=tuple(c for key,c in lookup.items() if solver.Value(variables[key]))
                rec["objective"]=solver.ObjectiveValue()
                with budget.measure("selected_validation"):
                    r,conflict=context.check_batch(selected)
                    evaluation=context.check_commit_prefix(selected) if r.accepted and budget.watchdog_ok() else None
                rec["validation_status"]=r.status
                rec["prefix_status"]=getattr(evaluation,"status","NOT_CHECKED")
                if not r.accepted or evaluation is None or not evaluation.accepted:
                    if len(conflict)==2 and r.status!="UNKNOWN": exclusions.append(tuple(c.signature for c in conflict))
                    elif len(conflict)==1: nogoods.append(tuple(c.signature for c in selected))
                    else: nogoods.append(tuple(c.signature for c in selected))
                    invalid=True; all_optimal=False; break
                key=_sum_key(selected)
                proposal=PlanningResult(selected,context.prefix(selected),evaluation,"FEASIBLE",key,certified=True)
                if phase==4:
                    final_solver_key=tuple(sum(solver_coefficients(c)[k] for c in selected) for k in range(5))
                if incumbent is None or proposal.objective<incumbent.objective: incumbent=proposal
                if status!=cp_model.OPTIMAL: all_optimal=False
                # Only a certified feasible value may constrain the next layer.
                model.Add(expression==int(round(solver.ObjectiveValue())))
            if not invalid: break
        if incumbent is None:
            keep=tuple(cols[0] for cols in candidates.columns.values())
            with budget.measure("fallback"):
                evaluation=context.check_commit_prefix(keep) if budget.watchdog_ok() else None
            incumbent=PlanningResult(keep,(),evaluation,"NO_VERIFIED_INCUMBENT",_sum_key(keep),certified=bool(evaluation and evaluation.accepted))
        incumbent_solver_key=tuple(sum(solver_coefficients(c)[k] for c in incumbent.selected) for k in range(5))
        complete=all_optimal and len(records)>=5 and records[-1]["phase"]==4 and records[-1]["status"]=="OPTIMAL" and incumbent_solver_key==final_solver_key
        return PlanningResult(incumbent.selected,incumbent.prefix,incumbent.evaluation,
                              "CATALOG_LEX_OPTIMAL" if complete else "LEX_LIMITED",incumbent.objective,
                              tuple(records),max((r["round"] for r in records),default=0),incumbent.certified,
                              MappingProxyType({"solver_wall_s":solve_used,"catalog_incomplete":candidates.incomplete,
                                                "fallback":incumbent.status=="NO_VERIFIED_INCUMBENT"}))


class NormalSchedulerCallback:
    """Callable SchedulerPort adapter; runner owns admission/emergency timing."""
    def __init__(self,method="energy_slack",candidate_provider=None):
        self.scheduler=EnergySlackScheduler() if method=="energy_slack" else RollingHorizonCPSATScheduler() if method=="rh_cp_sat" else None
        if self.scheduler is None: raise ValueError("unknown Stage2 baseline")
        self.provider=candidate_provider or CandidateProvider()
        self.records=[]
        self.last_result=None

    def state_dict(self):
        from .trajectory import TrajectoryPlan
        def reference(r):
            return {"status":r.status,"plans":[p.to_dict() for p in r.plans],"soc_required":r.soc_required,
                    "peak_required":r.peak_required,"duration":r.duration,"energy_j":r.energy_j,
                    "identity":r.identity,"source":r.source}
        state={"method":self.scheduler.method,"records":self.records,
                "previous":{k:[p.to_dict() for p in v] for k,v in self.provider.previous.items()},
                "references":[(key,reference(r)) for key,r in self.provider.references.cache.items()],
                "templates":[(key,p.to_dict() if p is not None else None) for key,p in self.provider.references.templates.items()],
                "reference_hits":self.provider.references.hits}
        return json.loads(json.dumps(state,allow_nan=False))

    def load_state_dict(self,state):
        from .trajectory import TrajectoryPlan
        if state["method"]!=self.scheduler.method: raise ValueError("scheduler method mismatch")
        self.records=list(state["records"])
        self.provider.previous={k:tuple(TrajectoryPlan.from_dict(p) for p in v) for k,v in state["previous"].items()}
        self.provider.references.cache={}
        for key,data in state["references"]:
            data=dict(data); data["plans"]=tuple(TrajectoryPlan.from_dict(p) for p in data["plans"])
            data["identity"]=tuple(data["identity"])
            self.provider.references.cache[tuple(key)]=NormalReference(**data)
        self.provider.references.hits=state["reference_hits"]
        self.provider.references.templates={tuple(key):TrajectoryPlan.from_dict(p) if p is not None else None for key,p in state.get("templates",())}

    def __call__(self,port,*args):
        if args: port=args[-1]  # executor callback(snapshot, port) convention
        budget=PlanningBudget()
        for restart in range(2):
            with budget.measure("snapshot_and_uniform_refinement"):
                snapshot=build_planning_snapshot(port)
            context=EvaluationContext(snapshot,budget=budget)
            with budget.measure("candidate_generation_and_single_vehicle"):
                catalog=self.provider.get_columns(snapshot,budget,context)
            with budget.measure("selection"):
                result=self.scheduler.plan(snapshot,catalog,context,budget)
            if not getattr(result.evaluation,"refinement_required",False): break
            if restart or not budget.search_available():
                result=PlanningResult(result.selected,(),None,"REFINEMENT_UNCERTAIN",result.objective)
                break
            with budget.measure("snapshot_and_uniform_refinement"):
                port.refine_planning_boundary()
        commit=None
        if result.certified and budget.watchdog_ok():
            with budget.measure("commit"):
                commit=port.commit_validated_prefix(result.evaluation,snapshot_version=snapshot.version)
        self.last_result=result
        committed_prefix=result.prefix if commit is not None and commit.accepted else ()
        self.provider.previous={c.aircraft_id:tuple(p for p in c.plans if p not in committed_prefix) for c in result.selected if c.kind=="MOVE"}
        record={"time":snapshot.time,"method":self.scheduler.method,"status":result.status,
                "objective":result.objective,"columns":sum(map(len,catalog.columns.values())),
                "aircraft":len(catalog.columns),"attempts":dict(catalog.attempts),"catalog_incomplete":catalog.incomplete,
                "truncations":sorted(budget.truncations),"timings":dict(budget.timings),
                "wall_s":budget.elapsed,"cpu_s":time.process_time()-budget.cpu_start,
                "deadline_miss":budget.elapsed>5.,"submitted":len(result.prefix) if commit and commit.accepted else 0,
                "commit_status":commit.status if commit is not None else "NOT_COMMITTED",
                "solver":result.solver_records,"refinement_rounds":result.refinement_rounds,
                "uniform_refinement_restarts":restart,
                "active_count":len(snapshot.active),"holding_count":sum(a.status.startswith("HOLD_") for a in snapshot.aircraft.values()),
                "upstream_count":len(snapshot.requests),"trigger_reason":"UNSPECIFIED_EXECUTOR_TRIGGER",
                "cache_hits":catalog.cache_hits,"trajectory_recomputes":sum(catalog.attempts.values()),
                "resource_aligned_starts":tuple(self.provider.aligned_starts),
                "fallback":bool(result.details.get("fallback",False)) or result.status in ("PREFIX_NOT_EXECUTABLE","NO_VERIFIED_INCUMBENT","REFINEMENT_UNCERTAIN"),
                "stale":bool(commit is not None and commit.status=="STALE"),"watchdog":"WATCHDOG" in budget.truncations,
                "noop":not result.prefix,"selected_key":result.objective,
                "keep":sum(c.kind=="KEEP" for c in result.selected),"wait":sum(c.kind=="WAIT" for c in result.selected)}
        self.records.append(record)
        return result
