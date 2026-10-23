"""Bounded, read-only multistep planning for FT-RS-S2-1.0.

The executor owns commitment.  Columns describe predictions, never commands for
their complete suffix; only a freshly certified first maneuver can be committed.
OR-Tools is an optional import, but is mandatory when selecting the CP method.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from collections import Counter
from itertools import combinations
from types import MappingProxyType
import math
import time
import heapq
import json
import hashlib

from .energy import integrate_energy, stable_power, practical_numeric_stats
from .config import service_duration
from .geometry import ring_config
from .models import ValidationResult
from .resources import Reservation
from .safety import validate_separation
from .trajectory import (generate_transfer, generate_final, make_stable_plan, make_ground_plan,
                         screen_normal_templates)

HORIZON = 240.0
OFFSETS = (0., 5., 10., 15., 20., 30.)
SERVICE_SECONDS = {"short": 60., "standard": 120., "long": 240.}
PHASE_SHARES = (.3, .8, .3, .4, .2)
QUALITY_BUDGET_PROFILE = "quality_v5"
CANDIDATE_POLICY_ID = "shared_multistep_columns_v6_reuse_diversity"
PLANNING_PROFILE_LEGACY = "progress_tail_v7"
PLANNING_PROFILE_DOWNSTREAM = "downstream_service_v8"
PLANNING_PROFILES = (PLANNING_PROFILE_LEGACY, PLANNING_PROFILE_DOWNSTREAM)


def candidate_policy_id(profile):
    if profile not in PLANNING_PROFILES: raise ValueError("unknown Stage2 planning profile: "+str(profile))
    return CANDIDATE_POLICY_ID if profile == PLANNING_PROFILE_LEGACY else "shared_downstream_columns_v8_bounded_delay"


def objective_schema_id(profile):
    if profile not in PLANNING_PROFILES: raise ValueError("unknown Stage2 planning profile: "+str(profile))
    return "reliability_throughput_progress_tail_v2" if profile == PLANNING_PROFILE_LEGACY else "downstream_service_v8_f_L_risk_Ehat_tau"
QUALITY_BUDGET_SETTINGS = MappingProxyType(dict(soft_seconds=30.,watchdog_seconds=60.,candidate_seconds=12.,
    hint_seconds=3.,solver_seconds=8.,solver_reserve_seconds=8.))
LEGACY_BUDGET_PROFILE = "legacy_v4"
UNKNOWN_SLACK_SOURCE = "holding_margin_minus_remaining_time_lower_v5"
QUALITY_MAX_REFINEMENT_ROUNDS = 64
QUALITY_HINT_VALIDATION_RESERVE_SECONDS = .5
VALIDATION_CACHE_LIMIT = 8192
MAX_BATCH_CONFLICTS = 16


def _v6(snapshot):
    from .rescue_rules import RESCUE_PROFILE_V6, RESCUE_PROFILE_V7
    return (getattr(snapshot,"emergency_monitor",None) or {}).get("rescue_rule_profile") in (RESCUE_PROFILE_V6, RESCUE_PROFILE_V7)


def _bounded_put(cache,key,value,limit=VALIDATION_CACHE_LIMIT):
    cache[key]=value
    if len(cache)>limit: cache.pop(next(iter(cache)))


@dataclass
class PlanningBudget:
    soft_seconds: float = 5.
    watchdog_seconds: float = 30.
    candidate_seconds: float = 3.
    max_per_aircraft: int = 64
    max_attempts: int = 2048
    profile: str = LEGACY_BUDGET_PROFILE
    hint_seconds: float | None = None
    solver_seconds: float = 2.
    solver_reserve_seconds: float = 0.
    solver_wall_used: float = 0.
    hint_wall_used: float = 0.
    max_refinement_rounds: int = 3
    refinement_rounds_used: int = 0
    hint_validation_reserve_seconds: float = 0.
    wall_start: float = field(default_factory=time.perf_counter)
    cpu_start: float = field(default_factory=time.process_time)
    attempts: Counter = field(default_factory=Counter)
    screening: Counter = field(default_factory=Counter)
    timings: dict = field(default_factory=dict)
    truncations: set = field(default_factory=set)
    _timing_stack: list = field(default_factory=list, repr=False)
    _stage_limits: list = field(default_factory=list, repr=False)

    @classmethod
    def for_profile(cls, profile):
        if profile == LEGACY_BUDGET_PROFILE: return cls()
        if profile == QUALITY_BUDGET_PROFILE:
            return cls(profile=profile,**QUALITY_BUDGET_SETTINGS,max_refinement_rounds=QUALITY_MAX_REFINEMENT_ROUNDS,
                hint_validation_reserve_seconds=QUALITY_HINT_VALIDATION_RESERVE_SECONDS)
        raise ValueError("unknown Stage2 planning budget profile: "+str(profile))

    def settings(self):
        return {name:getattr(self,name) for name in ("soft_seconds","watchdog_seconds","candidate_seconds",
            "hint_seconds","solver_seconds","solver_reserve_seconds","max_per_aircraft","max_attempts",
            "max_refinement_rounds","hint_validation_reserve_seconds")}

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

    def search_available(self, reserve_seconds=0.):
        if not self.evaluation_available(): return False
        if reserve_seconds and any(time.perf_counter()>=deadline-reserve_seconds for deadline,_ in self._stage_limits):
            self.truncations.add("HINT_VALIDATION_RESERVE"); return False
        if self.elapsed<self.soft_seconds: return True
        self.truncations.add("SOFT_BUDGET"); return False

    def evaluation_available(self):
        if not self.watchdog_ok(): return False
        for deadline,reason in self._stage_limits:
            if time.perf_counter()>=deadline:
                self.truncations.add(reason); return False
        return True

    def evaluation_stop_reason(self):
        if self.elapsed>=self.watchdog_seconds: return "WATCHDOG"
        for deadline,reason in self._stage_limits:
            if time.perf_counter()>=deadline: return reason
        return "EVALUATION_BUDGET"

    @contextmanager
    def stage_limit(self, seconds, reason, reserve_seconds=0.):
        # An absolute deadline on the original clock, never a fresh budget.
        deadline=self.wall_start+self.soft_seconds-reserve_seconds
        if seconds is not None: deadline=min(deadline,time.perf_counter()+max(0.,seconds))
        self._stage_limits.append((deadline,reason))
        try: yield
        finally: self._stage_limits.pop()

    def preparation_available(self):
        if not self.search_available(): return False
        if self.elapsed<self.soft_seconds-self.solver_reserve_seconds: return True
        self.truncations.add("CP_PREPARATION_CUTOFF"); return False

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
    downstream: object = field(default_factory=dict)

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
    slack_reference_statuses: object = field(default_factory=dict)
    downstream: object = None


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


def _selection_key(catalog, columns):
    return catalog.downstream.key(columns) if catalog.downstream is not None else _sum_key(columns)


def _selection_solver_key(catalog, columns):
    return (catalog.downstream.evaluate(columns)[0] if catalog.downstream is not None else
            tuple(sum(solver_coefficients(c)[k] for c in columns) for k in range(5)))


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


def remaining_normal_time_lower(snapshot, ring):
    """Queue heuristic from necessary canonical/grid times, not a certificate.

    This strengthens the original geometric heuristic by summing the
    necessary minimum duration of each remaining canonical maneuver.
    """
    original=ring_config(ring,snapshot.config)
    geometric=max((original.radius_m-100)/25,(original.height_agl_m-15)/2)+13.5
    remaining=13.5
    while ring is not None:
        source=ring_config(ring,snapshot.config)
        next_ring={"outer":"middle","middle":"inner","inner":None}[ring]
        target=ring_config(next_ring,snapshot.config) if next_ring else None
        radius=target.radius_m if target else snapshot.config.pads.layout_radius_m
        # The canonical radial quintic has r''(1/4)=-5.625*dr/T^2.
        # Centripetal acceleration has the same sign at that point.
        seconds=max(20.,math.sqrt(5.625*(source.radius_m-radius)/snapshot.config.vehicle.max_horizontal_acceleration_norm_mps2))
        if target:
            # The canonical transfer's midpoint vertical speed is
            # -1.875*dz/T. Finals use their own existing height profile.
            seconds=max(seconds,1.875*(source.height_agl_m-target.height_agl_m)/snapshot.config.vehicle.max_descent_speed_mps)
        remaining+=5*math.ceil((seconds-1.e-10)/5.)
        ring=next_ring
    return max(geometric,remaining)


def energy_slack(snapshot, a, reference):
    node = _node(snapshot, a)
    power = stable_power(node.ring, snapshot.config)
    rate = power.current_a/(snapshot.config.energy.reference_capacity_Ah*3600.)
    if reference.status != "KNOWN":
        margin=(node.soc-.25)/rate-remaining_normal_time_lower(snapshot,node.ring)
        return (1,margin,a.release_time,a.id)
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
        self.energy = {}
        self.motion_energy_index = {}
        self.energy_stats = Counter()
        self.advisor_runtime = None
        self._stage3_screened = set()

    def move_energy(self,node,plan,config):
        key=(config.config_hash,node,plan.trajectory_hash)
        if key in self.energy:
            self.energy_stats["hits"]+=1
            _bounded_put(self.motion_energy_index,(config.config_hash,plan.trajectory_hash),(node,self.energy[key]))
            if self.advisor_runtime is not None: self.advisor_runtime.record_energy(node,plan,self.energy[key],True)
            return self.energy[key]
        self.energy_stats["misses"]+=1
        energy_clock=(time.perf_counter(),time.process_time()) if self.advisor_runtime is not None else None
        q=_move_energy(node,plan,config)
        if all(math.isfinite(x) for x in q):
            _bounded_put(self.energy,key,q)
            _bounded_put(self.motion_energy_index,(config.config_hash,plan.trajectory_hash),(node,q))
        if self.advisor_runtime is not None:
            self.advisor_runtime.record_energy(node,plan,q,False,time.perf_counter()-energy_clock[0],time.process_time()-energy_clock[1])
        return q

    def screen(self, snapshot, node, start, budget):
        if not budget.available(): return None
        with budget.measure("geometry_template_screen"):
            table, hit = screen_normal_templates(node.ring, node.slot, start, snapshot.config)
        budget.screening["grid_calls"] += 1
        budget.screening["cache_hits"] += int(hit)
        budget.screening["templates"] += len(table)
        budget.screening["witness_rejected"] += sum(value is not None for value in table.values())
        if self.advisor_runtime is not None:
            self._stage3_screened.add((snapshot.config.config_hash,node.ring,node.slot,start))
            if len(self._stage3_screened)>4096: self._stage3_screened.clear()
        return table

    def generate(self,snapshot,a,node,target,start,duration,budget,*,construction_stage="reference",prefix=(),base=None,work_descriptor=None):
        runtime=self.advisor_runtime
        if runtime is None:
            return self._generate_unobserved(snapshot,a,node,target,start,duration,budget)
        descriptor=work_descriptor or runtime.descriptor(snapshot,a,node,target,start,duration,construction_stage,prefix,base)
        key=(snapshot.config.config_hash,a.id,node.ring,node.slot,node.version,target,start,duration)
        cache=dict(template_hit=self.templates.get(key) is not None,reference_hits=self.hits,
                   geometry_screened_before_query=(snapshot.config.config_hash,node.ring,node.slot,start) in self._stage3_screened,
                   geometry_cost_separate=True)
        token=runtime.record_before(snapshot,descriptor,budget,cache)
        ok,p=self._generate_unobserved(snapshot,a,node,target,start,duration,budget)
        runtime.record_after(descriptor,token,ok,p,budget,dict(cache,reference_hits_after=self.hits))
        return ok,p

    def _generate_unobserved(self,snapshot,a,node,target,start,duration,budget):
        key=(snapshot.config.config_hash,a.id,node.ring,node.slot,node.version,target,start,duration)
        if self.templates.get(key) is not None:
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
        # A bounded template miss is not an infeasibility certificate. Retain
        # only constructed plans, so an UNKNOWN/NOT_FOUND cannot poison reuse.
        if p is not None:
            self.templates[key]=p
            if len(self.templates)>4096: self.templates.pop(next(iter(self.templates)))
        return True,p

    def build_steps(self,snapshot,aircraft,node,budget):
        """Yield at every primitive attempt so callers can round-robin aircraft."""
        key=(snapshot.config.config_hash,node.ring,node.slot,node.time,node.version,aircraft.id)
        if key in self.cache:
            self.hits+=1
            if self.advisor_runtime is not None:
                self._register_reused_chain(snapshot,aircraft,node,self.cache[key].plans,budget)
            return self.cache[key]
        current,plans=node,[]
        for _ in range({"outer":3,"middle":2,"inner":1}[node.ring]):
            found=[]
            targets=range(6 if current.ring=="inner" else ring_config({"outer":"middle","middle":"inner"}[current.ring],snapshot.config).slot_count)
            screen=self.screen(snapshot,current,current.time,budget)
            if screen is None: return NormalReference("UNKNOWN",identity=key)
            yield None
            for duration in range(20,301 if current.ring=="inner" else 181,5):
                ready=[target for target in targets if screen[(target,duration)] is None]
                while ready:
                    descriptor=None
                    if self.advisor_runtime is not None:
                        runtime=self.advisor_runtime
                        work=[runtime.descriptor(snapshot,aircraft,current,target,current.time,duration,"reference",tuple(plans)) for target in ready[:64]]
                        descriptor=runtime.order(snapshot,work,budget)[0]
                        target=descriptor.target_id;ready.remove(target)
                    else: target=ready.pop(0)
                    if screen[(target,duration)] is not None: continue
                    if self.advisor_runtime is None:
                        ok,p=self.generate(snapshot,aircraft,current,target,current.time,duration,budget)
                    else:
                        ok,p=self.generate(snapshot,aircraft,current,target,current.time,duration,budget,
                                           prefix=tuple(plans),work_descriptor=descriptor)
                    if not ok: return NormalReference("UNKNOWN",identity=key)
                    yield None
                    if p is not None:
                        q=self.move_energy(current,p,snapshot.config)
                        if math.isfinite(q[0]): found.append((p.end_time,q[0],target,p))
                if found: break
            if not found: return NormalReference("UNKNOWN",identity=key)
            p=min(found,key=lambda row:row[:3])[3]; plans.append(p)
            if p.pad_id is not None: break
            q=self.move_energy(current,p,snapshot.config)
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
        if key in self.cache:
            self.hits += 1
            if self.advisor_runtime is not None:
                self._register_reused_chain(snapshot,aircraft,node,self.cache[key].plans,budget)
            return self.cache[key]
        gen=self.build_steps(snapshot,aircraft,node,budget or PlanningBudget())
        while True:
            try: next(gen)
            except StopIteration as stop: return stop.value

    def _register_reused_chain(self,snapshot,a,node,plans,budget):
        """Observe exact cached source epochs, without translating a plan."""
        current=node;prefix=()
        for p in plans:
            if (p.source_ring,p.source_slot,p.source_version)!=(current.ring,current.slot,current.version) or p.start_time<current.time-1.e-8:
                return
            q=self.move_energy(current,p,snapshot.config)
            if not all(math.isfinite(v) for v in q): return
            self.advisor_runtime.register_reuse(snapshot,a,current,p,prefix,q)
            prefix=(*prefix,p)
            if p.target_ring is None: break
            current=PlanningNode(p.target_ring,p.target_slot,p.end_time,current.soc-q[1],current.peak+q[2],
                current.version+1,current.error_j+q[3],current.peak_error_s+q[4])

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
    def __init__(self, snapshot, prefix_evaluator=None, budget=None, validation_cache=None):
        self.snapshot = snapshot
        self.prefix_evaluator = prefix_evaluator
        self.cache=validation_cache if validation_cache is not None else {}
        self.cache_stats=Counter()
        # Include the full immutable scheduling input, rather than the world
        # version alone: resources, SOC and fixed traffic are validation inputs.
        self.snapshot_key=hashlib.sha256(repr((snapshot,HORIZON)).encode()).hexdigest()
        previous=self.cache.get("snapshot_key")
        if previous is not None and previous!=self.snapshot_key: self.cache_stats["snapshot_invalidations"]+=1
        self.cache["snapshot_key"]=self.snapshot_key
        self.protection=self.cache.setdefault("protection",{})
        for key,value in snapshot.source_clear.items():
            _bounded_put(self.protection,(snapshot.config.config_hash,key),value)
        self.pairs=self.cache.setdefault("pairs",{})
        self.singles=self.cache.setdefault("singles",{})
        self.energies=self.cache.setdefault("energies",{})
        self.decisions = {a.id for a in _decision_aircraft(snapshot)}
        self.budget=budget
        self.rejection_reasons=Counter()
        self.rejections_by_aircraft={}
        self.pad_rejections={}

    def _time_ok(self): return self.budget is None or self.budget.evaluation_available()

    def _time_rejection(self):
        return ValidationResult(False,"UNKNOWN",self.budget.evaluation_stop_reason())

    def _observe(self,result,columns):
        if not result.accepted:
            reason=getattr(result,"reason",None) or result.status
            self.rejection_reasons[reason]+=1
            for c in columns:
                self.rejections_by_aircraft.setdefault(c.aircraft_id,Counter())[reason]+=1
                for pad in {p.pad_id for p in c.plans if getattr(p,"pad_id",None) is not None}:
                    row=self.pad_rejections.setdefault(pad,Counter())
                    row["unknown_count" if result.status=="UNKNOWN" else "explicit_conflict_count"]+=1
                    row["reason:"+reason]+=1
        return result

    def protect_times(self, p):
        key=(self.snapshot.config.config_hash,p.trajectory_hash)
        if key in self.protection:
            self.cache_stats["protection_hits"]+=1
            return self.protection[key]
        self.cache_stats["protection_misses"]+=1
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
        _bounded_put(self.protection,key,(clear,capture))
        return clear, capture

    def energy_portions(self,portions):
        key=(self.snapshot.config.config_hash,tuple((p.trajectory_hash,s,e) for p,s,e in portions))
        if key in self.energies:
            self.cache_stats["energy_hits"]+=1
            return self.energies[key]
        self.cache_stats["energy_misses"]+=1
        value=_energy_portions(portions,self.snapshot.config)
        if all(math.isfinite(x) for x in value): _bounded_put(self.energies,key,value)
        return value

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
                rows.append(Reservation(("pad",p.pad_id), a.id, p.end_time-13.5, p.end_time+service_duration(self.snapshot.config,a.service_class)))
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
        end = final.end_time+service_duration(self.snapshot.config,a.service_class) if final.pad_id is not None else math.inf
        if a.status == "LANDED_SERVICE": end = a.service_end
        return _Traffic(a.id,base,plans,self.snapshot.time,end)

    @staticmethod
    def resource_conflict(a, b):
        return any(x.resource == y.resource and x.start < y.end and y.start < x.end for x in a.resources for y in b.resources)

    def check_pair(self, a, b):
        if not self._time_ok(): return self._time_rejection()
        key = (self.snapshot_key,tuple(sorted(((a.signature,a.resources),(b.signature,b.resources)))))
        if key in self.pairs:
            self.cache_stats["pair_hits"]+=1
            return self.pairs[key]
        self.cache_stats["pair_misses"]+=1
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
            if end>self.snapshot.time:
                with (self.budget.measure("trajectory_validation") if self.budget else nullcontext()):
                    result=validate_separation(ta,tb,self.snapshot.time,end,self.snapshot.config)
            else: result=ValidationResult(True,"SAFE")
        if result.status!="UNKNOWN": _bounded_put(self.pairs,key,result)
        return result

    def check_single(self, column):
        if not self._time_ok(): return self._observe(self._time_rejection(),(column,))
        key=((self.snapshot_key,column.signature,column.resources)
             if all(getattr(p,"trajectory_hash",None) for p in column.plans) else None)
        if key is not None and key in self.singles:
            self.cache_stats["single_hits"]+=1
            return self._observe(self.singles[key],(column,))
        self.cache_stats["single_misses"]+=1
        result=self._check_single(column)
        if key is not None and result.status!="UNKNOWN": _bounded_put(self.singles,key,result)
        return result

    def _check_single(self, column):
        # Only the first action that prefix() would actually submit is guarded;
        # a future normal-final suffix cannot suppress an inward transfer.
        if (column.kind=="MOVE" and column.plans and column.aircraft_id not in self.snapshot.soft
            and column.plans[0].kind=="FINAL_NORMAL"
            and column.plans[0].start_time<=self.snapshot.time+5.+1.e-8):
            from .world import emergency_priority_blocker, STAGE2_POLICY_ID
            if (self.snapshot.emergency_monitor or {}).get("experiment_policy_id")==STAGE2_POLICY_ID:
                kwargs={}
                if _v6(self.snapshot):
                    from .rescue_rules import RESCUE_PROFILE_V6
                    kwargs=dict(profile=RESCUE_PROFILE_V6,proposed_plan=column.plans[0],
                        protected_plans=tuple(self.snapshot.soft.values()))
                blocked=emergency_priority_blocker(self.snapshot.aircraft,self.snapshot.active,
                    self.snapshot.config,column.aircraft_id,**kwargs)
                if blocked:
                    return self._observe(ValidationResult(False,"REJECTED","EMERGENCY_PRIORITY_WAIT",blocked),(column,))
        # Other decision aircraft are deliberately absent here. Joint checks
        # determine their departure or continued holding.
        for a in self.snapshot.aircraft.values():
            if not self._time_ok(): return self._observe(self._time_rejection(),(column,))
            if a.id == column.aircraft_id or a.id in self.decisions: continue
            if a.status not in ("LANDED_SERVICE","FINAL_NORMAL","FINAL_EMERGENCY","VERTICAL_DESCENT") and not a.status.startswith(("HOLD_","TRANSFER_","INBOUND")): continue
            background = PredictionColumn(a.id,"BACKGROUND",(),self.resources(a,()) or (), (0,0,0,0,0))
            r = self.check_pair(column, background)
            if not r.accepted: return self._observe(r,(column,))
        return ValidationResult(True,"SAFE")

    def check_batch(self, columns):
        result,conflicts=self.check_batch_conflicts(columns,max_conflicts=1)
        return result,conflicts[0] if conflicts else ()

    def check_batch_conflicts(self, columns, max_conflicts=MAX_BATCH_CONFLICTS):
        """Collect explicit conflicts; uncertainty never becomes an exclusion.

        The legacy first-conflict API remains available for hint construction.
        Selected CP batches amortize single checks and add several cuts at once.
        """
        columns=tuple(columns); conflicts=[]; unknown=None; first=None
        max_conflicts=max(1,min(MAX_BATCH_CONFLICTS,max_conflicts))
        for c in columns:
            if not self._time_ok():
                unknown=self._observe(self._time_rejection(),columns); break
            r = self.check_single(c)
            if not r.accepted:
                if r.status=="UNKNOWN": unknown=r
                else:
                    conflicts.append((c,)); first=first or r
                    if len(conflicts)>=max_conflicts: break
        if len(conflicts)>=max_conflicts:
            self.cache_stats["batch_conflicts"]+=len(conflicts)
            return first,tuple(conflicts)
        for a,b in combinations(columns,2):
            if not self._time_ok():
                unknown=self._observe(self._time_rejection(),columns); break
            r = self.check_pair(a,b)
            if not r.accepted:
                self._observe(r,(a,b))
                if r.status=="UNKNOWN": unknown=r
                else:
                    conflicts.append((a,b)); first=first or r
                    if len(conflicts)>=max_conflicts: break
        self.cache_stats["batch_conflicts"]+=len(conflicts)
        return first or unknown or ValidationResult(True,"SAFE"),tuple(conflicts)

    def prefix(self, columns):
        out = []
        for c in columns:
            if c.kind in ("KEEP","WAIT") or not c.plans: continue
            p = c.plans[0]
            if p.start_time <= self.snapshot.time+5.+1.e-8 and p.aircraft_id not in self.snapshot.soft:
                out.append(p)
        return tuple(out)

    def check_commit_prefix(self, columns):
        if not self._time_ok(): return self._observe(self._time_rejection(),columns)
        prefix = self.prefix(columns)
        with (self.budget.measure("prefix_validation") if self.budget else nullcontext()):
            if self.prefix_evaluator is not None: return self._observe(self.prefix_evaluator(prefix),columns)
            from .world import evaluate_planning_snapshot
            return self._observe(evaluate_planning_snapshot(self.snapshot,prefix),columns)


class CandidateProvider:
    def __init__(self, reference_provider=None, planning_profile=PLANNING_PROFILE_LEGACY, advisor_runtime=None):
        candidate_policy_id(planning_profile)
        self.planning_profile=planning_profile
        self.references = reference_provider or ReferenceProvider()
        self.previous = {}
        self.aligned_starts=[]
        self._aligned_by_node={}
        self.validation_cache={}
        self.advisor_runtime=advisor_runtime if advisor_runtime is not None and advisor_runtime.enabled else None
        self.references.advisor_runtime=self.advisor_runtime

    def _pop_ready(self, heap, snapshot, a, budget, stage, extract, is_ready=None):
        """Select from at most 64 existing ready items before generation.

        Structural heap entries are still expanded by Stage2. Every descriptor
        supplied here has a materialized node and a concrete generator tuple.
        Only one work item is consumed per rank, so coverage/parent changes
        invalidate the ranking before the next primitive attempt.
        """
        runtime=self.advisor_runtime
        if runtime is None or (is_ready is not None and not is_ready(heap[0])):
            return heapq.heappop(heap),None
        ready=heapq.nsmallest(64,(item for item in heap if is_ready is None or is_ready(item)))
        descriptors=[]
        for item in ready:
            node,chain,raw=extract(item)
            target,start,duration,base=raw
            descriptors.append(runtime.descriptor(snapshot,a,node,target,start,duration,stage,chain,base))
        ranked=runtime.order(snapshot,descriptors,budget)
        chosen=ready[descriptors.index(ranked[0])]
        index=heap.index(chosen)
        heap[index]=heap[-1];heap.pop()
        if heap: heapq.heapify(heap)
        return chosen,ranked[0]

    @staticmethod
    def _stop_prediction(snapshot,state):
        return state=="RESCUE_REQUIRED" if _v6(snapshot) else state in ("ELIGIBLE","UNKNOWN")

    def _emergency_prediction(self,snapshot,a,node):
        if _v6(snapshot):
            from .rescue_rules import assess_rescue, normal_energy_lower_bound
            rate=stable_power(node.ring,snapshot.config).current_a/(snapshot.config.energy.reference_capacity_Ah*3600.)
            key=(snapshot.config.config_hash,node.ring,node.slot,node.time,node.version,a.id)
            reference=self.references.cache.get(key)
            assessment=assess_rescue(node.soc,rate,
                profile=(snapshot.emergency_monitor or {}).get("rescue_rule_profile"),
                normal_energy_lower_bound=normal_energy_lower_bound(snapshot.config,node.ring),
                normal_path_soc=reference.soc_required if reference and reference.status=="KNOWN" else None)
            if assessment.rescue_required: return "RESCUE_REQUIRED"
            if assessment.normal_status=="UNKNOWN": return "WARNING_UNKNOWN" if assessment.warning else "UNKNOWN"
            return "WARNING" if assessment.warning else "KNOWN_CLEAR"
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
            q = self.references.move_energy(current,p,snapshot.config)
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
        qh = context.energy_portions(portions)
        soc_h = a.soc-qh[1]; peak_h = a.peak_seconds+qh[2]
        f = int(soc_h <= 0 or peak_h > 120.)
        l = int(landed and allplans[-1].end_time <= snapshot.time+HORIZON)
        if landed:
            u=0; tau=min(900.,allplans[-1].end_time-snapshot.time)
        else:
            tailtime=max(snapshot.time+HORIZON,current.time)
            tailkey=(snapshot.config.config_hash,current.ring,current.slot,tailtime,current.version,a.id)
            ref=self.references.cache.get(tailkey,NormalReference("UNKNOWN",identity=tailkey))
            # Constructing a partial column never starts a recursive reference
            # search. The source-reference pass below does not populate these
            # horizon-tail epochs; only an existing exact cache hit is KNOWN.
            reference=ref
            known = ref is not None and ref.status == "KNOWN"
            future_soc=min(soc_h,current.soc);future_peak=max(peak_h,current.peak)
            urgent = not known or future_soc-.25-ref.soc_required < 0 or 120.-future_peak < ref.peak_required
            u = ({"inner":1,"middle":2,"outer":3}[ring] if known else 3) if urgent else 0
            # Authorized progress-tail objective v2: UNKNOWN stays U=3 and
            # strictly worse than known tau, with a necessary-time heuristic
            # from the actual predicted terminal ring (not a certificate).
            tau = min(900.,HORIZON+ref.duration) if known else 901.+self._remaining_lower(snapshot,ring)
        rows = context.resources(a,plans)
        if rows is None: return None
        c=PredictionColumn(a.id,kind,tuple(plans),rows,(f,-l,u,tau,qh[0]),current,reference,emergency,energy,peak)
        if self.planning_profile==PLANNING_PROFILE_DOWNSTREAM:
            from .stage2_downstream import describe_column
            d=describe_column(snapshot,c,self.references,context)
            c=replace(c,downstream=d.diagnostics,objective=(f,-l,d.unknown+max(0,d.base_energy-d.reserve_energy)*.1/3600.,d.base_energy*.1,(d.touchdown-round(snapshot.time*10))/10.))
        return c

    _remaining_lower = staticmethod(remaining_normal_time_lower)

    def _paths(self,snapshot,a,node,budget,context,prefix=(),accepted_counts=None):
        """Materialized chains and unconstructed all-target work share one heap."""
        frontier=[]; serial=0;groups=[]
        root_counts=accepted_counts if accepted_counts is not None else Counter()
        lower={ring:self._remaining_lower(snapshot,ring) for ring in ("outer","middle","inner")}
        def chain_signature(chain):
            return tuple((p.target_ring or "pad",p.target_slot if p.target_slot is not None else p.pad_id,p.start_time,p.end_time) for p in chain)
        def emitted_column(plans):
            # Catalog construction supplies accepted counts, updated only after
            # check_single succeeds. Standalone enumeration has no such caller
            # and bounds its own emitted columns instead.
            if accepted_counts is None:
                p=plans[0]
                root_counts[(p.target_ring,p.target_slot,p.pad_id)]+=1
        def estimate(n):
            ref=self.references.cache.get((snapshot.config.config_hash,n.ring,n.slot,n.time,n.version,a.id))
            return n.time+(ref.duration if ref and ref.status=="KNOWN" else lower[n.ring])
        def push_state(n,chain,energy):
            nonlocal serial
            serial+=1
            heapq.heappush(frontier,((estimate(n),energy,chain_signature(chain)),serial,"STATE",n,chain,energy,None))
        def push_cursor(current,chain,energy,group,cursor):
            nonlocal serial
            target=cursor["target"]
            first=(None,None,target) if current.ring=="inner" else ({"outer":"middle","middle":"inner"}[current.ring],target,None)
            if not chain and root_counts[first]>=2 and self.planning_profile==PLANNING_PROFILE_LEGACY: return
            if cursor["aligned_end"] is None and group["feasible"][(target,cursor["start"])]>=2: return
            descriptor=self._next_descriptor(snapshot,a,current,budget,cursor,group["seen"],group["immediate"])
            if descriptor is None: return
            target,start,duration,base=descriptor
            next_ring=None if current.ring=="inner" else {"outer":"middle","middle":"inner"}[current.ring]
            end=start+duration+(13.5 if next_ring is None else 0.)
            eta=end+(lower[next_ring] if next_ring is not None else 0.)
            signature=(*chain_signature(chain),(next_ring or "pad",target,start,end))
            serial+=1; group["remaining"]+=1
            heapq.heappush(frontier,((eta,energy,signature),serial,"TEMPLATE",current,chain,energy,(descriptor,group,cursor)))
        initial_energy=self.references._reference(snapshot,node,prefix,"ordering").energy_j if prefix else 0.
        push_state(node,prefix,initial_energy)
        while frontier and budget.available(a.id):
            item,work_descriptor=self._pop_ready(frontier,snapshot,a,budget,"alternative",
                lambda item:(item[3],item[4],item[6][0]),lambda item:item[2]=="TEMPLATE")
            _,_,tag,current,chain,energy,work=item
            if tag=="STATE":
                if chain and (current.time>=snapshot.time+HORIZON or len(chain)>=3):
                    emitted_column(chain)
                    yield ("COLUMN",chain,"UNKNOWN"); continue
                state=self._emergency_prediction(snapshot,a,current) if chain else "NONE"
                if chain and current.ring!="inner" and self._stop_prediction(snapshot,state):
                    emitted_column(chain)
                    yield ("COLUMN",chain,state); continue
                immediate=bool(chain and current.ring=="inner" and self._stop_prediction(snapshot,state))
                group=dict(feasible=Counter(),remaining=0,emitted=False,immediate=immediate,state=state,chain=chain,seen=set(),enumerating=True)
                groups.append(group)
                offsets=tuple(offset for offset in self._search_offsets(snapshot,a,current) if not immediate or offset==0.)
                if offsets:
                    serial+=1
                    heapq.heappush(frontier,((estimate(current)+offsets[0],energy,chain_signature(chain)),serial,"ENUMERATE",current,chain,energy,(group,offsets,0)))
                else:
                    group["enumerating"]=False
                    if immediate:
                        group["emitted"]=True
                        yield ("COLUMN",chain,state)
                continue
            if tag=="ENUMERATE":
                group,offsets,index=work
                for cursor in self._offset_cursors(snapshot,a,current,budget,offsets[index],include_aligned=index==0):
                    push_cursor(current,chain,energy,group,cursor)
                if index+1<len(offsets) and budget.available(a.id):
                    serial+=1
                    heapq.heappush(frontier,((estimate(current)+offsets[index+1],energy,chain_signature(chain)),serial,"ENUMERATE",current,chain,energy,(group,offsets,index+1)))
                else:
                    group["enumerating"]=False
                if not group["remaining"] and not group["enumerating"] and group["immediate"]:
                    group["emitted"]=True
                    yield ("COLUMN",chain,group["state"])
                yield ("SCREEN",None,None)
                continue
            descriptor,group,cursor=work
            group["remaining"]-=1
            target,start,duration,base=descriptor
            aligned=self._aligned_by_node.setdefault((a.id,current.ring,current.slot,current.time,current.version),set())
            first=(None,None,target) if current.ring=="inner" else ({"outer":"middle","middle":"inner"}[current.ring],target,None)
            allowed=(group["feasible"][(target,start)]<2 and
                     (bool(chain) or root_counts[first]<2 or self.planning_profile==PLANNING_PROFILE_DOWNSTREAM) and
                     (start==base or start in aligned or len(aligned)<2))
            if allowed:
                if self.advisor_runtime is None:
                    ok,p=self.references.generate(snapshot,a,current,target,start,duration,budget)
                else:
                    ok,p=self.references.generate(snapshot,a,current,target,start,duration,budget,
                        construction_stage="alternative",prefix=chain,base=base,work_descriptor=work_descriptor)
                if not ok: break
                yield ("ATTEMPT",None,None)
                if p is not None and p.end_time<=snapshot.time+900:
                    group["feasible"][(target,start)]+=1
                    if start!=base and start not in aligned:
                        aligned.add(start); self.aligned_starts.append((a.id,current.ring,current.slot,current.time,start))
                    q=self.references.move_energy(current,p,snapshot.config)
                    if current.soc-q[1]>=(.25 if p.pad_id is not None else 0.) and current.peak+q[2]<=120:
                        plans=(*chain,p)
                        if p.pad_id is not None:
                            emitted_column(plans)
                            group["emitted"]=True
                            predicted=self._emergency_prediction(snapshot,a,current)
                            yield ("COLUMN",plans,predicted+"_IMMEDIATE_NORMAL_EXCEPTION" if group["immediate"] else predicted)
                        else:
                            child=PlanningNode(p.target_ring,p.target_slot,p.end_time,current.soc-q[1],current.peak+q[2],current.version+1,current.error_j+q[3],current.peak_error_s+q[4])
                            push_state(child,plans,energy+q[0])
                            # Retain executable progress independently of a
                            # later pad/slot conflict, then continue this chain.
                            emitted_column(plans)
                            yield ("COLUMN",plans,self._emergency_prediction(snapshot,a,child))
            push_cursor(current,chain,energy,group,cursor)
            if not group["remaining"] and not group["enumerating"] and group["immediate"] and not group["emitted"]:
                group["emitted"]=True
                yield ("COLUMN",chain,group["state"])
        for group in groups:
            if group["immediate"] and not group["emitted"]:
                group["emitted"]=True
                yield ("COLUMN",group["chain"],group["state"])

    def _reference_seed(self,snapshot,a,node,reference,budget):
        """Replay a normal reference with the path search's executable rules.

        References measure normal energy even when that chain cannot be flown
        with the current SOC, predicted emergency state, or fixed background.
        They therefore enter the catalog only through this separate replay.
        """
        current,chain=node,()
        for p in reference.plans:
            if chain and (current.time>=snapshot.time+HORIZON or len(chain)>=3):
                return chain,"UNKNOWN"
            state=self._emergency_prediction(snapshot,a,current) if chain else "NONE"
            if chain and current.ring!="inner" and self._stop_prediction(snapshot,state):
                return chain,state
            immediate=bool(chain and current.ring=="inner" and self._stop_prediction(snapshot,state))
            if (p.source_ring,p.source_slot,p.source_version)!=(current.ring,current.slot,current.version) or p.start_time<current.time:
                budget.screening["seed_state_rejected"]+=1
                return None
            if immediate and abs(p.start_time-current.time)>1.e-8:
                return chain,state
            if p.end_time>snapshot.time+900:
                budget.screening["seed_horizon_rejected"]+=1
                return (chain,state) if immediate else None
            q=self.references.move_energy(current,p,snapshot.config)
            if not (current.soc-q[1]>=(.25 if p.pad_id is not None else 0.) and current.peak+q[2]<=120.) or not math.isfinite(q[0]):
                budget.screening["seed_energy_rejected"]+=1
                return (chain,state) if immediate else None
            chain=(*chain,p)
            if p.pad_id is not None:
                predicted=self._emergency_prediction(snapshot,a,current)
                return chain,predicted+"_IMMEDIATE_NORMAL_EXCEPTION" if immediate else predicted
            current=PlanningNode(p.target_ring,p.target_slot,p.end_time,current.soc-q[1],current.peak+q[2],current.version+1,current.error_j+q[3],current.peak_error_s+q[4])
        return None

    def _search_offsets(self,snapshot,a,node):
        if self.planning_profile==PLANNING_PROFILE_LEGACY: return OFFSETS
        if node.ring=="inner": return OFFSETS
        next_ring={"outer":"middle","middle":"inner"}[node.ring]
        starts={r.end for r in snapshot.slots if isinstance(r.resource,tuple) and r.resource[0]==next_ring and math.isfinite(r.end)}
        # Back-project observed pad release through the cached normal travel
        # time when available; otherwise the necessary-time proxy is explicit.
        exact=self.references.cache.get((snapshot.config.config_hash,node.ring,node.slot,node.time,node.version,a.id))
        flight=exact.duration if exact and exact.status=="KNOWN" else self._remaining_lower(snapshot,node.ring)
        final_flight=(exact.plans[-1].end_time-exact.plans[-1].start_time if exact and exact.status=="KNOWN" and exact.plans else self._remaining_lower(snapshot,"inner"))
        releases=sorted({r.end for r in snapshot.pads if math.isfinite(r.end) and r.end>node.time})
        # The occupancy boundary and final-approach readiness boundary differ:
        # ground aircraft can conflict with an approaching final before descent.
        # Both epochs are regenerated and checked by the original validator.
        if releases:
            starts.add(releases[0]-flight+13.5)
            starts.add(releases[0]-max(0.,flight-final_flight))
        viable=sorted(x-node.time for x in starts if node.time+1.e-8<x<=snapshot.time+900.)
        return (0.,*viable[:2])

    def _offset_cursors(self,snapshot,a,node,budget,offset,include_aligned=True):
        """One offset's all-target cursors; only their next template is queued."""
        targets=range(6 if node.ring=="inner" else ring_config({"outer":"middle","middle":"inner"}[node.ring],snapshot.config).slot_count)
        rows=snapshot.pads if node.ring=="inner" else snapshot.slots
        next_ring=None if node.ring=="inner" else {"outer":"middle","middle":"inner"}[node.ring]
        opportunities=sorted({(r.end,str(r.resource),r.resource) for r in rows
                              if math.isfinite(r.end) and r.end>node.time and
                              (next_ring is None or (isinstance(r.resource,tuple) and r.resource[0]==next_ring))})[:2] if include_aligned and (self.planning_profile==PLANNING_PROFILE_LEGACY or node.ring=="inner") else ()
        self._aligned_by_node.setdefault((a.id,node.ring,node.slot,node.time,node.version),set())
        base=node.time+offset
        screen=self.references.screen(snapshot,node,base,budget)
        if screen is None: return
        for target in targets:
            yield dict(target=target,start=base,base=base,aligned_end=None,screen=screen,
                       durations=iter(range(20,301 if node.ring=="inner" else 181,5)))
            resource=target if node.ring=="inner" else (next_ring,target)
            for end,_,r in opportunities:
                if r!=resource: continue
                yield dict(target=target,start=end,base=base,
                           aligned_end=end if node.ring=="inner" else None,screen=screen,
                           durations=iter(range(20,301 if node.ring=="inner" else 181,5)))

    def _next_descriptor(self,snapshot,a,node,budget,cursor,seen,immediate_only=False):
        aligned=self._aligned_by_node[(a.id,node.ring,node.slot,node.time,node.version)]
        target,base=cursor["target"],cursor["base"]
        for duration in cursor["durations"]:
            if not budget.available(a.id): return None
            start=cursor["aligned_end"]-duration if cursor["aligned_end"] is not None else cursor["start"]
            if start<node.time or start>snapshot.time+900: continue
            if immediate_only and abs(start-node.time)>1.e-8: continue
            if start!=base and start not in aligned and len(aligned)>=2: continue
            if (target,start,duration) in seen: continue
            seen.add((target,start,duration))
            # Transfer geometry is epoch invariant. Aligned final epochs
            # remain UNKNOWN; every retained item uses the original generator.
            if (start==base or node.ring!="inner") and cursor["screen"][(target,duration)] is not None: continue
            budget.screening["descriptors_queued"]+=1
            return target,start,duration,base
        return None

    def _descriptors(self,snapshot,a,node,budget,immediate_only=False):
        # Compatibility iterator. The heap search advances these cursors after
        # the precise generator result, rather than enqueuing their full grids.
        seen=set()
        for index,offset in enumerate(self._search_offsets(snapshot,a,node)):
            if immediate_only and offset!=0.: break
            for cursor in self._offset_cursors(snapshot,a,node,budget,offset,include_aligned=index==0):
                while budget.available(a.id):
                    descriptor=self._next_descriptor(snapshot,a,node,budget,cursor,seen,immediate_only)
                    if descriptor is None: break
                    yield descriptor

    def _successors(self,snapshot,a,node,budget,root_counts=None,immediate_only=False):
        # Compatibility iterator: the path search itself queues descriptors so
        # an early target's subtree cannot hide the rest of the source frontier.
        feasible=Counter()
        for target,start,duration,base in self._descriptors(snapshot,a,node,budget,immediate_only):
            first=(None,None,target) if node.ring=="inner" else ({"outer":"middle","middle":"inner"}[node.ring],target,None)
            if feasible[(target,start)]>=2 or (root_counts is not None and root_counts[first]>=2): continue
            aligned=self._aligned_by_node[(a.id,node.ring,node.slot,node.time,node.version)]
            if start!=base and start not in aligned and len(aligned)>=2: continue
            ok,p=self.references.generate(snapshot,a,node,target,start,duration,budget)
            if not ok: return
            if p is None or p.end_time>snapshot.time+900:
                yield None; continue
            feasible[(target,start)]+=1
            if start!=base and start not in aligned:
                aligned.add(start); self.aligned_starts.append((a.id,node.ring,node.slot,node.time,start))
            yield p

    @staticmethod
    def _current_progress_reservation_conflict(snapshot,a,node,target,start,duration):
        """Prefer independently available current targets in the progress pass.

        A row may be released by another decision, so skipping it here says
        nothing about joint feasibility. This independent first pass leaves
        such moves to the unfiltered reference/multistep joint enumeration.
        Transfer capture precedes end_time; overlap even after end_time is a
        guaranteed current-reservation dependency for this single step.
        """
        next_ring={"outer":"middle","middle":"inner","inner":None}[node.ring]
        resource=target if next_ring is None else (next_ring,target)
        rows=snapshot.pads if next_ring is None else snapshot.slots
        lo=start+duration
        hi=lo+13.5+service_duration(snapshot.config,a.service_class) if next_ring is None else math.inf
        return any(r.resource==resource and r.aircraft_id!=a.id
                   and lo<r.end and r.start<hi for r in rows)

    def _progress_steps(self,snapshot,a,node,budget,context):
        """Immediate primitives in end-time order, yielding after each attempt.

        Only one accepted progress column per aircraft is needed in this pass.
        The caller owns the half-wall allocation and round-robin scheduling;
        full references and multistep expansion retain the remaining budget.
        """
        heap=[];serial=0;seen=set()
        def push(cursor):
            nonlocal serial
            descriptor=self._next_descriptor(snapshot,a,node,budget,cursor,seen,True)
            if descriptor is None: return
            target,start,duration,_=descriptor
            serial+=1
            heapq.heappush(heap,(start+duration,target,serial,descriptor,cursor))
        for cursor in self._offset_cursors(snapshot,a,node,budget,0.,include_aligned=False):
            push(cursor)
        yield ("SCREEN",None,None)
        while heap and budget.available(a.id):
            item,work_descriptor=self._pop_ready(heap,snapshot,a,budget,"progress",lambda item:(node,(),item[3]))
            _,_,_,descriptor,cursor=item
            target,start,duration,_=descriptor
            if self._current_progress_reservation_conflict(snapshot,a,node,target,start,duration):
                budget.screening["progress_existing_reservation_skipped"]+=1
                push(cursor)
                continue
            before=budget.attempts[a.id]
            if self.advisor_runtime is None:
                ok,p=self.references.generate(snapshot,a,node,target,start,duration,budget)
            else:
                ok,p=self.references.generate(snapshot,a,node,target,start,duration,budget,
                    construction_stage="progress",work_descriptor=work_descriptor)
            budget.screening["progress_attempts"]+=budget.attempts[a.id]-before
            if not ok: return
            push(cursor)
            if p is not None:
                q=self.references.move_energy(node,p,snapshot.config)
                if node.soc-q[1]>=(.25 if p.pad_id is not None else 0.) and node.peak+q[2]<=120:
                    terminal=(PlanningNode(p.target_ring,p.target_slot,p.end_time,node.soc-q[1],node.peak+q[2],
                                          node.version+1,node.error_j+q[3],node.peak_error_s+q[4])
                              if p.target_ring is not None else node)
                    yield ("COLUMN",(p,),self._emergency_prediction(snapshot,a,terminal))
                    continue
                budget.screening["progress_energy_rejected"]+=1
            yield ("ATTEMPT",None,None)

    def _delayed_steps(self,snapshot,a,node,budget,context):
        """A bounded independent delayed primitive pass before reference work."""
        if node.ring=="inner": return
        heap=[]; serial=0; seen=set()
        def push(cursor):
            nonlocal serial
            descriptor=self._next_descriptor(snapshot,a,node,budget,cursor,seen)
            if descriptor is not None:
                serial+=1
                heapq.heappush(heap,(descriptor[1]+descriptor[2],serial,descriptor,cursor))
        for offset in self._search_offsets(snapshot,a,node)[1:]:
            for cursor in self._offset_cursors(snapshot,a,node,budget,offset,include_aligned=False): push(cursor)
        while heap and budget.available(a.id):
            item,work_descriptor=self._pop_ready(heap,snapshot,a,budget,"alternative",lambda item:(node,(),item[2]))
            _,_,(target,start,duration,base),cursor=item
            if self.advisor_runtime is None:
                ok,p=self.references.generate(snapshot,a,node,target,start,duration,budget)
            else:
                ok,p=self.references.generate(snapshot,a,node,target,start,duration,budget,
                    construction_stage="alternative",base=base,work_descriptor=work_descriptor)
            if not ok: return
            push(cursor)
            if p is not None:
                q=self.references.move_energy(node,p,snapshot.config)
                if node.soc-q[1]>=0. and node.peak+q[2]<=120.:
                    yield ("COLUMN",(p,),"DOWNSTREAM_DELAY_ESTIMATE")
            yield ("ATTEMPT",None,None)

    def _delayed_reference_seed(self,snapshot,a,node,reference,budget):
        """Regenerate each leg at the new epoch; no translated certificates."""
        if node.ring=="inner" or reference.status!="KNOWN": return
        for offset in self._search_offsets(snapshot,a,node)[1:]:
            current=node; chain=[]; start=node.time+offset
            for old in reference.plans:
                target=old.pad_id if old.pad_id is not None else old.target_slot
                duration=old.end_time-old.start_time-(13.5 if old.pad_id is not None else 0.)
                options=[(target,duration)]
                if old.pad_id is not None:
                    screen=self.references.screen(snapshot,current,start,budget)
                    if screen is None: break
                    options=sorted(((pad,d) for (pad,d),witness in screen.items()
                                    if d>=duration-1.e-8 and witness is None),key=lambda item:(item[1],item[0]))
                p=None; ok=True
                for target,duration in options:
                    if self.advisor_runtime is None:
                        ok,p=self.references.generate(snapshot,a,current,target,start,duration,budget)
                    else:
                        ok,p=self.references.generate(snapshot,a,current,target,start,duration,budget,
                            construction_stage="reference",prefix=tuple(chain))
                    if not ok or p is not None: break
                if not ok or p is None: break
                q=self.references.move_energy(current,p,snapshot.config)
                if current.soc-q[1] < (.25 if p.pad_id is not None else 0.) or current.peak+q[2]>120.: break
                chain.append(p)
                if p.pad_id is not None:
                    yield tuple(chain),"DOWNSTREAM_DELAY_REGENERATED_CHAIN"
                    break
                current=PlanningNode(p.target_ring,p.target_slot,p.end_time,current.soc-q[1],current.peak+q[2],
                                     current.version+1,current.error_j+q[3],current.peak_error_s+q[4])
                start=current.time

    def _previous_chain(self,snapshot,a,node,budget):
        """Trim executed/fixed prefixes without moving any certified epoch."""
        previous=self.previous.get(a.id,())
        fixed={p.trajectory_hash for p in (snapshot.active.get(a.id),snapshot.soft.get(a.id)) if p is not None}
        plans=tuple(p for p in previous if p.trajectory_hash not in fixed and p.end_time>node.time+1.e-8)
        if not plans: return ()
        current=node
        for p in plans:
            if (p.config_hash!=snapshot.config.config_hash or p.aircraft_id!=a.id or
                (p.source_ring,p.source_slot,p.source_version)!=(current.ring,current.slot,current.version) or
                p.start_time<current.time-1.e-8):
                budget.screening["previous_state_invalidations"]+=1
                return ()
            q=self.references.move_energy(current,p,snapshot.config)
            if not (all(math.isfinite(x) for x in q) and current.soc-q[1]>=(.25 if p.pad_id is not None else 0.)
                    and current.peak+q[2]<=120.):
                budget.screening["previous_energy_invalidations"]+=1
                return ()
            if p.target_ring is not None:
                current=PlanningNode(p.target_ring,p.target_slot,p.end_time,current.soc-q[1],current.peak+q[2],
                    current.version+1,current.error_j+q[3],current.peak_error_s+q[4])
        budget.screening["previous_chain_hits"]+=1
        budget.screening["previous_prefix_trimmed"]+=len(previous)-len(plans)
        if self.advisor_runtime is not None:
            self.references._register_reused_chain(snapshot,a,node,plans,budget)
        return plans

    @staticmethod
    def _retain_column(columns,column,planning_profile=PLANNING_PROFILE_LEGACY):
        """Bound alternatives while retaining full chains and useful progress."""
        if column.signature in {c.signature for c in columns}: return False
        moves=[c for c in columns if c.kind=="MOVE" and c.plans]
        same=[c for c in moves if c.first_target==column.first_target]
        if len(columns)<10 and len(same)<2:
            columns.append(column); return True
        full=lambda c:c.plans[-1].pad_id is not None
        full_count=sum(full(c) for c in moves); partial_count=len(moves)-full_count
        choices=same if len(same)>=2 else moves
        choices=[c for c in choices if not (full(c) and full_count==1 and not full(column))
            and not (not full(c) and partial_count==1 and full(column))]
        if planning_profile==PLANNING_PROFILE_DOWNSTREAM:
            delayed=lambda c:c.downstream.get("candidate_timing")=="DELAYED"
            delayed_count=sum(delayed(c) for c in moves)
            choices=[c for c in choices if not (delayed(c) and delayed_count==1 and not delayed(column))
                and not (not delayed(c) and len(moves)-delayed_count==1 and delayed(column))]
        if not choices: return False
        # At capacity, preserve target/start coverage before adding durations.
        counts=Counter((c.first_target,c.plans[0].start_time,full(c)) for c in moves)
        def priority(c):
            category_needed=(full(column) and full_count==0) or (not full(column) and partial_count==0)
            same_category=full(c)==full(column)
            duplicate=counts[(c.first_target,c.plans[0].start_time,full(c))]>1
            return (category_needed,duplicate,c.first_target==column.first_target,not same_category,c.objective)
        replaceable=max(choices,key=priority)
        # A different duration alone does not evict a unique target/start.
        new_coverage=(column.first_target,column.plans[0].start_time,full(column)) not in counts
        missing_category=(full(column) and full_count==0) or (not full(column) and partial_count==0)
        if not (new_coverage or missing_category or column.objective<replaceable.objective): return False
        columns.remove(replaceable); columns.append(column)
        return True

    def get_columns(self,snapshot,budget=None,context=None):
        budget=budget or PlanningBudget(); context=context or EvaluationContext(snapshot,budget=budget)
        self.aligned_starts=[]
        self._aligned_by_node={}
        start_hits=self.references.hits
        aircraft=_decision_aircraft(snapshot)
        runtime=self.advisor_runtime
        if runtime is not None: runtime.begin_snapshot(snapshot,budget,[a.id for a in aircraft])
        refs={}; slacks={}; columns={}; incomplete=[]
        # Immediate progress and full references share the same fixed budget.
        for a in aircraft:
            node=_node(snapshot,a)
            key=(snapshot.config.config_hash,node.ring,node.slot,node.time,node.version,a.id)
            refs[a.id]=self.references.cache.get(key,NormalReference("UNKNOWN"))
            slacks[a.id]=energy_slack(snapshot,a,refs[a.id])
            columns[a.id]=[self._column(snapshot,a,(),"KEEP",context,refs[a.id])]
            columns[a.id].append(self._column(snapshot,a,(),"WAIT",context,refs[a.id]))
            previous=self._previous_chain(snapshot,a,node,budget)
            if previous:
                c=self._column(snapshot,a,previous,"MOVE",context,refs[a.id])
                if c is not None and context.check_single(c).accepted: columns[a.id].append(c)
                else: budget.screening["previous_safety_invalidations"]+=1
        ordered=sorted(aircraft,key=lambda a:slacks[a.id])
        counts={a.id:Counter(c.first_target for c in columns[a.id] if c.kind=="MOVE") for a in ordered}
        def add_column(a,plans,emergency):
            ref=(ReferenceProvider._reference(snapshot,_node(snapshot,a),plans,"catalog_minimum")
                 if plans[-1].pad_id is not None else refs[a.id])
            c=self._column(snapshot,a,plans,"MOVE",context,ref,emergency)
            if c is not None and c.signature not in {x.signature for x in columns[a.id]}:
                validation=context.check_single(c)
                if validation.accepted:
                    retained=self._retain_column(columns[a.id],c,self.planning_profile)
                    if runtime is not None: runtime.record_column(c,validation,retained)
                    if not retained: return False
                    counts[a.id].clear(); counts[a.id].update(x.first_target for x in columns[a.id] if x.kind=="MOVE")
                    if plans[-1].pad_id is not None and (refs[a.id].status!="KNOWN" or ref.energy_j<refs[a.id].energy_j): refs[a.id]=ref
                    return True
                if runtime is not None: runtime.record_column(c,validation,False)
            return False
        by_ring={ring:[a for a in ordered if _node(snapshot,a).ring==ring]
                 for ring in ("outer","middle","inner")}
        seed_order=[group[index] for index in range(max(map(len,by_ring.values()),default=0))
                    for group in by_ring.values() if index<len(group)]
        progress={a.id:self._progress_steps(snapshot,a,_node(snapshot,a),budget,context) for a in seed_order
                  if _node(snapshot,a).time<=snapshot.time+5. and a.id not in snapshot.soft}
        if runtime is not None and runtime.guided:
            # SCREEN is enumeration, not a materialization opportunity. Advance
            # each eligible aircraft to its first real query before learning.
            # Future nodes are the actual endpoints of existing commitments;
            # they remain predictions and still pass the common validator.
            with budget.measure("stage3_initial_coverage"):
                for a in seed_order:
                    if not budget.available(): break
                    gen=progress.get(a.id)
                    if gen is None: gen=self._progress_steps(snapshot,a,_node(snapshot,a),budget,context)
                    while budget.available(a.id):
                        try: tag,plans,emergency=next(gen)
                        except StopIteration:
                            progress.pop(a.id,None); break
                        if tag=="COLUMN":
                            if add_column(a,plans,emergency): progress.pop(a.id,None)
                            break
                        if tag=="ATTEMPT": break
            runtime.complete_initial_coverage()
        progress_start=Counter(budget.attempts)
        progress_share=.25 if self.planning_profile==PLANNING_PROFILE_DOWNSTREAM else .5
        with budget.measure("safe_progress_generation"):
            while progress and budget.available() and budget.elapsed<min(budget.candidate_seconds,budget.soft_seconds)*progress_share:
                for a in seed_order:
                    if budget.elapsed>=min(budget.candidate_seconds,budget.soft_seconds)*progress_share: break
                    gen=progress.get(a.id)
                    if gen is None: continue
                    # The same half allocation reserves physical attempts too,
                    # including under deterministic-clock regression tests.
                    attempt_share=.125 if self.planning_profile==PLANNING_PROFILE_DOWNSTREAM else progress_share
                    if (budget.attempts[a.id]-progress_start[a.id]>=int(budget.max_per_aircraft*attempt_share)
                        or sum(budget.attempts.values())-sum(progress_start.values())>=int(budget.max_attempts*progress_share)):
                        progress.pop(a.id,None);continue
                    try: tag,plans,emergency=next(gen)
                    except StopIteration: progress.pop(a.id,None);continue
                    if tag=="COLUMN":
                        budget.screening["progress_columns_checked"]+=1
                        if add_column(a,plans,emergency):
                            budget.screening["progress_retained"]+=1
                            progress.pop(a.id,None)
            if progress: budget.screening["progress_pass_cutoff"]+=1
        if self.planning_profile==PLANNING_PROFILE_DOWNSTREAM:
            delayed={a.id:self._delayed_steps(snapshot,a,_node(snapshot,a),budget,context) for a in seed_order}
            delayed_start=Counter(budget.attempts)
            with budget.measure("bounded_delayed_generation"):
                while delayed and budget.available() and budget.elapsed<min(budget.candidate_seconds,budget.soft_seconds)*.4:
                    for a in seed_order:
                        gen=delayed.get(a.id)
                        if gen is None: continue
                        if budget.attempts[a.id]-delayed_start[a.id]>=max(1,budget.max_per_aircraft//16):
                            delayed.pop(a.id,None); continue
                        try: tag,plans,emergency=next(gen)
                        except StopIteration: delayed.pop(a.id,None); continue
                        if tag=="COLUMN" and add_column(a,plans,emergency):
                            budget.screening["delayed_retained"]+=1; delayed.pop(a.id,None)
        with budget.measure("reference_seed_generation"):
            for a in seed_order:
                if not budget.available(): break
                gen=self.references.build_steps(snapshot,a,_node(snapshot,a),budget)
                # Spend this aircraft's bounded allowance on a complete chain
                # before moving to the next ring. Alternative paths below keep
                # their original primitive-level round-robin enumeration.
                while budget.available(a.id):
                    try: next(gen)
                    except StopIteration as stop:
                        reference=stop.value
                        if reference.status=="KNOWN":
                            refs[a.id]=reference
                            seed=self._reference_seed(snapshot,a,_node(snapshot,a),reference,budget)
                            if seed is not None: add_column(a,*seed)
                        break
        if self.planning_profile==PLANNING_PROFILE_DOWNSTREAM:
            with budget.measure("delayed_reference_regeneration"):
                for a in seed_order:
                    if not budget.available(a.id): continue
                    for plans,emergency in self._delayed_reference_seed(snapshot,a,_node(snapshot,a),refs[a.id],budget):
                        if add_column(a,plans,emergency): budget.screening["delayed_full_chain_retained"]+=1
        slacks={a.id:energy_slack(snapshot,a,refs[a.id]) for a in ordered}
        ordered=sorted(ordered,key=lambda a:slacks[a.id])
        generators={a.id:self._paths(snapshot,a,_node(snapshot,a),budget,context,accepted_counts=counts[a.id]) for a in ordered}
        while generators and budget.available():
            for a in ordered:
                gen=generators.get(a.id)
                if gen is None: continue
                try: tag,plans,emergency=next(gen)
                except StopIteration: generators.pop(a.id,None); continue
                if tag!="COLUMN": continue
                add_column(a,plans,emergency)
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
            if len(columns[a.id])<3 or a.id in generators or budget.truncations: incomplete.append(a.id)
        downstream=None
        if self.planning_profile==PLANNING_PROFILE_DOWNSTREAM:
            from .stage2_downstream import JointDownstream,describe_column,ticks
            demands={c.signature:describe_column(snapshot,c,self.references,context) for values in columns.values() for c in values}
            fixed=[]; all_order=list(a.id for a in sorted(ordered,key=lambda a:slacks[a.id]))
            # Declared but unreserved rescue traffic remains visible demand.
            # No future manifest or unreleased request is consulted.
            others=[]
            for a in snapshot.aircraft.values():
                if a.id in columns or _node(snapshot,a) is None: continue
                n=_node(snapshot,a)
                c=self._column(snapshot,a,(),"WAIT",context)
                if c is None: continue
                d=describe_column(snapshot,c,self.references,context)
                d=replace(d,unknown=1,diagnostics=dict(d.diagnostics,reference_status="UNKNOWN",reference_unknown_flag=True,
                    uncertainty_risk_penalty_wh=1.,reference_estimate_kind="noncandidate_known_aircraft_proxy"))
                fixed.append(d); others.append((energy_slack(snapshot,a,NormalReference("UNKNOWN")),a.id))
            all_order=sorted(all_order+[aid for _,aid in others],key=lambda aid:slacks.get(aid,next((s for s,i in others if i==aid),())))
            calendars=tuple((r.resource,ticks(r.start),ticks(r.end),r.aircraft_id) for r in snapshot.pads if math.isfinite(r.end))
            downstream=JointDownstream(demands,calendars,all_order,now=ticks(snapshot.time),pad_count=snapshot.config.pads.count,fixed_demands=fixed)
            baseline={aid:values[0] for aid,values in columns.items()}
            for aid,values in columns.items():
                columns[aid]=[replace(c,objective=downstream.key(tuple(dict(baseline,**{aid:c}).values())),downstream=demands[c.signature].diagnostics) for c in values]
        if runtime is not None:
            for values in columns.values():
                for column in values: runtime.record_column(column,retained=True)
            runtime.finish_snapshot(budget)
        return CandidateCatalog(MappingProxyType({k:tuple(v) for k,v in columns.items()}),MappingProxyType(slacks),
                                MappingProxyType(dict(budget.attempts)),tuple(incomplete),self.references.hits-start_hits,
                                MappingProxyType({aid:ref.status for aid,ref in refs.items()}),downstream)


class EnergySlackScheduler:
    method="energy_slack"
    def plan(self,snapshot,candidates,context,budget,*,validation_reserve_seconds=0.):
        chosen={i:cols[0] for i,cols in candidates.columns.items()}
        for i in sorted(chosen,key=lambda i:candidates.slacks[i]):
            if not budget.search_available(validation_reserve_seconds): break
            order_key=(lambda c:(_selection_key(candidates,tuple(dict(chosen,**{i:c}).values())),c.signature)) if candidates.downstream is not None else (lambda c:(c.objective,c.signature))
            for c in sorted(candidates.columns[i],key=order_key):
                if not budget.search_available(validation_reserve_seconds): break
                trial=dict(chosen); trial[i]=c
                r,_=context.check_batch(tuple(trial.values()))
                if r.accepted: chosen=trial; break
        selected=tuple(chosen.values())
        with budget.measure("selected_validation"):
            r,_=(context.check_batch(selected) if budget.evaluation_available() else
                 (ValidationResult(False,"UNKNOWN",budget.evaluation_stop_reason()),()))
            e=context.check_commit_prefix(selected) if r.accepted and budget.evaluation_available() else None
        if e is not None and e.accepted:
            return PlanningResult(selected,context.prefix(selected),e,"FEASIBLE",_selection_key(candidates,selected),certified=True)
        # One local repair: rejected predictive release cannot authorize execution.
        with budget.measure("fallback"):
            keep=tuple(cols[0] for cols in candidates.columns.values())
            e=context.check_commit_prefix(keep) if budget.evaluation_available() else None
        return PlanningResult(keep,(),e,"PREFIX_NOT_EXECUTABLE",_selection_key(candidates,keep),certified=bool(e and e.accepted))


class RollingHorizonCPSATScheduler:
    method="rh_cp_sat"
    def plan(self,snapshot,candidates,context,budget):
        try: from ortools.sat.python import cp_model
        except ImportError as exc:
            raise RuntimeError("RH-CP-SAT requires OR-Tools; no replacement solver is used") from exc
        # Bound the hint on the original normal-dispatch clock. EvaluationContext
        # also observes this phase limit, including nested safety checks.
        previous_budget=getattr(context,"budget",None)
        if isinstance(context,EvaluationContext): context.budget=budget
        hint_started=time.perf_counter()
        try:
            with budget.measure("hint_construction"), budget.stage_limit(
                None if budget.hint_seconds is None else max(0.,budget.hint_seconds-budget.hint_wall_used),
                "HINT_BUDGET",budget.solver_reserve_seconds):
                hint=EnergySlackScheduler().plan(snapshot,candidates,context,budget,
                    validation_reserve_seconds=budget.hint_validation_reserve_seconds)
        finally:
            budget.hint_wall_used+=time.perf_counter()-hint_started
            if isinstance(context,EvaluationContext): context.budget=previous_budget
        hint_key=hint.objective if hint.certified else None
        incumbent=hint if hint.certified else None
        exclusions=set(); single_exclusions=set(); nogoods=set(); model_single_results={}
        records=[]; solve_used=0.; all_optimal=True; final_solver_key=None
        batched_conflicts=0
        ids=tuple(candidates.columns)
        move_aircraft=sum(any(c.kind=="MOVE" and c.plans for c in columns) for columns in candidates.columns.values())
        skip_reason="EMPTY_CATALOG" if not ids else "NO_MOVE_OPTIONS" if not move_aircraft else None
        stop_reason=None; resource_complete=True; model_complete=True
        if skip_reason is None:
            with budget.measure("resource_conflict_build"):
                for i,j in combinations(ids,2):
                    if not budget.preparation_available(): resource_complete=False; break
                    for c in candidates.columns[i]:
                        if not budget.preparation_available(): resource_complete=False; break
                        for d in candidates.columns[j]:
                            if not budget.preparation_available(): resource_complete=False; break
                            if context.resource_conflict(c,d): exclusions.add(tuple(sorted((c.signature,d.signature))))
                        if not resource_complete: break
                    if not resource_complete: break
            for _ in range(max(0,budget.max_refinement_rounds-budget.refinement_rounds_used)):
                if not budget.search_available():
                    stop_reason=budget.evaluation_stop_reason() if not budget.evaluation_available() else "SOFT_BUDGET"
                    break
                if budget.solver_wall_used>=budget.solver_seconds:
                    stop_reason="SOLVER_ALLOCATION_EXHAUSTED"; break
                budget.refinement_rounds_used+=1
                refinement=budget.refinement_rounds_used
                with budget.measure("model_build"):
                    model=cp_model.CpModel(); variables={}; lookup={}
                    # Keep exactly-one constraints for every aircraft even if the
                    # optional model screening must stop to reserve solver time.
                    for i in ids:
                        vs=[]
                        for n,c in enumerate(candidates.columns[i]):
                            v=model.NewBoolVar(f"x_{i}_{n}"); variables[c.signature]=v; lookup[c.signature]=c; vs.append(v)
                        model.AddExactlyOne(vs)
                    for a,b in exclusions: model.Add(variables[a]+variables[b]<=1)
                    for key in single_exclusions: model.Add(variables[key]==0)
                    for keys in nogoods: model.Add(sum(variables[k] for k in keys)<=len(keys)-1)
                    for c in lookup.values():
                        if not budget.preparation_available(): model_complete=False; break
                        r=model_single_results.get(c.signature)
                        if r is None:
                            r=context.check_single(c)
                            if r.status!="UNKNOWN": model_single_results[c.signature]=r
                        if not r.accepted:
                            if r.status!="UNKNOWN": model.Add(variables[c.signature]==0)
                            else: model_complete=False
                    if incumbent:
                        chosen={c.signature for c in incumbent.selected}
                        for key,v in variables.items(): model.AddHint(v,int(key in chosen))
                    shared_expressions=(candidates.downstream.cp_expressions(model,variables,lookup) if candidates.downstream is not None else None)
                invalid=False; carry=0.
                for phase,share in enumerate(PHASE_SHARES):
                    if not budget.search_available():
                        all_optimal=False
                        stop_reason=budget.evaluation_stop_reason() if not budget.evaluation_available() else "SOFT_BUDGET"
                        break
                    if budget.solver_wall_used>=budget.solver_seconds:
                        all_optimal=False; stop_reason="SOLVER_ALLOCATION_EXHAUSTED"; break
                    limit=min(share*budget.solver_seconds/sum(PHASE_SHARES)+carry,
                        budget.solver_seconds-budget.solver_wall_used,max(0.,budget.soft_seconds-budget.elapsed))
                    if limit<=0:
                        all_optimal=False; stop_reason="NO_SOLVER_TIME"; break
                    expression=(shared_expressions[phase] if shared_expressions is not None else
                                sum(solver_coefficients(c)[phase]*variables[key] for key,c in lookup.items()))
                    model.Minimize(expression)
                    solver=cp_model.CpSolver(); solver.parameters.num_search_workers=1; solver.parameters.random_seed=71
                    solver.parameters.max_time_in_seconds=limit
                    with budget.measure("cp_solve"):
                        started=time.perf_counter(); status=solver.Solve(model); used=time.perf_counter()-started
                    solve_used+=used; budget.solver_wall_used+=used; carry=max(0.,limit-used)
                    rec={"round":refinement,"phase":phase,"status":solver.StatusName(status),"wall_s":used,
                         "allocated_seconds":limit,"bound":solver.BestObjectiveBound(),"branches":solver.NumBranches(),
                         "conflicts":solver.NumConflicts(),"objective":None}
                    records.append(rec)
                    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
                        all_optimal=False; stop_reason="SOLVER_"+solver.StatusName(status); break
                    selected=tuple(c for key,c in lookup.items() if solver.Value(variables[key]))
                    rec["objective"]=solver.ObjectiveValue()
                    if candidates.downstream is not None:
                        rec["canonical_selected_integer_key"]=_selection_solver_key(candidates,selected)
                        rec["model_integer_key"]=tuple(solver.Value(expr) for expr in shared_expressions)
                    with budget.measure("selected_validation"):
                        if hasattr(context,"check_batch_conflicts"):
                            r,conflicts=context.check_batch_conflicts(selected)
                        else:
                            r,conflict=context.check_batch(selected)
                            conflicts=(conflict,) if conflict and r.status!="UNKNOWN" else ()
                        evaluation=context.check_commit_prefix(selected) if r.accepted and budget.watchdog_ok() else None
                    rec["batch_conflict_count"]=len(conflicts)
                    batched_conflicts+=len(conflicts)
                    rec["validation_status"]=r.status; rec["validation_reason"]=r.reason
                    rec["prefix_status"]=getattr(evaluation,"status","NOT_CHECKED")
                    rec["prefix_reason"]=getattr(evaluation,"reason",None)
                    if not r.accepted or evaluation is None or not evaluation.accepted:
                        if r.status=="UNKNOWN":
                            stop_reason="VALIDATION_UNKNOWN"; all_optimal=False; break
                        if r.accepted and (evaluation is None or evaluation.status=="UNKNOWN"):
                            stop_reason="PREFIX_VALIDATION_UNKNOWN"; all_optimal=False; break
                        if not r.accepted:
                            for conflict in conflicts:
                                if len(conflict)==2: exclusions.add(tuple(sorted(c.signature for c in conflict)))
                                elif len(conflict)==1: single_exclusions.add(conflict[0].signature)
                            if not conflicts:
                                stop_reason="VALIDATION_NO_EXPLICIT_CONFLICT"; all_optimal=False; break
                        else:
                            # Prefix rejection is a joint transaction conflict;
                            # UNKNOWN was handled above and must never be cut.
                            nogoods.add(tuple(sorted(c.signature for c in selected)))
                        rec["cumulative_pair_cuts"]=len(exclusions)
                        rec["cumulative_single_cuts"]=len(single_exclusions)
                        rec["cumulative_prefix_nogoods"]=len(nogoods)
                        invalid=True; all_optimal=False; break
                    key=_selection_key(candidates,selected)
                    proposal=PlanningResult(selected,context.prefix(selected),evaluation,"FEASIBLE",key,certified=True)
                    if phase==4:
                        final_solver_key=(tuple(solver.Value(expr) for expr in shared_expressions) if shared_expressions is not None else _selection_solver_key(candidates,selected))
                    if incumbent is None or proposal.objective<incumbent.objective: incumbent=proposal
                    if status!=cp_model.OPTIMAL: all_optimal=False
                    # Only a freshly certified feasible value fixes the next layer.
                    model.Add(expression==(_selection_solver_key(candidates,selected)[phase] if shared_expressions is not None else int(round(solver.ObjectiveValue()))))
                if not invalid: break
            else:
                stop_reason="REFINEMENT_LIMIT"; all_optimal=False
                budget.truncations.add("REFINEMENT_LIMIT")
        if not records and skip_reason is None: skip_reason=stop_reason or "NO_SOLVER_CALL"
        if incumbent is None:
            keep=tuple(cols[0] for cols in candidates.columns.values())
            with budget.measure("fallback"):
                evaluation=context.check_commit_prefix(keep) if budget.watchdog_ok() else None
            incumbent=PlanningResult(keep,(),evaluation,"NO_VERIFIED_INCUMBENT",_selection_key(candidates,keep),certified=bool(evaluation and evaluation.accepted))
        incumbent_solver_key=_selection_solver_key(candidates,incumbent.selected)
        complete=(all_optimal and resource_complete and model_complete and len(records)>=5
            and records[-1]["phase"]==4 and records[-1]["status"]=="OPTIMAL" and incumbent_solver_key==final_solver_key)
        return PlanningResult(incumbent.selected,incumbent.prefix,incumbent.evaluation,
                              "CATALOG_LEX_OPTIMAL" if complete else "LEX_LIMITED",incumbent.objective,
                              tuple(records),max((r["round"] for r in records),default=0),incumbent.certified,
                              MappingProxyType({"solver_wall_s":solve_used,"catalog_incomplete":candidates.incomplete,
                                  "resource_conflicts_complete":resource_complete,"model_screen_complete":model_complete,
                                  "optimality_scope":"bounded_candidate_catalog_model","solver_entered":bool(records),
                                  "skip_reason":skip_reason,"solver_stop_reason":stop_reason,
                                  "stop_reason":stop_reason,"max_refinement_rounds":budget.max_refinement_rounds,
                                  "refinement_rounds_used":budget.refinement_rounds_used,
                                  "hint_objective":hint_key,"hint_certified":hint.certified,"final_objective":incumbent.objective,
                                  "strict_improvement_over_hint":None if hint_key is None else incumbent.objective<hint_key,
                                  "move_aircraft_count":move_aircraft,
                                  "batched_conflicts":batched_conflicts,"pair_cuts":len(exclusions),
                                  "single_cuts":len(single_exclusions),"prefix_nogoods":len(nogoods),
                                  "model_single_cache_entries":len(model_single_results),
                                  "fallback":incumbent.status=="NO_VERIFIED_INCUMBENT"}))


class NormalSchedulerCallback:
    """Callable SchedulerPort adapter; runner owns admission/emergency timing."""
    def __init__(self,method="energy_slack",candidate_provider=None,budget_profile=QUALITY_BUDGET_PROFILE,planning_profile=PLANNING_PROFILE_LEGACY,
                 advisor_runtime=None,budget_scale=1.):
        self.scheduler=EnergySlackScheduler() if method=="energy_slack" else RollingHorizonCPSATScheduler() if method=="rh_cp_sat" else None
        if self.scheduler is None: raise ValueError("unknown Stage2 baseline")
        candidate_policy_id(planning_profile)
        self.planning_profile=planning_profile
        self.provider=candidate_provider or CandidateProvider(planning_profile=planning_profile,advisor_runtime=advisor_runtime)
        if candidate_provider is not None and advisor_runtime is not None and self.provider.advisor_runtime is not advisor_runtime:
            raise ValueError("candidate provider advisor runtime mismatch")
        if budget_scale<=0 or not math.isfinite(budget_scale): raise ValueError("invalid candidate budget scale")
        self.budget_scale=budget_scale
        if self.provider.planning_profile!=planning_profile: raise ValueError("candidate provider planning profile mismatch")
        PlanningBudget.for_profile(budget_profile)  # fail closed on unknown names
        self.budget_profile=budget_profile
        self.records=[]
        self.last_result=None

    def state_dict(self):
        from .trajectory import TrajectoryPlan
        def reference(r):
            return {"status":r.status,"plans":[p.to_dict() for p in r.plans],"soc_required":r.soc_required,
                    "peak_required":r.peak_required,"duration":r.duration,"energy_j":r.energy_j,
                    "identity":r.identity,"source":r.source}
        state={"method":self.scheduler.method,"budget_profile":self.budget_profile,"records":self.records,
                "previous":{k:[p.to_dict() for p in v] for k,v in self.provider.previous.items()},
                "references":[(key,reference(r)) for key,r in self.provider.references.cache.items()],
                "templates":[(key,p.to_dict() if p is not None else None) for key,p in self.provider.references.templates.items()],
                "reference_hits":self.provider.references.hits}
        if self.planning_profile!=PLANNING_PROFILE_LEGACY: state["planning_profile"]=self.planning_profile
        if self.provider.advisor_runtime is not None or self.budget_scale!=1.:
            runtime=self.provider.advisor_runtime
            state["stage3"]={"budget_scale":self.budget_scale,"mode":runtime.mode if runtime is not None else "OFF",
                             "deterministic_only":runtime.deterministic_only if runtime is not None else False,
                             "fingerprint":runtime.fingerprint if runtime is not None else {}}
        return json.loads(json.dumps(state,allow_nan=False))

    def load_state_dict(self,state):
        from .trajectory import TrajectoryPlan
        runtime=self.provider.advisor_runtime
        binding={"budget_scale":self.budget_scale,"mode":runtime.mode if runtime is not None else "OFF",
                 "deterministic_only":runtime.deterministic_only if runtime is not None else False,
                 "fingerprint":runtime.fingerprint if runtime is not None else {}}
        saved=dict(state.get("stage3",{"budget_scale":1.,"mode":"OFF","fingerprint":{}}))
        saved.setdefault("deterministic_only",False)
        if saved!=binding: raise ValueError("Stage3 callback binding mismatch")
        if state["method"]!=self.scheduler.method: raise ValueError("scheduler method mismatch")
        if state.get("planning_profile",PLANNING_PROFILE_LEGACY)!=self.planning_profile: raise ValueError("scheduler planning profile mismatch")
        profile=state.get("budget_profile",LEGACY_BUDGET_PROFILE)
        PlanningBudget.for_profile(profile)
        self.budget_profile=profile
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
        budget=PlanningBudget.for_profile(self.budget_profile)
        if self.budget_scale!=1.:
            from .stage3_learning.runtime import scale_candidate_budget
            scale_candidate_budget(budget,self.budget_scale)
        solver_history=[]
        for restart in range(2):
            with budget.measure("snapshot_and_uniform_refinement"):
                snapshot=build_planning_snapshot(port)
            context=EvaluationContext(snapshot,budget=budget,validation_cache=self.provider.validation_cache)
            with budget.measure("candidate_generation_and_single_vehicle"):
                catalog=self.provider.get_columns(snapshot,budget,context)
            with budget.measure("selection"):
                result=self.scheduler.plan(snapshot,catalog,context,budget)
            solver_history.extend(dict(r,uniform_refinement_restart=restart) for r in result.solver_records)
            if not getattr(result.evaluation,"refinement_required",False): break
            if restart or not budget.search_available():
                result=replace(result,prefix=(),evaluation=None,status="REFINEMENT_UNCERTAIN",certified=False)
                break
            with budget.measure("snapshot_and_uniform_refinement"):
                port.refine_planning_boundary()
        commit=None
        if result.certified and budget.watchdog_ok():
            with budget.measure("commit"):
                commit=port.commit_validated_prefix(result.evaluation,snapshot_version=snapshot.version)
        runtime=self.provider.advisor_runtime
        if runtime is not None:
            from .stage3_learning.adapters import stable_id
            runtime.emit(dict(record_type="PlanningSelection",snapshot_id=runtime.snapshot_id,
                selected_column_ids=[stable_id("column",c.signature) for c in result.selected],
                selected_descriptor_ids=sorted({did for c in result.selected for p in c.plans for did in runtime._plans.get(p.trajectory_hash,())[-1:]}),
                certified=result.certified,objective=list(result.objective),
                submitted_trajectory_hashes=[p.trajectory_hash for p in result.prefix] if commit is not None and commit.accepted else [],
                commit_status=getattr(commit,"status","NOT_COMMITTED")))
        evidence_report=None
        report=getattr(port,"report_normal_evidence",None)
        if _v6(snapshot) and callable(report):
            rows=[]
            for a in _decision_aircraft(snapshot):
                node=_node(snapshot,a)
                key=(snapshot.config.config_hash,node.ring,node.slot,node.time,node.version,a.id)
                reference=self.provider.references.cache.get(key)
                if reference is not None and reference.status=="KNOWN":
                    rows.append(dict(aircraft_id=a.id,source_ring=node.ring,source_slot=node.slot,
                        source_version=node.version,source_time=node.time,normal_path_soc=reference.soc_required,
                        normal_path_wait_seconds=0.,reference_status=reference.status,reference_source=reference.source))
            if rows:
                # Warning metadata changes the visible snapshot identity. Send
                # it after the safety transaction, using a fresh version; the
                # executor rejects rows whose source changed at commitment.
                with budget.measure("normal_evidence_reporting"):
                    current=build_planning_snapshot(port)
                    evidence_report=report(tuple(rows),snapshot_version=current.version)
        self.last_result=result
        committed_prefix=result.prefix if commit is not None and commit.accepted else ()
        self.provider.previous={c.aircraft_id:tuple(p for p in c.plans if p not in committed_prefix) for c in result.selected if c.kind=="MOVE"}
        selected_by_id={c.aircraft_id:c for c in result.selected}
        aircraft_choices=[]
        for aid,columns in catalog.columns.items():
            a=snapshot.aircraft[aid]; selected=selected_by_id.get(aid)
            reference=columns[0].reference if columns else None
            source_status=catalog.slack_reference_statuses.get(aid)
            emergency=getattr(a,"emergency",None)
            moving=[c for c in columns if c.kind=="MOVE" and c.plans]
            full=sum(c.plans[-1].pad_id is not None for c in moving)
            insufficient=[]
            if not moving: insufficient.append("NO_SAFE_MOVE_OBSERVED")
            if not full: insufficient.append("NO_COMPLETE_NORMAL_CHAIN_OBSERVED")
            if source_status!="KNOWN": insufficient.append("NORMAL_REFERENCE_UNKNOWN")
            if aid in catalog.incomplete: insufficient.append("BOUNDED_CATALOG_INCOMPLETE")
            aircraft_choices.append(dict(aircraft_id=aid,ring=a.ring,soc=a.soc,candidate_count=len(columns),
                move_candidates=len(moving),full_normal_chain_count=full,partial_progress_count=len(moving)-full,
                only_keep_wait=not moving,candidate_insufficiency_reasons=insufficient,
                candidate_pad_ids=sorted({p.pad_id for c in moving for p in c.plans if p.pad_id is not None}),
                selected_pad_ids=sorted({p.pad_id for p in selected.plans if p.pad_id is not None}) if selected else [],
                immediate_move_candidates=sum(c.kind=="MOVE" and bool(c.plans) and c.plans[0].start_time<=snapshot.time+5.+1.e-8
                    and aid not in snapshot.soft for c in columns),
                selected_kind=None if selected is None else selected.kind,
                first_target=None if selected is None else selected.first_target,
                emergency_declared=bool(getattr(emergency,"declared",False)),
                emergency_eligible=bool(getattr(emergency,"eligible",False)),slack=catalog.slacks.get(aid),
                emergency_warning=bool(getattr(emergency,"warning",False)),
                normal_status=getattr(emergency,"normal_status","UNKNOWN"),normal_margin=getattr(emergency,"normal_margin",None),
                source_reference_status=source_status,keep_tail_reference_status=getattr(reference,"status",None),
                unknown_slack_source=UNKNOWN_SLACK_SOURCE if source_status is not None and source_status!="KNOWN" else None,
                rejection_reasons=dict(context.rejections_by_aircraft.get(aid,{}))))
        solver_entered=bool(solver_history)
        pad_opportunities={}
        for pad in range(snapshot.config.pads.count):
            available=[c for columns in catalog.columns.values() for c in columns
                if any(p.pad_id==pad for p in c.plans)]
            checked=[context.singles.get((context.snapshot_key,c.signature,c.resources)) for c in available]
            rejected=context.pad_rejections.get(pad,{})
            pad_opportunities[str(pad)]=dict(candidate_count=len(available),
                full_chain_count=sum(c.plans[-1].pad_id is not None for c in available),
                certified_single_count=sum(r is not None and r.accepted for r in checked),
                selected_count=sum(any(p.pad_id==pad for p in c.plans) for c in result.selected),
                submitted_count=sum(p.pad_id==pad for p in committed_prefix),
                explicit_conflict_count=rejected.get("explicit_conflict_count",0),unknown_count=rejected.get("unknown_count",0),
                rejection_reasons={key[7:]:value for key,value in rejected.items() if key.startswith("reason:")},
                validation_status="EXAMINED" if any(r is not None for r in checked) or rejected else "UNKNOWN_NOT_EXAMINED")
        record={"time":snapshot.time,"method":self.scheduler.method,"status":result.status,
                "candidate_policy_id":candidate_policy_id(self.planning_profile),"budget_profile":self.budget_profile,"planning_budget":budget.settings(),
                "deadline_reference_seconds":5.,"hint_wall_s":budget.hint_wall_used,"solver_wall_s":budget.solver_wall_used,
                "hint_objective":result.details.get("hint_objective"),"final_objective":result.objective,
                "hint_certified":result.details.get("hint_certified"),
                "strict_improvement_over_hint":result.details.get("strict_improvement_over_hint"),
                "solver_entered":solver_entered,"skip_reason":None if solver_entered else result.details.get("skip_reason",
                    "METHOD_ENERGY_SLACK" if self.scheduler.method=="energy_slack" else "NO_SOLVER_CALL"),
                "solver_stop_reason":result.details.get("solver_stop_reason"),
                "stop_reason":result.details.get("stop_reason"),
                "refinement_rounds_used":budget.refinement_rounds_used,
                "resource_conflicts_complete":result.details.get("resource_conflicts_complete"),
                "model_screen_complete":result.details.get("model_screen_complete"),
                "solver_optimality_scope":result.details.get("optimality_scope"),
                "move_aircraft_count":sum(row["move_candidates"]>0 for row in aircraft_choices),
                "aircraft_choices":aircraft_choices,"candidate_rejection_reasons":dict(context.rejection_reasons),
                "unknown_slack_source":UNKNOWN_SLACK_SOURCE,
                "objective":result.objective,"columns":sum(map(len,catalog.columns.values())),
                "aircraft":len(catalog.columns),"attempts":dict(catalog.attempts),"catalog_incomplete":catalog.incomplete,
                "geometry_screening":dict(budget.screening),
                "validation_cache":dict(context.cache_stats),"move_energy_cache":dict(self.provider.references.energy_stats),
                "batched_conflicts":result.details.get("batched_conflicts",0),
                "pad_opportunities":pad_opportunities,
                "normal_evidence_report_status":getattr(evidence_report,"status","NOT_REPORTED"),
                "truncations":sorted(budget.truncations),"timings":dict(budget.timings),
                "wall_s":budget.elapsed,"cpu_s":time.process_time()-budget.cpu_start,
                "deadline_miss":budget.elapsed>5.,"submitted":len(result.prefix) if commit and commit.accepted else 0,
                "commit_status":commit.status if commit is not None else "NOT_COMMITTED",
                "solver":tuple(solver_history),"refinement_rounds":result.refinement_rounds,
                "uniform_refinement_restarts":restart,
                "active_count":len(snapshot.active),"holding_count":sum(a.status.startswith("HOLD_") for a in snapshot.aircraft.values()),
                "upstream_count":len(snapshot.requests),"trigger_reason":"UNSPECIFIED_EXECUTOR_TRIGGER",
                "cache_hits":catalog.cache_hits,"trajectory_recomputes":sum(catalog.attempts.values()),
                "resource_aligned_starts":tuple(self.provider.aligned_starts),
                "fallback":bool(result.details.get("fallback",False)) or result.status in ("PREFIX_NOT_EXECUTABLE","NO_VERIFIED_INCUMBENT","REFINEMENT_UNCERTAIN"),
                "stale":bool(commit is not None and commit.status=="STALE"),"watchdog":"WATCHDOG" in budget.truncations,
                "noop":not result.prefix,"selected_key":result.objective,
                "keep":sum(c.kind=="KEEP" for c in result.selected),"wait":sum(c.kind=="WAIT" for c in result.selected)}
        if catalog.downstream is not None:
            with budget.measure("downstream_explanation"):
                selected_key,downstream_rows=catalog.downstream.evaluate(result.selected)
                for selected in result.selected:
                    timing=selected.downstream.get("candidate_timing")
                    alternatives=[c for c in catalog.columns[selected.aircraft_id] if c.plans and
                        (not selected.plans or c.downstream.get("candidate_timing")!=timing)]
                    comparison=None
                    if alternatives:
                        scored=[]
                        for c in alternatives:
                            trial=tuple(c if item.aircraft_id==c.aircraft_id else item for item in result.selected)
                            scored.append((catalog.downstream.evaluate(trial)[0],c.signature,c))
                        comparison_key,_,comparison=min(scored,key=lambda row:row[:2])
                        layer=next((name for name,a,b in zip(("F","L","risk","Ehat","tau"),selected_key,comparison_key) if a!=b),"TIE")
                        reason=("JOINT_VALIDATION_OR_GLOBAL_ALLOCATION" if comparison_key<selected_key else
                                "TIE_SIGNATURE_OR_ALLOCATION" if comparison_key==selected_key else layer)
                    else:
                        comparison_key=None; layer=None; reason="NO_OPPOSITE_TIMING_CANDIDATE"
                    submitted=any(p.aircraft_id==selected.aircraft_id for p in committed_prefix)
                    action=("IMMEDIATE_PREFIX_SUBMITTED" if submitted else "HOLD_CURRENT_SLOT" if not selected.plans else
                            "FUTURE_PREDICTION_NO_RESERVATION" if selected.plans[0].start_time>snapshot.time+5.+1.e-8 else "PREFIX_NOT_COMMITTED")
                    downstream_rows[selected.aircraft_id]["decision_explanation"]=dict(reason=reason,first_different_objective_layer=layer,
                        comparison_signature=None if comparison is None else comparison.signature,comparison_integer_key=comparison_key,
                        comparison_status="NOT_JOINT_VALIDATED" if comparison is not None else "NO_COMPARISON",actual_action=action,
                        observed_rejections=dict(context.rejections_by_aircraft.get(selected.aircraft_id,{})))
                record.update(planning_profile=self.planning_profile,objective_id=objective_schema_id(self.planning_profile),
                    downstream=downstream_rows,downstream_integer_key=selected_key,
                    virtual_pad_assignment=dict(catalog.downstream.assignment),virtual_queue_order=catalog.downstream.order)
            record.update(wall_s=budget.elapsed,cpu_s=time.process_time()-budget.cpu_start,
                timings=dict(budget.timings),deadline_miss=budget.elapsed>5.)
        self.records.append(record)
        return result
