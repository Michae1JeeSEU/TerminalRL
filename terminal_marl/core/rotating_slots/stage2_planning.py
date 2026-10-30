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
from .selective_replanning.policy import PROGRESSIVE_VARIANTS, REFERENCE_VARIANT, resolve_search_policy
from .stage2_downstream import (LEGACY_EVALUATION_PROFILE, STITCHED_EVALUATION_PROFILE,
    validate_evaluation_profile)

HORIZON = 240.0
OFFSETS = (0., 5., 10., 15., 20., 30.)
SERVICE_SECONDS = {"short": 60., "standard": 120., "long": 240.}
PHASE_SHARES = (.3, .8, .3, .4, .2)
QUALITY_BUDGET_PROFILE = "quality_v5"
CANDIDATE_POLICY_ID = "shared_multistep_columns_v6_reuse_diversity"
PLANNING_PROFILE_LEGACY = "progress_tail_v7"
PLANNING_PROFILE_DOWNSTREAM = "downstream_service_v8"
PLANNING_PROFILES = (PLANNING_PROFILE_LEGACY, PLANNING_PROFILE_DOWNSTREAM)


def candidate_policy_id(profile, evaluation_profile=LEGACY_EVALUATION_PROFILE):
    if profile not in PLANNING_PROFILES: raise ValueError("unknown Stage2 planning profile: "+str(profile))
    validate_evaluation_profile(evaluation_profile)
    if evaluation_profile==STITCHED_EVALUATION_PROFILE: return "bounded_suffix_completion_v1"
    return CANDIDATE_POLICY_ID if profile == PLANNING_PROFILE_LEGACY else "shared_downstream_columns_v8_bounded_delay"


def objective_schema_id(profile, evaluation_profile=LEGACY_EVALUATION_PROFILE):
    if profile not in PLANNING_PROFILES: raise ValueError("unknown Stage2 planning profile: "+str(profile))
    validate_evaluation_profile(evaluation_profile)
    if evaluation_profile==STITCHED_EVALUATION_PROFILE: return "stitched_continuation_v1_F_L_risk_E_tau"
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
    candidate_active_seconds: float | None = None
    _candidate_phase_started: float | None = field(default=None,repr=False)
    planning_wall_deadline: float | None = None
    search_wall_deadline: float | None = None
    planning_wall_origin: float | None = None
    search_handoff_reserve_seconds: float = 0.
    heavy_highwater: dict = field(default_factory=dict,repr=False)
    heavy_checks: dict = field(default_factory=dict,repr=False)
    certificate_events: list = field(default_factory=list,repr=False)
    first_nonempty_certified: dict | None = None
    _certificate_best: dict = field(default_factory=dict,repr=False)
    _prefix_receipts: dict = field(default_factory=dict,repr=False)
    _catalog: object = field(default=None,repr=False)
    _catalog_scope: str | None = field(default=None,repr=False)
    _certificate_context: str | None = field(default=None,repr=False)
    _catalog_counter: int = field(default=0,repr=False)
    certificate_source: str = "search_prefix"
    allowed_replan_ids: object = None
    diagnostics: object = None

    def search_deadline_ok(self):
        if self.search_wall_deadline is None or time.perf_counter()<self.search_wall_deadline: return True
        self.truncations.add("SEARCH_HANDOFF_DEADLINE"); return False

    def heavy_estimate(self,kind):
        return max(1.,1.25*self.heavy_highwater.get(kind,0.))

    def heavy_allowed(self,kind):
        if self.search_wall_deadline is None: return True
        remaining=max(0.,self.search_wall_deadline-time.perf_counter())
        estimate=self.heavy_estimate(kind)
        row=self.heavy_checks.setdefault(kind,dict(started=0,skipped=0,wall_s=0.,max_wall_s=0.,
            max_estimate_s=0.,overrun_count=0))
        row["max_estimate_s"]=max(row["max_estimate_s"],estimate)
        if remaining>=estimate: return True
        row["skipped"]+=1; row["last_skip_remaining_s"]=remaining
        self.truncations.add("SEARCH_HEAVY_HEADROOM"); return False

    @contextmanager
    def heavy_work(self,kind):
        started=time.perf_counter()
        try: yield
        finally:
            if self.search_wall_deadline is not None:
                used=time.perf_counter()-started
                self.heavy_highwater[kind]=max(self.heavy_highwater.get(kind,0.),used)
                row=self.heavy_checks.setdefault(kind,dict(started=0,skipped=0,wall_s=0.,max_wall_s=0.,
                    max_estimate_s=0.,overrun_count=0))
                row["started"]+=1;row["wall_s"]+=used;row["max_wall_s"]=max(row["max_wall_s"],used)
                row["overrun_count"]+=int(time.perf_counter()>self.search_wall_deadline)

    def bind_catalog(self,catalog):
        if self.search_wall_deadline is None: return
        if self._catalog is catalog: return
        self._catalog=catalog
        self._catalog_counter+=1
        self._catalog_scope="callback_catalog_"+str(self._catalog_counter)

    def solver_validation_headroom(self):
        # One new solution must still pass uncached selected single/pair and
        # whole-prefix checks. This is an estimate, not a safety certificate.
        return (sum(self.heavy_estimate(kind) for kind in ("prefix","single","geometry"))
                if self.search_wall_deadline is not None else 0.)

    def certificate_event(self,source,columns,prefix,*,event="CERTIFIED_PREFIX",comparison=None,objective=None):
        if self.search_wall_deadline is None: return
        row=dict(event=event,source=source,wall_elapsed_s=time.perf_counter()-(self.planning_wall_origin if self.planning_wall_origin is not None else self.wall_start),
            catalog_scope=self._catalog_scope,context_scope=self._certificate_context,objective=None if objective is None else list(objective),
            prefix_hashes=[p.trajectory_hash for p in prefix],search_certificate_not_publication=True)
        if comparison is not None: row["comparison_objective"]=list(comparison)
        key=(self._catalog_scope,tuple(row["prefix_hashes"]))
        if event=="CERTIFIED_PREFIX":
            self._prefix_receipts[key]=row
            if prefix and self.first_nonempty_certified is None:
                row["event"]="FIRST_NONEMPTY_CERTIFIED"
                self.first_nonempty_certified=row
                self.certificate_events.append(row)
            return
        if event=="CERTIFIED_OBJECTIVE":
            receipt=self._prefix_receipts.get(key)
            if receipt is None: return  # A rescore/cache reuse is not a new validation receipt.
            previous=self._certificate_best.get(self._catalog_scope)
            if previous is None or objective<previous: self._certificate_best[self._catalog_scope]=objective
            if receipt["objective"] is None: receipt["objective"]=list(objective)
            if previous is None or objective>=previous: return
            row.update(event="STRICT_CERTIFIED_IMPROVEMENT",certificate_wall_elapsed_s=receipt["wall_elapsed_s"])
        self.certificate_events.append(row)

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

    @property
    def candidate_elapsed(self):
        if self.candidate_active_seconds is None: return self.elapsed
        return self.candidate_active_seconds+(0. if self._candidate_phase_started is None else time.perf_counter()-self._candidate_phase_started)

    @contextmanager
    def candidate_work(self):
        if self.candidate_active_seconds is None:
            yield
            return
        if self._candidate_phase_started is not None: raise RuntimeError("nested candidate phase")
        self._candidate_phase_started=time.perf_counter()
        try: yield
        finally:
            self.candidate_active_seconds+=time.perf_counter()-self._candidate_phase_started
            self._candidate_phase_started=None

    def available(self, aircraft_id=None):
        if not self.search_deadline_ok(): return False
        if not self.ready_deadline_ok(): return False
        for deadline,reason in self._stage_limits:
            if time.perf_counter()>=deadline:
                self.truncations.add(reason); return False
        if self.elapsed>=self.soft_seconds:
            self.truncations.add("SOFT_BUDGET" if self.candidate_active_seconds is not None else "CANDIDATE_WORK_CUTOFF"); return False
        if self.candidate_elapsed >= self.candidate_seconds:
            self.truncations.add("CANDIDATE_WORK_CUTOFF"); return False
        if sum(self.attempts.values()) >= self.max_attempts:
            self.truncations.add("GLOBAL_ATTEMPT_LIMIT"); return False
        if aircraft_id is not None and self.attempts[aircraft_id] >= self.max_per_aircraft:
            self.truncations.add("AIRCRAFT_ATTEMPT_LIMIT"); return False
        return True

    def generation_stop_reason(self):
        """Candidate exhaustion does not consume the existing-catalog solver."""
        if not self.search_deadline_ok(): return "SEARCH_HANDOFF_DEADLINE"
        if not self.ready_deadline_ok(): return "READY_DEADLINE"
        for deadline,reason in self._stage_limits:
            if time.perf_counter()>=deadline: return reason
        if self.elapsed>=self.soft_seconds: return "SOFT_BUDGET"
        if self.candidate_elapsed>=self.candidate_seconds: return "CANDIDATE_WORK_CUTOFF"
        if sum(self.attempts.values())>=self.max_attempts: return "GLOBAL_ATTEMPT_LIMIT"
        return None

    def watchdog_ok(self):
        if self.elapsed < self.watchdog_seconds: return True
        self.truncations.add("WATCHDOG"); return False

    def ready_deadline_ok(self):
        if self.planning_wall_deadline is None or time.perf_counter()<self.planning_wall_deadline: return True
        self.truncations.add("READY_DEADLINE"); return False

    def remaining_seconds(self):
        """Remaining search time on the original clock, including active caps."""
        deadlines=[self.wall_start+self.soft_seconds,self.wall_start+self.watchdog_seconds]
        if self.planning_wall_deadline is not None: deadlines.append(self.planning_wall_deadline)
        if self.search_wall_deadline is not None: deadlines.append(self.search_wall_deadline)
        deadlines.extend(deadline for deadline,_ in self._stage_limits)
        return max(0.,min(deadlines)-time.perf_counter())

    def search_available(self, reserve_seconds=0.):
        if not self.evaluation_available(): return False
        reserves=[deadline for deadline,_ in self._stage_limits]
        if self.planning_wall_deadline is not None: reserves.append(self.planning_wall_deadline)
        if self.search_wall_deadline is not None: reserves.append(self.search_wall_deadline)
        if reserve_seconds and any(time.perf_counter()>=deadline-reserve_seconds for deadline in reserves):
            self.truncations.add("HINT_VALIDATION_RESERVE"); return False
        if self.elapsed<self.soft_seconds: return True
        self.truncations.add("SOFT_BUDGET"); return False

    def evaluation_available(self):
        if not self.search_deadline_ok(): return False
        if not self.ready_deadline_ok(): return False
        if not self.watchdog_ok(): return False
        for deadline,reason in self._stage_limits:
            if time.perf_counter()>=deadline:
                self.truncations.add(reason); return False
        return True

    def evaluation_stop_reason(self):
        if not self.search_deadline_ok(): return "SEARCH_HANDOFF_DEADLINE"
        if self.planning_wall_deadline is not None and time.perf_counter()>=self.planning_wall_deadline: return "READY_DEADLINE"
        if self.elapsed>=self.watchdog_seconds: return "WATCHDOG"
        for deadline,reason in self._stage_limits:
            if time.perf_counter()>=deadline: return reason
        return "EVALUATION_BUDGET"

    @contextmanager
    def stage_limit(self, seconds, reason, reserve_seconds=0.):
        # An absolute deadline on the original clock, never a fresh budget.
        deadline=self.wall_start+self.soft_seconds-reserve_seconds
        if self.planning_wall_deadline is not None: deadline=min(deadline,self.planning_wall_deadline-reserve_seconds)
        if self.search_wall_deadline is not None: deadline=min(deadline,self.search_wall_deadline-reserve_seconds)
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
    scope: object = None
    fixed_background: object = None
    generation_coverage: object = None


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


def _executable_start(snapshot, start):
    timing = getattr(snapshot,"planning_timing",None)
    return (abs(start-timing.effective_time)<=1.e-8 if timing is not None
            else start<=snapshot.time+5.+1.e-8)


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
        hit_clock=None
        if self.advisor_runtime is not None:
            from .stage3_learning.features import SCHEMA_V2
            if getattr(self.advisor_runtime,"feature_schema",None)==SCHEMA_V2:
                hit_clock=(time.perf_counter(),time.process_time())
        key=(config.config_hash,node,plan.trajectory_hash)
        if key in self.energy:
            q=self.energy[key]
            self.energy_stats["hits"]+=1
            _bounded_put(self.motion_energy_index,(config.config_hash,plan.trajectory_hash),(node,q))
            if self.advisor_runtime is not None:
                if hit_clock is None: self.advisor_runtime.record_energy(node,plan,q,True)
                else: self.advisor_runtime.record_energy(node,plan,q,True,
                    time.perf_counter()-hit_clock[0],time.process_time()-hit_clock[1])
            return q
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
        if budget.allowed_replan_ids is not None and a.id not in budget.allowed_replan_ids: raise ValueError("reference generation outside scope")
        context=getattr(budget,"_strict_resource_context",None)
        if context is not None and self.immutable_resource_conflict(snapshot,a,node,target,start,duration,context):
            budget.screening["strict_immutable_resource_pruned"]+=1
            return True,None
        runtime=self.advisor_runtime
        if (runtime is not None and (getattr(runtime,'ranking_scope','legacy') in ('comparable_group_v1','comparable_group_lazy_v2') or
                budget.planning_wall_deadline is not None) and not budget.available(a.id)):
            return False,None
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

    @staticmethod
    def immutable_resource_conflict(snapshot,a,node,target,start,duration,context):
        """Prove overlap using immutable committed finite occupancy only.

        A selected holder may leave its target slot jointly; its current
        uncommitted or open-ended occupancy is never a strict pruning proof.
        """
        next_ring={"outer":"middle","middle":"inner","inner":None}[node.ring]
        resource=target if next_ring is None else (next_ring,target)
        rows=snapshot.pads if next_ring is None else snapshot.slots
        lo=start+duration
        hi=lo+13.5+service_duration(snapshot.config,a.service_class) if next_ring is None else None
        return any(r.resource==resource and r.aircraft_id!=a.id and r.committed and math.isfinite(r.end)
            and (r.start<hi and lo<r.end if hi is not None else r.start<=lo<r.end) for r in rows)

    def _generate_unobserved(self,snapshot,a,node,target,start,duration,budget):
        if budget.allowed_replan_ids is not None and a.id not in budget.allowed_replan_ids: raise ValueError("trajectory materialization outside scope")
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
        if not budget.heavy_allowed("geometry"): return False,None
        with budget.heavy_work("geometry"):
            p=_generate(snapshot,a.id,node,target,start,duration)
        budget.screening["materialized:"+a.id]+=1
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
        if budget.allowed_replan_ids is not None and aircraft.id not in budget.allowed_replan_ids: raise ValueError("cold reference outside scope")
        key=(snapshot.config.config_hash,node.ring,node.slot,node.time,node.version,aircraft.id)
        if key in self.cache:
            self.hits+=1
            if self.advisor_runtime is not None:
                self._register_reused_chain(snapshot,aircraft,node,self.cache[key].plans,budget)
            return self.cache[key]
        current,plans=node,[]
        for _ in range({"outer":3,"middle":2,"inner":1}[node.ring]):
            found=[]
            targets=range(snapshot.config.pads.count if current.ring=="inner" else ring_config({"outer":"middle","middle":"inner"}[current.ring],snapshot.config).slot_count)
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
    def __init__(self, snapshot, prefix_evaluator=None, budget=None, validation_cache=None,
                 reuse_prefix_results=False, prefix_evaluator_identity=None, *, scope=None, global_frame=None):
        if scope is not None: scope.validate(snapshot)
        if global_frame is not None and (global_frame.snapshot_id != snapshot.version or global_frame.effective_time != snapshot.time):
            raise ValueError("global frame snapshot binding mismatch")
        if scope is not None and global_frame is not None and scope.objective_frame_id != global_frame.identity:
            raise ValueError("scope/frame binding mismatch")
        self.scope, self.global_frame = scope, global_frame
        self.snapshot = snapshot
        self.prefix_evaluator = prefix_evaluator
        self.reuse_prefix_results = bool(reuse_prefix_results)
        self.prefix_evaluator_identity = prefix_evaluator_identity
        self._prefix_results = {}
        self.observed_definite_conflicts = []
        self.joint_conflict_observations = []
        self.pair_conflict_evidence = []
        self.pair_conflict_evidence_dropped = 0
        self._pair_conflict_seen = {}
        self.cache=validation_cache if validation_cache is not None else {}
        self.cache_stats=Counter()
        if self.reuse_prefix_results:
            self.cache_stats["prefix_invalidations"] = int(bool(self.cache.get("prefix_context_seen")))
            self.cache["prefix_context_seen"] = True
        # Include the full immutable scheduling input, rather than the world
        # version alone: resources, SOC and fixed traffic are validation inputs.
        self.snapshot_key=hashlib.sha256(repr((snapshot,HORIZON)).encode()).hexdigest()
        if scope is not None:
            self.snapshot_key=hashlib.sha256(repr((self.snapshot_key,scope.identity,getattr(global_frame,"identity",None))).encode()).hexdigest()
        previous=self.cache.get("snapshot_key")
        if previous is not None and previous!=self.snapshot_key: self.cache_stats["snapshot_invalidations"]+=1
        self.cache["snapshot_key"]=self.snapshot_key
        self.protection=self.cache.setdefault("protection",{})
        for key,value in snapshot.source_clear.items():
            _bounded_put(self.protection,(snapshot.config.config_hash,key),value)
        self.pairs=self.cache.setdefault("pairs",{})
        self.singles=self.cache.setdefault("singles",{})
        self.energies=self.cache.setdefault("energies",{})
        self.decisions = set(scope.replan_ids) if scope is not None else {a.id for a in _decision_aircraft(snapshot)}
        self.completed_evaluations=set()
        self.background_blockers=[]
        self.budget=budget
        self.rejection_reasons=Counter()
        self.rejections_by_aircraft={}
        self.pad_rejections={}
        self.diagnostics=None
        if budget is not None and getattr(budget,"diagnostics",None) is not None:
            from .selective_replanning.diagnostics import CandidateDiagnostics
            self.diagnostics=CandidateDiagnostics(budget.diagnostics,snapshot)

    def _time_ok(self): return self.budget is None or self.budget.evaluation_available()

    def _time_rejection(self):
        return ValidationResult(False,"UNKNOWN",self.budget.evaluation_stop_reason())

    def _geometry_validation(self,*args):
        if self.budget is not None and self.budget.search_wall_deadline is not None and not self._time_ok():
            return self._time_rejection()
        if self.budget is not None and not self.budget.heavy_allowed("geometry"):
            return ValidationResult(False,"UNKNOWN","SEARCH_HEAVY_HEADROOM")
        with self.budget.heavy_work("geometry") if self.budget else nullcontext():
            return validate_separation(*args)

    def _observe(self,result,columns):
        if self.diagnostics is not None: self.diagnostics.query("OBSERVE",columns,result)
        if not result.accepted:
            reason=getattr(result,"reason",None) or result.status
            self.rejection_reasons[reason]+=1
            if result.status in ("CONFLICT","REJECTED","UNSAFE","INVALID") and len(self.observed_definite_conflicts)<128:
                self.observed_definite_conflicts.append(dict(
                    aircraft_ids=tuple(c.aircraft_id for c in columns),
                    resources=tuple(r.resource for c in columns for r in c.resources), reason=reason))
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
            if not self._geometry_validation(p, base, hi, horizon, self.snapshot.config).accepted: return None, None
            while hi-lo > .01:
                mid = (lo+hi)/2
                result=self._geometry_validation(p, base, mid, horizon, self.snapshot.config)
                if result.status=="UNKNOWN": return None,None
                if result.accepted: hi = mid
                else: lo = mid
            clear = hi
        if p.target_ring is not None:
            base = make_stable_plan("target", p.target_ring, p.target_slot, p.start_time, horizon, config=self.snapshot.config)
            lo, hi = p.start_time, p.end_time
            while hi-lo > .01:
                mid = (lo+hi)/2
                result=self._geometry_validation(p, base, p.start_time, mid, self.snapshot.config)
                if result.status=="UNKNOWN": return None,None
                if result.accepted: lo = mid
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
                if end is None: return None
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

    def _pair_observation(self,a,b,result,check_interval=None):
        if not result.accepted:
            interval=result.details.get("pair_evidence",{}).get("check_interval",check_interval)
            raw_details=tuple(sorted((k,v) for k,v in result.details.items() if k!="pair_evidence"))
            key=repr((self.snapshot_key,tuple(sorted((a.signature,b.signature))),result.status,result.reason,interval,raw_details))
            previous=self._pair_conflict_seen.get(key)
            if previous is not None:
                self.pair_conflict_evidence[previous]["count"]+=1
                if self.diagnostics is not None: self.diagnostics.query("PAIR",(a,b),result)
                return result
            if len(self.pair_conflict_evidence)>=128:
                self.pair_conflict_evidence_dropped+=1
                if self.diagnostics is None: return result
            from .selective_replanning.diagnostics import identity,plain
            if "pair_evidence" not in result.details:
                participants=[]
                for column in (a,b):
                    aircraft=self.snapshot.aircraft[column.aircraft_id]
                    participants.append(dict(identity(column),snapshot_ring=aircraft.ring,snapshot_slot=aircraft.slot,
                        committed_trajectory_hashes=[p.trajectory_hash for p in
                            (self.snapshot.active.get(column.aircraft_id),self.snapshot.soft.get(column.aircraft_id)) if p is not None]))
                result=replace(result,details=dict(result.details,pair_evidence=dict(
                    participants=participants,check_interval=check_interval,
                    hard_separation_m=self.snapshot.config.safety.hard_separation_m)))
            if len(self.pair_conflict_evidence)<128:
                self._pair_conflict_seen[key]=len(self.pair_conflict_evidence)
                self.pair_conflict_evidence.append(dict(status=result.status,reason=result.reason,
                    snapshot_id=self.snapshot.version,context_scope=getattr(self.scope,"identity",None),
                    local_round=getattr(self,"local_round",None),count=1,details=plain(result.details)))
        if self.diagnostics is not None: self.diagnostics.query("PAIR",(a,b),result)
        return result

    def check_pair(self, a, b):
        if not self._time_ok(): return self._pair_observation(a,b,self._time_rejection())
        key = (self.snapshot_key,tuple(sorted(((a.signature,a.resources),(b.signature,b.resources)))))
        if key in self.pairs:
            self.cache_stats["pair_hits"]+=1
            result=self.pairs[key]
            return self._pair_observation(a,b,result)
        self.cache_stats["pair_misses"]+=1
        self.cache_stats["pair_selected_background" if a.kind=="BACKGROUND" or b.kind=="BACKGROUND" else "pair_selected_selected"]+=1
        if self.resource_conflict(a,b):
            overlaps=[dict(resource=x.resource,interval=[max(x.start,y.start),min(x.end,y.end)],
                           first_interval=[x.start,x.end],second_interval=[y.start,y.end])
                for x in a.resources for y in b.resources
                if x.resource==y.resource and x.start<y.end and y.start<x.end]
            result = ValidationResult(False,"CONFLICT","RESOURCE_CONFLICT",{"resource_overlaps":overlaps})
            check_interval=None
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
            check_interval=[self.snapshot.time,end]
            if end>self.snapshot.time:
                with (self.budget.measure("trajectory_validation") if self.budget else nullcontext()):
                    result=self._geometry_validation(ta,tb,self.snapshot.time,end,self.snapshot.config)
            else: result=ValidationResult(True,"SAFE")
        # Retain the exact certificate details; no witness or distance is
        # inferred from a rejected interval. This metadata is not a new check.
        result=self._pair_observation(a,b,result,check_interval)
        if result.status!="UNKNOWN": _bounded_put(self.pairs,key,result)
        return result

    def check_single(self, column):
        if self.scope is not None and column.aircraft_id not in self.decisions:
            raise ValueError("candidate outside planning scope")
        if not self._time_ok(): return self._observe(self._time_rejection(),(column,))
        key=((self.snapshot_key,column.signature,column.resources)
             if all(getattr(p,"trajectory_hash",None) for p in column.plans) else None)
        if key is not None and key in self.singles:
            self.cache_stats["single_hits"]+=1
            if self.singles[key].status!="UNKNOWN": self.completed_evaluations.add(column.aircraft_id)
            return self._observe(self.singles[key],(column,))
        self.cache_stats["single_misses"]+=1
        if self.budget is not None and not self.budget.heavy_allowed("single"):
            return self._observe(ValidationResult(False,"UNKNOWN","SEARCH_HEAVY_HEADROOM"),(column,))
        with self.budget.heavy_work("single") if self.budget else nullcontext():
            result=self._check_single(column)
        if key is not None and result.status!="UNKNOWN": _bounded_put(self.singles,key,result)
        if result.status!="UNKNOWN": self.completed_evaluations.add(column.aircraft_id)
        if self.diagnostics is not None: self.diagnostics.query("SINGLE",(column,),result)
        return result

    def _check_single(self, column):
        # Only the first action that prefix() would actually submit is guarded;
        # a future normal-final suffix cannot suppress an inward transfer.
        if (column.kind=="MOVE" and column.plans and column.aircraft_id not in self.snapshot.soft
            and column.plans[0].kind=="FINAL_NORMAL"
            and _executable_start(self.snapshot,column.plans[0].start_time)):
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
            resources=self.resources(a,())
            if resources is None and self.budget is not None and self.budget.search_wall_deadline is not None:
                return self._observe(ValidationResult(False,"UNKNOWN","SEARCH_RESOURCE_PROTECTION_UNKNOWN"),(column,))
            background = PredictionColumn(a.id,"BACKGROUND",(),resources or (), (0,0,0,0,0))
            r = self.check_pair(column, background)
            if not r.accepted:
                if r.status in ("CONFLICT","REJECTED","UNSAFE","INVALID"):
                    self.background_blockers.append(dict(source=column.aircraft_id,target=a.id,
                        snapshot_id=self.snapshot.version,reason=r.reason,
                        resource=next((x.resource for x in column.resources for y in background.resources
                            if x.resource==y.resource and x.start<y.end and y.start<x.end),())))
                return self._observe(r,(column,))
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
                    if (a.aircraft_id in self.decisions and b.aircraft_id in self.decisions
                            and a.kind!="BACKGROUND" and b.kind!="BACKGROUND" and len(self.joint_conflict_observations)<128):
                        resource=next((x.resource for x in a.resources for y in b.resources
                            if x.resource==y.resource and x.start<y.end and y.start<x.end),None)
                        observation=dict(aircraft_ids=sorted((a.aircraft_id,b.aircraft_id)),reason=r.reason or r.status,
                            snapshot_id=self.snapshot.version,context_scope=getattr(self.scope,"identity",None),
                            local_round=getattr(self,"local_round",None),refinement_round=getattr(self.budget,"refinement_rounds_used",None),
                            source="SELECTED_SELECTED_BATCH")
                        if resource is not None: observation["resource"]=resource
                        if observation not in self.joint_conflict_observations: self.joint_conflict_observations.append(observation)
                    conflicts.append((a,b)); first=first or r
                    if len(conflicts)>=max_conflicts: break
        self.cache_stats["batch_conflicts"]+=len(conflicts)
        return first or unknown or ValidationResult(True,"SAFE"),tuple(conflicts)

    def prefix(self, columns):
        out = []
        for c in columns:
            if self.scope is not None and c.aircraft_id not in self.decisions: raise ValueError("prefix outside scope")
            if c.kind in ("KEEP","WAIT") or not c.plans: continue
            p = c.plans[0]
            if _executable_start(self.snapshot,p.start_time) and p.aircraft_id not in self.snapshot.soft:
                out.append(p)
        return tuple(out)

    def check_commit_prefix(self, columns):
        if not self._time_ok(): return self._observe(self._time_rejection(),columns)
        prefix = self.prefix(columns)
        reusable = self.reuse_prefix_results and (self.prefix_evaluator is None or self.prefix_evaluator_identity is not None)
        key = (self.snapshot_key, self.prefix_evaluator_identity, tuple((p.trajectory_hash,repr(p)) for p in prefix))
        if reusable and key in self._prefix_results:
            self.cache_stats["prefix_hits"] += 1
            return self._prefix_results[key]
        if self.reuse_prefix_results: self.cache_stats["prefix_misses"] += 1
        if self.budget is not None and not self.budget.heavy_allowed("prefix"):
            return self._observe(ValidationResult(False,"UNKNOWN","SEARCH_HEAVY_HEADROOM"),columns)
        with (self.budget.measure("prefix_validation") if self.budget else nullcontext()), (self.budget.heavy_work("prefix") if self.budget else nullcontext()):
            if self.prefix_evaluator is not None:
                result = self.prefix_evaluator(prefix)
            else:
                from .world import evaluate_planning_snapshot
                result = evaluate_planning_snapshot(self.snapshot,prefix)
        if result.accepted and not getattr(result,"refinement_required",False) and self.budget is not None:
            self.budget.certificate_event(self.budget.certificate_source,tuple(columns),prefix)
        if reusable and result.accepted and result.status in ("ACCEPTED","SAFE","FEASIBLE","VALID") and not getattr(result,"refinement_required",False):
            self._prefix_results[key] = result
        return self._observe(result,columns)


class CandidateProvider:
    def __init__(self, reference_provider=None, planning_profile=PLANNING_PROFILE_LEGACY, advisor_runtime=None,
                 evaluation_profile=LEGACY_EVALUATION_PROFILE):
        candidate_policy_id(planning_profile)
        self.planning_profile=planning_profile
        self.evaluation_profile=validate_evaluation_profile(evaluation_profile)
        if evaluation_profile!=LEGACY_EVALUATION_PROFILE and planning_profile!=PLANNING_PROFILE_DOWNSTREAM:
            raise ValueError("stitched evaluation requires downstream_service_v8")
        self.references = reference_provider or ReferenceProvider()
        self.previous = {}
        self.aligned_starts=[]
        self._aligned_by_node={}
        self.validation_cache={}
        self.advisor_runtime=advisor_runtime if advisor_runtime is not None and advisor_runtime.enabled else None
        self.references.advisor_runtime=self.advisor_runtime

    def _describe_column(self,snapshot,column,context):
        from .stage2_downstream import describe_column
        if self.evaluation_profile==LEGACY_EVALUATION_PROFILE:
            return describe_column(snapshot,column,self.references,context)
        return describe_column(snapshot,column,self.references,context,evaluation_profile=self.evaluation_profile)

    def _check_frame_profile(self,context):
        if context.global_frame is not None:
            from .selective_replanning.frame import frame_evaluation_profile
            if frame_evaluation_profile(context.global_frame.policy)!=self.evaluation_profile:
                raise ValueError("candidate evaluation frame profile mismatch")

    def reserve_suffix_completion(self,snapshot,catalog,budget):
        """Freeze at most two existing partials after the first real solve."""
        if self.evaluation_profile!=STITCHED_EVALUATION_PROFILE: return
        if getattr(budget,"_suffix_completion_state",None) is not None: return
        from .selective_replanning.diagnostics import candidate_hash
        candidates=[]
        for aid,rows in catalog.columns.items():
            partials=[c for c in rows if c.kind=="MOVE" and c.plans and c.plans[-1].pad_id is None
                and c.node is not None and aid not in snapshot.soft and _executable_start(snapshot,c.plans[0].start_time)
                and not any(full.plans[:len(c.plans)]==c.plans and full.plans[-1].pad_id is not None
                    for full in rows if full.plans)]
            if partials:
                best=min(partials,key=lambda c:(c.downstream.get("estimated_touchdown",math.inf),candidate_hash(c)))
                candidates.append(best)
        candidates.sort(key=lambda c:(c.downstream.get("estimated_touchdown",math.inf),c.aircraft_id,candidate_hash(c)))
        budget._suffix_completion_state=dict(snapshot_id=snapshot.version,
            records=[dict(aircraft_id=c.aircraft_id,prefix_hashes=[p.trajectory_hash for p in c.plans],
                suffix_status="RESERVED",complete_candidate_in_catalog=False,wall_s=0.,
                priority_slice_attempted=False,skip_reason="AWAITING_FIRST_CP_GROWTH",
                _partial=c,_cursor=None) for c in candidates[:2]],retained=dict(catalog.columns))

    @staticmethod
    def suffix_completion_report(budget):
        state=getattr(budget,"_suffix_completion_state",None)
        if state is None: return None
        rows=[{k:v for k,v in row.items() if not k.startswith("_")} for row in state["records"]]
        for row in rows:
            row["pending"]=row["suffix_status"] in ("RESERVED","RUNNING")
            if row["pending"]: row["budget_exit"]=budget.generation_stop_reason() or "BOUNDED_GENERATOR_SLICE_NOT_FINISHED"
        return dict(policy_id="bounded_suffix_completion_v1",max_aircraft=2,attempts=rows)

    def _suffix_completion_steps(self,snapshot,a,budget,row):
        """Advance one shared reference attempt at a time; preserve epochs."""
        if row["suffix_status"] not in ("RESERVED","RUNNING"): return
        partial=row["_partial"]
        if row["_cursor"] is None: row["_cursor"]=self.references.build_steps(snapshot,a,partial.node,budget)
        row["suffix_status"]="RUNNING"
        while budget.available(a.id):
            started=time.perf_counter()
            try: next(row["_cursor"])
            except StopIteration as stop:
                row["wall_s"]+=time.perf_counter()-started
                ref=stop.value
                row["suffix_status"]=getattr(ref,"status","UNKNOWN")
                if ref is None or ref.status!="KNOWN" or not ref.plans or ref.plans[-1].pad_id is None: return
                combined=(*partial.plans,*ref.plans)
                replay_started=time.perf_counter()
                checked=self._reference_seed(snapshot,a,_node(snapshot,a),replace(ref,plans=combined),budget)
                row["wall_s"]+=time.perf_counter()-replay_started
                row["completion_validation"]="EXISTING_REFERENCE_SEED_REPLAY"
                if checked is None or checked[0]!=combined or checked[0][-1].pad_id is None:
                    row["suffix_status"]="NOT_ADMISSIBLE"; return
                yield ("COLUMN",combined,checked[1])
                return
            row["wall_s"]+=time.perf_counter()-started
            yield ("ATTEMPT",None,None)
        row["suffix_status"]="BUDGET_EXIT"
        row["budget_exit"]=budget.generation_stop_reason() or "PER_AIRCRAFT_BUDGET"

    def _pop_ready(self, heap, snapshot, a, budget, stage, extract, is_ready=None):
        """Select from at most 64 existing ready items before generation.

        Structural heap entries are still expanded by Stage2. Every descriptor
        supplied here has a materialized node and a concrete generator tuple.
        One work item is consumed per selection. The runtime may reuse a
        ranking for its configured consumption allowance, while new ready work,
        changed source/parent state or new coverage invalidates that ranking.
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
        if plans and getattr(context,"scope",None) is not None and a.id not in context.decisions: raise ValueError("candidate materialization outside scope")
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
            d=self._describe_column(snapshot,c,context)
            estimated_f=(max(d.failure,int(d.base_energy>=d.diagnostics["available_energy_threshold"]))
                if self.evaluation_profile==STITCHED_EVALUATION_PROFILE else f)
            c=replace(c,downstream=d.diagnostics,objective=(estimated_f,-l,d.unknown+max(0,d.base_energy-d.reserve_energy)*.1/3600.,d.base_energy*.1,(d.touchdown-round(snapshot.time*10))/10.))
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
        targets=range(snapshot.config.pads.count if node.ring=="inner" else ring_config({"outer":"middle","middle":"inner"}[node.ring],snapshot.config).slot_count)
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
            blocked=(self.references.immutable_resource_conflict(snapshot,a,node,target,start,duration,context)
                if getattr(budget,"_selective_progressive_v2",False) else
                self._current_progress_reservation_conflict(snapshot,a,node,target,start,duration))
            if blocked:
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

    def _target_coverage_steps(self,snapshot,a,node,budget,context,state):
        """One immediate descriptor per target before duration refinement.

        This cursor is resumed round-robin across aircraft. Current occupancy
        changes ordering only; selected owners may release jointly.
        """
        seen=set(); pending=[]
        for cursor in self._offset_cursors(snapshot,a,node,budget,0.,include_aligned=False):
            descriptor=self._next_descriptor(snapshot,a,node,budget,cursor,seen,True)
            if descriptor is None:
                state["screened_without_descriptor_targets"].add(cursor["target"]); continue
            target,start,duration,_=descriptor
            state["viable_targets"].add(target)
            if target in state["attempted_targets"]: continue
            occupied=self._current_progress_reservation_conflict(snapshot,a,node,target,start,duration)
            pending.append((occupied,start+duration,target,descriptor))
        pending.sort(key=lambda row:row[:3])
        yield ("SCREEN",None,None)
        for _,_,target,descriptor in pending:
            if not budget.available(a.id): return
            _,start,duration,_=descriptor
            ok,p=self.references.generate(snapshot,a,node,target,start,duration,budget)
            if not ok: return
            state["attempted_targets"].add(target)
            budget.screening["executable_v3_target_queries"]+=1
            if p is not None:
                q=self.references.move_energy(node,p,snapshot.config)
                if node.soc-q[1]>=(.25 if p.pad_id is not None else 0.) and node.peak+q[2]<=120:
                    terminal=(PlanningNode(p.target_ring,p.target_slot,p.end_time,node.soc-q[1],node.peak+q[2],
                                          node.version+1,node.error_j+q[3],node.peak_error_s+q[4])
                              if p.target_ring is not None else node)
                    yield ("COLUMN",(p,),self._emergency_prediction(snapshot,a,terminal)); continue
                budget.screening["executable_v3_target_energy_rejected"]+=1
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

    def iter_progressive_columns(self,snapshot,budget,context,*,scope=None,early_publish=False):
        """Resume one finite search across up to three growing catalog batches."""
        from .stage3_learning.catalog_control import retain_validated
        self.aligned_starts=[]; self._aligned_by_node={}
        runtime=self.advisor_runtime
        if early_publish:
            budget._selective_progressive_v2=True
            budget._strict_resource_context=context
        scope=scope or getattr(context,"scope",None)
        self._check_frame_profile(context)
        if scope is not None:
            scope.validate(snapshot)
            if context.scope != scope: raise ValueError("candidate/context scope mismatch")
            budget.allowed_replan_ids=set(scope.replan_ids)
            if self.planning_profile==PLANNING_PROFILE_DOWNSTREAM and context.global_frame is None:
                raise ValueError("scoped downstream generation requires preselection frame")
        aircraft=[a for a in _decision_aircraft(snapshot) if scope is None or a.id in scope.replan_ids]
        nodes={a.id:_node(snapshot,a) for a in aircraft}
        refs={a.id:self.references.cache.get((snapshot.config.config_hash,nodes[a.id].ring,
            nodes[a.id].slot,nodes[a.id].time,nodes[a.id].version,a.id),NormalReference("UNKNOWN")) for a in aircraft}
        slacks={a.id:energy_slack(snapshot,a,refs[a.id]) for a in aircraft}
        ordered=sorted(aircraft,key=lambda a:slacks[a.id])
        groups={ring:[a for a in ordered if nodes[a.id].ring==ring] for ring in ("outer","middle","inner")}
        order=[group[index] for index in range(max(map(len,groups.values()),default=0))
            for group in groups.values() if index<len(group)]
        columns={a.id:[self._column(snapshot,a,(),kind,context,refs[a.id]) for kind in ("KEEP","WAIT")] for a in aircraft}
        if runtime is not None: runtime.begin_snapshot(snapshot,budget,[a.id for a in aircraft])
        target=2
        def publish(downstream=None, retained=None):
            hook=getattr(runtime,"update_retained_context",None)
            if hook is not None: hook(snapshot,columns if retained is None else retained,refs,context,downstream=downstream)
        def add(a,plans,emergency="NONE",preserve=False):
            if not plans: return False
            ref=(self.references._reference(snapshot,nodes[a.id],plans,"catalog_minimum")
                if plans[-1].pad_id is not None else refs[a.id])
            c=self._column(snapshot,a,plans,"MOVE",context,ref,emergency)
            if c is None or any(old.signature==c.signature for old in columns[a.id]): return False
            hits=context.cache_stats["single_hits"]; w=time.perf_counter(); cpu=time.process_time()
            validation=context.check_single(c)
            hook=getattr(runtime,"record_column_validation",None)
            if hook is not None: hook(c,validation,time.perf_counter()-w,time.process_time()-cpu,context.cache_stats["single_hits"]>hits)
            if preserve and validation.accepted:
                columns[a.id].append(c); retained=True
            else:
                retained=validation.accepted and retain_validated(columns[a.id],c,snapshot.time,target,budget.screening,
                    effective_time=getattr(getattr(snapshot,'planning_timing',None),'effective_time',None))
            if runtime is not None: runtime.record_column(c,validation,retained)
            if retained:
                if plans[-1].pad_id is not None and (refs[a.id].status!="KNOWN" or ref.energy_j<refs[a.id].energy_j): refs[a.id]=ref
            publish()
            return retained
        publish()
        for a in order:
            completion_state=getattr(budget,"_suffix_completion_state",None)
            if completion_state is not None and completion_state["snapshot_id"]==snapshot.version:
                for old in completion_state["retained"].get(a.id,()):
                    if old.plans: add(a,old.plans,old.emergency_prediction,preserve=True)
            previous=self._previous_chain(snapshot,a,nodes[a.id],budget)
            if previous: add(a,previous)
            retained=getattr(budget,"_selective_retained_columns",{}).get(a.id) if early_publish else None
            if (retained is not None and getattr(budget,"_selective_retained_snapshot",None)==snapshot.version
                    and retained.plans):
                # Already certified in this real immutable snapshot; recreate
                # and validate against the new scope, never translate epochs.
                add(a,retained.plans,retained.emergency_prediction)
        stages={}; target_states={}; family_turn={}
        for a in order:
            n=nodes[a.id]
            stages[a.id]=[("progress",self._progress_steps(snapshot,a,n,budget,context)),
                ("reference",self.references.build_steps(snapshot,a,n,budget))]
            if self.planning_profile==PLANNING_PROFILE_DOWNSTREAM:
                stages[a.id].append(("delayed",self._delayed_steps(snapshot,a,n,budget,context)))
            stages[a.id].append(("alternative",self._paths(snapshot,a,n,budget,context)))
            if getattr(budget,"_selective_target_coverage_order",getattr(budget,"_selective_executable_v3",False)):
                from .selective_replanning.near_term import adjustment_access
                access=adjustment_access(snapshot,a,n)
                reused={c.first_target[2] if n.ring=="inner" else c.first_target[1]
                    for c in columns[a.id] if c.kind=="MOVE" and c.plans and _executable_start(snapshot,c.plans[0].start_time)}
                state=dict(attempted_targets=set(reused),reused_targets=set(reused),viable_targets=set(),
                    screened_without_descriptor_targets=set(),adjustment_access=access)
                target_states[a.id]=state
                if access["adjustable"]:
                    stages[a.id].insert(0,("target_coverage",self._target_coverage_steps(snapshot,a,n,budget,context,state)))
        initial_visited=set()
        def catalog_with_coverage(phase):
            catalog=self._progressive_catalog(snapshot,budget,context,columns,refs,ordered)
            if early_publish:
                catalog=replace(catalog,generation_coverage=dict(phase=phase,
                    initial_visited_ids=sorted(initial_visited),initial_unvisited_ids=sorted(set(columns)-initial_visited),
                    all_initial_aircraft_visited=len(initial_visited)==len(columns),
                    claim="OBSERVED_BOUNDED_CATALOG_ONLY"))
                if target_states:
                    report={aid:dict(attempted_targets=sorted(row["attempted_targets"]),reused_targets=sorted(row["reused_targets"]),
                        viable_targets=sorted(row["viable_targets"]),remaining_unseen_targets=sorted(row["viable_targets"]-row["attempted_targets"]),
                        screened_without_descriptor_targets=sorted(row["screened_without_descriptor_targets"]),
                        adjustment_access=row["adjustment_access"],claim="BOUNDED_DESCRIPTOR_QUERIES_NOT_FEASIBILITY")
                        for aid,row in target_states.items()}
                    catalog=replace(catalog,generation_coverage=dict(catalog.generation_coverage,target_attempt_coverage=report))
            completion=self.suffix_completion_report(budget)
            if completion is not None:
                catalog=replace(catalog,generation_coverage=dict(catalog.generation_coverage or {},suffix_completion=completion))
            return catalog
        try:
            for a in order:
                if not budget.available(): break
                if early_publish and not budget.available(a.id): continue
                initial_kind,progress=stages[a.id][0]
                while budget.available(a.id):
                    try: item=next(progress)
                    except StopIteration:
                        stages[a.id].remove((initial_kind,progress)); break
                    if item[0]=="COLUMN":
                        add(a,item[1],item[2]); break
                    if item[0]=="ATTEMPT": break
                budget.screening["progressive_initial_aircraft_visited"]+=1
                initial_visited.add(a.id)
            hook=getattr(runtime,"complete_initial_coverage",None)
            if hook is not None: hook()
            completed_first_solve=getattr(budget,"_suffix_completion_state",None) is not None
            if (early_publish or self.evaluation_profile==STITCHED_EVALUATION_PROFILE) and not (
                    self.evaluation_profile==STITCHED_EVALUATION_PROFILE and completed_first_solve):
                catalog=catalog_with_coverage("INITIAL_PROGRESS")
                publish(catalog.downstream,catalog.columns)
                yield catalog
                publish(catalog.downstream,catalog.columns)
            state=getattr(budget,"_suffix_completion_state",None)
            completion_rows={} if state is None or state["snapshot_id"]!=snapshot.version else {row["aircraft_id"]:row for row in state["records"]}
            for a in order:
                row=completion_rows.get(a.id)
                if row is not None and row["suffix_status"] in ("RESERVED","RUNNING"):
                    stages[a.id].insert(0,("suffix_completion",self._suffix_completion_steps(snapshot,a,budget,row)))
            def add_completion(a,item):
                if item[0]!="COLUMN": return
                validation_started=time.perf_counter()
                retained=add(a,item[1],item[2])
                row=completion_rows[a.id]
                row["complete_candidate_in_catalog"]=retained or any(c.plans==item[1] for c in columns[a.id])
                row["single_validation_wall_s"]=time.perf_counter()-validation_started
                row["wall_s"]+=row["single_validation_wall_s"]
            if completion_rows and not state.get("priority_slice_completed",False):
                # One global micro-round, in the frozen reservation order,
                # before ordinary aircraft can consume the remaining budget.
                # Keep the existing cursor for later family round-robin work.
                state["priority_slice_completed"]=True
                aircraft_by_id={a.id:a for a in order}
                for row in state["records"]:
                    a=aircraft_by_id.get(row["aircraft_id"])
                    if a is None:
                        row["skip_reason"]="OUTSIDE_CURRENT_SCOPE"; continue
                    family=next((family for family in stages[a.id] if family[0]=="suffix_completion"),None)
                    if family is None:
                        row["skip_reason"]="SUFFIX_ALREADY_TERMINAL"; continue
                    if not budget.available(a.id):
                        row["skip_reason"]=budget.generation_stop_reason() or "PER_AIRCRAFT_BUDGET"; continue
                    row["priority_slice_attempted"]=True; row["skip_reason"]=None
                    try: item=next(family[1])
                    except StopIteration:
                        stages[a.id].remove(family)
                    else: add_completion(a,item)
            for round_index,target in enumerate((2,4,8)):
                quota=max(1,math.ceil(budget.max_per_aircraft*(.25,.5,1.)[round_index]))
                visited=set()
                # Each pass advances every aircraft and every construction family
                # once before spending another slice on any one aircraft.
                for _ in range(4096):
                    advanced=False
                    gaps=getattr(budget,"_selective_coverage_gaps",{}) if early_publish else {}
                    round_order=sorted(order,key=lambda a:(not bool(gaps.get(a.id)),order.index(a))) if gaps else order
                    for a in round_order:
                        if not budget.available(a.id) or budget.attempts[a.id]>=quota: continue
                        pending_completion=a.id in completion_rows and completion_rows[a.id]["suffix_status"] in ("RESERVED","RUNNING")
                        if (not target_states and not pending_completion and a.id in visited and sum(c.kind=="MOVE" for c in columns[a.id])>=target): continue
                        visited.add(a.id)
                        families=tuple(stages[a.id])
                        if target_states:
                            coverage=next((family for family in families if family[0]=="target_coverage"),None)
                            if coverage is not None and len(target_states[a.id]["attempted_targets"])<2:
                                families=(coverage,)
                            else:
                                # After two targets, keep full/delayed chains in
                                # the same round-robin as unseen targets. A new
                                # duration on an old target waits behind these.
                                pool=tuple(family for family in families if coverage is None or family[0] not in ("progress","alternative"))
                                if pool:
                                    turn=family_turn.get(a.id,0)%len(pool)
                                    families=(pool[turn],); family_turn[a.id]=turn+1
                                else: families=()
                        elif gaps:
                            missing=gaps.get(a.id,())
                            priority=("progress" if "NO_ELIGIBLE_IMMEDIATE_MOVE_OBSERVED" in missing else
                                "reference" if "NO_COMPLETE_CHAIN_OBSERVED" in missing else
                                "delayed" if "NO_DELAYED_MOVE_OBSERVED" in missing else "alternative")
                            families=tuple(sorted(families,key=lambda row:(row[0]!="suffix_completion",row[0]!=priority)))
                        for kind,gen in families:
                            if not budget.available(a.id) or budget.attempts[a.id]>=quota: break
                            advanced=True
                            try: item=next(gen)
                            except StopIteration as stop:
                                stages[a.id].remove((kind,gen))
                                if kind=="reference" and stop.value is not None and stop.value.status=="KNOWN":
                                    ref=stop.value; refs[a.id]=ref
                                    seed=self._reference_seed(snapshot,a,nodes[a.id],ref,budget)
                                    if seed is not None: add(a,*seed)
                                    if self.planning_profile==PLANNING_PROFILE_DOWNSTREAM:
                                        stages[a.id].insert(0,("delayed_reference",self._delayed_reference_seed(snapshot,a,nodes[a.id],ref,budget)))
                                continue
                            if kind=="delayed_reference": add(a,*item)
                            elif kind!="reference" and item[0]=="COLUMN":
                                if kind=="suffix_completion": add_completion(a,item)
                                else: add(a,item[1],item[2])
                    if not advanced or not budget.available(): break
                catalog=catalog_with_coverage("GROWTH_"+str(round_index))
                publish(catalog.downstream,catalog.columns)
                yield catalog
                publish(catalog.downstream,catalog.columns)
                if not budget.available() or not any(stages.values()): break
        finally:
            if runtime is not None: runtime.finish_snapshot(budget)

    def _progressive_catalog(self,snapshot,budget,context,columns,refs,ordered):
        """Materialize current estimates without restarting construction cursors."""
        slacks={a.id:energy_slack(snapshot,a,refs[a.id]) for a in ordered}
        for a in ordered:
            columns[a.id][:2]=[self._column(snapshot,a,(),k,context,refs[a.id]) for k in ("KEEP","WAIT")]
            if refs[a.id].status=="KNOWN":
                n=_node(snapshot,a)
                self.references.cache[(snapshot.config.config_hash,n.ring,n.slot,n.time,n.version,a.id)]=refs[a.id]
        values={aid:tuple(rows) for aid,rows in columns.items()}
        downstream=None
        if self.planning_profile==PLANNING_PROFILE_DOWNSTREAM:
            from .stage2_downstream import JointDownstream,describe_column,ticks
            self._check_frame_profile(context)
            demands={c.signature:self._describe_column(snapshot,c,context) for rows in values.values() for c in rows}
            if context.global_frame is not None:
                # FOLLOW is precisely the preselection committed continuation.
                # Newly searched references cannot improve a fixed background
                # only when it happens to have been selected by FULL_H.
                baseline_demands=context.global_frame.baseline_by_id
                demands={c.signature:(baseline_demands[c.aircraft_id] if not c.plans else demands[c.signature])
                    for rows in values.values() for c in rows}
                downstream=context.global_frame.downstream(demands,set(values))
                baseline={aid:rows[0] for aid,rows in values.items()}
                values={aid:tuple(replace(c,objective=downstream.key(tuple(dict(baseline,**{aid:c}).values())),
                    downstream=demands[c.signature].diagnostics) for c in rows) for aid,rows in values.items()}
                return CandidateCatalog(MappingProxyType(values),MappingProxyType(slacks),MappingProxyType(dict(budget.attempts)),
                    tuple(values),0,MappingProxyType({aid:ref.status for aid,ref in refs.items()}),downstream,
                    context.scope,MappingProxyType(dict(downstream.fixed_demands)))
            fixed=[]; order=[a.id for a in sorted(ordered,key=lambda a:slacks[a.id])]; others=[]
            for a in snapshot.aircraft.values():
                if a.id in values or _node(snapshot,a) is None: continue
                c=self._column(snapshot,a,(),"WAIT",context)
                if c is None: continue
                d=self._describe_column(snapshot,c,context)
                fixed.append(replace(d,unknown=1,diagnostics=dict(d.diagnostics,reference_status="UNKNOWN",reference_unknown_flag=True,
                    uncertainty_risk_penalty_wh=1.,reference_estimate_kind="noncandidate_known_aircraft_proxy")))
                others.append((energy_slack(snapshot,a,NormalReference("UNKNOWN")),a.id))
            scores={aid:score for score,aid in others}; scores.update(slacks)
            order=sorted(order+[aid for _,aid in others],key=lambda aid:scores[aid])
            calendars=tuple((r.resource,ticks(r.start),ticks(r.end),r.aircraft_id) for r in snapshot.pads if math.isfinite(r.end))
            downstream=JointDownstream(demands,calendars,order,now=ticks(snapshot.time),pad_count=snapshot.config.pads.count,fixed_demands=fixed,
                evaluation_profile=self.evaluation_profile)
            baseline={aid:rows[0] for aid,rows in values.items()}
            values={aid:tuple(replace(c,objective=downstream.key(tuple(dict(baseline,**{aid:c}).values())),
                downstream=demands[c.signature].diagnostics) for c in rows) for aid,rows in values.items()}
        return CandidateCatalog(MappingProxyType(values),MappingProxyType(slacks),MappingProxyType(dict(budget.attempts)),
            tuple(values),0,MappingProxyType({aid:ref.status for aid,ref in refs.items()}),downstream)

    def get_columns(self,snapshot,budget=None,context=None,*,scope=None):
        budget=budget or PlanningBudget(); context=context or EvaluationContext(snapshot,budget=budget,scope=scope)
        scope=scope or getattr(context,"scope",None)
        if scope is not None:
            scope.validate(snapshot)
            if context.scope != scope: raise ValueError("candidate/context scope mismatch")
            # Same scoped progressive generator, including all cold references;
            # the legacy alternative entry cannot bypass selection.
            latest=None
            with budget.candidate_work():
                for latest in self.iter_progressive_columns(snapshot,budget,context,scope=scope): pass
            return latest
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
                  if _executable_start(snapshot,_node(snapshot,a).time) and a.id not in snapshot.soft}
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
            demands={c.signature:self._describe_column(snapshot,c,context) for values in columns.values() for c in values}
            fixed=[]; all_order=list(a.id for a in sorted(ordered,key=lambda a:slacks[a.id]))
            # Declared but unreserved rescue traffic remains visible demand.
            # No future manifest or unreleased request is consulted.
            others=[]
            for a in snapshot.aircraft.values():
                if a.id in columns or _node(snapshot,a) is None: continue
                n=_node(snapshot,a)
                c=self._column(snapshot,a,(),"WAIT",context)
                if c is None: continue
                d=self._describe_column(snapshot,c,context)
                d=replace(d,unknown=1,diagnostics=dict(d.diagnostics,reference_status="UNKNOWN",reference_unknown_flag=True,
                    uncertainty_risk_penalty_wh=1.,reference_estimate_kind="noncandidate_known_aircraft_proxy"))
                fixed.append(d); others.append((energy_slack(snapshot,a,NormalReference("UNKNOWN")),a.id))
            all_order=sorted(all_order+[aid for _,aid in others],key=lambda aid:slacks.get(aid,next((s for s,i in others if i==aid),())))
            calendars=tuple((r.resource,ticks(r.start),ticks(r.end),r.aircraft_id) for r in snapshot.pads if math.isfinite(r.end))
            downstream=JointDownstream(demands,calendars,all_order,now=ticks(snapshot.time),pad_count=snapshot.config.pads.count,fixed_demands=fixed,
                evaluation_profile=self.evaluation_profile)
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
        budget.bind_catalog(candidates)
        budget.certificate_source="energy_slack_hint"
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
            objective=_selection_key(candidates,selected)
            prefix=context.prefix(selected)
            budget.certificate_event("energy_slack_hint",selected,prefix,event="CERTIFIED_OBJECTIVE",objective=objective)
            return PlanningResult(selected,prefix,e,"FEASIBLE",objective,certified=True)
        # One local repair: rejected predictive release cannot authorize execution.
        with budget.measure("fallback"):
            keep=tuple(cols[0] for cols in candidates.columns.values())
            e=context.check_commit_prefix(keep) if budget.evaluation_available() else None
        return PlanningResult(keep,(),e,"PREFIX_NOT_EXECUTABLE",_selection_key(candidates,keep),certified=bool(e and e.accepted))


class RollingHorizonCPSATScheduler:
    method="rh_cp_sat"

    @staticmethod
    def _continuation(snapshot,candidates,context):
        # Full column contents include resources, objective, node and energy
        # references. A trajectory-only signature is insufficient for proofs.
        downstream=candidates.downstream
        downstream_inputs=vars(downstream) if hasattr(downstream,'__dict__') else downstream
        evaluator_binding=(getattr(context,'prefix_evaluator_identity',None),id(getattr(context,'prefix_evaluator',None)))
        inputs=(getattr(context,'snapshot_key',None),snapshot,getattr(context,'snapshot',None),
            getattr(getattr(context,'scope',None),'identity',None),getattr(getattr(context,'global_frame',None),'identity',None),
            evaluator_binding,
            tuple((aid,tuple(rows)) for aid,rows in candidates.columns.items()),candidates.slacks,downstream_inputs)
        identity=hashlib.sha256(repr(inputs).encode()).hexdigest()
        previous=getattr(context,'_cp_continuation',None)
        reused=previous is not None and previous['identity']==identity
        invalidations=getattr(context,'_cp_continuation_invalidations',0)+int(previous is not None and not reused)
        context._cp_continuation_invalidations=invalidations
        if not reused:
            if previous is not None and previous['evaluator_binding']!=evaluator_binding and hasattr(context,'_prefix_results'):
                context._prefix_results.clear()
            context._cp_continuation=dict(identity=identity,incumbent=None,exclusions=set(),
                single_exclusions=set(),nogoods=set(),single_results={},resource_complete=False,evaluator_binding=evaluator_binding)
        state=context._cp_continuation
        report=dict(state_reused=reused,state_invalidated=previous is not None and not reused,
            state_invalidations=invalidations,incumbent_source=None,caller_incumbent_accepted=False,
            pair_cuts_reused=len(state['exclusions']) if reused else 0,
            single_cuts_reused=len(state['single_exclusions']) if reused else 0,
            prefix_nogoods_reused=len(state['nogoods']) if reused else 0,
            single_results_reused=len(state['single_results']) if reused else 0,solver_internal_state_reused=False)
        return state,report

    @staticmethod
    def _certified_seed(seed,candidates,context,budget,*,known=None):
        if seed is None or not seed.certified: return None
        if len(seed.selected)!=len(candidates.columns) or {c.aircraft_id for c in seed.selected}!=set(candidates.columns): return None
        selected=[]
        for column in seed.selected:
            match=next((c for c in candidates.columns[column.aircraft_id] if c.signature==column.signature and repr(c)==repr(column)),None)
            if match is None: return None
            selected.append(match)
        selected=tuple(selected)
        objective=_selection_key(candidates,selected)
        prefix=context.prefix(selected)
        if objective!=seed.objective or repr(prefix)!=repr(seed.prefix): return None
        # Only a certificate produced in this exact continuation can skip the
        # validators. Caller flags and trajectory membership do not certify it.
        if known is not None and seed.evaluation is known.evaluation and seed.objective==known.objective and repr(seed.selected)==repr(known.selected):
            return replace(seed,selected=selected,objective=objective)
        if not budget.evaluation_available(): return None
        result,_=context.check_batch(selected)
        evaluation=context.check_commit_prefix(selected) if result.accepted and budget.evaluation_available() else None
        if evaluation is None or not evaluation.accepted or getattr(evaluation,'refinement_required',False): return None
        return replace(seed,selected=selected,prefix=prefix,evaluation=evaluation,objective=objective,certified=True)

    def plan(self,snapshot,candidates,context,budget,*,incumbent=None):
        if getattr(context,"scope",None) is not None:
            if set(candidates.columns)!=set(context.scope.replan_ids) or candidates.scope!=context.scope:
                raise ValueError("CP catalog/scope binding mismatch")
        try: from ortools.sat.python import cp_model
        except ImportError as exc:
            raise RuntimeError("RH-CP-SAT requires OR-Tools; no replacement solver is used") from exc
        state,reuse=self._continuation(snapshot,candidates,context)
        budget.bind_catalog(candidates)
        budget.certificate_source="cp_seed"
        # Bound the hint on the original normal-dispatch clock. EvaluationContext
        # also observes this phase limit, including nested safety checks.
        previous_budget=getattr(context,"budget",None)
        if isinstance(context,EvaluationContext): context.budget=budget
        try:
            caller_incumbent=incumbent
            incumbent=self._certified_seed(state['incumbent'],candidates,context,budget,known=state['incumbent'])
            if incumbent is not None: reuse['incumbent_source']='continuation'
            caller=self._certified_seed(caller_incumbent,candidates,context,budget,known=state['incumbent'])
            reuse['caller_incumbent_accepted']=caller is not None
            if caller is not None and (incumbent is None or caller.objective<incumbent.objective):
                incumbent=caller;reuse['incumbent_source']='caller_revalidated'
            if incumbent is not None:
                hint=incumbent
            else:
                hint_started=time.perf_counter()
                try:
                    with budget.measure("hint_construction"), budget.stage_limit(
                        None if budget.hint_seconds is None else max(0.,budget.hint_seconds-budget.hint_wall_used),
                        "HINT_BUDGET",budget.solver_reserve_seconds):
                        hint=EnergySlackScheduler().plan(snapshot,candidates,context,budget,
                            validation_reserve_seconds=budget.hint_validation_reserve_seconds)
                finally:
                    budget.hint_wall_used+=time.perf_counter()-hint_started
                if hint.certified: reuse['incumbent_source']='energy_slack_hint'
        finally:
            if isinstance(context,EvaluationContext): context.budget=previous_budget
        hint_key=hint.objective if hint.certified else None
        incumbent=hint if hint.certified else None
        exclusions=state['exclusions'];single_exclusions=state['single_exclusions']
        nogoods=state['nogoods'];model_single_results=state['single_results']
        records=[]; solve_used=0.; all_optimal=True; final_solver_key=None
        call_limit=getattr(budget,"solver_call_seconds",None)
        def solver_remaining():
            return min(budget.solver_seconds-budget.solver_wall_used,
                math.inf if call_limit is None else max(0.,call_limit-solve_used))
        batched_conflicts=0
        ids=tuple(candidates.columns)
        move_aircraft=sum(any(c.kind=="MOVE" and c.plans for c in columns) for columns in candidates.columns.values())
        skip_reason="EMPTY_CATALOG" if not ids else "NO_MOVE_OPTIONS" if not move_aircraft else None
        stop_reason=None; resource_complete=state['resource_complete']; model_complete=True
        if skip_reason is None:
            with budget.measure("resource_conflict_build"):
                previously_complete=resource_complete
                resource_complete=True
                for i,j in combinations(ids,2):
                    if previously_complete: break
                    if not budget.preparation_available(): resource_complete=False; break
                    for c in candidates.columns[i]:
                        if not budget.preparation_available(): resource_complete=False; break
                        for d in candidates.columns[j]:
                            if not budget.preparation_available(): resource_complete=False; break
                            if context.resource_conflict(c,d): exclusions.add(tuple(sorted((c.signature,d.signature))))
                        if not resource_complete: break
                    if not resource_complete: break
                state['resource_complete']=resource_complete
            for _ in range(max(0,budget.max_refinement_rounds-budget.refinement_rounds_used)):
                if not budget.search_available():
                    stop_reason=budget.evaluation_stop_reason() if not budget.evaluation_available() else "SOFT_BUDGET"
                    break
                if solver_remaining()<=0:
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
                            if r.status in ("CONFLICT","REJECTED","UNSAFE","INVALID"):
                                single_exclusions.add(c.signature);model.Add(variables[c.signature]==0)
                            else: model_complete=False
                    if getattr(context,"diagnostics",None) is not None:
                        context.diagnostics.model(candidates.columns,exclusions,single_exclusions,nogoods,refinement)
                    if incumbent:
                        chosen={c.signature for c in incumbent.selected}
                        for key,v in variables.items(): model.AddHint(v,int(key in chosen))
                    shared_expressions=(candidates.downstream.cp_expressions(model,variables,lookup) if candidates.downstream is not None else None)
                    model_scale=dict(decision_variables=len(variables),auxiliary_variables=len(model.Proto().variables)-len(variables),constraints=len(model.Proto().constraints))
                invalid=False; carry=0.
                for phase,share in enumerate(PHASE_SHARES):
                    if not budget.search_available():
                        all_optimal=False
                        stop_reason=budget.evaluation_stop_reason() if not budget.evaluation_available() else "SOFT_BUDGET"
                        break
                    if solver_remaining()<=0:
                        all_optimal=False; stop_reason="SOLVER_ALLOCATION_EXHAUSTED"; break
                    validation_headroom=budget.solver_validation_headroom()
                    deadline_remaining=max(0.,budget.remaining_seconds()-validation_headroom)
                    limit=min(share*budget.solver_seconds/sum(PHASE_SHARES)+carry,solver_remaining(),deadline_remaining)
                    if limit<=0:
                        all_optimal=False; stop_reason="NO_SOLVER_TIME"; break
                    expression=(shared_expressions[phase] if shared_expressions is not None else
                                sum(solver_coefficients(c)[phase]*variables[key] for key,c in lookup.items()))
                    model.Minimize(expression)
                    solver=cp_model.CpSolver(); solver.parameters.num_search_workers=1; solver.parameters.random_seed=71
                    with budget.measure("cp_solve"):
                        # Model finalization and solver construction also use
                        # the same wall allowance; cap again at the Solve edge.
                        deadline_remaining=max(0.,budget.remaining_seconds()-validation_headroom)
                        limit=min(limit,solver_remaining(),deadline_remaining)
                        if limit<=0:
                            all_optimal=False
                            stop_reason=budget.evaluation_stop_reason() if not budget.evaluation_available() else "SOFT_BUDGET"
                            break
                        solver.parameters.max_time_in_seconds=limit
                        started=time.perf_counter(); status=solver.Solve(model); used=time.perf_counter()-started
                    solve_used+=used; budget.solver_wall_used+=used; carry=max(0.,limit-used)
                    rec={"round":refinement,"phase":phase,"status":solver.StatusName(status),"wall_s":used,
                         "model_scale":model_scale,
                         "allocated_seconds":limit,"remaining_deadline_s":deadline_remaining,
                         "search_validation_headroom_s":validation_headroom,
                         "bound":solver.BestObjectiveBound(),"branches":solver.NumBranches(),
                         "conflicts":solver.NumConflicts(),"objective":None}
                    records.append(rec)
                    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
                        all_optimal=False; stop_reason="SOLVER_"+solver.StatusName(status); break
                    selected=tuple(c for key,c in lookup.items() if solver.Value(variables[key]))
                    if getattr(context,"diagnostics",None) is not None:
                        rec.update(context.diagnostics.selection(candidates,selected))
                    rec["objective"]=solver.ObjectiveValue()
                    if candidates.downstream is not None:
                        rec["canonical_selected_integer_key"]=_selection_solver_key(candidates,selected)
                        rec["model_integer_key"]=tuple(solver.Value(expr) for expr in shared_expressions)
                    with budget.measure("selected_validation"):
                        budget.certificate_source="cp_proposal"
                        if not budget.evaluation_available():
                            r,conflicts=ValidationResult(False,"UNKNOWN",budget.evaluation_stop_reason()),()
                        elif hasattr(context,"check_batch_conflicts"):
                            r,conflicts=context.check_batch_conflicts(selected)
                        else:
                            r,conflict=context.check_batch(selected)
                            conflicts=(conflict,) if conflict and r.status!="UNKNOWN" else ()
                        evaluation=context.check_commit_prefix(selected) if r.accepted and budget.evaluation_available() else None
                    rec["batch_conflict_count"]=len(conflicts)
                    batched_conflicts+=len(conflicts)
                    rec["validation_status"]=r.status; rec["validation_reason"]=r.reason
                    rec["prefix_status"]=getattr(evaluation,"status","NOT_CHECKED")
                    rec["prefix_reason"]=getattr(evaluation,"reason",None)
                    if not r.accepted or evaluation is None or not evaluation.accepted:
                        if not r.accepted and r.status not in ("CONFLICT","REJECTED","UNSAFE","INVALID"):
                            stop_reason="VALIDATION_UNKNOWN"; all_optimal=False; break
                        if r.accepted and (evaluation is None or evaluation.status not in ("CONFLICT","REJECTED","UNSAFE","INVALID")):
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
                    budget.certificate_event("cp_proposal",selected,proposal.prefix,event="CERTIFIED_OBJECTIVE",objective=key)
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
                evaluation=context.check_commit_prefix(keep) if budget.evaluation_available() else None
            incumbent=PlanningResult(keep,(),evaluation,"NO_VERIFIED_INCUMBENT",_selection_key(candidates,keep),certified=bool(evaluation and evaluation.accepted))
        incumbent_solver_key=_selection_solver_key(candidates,incumbent.selected)
        if incumbent.certified: state['incumbent']=incumbent
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
                                  "continuation":reuse,
                                  "fallback":incumbent.status=="NO_VERIFIED_INCUMBENT"}))


class NormalSchedulerCallback:
    """Callable SchedulerPort adapter; runner owns admission/emergency timing."""
    def __init__(self,method="energy_slack",candidate_provider=None,budget_profile=QUALITY_BUDGET_PROFILE,planning_profile=PLANNING_PROFILE_LEGACY,
                 advisor_runtime=None,budget_scale=1.,online_policy=None,prefix_reuse=None,selective_method=None,global_frame=None,
                 selective_rule_variant="baseline",opportunity_seed_offset=0,diagnostics=None,search_policy=None,
                 evaluation_profile=LEGACY_EVALUATION_PROFILE):
        self.diagnostics=diagnostics
        self.evaluation_profile=validate_evaluation_profile(evaluation_profile)
        if evaluation_profile!=LEGACY_EVALUATION_PROFILE and planning_profile!=PLANNING_PROFILE_DOWNSTREAM:
            raise ValueError("stitched evaluation requires downstream_service_v8")
        if selective_method not in (None,"FULL_H","RULE_LOCAL_K8"): raise ValueError("unknown Stage A selective method")
        self.search_policy=resolve_search_policy(selective_rule_variant,search_policy)
        if opportunity_seed_offset not in (0,1,2): raise ValueError("diagnostic opportunity offset must be 0..2")
        self.opportunity_seed_offset=opportunity_seed_offset
        if selective_method is None and selective_rule_variant!="baseline": raise ValueError("selective rule variant requires selective method")
        if selective_method is not None:
            if method!="rh_cp_sat" or planning_profile!=PLANNING_PROFILE_DOWNSTREAM: raise ValueError("Stage A requires rh_cp_sat/downstream_service_v8")
            if advisor_runtime is not None or getattr(candidate_provider,"advisor_runtime",None) is not None: raise ValueError("Stage A disables old descriptor ranker")
            if budget_scale!=1.: raise ValueError("Stage A freezes candidate budget")
            if online_policy not in (None,"progressive_v1"): raise ValueError("Stage A requires progressive_v1")
            online_policy="progressive_v1"
        self.selective_method,self.global_frame=selective_method,global_frame
        self.selective_rule_variant=selective_rule_variant
        if selective_method is not None:
            from .selective_replanning.history import PlanningHistory
            self.selective_history=PlanningHistory(progress_policy="maneuver_started_v2" if selective_rule_variant in PROGRESSIVE_VARIANTS else "legacy")
            self.selective_blockers=[]
        self.scheduler=EnergySlackScheduler() if method=="energy_slack" else RollingHorizonCPSATScheduler() if method=="rh_cp_sat" else None
        if self.scheduler is None: raise ValueError("unknown Stage2 baseline")
        candidate_policy_id(planning_profile)
        self.planning_profile=planning_profile
        self.provider=candidate_provider or CandidateProvider(planning_profile=planning_profile,advisor_runtime=advisor_runtime,
            evaluation_profile=self.evaluation_profile)
        if candidate_provider is not None and advisor_runtime is not None and self.provider.advisor_runtime is not advisor_runtime:
            raise ValueError("candidate provider advisor runtime mismatch")
        if budget_scale<=0 or not math.isfinite(budget_scale): raise ValueError("invalid candidate budget scale")
        self.budget_scale=budget_scale
        if online_policy not in (None,"progressive_v1"): raise ValueError("unknown online catalog policy")
        self.online_policy=online_policy
        self.prefix_reuse=bool(online_policy) if prefix_reuse is None else bool(prefix_reuse)
        if self.provider.planning_profile!=planning_profile: raise ValueError("candidate provider planning profile mismatch")
        if getattr(self.provider,"evaluation_profile",LEGACY_EVALUATION_PROFILE)!=self.evaluation_profile: raise ValueError("candidate provider evaluation profile mismatch")
        PlanningBudget.for_profile(budget_profile)  # fail closed on unknown names
        self.budget_profile=budget_profile
        self.records=[]
        self.last_result=None
        self.last_diagnostic_problem=None

    def probe_last_choices(self,forced_choices,**kwargs):
        """Read-only counterfactual of the latest opt-in, actual local catalog."""
        if self.last_diagnostic_problem is None: raise ValueError("no retained diagnostic catalog")
        from .selective_replanning.diagnostics import probe_catalog_choices
        problem=self.last_diagnostic_problem
        kwargs.setdefault("baseline",problem["selected"])
        kwargs.setdefault("global_frame",problem["global_frame"])
        return probe_catalog_choices(problem["snapshot"],problem["catalog"],forced_choices,**kwargs)

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
        if self.selective_method is not None:
            state["selective_replanning"]=dict(method=self.selective_method,history=self.selective_history.state_dict(),blockers=self.selective_blockers,
                rule_variant=self.selective_rule_variant,opportunity_seed_offset=self.opportunity_seed_offset)
            if self.selective_rule_variant==REFERENCE_VARIANT:
                state["selective_replanning"]["search_policy"]=self.search_policy.to_dict()
        if self.planning_profile!=PLANNING_PROFILE_LEGACY: state["planning_profile"]=self.planning_profile
        if self.evaluation_profile!=LEGACY_EVALUATION_PROFILE: state["evaluation_profile"]=self.evaluation_profile
        if self.online_policy is not None: state["online_policy"]=self.online_policy
        if self.prefix_reuse or self.online_policy: state["prefix_reuse"]=self.prefix_reuse
        if self.provider.advisor_runtime is not None or self.budget_scale!=1.:
            runtime=self.provider.advisor_runtime
            state["stage3"]={"budget_scale":self.budget_scale,"mode":runtime.mode if runtime is not None else "OFF",
                             "deterministic_only":runtime.deterministic_only if runtime is not None else False,
                             "ranking_scope":getattr(runtime,"ranking_scope","legacy"),
                             "fingerprint":runtime.fingerprint if runtime is not None else {}}
        if self.selective_method is not None:
            # Frozen global-frame diagnostics can retain nested Section values
            # in FOLLOW records. Normalize through the existing archive view;
            # retain the original finite JSON boundary and the OFF path.
            from .replay import _plain
            state=_plain(state)
        return json.loads(json.dumps(state,allow_nan=False))

    def load_state_dict(self,state):
        from .trajectory import TrajectoryPlan
        if state.get("evaluation_profile",LEGACY_EVALUATION_PROFILE)!=self.evaluation_profile:
            raise ValueError("scheduler evaluation profile mismatch")
        selective=state.get("selective_replanning")
        if (selective or {}).get("method")!=self.selective_method: raise ValueError("selective method binding mismatch")
        if (selective or {}).get("rule_variant","baseline")!=self.selective_rule_variant: raise ValueError("selective rule variant binding mismatch")
        if (selective or {}).get("opportunity_seed_offset",0)!=self.opportunity_seed_offset: raise ValueError("opportunity seed binding mismatch")
        if self.selective_rule_variant==REFERENCE_VARIANT:
            saved_policy=(selective or {}).get("search_policy")
            if (not isinstance(saved_policy,dict) or any(type(value) is not bool for value in saved_policy.values())
                    or saved_policy!=self.search_policy.to_dict()):
                raise ValueError("selective search policy binding mismatch")
        elif selective is not None and "search_policy" in selective:
            if resolve_search_policy(self.selective_rule_variant,selective["search_policy"])!=self.search_policy:
                raise ValueError("selective search policy binding mismatch")
        if selective is not None:
            self.selective_history.load_state_dict(selective["history"])
            self.selective_blockers=list(selective.get("blockers",()))
        if state.get("online_policy")!=self.online_policy: raise ValueError("online catalog policy mismatch")
        if state.get("prefix_reuse",False)!=self.prefix_reuse: raise ValueError("prefix reuse policy mismatch")
        runtime=self.provider.advisor_runtime
        binding={"budget_scale":self.budget_scale,"mode":runtime.mode if runtime is not None else "OFF",
                 "deterministic_only":runtime.deterministic_only if runtime is not None else False,
                 "ranking_scope":getattr(runtime,"ranking_scope","legacy"),
                 "fingerprint":runtime.fingerprint if runtime is not None else {}}
        saved=dict(state.get("stage3",{"budget_scale":1.,"mode":"OFF","fingerprint":{}}))
        saved.setdefault("deterministic_only",False)
        saved.setdefault("ranking_scope","legacy")
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
        self.last_diagnostic_problem=None
        from .safety import geometry_evidence_counts
        geometry_before=geometry_evidence_counts()
        budget=PlanningBudget.for_profile(self.budget_profile)
        budget.diagnostics=self.diagnostics
        if self.selective_method is not None:
            budget.candidate_seconds=6.; budget.solver_seconds=8.; budget.hint_seconds=3.
            budget.soft_seconds=30.; budget.watchdog_seconds=60.
            budget.max_per_aircraft=32; budget.max_attempts=1024
        wall_deadline=getattr(port,'planning_wall_deadline',None)
        if wall_deadline is not None:
            if not math.isfinite(wall_deadline): raise ValueError('finite planning wall deadline required')
            budget.planning_wall_deadline=float(wall_deadline)
        search_deadline=getattr(port,'search_wall_deadline',None)
        if search_deadline is not None:
            require_search=(math.isfinite(search_deadline) and wall_deadline is not None and search_deadline<=wall_deadline)
            if not require_search: raise ValueError('search deadline must not exceed ready deadline')
            budget.search_wall_deadline=float(search_deadline)
            budget.planning_wall_origin=float(getattr(port,'planning_wall_origin',budget.wall_start))
            budget.search_handoff_reserve_seconds=float(getattr(port,'search_handoff_reserve_seconds',wall_deadline-search_deadline))
        ready_remaining_at_start=(None if wall_deadline is None else max(0.,wall_deadline-time.perf_counter()))
        effective_scale = .5 if self.selective_method is None and self.online_policy and self.budget_scale == 1. else self.budget_scale
        if effective_scale!=1.:
            from .stage3_learning.runtime import scale_candidate_budget
            scale_candidate_budget(budget,effective_scale)
        if self.online_policy: budget.candidate_active_seconds=0.
        solver_history=[]
        catalog_round_records=[]
        cp_continuation_records=[]
        generation_stop_reason=None; improvement_stop_reason=None; same_catalog_improvement=False
        comparable_scope=getattr(self.provider.advisor_runtime,'ranking_scope','legacy') in ('comparable_group_v1','comparable_group_lazy_v2')
        selective_telemetry=None; selective_completed=set()
        for restart in range(2):
            with budget.measure("snapshot_and_uniform_refinement"):
                snapshot=build_planning_snapshot(port)
            context=EvaluationContext(snapshot,budget=budget,validation_cache=self.provider.validation_cache,
                reuse_prefix_results=self.prefix_reuse)
            budget._certificate_context=context.snapshot_key
            budget._catalog=None
            if self.selective_method is not None:
                from .selective_replanning import build_global_frame, RuleSelector, LocalPlanner
                from .selective_replanning.frame import FrameIncomplete
                from .selective_replanning.relations import build_relations
                source=getattr(port,"_source",snapshot)
                if getattr(source,"planning_timing",None) is None: self.selective_history.observe(source)
                now=source.time
                with budget.measure("global_evaluation_frame"):
                    try: frame=self.global_frame or build_global_frame(snapshot,self.provider,context)
                    except FrameIncomplete as exc:
                        access_failure={}
                        if self.selective_rule_variant==REFERENCE_VARIANT:
                            from .selective_replanning.near_term import adjustment_access
                            access_failure={a.id:adjustment_access(snapshot,a) for a in _decision_aircraft(snapshot)}
                        result=PlanningResult((),(),None,"GLOBAL_FRAME_INCOMPLETE",(),certified=False,details={"reason":str(exc)})
                        self.last_result=result
                        geometry_after=geometry_evidence_counts()
                        self.records.append(dict(time=snapshot.time,method=self.scheduler.method,status=result.status,
                            wall_s=budget.elapsed,cpu_s=time.process_time()-budget.cpu_start,planning_budget=budget.settings(),
                            timings=dict(budget.timings),submitted=0,commit_status="NOT_COMMITTED",attempts=dict(budget.attempts),
                            geometry_evidence_counts=dict(before=geometry_before,after=geometry_after,
                                delta={key:geometry_after[key]-geometry_before[key] for key in geometry_after}),
                            selective_replanning=dict(method=self.selective_method,status="LOCAL_SEARCH_INCOMPLETE",reason=str(exc),
                                solve_calls=0,scope_expansions=0,objective_frame_id=None,scope=None,rule_variant=self.selective_rule_variant,
                                **(dict(search_policy=self.search_policy.to_dict(),adjustment_access=access_failure,
                                    round_execution={}) if self.selective_rule_variant==REFERENCE_VARIANT else {}))))
                        return result
                if frame.snapshot_id!=snapshot.version or frame.effective_time!=snapshot.time: raise ValueError("injected frame snapshot mismatch")
                from .selective_replanning.frame import frame_evaluation_profile
                if frame_evaluation_profile(frame.policy)!=self.evaluation_profile: raise ValueError("injected frame evaluation profile mismatch")
                with budget.measure("selective_selection"):
                    relations=build_relations(snapshot,tuple(a.id for a in _decision_aircraft(snapshot)),self.selective_blockers,frame=frame,variant=self.selective_rule_variant)
                    scope,selection=RuleSelector(variant=self.selective_rule_variant,search_policy=self.search_policy).select(snapshot,frame,self.selective_history,relations,
                        method=self.selective_method,now=now,references=self.provider.references,opportunity_seed_offset=self.opportunity_seed_offset)
                local=LocalPlanner(self.provider,self.scheduler,
                    progressive_v2=self.selective_rule_variant in PROGRESSIVE_VARIANTS,
                    executable_v3=self.selective_rule_variant=="opportunity_executable_v3",search_policy=self.search_policy)
                result=local.plan(snapshot,scope,frame,budget)
                catalog,context=local.last_catalog,local.last_context
                if self.diagnostics is not None:
                    self.last_diagnostic_problem=dict(snapshot=snapshot,catalog=catalog,selected=result.selected,global_frame=frame)
                selection["selection_status"]=selection.pop("status")
                selective_telemetry=dict(local.telemetry,**selection,method=self.selective_method,
                    relations=[vars(r) for r in relations],selection_reasons=dict(local.last_scope.selection_reasons),
                    frame_wall_s=budget.timings.get("global_evaluation_frame",(0.,0.))[0],selection_wall_s=budget.timings.get("selective_selection",(0.,0.))[0])
                selective_completed.update(local.completed_evaluations)
                self.selective_blockers=list(local.telemetry["background_blockers"])
                for call in local.telemetry["calls"]: solver_history.extend(call["solver_records"])
                generation_stop_reason=budget.generation_stop_reason() or "SCOPED_CATALOG_COMPLETE"
                improvement_stop_reason=result.details.get("solver_stop_reason") or "LOCAL_SOLVE_LIMIT"
            elif self.online_policy:
                from .stage3_learning.catalog_control import coverage_report, expansion_reasons
                catalogs=self.provider.iter_progressive_columns(snapshot,budget,context)
                best=None
                for catalog_round in range(3):
                    if budget.search_wall_deadline is not None and best is not None and not budget.search_available():
                        generation_stop_reason=improvement_stop_reason=budget.evaluation_stop_reason()
                        break
                    with budget.measure("candidate_generation_and_single_vehicle"),budget.candidate_work():
                        try: catalog=next(catalogs)
                        except StopIteration: break
                    if best is not None:
                        current={c.signature:c for rows in catalog.columns.values() for c in rows}
                        remapped=tuple(current[c.signature] for c in best.selected)
                        best=replace(best,selected=remapped,objective=_selection_key(catalog,remapped))
                        budget.bind_catalog(catalog)
                        budget.certificate_event("catalog_canonical_rescore",best.selected,best.prefix,event="CATALOG_RESCORE",objective=best.objective)
                    budget.bind_catalog(catalog)
                    before=budget.solver_wall_used
                    budget.solver_call_seconds=2. if catalog_round<2 else None
                    try:
                        with budget.measure("selection"):
                            options=({'incumbent':best} if comparable_scope and best is not None and self.scheduler.method=='rh_cp_sat' else {})
                            proposal=self.scheduler.plan(snapshot,catalog,context,budget,**options)
                    finally: budget.solver_call_seconds=None
                    reserve=getattr(self.provider,"reserve_suffix_completion",None)
                    if reserve is not None: reserve(snapshot,catalog,budget)
                    solver_history.extend(dict(r,uniform_refinement_restart=restart,catalog_round=catalog_round) for r in proposal.solver_records)
                    if proposal.certified:
                        budget.certificate_event("catalog_selection",proposal.selected,proposal.prefix,
                            event="CERTIFIED_OBJECTIVE",objective=proposal.objective)
                    if 'continuation' in proposal.details: cp_continuation_records.append(dict(proposal.details['continuation']))
                    if proposal.certified and (best is None or proposal.objective<best.objective or
                            (proposal.objective==best.objective and proposal.prefix and not best.prefix)):
                        budget.certificate_event("catalog_selection",proposal.selected,proposal.prefix,event="CATALOG_BEST_UPDATE",
                            comparison=best.objective if best is not None else None,objective=proposal.objective)
                        best=proposal
                    result=best or proposal
                    refining=getattr(proposal.evaluation,"refinement_required",False)
                    if refining and (not comparable_scope or best is None): result=proposal
                    coverage=coverage_report(snapshot,catalog.columns,
                        effective_time=getattr(getattr(snapshot,'planning_timing',None),'effective_time',None))
                    reasons=expansion_reasons(snapshot,catalog,proposal,coverage,context)
                    catalog_round_records.append(dict(round=catalog_round,columns=sum(map(len,catalog.columns.values())),
                        moving_columns={aid:sum(c.kind=="MOVE" for c in values) for aid,values in catalog.columns.items()},
                        coverage=coverage,expansion_reasons=reasons,pruned_columns=budget.screening["catalog_similar_pruned"],
                        cp_wall_s=budget.solver_wall_used-before,cumulative_cp_wall_s=budget.solver_wall_used,
                        same_catalog_improvement=False))
                    if refining:
                        generation_stop_reason="UNIFORM_REFINEMENT_REQUIRED"
                        improvement_stop_reason="UNIFORM_REFINEMENT_REQUIRED"; break
                    if not reasons:
                        generation_stop_reason="NO_EXPANSION_REASON"
                        improvement_stop_reason="NO_EXPANSION_REASON"; break
                    if not budget.available():
                        generation_stop_reason=budget.generation_stop_reason() or "CANDIDATE_UNAVAILABLE"
                        # One final call may improve the already materialized
                        # catalog after an early 2s slice. No generator resumes,
                        # clock resets, extra solver allocation or new rounds.
                        can_improve=(comparable_scope and catalog_round<2 and
                            self.scheduler.method=="rh_cp_sat" and
                            any(c.kind=="MOVE" for rows in catalog.columns.values() for c in rows) and
                            budget.solver_wall_used<budget.solver_seconds and
                            budget.refinement_rounds_used<budget.max_refinement_rounds and budget.search_available())
                        if can_improve:
                            before=budget.solver_wall_used
                            with budget.measure("selection"):
                                final=self.scheduler.plan(snapshot,catalog,context,budget,incumbent=best)
                            if 'continuation' in final.details: cp_continuation_records.append(dict(final.details['continuation']))
                            same_catalog_improvement=True
                            solver_history.extend(dict(r,uniform_refinement_restart=restart,catalog_round=catalog_round,
                                same_catalog_improvement=True) for r in final.solver_records)
                            if final.certified and (best is None or final.objective<best.objective or
                                    (final.objective==best.objective and final.prefix and not best.prefix)):
                                budget.certificate_event("existing_catalog_improvement",final.selected,final.prefix,event="CATALOG_BEST_UPDATE",
                                    comparison=best.objective if best is not None else None,objective=final.objective)
                                best=final
                            result=best or final
                            catalog_round_records.append(dict(round=catalog_round,columns=sum(map(len,catalog.columns.values())),
                                moving_columns={aid:sum(c.kind=="MOVE" for c in values) for aid,values in catalog.columns.items()},
                                coverage=coverage,expansion_reasons=reasons,pruned_columns=budget.screening["catalog_similar_pruned"],
                                cp_wall_s=budget.solver_wall_used-before,cumulative_cp_wall_s=budget.solver_wall_used,
                                same_catalog_improvement=True))
                            improvement_stop_reason=final.details.get("solver_stop_reason") or "FINAL_EXISTING_CATALOG_CALL_COMPLETE"
                        else:
                            improvement_stop_reason=("SOLVER_ALLOCATION_EXHAUSTED" if budget.solver_wall_used>=budget.solver_seconds else
                                "REFINEMENT_LIMIT" if budget.refinement_rounds_used>=budget.max_refinement_rounds else
                                "SOFT_BUDGET" if budget.elapsed>=budget.soft_seconds else "NO_FINAL_EXISTING_CATALOG_CALL")
                        break
                catalogs.close()
                generation_stop_reason=generation_stop_reason or "CATALOG_ROUNDS_OR_GENERATORS_COMPLETE"
                improvement_stop_reason=improvement_stop_reason or result.details.get("solver_stop_reason") or "CATALOG_ROUNDS_COMPLETE"
            else:
                with budget.measure("candidate_generation_and_single_vehicle"):
                    catalog=self.provider.get_columns(snapshot,budget,context)
                with budget.measure("selection"):
                    result=self.scheduler.plan(snapshot,catalog,context,budget)
                if 'continuation' in result.details: cp_continuation_records.append(dict(result.details['continuation']))
                solver_history.extend(dict(r,uniform_refinement_restart=restart) for r in result.solver_records)
            if not getattr(result.evaluation,"refinement_required",False): break
            if restart or self.selective_method is not None or not budget.search_available():
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
        if selective_telemetry is not None and self.selective_rule_variant==REFERENCE_VARIANT:
            executable_ids={p.aircraft_id for p in context.prefix(result.selected)}
            committed_ids={p.aircraft_id for p in committed_prefix}
            selective_telemetry["round_execution"]={c.aircraft_id:dict(
                evidence_phase="CALLBACK_RETURN",
                kind=c.kind,first_start_time=c.plans[0].start_time if c.plans else None,
                executable_prefix=c.aircraft_id in executable_ids,
                prefix_committed=c.aircraft_id in committed_ids,
                certified=result.certified) for c in result.selected}
        updated_previous={c.aircraft_id:tuple(p for p in c.plans if p not in committed_prefix) for c in result.selected if c.kind=="MOVE"}
        if self.selective_method is None: self.provider.previous=updated_previous
        else:
            for aid in context.decisions: self.provider.previous.pop(aid,None)
            self.provider.previous.update(updated_previous)
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
                immediate_move_candidates=sum(c.kind=="MOVE" and bool(c.plans) and _executable_start(snapshot,c.plans[0].start_time)
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
                "evaluation_profile":self.evaluation_profile,"objective_id":objective_schema_id(self.planning_profile,self.evaluation_profile),
                "candidate_policy_id":candidate_policy_id(self.planning_profile,self.evaluation_profile),"budget_profile":self.budget_profile,"planning_budget":budget.settings(),
                "deadline_reference_seconds":5.,"hint_wall_s":budget.hint_wall_used,"solver_wall_s":budget.solver_wall_used,
                "hint_objective":result.details.get("hint_objective"),"final_objective":result.objective,
                "hint_certified":result.details.get("hint_certified"),
                "strict_improvement_over_hint":result.details.get("strict_improvement_over_hint"),
                "solver_entered":solver_entered,"skip_reason":None if solver_entered else result.details.get("skip_reason",
                    "METHOD_ENERGY_SLACK" if self.scheduler.method=="energy_slack" else "NO_SOLVER_CALL"),
                "solver_stop_reason":result.details.get("solver_stop_reason"),
                "stop_reason":result.details.get("stop_reason"),
                "generation_stop_reason":generation_stop_reason or budget.generation_stop_reason(),
                "improvement_stop_reason":improvement_stop_reason or result.details.get("solver_stop_reason"),
                "same_catalog_improvement":same_catalog_improvement,
                "cp_continuation":cp_continuation_records,
                "ready_budget":{"planning_wall_deadline":budget.planning_wall_deadline,
                    "remaining_at_start_s":ready_remaining_at_start,
                    "remaining_at_return_s":None if wall_deadline is None else max(0.,wall_deadline-time.perf_counter())},
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
                "online_policy":self.online_policy,"prefix_reuse_enabled":self.prefix_reuse,"catalog_rounds":catalog_round_records,
                "candidate_active_wall_s":budget.candidate_elapsed,
                "candidate_noncandidate_excluded_wall_s":max(0.,budget.elapsed-budget.candidate_elapsed) if self.online_policy else 0.,
                "candidate_timing_regime":"cumulative_active_generation" if self.online_policy else "original_callback_elapsed",
                "prefix_reuse":{k:v for k,v in context.cache_stats.items() if k.startswith("prefix_")},
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
        completion=getattr(self.provider,"suffix_completion_report",lambda budget:None)(budget)
        if completion is not None: record["suffix_completion"]=completion
        elif self.evaluation_profile==STITCHED_EVALUATION_PROFILE and self.online_policy is None:
            record["suffix_completion"]=dict(policy_id="bounded_suffix_completion_v1",
                status="NOT_ENABLED_NON_PROGRESSIVE",enabled=False,attempts=[])
            record["candidate_policy_id"]=candidate_policy_id(self.planning_profile)
        if selective_telemetry is not None:
            record["selective_replanning"]=selective_telemetry
            source=getattr(port,"_source",snapshot)
            timing=getattr(snapshot,"planning_timing",None)
            review_key=self._selective_review_key(timing) if timing is not None else "immediate:"+snapshot.version
            unknown_reviews=()
            if self.selective_rule_variant in PROGRESSIVE_VARIANTS:
                # A completed safety check with UNKNOWN normal continuation is
                # not a delivered normal-path review certificate.
                unknown_reviews={aid for aid in selective_completed if not any(
                    getattr(c.reference,"status",None)=="KNOWN" for c in catalog.columns.get(aid,()))}
                record["selective_replanning"]["review_unknown_ids"]=sorted(unknown_reviews)
            self.selective_history.queue_reviews(review_key,source,selective_completed,unknown_ids=unknown_reviews)
            if timing is None: self.selective_history.deliver(review_key,build_planning_snapshot(port),port.time)
            record["selective_replanning"]["review_delivery_key"]=review_key
        explanation_stopped=search_deadline is not None and not budget.search_deadline_ok()
        if catalog.downstream is not None and not explanation_stopped:
            with budget.measure("downstream_explanation"):
                selected_key,downstream_rows=catalog.downstream.evaluate(result.selected)
                for selected in result.selected:
                    if search_deadline is not None and not budget.search_deadline_ok():
                        explanation_stopped=True; break
                    timing=selected.downstream.get("candidate_timing")
                    alternatives=[c for c in catalog.columns[selected.aircraft_id] if c.plans and
                        (not selected.plans or c.downstream.get("candidate_timing")!=timing)]
                    comparison=None
                    if alternatives:
                        scored=[]
                        for c in alternatives:
                            if search_deadline is not None and not budget.search_deadline_ok():
                                explanation_stopped=True; break
                            trial=tuple(c if item.aircraft_id==c.aircraft_id else item for item in result.selected)
                            scored.append((catalog.downstream.evaluate(trial)[0],c.signature,c))
                        if explanation_stopped: break
                        comparison_key,_,comparison=min(scored,key=lambda row:row[:2])
                        layer=next((name for name,a,b in zip(("F","L","risk","Ehat","tau"),selected_key,comparison_key) if a!=b),"TIE")
                        reason=("JOINT_VALIDATION_OR_GLOBAL_ALLOCATION" if comparison_key<selected_key else
                                "TIE_SIGNATURE_OR_ALLOCATION" if comparison_key==selected_key else layer)
                    else:
                        comparison_key=None; layer=None; reason="NO_OPPOSITE_TIMING_CANDIDATE"
                    submitted=any(p.aircraft_id==selected.aircraft_id for p in committed_prefix)
                    action=("IMMEDIATE_PREFIX_SUBMITTED" if submitted else "HOLD_CURRENT_SLOT" if not selected.plans else
                            "FUTURE_PREDICTION_NO_RESERVATION" if not _executable_start(snapshot,selected.plans[0].start_time) else "PREFIX_NOT_COMMITTED")
                    downstream_rows[selected.aircraft_id]["decision_explanation"]=dict(reason=reason,first_different_objective_layer=layer,
                        comparison_signature=None if comparison is None else comparison.signature,comparison_integer_key=comparison_key,
                        comparison_status="NOT_JOINT_VALIDATED" if comparison is not None else "NO_COMPARISON",actual_action=action,
                        observed_rejections=dict(context.rejections_by_aircraft.get(selected.aircraft_id,{})))
                record.update(planning_profile=self.planning_profile,evaluation_profile=self.evaluation_profile,
                    objective_id=objective_schema_id(self.planning_profile,self.evaluation_profile),
                    downstream=downstream_rows,downstream_integer_key=selected_key,
                    virtual_pad_assignment=dict(catalog.downstream.assignment),virtual_queue_order=catalog.downstream.order)
            record.update(wall_s=budget.elapsed,cpu_s=time.process_time()-budget.cpu_start,
                timings=dict(budget.timings),deadline_miss=budget.elapsed>5.)
        if search_deadline is not None:
            record['explanation_skipped_search_handoff']=explanation_stopped
            from .planning_latency import SEARCH_HANDOFF_VERSION
            trace=dict(protocol=SEARCH_HANDOFF_VERSION,reserve_seconds=budget.search_handoff_reserve_seconds,
                planning_wall_origin=budget.planning_wall_origin,planning_wall_deadline=wall_deadline,
                search_wall_deadline=search_deadline,first_nonempty_certified=budget.first_nonempty_certified,
                events=budget.certificate_events,heavy_checks=budget.heavy_checks,
                callback_return=dict(wall_elapsed_s=time.perf_counter()-budget.planning_wall_origin,
                    certified=result.certified,objective=list(result.objective),prefix_hashes=[p.trajectory_hash for p in result.prefix]))
            record['search_handoff']=trace
            port.search_handoff=trace
        geometry_after=geometry_evidence_counts()
        record["geometry_evidence_counts"]=dict(before=geometry_before,after=geometry_after,
            delta={key:geometry_after[key]-geometry_before[key] for key in geometry_after})
        if selective_telemetry is not None:
            record["joint_conflict_observations"]=selective_telemetry.get("joint_conflict_observations",[])
        self.records.append(record)
        if runtime is not None and getattr(runtime,'ranking_scope','legacy') in ('comparable_group_v1','comparable_group_lazy_v2'):
            record['rank_batch_cache']=dict(hits=runtime.coverage['rank_cache_hits'],
                miss_reasons={k.split(':',1)[1]:v for k,v in runtime.coverage.items() if k.startswith('rank_cache_miss:')},
                max_budget_fraction_drift=runtime.coverage.get('rank_budget_fraction_max_drift'))
        if getattr(snapshot,"planning_timing",None) is not None:
            from dataclasses import asdict
            record.update(planning_timing=asdict(snapshot.planning_timing),
                effective_state_version=snapshot.effective_state_version,
                deferred=commit is not None and commit.status=="DEFERRED",
                actual_submitted_trajectory_hashes=[])
        if wall_deadline is not None:
            record['ready_budget']['remaining_at_return_s']=max(0.,wall_deadline-time.perf_counter())
        if search_deadline is not None:
            trace['callback_return']['wall_elapsed_s']=time.perf_counter()-budget.planning_wall_origin
        return result

    @staticmethod
    def _selective_review_key(timing):
        return "effective:"+repr((timing.snapshot_time,timing.effective_time,timing.ready_deadline))

    def observe_execution_snapshot(self, snapshot, now=None, *, events=()):
        if self.selective_method is not None: self.selective_history.observe(snapshot,now,events=events)

    def initialize_imported_history(self, snapshot, events=()):
        if self.selective_method is None: raise ValueError("selective history requires Stage A method")
        self.selective_history.initialize_imported(snapshot,events)

    def record_evaluation_delivery(self, timing, snapshot, now):
        if self.selective_method is None: return
        updated=self.selective_history.deliver(self._selective_review_key(timing),snapshot,now)
        for record in reversed(self.records):
            if record.get("selective_replanning",{}).get("review_delivery_key")==self._selective_review_key(timing):
                record["selective_replanning"].update(review_delivered_time=now,review_updated_ids=list(updated)); break

    def record_latency_ack(self, timing, delivery, plans):
        """Attach actual executor acknowledgement after a deferred proposal."""
        from dataclasses import asdict
        key = asdict(timing)
        record = next((r for r in reversed(self.records) if r.get("planning_timing")==key),None)
        if record is None: return
        record.update(commit_status=delivery["commit_status"],submitted=delivery["submitted"],
            actual_delivery_status=delivery["status"],actual_delivery_reason=delivery["reason"],
            actual_delivery_time=delivery["finished_time"],
            actual_submitted_trajectory_hashes=[p.trajectory_hash for p in plans] if delivery["submitted"] else [])
        if delivery["submitted"]:
            committed = {p.trajectory_hash for p in plans}
            self.provider.previous = {aid:tuple(p for p in rows if p.trajectory_hash not in committed)
                                      for aid,rows in self.provider.previous.items()}
        runtime = self.provider.advisor_runtime
        if runtime is not None:
            runtime.emit(dict(record_type="PlanningDelivery",planning_timing=key,
                status=delivery["status"],reason=delivery["reason"],
                commit_status=delivery["commit_status"],
                submitted_trajectory_hashes=record["actual_submitted_trajectory_hashes"]))
