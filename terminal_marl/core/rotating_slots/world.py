"""Deterministic event executor for FT-RS-S1-1.0.

The scheduler supplies immutable trajectories.  Resource calendars and the entire
traffic batch are validated before any command changes the world.
"""
from dataclasses import asdict, dataclass, field, replace
from collections import OrderedDict
from itertools import combinations
from types import MappingProxyType
from hashlib import sha256
import math
import json
from pathlib import Path
import time as walltime

from .config import freeze, load_config, thaw, service_duration
from .demand import ArrivalManifest, ArrivalRequest
from .models import ValidationResult
from .resources import PadCalendar, ResourceConflict, SlotCalendar
from .trajectory import (generate_final, generate_inbound, generate_transfer,
                         make_ground_plan, make_stable_plan, slot_state, TrajectoryPlan)
from .energy import evaluate_power, integrate_energy, stable_power
from .safety import validate_plan, validate_separation, _ExecutionQueryCache, _execution_distance_screen
from .rescue_rules import (LEGACY_PROFILE, RESCUE_PROFILE_V6, RESCUE_PROFILE_V7, assess_rescue,
    normal_energy_lower_bound, pad_opportunities_overlap, MONITOR_SECONDS_V6, RESCUE_LOOKAHEAD_SECONDS_V6,
    RESCUE_DURATION_ALTERNATIVES_V6)


AIRBORNE = {"INBOUND", "HOLD_OUTER", "HOLD_MIDDLE", "HOLD_INNER", "TRANSFER_O_M",
            "TRANSFER_M_I", "FINAL_NORMAL", "FINAL_EMERGENCY", "VERTICAL_DESCENT"}
HOLD = {"outer": "HOLD_OUTER", "middle": "HOLD_MIDDLE", "inner": "HOLD_INNER"}
# Legacy import compatibility only; execution reads the frozen configuration.
SERVICE_SECONDS = {"short": 60., "standard": 120., "long": 240.}
POLICY_ID = "upstream_capacity_only_pad_arrival_v1_20261018"
STAGE2_POLICY_ID = "rs_s2_loss_continue_throughput_v1_0"


def emergency_threshold(soc, q_e, qhold):
    """The shared public emergency predicate, without declaring an emergency."""
    return soc <= .35 or (q_e is not None and soc-q_e-qhold <= .25)


def emergency_priority(aircraft, config):
    """Pure urgency estimate from the revealed rescue cost and holding power.

    Missing rescue cost deliberately falls back to time until SOC depletion;
    neither estimate certifies that a rescue trajectory is currently feasible.
    """
    power = stable_power(aircraft.ring, config)
    rate = power.current_a / (config.energy.reference_capacity_Ah * 3600.)
    q = aircraft.emergency.q_e
    known = q is not None and math.isfinite(q) and q >= 0.
    seconds = (aircraft.soc - (q if known else 0.)) / rate
    declared = aircraft.emergency.declaration_time
    key = (seconds, aircraft.soc, declared is None, 0. if declared is None else declared, aircraft.id)
    return dict(key=key, holding_seconds=seconds, holding_soc_per_second=rate,
                q_e=q if known else None, q_e_unknown=not known,
                fallback=None if known else "soc_over_holding_rate", soc=aircraft.soc,
                declaration_time=declared)


def emergency_waiting_queue(aircraft, active, config, *, profile=LEGACY_PROFILE):
    """Pure queue, including revealed thresholds before monitor flag refresh."""
    candidates = []
    for a in aircraft.values():
        if a.status not in HOLD.values() or a.id in active:
            continue
        priority = emergency_priority(a, config)
        if (a.emergency.eligible or a.emergency.declared
                or (profile in (RESCUE_PROFILE_V6, RESCUE_PROFILE_V7) and a.emergency.pending)
                or (profile not in (RESCUE_PROFILE_V6, RESCUE_PROFILE_V7) and emergency_threshold(a.soc, priority["q_e"],
                                       10.*priority["holding_soc_per_second"]))):
            candidates.append((priority["key"], a))
    return [a for _, a in sorted(candidates, key=lambda item: item[0])]


def emergency_priority_blocker(aircraft, active, config, candidate_id, *,
                               profile=LEGACY_PROFILE, proposed_plan=None, protected_plans=()):
    """Pure normal-final priority guard for world or immutable planning inputs."""
    if profile in (RESCUE_PROFILE_V6, RESCUE_PROFILE_V7):
        if proposed_plan is None or proposed_plan.kind != "FINAL_NORMAL":
            return None
        for p in sorted(protected_plans, key=lambda p: (p.start_time, p.aircraft_id)):
            if (p.aircraft_id != candidate_id and p.kind == "FINAL_EMERGENCY"
                    and pad_opportunities_overlap(proposed_plan, p, aircraft, config)):
                return dict(blocked_by=p.aircraft_id, protected_opportunity=p.trajectory_hash)
        return None
    waiting = emergency_waiting_queue(aircraft, active, config)
    if not waiting:
        return None
    a, first = aircraft[candidate_id], waiting[0]
    if (a.id == first.id and a.ring == "inner"
            and not a.emergency.declared and not a.emergency.pending):
        return None
    return dict(blocked_by=first.id, priority=emergency_priority(a, config)
                if a.status in HOLD.values() else None,
                blocking_priority=emergency_priority(first, config))


@dataclass(frozen=True)
class PlanningSnapshot:
    """Revealed immutable planning input; deliberately contains no manifest/world."""
    time: float
    version: str
    config: object
    aircraft: object
    active: object
    soft: object
    requests: object
    slots: tuple
    pads: tuple
    dependencies: object
    source_clear: object
    trajectories: object
    energy_ledger: object
    emergency_monitor: object = None


@dataclass(frozen=True)
class SubmissionEvaluation:
    accepted: bool
    status: str
    reason: str
    details: object
    time: float
    snapshot_version: str
    plans: tuple
    cancel: tuple
    refinement_required: bool = False
    _prepared: object = field(default=None, repr=False, compare=False)


def evaluate_planning_snapshot(snapshot, plans, *, cancel=()):
    """Certify prefixes in a detached visible-state projection, without execution.

    This copies only the finite scheduling input and numerical ledger. It never
    clones a world, possesses demand history, or runs an execution branch.
    """
    context = object.__new__(RotatingSlotWorld)
    context.config, context.time = snapshot.config, snapshot.time
    context.experiment_policy_id = (snapshot.emergency_monitor or {}).get("experiment_policy_id")
    context.rescue_rule_profile = (snapshot.emergency_monitor or {}).get("rescue_rule_profile", LEGACY_PROFILE)
    context.aircraft = dict(snapshot.aircraft)
    context.active, context.soft = dict(snapshot.active), dict(snapshot.soft)
    context.dependencies, context._source_clear = dict(snapshot.dependencies), dict(snapshot.source_clear)
    context.slots, context.pads = SlotCalendar(), PadCalendar()
    context.slots._entries, context.pads._entries = list(snapshot.slots), list(snapshot.pads)
    context.trajectories = {h: thaw(v) for h,v in snapshot.trajectories.items()}
    context._targeted_energy = thaw(snapshot.energy_ledger)
    context.failed, context.events, context.rejections = False, [], []
    context.cache_enabled, context.cache_hits = True, 0
    context._energy_cache, context._q_cache = OrderedDict(), OrderedDict()
    context._reference_templates = {}
    context._last_reference_fallback = None
    context.reference_counts = dict(initial_search=0, fallback_search=0, template_instantiation=0)
    context.cpu = {k: 0. for k in ("energy", "validation", "reference_template_search", "reference_fallback_search")}
    plans, cancel = tuple(plans), tuple(cancel)
    prepared = context._prepare_submission(plans, cancel=cancel)
    if isinstance(prepared, ValidationResult):
        result, prepared = prepared, None
    else:
        result = context._result(True)
    refined = (context.aircraft != dict(snapshot.aircraft)
               or freeze(context._targeted_energy["executed"]) != snapshot.energy_ledger["executed"])
    return SubmissionEvaluation(result.accepted, result.status, result.reason, freeze(result.details),
                                snapshot.time, snapshot.version, plans, cancel, refined, prepared)


class EmergencyDependencyConflict(RuntimeError):
    """Emergency disposition cannot preserve an existing safety commitment.

    The executor leaves the old reservation intact and stops at the decision
    timestamp.  This is an explicit dispatch conflict, not a physical failure.
    """

    def __init__(self, aircraft_id, reason, details):
        self.aircraft_id, self.reason, self.details = aircraft_id, reason, details
        super().__init__(f"EMERGENCY_DEPENDENCY_CONFLICT: {aircraft_id}: {reason}: {details}")


@dataclass(frozen=True)
class EmergencyStatus:
    eligible: bool = False
    declared: bool = False
    pending: bool = False
    declaration_time: float | None = None
    declaration_ring: str | None = None
    start_time: float | None = None
    start_ring: str | None = None
    charged: bool = False
    state: str = "NONE"
    q_e: float | None = None
    reference_missing: bool = False
    deadline_missed: bool = False
    warning: bool = False
    normal_status: str = "UNKNOWN"
    normal_margin: float | None = None
    rescue_wait_seconds: float | None = None
    rescue_reason: str | None = None
    rescue_slack_seconds: float | None = None
    normal_path_soc: float | None = None
    normal_path_wait_seconds: float = 0.
    normal_estimate_epoch: float | None = None
    normal_estimate_source: tuple | None = None


@dataclass(frozen=True)
class AircraftState:
    id: str
    status: str
    soc: float
    p: tuple
    v: tuple
    a: tuple
    jerk: tuple
    service_class: str = "standard"
    ring: str | None = None
    slot: int | None = None
    pad_id: int | None = None
    version: int = 0
    emergency: EmergencyStatus = field(default_factory=EmergencyStatus)
    trajectory_hash: str | None = None
    energy_j: float = 0.
    peak_seconds: float = 0.
    release_time: float = 0.
    touchdown_time: float | None = None
    service_end: float | None = None
    admitted: bool = True
    fixture: bool = False
    airborne_time: float = 0.
    holding_time: float = 0.
    transfer_time: float = 0.
    final_time: float = 0.
    descent_time: float = 0.
    battery_power_w: float = 0.
    normal_final_touchdown_soc: float | None = None
    admission_time: float | None = None
    upstream_wait_seconds: float = 0.

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class WorldSnapshot:
    time: float
    aircraft: tuple
    requests: tuple
    slots: tuple
    pads: tuple
    metrics: object
    failed: bool = False
    upstream: tuple = ()

    def to_dict(self):
        return {"time": self.time, "aircraft": [a.to_dict() for a in self.aircraft],
                "requests": [r.to_dict() for r in self.requests],
                "slots": [r.to_dict() for r in self.slots], "pads": [r.to_dict() for r in self.pads],
                "metrics": thaw(self.metrics), "failed": self.failed,
                "policy_id": self.metrics.get("experiment_policy_id", POLICY_ID), "upstream": thaw(self.upstream)}


class _Traffic:
    """A continuous hold/current/future chain; bounds include every junction."""
    def __init__(self, aircraft_id, base, plans, start, end):
        self.aircraft_id = aircraft_id
        self.base = base
        self.plans = tuple(sorted(plans, key=lambda p: p.start_time))
        self.start_time, self.end_time = start, end

    def _at(self, t):
        result = self.base
        for p in self.plans:
            if p.start_time <= t:
                result = p
            else:
                break
        return result

    def state(self, t):
        return self._at(t).state(t)

    def bounds(self, start, end, derivative=0):
        cuts = sorted({start, end, *(p.start_time for p in self.plans if start < p.start_time < end)})
        boxes = [self._at((x+y)/2).bounds(x, y, derivative=derivative) for x, y in zip(cuts, cuts[1:])]
        if not boxes:
            return self._at(start).bounds(start, end, derivative=derivative)
        return tuple((min(b[j][0] for b in boxes), max(b[j][1] for b in boxes)) for j in range(3))


class SchedulerPort:
    """Public scheduling commands and revealed state only.

    This is an API boundary, not a Python security sandbox. The reference is
    deliberately name-mangled and there is no raw-world attribute forwarding.
    """
    __slots__=("__world",)

    def __init__(self,world):
        self.__world=world

    @property
    def time(self):
        return self.__world.time

    @property
    def config(self):
        return self.__world.config

    @property
    def aircraft(self):
        return MappingProxyType(dict(self.__world.aircraft))

    @property
    def active(self):
        return MappingProxyType(dict(self.__world.active))

    @property
    def soft(self):
        return MappingProxyType(dict(self.__world.soft))

    @property
    def requests(self):
        return MappingProxyType(dict(self.__world.requests))

    def snapshot(self):
        return self.__world.snapshot()

    def build_planning_snapshot(self):
        return self.__world.build_planning_snapshot()

    def evaluate_submission(self, plans, *, cancel=(), snapshot=None):
        return self.__world.evaluate_submission(plans, cancel=cancel, snapshot=snapshot)

    def commit_validated_prefix(self, result, snapshot_version=None):
        return self.__world.commit_validated_prefix(result, snapshot_version)

    def report_normal_evidence(self, rows, snapshot_version):
        return self.__world.report_normal_evidence(rows, snapshot_version)

    def refine_planning_boundary(self):
        return self.__world.refine_planning_boundary()

    def metrics(self):
        return freeze(self.__world._scheduler_metrics())

    def submit(self,plans,*,cancel=(),simulated_latency=0.):
        return self.__world.submit(plans,cancel=cancel,simulated_latency=simulated_latency)

    def cancel(self,aircraft_id):
        return self.__world.cancel(aircraft_id)

    def admit(self,request_id,plan=None):
        return self.__world.admit(request_id,plan)

    def reroute(self,request_id,reason="NO_PLAN_FOUND"):
        return self.__world.reroute(request_id,reason)

    def try_final(self,aircraft,emergency=False):
        return self.__world.try_final(aircraft,emergency=emergency)

    def reject(self,reason,aircraft_id=None,**details):
        return self.__world.reject(reason,aircraft_id,**details)


class RotatingSlotWorld:
    Q_CACHE_CAPACITY = 512
    ENERGY_CACHE_CAPACITY = 4096
    REFERENCE_PHASE_BINS = 24

    def __init__(self, config=None, manifest=None, physics_dt=.2, scheduler_period=5., cache_enabled=True,
                 state_sink=None, practical_runtime=True, reference_cache_path=None, experiment_policy_id=None):
        self.config = config or load_config()
        if experiment_policy_id not in (None, STAGE2_POLICY_ID):
            raise ValueError("unsupported experiment_policy_id")
        if experiment_policy_id is not None:
            self.experiment_policy_id = experiment_policy_id
            self.rescue_rule_profile = LEGACY_PROFILE
            self.has_energy_losses = False
            self.terminated_invalid = False
            self.failure_archive = {}
            self.failure_resources_released = 0
            self.dispatch_timings = []
            self.admission_callback = None
            self._pending_dispatch = False
            self._refinement_time = None
            self._monitor_timing_pending = (0., 0.)
        if not 0 < physics_dt <= .2 or scheduler_period <= 0:
            raise ValueError("physics_dt must be in (0,.2], scheduler_period positive")
        self.manifest = manifest or ArrivalManifest()
        self.physics_dt, self.scheduler_period = float(physics_dt), float(scheduler_period)
        self.cache_enabled = cache_enabled
        if state_sink is not None and not callable(state_sink):
            raise TypeError("state_sink must be callable")
        self.state_sink = state_sink
        self._record_count = 0
        self._last_record_time = None
        self.time = 0.
        self.aircraft = {}
        self._targeted_energy = {"fine_plans": {}, "executed": {}}
        self.requests = {}
        self.upstream_reasons = {}
        self.slots, self.pads = SlotCalendar(), PadCalendar()
        self.active, self.soft = {}, {}
        self.dependencies = {}
        self._source_clear = {}
        self._manifest_index = 0
        self._next_heartbeat = 0.
        self._initial_settled = False
        self._q_cache = OrderedDict()
        # This finite canonical template table defines the candidate policy,
        # rather than a history-dependent optimization. It is memoized in both
        # cache modes; cache_enabled controls only exact query/energy results.
        self._reference_templates = {}
        self._energy_cache = OrderedDict()
        self._last_reference_fallback = None
        self.failed = False
        self.failure_time = None
        self.events, self.records, self.rejections = [], [], []
        self.trajectories = {}
        self.decision_times = []
        self.decision_callback = None
        self.auto_emergency = True
        self.cpu = {"energy": 0., "validation": 0., "emergency": 0., "decision": 0., "serialization": 0.}
        self.cpu.update(reference_template_search=0.,reference_fallback_search=0.)
        self.cache_hits = 0
        self.reference_counts = {"initial_search": 0, "fallback_search": 0, "template_instantiation": 0}
        self.minimum_separation = None
        self.costs = {"reroute": 0., "emergency": 0., "failure": 0.}
        self.pad_busy_seconds = {j:0. for j in range(6)}
        self._scheduler_port=SchedulerPort(self)
        self.configure_practical_runtime(practical_runtime, reference_cache_path=reference_cache_path)
        self._record(force=True)

    def configure_practical_runtime(self, enabled=True, *, reference_cache_path=None,
                                    soc_threshold=.35, telemetry_period=1., emergency_period=1.):
        """Configure optional derived work; physical and resource rules stay fixed."""
        if type(enabled) is not bool or not .35 <= soc_threshold <= 1:
            raise ValueError("invalid practical runtime configuration")
        if not self.physics_dt <= telemetry_period <= 1 or not 0 < emergency_period <= 1:
            raise ValueError("runtime periods must be in (0,1]")
        old = getattr(self, "_practical_runtime", {})
        self._practical_runtime = dict(enabled=enabled, soc_threshold=float(soc_threshold),
            telemetry_period=float(telemetry_period), emergency_period=float(emergency_period),
            reference_cache_path=None if reference_cache_path is None else str(reference_cache_path),
            base_plans=old.get("base_plans", {}), monitor=old.get("monitor", {}),
            imminent=old.get("imminent", {}),
            table=old.get("table", {}), counters=old.get("counters", {}),
            precompute_seconds=old.get("precompute_seconds", 0.))
        for name in ("monitor_calls", "monitor_throttled", "soc_gate_skips", "projected_bound_skips",
                     "exact_reference_calls", "table_hits", "table_misses", "table_uncertain",
                     "table_samples", "table_loaded", "table_identity_rejected", "base_plan_hits", "imminent_hint_checks"):
            self._practical_runtime["counters"].setdefault(name, 0)
        if reference_cache_path is not None:
            self._load_reference_table()
        return self

    def _runtime_enabled(self):
        return getattr(self, "_practical_runtime", {}).get("enabled", False)

    def _table_identity(self):
        return dict(schema="emergency_piecewise_v1", config_hash=self.config.config_hash,
                    profile_hash=self.config.profile_hash, bins=240, soc_error_budget=2e-5,
                    sampling="endpoints_midpoint_same_duration", reference_policy="canonical_exact_v1")

    def _load_reference_table(self):
        rt = self._practical_runtime
        path = Path(rt["reference_cache_path"])
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("identity") != self._table_identity():
                raise ValueError("reference identity mismatch")
            table = data["table"]
            if not isinstance(table, dict):
                raise ValueError("invalid table")
            for key, samples in table.items():
                ring, cell = key.split(":")
                if ring not in HOLD or not 0 <= int(cell) < 240 or len(samples) != 3:
                    raise ValueError("invalid table cell")
                for q, duration in samples:
                    if q is not None and (not math.isfinite(q) or q < 0 or duration is None or not math.isfinite(duration)):
                        raise ValueError("invalid sample")
            rt["table"].update(table)
            rt["counters"]["table_loaded"] += len(table)
        except (OSError, ValueError, TypeError, KeyError):
            rt["counters"]["table_identity_rejected"] += 1

    def flush_reference_table(self):
        """Persist only lazily sampled cells, with physical/profile identity."""
        rt = getattr(self, "_practical_runtime", {})
        if rt.get("reference_cache_path") is None:
            return
        path = Path(rt["reference_cache_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(dict(identity=self._table_identity(), table=rt["table"]),
                                        separators=(",", ":"), allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def practical_runtime_stats(self):
        rt = getattr(self, "_practical_runtime", {})
        return dict(enabled=rt.get("enabled", False), counters=dict(rt.get("counters", {})),
                    table_cells=len(rt.get("table", {})), precompute_seconds=rt.get("precompute_seconds", 0.),
                    soc_error_budget=2e-5, telemetry_period=rt.get("telemetry_period", self.physics_dt))

    def _event(self, kind, aircraft_id=None, **details):
        self.events.append({"time": self.time, "type": kind, "aircraft_id": aircraft_id, "details": details})

    def _result(self, accepted, reason="", **details):
        return ValidationResult(accepted, "ACCEPTED" if accepted else "REJECTED", reason, details)

    def _reject(self, reason, aircraft_id=None, **details):
        self.rejections.append({"time": self.time, "aircraft_id": aircraft_id, "reason": reason, "details": details})
        return self._result(False, reason, **details)

    def reject(self,reason,aircraft_id=None,**details):
        """Record a test driver's rejected candidate through the public port."""
        return self._reject(reason,aircraft_id,**details)

    def _energy_state(self):
        if not hasattr(self, "_targeted_energy"):
            self._targeted_energy = {"fine_plans": {}, "executed": {}}
        return self._targeted_energy

    def _energy(self, plan, start, end, *, fine=None):
        identity = getattr(plan, "trajectory_hash", id(plan))
        state = self._energy_state()
        if fine is None:
            fine = state["fine_plans"].get(identity, False)
        if fine and plan.kind not in ("STABLE", "GROUND"):
            state["fine_plans"][identity] = True
        key = (identity, float(start), float(end), self.config.profile_hash, bool(fine))
        if self.cache_enabled and key in self._energy_cache:
            self.cache_hits += 1
            return self._energy_cache[key]
        t = walltime.perf_counter()
        value = integrate_energy(plan, start, end, max_dt=1., config=self.config, fine=bool(fine))
        self.cpu["energy"] += walltime.perf_counter() - t
        if self.cache_enabled:
            self._store_cache(self._energy_cache,key,value,self.ENERGY_CACHE_CAPACITY)
        return value

    @staticmethod
    def _sum_energy(values):
        from .energy import EnergyResult
        values = tuple(values)
        return EnergyResult(math.fsum(q.joules for q in values), math.fsum(q.soc_used for q in values),
            math.fsum(q.peak_seconds for q in values), all(q.valid for q in values),
            reason=next((q.reason for q in values if q.reason), ""),
            error_j=math.fsum(q.error_j for q in values),
            peak_error_s=math.fsum(q.peak_error_s for q in values))

    def _energy_capacity(self):
        return self.config.energy.equivalent_voltage_V*self.config.energy.reference_capacity_Ah*3600

    def _refine_executed_energy(self, aircraft_id):
        """Correct only numerical charges accumulated since this world's origin.

        This correction and fine selection remain valid even if a later candidate
        is rejected; time, physical trajectories and reservations do not change.
        """
        a = self.aircraft[aircraft_id]
        delta_j, delta_peak = [], []
        for record in self._energy_state()["executed"].get(aircraft_id, ()):
            if record["fine"]:
                continue
            plan = TrajectoryPlan.from_dict(self.trajectories[record["trajectory_hash"]], config=self.config)
            q = self._energy(plan, record["start"], record["end"], fine=True)
            if not q.valid:
                raise RuntimeError("ENERGY_REFINEMENT_INVALID: " + q.reason)
            delta_j.append(q.joules-record["charged_j"])
            delta_peak.append(q.peak_seconds-record["charged_peak"])
            record.update(charged_j=q.joules, charged_peak=q.peak_seconds, error_j=q.error_j,
                          peak_error_s=q.peak_error_s, fine=True)
        correction = math.fsum(delta_j)
        self.aircraft[aircraft_id] = replace(a, energy_j=a.energy_j+correction,
            soc=a.soc-correction/self._energy_capacity(), peak_seconds=a.peak_seconds+math.fsum(delta_peak))
        return correction

    def _budget_energy(self, aircraft_id, portions, *, soc=None, peak=0., reserve=False):
        """Select precision from whole committed/candidate budgets, never a step.

        Full-plan error allocations plus executed allowances define the narrow
        refinement band. Clearly feasible/infeasible remote budgets stay coarse.
        """
        portions = tuple(portions)
        values = tuple(self._energy(p, start, end) for p, start, end in portions)
        if not all(q.valid for q in values):
            return values
        history = ()
        if aircraft_id is not None:
            a = self.aircraft[aircraft_id]
            soc, peak = a.soc, a.peak_seconds
            history = self._energy_state()["executed"].get(aircraft_id, ())
        total = self._sum_energy(values)
        soc_width = (total.error_j+math.fsum(r["error_j"] for r in history))/self._energy_capacity()+1e-5
        peak_width = total.peak_error_s+math.fsum(r["peak_error_s"] for r in history)+.05
        remaining = soc-total.soc_used
        near = abs(remaining) <= soc_width or (reserve and abs(remaining-.25) <= soc_width)
        near |= abs(120.-peak-total.peak_seconds) <= peak_width
        if not near:
            return values
        active = self.active.get(aircraft_id)
        active_was_fine = (active is not None and
                          self._energy_state()["fine_plans"].get(active.trajectory_hash,False))
        correction = 0.
        if aircraft_id is not None:
            correction = self._refine_executed_energy(aircraft_id)
        values = tuple(self._energy(p, start, end, fine=True) for p, start, end in portions)
        if aircraft_id is not None:
            a = self.aircraft[aircraft_id]
            active = self.active.get(aircraft_id)
            if (active is not None and active.kind == "FINAL_NORMAL" and a.normal_final_touchdown_soc is not None
                    and (not active_was_fine or correction != 0.)):
                rest = self._energy(active, self.time, active.end_time, fine=True)
                required_soc = math.fsum((.25,rest.soc_used))
                self.aircraft[aircraft_id] = replace(a, normal_final_touchdown_soc=.25+(a.soc-required_soc))
        return values

    def _record_energy_charge(self, aircraft_id, plan, start, end, q):
        if plan.kind in ("STABLE", "GROUND") or not (q.error_j or q.peak_error_s):
            return
        state = self._energy_state()
        records = state["executed"].setdefault(aircraft_id, [])
        fine = state["fine_plans"].get(plan.trajectory_hash, False)
        if (records and records[-1]["trajectory_hash"] == plan.trajectory_hash
                and records[-1]["fine"] == fine and abs(records[-1]["end"]-start) <= 1e-9):
            record = records[-1]
            record.update(end=end, charged_j=math.fsum((record["charged_j"],q.joules)),
                charged_peak=math.fsum((record["charged_peak"],q.peak_seconds)),
                error_j=math.fsum((record["error_j"],q.error_j)),
                peak_error_s=math.fsum((record["peak_error_s"],q.peak_error_s)))
        else:
            records.append(dict(trajectory_hash=plan.trajectory_hash,start=start,end=end,
                charged_j=q.joules,charged_peak=q.peak_seconds,error_j=q.error_j,
                peak_error_s=q.peak_error_s,fine=fine))

    def _execution_energy_portions(self, a, plan, end):
        """All already committed remaining flight, including a soft successor."""
        future = self.soft.get(a.id)
        current_end = plan.end_time if a.id in self.active else (future.start_time if future else end)
        portions = [(plan, self.time, current_end)]
        if future is not None:
            if future.start_time > current_end:
                hold = make_stable_plan(a.id,future.source_ring,future.source_slot,
                    start_time=current_end,end_time=future.start_time,config=self.config)
                portions.append((hold,current_end,future.start_time))
            portions.append((future,future.start_time,future.end_time))
        return portions

    @staticmethod
    def _store_cache(cache,key,value,capacity):
        """Bound exact-result caches with FIFO eviction; no candidate selection."""
        cache[key]=value
        while len(cache)>capacity:
            cache.popitem(last=False)

    def _stable(self, a, start=None, end=None):
        return make_stable_plan(a.id, a.ring, a.slot, start_time=self.time if start is None else start,
                                end_time=self.time + 1000 if end is None else end,
                                source_version=a.version, config=self.config)

    def _base(self, a):
        if a.id in self.active:
            return self.active[a.id]
        if not self._runtime_enabled():
            if a.ring is not None and a.status in AIRBORNE:
                return self._stable(a)
            return make_ground_plan(a.id, a.pad_id, start_time=self.time,
                                    end_time=a.service_end or self.time + 1000, config=self.config)
        rt = self._practical_runtime
        key = (a.id, a.version, a.status, a.ring, a.slot, a.pad_id, a.trajectory_hash)
        if key in rt["base_plans"]:
            p = rt["base_plans"][key]
            rt["counters"]["base_plan_hits"] += 1
            return p
        archive = self.trajectories.get(a.trajectory_hash)
        p = None
        if archive is not None and archive["kind"] in ("STABLE", "GROUND"):
            p = TrajectoryPlan.from_dict(archive, config=self.config)
        if p is None:
            p = (self._stable(a) if a.ring is not None and a.status in AIRBORNE else
                 make_ground_plan(a.id, a.pad_id, start_time=self.time,
                                  end_time=a.service_end or self.time + 1000, config=self.config))
        rt["base_plans"] = {k:v for k,v in rt["base_plans"].items() if k[0] != a.id}
        rt["base_plans"][key] = p
        return p

    def _traffic(self, a, soft=None, extra=None):
        plans = []
        p = (self.soft if soft is None else soft).get(a.id)
        if p is not None:
            plans.append(p)
        if extra is not None:
            plans.append(extra)
        base = self._base(a)
        final = plans[-1] if plans else base
        end = (final.end_time + service_duration(self.config, a.service_class)) if final.pad_id is not None else math.inf
        if a.status == "LANDED_SERVICE":
            end = a.service_end
        return _Traffic(a.id, base, plans, self.time, end)

    def seed_hold(self, aircraft_id, ring="outer", slot=0, soc=.75, service_class="standard", emergency=False):
        """Named fixtures only; ordinary demand cannot spawn directly into a ring."""
        if self.time != 0 or self._initial_settled or aircraft_id in self.aircraft:
            raise ValueError("fixtures must be seeded before time starts")
        p = make_stable_plan(aircraft_id, ring, slot, start_time=0, end_time=1000, config=self.config)
        s = p.state(0)
        self.slots.reserve((ring, slot), aircraft_id, 0)
        a = AircraftState(aircraft_id, HOLD[ring], soc, s.p, s.v, s.a, s.jerk,
                          service_class=service_class, ring=ring, slot=slot, fixture=True,
                          trajectory_hash=p.trajectory_hash,
                          battery_power_w=evaluate_power(s.v,s.a,self.config).battery_w,
                          emergency=EmergencyStatus(eligible=emergency))
        for b in self.aircraft.values():
            result = validate_separation(p, self._base(b), 0, 180, config=self.config)
            if not result.accepted:
                self.slots._entries = [r for r in self.slots.reservations if r.aircraft_id != aircraft_id]
                raise ValueError(f"unsafe fixture: {result.reason}")
        self.aircraft[aircraft_id] = a
        self.trajectories[p.trajectory_hash] = p.to_dict()
        self._record(force=True)
        return a

    def seed_service(self, aircraft_id, pad_id=0, remaining=60., service_class="short", soc=.5):
        if self.time != 0 or self._initial_settled or remaining <= 0 or aircraft_id in self.aircraft:
            raise ValueError("invalid service fixture")
        p = make_ground_plan(aircraft_id, pad_id, start_time=0, end_time=remaining, config=self.config)
        self.pads.reserve(pad_id, aircraft_id, 0, remaining, kind="service", committed=True)
        s = p.state(0)
        for b in self.aircraft.values():
            result = validate_separation(p,self._base(b),0,min(remaining,180.),config=self.config)
            if not result.accepted:
                self.pads._entries = [r for r in self.pads.reservations if r.aircraft_id != aircraft_id]
                raise ValueError(f"unsafe service fixture: {result.reason}")
        a = AircraftState(aircraft_id, "LANDED_SERVICE", soc, s.p, s.v, s.a, s.jerk,
                          service_class=service_class, pad_id=pad_id, touchdown_time=0,
                          service_end=remaining, fixture=True, trajectory_hash=p.trajectory_hash)
        self.aircraft[aircraft_id] = a
        self.trajectories[p.trajectory_hash] = p.to_dict()
        self._record(force=True)
        return a

    def snapshot(self):
        return self._snapshot(self._scheduler_metrics())

    def _stage2_enabled(self):
        return getattr(self, "experiment_policy_id", None) == STAGE2_POLICY_ID

    def _rescue_v6(self):
        return getattr(self, "rescue_rule_profile", LEGACY_PROFILE) in (RESCUE_PROFILE_V6, RESCUE_PROFILE_V7)

    def _rescue_need_v7(self):
        return getattr(self, "rescue_rule_profile", LEGACY_PROFILE) == RESCUE_PROFILE_V7

    def _declare_rescue_need(self, key, *, pending=False, ring=None, trajectory_hash=None):
        """Record v7 need independently of an available rescue trajectory."""
        a = self.aircraft[key]
        if a.emergency.declared:
            return
        e = replace(a.emergency, declared=True, declaration_time=self.time,
                    declaration_ring=ring or a.ring, pending=pending,
                    eligible=not pending, state="EMERGENCY_WAITING")
        self.aircraft[key] = replace(a, emergency=e)
        self._event("emergency_declared", key, ring=e.declaration_ring,
                    reason=e.rescue_reason, pending=pending,
                    opportunity_status="UNKNOWN", trajectory_hash=trajectory_hash)

    def build_planning_snapshot(self):
        ledger = freeze(self._energy_state())
        runtime=getattr(self,"_practical_runtime",{})
        emergency_monitor=freeze(dict(enabled=bool(runtime.get("enabled",False)),
            experiment_policy_id=getattr(self,"experiment_policy_id",None),
            rescue_rule_profile=getattr(self,"rescue_rule_profile",LEGACY_PROFILE),
            protected_opportunities=tuple(sorted(p.trajectory_hash for p in
                (*self.active.values(), *self.soft.values()) if p.kind == "FINAL_EMERGENCY")),
            soc_threshold=runtime.get("soc_threshold",.35),
            imminent=dict(runtime.get("imminent",{})),monitor=dict(runtime.get("monitor",{}))))
        needed = {a.trajectory_hash for a in self.aircraft.values() if a.status in AIRBORNE}
        needed.update(p.trajectory_hash for p in (*self.active.values(), *self.soft.values()))
        needed.update(r["trajectory_hash"] for rows in self._energy_state()["executed"].values() for r in rows)
        trajectories = freeze({h: self.trajectories[h] for h in sorted(needed) if h in self.trajectories})
        fields = (self.time, tuple(sorted(self.aircraft.items())), tuple(sorted(self.active.items())),
                  tuple(sorted(self.soft.items())), self.slots.reservations, self.pads.reservations,
                  tuple(sorted(self.dependencies.items())), tuple(sorted(self._source_clear.items())),
                  tuple(sorted(self.requests.items())), thaw(ledger), self.failed, thaw(emergency_monitor))
        version = sha256(repr(fields).encode()).hexdigest()
        return PlanningSnapshot(self.time, version, self.config,
            MappingProxyType(dict(self.aircraft)), MappingProxyType(dict(self.active)),
            MappingProxyType(dict(self.soft)), MappingProxyType(dict(self.requests)),
            self.slots.reservations, self.pads.reservations, freeze(self.dependencies),
            freeze(self._source_clear), trajectories, ledger, emergency_monitor)

    def evaluate_submission(self, plans, *, cancel=(), snapshot=None):
        return evaluate_planning_snapshot(snapshot or self.build_planning_snapshot(), plans, cancel=cancel)

    def _normal_estimate(self, a, current=None):
        """Five-second route estimate for warnings only; never a safety proof."""
        e = a.emergency
        if (e.normal_estimate_epoch is None or self.time-e.normal_estimate_epoch > 5.+1e-8
                or e.normal_estimate_source is None or e.normal_path_soc is None):
            return {}
        ring, slot, version, source_time = e.normal_estimate_source
        if current is None:
            match = a.status in HOLD.values() and (a.ring,a.slot,a.version) == (ring,slot,version)
        else:
            match = (current.target_ring,current.target_slot,a.version+1,current.end_time) == (ring,slot,version,source_time)
        if not match:
            return {}
        return dict(normal_path_soc=e.normal_path_soc, normal_path_wait_seconds=e.normal_path_wait_seconds)

    def report_normal_evidence(self, rows, snapshot_version):
        """Accept source-matched planner estimates after its commit transaction.

        Warning metadata participates in snapshot identity. Callers therefore
        report against a fresh snapshot, after committing their validated prefix.
        A route estimate cannot prove need or normal feasibility. Under v7 the
        independent necessary energy bound can declare need at this first check.
        """
        if not self._rescue_v6():
            return self._result(True, accepted_ids=(), ignored_ids=())
        if snapshot_version != self.build_planning_snapshot().version:
            return self._result(False,"STALE_PLANNING_SNAPSHOT")
        accepted, ignored = [], []
        for row in rows:
            key = row.get("aircraft_id")
            a = self.aircraft.get(key)
            if a is None or a.status not in AIRBORNE or a.status.startswith("FINAL") or a.emergency.declared:
                ignored.append(key)
                continue
            current = self.active.get(key)
            source = ((a.ring,a.slot,a.version,self.time) if current is None else
                      (current.target_ring,current.target_slot,a.version+1,current.end_time))
            supplied = tuple(row.get(name) for name in ("source_ring","source_slot","source_version","source_time"))
            q, wait = row.get("normal_path_soc"), row.get("normal_path_wait_seconds",0.)
            if (source != supplied or not isinstance(q,(int,float)) or not math.isfinite(q) or q < 0
                    or not isinstance(wait,(int,float)) or not math.isfinite(wait) or wait < 0):
                ignored.append(key)
                continue
            soc = a.soc if current is None else a.soc-self._energy(current,self.time,current.end_time).soc_used
            rate = stable_power(source[0],self.config).current_a/(self.config.energy.reference_capacity_Ah*3600.)
            assessed = assess_rescue(soc,rate,profile=self.rescue_rule_profile,
                                    normal_energy_lower_bound=normal_energy_lower_bound(self.config,source[0]),
                                    normal_path_soc=q,normal_path_wait_seconds=wait)
            e = replace(a.emergency, warning=assessed.warning, normal_status=assessed.normal_status,
                normal_margin=assessed.normal_margin, normal_path_soc=q,normal_path_wait_seconds=wait,
                normal_estimate_epoch=self.time,normal_estimate_source=source)
            self.aircraft[key] = replace(a,emergency=e)
            self._event("normal_path_estimate",key,estimate_status="PATH_ESTIMATE",normal_margin=assessed.normal_margin,
                        estimate_epoch=self.time,source=source,normal_path_soc=q,normal_path_wait_seconds=wait)
            if e.warning and not a.emergency.warning:
                self._event("rescue_warning",key,normal_status=assessed.normal_status,
                            normal_margin=assessed.normal_margin,soc=soc,estimate_status="PATH_ESTIMATE")
            if self._rescue_need_v7() and assessed.rescue_required:
                if not (a.emergency.eligible or a.emergency.pending):
                    self._event("emergency_triggered",key,pending=current is not None,
                                reason=assessed.rescue_reason,normal_status=assessed.normal_status,
                                deadline_basis="necessary_normal_energy_bound",
                                assessment_ring=source[0],assessment_time=source[3],assessment_soc=soc)
                self.aircraft[key] = replace(self.aircraft[key], emergency=replace(e,
                    rescue_reason=assessed.rescue_reason, eligible=current is None, pending=current is not None))
                self._declare_rescue_need(key,pending=current is not None,ring=source[0])
            accepted.append(key)
        return self._result(True,accepted_ids=tuple(accepted),ignored_ids=tuple(ignored))

    def _emergency_waiting_queue(self):
        """Read-only revealed waiting queue, shared by dispatch and evaluation."""
        return emergency_waiting_queue(self.aircraft, self.active, self.config,
                                       profile=getattr(self,"rescue_rule_profile",LEGACY_PROFILE))

    def _emergency_priority_wait(self, plan_or_aircraft_id):
        """Return a normal-final blocker without mutating scheduling state."""
        if not self._stage2_enabled():
            return None
        if not isinstance(plan_or_aircraft_id, str) and plan_or_aircraft_id.kind != "FINAL_NORMAL":
            return None
        key = (plan_or_aircraft_id if isinstance(plan_or_aircraft_id, str)
               else plan_or_aircraft_id.aircraft_id)
        # Only an urgent, undeclared inner-ring aircraft may use the existing
        # reserve-certified normal exception, and only ahead of other waiters.
        return emergency_priority_blocker(self.aircraft, self.active, self.config, key,
            profile=getattr(self,"rescue_rule_profile",LEGACY_PROFILE),
            proposed_plan=None if isinstance(plan_or_aircraft_id,str) else plan_or_aircraft_id,
            protected_plans=(*self.active.values(), *self.soft.values()))

    def _yield_normal_finals(self):
        """Attempt ordinary soft cancellations through the existing transaction."""
        if not self._stage2_enabled():
            return
        if self._rescue_v6():
            # Rescue replacement/preemption is one complete validated batch in
            # _prepare_submission; urgent presence alone cannot cancel a plan.
            return
        for key, plan in sorted(tuple(self.soft.items())):
            blocked = self._emergency_priority_wait(plan)
            if blocked is None:
                continue
            result = self.cancel(key)
            protected_by = sorted(h for h, deps in self.dependencies.items()
                                  if plan.trajectory_hash in deps)
            self._event("emergency_priority_yielded" if result.accepted else "emergency_priority_yield_blocked",
                        key, trajectory_hash=plan.trajectory_hash,
                        reason="EMERGENCY_PRIORITY_WAIT" if result.accepted else result.reason,
                        dependent_started_plans=protected_by, **blocked)
        if self._emergency_waiting_queue():
            for key, plan in sorted(self.active.items()):
                if plan.kind == "FINAL_NORMAL":
                    self._event("emergency_priority_yield_blocked", key,
                                trajectory_hash=plan.trajectory_hash, reason="MANEUVER_ALREADY_STARTED",
                                **self._emergency_priority_wait(plan))

    def refine_planning_boundary(self):
        """One uniform executed-ledger refinement per decision timestamp."""
        if getattr(self, "_refinement_time", None) != self.time:
            for key,a in tuple(self.aircraft.items()):
                if a.status in AIRBORNE:
                    self._refine_executed_energy(key)
            if self._stage2_enabled():
                self._refinement_time = self.time
        return self.build_planning_snapshot()

    def commit_validated_prefix(self, result, snapshot_version=None):
        if not isinstance(result, SubmissionEvaluation):
            raise TypeError("SubmissionEvaluation required")
        version = result.snapshot_version if snapshot_version is None else snapshot_version
        if self.failed:
            return self._reject("EPISODE_STOPPED")
        if (result.time != self.time or result.snapshot_version != version
                or self.build_planning_snapshot().version != version):
            return self._reject("STALE_PLANNING_SNAPSHOT")
        if not result.accepted:
            return self._reject(result.reason, **thaw(result.details))
        if result.refinement_required:
            return self._reject("UNIFORM_REFINEMENT_REQUIRED")
        # Revalidate the public result rather than trusting its opaque payload;
        # callers cannot manufacture a bypass by constructing this dataclass.
        checked = evaluate_planning_snapshot(self.build_planning_snapshot(), result.plans, cancel=result.cancel)
        if not checked.accepted:
            return self._reject(checked.reason, **thaw(checked.details))
        if checked.refinement_required:
            return self._reject("UNIFORM_REFINEMENT_REQUIRED")
        return self._commit_submission(checked.plans, checked._prepared)

    def _snapshot(self,metrics):
        return WorldSnapshot(self.time, tuple(self.aircraft[k] for k in sorted(self.aircraft)),
                             tuple(self.requests[k] for k in sorted(self.requests)), self.slots.reservations,
                             self.pads.reservations, freeze(metrics), self.failed,
                             tuple(freeze(r) for r in self._upstream_rows()))

    def _upstream_rows(self):
        return [dict(r.to_dict(), status="CENSORED_UPSTREAM" if self.failed else "WAITING",
                     wait_seconds=max(0., self.time-r.release_time),
                     **self.upstream_reasons.get(r.id, {"reason": "AWAITING_DECISION",
                         "reason_categories": [], "last_attempt_time": None}))
                for r in sorted(self.requests.values(), key=lambda r: (r.release_time, r.id))
                if r.id not in self.aircraft]

    def _record(self, force=False):
        if not force and self._runtime_enabled():
            period = self._practical_runtime["telemetry_period"]
            on_tick = abs(self.time/period-round(self.time/period)) <= 1e-8
            event_now = bool(self.events and abs(self.events[-1]["time"]-self.time) <= 1e-8)
            junction = any(self._descent_start(p) == self.time for p in self.active.values())
            if not (on_tick or event_now or junction or self.failed):
                return
        # Replay/audit output retains demand conservation totals. Scheduling
        # snapshots deliberately omit unrevealed manifest cardinalities.
        record = self._snapshot(self.metrics()).to_dict()
        if self._last_record_time != self.time:
            self._record_count += 1
            self._last_record_time = self.time
        if self.state_sink is not None:
            self.state_sink(record)
            # The sink owns finalized history; retain the mutable timestamp's
            # latest complete frame only. Default recording remains unchanged.
            self.records[:] = [record]
        elif self.records and self.records[-1]["time"] == self.time:
            self.records[-1] = record
        else:
            self.records.append(record)

    def flush_records(self):
        """Flush an external recorder without materializing its frame history."""
        if self.state_sink is not None:
            flush = getattr(self.state_sink,"flush",None)
            if flush is not None:
                flush()

    def _protect_times(self, plan):
        """Continuous suffix/prefix certificates define conservative clear/capture events."""
        clear, capture = plan.start_time, plan.start_time
        horizon = plan.end_time + 2 * math.pi / .035
        if plan.source_ring is not None:
            source = make_stable_plan("source_reference", plan.source_ring, plan.source_slot,
                                      start_time=plan.start_time, end_time=horizon, config=self.config)
            lo, hi = plan.start_time, plan.end_time
            result = validate_separation(plan, source, hi, horizon, config=self.config)
            if not result.accepted:
                return None, None
            while hi - lo > .01:
                mid = (lo + hi) / 2
                if validate_separation(plan, source, mid, horizon, config=self.config).accepted:
                    hi = mid
                else:
                    lo = mid
            clear = hi
        if plan.target_ring is not None:
            target = make_stable_plan("target_reference", plan.target_ring, plan.target_slot,
                                      start_time=plan.start_time, end_time=horizon, config=self.config)
            lo, hi = plan.start_time, plan.end_time
            while hi - lo > .01:
                mid = (lo + hi) / 2
                if validate_separation(plan, target, plan.start_time, mid, config=self.config).accepted:
                    lo = mid
                else:
                    hi = mid
            capture = lo
        return clear, capture

    def _calendar_stage(self, soft, new_plans):
        slots, pads = SlotCalendar(), PadCalendar()
        all_plans = tuple(self.active.values()) + tuple(soft.values())
        live_hashes = {p.trajectory_hash for p in all_plans}
        times = {h: value for h,value in self._source_clear.items() if h in live_hashes}
        for p in new_plans:
            times[p.trajectory_hash] = self._protect_times(p)
            if times[p.trajectory_hash][0] is None:
                raise ResourceConflict("source continuation cannot be certified clear")
        for a in self.aircraft.values():
            if a.status not in AIRBORNE and a.status != "LANDED_SERVICE":
                continue
            current = self.active.get(a.id)
            future = soft.get(a.id)
            if a.ring is not None:
                departure = current or future
                end = times[departure.trajectory_hash][0] if departure is not None else math.inf
                fallback = a.admission_time if a.admission_time is not None else a.release_time
                # Admission precedes capture for inbound flights. Preserve the
                # actual slot interval when rebuilding a reused source slot.
                entered = min((r.start for r in self.slots.reservations
                               if r.aircraft_id == a.id and r.resource == (a.ring,a.slot)),
                              default=fallback)
                slots.reserve((a.ring, a.slot), a.id, entered, end,
                              kind="source" if departure else "occupied",
                              plan_hash=departure.trajectory_hash if departure else None,
                              committed=current is not None)
            if a.status == "LANDED_SERVICE":
                pads.reserve(a.pad_id, a.id, a.touchdown_time, a.service_end, kind="service", committed=True)
            for p in (current, future):
                if p is None:
                    continue
                clear, capture = times[p.trajectory_hash]
                if p.target_ring is not None:
                    nextp = future if p is current else None
                    end = times[nextp.trajectory_hash][0] if nextp else math.inf
                    slots.reserve((p.target_ring, p.target_slot), a.id, capture, end, kind="target",
                                  plan_hash=p.trajectory_hash, committed=p is current)
                if p.pad_id is not None:
                    start = p.end_time - 13.5
                    pads.reserve(p.pad_id, a.id, start, p.end_time + service_duration(self.config, a.service_class),
                                 kind="emergency" if p.kind == "FINAL_EMERGENCY" else "final",
                                 plan_hash=p.trajectory_hash, committed=p is current)
        isolated = {a.id for a in self.aircraft.values() if a.status == "ENERGY_MISSION_FAILED"}
        for before, after in ((self.slots, slots), (self.pads, pads)):
            after._entries.extend(r for r in before.reservations if r.aircraft_id in isolated)
        return slots, pads, times

    def _future_support_witness(self, plan):
        """Forecast known emergency disposition before binding a soft release.

        This screens the current target and scheduled source timestamps only;
        emergency handling still guards changes that this forecast cannot prove.
        """
        if plan.kind == "FINAL_EMERGENCY" or plan.start_time <= self.time:
            return None
        a = self.aircraft[plan.aircraft_id]
        current = self.active.get(a.id)
        source_time, predicted_soc = self.time, a.soc
        checkpoints = []
        if current is not None:
            rest = self._energy(current, self.time, current.end_time)
            predicted_soc -= rest.soc_used
            source_time = current.end_time
            checkpoints.append((source_time, predicted_soc, "current_target"))
        if plan.start_time > source_time:
            hold = make_stable_plan(a.id, plan.source_ring, plan.source_slot,
                start_time=source_time, end_time=plan.start_time,
                source_version=plan.source_version, config=self.config)
            predicted_soc -= self._energy(hold, source_time, plan.start_time).soc_used
        checkpoints.append((plan.start_time, predicted_soc, "scheduled_start"))
        for t, soc, witness in checkpoints:
            q, _ = self._reference(plan.source_ring, plan.source_slot, t, plan.source_version)
            hold = make_stable_plan("reference_hold", plan.source_ring, plan.source_slot,
                                   start_time=t, end_time=t+10, config=self.config)
            qhold = self._energy(hold, t, t+10).soc_used
            alert = (assess_rescue(soc, qhold/10., rescue_soc=q, profile=self.rescue_rule_profile,
                                  normal_energy_lower_bound=normal_energy_lower_bound(self.config, plan.source_ring),
                                  declared=a.emergency.declared,
                                  pending=a.emergency.pending).rescue_required
                     if self._rescue_v6() else emergency_threshold(soc, q, qhold))
            if alert:
                if self._runtime_enabled():
                    self._practical_runtime["imminent"][a.id] = dict(time=t,
                        ring=plan.source_ring, slot=plan.source_slot, version=plan.source_version)
                return {"supporting_aircraft_id": a.id, "trajectory_hash": plan.trajectory_hash,
                        "witness": witness, "witness_time": t, "predicted_soc": soc,
                        "q_e": q, "q_hold": qhold}
        return None

    def _release_supports(self, ring, slot, aircraft_id, slots, soft):
        supports = set()
        for reservation in slots.reservations:
            if (reservation.resource != (ring, slot) or reservation.aircraft_id == aircraft_id
                    or not reservation.plan_hash):
                continue
            supports.add(reservation.plan_hash)
            # An active plan's target reservation can end at its soft successor's
            # source clear; that successor is the release supporting this claim.
            current = self.active.get(reservation.aircraft_id)
            future = soft.get(reservation.aircraft_id)
            if (current is not None and future is not None
                    and reservation.plan_hash == current.trajectory_hash
                    and reservation.resource == (current.target_ring, current.target_slot)
                    and reservation.resource == (future.source_ring, future.source_slot)):
                supports.add(future.trajectory_hash)
        return supports

    def _future_dependency_witness(self, dependencies, soft, slots):
        by_hash = {p.trajectory_hash: p for p in soft.values()}
        pending, seen = set(dependencies), set()
        while pending:
            plan_hash = min(pending)
            pending.remove(plan_hash)
            if plan_hash in seen:
                continue
            seen.add(plan_hash)
            plan = by_hash.get(plan_hash)
            if plan is None:
                continue
            witness = self._future_support_witness(plan)
            if witness is not None:
                return witness
            if plan.target_ring is not None:
                pending.update(self._release_supports(plan.target_ring, plan.target_slot,
                                                     plan.aircraft_id, slots, soft)-seen)
        return None

    def submit(self, plans, *, cancel=(), simulated_latency=0.):
        if simulated_latency < 0:
            return self._reject("INVALID_INPUT")
        if simulated_latency:
            self.advance(self.time + simulated_latency, decision_callback=False)
            self._event("decision_late", latency=simulated_latency)
        if self.failed:
            return self._reject("EPISODE_STOPPED")
        if self._stage2_enabled():
            result = self.evaluate_submission(plans, cancel=cancel)
            if result.refinement_required:
                snapshot = self.refine_planning_boundary()
                result = self.evaluate_submission(plans, cancel=cancel, snapshot=snapshot)
            return self.commit_validated_prefix(result)
        plans = tuple(plans)
        prepared = self._prepare_submission(plans, cancel=cancel)
        if isinstance(prepared, ValidationResult):
            return prepared
        return self._commit_submission(tuple(plans), prepared)

    def _prepare_submission(self, plans, *, cancel=()):
        plans = tuple(plans)
        ids = [p.aircraft_id for p in plans]
        if len(ids) != len(set(ids)):
            return self._reject("DUPLICATE_AIRCRAFT")
        soft = dict(self.soft)
        changed = set(cancel) | set(ids)
        for key in cancel:
            if key in self.active and key not in self.soft:
                return self._reject("COMMITTED_MODIFICATION",key)
        protected = {h for deps in self.dependencies.values() for h in deps}
        redirects = {}
        for aircraft_id in changed:
            old = soft.get(aircraft_id)
            if old is not None and old.trajectory_hash in protected:
                replacement = next((p for p in plans if p.aircraft_id == aircraft_id), None)
                equivalent = replacement is not None and (replacement.source_ring, replacement.source_slot) == (old.source_ring, old.source_slot)
                if equivalent and replacement.trajectory_hash != old.trajectory_hash:
                    old_clear = self._source_clear[old.trajectory_hash][0]
                    replacement_clear, _ = self._protect_times(replacement)
                    equivalent = replacement_clear is not None and replacement_clear <= old_clear
                    if equivalent:
                        redirects[old.trajectory_hash] = replacement.trajectory_hash
                if not equivalent:
                    return self._reject("RESOURCE_DEPENDENCY", aircraft_id)
            soft.pop(aircraft_id, None)
        for p in plans:
            a = self.aircraft.get(p.aircraft_id)
            if a is None:
                return self._reject("UNKNOWN_AIRCRAFT", p.aircraft_id)
            if p.start_time < self.time - 1e-9:
                return self._reject("EXPIRED_TIME", a.id)
            current = self.active.get(a.id)
            if current is not None:
                if current.pad_id is not None or p.start_time < current.end_time:
                    return self._reject("COMMITTED_MODIFICATION", a.id)
                source_ring, source_slot, source_version = current.target_ring, current.target_slot, a.version+1
                source_time = current.end_time
            else:
                source_ring, source_slot, source_version, source_time = a.ring, a.slot, a.version, self.time
            if p.source_version != source_version:
                return self._reject("STALE_STATE", a.id)
            if current is None and a.status not in HOLD.values():
                return self._reject("INVALID_SOURCE_STATE", a.id)
            if p.source_ring != source_ring or p.source_slot != source_slot:
                return self._reject("STALE_STATE", a.id)
            if p.kind not in ("TRANSFER", "FINAL_NORMAL", "FINAL_EMERGENCY"):
                return self._reject("INVALID_ACTION", a.id)
            emergency = p.kind == "FINAL_EMERGENCY"
            if emergency and not a.emergency.eligible:
                return self._reject("NO_EMERGENCY_ELIGIBILITY", a.id)
            if a.emergency.declared and not emergency:
                return self._reject("EMERGENCY_DECLARED", a.id)
            normal_exception = (p.kind == "FINAL_NORMAL" and source_ring == "inner" and
                                current is None and p.start_time == self.time and not a.emergency.declared
                                and not self._rescue_v6())
            priority_wait = self._emergency_priority_wait(p)
            if priority_wait is not None:
                return self._reject("EMERGENCY_PRIORITY_WAIT", a.id, **priority_wait)
            if (a.emergency.eligible or a.emergency.pending) and not emergency and not normal_exception:
                return self._reject("EMERGENCY_DISPOSITION_REQUIRED", a.id)
            if p.kind == "FINAL_NORMAL" and source_ring != "inner":
                return self._reject("ILLEGAL_NORMAL_TOPOLOGY", a.id)
            if p.kind.startswith("TRANSFER") and (source_ring, p.target_ring) not in (("outer", "middle"), ("middle", "inner")):
                return self._reject("ILLEGAL_NORMAL_TOPOLOGY", a.id)
            if emergency:
                if p.start_time != self.time and not self._rescue_v6():
                    return self._reject("EMERGENCY_MUST_START_NOW", a.id)
                # Ordinary soft final may be preempted transactionally.
                for other_id, old in tuple(soft.items()):
                    if (old.pad_id == p.pad_id and old.kind == "FINAL_NORMAL" and
                            old.end_time-13.5 < p.end_time+service_duration(self.config, a.service_class) and
                            p.end_time-13.5 < old.end_time+service_duration(self.config, self.aircraft[other_id].service_class)):
                        if old.trajectory_hash in protected:
                            return self._reject("RESOURCE_DEPENDENCY", a.id)
                        del soft[other_id]
                        changed.add(other_id)
            result = validate_plan(p, config=self.config)
            if not result.accepted:
                return self._reject(result.reason, a.id, validation=result.details)
            portions = []
            if p.start_time > source_time:
                hold = make_stable_plan(a.id, source_ring, source_slot, start_time=source_time,
                                        end_time=p.start_time, source_version=source_version, config=self.config)
                portions.append((hold, source_time, p.start_time))
            if current is not None:
                portions.append((current, self.time, current.end_time))
            portions.append((p, p.start_time, p.end_time))
            estimates = self._budget_energy(a.id, portions, reserve=p.kind == "FINAL_NORMAL")
            a = self.aircraft[a.id]
            holding, energy = self._sum_energy(estimates[:-1]), estimates[-1]
            if not holding.valid or not energy.valid:
                return self._reject(energy.reason or holding.reason, a.id)
            remaining = a.soc - holding.soc_used - energy.soc_used
            if remaining <= 0:
                return self._reject("SOC_INSUFFICIENT", a.id)
            if a.peak_seconds + holding.peak_seconds + energy.peak_seconds > 120:
                return self._reject("PEAK_BUDGET", a.id)
            if p.kind == "FINAL_NORMAL":
                required_soc = math.fsum((.25, holding.soc_used, energy.soc_used))
                predicted_soc = .25 + (a.soc-required_soc)
                if a.soc < required_soc:
                    return self._reject("LANDING_RESERVE", a.id, predicted_soc=predicted_soc,
                                        required_soc=required_soc)
            soft[a.id] = p
        try:
            slots, pads, times = self._calendar_stage(soft, plans)
        except (ResourceConflict, ValueError) as exc:
            blocking = [r.to_dict() for p in plans if p.pad_id is not None
                        for r in self.pads.conflicts(p.pad_id, p.end_time-13.5,
                            p.end_time+service_duration(self.config, self.aircraft[p.aircraft_id].service_class), p.aircraft_id)]
            return self._reject("RESOURCE_CONFLICT", detail=str(exc), blocking_reservations=blocking)
        traffic = [self._traffic(a, soft) for a in self.aircraft.values()
                   if a.status in AIRBORNE or a.status == "LANDED_SERVICE"]
        start_cpu = walltime.perf_counter()
        for x, y in combinations(traffic, 2):
            if x.aircraft_id not in changed and y.aircraft_id not in changed:
                continue
            horizon = max([self.time, *(p.end_time for p in self.active.values()),
                           *(p.end_time for p in soft.values())]) + 2*math.pi/.035
            end = min(horizon, x.end_time, y.end_time)
            if end > self.time:
                result = validate_separation(x, y, self.time, end, config=self.config)
                if not result.accepted:
                    self.cpu["validation"] += walltime.perf_counter() - start_cpu
                    return self._reject(result.reason, pair=[x.aircraft_id, y.aircraft_id], validation=result.details)
        self.cpu["validation"] += walltime.perf_counter() - start_cpu
        # Bind a future release only after forecasting its known emergency
        # disposition. Unprotected soft plans remain editable and preemptible.
        binding = set(redirects.values())
        for p in plans:
            if p.start_time <= self.time+1e-9 and p.target_ring is not None:
                binding.update(self._release_supports(p.target_ring, p.target_slot,
                                                      p.aircraft_id, slots, soft))
        witness = self._future_dependency_witness(binding, soft, slots)
        if witness is not None:
            return self._reject("FUTURE_EMERGENCY_DISPOSITION", **witness)
        return (soft, slots, pads, times, redirects, dict(self._energy_state()["fine_plans"]))

    def _commit_submission(self, plans, prepared):
        soft, slots, pads, times, redirects, fine_plans = prepared
        self._energy_state()["fine_plans"].update(fine_plans)
        ids = [p.aircraft_id for p in plans]
        removed = [old for key,old in self.soft.items() if key not in soft or soft[key].trajectory_hash != old.trajectory_hash]
        self.soft, self.slots, self.pads, self._source_clear = soft, slots, pads, times
        for old in removed:
            self._event("soft_plan_cancelled",old.aircraft_id,trajectory_hash=old.trajectory_hash,
                        reason="atomic_replacement_or_preemption")
            if (self._rescue_need_v7() and old.kind == "FINAL_EMERGENCY"
                    and old.aircraft_id not in self.active
                    and (old.aircraft_id not in soft or soft[old.aircraft_id].kind != "FINAL_EMERGENCY")):
                a = self.aircraft[old.aircraft_id]
                self.aircraft[a.id] = replace(a, emergency=replace(a.emergency, state="EMERGENCY_WAITING"))
        if redirects:
            self.dependencies = {h: tuple(redirects.get(d,d) for d in ds) for h,ds in self.dependencies.items()}
        for p in plans:
            self.trajectories[p.trajectory_hash] = p.to_dict()
            if self._rescue_v6() and p.kind == "FINAL_EMERGENCY":
                a = self.aircraft[p.aircraft_id]
                if not a.emergency.declared:
                    self.aircraft[a.id] = replace(a, emergency=replace(a.emergency,
                        declared=True, declaration_time=self.time, declaration_ring=a.ring,
                        state="EMERGENCY_WAITING"))
                    self._event("emergency_declared", a.id, ring=a.ring,
                                reason=a.emergency.rescue_reason, trajectory_hash=p.trajectory_hash)
                if self._rescue_need_v7():
                    a = self.aircraft[p.aircraft_id]
                    self.aircraft[a.id] = replace(a, emergency=replace(a.emergency,
                        state="EMERGENCY_RESERVED", pending=False))
            self._event("plan_committed" if p.start_time == self.time else "soft_plan_reserved", p.aircraft_id,
                        trajectory_hash=p.trajectory_hash, start_time=p.start_time, end_time=p.end_time)
        # A submit within emergency handling must not start another aircraft's
        # scheduled ordinary maneuver before that aircraft receives disposition.
        self._start_due(ids)
        self._record(force=True)
        return self._result(True)

    def cancel(self, aircraft_id):
        return self.submit((), cancel=(aircraft_id,))

    def _start_due(self, aircraft_ids=None):
        selected = None if aircraft_ids is None else set(aircraft_ids)
        for aircraft_id, p in sorted(tuple(self.soft.items())):
            if selected is not None and aircraft_id not in selected:
                continue
            if p.start_time > self.time:
                continue
            # A nested cancellation/submit may have changed this finite copy.
            if self.soft.get(aircraft_id) != p:
                continue
            a = self.aircraft[aircraft_id]
            priority_wait = self._emergency_priority_wait(p)
            if priority_wait is not None:
                result = self.cancel(a.id)
                protected_by = sorted(h for h, deps in self.dependencies.items()
                                      if p.trajectory_hash in deps)
                self._event("emergency_priority_yielded" if result.accepted else "emergency_priority_yield_blocked",
                            a.id, trajectory_hash=p.trajectory_hash,
                            reason="EMERGENCY_PRIORITY_WAIT" if result.accepted else result.reason,
                            dependent_started_plans=protected_by, **priority_wait)
                if result.accepted:
                    continue
                # A failed cancellation does not certify safe holding. Keep
                # the original certified departure, including geometric
                # commitments not represented by an explicit slot dependency.
            normal_exception = (p.kind == "FINAL_NORMAL" and a.ring == "inner" and
                                not a.emergency.declared and not a.emergency.pending and not self._rescue_v6())
            if p.kind != "FINAL_EMERGENCY" and (a.emergency.declared or
                    ((a.emergency.eligible or a.emergency.pending) and not normal_exception)):
                self._reject("EMERGENCY_DISPOSITION_REQUIRED", a.id, trajectory_hash=p.trajectory_hash)
                self._event("commit_invalidated", a.id, reason="EMERGENCY_DISPOSITION_REQUIRED",
                            trajectory_hash=p.trajectory_hash)
                result = self.cancel(a.id)
                if not result.accepted:
                    if self._rescue_v6():
                        self._event("rescue_commitment_protected", a.id, reason=result.reason,
                                    trajectory_hash=p.trajectory_hash)
                    else:
                        self._emergency_conflict(a.id, result)
                else:
                    continue
            # Energy was forecast at submission; revalidate the exact source/version.
            if a.id in self.active or a.version != p.source_version:
                self._event("commit_invalidated", a.id, reason="STALE_STATE")
                del self.soft[a.id]
                continue
            binding = (self._release_supports(p.target_ring, p.target_slot, a.id,
                                             self.slots, self.soft)
                       if p.target_ring is not None else set())
            witness = self._future_dependency_witness(binding, self.soft, self.slots)
            if witness is not None:
                self._reject("FUTURE_EMERGENCY_DISPOSITION", a.id, **witness)
                self._event("commit_invalidated", a.id, reason="FUTURE_EMERGENCY_DISPOSITION", **witness)
                result = self.cancel(a.id)
                if not result.accepted:
                    self._emergency_conflict(a.id, result)
                continue
            energy = self._sum_energy(self._budget_energy(a.id, ((p,self.time,p.end_time),),
                                                       reserve=p.kind == "FINAL_NORMAL"))
            a = self.aircraft[a.id]
            required_soc = math.fsum((.25 if p.kind == "FINAL_NORMAL" else 0.,energy.soc_used))
            reason = (energy.reason if not energy.valid else
                      "SOC_INSUFFICIENT" if a.soc-energy.soc_used <= 0 else
                      "PEAK_BUDGET" if a.peak_seconds+energy.peak_seconds > 120 else
                      "LANDING_RESERVE" if p.kind == "FINAL_NORMAL" and a.soc < required_soc else "")
            if reason:
                self._reject(reason,a.id,trajectory_hash=p.trajectory_hash)
                self._event("commit_invalidated",a.id,reason=reason,trajectory_hash=p.trajectory_hash)
                result = self.cancel(a.id)
                if not result.accepted:
                    self._emergency_conflict(a.id,result)
                continue
            self.active[a.id] = p
            del self.soft[a.id]
            self.slots.mark_committed(p.trajectory_hash)
            self.pads.mark_committed(p.trajectory_hash)
            deps = (self._release_supports(p.target_ring, p.target_slot, a.id,
                                          self.slots, self.soft)
                    if p.target_ring is not None else set())
            self.dependencies[p.trajectory_hash] = tuple(sorted(deps))
            e = a.emergency
            if self._rescue_v6():
                e = replace(e,normal_path_soc=None,normal_path_wait_seconds=0.,
                            normal_estimate_epoch=None,normal_estimate_source=None)
            if p.kind == "FINAL_NORMAL" and a.ring == "inner" and not e.declared:
                e=replace(e,eligible=False,pending=False,state="NONE")
            if p.kind == "FINAL_EMERGENCY" and not e.charged:
                self.costs["emergency"] += {"inner": 1200., "middle": 2400., "outer": 3600.}[a.ring]
                e = replace(e, eligible=True, declared=True, pending=False, charged=True,
                            declaration_time=e.declaration_time if e.declaration_time is not None else self.time,
                            declaration_ring=e.declaration_ring or a.ring,
                            start_time=self.time, start_ring=a.ring, state="FINAL_EMERGENCY")
            status = ("TRANSFER_O_M" if p.source_ring == "outer" else "TRANSFER_M_I") if p.kind == "TRANSFER" else p.kind
            start_details = {}
            if self._rescue_need_v7() and p.kind == "FINAL_EMERGENCY":
                start_details["opportunity_status"] = "ACTIVE"
            touchdown_soc = None
            if p.kind == "FINAL_NORMAL":
                required_soc = math.fsum((.25, energy.soc_used))
                touchdown_soc = .25 + (a.soc-required_soc)
                start_details = {"start_soc": a.soc, "required_soc": required_soc,
                                 "predicted_touchdown_soc": touchdown_soc}
            self.aircraft[a.id] = replace(a, status=status, emergency=e, trajectory_hash=p.trajectory_hash,
                                         normal_final_touchdown_soc=touchdown_soc)
            self._event("maneuver_started", a.id, trajectory_hash=p.trajectory_hash,
                        dependencies=sorted(deps), maneuver_kind=p.kind, **start_details)

    def _fail(self, aircraft_id, reason):
        if self._stage2_enabled():
            self.failed, self.terminated_invalid = True, True
            self._event("invalid_run_stopped", aircraft_id, reason=reason)
            return
        self.failed, self.failure_time = True, self.time
        self.costs["failure"] += 1000000.
        for key, a in tuple(self.aircraft.items()):
            if key == aircraft_id:
                self.aircraft[key] = replace(a, status=reason, soc=max(0., a.soc))
            elif a.status in AIRBORNE:
                self.aircraft[key] = replace(a, status="CENSORED_DUE_TO_FAILURE")
        a = self.aircraft[aircraft_id]
        self._event(reason, aircraft_id, p=a.p, soc=a.soc)
        self._event("episode_stopped", unreleased=len(self.manifest.requests)-self._manifest_index)

    def _isolate_energy_failures(self, failures):
        """Apply the explicitly opted-in Stage2 task-loss abstraction atomically."""
        for key, reason in failures:
            if key in self.failure_archive:
                continue
            a = self.aircraft[key]
            own_active, own_soft = self.active.pop(key, None), self.soft.pop(key, None)
            hashes = {p.trajectory_hash for p in (own_active, own_soft) if p is not None}
            reservations = tuple(r for r in (*self.slots.reservations, *self.pads.reservations)
                                 if r.aircraft_id == key)
            self.failure_archive[key] = dict(time=self.time, reason=reason, last_state=a.to_dict(),
                active=None if own_active is None else own_active.to_dict(),
                soft=None if own_soft is None else own_soft.to_dict(),
                reservations=[r.to_dict() for r in reservations])
            released = self.slots.isolate(key, self.time)+self.pads.isolate(key, self.time)
            self.failure_resources_released += released
            self.dependencies = {h: tuple(d for d in ds if d not in hashes)
                                 for h, ds in self.dependencies.items() if h not in hashes}
            self._source_clear = {h:v for h,v in self._source_clear.items() if h not in hashes}
            self.aircraft[key] = replace(a, status="ENERGY_MISSION_FAILED",
                soc=0. if reason == "BATTERY_DEPLETED" else max(0.,a.soc), battery_power_w=0.)
            self.costs["failure"] += 1000000.
            self.has_energy_losses = True
            if self.failure_time is None:
                self.failure_time = self.time
            self._event("aircraft_energy_failure", key, reason=reason)
            self._event("failure_isolated", key, reason=reason, resources_released=released)
        self._pending_dispatch = True

    @staticmethod
    def _descent_start(plan):
        """Use the executed final's serialized phase junction."""
        if plan.kind in {"FINAL_NORMAL", "FINAL_EMERGENCY"}:
            return next(s.start_time for s in plan.segments if s.kind == "DESCENT")
        return None

    def _advance_energy(self, end):
        initial, duration = self.time, end-self.time
        results, fail_at = {}, None
        failures = ()
        if self._stage2_enabled():
            end, failures = self._energy_failure_events(end)
            duration = end-initial
        for a in tuple(self.aircraft.values()):
            if a.status not in AIRBORNE:
                continue
            p = self._base(a)
            portions = self._execution_energy_portions(a,p,end)
            state = self._energy_state()
            if not (all(part.kind in ("STABLE","GROUND") or state["fine_plans"].get(part.trajectory_hash,False)
                        for part,_,_ in portions)
                    and all(record["fine"] for record in state["executed"].get(a.id,()))):
                self._budget_energy(a.id,portions,reserve=any(part.kind == "FINAL_NORMAL" for part,_,_ in portions))
            a = self.aircraft[a.id]
            result = self._energy(p, initial, end)
            if not self._stage2_enabled() and (not result.valid or a.soc - result.soc_used <= 0 or a.peak_seconds + result.peak_seconds > 120):
                lo, hi = initial, end
                for _ in range(35):
                    mid = (lo+hi)/2
                    q = self._energy(p, initial, mid)
                    if q.valid and a.soc-q.soc_used > 0 and a.peak_seconds+q.peak_seconds <= 120:
                        lo = mid
                    else:
                        hi = mid
                if fail_at is None or hi < fail_at[0]:
                    fail_at = (hi, a.id)
            results[a.id] = (p, result)
        if fail_at:
            end, _ = fail_at
            duration = end-initial
            results = {key: (p, self._energy(p, initial, end)) for key, (p, _) in results.items()}
        self.time = end
        for pad in range(6):
            if any(r.resource == pad and r.start <= initial < r.end for r in self.pads.reservations):
                self.pad_busy_seconds[pad] += duration
        for key, (p, q) in results.items():
            a = self.aircraft[key]
            s = p.state(end)
            status = a.status
            descent_start = self._descent_start(p)
            if descent_start is not None:
                final_duration = max(0., min(end, descent_start)-max(initial, p.start_time))
                descent_duration = max(0., min(end, p.end_time)-max(initial, descent_start))
                if end >= descent_start:
                    status = "VERTICAL_DESCENT"
            else:
                final_duration = duration if a.status.startswith("FINAL") else 0.
                descent_duration = duration if a.status == "VERTICAL_DESCENT" else 0.
            soc = a.soc-q.soc_used
            # The fixed normal final has an authoritative start SOC and complete
            # energy forecast; preserve its additive reserve boundary at touchdown.
            if (p.kind == "FINAL_NORMAL" and end == p.end_time
                    and a.normal_final_touchdown_soc is not None):
                soc = a.normal_final_touchdown_soc
            self.aircraft[key] = replace(a, p=s.p, v=s.v, a=s.a, jerk=s.jerk,
                soc=soc, energy_j=a.energy_j+q.joules,
                peak_seconds=a.peak_seconds+q.peak_seconds, status=status,
                airborne_time=a.airborne_time+duration,
                holding_time=a.holding_time+(duration if a.status.startswith("HOLD") else 0),
                transfer_time=a.transfer_time+(duration if a.status.startswith("TRANSFER") else 0),
                final_time=a.final_time+final_duration,
                descent_time=a.descent_time+descent_duration,
                battery_power_w=(stable_power(a.ring,self.config).battery_w
                    if p.kind == "STABLE" and self._runtime_enabled()
                    else evaluate_power(s.v,s.a,self.config).battery_w))
            self._record_energy_charge(key,p,initial,end,q)
        if fail_at:
            self._fail(fail_at[1], "ENERGY_FAILURE")
        if failures:
            self._isolate_energy_failures(failures)

    def _energy_failure_events(self, end):
        """Locate the first physical loss before checking later traffic geometry."""
        initial, events = self.time, []
        for a in tuple(self.aircraft.values()):
            if a.status not in AIRBORNE:
                continue
            if not all(math.isfinite(x) for x in (a.soc,a.energy_j,a.peak_seconds,*a.p,*a.v,*a.a)):
                raise RuntimeError("INVALID_PHYSICAL_STATE: "+a.id)
            plan = self._base(a)
            portions = self._execution_energy_portions(a,plan,end)
            budget = self._budget_energy(a.id,portions,reserve=any(p.kind == "FINAL_NORMAL" for p,_,_ in portions))
            if any(not q.valid for q in budget):
                raise RuntimeError("ENERGY_MODEL_INVALID: "+a.id+": "+next(q.reason for q in budget if not q.valid))
            a = self.aircraft[a.id]
            total = self._energy(plan,initial,end)
            if not total.valid or not all(math.isfinite(x) for x in (total.joules,total.soc_used,total.peak_seconds)):
                raise RuntimeError("ENERGY_MODEL_INVALID: "+a.id+": "+total.reason)
            for reason, exhausted in (
                    ("BATTERY_DEPLETED", lambda q: a.soc-q.soc_used <= 0),
                    ("PEAK_BUDGET_EXHAUSTED", lambda q: a.peak_seconds+q.peak_seconds > 120)):
                if not exhausted(total):
                    continue
                lo, hi = initial, end
                if exhausted(self._energy(plan,initial,initial)):
                    hi = initial
                else:
                    for _ in range(40):
                        mid = (lo+hi)/2
                        q = self._energy(plan,initial,mid)
                        if not q.valid:
                            raise RuntimeError("ENERGY_MODEL_INVALID: "+a.id+": "+q.reason)
                        if exhausted(q):
                            hi = mid
                        else:
                            lo = mid
                events.append((hi,a.id,reason))
        if not events:
            return end, ()
        first = min(t for t,_,_ in events)
        simultaneous = {}
        for t,key,reason in sorted(events):
            if abs(t-first) <= 1e-8:
                simultaneous.setdefault(key,reason)
        return first, tuple(sorted(simultaneous.items()))

    def _check_execution(self, end):
        """Check the full physical interval, with exact stable structural proofs."""
        if self._stage2_enabled():
            for key, a in self.aircraft.items():
                if a.status not in AIRBORNE:
                    continue
                plan = self._base(a)
                state = plan.state(self.time)
                if ((key in self.active and a.trajectory_hash != plan.trajectory_hash)
                        or math.dist(a.p,state.p) > 1e-7 or math.dist(a.v,state.v) > 1e-7):
                    raise RuntimeError("AUTHORITATIVE_TRAJECTORY_MISMATCH: "+key)
        bodies = [a for a in self.aircraft.values() if a.status in AIRBORNE or a.status == "LANDED_SERVICE"]
        if len(bodies) < 2:
            return
        bases = {a.id: self._base(a) for a in bodies}
        screened = _execution_distance_screen(bases, self.time, end, self.config)
        query_options = ({"_query_cache": _ExecutionQueryCache(bases.values())}
                         if getattr(self,"cache_enabled",False) else {})
        first = None
        for a,b in combinations(bodies,2):
            x,y = bases[a.id],bases[b.id]
            result = screened.get((a.id,b.id)) or screened.get((b.id,a.id))
            if result is None:
                result = validate_separation(x,y,self.time,end,config=self.config,**query_options)
            upper = result.details.get("upper_bound_m", result.details.get("distance_m"))
            if upper is not None and (self.minimum_separation is None or upper < self.minimum_separation["upper_bound_m"]):
                self.minimum_separation = {"upper_bound_m": upper, "lower_bound_m": result.details.get("lower_bound_m"),
                                           "time": result.details.get("witness_time",(self.time+end)/2), "pair":[a.id,b.id]}
            if result.accepted:
                continue
            if result.status == "UNKNOWN":
                raise RuntimeError(f"EXECUTION_SAFETY_UNKNOWN: {a.id},{b.id}: {result.details}")
            lo,hi = self.time,result.details.get("witness_time",end)
            while hi-lo > .00001:
                mid=(lo+hi)/2
                check=validate_separation(x,y,self.time,mid,config=self.config,**query_options)
                if check.accepted:
                    lo=mid
                else:
                    hi=mid
            if first is None or hi < first[0]:
                first=(hi,a.id,b.id)
        if first:
            self._advance_energy(first[0])
            if not self.failed:
                self._event("hard_separation_pair", first[1], other_id=first[2],
                            distance_m=math.dist(self.aircraft[first[1]].p,self.aircraft[first[2]].p))
                self._fail(first[1],"EXECUTED_HARD_SEPARATION_VIOLATION")

    def _settle(self):
        triggered = False
        # Touchdown/completion precede services/releases at the same timestamp.
        for key, p in sorted(tuple(self.active.items())):
            if p.end_time > self.time:
                continue
            a = self.aircraft[key]
            del self.active[key]
            self.dependencies.pop(p.trajectory_hash, None)
            if p.pad_id is not None:
                service_end = p.end_time+service_duration(self.config, a.service_class)
                self.aircraft[key] = replace(a, status="LANDED_SERVICE", ring=None, slot=None,
                    pad_id=p.pad_id, touchdown_time=p.end_time, service_end=service_end, version=a.version+1,
                    battery_power_w=0.)
                self._event("touchdown", key, soc=a.soc, below_reserve=a.soc < .25,
                            emergency=a.emergency.charged)
            else:
                hold = make_stable_plan(a.id, p.target_ring, p.target_slot, start_time=p.end_time,
                                        end_time=p.end_time+1000, source_version=a.version+1, config=self.config)
                self.trajectories[hold.trajectory_hash] = hold.to_dict()
                self.aircraft[key] = replace(a, status=HOLD[p.target_ring], ring=p.target_ring,
                    slot=p.target_slot, version=a.version+1, trajectory_hash=hold.trajectory_hash)
                self._event("inbound_completed" if p.kind == "INBOUND" else "transfer_completed", key)
            triggered = True
        for h, times in sorted(self._source_clear.items()):
            clear = times[0]
            if abs(clear-self.time) <= 1e-8 and not any(e["type"] == "source_slot_cleared" and e["details"].get("trajectory_hash") == h for e in self.events):
                self._event("source_slot_cleared", trajectory_hash=h)
                triggered = True
        for key, a in sorted(tuple(self.aircraft.items())):
            if a.status == "LANDED_SERVICE" and a.service_end <= self.time+1e-8:
                self.aircraft[key] = replace(a, status="CLEARED", version=a.version+1)
                self._event("pad_service_completed", key)
                triggered = True
        while self._manifest_index < len(self.manifest.requests):
            request = self.manifest.requests[self._manifest_index]
            if request.release_time > self.time+1e-8:
                break
            self._manifest_index += 1
            self.requests[request.id] = request
            self._event("request_released", request.id)
            triggered = True
        if any(p.start_time <= self.time for p in self.soft.values()):
            triggered = True
        return triggered

    def _canonical_reference_duration(self,ring,phase_bin):
        template_key=(ring,phase_bin,self.config.profile_hash)
        if template_key not in self._reference_templates:
            # Slot0/pad0 have relative phase omega*t. This fixed bin-center
            # representative is independent of the first caller's aircraft,
            # time, phase, cache mode, and the order in which bins are visited.
            phase=(phase_bin+.5)*2*math.pi/self.REFERENCE_PHASE_BINS
            representative_time=phase/self.config.airspace.common_angular_speed_radps
            cpu=walltime.perf_counter()
            candidate=generate_final("canonical_reference",ring,0,0,representative_time,
                                     emergency=True,config=self.config)
            self.cpu["reference_template_search"]+=walltime.perf_counter()-cpu
            self.reference_counts["initial_search"]+=1
            duration=(candidate.plan.segments[0].duration if candidate.plan is not None
                      and candidate.status=="FEASIBLE" else None)
            self._reference_templates[template_key]=duration
        return self._reference_templates[template_key]

    def _reference(self, ring, slot, t, version=0):
        key = (ring, slot, float(t), version, self.config.profile_hash)
        if self.cache_enabled and key in self._q_cache:
            self.cache_hits += 1
            return self._q_cache[key]
        choices = []
        for pad in range(6):
            q, plan = self._reference_pad(ring, slot, pad, t, version)
            if q is not None:
                choices.append((q, plan))
        result = min(choices, key=lambda x: x[0]) if choices else (None, None)
        if self.cache_enabled:
            self._store_cache(self._q_cache,key,result,self.Q_CACHE_CAPACITY)
        return result

    def _reference_pad(self, ring, slot, pad, t, version=0):
        phase = (pad*math.pi/3 - math.atan2(*slot_state(ring,slot,t,self.config).p[1::-1])) % (2*math.pi)
        phase_bin=min(self.REFERENCE_PHASE_BINS-1,int(phase/(2*math.pi)*self.REFERENCE_PHASE_BINS))
        duration=self._canonical_reference_duration(ring,phase_bin)
        result=None
        if duration is not None:
            self.reference_counts["template_instantiation"]+=1
            result=generate_final("reference",ring,slot,pad,t,emergency=True,
                                  source_version=version,config=self.config,durations=[duration])
        if result is None or result.plan is None or result.status!="FEASIBLE":
            self.reference_counts["fallback_search"] += 1
            self._last_reference_fallback={"ring":ring,"slot":slot,"query_time":t,"pad_id":pad,
                "phase_bin":phase_bin,"canonical_duration":duration,
                "reason":"canonical template missing or invalid at exact current phase"}
            cpu=walltime.perf_counter()
            result = generate_final("reference", ring, slot, pad, t, emergency=True,
                                    source_version=version, config=self.config)
            self.cpu["reference_fallback_search"]+=walltime.perf_counter()-cpu
        if result.plan is not None and result.status == "FEASIBLE":
            q = self._energy(result.plan, t, result.plan.end_time)
            if q.valid:
                return q.soc_used, result.plan
        return None, None

    def _monitor_reference(self, ring, slot, t, version, soc, qhold):
        """Use sampled interpolation only outside the critical reserve margin.

        No sampled estimate certifies trajectory safety or planned reserve.
        Switches, uncovered cells, excessive sampled residual, and energy-critical
        queries use the unchanged exact reference. Rescue plans pass submit().
        """
        rt = self._practical_runtime
        counts = rt["counters"]
        phase0 = -math.atan2(*slot_state(ring,slot,t,self.config).p[1::-1])
        estimates, uncertain = [], False
        width = 2*math.pi/240
        for pad in range(6):
            phase = (pad*math.pi/3 + phase0) % (2*math.pi)
            cell = min(239, int(phase/width))
            key = f"{ring}:{cell}"
            samples = rt["table"].get(key)
            if samples is None:
                counts["table_misses"] += 1
                uncertain = True  # first uncovered query always uses exact result
                cpu = walltime.perf_counter()
                samples = []
                for fraction in (0., .5, 1.):
                    # Clockwise slot0 angle=-omega*t, relative pad0 phase=+omega*t.
                    sample_t = ((cell+fraction)*width)/self.config.airspace.common_angular_speed_radps
                    q, plan = self._reference_pad(ring, 0, 0, sample_t)
                    samples.append([q, None if plan is None else plan.segments[0].duration])
                    counts["table_samples"] += 1
                rt["table"][key] = samples
                rt["precompute_seconds"] += walltime.perf_counter()-cpu
            values = [row[0] for row in samples]
            durations = [row[1] for row in samples]
            if any(q is None for q in values) or len(set(durations)) != 1:
                uncertain = True
                continue
            residual = abs(values[1]-(values[0]+values[2])/2)
            if residual > 2e-5:
                uncertain = True
                continue
            u = (phase-cell*width)/width
            j = 0 if u <= .5 else 1
            fraction = 2*u-j
            estimates.append(values[j] + fraction*(values[j+1]-values[j]))
        q = min(estimates) if len(estimates) == 6 else None
        # The sampled residual is an explicit allowance, not a certified global
        # bound. Critical decisions always request the original exact routine.
        critical = q is None or soc > .35 or soc-q-qhold-.25 <= 2e-5 + .002
        if uncertain or critical:
            counts["table_uncertain"] += 1
            counts["exact_reference_calls"] += 1
            return self._reference(ring, slot, t, version)[0]
        counts["table_hits"] += 1
        return q

    def _search_rescue_opportunity(self, a):
        """Bounded exact candidate search; exhaustion means UNKNOWN, never impossible.

        Pads receive one duration alternative each before the next alternative.
        Every returned opportunity passed the entire replacement transaction;
        current execution still revalidates it through submit().
        """
        if a.id in self.active or a.status not in HOLD.values():
            return None
        snapshot = self.build_planning_snapshot()
        cache = getattr(self, "_rescue_search_cache", None)
        if cache is None:
            self._rescue_search_cache = cache = OrderedDict()
        cache_key = (snapshot.version, a.id)
        if cache_key in cache:
            return cache[cache_key]
        visible = dict(snapshot.aircraft)
        visible[a.id] = replace(a, emergency=replace(a.emergency, eligible=True))
        probe = replace(snapshot, aircraft=MappingProxyType(visible))
        grid = tuple(range(20, 301, 5))
        next_duration = {pad: 20. for pad in range(6)}
        for alternative in range(RESCUE_DURATION_ALTERNATIVES_V6):
            for pad in range(6):
                first = generate_final(a.id, a.ring, a.slot, pad, self.time,
                    emergency=True, source_version=a.version, config=self.config,
                    durations=[d for d in grid if d >= next_duration[pad]])
                if first.plan is None or first.status != "FEASIBLE":
                    next_duration[pad] = 305.
                    continue
                duration = first.plan.segments[0].duration
                next_duration[pad] = duration+5.
                releases = sorted({start for r in self.pads.reservations
                    if r.resource == pad and math.isfinite(r.end)
                    for start in (r.end-duration,r.end)
                    if self.time < start <= self.time+RESCUE_LOOKAHEAD_SECONDS_V6})[:2]
                for start in (self.time, *releases):
                    candidate = first if start == self.time else generate_final(a.id,
                        a.ring, a.slot, pad, start, emergency=True, source_version=a.version,
                        config=self.config, durations=[d for d in grid if d >= duration])
                    if candidate.plan is None or candidate.status != "FEASIBLE":
                        continue
                    p = candidate.plan
                    checked = evaluate_planning_snapshot(probe, (p,))
                    self._event("rescue_opportunity_attempt", a.id, pad_id=pad,
                        start_time=start, trajectory_hash=p.trajectory_hash,
                        end_time=p.end_time, pad_window_start=p.end_time-13.5,
                        pad_window_end=p.end_time+service_duration(self.config,a.service_class),
                        alternative=alternative, status=checked.status,
                        reason=checked.reason, validation=thaw(checked.details))
                    if not checked.accepted:
                        continue
                    # Match the energy resolution certified by the projection.
                    fine = checked._prepared[-1].get(p.trajectory_hash, False)
                    q = self._energy(p, start, p.end_time, fine=fine)
                    opportunity = (p, q.soc_used, start-self.time)
                    self._store_cache(cache, cache_key, opportunity, 32)
                    return opportunity
        self._event("rescue_opportunity_search_exhausted", a.id, status="UNKNOWN",
                    reason="bounded_search_exhausted_no_infeasibility_proof")
        self._store_cache(cache, cache_key, None, 32)
        return None

    def _update_emergency_v6(self):
        triggered = False
        monitor = getattr(self, "_rescue_monitor_v6", None)
        if monitor is None:
            self._rescue_monitor_v6 = monitor = {}
        for key, a in sorted(tuple(self.aircraft.items())):
            if a.status not in AIRBORNE or a.emergency.declared or a.status.startswith("FINAL") or a.status == "VERTICAL_DESCENT":
                continue
            current = self.active.get(key)
            signature = (a.version, a.status, None if current is None else current.trajectory_hash)
            previous = monitor.get(key)
            if previous is not None and previous[1] == signature and self.time-previous[0] < MONITOR_SECONDS_V6-1e-8:
                continue
            monitor[key] = (self.time, signature)
            if current is not None:
                ring, slot, t = current.target_ring, current.target_slot, current.end_time
                soc = a.soc-self._energy(current, self.time, t).soc_used
            else:
                ring, slot, t, soc = a.ring, a.slot, self.time, a.soc
            rate = stable_power(ring, self.config).current_a / (self.config.energy.reference_capacity_Ah*3600.)
            # Physics monitoring uses necessary reserve bounds. Exact opportunity
            # search runs at dispatch boundaries in handle_emergencies().
            q, wait = a.emergency.q_e, a.emergency.rescue_wait_seconds or 0.
            assessed = assess_rescue(soc, rate, profile=self.rescue_rule_profile,
                rescue_soc=q, rescue_wait_seconds=wait,
                normal_energy_lower_bound=normal_energy_lower_bound(self.config, ring),
                rescue_feasible=False, declared=a.emergency.declared,
                pending=a.emergency.pending, **self._normal_estimate(a,current))
            old = a.emergency
            required = assessed.rescue_required or old.eligible
            e = replace(old, warning=assessed.warning, normal_status=assessed.normal_status,
                normal_margin=assessed.normal_margin, q_e=q,
                rescue_wait_seconds=a.emergency.rescue_wait_seconds,
                rescue_slack_seconds=assessed.rescue_slack_seconds,
                rescue_reason=assessed.rescue_reason or old.rescue_reason,
                eligible=required and current is None, pending=required and current is not None)
            self.aircraft[key] = replace(a, emergency=e)
            if assessed.warning and not old.warning:
                self._event("rescue_warning", key, normal_status=assessed.normal_status,
                            normal_margin=assessed.normal_margin, soc=soc)
            if required and not (old.eligible or old.pending):
                self._event("emergency_triggered", key, pending=current is not None,
                    q_e=q, reason=assessed.rescue_reason,
                    rescue_wait_seconds=wait, rescue_slack_seconds=assessed.rescue_slack_seconds,
                    **(dict(deadline_basis="necessary_normal_energy_bound",
                            assessment_ring=ring,assessment_time=t,assessment_soc=soc)
                       if self._rescue_need_v7() else {}))
                triggered = True
            if self._rescue_need_v7() and required:
                self._declare_rescue_need(key, pending=current is not None, ring=ring)
        return triggered

    def _update_emergency(self):
        if self._rescue_v6():
            start_cpu = walltime.perf_counter()
            result = self._update_emergency_v6()
            self.cpu["emergency"] += walltime.perf_counter()-start_cpu
            return result
        triggered = False
        start_cpu = walltime.perf_counter()
        practical = self._runtime_enabled()
        rt = getattr(self, "_practical_runtime", {})
        for key, a in sorted(tuple(self.aircraft.items())):
            if a.status not in AIRBORNE or a.emergency.declared or a.status.startswith("FINAL") or a.status == "VERTICAL_DESCENT":
                continue
            p = self.active.get(key)
            if p is not None and p.pad_id is not None:
                continue
            known_imminent = False
            force_exact = practical and a.emergency.reference_missing
            if practical:
                hint = rt["imminent"].get(key)
                query_ring = a.ring if p is None else p.target_ring
                query_slot = a.slot if p is None else p.target_slot
                known_imminent = bool(hint and a.version <= hint["version"]
                    and query_ring == hint["ring"] and query_slot == hint["slot"]
                    and self.time >= hint["time"]-self.physics_dt-1e-8)
                counts = rt["counters"]
                counts["monitor_calls"] += 1
                signature = (a.version, a.status, None if p is None else p.trajectory_hash)
                previous = rt["monitor"].get(key)
                urgent = (known_imminent or a.soc <= rt["soc_threshold"] or a.emergency.eligible or a.emergency.pending
                          or a.emergency.reference_missing)
                if p is not None:
                    # Accepted plans cannot exceed this frozen current ceiling.
                    # Subtracting an upper bound gives a conservative SOC floor.
                    worst = (p.end_time-self.time)*self.config.energy.reference_max_battery_current_A/(self.config.energy.reference_capacity_Ah*3600)
                    urgent = urgent or a.soc-worst <= rt["soc_threshold"]
                event_now = bool(self.events and self.events[-1]["time"] == self.time)
                if (not urgent and not event_now and previous is not None and previous[1] == signature
                        and self.time-previous[0] < rt["emergency_period"]-1e-8):
                    counts["monitor_throttled"] += 1
                    continue
                rt["monitor"][key] = (self.time, signature)
                if not (known_imminent or force_exact) and a.soc > rt["soc_threshold"] and (p is None or a.soc-worst > rt["soc_threshold"]):
                    counts["soc_gate_skips"] += 1
                    if p is not None:
                        counts["projected_bound_skips"] += 1
                    continue
            if p is not None:
                ring, slot, t = p.target_ring, p.target_slot, p.end_time
                soc = a.soc-self._energy(p, self.time, t).soc_used
            else:
                ring, slot, t, soc = a.ring, a.slot, self.time, a.soc
            if practical and not (known_imminent or force_exact) and soc > rt["soc_threshold"]:
                rt["counters"]["soc_gate_skips"] += 1
                continue
            hold = make_stable_plan("reference_hold", ring, slot, start_time=t, end_time=t+10, config=self.config)
            qhold = self._energy(hold, t, t+10).soc_used
            if practical and (known_imminent or force_exact):
                if known_imminent:
                    rt["counters"]["imminent_hint_checks"] += 1
                rt["counters"]["exact_reference_calls"] += 1
                q, _ = self._reference(ring, slot, t, a.version)
            elif practical:
                q = self._monitor_reference(ring, slot, t, a.version, soc, qhold)
            else:
                q, _ = self._reference(ring, slot, t, a.version)
            alert = emergency_threshold(soc, q, qhold)
            old = a.emergency
            e = replace(old, q_e=q, reference_missing=q is None)
            if q is None and not old.reference_missing:
                self._event("rescue_reference_missing", key, ring=ring)
            if alert:
                e = replace(e, pending=p is not None, eligible=p is None)
                if not old.pending and not old.eligible:
                    self._event("emergency_triggered", key, pending=p is not None, q_e=q)
                    triggered = True
            self.aircraft[key] = replace(a, emergency=e)
        self.cpu["emergency"] += walltime.perf_counter()-start_cpu
        return triggered

    def handle_emergencies(self):
        """Test-only deterministic rescue interface; all attempts pass submit()."""
        if self._rescue_v6():
            candidates = sorted((a for a in self.aircraft.values()
                if a.status in HOLD.values() and a.id not in self.active and
                (a.soc <= .35 or a.emergency.warning or a.emergency.eligible or a.emergency.pending or a.emergency.declared)),
                key=lambda a: emergency_priority(a, self.config)["key"])
            required_opportunities = []
            for a in candidates:
                if a.id in self.soft and self.soft[a.id].kind == "FINAL_EMERGENCY":
                    continue
                opportunity = self._search_rescue_opportunity(a)
                rate = emergency_priority(a, self.config)["holding_soc_per_second"]
                q, wait = (opportunity[1], opportunity[2]) if opportunity else (None, 0.)
                assessed = assess_rescue(a.soc, rate, profile=self.rescue_rule_profile,
                    normal_energy_lower_bound=normal_energy_lower_bound(self.config, a.ring),
                    rescue_soc=q, rescue_wait_seconds=wait, rescue_feasible=opportunity is not None,
                    declared=a.emergency.declared, pending=a.emergency.pending, **self._normal_estimate(a))
                old = a.emergency
                required = assessed.rescue_required or old.eligible
                e = replace(old, warning=assessed.warning, normal_status=assessed.normal_status,
                    normal_margin=assessed.normal_margin, q_e=q, reference_missing=opportunity is None,
                    rescue_wait_seconds=wait if opportunity else None,
                    rescue_slack_seconds=assessed.rescue_slack_seconds,
                    rescue_reason=old.rescue_reason if self._rescue_need_v7() and old.declared else assessed.rescue_reason or old.rescue_reason,
                    eligible=required, pending=False if self._rescue_need_v7() else old.pending)
                self.aircraft[a.id] = replace(a, emergency=e)
                if assessed.warning and not old.warning:
                    self._event("rescue_warning", a.id, normal_status=assessed.normal_status,
                                normal_margin=assessed.normal_margin, soc=a.soc)
                if required and not (old.eligible or old.pending or old.declared):
                    self._event("emergency_triggered", a.id, pending=False,
                        q_e=q, reason=assessed.rescue_reason, rescue_wait_seconds=wait,
                        rescue_slack_seconds=assessed.rescue_slack_seconds,
                        deadline_basis=("necessary_normal_energy_bound"
                            if self._rescue_need_v7() and assessed.rescue_reason == "NORMAL_RESERVE_IMPOSSIBLE"
                            else "certified_opportunity_estimate"))
                if not required:
                    continue
                if self._rescue_need_v7() and not old.declared:
                    self._declare_rescue_need(a.id, trajectory_hash=opportunity[0].trajectory_hash if opportunity else None)
                    e = self.aircraft[a.id].emergency
                if not old.declared and not self._rescue_need_v7():
                    # Required disposition is already a rescue takeover, even
                    # if a preceding rescue later consumes this opportunity.
                    # Declaration counts now; start and charge still require
                    # the actual maneuver to execute.
                    e = replace(e,declared=True,declaration_time=self.time,
                        declaration_ring=a.ring,state="EMERGENCY_WAITING",
                        rescue_reason=e.rescue_reason or "EXISTING_RESCUE_ELIGIBILITY")
                    self.aircraft[a.id] = replace(self.aircraft[a.id],emergency=e)
                    self._event("emergency_declared",a.id,ring=a.ring,
                                reason=e.rescue_reason,
                                trajectory_hash=opportunity[0].trajectory_hash if opportunity else None)
                priority = emergency_priority(self.aircraft[a.id], self.config)
                urgency = (assessed.rescue_slack_seconds if assessed.rescue_slack_seconds is not None
                           else priority["holding_seconds"])
                fresh_key = (urgency, assessed.normal_margin if assessed.normal_margin is not None else math.inf,
                             *priority["key"][1:])
                required_opportunities.append((fresh_key, a.id, opportunity))
            # All ordering uses this dispatch's exact opportunities, including
            # their future hold costs. Probing itself protects no resources.
            for fresh_key, key, opportunity in sorted(required_opportunities):
                a = self.aircraft[key]
                self._event("emergency_priority_selected", key,
                            fresh_key=tuple(None if isinstance(v,float) and not math.isfinite(v) else v for v in fresh_key),
                            rescue_wait_seconds=a.emergency.rescue_wait_seconds,
                            rescue_slack_seconds=a.emergency.rescue_slack_seconds,
                            normal_margin=a.emergency.normal_margin,
                            **emergency_priority(a, self.config))
                for attempt in range(2):
                    if opportunity is None:
                        break
                    result = self.submit((opportunity[0],))
                    if result.accepted:
                        p = opportunity[0]
                        self._event("rescue_opportunity_protected", a.id, pad_id=p.pad_id,
                            trajectory_hash=p.trajectory_hash, start_time=p.start_time,
                            end_time=p.end_time, pad_window_start=p.end_time-13.5,
                            pad_window_end=p.end_time+service_duration(self.config,a.service_class),
                            **({"opportunity_status": "PROTECTED"} if self._rescue_need_v7() else {}))
                        break
                    self._event("rescue_commitment_protected", a.id, reason=result.reason,
                                validation=result.details)
                    # A preceding rescue may have changed calendars/traffic.
                    # One fresh bounded search replaces a stale opportunity;
                    # UNKNOWN retains all existing commitments.
                    opportunity = self._search_rescue_opportunity(self.aircraft[key]) if attempt == 0 else None
                # Preserve every existing commitment when a safe replacement
                # could not be certified. No standalone speculative cancellation.
            return
        if self._stage2_enabled():
            queue = self._emergency_waiting_queue()
            if queue:
                self._event("emergency_priority_queue", queue=[dict(aircraft_id=a.id,
                            **emergency_priority(a, self.config)) for a in queue])
                self._yield_normal_finals()
            keys = [a.id for a in queue]
        else:
            keys = sorted(self.aircraft)
        for key in keys:
            a = self.aircraft[key]
            if (self._stage2_enabled() and a.status in HOLD.values() and key not in self.active
                    and not a.emergency.eligible):
                a = replace(a, emergency=replace(a.emergency, eligible=True))
                self.aircraft[key] = a
                if not a.emergency.declared:
                    self._event("emergency_triggered", key, pending=False, q_e=a.emergency.q_e,
                                trigger="revealed_threshold_priority")
            if not (a.emergency.eligible or (self._stage2_enabled() and a.emergency.declared)) or key in self.active or a.status not in HOLD.values():
                continue
            if self._stage2_enabled():
                self._event("emergency_priority_selected", key, **emergency_priority(a, self.config))
            if a.ring == "inner" and not a.emergency.declared:
                if self._try_final(a, emergency=False):
                    continue
            if not a.emergency.declared:
                e = replace(a.emergency, declared=True, declaration_time=self.time,
                            declaration_ring=a.ring, state="EMERGENCY_WAITING")
                a = replace(a, emergency=e)
                self.aircraft[key] = a
                self._event("emergency_declared", key, ring=a.ring)
            if self._try_final(a, emergency=True):
                continue
            # If rescue cannot replace a scheduled ordinary maneuver, retain
            # holding only when the whole traffic batch still certifies safe.
            old = self.soft.get(key)
            if old is not None and old.kind != "FINAL_EMERGENCY":
                result = self.cancel(key)
                if not result.accepted:
                    self._emergency_conflict(key, result)
            a = self.aircraft[key]
            self.aircraft[key] = replace(a, emergency=replace(a.emergency, state="EMERGENCY_WAITING"))
            # A rescue may depart before a pad is free. Waiting until its release
            # is not a mandatory energy cost and cannot prove a missed deadline.

    def _emergency_conflict(self, aircraft_id, result):
        old = self.soft.get(aircraft_id)
        dependencies = sorted(h for h, ds in self.dependencies.items()
                              if old is not None and old.trajectory_hash in ds)
        details = {"trajectory_hash": old.trajectory_hash if old else None,
                   "dependent_started_plans": dependencies, "validation": result.details}
        self._event("emergency_dependency_conflict", aircraft_id, reason=result.reason, **details)
        self._record(force=True)
        raise EmergencyDependencyConflict(aircraft_id, result.reason, details)

    def _try_final(self, a, emergency):
        if emergency and self._rescue_v6():
            opportunity = self._search_rescue_opportunity(a)
            return bool(opportunity is not None and self.submit((opportunity[0],)).accepted)
        for pad in range(6):
            result = generate_final(a.id, a.ring, a.slot, pad, self.time,
                                    emergency=emergency, source_version=a.version, config=self.config)
            if result.plan is None or result.status != "FEASIBLE":
                self._reject(result.reason or result.status, a.id, pad_id=pad, candidate=True)
                continue
            before = len(self.rejections)
            accepted = self.submit((result.plan,))
            if accepted.accepted:
                return True
            self._event("final_attempt_rejected", a.id, pad_id=pad,
                        trajectory_hash=result.plan.trajectory_hash, emergency=emergency,
                        reason=accepted.reason, validation=accepted.details,
                        rejection_indices=list(range(before, len(self.rejections))))
        return False

    def try_final(self,aircraft,emergency=False):
        """Public deterministic-driver attempt; final submission remains joint."""
        return self._try_final(aircraft,emergency)

    def _prove_pad_deadline(self, a):
        """Legacy hook retained for callers; the old departure-time proof is invalid."""
        return None

    def advance(self, until, decision_callback=None, *, settle_only_at_end=None):
        if settle_only_at_end is None:
            settle_only_at_end = self._stage2_enabled()
        try:
            return self._advance(until, decision_callback, settle_only_at_end=settle_only_at_end)
        except (RuntimeError, ValueError, ArithmeticError) as exc:
            if self._stage2_enabled() and not self.terminated_invalid:
                self._fail(None, str(exc))
            raise

    def _advance(self, until, decision_callback=None, *, settle_only_at_end=False):
        if until < self.time:
            raise ValueError("time cannot run backward")
        callback = self.decision_callback if decision_callback is None else decision_callback
        while not self.failed:
            triggers = self._settle()
            if settle_only_at_end and 0. < until-self.time <= 1e-9:
                # Finish a positive roundoff-sized physical interval before
                # settling the requested boundary. It is not a fresh dispatch.
                if self._stage2_enabled():
                    self._pending_dispatch = self._pending_dispatch or triggers
                next_time = until
                if self._stage2_enabled():
                    next_time, _ = self._energy_failure_events(next_time)
                self._check_execution(next_time)
                if not self.failed:
                    self._advance_energy(next_time)
                self._record()
                continue
            if settle_only_at_end and self.time >= until:
                if self._stage2_enabled():
                    self._pending_dispatch = self._pending_dispatch or triggers
                break
            monitor_event_count = len(self.events)
            monitor_wall, monitor_cpu = walltime.perf_counter(), walltime.process_time()
            triggers = self._update_emergency() or triggers
            monitor_wall, monitor_cpu = walltime.perf_counter()-monitor_wall, walltime.process_time()-monitor_cpu
            if self._stage2_enabled():
                pending_wall, pending_cpu = self._monitor_timing_pending
                self._monitor_timing_pending = pending_wall+monitor_wall, pending_cpu+monitor_cpu
                triggers = self._pending_dispatch or triggers
            heartbeat = self._next_heartbeat <= self.time+1e-8
            if heartbeat:
                self._next_heartbeat = (math.floor((self.time+1e-8)/self.scheduler_period)+1)*self.scheduler_period
            if triggers or heartbeat or not self._initial_settled:
                self._initial_settled = True
                self.decision_times.append(self.time)
                self._record(force=True)
                cpu = walltime.perf_counter()
                process_cpu = walltime.process_time()
                emergency_wall, emergency_cpu = walltime.perf_counter(), walltime.process_time()
                if self.auto_emergency:
                    self.handle_emergencies()
                emergency_wall = walltime.perf_counter()-emergency_wall
                emergency_cpu = walltime.process_time()-emergency_cpu
                admission_wall, admission_cpu = walltime.perf_counter(), walltime.process_time()
                admission = getattr(self,"admission_callback",None)
                if admission:
                    admission(self.snapshot(), self._scheduler_port)
                admission_wall = walltime.perf_counter()-admission_wall
                admission_cpu = walltime.process_time()-admission_cpu
                normal_wall, normal_cpu = walltime.perf_counter(), walltime.process_time()
                if callback and callback is not False:
                    callback(self.snapshot(), self._scheduler_port)
                self._start_due()
                normal_wall = walltime.perf_counter()-normal_wall
                normal_cpu = walltime.process_time()-normal_cpu
                if self._stage2_enabled():
                    mw, mc = self._monitor_timing_pending
                    self.dispatch_timings.append(dict(time=self.time,
                        emergency_wall_s=mw+emergency_wall, emergency_cpu_s=mc+emergency_cpu,
                        admission_wall_s=admission_wall, admission_cpu_s=admission_cpu,
                        normal_wall_s=normal_wall, normal_cpu_s=normal_cpu,
                        dispatch_wall_s=mw+walltime.perf_counter()-cpu,
                        dispatch_cpu_s=mc+walltime.process_time()-process_cpu))
                    self._monitor_timing_pending = (0.,0.)
                    self._pending_dispatch = False
                self.cpu["decision"] += walltime.perf_counter()-cpu
                self._record(force=True)
            elif len(self.events) > monitor_event_count:
                # Monitor-only events need the exact pre-advance physical frame.
                # Dispatch already records the final state at this timestamp.
                self._record(force=True)
            descent_starts = [self._descent_start(p) for p in self.active.values()]
            descent_starts = [t for t in descent_starts if t is not None]
            active_ends = [p.end_time for p in self.active.values()]
            soft_starts = [p.start_time for p in self.soft.values()]
            # A positive phase interval must reach its exact junction, even if
            # accumulated step roundoff is smaller than the usual stop tolerance.
            if (self.time >= until-1e-9
                    and not any(self.time < t <= until for t in descent_starts+active_ends+soft_starts)):
                break
            next_times = [until, self.time+self.physics_dt, self._next_heartbeat]
            if self._runtime_enabled():
                period = self._practical_runtime["telemetry_period"]
                next_times.append((math.floor((self.time+1e-8)/period)+1)*period)
            if self._manifest_index < len(self.manifest.requests):
                next_times.append(self.manifest.requests[self._manifest_index].release_time)
            next_times.extend(t for t in active_ends if t > self.time)
            next_times.extend(t for t in descent_starts if t > self.time)
            next_times.extend(t for t in soft_starts if t > self.time)
            next_times.extend(a.service_end for a in self.aircraft.values() if a.status == "LANDED_SERVICE" and a.service_end > self.time+1e-8)
            next_times.extend(x[0] for x in self._source_clear.values() if x[0] > self.time+1e-8)
            next_time = min(next_times)
            if self._stage2_enabled():
                next_time, _ = self._energy_failure_events(next_time)
            self._check_execution(next_time)
            if not self.failed:
                self._advance_energy(next_time)
            self._record()
        self._record(force=True)
        self.flush_reference_table()
        return self.snapshot()

    def _outer_capacity(self):
        """Only perpetual reservations without any recorded release prove capacity."""
        evidence = []
        for slot in range(37):
            reservations = [r for r in self.slots.reservations if r.resource == ("outer", slot)
                            and r.end > self.time]
            if not reservations or any(math.isfinite(r.end) for r in reservations):
                return None
            if any(r.aircraft_id not in self.aircraft or self.aircraft[r.aircraft_id].status not in AIRBORNE
                   for r in reservations):
                return None
            evidence.append({"slot": slot, "reservations": [r.to_dict() for r in reservations]})
        return evidence

    def _wait_upstream(self, request_id, attempts, reason="NO_PLAN_FOUND"):
        categories = sorted({a["reason"] for a in attempts})
        self.upstream_reasons[request_id] = {"reason": reason, "reason_categories": categories,
                                             "last_attempt_time": self.time}
        self._event("admission_waiting", request_id, reason=reason, reason_categories=categories,
                    attempts=attempts, policy_id=POLICY_ID)
        return ValidationResult(False, "WAITING", reason,
                                {"attempts": attempts, "reason_categories": categories})

    def reroute(self, request_id, reason="OUTER_CAPACITY"):
        if self.failed:
            return self._reject("EPISODE_STOPPED", request_id)
        if request_id not in self.requests or request_id in self.aircraft:
            return self._reject("INVALID_REROUTE", request_id)
        capacity = self._outer_capacity()
        if capacity is None:
            return self._reject("REROUTE_REQUIRES_OUTER_CAPACITY", request_id)
        r = self.requests[request_id]
        self.aircraft[r.id] = AircraftState(r.id, "REROUTED", r.initial_soc, (), (), (), (),
                                           r.service_class, admitted=False, release_time=r.release_time,
                                           upstream_wait_seconds=max(0., self.time-r.release_time))
        self.costs["reroute"] += 600
        self.upstream_reasons.pop(r.id, None)
        self._event("rerouted", r.id, reason="OUTER_CAPACITY", capacity_evidence=capacity,
                    upstream_wait_seconds=max(0., self.time-r.release_time), policy_id=POLICY_ID)
        return self._result(True, "OUTER_CAPACITY", capacity_evidence=capacity)

    def _freeflow_screen(self, inbound, initial_soc):
        """Construct one complete single-aircraft normal reference; no fleet certificate."""
        portions = [(inbound, inbound.start_time, inbound.end_time)]
        ring, slot, t = "outer", inbound.target_slot, inbound.end_time
        for target_ring, count in (("middle", 28), ("inner", 18)):
            found = None
            for target in range(count):
                candidate = generate_transfer("freeflow", ring, slot, target, t, config=self.config)
                if candidate.plan is not None and candidate.status == "FEASIBLE":
                    q = self._energy(candidate.plan, t, candidate.plan.end_time)
                    if q.valid:
                        found = candidate.plan, q
                        break
            if found is None:
                return self._result(False, "NO_PLAN_FOUND", stage="freeflow_reference")
            p, q = found
            portions.append((p,p.start_time,p.end_time))
            ring, slot, t = target_ring, p.target_slot, p.end_time
        for pad in range(6):
            candidate = generate_final("freeflow", ring, slot, pad, t, config=self.config)
            if candidate.plan is not None and candidate.status == "FEASIBLE":
                p = candidate.plan
                q = self._sum_energy(self._budget_energy(None, (*portions,(p,t,p.end_time)),
                                                       soc=initial_soc,reserve=True))
                if q.valid and initial_soc-q.soc_used >= .25 and q.peak_seconds <= 120:
                    return self._result(True)
        return self._result(False, "INITIAL_ENERGY_SCREEN", stage="freeflow_reference")

    def admit(self, request_id, plan=None):
        if self.failed:
            return self._reject("EPISODE_STOPPED", request_id)
        r = self.requests.get(request_id)
        if r is None or request_id in self.aircraft:
            return self._reject("INVALID_REQUEST", request_id)
        if r.release_time > self.time+1e-8:
            return self._reject("REQUEST_NOT_RELEASED", request_id)
        earlier = [q for q in self.requests.values() if q.gate_id == r.gate_id and q.id not in self.aircraft
                   and (q.release_time, q.id) < (r.release_time, r.id)]
        if earlier:
            return self._wait_upstream(r.id, [], "UPSTREAM_FIFO_WAIT")
        if self._outer_capacity() is not None:
            rerouted = self.reroute(r.id)
            return ValidationResult(False, "REROUTED", "OUTER_CAPACITY", rerouted.details)
        attempts = []
        def rejected(reason, p, stage, **details):
            evidence = {"slot": p.target_slot, "trajectory_hash": p.trajectory_hash,
                        "stage": stage, "reason": reason, **details}
            attempts.append(evidence)
            return self._reject(reason, r.id, **{k: v for k, v in evidence.items() if k != "reason"})
        candidates = [plan] if plan is not None else None
        if candidates is None:
            candidates = []
            for slot in range(37):
                c = generate_inbound(r.id, r.gate_id, slot, self.time, config=self.config)
                if c.plan is not None and c.status == "FEASIBLE":
                    candidates.append(c.plan)
                else:
                    failures = c.details.get("attempts", ())
                    reason_counts = {}
                    for failure in failures:
                        why = failure.get("reason", "UNKNOWN")
                        reason_counts[why] = reason_counts.get(why, 0)+1
                    summary = {"status": c.status, "attempt_count": len(failures),
                               "reason_counts": reason_counts,
                               "duration_grid": {"start": 20, "stop_inclusive": 180, "step": 5},
                               "phase_options_per_duration": 2}
                    evidence = {"slot": slot, "trajectory_hash": None, "stage": "generation",
                                "reason": c.reason or c.status, "validation": summary}
                    attempts.append(evidence)
                    self._reject(evidence["reason"], r.id, slot=slot, stage="generation", validation=summary)
        last = self._result(False, "NO_PLAN_FOUND")
        for p in candidates:
            if p.aircraft_id != r.id or p.kind != "INBOUND" or p.start_time != self.time or p.target_ring != "outer":
                last = rejected("INVALID_INBOUND", p, "input")
                continue
            theta = r.gate_id*math.pi/2
            s0 = p.state(self.time)
            expected_p = (900*math.cos(theta),900*math.sin(theta),110.)
            expected_v = (21*(-math.cos(theta)+math.sin(theta))/math.sqrt(2),
                          21*(-math.sin(theta)-math.cos(theta))/math.sqrt(2),0.)
            if math.dist(s0.p,expected_p) > .01 or math.dist(s0.v,expected_v) > .01 or math.dist(s0.a,(0.,0.,0.)) > .02:
                last = rejected("INVALID_ENTRY_STATE", p, "entry_state")
                continue
            result = validate_plan(p, config=self.config)
            if not result.accepted:
                last = rejected(result.reason, p, "plan", validation=result.details)
                continue
            q = self._sum_energy(self._budget_energy(None, ((p,p.start_time,p.end_time),),soc=r.initial_soc))
            if not q.valid or r.initial_soc-q.soc_used <= 0 or q.peak_seconds > 120:
                last = rejected("SOC_INSUFFICIENT", p, "inbound_energy", energy=asdict(q))
                continue
            screen = self._freeflow_screen(p, r.initial_soc)
            if not screen.accepted:
                last = rejected(screen.reason, p, "freeflow_reference", validation=screen.details)
                continue
            s = p.state(self.time)
            a = AircraftState(r.id, "INBOUND", r.initial_soc, s.p, s.v, s.a, s.jerk,
                               r.service_class, release_time=r.release_time, trajectory_hash=p.trajectory_hash,
                               admission_time=self.time, upstream_wait_seconds=max(0., self.time-r.release_time),
                              battery_power_w=evaluate_power(s.v,s.a,self.config).battery_w)
            traffic = _Traffic(a.id, p, (), self.time, math.inf)
            conflict = None
            conflict_pair = None
            horizon = max([p.end_time, *(x.end_time for x in self.active.values()),
                           *(x.end_time for x in self.soft.values())])+2*math.pi/.035
            for b in self.aircraft.values():
                if b.status not in AIRBORNE and b.status != "LANDED_SERVICE":
                    continue
                other = self._traffic(b)
                result = validate_separation(traffic, other, self.time, min(horizon, other.end_time), config=self.config)
                if not result.accepted:
                    conflict = result
                    conflict_pair = [r.id, b.id]
                    break
            if conflict:
                last = rejected(conflict.reason, p, "traffic", pair=conflict_pair, validation=conflict.details)
                continue
            binding = self._release_supports("outer", p.target_slot, r.id, self.slots, self.soft)
            witness = self._future_dependency_witness(binding, self.soft, self.slots)
            if witness is not None:
                last = rejected("FUTURE_EMERGENCY_DISPOSITION", p, "future_emergency", **witness)
                continue
            self.aircraft[r.id] = a
            self.active[r.id] = p
            try:
                slots, pads, times = self._calendar_stage(self.soft, [p])
            except (ResourceConflict, ValueError) as exc:
                del self.aircraft[r.id]
                del self.active[r.id]
                last = rejected("SLOT_CONFLICT", p, "resources", detail=str(exc))
                continue
            self.slots, self.pads, self._source_clear = slots, pads, times
            self.trajectories[p.trajectory_hash] = p.to_dict()
            deps = self._release_supports("outer", p.target_slot, r.id, slots, self.soft)
            self.dependencies[p.trajectory_hash] = tuple(sorted(deps))
            self.upstream_reasons.pop(r.id, None)
            self._event("admitted", r.id, trajectory_hash=p.trajectory_hash, dependencies=sorted(deps),
                        release_time=r.release_time, admission_time=self.time,
                        upstream_wait_seconds=max(0., self.time-r.release_time), policy_id=POLICY_ID,
                        prior_rejections=attempts)
            return self._result(True)
        return self._wait_upstream(r.id, attempts)

    def _scheduler_metrics(self):
        return {key:value for key,value in self.metrics().items()
                if key not in ("planned_manifest","unreleased")}

    def metrics(self):
        """Full audit metrics; schedulers receive the redacted port view."""
        values = list(self.aircraft.values())
        admitted = [a for a in values if a.admitted]
        landed = [a for a in admitted if a.touchdown_time is not None]
        failed = [a for a in admitted if a.status in ("ENERGY_FAILURE", "EXECUTED_HARD_SEPARATION_VIOLATION")]
        rerouted = sum(a.status == "REROUTED" for a in values)
        offered = self._manifest_index + sum(a.fixture for a in values)
        airborne = len(admitted)-len(landed)-len(failed)
        normal = sum(not a.emergency.charged for a in landed)
        emergency = len(landed)-normal
        upstream = self._upstream_rows()
        wait = sum(a.upstream_wait_seconds for a in values)+sum(r["wait_seconds"] for r in upstream)
        result = {"policy_id": POLICY_ID, "planned_manifest": len(self.manifest.requests), "offered": offered,
                "manifest_offered": self._manifest_index, "fixture_count": sum(a.fixture for a in values),
                "unreleased": len(self.manifest.requests)-self._manifest_index,
                "admitted": len(admitted), "rerouted": rerouted, "physically_landed": len(landed),
                "upstream_waiting": len(upstream), "censored_upstream": len(upstream) if self.failed else 0,
                "upstream_wait_time": wait, "cost_upstream_wait": wait,
                "normal_landed": normal, "emergency_landed": emergency,
                "landing_below_reserve": sum(a.soc < .25 for a in landed),
                "failed_before_landing": len(failed), "airborne_or_censored": airborne,
                "energy_failures": sum(a.status == "ENERGY_FAILURE" for a in values),
                "censored": sum(a.status == "CENSORED_DUE_TO_FAILURE" for a in values),
                "pad_deadline_missed": sum(a.emergency.deadline_missed for a in values),
                "emergency_declared": sum(a.emergency.declared for a in values),
                "emergency_unresolved": sum(a.emergency.declared and a.touchdown_time is None and a.status not in ("ENERGY_FAILURE", "EXECUTED_HARD_SEPARATION_VIOLATION") for a in values),
                "energy_j": sum(a.energy_j for a in values),
                "energy_kwh": sum(a.energy_j for a in values)/3600000,
                "airborne_time": sum(a.airborne_time for a in values),
                "holding_time": sum(a.holding_time for a in values),
                "physical_airborne_count": sum(a.status in AIRBORNE for a in values),
                "slot_reservations_active": sum(r.start <= self.time < r.end for r in self.slots.reservations),
                "pad_utilization_seconds": sum(self.pad_busy_seconds.values()),
                "pad_busy_seconds": dict(self.pad_busy_seconds),
                "decision_count": len(self.decision_times), "cost_reroute": self.costs["reroute"],
                "cost_emergency": self.costs["emergency"], "cost_failure": self.costs["failure"],
                "total_cost": sum(a.airborne_time for a in values)+sum(self.costs.values())+wait,
                "truncated": self.failed, "cost_is_lower_bound": self.failed or bool(upstream),
                "first_failure_time": self.failure_time,
                "minimum_separation": self.minimum_separation,
                "conservation": offered == len(admitted)+rerouted+len(upstream) and len(admitted) == len(landed)+len(failed)+airborne}
        result.update(rescue_rule_profile=getattr(self,"rescue_rule_profile",LEGACY_PROFILE),
            rescue_warning=sum(a.emergency.warning for a in values),
            emergency_trigger_count=sum(e["type"] == "emergency_triggered" for e in self.events),
            rescue_search_exhausted=sum(e["type"] == "rescue_opportunity_search_exhausted" for e in self.events),
            protected_rescue_opportunities=sum(p.kind == "FINAL_EMERGENCY"
                for p in (*self.active.values(), *self.soft.values())))
        if self._stage2_enabled():
            result.update(self._stage2_metrics(values, upstream))
        return result

    def _stage2_metrics(self, values, upstream):
        ground = {a.id for a in values if a.fixture and a.touchdown_time == 0. and a.airborne_time == 0.}
        initial = {a.id for a in values if a.fixture and a.id not in ground}
        admitted = [a for a in values if a.admitted and a.id not in ground]
        landed = [a for a in admitted if a.touchdown_time is not None]
        pending = [a for a in admitted if a.status in AIRBORNE]
        failures = len(self.failure_archive)
        rerouted = sum(a.status == "REROUTED" for a in values)
        normal = sum(not a.emergency.charged for a in landed)
        offered = len(initial)+self._manifest_index
        before = sum(a.touchdown_time < self.failure_time for a in landed) if self.failure_time is not None else len(landed)
        cohorts = {}
        for name, ids in (("initial_airborne", initial), ("new_requests", {a.id for a in values if not a.fixture})):
            rows = [a for a in admitted if a.id in ids]
            cohorts[name] = dict(admitted=len(rows), success=sum(a.touchdown_time is not None for a in rows),
                failed=sum(a.id in self.failure_archive for a in rows),
                airborne_pending=sum(a.status in AIRBORNE for a in rows),
                upstream_pending=len(upstream) if name == "new_requests" else 0,
                rerouted=rerouted if name == "new_requests" else 0,
                airborne_time_s=sum(a.airborne_time for a in rows))
        key = (failures, -len(landed), self.costs["emergency"]+self.costs["reroute"],
               sum(a.energy_j for a in values), sum(a.airborne_time for a in values)
               +sum(a.upstream_wait_seconds for a in values)+sum(r["wait_seconds"] for r in upstream))
        return dict(policy_id=STAGE2_POLICY_ID, base_policy_id=POLICY_ID,
            experiment_policy_id=STAGE2_POLICY_ID, post_failure_model="abstract_isolation_no_cascade",
            has_energy_losses=self.has_energy_losses, terminated_invalid=self.terminated_invalid,
            initial_airborne_count=len(initial), initial_ground_count=len(ground),
            offered=offered, admitted=len(admitted), physically_landed=len(landed),
            normal_landed=normal, emergency_landed=len(landed)-normal,
            landing_below_reserve=sum(a.soc < .25 for a in landed),
            energy_failures=failures, failed_before_landing=failures,
            F_energy_total=failures,
            F_battery=sum(r["reason"] == "BATTERY_DEPLETED" for r in self.failure_archive.values()),
            F_peak=sum(r["reason"] == "PEAK_BUDGET_EXHAUSTED" for r in self.failure_archive.values()),
            L_success=len(landed), L_normal=normal, L_emergency=len(landed)-normal,
            N_offered=offered, N_admitted_total=len(admitted), N_rerouted=rerouted,
            N_upstream_pending=len(upstream), N_airborne_pending=len(pending),
            emergency_unresolved=sum(a.emergency.declared and a.status in AIRBORNE for a in admitted),
            airborne_or_censored=len(pending), successful_touchdowns_before_first_failure=before,
            successful_touchdowns_after_first_failure=len(landed)-before,
            failure_resources_released=self.failure_resources_released,
            cohorts=cohorts, K_episode=key, E_terminal_J=key[-2], W_unresolved_plus_airborne_seconds=key[-1],
            C_legacy=sum(self.costs.values())+key[-1],
            conservation=(offered == len(admitted)+rerouted+len(upstream)
                          and len(admitted) == len(landed)+failures+len(pending)))

    def to_dict(self):
        self._record(force=True)
        self.flush_records()
        result = {"time": self.time, "config_hash": self.config.config_hash, "policy_id": POLICY_ID,
                "manifest": self.manifest.to_dict(), "states": list(self.records),
                "events": list(self.events), "trajectories": dict(self.trajectories),
                "metrics": self.metrics(), "rejections": list(self.rejections),
                "cpu_seconds": dict(self.cpu), "cache_hits": self.cache_hits,
                "reference_counts": dict(self.reference_counts),
                "last_reference_fallback": self._last_reference_fallback,
                "cache": {"enabled": self.cache_enabled,"q_size":len(self._q_cache),
                    "q_capacity":self.Q_CACHE_CAPACITY,"energy_size":len(self._energy_cache),
                    "energy_capacity":self.ENERGY_CACHE_CAPACITY,
                    "canonical_template_count":len(self._reference_templates),
                    "canonical_template_capacity":3*self.REFERENCE_PHASE_BINS,
                    "canonical_templates_always_memoized":True,
                    "practical_runtime":self.practical_runtime_stats()},
                "sample_dt_s": self.physics_dt, "execution_dt": self.physics_dt,
                "telemetry_dt_s": (self._practical_runtime["telemetry_period"] if self._runtime_enabled() else self.physics_dt)}
        if self.state_sink is not None:
            result["states_streamed"] = True
            result["states_count"] = self._record_count
            path = getattr(self.state_sink,"path",None)
            result["states_complete"] = path is not None
            if path is not None:
                result["states_jsonl"] = str(path)
        if self._stage2_enabled():
            result.update(policy_id=STAGE2_POLICY_ID, base_policy_id=POLICY_ID, experiment_policy_id=STAGE2_POLICY_ID,
                          post_failure_model="abstract_isolation_no_cascade",
                          failure_archive=dict(self.failure_archive), dispatch_timings=list(self.dispatch_timings),
                          has_energy_losses=self.has_energy_losses, terminated_invalid=self.terminated_invalid)
        return result
