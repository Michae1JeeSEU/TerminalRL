"""Deterministic event executor for FT-RS-S1-1.0.

The scheduler supplies immutable trajectories.  Resource calendars and the entire
traffic batch are validated before any command changes the world.
"""
from dataclasses import asdict, dataclass, field, replace
from collections import OrderedDict
from itertools import combinations
from types import MappingProxyType
import math
import time as walltime

from .config import freeze, load_config, thaw
from .demand import ArrivalManifest, ArrivalRequest
from .models import ValidationResult
from .resources import PadCalendar, ResourceConflict, SlotCalendar
from .trajectory import (generate_final, generate_inbound, generate_transfer,
                         make_ground_plan, make_stable_plan, slot_state)
from .energy import evaluate_power, integrate_energy
from .safety import validate_plan, validate_separation


AIRBORNE = {"INBOUND", "HOLD_OUTER", "HOLD_MIDDLE", "HOLD_INNER", "TRANSFER_O_M",
            "TRANSFER_M_I", "FINAL_NORMAL", "FINAL_EMERGENCY", "VERTICAL_DESCENT"}
HOLD = {"outer": "HOLD_OUTER", "middle": "HOLD_MIDDLE", "inner": "HOLD_INNER"}
SERVICE_SECONDS = {"short": 60., "standard": 120., "long": 240.}
POLICY_ID = "upstream_capacity_only_pad_arrival_v1_20261018"


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
                "policy_id": POLICY_ID, "upstream": thaw(self.upstream)}


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
                 state_sink=None):
        self.config = config or load_config()
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
        self._record()

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

    def _energy(self, plan, start, end):
        key = (getattr(plan, "trajectory_hash", id(plan)), float(start), float(end), self.config.profile_hash)
        if self.cache_enabled and key in self._energy_cache:
            self.cache_hits += 1
            return self._energy_cache[key]
        t = walltime.perf_counter()
        value = integrate_energy(plan, start, end, max_dt=.2, config=self.config)
        self.cpu["energy"] += walltime.perf_counter() - t
        if self.cache_enabled:
            self._store_cache(self._energy_cache,key,value,self.ENERGY_CACHE_CAPACITY)
        return value

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
        if a.ring is not None and a.status in AIRBORNE:
            return self._stable(a)
        return make_ground_plan(a.id, a.pad_id, start_time=self.time,
                                end_time=a.service_end or self.time + 1000, config=self.config)

    def _traffic(self, a, soft=None, extra=None):
        plans = []
        p = (self.soft if soft is None else soft).get(a.id)
        if p is not None:
            plans.append(p)
        if extra is not None:
            plans.append(extra)
        base = self._base(a)
        final = plans[-1] if plans else base
        end = (final.end_time + SERVICE_SECONDS[a.service_class]) if final.pad_id is not None else math.inf
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
        self._record()
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
        self._record()
        return a

    def snapshot(self):
        return self._snapshot(self._scheduler_metrics())

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

    def _record(self):
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
            while hi - lo > .0005:
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
            while hi - lo > .0005:
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
                entered = a.admission_time if a.admission_time is not None else a.release_time
                slots.reserve((a.ring, a.slot), a.id, min(entered, self.time), end,
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
                    pads.reserve(p.pad_id, a.id, start, p.end_time + SERVICE_SECONDS[a.service_class],
                                 kind="emergency" if p.kind == "FINAL_EMERGENCY" else "final",
                                 plan_hash=p.trajectory_hash, committed=p is current)
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
            if soc <= .35 or (q is not None and soc-q-qhold <= .25):
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
                                current is None and p.start_time == self.time and not a.emergency.declared)
            if (a.emergency.eligible or a.emergency.pending) and not emergency and not normal_exception:
                return self._reject("EMERGENCY_DISPOSITION_REQUIRED", a.id)
            if p.kind == "FINAL_NORMAL" and source_ring != "inner":
                return self._reject("ILLEGAL_NORMAL_TOPOLOGY", a.id)
            if p.kind.startswith("TRANSFER") and (source_ring, p.target_ring) not in (("outer", "middle"), ("middle", "inner")):
                return self._reject("ILLEGAL_NORMAL_TOPOLOGY", a.id)
            if emergency:
                if p.start_time != self.time:
                    return self._reject("EMERGENCY_MUST_START_NOW", a.id)
                # Ordinary soft final may be preempted transactionally.
                for other_id, old in tuple(soft.items()):
                    if (old.pad_id == p.pad_id and old.kind == "FINAL_NORMAL" and
                            old.end_time-13.5 < p.end_time+SERVICE_SECONDS[a.service_class] and
                            p.end_time-13.5 < old.end_time+SERVICE_SECONDS[self.aircraft[other_id].service_class]):
                        if old.trajectory_hash in protected:
                            return self._reject("RESOURCE_DEPENDENCY", a.id)
                        del soft[other_id]
                        changed.add(other_id)
            result = validate_plan(p, config=self.config)
            if not result.accepted:
                return self._reject(result.reason, a.id, validation=result.details)
            from .energy import EnergyResult
            holding = EnergyResult(0., 0., 0., True)
            if p.start_time > source_time:
                hold = make_stable_plan(a.id, source_ring, source_slot, start_time=source_time,
                                        end_time=p.start_time, source_version=source_version, config=self.config)
                holding = self._energy(hold, source_time, p.start_time)
            if current is not None:
                rest = self._energy(current, self.time, current.end_time)
                holding = EnergyResult(holding.joules+rest.joules, holding.soc_used+rest.soc_used,
                                       holding.peak_seconds+rest.peak_seconds, holding.valid and rest.valid,
                                       reason=holding.reason or rest.reason)
            energy = self._energy(p, p.start_time, p.end_time)
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
                            p.end_time+SERVICE_SECONDS[self.aircraft[p.aircraft_id].service_class], p.aircraft_id)]
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
        removed = [old for key,old in self.soft.items() if key not in soft or soft[key].trajectory_hash != old.trajectory_hash]
        self.soft, self.slots, self.pads, self._source_clear = soft, slots, pads, times
        for old in removed:
            self._event("soft_plan_cancelled",old.aircraft_id,trajectory_hash=old.trajectory_hash,
                        reason="atomic_replacement_or_preemption")
        if redirects:
            self.dependencies = {h: tuple(redirects.get(d,d) for d in ds) for h,ds in self.dependencies.items()}
        for p in plans:
            self.trajectories[p.trajectory_hash] = p.to_dict()
            self._event("plan_committed" if p.start_time == self.time else "soft_plan_reserved", p.aircraft_id,
                        trajectory_hash=p.trajectory_hash, start_time=p.start_time, end_time=p.end_time)
        # A submit within emergency handling must not start another aircraft's
        # scheduled ordinary maneuver before that aircraft receives disposition.
        self._start_due(ids)
        self._record()
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
            a = self.aircraft[aircraft_id]
            normal_exception = (p.kind == "FINAL_NORMAL" and a.ring == "inner" and
                                not a.emergency.declared and not a.emergency.pending)
            if p.kind != "FINAL_EMERGENCY" and (a.emergency.declared or
                    ((a.emergency.eligible or a.emergency.pending) and not normal_exception)):
                self._reject("EMERGENCY_DISPOSITION_REQUIRED", a.id, trajectory_hash=p.trajectory_hash)
                self._event("commit_invalidated", a.id, reason="EMERGENCY_DISPOSITION_REQUIRED",
                            trajectory_hash=p.trajectory_hash)
                result = self.cancel(a.id)
                if not result.accepted:
                    self._emergency_conflict(a.id, result)
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
            self.active[a.id] = p
            del self.soft[a.id]
            self.slots.mark_committed(p.trajectory_hash)
            self.pads.mark_committed(p.trajectory_hash)
            deps = (self._release_supports(p.target_ring, p.target_slot, a.id,
                                          self.slots, self.soft)
                    if p.target_ring is not None else set())
            self.dependencies[p.trajectory_hash] = tuple(sorted(deps))
            e = a.emergency
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
            touchdown_soc = None
            if p.kind == "FINAL_NORMAL":
                energy = self._energy(p, self.time, p.end_time)
                required_soc = math.fsum((.25, energy.soc_used))
                touchdown_soc = .25 + (a.soc-required_soc)
                start_details = {"start_soc": a.soc, "required_soc": required_soc,
                                 "predicted_touchdown_soc": touchdown_soc}
            self.aircraft[a.id] = replace(a, status=status, emergency=e, trajectory_hash=p.trajectory_hash,
                                         normal_final_touchdown_soc=touchdown_soc)
            self._event("maneuver_started", a.id, trajectory_hash=p.trajectory_hash,
                        dependencies=sorted(deps), **start_details)

    def _fail(self, aircraft_id, reason):
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

    @staticmethod
    def _descent_start(plan):
        """Use the executed final's serialized phase junction."""
        if plan.kind in {"FINAL_NORMAL", "FINAL_EMERGENCY"}:
            return next(s.start_time for s in plan.segments if s.kind == "DESCENT")
        return None

    def _advance_energy(self, end):
        initial, duration = self.time, end-self.time
        results, fail_at = {}, None
        for a in self.aircraft.values():
            if a.status not in AIRBORNE:
                continue
            p = self._base(a)
            result = self._energy(p, initial, end)
            if not result.valid or a.soc - result.soc_used <= 0 or a.peak_seconds + result.peak_seconds > 120:
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
                battery_power_w=evaluate_power(s.v,s.a,self.config).battery_w)
        if fail_at:
            self._fail(fail_at[1], "ENERGY_FAILURE")

    def _check_execution(self, end):
        """Check the full physical interval, with exact stable structural proofs."""
        bodies = [a for a in self.aircraft.values() if a.status in AIRBORNE or a.status == "LANDED_SERVICE"]
        if len(bodies) < 2:
            return
        bases = {a.id: self._base(a) for a in bodies}
        first = None
        for a,b in combinations(bodies,2):
            x,y = bases[a.id],bases[b.id]
            result = validate_separation(x,y,self.time,end,config=self.config)
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
                check=validate_separation(x,y,self.time,mid,config=self.config)
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
                service_end = p.end_time+SERVICE_SECONDS[a.service_class]
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
                    choices.append((q.soc_used, result.plan))
        result = min(choices, key=lambda x: x[0]) if choices else (None, None)
        if self.cache_enabled:
            self._store_cache(self._q_cache,key,result,self.Q_CACHE_CAPACITY)
        return result

    def _update_emergency(self):
        triggered = False
        start_cpu = walltime.perf_counter()
        for key, a in sorted(tuple(self.aircraft.items())):
            if a.status not in AIRBORNE or a.emergency.declared or a.status.startswith("FINAL") or a.status == "VERTICAL_DESCENT":
                continue
            p = self.active.get(key)
            if p is not None:
                if p.pad_id is not None:
                    continue
                ring, slot, t = p.target_ring, p.target_slot, p.end_time
                soc = a.soc-self._energy(p, self.time, t).soc_used
            else:
                ring, slot, t, soc = a.ring, a.slot, self.time, a.soc
            q, _ = self._reference(ring, slot, t, a.version)
            hold = make_stable_plan("reference_hold", ring, slot, start_time=t, end_time=t+10, config=self.config)
            qhold = self._energy(hold, t, t+10).soc_used
            alert = soc <= .35 or (q is not None and soc-q-qhold <= .25)
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
        for key in sorted(self.aircraft):
            a = self.aircraft[key]
            if not a.emergency.eligible or key in self.active or a.status not in HOLD.values():
                continue
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
        self._record()
        raise EmergencyDependencyConflict(aircraft_id, result.reason, details)

    def _try_final(self, a, emergency):
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

    def advance(self, until, decision_callback=None):
        if until < self.time:
            raise ValueError("time cannot run backward")
        callback = self.decision_callback if decision_callback is None else decision_callback
        while not self.failed:
            triggers = self._settle()
            triggers = self._update_emergency() or triggers
            heartbeat = self._next_heartbeat <= self.time+1e-8
            if heartbeat:
                self._next_heartbeat = (math.floor((self.time+1e-8)/self.scheduler_period)+1)*self.scheduler_period
            if triggers or heartbeat or not self._initial_settled:
                self._initial_settled = True
                self.decision_times.append(self.time)
                self._record()
                cpu = walltime.perf_counter()
                if self.auto_emergency:
                    self.handle_emergencies()
                if callback and callback is not False:
                    callback(self.snapshot(), self._scheduler_port)
                self._start_due()
                self.cpu["decision"] += walltime.perf_counter()-cpu
                self._record()
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
            if self._manifest_index < len(self.manifest.requests):
                next_times.append(self.manifest.requests[self._manifest_index].release_time)
            next_times.extend(t for t in active_ends if t > self.time)
            next_times.extend(t for t in descent_starts if t > self.time)
            next_times.extend(t for t in soft_starts if t > self.time)
            next_times.extend(a.service_end for a in self.aircraft.values() if a.status == "LANDED_SERVICE" and a.service_end > self.time+1e-8)
            next_times.extend(x[0] for x in self._source_clear.values() if x[0] > self.time+1e-8)
            next_time = min(next_times)
            self._check_execution(next_time)
            if not self.failed:
                self._advance_energy(next_time)
            self._record()
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
        inbound_energy = self._energy(inbound, inbound.start_time, inbound.end_time)
        soc = initial_soc-inbound_energy.soc_used
        peak = inbound_energy.peak_seconds
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
            soc -= q.soc_used
            peak += q.peak_seconds
            ring, slot, t = target_ring, p.target_slot, p.end_time
        for pad in range(6):
            candidate = generate_final("freeflow", ring, slot, pad, t, config=self.config)
            if candidate.plan is not None and candidate.status == "FEASIBLE":
                q = self._energy(candidate.plan, t, candidate.plan.end_time)
                if q.valid and soc-q.soc_used >= .25 and peak+q.peak_seconds <= 120:
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
            q = self._energy(p, p.start_time, p.end_time)
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
        return {"policy_id": POLICY_ID, "planned_manifest": len(self.manifest.requests), "offered": offered,
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

    def to_dict(self):
        self._record()
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
                    "canonical_templates_always_memoized":True},
                "sample_dt_s": self.physics_dt, "execution_dt": self.physics_dt}
        if self.state_sink is not None:
            result["states_streamed"] = True
            result["states_count"] = self._record_count
            path = getattr(self.state_sink,"path",None)
            result["states_complete"] = path is not None
            if path is not None:
                result["states_jsonl"] = str(path)
        return result
