"""Detached arrival dispatch policies for the native terminal environment.

The dispatcher only chooses a deterministic order and the existing nominal
release request.  The environment remains responsible for admission,
trajectory generation, and the authoritative safety filter.  Route-energy
numbers in this module are detached scheduling estimates; they are not safety
certificates and do not reserve resources.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time
from types import SimpleNamespace
from typing import Any, Iterable, TYPE_CHECKING

import numpy as np

from .config import RingRole
from .energy import EnergyModel
from .energy_reachability import _advance_ring_holder, _copy_state_at_ring, _future_ring_holder
from .merge_scheduler import MergeReservation, MergeScheduler
from .priority import PriorityEngine
from .trajectory_generator import TrajectoryGenerator
from .trajectory_time_parameterization import TrajectoryInfeasibleError
from .uav import FlightPhase

if TYPE_CHECKING:  # pragma: no cover - imported only by static type checkers.
    from ..envs.parallel_env import TerminalParallelEnv


NOMINAL_K3 = (
    (1.0, 0.0, 0.0),
    (1.0, -1.0, 0.0),
    (1.0, 1.0, 0.0),
)

_TARGETS_BY_ROLE = {
    RingRole.OUTER: (RingRole.MIDDLE, RingRole.INNER, "final"),
    RingRole.MIDDLE: (RingRole.INNER, "final"),
    RingRole.INNER: ("final",),
}
_RING_PHASES = frozenset(
    (FlightPhase.RING_OUTER, FlightPhase.RING_MIDDLE, FlightPhase.RING_INNER)
)
_PHASE_PROGRESS = {
    FlightPhase.INBOUND: 0,
    FlightPhase.RING_OUTER: 1,
    FlightPhase.TRANSFER_O2M: 2,
    FlightPhase.RING_MIDDLE: 3,
    FlightPhase.TRANSFER_M2I: 4,
    FlightPhase.RING_INNER: 5,
    FlightPhase.FINAL: 6,
    FlightPhase.EMERGENCY_LANDING: 6,
    FlightPhase.LANDING: 7,
    FlightPhase.SERVICE: 8,
    FlightPhase.COMPLETED: 9,
}


@dataclass(frozen=True)
class DispatchDecision:
    """One deterministic native dispatch request."""

    order: tuple[int, ...]
    actions: dict[str, np.ndarray]
    shortlists: dict[str, list[np.ndarray]]
    diagnostics: dict[str, Any]


def _uid(value: Any) -> int:
    return int(value)


def _active_rows(env: Any) -> list[tuple[int, Any]]:
    active = getattr(env, "active", {})
    return sorted((int(uid), uav) for uid, uav in active.items())


def _uav_for_uid(env: Any, uid: int) -> Any:
    try:
        return env.active[uid]
    except (KeyError, TypeError):
        return env.active[str(uid)]


def _eligible(env: Any, uav: Any) -> bool:
    # Frozen nominal K3 contract from collect_p3_v3_sparse_expert.rule_tape.
    return bool(
        getattr(uav, "committed_plan", None) is None
        and getattr(uav, "ring_role", None) is not None
        and float(getattr(env, "now", 0.0)) >= float(getattr(uav, "safety_retry_after", 0.0))
    )


def _array(values: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).copy()


def nominal_requests(
    env: "TerminalParallelEnv",
    release_ids: Iterable[int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    """Return the standard nominal release/WAIT action and K3 shortlist.

    Every current active UID receives an explicit action.  Missing actions in
    the native environment have historical release semantics, so explicit
    WAIT rows are retained for non-selected or ineligible UAVs.
    """

    rows = _active_rows(env)
    active_ids = {uid for uid, _uav in rows}
    if release_ids is None:
        selected = {uid for uid, uav in rows if _eligible(env, uav)}
    else:
        selected = {_uid(value) for value in release_ids}
        unknown = selected - active_ids
        if unknown:
            raise ValueError(f"release_ids contain unknown active UIDs: {sorted(unknown)}")

    actions: dict[str, np.ndarray] = {}
    shortlists: dict[str, list[np.ndarray]] = {}
    for uid, uav in rows:
        key = str(uid)
        release = uid in selected and _eligible(env, uav)
        actions[key] = _array((1.0, 0.0, 0.0) if release else (0.0, 0.0, 0.0))
        if release:
            shortlists[key] = [_array(candidate) for candidate in NOMINAL_K3]
    return actions, shortlists


def downstream_soc_external_decision(env: "TerminalParallelEnv") -> DispatchDecision:
    """Recompute the native downstream/SOC rule on pre-step active state.

    This uses the legacy tuple on the same state, but is not a claim of
    stepwise equivalence: native admission occurs inside step, and external
    dispatch appends new same-tick admissions in UID order.
    """
    rank = {RingRole.INNER: 0, RingRole.MIDDLE: 1, RingRole.OUTER: 2}
    priority = PriorityEngine(env.config)
    keys = {
        uid: (rank.get(uav.ring_role, 3), float(uav.soc),
              -float(priority.compute(uav, env.now)),
              -float(uav.external_wait_time + uav.internal_wait_time), uid)
        for uid, uav in _active_rows(env)
    }
    order = tuple(sorted(keys, key=keys.__getitem__))
    actions, shortlists = nominal_requests(env)
    return DispatchDecision(order, actions, shortlists, {
        "controller": "downstream_soc_external",
        "ordering_state": "pre_step_active",
        "same_tick_admissions": "appended_in_uid_order_by_external_interface",
        "legacy_stepwise_equivalence": False,
        "ordering_keys": {str(uid): list(keys[uid]) for uid in order},
        "native_budget_queries": 0,
        "native_rollout_steps": 0,
        "cache_used": False,
    })


def _reservation(uid: int, holder: Any, target: RingRole | str,
                 now: float, duration: float) -> MergeReservation:
    ring_progress = float(getattr(holder, "ring_progress", 0.0) or 0.0)
    target_name = target.value if isinstance(target, RingRole) else str(target)
    return MergeReservation(
        f"dispatch-budget-{uid}-{target_name}-{now:.9f}",
        uid,
        getattr(getattr(holder, "phase", None), "value", str(getattr(holder, "phase", ""))),
        target_name,
        float(now + 6.0 + duration),
        ring_progress + math.pi / 2.0,
        None,
        None,
    )


def _normal_route_budget_details_unchecked(env: Any, uav: Any) -> dict[str, Any]:
    """Return a detached normal-route estimate and its bookkeeping fields."""

    role = getattr(uav, "ring_role", None)
    if role not in _TARGETS_BY_ROLE or getattr(uav, "committed_plan", None) is not None:
        return {
            "required_energy": math.inf,
            "route_energy": math.inf,
            "pad_wait_seconds": math.inf,
            "selected_lz_id": None,
            "status": "no_route",
            "physical_infeasibility_claim": False,
        }

    # Fresh components keep candidate queries away from live generator caches,
    # mutable geometry, telemetry, and scheduler state.
    config = deepcopy(env.config)
    airspace = deepcopy(env.airspace)
    generator = TrajectoryGenerator(config)
    energy = EnergyModel(config)
    scheduler = MergeScheduler(config)
    route_now = float(env.now)
    # The live position is already at env.now, even if entry into this ring
    # happened earlier. Anchor the detached holder to the current timestamp.
    holder = _copy_state_at_ring(uav, role, route_now)
    route_energy = 0.0

    for target in _TARGETS_BY_ROLE[role]:
        if target == "final":
            break
        snapshot = SimpleNamespace(now=route_now, active={}, airspace=airspace)
        duration = scheduler.transfer_duration_estimate(holder, target, snapshot)
        plan = generator.ring_transfer(
            holder,
            holder.ring_role,
            target,
            _reservation(int(uav.uid), holder, target, route_now, duration),
            airspace,
            route_now,
            0.0,
        )
        route_energy += float(energy.estimate_plan_energy(plan))
        holder = _future_ring_holder(holder, plan, target)
        route_now = float(plan.end_time)

    reserve_energy = float(energy.min_landing_soc * uav.energy_capacity)
    margin_energy = float(config.emergency_energy_margin_seconds * config.power_reference)
    best: tuple[float, int, float, float] | None = None
    lz_manager = deepcopy(env.lz_manager)
    for zone in lz_manager.zones:
        if uav.mission_type not in zone.compatible_missions:
            continue
        ready = max(
            float(env.now),
            float(lz_manager.earliest_release_time(zone.lz_id, now=float(env.now))),
        )
        if not math.isfinite(ready):
            continue
        wait_seconds = max(0.0, ready - route_now)
        final_now = route_now + wait_seconds
        final_holder = _advance_ring_holder(holder, RingRole.INNER, final_now, airspace, generator)
        try:
            final_plan = generator.final_approach(
                final_holder, zone,
                MergeReservation("dispatch-final-budget", int(uav.uid),
                                 final_holder.phase.value, "final", final_now,
                                 None, None, None),
                final_now, speed_residual_mps=0.0, airspace=airspace,
            )
        except TrajectoryInfeasibleError:
            continue
        final_energy = float(energy.estimate_plan_energy(final_plan))
        wait_energy = float(energy.phase_power(holder.phase) * wait_seconds)
        required = route_energy + final_energy + wait_energy + reserve_energy + margin_energy
        candidate = (float(required), int(zone.lz_id), float(wait_seconds), float(final_energy))
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None:
        return {
            "required_energy": math.inf,
            "route_energy": route_energy,
            "pad_wait_seconds": math.inf,
            "selected_lz_id": None,
            "status": "no_route",
            "physical_infeasibility_claim": False,
        }
    required, lz_id, wait_seconds, final_energy = best
    return {
        "required_energy": required,
        "route_energy": route_energy + final_energy,
        "pad_wait_seconds": wait_seconds,
        "selected_lz_id": lz_id,
        "status": "estimated",
        "physical_infeasibility_claim": False,
    }


def _normal_route_budget_details(env: Any, uav: Any) -> dict[str, Any]:
    try:
        return _normal_route_budget_details_unchecked(env, uav)
    except TrajectoryInfeasibleError as exc:
        return {"required_energy": math.inf, "status": "no_route",
                "reason": str(exc), "physical_infeasibility_claim": False}


def normal_route_budget(env: "TerminalParallelEnv", uav: Any) -> float:
    """Return detached normal remaining-route energy including reserve/margin.

    The estimate includes actual generated remaining normal transfers, a
    generated normal final approach, reported pad wait, landing reserve, and
    configured emergency margin.  It does not reserve an LZ or run SafetyFilter.
    ``math.inf`` means that no normal route witness was found.
    """

    return float(_normal_route_budget_details(env, uav)["required_energy"])


def _energy_rows(env: Any) -> tuple[list[tuple[int, Any, float, float, dict[str, Any]]], tuple[int, ...]]:
    rows: list[tuple[int, Any, float, float, dict[str, Any]]] = []
    for uid, uav in _active_rows(env):
        details = _normal_route_budget_details(env, uav)
        budget = float(details["required_energy"])
        energy_remaining = float(getattr(uav, "energy_remaining", 0.0))
        slack_energy = energy_remaining - budget if math.isfinite(budget) else -math.inf
        power = float(EnergyModel(env.config).phase_power(uav.phase))
        slack = slack_energy / power if power > 0 else -math.inf
        wait = float(getattr(uav, "external_wait_time", 0.0)) + float(
            getattr(uav, "internal_wait_time", 0.0)
        )
        details = {**details,
            "required_energy": budget,
            "energy_remaining": energy_remaining,
            "slack_energy": slack_energy,
            "slack_seconds": slack,
            "waiting_power": power,
            "wait_seconds": wait,
            "status": "estimated" if math.isfinite(budget) else "no_route",
            "physical_infeasibility_claim": False,
        }
        rows.append((uid, uav, slack, wait, details))
    rows.sort(key=lambda row: (
        not (row[1].phase in _RING_PHASES and row[1].committed_plan is None),
        row[2], -row[3], row[0]))
    return rows, tuple(uid for uid, _uav, _slack, _wait, _details in rows)


def energy_slack_order(env: "TerminalParallelEnv") -> tuple[int, ...]:
    """Order every current active UID by detached normal-route energy slack."""

    return _energy_rows(env)[1]


def _resource_signature(env: Any) -> tuple[Any, ...]:
    active = []
    for uid, uav in _active_rows(env):
        plan = getattr(uav, "committed_plan", None)
        active.append(
            (
                uid,
                getattr(getattr(uav, "ring_role", None), "value", None),
                None if plan is None else str(getattr(plan, "plan_id", "")),
                getattr(uav, "assigned_lz", None),
            )
        )
    queue = []
    for item in getattr(getattr(env, "admission", None), "queue", SimpleNamespace(items=[])).items:
        request = item.request
        queue.append(int(request.arrival_id))
    zones = []
    for zone in getattr(getattr(env, "lz_manager", None), "zones", ()):
        # A free pad's ready time tracks now, which is not a resource event.
        ready = None if zone.status.value == "free" else env.lz_manager.earliest_release_time(zone.lz_id, now=env.now)
        zones.append((int(zone.lz_id), zone.status.value, zone.reserved_by,
                      zone.occupied_by, ready))
    scheduler = getattr(env, "scheduler", None)
    reservations = tuple(sorted((key, tuple(sorted(vars(value).items())))
                                for key, value in getattr(scheduler, "reservations", {}).items()))
    return (tuple(active), tuple(queue), tuple(zones), reservations,
            tuple(sorted(getattr(scheduler, "last_executed_release", {}).items())))


def _freeze_future_arrivals(clone: Any) -> None:
    arrivals = getattr(clone, "arrivals", None)
    if arrivals is None:
        return
    if hasattr(arrivals, "_schedule"):
        arrivals._schedule = ()
    if hasattr(arrivals, "index"):
        arrivals.index = 0
    if hasattr(arrivals, "next_time"):
        arrivals.next_time = math.inf


def _clone_for_planning(env: Any) -> Any:
    # Observers can contain open output streams or local functions; neither
    # belongs in a detached rollout or should receive planning-only steps.
    observer = getattr(env, "_step_observer", None)
    clone = deepcopy(env, {} if observer is None else {id(observer): None})
    if hasattr(clone, "clear_step_observer"):
        clone.clear_step_observer()
    _freeze_future_arrivals(clone)
    return clone


def _retained_order(order: Iterable[int], current_order: Iterable[int]) -> tuple[int, ...]:
    current = tuple(int(uid) for uid in current_order)
    current_set = set(current)
    retained = [int(uid) for uid in order if int(uid) in current_set]
    retained_set = set(retained)
    retained.extend(uid for uid in current if uid not in retained_set)
    return tuple(retained)


def _candidate_orders(base: tuple[int, ...], env: Any, max_orders: int) -> list[tuple[int, ...]]:
    candidates = [base]
    if max_orders == 1:
        return candidates
    eligible_positions = [index for index, uid in enumerate(base) if _eligible(env, _uav_for_uid(env, uid))]
    for left, right in zip(eligible_positions, eligible_positions[1:]):
        candidate = list(base)
        candidate[left], candidate[right] = candidate[right], candidate[left]
        value = tuple(candidate)
        if value not in candidates:
            candidates.append(value)
        if len(candidates) >= max_orders:
            break
    return candidates


def _outcome_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _progress_score(env: Any) -> int:
    total = 0
    for _uid, uav in _active_rows(env):
        phase = getattr(uav, "phase", None)
        total += 100 * _PHASE_PROGRESS.get(phase, 0)
        total += int(getattr(uav, "committed_plan", None) is not None)
    for uav in getattr(env, "service_uavs", {}).values():
        total += 100 * _PHASE_PROGRESS.get(getattr(uav, "phase", None), 8)
    total += 900 * len(getattr(env, "completed_uavs", ()))
    return int(total)


def _wait_total(env: Any) -> float:
    active_wait = sum(
        float(getattr(uav, "external_wait_time", 0.0))
        + float(getattr(uav, "internal_wait_time", 0.0))
        for _uid, uav in _active_rows(env)
    )
    queue_wait = sum(float(getattr(item, "external_wait_time", 0.0)) for item in getattr(getattr(env, "admission", None), "queue", SimpleNamespace(items=[])).items)
    return float(active_wait + queue_wait)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _integrity_errors(env: Any) -> list[str]:
    ledger = env.ledger
    errors = []
    if not set(ledger.outcomes).issubset(ledger.requested):
        errors.append("outcome_without_request")
    pending = [u.arrival_id for u in env.active.values()]
    pending += [u.arrival_id for u in env.service_uavs.values()]
    pending += [q.request.arrival_id for q in env.admission.queue.items]
    if len(pending) != len(set(pending)):
        errors.append("duplicate_pending_request")
    if set(pending) != ledger.unresolved:
        errors.append("unresolved_ledger_mismatch")
    return errors


def _evaluate_order(env: Any, order: tuple[int, ...], lookahead_steps: int,
                    deadline: float) -> dict[str, Any]:
    started = time.monotonic()
    clone = _clone_for_planning(env)
    before_separation = dict(clone.separation_monitor.telemetry)
    before_hash_mismatches = int(clone.telemetry.get("checked_executed_plan_hash_mismatches", 0))
    before_outcomes = set(clone.ledger.outcomes)
    before_integrity = set(_integrity_errors(clone))
    native_steps = 0
    tick_order = order
    try:
        for _tick in range(lookahead_steps):
            if time.monotonic() >= deadline:
                return {"status": "timeout", "order": order,
                        "native_steps": native_steps, "elapsed_seconds": time.monotonic()-started}
            actions, shortlists = nominal_requests(clone)
            # Retain the candidate order; newly admitted observed requests go
            # last in UID order. No fresh hidden-demand or budget query here.
            tick_order = _retained_order(tick_order, (uid for uid, _ in _active_rows(clone)))
            result = clone.step(actions, proposal_shortlists=shortlists, release_order=tick_order)
            native_steps += 1
            if not isinstance(result, tuple) or len(result) != 5:
                raise ValueError("native step must return its five-tuple")
            if time.monotonic() >= deadline:
                return {"status": "timeout", "order": order,
                        "native_steps": native_steps, "elapsed_seconds": time.monotonic()-started}
            if clone.now >= clone.config.max_simulation_seconds:
                break
            if not clone.active and not clone.service_uavs and not clone.admission.queue.items:
                break
        separation = clone.separation_monitor.telemetry
        deltas = {name: int(separation.get(name, 0))-int(before_separation.get(name, 0))
                  for name in ("collisions", "nmac", "hard_separation_violations")}
        values = [_outcome_value(value) for key, value in clone.ledger.outcomes.items()
                  if key not in before_outcomes]
        failed_energy = values.count("failed_energy")
        failed_collision = values.count("failed_collision")
        emergency_ids = {u.arrival_id for u in clone.active.values() if u.emergency_landing}
        emergency_ids.update(u.arrival_id for u in clone.service_uavs.values() if u.emergency_landing)
        emergency = len(emergency_ids) + values.count("emergency_landed")
        bad_terminal = sum(v.startswith(("failed_", "diverted_")) for v in values) + emergency
        normal_completed = values.count("completed")
        integrity = sorted(set(_integrity_errors(clone))-before_integrity)
        hash_mismatch_delta = int(clone.telemetry.get("checked_executed_plan_hash_mismatches", 0))-before_hash_mismatches
        valid = not (any(v > 0 for v in deltas.values()) or failed_energy or failed_collision
                     or integrity or hash_mismatch_delta > 0 or "invalid" in values)
        score = (int(valid), -int(bad_terminal), normal_completed,
                 _progress_score(clone), -_wait_total(clone))
        return {"status": "evaluated" if valid else "invalid_native_outcome", "order": order,
                "score": score, "separation_deltas": deltas, "failed_energy": failed_energy,
                "failed_collision": failed_collision, "integrity_errors": integrity,
                "checked_executed_plan_hash_mismatches_delta": hash_mismatch_delta,
                "bad_terminal": bad_terminal, "normal_completed": normal_completed,
                "native_steps": native_steps, "elapsed_seconds": time.monotonic()-started}
    except TrajectoryInfeasibleError as exc:
        return {"status": "trajectory_infeasible", "order": order,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "native_steps": native_steps, "elapsed_seconds": time.monotonic()-started,
                "physical_infeasibility_claim": False}


class ArrivalDispatcher:
    """Deterministic energy-slack or bounded rolling order controller."""

    def __init__(
        self,
        controller: str = "energy_slack",
        lookahead_steps: int = 4,
        max_orders: int = 4,
        replan_interval: int = 10,
        planning_wall_seconds: float = 60,
    ) -> None:
        if controller not in {"energy_slack", "rolling_reorder"}:
            raise ValueError("controller must be 'energy_slack' or 'rolling_reorder'")
        if not 1 <= int(lookahead_steps) <= 4 or not 1 <= int(max_orders) <= 4 or int(replan_interval) < 1:
            raise ValueError("lookahead_steps/max_orders must be 1..4 and replan_interval positive")
        if float(planning_wall_seconds) <= 0.0 or not math.isfinite(float(planning_wall_seconds)):
            raise ValueError("planning_wall_seconds must be finite and positive")
        self.controller = controller
        self.lookahead_steps = int(lookahead_steps)
        self.max_orders = int(max_orders)
        self.replan_interval = int(replan_interval)
        self.planning_wall_seconds = float(planning_wall_seconds)
        self._last_order: tuple[int, ...] | None = None
        self._last_signature: tuple[Any, ...] | None = None
        self._last_replan_step: int | None = None

    def decide(self, env: "TerminalParallelEnv") -> DispatchDecision:
        started = time.monotonic()
        signature = _resource_signature(env)
        step_count = int(env.step_count)
        interval_due = (self._last_replan_step is None or step_count < self._last_replan_step
                        or step_count-self._last_replan_step >= self.replan_interval)
        signature_changed = self._last_signature is None or signature != self._last_signature
        refreshed = interval_due or signature_changed or self._last_order is None
        event = "initial" if self._last_order is None else (
            "resource_change" if signature_changed else "interval" if interval_due else "cached")
        native_budget_queries = 0
        budget_started = time.monotonic()
        if refreshed:
            energy_rows, energy_order = _energy_rows(env)
            self._cached_energy_rows = energy_rows
            self._cached_energy_order = energy_order
            native_budget_queries = sum(u.phase in _RING_PHASES and u.committed_plan is None
                                        for _, u in _active_rows(env))
        else:
            energy_rows, energy_order = self._cached_energy_rows, self._cached_energy_order
        budget_elapsed = time.monotonic()-budget_started
        candidate_results = []
        fallback = False
        fallback_reason = None
        if not refreshed:
            order = self._last_order
        elif self.controller == "energy_slack":
            order = energy_order
        else:
            candidates = _candidate_orders(energy_order, env, self.max_orders)
            deadline = started + self.planning_wall_seconds
            best = None
            for candidate in candidates:
                if time.monotonic() >= deadline:
                    result = {"status": "timeout", "order": candidate, "native_steps": 0}
                else:
                    result = _evaluate_order(env, candidate, self.lookahead_steps, deadline)
                candidate_results.append(result)
                if result["status"] == "evaluated":
                    if best is None or tuple(result["score"]) > tuple(best["score"]):
                        best = result
            if best is None:
                order = energy_order
                fallback = True
                fallback_reason = "no_valid_candidate_within_budget"
            else:
                order = tuple(best["order"])
        if refreshed:
            self._last_replan_step = step_count
        self._last_order = tuple(order)
        self._last_signature = signature
        actions, shortlists = nominal_requests(env)
        diagnostics = _json_safe({
            "controller": self.controller, "replanned": refreshed, "replan_reason": event,
            "cache_refreshed": refreshed, "cache_age_ticks": step_count-self._last_replan_step,
            "energy_order": energy_order,
            "energy_rows": {str(uid): details for uid, _, _, _, details in energy_rows},
            "candidate_results": candidate_results, "fallback": fallback,
            "fallback_reason": fallback_reason, "fallback_guarantees_complete_suffix": False,
            "native_budget_queries": native_budget_queries,
            "native_rollout_steps": sum(r.get("native_steps", 0) for r in candidate_results),
            "budget_elapsed_seconds": budget_elapsed,
            "planning_elapsed_seconds": time.monotonic()-started,
            "planning_wall_seconds": self.planning_wall_seconds,
            "lookahead_steps": self.lookahead_steps, "max_orders": self.max_orders,
            "replan_interval": self.replan_interval,
            "physical_infeasibility_claim": False,
            "normal_route_estimate_is_safety_certificate": False,
        })
        return DispatchDecision(tuple(order), actions, shortlists, diagnostics)


__all__ = [
    "ArrivalDispatcher",
    "DispatchDecision",
    "NOMINAL_K3",
    "downstream_soc_external_decision",
    "energy_slack_order",
    "nominal_requests",
    "normal_route_budget",
]
