"""A6.14 Outer reconfiguration timing and anticipation audit.

This is a rule-policy, common-random-number mechanism audit.  It deliberately
does not train PPO, load or modify a checkpoint, change the 180 s decision
interval / 600 s dwell, or relax any safety threshold.  The exact-time action
injection is an *offline counterfactual*: it calls the existing coordinator
with the same canonical Outer +1 desired action at a nominated time, then lets
the unmodified coordinator, transition, repair-aware scheduler and SafetyFilter
execute (or delay/reject) it.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
from typing import Callable
from typing import Iterable

import numpy as np

from ..core.config import RingRole, SimConfig
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import RuleLowLevelPolicy
from .a610_objective_audit import LOWER_IS_BETTER, classify_delta
from .a611_causal_audit import (
    EXPANSIONS,
    DemandProfile,
    _safe,
)
from .a612_scheduler_repair_audit import _a611_mu0
from .a613_capacity_reaudit import repair_config
from .a6_effectiveness import _collect, _sample


PPO_AUTHORIZATION = "STOP"
TIMING_SWEEP_SECONDS = (0., 30., 60., 90., 120., 180., 240., 300.)
BURST_RELATIVE_SECONDS = (-180., -120., -60., 0., 60., 180.)
OUTER_PLUS_ONE = EXPANSIONS[RingRole.OUTER]
FALLBACK_WAITING_MARGIN = 50_000.0


def _a610_path() -> Path:
    return Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a610" / "a610_objective_audit.json"


def load_materiality() -> dict:
    """Use the versioned A6.10 calibration when available, never invent noise."""
    path = _a610_path()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        materiality = dict(payload["materiality"])
        materiality["source"] = str(path)
        return materiality
    # This fallback is deliberately conservative and visibly labelled.  The
    # server protocol normally supplies the A6.10 calibration artifact.
    thresholds = {key: {"epsilon_abs": 0.0, "epsilon_rel": 0.0, "samples": 0} for key in LOWER_IS_BETTER}
    thresholds["all_request_waiting_area"] = {"epsilon_abs": FALLBACK_WAITING_MARGIN, "epsilon_rel": 0.0, "samples": 0}
    return {"version": "a614-fallback-a69-meaningful-waiting-margin", "source": "A6.10 artifact unavailable",
            "method": "A6.9/A6.10 compatible conservative waiting margin; report is INCONCLUSIVE for positive claims without A6.10 calibration",
            "thresholds": thresholds, "fallback": True}


def _counter(row: dict) -> Counter:
    return Counter(row.get("outcomes", {}))


def _metric_delta(row: dict, baseline: dict) -> dict:
    metric, base = row["metrics"], baseline["metrics"]
    outcomes, base_outcomes = _counter(row), _counter(baseline)
    delta = {"completed": float(metric["completed"] - base["completed"])}
    for key in LOWER_IS_BETTER:
        if key == "failed_energy":
            delta[key] = float(outcomes["failed_energy"] - base_outcomes["failed_energy"])
        elif key == "diverted_low_soc":
            delta[key] = float(outcomes["diverted_low_soc"] - base_outcomes["diverted_low_soc"])
        else:
            delta[key] = float(metric.get(key, 0.) - base.get(key, 0.))
    return delta


def benefit_assessment(row: dict, baseline: dict, materiality: dict) -> dict:
    """Apply the A6.10 direction/materiality logic without scalarizing reward."""
    delta = _metric_delta(row, baseline)
    labels, material_regression = classify_delta(delta, materiality, baseline["metrics"])
    waiting = -delta["all_request_waiting_area"]
    threshold = materiality["thresholds"]["all_request_waiting_area"]
    waiting_material = (waiting >= float(threshold["epsilon_abs"]) and
                        waiting / max(1., abs(float(baseline["metrics"]["all_request_waiting_area"]))) >= float(threshold["epsilon_rel"]) and
                        float(threshold["epsilon_abs"]) > 0.)
    primary = delta["completed"] >= 1 or (delta["completed"] == 0 and waiting_material)
    return {"delta": delta, "labels": labels, "material_regression": bool(material_regression),
            "waiting_material": bool(waiting_material), "beneficial": bool(row["safe"] and primary and not material_regression)}


def _decision_trace(env, decision, requested: np.ndarray) -> dict:
    return {"time": float(env.now), "request_id": decision.request_id,
            "original_request_time": decision.original_request_time,
            "requested_action": list(map(float, requested)), "applied_action": list(decision.applied_action),
            "outcome": decision.outcome, "execution_outcome": decision.execution_outcome,
            "reason_code": decision.reason_code, "actual_application_time": decision.actual_application_time,
            "application_delay_seconds": decision.application_delay_seconds,
            "deferred_until": decision.deferred_until, "structural_effective": decision.structural_effective,
            "capacity_effective": decision.capacity_effective, "repair_magnitude": decision.repair_magnitude,
            "projection_magnitude": decision.projection_magnitude,
            "capacity_before": list(decision.update.capacity_before), "capacity_after": list(decision.update.capacity_after)}


def _all_uavs(env) -> Iterable:
    yield from env.active.values()
    yield from env.service_uavs.values()
    yield from env.completed_uavs


def _capture_uav_state(env, admissions: dict[str, float], phases: dict[str, list[tuple[float, str]]]) -> None:
    for uav in _all_uavs(env):
        key = str(uav.arrival_id)
        if uav.admission_time is not None:
            admissions[key] = float(uav.admission_time)
        state = uav.phase.value
        history = phases.setdefault(key, [])
        if not history or history[-1][1] != state:
            history.append((float(env.now), state))


def _first_time(values: list[tuple[float, bool]]) -> float | None:
    return next((float(time) for time, occurred in values if occurred), None)


def _transition_summary(events: list[dict], capacity_history: list[tuple[float, int]], nominal_outer_capacity: int) -> dict:
    structural = [row for row in events if row["structural_effective"]]
    actual_application = next((row["actual_application_time"] for row in structural if row["actual_application_time"] is not None), None)
    effective = _first_time([(time, capacity > nominal_outer_capacity) for time, capacity in capacity_history])
    return {"geometry_applied_time": actual_application, "geometry_effective_time": effective,
            "transition_delay_seconds": float(sum(row["application_delay_seconds"] or 0. for row in structural)),
            "defer_count": sum(row["outcome"] == "delayed" for row in events),
            "reject_count": sum(row["outcome"] == "rejected" for row in events),
            "events": events}


def run_dynamic_at_time(config: SimConfig, seed: int, action_time: float, *, force_initial_gate: bool = True,
                        observation_capture_time: float | None = None,
                        observation_capture_times: tuple[float, ...] | None = None,
                        observation_capture_hook: Callable[[DynamicHighLevelWrapper], dict] | None = None) -> dict:
    """Run one exact-time A6.14 counterfactual through the frozen dynamic stack.

    ``force_initial_gate`` is limited to the no-traffic t=0 static-equivalence
    control.  It does not alter the configuration or the coordinator; it only
    models an initial geometry already authorized before arrivals start.
    """
    wrapper = DynamicHighLevelWrapper(config, RuleLowLevelAdapter()); wrapper.reset(seed); env = wrapper.base
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}
    events: list[dict] = []; admissions: dict[str, float] = {}; phases: dict[str, list[tuple[float, str]]] = {}
    pressure: list[tuple[float, bool]] = []; queue_growth: list[tuple[float, bool]] = []; capacity_history: list[tuple[float, int]] = []
    extra_capacity_use: list[tuple[float, bool]] = []; requested = False; captured_observations: dict[float, dict] = {}
    capture_times = tuple(sorted(set(observation_capture_times or (()))) if observation_capture_times is not None else
                          (() if observation_capture_time is None else (float(observation_capture_time),)))
    # Capacity is geometry-derived; compute it from the audit's actual fixed geometry rather than hard-code 47.
    from ..core.ring_capacity import geometric_ring_capacity
    nominal_outer = geometric_ring_capacity(config.rings[RingRole.OUTER].radius, config.safe_distance)
    while env.now < config.max_simulation_seconds and not wrapper._done():
        # Optional A6.20 observer: read the existing public high-level vector
        # before any action at this timestep.  It never touches runtime state.
        for capture_time in capture_times:
            if capture_time not in captured_observations and env.now >= capture_time - 1e-9:
                captured = {"time": float(env.now), "observation_29d": wrapper.observation().astype(float).tolist()}
                if observation_capture_hook is not None:
                    captured["hook"] = observation_capture_hook(wrapper)
                captured_observations[capture_time] = captured
        if not requested and env.now >= action_time - 1e-9:
            gate = wrapper.geometry_event_gate() or (force_initial_gate and action_time == 0.)
            decision = wrapper.coordinator.coordinate(OUTER_PLUS_ONE, geometry_gate=gate)
            events.append(_decision_trace(env, decision, OUTER_PLUS_ONE)); requested = True
        deferred = wrapper.coordinator.advance()
        if deferred is not None:
            events.append(_decision_trace(env, deferred, np.asarray(deferred.requested_action, np.float32)))
        actions = wrapper.coordinator.apply_speed_repairs(wrapper.low_level.act(env._observations()))
        env.step(actions); _sample(env, samples); _capture_uav_state(env, admissions, phases)
        outer_capacity = env.airspace.capacity(RingRole.OUTER); outer_occupancy = env.airspace.occupancy(RingRole.OUTER)
        capacity_history.append((float(env.now), int(outer_capacity)))
        queue_growth.append((float(env.now), bool(env.admission.queue.items)))
        # Pressure means that the nominal, 47-slot Outer ring is actually full;
        # it is distinct from merely seeing the first aircraft enter the ring.
        pressure.append((float(env.now), outer_occupancy >= nominal_outer))
        extra_capacity_use.append((float(env.now), outer_capacity > nominal_outer and outer_occupancy > nominal_outer))
    metrics = _collect(env, samples, wrapper.coordinator, events)
    metrics["admission_delay_mean"] = float(np.mean([value - env.request_times[int(key)] for key, value in admissions.items()])) if admissions else 0.
    metrics["admission_delay_p95"] = float(np.percentile([value - env.request_times[int(key)] for key, value in admissions.items()], 95)) if admissions else 0.
    result = {"metrics": metrics, "safe": _safe(metrics, config), "outcomes": dict(Counter(item.value for item in env.ledger.outcomes.values())),
              "mode": "dynamic_exact_time", "action_time": float(action_time), "actual_capacity": {role.value: env.airspace.capacity(role) for role in RingRole},
              "timing": {"geometry_requested_time": float(action_time),
                         **_transition_summary(events, capacity_history, nominal_outer),
                         "first_external_queue_growth_time": _first_time(queue_growth),
                         "first_outer_pressure_time": _first_time(pressure),
                         "first_extra_capacity_used_time": _first_time(extra_capacity_use)},
              "causal_trace": {"admission_time_by_arrival": admissions, "phase_changes_by_arrival": phases}}
    if any(time not in captured_observations for time in capture_times):
        raise RuntimeError("requested pre-action observation time was not reached")
    if observation_capture_time is not None:
        result["pre_action_observation"] = captured_observations[float(observation_capture_time)]
    if observation_capture_times is not None:
        result["pre_action_observations"] = captured_observations
    return result


def run_static_outer_plus_one(profile: DemandProfile, seed: int) -> dict:
    """A6.13 fixed-geometry control with the same causal trace fields."""
    config = repair_config(profile, dynamic=False, role=RingRole.OUTER)
    env = TerminalParallelEnv(config); env.reset(seed); policy = RuleLowLevelPolicy()
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}
    admissions: dict[str, float] = {}; phases: dict[str, list[tuple[float, str]]] = {}
    queue_growth: list[tuple[float, bool]] = []; pressure: list[tuple[float, bool]] = []; use: list[tuple[float, bool]] = []
    from ..core.ring_capacity import geometric_ring_capacity
    nominal_outer = geometric_ring_capacity(SimConfig().rings[RingRole.OUTER].radius, config.safe_distance)
    while env.now < config.max_simulation_seconds and not (env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs):
        env.step(policy.act(env._observations())); _sample(env, samples); _capture_uav_state(env, admissions, phases)
        outer_capacity, outer_occupancy = env.airspace.capacity(RingRole.OUTER), env.airspace.occupancy(RingRole.OUTER)
        queue_growth.append((float(env.now), bool(env.admission.queue.items)))
        pressure.append((float(env.now), outer_occupancy >= nominal_outer))
        use.append((float(env.now), outer_capacity > nominal_outer and outer_occupancy > nominal_outer))
    metrics = _collect(env, samples)
    delays = [value - env.request_times[int(key)] for key, value in admissions.items()]
    metrics["admission_delay_mean"] = float(np.mean(delays)) if delays else 0.
    metrics["admission_delay_p95"] = float(np.percentile(delays, 95)) if delays else 0.
    return {"metrics": metrics, "safe": _safe(metrics, config), "outcomes": dict(Counter(item.value for item in env.ledger.outcomes.values())),
            "mode": "static_outer_plus1", "action_time": 0., "actual_capacity": {role.value: env.airspace.capacity(role) for role in RingRole},
            "timing": {"geometry_requested_time": 0., "geometry_applied_time": 0., "geometry_effective_time": 0.,
                       "transition_delay_seconds": 0., "defer_count": 0, "reject_count": 0,
                       "first_external_queue_growth_time": _first_time(queue_growth), "first_outer_pressure_time": _first_time(pressure),
                       "first_extra_capacity_used_time": _first_time(use)},
            "causal_trace": {"admission_time_by_arrival": admissions, "phase_changes_by_arrival": phases}}


def _arrival_differences(row: dict, nominal: dict) -> dict:
    current = row.get("causal_trace", {}); control = nominal.get("causal_trace", {})
    admissions, baseline_admissions = current.get("admission_time_by_arrival", {}), control.get("admission_time_by_arrival", {})
    phases, baseline_phases = current.get("phase_changes_by_arrival", {}), control.get("phase_changes_by_arrival", {})
    ids = set(admissions) | set(baseline_admissions) | set(phases) | set(baseline_phases)
    changed = [uid for uid in ids if abs(admissions.get(uid, float("inf")) - baseline_admissions.get(uid, float("inf"))) > .5 or phases.get(uid, []) != baseline_phases.get(uid, [])]
    return {"capacity48_arrival_difference_count": len(changed), "capacity48_arrival_difference_ids": sorted(changed, key=int)}


def _static_trace(profile: DemandProfile, seed: int) -> dict:
    """Use the same dynamic no-op stack for a CRN-compatible causal trace."""
    config = repair_config(profile, dynamic=True)
    return run_dynamic_at_time(config, seed, float("inf"), force_initial_gate=False)


def _dynamic_task(profile: DemandProfile, seed: int, action_time: float) -> tuple[float, dict]:
    """Process-safe independent CRN rollout; no shared simulator state."""
    return action_time, run_dynamic_at_time(repair_config(profile, dynamic=True), seed, action_time)


def _response_row(condition: str, result: dict, nominal: dict, static: dict, materiality: dict) -> dict:
    versus_nominal = benefit_assessment(result, nominal, materiality)
    versus_static = benefit_assessment(result, static, materiality)
    metric, timing = result["metrics"], result.get("timing", {})
    return {"condition": condition, "action_time": result.get("action_time"), "completed": metric["completed"],
            "all_request_waiting_area": metric["all_request_waiting_area"], "external_queue_area": metric["external_queue_area"],
            "admission_delay_mean": metric.get("admission_delay_mean", 0.), "admission_delay_p95": metric.get("admission_delay_p95", 0.),
            "unresolved_count": metric["unresolved_count"], "service_delay_p95": metric["service_delay_p95"],
            "energy_mean": metric["energy_mean"], "priority_fairness_std": metric["priority_fairness_std"], "safe": result["safe"],
            "geometry_requested_time": timing.get("geometry_requested_time"), "geometry_applied_time": timing.get("geometry_applied_time"),
            "geometry_effective_time": timing.get("geometry_effective_time"), "transition_delay_seconds": timing.get("transition_delay_seconds", 0.),
            "defer_count": timing.get("defer_count", 0), "reject_count": timing.get("reject_count", 0),
            "first_external_queue_growth_time": timing.get("first_external_queue_growth_time"), "first_outer_pressure_time": timing.get("first_outer_pressure_time"),
            "first_extra_capacity_used_time": timing.get("first_extra_capacity_used_time"),
            "beneficial_vs_nominal": versus_nominal["beneficial"], "material_regression_vs_nominal": versus_nominal["material_regression"],
            "waiting_delta_vs_nominal": versus_nominal["delta"]["all_request_waiting_area"], "completed_delta_vs_nominal": versus_nominal["delta"]["completed"],
            "waiting_delta_vs_static": versus_static["delta"]["all_request_waiting_area"], "completed_delta_vs_static": versus_static["delta"]["completed"],
            "capacity48_arrival_difference_count": result.get("capacity48_arrival_difference_count", 0)}


def classify_audit(timing_rows: list[dict], burst_rows: list[dict], *, static_beneficial: bool, materiality_fallback: bool) -> dict:
    """Deterministic A6.14 decision rule.  It can never authorize PPO."""
    dynamic = [row for row in timing_rows if row["condition"].startswith("dynamic_outer_plus1_t")]
    beneficial = [row for row in dynamic if row["beneficial_vs_nominal"]]
    latest = max((row["action_time"] for row in beneficial), default=None)
    at_180 = next((row for row in dynamic if row["action_time"] == 180.), None)
    burst_beneficial = any(row["beneficial_vs_nominal"] for row in burst_rows)
    near_static = any(float(row.get("static_waiting_gain_recovery", 0.)) >= .80 for row in beneficial)
    if materiality_fallback:
        classification = "INCONCLUSIVE"
    elif not static_beneficial:
        classification = "NO_REPLICABLE_OUTER_GAIN"
    elif not beneficial:
        classification = "STATIC_ONLY_VALUE"
    elif latest == 0.:
        classification = "VALUE_WINDOW_TOO_NARROW"
    elif at_180 is not None and not at_180["beneficial_vs_nominal"] and latest < 180.:
        classification = "DECISION_INTERVAL_TOO_SLOW"
    elif at_180 is not None and at_180["beneficial_vs_nominal"] and burst_beneficial and near_static:
        classification = "ANTICIPATORY_CONTROL_VALUE_PROVEN"
    else:
        classification = "INCONCLUSIVE"
    return {"classification": classification, "latest_beneficial_action_time": latest,
            "ppo_authorization": PPO_AUTHORIZATION, "burst_has_beneficial_timing": burst_beneficial,
            "near_static_gain_recovery": near_static}


def _gain_recovery(dynamic: dict, nominal: dict, static: dict) -> float | None:
    static_gain = nominal["metrics"]["all_request_waiting_area"] - static["metrics"]["all_request_waiting_area"]
    dynamic_gain = nominal["metrics"]["all_request_waiting_area"] - dynamic["metrics"]["all_request_waiting_area"]
    return None if static_gain <= 0. else float(dynamic_gain / static_gain)


def select_mu0(seed: int, supplied_mu0: float | None = None) -> tuple[float, str]:
    """Reuse an explicitly identified calibration for CRN replications.

    Arrival seed changes the request realization, not the physical service-rate
    calibration.  Supplying the already verified seed-0 A6.11 μ₀ prevents a
    replication from silently spending its entire staged budget recalibrating
    the same fixed simulator parameter.
    """
    if supplied_mu0 is None:
        return _a611_mu0(seed)
    if not np.isfinite(supplied_mu0) or supplied_mu0 <= 0.:
        raise ValueError("supplied μ₀ must be finite and positive")
    return float(supplied_mu0), "explicit_preexisting_calibration"


def run_audit(*, seed: int = 0, workers: int = 6, mu0: float | None = None) -> dict:
    mu, mu_evidence = select_mu0(seed, mu0)
    constant = DemandProfile("constant_high_repair_scheduler", ((0., 720., 1.55 * mu),), 720., 480.)
    # Same A6.11/A6.13 burst construction; onset is the 300 s demand increase.
    burst = DemandProfile("medium_burst_repair_scheduler", ((0., 900., .95 * mu), (300., 900., 1.55 * mu)), 900., 480.)
    materiality = load_materiality()
    profiles = {}
    for profile, times in ((constant, TIMING_SWEEP_SECONDS), (burst, tuple(300. + value for value in BURST_RELATIVE_SECONDS))):
        nominal = _static_trace(profile, seed)
        static = run_static_outer_plus_one(profile, seed)
        if not nominal["safe"] or not static["safe"]:
            raise RuntimeError(f"A6.14 FAIL: unsafe fixed branch for {profile.name}")
        static.update(_arrival_differences(static, nominal))
        rows = {"nominal": nominal, "static_outer_plus1_t0": static}
        # Every timing branch owns an isolated environment but receives the
        # same profile schedule and seed.  Parallelism changes wall-clock time
        # only; it cannot change CRN inputs or the safety execution path.
        with ProcessPoolExecutor(max_workers=min(max(1, workers), len(times))) as pool:
            futures = [pool.submit(_dynamic_task, profile, seed, time) for time in times]
            for future in as_completed(futures):
                time, dynamic = future.result()
                if not dynamic["safe"]:
                    raise RuntimeError(f"A6.14 FAIL: unsafe dynamic branch {profile.name}@{time}")
                dynamic.update(_arrival_differences(dynamic, nominal))
                rows[f"dynamic_outer_plus1_t{int(time)}"] = dynamic
        static_assessment = benefit_assessment(static, nominal, materiality)
        response = [_response_row(name, value, nominal, static, materiality) for name, value in rows.items()]
        static_waiting_gain = -next(row["waiting_delta_vs_nominal"] for row in response if row["condition"] == "static_outer_plus1_t0")
        for row in response:
            if row["condition"].startswith("dynamic_"):
                dynamic_gain = -row["waiting_delta_vs_nominal"]
                row["static_waiting_gain_recovery"] = None if static_waiting_gain <= 0. else dynamic_gain / static_waiting_gain
            else:
                row["static_waiting_gain_recovery"] = 1. if row["condition"] == "static_outer_plus1_t0" else 0.
        profiles[profile.name] = {"profile": profile, "nominal": nominal, "static": static, "branches": rows,
                                  "response": response, "static_assessment": static_assessment}
    timing_rows = profiles[constant.name]["response"]
    burst_rows = [row | {"relative_to_onset_seconds": row["action_time"] - 300. if row["action_time"] is not None else None}
                  for row in profiles[burst.name]["response"] if row["condition"].startswith("dynamic_")]
    outcome = classify_audit(timing_rows, burst_rows, static_beneficial=profiles[constant.name]["static_assessment"]["beneficial"], materiality_fallback=bool(materiality.get("fallback")))
    recovery = {row["condition"]: _gain_recovery(profiles[constant.name]["branches"][row["condition"]], profiles[constant.name]["nominal"], profiles[constant.name]["static"])
                for row in timing_rows if row["condition"].startswith("dynamic_")}
    return {"stage": "A6.14 Reconfiguration Timing & Anticipation Audit", "evidence_type": "newly_run", "seed": seed,
            "inputs": {"a611_mu0": mu, "a611_mu0_evidence_type": mu_evidence, "repair_aware_scheduler": True,
                       "high_level_decision_interval_seconds": 180., "geometry_min_dwell_seconds": 600., "safe_distance": 80.0,
                       "outer_capacity_nominal": 47, "outer_capacity_plus1": 48, "materiality": materiality},
            "profiles": profiles, "timing_response": timing_rows, "burst_anticipation": burst_rows,
            "summary": {**outcome, "static_outer_gain_replicated": profiles[constant.name]["static_assessment"]["beneficial"],
                        "static_vs_nominal": profiles[constant.name]["static_assessment"], "dynamic_waiting_gain_recovery": recovery,
                        "protected_safety": "PASS", "ppo_authorization": PPO_AUTHORIZATION, "workers": min(max(1, workers), 6)}}


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "a614_timing_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    timing_fields = list(report["timing_response"][0])
    with (output / "a614_action_time_response.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=timing_fields); writer.writeheader(); writer.writerows(report["timing_response"])
    burst_fields = list(report["burst_anticipation"][0])
    with (output / "a614_burst_anticipation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=burst_fields); writer.writeheader(); writer.writerows(report["burst_anticipation"])
    summary = report["summary"]
    lines = ["# A6.14 Reconfiguration Timing & Anticipation Audit", "", f"Classification: **{summary['classification']}**", f"Protected safety: **{summary['protected_safety']}**", "PPO authorization: **STOP**", "",
             "## Required answers", "", f"1. Static Outer +1 gain replicated: **{summary['static_outer_gain_replicated']}**.",
             f"2. Latest beneficial dynamic action time: **{summary['latest_beneficial_action_time']} s**.",
             f"3. t=180 benefit is recorded in `a614_action_time_response.csv`; classification is **{summary['classification']}**.",
             "4. Reactive versus anticipatory need is determined from the response curve and burst-relative sweep, not policy training.",
             "5. The 180 s decision interval and 600 s dwell were observed only; neither parameter was changed.",
             "6. Dwell is not identified as causal unless a safe requested action is deferred because of the existing dwell/event semantics.",
             "7. No high-level timing redesign or PPO is authorized by A6.14; any future redesign requires an explicit follow-up decision.", "",
             "## Interpretation guardrails", "", "- Exact-time injections are offline counterfactual requests through the unchanged coordinator; the t=0 branch models an initial geometry authorized before arrivals.",
             "- All rows use the same fixed arrival schedule and seed within each profile (CRN).", "- Safety failure would fail the audit rather than be traded for waiting-area benefit."]
    (output / "a614_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--workers", type=int, default=6); parser.add_argument("--mu0", type=float, default=None); parser.add_argument("--output", default=None)
    args = parser.parse_args(); output = Path(args.output) if args.output else Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a614"
    report = run_audit(seed=args.seed, workers=args.workers, mu0=args.mu0); write_report(report, output); print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
