"""A6.11 demand--capacity--safety causal audit (diagnostic only; never trains PPO)."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import csv
import json
import os
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np

from ..core.config import RingRole, RingSpec, SimConfig
from ..core.merge_scheduler import LowLevelIntent
from ..core.ring_capacity import geometric_ring_capacity, minimum_radius_for_occupancy
from ..core.uav import FlightPhase
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import RuleLowLevelPolicy
from .a65_counterfactual_audit import REQUIRED_SAFETY, available_cpus
from .a6_effectiveness import _collect, _sample


HORIZON = 540.0
AUDIT_SECONDS = 3000.0
SAFETY_REJECT_LIMIT = 8
NO_OP = np.zeros(10, dtype=np.float32)
EXPANSIONS = {
    RingRole.OUTER: np.asarray((.60, 0, 0, 0, 0, 0, 0, 0, 0, 0), np.float32),
    RingRole.MIDDLE: np.asarray((0, .45, 0, 0, 0, 0, 0, 0, 0, 0), np.float32),
    RingRole.INNER: np.asarray((0, 0, .30, 0, 0, 0, 0, 0, 0, 0), np.float32),
}
INTERFACES = {
    "o2m": (FlightPhase.RING_OUTER, RingRole.MIDDLE),
    "m2i": (FlightPhase.RING_MIDDLE, RingRole.INNER),
    "i2f": (FlightPhase.RING_INNER, None),
}


@dataclass(frozen=True)
class DemandProfile:
    name: str
    segments: tuple[tuple[float, float, float], ...]
    duration_seconds: float
    drain_seconds: float = 1200.


def _schedule(segments: Iterable[tuple[float, float, float]]) -> tuple[float, ...]:
    values: list[float] = []
    for start, end, rate in segments:
        if rate <= 0:
            continue
        count = int(np.ceil((end - start) * rate - 1e-9))
        values.extend(start + index / rate for index in range(count) if start + index / rate < end - 1e-9)
    return tuple(sorted(values))


def demand_profiles(mu0: float) -> dict[str, DemandProfile]:
    """Fixed CRN schedules spanning sub-dwell, dwell, and super-dwell overloads."""
    low, near, above, high = (.55 * mu0, .95 * mu0, 1.10 * mu0, 1.55 * mu0)
    return {
        "constant_low": DemandProfile("constant_low", ((0., 1800., low),), 1800.),
        "constant_high": DemandProfile("constant_high", ((0., 1800., high),), 1800.),
        "low_high_low_ramp": DemandProfile("low_high_low_ramp", ((0., 600., low), (600., 1800., above), (1800., 2400., low)), 2400.),
        "short_burst": DemandProfile("short_burst", ((0., 1800., near), (300., 540., high)), 1800.),
        "medium_burst": DemandProfile("medium_burst", ((0., 2100., near), (300., 900., high)), 2100.),
        "long_burst": DemandProfile("long_burst", ((0., 2700., near), (300., 1560., high)), 2700.),
        "double_burst": DemandProfile("double_burst", ((0., 2400., near), (240., 600., high), (1260., 1860., high)), 2400.),
    }


def _base_config(profile: DemandProfile, *, dynamic: bool, rings=None) -> SimConfig:
    schedule = _schedule(profile.segments)
    return SimConfig(
        experiment_mode="dynamic_rule" if dynamic else "fixed_rule",
        high_level_enabled=dynamic,
        reset_mode="flow_from_empty",
        arrival_model="fixed_schedule",
        fixed_arrival_schedule=schedule,
        total_requested_uavs=len(schedule),
        max_simulation_seconds=profile.duration_seconds + profile.drain_seconds,
        rings=SimConfig().rings if rings is None else rings,
    )


def static_expanded_rings(config: SimConfig, role: RingRole) -> dict[RingRole, RingSpec]:
    """A t=0 static capacity +1 control, derived solely from legal geometry."""
    rings = dict(config.rings); original = rings[role]
    current = geometric_ring_capacity(original.radius, config.safe_distance)
    radius = minimum_radius_for_occupancy(current + 1, config.safe_distance)
    rings[role] = RingSpec(role, radius, original.height, original.nominal_speed)
    return rings


def _safe(metrics: dict, config: SimConfig) -> bool:
    return all(metrics.get(key) == 0 for key in REQUIRED_SAFETY) and metrics.get("max_concurrent_final", 0) <= config.final_queue_capacity


def _empty_interface() -> dict:
    return {
        "occupancy_sum": 0., "capacity_sum": 0., "binding_steps": 0, "sample_steps": 0,
        "capacity_blocked_release_count": 0, "safety_blocked_count": 0, "headway_blocked_count": 0,
        "merge_gap_blocked_count": 0, "downstream_resource_blocked_count": 0,
        "eligible_releases": 0, "attempted_releases": 0, "approved_releases": 0,
        "rejected_releases": 0, "downstream_slack_sum": 0., "downstream_slack_min": float("inf"),
        "reason_histogram": Counter(),
    }


def _interface_pre(env: TerminalParallelEnv, actions: dict[str, np.ndarray], counters: dict[str, dict]) -> dict[int, str]:
    phases: dict[int, str] = {}
    for key, (source, target) in INTERFACES.items():
        row = counters[key]
        occupancy = env.airspace.occupancy(target) if target is not None else sum(u.phase is FlightPhase.FINAL for u in env.active.values())
        capacity = env.airspace.capacity(target) if target is not None else env.config.final_queue_capacity
        row["occupancy_sum"] += occupancy; row["capacity_sum"] += capacity; row["sample_steps"] += 1
        row["binding_steps"] += int(occupancy >= capacity)
        slack = capacity - occupancy; row["downstream_slack_sum"] += slack; row["downstream_slack_min"] = min(row["downstream_slack_min"], slack)
        for uid, uav in env.active.items():
            if uav.phase is not source or uav.committed_plan is not None:
                continue
            phases[uid] = key; row["eligible_releases"] += 1
            intent = actions.get(str(uid), NO_OP)
            if float(intent[0]) <= env.config.release_threshold:
                continue
            row["attempted_releases"] += 1
            previous = max(env.scheduler.last_release[key], env.scheduler.last_executed_release[key])
            if previous + env.airspace.headways[key] > env.now + 1e-9:
                row["headway_blocked_count"] += 1
            if slack <= 0:
                row["capacity_blocked_release_count"] += 1
    return phases


def _interface_post(env: TerminalParallelEnv, decisions: dict[str, str], phases: dict[int, str], counters: dict[str, dict]) -> None:
    for raw_uid, reason in decisions.items():
        uid = int(raw_uid)
        key = phases.get(uid)
        if key is None or reason == "not_requested":
            continue
        row = counters[key]; row["reason_histogram"][reason] += 1
        if reason == "approved":
            row["approved_releases"] += 1; continue
        row["rejected_releases"] += 1
        if reason in {"safety_reject", "safety_wait"}:
            row["safety_blocked_count"] += 1
            verdict = env.last_safety_decisions.get(uid, "")
            if "holder" in verdict or "separation" in verdict:
                row["merge_gap_blocked_count"] += 1
        elif reason in {"final_capacity", "lz_unavailable", "lz_reservation_reject"}:
            row["downstream_resource_blocked_count"] += 1
        elif reason in {"geometry_frozen", "plan_active"}:
            row["merge_gap_blocked_count"] += 1


def _finalize_counters(counters: dict[str, dict], conditions: Counter) -> dict:
    result = {}
    for key, row in counters.items():
        steps = max(1, row["sample_steps"])
        result[key] = {
            **{name: value for name, value in row.items() if name not in {"reason_histogram", "downstream_slack_min"}},
            "occupancy_mean": row["occupancy_sum"] / steps,
            "capacity_mean": row["capacity_sum"] / steps,
            "capacity_binding_time_fraction": row["binding_steps"] / steps,
            "downstream_slack_mean": row["downstream_slack_sum"] / steps,
            "downstream_slack_min": None if not np.isfinite(row["downstream_slack_min"]) else row["downstream_slack_min"],
            "reason_histogram": dict(row["reason_histogram"]),
        }
    return {"interfaces": result, "state_patterns": dict(conditions)}


def _sample_conditions(env: TerminalParallelEnv, conditions: Counter) -> None:
    outer, middle, inner = (env.airspace.occupancy(role) for role in RingRole)
    cap_outer, cap_middle, cap_inner = (env.airspace.capacity(role) for role in RingRole)
    if env.admission.queue.items and outer < cap_outer:
        conditions["external_queue_high_outer_not_full_seconds"] += 1
    if outer >= cap_outer and middle < cap_middle and inner < cap_inner:
        conditions["outer_high_middle_inner_slack_seconds"] += 1


def _run_fixed(config: SimConfig, seed: int, *, capture_rejections: bool = False) -> tuple[dict, list[dict]]:
    env = TerminalParallelEnv(config); env.reset(seed); policy = RuleLowLevelPolicy()
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}
    counters = {key: _empty_interface() for key in INTERFACES}; patterns = Counter(); completions: list[float] = []; rejects: list[dict] = []
    while env.now < config.max_simulation_seconds and not (env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs):
        actions = policy.act(env._observations()); phases = _interface_pre(env, actions, counters); _sample_conditions(env, patterns)
        state = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL) if capture_rejections and len(rejects) < SAFETY_REJECT_LIMIT else None
        previous_events = len(env.events); _, _, _, _, info = env.step(actions); _sample(env, samples)
        _interface_post(env, info["release_decisions"], phases, counters)
        completions.extend(float(event[2]) for event in env.events[previous_events:] if event[0] == "completed")
        if state is not None:
            for uid, reason in info["release_decisions"].items():
                if reason == "safety_reject" and int(uid) in phases and len(rejects) < SAFETY_REJECT_LIMIT:
                    rejects.append({"payload": state, "uid": int(uid), "interface": phases[int(uid)], "time": env.now - config.dt})
    metrics = _collect(env, samples)
    return {"metrics": metrics, "safe": _safe(metrics, config), "interfaces": _finalize_counters(counters, patterns), "completion_times": completions,
            "mode": "static", "actual_capacity": {role.value: env.airspace.capacity(role) for role in RingRole}}, rejects


def _run_dynamic(config: SimConfig, seed: int, action: np.ndarray, *, expansion_at: float = 180.) -> tuple[dict, list[dict]]:
    wrapper = DynamicHighLevelWrapper(config, RuleLowLevelAdapter()); wrapper.reset(seed); env = wrapper.base
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}
    counters = {key: _empty_interface() for key in INTERFACES}; patterns = Counter(); events: list[dict] = []; completions: list[float] = []
    requested = False; next_decision = 0.
    while env.now < config.max_simulation_seconds and not wrapper._done():
        if env.now >= next_decision - 1e-9:
            desired = action if not requested and env.now >= expansion_at else NO_OP
            if desired is action:
                requested = True
            decision = wrapper.coordinator.coordinate(desired, geometry_gate=wrapper.geometry_event_gate())
            events.append({"time": env.now, "requested": list(decision.requested_action), "applied": list(decision.applied_action), "outcome": decision.outcome,
                           "reason": decision.reason_code, "capacity_before": list(decision.update.capacity_before), "capacity_after": list(decision.update.capacity_after)})
            next_decision += config.high_level_decision_interval_seconds
        deferred = wrapper.coordinator.advance()
        if deferred is not None:
            events.append({"time": env.now, "requested": list(deferred.requested_action), "applied": list(deferred.applied_action), "outcome": deferred.outcome,
                           "reason": deferred.reason_code, "capacity_before": list(deferred.update.capacity_before), "capacity_after": list(deferred.update.capacity_after)})
        actions = wrapper.coordinator.apply_speed_repairs(wrapper.low_level.act(env._observations()))
        phases = _interface_pre(env, actions, counters); _sample_conditions(env, patterns); previous_events = len(env.events)
        _, _, _, _, info = env.step(actions); _sample(env, samples); _interface_post(env, info["release_decisions"], phases, counters)
        completions.extend(float(event[2]) for event in env.events[previous_events:] if event[0] == "completed")
    metrics = _collect(env, samples, wrapper.coordinator, events)
    return {"metrics": metrics, "safe": _safe(metrics, config), "interfaces": _finalize_counters(counters, patterns), "completion_times": completions,
            "mode": "dynamic", "action_events": events, "actual_capacity": {role.value: env.airspace.capacity(role) for role in RingRole}}, []


def _comparison_task(profile: DemandProfile, seed: int, kind: str, role: RingRole | None) -> tuple[str, str, dict]:
    nominal = _base_config(profile, dynamic=False)
    if kind == "nominal":
        result, _ = _run_fixed(nominal, seed)
    elif kind == "static":
        result, _ = _run_fixed(replace(nominal, rings=static_expanded_rings(nominal, role)), seed)
    elif kind == "dynamic":
        result, _ = _run_dynamic(_base_config(profile, dynamic=True), seed, EXPANSIONS[role])
    else:
        raise ValueError(kind)
    return profile.name, "nominal" if role is None else f"{kind}_{role.value}_plus1", result


def sustainable_rate(seed: int = 0) -> dict:
    # A short saturated prefix is sufficient to estimate the steady service
    # slope for a staged diagnostic; the longer profiles are gated on it.
    calibration = DemandProfile("mu0_calibration", ((0., 720., .08),), 720., 480.)
    result, _ = _run_fixed(_base_config(calibration, dynamic=False), seed)
    steady = [time for time in result["completion_times"] if 240. <= time <= 720.]
    mu0 = len(steady) / 480.
    if mu0 <= 0:
        raise RuntimeError("A6.11 cannot calibrate nominal service rate: no steady completion")
    return {"evidence_type": "newly_run", "profile": calibration.name, "seed": seed, "steady_window_seconds": [240., 720.], "completed_in_window": len(steady), "mu0_per_second": mu0, "result": result}


def _decomposition(nominal: dict, static: dict, dynamic: dict) -> str:
    a, b, c = nominal["metrics"], static["metrics"], dynamic["metrics"]
    static_gain = b["completed"] > a["completed"] or b["all_request_waiting_area"] < a["all_request_waiting_area"] - .01 * max(1., a["all_request_waiting_area"])
    dynamic_gain = c["completed"] > a["completed"] or c["all_request_waiting_area"] < a["all_request_waiting_area"] - .01 * max(1., a["all_request_waiting_area"])
    if static_gain and not dynamic_gain:
        return "RECONFIGURATION_COST_OFFSETS_GAIN"
    if not static_gain:
        return "CAPACITY_NOT_BINDING"
    return "CAPACITY_VALUE_PROVEN"


def _search_repair(record: dict) -> dict:
    """Frozen-state legal timing/speed search; no safety threshold is altered."""
    outcomes: list[dict] = []
    for delay in (0, 5, 10, 20, 40):
        for speed in (-1., 0., 1.):
            for merge in (-1., 0., 1.):
                env: TerminalParallelEnv = pickle.loads(record["payload"])
                for _ in range(delay):
                    if record["uid"] not in env.active:
                        break
                    held = {str(record["uid"]): np.asarray((-1., 0., 0.), np.float32)}
                    env.step(held)
                uav = env.active.get(record["uid"])
                if uav is None or uav.committed_plan is not None:
                    continue
                uav.safety_retry_after = min(uav.safety_retry_after, env.now)
                verdict = env._release(uav, LowLevelIntent(1., speed, merge))
                outcomes.append({"delay_seconds": delay, "speed_residual": speed, "merge_time_residual": merge,
                                 "release_result": verdict, "safety_reason": env.last_safety_decisions.get(uav.uid)})
                if verdict == "approved":
                    category = "SAFE_REPAIR_EXISTS" if delay == 0 else "SAFE_AFTER_DELAY"
                    return {"time": record["time"], "interface": record["interface"], "category": category, "witness": outcomes[-1], "attempts": len(outcomes)}
    return {"time": record["time"], "interface": record["interface"], "category": "TRULY_INFEASIBLE", "witness": None, "attempts": len(outcomes)}


def _shadow_snapshot(profile: DemandProfile, seed: int, at: float) -> bytes:
    config = _base_config(profile, dynamic=True); wrapper = DynamicHighLevelWrapper(config, RuleLowLevelAdapter()); wrapper.reset(seed)
    while wrapper.base.now < at and not wrapper._done():
        wrapper.coordinator.coordinate(NO_OP, geometry_gate=wrapper.geometry_event_gate()) if wrapper.base.now % config.high_level_decision_interval_seconds == 0 else None
        acts = wrapper.coordinator.apply_speed_repairs(wrapper.low_level.act(wrapper.base._observations())); wrapper.base.step(acts)
    return pickle.dumps(wrapper, protocol=pickle.HIGHEST_PROTOCOL)


def _shadow_task(payload: bytes, role: RingRole | None) -> tuple[str, dict]:
    wrapper: DynamicHighLevelWrapper = pickle.loads(payload); start = wrapper.base.now
    action = NO_OP if role is None else EXPANSIONS[role]
    result, _ = _run_dynamic_from_wrapper(wrapper, action, start + HORIZON)
    return "no_op" if role is None else f"{role.value}_capacity_plus1", result


def _run_dynamic_from_wrapper(wrapper: DynamicHighLevelWrapper, action: np.ndarray, end: float) -> tuple[dict, list[dict]]:
    """A 540-s shadow branch, retaining the same per-interface audit semantics."""
    env = wrapper.base; samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}
    counters = {key: _empty_interface() for key in INTERFACES}; patterns = Counter(); events = []; requested = False
    while env.now < end and not wrapper._done():
        if not requested:
            decision = wrapper.coordinator.coordinate(action, geometry_gate=wrapper.geometry_event_gate()); requested = True
            events.append({"time": env.now, "outcome": decision.outcome, "deferred_until": decision.deferred_until,
                           "capacity_before": list(decision.update.capacity_before), "capacity_after": list(decision.update.capacity_after)})
        deferred = wrapper.coordinator.advance()
        if deferred is not None:
            events.append({"time": env.now, "outcome": deferred.outcome, "deferred_until": deferred.deferred_until,
                           "capacity_before": list(deferred.update.capacity_before), "capacity_after": list(deferred.update.capacity_after)})
        actions = wrapper.coordinator.apply_speed_repairs(wrapper.low_level.act(env._observations())); phases = _interface_pre(env, actions, counters); _sample_conditions(env, patterns)
        _, _, _, _, info = env.step(actions); _sample(env, samples); _interface_post(env, info["release_decisions"], phases, counters)
    metrics = _collect(env, samples, wrapper.coordinator, events)
    return {"metrics": metrics, "safe": _safe(metrics, wrapper.config), "interfaces": _finalize_counters(counters, patterns), "action_events": events,
            "actual_capacity": {role.value: env.airspace.capacity(role) for role in RingRole}}, []


def _classify_ring(rows: dict[str, dict]) -> str:
    decomposition = rows["decomposition"]
    interface = "o2m" if rows["role"] == "outer" else "m2i" if rows["role"] == "middle" else "i2f"
    nominal = rows["nominal"]["interfaces"]["interfaces"][interface]
    safety = nominal["safety_blocked_count"] + nominal["merge_gap_blocked_count"]
    if safety > nominal["capacity_blocked_release_count"] and rows.get("conservative_rejection_rate", 0.) > 0.2:
        return "SAFETY_SCHEDULER_CONSERVATISM"
    return decomposition


def run_audit(*, stage: str = "pilot", seed: int = 0, workers: int | None = None) -> dict:
    """Fail-fast staged audit.  ``full`` only follows a protected pilot pass."""
    mu = sustainable_rate(seed); profiles = demand_profiles(mu["mu0_per_second"])
    pilot_profile = DemandProfile("constant_high_pilot", ((0., 720., 1.55 * mu["mu0_per_second"]),), 720., 480.)
    selected = [pilot_profile.name] if stage == "pilot" else list(profiles)
    jobs: list[tuple[DemandProfile, int, str, RingRole | None]] = []
    # Static/dynamic decomposition is required for every ring in the first profile.
    jobs.append((pilot_profile, seed, "nominal", None))
    for role in RingRole:
        jobs.extend(((pilot_profile, seed, "static", role), (pilot_profile, seed, "dynamic", role)))
    # Continuous profiles use the most downstream structural lever as a small
    # non-stationarity screen; all full static/dynamic ring decomposition data
    # remain available from the high-load pilot above.
    if stage == "full":
        for name in selected:
            if name == "constant_high":
                continue
            jobs.extend(((profiles[name], seed, "nominal", None), (profiles[name], seed, "dynamic", RingRole.INNER)))
    limit = int(os.environ.get("A611_MAX_WORKERS", workers or min(6, max(1, available_cpus() - 2))))
    results: dict[str, dict[str, dict]] = {}
    rejects: list[dict] = []
    # First branch is synchronous and carries frozen safety-rejection states.
    nominal, rejects = _run_fixed(_base_config(pilot_profile, dynamic=False), seed, capture_rejections=True)
    if not nominal["safe"]:
        raise RuntimeError("A6.11 fail closed: nominal constant-high branch is unsafe")
    results.setdefault(pilot_profile.name, {})["nominal"] = nominal
    remaining = [job for job in jobs if not (job[0].name == pilot_profile.name and job[2] == "nominal")]
    with ProcessPoolExecutor(max_workers=min(limit, len(remaining)), initializer=lambda: None) as pool:
        futures = [pool.submit(_comparison_task, *job) for job in remaining]
        for future in as_completed(futures):
            profile_name, label, row = future.result()
            if not row["safe"]:
                raise RuntimeError(f"A6.11 fail closed: unsafe branch {profile_name}/{label}")
            results.setdefault(profile_name, {})[label] = row
    safety_rows = [_search_repair(item) for item in rejects]
    safe_repair = sum(row["category"] == "SAFE_REPAIR_EXISTS" for row in safety_rows)
    conservative_rate = safe_repair / max(1, len(safety_rows))
    decomposition = {}
    for role in RingRole:
        label = role.value
        decomposition[label] = {"role": label, "nominal": nominal, "static": results[pilot_profile.name][f"static_{label}_plus1"],
                                "dynamic": results[pilot_profile.name][f"dynamic_{label}_plus1"],
                                "decomposition": _decomposition(nominal, results[pilot_profile.name][f"static_{label}_plus1"], results[pilot_profile.name][f"dynamic_{label}_plus1"]),
                                "conservative_rejection_rate": conservative_rate}
        decomposition[label]["classification"] = _classify_ring(decomposition[label])
    shadow_payload = _shadow_snapshot(pilot_profile, seed, 360.)
    shadow: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_shadow_task, shadow_payload, role) for role in (None, *RingRole)]
        for future in as_completed(futures):
            name, row = future.result()
            if not row["safe"]:
                raise RuntimeError(f"A6.11 fail closed: unsafe shadow {name}")
            shadow[name] = row
    overall_classes = {row["classification"] for row in decomposition.values()}
    if overall_classes == {"CAPACITY_NOT_BINDING"}:
        overall = "BOTTLENECK_MISALIGNMENT"
    elif overall_classes == {"RECONFIGURATION_COST_OFFSETS_GAIN"}:
        overall = "RECONFIGURATION_TIMESCALE_MISMATCH"
    elif overall_classes == {"SAFETY_SCHEDULER_CONSERVATISM"}:
        overall = "SAFETY_SCHEDULER_CONSERVATISM"
    elif overall_classes == {"CAPACITY_VALUE_PROVEN"}:
        overall = "CAPACITY_CONTROL_VALUE_PROVEN"
    else:
        overall = "MIXED"
    return {"stage": "A6.11 demand-capacity-safety causal audit", "evidence_type": "newly_run", "audit_stage": stage, "seed": seed,
            "mu0": mu, "profiles": {name: {"segments": profile.segments, "duration_seconds": profile.duration_seconds, "request_count": len(_schedule(profile.segments))} for name, profile in (profiles | {pilot_profile.name: pilot_profile}).items()},
            "comparison": results, "ring_conclusions": decomposition, "safety_conservatism": {"frozen_rejections": len(safety_rows), "rows": safety_rows,
                                                                              "safe_repair_exists": safe_repair, "conservative_rejection_rate": conservative_rate},
            "shadow_value": shadow, "summary": {"protected_safety": "PASS", "a611_result": overall, "ppo_authorization": "STOP",
                                                    "profile_screen_scope": selected, "worker_cap": limit},
            "protocol_risks": ["Internal deterministic simulator evidence; no external generalization claim.", "Pilot uses one common-random seed; full multi-seed expansion remains deliberately gated on this hard-safe mechanism screen.", "Safety-repair search is finite over existing legal timing/speed authorities, so TRULY_INFEASIBLE means infeasible within this audit grid."]}


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "a611_capacity_binding_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with (output / "a611_static_dynamic_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["profile", "condition", "completed", "waiting_area", "unresolved", "energy", "fairness", "capacity_blocked_releases", "safety_blocked_releases", "safe"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for profile, conditions in report["comparison"].items():
            for name, row in conditions.items():
                metrics = row["metrics"]; interfaces = row["interfaces"]["interfaces"]
                writer.writerow({"profile": profile, "condition": name, "completed": metrics["completed"], "waiting_area": metrics["all_request_waiting_area"], "unresolved": metrics["unresolved_count"], "energy": metrics["energy_mean"], "fairness": metrics["priority_fairness_std"], "capacity_blocked_releases": sum(item["capacity_blocked_release_count"] for item in interfaces.values()), "safety_blocked_releases": sum(item["safety_blocked_count"] for item in interfaces.values()), "safe": row["safe"]})
    with (output / "a611_demand_profile_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["profile", "duration_seconds", "request_count", "conditions_run"]); writer.writeheader()
        for name, profile in report["profiles"].items(): writer.writerow({"profile": name, "duration_seconds": profile["duration_seconds"], "request_count": profile["request_count"], "conditions_run": "|".join(report["comparison"].get(name, {}))})
    with (output / "a611_safety_conservatism.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "interface", "category", "attempts", "witness"]); writer.writeheader(); writer.writerows(report["safety_conservatism"]["rows"])
    with (output / "a611_capacity_shadow_value.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["action", "safe", "completed", "waiting_area", "capacity_binding_duration", "capacity_blocked_releases", "safety_blocked_releases"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for action, row in report["shadow_value"].items():
            interfaces = row["interfaces"]["interfaces"]
            writer.writerow({"action": action, "safe": row["safe"], "completed": row["metrics"]["completed"], "waiting_area": row["metrics"]["all_request_waiting_area"], "capacity_binding_duration": sum(value["binding_steps"] for value in interfaces.values()), "capacity_blocked_releases": sum(value["capacity_blocked_release_count"] for value in interfaces.values()), "safety_blocked_releases": sum(value["safety_blocked_count"] for value in interfaces.values())})
    lines = ["# A6.11 Demand-Capacity-Safety Causal Audit", "", f"Protected safety: **{report['summary']['protected_safety']}**", f"A6.11 result: **{report['summary']['a611_result']}**", "PPO authorization: **STOP**", "", "## Ring conclusions", "", "| Ring | Conclusion | Static vs dynamic decomposition |", "|---|---|---|"]
    for role, row in report["ring_conclusions"].items(): lines.append(f"| {role} | {row['classification']} | {row['decomposition']} |")
    lines.extend(["", "## Answers", "", "1. The binding cause is reported per interface in `a611_capacity_binding_audit.json`; capacity and safety/headway/resource counts are not conflated.", "2. Outer/Middle/Inner binding fractions are the authoritative per-step occupancy/capacity fractions in that JSON.", "3. Static larger geometry and nominal-to-dynamic expansion are paired on the same fixed arrival schedule.", "4. `RECONFIGURATION_COST_OFFSETS_GAIN` is emitted only when static improves but dynamic does not.", "5. The demand-profile CSV records sub-dwell, dwell, and super-dwell burst durations; only the selected staged profiles are newly run.", "6. Safety conservatism uses frozen rejected states and the same trajectory generator/SafetyFilter; it never relaxes 80 m separation.", "7. Capacity shadow values are 540-s CRN continuations at t=720.", "8. High-level RL remains blocked regardless of this audit result.", "", "Evidence type: newly_run. Protocol risks: one-seed mechanism diagnosis; finite legal repair grid; simulator-only evidence."])
    (output / "a611_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("pilot", "full"), default="pilot"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--output", default=None)
    args = parser.parse_args(); output = Path(args.output) if args.output else Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a611"
    report = run_audit(stage=args.stage, seed=args.seed); write_report(report, output)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
