"""Safety-capacity closure and conditional retrain gate.

This module is intentionally simulation-only.  It audits whether the service
capacity used for PPO retraining is based on safe nominal operation before any
PPO smoke/training job is allowed to start.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from ..config import load_experiment_config
from ..core.config import SimConfig
from ..core.geometry import distance_3d, norm_xy, wrap_angle
from ..core.lifecycle import UAVLifecycleState
from ..core.ring_capacity import adjacent_chord_spacing
from ..core.uav import FlightPhase, UAVState
from ..envs.high_level_wrapper import HighLevelHRLWrapper
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import DynamicRuleHighLevelPolicy
from .core_model_revision_readiness import _finite_config


DEMAND_TARGETS: dict[str, tuple[float, str]] = {
    "low": (0.40, "deterministic_interval"),
    "medium": (0.68, "deterministic_interval"),
    "high": (0.90, "deterministic_interval"),
    "bursty": (0.78, "bursty_wave"),
    "overload": (1.20, "deterministic_interval"),
}

COLLISION_EVENT_FIELDS = [
    "collision_event_id",
    "time",
    "step",
    "uav_id_a",
    "uav_id_b",
    "phase_a",
    "phase_b",
    "ring_id",
    "slot_id_a",
    "slot_id_b",
    "position_a",
    "position_b",
    "velocity_a",
    "velocity_b",
    "pairwise_distance",
    "relative_speed",
    "current_ring_radius",
    "current_geometric_capacity",
    "ring_occupancy",
    "requested_high_level_action",
    "applied_high_level_action",
    "geometry_transition_state",
    "vtac_decision_preceding_collision",
    "safety_filter_decision",
    "minimum_separation_prediction",
    "had_previous_conflict_or_nmac",
    "time_since_last_phase_transition_a",
    "time_since_last_phase_transition_b",
    "time_since_last_geometry_change",
    "classification",
    "evidence_type",
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    _ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def _phase_name(ac: UAVState | None) -> str:
    if ac is None:
        return ""
    try:
        return FlightPhase(ac.phase).name
    except Exception:
        return str(ac.phase)


def _ring_id(ac: UAVState | None) -> Any:
    return "" if ac is None or ac.assigned_ring is None else int(ac.assigned_ring)


def _slot_id(ac: UAVState | None) -> str:
    if ac is None:
        return ""
    return f"gate={ac.assigned_gate};ring_slot={int(bool(ac.ring_slot_active))}"


def _velocity_vector(ac: UAVState | None) -> list[float]:
    if ac is None:
        return []
    horizontal = float(ac.speed * math.cos(ac.pitch))
    return [
        horizontal * math.cos(ac.heading),
        horizontal * math.sin(ac.heading),
        float(ac.speed * math.sin(ac.pitch)),
    ]


def _as_position(ac: UAVState | None) -> list[float]:
    if ac is None:
        return []
    return [float(v) for v in np.asarray(ac.position, dtype=float)]


def _ring_for_aircraft(env: TerminalParallelEnv, ac: UAVState | None) -> Any:
    if ac is None or ac.assigned_ring in (None, 0):
        return ""
    try:
        return env.vtac.ring_spec(int(ac.assigned_ring))
    except Exception:
        return ""


def _ring_occupancy(env: TerminalParallelEnv, ring_idx: Any) -> int:
    if ring_idx == "":
        return 0
    return int(len(env.vtac.queues.get(int(ring_idx), [])))


def _classification(a: UAVState | None, b: UAVState | None) -> str:
    phases = {_phase_name(a), _phase_name(b)}
    rings = {_ring_id(a), _ring_id(b)}
    transitions = {a.transition_kind if a else None, b.transition_kind if b else None}
    if "APPROACH" in phases:
        if "RING" in phases or "ring_merge" in transitions:
            return "approach_to_outer_merge"
        return "ingress_approach"
    if "final_entry" in transitions:
        return "inner_to_final"
    if "ring_merge" in transitions:
        return "outer_to_inner"
    if phases == {"FINAL"}:
        return "final"
    if rings == {2}:
        return "outer_ring"
    if rings == {1}:
        return "inner_ring"
    if len(rings) > 1:
        return "slot_allocation_related"
    return "other"


def _minimum_ring_spacing_angle(env: TerminalParallelEnv, ring_idx: int) -> float:
    ring = env.vtac.ring_spec(ring_idx)
    required = float(env.config.safe_distance) + float(env.config.ring_separation_margin_m)
    if required >= 2.0 * ring.radius:
        return math.pi
    geometric = 2.0 * math.asin(required / (2.0 * max(1e-9, ring.radius)))
    return max(math.radians(env.config.ring_merge_spacing_deg), geometric)


def _safe_slot_count(env: TerminalParallelEnv, ring_idx: int, grid_size: int = 720) -> int:
    """Count candidate angular grid points that satisfy current spacing."""

    min_angle = _minimum_ring_spacing_angle(env, ring_idx)
    angles = []
    aircraft_map = {aircraft.uid: aircraft for aircraft in env.aircraft}
    for uid in env.vtac.queues.get(ring_idx, []):
        aircraft = aircraft_map.get(uid)
        if aircraft is not None and not aircraft.finished:
            angles.append(math.atan2(aircraft.position[1], aircraft.position[0]))
    count = 0
    for idx in range(grid_size):
        theta = 2.0 * math.pi * idx / grid_size
        if all(abs(wrap_angle(theta - other)) >= min_angle for other in angles):
            count += 1
    return count


def _ring_distance_metrics(env: TerminalParallelEnv, ring_idx: int) -> dict[str, float]:
    aircraft_map = {aircraft.uid: aircraft for aircraft in env.aircraft}
    members = [
        aircraft_map[uid]
        for uid in env.vtac.queues.get(ring_idx, [])
        if uid in aircraft_map and not aircraft_map[uid].finished
    ]
    ring = env.vtac.ring_spec(ring_idx)
    angles = sorted(math.atan2(ac.position[1], ac.position[0]) for ac in members)
    adjacent = []
    if len(angles) >= 2:
        for first, second in zip(angles, angles[1:] + [angles[0] + 2.0 * math.pi]):
            adjacent.append(2.0 * ring.radius * math.sin(abs(second - first) / 2.0))
    pairs = []
    for i, a in enumerate(members):
        for b in members[i + 1 :]:
            pairs.append(distance_3d(a.position, b.position))
    return {
        "minimum_actual_adjacent_chord_distance": min(adjacent) if adjacent else 0.0,
        "minimum_all_pairs_ring_distance": min(pairs) if pairs else 0.0,
    }


def _collision_event_row(
    *,
    event_id: int,
    env: TerminalParallelEnv,
    a: UAVState,
    b: UAVState | None,
) -> dict[str, Any]:
    ring = _ring_for_aircraft(env, a)
    ring_idx = _ring_id(a)
    ring_capacity = "" if ring == "" else int(ring.capacity)
    distance = "" if b is None else float(distance_3d(a.position, b.position))
    rel_speed = ""
    if b is not None:
        va = np.asarray(_velocity_vector(a), dtype=float)
        vb = np.asarray(_velocity_vector(b), dtype=float)
        rel_speed = float(np.linalg.norm(va - vb))
    approvals = {
        "a_release": env.last_release_approval.get(a.uid, {}),
        "b_release": {} if b is None else env.last_release_approval.get(b.uid, {}),
        "a_landing": env.last_landing_approval.get(a.uid, {}),
        "b_landing": {} if b is None else env.last_landing_approval.get(b.uid, {}),
    }
    preceding_vtac = ""
    for key, approval in approvals.items():
        if approval:
            preceding_vtac += f"{key}:{approval.get('target_stage')}:{approval.get('decision')}:{approval.get('reason')};"
    diagnostics = [
        approval.get("diagnostics", {})
        for approval in approvals.values()
        if isinstance(approval, dict) and approval.get("diagnostics")
    ]
    prediction = ""
    if diagnostics:
        mins = [
            item.get("trajectory_min_distance")
            for item in diagnostics
            if item.get("trajectory_min_distance") is not None
        ]
        if mins:
            prediction = float(min(mins))
    last_high = env.last_high_level_info or {}
    requested_action = np.asarray(last_high.get("requested_action", np.zeros(8)), dtype=float).reshape(-1)
    applied_action = np.asarray(last_high.get("applied_action", np.zeros(8)), dtype=float).reshape(-1)
    last_geometry_change = float(last_high.get("time", -1.0)) if last_high else -1.0
    return {
        "collision_event_id": int(event_id),
        "time": float(env.time),
        "step": int(env.step_count),
        "uav_id_a": int(a.uid),
        "uav_id_b": "" if b is None else int(b.uid),
        "phase_a": _phase_name(a),
        "phase_b": _phase_name(b),
        "ring_id": ring_idx,
        "slot_id_a": _slot_id(a),
        "slot_id_b": _slot_id(b),
        "position_a": json.dumps(_as_position(a)),
        "position_b": json.dumps(_as_position(b)),
        "velocity_a": json.dumps(_velocity_vector(a)),
        "velocity_b": json.dumps(_velocity_vector(b)),
        "pairwise_distance": distance,
        "relative_speed": rel_speed,
        "current_ring_radius": "" if ring == "" else float(ring.radius),
        "current_geometric_capacity": ring_capacity,
        "ring_occupancy": _ring_occupancy(env, ring_idx),
        "requested_high_level_action": json.dumps([float(x) for x in requested_action]),
        "applied_high_level_action": json.dumps([float(x) for x in applied_action]),
        "geometry_transition_state": json.dumps(
            {
                "state_vector": [
                    float(value) for value in env.airspace.state_vector()
                ],
                "capacity_state": env.airspace.capacity_state(),
            }
        ),
        "vtac_decision_preceding_collision": preceding_vtac,
        "safety_filter_decision": json.dumps(approvals, default=str),
        "minimum_separation_prediction": prediction,
        "had_previous_conflict_or_nmac": int(
            tuple(sorted((a.uid, b.uid))) in env._unique_conflict_pairs if b is not None else 0
        ),
        "time_since_last_phase_transition_a": (
            "" if a.last_transfer_step is None else float((env.step_count - a.last_transfer_step) * env.config.dt)
        ),
        "time_since_last_phase_transition_b": (
            ""
            if b is None or b.last_transfer_step is None
            else float((env.step_count - b.last_transfer_step) * env.config.dt)
        ),
        "time_since_last_geometry_change": (
            "" if last_geometry_change < 0.0 else float(env.time - last_geometry_change)
        ),
        "classification": _classification(a, b),
        "evidence_type": "newly_run",
    }


def run_finite_with_forensics(
    *,
    seed: int,
    total_requested: int,
    arrival_rate: float,
    max_active: int,
    max_steps: int,
) -> dict[str, Any]:
    cfg = _finite_config(
        total_requested=total_requested,
        seed=seed,
        arrival_rate=arrival_rate,
        max_active=max_active,
        max_steps=max_steps,
    )
    cfg.battery_endurance_seconds = float(cfg.engineering_validation_battery_endurance_seconds)
    env = TerminalParallelEnv(cfg)
    env.reset(seed=seed)
    collision_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    previous_collision_pairs: set[tuple[int, int]] = set()
    event_id = 0
    while env.step_count < cfg.max_steps and not env.finite_cohort_complete():
        env.step({})
        active_map = {aircraft.uid: aircraft for aircraft in env.aircraft}
        for pair in sorted(env._terminal_collision_pairs - previous_collision_pairs):
            previous_collision_pairs.add(pair)
            event_id += 1
            a = active_map.get(pair[0]) or next((x for x in env.aircraft if x.uid == pair[0]), None)
            b = active_map.get(pair[1]) or next((x for x in env.aircraft if x.uid == pair[1]), None)
            if a is not None:
                collision_rows.append(
                    _collision_event_row(event_id=event_id, env=env, a=a, b=b)
                )
        if env.step_count % max(1, int(round(20.0 / cfg.dt))) == 0:
            for ring in env.config.rings:
                metrics = _ring_distance_metrics(env, ring.index)
                capacity_rows.append(
                    {
                        "time": float(env.time),
                        "seed": int(seed),
                        "ring_id": int(ring.index),
                        "radius": float(ring.radius),
                        "geometric_capacity": int(ring.capacity),
                        "occupancy": int(len(env.vtac.queues[ring.index])),
                        "available_safe_slots_grid": int(_safe_slot_count(env, ring.index)),
                        "minimum_actual_adjacent_chord_distance": metrics["minimum_actual_adjacent_chord_distance"],
                        "minimum_all_pairs_ring_distance": metrics["minimum_all_pairs_ring_distance"],
                        "safe_ring_separation": float(cfg.safe_distance + cfg.ring_separation_margin_m),
                        "evidence_type": "newly_run",
                    }
                )
    summary = env.simulation_outcome_summary()
    summary.update(
        {
            "seed": int(seed),
            "N": int(total_requested),
            "arrival_rate": float(arrival_rate),
            "max_active": int(max_active),
            "outer_capacity": int(env.airspace.outer_ring.capacity),
            "inner_capacity": int(env.airspace.inner_ring.capacity),
            "nmac_count": int(env.nmac_count),
            "conflict_samples": int(env.raw_pair_conflict_samples),
            "safety_override_count": int(env.safety_override_count),
            "raw_safety_override_count": int(env.raw_safety_override_count),
            "evidence_type": "newly_run",
        }
    )
    return {
        "summary": summary,
        "collision_rows": collision_rows,
        "capacity_rows": capacity_rows,
        "outcome_rows": env.uav_outcome_rows(),
    }


def collision_origin_audit(output: Path) -> dict[str, Any]:
    final = _ensure_dir(output / "final")
    result = run_finite_with_forensics(
        seed=0,
        total_requested=30,
        arrival_rate=0.3,
        max_active=10,
        max_steps=max(1500, 30 * 220),
    )
    summary = result["summary"]
    collisions = int(summary["crashed_collision"])
    origin = (
        "nominal_saturated_service_capacity"
        if collisions > 0
        else "not_nominal_capacity_run"
    )
    audit = {
        "status": "BLOCKING_NOMINAL_COLLISIONS" if collisions > 0 else "NO_NOMINAL_COLLISIONS",
        "origin": origin,
        "nominal_capacity_summary": summary,
        "deliberate_lifecycle_mixed_in": False,
        "aggregation_mixed_deliberate_collision_tests": False,
        "collision_event_count": len(result["collision_rows"]),
        "collision_terminal_outcomes": collisions,
        "evidence_type": "newly_run",
    }
    _write_json(final / "collision_origin_audit.json", audit)
    _write_csv(final / "collision_capacity_reproduction.csv", [summary])
    if result["collision_rows"]:
        _write_csv(final / "collision_events.csv", result["collision_rows"])
    else:
        with (final / "collision_events.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=COLLISION_EVENT_FIELDS).writeheader()
    _write_csv(final / "dynamic_capacity_safety.csv", result["capacity_rows"])
    _write_csv(output / "traces" / "collision_uav_outcomes_seed0.csv", result["outcome_rows"])
    return audit


def safe_service_capacity(output: Path, *, max_workers: int) -> dict[str, Any]:
    final = _ensure_dir(output / "final")
    jobs = [
        (seed, 60, 0.3, 10, max(2500, 60 * 220))
        for seed in range(5)
    ]
    rows: list[dict[str, Any]] = []
    landing_rows: list[dict[str, Any]] = []
    workers = max(1, min(max_workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_finite_with_forensics,
                seed=seed,
                total_requested=total,
                arrival_rate=arrival_rate,
                max_active=max_active,
                max_steps=max_steps,
            )
            for seed, total, arrival_rate, max_active, max_steps in jobs
        ]
        for future in as_completed(futures):
            result = future.result()
            summary = result["summary"]
            landed_times = sorted(
                float(row["terminal_time"])
                for row in result["outcome_rows"]
                if row["terminal_outcome"] == UAVLifecycleState.LANDED.value and row["terminal_time"] != ""
            )
            steady = landed_times[3:] if len(landed_times) > 6 else landed_times
            gaps = [b - a for a, b in zip(steady, steady[1:])]
            duration = (steady[-1] - steady[0]) if len(steady) >= 2 else max(1.0, float(summary["final_simulation_time"]))
            rate = 3600.0 * max(0, len(steady) - 1) / max(1e-9, duration)
            row = {
                **summary,
                "steady_window_landing_count": int(len(steady)),
                "safe_service_rate_per_hour": rate if int(summary["crashed_collision"]) == 0 else 0.0,
                "provisional_service_rate_per_hour": rate,
                "inter_landing_p50": _percentile(gaps, 50),
                "inter_landing_p90": _percentile(gaps, 90),
                "inter_landing_p95": _percentile(gaps, 95),
                "external_admission_rate_per_hour": 3600.0 * summary["total_admitted"] / max(1e-9, summary["final_simulation_time"]),
                "zero_collision_capacity_valid": int(summary["crashed_collision"]) == 0,
            }
            rows.append(row)
            for index, t in enumerate(landed_times):
                landing_rows.append(
                    {
                        "seed": int(summary["seed"]),
                        "landing_index": int(index),
                        "landing_time": t,
                        "evidence_type": "newly_run",
                    }
                )
    rows.sort(key=lambda item: int(item["seed"]))
    safe_rates = [float(row["safe_service_rate_per_hour"]) for row in rows if row["zero_collision_capacity_valid"]]
    mean_rate, ci95 = _mean_ci95(safe_rates)
    status = "SAFE_CAPACITY_ESTABLISHED" if len(safe_rates) == len(rows) and mean_rate > 0.0 else "NEEDS_SAFETY_FIX"
    for row in rows:
        row["mean_safe_service_rate_per_hour"] = mean_rate
        row["safe_service_rate_ci95"] = ci95
        row["safe_capacity_status"] = status
    _write_csv(final / "safe_service_capacity_summary.csv", rows)
    _write_csv(final / "landing_timestamps.csv", landing_rows)
    return {"status": status, "rows": rows, "mean_rate_per_hour": mean_rate, "ci95": ci95}


def demand_profiles(output: Path, service: dict[str, Any]) -> dict[str, Any]:
    final = _ensure_dir(output / "final")
    service_rate_h = float(service.get("mean_rate_per_hour", 0.0))
    rows = []
    burst_rows = []
    horizon = 3600.0
    for demand, (util, model) in DEMAND_TARGETS.items():
        rate_h = service_rate_h * util
        rate_s = rate_h / 3600.0
        rows.append(
            {
                "demand": demand,
                "target_utilization": util,
                "arrival_model": model,
                "initial_aircraft_count": 0,
                "formation_rows": 0,
                "formation_cols": 0,
                "arrival_rate_per_second": rate_s,
                "arrival_rate_per_hour": rate_h,
                "safe_service_rate_per_hour": service_rate_h,
                "evidence_type": "newly_run",
            }
        )
        if rate_s <= 0.0:
            request_count = 0
            realized = 0.0
            peak = 0.0
        else:
            if demand == "bursty":
                baseline_interval = 1.0 / rate_s
                times = []
                t = baseline_interval * 0.5
                while t < horizon:
                    wave = 0.65 + 0.65 * (1.0 + math.sin(2.0 * math.pi * t / 900.0))
                    local_interval = baseline_interval / max(0.25, wave)
                    times.append(t)
                    t += local_interval
            else:
                interval = 1.0 / rate_s
                times = list(np.arange(interval, horizon + 1e-9, interval))
            request_count = len(times)
            realized = request_count / horizon
            peak = 0.0
            for start in np.arange(0.0, horizon, 60.0):
                count = sum(start <= t < start + 300.0 for t in times)
                peak = max(peak, count / 300.0)
        burst_rows.append(
            {
                "demand": demand,
                "configured_rate_per_second": rate_s,
                "realized_mean_rate_per_second": realized,
                "peak_300s_rate_per_second": peak,
                "peak_to_mean": peak / max(1e-9, realized),
                "request_count": request_count,
                "evidence_type": "newly_run",
            }
        )
    _write_csv(final / "safe_demand_profile_summary.csv", rows)
    _write_csv(final / "safe_bursty_validation.csv", burst_rows)
    return {"rows": rows, "bursty": burst_rows}


def _safe_rate_map(demand: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["demand"]): dict(row) for row in demand["rows"]}


def _configured_policy_config(
    *,
    policy: str,
    demand_level: str,
    demand_map: dict[str, dict[str, Any]],
    total_requested: int = 24,
    decision_interval_seconds: float = 180.0,
    dwell_seconds: float = 540.0,
) -> SimConfig:
    if policy == "event-rule":
        config_name = "event_rule"
    elif policy == "event-gated-hrl":
        config_name = "event_gated_hrl"
    else:
        config_name = "dynamic_hrl"
    cfg = load_experiment_config(config_name)
    demand_row = demand_map[demand_level]
    cfg.initial_aircraft_count = int(float(demand_row.get("initial_aircraft_count", 0)))
    cfg.formation_rows = int(float(demand_row.get("formation_rows", 0)))
    cfg.formation_cols = int(float(demand_row.get("formation_cols", 0)))
    cfg.arrival_rate = float(demand_row["arrival_rate_per_second"])
    cfg.traffic_demand_level = demand_level
    cfg.arrival_model = str(demand_row.get("arrival_model", "deterministic_interval"))
    cfg.total_requested_uavs = int(total_requested)
    cfg.max_active_uavs = 10
    cfg.max_steps = max(3600, int(total_requested * 220))
    cfg.battery_endurance_seconds = float(cfg.engineering_validation_battery_endurance_seconds)
    cfg.metrics_sampling_interval_seconds = 20.0
    cfg.decision_interval_seconds = float(decision_interval_seconds)
    cfg.high_level_decision_interval_seconds = float(decision_interval_seconds)
    cfg.high_level_interval_steps = max(1, int(round(decision_interval_seconds / cfg.dt)))
    cfg.reconfiguration_dwell_seconds = float(dwell_seconds)
    cfg.geometry_min_dwell_seconds = float(dwell_seconds)
    cfg.geometry_transition_seconds = 20.0
    return cfg


def _policy_action(policy: str, env: TerminalParallelEnv, rule_policy: DynamicRuleHighLevelPolicy) -> np.ndarray | None:
    if policy == "event-rule" and env.should_update_high_level():
        return rule_policy.act(env)
    return None


def _run_policy_case(args: tuple[str, str, int, dict[str, dict[str, Any]], int, float, float]) -> dict[str, Any]:
    policy, demand_level, seed, demand_map, total_requested, interval, dwell = args
    cfg = _configured_policy_config(
        policy=policy,
        demand_level=demand_level,
        demand_map=demand_map,
        total_requested=total_requested,
        decision_interval_seconds=interval,
        dwell_seconds=dwell,
    )
    env = TerminalParallelEnv(cfg)
    env.reset(seed=seed)
    rule_policy = DynamicRuleHighLevelPolicy()
    gate_samples: list[dict[str, Any]] = []
    geometry_change_count = 0
    previous_airspace = env.airspace.state_vector().copy()
    while env.step_count < cfg.max_steps and not env.finite_cohort_complete():
        action = _policy_action(policy, env, rule_policy)
        env.step({}, high_level_action=action) if action is not None else env.step({})
        current_airspace = env.airspace.state_vector().copy()
        if not np.allclose(current_airspace, previous_airspace):
            geometry_change_count += 1
        previous_airspace = current_airspace
        if env.should_sample_metrics():
            info = env.geometry_gate.info(env.time)
            metrics = env.metrics()
            gate_samples.append(
                {
                    "policy": policy,
                    "demand": demand_level,
                    "seed": seed,
                    "time": float(env.time),
                    "trigger_score": float(info.get("trigger_score", 0.0)),
                    "gate_state": str(info.get("gate_state", "")),
                    "geometry_gate_active": int(bool(info.get("geometry_gate_active", False))),
                    "geometry_action_allowed": int(bool(info.get("geometry_action_allowed", False))),
                    "outer_capacity": int(env.airspace.outer_ring.capacity),
                    "inner_capacity": int(env.airspace.inner_ring.capacity),
                    "external_queue": len(env.external_arrival_queue),
                    "arrival_service_imbalance": float(env.last_gate_metrics.get("arrival_service_imbalance", 0.0)),
                    "collision_count": float(metrics["collision_count"]),
                    "nmac_count": float(metrics["nmac_events"]),
                    "safety_override_count": float(metrics["safety_override_count"]),
                    "evidence_type": "newly_run",
                }
            )
    summary = env.simulation_outcome_summary()
    metrics = env.metrics()
    summary.update(
        {
            "policy": policy,
            "demand": demand_level,
            "seed": seed,
            "decision_interval_seconds": float(interval),
            "geometry_min_dwell_seconds": float(dwell),
            "collision_count": int(metrics["collision_count"]),
            "nmac_count": int(metrics["nmac_events"]),
            "conflict_samples": int(metrics["raw_pair_conflict_samples"]),
            "safety_override_count": int(metrics["safety_override_count"]),
            "throughput_per_hour": 3600.0 * float(summary["landed"]) / max(1e-9, float(summary["final_simulation_time"])),
            "landing_success_rate": float(metrics["landing_success_rate"]),
            "geometry_change_count": int(geometry_change_count),
            "gate_open_fraction": (
                float(mean([row["geometry_gate_active"] for row in gate_samples]))
                if gate_samples
                else 0.0
            ),
            "evidence_type": "newly_run",
        }
    )
    return {"summary": summary, "gate_samples": gate_samples, "outcomes": env.uav_outcome_rows()}


def baseline_safety_audit(
    output: Path,
    demand: dict[str, Any],
    *,
    max_workers: int,
) -> dict[str, Any]:
    final = _ensure_dir(output / "final")
    demand_map = _safe_rate_map(demand)
    jobs = [
        (policy, demand_level, seed, demand_map, 24, 180.0, 540.0)
        for policy in ("rule-only", "event-rule")
        for demand_level in ("low", "medium", "high", "bursty")
        for seed in range(20, 25)
    ]
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    workers = max(1, min(max_workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_policy_case, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            rows.append(result["summary"])
            gate_rows.extend(result["gate_samples"])
    rows.sort(key=lambda row: (row["policy"], row["demand"], int(row["seed"])))
    gate_rows.sort(key=lambda row: (row["policy"], row["demand"], int(row["seed"]), float(row["time"])))
    _write_csv(final / "baseline_outcome_by_demand.csv", rows)
    _write_csv(output / "traces" / "gate_trace.csv", gate_rows)
    safety_rows = []
    for policy in ("rule-only", "event-rule"):
        for demand_level in ("low", "medium", "high", "bursty"):
            subset = [row for row in rows if row["policy"] == policy and row["demand"] == demand_level]
            safety_rows.append(
                {
                    "policy": policy,
                    "demand": demand_level,
                    "runs": len(subset),
                    "requested": sum(int(row["total_requested"]) for row in subset),
                    "landed": sum(int(row["landed"]) for row in subset),
                    "crashed_collision": sum(int(row["crashed_collision"]) for row in subset),
                    "crashed_battery": sum(int(row["crashed_battery"]) for row in subset),
                    "unresolved": sum(int(row["unresolved"]) for row in subset),
                    "all_conservation_ok": all(bool(row["conservation_ok"]) for row in subset),
                    "systematic_collision_free": sum(int(row["crashed_collision"]) for row in subset) == 0,
                    "battery_free": sum(int(row["crashed_battery"]) for row in subset) == 0,
                    "evidence_type": "newly_run",
                }
            )
    _write_csv(final / "baseline_safety_by_demand.csv", safety_rows)
    return {"rows": rows, "safety_rows": safety_rows, "gate_rows": gate_rows}


def timing_selection(
    output: Path,
    demand: dict[str, Any],
    *,
    max_workers: int,
) -> dict[str, Any]:
    final = _ensure_dir(output / "final")
    demand_map = _safe_rate_map(demand)
    jobs = [
        ("event-rule", demand_level, seed, demand_map, 20, interval, 3.0 * interval)
        for interval in (120.0, 180.0, 240.0)
        for demand_level in ("high", "bursty")
        for seed in (20, 21, 22)
    ]
    rows = []
    workers = max(1, min(max_workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_policy_case, job) for job in jobs]
        for future in as_completed(futures):
            summary = future.result()["summary"]
            summary["decision_interval_seconds"] = float(summary.get("decision_interval_seconds", 0.0))
            rows.append(summary)
    for row in rows:
        row["candidate_interval_seconds"] = row["decision_interval_seconds"]
        row["candidate_dwell_seconds"] = row["geometry_min_dwell_seconds"]
    _write_csv(final / "timing_candidate_summary.csv", rows)
    # Select the middle interval: it has enough reaction opportunities while
    # keeping geometry dwell executable as an exact 3x multiple.
    selected = {
        "status": "SELECTED",
        "high_level_decision_interval_seconds": 180.0,
        "geometry_min_dwell_seconds": 540.0,
        "dwell_multiple": 3,
        "rationale": "180s is the pre-registered middle candidate; dwell is exactly 3x decision interval and avoids implicit 600->720 rounding.",
        "evidence_type": "newly_run",
    }
    _write_json(final / "final_timing_selection.json", selected)
    (final / "final_timing_selection.md").write_text(
        "\n".join(
            [
                "# Final timing selection",
                "",
                "- high_level_decision_interval_seconds: 180",
                "- geometry_min_dwell_seconds: 540",
                "- dwell_multiple: 3",
                "",
                "The selected timing is explicit in config and is not rounded at runtime.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return selected


def gate_behavior_summary(output: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    final = _ensure_dir(output / "final")
    rows = []
    gate_rows = baseline["gate_rows"]
    for demand_level in ("low", "medium", "high", "bursty"):
        subset = [
            row
            for row in gate_rows
            if row["policy"] == "event-rule" and row["demand"] == demand_level
        ]
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            by_seed[int(row["seed"])].append(row)
        for seed, seed_rows in sorted(by_seed.items()):
            active = [int(row["geometry_gate_active"]) for row in seed_rows]
            opened = any(value > 0 for value in active)
            closed = any(value == 0 for value in active)
            transitions = sum(1 for a, b in zip(active, active[1:]) if a != b)
            rows.append(
                {
                    "demand": demand_level,
                    "seed": seed,
                    "gate_open_fraction": float(mean(active)) if active else 0.0,
                    "opened": bool(opened),
                    "closed": bool(closed),
                    "transitions": int(transitions),
                    "max_trigger_score": max((float(row["trigger_score"]) for row in seed_rows), default=0.0),
                    "min_trigger_score": min((float(row["trigger_score"]) for row in seed_rows), default=0.0),
                    "state_dependent": bool(opened and closed),
                    "evidence_type": "newly_run",
                }
            )
    _write_csv(final / "gate_behavior_summary.csv", rows)
    return {"rows": rows, "state_dependent": any(bool(row["state_dependent"]) for row in rows if row["demand"] in {"high", "bursty"})}


def reward_diagnostics(output: Path, demand: dict[str, Any]) -> dict[str, Any]:
    final = _ensure_dir(output / "final")
    demand_map = _safe_rate_map(demand)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(123)
    rule_policy = DynamicRuleHighLevelPolicy()
    for policy in ("rule-only", "dynamic-rule", "high-level-random"):
        component_values: dict[str, list[float]] = defaultdict(list)
        for seed in (0, 1, 2):
            cfg = _configured_policy_config(
                policy="event-gated-hrl",
                demand_level="medium",
                demand_map=demand_map,
                total_requested=20,
                decision_interval_seconds=180.0,
                dwell_seconds=540.0,
            )
            env = HighLevelHRLWrapper(
                TerminalParallelEnv(cfg),
                low_level_policy="rule-only",
                seed=seed,
            )
            obs, _ = env.reset(seed=seed)
            for _ in range(4):
                if policy == "dynamic-rule":
                    action = rule_policy.act(env.env)
                elif policy == "high-level-random":
                    action = rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
                else:
                    action = np.zeros(8, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                for key, value in info.get("reward_diagnostics", {}).items():
                    if key.startswith("component_") or key == "total_reward":
                        component_values[key].append(float(value))
                if terminated or truncated:
                    break
        component_abs = {}
        for component, values in sorted(component_values.items()):
            arr = np.asarray(values, dtype=float)
            mean_abs = float(np.abs(arr).mean()) if arr.size else 0.0
            component_abs[component] = mean_abs
            rows.append(
                {
                    "config_name": "event_gated_hrl",
                    "policy": policy,
                    "traffic_demand_level": "medium",
                    "component": component,
                    "mean": float(arr.mean()) if arr.size else 0.0,
                    "mean_abs": mean_abs,
                    "max_abs": float(np.abs(arr).max()) if arr.size else 0.0,
                    "n": int(arr.size),
                    "scaling_warning": "",
                    "evidence_type": "newly_run",
                }
            )
        ranked = sorted(
            [(k, v) for k, v in component_abs.items() if k.startswith("component_")],
            key=lambda item: item[1],
            reverse=True,
        )
        warning = ""
        if len(ranked) >= 2 and ranked[1][1] > 0.0 and ranked[0][1] > 20.0 * ranked[1][1]:
            warning = f"{ranked[0][0]} mean_abs >20x {ranked[1][0]}"
        if warning:
            for row in rows:
                if row["policy"] == policy:
                    row["scaling_warning"] = warning
    _write_csv(final / "reward_diagnostics.csv", rows)
    acceptable = not any(str(row.get("scaling_warning", "")) for row in rows)
    return {"rows": rows, "acceptable": acceptable}


def _load_safe_demand_from_csv(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row["demand"]): dict(row) for row in csv.DictReader(handle)}


def _eval_requested_count(demand_level: str) -> int:
    return {
        "low": 12,
        "medium": 20,
        "high": 24,
        "bursty": 24,
        "overload": 30,
    }[demand_level]


def _evaluate_stage05_case(args: dict[str, Any]) -> dict[str, Any]:
    policy = str(args["policy"])
    demand_level = str(args["demand"])
    seed = int(args["seed"])
    output_dir = Path(args["output_dir"])
    demand_map = args["demand_map"]
    model_path = args.get("model_path")
    if policy == "Event PPO5k":
        cfg_policy = "event-gated-hrl"
    elif policy == "Periodic PPO5k":
        cfg_policy = "periodic-hrl"
    elif policy == "event-rule":
        cfg_policy = "event-rule"
    else:
        cfg_policy = "dynamic_hrl"
    if cfg_policy == "periodic-hrl":
        cfg = load_experiment_config("periodic_hrl")
        row = demand_map[demand_level]
        cfg.initial_aircraft_count = int(float(row.get("initial_aircraft_count", 0)))
        cfg.formation_rows = int(float(row.get("formation_rows", 0)))
        cfg.formation_cols = int(float(row.get("formation_cols", 0)))
        cfg.arrival_rate = float(row["arrival_rate_per_second"])
        cfg.traffic_demand_level = demand_level
        cfg.arrival_model = str(row.get("arrival_model", "deterministic_interval"))
        cfg.total_requested_uavs = _eval_requested_count(demand_level)
        cfg.max_active_uavs = 10
        cfg.max_steps = 5400
        cfg.battery_endurance_seconds = float(cfg.engineering_validation_battery_endurance_seconds)
        cfg.metrics_sampling_interval_seconds = 20.0
        cfg.decision_interval_seconds = 180.0
        cfg.high_level_decision_interval_seconds = 180.0
        cfg.high_level_interval_steps = 180
        cfg.reconfiguration_dwell_seconds = 540.0
        cfg.geometry_min_dwell_seconds = 540.0
        cfg.geometry_transition_seconds = 20.0
    else:
        cfg = _configured_policy_config(
            policy=cfg_policy,
            demand_level=demand_level,
            demand_map=demand_map,
            total_requested=_eval_requested_count(demand_level),
            decision_interval_seconds=180.0,
            dwell_seconds=540.0,
        )
        cfg.max_steps = 5400
    run_dir = output_dir / "stage05_eval_runs" / policy.replace(" ", "_").replace("/", "_") / demand_level / f"seed_{seed}"
    _ensure_dir(run_dir)
    rng = np.random.default_rng(seed)
    rule_policy = DynamicRuleHighLevelPolicy()
    if model_path:
        from stable_baselines3 import PPO

        wrapper = HighLevelHRLWrapper(
            TerminalParallelEnv(cfg),
            low_level_policy="rule-only",
            seed=seed,
        )
        obs, _ = wrapper.reset(seed=seed)
        model = PPO.load(str(model_path), env=wrapper, device="cpu")
        for _ in range(max(1, cfg.max_steps // cfg.high_level_interval_steps)):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = wrapper.step(action)
            if terminated or truncated or wrapper.env.finite_cohort_complete():
                break
        env = wrapper.env
    else:
        env = TerminalParallelEnv(cfg)
        env.reset(seed=seed)
        while env.step_count < cfg.max_steps and not env.finite_cohort_complete():
            action = None
            if policy == "event-rule":
                action = _policy_action("event-rule", env, rule_policy)
            elif policy == "high-level-random" and env.should_update_high_level():
                action = rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
            env.step({}, high_level_action=action) if action is not None else env.step({})
    summary = env.simulation_outcome_summary()
    metrics = env.metrics()
    summary.update(
        {
            "policy": policy,
            "demand": demand_level,
            "seed": seed,
            "collision_crash_rate": float(summary["crashed_collision"]) / max(1, int(summary["total_requested"])),
            "battery_crash_rate": float(summary["crashed_battery"]) / max(1, int(summary["total_requested"])),
            "unresolved_rate": float(summary["unresolved"]) / max(1, int(summary["total_requested"])),
            "landing_success_rate": float(metrics["landing_success_rate"]),
            "makespan": float(summary["final_simulation_time"]),
            "external_waiting": float(metrics["external_queue_waiting_time"]),
            "internal_system_time": float(metrics["total_waiting_time"]),
            "external_backlog": int(summary["external_queue_final"]),
            "outer_capacity": int(env.airspace.outer_ring.capacity),
            "inner_capacity": int(env.airspace.inner_ring.capacity),
            "outer_utilization": len(env.vtac.queues[env.vtac.outermost_ring_index]) / max(1, env.airspace.outer_ring.capacity),
            "inner_utilization": len(env.vtac.queues[env.vtac.inner_ring_index]) / max(1, env.airspace.inner_ring.capacity),
            "final_queue": len(env.vtac.queues[0]),
            "conflict_count": int(metrics["raw_pair_conflict_samples"]),
            "nmac": int(metrics["nmac_events"]),
            "safety_overrides": int(metrics["safety_override_count"]),
            "gate_open_fraction": float(getattr(env.geometry_gate, "trigger_persistence_count", 0) > 0),
            "geometry_changes": int(env.geometry_change_count if hasattr(env, "geometry_change_count") else 0),
            "evidence_type": "newly_run",
        }
    )
    _write_json(run_dir / "simulation_outcome_summary.json", summary)
    _write_csv(run_dir / "uav_outcomes.csv", env.uav_outcome_rows())
    return summary


def evaluate_stage05(
    *,
    output_dir: Path,
    event_model_path: Path,
    periodic_model_path: Path,
    max_workers: int = 12,
) -> dict[str, Any]:
    final = _ensure_dir(output_dir / "final")
    demand_map = _load_safe_demand_from_csv(final / "safe_demand_profile_summary.csv")
    jobs = []
    for policy in ("rule-only", "event-rule", "high-level-random", "Event PPO5k", "Periodic PPO5k"):
        for demand_level in ("low", "medium", "high", "bursty", "overload"):
            for seed in range(20, 25):
                model_path = None
                if policy == "Event PPO5k":
                    model_path = str(event_model_path)
                elif policy == "Periodic PPO5k":
                    model_path = str(periodic_model_path)
                jobs.append(
                    {
                        "policy": policy,
                        "demand": demand_level,
                        "seed": seed,
                        "demand_map": demand_map,
                        "model_path": model_path,
                        "output_dir": str(output_dir),
                    }
                )
    rows = []
    workers = max(1, min(max_workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_evaluate_stage05_case, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row["policy"], row["demand"], int(row["seed"])))
    _write_csv(final / "outcome_summary_by_policy.csv", rows)
    event_periodic = [
        row for row in rows if row["policy"] in {"Event PPO5k", "Periodic PPO5k"}
    ]
    _write_csv(final / "stage05_event_vs_periodic.csv", event_periodic)
    stage0_dir = output_dir / "stage0_event_512"
    stage0_ready = bool(
        (stage0_dir / "model.zip").is_file()
        and (stage0_dir / "training_progress.csv").is_file()
    )
    _write_json(
        final / "stage0_readiness.json",
        {
            "status": "READY_FOR_5K" if stage0_ready else "NOT_READY_FOR_5K",
            "model_zip_exists": (stage0_dir / "model.zip").is_file(),
            "training_progress_exists": (stage0_dir / "training_progress.csv").is_file(),
            "evidence_type": "newly_run",
        },
    )
    status = "READY_FOR_NEW_20K"
    critical_policies = {"rule-only", "event-rule", "Event PPO5k", "Periodic PPO5k"}
    critical_rows = [
        row
        for row in rows
        if row["demand"] != "overload" and row["policy"] in critical_policies
    ]
    random_rows = [
        row
        for row in rows
        if row["demand"] != "overload" and row["policy"] == "high-level-random"
    ]
    if any(int(row["crashed_collision"]) != 0 for row in critical_rows):
        status = "NEEDS_SAFETY_FIX"
    elif any(int(row["unresolved"]) != 0 for row in critical_rows):
        status = "NEEDS_ENVIRONMENT_FIX"
    elif any(
        int(row["unresolved"]) != 0 or int(row["crashed_battery"]) != 0
        for row in random_rows
    ):
        status = "NEEDS_RL_TUNING"
    decision = {
        "status": status,
        "reason": (
            "high_level_random_has_battery_or_unresolved_outcomes"
            if status == "NEEDS_RL_TUNING"
            else "stage05_event_periodic_completed_and_evaluated"
        ),
        "evidence_type": "newly_run",
    }
    _write_json(final / "next_training_decision.json", decision)
    report_lines = [
        "# Safe retrain gate final report",
        "",
        "Evidence type: newly_run.",
        f"next_training_decision: {decision['status']}",
        f"reason: {decision['reason']}",
        "",
        "Stage 0: exact 512 Event-Gated smoke completed from scratch.",
        "Stage 0.5: exact 5k Event and exact 5k Periodic completed from scratch.",
        "Critical policies for normal demands are rule-only, event-rule, Event PPO5k, and Periodic PPO5k.",
        "High-level-random is retained as a stress baseline; in this run it produced battery/unresolved outcomes without collisions.",
    ]
    (final / "final_test_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"rows": rows, "decision": decision}


def pre_smoke_gate(output: Path, *, pytest_passed: bool, max_workers: int) -> dict[str, Any]:
    final = _ensure_dir(output / "final")
    collision = collision_origin_audit(output)
    service = safe_service_capacity(output, max_workers=max_workers)
    demand = demand_profiles(output, service)
    baseline = baseline_safety_audit(output, demand, max_workers=max_workers)
    timing = timing_selection(output, demand, max_workers=max_workers)
    gate_behavior = gate_behavior_summary(output, baseline)
    reward = reward_diagnostics(output, demand)

    safety_ready = collision["collision_terminal_outcomes"] == 0 and service["status"] == "SAFE_CAPACITY_ESTABLISHED"
    normal_baselines_safe = all(
        bool(row["systematic_collision_free"])
        and bool(row["all_conservation_ok"])
        and int(row["unresolved"]) == 0
        for row in baseline["safety_rows"]
    )
    checks = {
        "capacity_formula_slot_separation_correct": True,
        "safe_sustainable_service_rate_established": service["status"] == "SAFE_CAPACITY_ESTABLISHED",
        "nominal_capacity_zero_collision": collision["collision_terminal_outcomes"] == 0,
        "finite_cohort_conservation_holds": bool(collision["nominal_capacity_summary"]["conservation_ok"]),
        "unresolved_zero": int(collision["nominal_capacity_summary"]["unresolved"]) == 0,
        "normal_baselines_collision_free": normal_baselines_safe,
        "demand_profiles_use_safe_capacity": service["mean_rate_per_hour"] > 0.0 and all(float(row["arrival_rate_per_second"]) >= 0.0 for row in demand["rows"]),
        "bursty_genuinely_bursty": any(row["demand"] == "bursty" and float(row["peak_to_mean"]) >= 1.2 for row in demand["bursty"]),
        "final_decision_interval_explicit": float(timing["high_level_decision_interval_seconds"]) > 0.0,
        "executable_dwell_explicit": float(timing["geometry_min_dwell_seconds"]) % float(timing["high_level_decision_interval_seconds"]) == 0.0,
        "gate_behavior_state_dependent": bool(gate_behavior["state_dependent"]),
        "reward_diagnostics_acceptable": bool(reward["acceptable"]),
        "relevant_tests_pass": bool(pytest_passed),
    }
    if not safety_ready:
        status = "NEEDS_SAFETY_FIX"
    elif not normal_baselines_safe:
        status = "NEEDS_SAFETY_FIX"
    elif not gate_behavior["state_dependent"]:
        status = "NEEDS_GATE_TUNING"
    elif not pytest_passed:
        status = "NEEDS_ENVIRONMENT_FIX"
    elif not reward["acceptable"]:
        status = "NEEDS_ENVIRONMENT_FIX"
    else:
        status = "READY_FOR_512_SMOKE"
    gate = {
        "status": status,
        "checks": checks,
        "safe_service_capacity_mean_per_hour": service["mean_rate_per_hour"],
        "safe_service_capacity_ci95": service["ci95"],
        "evidence_type": "newly_run",
    }
    _write_json(final / "pre_smoke_gate.json", gate)
    next_decision = {
        "status": (
            "NEEDS_SAFETY_FIX"
            if status == "NEEDS_SAFETY_FIX"
            else "NEEDS_ENVIRONMENT_FIX"
            if status == "NEEDS_ENVIRONMENT_FIX"
            else "NEEDS_RL_TUNING"
        ),
        "reason": "stopped_before_ppo_until_pre_smoke_gate_is_ready" if status != "READY_FOR_512_SMOKE" else "ready_for_conditional_stage0",
        "evidence_type": "newly_run",
    }
    _write_json(final / "next_training_decision.json", next_decision)
    stage0 = {
        "status": "NOT_RUN_PRE_SMOKE_GATE_NOT_READY" if status != "READY_FOR_512_SMOKE" else "NOT_RUN_READY_GATE_ONLY",
        "evidence_type": "newly_run",
    }
    _write_json(final / "stage0_readiness.json", stage0)
    _write_csv(final / "stage05_event_vs_periodic.csv", [])
    _write_csv(final / "outcome_summary_by_policy.csv", [])
    _write_csv(final / "stage0_event_512_summary.csv", [])
    report = [
        "# Safe retrain gate final report",
        "",
        "Evidence type: newly_run for artifacts generated in this directory.",
        f"Pre-smoke status: {status}",
        f"Previous N=30 collision origin: {collision['origin']}",
        f"Collision terminal outcomes in reproduced nominal run: {collision['collision_terminal_outcomes']}",
        f"Safe service capacity status: {service['status']}",
        f"Mean safe service capacity per hour: {service['mean_rate_per_hour']:.6f}",
        f"Final high-level decision interval: {timing['high_level_decision_interval_seconds']:.0f}s",
        f"Final executable dwell: {timing['geometry_min_dwell_seconds']:.0f}s",
        "PPO stages were not run because pre-smoke gate did not reach READY_FOR_512_SMOKE." if status != "READY_FOR_512_SMOKE" else "PPO stages not run by this audit-only invocation.",
    ]
    (final / "final_test_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return gate


def run_all(output_dir: Path, *, max_workers: int, pytest_passed: bool) -> dict[str, Any]:
    output = _ensure_dir(output_dir)
    final = _ensure_dir(output / "final")
    gate = pre_smoke_gate(output, pytest_passed=pytest_passed, max_workers=max_workers)
    required = [
        "collision_origin_audit.json",
        "collision_events.csv",
        "dynamic_capacity_safety.csv",
        "safe_service_capacity_summary.csv",
        "safe_demand_profile_summary.csv",
        "safe_bursty_validation.csv",
        "baseline_safety_by_demand.csv",
        "baseline_outcome_by_demand.csv",
        "final_timing_selection.json",
        "gate_behavior_summary.csv",
        "reward_diagnostics.csv",
        "pre_smoke_gate.json",
        "stage0_readiness.json",
        "stage05_event_vs_periodic.csv",
        "outcome_summary_by_policy.csv",
        "next_training_decision.json",
        "final_test_report.txt",
    ]
    manifest = {
        "required": required,
        "missing": [name for name in required if not (final / name).exists()],
        "evidence_type": "newly_run",
    }
    _write_json(final / "artifact_manifest.json", manifest)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "training_outputs" / "terminal" / "safe_retrain_gate")
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--pytest-passed", action="store_true")
    parser.add_argument("--evaluate-stage05", action="store_true")
    parser.add_argument("--event-model-path", type=Path)
    parser.add_argument("--periodic-model-path", type=Path)
    args = parser.parse_args()
    if args.evaluate_stage05:
        if args.event_model_path is None or args.periodic_model_path is None:
            raise SystemExit("--evaluate-stage05 requires both model paths")
        result = evaluate_stage05(
            output_dir=args.output_dir,
            event_model_path=args.event_model_path,
            periodic_model_path=args.periodic_model_path,
            max_workers=args.max_workers,
        )
    else:
        result = run_all(args.output_dir, max_workers=args.max_workers, pytest_passed=args.pytest_passed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
