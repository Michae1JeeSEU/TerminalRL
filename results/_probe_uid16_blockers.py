"""Bounded, read-only UID16 blocker diagnostic.

This script replays the dense 30-UAV coordinated scene with the two local
wrappers used by the downstream urgency ablation.  At t=200/250/300 it
captures the pre-release state, checks the complete normal M->I candidate
catalogue and the emergency duration catalogue, then resumes the replay.
Candidates are never reserved or committed.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from terminal_marl.core.config import RingRole
from terminal_marl.core.landing_flow import LandingFlowCoordinator
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.core.uav import FlightPhase
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.diagnose_energy_reachability_guard_smoke import (
    _jsonable,
    build_config,
)


SEED = 841501
SNAPSHOT_TIMES = (200.0, 250.0, 300.0)
STOP_TIME = 301.0
MAX_OUTPUT_BYTES = 150 * 1024
MAX_SNAPSHOT_BYTES = 100 * 1024 * 1024


def _finite(value: Any) -> Any:
    """Return a compact JSON-safe scalar."""

    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return value if math.isfinite(value) else None


def _vec(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in np.asarray(value, dtype=float).tolist()]


def _phase(uav: Any) -> str | None:
    value = getattr(uav, "phase", None)
    return None if value is None else getattr(value, "value", str(value))


def _role(uav: Any) -> str | None:
    value = getattr(uav, "ring_role", None)
    return None if value is None else getattr(value, "value", str(value))


def _plan_row(uav: Any) -> dict[str, Any] | None:
    plan = getattr(uav, "committed_plan", None)
    if plan is None:
        return None
    metadata = dict(getattr(plan, "metadata", {}) or {})
    return {
        "plan_id": str(getattr(plan, "plan_id", "")),
        "start": _finite(plan.start_time),
        "end": _finite(plan.end_time),
        "mode": str(getattr(plan, "mode", "")),
        "target_phase": metadata.get("target_phase"),
        "source_ring": metadata.get("source_ring"),
        "target_ring": metadata.get("target_ring"),
        "position_end": _vec(plan.points[-1].position),
        "velocity_end": _vec(plan.points[-1].velocity),
    }


def _uav_row(uav: Any, *, include_plan: bool = True) -> dict[str, Any]:
    return {
        "uid": int(uav.uid),
        "phase": _phase(uav),
        "ring_role": _role(uav),
        "position": _vec(uav.position),
        "velocity": _vec(uav.velocity),
        "speed": _finite(uav.speed),
        "soc": _finite(uav.soc),
        "energy_remaining": _finite(uav.energy_remaining),
        "ring_progress": _finite(uav.ring_progress),
        "committed_plan": _plan_row(uav) if include_plan else None,
    }


def _snapshot_state(env: TerminalParallelEnv, uid: int = 16) -> dict[str, Any]:
    uav = env.active.get(uid)
    rows: list[dict[str, Any]] = []
    incoming: list[dict[str, Any]] = []
    for other in sorted(env.active.values(), key=lambda item: int(item.uid)):
        plan = getattr(other, "committed_plan", None)
        if other.ring_role is RingRole.MIDDLE or plan is not None:
            row = _uav_row(other)
            rows.append(row)
        if plan is not None:
            metadata = dict(getattr(plan, "metadata", {}) or {})
            incoming.append({
                "uid": int(other.uid),
                "phase": _phase(other),
                "soc": _finite(other.soc),
                "position": _vec(other.position),
                "target_phase": metadata.get("target_phase"),
                "source_ring": metadata.get("source_ring"),
                "target_ring": metadata.get("target_ring"),
                "start": _finite(plan.start_time),
                "end": _finite(plan.end_time),
                "plan_id": str(plan.plan_id),
                "position_end": _vec(plan.points[-1].position),
            })
    return {
        "time": _finite(env.now),
        "uid16": None if uav is None else _uav_row(uav),
        "ring_holders_and_plans": rows,
        "incoming_plans": incoming,
        "pads": [
            {
                "id": int(zone.lz_id),
                "status": getattr(zone.status, "value", str(zone.status)),
                "reserved_by": None if zone.reserved_by is None else int(zone.reserved_by),
                "occupied_by": None if zone.occupied_by is None else int(zone.occupied_by),
            }
            for zone in env.lz_manager.zones
        ],
    }


def _source_neighbors(env: TerminalParallelEnv, uid: int = 16) -> dict[str, Any]:
    uav = env.active.get(uid)
    if uav is None:
        return {"reference_uid": uid, "ahead": None, "behind": None, "all": []}
    theta0 = float(math.atan2(float(uav.position[1]), float(uav.position[0])))
    rows: list[dict[str, Any]] = []
    for other in env.active.values():
        if other.uid == uid or other.ring_role is not RingRole.MIDDLE:
            continue
        if other.committed_plan is not None:
            continue
        theta = float(math.atan2(float(other.position[1]), float(other.position[0])))
        signed = (theta - theta0 + math.pi) % (2.0 * math.pi) - math.pi
        rows.append({
            "uid": int(other.uid),
            "phase": _phase(other),
            "soc": _finite(other.soc),
            "signed_angle": _finite(signed),
            "absolute_angle": _finite(abs(signed)),
            "chord_gap": _finite(np.linalg.norm(np.asarray(other.position) - np.asarray(uav.position))),
        })
    ahead = sorted((row for row in rows if float(row["signed_angle"]) > 0.0), key=lambda row: float(row["signed_angle"]))
    behind = sorted((row for row in rows if float(row["signed_angle"]) < 0.0), key=lambda row: abs(float(row["signed_angle"])))
    return {
        "reference_uid": uid,
        "ahead": ahead[0] if ahead else None,
        "behind": behind[0] if behind else None,
        "all": sorted(rows, key=lambda row: float(row["absolute_angle"])),
    }


def _snapshot_side_effects(safety: Any, env: TerminalParallelEnv) -> dict[str, Any]:
    return {
        "checked_plan_ids": list(safety.checked_plan_ids),
        "checked_trajectory_hashes": dict(safety.checked_trajectory_hashes),
        "holder_track_cache": copy.deepcopy(safety._holder_track_cache),
        "holder_tube_cache": copy.deepcopy(safety._holder_tube_cache),
        "committed_track_cache": copy.deepcopy(safety._committed_track_cache),
        "last_predicted_track_audit": copy.deepcopy(safety.last_predicted_track_audit),
        "telemetry": dict(env.telemetry),
        "energy_diagnostics": copy.deepcopy(env.energy_reachability_diagnostics),
    }


def _restore_side_effects(safety: Any, env: TerminalParallelEnv, state: Mapping[str, Any]) -> None:
    safety.checked_plan_ids[:] = state["checked_plan_ids"]
    safety.checked_trajectory_hashes.clear()
    safety.checked_trajectory_hashes.update(state["checked_trajectory_hashes"])
    safety._holder_track_cache.clear()
    safety._holder_track_cache.update(state["holder_track_cache"])
    safety._holder_tube_cache.clear()
    safety._holder_tube_cache.update(state["holder_tube_cache"])
    safety._committed_track_cache.clear()
    safety._committed_track_cache.update(state["committed_track_cache"])
    safety.last_predicted_track_audit = copy.deepcopy(state["last_predicted_track_audit"])
    env.telemetry.clear()
    env.telemetry.update(state["telemetry"])
    env.energy_reachability_diagnostics.clear()
    env.energy_reachability_diagnostics.update(state["energy_diagnostics"])


def _candidate_conflicts(env: TerminalParallelEnv, uav: Any, plan: Any) -> dict[str, Any]:
    """Record point/swept/tube minima using SafetyFilter's own samplers."""

    safety = env.safety
    flow = env.flow_coordinator
    snapshot = flow._snapshot()
    samples = np.arange(
        float(plan.start_time),
        float(plan.end_time) + 1e-9,
        float(env.config.safety_sampling_dt),
    )
    if len(samples) == 0 or samples[-1] < float(plan.end_time) - 1e-9:
        samples = np.append(samples, float(plan.end_time))
    candidate = np.asarray(plan.sample_positions(samples), dtype=float)
    rows: list[dict[str, Any]] = []
    for other in sorted(env.active.values(), key=lambda item: int(item.uid)):
        if other.uid == uav.uid:
            continue
        other_plan = other.committed_plan
        if other_plan is not None:
            other_positions = np.asarray(
                safety._committed_positions(other_plan, snapshot, samples), dtype=float
            )
            source_kind = "committed_plan"
        else:
            other_positions = np.asarray(
                safety._holder_positions(other, snapshot, samples, 0.0), dtype=float
            )
            source_kind = "holder"
        point, swept = safety._track_distances(candidate, other_positions)
        metrics: list[tuple[float, str, float]] = []
        if len(point):
            index = int(np.argmin(point))
            metrics.append((float(point[index]), "point", float(samples[index])))
        if len(swept):
            index = int(np.argmin(swept))
            metrics.append((float(swept[index]), "swept", float((samples[index] + samples[index + 1]) * 0.5)))
        tube_min = None
        tube_time = None
        if other_plan is None and other.ring_role is not None:
            try:
                tube_times = np.arange(
                    float(plan.start_time),
                    float(plan.end_time) + 1e-9,
                    float(env.config.physics_dt),
                )
                if len(tube_times) == 0 or tube_times[-1] < float(plan.end_time) - 1e-9:
                    tube_times = np.append(tube_times, float(plan.end_time))
                tube_candidate = np.asarray(plan.sample_positions(tube_times), dtype=float)
                lower, upper = safety.holder_reachable_tube(other, snapshot, tube_times)
                ring = snapshot.airspace.actual_ring(other.ring_role)
                tube_distances = safety._tube_point_distances(tube_candidate, ring, lower, upper)
                if len(tube_distances):
                    index = int(np.argmin(tube_distances))
                    tube_min = float(tube_distances[index])
                    tube_time = float(tube_times[index])
                    metrics.append((tube_min, "reachable_tube", tube_time))
            except (RuntimeError, ValueError, KeyError, TypeError):
                tube_min = None
        if not metrics:
            continue
        minimum, metric, moment = min(metrics, key=lambda row: row[0])
        rows.append({
            "uid": int(other.uid),
            "phase": _phase(other),
            "ring_role": _role(other),
            "source_kind": source_kind,
            "minimum_separation": _finite(minimum),
            "metric": metric,
            "time": _finite(moment),
            "point_minimum": None if not len(point) else _finite(np.min(point)),
            "swept_minimum": None if not len(swept) else _finite(np.min(swept)),
            "reachable_tube_minimum": _finite(tube_min),
            "reachable_tube_time": _finite(tube_time),
        })
    rows.sort(key=lambda row: float(row["minimum_separation"]))
    return {
        "conflict_count": len(rows),
        "worst_conflict": rows[0] if rows else None,
        "top_conflicts": rows[:3],
    }


def _filter_read_only(env: TerminalParallelEnv, plan: Any) -> dict[str, Any]:
    state = _snapshot_side_effects(env.safety, env)
    try:
        result = env.safety.filter_plan(
            plan,
            env.flow_coordinator._committed_plans(),
            env.flow_coordinator._snapshot(),
        )
        return {
            "approved": bool(result.approved),
            "reason": result.reason or None,
            "checked_plan_id": result.checked_plan_id,
        }
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        return {
            "approved": False,
            "reason": f"{type(exc).__name__}:{exc}",
            "checked_plan_id": None,
        }
    finally:
        _restore_side_effects(env.safety, env, state)


def _energy_read_only(env: TerminalParallelEnv, uav: Any, plan: Any, target: Any) -> dict[str, Any]:
    state = _snapshot_side_effects(env.safety, env)
    try:
        reason = env._candidate_energy_guard_reason(uav, plan, target)
        diagnostic = copy.deepcopy(env.energy_reachability_diagnostics.get(uav.uid))
        return {
            "pass": reason is None,
            "reason": reason,
            "diagnostic": _jsonable(diagnostic),
        }
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        return {
            "pass": False,
            "reason": f"{type(exc).__name__}:{exc}",
            "diagnostic": None,
        }
    finally:
        _restore_side_effects(env.safety, env, state)


def _normal_candidate(env: TerminalParallelEnv, uav: Any, duration: float, bias: float) -> dict[str, Any]:
    flow = env.flow_coordinator
    row: dict[str, Any] = {
        "duration": float(duration),
        "bias_mps": float(bias),
        "dynamics_pass": False,
        "dynamics_reason": None,
        "scheduler_guard": None,
        "inner_departure_headway": None,
        "energy": None,
        "safety": None,
        "safety_debug": None,
        "source_neighbor_worst": None,
    }
    try:
        plan = flow.trajectory.ring_transfer(
            uav, RingRole.INNER, env.airspace, env.now, duration,
            tangential_speed_bias_mps=bias,
        )
        errors = flow.trajectory.dynamics.validate_plan(plan.points)
        row["dynamics_pass"] = not bool(errors)
        row["dynamics_reason"] = None if not errors else list(errors)
    except (RuntimeError, ValueError, KeyError, AssertionError) as exc:
        row["dynamics_reason"] = f"{type(exc).__name__}:{exc}"
        return row

    proposal = flow._proposal(RingRole.INNER, plan)
    pre = env.scheduler.can_reserve_at(
        RingRole.INNER, env.now + float(duration),
        headway=float(env.airspace.headways["m2i"]),
    )
    pairwise = env.scheduler.can_reserve(proposal)
    row["scheduler_guard"] = {
        "can_reserve_at_requested_end": bool(pre),
        "can_reserve_proposal": bool(pairwise),
        "proposal_time": _finite(proposal.merge_time),
        "headway": _finite(proposal.headway),
    }
    try:
        row["inner_departure_headway"] = {
            "pass": flow._inner_departure_headway_reason(uav, plan) is None,
            "reason": flow._inner_departure_headway_reason(uav, plan),
        }
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        row["inner_departure_headway"] = {
            "pass": False,
            "reason": f"{type(exc).__name__}:{exc}",
        }
    row["energy"] = _energy_read_only(env, uav, plan, RingRole.INNER)
    row["safety"] = _filter_read_only(env, plan)
    row["safety_debug"] = _candidate_conflicts(env, uav, plan)
    row["source_neighbor_worst"] = _source_neighbors(env, uav.uid)
    return row


def _emergency_candidate(env: TerminalParallelEnv, uav: Any, duration: float, lz_id: int) -> dict[str, Any]:
    flow = env.flow_coordinator
    zone = env.lz_manager.zones[int(lz_id)]
    row: dict[str, Any] = {
        "duration": float(duration),
        "pad_id": int(lz_id),
        "dynamics_pass": False,
        "dynamics_reason": None,
        "energy": None,
        "slack": None,
        "safety": None,
        "safety_debug": None,
    }
    try:
        plan = flow.trajectory.final_approach(
            uav, zone, env.airspace, env.now, duration, emergency=True,
        )
        errors = flow.trajectory.dynamics.validate_plan(plan.points)
        row["dynamics_pass"] = not bool(errors)
        row["dynamics_reason"] = None if not errors else list(errors)
    except (RuntimeError, ValueError, KeyError, AssertionError) as exc:
        row["dynamics_reason"] = f"{type(exc).__name__}:{exc}"
        return row
    try:
        required, slack, wait_seconds, slack_limit = flow._emergency_slack_budget(uav, plan)
        row["energy"] = {
            "pass": bool(uav.energy_remaining + 1e-9 >= required),
            "energy_remaining": _finite(uav.energy_remaining),
            "required_energy": _finite(required),
        }
        row["slack"] = {
            "pass": bool(slack <= slack_limit + 1e-9),
            "slack_energy": _finite(slack),
            "wait_seconds": _finite(wait_seconds),
            "slack_limit": _finite(slack_limit),
        }
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        row["energy"] = {"pass": False, "reason": f"{type(exc).__name__}:{exc}"}
        row["slack"] = {"pass": False, "reason": f"{type(exc).__name__}:{exc}"}
    row["safety"] = _filter_read_only(env, plan)
    row["safety_debug"] = _candidate_conflicts(env, uav, plan)
    return row


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(json.dumps(_jsonable(payload), sort_keys=True), flush=True)


def _save_snapshot(env: TerminalParallelEnv, path: Path) -> dict[str, Any]:
    flow = env.flow_coordinator
    flow_wrapper = flow.__dict__.pop("processing_order", None)
    emergency_wrapper = flow.__dict__.pop("emergency_processing_order", None)
    step_wrapper = env.__dict__.pop("step", None)
    try:
        with path.open("wb") as handle:
            pickle.dump(env, handle, protocol=pickle.HIGHEST_PROTOCOL)
        size = path.stat().st_size
        if size > MAX_SNAPSHOT_BYTES:
            path.unlink()
            return {"path": str(path), "saved": False, "bytes": int(size), "error": "snapshot_exceeds_100MB"}
        return {"path": str(path), "saved": True, "bytes": int(size)}
    except Exception as exc:  # pragma: no cover - environment-specific pickle failures
        if path.exists():
            path.unlink()
        return {"path": str(path), "saved": False, "error": f"{type(exc).__name__}:{exc}"}
    finally:
        if flow_wrapper is not None:
            flow.__dict__["processing_order"] = flow_wrapper
        if emergency_wrapper is not None:
            flow.__dict__["emergency_processing_order"] = emergency_wrapper
        if step_wrapper is not None:
            env.__dict__["step"] = step_wrapper


def _apply_downstream_urgency_wrappers(env: TerminalParallelEnv) -> dict[str, Any]:
    flow = env.flow_coordinator
    if flow is None:
        raise RuntimeError("flow coordinator is required for this diagnostic")
    original_trigger = float(env.config.emergency_trigger_soc)
    original_processing = flow.processing_order
    original_emergency = flow.emergency_processing_order
    original_step = env.step

    def downstream_processing_order():
        for current in env.active.values():
            env.priority.update(current, env.now)
        rank = {RingRole.INNER: 0, RingRole.MIDDLE: 1, RingRole.OUTER: 2}
        rows = []
        for current in env.active.items():
            _uid, item = current
            slack, _old_rank, priority, uid = flow._energy_slack_key(item)
            rows.append((
                rank.get(item.ring_role, 3),
                slack,
                priority,
                uid,
                current,
            ))
        rows.sort(key=lambda row: row[:4])
        return [row[-1] for row in rows]

    def emergency_processing_order():
        result = original_emergency()
        env.config.emergency_trigger_soc = 1.0
        return result

    def wrapped_step(*args, **kwargs):
        try:
            return original_step(*args, **kwargs)
        finally:
            env.config.emergency_trigger_soc = original_trigger

    flow.processing_order = downstream_processing_order
    flow.emergency_processing_order = emergency_processing_order
    env.step = wrapped_step
    return {
        "normal_order": "(downstream_rank, energy_slack, priority, uid)",
        "emergency_order": "original emergency order; trigger set to 1.0 after ordering",
        "trigger_original": original_trigger,
        "trigger_during_emergency_step": 1.0,
        "restore_after_step": True,
        "source_original_processing": getattr(original_processing, "__qualname__", str(original_processing)),
        "source_original_emergency": getattr(original_emergency, "__qualname__", str(original_emergency)),
    }


def _config():
    base = build_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=True,
        max_simulation_seconds=301.0,
        dense_warm_start_uavs=30,
        final_approach_geometry_mode="continuous_inward_v1",
        release_arbitration_mode="downstream_priority",
        reconcile_late_merge_slots_enabled=False,
    )
    return replace(base, rule_flow_coordination_enabled=True)


def run(output: str | Path = "results/flow_uid16_blockers.json") -> dict[str, Any]:
    output = Path(output)
    progress_path = output.with_name(f"{output.name}.progress.json")
    if output.exists() or progress_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    for moment in SNAPSHOT_TIMES:
        snapshot_path = output.with_name(f"{output.stem}.t{int(moment)}.pkl")
        if snapshot_path.exists():
            raise FileExistsError(f"refusing to overwrite snapshot: {snapshot_path}")

    config = _config()
    env = TerminalParallelEnv(config)
    env.reset(seed=SEED)
    pickle_roundtrip: dict[str, Any]
    try:
        payload = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
        restored = pickle.loads(payload)
        pickle_roundtrip = {"ok": True, "bytes": int(len(payload))}
        del restored
    except Exception as exc:
        pickle_roundtrip = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}

    wrapper_info = _apply_downstream_urgency_wrappers(env)
    snapshots: dict[str, Any] = {}
    candidate_records: dict[str, Any] = {}
    error: str | None = None
    step = 0
    start_wall = time.perf_counter()
    progress = {
        "step": 0,
        "sim_time": 0.0,
        "snapshots": [],
        "status": "running",
        "error": None,
    }
    _write_progress(progress_path, progress)

    while float(env.now) < STOP_TIME - 1e-9:
        now = float(env.now)
        if any(abs(now - moment) <= 1e-9 for moment in SNAPSHOT_TIMES):
            key = str(int(now))
            snapshots[key] = _snapshot_state(env)
            snapshots[key]["source_neighbors"] = _source_neighbors(env, 16)
            if pickle_roundtrip.get("ok"):
                snapshot_path = output.with_name(f"{output.stem}.t{int(now)}.pkl")
                snapshots[key]["pickle"] = _save_snapshot(env, snapshot_path)
            uav = env.active.get(16)
            if uav is None:
                candidate_records[key] = {"uid16_missing": True}
            elif uav.ring_role is not RingRole.MIDDLE:
                candidate_records[key] = {
                    "uid16_missing": False,
                    "uid16_ring_role": _role(uav),
                    "normal_m2i": [],
                    "emergency": [],
                }
            else:
                normal = []
                for duration in (60.0, 72.0, 84.0, 96.0):
                    for bias in (0.0, -4.0, -8.0):
                        normal.append(_normal_candidate(env, uav, duration, bias))
                free_pads = list(env.flow_coordinator._available_lz_ids(uav))
                emergency = []
                for duration in (60.0, 72.0, 84.0, 90.0):
                    for lz_id in free_pads:
                        emergency.append(_emergency_candidate(env, uav, duration, int(lz_id)))
                candidate_records[key] = {
                    "uid16_missing": False,
                    "uid16_ring_role": _role(uav),
                    "normal_m2i": normal,
                    "emergency": emergency,
                    "emergency_free_pads": [int(item) for item in free_pads],
                }
            snapshots[key]["candidate_record_key"] = key
            progress["snapshots"] = sorted(snapshots)
            progress["sim_time"] = now
            _write_progress(progress_path, progress)
        try:
            env.step()
        except SafetyInvariantError as exc:
            error = f"SafetyInvariantError:{exc}"
            progress["status"] = "safety_error"
            progress["error"] = error
            break
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            progress["status"] = "error"
            progress["error"] = error
            break
        step += 1
        if step % 50 == 0:
            progress.update({
                "step": int(step),
                "sim_time": float(env.now),
                "active": int(len(env.active)),
                "resolved": int(len(env.ledger.outcomes)),
                "snapshots": sorted(snapshots),
                "status": "running",
            })
            _write_progress(progress_path, progress)

    if error is None:
        progress.update({"step": int(step), "sim_time": float(env.now), "status": "max_time"})
        _write_progress(progress_path, progress)

    result = {
        "seed": SEED,
        "diagnostic_only": True,
        "scenario": "30 warm-start continuous_inward_v1 route_guard downstream_urgency_v2",
        "wrapper_info": wrapper_info,
        "config": _jsonable(asdict(config)),
        "pickle_roundtrip": pickle_roundtrip,
        "steps": int(step),
        "sim_time_seconds": float(env.now),
        "wall_seconds": float(time.perf_counter() - start_wall),
        "stop_reason": "safety_error" if error and error.startswith("SafetyInvariantError") else ("error" if error else "max_time"),
        "error": error,
        "snapshots": snapshots,
        "candidate_records": candidate_records,
        "safety_summary": {
            "minimum_actual_separation": _finite(env.telemetry.get("minimum_actual_separation")),
            "minimum_swept_separation": _finite(env.telemetry.get("minimum_swept_separation")),
            "collisions": int(env.telemetry.get("collisions", 0)),
            "nmac": int(env.telemetry.get("nmac", 0)),
            "shield_fail_closed": int(env.telemetry.get("shield_fail_closed", 0)),
            "joint_safety_fail_closed": int(env.telemetry.get("joint_safety_fail_closed", 0)),
            "requested": int(len(env.ledger.outcomes) + len(env.active) + len(env.admission.queue.items)),
            "resolved": int(len(env.ledger.outcomes)),
        },
        "telemetry": _jsonable(env.telemetry),
    }
    encoded = json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise RuntimeError(f"diagnostic output exceeds 150KB: {len(encoded.encode('utf-8'))}")
    output.write_text(encoded + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/flow_uid16_blockers.json")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "output": args.output,
        "steps": result["steps"],
        "sim_time": result["sim_time_seconds"],
        "stop_reason": result["stop_reason"],
        "snapshots": sorted(result["snapshots"]),
        "pickle_roundtrip": result["pickle_roundtrip"],
    }, sort_keys=True), flush=True)
