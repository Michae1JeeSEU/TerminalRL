"""Bounded comparison diagnostic for the opt-in landing-flow coordinator.

The command replays the fixed 30-UAV warm-start scene once with the legacy
release path or once with ``rule_flow_coordination_enabled`` enabled.  It
keeps only compact event, blocker, safety, energy, and t=552 s records so the
result remains suitable for source-controlled diagnostics and visualization.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from enum import Enum
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.diagnose_energy_reachability_guard_smoke import (
    _jsonable,
    build_config,
)


TARGET_UIDS = (3, 4)
PAIR_FOCUS_UIDS = frozenset((19, 21))
TARGET_DIAGNOSTIC_TIME = 552.0
PROGRESS_INTERVAL = 50
MAX_OUTPUT_BYTES = 500 * 1024


def _terminal(env: TerminalParallelEnv) -> bool:
    """Return whether the complete cycle, including ground service, ended."""

    return bool(
        env.arrivals.exhausted()
        and not env.active
        and not env.admission.queue.items
        and not env.service_uavs
    )


def _build_diagnostic_config(coordinated: bool):
    base = build_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=True,
        max_simulation_seconds=1800.0,
        dense_warm_start_uavs=30,
        final_approach_geometry_mode="continuous_inward_v1",
        release_arbitration_mode="downstream_priority",
        reconcile_late_merge_slots_enabled=False,
    )
    return replace(base, rule_flow_coordination_enabled=bool(coordinated))


def _state_row(uav) -> dict[str, Any]:
    plan = uav.committed_plan
    return {
        "uid": int(uav.uid),
        "arrival_id": int(uav.arrival_id),
        "phase": uav.phase.value,
        "ring_role": None if uav.ring_role is None else uav.ring_role.value,
        "position": [float(value) for value in uav.position],
        "velocity": [float(value) for value in uav.velocity],
        "speed": float(uav.speed),
        "soc": float(uav.soc),
        "energy_remaining": float(uav.energy_remaining),
        "energy_consumed": float(uav.energy_consumed),
        "phase_entry_time": float(uav.phase_entry_time),
        "assigned_lz": None if uav.assigned_lz is None else int(uav.assigned_lz),
        "emergency_landing": bool(uav.emergency_landing),
        "committed": plan is not None,
        "trajectory_plan_id": None if plan is None else plan.plan_id,
        "plan_end_time": None if plan is None else float(plan.end_time),
    }


def _live_uavs(env: TerminalParallelEnv) -> dict[int, Any]:
    values: dict[int, Any] = {}
    values.update({int(uid): uav for uid, uav in env.active.items()})
    values.update({int(uid): uav for uid, uav in env.service_uavs.items()})
    for uav in getattr(env, "completed_uavs", ()):
        values.setdefault(int(uav.uid), uav)
    return values


def _state_map(env: TerminalParallelEnv) -> dict[int, dict[str, Any]]:
    return {uid: _state_row(uav) for uid, uav in _live_uavs(env).items()}


def _pad_state(env: TerminalParallelEnv) -> list[dict[str, Any]]:
    return [
        {
            "id": int(zone.lz_id),
            "status": zone.status.value,
            "reserved_by": None if zone.reserved_by is None else int(zone.reserved_by),
            "occupied_by": None if zone.occupied_by is None else int(zone.occupied_by),
            "reservation_start": zone.reservation_start,
            "expected_release_at": zone.expected_release_at,
            "service_remaining": float(zone.service_remaining),
        }
        for zone in env.lz_manager.zones
    ]


def _complete_state_at_552(env: TerminalParallelEnv) -> dict[str, Any]:
    rows = _state_map(env)
    for row in env.energy_outcomes:
        uid = row.get("uid")
        if uid is None or int(uid) in rows:
            continue
        rows[int(uid)] = {
            "uid": int(uid),
            "arrival_id": row.get("arrival_id"),
            "phase": row.get("phase"),
            "ring_role": None,
            "position": row.get("position"),
            "velocity": None,
            "speed": None,
            "soc": row.get("soc"),
            "energy_remaining": row.get("energy_remaining"),
            "energy_consumed": row.get("energy_consumed"),
            "phase_entry_time": None,
            "assigned_lz": row.get("assigned_lz"),
            "emergency_landing": row.get("emergency_landing", False),
            "committed": False,
            "trajectory_plan_id": None,
            "plan_end_time": None,
            "terminal_outcome": row.get("outcome"),
            "terminal_reason": row.get("reason"),
        }
    return {
        "time": float(env.now),
        "uavs": [rows[uid] for uid in sorted(rows)],
        "pads": _pad_state(env),
    }


def _outcome_counts(env: TerminalParallelEnv) -> dict[str, int]:
    return dict(sorted(Counter(
        outcome.value for outcome in env.ledger.outcomes.values()
    ).items()))


def _phase_counts(env: TerminalParallelEnv) -> dict[str, int]:
    rows = _state_map(env)
    counts = Counter(
        str(row.get("phase")) for row in rows.values()
        if row.get("phase") is not None
    )
    live_ids = set(rows)
    for row in env.energy_outcomes:
        uid = row.get("uid")
        phase = row.get("phase")
        if uid is not None and int(uid) not in live_ids and phase is not None:
            counts[str(phase)] += 1
    return dict(sorted(counts.items()))


def _progress_payload(env: TerminalParallelEnv, *, step: int, elapsed: float,
                      stop_reason: str | None, normal_commits: int = 0,
                      emergency_commits: int = 0) -> dict[str, Any]:
    active_socs = [
        float(uav.soc)
        for uav in env.active.values()
        if getattr(uav, "soc", None) is not None
    ]
    return {
        "step": int(step),
        "sim_time": float(env.now),
        "wall_seconds": float(elapsed),
        "active": int(len(env.active)),
        "minimum_active_soc": min(active_socs) if active_socs else None,
        "service": int(len(env.service_uavs)),
        "queue": int(len(env.admission.queue.items)),
        "resolved": int(len(env.ledger.outcomes)),
        "outcome_counts": _outcome_counts(env),
        "phases": _phase_counts(env),
        "normal_commits": int(normal_commits),
        "emergency_commits": int(emergency_commits),
        "terminal": bool(_terminal(env)),
        "stop_reason": stop_reason,
    }


def _write_progress_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(json.dumps(_jsonable(payload), sort_keys=True), flush=True)


def _record_phase_observations(
    previous: Mapping[int, dict[str, Any]],
    current: Mapping[int, dict[str, Any]],
    *,
    now: float,
    phase_transitions: dict[str, list[dict[str, Any]]],
    first_low_soc: dict[str, dict[str, Any] | None],
    low_soc_threshold: float,
) -> None:
    for uid in TARGET_UIDS:
        key = str(uid)
        before = previous.get(uid)
        after = current.get(uid)
        if after is None:
            continue
        if before is not None and before.get("phase") != after.get("phase"):
            phase_transitions[key].append({
                "time": float(now),
                "from_phase": before.get("phase"),
                "to_phase": after.get("phase"),
                "soc": float(after["soc"]),
                "ring_role": after.get("ring_role"),
            })
        if first_low_soc[key] is None and float(after["soc"]) <= low_soc_threshold + 1e-12:
            first_low_soc[key] = {
                "time": float(now),
                "soc": float(after["soc"]),
                "phase": after.get("phase"),
                "ring_role": after.get("ring_role"),
            }


def _commit_record(uid: int, before: Mapping[int, dict[str, Any]], after_uavs: Mapping[int, Any],
                   *, time_value: float, kind: str, coordinated: bool) -> dict[str, Any]:
    before_row = before.get(uid, {})
    uav = after_uavs.get(uid)
    plan = None if uav is None else uav.committed_plan
    plan_metadata = {} if plan is None else (getattr(plan, "metadata", {}) or {})
    tangential_speed_bias = (
        None
        if plan is None
        else float(plan_metadata.get("flow_tangential_speed_bias_mps", 0.0))
    )
    return {
        "uid": int(uid),
        "time": float(time_value),
        "soc": before_row.get("soc"),
        "from_phase": before_row.get("phase"),
        "end_time": None if plan is None else float(plan.end_time),
        "pad_id": None if uav is None or uav.assigned_lz is None else int(uav.assigned_lz),
        "trajectory_plan_id": None if plan is None else plan.plan_id,
        "tangential_speed_bias_mps": tangential_speed_bias,
        "kind": kind,
        "coordinated": bool(coordinated),
    }


def _plan_summary(plan: Any) -> dict[str, Any] | None:
    if plan is None:
        return None
    metadata = getattr(plan, "metadata", {}) or {}
    source = metadata.get("source_ring", metadata.get("source_ring_role"))
    target = metadata.get(
        "target_ring",
        metadata.get("target_ring_role", metadata.get("target_phase")),
    )
    return {
        "plan_id": getattr(plan, "plan_id", None),
        "mode": getattr(plan, "mode", None),
        "start_time": float(plan.start_time),
        "end_time": float(plan.end_time),
        "source": source,
        "target": target,
        "source_ring": metadata.get("source_ring", metadata.get("source_ring_role")),
        "target_ring": metadata.get("target_ring", metadata.get("target_ring_role")),
        "target_phase": metadata.get("target_phase"),
        "trajectory_hash": getattr(plan, "trajectory_hash", metadata.get("trajectory_hash")),
    }


def _diagnostic_pairs(diagnostic: Mapping[str, Any] | None) -> list[list[int]]:
    if not isinstance(diagnostic, Mapping):
        return []
    pairs: list[list[int]] = []
    for raw_pair in diagnostic.get("unsafe_pairs", ()) or ():
        if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
            continue
        try:
            pair = [int(raw_pair[0]), int(raw_pair[1])]
        except (TypeError, ValueError):
            continue
        pairs.append(pair)
    return pairs


def _pair_state_context(env: TerminalParallelEnv,
                        diagnostic: Mapping[str, Any] | None) -> dict[str, Any]:
    pairs = _diagnostic_pairs(diagnostic)
    live = _live_uavs(env)
    release_telemetry = getattr(env, "last_release_telemetry", {})
    relevant_release_telemetry: dict[str, Any] = {}
    if isinstance(release_telemetry, Mapping):
        for uid in sorted({uid for pair in pairs for uid in pair}):
            row = release_telemetry.get(uid, release_telemetry.get(str(uid)))
            if row is not None:
                relevant_release_telemetry[str(uid)] = row
    pair_rows: list[dict[str, Any]] = []
    for pair in pairs:
        uavs: list[dict[str, Any]] = []
        for uid in pair:
            uav = live.get(uid)
            if uav is None:
                uavs.append({"uid": int(uid), "missing": True})
                continue
            row = _state_row(uav)
            row["committed_plan"] = _plan_summary(uav.committed_plan)
            uavs.append(row)
        pair_rows.append({"pair": pair, "uavs": uavs})
    raw = diagnostic if isinstance(diagnostic, Mapping) else {}
    return {
        "time": float(env.now),
        "unsafe_pairs": pairs,
        "minimum_predicted_separation": raw.get("initial_minimum_separation"),
        "recursive_gate": raw.get("recursive_gate"),
        "last_release_telemetry": relevant_release_telemetry,
        "pairs": pair_rows,
    }


def _pair_commit_record(uid: int, before: Mapping[int, dict[str, Any]],
                        after_uavs: Mapping[int, Any], *, time_value: float,
                        kind: str, coordinated: bool) -> dict[str, Any]:
    record = _commit_record(
        uid, before, after_uavs, time_value=time_value,
        kind=kind, coordinated=coordinated,
    )
    uav = after_uavs.get(uid)
    record["state"] = None if uav is None else _state_row(uav)
    record["committed_plan"] = None if uav is None else _plan_summary(uav.committed_plan)
    return record


def run_diagnostic(*, seed: int, coordinated: bool, max_steps: int = 1800,
                   output: Path) -> dict[str, Any]:
    """Run one bounded comparison case and write a compact JSON artifact."""

    output = Path(output)
    progress_path = Path(f"{output}.progress.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if progress_path.exists():
        raise FileExistsError(f"refusing to overwrite existing progress output: {progress_path}")
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")

    config = _build_diagnostic_config(bool(coordinated))
    started = time.perf_counter()
    env = TerminalParallelEnv(config)
    env.reset(seed=int(seed))
    phase_transitions = {str(uid): [] for uid in TARGET_UIDS}
    first_low_soc: dict[str, dict[str, Any] | None] = {str(uid): None for uid in TARGET_UIDS}
    initial_states = _state_map(env)
    _record_phase_observations(
        {}, initial_states, now=float(env.now),
        phase_transitions=phase_transitions, first_low_soc=first_low_soc,
        low_soc_threshold=float(config.emergency_trigger_soc),
    )
    normal_commits: list[dict[str, Any]] = []
    emergency_commits: list[dict[str, Any]] = []
    pair_focus_commits: list[dict[str, Any]] = []
    seen_normal: set[tuple[int, str]] = set()
    seen_emergency: set[tuple[int, str | None]] = set()
    candidate_rejections = {
        "coordinated_flow_candidates": Counter(),
        "legacy_release_rows": Counter(),
    }
    release_row_reasons = {"legacy": Counter()}
    touchdowns: list[dict[str, Any]] = []
    snapshot_552: dict[str, Any] | None = None
    safety_error: str | None = None
    safety_error_diagnostic: dict[str, Any] | None = None
    safety_error_pair_context: dict[str, Any] | None = None
    steps = 0
    stop_reason: str | None = None
    previous_event_count = len(env.events)

    _write_progress_atomic(
        progress_path,
        _progress_payload(env, step=0, elapsed=time.perf_counter() - started,
                          stop_reason=None, normal_commits=0, emergency_commits=0),
    )

    while steps < int(max_steps) and not _terminal(env):
        before_states = _state_map(env)
        before_landing_plans = {
            int(uid): (uav.committed_plan, bool(uav.emergency_landing))
            for uid, uav in _live_uavs(env).items()
            if uav.committed_plan is not None
            and uav.committed_plan.metadata.get("target_phase") in {"landing", "emergency_landing"}
        }
        actions = {
            str(uid): np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            for uid in env.active
        }
        try:
            _, _, _, _, info = env.step(actions)
        except SafetyInvariantError as exc:
            safety_error = str(exc)
            safety_error_diagnostic = dict(getattr(exc, "diagnostic", {}) or {})
            safety_error_pair_context = _pair_state_context(
                env, safety_error_diagnostic,
            )
            steps += 1
            stop_reason = "safety_error"
            break
        steps += 1
        current_states = _state_map(env)
        after_uavs = _live_uavs(env)
        _record_phase_observations(
            before_states, current_states, now=float(env.now),
            phase_transitions=phase_transitions, first_low_soc=first_low_soc,
            low_soc_threshold=float(config.emergency_trigger_soc),
        )

        release_rows = info.get("release_intent_telemetry", {})
        release_decisions = info.get("release_decisions", {})
        for raw_uid, raw_row in release_rows.items():
            if not isinstance(raw_row, Mapping):
                continue
            uid = int(raw_uid)
            decision = release_decisions.get(str(raw_uid), release_decisions.get(uid))
            coordinated_row = bool(raw_row.get("flow_coordination"))
            reasons = (
                raw_row.get("flow_candidate_rejection_reasons", ())
                if coordinated_row
                else raw_row.get("secondary_constraints", ())
            )
            if isinstance(reasons, (list, tuple, set)):
                if coordinated_row:
                    # Flow rows contain the candidate reasons generated in
                    # this attempt, so a safety-wait retry is still a fresh
                    # candidate observation.
                    candidate_rejections["coordinated_flow_candidates"].update(
                        str(reason) for reason in reasons if reason is not None
                    )
                elif decision not in {None, "safety_wait"}:
                    # Legacy secondary_constraints are row-level reasons.
                    # A safety_wait row may persist across ticks and must not
                    # be counted as a new candidate rejection each time.
                    candidate_rejections["legacy_release_rows"].update(
                        str(reason) for reason in reasons if reason is not None
                    )
                    release_row_reasons["legacy"].update(
                        str(reason) for reason in reasons if reason is not None
                    )
            plan_id = raw_row.get("trajectory_plan_id")
            if decision != "approved" or plan_id is None:
                continue
            marker = (uid, str(plan_id))
            if marker in seen_normal:
                continue
            seen_normal.add(marker)
            normal_commits.append(
                _commit_record(
                    uid, before_states, after_uavs,
                    time_value=float(raw_row.get("approval_time", env.now - config.dt)),
                    kind="normal", coordinated=bool(raw_row.get("flow_coordination")),
                )
            )
            if uid in PAIR_FOCUS_UIDS:
                pair_focus_commits.append(
                    _pair_commit_record(
                        uid, before_states, after_uavs,
                        time_value=float(raw_row.get("approval_time", env.now - config.dt)),
                        kind="normal", coordinated=bool(raw_row.get("flow_coordination")),
                    )
                )

        for event in env.events[previous_event_count:]:
            if not isinstance(event, (tuple, list)) or len(event) < 3:
                continue
            if str(event[0]) != "emergency_departure":
                continue
            uid = int(event[1])
            plan = after_uavs.get(uid)
            plan_id = None if plan is None or plan.committed_plan is None else plan.committed_plan.plan_id
            marker = (uid, plan_id)
            if marker in seen_emergency:
                continue
            seen_emergency.add(marker)
            emergency_commits.append(
                _commit_record(
                    uid, before_states, after_uavs, time_value=float(event[2]),
                    kind="emergency", coordinated=bool(coordinated),
                )
            )
            if uid in PAIR_FOCUS_UIDS:
                pair_focus_commits.append(
                    _pair_commit_record(
                        uid, before_states, after_uavs,
                        time_value=float(event[2]), kind="emergency",
                        coordinated=bool(coordinated),
                    )
                )
        for uid, (before_plan, before_emergency) in before_landing_plans.items():
            after_uav = after_uavs.get(uid)
            if (
                after_uav is not None
                and after_uav.phase.value == "service"
            ):
                touchdowns.append({
                    "uid": int(uid),
                    "time": float(env.now),
                    "soc": float(after_uav.soc),
                    "emergency": bool(before_emergency),
                    "plan_end_time": float(before_plan.end_time),
                    "pad_id": None if after_uav.assigned_lz is None else int(after_uav.assigned_lz),
                })
        previous_event_count = len(env.events)

        if snapshot_552 is None and float(env.now) >= TARGET_DIAGNOSTIC_TIME - 1e-9:
            snapshot_552 = _complete_state_at_552(env)
        if steps % PROGRESS_INTERVAL == 0:
            _write_progress_atomic(
                progress_path,
                _progress_payload(env, step=steps, elapsed=time.perf_counter() - started,
                                  stop_reason=None, normal_commits=len(normal_commits),
                                  emergency_commits=len(emergency_commits)),
            )

    if stop_reason is None:
        if _terminal(env):
            stop_reason = "terminal"
        elif steps >= int(max_steps):
            stop_reason = "max_steps"
        elif float(env.now) >= float(config.max_simulation_seconds):
            stop_reason = "max_simulation_seconds"
        else:
            stop_reason = "stopped"
    _write_progress_atomic(
        progress_path,
        _progress_payload(env, step=steps, elapsed=time.perf_counter() - started,
                          stop_reason=stop_reason, normal_commits=len(normal_commits),
                          emergency_commits=len(emergency_commits)),
    )

    outcome_counts = _outcome_counts(env)
    # ``env.step`` normally copies runtime-traffic counters into
    # ``env.telemetry`` after protection returns.  A SafetyInvariantError is
    # raised before that copy, so merge the shield-owned counters explicitly
    # for failed diagnostic ticks as well.
    telemetry = dict(env.telemetry)
    runtime_traffic = getattr(env, "runtime_traffic", None)
    runtime_telemetry = getattr(runtime_traffic, "telemetry", {})
    if isinstance(runtime_telemetry, Mapping):
        telemetry.update(runtime_telemetry)
    requested_ids = {int(value) for value in env.ledger.requested}
    resolved_ids = {int(value) for value in env.ledger.outcomes}
    unresolved_ids = {int(value) for value in env.ledger.unresolved}
    def _finite_metric(key: str) -> float | None:
        value = telemetry.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    safety_summary = {
        "safety_error": safety_error,
        "safety_interventions": int(telemetry.get("safety_interventions", 0)),
        "hard_separation_violations": int(telemetry.get("hard_separation_violations", 0)),
        "nmac": int(telemetry.get("nmac", 0)),
        "collisions": int(telemetry.get("collisions", 0)),
        "operational_buffer_entries": int(telemetry.get("operational_buffer_entries", 0)),
        "minimum_actual_separation": _finite_metric("minimum_actual_separation"),
        "minimum_swept_separation": _finite_metric("minimum_swept_separation"),
        "shield_fail_closed": int(telemetry.get("shield_fail_closed", 0)),
        "joint_safety_fail_closed": int(telemetry.get("joint_safety_fail_closed", 0)),
        "checked_executed_plan_hash_mismatches": int(
            telemetry.get("checked_executed_plan_hash_mismatches", 0)
        ),
        "requested": len(requested_ids),
        "resolved": len(resolved_ids),
        "unresolved": len(unresolved_ids),
        "conflict_event_count": len(env.conflict_events),
        "first_conflict": None if not env.conflict_events else env.conflict_events[0],
    }
    wall_seconds = time.perf_counter() - started
    result = {
        "diagnostic_only": True,
        "training_called": False,
        "PPO_ALLOWED": False,
        "seed": int(seed),
        "coordinated": bool(coordinated),
        "config": asdict(config),
        "wall_seconds": float(wall_seconds),
        "steps": int(steps),
        "sim_time_seconds": float(env.now),
        "stop_reason": stop_reason,
        "outcome_counts": outcome_counts,
        "telemetry": telemetry,
        "safety_summary": safety_summary,
        "safety_error_diagnostic": safety_error_diagnostic,
        "safety_error_pair_context": safety_error_pair_context,
        "energy_outcomes": list(env.energy_outcomes),
        "uid3_uid4": {
            "low_soc_threshold": float(config.emergency_trigger_soc),
            "phase_transitions": phase_transitions,
            "first_low_soc": first_low_soc,
        },
        "normal_commits": normal_commits,
        "emergency_commits": emergency_commits,
        "pair_focus_commits": pair_focus_commits,
        "touchdowns": touchdowns,
        "state_at_552": snapshot_552,
        "candidate_rejection_counts": {
            bucket: dict(sorted(counter.items()))
            for bucket, counter in candidate_rejections.items()
        },
        "release_row_reason_counts": {
            bucket: dict(sorted(counter.items()))
            for bucket, counter in release_row_reasons.items()
        },
        "progress_path": str(progress_path),
    }
    serialized = json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    encoded = serialized.encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise RuntimeError(
            f"diagnostic output exceeds {MAX_OUTPUT_BYTES} bytes: {len(encoded)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return _jsonable(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--coordinated", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_diagnostic(
        seed=args.seed,
        coordinated=bool(args.coordinated),
        max_steps=args.max_steps,
        output=args.output,
    )
    print(json.dumps({
        "output": str(args.output),
        "coordinated": result["coordinated"],
        "steps": result["steps"],
        "stop_reason": result["stop_reason"],
        "outcome_counts": result["outcome_counts"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
