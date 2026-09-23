"""UID-specific forensic trace for the dense continuous final-approach run.

This diagnostic replays the frozen rule-only scenario used by the dense30
preflight and instruments only two target UAVs.  The wrappers record emergency
landing attempts and emergency-plan safety commits without changing simulator
decisions.  The normal environment outcome is still the authority for the
terminal counts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from enum import Enum
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from terminal_marl.core.config import RingRole
from terminal_marl.core.landing_zone import LZStatus
from terminal_marl.core.low_level_joint_action_v4_1 import RESIDUAL_VALUES
from terminal_marl.core.merge_scheduler import LowLevelIntent, RepairCandidate
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.core.uav import FlightPhase
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from terminal_marl.training.diagnose_energy_reachability_guard_smoke import (
    build_config,
)


TARGET_UIDS = (13, 14)
TELEMETRY_KEYS = (
    "emergency_attempts",
    "emergency_blocked_lz",
    "emergency_blocked_dynamics",
    "emergency_blocked_energy",
    "emergency_blocked_safety",
    "emergency_committed",
)
EXPECTED_TERMINAL = {
    "steps_executed": 1117,
    "outcome_counts": {"completed": 20, "emergency_landed": 8, "failed_energy": 2},
}
EXPECTED_MIXED_TERMINAL = {
    "steps_executed": 1032,
    "outcome_counts": {"completed": 23, "emergency_landed": 6, "failed_energy": 1},
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _terminal(env: TerminalParallelEnv) -> bool:
    return bool(
        env.arrivals.exhausted()
        and not env.active
        and not env.admission.queue.items
        and not env.service_uavs
    )


def _telemetry_snapshot(env: TerminalParallelEnv) -> dict[str, int]:
    return {key: int(env.telemetry.get(key, 0)) for key in TELEMETRY_KEYS}


def _telemetry_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in TELEMETRY_KEYS}


def _lz_state(env: TerminalParallelEnv, uav: Any) -> dict[str, list[int]]:
    free = [int(zone.lz_id) for zone in env.lz_manager.zones if zone.status is LZStatus.FREE]
    compatible = [
        int(zone_id)
        for zone_id in env.lz_manager.available_zones(uav, env.now)
    ]
    return {"free_lz_ids": free, "compatible_lz_ids": compatible}


def _uav_state(uav: Any) -> dict[str, Any]:
    return {
        "uid": int(uav.uid),
        "arrival_id": int(uav.arrival_id),
        "soc": float(uav.soc),
        "energy_remaining": float(uav.energy_remaining),
        "energy_consumed": float(uav.energy_consumed),
        "phase": None if uav.phase is None else uav.phase.value,
        "ring_role": None if uav.ring_role is None else uav.ring_role.value,
        "assigned_lz": None if uav.assigned_lz is None else int(uav.assigned_lz),
        "committed": bool(uav.committed_plan is not None),
        "emergency_landing": bool(uav.emergency_landing),
        "safety_retry_after": float(uav.safety_retry_after),
        "emergency_retry_after": float(uav.emergency_retry_after),
        "terminal_outcome": (
            None if uav.terminal_outcome is None else uav.terminal_outcome.value
        ),
        "terminal_reason": uav.terminal_reason,
    }


def _attach_targets(
    env: TerminalParallelEnv,
    target_uids: tuple[int, ...],
    tracked: dict[int, Any],
) -> tuple[int, ...]:
    """Attach target objects when a mixed-arrival UAV is admitted."""

    attached: list[int] = []
    for uid in target_uids:
        if uid not in tracked and uid in env.active:
            tracked[uid] = env.active[uid]
            attached.append(uid)
    return tuple(attached)


def _emergency_call_record(
    env: TerminalParallelEnv,
    uav: Any,
    *,
    force: bool,
    before: Mapping[str, Any],
    before_telemetry: Mapping[str, int],
    result: bool | None,
    error: str | None,
) -> dict[str, Any]:
    after_telemetry = _telemetry_snapshot(env)
    after = _uav_state(uav)
    return {
        "time": float(before["time"]),
        "uid": int(uav.uid),
        "force": bool(force),
        "before": before,
        "after": after,
        "before_lz": before["lz"],
        "after_lz": _lz_state(env, uav),
        "telemetry_delta": _telemetry_delta(before_telemetry, after_telemetry),
        "result": None if result is None else bool(result),
        "error": error,
    }


def _install_auditors(
    env: TerminalParallelEnv,
    *,
    target_uids: tuple[int, ...],
    emergency_calls: list[dict[str, Any]],
    emergency_commits: list[dict[str, Any]],
    on_first_emergency: Any | None = None,
) -> None:
    """Install instance-local wrappers; both wrappers preserve original calls."""

    original_try = env._try_emergency_landing
    original_commit = env._commit
    target_set = set(int(uid) for uid in target_uids)
    first_emergency_seen: set[int] = set()

    def audited_try(uav: Any, *, force: bool = False) -> bool:
        if int(uav.uid) not in target_set:
            return original_try(uav, force=force)
        emergency_eligible = (
            uav.ring_role in {RingRole.OUTER, RingRole.MIDDLE, RingRole.INNER}
            and (force or float(uav.soc) <= float(env.config.emergency_trigger_soc) + 1e-12)
        )
        if emergency_eligible and int(uav.uid) not in first_emergency_seen:
            first_emergency_seen.add(int(uav.uid))
            if on_first_emergency is not None:
                on_first_emergency(uav)
        before_telemetry = _telemetry_snapshot(env)
        before = _uav_state(uav)
        before["time"] = float(env.now)
        before["lz"] = _lz_state(env, uav)
        result: bool | None = None
        error: str | None = None
        try:
            result = bool(original_try(uav, force=force))
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            emergency_calls.append(
                _emergency_call_record(
                    env,
                    uav,
                    force=force,
                    before=before,
                    before_telemetry=before_telemetry,
                    result=result,
                    error=error,
                )
            )

    def audited_commit(uav: Any, plan: Any) -> bool:
        metadata = getattr(plan, "metadata", {}) or {}
        is_emergency = bool(
            metadata.get("target_phase") == "emergency_landing"
            or getattr(plan, "mode", None) == "emergency_landing"
        )
        if int(uav.uid) not in target_set or not is_emergency:
            return original_commit(uav, plan)
        before_telemetry = _telemetry_snapshot(env)
        before = _uav_state(uav)
        before["time"] = float(env.now)
        before["lz"] = _lz_state(env, uav)
        result: bool | None = None
        error: str | None = None
        try:
            result = bool(original_commit(uav, plan))
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            after_telemetry = _telemetry_snapshot(env)
            emergency_commits.append(
                {
                    "time": float(before["time"]),
                    "uid": int(uav.uid),
                    "plan_id": getattr(plan, "plan_id", None),
                    "trajectory_hash": getattr(plan, "trajectory_hash", None),
                    "target_phase": metadata.get("target_phase"),
                    "lz_id": metadata.get("lz_id"),
                    "source_ring_role": metadata.get("source_ring_role"),
                    "start_time": getattr(plan, "start_time", None),
                    "end_time": getattr(plan, "end_time", None),
                    "point_count": len(getattr(plan, "points", ())),
                    "before": before,
                    "after": _uav_state(uav),
                    "after_lz": _lz_state(env, uav),
                    "last_safety_decision": env.last_safety_decisions.get(uav.uid),
                    "telemetry_delta": _telemetry_delta(before_telemetry, after_telemetry),
                    "result": None if result is None else bool(result),
                    "error": error,
                }
            )

    env._try_emergency_landing = audited_try
    env._commit = audited_commit


def _release_row(env: TerminalParallelEnv, uid: int, info: Mapping[str, Any]) -> dict[str, Any] | None:
    telemetry = info.get("release_intent_telemetry", {})
    row = telemetry.get(str(uid), telemetry.get(uid)) if isinstance(telemetry, Mapping) else None
    if not isinstance(row, Mapping):
        row = env.last_release_telemetry.get(uid)
    if not isinstance(row, Mapping):
        return None
    secondary = row.get("secondary_constraints", [])
    return {
        "primary_blocker": row.get("primary_blocker"),
        "secondary_constraints": [str(item) for item in secondary] if isinstance(secondary, (list, tuple, set)) else [],
        "decision_history_tail": [str(item) for item in row.get("decision_history", [])[-3:]] if isinstance(row.get("decision_history", []), (list, tuple)) else [],
        "first_attempt_time": row.get("first_attempt_time"),
        "attempt_count": row.get("attempt_count"),
        "retry_count": row.get("retry_count"),
    }


def _step_record(
    env: TerminalParallelEnv,
    uav: Any,
    *,
    step: int,
    info: Mapping[str, Any],
    release_decisions: dict[int, str | None],
) -> dict[str, Any]:
    uid = int(uav.uid)
    row = _uav_state(uav)
    row.update(
        {
            "step": int(step),
            "time": float(env.now),
            "release_decision": release_decisions.get(uid),
            "release_telemetry": _release_row(env, uid, info),
            "lz": _lz_state(env, uav),
        }
    )
    return row


def _record_target_states(
    env: TerminalParallelEnv,
    *,
    step: int,
    info: Mapping[str, Any],
    tracked: Mapping[int, Any],
    release_decisions: dict[int, str | None],
    records: dict[str, list[dict[str, Any]]],
    last_signature: dict[int, tuple[Any, ...]],
    last_record_time: dict[int, float],
) -> None:
    for uid, uav in tracked.items():
        release_decisions[uid] = env.last_release_decisions.get(uid, release_decisions.get(uid))
        row = _step_record(
            env,
            uav,
            step=step,
            info=info,
            release_decisions=release_decisions,
        )
        release = row["release_telemetry"] or {}
        signature = (
            row["phase"],
            row["ring_role"],
            row["terminal_outcome"],
            row["release_decision"],
            release.get("primary_blocker"),
            tuple(release.get("secondary_constraints", [])),
        )
        changed = last_signature.get(uid) != signature
        interval = float(env.now) - float(last_record_time.get(uid, -math.inf)) >= 10.0 - 1e-9
        if not records[str(uid)] or changed or interval:
            records[str(uid)].append(row)
            last_signature[uid] = signature
            last_record_time[uid] = float(env.now)


def _scan_target_proposals(env: TerminalParallelEnv, uav: Any) -> dict[str, Any]:
    """Probe the unchanged 25-proposal catalogue without committing a plan."""

    if uav.uid not in env.active or uav.ring_role not in {RingRole.OUTER, RingRole.MIDDLE, RingRole.INNER}:
        return {"uid": int(uav.uid), "time": float(env.now), "status": "not_active_ring_holder", "rows": []}
    target_phase = {
        RingRole.OUTER: (RingRole.MIDDLE, "o2m", FlightPhase.TRANSFER_O2M),
        RingRole.MIDDLE: (RingRole.INNER, "m2i", FlightPhase.TRANSFER_M2I),
        RingRole.INNER: ("final", "i2f", FlightPhase.FINAL),
    }[uav.ring_role]
    target, interface, _phase = target_phase
    lz_id = None
    if target == "final":
        lz_id = env.lz_manager.choose_rule_zone(uav, env.now)
        if lz_id is None:
            return {"uid": int(uav.uid), "time": float(env.now), "status": "no_lz", "rows": []}
    before_telemetry = dict(env.telemetry)
    rows: list[dict[str, Any]] = []
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    for speed in RESIDUAL_VALUES:
        for merge in RESIDUAL_VALUES:
            intent = LowLevelIntent(1.0, float(speed), float(merge))
            candidate = RepairCandidate("diagnostic_catalogue", intent)
            try:
                proposal, plan, reason = env._candidate_plan(uav, candidate, target, snapshot, lz_id)
                rows.append(
                    {
                        "speed_residual": float(speed),
                        "merge_time_residual": float(merge),
                        "merge_time": None if proposal is None else float(proposal.merge_time),
                        "reason": reason,
                        "plan_id": None if plan is None else plan.plan_id,
                        "certified": bool(proposal is not None and plan is not None and reason is None),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "speed_residual": float(speed),
                        "merge_time_residual": float(merge),
                        "merge_time": None,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "plan_id": None,
                        "certified": False,
                    }
                )
    env.telemetry.clear()
    env.telemetry.update(before_telemetry)
    counts = Counter(str(row["reason"]) for row in rows)
    return {
        "uid": int(uav.uid),
        "time": float(env.now),
        "status": "scanned",
        "phase": uav.phase.value,
        "ring_role": uav.ring_role.value if uav.ring_role is not None else None,
        "target": target.value if isinstance(target, RingRole) else target,
        "interface": interface,
        "lz_id": lz_id,
        "proposal_count": len(rows),
        "certified_count": sum(bool(row["certified"]) for row in rows),
        "reason_counts": dict(sorted(counts.items())),
        "rows": rows,
    }


def _counterfactual_emergency_lz_scan(env: TerminalParallelEnv, uav: Any) -> dict[str, Any]:
    """Evaluate every LZ as an emergency candidate without reserving or committing."""

    current = _uav_state(uav)
    current.update({"time": float(env.now), "lz": _lz_state(env, uav)})
    committed = {
        uid: other.committed_plan
        for uid, other in env.active.items()
        if other.committed_plan is not None
    }
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    checked_plan_ids = list(env.safety.checked_plan_ids)
    checked_hashes = dict(env.safety.checked_trajectory_hashes)
    telemetry = dict(env.telemetry)
    energy_diagnostics = dict(env.energy_reachability_diagnostics)
    rows: list[dict[str, Any]] = []
    try:
        for zone in env.lz_manager.zones:
            row: dict[str, Any] = {
                "lz_id": int(zone.lz_id),
                "status": zone.status.value,
                "reserved_by": None if zone.reserved_by is None else int(zone.reserved_by),
                "occupied_by": None if zone.occupied_by is None else int(zone.occupied_by),
                "mission_compatible": bool(uav.mission_type in zone.compatible_missions),
                "plan_id": None,
                "plan_duration_seconds": None,
                "energy_required": None,
                "energy_slack": None,
                "energy_pass": False,
                "safety_pass": False,
                "both_pass": False,
                "verdict_reason": None,
                "plan_error": None,
            }
            try:
                plan = env.generator.emergency_landing(uav, zone, env.airspace, env.now)
                required = float(env.energy.emergency_required_energy(uav, plan))
                verdict = env.safety.filter_plan(plan, committed, snapshot)
                row.update(
                    {
                        "plan_id": plan.plan_id,
                        "plan_duration_seconds": float(plan.end_time - plan.start_time),
                        "energy_required": required,
                        "energy_slack": float(uav.energy_remaining - required),
                        "energy_pass": bool(uav.energy_remaining + 1e-9 >= required),
                        "safety_pass": bool(verdict.approved),
                        "both_pass": bool(uav.energy_remaining + 1e-9 >= required and verdict.approved),
                        "verdict_reason": verdict.reason or "approved",
                    }
                )
            except Exception as exc:
                row["plan_error"] = f"{type(exc).__name__}: {exc}"
                row["verdict_reason"] = "plan_error"
            rows.append(row)
    finally:
        # filter_plan records successful probe IDs for audit bookkeeping.  A
        # counterfactual must leave those counters identical to the replay.
        env.safety.checked_plan_ids[:] = checked_plan_ids
        env.safety.checked_trajectory_hashes.clear()
        env.safety.checked_trajectory_hashes.update(checked_hashes)
        env.telemetry.clear()
        env.telemetry.update(telemetry)
        env.energy_reachability_diagnostics.clear()
        env.energy_reachability_diagnostics.update(energy_diagnostics)
    reasons = Counter(str(row["verdict_reason"]) for row in rows)
    return {
        "uid": int(uav.uid),
        "time": float(env.now),
        "phase": current["phase"],
        "ring_role": current["ring_role"],
        "soc": current["soc"],
        "energy_remaining": current["energy_remaining"],
        "free_compatible_lz_ids": current["lz"]["compatible_lz_ids"],
        "free_lz_ids": current["lz"]["free_lz_ids"],
        "lz_count": len(rows),
        "energy_pass_count": sum(bool(row["energy_pass"]) for row in rows),
        "safety_pass_count": sum(bool(row["safety_pass"]) for row in rows),
        "both_pass_count": sum(bool(row["both_pass"]) for row in rows),
        "verdict_reason_counts": dict(sorted(reasons.items())),
        "rows": rows,
    }


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(json.dumps(_jsonable(payload), sort_keys=True), flush=True)


def run_trace(
    *,
    seed: int = 840626,
    max_steps: int = 1200,
    progress_every_steps: int = 100,
    output: Path,
    compare_expected: bool | None = None,
    mixed_total_uavs: int | None = None,
    mixed_warm_start_uavs: int = 12,
    mixed_arrival_interval_seconds: float = 8.0,
    target_uids: Sequence[int] | None = None,
    proposal_scan_steps: Sequence[int] | None = None,
    counterfactual_scan_steps: Sequence[int] | None = None,
    max_scans_per_kind: int = 6,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    progress_path = Path(f"{output}.progress.json")
    if progress_path.exists():
        raise FileExistsError(f"refusing to overwrite existing progress output: {progress_path}")
    if max_steps <= 0 or progress_every_steps <= 0:
        raise ValueError("max_steps and progress_every_steps must be positive")
    if max_scans_per_kind <= 0:
        raise ValueError("max_scans_per_kind must be positive")
    selected_targets = tuple(
        TARGET_UIDS if target_uids is None else sorted({int(uid) for uid in target_uids})
    )
    if not selected_targets:
        raise ValueError("target_uids must not be empty")
    mixed_mode = mixed_total_uavs is not None
    if compare_expected is None:
        expected_for_mode = EXPECTED_MIXED_TERMINAL if mixed_mode else EXPECTED_TERMINAL
        compare_expected = max_steps >= expected_for_mode["steps_executed"]

    config = build_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=True,
        max_simulation_seconds=float(max_steps),
        dense_warm_start_uavs=None if mixed_mode else 30,
        mixed_total_uavs=mixed_total_uavs,
        mixed_warm_start_uavs=mixed_warm_start_uavs,
        mixed_arrival_interval_seconds=mixed_arrival_interval_seconds,
        final_approach_geometry_mode="continuous_inward_v1",
    )
    started = time.perf_counter()
    env = TerminalParallelEnv(config)
    env.reset(seed=int(seed))
    tracked: dict[int, Any] = {}
    _attach_targets(env, selected_targets, tracked)
    if not mixed_mode and set(tracked) != set(selected_targets):
        raise RuntimeError(f"target UIDs not present after reset: {sorted(tracked)}")
    emergency_calls: list[dict[str, Any]] = []
    emergency_commits: list[dict[str, Any]] = []
    target_records: dict[str, list[dict[str, Any]]] = {str(uid): [] for uid in selected_targets}
    release_decisions: dict[int, str | None] = {uid: None for uid in selected_targets}
    last_signature: dict[int, tuple[Any, ...]] = {}
    last_record_time: dict[int, float] = {}
    proposal_scans: list[dict[str, Any]] = []
    scan_counts = {"release_proposal_catalogue": 0, "emergency_lz_counterfactual": 0}
    scan_targets = tuple(
        (100, 200, 500, 750)
        if proposal_scan_steps is None and not mixed_mode
        else (() if proposal_scan_steps is None else tuple(int(step) for step in proposal_scan_steps))
        if proposal_scan_steps is not None or not mixed_mode
        else (500, 750, 900)
    )
    counterfactual_targets = (
        {200: (13,), 290: (13,), 400: (14,), 495: (14,)}
        if counterfactual_scan_steps is None and not mixed_mode
        else {}
    )
    mixed_counterfactual_steps = tuple(
        (500, 750, 900)
        if counterfactual_scan_steps is None
        else tuple(int(step) for step in counterfactual_scan_steps)
    ) if mixed_mode else ()
    scans_done: set[int] = set()
    counterfactual_scans_done: set[int] = set()
    peak_active = 0
    peak_ring_occupancy = {role.value: 0 for role in RingRole}

    def update_peak_occupancy() -> None:
        nonlocal peak_active
        peak_active = max(peak_active, len(env.active))
        for role in RingRole:
            count = sum(1 for item in env.active.values() if item.ring_role is role)
            peak_ring_occupancy[role.value] = max(peak_ring_occupancy[role.value], count)

    def append_scan(kind: str, uav: Any) -> None:
        if scan_counts[kind] >= max_scans_per_kind:
            return
        if uav.uid not in env.active or uav.ring_role not in {RingRole.OUTER, RingRole.MIDDLE, RingRole.INNER}:
            return
        if kind == "release_proposal_catalogue":
            payload = _scan_target_proposals(env, uav)
        else:
            payload = _counterfactual_emergency_lz_scan(env, uav)
        proposal_scans.append({"kind": kind, **payload})
        scan_counts[kind] += 1

    def on_first_emergency(uav: Any) -> None:
        if not mixed_mode:
            return
        tracked.setdefault(int(uav.uid), uav)
        append_scan("release_proposal_catalogue", uav)
        append_scan("emergency_lz_counterfactual", uav)

    _install_auditors(
        env,
        target_uids=selected_targets,
        emergency_calls=emergency_calls,
        emergency_commits=emergency_commits,
        on_first_emergency=on_first_emergency,
    )
    update_peak_occupancy()
    safety_error: str | None = None
    steps = 0
    info: Mapping[str, Any] = {}
    _record_target_states(
        env,
        step=0,
        info=info,
        tracked=tracked,
        release_decisions=release_decisions,
        records=target_records,
        last_signature=last_signature,
        last_record_time=last_record_time,
    )
    _write_progress(
        progress_path,
        {
            "step": 0,
            "sim_time": float(env.now),
            "wall_elapsed": time.perf_counter() - started,
            "targets": {str(uid): _uav_state(uav) for uid, uav in tracked.items()},
            "terminal": _terminal(env),
            "safety_error": None,
            "status": "running",
        },
    )
    terminal = _terminal(env)
    while steps < max_steps and not terminal:
        if not mixed_mode and int(round(env.now)) in counterfactual_targets:
            for uid in counterfactual_targets[int(round(env.now))]:
                if uid in tracked:
                    append_scan("emergency_lz_counterfactual", tracked[uid])
        actions = {
            str(uid): np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            for uid in env.active
        }
        try:
            _, _, _, _, info = env.step(actions)
        except SafetyInvariantError as exc:
            safety_error = str(exc)
            steps += 1
            break
        steps += 1
        _attach_targets(env, selected_targets, tracked)
        for raw_uid, decision in info.get("release_decisions", {}).items():
            uid = int(raw_uid)
            if uid in release_decisions:
                release_decisions[uid] = str(decision)
        _record_target_states(
            env,
            step=steps,
            info=info,
            tracked=tracked,
            release_decisions=release_decisions,
            records=target_records,
            last_signature=last_signature,
            last_record_time=last_record_time,
        )
        if steps in scan_targets:
            for uid in selected_targets:
                if uid in tracked:
                    append_scan("release_proposal_catalogue", tracked[uid])
            scans_done.add(steps)
        if mixed_mode and steps in mixed_counterfactual_steps:
            for uid in selected_targets:
                if uid in tracked:
                    append_scan("emergency_lz_counterfactual", tracked[uid])
            counterfactual_scans_done.add(steps)
        update_peak_occupancy()
        terminal = _terminal(env)
        if steps % progress_every_steps == 0 and not terminal and steps < max_steps:
            _write_progress(
                progress_path,
                {
                    "step": int(steps),
                    "sim_time": float(env.now),
                    "wall_elapsed": time.perf_counter() - started,
                    "targets": {str(uid): _uav_state(uav) for uid, uav in tracked.items()},
                    "terminal": bool(terminal),
                    "safety_error": safety_error,
                    "status": "running",
                },
            )

    terminal = _terminal(env)
    outcomes = {str(k): v.value for k, v in env.ledger.outcomes.items()}
    outcome_counts = Counter(outcomes.values())
    expected_terminal = EXPECTED_MIXED_TERMINAL if mixed_mode else EXPECTED_TERMINAL
    observed_terminal = {
        "steps_executed": int(steps),
        "outcome_counts": dict(sorted(outcome_counts.items())),
    }
    comparison = {
        "enabled": bool(compare_expected),
        "expected": expected_terminal,
        "observed": observed_terminal,
        "matches": bool(observed_terminal == expected_terminal) if compare_expected else None,
    }
    result = {
        "diagnostic_only": True,
        "training_called": False,
        "teacher_used": False,
        "PPO_ALLOWED": False,
        "config": {
            "seed": int(seed),
            "warm_start_uavs": int(config.warm_start_uavs),
            "total_requested_uavs": int(config.total_requested_uavs or 0),
            "initial_soc_min": float(config.initial_soc_min),
            "initial_soc_max": float(config.initial_soc_max),
            "energy_model_mode": config.energy_model_mode,
            "final_approach_geometry_mode": config.final_approach_geometry_mode,
            "energy_reachability_guard_enabled": bool(config.energy_reachability_guard_enabled),
            "emergency_priority_guard_enabled": bool(config.emergency_priority_guard_enabled),
            "fixed_action": [1.0, 0.0, 0.0],
            "target_uids": list(selected_targets),
            "mixed_total_uavs": None if mixed_total_uavs is None else int(mixed_total_uavs),
            "mixed_warm_start_uavs": int(mixed_warm_start_uavs),
            "mixed_arrival_interval_seconds": float(mixed_arrival_interval_seconds),
            "proposal_scan_steps": list(scan_targets),
            "counterfactual_scan_steps": list(
                counterfactual_targets if not mixed_mode else mixed_counterfactual_steps
            ),
            "max_scans_per_kind": int(max_scans_per_kind),
        },
        "steps_executed": int(steps),
        "sim_time_seconds": float(env.now),
        "terminal": bool(terminal),
        "horizon_reached": bool(not terminal and steps >= max_steps),
        "safety_error": safety_error,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "outcomes": outcomes,
        "comparison": comparison,
        "target_records": target_records,
        "emergency_attempts_by_target": emergency_calls,
        "emergency_plan_commits_by_target": emergency_commits,
        "proposal_scans": proposal_scans,
        "scan_steps": sorted(scans_done),
        "counterfactual_scan_steps": sorted(counterfactual_scans_done),
        "scan_counts": dict(scan_counts),
        "energy_outcomes": list(env.energy_outcomes),
        "safety_telemetry": dict(env.telemetry),
        "failed_energy": int(outcome_counts.get("failed_energy", 0)),
        "emergency_committed": int(env.telemetry.get("emergency_committed", 0)),
        "peak_active": int(peak_active),
        "peak_ring_occupancy": dict(peak_ring_occupancy),
        "simulation_wall_time_seconds": float(time.perf_counter() - started),
        "progress_path": str(progress_path),
        "progress_every_steps": int(progress_every_steps),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_progress(
        progress_path,
        {
            "step": int(steps),
            "sim_time": float(env.now),
            "wall_elapsed": result["simulation_wall_time_seconds"],
            "targets": {str(uid): _uav_state(uav) for uid, uav in tracked.items()},
            "terminal": bool(terminal),
            "safety_error": safety_error,
            "status": "terminal" if terminal else ("safety_error" if safety_error else "horizon"),
            "comparison_matches": comparison["matches"],
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=840626)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--progress-every-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mixed-total-uavs", type=int, default=None)
    parser.add_argument("--mixed-warm-start-uavs", type=int, default=12)
    parser.add_argument("--mixed-arrival-interval-seconds", type=float, default=8.0)
    parser.add_argument("--target-uids", type=int, nargs="+", default=None)
    parser.add_argument("--proposal-scan-steps", type=int, nargs="*", default=None)
    parser.add_argument("--counterfactual-scan-steps", type=int, nargs="*", default=None)
    parser.add_argument("--max-scans-per-kind", type=int, default=6)
    parser.add_argument(
        "--compare-expected",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compare terminal counts with the recorded outcome for the selected scenario.",
    )
    args = parser.parse_args(argv)
    result = run_trace(
        seed=args.seed,
        max_steps=args.max_steps,
        progress_every_steps=args.progress_every_steps,
        output=args.output,
        compare_expected=args.compare_expected,
        mixed_total_uavs=args.mixed_total_uavs,
        mixed_warm_start_uavs=args.mixed_warm_start_uavs,
        mixed_arrival_interval_seconds=args.mixed_arrival_interval_seconds,
        target_uids=args.target_uids,
        proposal_scan_steps=args.proposal_scan_steps,
        counterfactual_scan_steps=args.counterfactual_scan_steps,
        max_scans_per_kind=args.max_scans_per_kind,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "steps_executed": result["steps_executed"],
                "terminal": result["terminal"],
                "outcome_counts": result["outcome_counts"],
                "failed_energy": result["failed_energy"],
                "comparison": result["comparison"],
                "target_emergency_calls": len(result["emergency_attempts_by_target"]),
                "target_emergency_commits": len(result["emergency_plan_commits_by_target"]),
                "peak_active": result["peak_active"],
                "peak_ring_occupancy": result["peak_ring_occupancy"],
                "scan_counts": result["scan_counts"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["safety_error"] is None and result["comparison"]["matches"] is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
