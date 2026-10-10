"""Read-only exact replay evidence for the V7 run0 dev-after-0080 failure."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import (
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import run_p3_adaptive_kl_formal_v7 as formal_v7


STAGE = "P3_V7_RUN0_DEV80_ENERGY_FORENSICS_V1"
TARGET_ARRIVAL_ID = 15
EXPECTED_STEPS = 897
THRESHOLDS = (0.65, 0.40, 0.25)


def _jsonable(value: Any) -> Any:
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return repr(value)


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_by_uid(env: TerminalParallelEnv, uid: int) -> Any | None:
    if int(uid) in env.active:
        return env.active[int(uid)]
    if int(uid) in env.service_uavs:
        return env.service_uavs[int(uid)]
    for uav in env.completed_uavs:
        if int(uav.uid) == int(uid):
            return uav
    return None


def _find_by_arrival(env: TerminalParallelEnv, arrival_id: int) -> Any | None:
    candidates = [*env.active.values(), *env.service_uavs.values(), *env.completed_uavs]
    for uav in candidates:
        if int(uav.arrival_id) == int(arrival_id):
            return uav
    return None


def _uav_state(env: TerminalParallelEnv, uav: Any | None) -> dict[str, Any]:
    if uav is None:
        return {"present": False}
    plan = uav.committed_plan
    return {
        "present": True,
        "uid": int(uav.uid),
        "arrival_id": int(uav.arrival_id),
        "phase": getattr(uav.phase, "value", uav.phase),
        "ring_role": None if uav.ring_role is None else getattr(uav.ring_role, "value", uav.ring_role),
        "soc": float(uav.soc),
        "energy_remaining": float(uav.energy_remaining),
        "energy_consumed": float(uav.energy_consumed),
        "committed": plan is not None,
        "landing_intent": None if plan is None else {
            "plan_id": getattr(plan, "plan_id", None),
            "mode": getattr(plan, "mode", None),
            "target_phase": getattr(plan, "metadata", {}).get("target_phase"),
            "end_time": getattr(plan, "end_time", None),
        },
        "assigned_lz": None if uav.assigned_lz is None else int(uav.assigned_lz),
        "emergency_landing": bool(uav.emergency_landing),
        "safety_retry_after": float(uav.safety_retry_after),
        "emergency_retry_after": float(uav.emergency_retry_after),
        "terminal_outcome": None if uav.terminal_outcome is None else getattr(uav.terminal_outcome, "value", uav.terminal_outcome),
        "terminal_reason": uav.terminal_reason,
        "completion_time": getattr(uav, "completion_time", None),
    }


def _occupancy(env: TerminalParallelEnv) -> dict[str, Any]:
    roles = list(env.airspace._committed)
    return {
        "actual": {str(role.value): int(env.airspace.occupancy(role)) for role in roles},
        "committed_incoming": {
            str(role.value): int(env.airspace._committed.get(role, 0)) for role in roles
        },
        "active_ring_holders": {
            str(role.value): sum(
                1 for uav in env.active.values() if uav.ring_role is role
            )
            for role in roles
        },
    }


def _energy(env: TerminalParallelEnv, uid: int | None) -> dict[str, Any]:
    if uid is None:
        return {}
    uav = _find_by_uid(env, uid)
    result = dict(env.energy_reachability_diagnostics.get(uid, {}))
    result.update({
        "current_soc": None if uav is None else float(uav.soc),
        "current_energy": None if uav is None else float(uav.energy_remaining),
    })
    return _copy_json(result)


def _telemetry_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(set(before) | set(after), key=str):
        left, right = before.get(key, 0), after.get(key, 0)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            result[str(key)] = right - left
    return result


def _requested(actions: Any, uid: int | None) -> dict[str, Any] | None:
    if uid is None or not isinstance(actions, Mapping):
        return None
    raw = actions.get(str(uid), actions.get(uid))
    if raw is None:
        return None
    result: dict[str, Any] = {"raw": _copy_json(raw)}
    try:
        decoded = hierarchical_from_normalized_v4_2(np.asarray(raw, dtype=np.float32))
        result.update({
            "gate": int(decoded.gate),
            "proposal": None if decoded.proposal is None else int(decoded.proposal),
        })
    except (TypeError, ValueError) as exc:
        result["decode_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _intent(intent: Any) -> dict[str, Any]:
    return {
        "release_request": float(intent.release_request),
        "speed_residual": float(intent.speed_residual),
        "merge_time_residual": float(intent.merge_time_residual),
    }


def _shortlist(shortlist: Any) -> list[dict[str, Any]] | None:
    if shortlist is None:
        return None
    return [_intent(item) for item in tuple(shortlist)]


def _release_row(env: TerminalParallelEnv, uid: int) -> Any:
    return _copy_json(env.last_release_telemetry.get(uid))


def _install_observers(
    env: TerminalParallelEnv,
    *,
    timeline: list[dict[str, Any]],
    release_attempts: list[dict[str, Any]],
    emergency_attempts: list[dict[str, Any]],
    energy_checks: list[dict[str, Any]],
) -> None:
    target_uid: int | None = None

    def target() -> Any | None:
        nonlocal target_uid
        if target_uid is not None:
            found = _find_by_uid(env, target_uid)
            if found is not None:
                return found
        found = _find_by_arrival(env, TARGET_ARRIVAL_ID)
        if found is not None:
            target_uid = int(found.uid)
        return found

    def snapshot(actions: Any = None) -> dict[str, Any]:
        uav = target()
        uid = None if uav is None else int(uav.uid)
        return {
            "time": float(env.now),
            "step": int(env.step_count),
            "target_uid": uid,
            "target": _uav_state(env, uav),
            "requested": _requested(actions, uid),
            "ring_occupancy": _occupancy(env),
            "energy_diagnostic": _energy(env, uid),
            "last_safety_reason": None if uid is None else env.last_safety_decisions.get(uid),
            "last_release_telemetry": None if uid is None else _release_row(env, uid),
        }

    original_step = env.step

    def audited_step(*args: Any, **kwargs: Any) -> Any:
        actions = args[0] if args else kwargs.get("actions")
        pre = snapshot(actions)
        error: str | None = None
        try:
            return original_step(*args, **kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            post = snapshot()
            timeline.append({
                "step": pre["step"],
                "time_before": pre["time"],
                "time_after": post["time"],
                "pre": pre,
                "post": post,
                **({"error": error} if error is not None else {}),
            })

    original_release = env._release

    def audited_release(uav: Any, intent: Any, shortlist: Any = None) -> Any:
        if int(uav.arrival_id) != TARGET_ARRIVAL_ID:
            return original_release(uav, intent, shortlist)
        uid = int(uav.uid)
        before_telemetry = dict(env.telemetry)
        before = {
            "time": float(env.now),
            "step": int(env.step_count),
            "state": _uav_state(env, uav),
            "intent": _intent(intent),
            "shortlist": _shortlist(shortlist),
            "safety_reason": env.last_safety_decisions.get(uid),
            "release_telemetry": _release_row(env, uid),
            "energy": _energy(env, uid),
        }
        result: Any = None
        error: str | None = None
        try:
            result = original_release(uav, intent, shortlist)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            after_telemetry = dict(env.telemetry)
            release_attempts.append({
                "time": before["time"],
                "step": before["step"],
                "uid": uid,
                "ring_role": before["state"].get("ring_role"),
                "returned": result,
                "error": error,
                "before": before,
                "after": {
                    "state": _uav_state(env, uav),
                    "safety_reason": env.last_safety_decisions.get(uid),
                    "release_telemetry": _release_row(env, uid),
                    "energy": _energy(env, uid),
                    "telemetry_delta": _telemetry_delta(before_telemetry, after_telemetry),
                },
            })

    original_emergency = env._try_emergency_landing

    def audited_emergency(uav: Any, *, force: bool = False) -> bool:
        if int(uav.arrival_id) != TARGET_ARRIVAL_ID:
            return original_emergency(uav, force=force)
        uid = int(uav.uid)
        before_telemetry = dict(env.telemetry)
        before = {
            "time": float(env.now),
            "step": int(env.step_count),
            "uid": uid,
            "force": bool(force),
            "state": _uav_state(env, uav),
            "safety_reason": env.last_safety_decisions.get(uid),
            "energy": _energy(env, uid),
        }
        result: bool | None = None
        error: str | None = None
        try:
            result = bool(original_emergency(uav, force=force))
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            emergency_attempts.append({
                "time": before["time"],
                "step": before["step"],
                "uid": uid,
                "force": bool(force),
                "returned": result,
                "error": error,
                "before": before,
                "after": {
                    "state": _uav_state(env, uav),
                    "safety_reason": env.last_safety_decisions.get(uid),
                    "energy": _energy(env, uid),
                    "telemetry_delta": _telemetry_delta(before_telemetry, dict(env.telemetry)),
                },
            })

    original_energy_guard = env._candidate_energy_guard_reason

    def audited_energy_guard(uav: Any, plan: Any, target_role: Any) -> Any:
        result = original_energy_guard(uav, plan, target_role)
        if int(uav.arrival_id) == TARGET_ARRIVAL_ID:
            energy_checks.append({
                "time": float(env.now),
                "step": int(env.step_count),
                "uid": int(uav.uid),
                "target": getattr(target_role, "value", target_role),
                "returned": result,
                "plan_id": getattr(plan, "plan_id", None),
                "energy": _energy(env, int(uav.uid)),
            })
        return result

    env.step = audited_step  # type: ignore[method-assign]
    env._release = audited_release  # type: ignore[method-assign]
    env._try_emergency_landing = audited_emergency  # type: ignore[method-assign]
    env._candidate_energy_guard_reason = audited_energy_guard  # type: ignore[method-assign]


def _prefix_check(tape_path: Path, config: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    prefix = {
        key: value for key, value in json.loads(tape_path.read_text(encoding="utf-8")).items()
        if key not in {"action_tape", "records"}
    }
    prefix["action_tape"] = rows[:2]
    with tempfile.TemporaryDirectory(prefix="p3_v7_prefix_") as directory:
        path = Path(directory) / "prefix.json"
        path.write_text(json.dumps(_jsonable(prefix), allow_nan=False), encoding="utf-8")
        replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), path)
    return {
        "rows": 2,
        "input_fields": sorted(rows[0]),
        "serialized_json": True,
        "exact": bool(replay.get("exact")),
        "steps_replayed": replay.get("steps_replayed"),
        "steps_expected": replay.get("steps_expected"),
    }


def _phase_transitions(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous: tuple[Any, Any] | None = None
    for row in timeline:
        state = row["post"]["target"]
        if not state.get("present"):
            current = (None, None)
        else:
            current = (state.get("phase"), state.get("ring_role"))
        if current != previous:
            result.append({
                "step": row["step"],
                "time": row["time_after"],
                "phase": current[0],
                "ring_role": current[1],
                "soc": state.get("soc"),
                "terminal_outcome": state.get("terminal_outcome"),
                "terminal_reason": state.get("terminal_reason"),
            })
            previous = current
    return result


def _threshold_crossings(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        first: dict[str, Any] | None = None
        previous_soc: float | None = None
        for row in timeline:
            for when, state in (
                (row["time_before"], row["pre"]["target"]),
                (row["time_after"], row["post"]["target"]),
            ):
                soc = state.get("soc") if state.get("present") else None
                if soc is None:
                    continue
                if first is None and float(soc) <= threshold:
                    if previous_soc is None or float(previous_soc) > threshold:
                        first = {"time": float(when), "step": int(row["step"]), "soc": float(soc)}
                previous_soc = float(soc)
        result[str(threshold)] = first
    return result


def _ring_aggregate(timeline: list[dict[str, Any]], releases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, dict[str, int]] = defaultdict(lambda: {
        "WAIT": 0, "RELEASE": 0, "approved": 0, "rejected": 0,
    })
    for row in timeline:
        state = row["pre"]["target"]
        requested = row["pre"].get("requested")
        if not state.get("present") or not requested:
            continue
        ring = str(state.get("ring_role"))
        gate = "RELEASE" if int(requested.get("gate", 0)) == 1 else "WAIT"
        result[ring][gate] += 1
    for row in releases:
        ring = str(row.get("ring_role"))
        result[ring]["approved" if row.get("returned") in {"approved", "emergency_redirected"} else "rejected"] += 1
    return {key: dict(value) for key, value in sorted(result.items())}


def _summary_outcomes(summary: Mapping[str, Any]) -> dict[str, Any]:
    value = summary.get("outcomes") or {}
    return {str(key): value for key, value in value.items()} if isinstance(value, Mapping) else {}


def _replay_outcomes(env: TerminalParallelEnv) -> dict[str, Any]:
    return {
        str(arrival_id): getattr(outcome, "value", outcome)
        for arrival_id, outcome in env.ledger.outcomes.items()
    }


def _target_energy_outcomes(env: TerminalParallelEnv) -> list[dict[str, Any]]:
    return [
        _copy_json(row)
        for row in env.energy_outcomes
        if int(row.get("arrival_id", -1)) == TARGET_ARRIVAL_ID
    ]


def run(tape_path: Path, summary_path: Path, output_path: Path) -> dict[str, Any]:
    prior_report = output_path / "diagnostic.json"
    if output_path.exists() and not prior_report.is_file():
        raise RuntimeError(f"refusing to overwrite existing output: {output_path}")
    tape_payload = json.loads(tape_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(tape_payload.get("action_tape", []))
    if len(rows) != EXPECTED_STEPS:
        raise RuntimeError(f"expected {EXPECTED_STEPS} tape rows, got {len(rows)}")
    protocol = formal_v7._protocol()
    configs, _snapshots = formal_v7._build_configs(protocol)
    config = configs["dense24"]
    target_hints = [int(uid) for uid, outcome in _summary_outcomes(summary).items() if outcome == "failed_energy"]
    if TARGET_ARRIVAL_ID not in target_hints:
        raise RuntimeError(f"summary does not bind failed-energy arrival {TARGET_ARRIVAL_ID}: {target_hints}")
    prefix = _prefix_check(tape_path, config, rows)

    timeline: list[dict[str, Any]] = []
    release_attempts: list[dict[str, Any]] = []
    emergency_attempts: list[dict[str, Any]] = []
    energy_checks: list[dict[str, Any]] = []
    env = TerminalParallelEnv(config)
    _install_observers(
        env,
        timeline=timeline,
        release_attempts=release_attempts,
        emergency_attempts=emergency_attempts,
        energy_checks=energy_checks,
    )
    replay = SafetyTraceRecorder.replay_action_tape(env, tape_path)
    replay_steps = replay.get("steps", [])
    mismatches = [
        int(row.get("step", -1))
        for row in replay_steps
        if not all(bool(row.get(key)) for key in (
            "observation_match", "requested_category_match", "post_state_match",
            "executed_mapped_action_match", "runtime_shortlist_match",
        ))
    ]
    original_outcomes = _summary_outcomes(summary)
    replay_outcomes = _replay_outcomes(env)
    outcomes_match = original_outcomes == replay_outcomes
    target = _find_by_arrival(env, TARGET_ARRIVAL_ID)
    observed_uids = [
        int(state.get("target_uid"))
        for row in timeline
        for state in (row.get("pre", {}), row.get("post", {}))
        if state.get("target_uid") is not None
    ]
    target_uid = None if not observed_uids else observed_uids[-1]
    final_state = _uav_state(env, target)
    if not final_state.get("present"):
        for row in reversed(timeline):
            candidate = row.get("pre", {}).get("target", {})
            if candidate.get("present"):
                final_state = candidate
                break
    approved = [row for row in release_attempts if row.get("returned") in {"approved", "emergency_redirected"}]
    target_terminal_outcomes = _target_energy_outcomes(env)
    report = {
        "stage": STAGE,
        "tape": str(tape_path),
        "summary": str(summary_path),
        "checkpoint_context": str(tape_path.parents[3] / "checkpoints" / "checkpoint_episode_0080.pt"),
        "checkpoint_context_sha256": _sha256(tape_path.parents[3] / "checkpoints" / "checkpoint_episode_0080.pt"),
        "target_arrival_id": TARGET_ARRIVAL_ID,
        "target_uid": target_uid,
        "target_uid_from_summary_failed_energy": target_hints,
        "input": {"tape_rows": len(rows), "schema_version": tape_payload.get("schema_version"), "prefix_check": prefix},
        "replay": {
            key: value for key, value in replay.items() if key != "steps"
        },
        "exact": bool(replay.get("exact")) and not mismatches,
        "replay_mismatch_steps": mismatches,
        "original_outcomes": original_outcomes,
        "replay_outcomes": replay_outcomes,
        "outcomes_match": outcomes_match,
        "target": {
            "final_state": final_state,
            "phase_transitions": _phase_transitions(timeline),
            "soc_threshold_crossings": _threshold_crossings(timeline),
            "last_approved_progress": approved[-1] if approved else None,
            "terminal_context": {
                "time": timeline[-1]["time_after"] if timeline else None,
                "step": timeline[-1]["step"] if timeline else None,
                "state": final_state,
                "energy_outcomes": target_terminal_outcomes,
            },
            "ring_aggregate": _ring_aggregate(timeline, release_attempts),
            "release_attempts": release_attempts,
            "emergency_attempts": emergency_attempts,
            "energy_guard_checks": energy_checks,
            "timeline": timeline,
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / ("diagnostic_v2.json" if prior_report.is_file() else "diagnostic.json")
    report["report_path"] = str(report_path)
    report["prior_report_preserved"] = bool(prior_report.is_file())
    report_path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.tape, args.summary, args.output)
    except Exception as exc:
        print(json.dumps({"stage": STAGE, "exact": False, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2
    print(json.dumps({
        "stage": STAGE,
        "exact": report["exact"],
        "steps_replayed": report["replay"].get("steps_replayed"),
        "steps_expected": report["replay"].get("steps_expected"),
        "target_arrival_id": report["target_arrival_id"],
        "target_uid": report["target_uid"],
        "outcomes_match": report["outcomes_match"],
    }, sort_keys=True))
    return 0 if report["exact"] and report["outcomes_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
