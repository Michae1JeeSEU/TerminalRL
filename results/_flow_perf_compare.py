"""Compare scalar and vectorized dynamics-plan validation on one fixed run.

This is an offline performance/equivalence probe.  It runs the same
coordinated seed twice for the first 100 simulator steps.  The scalar case
replaces only ``VehicleDynamics.validate_plan`` with the pre-fast-path
transition loop; the vectorized case uses the current production method.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.diagnose_landing_flow import (
    _build_diagnostic_config,
    _jsonable,
    _terminal,
)


SEED = 841301
MAX_STEPS = 100
CELL_TIMEOUT_SECONDS = 180.0
OUTPUT = Path("results/flow_perf_equivalent_100.json")


def _scalar_validate_plan(self: VehicleDynamics, points) -> list[str]:
    """The pre-vectorized validate_plan oracle, preserving scalar order."""

    rows = list(points)
    if len(rows) < 2:
        return ["invalid_time_parameterization"]
    errors: list[str] = []
    for previous, current in zip(rows, rows[1:]):
        errors.extend(self.validate_transition(previous, current))
    return list(dict.fromkeys(errors))


def _float_list(values: Any) -> list[float]:
    return [round(float(value), 9) for value in np.asarray(values, dtype=float).tolist()]


def _state_trace(env: TerminalParallelEnv) -> dict[str, Any]:
    rows = []
    for uid, uav in sorted(env.active.items()):
        rows.append(
            {
                "uid": int(uid),
                "phase": uav.phase.value,
                "position": _float_list(uav.position),
                "velocity": _float_list(uav.velocity),
                "soc": round(float(uav.soc), 12),
            }
        )
    return {"time": float(env.now), "uavs": rows}


def _safety_trace(env: TerminalParallelEnv) -> dict[str, Any]:
    keys = (
        "safety_interventions",
        "hard_separation_violations",
        "nmac",
        "collisions",
        "operational_buffer_entries",
        "shield_fail_closed",
        "joint_safety_fail_closed",
        "checked_executed_plan_hash_mismatches",
    )
    return {key: int(env.telemetry.get(key, 0)) for key in keys}


def _plan_hash_from_row(row: Mapping[str, Any], env: TerminalParallelEnv, uid: int) -> str | None:
    for key in ("trajectory_hash", "checked_trajectory_hash"):
        value = row.get(key)
        if value:
            return str(value)
    uav = env.active.get(uid) or env.service_uavs.get(uid)
    plan = None if uav is None else uav.committed_plan
    return None if plan is None else str(plan.trajectory_hash)


def _commit_records(
    env: TerminalParallelEnv,
    info: Mapping[str, Any],
    before_event_count: int,
    seen_commit_keys: set[tuple[int, str, str | None]],
    *,
    step: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    release_rows = info.get("release_intent_telemetry", {})
    release_decisions = info.get("release_decisions", {})
    for raw_uid, raw_row in release_rows.items():
        if not isinstance(raw_row, Mapping):
            continue
        uid = int(raw_uid)
        decision = release_decisions.get(str(raw_uid), release_decisions.get(uid))
        if decision != "approved":
            continue
        plan_hash = _plan_hash_from_row(raw_row, env, uid)
        if plan_hash is None:
            continue
        marker = (uid, "normal", plan_hash)
        if marker in seen_commit_keys:
            continue
        seen_commit_keys.add(marker)
        records.append(
            {
                "step": int(step),
                "time": float(raw_row.get("approval_time", env.now)),
                "uid": uid,
                "mode": "normal",
                "geometry_hash": plan_hash,
            }
        )
    for event in env.events[before_event_count:]:
        if not isinstance(event, (tuple, list)) or len(event) < 3:
            continue
        if str(event[0]) != "emergency_departure":
            continue
        uid = int(event[1])
        uav = env.active.get(uid) or env.service_uavs.get(uid)
        plan = None if uav is None else uav.committed_plan
        plan_hash = None if plan is None else str(plan.trajectory_hash)
        marker = (uid, "emergency", plan_hash)
        if marker in seen_commit_keys:
            continue
        seen_commit_keys.add(marker)
        records.append(
            {
                "step": int(step),
                "time": float(event[2]),
                "uid": uid,
                "mode": "emergency",
                "geometry_hash": plan_hash,
            }
        )
    return records


def _outcome_counts(env: TerminalParallelEnv) -> dict[str, int]:
    return dict(
        sorted(Counter(outcome.value for outcome in env.ledger.outcomes.values()).items())
    )


def _safety_summary(env: TerminalParallelEnv) -> dict[str, Any]:
    keys = (
        "safety_interventions",
        "hard_separation_violations",
        "nmac",
        "collisions",
        "operational_buffer_entries",
        "minimum_actual_separation",
        "minimum_swept_separation",
        "shield_fail_closed",
        "joint_safety_fail_closed",
        "checked_executed_plan_hash_mismatches",
    )
    result: dict[str, Any] = {}
    for key in keys:
        value = env.telemetry.get(key)
        if isinstance(value, (int, np.integer)):
            result[key] = int(value)
        elif value is None:
            result[key] = None
        else:
            result[key] = float(value)
    return result


def _trace_digest(trace: list[Mapping[str, Any]]) -> str:
    payload = json.dumps(trace, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_case(config, *, label: str, scalar: bool) -> dict[str, Any]:
    started = time.perf_counter()
    original_validate_plan = VehicleDynamics.validate_plan
    if scalar:
        VehicleDynamics.validate_plan = _scalar_validate_plan
    try:
        env = TerminalParallelEnv(config)
        env.reset(seed=SEED)
        states: list[dict[str, Any]] = []
        safety: list[dict[str, Any]] = []
        commits: list[dict[str, Any]] = []
        seen_commit_keys: set[tuple[int, str, str | None]] = set()
        steps = 0
        stop_reason: str | None = None
        while steps < MAX_STEPS and not _terminal(env):
            if time.perf_counter() - started > CELL_TIMEOUT_SECONDS:
                raise TimeoutError(f"{label} exceeded {CELL_TIMEOUT_SECONDS}s")
            before_event_count = len(env.events)
            actions = {
                str(uid): np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                for uid in env.active
            }
            _observations, _rewards, _terminations, _truncations, info = env.step(actions)
            steps += 1
            states.append(_state_trace(env))
            safety.append(_safety_trace(env))
            commits.extend(
                _commit_records(
                    env,
                    info,
                    before_event_count,
                    seen_commit_keys,
                    step=steps,
                )
            )
            if steps % 50 == 0:
                print(
                    json.dumps(
                        {
                            "label": label,
                            "step": steps,
                            "sim_time": float(env.now),
                            "wall_seconds": time.perf_counter() - started,
                            "active": len(env.active),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if _terminal(env):
            stop_reason = "terminal"
        elif steps >= MAX_STEPS:
            stop_reason = "max_steps"
        else:
            stop_reason = "stopped"
        elapsed = time.perf_counter() - started
        return {
            "label": label,
            "scalar_oracle": bool(scalar),
            "wall_seconds": float(elapsed),
            "steps": int(steps),
            "sim_time_seconds": float(env.now),
            "stop_reason": stop_reason,
            "states": states,
            "safety_trace": safety,
            "commits": commits,
            "terminal": bool(_terminal(env)),
            "outcome_counts": _outcome_counts(env),
            "safety_summary": _safety_summary(env),
            "active_uids_final": sorted(int(uid) for uid in env.active),
            "trace_sha256": _trace_digest(states),
            "safety_trace_sha256": _trace_digest(safety),
        }
    finally:
        VehicleDynamics.validate_plan = original_validate_plan


def _compare_traces(scalar: Mapping[str, Any], vectorized: Mapping[str, Any]) -> dict[str, Any]:
    scalar_states = scalar["states"]
    vector_states = vectorized["states"]
    scalar_safety = scalar["safety_trace"]
    vector_safety = vectorized["safety_trace"]
    mismatches: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    for step, (left, right) in enumerate(zip(scalar_states, vector_states), start=1):
        if left.get("time") != right.get("time") or len(left["uavs"]) != len(right["uavs"]):
            mismatches.append({"step": step, "kind": "state_shape_or_time", "scalar": left, "vectorized": right})
            break
        for left_row, right_row in zip(left["uavs"], right["uavs"]):
            if left_row["uid"] != right_row["uid"] or left_row["phase"] != right_row["phase"]:
                mismatches.append({"step": step, "kind": "uid_or_phase", "scalar": left_row, "vectorized": right_row})
                break
            for field in ("position", "velocity", "soc"):
                delta = np.max(
                    np.abs(
                        np.asarray(left_row[field], dtype=float)
                        - np.asarray(right_row[field], dtype=float)
                    )
                )
                max_abs_delta = max(max_abs_delta, float(delta))
                if delta > 1e-12:
                    mismatches.append({"step": step, "kind": field, "scalar": left_row, "vectorized": right_row})
                    break
            if mismatches:
                break
        if mismatches:
            break
        if left_safety := (scalar_safety[step - 1] != vector_safety[step - 1]):
            mismatches.append(
                {
                    "step": step,
                    "kind": "safety",
                    "scalar": scalar_safety[step - 1],
                    "vectorized": vector_safety[step - 1],
                }
            )
            break
    if len(scalar_states) != len(vector_states):
        mismatches.append(
            {
                "step": min(len(scalar_states), len(vector_states)) + 1,
                "kind": "trace_length",
                "scalar": len(scalar_states),
                "vectorized": len(vector_states),
            }
        )
    scalar_commits = [
        {key: record.get(key) for key in ("time", "uid", "mode", "geometry_hash")}
        for record in scalar["commits"]
    ]
    vector_commits = [
        {key: record.get(key) for key in ("time", "uid", "mode", "geometry_hash")}
        for record in vectorized["commits"]
    ]
    if scalar_commits != vector_commits:
        mismatches.append(
            {"step": None, "kind": "commits", "scalar": scalar_commits, "vectorized": vector_commits}
        )
    for key in ("terminal", "outcome_counts", "safety_summary", "stop_reason"):
        if scalar[key] != vectorized[key]:
            mismatches.append(
                {"step": None, "kind": key, "scalar": scalar[key], "vectorized": vectorized[key]}
            )
    return {
        "equivalent": not mismatches,
        "max_abs_state_delta": float(max_abs_delta),
        "scalar_trace_sha256": scalar["trace_sha256"],
        "vectorized_trace_sha256": vectorized["trace_sha256"],
        "scalar_safety_trace_sha256": scalar["safety_trace_sha256"],
        "vectorized_safety_trace_sha256": vectorized["safety_trace_sha256"],
        "mismatches": mismatches[:1],
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    config = _build_diagnostic_config(True)
    config_json = json.dumps(_jsonable(asdict(config)), sort_keys=True, separators=(",", ":"))
    scalar = _run_case(config, label="scalar_validate_plan", scalar=True)
    vectorized = _run_case(config, label="vectorized_validate_plan", scalar=False)
    comparison = _compare_traces(scalar, vectorized)
    reference_trace = scalar.pop("states")
    reference_safety_trace = scalar.pop("safety_trace")
    vectorized.pop("states")
    vectorized.pop("safety_trace")
    result = {
        "diagnostic_only": True,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "coordinated": True,
        "comparison_scope": "validate_plan_fastpath_only",
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "config_identical_across_cells": True,
        "cells": {
            "scalar_validate_plan": scalar,
            "vectorized_validate_plan": vectorized,
        },
        "per_step_state_trace": reference_trace,
        "per_step_safety_trace": reference_safety_trace,
        "per_step_trace_reference": "scalar_validate_plan (identical to vectorized; hashes in comparison)",
        "comparison": comparison,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            _jsonable(result),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "equivalent": comparison["equivalent"],
                "scalar_wall_seconds": scalar["wall_seconds"],
                "vectorized_wall_seconds": vectorized["wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not comparison["equivalent"]:
        raise RuntimeError("semantic difference detected; see comparison.mismatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
