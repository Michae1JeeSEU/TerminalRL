"""Small read-only trace of actual final/emergency vertical execution.

The simulator itself is unchanged.  The instance method ``_advance_plan`` is
wrapped only for this process, and only vertical samples from landing plans
are retained.
"""

from __future__ import annotations

import argparse
from collections import Counter
from enum import Enum
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.diagnose_energy_reachability_guard_smoke import build_config


LANDING_MODES = {"final_approach", "emergency_landing"}
LANDING_TARGETS = {"landing", "emergency_landing"}
XY_TOLERANCE = 1.0e-6


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def terminal_now(env: TerminalParallelEnv) -> bool:
    return bool(
        env.arrivals.exhausted()
        and not env.active
        and not env.admission.queue.items
        and not env.service_uavs
    )


def vertical_geometry(plan: Any) -> dict[str, Any]:
    endpoint = np.asarray(plan.points[-1].position, dtype=float)
    start_index = len(plan.points) - 1
    while start_index > 0:
        previous = np.asarray(plan.points[start_index - 1].position, dtype=float)
        if float(np.linalg.norm(previous[:2] - endpoint[:2])) > XY_TOLERANCE:
            break
        start_index -= 1
    vertical_start = plan.points[start_index]
    metadata_staging = plan.metadata.get("staging_point")
    staging = (np.asarray(metadata_staging, dtype=float)
               if metadata_staging is not None else np.asarray(vertical_start.position, dtype=float))
    return {
        "vertical_start_time": float(vertical_start.t),
        "vertical_end_time": float(plan.end_time),
        "vertical_duration": float(plan.end_time - vertical_start.t),
        "staging_point": staging.tolist(),
        "pad_position": endpoint.tolist(),
        "planned_vertical_point_count": int(len(plan.points) - start_index),
    }


def offline_full_summary() -> dict[str, Any]:
    path = Path("results/viewer_trace_841301_full.json")
    if not path.exists():
        return {"status": "unavailable", "reason": str(path) + " missing"}
    data = json.loads(path.read_text(encoding="utf-8"))

    def sample(uid: int, moment: int) -> dict[str, Any] | None:
        values = data.get("tracked_samples", {}).get(str(uid), ())
        return next((row for row in values if abs(float(row["time"]) - moment) <= 1e-9), None)

    uid12 = {
        str(moment): ({
            "soc": float(sample(12, moment)["soc"]),
            "energy_remaining": float(sample(12, moment)["energy_remaining"]),
            "phase": sample(12, moment)["phase"],
            "plan_id": sample(12, moment)["trajectory_plan_id"],
            "last_release_decision": sample(12, moment)["last_release_decision"],
            "last_safety_decision": sample(12, moment)["last_safety_decision"],
            "blocker": ((sample(12, moment).get("last_release_telemetry") or {}).get("primary_blocker")),
        } if sample(12, moment) is not None else None)
        for moment in (991, 993)
    }

    mids: dict[str, list[float] | None] = {}
    for uid in (4, 11, 12):
        left = sample(uid, 773)
        right = sample(uid, 813)
        if left is None or right is None:
            mids[str(uid)] = None
        else:
            mids[str(uid)] = [
                (float(left["position"][axis]) + float(right["position"][axis])) / 2.0
                for axis in range(3)
            ]
    pair_distances: dict[str, float] = {}
    for first, second in ((4, 11), (4, 12), (11, 12)):
        left = np.asarray(mids[str(first)], dtype=float)
        right = np.asarray(mids[str(second)], dtype=float)
        pair_distances[f"{first}-{second}"] = float(np.linalg.norm(left - right))
    midpoint_radii = {
        uid: (None if position is None else float(np.linalg.norm(np.asarray(position[:2], dtype=float))))
        for uid, position in mids.items()
    }
    return {
        "status": "available",
        "uid12_at_991_993": uid12,
        "tracked_midpoints_from_773_to_813": mids,
        "midpoint_radii": midpoint_radii,
        "midpoint_pair_distances": pair_distances,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=841301)
    args = parser.parse_args()

    config = build_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=True,
        max_simulation_seconds=1800.0,
        dense_warm_start_uavs=30,
        final_approach_geometry_mode="continuous_inward_v1",
        release_arbitration_mode="downstream_priority",
        reconcile_late_merge_slots_enabled=False,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=int(args.seed))
    records: dict[str, dict[str, Any]] = {}
    calls = 0
    touchdown_count = 0
    original_advance_plan = env._advance_plan

    def traced_advance_plan(uav: Any) -> tuple[float, bool]:
        nonlocal calls, touchdown_count
        plan = uav.committed_plan
        if plan is None or plan.mode not in LANDING_MODES or plan.metadata.get("target_phase") not in LANDING_TARGETS:
            return original_advance_plan(uav)

        plan_id = str(plan.plan_id)
        if plan_id not in records:
            records[plan_id] = {
                "plan_id": plan_id,
                "uid": int(uav.uid),
                "mode": plan.mode,
                "target_phase": plan.metadata.get("target_phase"),
                "plan_start_time": float(plan.start_time),
                "plan_end_time": float(plan.end_time),
                "plan_duration": float(plan.end_time - plan.start_time),
                **vertical_geometry(plan),
                "vertical_samples": [],
                "touchdown": None,
            }
        row = records[plan_id]
        before_time = float(env.now)
        distance, touchdown = original_advance_plan(uav)
        calls += 1
        after_position = np.asarray(uav.position, dtype=float).copy()
        after_velocity = np.asarray(uav.velocity, dtype=float).copy()
        pad = np.asarray(row["pad_position"], dtype=float)
        staging = np.asarray(row["staging_point"], dtype=float)
        is_vertical = (
            float(row["vertical_start_time"]) <= before_time + config.dt + 1e-9
            and float(np.linalg.norm(after_position[:2] - pad[:2])) <= XY_TOLERANCE
            and -1e-9 <= float(after_position[2]) <= float(staging[2]) + 1e-9
        )
        if is_vertical:
            row["vertical_samples"].append({
                "time_before": before_time,
                "time_after": float(before_time + config.dt),
                "position": after_position.tolist(),
                "velocity": after_velocity.tolist(),
                "distance_this_call": float(distance),
                "touchdown_return": bool(touchdown),
            })
        if touchdown:
            touchdown_count += 1
            row["touchdown"] = {
                "return_true": True,
                "time": float(before_time + config.dt),
                "position": after_position.tolist(),
                "velocity": after_velocity.tolist(),
            }
        return distance, touchdown

    env._advance_plan = traced_advance_plan
    started = time.perf_counter()
    stop_reason = "max_steps"
    for step_index in range(int(args.max_steps)):
        env.step()
        if (step_index + 1) % 200 == 0:
            print(json.dumps({"step": step_index + 1, "sim_time": env.now,
                              "active": len(env.active), "landing_plans": len(records)}, sort_keys=True), flush=True)
        if terminal_now(env):
            stop_reason = "terminal"
            break

    summary = []
    for row in sorted(records.values(), key=lambda item: (item["uid"], item["plan_start_time"])):
        samples = row["vertical_samples"]
        heights = [float(item["position"][2]) for item in samples]
        pad_xy = np.asarray(row["pad_position"][:2], dtype=float)
        xy_error = [float(np.linalg.norm(np.asarray(item["position"][:2], dtype=float) - pad_xy)) for item in samples]
        monotonic = all(right <= left + 1e-8 for left, right in zip(heights, heights[1:]))
        row["actual_vertical_sample_count"] = len(samples)
        row["actual_first_sample"] = None if not samples else samples[0]
        row["actual_last_sample"] = None if not samples else samples[-1]
        row["actual_height_first"] = None if not heights else heights[0]
        row["actual_height_last"] = None if not heights else heights[-1]
        row["actual_height_monotonic_nonincreasing"] = bool(monotonic)
        row["actual_pad_xy_max_error"] = None if not xy_error else max(xy_error)
        row["touchdown_z_zero"] = bool(row["touchdown"] is not None and abs(float(row["touchdown"]["position"][2])) <= 1e-8)
        summary.append({
            key: row[key] for key in (
                "plan_id", "uid", "mode", "target_phase", "plan_start_time", "plan_end_time",
                "plan_duration", "vertical_start_time", "vertical_end_time", "vertical_duration",
                "staging_point", "pad_position", "planned_vertical_point_count",
                "actual_vertical_sample_count", "actual_height_first", "actual_height_last",
                "actual_height_monotonic_nonincreasing", "actual_pad_xy_max_error",
                "touchdown", "touchdown_z_zero", "vertical_samples",
            )
        })

    outcome_counts = Counter(str(value.value if isinstance(value, Enum) else value)
                             for value in env.ledger.outcomes.values())
    successful = [row for row in summary if row["touchdown"] is not None and row["touchdown"].get("return_true")]
    payload = {
        "schema_version": "viewer-vertical-execution-841301-v1",
        "seed": int(args.seed),
        "config": {
            "initial_soc_min": 0.55, "initial_soc_max": 0.90,
            "route_guard_enabled": True, "max_simulation_seconds": 1800.0,
            "dense_warm_start_uavs": 30,
            "final_approach_geometry_mode": "continuous_inward_v1",
            "release_arbitration_mode": "downstream_priority",
            "reconcile_late_merge_slots_enabled": False,
        },
        "run": {
            "steps": int(env.step_count), "sim_time": float(env.now),
            "wall_seconds": float(time.perf_counter() - started),
            "stop_reason": stop_reason, "advance_plan_landing_calls": int(calls),
            "touchdown_return_true_count": int(touchdown_count),
        },
        "outcome_counts": dict(outcome_counts),
        "landing_plan_count": len(summary),
        "successful_landing_plan_count": len(successful),
        "landing_plans": summary,
        "offline_full_trace_summary": offline_full_summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "steps": env.step_count, "sim_time": env.now,
        "stop_reason": stop_reason, "outcome_counts": dict(outcome_counts),
        "landing_plans": len(summary), "successful_landing_plans": len(successful),
        "touchdown_return_true": touchdown_count,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
