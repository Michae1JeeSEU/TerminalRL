"""Read-only deterministic trace collector for the dense 30-UAV viewer scene.

This file is a temporary diagnostic helper.  It does not attach a viewer or
alter simulator code.  The output is deliberately JSON so a later renderer or
analysis script can inspect the exact state around a selected simulation time.
"""

from __future__ import annotations

import argparse
from collections import Counter
from enum import Enum
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terminal_marl.core.config import RingRole
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.diagnose_energy_reachability_guard_smoke import build_config


TRACKED_UIDS = (4, 11, 12)
WINDOW_START = 780.0
WINDOW_END = 850.0


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


def plan_payload(plan: Any, observed_at: float) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "uav_id": int(plan.uav_id),
        "mode": plan.mode,
        "start_time": float(plan.start_time),
        "end_time": float(plan.end_time),
        "reservation_ids": list(plan.reservation_ids),
        "metadata": jsonable(plan.metadata),
        "trajectory_hash": plan.trajectory_hash,
        "observed_at": float(observed_at),
        "points": [
            {
                "t": float(point.t),
                "position": jsonable(point.position),
                "velocity": jsonable(point.velocity),
                "acceleration": jsonable(point.acceleration),
                "yaw": None if point.yaw is None else float(point.yaw),
                "yaw_rate": None if point.yaw_rate is None else float(point.yaw_rate),
            }
            for point in plan.points
        ],
    }


def state_payload(env: TerminalParallelEnv, uav: Any) -> dict[str, Any]:
    plan = uav.committed_plan
    return {
        "uid": int(uav.uid),
        "arrival_id": int(uav.arrival_id),
        "phase": uav.phase.value,
        "ring_role": None if uav.ring_role is None else uav.ring_role.value,
        "position": jsonable(uav.position),
        "velocity": jsonable(uav.velocity),
        "acceleration": jsonable(uav.acceleration),
        "speed": float(uav.speed),
        "heading": float(uav.heading),
        "ring_progress": None if uav.ring_progress is None else float(uav.ring_progress),
        "soc": float(uav.soc),
        "energy_remaining": float(uav.energy_remaining),
        "energy_capacity": float(uav.energy_capacity),
        "energy_consumed": float(uav.energy_consumed),
        "assigned_lz": uav.assigned_lz,
        "emergency_landing": bool(uav.emergency_landing),
        "terminal_outcome": None if uav.terminal_outcome is None else uav.terminal_outcome.value,
        "terminal_reason": uav.terminal_reason,
        "phase_entry_time": float(uav.phase_entry_time),
        "assigned_gate": int(uav.assigned_gate),
        "trajectory_plan_id": None if plan is None else plan.plan_id,
        "trajectory_plan_mode": None if plan is None else plan.mode,
        "trajectory_plan_start_time": None if plan is None else float(plan.start_time),
        "trajectory_plan_end_time": None if plan is None else float(plan.end_time),
        "trajectory_plan_metadata": None if plan is None else jsonable(plan.metadata),
        "trajectory_hash": None if plan is None else plan.trajectory_hash,
        "merge_reservation_id": uav.merge_reservation_id,
        "committed": plan is not None,
        "last_release_decision": env.last_release_decisions.get(uav.uid),
        "last_safety_decision": env.last_safety_decisions.get(uav.uid),
        "last_release_telemetry": jsonable(env.last_release_telemetry.get(uav.uid)),
        "emergency_retry_after": (None if not env.config.emergency_priority_guard_enabled
                                   else float(uav.emergency_retry_after)),
    }


def in_window(sim_time: float) -> bool:
    return WINDOW_START <= sim_time <= WINDOW_END


def terminal_now(env: TerminalParallelEnv) -> bool:
    return bool(
        env.arrivals.exhausted()
        and not env.active
        and not env.admission.queue.items
        and not env.service_uavs
    )


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

    tracked_samples: dict[str, list[dict[str, Any]]] = {str(uid): [] for uid in TRACKED_UIDS}
    window_samples: list[dict[str, Any]] = []
    plans: dict[str, dict[str, Any]] = {}
    disappearances: list[dict[str, Any]] = []
    control_events: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    probe_813: dict[str, Any] | None = None
    previous_active: dict[int, dict[str, Any]] = {}
    previous_plan_ids: dict[int, str | None] = {}
    previous_energy_count = 0
    previous_event_count = 0
    wall_start = __import__("time").perf_counter()
    stopped_reason = "max_steps"

    for _ in range(int(args.max_steps)):
        before_time = float(env.now)
        before_active = {int(uid): state_payload(env, uav) for uid, uav in env.active.items()}
        before_objects = {int(uid): uav for uid, uav in env.active.items()}

        env.step()
        after_time = float(env.now)
        active_now = {int(uid): state_payload(env, uav) for uid, uav in env.active.items()}

        # Capture full tracked histories and all aircraft in the requested
        # diagnostic window.  Every live committed plan is copied once.
        for uid in TRACKED_UIDS:
            state = active_now.get(uid)
            if state is not None:
                tracked_samples[str(uid)].append({"time": after_time, **state})
        if in_window(after_time):
            window_samples.append({
                "time": after_time,
                "active": [active_now[uid] for uid in sorted(active_now)],
                "service_uids": sorted(int(uid) for uid in env.service_uavs),
                "completed_uids": sorted(int(uav.uid) for uav in env.completed_uavs),
                "landing_zones": [
                    {
                        "lz_id": int(zone.lz_id),
                        "status": zone.status.value,
                        "reserved_by": zone.reserved_by,
                        "occupied_by": zone.occupied_by,
                        "reservation_start": zone.reservation_start,
                        "service_remaining": float(zone.service_remaining),
                    }
                    for zone in env.lz_manager.zones
                ],
            })

        if abs(after_time - 813.0) <= 1e-9:
            probe_813 = {
                "time": after_time,
                "available_zone_ids": {},
                "candidates": {},
            }
            for uid in TRACKED_UIDS:
                uav = env.active.get(uid)
                if uav is None:
                    probe_813["available_zone_ids"][str(uid)] = []
                    probe_813["candidates"][str(uid)] = {"status": "uid_not_active"}
                    continue
                zone_ids = [int(item) for item in env.lz_manager.rank_available_zones(uav, env.now)]
                probe_813["available_zone_ids"][str(uid)] = zone_ids
                rows = []
                for lz_id in zone_ids:
                    zone = env.lz_manager.zones[lz_id]
                    try:
                        candidate = env.generator.emergency_landing(uav, zone, env.airspace, env.now)
                        required = env.energy.emergency_required_energy(uav, candidate)
                        rows.append({
                            "lz_id": lz_id,
                            "status": "generated",
                            "duration": float(candidate.end_time - candidate.start_time),
                            "start_time": float(candidate.start_time),
                            "end_time": float(candidate.end_time),
                            "energy_required": float(required),
                            "energy_remaining": float(uav.energy_remaining),
                            "soc": float(uav.soc),
                            "plan_mode": candidate.mode,
                            "metadata": jsonable(candidate.metadata),
                            "points": [
                                {"t": float(point.t), "position": jsonable(point.position),
                                 "velocity": jsonable(point.velocity)}
                                for point in candidate.points
                            ],
                        })
                    except Exception as exc:  # evidence only; preserve exact exception text
                        rows.append({"lz_id": lz_id, "status": "exception", "exception": repr(exc)})
                probe_813["candidates"][str(uid)] = {
                    "phase": uav.phase.value,
                    "ring_role": None if uav.ring_role is None else uav.ring_role.value,
                    "committed_plan_id": None if uav.committed_plan is None else uav.committed_plan.plan_id,
                    "energy_remaining": float(uav.energy_remaining),
                    "soc": float(uav.soc),
                    "rows": rows,
                }

        for uid, uav in env.active.items():
            plan = uav.committed_plan
            if plan is not None and plan.plan_id not in plans:
                plans[plan.plan_id] = plan_payload(plan, after_time)
        for uid, old in before_objects.items():
            plan = old.committed_plan
            if plan is not None and plan.plan_id not in plans:
                plans[plan.plan_id] = plan_payload(plan, before_time)

        new_energy = env.energy_outcomes[previous_energy_count:]
        new_events = env.events[previous_event_count:]
        post_terminal = {
            "ledger_outcomes": {str(k): jsonable(v) for k, v in env.ledger.outcomes.items()},
            "service_uids": sorted(int(uid) for uid in env.service_uavs),
            "completed_uids": sorted(int(uav.uid) for uav in env.completed_uavs),
            "energy_outcomes": jsonable(new_energy),
            "events": jsonable(new_events),
        }
        disappeared = sorted(set(before_active) - set(active_now))
        for uid in disappeared:
            old = before_active[uid]
            matching_energy = [row for row in new_energy if row.get("uid") == uid]
            matching_events = [row for row in new_events if len(row) > 1 and row[1] == uid]
            disappearances.append({
                "time_before": before_time,
                "time_after": after_time,
                "uid": uid,
                "last_active_state": old,
                "post_state": post_terminal,
                "matching_energy_outcomes": jsonable(matching_energy),
                "matching_events": jsonable(matching_events),
                "ledger_outcome_for_arrival": jsonable(env.ledger.outcomes.get(old["arrival_id"])),
            })

        changed_plans = [
            {"uid": uid, "from": previous_plan_ids.get(uid), "to": state.get("trajectory_plan_id")}
            for uid, state in sorted(active_now.items())
            if previous_plan_ids.get(uid) != state.get("trajectory_plan_id")
        ]
        changed_plans.extend(
            {"uid": uid, "from": previous_plan_ids.get(uid), "to": None}
            for uid in disappeared
            if previous_plan_ids.get(uid) is not None
        )
        decision_uids = set(env.last_safety_decisions) | set(env.last_release_decisions)
        if new_events or new_energy or changed_plans or decision_uids:
            control_events.append({
                "time": after_time,
                "events": jsonable(new_events),
                "energy_outcomes": jsonable(new_energy),
                "plan_changes": jsonable(changed_plans),
                "safety_decisions": jsonable(env.last_safety_decisions),
                "release_decisions": jsonable(env.last_release_decisions),
                "release_telemetry": jsonable(env.last_release_telemetry),
                "action_traces": jsonable({str(uid): trace.as_dict() for uid, trace in env.last_action_traces.items()}),
                "conflict_events": jsonable(env.conflict_events),
            })

        if in_window(after_time) or disappeared or any(uid in TRACKED_UIDS for uid in changed_plans):
            step_records.append({
                "time_before": before_time,
                "time_after": after_time,
                "active_uids": sorted(active_now),
                "queue_size": len(env.admission.queue.items),
                "service_uids": sorted(int(uid) for uid in env.service_uavs),
                "completed_uids": sorted(int(uav.uid) for uav in env.completed_uavs),
                "disappeared_uids": disappeared,
                "new_events": jsonable(new_events),
                "new_energy_outcomes": jsonable(new_energy),
                "ledger_outcomes": {str(k): jsonable(v) for k, v in env.ledger.outcomes.items()},
                "telemetry": jsonable(env.telemetry),
            })

        previous_active = active_now
        previous_plan_ids = {uid: state.get("trajectory_plan_id") for uid, state in active_now.items()}
        previous_energy_count = len(env.energy_outcomes)
        previous_event_count = len(env.events)
        if terminal_now(env):
            stopped_reason = "terminal"
            break

    elapsed = __import__("time").perf_counter() - wall_start
    outcome_counts = Counter(str(value.value if isinstance(value, Enum) else value) for value in env.ledger.outcomes.values())
    payload = {
        "schema_version": "viewer-trace-841301-v1",
        "seed": int(args.seed),
        "config": {
            "initial_soc_min": 0.55,
            "initial_soc_max": 0.90,
            "route_guard_enabled": True,
            "max_simulation_seconds": 1800.0,
            "dense_warm_start_uavs": 30,
            "final_approach_geometry_mode": "continuous_inward_v1",
            "release_arbitration_mode": "downstream_priority",
            "reconcile_late_merge_slots_enabled": False,
        },
        "run": {
            "steps": int(env.step_count),
            "sim_time": float(env.now),
            "wall_seconds": float(elapsed),
            "stop_reason": stopped_reason,
            "active_uids": sorted(int(uid) for uid in env.active),
            "queue_size": len(env.admission.queue.items),
            "service_uids": sorted(int(uid) for uid in env.service_uavs),
            "completed_uids": sorted(int(uav.uid) for uav in env.completed_uavs),
        },
        "outcome_counts": dict(outcome_counts),
        "ledger_outcomes": {str(k): jsonable(v) for k, v in env.ledger.outcomes.items()},
        "telemetry": jsonable(env.telemetry),
        "energy_outcomes": jsonable(env.energy_outcomes),
        "tracked_uids": list(TRACKED_UIDS),
        "tracked_samples": tracked_samples,
        "probe_813": probe_813,
        "window": {"start": WINDOW_START, "end": WINDOW_END, "samples": window_samples},
        "plans": plans,
        "disappearances": disappearances,
        "control_events": control_events,
        "step_records": step_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "steps": env.step_count,
        "sim_time": env.now,
        "wall_seconds": elapsed,
        "stop_reason": stopped_reason,
        "outcome_counts": dict(outcome_counts),
        "plans": len(plans),
        "disappearances": len(disappearances),
        "window_samples": len(window_samples),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
