"""Bounded, headless baseline trace for the dense seed-841301 flow.

The simulator is used as-is.  This helper records the actual per-step return
information from ``env.step()`` and the event/outcome deltas, rather than
reusing a stale last-state dictionary.  It is intentionally a temporary
diagnostic script and does not attach a viewer or patch production code.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
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


CHECKPOINTS = {551.0, 552.0, 553.0}
TRACKED_UIDS = (3, 4)
LOW_SOC_THRESHOLD = 0.40


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


def state_payload(uav: Any) -> dict[str, Any]:
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
        "ring_progress": None if uav.ring_progress is None else float(uav.ring_progress),
        "soc": float(uav.soc),
        "energy_remaining": float(uav.energy_remaining),
        "energy_consumed": float(uav.energy_consumed),
        "energy_capacity": float(uav.energy_capacity),
        "assigned_lz": uav.assigned_lz,
        "emergency_landing": bool(uav.emergency_landing),
        "phase_entry_time": float(uav.phase_entry_time),
        "assigned_gate": int(uav.assigned_gate),
        "trajectory_plan_id": None if plan is None else plan.plan_id,
        "trajectory_plan_mode": None if plan is None else plan.mode,
        "trajectory_plan_start_time": None if plan is None else float(plan.start_time),
        "trajectory_plan_end_time": None if plan is None else float(plan.end_time),
        "trajectory_plan_target_phase": (
            None if plan is None else plan.metadata.get("target_phase")
        ),
        "merge_reservation_id": uav.merge_reservation_id,
        "committed": plan is not None,
    }


def zone_payload(env: TerminalParallelEnv) -> list[dict[str, Any]]:
    return [
        {
            "lz_id": int(zone.lz_id),
            "status": zone.status.value,
            "reserved_by": zone.reserved_by,
            "occupied_by": zone.occupied_by,
            "reservation_start": zone.reservation_start,
            "service_remaining": float(zone.service_remaining),
        }
        for zone in env.lz_manager.zones
    ]


def telemetry_payload(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    final_pairs = row.get("final_lz_pairs_checked")
    return {
        "arrival_id": row.get("arrival_id"),
        "uav_id": row.get("uav_id"),
        "interface": row.get("interface"),
        "attempt_count": row.get("attempt_count"),
        "retry_count": row.get("retry_count"),
        "first_attempt_time": row.get("first_attempt_time"),
        "primary_blocker": row.get("primary_blocker"),
        "secondary_constraints": list(row.get("secondary_constraints") or []),
        "runtime_shortlist_candidates_checked": row.get(
            "runtime_shortlist_candidates_checked"
        ),
        "runtime_shortlist_rank": row.get("runtime_shortlist_rank"),
        "runtime_shortlist_size": row.get("runtime_shortlist_size"),
        "trajectory_plan_id": row.get("trajectory_plan_id"),
        "final_lz_selected_id": row.get("final_lz_selected_id"),
        "final_lz_pairs_checked": None
        if final_pairs is None
        else [
            {
                "lz_id": item.get("lz_id"),
                "proposal_rank": item.get("proposal_rank"),
                "status": item.get("status"),
                "reason": item.get("reason"),
            }
            for item in final_pairs
        ],
    }


def actual_release_rows(info: Mapping[str, Any], call_time: float) -> dict[str, dict[str, Any]]:
    decisions = info.get("release_decisions") or {}
    telemetries = info.get("release_intent_telemetry") or {}
    rows: dict[str, dict[str, Any]] = {}
    for uid, decision in decisions.items():
        uid_text = str(uid)
        telemetry = telemetries.get(uid_text)
        rows[uid_text] = {
            "time": float(call_time),
            "uid": int(uid),
            "decision": decision,
            "telemetry": telemetry_payload(telemetry),
        }
    return rows


def lz_compact_status(env: TerminalParallelEnv) -> dict[str, Any]:
    zones = zone_payload(env)
    return {
        "zones": zones,
        "free_lz_ids": [
            int(item["lz_id"]) for item in zones if item["status"] == "free"
        ],
        "reserved_lz_ids": [
            int(item["lz_id"]) for item in zones if item["status"] == "reserved"
        ],
        "occupied_lz_ids": [
            int(item["lz_id"]) for item in zones if item["status"] == "occupied"
        ],
    }


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(payload), ensure_ascii=False, sort_keys=True), flush=True)


def build_payload(
    *,
    env: TerminalParallelEnv,
    seed: int,
    wall_seconds: float,
    stop_reason: str,
    checkpoints: Mapping[str, Any],
    phase_changes: Mapping[str, list[dict[str, Any]]],
    first_low_soc: Mapping[str, Any],
    release_counts: Mapping[str, Any],
    latest_reasons: Mapping[str, list[dict[str, Any]]],
    reject_reason_evidence: Mapping[str, Any],
    emergency_commits: list[dict[str, Any]],
    touchdown_events: list[dict[str, Any]],
    event_log: list[dict[str, Any]],
    safety_observations: Mapping[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    outcomes = [jsonable(item) for item in env.energy_outcomes]
    outcome_counts = Counter(str(item.get("outcome")) for item in outcomes)
    return {
        "schema_version": "flow-baseline-841301-v1",
        "seed": int(seed),
        "config": {
            "initial_soc_min": 0.55,
            "initial_soc_max": 0.90,
            "route_guard_enabled": True,
            "max_simulation_seconds": 1800.0,
            "dense_warm_start_uavs": 30,
            "final_approach_geometry_mode": "continuous_inward_v1",
            "release_arbitration_mode": "downstream_priority",
            "reconcile_late_merge_slots_enabled": False,
            "emergency_trigger_soc": float(env.config.emergency_trigger_soc),
            "safe_distance": float(env.config.safe_distance),
            "nmac_distance": float(env.config.nmac_distance),
            "collision_distance": float(env.config.collision_distance),
        },
        "run": {
            "steps": int(env.step_count),
            "sim_time": float(env.now),
            "wall_seconds": float(wall_seconds),
            "stop_reason": stop_reason,
            "active_uids": sorted(int(uid) for uid in env.active),
            "service_uids": sorted(int(uid) for uid in env.service_uavs),
            "queue_size": len(env.admission.queue.items),
            "resolved_count": len(env.ledger.outcomes),
            "error": error,
        },
        "checkpoints_551_552_553": checkpoints,
        "uid_3_4": {
            "phase_changes": phase_changes,
            "first_low_soc_at_or_below_0_40": first_low_soc,
            "release_counts": release_counts,
            "latest_10_actual_release_reasons": latest_reasons,
            "candidate_reject_evidence_while_soc_gt_0_40": reject_reason_evidence,
        },
        "emergency_commits": emergency_commits,
        "touchdown_events": touchdown_events,
        "energy_outcomes": outcomes,
        "outcome_counts": dict(outcome_counts),
        "ledger_outcomes": {str(key): jsonable(value) for key, value in env.ledger.outcomes.items()},
        "safety": safety_observations,
        "events_of_interest": event_log,
        "telemetry": jsonable(env.telemetry),
        "conflict_events": jsonable(env.conflict_events),
    }


def run_trace(*, seed: int, max_steps: int, output: Path | None, progress: Path | None) -> dict[str, Any]:
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
    env.reset(seed=int(seed))
    started = time.perf_counter()
    checkpoints: dict[str, Any] = {}
    phase_changes: dict[str, list[dict[str, Any]]] = {str(uid): [] for uid in TRACKED_UIDS}
    first_low_soc: dict[str, Any] = {str(uid): None for uid in TRACKED_UIDS}
    release_counts: dict[str, dict[str, Any]] = {
        str(uid): {
            "decision_counts": Counter(),
            "primary_blocker_counts": Counter(),
            "secondary_constraint_counts": Counter(),
            "safety_decision_after_call_counts": Counter(),
        }
        for uid in TRACKED_UIDS
    }
    latest_reasons: dict[str, deque[dict[str, Any]]] = {
        str(uid): deque(maxlen=10) for uid in TRACKED_UIDS
    }
    reject_reason_evidence: dict[str, dict[str, Any]] = {
        str(uid): {
            "soc_gt_0_40_actual_release_call_count": 0,
            "candidate_evaluation_call_count": 0,
            "decision_counts": Counter(),
            "primary_blocker_counts": Counter(),
            "secondary_constraint_counts": Counter(),
            "candidate_pair_reason_counts": Counter(),
            "reason_time_ranges": {},
            "earliest_by_reason": {},
            "first_examples": [],
        }
        for uid in TRACKED_UIDS
    }
    emergency_commits: list[dict[str, Any]] = []
    touchdown_events: list[dict[str, Any]] = []
    event_log: list[dict[str, Any]] = []
    safety_observations: dict[str, Any] = {
        "per_step_actual_release_calls_recorded": True,
        "safety_decision_field": "env.last_safety_decisions sampled only alongside current info.release_decisions",
        "hard_separation_violations": 0,
        "nmac_violations": 0,
        "collision_events": 0,
    }
    previous_phases: dict[int, tuple[Any, Any]] = {}
    for uid in TRACKED_UIDS:
        if uid in env.active:
            initial = state_payload(env.active[uid])
            previous_phases[uid] = (initial["phase"], initial["ring_role"])
            phase_changes[str(uid)].append({"time": 0.0, **initial})

    event_count = len(env.events)
    outcome_count = len(env.energy_outcomes)
    stop_reason = "max_steps"
    caught_error: str | None = None
    steps = 0
    try:
        for step_index in range(int(max_steps)):
            before_time = float(env.now)
            before_states = {
                int(uid): state_payload(uav) for uid, uav in env.active.items()
            }
            _, _, _, _, info = env.step()
            steps = step_index + 1
            after_time = float(env.now)

            current_release_rows = actual_release_rows(info, before_time)
            for uid in TRACKED_UIDS:
                uid_text = str(uid)
                row = current_release_rows.get(uid_text)
                if row is None:
                    continue
                counts = release_counts[uid_text]
                counts["decision_counts"][str(row["decision"])] += 1
                telemetry = row.get("telemetry") or {}
                blocker = telemetry.get("primary_blocker")
                if blocker is not None:
                    counts["primary_blocker_counts"][str(blocker)] += 1
                for reason in telemetry.get("secondary_constraints") or []:
                    counts["secondary_constraint_counts"][str(reason)] += 1
                if uid in env.last_safety_decisions:
                    counts["safety_decision_after_call_counts"][
                        str(env.last_safety_decisions[uid])
                    ] += 1
                latest_reasons[uid_text].append(
                    {
                        "time": float(before_time),
                        "decision": row["decision"],
                        "primary_blocker": blocker,
                        "secondary_constraints": telemetry.get("secondary_constraints") or [],
                        "final_lz_pairs_checked": telemetry.get("final_lz_pairs_checked"),
                    }
                )
                before = before_states.get(uid)
                if before is not None and float(before["soc"]) > LOW_SOC_THRESHOLD:
                    evidence = reject_reason_evidence[uid_text]
                    evidence["soc_gt_0_40_actual_release_call_count"] += 1
                    decision = str(row["decision"])
                    candidate_pairs = telemetry.get("final_lz_pairs_checked") or []
                    candidate_reasons = []
                    if blocker is not None:
                        candidate_reasons.append(str(blocker))
                    candidate_reasons.extend(
                        str(reason) for reason in telemetry.get("secondary_constraints") or []
                    )
                    candidate_reasons.extend(
                        str(item.get("reason"))
                        for item in candidate_pairs
                        if item.get("reason") is not None
                    )
                    is_candidate_call = decision not in {"plan_active", "safety_wait"}
                    if is_candidate_call:
                        evidence["candidate_evaluation_call_count"] += 1
                        evidence["decision_counts"][decision] += 1
                        for reason in candidate_reasons:
                            evidence["candidate_pair_reason_counts"][reason] += 1
                            range_row = evidence["reason_time_ranges"].setdefault(
                                reason, {"first_time": before_time, "last_time": before_time}
                            )
                            range_row["first_time"] = min(range_row["first_time"], before_time)
                            range_row["last_time"] = max(range_row["last_time"], before_time)
                            if reason not in evidence["earliest_by_reason"]:
                                evidence["earliest_by_reason"][reason] = {
                                    "time": float(before_time),
                                    "soc": before["soc"],
                                    "phase": before["phase"],
                                    "ring_role": before["ring_role"],
                                    "decision": decision,
                                    "primary_blocker": blocker,
                                    "secondary_constraints": telemetry.get("secondary_constraints") or [],
                                    "final_lz_pairs_checked": candidate_pairs,
                                }
                        if len(evidence["first_examples"]) < 8:
                            evidence["first_examples"].append(
                                {
                                    "time": float(before_time),
                                    "soc": before["soc"],
                                    "phase": before["phase"],
                                    "ring_role": before["ring_role"],
                                    "decision": decision,
                                    "primary_blocker": blocker,
                                    "secondary_constraints": telemetry.get("secondary_constraints") or [],
                                    "final_lz_pairs_checked": candidate_pairs,
                                }
                            )

            for uid in TRACKED_UIDS:
                if uid in env.active:
                    state = state_payload(env.active[uid])
                    phase_key = (state["phase"], state["ring_role"])
                    if previous_phases.get(uid) != phase_key:
                        phase_changes[str(uid)].append(state)
                        previous_phases[uid] = phase_key
                    if first_low_soc[str(uid)] is None and state["soc"] <= LOW_SOC_THRESHOLD:
                        first_low_soc[str(uid)] = {"time": float(after_time), **state}

            new_events = env.events[event_count:]
            event_count = len(env.events)
            for event in new_events:
                if not isinstance(event, (tuple, list)) or not event:
                    continue
                kind = str(event[0])
                uid = None if len(event) < 2 else int(event[1])
                event_time = None if len(event) < 3 else float(event[2])
                if kind not in {"emergency_departure", "touchdown", "completed"}:
                    continue
                record: dict[str, Any] = {
                    "kind": kind,
                    "uid": uid,
                    "time": event_time,
                    "step_time_before": before_time,
                    "step_time_after": after_time,
                }
                if kind == "emergency_departure" and uid is not None:
                    before = before_states.get(uid)
                    after_uav = env.active.get(uid) or env.service_uavs.get(uid)
                    after = None if after_uav is None else state_payload(after_uav)
                    plan = None if after_uav is None else after_uav.committed_plan
                    record.update(
                        {
                            "soc_at_commit": None if before is None else before.get("soc"),
                            "phase_before_commit": None if before is None else before.get("phase"),
                            "ring_role_before_commit": None if before is None else before.get("ring_role"),
                            "position_at_commit": None if before is None else before.get("position"),
                            "phase_after_commit": None if after is None else after.get("phase"),
                            "ring_role_after_commit": None if after is None else after.get("ring_role"),
                            "assigned_lz": None if after is None else after.get("assigned_lz"),
                            "committed_plan_id": None if plan is None else plan.plan_id,
                            "committed_plan_mode": None if plan is None else plan.mode,
                            "committed_plan_start_time": None if plan is None else float(plan.start_time),
                            "committed_plan_end_time": None if plan is None else float(plan.end_time),
                        }
                    )
                    emergency_commits.append(record)
                elif kind == "touchdown":
                    touchdown_events.append(record)
                event_log.append(record)

            new_outcomes = env.energy_outcomes[outcome_count:]
            outcome_count = len(env.energy_outcomes)
            for row in new_outcomes:
                if row.get("uid") in TRACKED_UIDS and first_low_soc[str(row["uid"])] is None:
                    if row.get("soc", 1.0) <= LOW_SOC_THRESHOLD:
                        first_low_soc[str(row["uid"])] = {
                            "time": float(row.get("time", after_time)),
                            **jsonable(row),
                        }

            safety_observations["hard_separation_violations"] = int(
                env.telemetry.get("hard_separation_violations", 0)
            )
            safety_observations["nmac_violations"] = int(
                env.telemetry.get("nmac_entries", 0)
            )
            safety_observations["collision_events"] = sum(
                item.get("kind") == "collision" for item in env.conflict_events
            )

            if after_time in CHECKPOINTS:
                checkpoints[str(int(after_time))] = {
                    "time": after_time,
                    "step_time_before": before_time,
                    "active": [
                        state_payload(env.active[uid])
                        for uid in sorted(env.active)
                    ],
                    "service_uids": sorted(int(uid) for uid in env.service_uavs),
                    "queue_size": len(env.admission.queue.items),
                    "landing_zones": lz_compact_status(env),
                    "actual_release_calls": current_release_rows,
                    "new_events": jsonable(new_events),
                    "new_energy_outcomes": jsonable(new_outcomes),
                    "safety_decisions_after_call": {
                        str(uid): env.last_safety_decisions.get(uid)
                        for uid in sorted(int(key) for key in current_release_rows)
                    },
                }

            if progress is not None and (steps % 100 == 0 or terminal_now(env)):
                _write_progress(
                    progress,
                    {
                        "step": steps,
                        "sim_time": float(env.now),
                        "wall_elapsed": time.perf_counter() - started,
                        "active": len(env.active),
                        "service": len(env.service_uavs),
                        "resolved": len(env.ledger.outcomes),
                        "failed_energy": sum(
                            item.get("outcome") == "failed_energy"
                            for item in env.energy_outcomes
                        ),
                        "status": "terminal" if terminal_now(env) else "running",
                    },
                )
            if terminal_now(env):
                stop_reason = "terminal"
                break
    except Exception as exc:  # preserve exact error text in the artifact
        caught_error = repr(exc)
        stop_reason = "exception"
        if progress is not None:
            _write_progress(
                progress,
                {
                    "step": steps,
                    "sim_time": float(env.now),
                    "wall_elapsed": time.perf_counter() - started,
                    "status": "exception",
                    "exception": caught_error,
                },
            )

    for uid in TRACKED_UIDS:
        release_counts[str(uid)] = {
            key: dict(value) for key, value in release_counts[str(uid)].items()
        }

    payload = build_payload(
        env=env,
        seed=seed,
        wall_seconds=time.perf_counter() - started,
        stop_reason=stop_reason,
        checkpoints=checkpoints,
        phase_changes=phase_changes,
        first_low_soc=first_low_soc,
        release_counts=release_counts,
        latest_reasons={uid: list(rows) for uid, rows in latest_reasons.items()},
        reject_reason_evidence=reject_reason_evidence,
        emergency_commits=emergency_commits,
        touchdown_events=touchdown_events,
        event_log=event_log,
        safety_observations=safety_observations,
        error=caught_error,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        output.write_text(
            json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/flow_baseline_841301.json"))
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=841301)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    if args.preflight:
        payload = run_trace(seed=int(args.seed), max_steps=1, output=None, progress=None)
        encoded = json.dumps(jsonable(payload), ensure_ascii=False, sort_keys=True)
        print(json.dumps({
            "preflight": "ok",
            "steps": payload["run"]["steps"],
            "serialized_bytes": len(encoded.encode("utf-8")),
            "checkpoints": sorted(payload["checkpoints_551_552_553"]),
            "error": payload["run"]["error"],
        }, ensure_ascii=False, sort_keys=True))
        return

    if args.progress is None:
        args.progress = Path(f"{args.output}.progress.json")
    payload = run_trace(
        seed=int(args.seed),
        max_steps=int(args.max_steps),
        output=args.output,
        progress=args.progress,
    )
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size if args.output.exists() else None,
        "steps": payload["run"]["steps"],
        "sim_time": payload["run"]["sim_time"],
        "stop_reason": payload["run"]["stop_reason"],
        "outcome_counts": payload["outcome_counts"],
        "checkpoints": sorted(payload["checkpoints_551_552_553"]),
        "emergency_commits": len(payload["emergency_commits"]),
        "error": payload["run"]["error"],
    }, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
