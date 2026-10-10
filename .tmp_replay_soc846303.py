"""One bounded, read-only replay trace for candidate arrival 20."""
from __future__ import annotations

from collections import Counter, defaultdict
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import (
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import diagnose_trained_dense30_v2 as diagnostic


TAPE = Path(
    "/home/user/training_outputs/terminal/"
    "random_soc_gate_repair_v1_20260925/pair_dense24_846303/"
    "candidate/action_tapes/safety_trace_seed_846303_step_794.json"
)
OUTPUT = Path(
    "/home/user/training_outputs/terminal/"
    "random_soc_gate_repair_v1_20260925/replay_candidate_emergency846303.json"
)
RAW_OUTPUT = Path(
    "/home/user/training_outputs/terminal/"
    "random_soc_gate_repair_v1_20260925/replay_candidate_emergency846303_raw.json"
)
TARGET_ARRIVAL_ID = 20


class ReplayCaptureMismatch(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not np.isfinite(value):
        return "-Infinity" if value < 0 else "Infinity" if value > 0 else "NaN"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _copy_row(value: Any) -> Any:
    return json.loads(json.dumps(_jsonable(value), sort_keys=True))


def _plan_summary(plan: Any | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    result: dict[str, Any] = {"type": type(plan).__name__}
    for name in (
        "plan_id", "trajectory_hash", "source_stage", "target_stage", "start_time",
        "end_time", "merge_time", "reservation_ids", "metadata",
    ):
        if hasattr(plan, name):
            result[name] = _jsonable(getattr(plan, name))
    return result


def _uav_snapshot(uav: Any) -> dict[str, Any]:
    capacity = float(uav.energy_capacity)
    initial_soc_estimate = (
        float(uav.energy_remaining + uav.energy_consumed) / capacity
        if capacity else None
    )
    return {
        "uid": int(uav.uid),
        "arrival_id": int(uav.arrival_id),
        "request_time": float(uav.request_time),
        "base_priority": int(uav.base_priority),
        "mission_type": _jsonable(uav.mission_type),
        "phase": uav.phase.value,
        "ring_role": None if uav.ring_role is None else uav.ring_role.value,
        "phase_entry_time": float(uav.phase_entry_time),
        "soc": float(uav.soc),
        "energy_remaining": float(uav.energy_remaining),
        "energy_capacity": capacity,
        "energy_consumed": float(uav.energy_consumed),
        "initial_soc_estimate": initial_soc_estimate,
        "dynamic_priority": float(uav.dynamic_priority),
        "external_wait_time": float(uav.external_wait_time),
        "internal_wait_time": float(uav.internal_wait_time),
        "admission_time": _jsonable(uav.admission_time),
        "outer_entry_time": _jsonable(uav.outer_entry_time),
        "final_entry_time": _jsonable(uav.final_entry_time),
        "touchdown_time": _jsonable(uav.touchdown_time),
        "completion_time": _jsonable(uav.completion_time),
        "committed": bool(uav.committed_plan is not None),
        "committed_plan": _plan_summary(uav.committed_plan),
        "trajectory_plan_id": _jsonable(uav.trajectory_plan_id),
        "merge_reservation_id": _jsonable(uav.merge_reservation_id),
        "assigned_lz": _jsonable(uav.assigned_lz),
        "ring_progress": _jsonable(uav.ring_progress),
        "emergency_landing": bool(uav.emergency_landing),
        "terminal_outcome": _jsonable(uav.terminal_outcome),
        "terminal_reason": _jsonable(uav.terminal_reason),
        "safety_retry_after": float(uav.safety_retry_after),
        "emergency_retry_after": float(uav.emergency_retry_after),
    }


def _all_uavs(env: Any) -> list[Any]:
    values: list[Any] = list(env.active.values())
    values.extend(getattr(env, "service_uavs", {}).values())
    values.extend(getattr(env, "completed_uavs", ()))
    by_uid: dict[int, Any] = {}
    for uav in values:
        by_uid[int(uav.uid)] = uav
    return [by_uid[uid] for uid in sorted(by_uid)]


def _target_objects(env: Any) -> list[Any]:
    return [uav for uav in _all_uavs(env) if int(uav.arrival_id) == TARGET_ARRIVAL_ID]


def _target_snapshot(env: Any) -> list[dict[str, Any]]:
    return [_uav_snapshot(uav) for uav in _target_objects(env)]


def _release_state(env: Any) -> dict[str, dict[str, Any]]:
    return {str(key): _copy_row(value) for key, value in env.release_intents.items()}


def _release_diff(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in sorted(set(before) | set(after), key=lambda value: int(value)):
        old = before.get(key)
        new = after.get(key)
        if old == new:
            continue
        changed: dict[str, Any] = {}
        if old is None:
            changed = {field: value for field, value in (new or {}).items()}
        elif new is None:
            changed = {"deleted": True}
        else:
            for field in sorted(set(old) | set(new)):
                if old.get(field) != new.get(field):
                    changed[field] = {"before": old.get(field), "after": new.get(field)}
        result.append({"release_intent_id": int(key), "changed": changed})
    return result


def _decode(action: Any) -> dict[str, Any]:
    try:
        return hierarchical_from_normalized_v4_2(np.asarray(action, dtype=np.float32)).as_dict()
    except Exception as exc:  # raw evidence only; replay still performs its own validation
        return {"decode_error": str(exc), "raw": _jsonable(action)}


def _actor_row(env: Any, uid_key: str, action: Any, shortlist: Any, *, step: int, time: float) -> dict[str, Any] | None:
    uid = int(uid_key)
    uav = env.active.get(uid)
    if uav is None:
        return None
    eligible = (
        uav.committed_plan is None
        and uav.phase.value in {"ring_outer", "ring_middle", "ring_inner"}
        and uid not in env.future_release_reservations
    )
    return {
        "step": int(step),
        "time_before": float(time),
        "uid": uid,
        "arrival_id": int(uav.arrival_id),
        "phase": uav.phase.value,
        "ring_role": None if uav.ring_role is None else uav.ring_role.value,
        "phase_entry_time": float(uav.phase_entry_time),
        "soc": float(uav.soc),
        "energy_remaining": float(uav.energy_remaining),
        "base_priority": int(uav.base_priority),
        "actor_mask": bool(eligible),
        "requested_action": _jsonable(action),
        "decoded_requested_action": _decode(action),
        "proposal_shortlist": None if shortlist is None else [_decode(item) for item in shortlist],
    }


def _trace_subset(trace: Any) -> dict[str, Any]:
    keys = (
        "requested_low_action", "decoded_joint_action", "execution_applied_action",
        "shield_applied_action", "shield_used", "shield_reason", "teacher_used",
        "runtime_shortlist_used", "runtime_shortlist_rank",
        "runtime_shortlist_candidates_checked",
    )
    return {key: _copy_row(trace[key]) for key in keys if key in trace}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    if RAW_OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {RAW_OUTPUT}")
    payload = json.loads(TAPE.read_text(encoding="utf-8"))
    rows = payload.get("action_tape", [])
    if len(rows) != 795:
        raise ValueError(f"expected 795 action rows, got {len(rows)}")

    config = diagnostic.build_scenario_config(
        "dense24", reconcile_late_merge_slots=False,
    )
    env = TerminalParallelEnv(config)
    original_step = env.step
    capture: dict[str, Any] = {
        "target_states": [],
        "target_eligible_decisions": [],
        "actual_releases": [],
        "step_events": [],
        "commit_transitions": [],
        "mismatches": [],
        "per_phase_actor_gate_counts": defaultdict(lambda: Counter()),
        "target_uid": None,
        "calls": 0,
    }

    def fail(kind: str, step: int, detail: Any) -> None:
        record = {"kind": kind, "step": int(step), "detail": _jsonable(detail)}
        capture["mismatches"].append(record)
        raise ReplayCaptureMismatch(json.dumps(record, sort_keys=True))

    def wrapped_step(actions: dict[str, np.ndarray] | None = None,
                     proposal_shortlists: dict[str, Any] | None = None):
        index = int(capture["calls"])
        if index >= len(rows):
            fail("extra_step", index, "replay called beyond action tape")
        row = rows[index]
        actions = actions or {}
        proposal_shortlists = proposal_shortlists or {}
        step = int(row["step"])
        time_before = float(env.now)
        pre_target = _target_snapshot(env)
        pre_release = _release_state(env)
        pre_events = [_copy_row(item) for item in env.events]
        pre_energy_outcomes = [_copy_row(item) for item in env.energy_outcomes]
        actor_rows: list[dict[str, Any]] = []
        pending_release_indexes: list[int] = []
        for uid_key in sorted(actions):
            actor = _actor_row(
                env, uid_key, actions[uid_key], proposal_shortlists.get(uid_key),
                step=step, time=time_before,
            )
            if actor is None:
                continue
            actor_rows.append(actor)
            if actor["actor_mask"]:
                phase = str(actor["phase"])
                gate = str(actor["decoded_requested_action"].get("gate"))
                capture["per_phase_actor_gate_counts"][phase][gate] += 1
                if int(actor["arrival_id"]) == TARGET_ARRIVAL_ID:
                    capture["target_eligible_decisions"].append(actor)
            if actor["decoded_requested_action"].get("gate") == "RELEASE":
                capture["actual_releases"].append({
                    "step": step,
                    "time_before": time_before,
                    "uid": int(actor["uid"]),
                    "arrival_id": int(actor["arrival_id"]),
                    "actor_mask": bool(actor["actor_mask"]),
                    "phase": actor["phase"],
                    "ring_role": actor["ring_role"],
                    "soc_before": actor["soc"],
                    "requested_action": actor["requested_action"],
                    "decoded_requested_action": actor["decoded_requested_action"],
                    "proposal_shortlist": actor["proposal_shortlist"],
                    "current_rejection": None,
                    "info_action_trace": None,
                    "release_intent_telemetry": None,
                    "release_intent_diff": None,
                })
                pending_release_indexes.append(len(capture["actual_releases"]) - 1)

        # Match the frozen replay's pre-step observation/category checks before stepping.
        observed = env._observations()
        observed_hash = _digest({key: value for key, value in sorted(observed.items())})
        if observed_hash != row["observation_hash"]:
            fail("observation_hash_mismatch", step, {"expected": row["observation_hash"], "actual": observed_hash})
        for uid, action in actions.items():
            decoded = _decode(action)
            expected_gate = row.get("requested_gate", {}).get(str(uid))
            expected_proposal = row.get("requested_proposal", {}).get(str(uid))
            if decoded.get("gate_index") != expected_gate or decoded.get("proposal") != expected_proposal:
                fail("requested_category_mismatch", step, {"uid": uid, "expected": [expected_gate, expected_proposal], "actual": decoded})

        result = original_step(actions, proposal_shortlists)
        observations, _, _, _, info = result
        post_target = _target_snapshot(env)
        post_release = _release_state(env)
        post_events = [_copy_row(item) for item in env.events]
        post_energy_outcomes = [_copy_row(item) for item in env.energy_outcomes]
        target_rows = {
            "step": step,
            "time_before": time_before,
            "time_after": float(env.now),
            "pre": pre_target,
            "post": post_target,
        }
        if pre_target or post_target:
            capture["target_states"].append(target_rows)
            all_target = pre_target + post_target
            if capture["target_uid"] is None and all_target:
                capture["target_uid"] = int(all_target[0]["uid"])
        new_events = post_events[len(pre_events):]
        new_energy = post_energy_outcomes[len(pre_energy_outcomes):]
        if new_events or new_energy:
            capture["step_events"].append({
                "step": step, "time_before": time_before, "time_after": float(env.now),
                "events": new_events, "energy_outcomes": new_energy,
            })

        info_traces = info.get("action_traces", {})
        info_releases = info.get("release_intent_telemetry", {})
        decisions = info.get("release_decisions", {})
        for release_index in pending_release_indexes:
            release = capture["actual_releases"][release_index]
            uid_key = str(release["uid"])
            decision = decisions.get(uid_key)
            telemetry = info_releases.get(uid_key)
            release["decision"] = decision
            release["info_action_trace"] = _trace_subset(info_traces[uid_key]) if uid_key in info_traces else None
            release["release_intent_telemetry"] = _copy_row(telemetry) if telemetry is not None else None
            release["release_intent_diff"] = _release_diff(pre_release, post_release)
            if decision not in {"approved", "future_reserved", "future_reservation_wait", "emergency_redirected", "not_requested"}:
                release["current_rejection"] = None if telemetry is None else {
                    "decision": decision,
                    "primary_blocker": telemetry.get("primary_blocker"),
                    "secondary_constraints": telemetry.get("secondary_constraints"),
                }

        # Track target commit transitions from the actual object state.
        pre_by_uid = {int(item["uid"]): item for item in pre_target}
        post_by_uid = {int(item["uid"]): item for item in post_target}
        for uid in sorted(set(pre_by_uid) | set(post_by_uid)):
            before = pre_by_uid.get(uid)
            after = post_by_uid.get(uid)
            if before is None or after is None:
                continue
            if before.get("committed_plan") != after.get("committed_plan") or before.get("phase") != after.get("phase"):
                capture["commit_transitions"].append({
                    "step": step, "time_before": time_before, "time_after": float(env.now),
                    "uid": uid, "arrival_id": TARGET_ARRIVAL_ID,
                    "before": {key: before.get(key) for key in ("phase", "phase_entry_time", "soc", "committed", "committed_plan", "trajectory_plan_id")},
                    "after": {key: after.get(key) for key in ("phase", "phase_entry_time", "soc", "committed", "committed_plan", "trajectory_plan_id")},
                })

        post_hash = _digest(SafetyTraceRecorder._state(env))
        if post_hash != row.get("post_state_hash"):
            fail("post_state_hash_mismatch", step, {"expected": row.get("post_state_hash"), "actual": post_hash})
        executed = _jsonable({uid: trace.get("execution_applied_action") for uid, trace in info_traces.items()})
        if executed != row.get("executed_mapped_action"):
            fail("executed_action_mismatch", step, {"expected": row.get("executed_mapped_action"), "actual": executed})
        runtime = _jsonable({
            uid: {
                "used": bool(trace.get("runtime_shortlist_used")),
                "rank": trace.get("runtime_shortlist_rank"),
                "candidates_checked": int(trace.get("runtime_shortlist_candidates_checked", 0)),
            }
            for uid, trace in info_traces.items()
        })
        if runtime != row.get("runtime_shortlist_result"):
            fail("runtime_shortlist_mismatch", step, {"expected": row.get("runtime_shortlist_result"), "actual": runtime})
        capture["calls"] = index + 1
        return result

    env.step = wrapped_step
    replay = SafetyTraceRecorder.replay_action_tape(env, TAPE)
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT.write_text(
        json.dumps(_jsonable({"replay": replay, "capture": capture}), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    target_states = capture["target_states"]
    target_snapshots: list[dict[str, Any]] = []
    for row in target_states:
        for side in ("pre", "post"):
            for state in row[side]:
                target_snapshots.append({
                    "step": row["step"], "time": row["time_before"] if side == "pre" else row["time_after"],
                    "side": side, **state,
                })
    target_snapshots.sort(key=lambda item: (float(item["time"]), str(item["side"])))
    phase_first: dict[str, dict[str, Any]] = {}
    first_low: dict[str, Any] | None = None
    for state in target_snapshots:
        phase = str(state["phase"])
        phase_first.setdefault(phase, {
            "first_observed_time": state["time"],
            "phase_entry_time": state["phase_entry_time"],
            "step": state["step"],
            "side": state["side"],
            "soc": state["soc"],
        })
        if float(state["soc"]) <= 0.40 and first_low is None:
            first_low = state

    emergency_events = [
        event for item in capture["step_events"]
        for event in item["events"]
        if isinstance(event, list) and len(event) >= 2 and int(event[1]) == int(capture["target_uid"] or -1)
    ]
    target_uid = capture["target_uid"]
    target_eligible = [row for row in capture["target_eligible_decisions"] if int(row["uid"]) == int(target_uid or -1)]
    emergency_time = next((float(event[2]) for event in emergency_events if event[0] == "emergency_departure"), None)
    pre_emergency = [row for row in target_eligible if emergency_time is None or float(row["time_before"]) < emergency_time]

    for phase, counts in capture["per_phase_actor_gate_counts"].items():
        capture["per_phase_actor_gate_counts"][phase] = dict(counts)

    result = {
        "source": {"tape": str(TAPE), "schema_version": payload.get("schema_version"), "episode_seed": payload.get("episode_seed")},
        "config": {"scenario": "dense24", "reconcile_late_merge_slots_enabled": bool(config.reconcile_late_merge_slots_enabled)},
        "replay": replay,
        "exact": bool(replay.get("exact")) and not capture["mismatches"],
        "steps_captured": int(capture["calls"]),
        "target": {
            "arrival_id": TARGET_ARRIVAL_ID,
            "uid": target_uid,
            "initial_request_and_state": next((row for row in target_snapshots if row["side"] == "post" and row["step"] == min(item["step"] for item in target_states)), None) if target_states else None,
            "phase_first_observed": phase_first,
            "first_soc_le_0_40": first_low,
            "emergency_events": emergency_events,
            "target_state_trace": target_states,
            "eligible_decisions": target_eligible,
            "eligible_decisions_before_emergency_last20": pre_emergency[-20:],
            "commit_transitions": [row for row in capture["commit_transitions"] if int(row["uid"]) == int(target_uid or -1)],
        },
        "per_phase_actor_gate_counts": dict(capture["per_phase_actor_gate_counts"]),
        "actual_releases": capture["actual_releases"],
        "mismatches": capture["mismatches"],
        "raw_availability": {
            "target_state_fields": sorted(target_snapshots[0].keys()) if target_snapshots else [],
            "replay_action_rows": len(rows),
            "replay_function": "SafetyTraceRecorder.replay_action_tape",
            "current_rejection_fields": ["decision", "primary_blocker", "secondary_constraints"] if capture["actual_releases"] else [],
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    replay_brief = {key: replay.get(key) for key in (
        "exact", "reset_state_match", "config_match", "reset_recipe_match",
        "base_reset_state_match", "steps_replayed", "steps_expected",
    )}
    print(json.dumps({
        "output": str(OUTPUT), "exact": result["exact"], "replay": replay_brief,
        "steps_captured": capture["calls"], "target_uid": target_uid,
        "target_states": len(target_states), "target_eligible": len(target_eligible),
        "actual_releases": len(capture["actual_releases"]),
        "phase_first_observed": phase_first, "first_soc_le_0_40": first_low,
        "emergency_events": emergency_events,
        "per_phase_actor_gate_counts": result["per_phase_actor_gate_counts"],
        "mismatches": capture["mismatches"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
