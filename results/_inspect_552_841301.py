"""Compact, read-only inspection of the existing seed-841301 viewer trace.

This helper intentionally does not rerun the simulator.  The source trace only
stores full state samples for UIDs 4, 11, and 12 plus a 780--850 s window, so
the output records missing fields explicitly instead of reconstructing them
from stale control dictionaries.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "viewer_trace_841301_full.json"
OUTPUT = ROOT / "results" / "landing_blockers_841301_baseline.json"
PROBE_TIME = 552.0
FOCUS_UIDS = (3, 4, 7, 23)
THRESHOLDS = (0.40, 0.30, 0.25, 0.20, 0.10)


def compact_state(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "time", "uid", "phase", "ring_role", "position", "velocity", "speed",
        "soc", "energy_remaining", "committed", "trajectory_plan_id",
        "trajectory_plan_mode", "trajectory_plan_start_time",
        "trajectory_plan_end_time", "last_release_decision", "last_safety_decision",
        "assigned_lz", "emergency_landing", "emergency_retry_after",
    )
    return {key: row.get(key) for key in fields}


def compact_telemetry(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    pairs = row.get("final_lz_pairs_checked")
    return {
        "primary_blocker": row.get("primary_blocker"),
        "secondary_constraints": row.get("secondary_constraints"),
        "attempt_count": row.get("attempt_count"),
        "retry_count": row.get("retry_count"),
        "first_attempt_time": row.get("first_attempt_time"),
        "release_intent_id": row.get("release_intent_id"),
        "trajectory_plan_id": row.get("trajectory_plan_id"),
        "runtime_shortlist_candidates_checked": row.get(
            "runtime_shortlist_candidates_checked"
        ),
        "final_lz_selected_id": row.get("final_lz_selected_id"),
        "final_lz_pairs_checked": None
        if pairs is None
        else [
            {
                "lz_id": item.get("lz_id"),
                "proposal_rank": item.get("proposal_rank"),
                "reason": item.get("reason"),
                "status": item.get("status"),
            }
            for item in pairs
        ],
    }


def plan_position(plan: dict[str, Any], at_time: float) -> list[float] | None:
    points = plan.get("points") or []
    if not points or at_time < points[0]["t"] or at_time > points[-1]["t"]:
        return None
    for left, right in zip(points, points[1:]):
        if left["t"] <= at_time <= right["t"]:
            span = float(right["t"] - left["t"])
            alpha = 0.0 if span == 0.0 else (at_time - left["t"]) / span
            return [
                float(a + alpha * (b - a))
                for a, b in zip(left["position"], right["position"])
            ]
    if points[-1]["t"] == at_time:
        return [float(value) for value in points[-1]["position"]]
    return None


def main() -> None:
    trace = json.loads(SOURCE.read_text(encoding="utf-8"))
    events = trace["control_events"]
    at_probe = next((item for item in events if item.get("time") == PROBE_TIME), None)

    disappearance_by_uid = {int(item["uid"]): item for item in trace["disappearances"]}
    active_uids = sorted(
        uid for uid, item in disappearance_by_uid.items()
        if float(item["time_after"]) > PROBE_TIME
    )

    direct_states: dict[str, dict[str, Any]] = {}
    for uid_text, rows in trace.get("tracked_samples", {}).items():
        match = next((row for row in rows if row.get("time") == PROBE_TIME), None)
        if match is not None:
            direct_states[uid_text] = compact_state(match)

    plans = list(trace.get("plans", {}).values())
    plans_at_probe: dict[str, list[dict[str, Any]]] = {}
    for plan in plans:
        if float(plan["start_time"]) <= PROBE_TIME <= float(plan["end_time"]):
            metadata = plan.get("metadata") or {}
            plans_at_probe.setdefault(str(plan["uav_id"]), []).append(
                {
                    "plan_id": plan.get("plan_id"),
                    "mode": plan.get("mode"),
                    "start_time": plan.get("start_time"),
                    "end_time": plan.get("end_time"),
                    "position_from_committed_plan": plan_position(plan, PROBE_TIME),
                    "target_phase": metadata.get("target_phase"),
                    "source_ring_role": metadata.get("source_ring_role"),
                    "target_ring_role": metadata.get("target_ring_role"),
                    "trajectory_schema_version": metadata.get("trajectory_schema_version"),
                }
            )

    release_snapshot: dict[str, Any] = {}
    if at_probe is None:
        for uid in FOCUS_UIDS:
            release_snapshot[str(uid)] = {"status": "missing_control_event_at_552"}
    else:
        for uid in FOCUS_UIDS:
            uid_text = str(uid)
            telemetry = (at_probe.get("release_telemetry") or {}).get(uid_text)
            history = [] if telemetry is None else telemetry.get("decision_history") or []
            release_snapshot[uid_text] = {
                "release_decision": (at_probe.get("release_decisions") or {}).get(uid_text),
                "safety_decision": (at_probe.get("safety_decisions") or {}).get(uid_text),
                "latest_decision_history_entry": history[-1] if history else None,
                "latest_candidate_reason": [
                    item.get("reason")
                    for item in (telemetry or {}).get("final_lz_pairs_checked") or []
                ],
                "release_telemetry": compact_telemetry(telemetry),
            }

    uid4_rows = trace.get("tracked_samples", {}).get("4", [])
    threshold_crossings: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        match = next((row for row in uid4_rows if float(row["soc"]) <= threshold), None)
        threshold_crossings[str(threshold)] = None if match is None else {
            "time": match.get("time"),
            "soc": match.get("soc"),
            "phase": match.get("phase"),
            "ring_role": match.get("ring_role"),
            "last_release_decision": match.get("last_release_decision"),
        }

    phase_changes: list[dict[str, Any]] = []
    previous = None
    for row in uid4_rows:
        current = (row.get("phase"), row.get("ring_role"))
        if current != previous:
            phase_changes.append(
                {
                    "time": row.get("time"),
                    "phase": row.get("phase"),
                    "ring_role": row.get("ring_role"),
                    "soc": row.get("soc"),
                    "last_release_decision": row.get("last_release_decision"),
                }
            )
            previous = current

    history_summary: dict[str, Any] = {}
    for uid in (3, 4):
        uid_text = str(uid)
        rows = trace.get("tracked_samples", {}).get(uid_text, [])
        telemetry_events = [
            event for event in events
            if uid_text in (event.get("release_telemetry") or {})
        ]
        latest_event = telemetry_events[-1] if telemetry_events else None
        latest_telemetry = (
            None if latest_event is None
            else latest_event["release_telemetry"].get(uid_text)
        )
        history = [] if latest_telemetry is None else latest_telemetry.get("decision_history") or []
        terminal = next(
            (item for item in trace.get("energy_outcomes", []) if item.get("uid") == uid),
            None,
        )
        summary: dict[str, Any] = {
            "sample_count": len(rows),
            "state_history_available": bool(rows),
            "latest_release_telemetry_time": None if latest_event is None else latest_event.get("time"),
            "release_reason_counts": dict(Counter(history)),
            "latest_primary_blocker": None if latest_telemetry is None else latest_telemetry.get("primary_blocker"),
            "latest_secondary_constraints": None if latest_telemetry is None else latest_telemetry.get("secondary_constraints"),
            "terminal_energy_outcome": terminal,
        }
        if uid == 4:
            summary["soc_threshold_crossings"] = threshold_crossings
            summary["phase_changes"] = phase_changes
        else:
            summary["soc_threshold_crossings"] = None
            summary["phase_changes"] = None
            summary["missing_fields"] = ["phase_history", "ring_history", "soc_history", "position_history"]
        history_summary[uid_text] = summary

    emergency_rows: list[dict[str, Any]] = []
    for outcome in trace.get("energy_outcomes", []):
        if outcome.get("outcome") != "emergency_landed":
            continue
        uid = int(outcome["uid"])
        emergency_plans = sorted(
            (plan for plan in plans if int(plan["uav_id"]) == uid and plan.get("mode") == "emergency_landing"),
            key=lambda plan: float(plan["start_time"]),
        )
        emergency_plan = emergency_plans[0] if emergency_plans else None
        prior_ring = None
        if emergency_plan is not None:
            prior = [
                plan for plan in plans
                if int(plan["uav_id"]) == uid
                and float(plan["end_time"]) <= float(emergency_plan["start_time"])
                and plan.get("mode") == "ring_transfer"
            ]
            if prior:
                prior_ring = sorted(prior, key=lambda plan: float(plan["end_time"]))[-1].get("metadata", {}).get("target_phase")
        emergency_rows.append(
            {
                "uid": uid,
                "emergency_start_time": None if emergency_plan is None else emergency_plan.get("start_time"),
                "from_ring": prior_ring,
                "touchdown_time_from_plan_end": None if emergency_plan is None else emergency_plan.get("end_time"),
                "soc_at_emergency_start": None,
                "touchdown_soc": None,
                "recorded_energy_outcome_soc": outcome.get("soc"),
                "recorded_energy_outcome_time": outcome.get("time"),
                "energy_remaining": outcome.get("energy_remaining"),
                "energy_consumed": outcome.get("energy_consumed"),
                "missing_fields": ["soc_at_emergency_start", "soc_at_touchdown"],
            }
        )

    output = {
        "schema_version": "landing-blockers-841301-v1",
        "source_trace": str(SOURCE.relative_to(ROOT)),
        "baseline": trace.get("config"),
        "trace_run": trace.get("run"),
        "at_552": {
            "time": PROBE_TIME,
            "active_uids_inferred_from_disappearance_time": active_uids,
            "direct_state_snapshot_available_for_uids": sorted(int(uid) for uid in direct_states),
            "direct_state_by_uid": direct_states,
            "committed_plans_covering_552": plans_at_probe,
            "release_snapshot_3_4_7_23": release_snapshot,
            "landing_zone_status": None,
            "landing_zone_status_missing_because_trace_window_starts_at": trace.get("window", {}).get("start"),
            "missing_fields_for_all_active_552_states": [
                "position", "phase", "ring_role", "soc", "committed_plan_id"
            ],
            "missing_state_uids": [
                uid for uid in active_uids if str(uid) not in direct_states
            ],
        },
        "uid_3_4_history": history_summary,
        "emergency_landing_8_events": sorted(emergency_rows, key=lambda row: row["emergency_start_time"] or 1e99),
        "unavailable_from_existing_trace": [
            "complete 552 s state rows for UIDs other than 4, 11, and 12",
            "landing-zone status at 552 s",
            "SOC at emergency takeover and at touchdown",
        ],
    }
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "active_uids_at_552": active_uids,
        "direct_states": sorted(int(uid) for uid in direct_states),
        "emergency_events": len(emergency_rows),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
