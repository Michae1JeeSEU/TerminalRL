"""Extract the bounded landing-flow evidence summary after fresh cells exist."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "docs" / "evidence" / "v4_2" / "landing_flow_repair_20260924.json"

TRIAL_PATHS = {
    "baseline_841301": RESULTS / "flow_baseline_841301.json",
    "dev_v1_841301": RESULTS / "flow_dev_841301_v1.json",
    "dev_v2_841301": RESULTS / "flow_dev_841301_v2.json",
    "dev_v5_841301": RESULTS / "flow_dev_841301_v5.json",
    "dev_v6_841301": RESULTS / "flow_dev_841301_v6.json",
    "dev_v7_841301": RESULTS / "flow_dev_841301_v7.json",
    "fresh_841501_baseline": RESULTS / "flow_fresh_841501_baseline.json",
    "fresh_841501_coordinated": RESULTS / "flow_fresh_841501_coordinated.json",
    "fresh_841502_baseline": RESULTS / "flow_fresh_841502_baseline.json",
    "fresh_841502_coordinated": RESULTS / "flow_fresh_841502_coordinated.json",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object JSON: {path}")
    return value


def pick(mapping: Mapping[str, Any], *keys: str, default=None):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def min_landing_soc(raw: Mapping[str, Any]):
    rows = raw.get("touchdowns") or []
    values = [float(row["soc"]) for row in rows if isinstance(row, Mapping) and row.get("soc") is not None]
    if not values:
        rows = raw.get("energy_outcomes") or []
        values = [
            float(row["soc"])
            for row in rows
            if isinstance(row, Mapping)
            and row.get("outcome") in {"completed", "emergency_landed"}
            and row.get("soc") is not None
        ]
    return min(values) if values else None


def normalized_safety(raw: Mapping[str, Any]) -> dict[str, Any]:
    existing = raw.get("safety_summary")
    if isinstance(existing, Mapping) and existing:
        return dict(existing)
    cfg = raw.get("config") or {}
    run = raw.get("run") or {}
    telemetry = raw.get("telemetry") or {}
    requested = pick(cfg, "total_requested_uavs", "dense_warm_start_uavs")
    resolved = run.get("resolved_count")
    return {
        "checked_executed_plan_hash_mismatches": telemetry.get("checked_executed_plan_hash_mismatches"),
        "collisions": pick(telemetry, "collisions", default=0),
        "conflict_event_count": len(raw.get("conflict_events") or []),
        "hard_separation_violations": pick(telemetry, "hard_separation_violations", default=0),
        "joint_safety_fail_closed": pick(telemetry, "joint_safety_fail_closed", default=0),
        "minimum_actual_separation": telemetry.get("minimum_actual_separation"),
        "minimum_swept_separation": telemetry.get("minimum_swept_separation"),
        "nmac": pick(telemetry, "nmac", default=0),
        "requested": requested,
        "resolved": resolved,
        "safety_error": run.get("error"),
        "safety_interventions": telemetry.get("safety_interventions"),
        "shield_fail_closed": pick(telemetry, "shield_fail_closed", default=0),
        "unresolved": None if requested is None or resolved is None else int(requested) - int(resolved),
    }


def trial_record(label: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    safety = normalized_safety(raw)
    outcomes = dict(raw.get("outcome_counts") or {})
    cfg = raw.get("config") or {}
    run = raw.get("run") or {}
    requested = pick(safety, "requested", default=pick(cfg, "total_requested_uavs", "dense_warm_start_uavs"))
    resolved = pick(safety, "resolved", default=run.get("resolved_count"))
    if resolved is None and outcomes:
        resolved = sum(int(value) for value in outcomes.values() if isinstance(value, (int, float)))
    if requested is None and resolved is not None:
        requested = resolved
    safety.setdefault("requested", requested)
    safety.setdefault("resolved", resolved)
    if safety.get("unresolved") is None and requested is not None and resolved is not None:
        safety["unresolved"] = int(requested) - int(resolved)
    stop_reason = pick(raw, "stop_reason", default=run.get("stop_reason"))
    coordinated = raw.get("coordinated")
    if coordinated is None:
        coordinated = cfg.get("rule_flow_coordination_enabled", False)
    return {
        "label": label,
        "seed": raw.get("seed"),
        "coordinated": bool(coordinated),
        "status": stop_reason or "unknown",
        "outcomes": outcomes,
        "sim_time_seconds": pick(raw, "sim_time_seconds", default=run.get("sim_time")),
        "wall_seconds": pick(raw, "wall_seconds", default=run.get("wall_seconds")),
        "minimum_landing_soc": min_landing_soc(raw),
        "safety_summary": safety,
        "requested": requested,
        "resolved": resolved,
    }


def fixed_conditions(raw: Mapping[str, Any]) -> dict[str, Any]:
    cfg = raw.get("config") or {}
    keys = (
        "action_schema_version",
        "energy_model_mode",
        "warm_start_uavs",
        "total_requested_uavs",
        "warm_start_steady_motion",
        "initial_soc_mode",
        "initial_soc_min",
        "initial_soc_max",
        "energy_reachability_guard_enabled",
        "emergency_priority_guard_enabled",
        "emergency_trigger_soc",
        "emergency_energy_margin_seconds",
        "final_approach_geometry_mode",
        "release_arbitration_mode",
        "reconcile_late_merge_slots_enabled",
        "final_queue_capacity",
        "safe_distance",
        "nmac_distance",
        "collision_distance",
        "dt",
        "physics_dt",
        "safety_sampling_dt",
        "max_simulation_seconds",
        "rule_flow_coordination_enabled",
        "max_speed_residual_mps",
    )
    result = {key: cfg[key] for key in keys if key in cfg}
    if "energy_reachability_guard_enabled" in result:
        result["route_guard_enabled"] = result["energy_reachability_guard_enabled"]
    result["source"] = "flow_dev_841301_v7.json"
    return result


def v4_aborted_record() -> dict[str, Any]:
    raw = load_json(RESULTS / "flow_dev_841301_v4_aborted.json")
    progress = dict(raw.get("last_progress") or {})
    return {
        "label": "dev_v4_841301_aborted",
        "status": raw.get("status"),
        "complete": False,
        "reason": raw.get("reason"),
        "sim_time_seconds": progress.get("sim_time"),
        "wall_seconds": progress.get("wall_seconds"),
        "steps": progress.get("step"),
        "requested": 30,
        "resolved": progress.get("resolved"),
        "outcomes": progress.get("outcome_counts"),
        "observed_state_counts": raw.get("observed_state_counts"),
        "failed_gates": raw.get("failed_gates"),
        "incomplete_gates": raw.get("incomplete_gates"),
        "unavailable_metrics": raw.get("unavailable_metrics"),
    }


def diagnosis_552() -> dict[str, Any]:
    raw = load_json(RESULTS / "landing_blockers_841301_baseline.json")
    at = raw.get("at_552") or {}
    direct = at.get("direct_state_by_uid") or {}
    release = at.get("release_snapshot_3_4_7_23") or {}
    selected_direct = {}
    for uid in ("3", "4"):
        row = direct.get(uid) or {}
        selected_direct[uid] = {
            key: row.get(key)
            for key in ("uid", "time", "phase", "ring_role", "soc", "energy_remaining",
                        "committed", "assigned_lz", "last_release_decision", "last_safety_decision")
            if key in row
        }
    selected_release = {}
    for uid in ("7", "23"):
        row = release.get(uid) or {}
        telemetry = row.get("release_telemetry") or {}
        selected_release[uid] = {
            "release_decision": row.get("release_decision"),
            "latest_decision_history_entry": row.get("latest_decision_history_entry"),
            "latest_candidate_reason": row.get("latest_candidate_reason"),
            "runtime_shortlist_candidates_checked": telemetry.get("runtime_shortlist_candidates_checked"),
            "primary_blocker": telemetry.get("primary_blocker"),
            "safety_decision": row.get("safety_decision"),
            "final_lz_pairs_checked_note": "recorded prior telemetry; not treated as same-step safety rejection",
        }
    return {
        "source": "results/landing_blockers_841301_baseline.json",
        "time": at.get("time"),
        "baseline_conditions": raw.get("baseline"),
        "uid3_uid4_direct_state": selected_direct,
        "uid7_uid23_release_snapshot": selected_release,
        "landing_zone_status": "unavailable_from_existing_trace",
        "unavailable_from_existing_trace": raw.get("unavailable_from_existing_trace"),
    }


def v3_pair_witness() -> dict[str, Any]:
    raw = load_json(RESULTS / "flow_dev_841301_v3_original_pair_probe.json")
    ctx = raw.get("safety_error_pair_context") or {}
    diag = raw.get("safety_error_diagnostic") or {}
    pair = (ctx.get("pairs") or [{}])[0]
    uavs = pair.get("uavs") or []
    return {
        "source": "results/flow_dev_841301_v3_original_pair_probe.json",
        "time": ctx.get("time", diag.get("time")),
        "pair": pair.get("pair"),
        "uid19": {
            key: next((row.get(key) for row in uavs if row.get("uid") == 19), None)
            for key in ("uid", "phase", "assigned_lz", "committed", "plan_end_time", "soc")
        },
        "uid21": {
            key: next((row.get(key) for row in uavs if row.get("uid") == 21), None)
            for key in ("uid", "phase", "ring_role", "committed", "soc")
        },
        "minimum_predicted_separation": ctx.get("minimum_predicted_separation"),
        "minimum_actual_separation": raw.get("safety_summary", {}).get("minimum_actual_separation"),
        "minimum_swept_separation": raw.get("safety_summary", {}).get("minimum_swept_separation"),
        "old_filter_point_tube": "passed",
        "runtime_pair_swept_check": "79.92809338226441 m",
    }


def v5_missed_emergency() -> dict[str, Any]:
    raw = load_json(RESULTS / "flow_v5_missed_emergency_probe.json")
    return {
        "source": "results/flow_v5_missed_emergency_probe.json",
        "steps": (raw.get("stop") or {}).get("steps"),
        "sim_time_seconds": (raw.get("stop") or {}).get("sim_time"),
        "witnesses": raw.get("witnesses") or [],
        "witness_counts": (raw.get("stop") or {}).get("witness_counts"),
        "earliest_normal_energy_guard_rejection": raw.get("earliest_normal_energy_guard_rejection"),
    }


def v7_uid3_uid4_timeline(raw: Mapping[str, Any]) -> dict[str, Any]:
    def rows(source: str, uid: int, fields: tuple[str, ...]):
        result = []
        for row in raw.get(source) or []:
            if int(row.get("uid", -1)) != uid:
                continue
            result.append({key: row.get(key) for key in fields if key in row})
        return result

    normal_fields = ("uid", "time", "from_phase", "end_time", "pad_id", "tangential_speed_bias_mps", "soc")
    emergency_fields = ("uid", "time", "from_phase", "end_time", "pad_id", "tangential_speed_bias_mps", "soc")
    touchdown_fields = ("uid", "time", "plan_end_time", "pad_id", "emergency", "soc")
    result = {}
    for uid in (3, 4):
        result[str(uid)] = {
            "normal": rows("normal_commits", uid, normal_fields),
            "emergency": rows("emergency_commits", uid, emergency_fields),
            "touchdown": rows("touchdowns", uid, touchdown_fields),
        }
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    raw_trials = {label: load_json(path) for label, path in TRIAL_PATHS.items()}
    records = [trial_record(label, raw) for label, raw in raw_trials.items()]
    v7 = raw_trials["dev_v7_841301"]
    fixed = fixed_conditions(v7)
    result = {
        "schema_version": "landing-flow-evidence-summary-v1",
        "source_policy": "compact development, baseline, and fresh trial records; no full configs or all-state rows",
        "scenario_fixed_conditions": fixed,
        "trials": records,
        "v4_aborted": v4_aborted_record(),
        "diagnosis_552": diagnosis_552(),
        "v3_pair_witness": v3_pair_witness(),
        "v5_missed_emergency": v5_missed_emergency(),
        "v7_uid3_uid4_timeline": v7_uid3_uid4_timeline(v7),
        "fresh_status": "fresh records present; compact summary records observed metrics without pass/fail reinterpretation",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if len(text.encode("utf-8")) >= 50 * 1024:
        raise RuntimeError("compact evidence summary exceeds 50KB")
    OUTPUT.write_text(text, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "bytes": len(text.encode("utf-8")), "trials": len(records)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
