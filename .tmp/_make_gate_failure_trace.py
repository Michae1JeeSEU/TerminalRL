import hashlib
import json
from pathlib import Path

TRAIN = Path(r"D:\7_codespace\training_outputs\terminal\p3_followup_20260930")
GATE = TRAIN / "gate_release_control"
LEGACY = TRAIN / "forced_wait_frozen"
FULL_PROOF = Path(r"D:\7_codespace\terminal\.tmp\p3_gate_control_full_proof.json")
OUT_JSON = Path(r"D:\7_codespace\terminal\.tmp\p3_gate_control_failure_trace.json")
OUT_TXT = Path(r"D:\7_codespace\terminal\.tmp\p3_gate_control_failure_trace.txt")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(metrics):
    return {key: metrics.get(key) for key in ("normal", "emergency", "failed", "diverted", "unresolved")}


def denominator(metrics):
    return {
        key: metrics.get(key)
        for key in ("generated", "admitted", "completed", "resolved")
    }


def request_group_summary(arm):
    groups = arm.get("request_groups", [])
    nonempty = [slot for slot in groups if slot]
    requests = []
    for slot in groups:
        if not isinstance(slot, list):
            continue
        for group in slot:
            if isinstance(group, dict):
                requests.append(group)
            elif isinstance(group, list):
                requests.extend(item for item in group if isinstance(item, dict))
    target = [item for item in requests if item.get("arrival_id") == 21]
    return {
        "slots": len(groups),
        "nonempty_slots": len(nonempty),
        "request_records": len(requests),
        "arrival21_request_signatures": target,
        "contains_actions_or_decisions": any(
            any(key in item for key in ("action", "decision", "phase", "soc"))
            for item in requests
        ),
    }


def relevant_telemetry(arm):
    keys = (
        "release_approved",
        "release_rejected",
        "emergency_attempts",
        "emergency_committed",
        "emergency_touchdowns",
        "emergency_blocked_dynamics",
        "emergency_blocked_energy",
        "emergency_blocked_lz",
        "emergency_blocked_safety",
        "emergency_blocked_slack",
        "emergency_energy_guard_rejections",
        "emergency_lz_priority_rejections",
        "runtime_shortlist_candidates_checked",
        "runtime_shortlist_exhausted",
        "runtime_shortlist_fallback_selected",
        "energy_reachability_checks",
        "energy_reachability_negative_budgets",
        "recursive_feasibility_checks",
        "recursive_feasibility_rejections",
        "safety_interventions",
        "collisions",
        "nmac",
        "hard_separation_violations",
    )
    telemetry = arm.get("telemetry", {})
    return {key: telemetry.get(key) for key in keys}


def failure_row(arm, arrival_id):
    return next(
        (row for row in arm.get("ledger", {}).get("terminal_rows", []) if row.get("arrival_id") == arrival_id),
        None,
    )


full_proof = load(FULL_PROOF)
preflight = load(GATE / "preflight.json")
source = preflight["runtime"]["source_binding"]
config = preflight["runtime"]["config_snapshot"]
control_path = GATE / "arms" / "pair_0015_control.json"
reference_path = LEGACY / "arms" / "pair_0015_legacy.json"
control = load(control_path)
reference = load(reference_path)
control_failure = failure_row(control, 21)
reference_same_arrival = failure_row(reference, 21)

pair15 = next(item for item in full_proof["pairs"] if item["pair"]["pair_id"] == 15)
arrival13 = full_proof["arrival13_env877104"]

table = []
for item in full_proof["pairs"]:
    control_metrics = item["control"]["metrics"]
    reference_metrics = item["reference"]["metrics"]
    checks = item["reference_control_checks"]
    table.append(
        {
            "pair_id": item["pair"]["pair_id"],
            "env_seed": item["pair"]["env_seed"],
            "keyed_policy_seed": item["pair"]["keyed_policy_seed"],
            "status": item["status"],
            "control_NEFDU": counts(control_metrics),
            "reference_NEFDU": counts(reference_metrics),
            "control_denominator": denominator(control_metrics),
            "reference_denominator": denominator(reference_metrics),
            "control_wait_seconds": control_metrics.get("wait_seconds"),
            "reference_wait_seconds": reference_metrics.get("wait_seconds"),
            "delta_wait_seconds": item["delta"].get("wait_seconds"),
            "control_request_elapsed_seconds": control_metrics.get("request_elapsed_total_seconds"),
            "reference_request_elapsed_seconds": reference_metrics.get("request_elapsed_total_seconds"),
            "delta_request_elapsed_seconds": item["delta"].get("request_elapsed_total_seconds"),
            "control_energy": control_metrics.get("energy_total_consumed"),
            "reference_energy": reference_metrics.get("energy_total_consumed"),
            "delta_energy": item["delta"].get("energy_total_consumed"),
            "control_raw_reward_excluding_penalty": control_metrics.get("raw_reward_excluding_rejection_penalty"),
            "reference_raw_reward_excluding_penalty": reference_metrics.get("raw_reward_excluding_rejection_penalty"),
            "delta_raw_reward_excluding_penalty": item["delta"].get("raw_reward_excluding_rejection_penalty"),
            "control_penalty": control_metrics.get("release_rejection_penalty"),
            "reference_penalty": reference_metrics.get("release_rejection_penalty"),
            "delta_penalty": item["delta"].get("release_rejection_penalty"),
            "technical_ok": checks.get("technical_ok"),
            "source_model_digest_match": checks.get("source_model_digest_match"),
            "caller_rng_unchanged": checks.get("caller_rng_unchanged"),
            "hard_safety_clear": checks.get("hard_safety_clear"),
        }
    )

true_counts = {
    key: sum(bool(row[key]) for row in table)
    for key in ("technical_ok", "source_model_digest_match", "caller_rng_unchanged", "hard_safety_clear")
}

trace = {
    "evidence_type": "preexisting_artifact",
    "full_artifact": {
        "path": str(GATE / "full.json"),
        "status": full_proof["full_status"],
        "sha256": full_proof["full_sha256"],
        "pair_count": full_proof["pair_count"],
        "PPO_ALLOWED": full_proof["PPO_ALLOWED"],
        "off_policy": full_proof["off_policy"],
    },
    "binding": {
        "preflight_path": str(GATE / "preflight.json"),
        "preflight_status": preflight["status"],
        "runner_sha256": preflight["runner_sha256"],
        "source_checkpoint": source["resolved_checkpoint"],
        "source_checkpoint_sha256": source["binding"]["checkpoint_sha256"],
        "warmup_checkpoint": source["warmup_checkpoint"],
        "warmup_checkpoint_sha256": source["warmup"]["checkpoint_sha256"],
        "source_model_digest": source["source_model_digest"],
        "observation_schema_version": config["observation_schema_version"],
        "action_schema_version": config["action_schema_version"],
        "runtime_proposal_shortlist_k": config["runtime_proposal_shortlist_k"],
        "student_decision_mask_mode": config["student_decision_mask_mode"],
        "max_simulation_seconds": config["max_simulation_seconds"],
    },
    "pair15_failure": {
        "pair_id": 15,
        "env_seed": 877104,
        "keyed_policy_seed": 9303103,
        "control": {
            "path": str(control_path),
            "sha256": sha(control_path),
            "status": control["status"],
            "censored": control["censored"],
            "error": control["error"],
            "ticks": control["ticks"],
            "simulation_seconds": control["simulation_seconds"],
            "control_metadata": control["control"],
            "mask_counts": control["mask_counts"],
            "terminal_row_arrival21": control_failure,
            "telemetry_selected": relevant_telemetry(control),
            "request_groups": request_group_summary(control),
            "phase_trace": {
                "terminal_phase_before_terminal": control_failure.get("phase_before_terminal"),
                "terminal_time": control_failure.get("terminal_time"),
                "first_relevant_tick": None,
                "last_relevant_tick": None,
                "per_step_phase_durations": None,
                "per_step_soc": None,
                "per_step_release_reason": None,
            },
        },
        "reference_legacy": {
            "path": str(reference_path),
            "sha256": sha(reference_path),
            "status": reference["status"],
            "censored": reference["censored"],
            "error": reference["error"],
            "ticks": reference["ticks"],
            "simulation_seconds": reference["simulation_seconds"],
            "terminal_row_arrival21": reference_same_arrival,
            "telemetry_selected": relevant_telemetry(reference),
            "request_groups": request_group_summary(reference),
        },
        "summary_delta": pair15["delta"],
    },
    "arrival13_env877104_four_continuations": arrival13,
    "all9_table": table,
    "all9_true_check_counts": true_counts,
    "trace_availability": {
        "per_step_action_decision_phase_soc_trajectory_in_arm_json": False,
        "request_groups_semantics": "arrival generation groups/signatures, not per-step action or decision trace",
        "available_aggregate_fields": [
            "arm.control.requests/forced_release/canonical_wait",
            "arm.telemetry.release_approved/release_rejected",
            "arm.telemetry.emergency_* counters",
            "arm.telemetry.runtime_shortlist_* counters",
            "arm.request_groups request signatures by slot",
            "arm.ledger.terminal_rows",
        ],
        "candidate_ids": None,
        "per_step_release_reasons": None,
        "first_last_relevant_ticks": None,
        "k3_binding_observed": config["runtime_proposal_shortlist_k"],
    },
    "exact_replay_artifact_schema_paths": {
        "gate_full_summary": str(GATE / "full.json"),
        "gate_pair_summary": str(GATE / "pairs" / "pair_0015.json"),
        "gate_control_arm": str(control_path),
        "legacy_reference_arm": str(reference_path),
        "gate_preflight_binding": str(GATE / "preflight.json"),
        "source_checkpoint": source["resolved_checkpoint"],
        "warmup_checkpoint": source["warmup_checkpoint"],
        "runner_interfaces": {
            "gate_control_request_wrapper": "terminal_marl/training/run_p3_gate_release_control_20260930.py:_control_keyed_request (lines 180-247)",
            "gate_control_arm_runner": "terminal_marl/training/run_p3_gate_release_control_20260930.py:_run_control_arm (lines 578-620)",
            "full_phase_reuse": "terminal_marl/training/run_p3_gate_release_control_20260930.py:run_full (lines 771-799)",
            "request_capture": "terminal_marl/training/paired_gate_diagnostic.py:_install_request_capture (lines 369-390)",
            "passive_branch_schema": "terminal_marl/training/paired_gate_diagnostic.py:_branch_outcome (lines 780-930)",
            "snapshot_hook_restore": "terminal_marl/training/paired_gate_diagnostic.py:restore_diagnostic_snapshot (lines 395-430)",
        },
        "raw_step_trace_in_current_arm_artifacts": "absent; the listed runner/helper interfaces are the existing schema locations for a separately captured replay",
    },
}
OUT_JSON.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "P3 gate-control failure trace (artifact-only)",
    f"full_status={trace['full_artifact']['status']} full_sha256={trace['full_artifact']['sha256']}",
    "",
    "pair15 control arrival21:",
    json.dumps(trace["pair15_failure"]["control"]["terminal_row_arrival21"], ensure_ascii=False, sort_keys=True),
    "pair15 legacy arrival21:",
    json.dumps(trace["pair15_failure"]["reference_legacy"]["terminal_row_arrival21"], ensure_ascii=False, sort_keys=True),
    "",
    "ALL9 table: pair env policy | control N/E/F/D/U | reference N/E/F/D/U | delta wait elapsed energy raw penalty",
]
for row in table:
    c = row["control_NEFDU"]
    r = row["reference_NEFDU"]
    lines.append(
        f"{row['pair_id']} {row['env_seed']} {row['keyed_policy_seed']} | "
        f"{c['normal']}/{c['emergency']}/{c['failed']}/{c['diverted']}/{c['unresolved']} | "
        f"{r['normal']}/{r['emergency']}/{r['failed']}/{r['diverted']}/{r['unresolved']} | "
        f"wait={row['delta_wait_seconds']} elapsed={row['delta_request_elapsed_seconds']} "
        f"energy={row['delta_energy']} raw={row['delta_raw_reward_excluding_penalty']} penalty={row['delta_penalty']}"
    )
lines += [
    "",
    f"all9 true checks: {json.dumps(true_counts, sort_keys=True)}",
    "arrival13 env877104 rows: " + str(len(arrival13)) + " (control/reference for pair 13,14,15,16)",
    "per-step action/decision/phase/SOC trace in current arm JSON: absent/unknown",
    "exact replay artifact paths and existing helper interfaces: see JSON exact_replay_artifact_schema_paths",
]
OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({
    "json": str(OUT_JSON),
    "text": str(OUT_TXT),
    "pair15_control_failure": {
        "arrival_id": control_failure.get("arrival_id"),
        "uid": control_failure.get("uid"),
        "outcome": control_failure.get("outcome"),
        "terminal_time": control_failure.get("terminal_time"),
    },
    "pair15_reference_arrival21": {
        "outcome": reference_same_arrival.get("outcome"),
        "terminal_time": reference_same_arrival.get("terminal_time"),
    },
    "all9": len(table),
    "arrival13_rows": len(arrival13),
    "true_counts": true_counts,
}, ensure_ascii=False, sort_keys=True))
