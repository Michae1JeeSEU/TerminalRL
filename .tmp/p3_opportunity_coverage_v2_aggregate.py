from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


ROOT = Path("D:/7_codespace/training_outputs/terminal/p3_followup_20260930/opportunity_coverage_v2")
OUT_JSON = Path(".tmp/p3_opportunity_coverage_v2_evidence.json")
OUT_TXT = Path(".tmp/p3_opportunity_coverage_v2_evidence.txt")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lite_state(value):
    if not isinstance(value, dict):
        return None
    keys = (
        "active", "arrival_id", "uid", "phase", "ring_role", "interface",
        "committed", "soc", "energy_consumed", "energy_remaining",
        "external_wait_time", "internal_wait_time", "safety_retry_after",
        "emergency_retry_after",
    )
    return {key: value.get(key) for key in keys if key in value}


def state_summary(path: Path) -> dict:
    data = read(path)
    shadow = data["shadow"]
    branches = shadow["branches"]
    proposal_branches = branches[1:]
    approved = [
        {
            "proposal": branch.get("proposal"),
            "q": branch.get("proposal_probability"),
            "decision": branch.get("decision"),
            "native_blocker_reasons": branch.get("native_blocker_reasons", []),
        }
        for branch in proposal_branches
        if branch.get("approved") is True
    ]
    decision_counts = collections.Counter(
        branch.get("decision") for branch in branches
    )
    status_counts = collections.Counter(branch.get("status") for branch in branches)
    error_count = sum(branch.get("error") is not None for branch in branches)
    def is_truncated(value) -> bool:
        if isinstance(value, dict):
            return any(bool(item) for item in value.values())
        return bool(value)

    truncated_count = sum(is_truncated(branch.get("truncated")) for branch in branches)
    chosen = data.get("selection", {}).get("selection")
    return {
        "file": str(path),
        "pair_id": int(path.stem.split("_")[1]),
        "window_index": data.get("window_index"),
        "window": data.get("window"),
        "absolute_tick": data.get("absolute_tick"),
        "target_arrival_id": data.get("target_arrival_id"),
        "target_uid": data.get("target_uid"),
        "selected_candidate": chosen,
        "eligible_candidates": data.get("selection", {}).get("eligible", []),
        "target_before": lite_state(branches[0].get("before_target")),
        "branch_count": shadow.get("branch_count"),
        "branch_status_counts": {
            str(key): value for key, value in shadow.get("branch_status_counts", {}).items()
        },
        "branch_error_count": error_count,
        "branch_truncated_count": truncated_count,
        "decision_counts": {str(key): value for key, value in decision_counts.items()},
        "C": shadow.get("C"),
        "pC": shadow.get("pC"),
        "p_release": shadow.get("p_release"),
        "q_max": shadow.get("q_max"),
        "proposal_q": [branch.get("proposal_probability") for branch in proposal_branches],
        "proposal_q_sum": sum(
            float(branch["proposal_probability"])
            for branch in proposal_branches
            if branch.get("proposal_probability") is not None
        ),
        "approved_proposals": approved,
        "full_cycle_value_present": shadow.get("contains_full_cycle_value"),
        "m_present": data.get("m_present"),
        "live_environment_unchanged": shadow.get("live_environment_unchanged"),
    }


def trace_event(event: dict) -> dict:
    target = event.get("target") or {}
    after = event.get("actual_target_after") or {}
    return {
        "tick": event.get("tick"),
        "now": event.get("now"),
        "actual_decision": event.get("actual_decision"),
        "actual_error": event.get("actual_error"),
        "target_uid": event.get("target_uid"),
        "phase": target.get("phase"),
        "soc": target.get("soc"),
        "energy_remaining": target.get("energy_remaining"),
        "p_release": event.get("p_release"),
        "q_max": event.get("q_max"),
        "target_after_active": after.get("active"),
        "target_after_arrival_id": after.get("arrival_id"),
    }


def target13_trace_summary(arm: dict) -> dict:
    trace = arm.get("opportunity_diagnostic", {}).get("target13_trace", [])
    decision_counts = collections.Counter(event.get("actual_decision") for event in trace)
    phase_counts = collections.Counter(
        (event.get("target") or {}).get("phase")
        for event in trace
        if (event.get("target") or {}).get("active") is True
    )
    normal_events = [
        event for event in trace
        if event.get("actual_decision") in {"safety_reject", "approved"}
    ]
    approved_events = [
        event for event in trace if event.get("actual_decision") == "approved"
    ]
    return {
        "trace_length": len(trace),
        "trace_first_tick": trace[0].get("tick") if trace else None,
        "trace_last_tick": trace[-1].get("tick") if trace else None,
        "actual_decision_counts": {str(key): value for key, value in decision_counts.items()},
        "active_phase_counts": {str(key): value for key, value in phase_counts.items()},
        "active_trace_count": sum(
            (event.get("target") or {}).get("active") is True for event in trace
        ),
        "normal_release_event_count": len(normal_events),
        "first_normal_release_event": trace_event(normal_events[0]) if normal_events else None,
        "last_normal_release_event": trace_event(normal_events[-1]) if normal_events else None,
        "approved_event_count": len(approved_events),
        "approved_events": [trace_event(event) for event in approved_events],
        "trace_source": "opportunity_diagnostic.target13_trace",
    }


def arm_summary(arm: dict, pair: dict) -> dict:
    ledger = arm.get("ledger", {})
    energy = arm.get("energy_ledger", {})
    energy_rows = energy.get("rows", {})
    energy_total = sum(float(row.get("energy_consumed", 0.0)) for row in energy_rows.values())
    wait = arm.get("wait_metrics", {})
    config = pair.get("config_snapshot", {})
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "status": arm.get("status"),
        "censored": arm.get("censored"),
        "error": arm.get("error"),
        "env_seed": arm.get("env_seed"),
        "keyed_policy_seed": arm.get("keyed_policy_seed"),
        "ticks": arm.get("ticks"),
        "simulation_seconds": arm.get("simulation_seconds"),
        "model_digest_before": arm.get("model_digest_before"),
        "model_digest_after": arm.get("model_digest_after"),
        "model_unchanged": arm.get("model_unchanged"),
        "caller_rng_unchanged": arm.get("caller_rng_unchanged"),
        "request_identity_count": arm.get("request_identity_count"),
        "request_identity_unique": arm.get("request_identity_unique"),
        "request_capture_integrity": arm.get("request_capture_integrity"),
        "generated_process_count": arm.get("generated_process_count"),
        "admitted_count": arm.get("admitted_count"),
        "completed_count": arm.get("completed_count"),
        "resolved_count": arm.get("resolved_count"),
        "outcome_counts": arm.get("outcome_counts"),
        "safety": arm.get("safety"),
        "ledger_conservation": ledger.get("conservation"),
        "ledger_terminal_alignment": ledger.get("terminal_alignment"),
        "energy_coverage": energy.get("coverage"),
        "energy_total_consumed": energy_total,
        "wait_integrity": arm.get("wait_integrity"),
        "wait_metrics": {
            "all_request_cumulative_wait_seconds": wait.get("all_request_cumulative_wait_seconds"),
            "external_queue_integral_seconds": wait.get("external_queue_integral_seconds"),
            "internal_holding_sum_seconds": wait.get("internal_holding_sum_seconds"),
            "missing_internal_wait_ids": wait.get("missing_internal_wait_ids"),
            "time_scope": wait.get("time_scope"),
        },
        "reward_total": arm.get("reward_total"),
        "raw_reward_excluding_rejection_penalty": arm.get("raw_reward_excluding_rejection_penalty"),
        "release_rejection_penalty": arm.get("release_rejection_penalty"),
        "reference_exact": pair.get("reference_exact"),
        "config_snapshot_sha256": config_hash,
        "canonical_source_model_digest": pair.get("canonical_source_model_digest"),
        "canonical_source_model_digest_algorithm": pair.get("canonical_source_model_digest_algorithm"),
        "target13_trace": target13_trace_summary(arm),
    }


def main() -> None:
    full = read(ROOT / "full.json")
    preflight = read(ROOT / "preflight.json")
    pairs = []
    states = []
    env877104_arrival13_rows = []
    for summary in full["pair_summaries"]:
        pair_id = int(summary["pair"]["pair_id"])
        arm_path = ROOT / "arms" / f"pair_{pair_id:04d}_legacy.json"
        arm = read(arm_path)
        if summary["pair"]["env_seed"] == 877104:
            terminal_rows = arm.get("terminal_summary", {}).get("rows", [])
            env877104_arrival13_rows.append({
                "pair_id": summary["pair"]["pair_id"],
                "env_seed": summary["pair"]["env_seed"],
                "keyed_policy_seed": summary["pair"]["keyed_policy_seed"],
                "row": next(row for row in terminal_rows if row.get("arrival_id") == 13),
            })
        pairs.append({
            "pair": summary["pair"],
            "status": summary.get("status"),
            "selected_state_count": summary.get("selected_state_count"),
            "missing_window_indices": summary.get("missing_window_indices"),
            "source_model_digest": summary.get("source_model_digest"),
            "canonical_source_model_digest": summary.get("canonical_source_model_digest"),
            "reference_exact": summary.get("reference_exact"),
            "arm": arm_summary(arm, summary),
        })
        for state_path in sorted((ROOT / "states").glob(f"pair_{pair_id:04d}_window_*.json")):
            states.append(state_summary(state_path))

    config_hashes = sorted({item["arm"]["config_snapshot_sha256"] for item in pairs})
    paired_digests = sorted({item["source_model_digest"] for item in pairs})
    canonical_digests = sorted({item["canonical_source_model_digest"] for item in pairs})
    all_reference_exact = all(item["reference_exact"].get("exact") is True for item in pairs)
    all_arm_ok = all(
        item["arm"]["status"] == "completed"
        and item["arm"]["censored"] is False
        and item["arm"]["error"] is None
        for item in pairs
    )
    evidence = {
        "artifact": {
            "root": str(ROOT),
            "full_json": str(ROOT / "full.json"),
            "preflight_json": str(ROOT / "preflight.json"),
            "full_json_sha256": sha256(ROOT / "full.json"),
            "preflight_json_sha256": sha256(ROOT / "preflight.json"),
        },
        "full": {
            "status": full.get("status"),
            "pair_count": full.get("pair_count"),
            "selected_state_count": full.get("selected_state_count"),
            "max_selected_states": full.get("max_selected_states"),
            "PPO_ALLOWED": full.get("PPO_ALLOWED"),
            "preflight_reference_sha256": full.get("preflight_reference_sha256"),
            "qualification": full.get("qualification"),
        },
        "preflight": {
            "status": preflight.get("status"),
            "runner_sha256": preflight.get("runner_sha256"),
            "PPO_ALLOWED": preflight.get("PPO_ALLOWED"),
            "heldout_used": preflight.get("heldout_used"),
            "optimizer_used": preflight.get("optimizer_used"),
            "teacher_used": preflight.get("teacher_used"),
            "protocol": preflight.get("protocol"),
            "source_binding": preflight.get("source_binding"),
            "runtime_versions": preflight.get("runtime", {}).get("versions"),
            "runtime_manifest_count": preflight.get("runtime", {}).get("existing_code_manifest", {}).get("count"),
        },
        "consistency_checks": {
            "all_pair_status_pair_ok": all(item["status"] == "PAIR_OK" for item in pairs),
            "all_arm_completed_uncensored_error_null": all_arm_ok,
            "all_reference_exact": all_reference_exact,
            "config_snapshot_hashes": config_hashes,
            "all_config_snapshots_equal": len(config_hashes) == 1,
            "paired_execution_digests": paired_digests,
            "all_paired_execution_digests_equal": len(paired_digests) == 1,
            "canonical_source_digests": canonical_digests,
            "all_canonical_source_digests_equal": len(canonical_digests) == 1,
            "all_model_unchanged": all(item["arm"]["model_unchanged"] is True for item in pairs),
            "all_caller_rng_unchanged": all(item["arm"]["caller_rng_unchanged"] is True for item in pairs),
            "all_safety_violation_false": all(not item["arm"]["safety"].get("safety_violation") for item in pairs),
            "all_selected_branches_ok": all(
                state["branch_error_count"] == 0
                and state["branch_truncated_count"] == 0
                and state["branch_status_counts"].get("ok") == state["branch_count"]
                for state in states
            ),
        },
        "pairs": pairs,
        "states": states,
        "env877104_arrival13_terminal_rows": env877104_arrival13_rows,
        "trace_scope_note": (
            "target13_trace is the runner-emitted per-tick trace under each arm's "
            "opportunity_diagnostic; no raw branch files were read."
        ),
    }
    OUT_JSON.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"status={evidence['full']['status']} pair_count={evidence['full']['pair_count']} "
        f"selected_states={evidence['full']['selected_state_count']}/{evidence['full']['max_selected_states']}",
        f"all_pair_ok={evidence['consistency_checks']['all_pair_status_pair_ok']} "
        f"all_reference_exact={evidence['consistency_checks']['all_reference_exact']} "
        f"all_selected_branches_ok={evidence['consistency_checks']['all_selected_branches_ok']}",
        "pair_id env keyed states missing status C_values approved_count",
    ]
    for item in pairs:
        pair_id = item["pair"]["pair_id"]
        pair_states = [state for state in states if state["pair_id"] == pair_id]
        lines.append(
            f"{pair_id} {item['pair']['env_seed']} {item['pair']['keyed_policy_seed']} "
            f"{len(pair_states)} {item['missing_window_indices']} {item['status']} "
            f"{[state['C'] for state in pair_states]} "
            f"{sum(len(state['approved_proposals']) for state in pair_states)}"
        )
    lines.append("env877104 arrival13 terminal rows: " + ",".join(
        f"pair{row['pair_id']}/keyed{row['keyed_policy_seed']}/uid{row['row'].get('uid')}/"
        f"{row['row'].get('outcome')}@{row['row'].get('terminal_time')}"
        for row in evidence["env877104_arrival13_terminal_rows"]
    ))
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "json": str(OUT_JSON),
        "text": str(OUT_TXT),
        "pairs": len(pairs),
        "states": len(states),
        "status": full.get("status"),
        "all_pair_ok": evidence["consistency_checks"]["all_pair_status_pair_ok"],
        "all_reference_exact": evidence["consistency_checks"]["all_reference_exact"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
