import hashlib
import json
from pathlib import Path

OUT = Path(r"D:\7_codespace\training_outputs\terminal\p3_followup_20260930\gate_release_control")
REF = Path(r"D:\7_codespace\training_outputs\terminal\p3_followup_20260930\forced_wait_frozen")
PROOF = Path(r"D:\7_codespace\terminal\.tmp\p3_gate_control_full_proof.json")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_sha(rows):
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def metrics(summary, arm_key):
    return summary[f"{arm_key}_metrics"]


def arm_snapshot(arm, summary_metrics, summary_checks):
    coverage = arm["energy_ledger"]["coverage"]
    return {
        "status": arm["status"],
        "censored": arm["censored"],
        "error": arm["error"],
        "stop_reason": arm["stop_reason"],
        "ticks": arm["ticks"],
        "simulation_seconds": arm["simulation_seconds"],
        "budget_cap_is_not_natural_terminal": arm["budget_cap_is_not_natural_terminal"],
        "metrics": summary_metrics,
        "outcome_counts": arm["outcome_counts"],
        "reward_total": arm["reward_total"],
        "raw_reward_excluding_rejection_penalty": arm[
            "raw_reward_excluding_rejection_penalty"
        ],
        "release_rejection_penalty": arm["release_rejection_penalty"],
        "request_identity_count": arm["request_identity_count"],
        "request_identity_unique": arm["request_identity_unique"],
        "request_identity_sha256": identity_sha(arm["request_identity"]),
        "request_capture_integrity": arm["request_capture_integrity"],
        "energy_coverage": coverage,
        "energy_unique_arrival_ids": arm["energy_ledger"]["unique_arrival_ids"],
        "wait_integrity": arm["wait_integrity"],
        "wait_metrics": arm["wait_metrics"],
        "ledger_conservation": arm["ledger"]["conservation"],
        "ledger_terminal_alignment": arm["ledger"]["terminal_alignment"],
        "model_digest_before": arm["model_digest_before"],
        "model_digest_after": arm["model_digest_after"],
        "model_unchanged": arm["model_unchanged"],
        "caller_rng_unchanged": arm["caller_rng_unchanged"],
        "safety": arm["safety"],
        "checks": arm.get("reference_control_checks", summary_checks),
        "failed_terminal_rows": [
            row for row in arm["ledger"]["terminal_rows"] if row.get("outcome") == "failed_energy"
        ],
    }


full_path = OUT / "full.json"
full = load(full_path)
preflight = load(OUT / "preflight.json")
source = preflight["runtime"]["source_binding"]
config = preflight["runtime"]["config_snapshot"]

pairs = []
arrival13 = []
for summary in full["pair_summaries"]:
    pair_id = int(summary["pair"]["pair_id"])
    control_path = OUT / "arms" / f"pair_{pair_id:04d}_control.json"
    reference_path = REF / "arms" / f"pair_{pair_id:04d}_legacy.json"
    control = load(control_path)
    reference = load(reference_path)
    control_snapshot = arm_snapshot(
        control, metrics(summary, "control"), summary["reference_control_checks"]
    )
    reference_snapshot = arm_snapshot(
        reference, metrics(summary, "reference"), summary["reference_control_checks"]
    )
    for label, arm in (("control", control), ("reference", reference)):
        if int(summary["pair"]["env_seed"]) == 877104:
            for row in arm["ledger"]["terminal_rows"]:
                if int(row["arrival_id"]) == 13:
                    arrival13.append(
                        {
                            "pair_id": pair_id,
                            "arm": label,
                            "env_seed": summary["pair"]["env_seed"],
                            "keyed_policy_seed": summary["pair"]["keyed_policy_seed"],
                            "terminal_row": row,
                        }
                    )
    pairs.append(
        {
            "pair": summary["pair"],
            "status": summary["status"],
            "training": summary["training"],
            "off_policy": summary["off_policy"],
            "control": control_snapshot,
            "reference": reference_snapshot,
            "delta": summary["complete_scene_delta"],
            "reference_control_checks": summary["reference_control_checks"],
            "new_failed_arrival_ids": summary["new_failed_arrival_ids"],
            "new_diverted_arrival_ids": summary["new_diverted_arrival_ids"],
            "new_unresolved_arrival_ids": summary["new_unresolved_arrival_ids"],
            "failure_comparison_known": summary["failure_comparison_known"],
            "control_arm_sha256": sha(control_path),
            "reference_arm_sha256": sha(reference_path),
        }
    )

proof = {
    "source": "newly_run_server_full_artifact_synced_local",
    "full_status": full["status"],
    "full_phase": full["phase"],
    "full_sha256": sha(full_path),
    "pair_count": full["pair_count"],
    "PPO_ALLOWED": full["PPO_ALLOWED"],
    "off_policy": full["off_policy"],
    "qualification": full["qualification"],
    "business_review": full["business_review"],
    "preflight_binding": {
        "status": preflight["status"],
        "runner_sha256": preflight["runner_sha256"],
        "source_checkpoint_sha256": source["binding"]["checkpoint_sha256"],
        "warmup_checkpoint_sha256": source["warmup"]["checkpoint_sha256"],
        "source_model_digest": source["source_model_digest"],
        "pair_ids": preflight["control"]["pair_ids"],
        "proposal_sampling": preflight["control"]["proposal_sampling"],
        "shortlist": preflight["control"]["shortlist"],
        "max_steps": preflight["control"]["max_steps"],
        "runtime": {
            key: config[key]
            for key in (
                "runtime_proposal_shortlist_k",
                "student_decision_mask_mode",
                "max_simulation_seconds",
                "observation_schema_version",
                "action_schema_version",
                "teacher_enabled",
            )
        },
    },
    "pairs": pairs,
    "arrival13_env877104": arrival13,
    "artifacts": {
        "full": str(full_path),
        "preflight": str(OUT / "preflight.json"),
        "control_arms": [str(OUT / "arms" / f"pair_{int(s['pair']['pair_id']):04d}_control.json") for s in full["pair_summaries"]],
        "reference_arms": [str(REF / "arms" / f"pair_{int(s['pair']['pair_id']):04d}_legacy.json") for s in full["pair_summaries"]],
    },
}
PROOF.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "proof": str(PROOF),
    "full_status": proof["full_status"],
    "full_sha256": proof["full_sha256"],
    "pair_count": proof["pair_count"],
    "arrival13_rows": len(arrival13),
    "control_failed_pairs": [p["pair"]["pair_id"] for p in pairs if p["control"]["metrics"]["failed"]],
    "reference_failed_pairs": [p["pair"]["pair_id"] for p in pairs if p["reference"]["metrics"]["failed"]],
}, ensure_ascii=False, sort_keys=True))
