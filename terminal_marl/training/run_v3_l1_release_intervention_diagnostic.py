"""Release-intervention coverage diagnostic for the frozen V3 L1 pilot.

This is an evaluation-only diagnostic.  It freezes a deterministic, warm-start
cohort from the safety-qualified V3 L0 *raw* collection and evaluates a V3
categorical actor directly on the saved Observation V2 states.  It does not
step an environment, alter a policy, or create a new PPO update.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from ..algo.shared_ppo import SharedPPO
from ..core.checkpoints import load_checkpoint
from ..core.observations import LOW_OBS_V2
from .train_low_v2_ppo import l1_full_teacher_config


COHORT_KINDS = ("RELEASE_to_WAIT", "WAIT_to_RELEASE", "correct_RELEASE", "correct_WAIT")
COHORT_LIMITS = {"RELEASE_to_WAIT": 64, "WAIT_to_RELEASE": 32, "correct_RELEASE": 64, "correct_WAIT": 64}


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _kind(row: dict) -> str | None:
    requested, teacher = row["a_requested"]["release_index"], row["a_teacher"]["release_index"]
    if requested == 1 and teacher == 0:
        return "RELEASE_to_WAIT"
    if requested == 0 and teacher == 1:
        return "WAIT_to_RELEASE"
    if requested == 1 and teacher == 1:
        return "correct_RELEASE"
    if requested == 0 and teacher == 0:
        return "correct_WAIT"
    return None


def _reset_hash(row: dict) -> str:
    """Stable identity of the deterministic reset, separate from state hash."""
    payload = {"cohort": row["cohort"], "episode": row["episode"], "seed": row["seed"],
               "scenario": row["scenario"]}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _state_hash(row: dict) -> str:
    payload = {"reset_state_hash": _reset_hash(row), "step": row["step"],
               "observation_v2": row["observation_v2"]}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_release_cohort(raw_dataset: str | Path, output_path: str | Path) -> dict:
    """Select a fixed warm-start cohort without relabelling or balancing training."""
    groups = {name: [] for name in COHORT_KINDS}
    for row in _read_jsonl(raw_dataset):
        # This diagnostic deliberately stays inside the qualified warm-start
        # domain; no overload or altered traffic configuration is introduced.
        if not str(row.get("cohort", "")).startswith("warm_") or row.get("low_action_schema") != "V3":
            continue
        name = _kind(row)
        if name is not None:
            groups[name].append(row)
    selected: list[dict] = []
    for name in COHORT_KINDS:
        # A content hash, not collection order, makes selection deterministic.
        ordered = sorted(groups[name], key=lambda row: _state_hash(row))
        chosen = ordered[:COHORT_LIMITS[name]]
        if len(chosen) < COHORT_LIMITS[name]:
            raise RuntimeError(f"release cohort coverage insufficient for {name}: {len(chosen)}")
        for row in chosen:
            selected.append({
                "kind": name, "low_action_schema": row["low_action_schema"], "cohort": row["cohort"],
                "episode": row["episode"], "step": row["step"], "seed": row["seed"],
                "reset_state_hash": _reset_hash(row), "state_hash": _state_hash(row),
                "observation_v2": row["observation_v2"], "a_requested": row["a_requested"],
                "a_teacher": row["a_teacher"], "teacher_used": row["teacher_used"],
                "teacher_correction_heads": row["teacher_correction_heads"],
                "teacher_correction_reason": row["teacher_correction_reason"],
                "release_decision": row["release_decision"], "scenario": row["scenario"],
            })
    artifact = {
        "stage": "V3_L1_R_release_intervention_validation_cohort", "evidence_type": "preexisting_artifact",
        "source_dataset": str(raw_dataset), "selection": "warm_start V3 L0 raw rows; hash-ordered; labels unchanged",
        "limits": COHORT_LIMITS, "rows": selected,
        "composition": {name: sum(row["kind"] == name for row in selected) for name in COHORT_KINDS},
    }
    if artifact["composition"]["RELEASE_to_WAIT"] == 0 or artifact["composition"]["WAIT_to_RELEASE"] == 0:
        raise RuntimeError("release correction coverage gate failed")
    Path(output_path).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def _policy_from_initialization() -> SharedPPO:
    # Exact initialisation sequence from the frozen short-pilot runner.
    torch.manual_seed(2401)
    np.random.seed(2401)
    return SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05)


def _policy_from_checkpoint(path: str | Path) -> SharedPPO:
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05)
    config = l1_full_teacher_config(medium_rate=.001575, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                    warm_start_uavs=8, cohort_uavs=8, horizon_seconds=1200.0)
    load_checkpoint(path, policy, config)
    return policy


def _rate(mask: np.ndarray, values: np.ndarray) -> float | None:
    return None if not np.any(mask) else float(values[mask].mean())


def evaluate_release_cohort(policy: SharedPPO, cohort: dict) -> dict:
    rows = cohort["rows"]
    observations = torch.as_tensor(np.asarray([row["observation_v2"] for row in rows], dtype=np.float32), device=policy.device)
    targets = np.asarray([[row["a_teacher"]["release_index"], row["a_teacher"]["speed_index"], row["a_teacher"]["merge_time_index"]]
                          for row in rows], dtype=np.int64)
    kinds = np.asarray([row["kind"] for row in rows])
    with torch.no_grad():
        distribution = policy.model.actor_distribution(observations)
        logits = distribution.logits[0].cpu().numpy()
        probabilities = torch.softmax(distribution.logits[0], dim=-1).cpu().numpy()
        predicted = distribution.mode().cpu().numpy()
        entropy = distribution.entropy_per_head().cpu().numpy()
    release_target, release_predicted = targets[:, 0], predicted[:, 0]
    target_wait, target_release = release_target == 0, release_target == 1
    per_kind = {}
    for name in COHORT_KINDS:
        mask = kinds == name
        per_kind[name] = {
            "samples": int(mask.sum()), "release_accuracy": _rate(mask, (release_predicted == release_target).astype(float)),
            "predicted_RELEASE_rate": _rate(mask, (release_predicted == 1).astype(float)),
            "P_RELEASE": _rate(mask, probabilities[:, 1]), "release_entropy": _rate(mask, entropy[:, 0]),
            "release_logit_margin_RELEASE_minus_WAIT": _rate(mask, logits[:, 1] - logits[:, 0]),
        }
    return {
        "samples": len(rows), "release_accuracy": float(np.mean(release_predicted == release_target)),
        "false_release_rate": _rate(target_wait, (release_predicted == 1).astype(float)),
        "false_wait_rate": _rate(target_release, (release_predicted == 0).astype(float)),
        "P_RELEASE_given_teacher_WAIT": _rate(target_wait, probabilities[:, 1]),
        "P_RELEASE_given_teacher_RELEASE": _rate(target_release, probabilities[:, 1]),
        "release_entropy": float(entropy[:, 0].mean()),
        "release_logit_margin_RELEASE_minus_WAIT": float((logits[:, 1] - logits[:, 0]).mean()),
        # This is an offline teacher-disagreement proxy, never an executed intervention count.
        "offline_teacher_disagreement_rate": float(np.mean(np.any(predicted != targets, axis=1))),
        "speed_target_mismatch_rate": float(np.mean(predicted[:, 1] != targets[:, 1])),
        "merge_target_mismatch_rate": float(np.mean(predicted[:, 2] != targets[:, 2])),
        "per_kind": per_kind,
    }


def _pilot_validation_metrics(pilot: dict, label: str) -> dict | None:
    for record in pilot.get("validation", []):
        if record.get("label") == label:
            overall = record["metrics"]["overall"]
            warm = record["metrics"].get("warm_start", overall)
            return {"overall_teacher_burden": overall["environment"]["teacher_interventions_per_100_release_intents"],
                    "warm_teacher_burden": warm["environment"]["teacher_interventions_per_100_release_intents"],
                    "shield_burden": overall["environment"]["shield_interventions_per_100_release_intents"],
                    "completion": overall["environment"]["completed"], "throughput": overall["environment"]["throughput_per_second"],
                    "safety": {key: overall["environment"][key] for key in ("collisions", "nmac", "hard_separation_violations")}}
    return None


def _training_coverage(pilot: dict) -> dict:
    updates = []
    for row in pilot["updates"]:
        release = row["distribution"]["heads"]["release"]
        updates.append({"update": row["update"] + 1, "release_teacher_correction_count": int(round(
            row["distribution"]["transitions"] * release["teacher_correction_rate"])),
            "RELEASE_to_WAIT": release["RELEASE_to_WAIT"], "WAIT_to_RELEASE": release["WAIT_to_RELEASE"],
            "speed_teacher_correction_rate": row["distribution"]["heads"]["speed"]["teacher_correction_rate"],
            "merge_teacher_correction_rate": row["distribution"]["heads"]["merge_time"]["teacher_correction_rate"]})
    return {"per_update": updates, "total_release_teacher_corrections": sum(row["release_teacher_correction_count"] for row in updates),
            "total_RELEASE_to_WAIT": sum(row["RELEASE_to_WAIT"] for row in updates),
            "total_WAIT_to_RELEASE": sum(row["WAIT_to_RELEASE"] for row in updates)}


def _ppo_health(pilot: dict) -> dict:
    return {"passed": bool(pilot.get("ppo_health_passed", False)), "per_update": [
        {"update": row["update"] + 1, "approx_kl": row["ppo"]["approx_kl"],
         "clip_fraction": row["ppo"]["clip_fraction"], "entropy": row["ppo"]["entropy"],
         "entropy_release": row["ppo"]["entropy_release"], "grad_norm": row["ppo"]["grad_norm"]}
        for row in pilot["updates"]]}


def _release_trend(evaluation: dict) -> dict:
    """Use the decision boundary, not a small common-prior shift, as evidence."""
    baseline = evaluation["baseline"]["release_cohort"]
    final = evaluation["update6"]["release_cohort"]
    return {
        "RELEASE_to_WAIT": "NO_DEMONSTRABLE_IMPROVEMENT" if
            final["per_kind"]["RELEASE_to_WAIT"]["predicted_RELEASE_rate"] >=
            baseline["per_kind"]["RELEASE_to_WAIT"]["predicted_RELEASE_rate"] else "IMPROVING",
        "WAIT_to_RELEASE": "NO_DEMONSTRABLE_CHANGE" if
            final["per_kind"]["WAIT_to_RELEASE"]["predicted_RELEASE_rate"] ==
            baseline["per_kind"]["WAIT_to_RELEASE"]["predicted_RELEASE_rate"] else "CHANGED",
        "conditional_calibration_gap_P_RELEASE_teacher_RELEASE_minus_WAIT": {
            "baseline": baseline["P_RELEASE_given_teacher_RELEASE"] - baseline["P_RELEASE_given_teacher_WAIT"],
            "update6": final["P_RELEASE_given_teacher_RELEASE"] - final["P_RELEASE_given_teacher_WAIT"],
        },
    }


def _markdown(report: dict) -> str:
    lines = ["# V3 L1 Release Intervention Coverage + Learning Diagnostic", "",
             "Evidence types: fixed cohort = preexisting V3 L0 safety-qualified artifact; pilot rollout/health = newly-run pilot artifact.", "",
             "## Release validation cohort", "",
             f"- Composition: {report['cohort']['composition']}",
             f"- Frozen reset identities: {report['cohort']['unique_reset_state_hashes']}; immutable state hashes: {report['cohort']['unique_state_hashes']}",
             "- Labels retain the V3 L0 TeacherScheduler/Safety qualification; no traffic difficulty, teacher policy, reward, or actor objective was changed.", "",
             "## Checkpoint evaluation", ""]
    for label, entry in report["checkpoint_evaluation"].items():
        lines += [f"### {label}", ""]
        if entry["availability"] != "AVAILABLE":
            lines.append(f"- {entry['availability']}: {entry['reason']}")
        else:
            metric = entry["release_cohort"]
            lines += [f"- Release accuracy: {metric['release_accuracy']:.4f}; false-release / false-wait: {metric['false_release_rate']:.4f} / {metric['false_wait_rate']:.4f}",
                      f"- P(RELEASE | Teacher WAIT/RELEASE): {metric['P_RELEASE_given_teacher_WAIT']:.4f} / {metric['P_RELEASE_given_teacher_RELEASE']:.4f}",
                      f"- Release entropy / logit margin: {metric['release_entropy']:.4f} / {metric['release_logit_margin_RELEASE_minus_WAIT']:.4f}",
                      f"- RELEASE→WAIT predicted-release rate: {metric['per_kind']['RELEASE_to_WAIT']['predicted_RELEASE_rate']:.4f}; WAIT→RELEASE predicted-release rate: {metric['per_kind']['WAIT_to_RELEASE']['predicted_RELEASE_rate']:.4f}"]
        if entry.get("pilot_dynamic"):
            lines.append(f"- Existing pilot overall/warm teacher burden: {entry['pilot_dynamic']['overall_teacher_burden']} / {entry['pilot_dynamic']['warm_teacher_burden']}; shield: {entry['pilot_dynamic']['shield_burden']}")
        lines.append("")
    coverage = report["training_rollout_release_coverage"]
    lines += ["## Training rollout release supervision", "",
              f"- Total release correction / RELEASE→WAIT / WAIT→RELEASE: {coverage['total_release_teacher_corrections']} / {coverage['total_RELEASE_to_WAIT']} / {coverage['total_WAIT_to_RELEASE']}",
              f"- Per update: {coverage['per_update']}", "", "## Decision", "",
              f"- RELEASE→WAIT trend: {report['release_learning_trend']['RELEASE_to_WAIT']}; WAIT→RELEASE trend: {report['release_learning_trend']['WAIT_to_RELEASE']}",
              f"- Conditional calibration gap P(RELEASE|Teacher RELEASE) − P(RELEASE|Teacher WAIT), baseline/update6: {report['release_learning_trend']['conditional_calibration_gap_P_RELEASE_teacher_RELEASE_minus_WAIT']}",
              f"- PPO health: {'PASS' if report['ppo_health']['passed'] else 'FAIL'}; per update KL/clip: {[(item['approx_kl'], item['clip_fraction']) for item in report['ppo_health']['per_update']]}",
              f"release_learning = {report['release_learning']}", f"root_cause = {report['root_cause']}",
              f"longer_L1_continuation_allowed = {str(report['longer_L1_continuation_allowed']).lower()}",
              "formal_L1_training_allowed = false", "", f"Safety = {'PASS' if report['safety_passed'] else 'FAIL'}", ""]
    return "\n".join(lines)


def run_release_intervention_diagnostic(*, raw_dataset: str | Path, pilot_report: str | Path,
                                        final_checkpoint: str | Path, output_dir: str | Path,
                                        update3_checkpoint: str | Path | None = None) -> dict:
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    cohort = freeze_release_cohort(raw_dataset, destination / "v3_l1_release_intervention_cohort.json")
    cohort_summary = {"composition": cohort["composition"], "unique_reset_state_hashes": len({row["reset_state_hash"] for row in cohort["rows"]}),
                      "unique_state_hashes": len({row["state_hash"] for row in cohort["rows"]}), "path": str(destination / "v3_l1_release_intervention_cohort.json")}
    pilot = json.loads(Path(pilot_report).read_text(encoding="utf-8"))
    coverage = _training_coverage(pilot)
    evaluation = {
        "baseline": {"availability": "AVAILABLE", "release_cohort": evaluate_release_cohort(_policy_from_initialization(), cohort),
                     "pilot_dynamic": _pilot_validation_metrics(pilot, "baseline")},
        "update3": {"availability": "UNAVAILABLE", "reason": "The frozen pilot retained no update-3 model snapshot; metrics cannot be reconstructed without rerunning PPO, which this evaluation-only diagnostic forbids.",
                    "pilot_dynamic": _pilot_validation_metrics(pilot, "update3")},
        "update6": {"availability": "AVAILABLE", "release_cohort": evaluate_release_cohort(_policy_from_checkpoint(final_checkpoint), cohort),
                    "pilot_dynamic": _pilot_validation_metrics(pilot, "update6")},
    }
    if update3_checkpoint is not None:
        path = Path(update3_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        evaluation["update3"] = {"availability": "AVAILABLE", "release_cohort": evaluate_release_cohort(_policy_from_checkpoint(path), cohort),
                                 "pilot_dynamic": _pilot_validation_metrics(pilot, "update3")}
    # Case C has priority: without any release correction, an actor trend on
    # the intervention cohort cannot establish (or falsify) learning.
    no_release_supervision = coverage["total_release_teacher_corrections"] == 0
    root_cause = "TRAINING_COVERAGE_INSUFFICIENT" if no_release_supervision else "OTHER"
    release_trend = _release_trend(evaluation)
    report = {"stage": "V3_L1_R_release_intervention_coverage_learning_diagnostic", "evidence_type": "mixed",
              "frozen": {"action_schema": "V3_factorized_categorical", "teacher": "FULL", "reward_changed": False,
                         "teacher_bc_weight": .05, "mu_hat": .003, "lambda_training": .001575},
              "cohort": cohort_summary, "checkpoint_evaluation": evaluation, "training_rollout_release_coverage": coverage,
              "release_learning_trend": release_trend, "ppo_health": _ppo_health(pilot),
              "release_learning": "INCONCLUSIVE" if no_release_supervision else "INCONCLUSIVE", "root_cause": root_cause,
              "safety_passed": bool(pilot.get("safety_passed", False)), "ppo_health_passed": bool(pilot.get("ppo_health_passed", False)),
              "longer_L1_continuation_allowed": False, "formal_L1_training_allowed": False,
              "next_diagnostic": "Modify only the L1 training scenario mixture to include qualified warm-start release-WAIT states, then rerun this fixed cohort; retain lambda, geometry, safety, curriculum domain, reward, BC weight, and PPO hyperparameters."}
    (destination / "v3_l1_release_intervention_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (destination / "v3_l1_release_intervention_diagnostic.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation-only V3 L1 release-intervention diagnostic.")
    parser.add_argument("--raw-dataset", required=True); parser.add_argument("--pilot-report", required=True)
    parser.add_argument("--final-checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--update3-checkpoint")
    args = parser.parse_args()
    report = run_release_intervention_diagnostic(raw_dataset=args.raw_dataset, pilot_report=args.pilot_report,
        final_checkpoint=args.final_checkpoint, output_dir=args.output, update3_checkpoint=args.update3_checkpoint)
    print(json.dumps({key: report[key] for key in ("release_learning", "root_cause", "safety_passed", "longer_L1_continuation_allowed")}, indent=2))


if __name__ == "__main__":
    _main()
