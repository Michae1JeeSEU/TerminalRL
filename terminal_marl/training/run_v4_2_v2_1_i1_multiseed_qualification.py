"""Fresh multi-seed V4.2/V2.1 I1 training and one-shot Q qualification.

The program deliberately has two irreversible phases.  During TRAINING an
audit hook rejects every attempt to open either immutable Q artifact.  Only
after all three final-epoch checkpoints and training logs have been written
and hashed does the process enter Q_EVALUATION and read the cohort once for a
single, conjunctive qualification decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..algo.joint_ppo_v4_2 import JointPPOV42, hierarchical_bc_loss
from ..core.checkpoints import load_checkpoint, save_checkpoint
from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, RELEASE_GATE
from .run_v4_2_hierarchical_s0_qualification import _config as _base_config
from .run_v4_2_v2_1_i1_bc import (
    OBSERVATION_DIM,
    OBSERVATION_SCHEMA,
    _architecture_fingerprint,
    _arrays,
    _family_metrics,
    _metrics,
    _retry_metrics,
    _rows,
    _sha256,
)


EXPECTED_TRAIN_SHA256 = "6bcbc1782a171f909282126c4f6d27dd93355b09513e2827093228af4fe05f05"
EXPECTED_I0_MANIFEST_SHA256 = "51868f7892395d7d5a369d956f6c374ba536d3c8023ec5d0f42a2d9f208f8a31"
EXPECTED_Q_SHA256 = "8d1e80bbf7efbf8bdada1814efc3c435daa80dd00564935c2e9dd3ba466dffe5"
EXPECTED_Q_MANIFEST_SHA256 = "a24ef05fb78093a81945f28d91bb16907ebf443c127f2158651613ce401c40f2"
EXPECTED_PROTOCOL_SHA256 = "6da3ff8f3c790428c6e16d093a63c4e09c1db44e5ff2676e4abfb95fa3d47d11"
EXPECTED_TRAIN_ROWS = 2020
EXPECTED_Q_ROWS = 1124
SEEDS = (4202, 4203, 4204)
CANONICAL_SEED = 4202
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STABLE_JOINT_SUPPORT = (11, 12, 13, 14)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _install_q_access_guard(q_paths: tuple[Path, ...], state: dict[str, Any]) -> None:
    blocked = {os.path.normcase(os.path.abspath(str(path))) for path in q_paths}

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or state["phase"] != "TRAINING" or not args:
            return
        candidate = args[0]
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        resolved = os.path.normcase(os.path.abspath(os.fsdecode(candidate)))
        if resolved in blocked:
            state["q_access_during_training"] = True
            raise RuntimeError("Q cohort access attempted during training; qualification invalid")

    sys.addaudithook(audit)


def _validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("qualification protocol SHA256 mismatch")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    expected = {
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_dim": OBSERVATION_DIM,
        "action_schema": ACTION_SCHEMA_V4_2,
        "training_dataset_sha256": EXPECTED_TRAIN_SHA256,
        "training_rows": EXPECTED_TRAIN_ROWS,
        "fresh_actor_seeds": list(SEEDS),
        "canonical_checkpoint_seed": CANONICAL_SEED,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "checkpoint_selection": "final_epoch",
    }
    mismatch = {key: (protocol.get(key), value) for key, value in expected.items() if protocol.get(key) != value}
    if mismatch:
        raise ValueError(f"preregistered protocol mismatch: {mismatch}")
    return protocol


def _validate_i0(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    if _sha256(manifest_path) != EXPECTED_I0_MANIFEST_SHA256:
        raise ValueError("formal I0-R manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("V4_2_V2_1_I0_R") != "PASS" or not manifest.get("V4_2_V2_1_I1_ALLOWED"):
        raise ValueError("I0-R did not authorize I1")
    if manifest.get("observation_schema") != OBSERVATION_SCHEMA or manifest.get("observation_dim") != OBSERVATION_DIM:
        raise ValueError("I0-R observation schema mismatch")
    if manifest.get("action_schema") != ACTION_SCHEMA_V4_2:
        raise ValueError("I0-R action schema mismatch")
    if not all(bool(value) for value in manifest.get("qualification_gates", {}).values()):
        raise ValueError("I0-R qualification/safety provenance regression")
    train_path = Path(manifest["formal_training_path"])
    if _sha256(train_path) != EXPECTED_TRAIN_SHA256:
        raise ValueError("formal training dataset SHA256 mismatch")
    return manifest, train_path


def _train_seed(seed: int, train_rows: list[dict], output: Path, i0_manifest: Path,
                protocol_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    observations, gates, proposals, joint = _arrays(train_rows)
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = JointPPOV42(OBSERVATION_DIM, learning_rate=LEARNING_RATE)
    fingerprint = _architecture_fingerprint(policy)
    before = _metrics(policy, observations, gates, proposals, joint)
    epochs: list[dict[str, Any]] = []
    gradient_norms: list[float] = []
    for epoch in range(1, EPOCHS + 1):
        epoch_norms: list[float] = []
        for offset in range(0, len(observations), BATCH_SIZE):
            ids = np.random.permutation(len(observations)) if offset == 0 else permutation
            permutation = ids
            batch = permutation[offset:offset + BATCH_SIZE]
            gate_dist, proposal_dist = policy.model.distributions(observations[batch])
            loss, _ = hierarchical_bc_loss(gate_dist.logits, proposal_dist.logits, gates[batch], proposals[batch])
            policy.optimizer.zero_grad()
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0))
            if not np.isfinite(norm):
                raise FloatingPointError(f"seed {seed}: non-finite gradient")
            policy.optimizer.step()
            epoch_norms.append(norm)
            gradient_norms.append(norm)
        metric = _metrics(policy, observations, gates, proposals, joint)
        release_share = metric["gate"]["prediction_RELEASE"] / len(train_rows)
        epochs.append({
            "epoch": epoch,
            "total_loss": metric["total_BC_loss"],
            "gate_CE": metric["gate_CE"],
            "proposal_CE": metric["proposal_CE"],
            "WAIT_recall": metric["gate"]["WAIT_recall"],
            "RELEASE_recall": metric["gate"]["RELEASE_recall"],
            "gate_macro_F1": metric["gate"]["macro_F1"],
            "proposal_top1": metric["proposal"]["top_1"],
            "proposal_top3": metric["proposal"]["top_3"],
            "decoded_accuracy": metric["decoded_joint_action_accuracy"],
            "gradient_norm_mean": float(np.mean(epoch_norms)),
            "finite_gradients": bool(np.isfinite(epoch_norms).all()),
            "WAIT_prediction_share": 1.0 - release_share,
            "RELEASE_prediction_share": release_share,
            "dominant_proposal_share": metric["proposal"]["dominant_prediction_share"],
        })
    final = _metrics(policy, observations, gates, proposals, joint)
    config = replace(_base_config(), observation_schema_version=OBSERVATION_SCHEMA,
                     action_schema_version=ACTION_SCHEMA_V4_2, retry_norm_horizon_seconds=30.0)
    checkpoint = output / f"v4_2_v2_1_i1_seed{seed}_final.pt"
    save_checkpoint(
        checkpoint, policy, config,
        stage="V4_2_V2_1_I1_MULTI_SEED_FORMAL", seed=seed, final_epoch=EPOCHS,
        fresh_actor=True, resumed_checkpoint=None, ppo_called=False,
        architecture_fingerprint=fingerprint,
        training_dataset_sha256=EXPECTED_TRAIN_SHA256,
        i0_r_manifest_sha256=EXPECTED_I0_MANIFEST_SHA256,
        qualification_protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        checkpoint_selection="final_epoch",
    )
    checkpoint_sha = _sha256(checkpoint)
    log_path = output / f"v4_2_v2_1_i1_seed{seed}_training_log.json"
    log = {
        "evidence_type": "newly_run", "seed": seed, "fresh_actor": True,
        "checkpoint_resume": False, "Q_DATASET_ACCESS_DURING_TRAINING": False,
        "observation_schema": OBSERVATION_SCHEMA, "observation_dim": OBSERVATION_DIM,
        "action_schema": ACTION_SCHEMA_V4_2, "architecture_fingerprint": fingerprint,
        "training_rows": len(train_rows), "training_dataset_sha256": EXPECTED_TRAIN_SHA256,
        "i0_r_manifest": str(i0_manifest), "i0_r_manifest_sha256": EXPECTED_I0_MANIFEST_SHA256,
        "qualification_protocol": str(protocol_path), "qualification_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "checkpoint_selection": "final_epoch",
        "loss": "mean_CE_gate_all + mean_CE_proposal_Teacher_RELEASE_only",
        "forbidden_manipulations": {name: False for name in (
            "class_weighting", "focal_loss", "oversampling", "undersampling", "balanced_sampler",
            "row_duplication", "label_smoothing", "proposal_weighting", "hyperparameter_search")},
        "before_training": before, "epoch_metrics": epochs, "final_training": final,
        "gradient_norm": {"mean": float(np.mean(gradient_norms)), "max": float(np.max(gradient_norms)),
                          "finite": bool(np.isfinite(gradient_norms).all())},
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
    }
    _write_json(log_path, log)
    return checkpoint, log_path, log


def _load_frozen_policy(checkpoint: Path, seed: int, fingerprint: str) -> JointPPOV42:
    config = replace(_base_config(), observation_schema_version=OBSERVATION_SCHEMA,
                     action_schema_version=ACTION_SCHEMA_V4_2, retry_norm_horizon_seconds=30.0)
    policy = JointPPOV42(OBSERVATION_DIM, learning_rate=LEARNING_RATE)
    payload = load_checkpoint(checkpoint, policy, config, load_optimizer=False)
    expected = {
        "seed": seed, "final_epoch": EPOCHS, "fresh_actor": True, "resumed_checkpoint": None,
        "architecture_fingerprint": fingerprint, "training_dataset_sha256": EXPECTED_TRAIN_SHA256,
        "i0_r_manifest_sha256": EXPECTED_I0_MANIFEST_SHA256,
        "qualification_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatch:
        raise ValueError(f"seed {seed} checkpoint lineage mismatch: {mismatch}")
    return policy


def _evaluate_seed(policy: JointPPOV42, rows: list[dict], protocol: dict[str, Any],
                   finite_gradients: bool) -> dict[str, Any]:
    observations, gates, proposals, joint = _arrays(rows)
    metric = _metrics(policy, observations, gates, proposals, joint)
    with torch.no_grad():
        gate_dist, _ = policy.model.distributions(observations)
        predicted_gate = gate_dist.logits.argmax(-1)
    family = _family_metrics(rows, predicted_gate)
    retry = _retry_metrics(rows, predicted_gate)
    predicted = {int(key) for key, value in metric["proposal"]["prediction_counts"].items() if value > 0}
    release_share = metric["gate"]["prediction_RELEASE"] / len(rows)
    wait_share = 1.0 - release_share
    limits = protocol["per_seed_hard_gates"]
    gates_result = {
        "WAIT_recall": metric["gate"]["WAIT_recall"] >= limits["wait_recall_min"],
        "RELEASE_recall": metric["gate"]["RELEASE_recall"] >= limits["release_recall_min"],
        "gate_macro_F1": metric["gate"]["macro_F1"] >= limits["gate_macro_f1_min"],
        "proposal_top1": metric["proposal"]["top_1"] >= limits["proposal_top1_min"],
        "proposal_top3": metric["proposal"]["top_3"] >= limits["proposal_top3_min"],
        "decoded_accuracy": metric["decoded_joint_action_accuracy"] >= limits["decoded_joint_accuracy_min"],
        "required_proposals_predicted": set(limits["stable_teacher_proposals_required"]).issubset(predicted),
        "finite_gradients": finite_gradients,
        "WAIT_prediction_share": wait_share >= limits["gate_class_prediction_share_min"],
        "RELEASE_prediction_share": release_share >= limits["gate_class_prediction_share_min"],
        "dominant_proposal_share": metric["proposal"]["dominant_prediction_share"] < limits["proposal_dominant_prediction_share_max_exclusive"],
    }
    return {
        "WAIT_recall": metric["gate"]["WAIT_recall"],
        "RELEASE_recall": metric["gate"]["RELEASE_recall"],
        "gate_macro_F1": metric["gate"]["macro_F1"],
        "proposal_top1": metric["proposal"]["top_1"],
        "proposal_top3": metric["proposal"]["top_3"],
        "decoded_accuracy": metric["decoded_joint_action_accuracy"],
        "proposal_predictions": sorted(predicted),
        "proposal_prediction_histogram": metric["proposal"]["prediction_counts"],
        "WAIT_prediction_share": wait_share, "RELEASE_prediction_share": release_share,
        "dominant_proposal_share": metric["proposal"]["dominant_prediction_share"],
        "finite_gradients": finite_gradients, "hard_gates": gates_result,
        "PASS": all(gates_result.values()),
        "boundary_diagnostics": {
            "clear_outer_RELEASE_recall": family["clear_outer"]["RELEASE_recall"],
            "middle_interface_RELEASE_recall": family["middle_interface"]["RELEASE_recall"],
            "post_WAIT_RELEASE_recall": family["post_WAIT"]["RELEASE_recall"],
            "post_conflict_RELEASE_recall": family["post_conflict"]["RELEASE_recall"],
            "retry_active_false_RELEASE_rate": retry["retry_active_false_RELEASE_rate"],
            "retry_expired_RELEASE_recall": retry["retry_expired_RELEASE_recall"],
            "proposal_recall": {key: value["recall"] for key, value in metric["proposal"]["per_stable_class"].items()},
        },
    }


def _aggregate(seed_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    names = ("WAIT_recall", "RELEASE_recall", "gate_macro_F1", "proposal_top1", "proposal_top3",
             "decoded_accuracy", "WAIT_prediction_share", "RELEASE_prediction_share", "dominant_proposal_share")
    result: dict[str, dict[str, float]] = {}
    for name in names:
        values = np.asarray([seed_results[str(seed)][name] for seed in SEEDS], dtype=np.float64)
        result[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
                        "min": float(values.min()), "max": float(values.max())}
    return result


def _render_training_summary(frozen: dict[str, Any]) -> str:
    lines = ["# V4.2 / Observation V2.1 Multi-Seed I1 Training Summary", "",
             "Evidence type: `newly_run`.", "", "```text",
             "Q_DATASET_ACCESS_DURING_TRAINING = false"]
    for seed in SEEDS:
        item = frozen["seeds"][str(seed)]
        lines += [f"TRAINING_{seed}_FROZEN = true", f"seed{seed}_checkpoint_sha256 = {item['checkpoint_sha256']}",
                  f"seed{seed}_training_log_sha256 = {item['training_log_sha256']}"]
    lines += ["```", "", "All actors used fresh initialization, the same immutable 2,020-row training set,",
              "Adam 3e-4, batch size 64, 100 epochs, final-epoch selection, and unweighted hierarchical CE.",
              "No Q artifact was opened before every checkpoint and training log was frozen and hashed.", ""]
    return "\n".join(lines)


def _render_qualification_summary(report: dict[str, Any]) -> str:
    lines = ["# V4.2 / Observation V2.1 I1 Formal Qualification", "", "Evidence type: `newly_run`.", "",
             "The immutable Q cohort was opened in one evaluation invocation only after all three trainings were frozen.", ""]
    for seed in SEEDS:
        r = report["seed_results"][str(seed)]
        lines += [f"## Seed {seed}", "", "```text",
                  f"WAIT_recall = {r['WAIT_recall']:.6f}", f"RELEASE_recall = {r['RELEASE_recall']:.6f}",
                  f"gate_macro_F1 = {r['gate_macro_F1']:.6f}", f"proposal_top1 = {r['proposal_top1']:.6f}",
                  f"proposal_top3 = {r['proposal_top3']:.6f}", f"decoded_accuracy = {r['decoded_accuracy']:.6f}",
                  f"proposal_predictions = {r['proposal_predictions']}", f"WAIT_prediction_share = {r['WAIT_prediction_share']:.6f}",
                  f"RELEASE_prediction_share = {r['RELEASE_prediction_share']:.6f}",
                  f"dominant_proposal_share = {r['dominant_proposal_share']:.6f}",
                  f"PASS = {str(r['PASS']).upper()}", "```", ""]
    lines += ["## Decision", "", "```text", "Q_EVALUATION_COUNT = 1",
              f"V4_2_V2_1_I1 = {report['V4_2_V2_1_I1']}",
              f"CANONICAL_I1_CHECKPOINT = {report['CANONICAL_I1_CHECKPOINT']}",
              f"V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = {str(report['V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED']).lower()}",
              "PPO_ALLOWED = false", "```", "", "## Protocol risks", "",
              "- This is an internal engineering qualification cohort, not an external generalization test.",
              "- Boundary-family diagnostics are informative only and did not alter preregistered hard gates.",
              "- The conjunctive three-seed rule was applied; means cannot rescue a failed seed.", ""]
    return "\n".join(lines)


def run(i0_manifest: str | Path, q_dataset: str | Path, q_manifest: str | Path,
        protocol_config: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    existing = list(output.iterdir()) if output.exists() else []
    # A nohup launcher creates/truncates run.log before Python enters run().
    # It is not a formal artifact and must not defeat the no-rerun guard.
    formal_existing = [path for path in existing if path.name != "run.log"]
    if formal_existing:
        raise FileExistsError("formal output directory already has artifacts; refusing a second Q evaluation")
    output.mkdir(parents=True, exist_ok=True)
    q_dataset_path, q_manifest_path = Path(q_dataset), Path(q_manifest)
    state = {"phase": "TRAINING", "q_access_during_training": False, "q_evaluation_count": 0}
    _install_q_access_guard((q_dataset_path, q_manifest_path), state)

    protocol_path = Path(protocol_config)
    protocol = _validate_protocol(protocol_path)
    i0_path = Path(i0_manifest)
    i0, train_path = _validate_i0(i0_path)
    train_rows = _rows(train_path)
    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError("formal training row count mismatch")

    frozen_training: dict[str, Any] = {
        "phase": "ALL_TRAINING_FROZEN_BEFORE_Q", "evidence_type": "newly_run",
        "Q_DATASET_ACCESS_DURING_TRAINING": False, "seeds": {},
    }
    training_logs: dict[int, dict[str, Any]] = {}
    checkpoints: dict[int, Path] = {}
    for seed in SEEDS:
        checkpoint, log_path, log = _train_seed(seed, train_rows, output, i0_path, protocol_path)
        checkpoints[seed] = checkpoint
        training_logs[seed] = log
        frozen_training["seeds"][str(seed)] = {
            "TRAINING_FROZEN": True, "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint), "training_log": str(log_path),
            "training_log_sha256": _sha256(log_path),
        }
    freeze_path = output / "v4_2_v2_1_i1_training_freeze_manifest.json"
    _write_json(freeze_path, frozen_training)
    frozen_training["training_freeze_manifest_sha256"] = _sha256(freeze_path)
    training_summary = output / "v4_2_v2_1_i1_multiseed_training_summary.md"
    training_summary.write_text(_render_training_summary(frozen_training), encoding="utf-8")

    if state["q_access_during_training"]:
        raise RuntimeError("Q_DATASET_ACCESS_DURING_TRAINING must remain false")
    state["phase"] = "Q_EVALUATION"
    state["q_evaluation_count"] += 1
    if state["q_evaluation_count"] != 1:
        raise RuntimeError("formal Q evaluation count must be exactly one")

    if _sha256(q_manifest_path) != EXPECTED_Q_MANIFEST_SHA256 or _sha256(q_dataset_path) != EXPECTED_Q_SHA256:
        raise ValueError("immutable Q artifact hash mismatch")
    q_meta = json.loads(q_manifest_path.read_text(encoding="utf-8"))
    if not q_meta.get("I1_Q_COHORT_FROZEN") or not q_meta.get("I1_FRESH_MULTI_SEED_TRAINING_ALLOWED"):
        raise ValueError("Q cohort is not formally frozen")
    if q_meta.get("observation_schema") != OBSERVATION_SCHEMA or q_meta.get("observation_dim") != OBSERVATION_DIM:
        raise ValueError("Q observation schema mismatch")
    if q_meta.get("action_schema") != ACTION_SCHEMA_V4_2:
        raise ValueError("Q action schema mismatch")
    if not all(bool(value) for value in q_meta.get("freeze_gates", {}).values()):
        raise ValueError("Q provenance/safety/schema regression")
    if q_meta.get("observation_aliasing", {}).get("exact_observation_hash_opposite_gate_conflicts") != 0:
        raise ValueError("opposite-gate exact observation contradiction")
    q_rows = _rows(q_dataset_path)
    if len(q_rows) != EXPECTED_Q_ROWS:
        raise ValueError("Q row count mismatch")

    seed_results: dict[str, Any] = {}
    for seed in SEEDS:
        log = training_logs[seed]
        policy = _load_frozen_policy(checkpoints[seed], seed, log["architecture_fingerprint"])
        seed_results[str(seed)] = _evaluate_seed(policy, q_rows, protocol, log["gradient_norm"]["finite"])
    passed = all(seed_results[str(seed)]["PASS"] for seed in SEEDS)
    result = {
        "stage": "V4_2_V2_1_I1_FORMAL_MULTI_SEED_QUALIFICATION", "evidence_type": "newly_run",
        "observation_schema": OBSERVATION_SCHEMA, "observation_dim": OBSERVATION_DIM,
        "action_schema": ACTION_SCHEMA_V4_2, "training_rows": len(train_rows), "Q_rows": len(q_rows),
        "lineage": {"training_dataset_sha256": EXPECTED_TRAIN_SHA256,
                    "i0_r_manifest_sha256": EXPECTED_I0_MANIFEST_SHA256,
                    "Q_dataset_sha256": EXPECTED_Q_SHA256, "Q_manifest_sha256": EXPECTED_Q_MANIFEST_SHA256,
                    "qualification_protocol_sha256": EXPECTED_PROTOCOL_SHA256},
        "Q_DATASET_ACCESS_DURING_TRAINING": False, "Q_EVALUATION_COUNT": 1,
        "training_freeze_manifest": str(freeze_path),
        "training_freeze_manifest_sha256": frozen_training["training_freeze_manifest_sha256"],
        "checkpoints": frozen_training["seeds"], "seed_results": seed_results,
        "multi_seed_aggregate": _aggregate(seed_results),
        "multi_seed_rule": "PASS iff seed4202 AND seed4203 AND seed4204 pass every hard gate",
        "canonical_seed_preregistered": CANONICAL_SEED,
        "V4_2_V2_1_I1": "PASS" if passed else "FAIL",
        "CANONICAL_I1_CHECKPOINT": str(checkpoints[CANONICAL_SEED]) if passed else "none",
        "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": bool(passed),
        "DAgger_run": False, "PPO_ALLOWED": False, "ppo_called": False,
        "protocol_risks": [
            "Internal engineering qualification cohort; not an external generalization test.",
            "Boundary-family diagnostics are not hard gates and did not select models or thresholds.",
            "All three seeds must pass; aggregate statistics cannot rescue a failed seed.",
        ],
    }
    manifest_out = output / "v4_2_v2_1_i1_formal_qualification_manifest.json"
    _write_json(manifest_out, result)
    qualification_summary = output / "v4_2_v2_1_i1_formal_qualification_summary.md"
    qualification_summary.write_text(_render_qualification_summary(result), encoding="utf-8")
    result["manifest_sha256"] = _sha256(manifest_out)
    result["training_summary_sha256"] = _sha256(training_summary)
    result["qualification_summary_sha256"] = _sha256(qualification_summary)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i0-manifest", required=True)
    parser.add_argument("--q-dataset", required=True)
    parser.add_argument("--q-manifest", required=True)
    parser.add_argument("--protocol-config", default="terminal_marl/configs/low_v4_2/i1_v2_1_qualification_protocol.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = run(args.i0_manifest, args.q_dataset, args.q_manifest, args.protocol_config, args.output_dir)
    print(json.dumps({
        "checkpoint_sha256": {seed: report["checkpoints"][str(seed)]["checkpoint_sha256"] for seed in SEEDS},
        "Q_EVALUATION_COUNT": report["Q_EVALUATION_COUNT"], "seed_results": report["seed_results"],
        "multi_seed_aggregate": report["multi_seed_aggregate"], "V4_2_V2_1_I1": report["V4_2_V2_1_I1"],
        "CANONICAL_I1_CHECKPOINT": report["CANONICAL_I1_CHECKPOINT"],
        "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": report["V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED"],
        "PPO_ALLOWED": False,
    }, indent=2))
