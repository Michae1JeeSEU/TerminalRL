"""Fresh formal V4.2 hierarchical BC on the qualified V2.1 I0-R lineage.

The frozen cohort is loaded and evaluated exactly once after final-epoch
training.  It never contributes gradients, checkpoint selection, thresholds,
or hyperparameter choices.  No V2.1 I1 thresholds existed before this run, so
the script reports full diagnostics but fails closed on I2 authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..algo.joint_ppo_v4_2 import JointPPOV42, hierarchical_bc_loss
from ..core.checkpoints import IncompatibleCheckpointError, load_checkpoint, save_checkpoint
from ..core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from ..core.low_level_joint_action_v4_2 import (
    ACTION_SCHEMA_V4_2, RELEASE_GATE, hierarchical_from_joint_index_v4_1,
)
from .run_v4_2_hierarchical_s0_qualification import _config as _base_config


OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
OBSERVATION_DIM = 88
EXPECTED_TRAIN_SHA256 = "6bcbc1782a171f909282126c4f6d27dd93355b09513e2827093228af4fe05f05"
EXPECTED_FROZEN_SHA256 = "229e9d0c087eac1e238c90d857e74dc71525d4eac1b343ed88f33fa7e0695aa2"
EXPECTED_I0_MANIFEST_SHA256 = "51868f7892395d7d5a369d956f6c374ba536d3c8023ec5d0f42a2d9f208f8a31"
EXPECTED_TRAIN_ROWS = 2020
EXPECTED_FROZEN_ROWS = 643
STABLE_JOINT_SUPPORT = (11, 12, 13, 14)
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_SEED = 4201
COLLAPSE_SHARE = 0.95


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _arrays(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.as_tensor(np.asarray([row["observation_v2_1"] for row in rows], dtype=np.float32))
    gates = torch.as_tensor(np.asarray([row["teacher_gate"] for row in rows], dtype=np.int64))
    proposals = torch.as_tensor(np.asarray([
        -1 if row["teacher_proposal"] is None else int(row["teacher_proposal"])
        for row in rows
    ], dtype=np.int64))
    joint = torch.as_tensor(np.asarray([row["teacher_joint_index"] for row in rows], dtype=np.int64))
    if observations.shape != (len(rows), OBSERVATION_DIM):
        raise ValueError(f"formal I1 requires {OBSERVATION_DIM}D observations")
    expected_gate = (joint > 0).long()
    expected_proposal = torch.where(joint > 0, joint - 1, torch.full_like(joint, -1))
    if not torch.equal(gates, expected_gate) or not torch.equal(proposals, expected_proposal):
        raise ValueError("canonical joint-to-hierarchical target mapping mismatch")
    # Explicitly freeze the audited 11->10 ... 14->13 mapping.
    for joint_index in STABLE_JOINT_SUPPORT:
        mapped = hierarchical_from_joint_index_v4_1(joint_index)
        if mapped.gate != RELEASE_GATE or mapped.proposal != joint_index - 1:
            raise AssertionError("stable proposal mapping is off by one")
    return observations, gates, proposals, joint


def _gate_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    prediction = logits.argmax(-1)
    wait = target == 0
    release = target == RELEASE_GATE
    predicted_wait = prediction == 0
    predicted_release = prediction == RELEASE_GATE
    wait_recall = float((prediction[wait] == 0).float().mean()) if bool(wait.any()) else 0.0
    release_recall = float((prediction[release] == RELEASE_GATE).float().mean()) if bool(release.any()) else 0.0
    release_precision = float((predicted_release & release).sum() / predicted_release.sum().clamp_min(1))
    wait_precision = float((predicted_wait & wait).sum() / predicted_wait.sum().clamp_min(1))
    wait_f1 = 2 * wait_precision * wait_recall / max(1e-12, wait_precision + wait_recall)
    release_f1 = 2 * release_precision * release_recall / max(1e-12, release_precision + release_recall)
    return {
        "accuracy": float((prediction == target).float().mean()),
        "WAIT_recall": wait_recall,
        "RELEASE_recall": release_recall,
        "RELEASE_precision": release_precision,
        "macro_F1": float((wait_f1 + release_f1) / 2),
        "prediction_WAIT": int(predicted_wait.sum()),
        "prediction_RELEASE": int(predicted_release.sum()),
        "confusion": {
            "Teacher_WAIT_predicted_WAIT": int((wait & predicted_wait).sum()),
            "Teacher_WAIT_predicted_RELEASE": int((wait & predicted_release).sum()),
            "Teacher_RELEASE_predicted_WAIT": int((release & predicted_wait).sum()),
            "Teacher_RELEASE_predicted_RELEASE": int((release & predicted_release).sum()),
        },
    }


def _proposal_metrics(logits: torch.Tensor, proposal_target: torch.Tensor) -> dict:
    valid = proposal_target >= 0
    logits = logits[valid]
    target = proposal_target[valid]
    prediction = logits.argmax(-1)
    top3 = (logits.topk(3, dim=-1).indices == target[:, None]).any(-1)
    predicted_joint = prediction + 1
    target_joint = target + 1
    predicted_actions = [JOINT_ACTION_CATALOG_V4_1[int(value)] for value in predicted_joint]
    target_actions = [JOINT_ACTION_CATALOG_V4_1[int(value)] for value in target_joint]
    prediction_counts = {str(index): int((predicted_joint == index).sum()) for index in sorted(set(predicted_joint.tolist()))}
    per_class = {}
    for joint_index in STABLE_JOINT_SUPPORT:
        mask = target_joint == joint_index
        per_class[str(joint_index)] = {
            "target_count": int(mask.sum()),
            "recall": float((predicted_joint[mask] == joint_index).float().mean()) if bool(mask.any()) else None,
            "prediction_count": int((predicted_joint == joint_index).sum()),
        }
    dominant_joint, dominant_count = Counter(int(value) for value in predicted_joint.tolist()).most_common(1)[0]
    dominant_share = float(dominant_count / max(1, len(predicted_joint)))
    return {
        "top_1": float((prediction == target).float().mean()),
        "top_3": float(top3.float().mean()),
        "speed_agreement": float(np.mean([left.speed_residual == right.speed_residual for left, right in zip(predicted_actions, target_actions)])),
        "merge_agreement": float(np.mean([left.merge_time_residual == right.merge_time_residual for left, right in zip(predicted_actions, target_actions)])),
        "prediction_counts": prediction_counts,
        "target_joint_classes": sorted(set(int(value) for value in target_joint.tolist())),
        "per_stable_class": per_class,
        "dominant_prediction_joint": dominant_joint,
        "dominant_prediction_share": dominant_share,
        "single_proposal_collapse": len(prediction_counts) == 1,
        "near_single_proposal_collapse": dominant_share >= COLLAPSE_SHARE,
        "proposal_13_share": float((predicted_joint == 13).float().mean()),
    }


def _metrics(policy: JointPPOV42, observations: torch.Tensor, gates: torch.Tensor,
             proposals: torch.Tensor, joint_target: torch.Tensor) -> dict:
    with torch.no_grad():
        gate_dist, proposal_dist = policy.model.distributions(observations)
        total, parts = hierarchical_bc_loss(gate_dist.logits, proposal_dist.logits, gates, proposals)
        predicted_gate = gate_dist.logits.argmax(-1)
        predicted_proposal = proposal_dist.logits.argmax(-1)
        decoded = torch.where(predicted_gate == RELEASE_GATE, predicted_proposal + 1, torch.zeros_like(predicted_proposal))
        entropy = policy.model.conditional_entropy(observations)
    return {
        "total_BC_loss": float(total),
        "gate_CE": float(parts["gate"]),
        "proposal_CE": float(parts["proposal"]),
        "gate": _gate_metrics(gate_dist.logits, gates),
        "proposal": _proposal_metrics(proposal_dist.logits, proposals),
        "decoded_joint_action_accuracy": float((decoded == joint_target).float().mean()),
        "conditional_entropy": float(entropy.mean()),
    }


def _family_metrics(rows: list[dict], predicted_gate: torch.Tensor) -> dict:
    result = {}
    for key, display in (("clear_outer", "clear_outer"), ("middle_interface", "middle_interface"),
                         ("post_wait", "post_WAIT"), ("post_conflict", "post_conflict")):
        indices = [index for index, row in enumerate(rows) if bool(row["family_flags"][key])]
        target = torch.as_tensor([int(rows[index]["teacher_gate"]) for index in indices])
        prediction = predicted_gate[indices]
        release = target == RELEASE_GATE
        result[display] = {
            "sample_count": len(indices),
            "lifecycle_count": len({rows[index]["lifecycle_event_id"] for index in indices}),
            "target_WAIT": int((target == 0).sum()),
            "target_RELEASE": int(release.sum()),
            "gate_accuracy": float((prediction == target).float().mean()) if indices else None,
            "RELEASE_recall": float((prediction[release] == RELEASE_GATE).float().mean()) if bool(release.any()) else None,
        }
    return result


def _retry_metrics(rows: list[dict], predicted_gate: torch.Tensor) -> dict:
    retry_active = torch.as_tensor([bool(row["retry_active"]) for row in rows])
    target_gate = torch.as_tensor([int(row["teacher_gate"]) for row in rows])
    retry_inactive = ~retry_active
    expired_release = retry_inactive & (target_gate == RELEASE_GATE)
    return {
        "retry_active_states": int(retry_active.sum()),
        "retry_inactive_states": int(retry_inactive.sum()),
        "retry_active_false_RELEASE_rate": float((predicted_gate[retry_active] == RELEASE_GATE).float().mean()) if bool(retry_active.any()) else None,
        "retry_expired_RELEASE_states": int(expired_release.sum()),
        "retry_expired_RELEASE_recall": float((predicted_gate[expired_release] == RELEASE_GATE).float().mean()) if bool(expired_release.any()) else None,
    }


def _architecture_fingerprint(policy: JointPPOV42) -> str:
    shape_map = {name: list(parameter.shape) for name, parameter in policy.model.state_dict().items()}
    payload = {"class": policy.model.__class__.__qualname__, "observation_dim": OBSERVATION_DIM, "parameters": shape_map}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_formal_i1_checkpoint(path: str | Path, policy: JointPPOV42, config,
                              *, load_optimizer: bool = True) -> dict:
    """Schema- and lineage-strict loader for any future formal resume attempt."""
    payload = load_checkpoint(path, policy, config, load_optimizer=load_optimizer)
    expected = {
        "training_dataset_sha256": EXPECTED_TRAIN_SHA256,
        "frozen_dataset_sha256": EXPECTED_FROZEN_SHA256,
        "i0_r_manifest_sha256": EXPECTED_I0_MANIFEST_SHA256,
        "architecture_fingerprint": _architecture_fingerprint(policy),
    }
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatch:
        raise IncompatibleCheckpointError(f"formal V2.1 I1 checkpoint lineage mismatch: {mismatch}")
    return payload


def _validate_lineage(manifest_path: Path) -> tuple[dict, Path, Path]:
    if _sha256(manifest_path) != EXPECTED_I0_MANIFEST_SHA256:
        raise ValueError("formal I0-R manifest SHA256 mismatch")
    lineage = json.loads(manifest_path.read_text(encoding="utf-8"))
    if lineage.get("V4_2_V2_1_I0_R") != "PASS" or not lineage.get("V4_2_V2_1_I1_ALLOWED"):
        raise ValueError("formal I0-R did not authorize I1")
    if lineage.get("observation_schema") != OBSERVATION_SCHEMA or lineage.get("observation_dim") != OBSERVATION_DIM:
        raise ValueError("formal I0-R observation schema mismatch")
    if lineage.get("action_schema") != ACTION_SCHEMA_V4_2:
        raise ValueError("formal I0-R action schema mismatch")
    train_path = Path(lineage["formal_training_path"])
    frozen_path = Path(lineage["formal_frozen_path"])
    if _sha256(train_path) != EXPECTED_TRAIN_SHA256 or _sha256(frozen_path) != EXPECTED_FROZEN_SHA256:
        raise ValueError("formal I0-R dataset SHA256 mismatch")
    return lineage, train_path, frozen_path


def run(i0_manifest: str | Path, output_dir: str | Path, *, epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE, learning_rate: float = DEFAULT_LEARNING_RATE,
        seed: int = DEFAULT_SEED) -> dict:
    if (epochs, batch_size, learning_rate, seed) != (DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_SEED):
        raise ValueError("formal I1 hyperparameters are frozen; use a separate diagnostic entrypoint for deviations")
    manifest_path = Path(i0_manifest)
    lineage, train_path, frozen_path = _validate_lineage(manifest_path)
    train_rows = _rows(train_path)
    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError("formal I1 training row count mismatch")
    observations, gate_target, proposal_target, joint_target = _arrays(train_rows)
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = JointPPOV42(OBSERVATION_DIM, learning_rate=learning_rate)
    architecture_fingerprint = _architecture_fingerprint(policy)
    before_train = _metrics(policy, observations, gate_target, proposal_target, joint_target)
    epoch_metrics = []
    all_gradient_norms = []
    for epoch in range(1, epochs + 1):
        epoch_gradients = []
        permutation = np.random.permutation(len(observations))
        for offset in range(0, len(observations), batch_size):
            ids = permutation[offset:offset + batch_size]
            gate_dist, proposal_dist = policy.model.distributions(observations[ids])
            loss, _ = hierarchical_bc_loss(
                gate_dist.logits, proposal_dist.logits,
                gate_target[ids], proposal_target[ids],
            )
            policy.optimizer.zero_grad()
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0))
            if not np.isfinite(norm):
                raise FloatingPointError("non-finite BC gradient")
            epoch_gradients.append(norm)
            all_gradient_norms.append(norm)
            policy.optimizer.step()
        values = _metrics(policy, observations, gate_target, proposal_target, joint_target)
        epoch_metrics.append({
            "epoch": epoch,
            "total_BC_loss": values["total_BC_loss"],
            "gate_CE": values["gate_CE"],
            "proposal_CE": values["proposal_CE"],
            "training_gate_accuracy": values["gate"]["accuracy"],
            "training_WAIT_recall": values["gate"]["WAIT_recall"],
            "training_RELEASE_recall": values["gate"]["RELEASE_recall"],
            "training_gate_macro_F1": values["gate"]["macro_F1"],
            "training_proposal_top_1": values["proposal"]["top_1"],
            "training_proposal_top_3": values["proposal"]["top_3"],
            "gradient_norm_mean": float(np.mean(epoch_gradients)),
        })
    final_train = _metrics(policy, observations, gate_target, proposal_target, joint_target)

    # Formal frozen cohort use begins here and occurs exactly once, after the
    # final epoch has already been selected independently of frozen metrics.
    frozen_rows = _rows(frozen_path)
    if len(frozen_rows) != EXPECTED_FROZEN_ROWS:
        raise ValueError("formal I1 frozen row count mismatch")
    frozen_observations, frozen_gates, frozen_proposals, frozen_joint = _arrays(frozen_rows)
    frozen = _metrics(policy, frozen_observations, frozen_gates, frozen_proposals, frozen_joint)
    with torch.no_grad():
        frozen_gate_dist, _ = policy.model.distributions(frozen_observations)
        frozen_predicted_gate = frozen_gate_dist.logits.argmax(-1)
    family = _family_metrics(frozen_rows, frozen_predicted_gate)
    retry = _retry_metrics(frozen_rows, frozen_predicted_gate)
    gate_share_release = frozen["gate"]["prediction_RELEASE"] / len(frozen_rows)
    gate_collapse = bool(gate_share_release <= 1.0 - COLLAPSE_SHARE or gate_share_release >= COLLAPSE_SHARE)
    proposal_collapse = bool(frozen["proposal"]["near_single_proposal_collapse"])
    represented = {int(key) for key, value in frozen["proposal"]["prediction_counts"].items() if value > 0}
    diagnostics = {
        "gate_CE_decreased": final_train["gate_CE"] < before_train["gate_CE"],
        "proposal_CE_decreased": final_train["proposal_CE"] < before_train["proposal_CE"],
        "WAIT_and_RELEASE_both_predicted": frozen["gate"]["prediction_WAIT"] > 0 and frozen["gate"]["prediction_RELEASE"] > 0,
        "all_stable_classes_represented_in_predictions": set(STABLE_JOINT_SUPPORT).issubset(represented),
        "gate_collapse": gate_collapse,
        "proposal_collapse": proposal_collapse,
        "post_WAIT_RELEASE_recall": family["post_WAIT"]["RELEASE_recall"],
        "post_conflict_RELEASE_recall": family["post_conflict"]["RELEASE_recall"],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = replace(
        _base_config(), observation_schema_version=OBSERVATION_SCHEMA,
        action_schema_version=ACTION_SCHEMA_V4_2, retry_norm_horizon_seconds=30.0,
    )
    checkpoint = output / "v4_2_v2_1_i1_final_epoch.pt"
    checkpoint_extra = {
        "stage": "V4_2_V2_1_I1_BC",
        "fresh_actor": True,
        "resumed_checkpoint": None,
        "ppo_called": False,
        "final_epoch": epochs,
        "architecture_fingerprint": architecture_fingerprint,
        "training_dataset_sha256": EXPECTED_TRAIN_SHA256,
        "frozen_dataset_sha256": EXPECTED_FROZEN_SHA256,
        "i0_r_manifest_sha256": EXPECTED_I0_MANIFEST_SHA256,
        "optimizer_config": {"name": "Adam", "learning_rate": learning_rate},
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    save_checkpoint(checkpoint, policy, config, **checkpoint_extra)
    checkpoint_sha = _sha256(checkpoint)
    source_path = Path(__file__)
    result = {
        "stage": "FRESH_V4_2_V2_1_I1_BC",
        "evidence_type": "newly_run",
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_dim": OBSERVATION_DIM,
        "action_schema": ACTION_SCHEMA_V4_2,
        "architecture_fingerprint": architecture_fingerprint,
        "fresh_actor": True,
        "resumed_checkpoint": None,
        "training_partition_only": True,
        "frozen_backprop": False,
        "frozen_model_selection": False,
        "frozen_threshold_tuning": False,
        "checkpoint_selection_rule": "final_epoch",
        "lineage": {
            "i0_r_manifest": str(manifest_path),
            "i0_r_manifest_sha256": EXPECTED_I0_MANIFEST_SHA256,
            "training_dataset": str(train_path),
            "training_dataset_sha256": EXPECTED_TRAIN_SHA256,
            "frozen_dataset": str(frozen_path),
            "frozen_dataset_sha256": EXPECTED_FROZEN_SHA256,
            "training_rows": len(train_rows),
            "frozen_rows": len(frozen_rows),
        },
        "hyperparameters": {"optimizer": "Adam", "learning_rate": learning_rate, "batch_size": batch_size, "epochs": epochs, "seed": seed},
        "forbidden_manipulations": {"class_weighting": False, "focal_loss": False, "oversampling": False, "undersampling": False, "row_duplication": False, "balanced_sampler": False, "threshold_tuning": False, "label_smoothing": False, "proposal_reweighting": False},
        "target_mapping_assertion": {"11": 10, "12": 11, "13": 12, "14": 13, "PASS": True},
        "before_train": before_train,
        "epoch_metrics": epoch_metrics,
        "final_training": final_train,
        "frozen_qualification": frozen,
        "family_stratified_gate": family,
        "retry_stratified": retry,
        "collapse_diagnostics": {"definition_share": COLLAPSE_SHARE, "gate_RELEASE_prediction_share": gate_share_release, **diagnostics},
        "gradient_norm": {"mean": float(np.mean(all_gradient_norms)), "max": float(np.max(all_gradient_norms)), "finite": bool(np.isfinite(all_gradient_norms).all())},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_extra": checkpoint_extra,
        "training_code": str(source_path),
        "training_code_sha256": _sha256(source_path),
        "formal_config_basis": "terminal_marl/configs/low_v4_2/i1_hierarchical_bc.yaml",
        "qualification_thresholds": None,
        "qualification_note": "No formal V2.1 I1 thresholds were preregistered before this run; old Observation-V2 I1 values are diagnostic references only.",
        "V4_2_V2_1_I1": "QUALIFICATION_THRESHOLD_NOT_PREREGISTERED",
        "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": False,
        "DAgger_rollout_run": False,
        "PPO_ALLOWED": False,
        "ppo_called": False,
        "protocol_risks": [
            "Single fixed training seed; no variance estimate was authorized.",
            "Frozen cohort is an internal qualification cohort, not an external generalization test.",
            "No V2.1 I1 numeric qualification threshold was preregistered; promotion is fail-closed.",
        ],
    }
    manifest_out = output / "v4_2_v2_1_i1_checkpoint_manifest.json"
    manifest_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i0-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    report = run(args.i0_manifest, args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
                 learning_rate=args.learning_rate, seed=args.seed)
    print(json.dumps({
        "checkpoint_sha256": report["checkpoint_sha256"],
        "final_training": report["final_training"],
        "frozen_qualification": report["frozen_qualification"],
        "family_stratified_gate": report["family_stratified_gate"],
        "retry_stratified": report["retry_stratified"],
        "collapse_diagnostics": report["collapse_diagnostics"],
        "V4_2_V2_1_I1": report["V4_2_V2_1_I1"],
        "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": False,
        "PPO_ALLOWED": False,
    }, indent=2))
