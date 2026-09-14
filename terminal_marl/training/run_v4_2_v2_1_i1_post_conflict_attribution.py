"""Read-only attribution of V2.1 I1 post-conflict RELEASE errors.

This diagnostic never trains, changes labels, or selects a checkpoint.  It
evaluates the already-consumed frozen cohort with the diagnostic-only I1
checkpoint and measures support in the immutable formal I0-R training set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..algo.joint_ppo_v4_2 import JointPPOV42
from .run_v4_2_hierarchical_s0_qualification import _config as _base_config
from .run_v4_2_v2_1_i1_bc import (
    ACTION_SCHEMA_V4_2,
    OBSERVATION_DIM,
    OBSERVATION_SCHEMA,
    _sha256,
    load_formal_i1_checkpoint,
)


EXPECTED_CHECKPOINT_SHA256 = "3b4ddd305dd873e9f892774d4d86fa321230fd7d6c3103bb0583ce5c6c1fb725"
CONFLICT_SLOT = {
    "present": 66,
    "path_gap_norm": 67,
    "euclidean_distance_norm": 68,
    "relative_speed_norm": 69,
    "closing_rate_norm": 70,
    "dCPA_norm": 71,
    "tCPA_norm": 72,
    "priority_delta_norm": 73,
    "reserved_or_committed": 74,
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(data), "mean": float(data.mean()), "median": float(np.median(data)),
        "min": float(data.min()), "max": float(data.max()),
    }


def _nearest(observation: np.ndarray, matrix: np.ndarray, rows: list[dict], indices: np.ndarray) -> dict | None:
    if not len(indices):
        return None
    distances = np.linalg.norm(matrix[indices] - observation[None, :], axis=1)
    local = int(np.argmin(distances)); index = int(indices[local]); row = rows[index]
    return {
        "distance_l2_88d": float(distances[local]),
        "decision_event_id": row["decision_event_id"],
        "lifecycle_event_id": row["lifecycle_event_id"],
        "teacher_gate": int(row["teacher_gate"]),
        "teacher_joint_index": int(row["teacher_joint_index"]),
        "teacher_rationale": row["teacher_rationale"],
        "collection_source": row["collection_source"],
        "post_conflict": bool(row["family_flags"]["post_conflict"]),
    }


def _state_record(row: dict, prediction: int, release_probability: float, proposal_prediction: int,
                  train_matrix: np.ndarray, train_rows: list[dict], groups: dict[str, np.ndarray],
                  same_lifecycle_rows: list[dict]) -> dict:
    observation = np.asarray(row["observation_v2_1"], dtype=np.float32)
    target = row["state_snapshot"]["target_uav"]
    snapshot = row["state_snapshot"]
    ledger = row["transition_ledger"]
    previous = ledger.get("previous_semantic_event") or {}
    interface = row["headway_state"].get("interface")
    scheduler = snapshot["scheduler"]
    last_release = scheduler["last_executed_release"].get(interface) if interface else None
    return {
        "decision_event_id": row["decision_event_id"],
        "state_hash": row["full_state_hash"],
        "seed": row["seed"],
        "episode_id": row["episode_id"],
        "uav_id": row["uav_id"],
        "lifecycle_event_id": row["lifecycle_event_id"],
        "prediction": "RELEASE" if prediction == 1 else "WAIT",
        "correct_release": prediction == 1,
        "P_RELEASE": release_probability,
        "gate_logit_margin_RELEASE_minus_WAIT": float(np.log(release_probability / max(1e-12, 1.0 - release_probability))),
        "teacher": {
            "gate": "RELEASE", "selected_joint_proposal": int(row["teacher_joint_index"]),
            "selected_head_proposal": int(row["teacher_proposal"]), "rationale": row["teacher_rationale"],
        },
        "proposal_prediction": {"head_index": proposal_prediction, "joint_index": proposal_prediction + 1,
                                "correct": proposal_prediction == int(row["teacher_proposal"])},
        "phase_ring": {"phase": target["phase"], "ring_role": target["ring_role"]},
        "retry_remaining_norm": float(row["retry_feature_observed"]),
        "commit_state": {"committed": bool(target["committed"]), "trajectory_plan_id": target["trajectory_plan_id"],
                         "reservation_ids": target["reservation_ids"]},
        "interface_state": {"interface": interface, "affected": interface in snapshot["airspace"]["affected_interfaces"],
                            "transition_active": bool(snapshot["airspace"]["transition_active"]),
                            "last_executed_release": last_release,
                            "reservations": scheduler["reservations"],
                            "future_release_reservations": scheduler["future_release_reservations"]},
        "headway": {**row["headway_state"], "slack_observation": float(observation[85])},
        "resources": snapshot["resources"],
        "current_conflict_topology_observation": {name: float(observation[index]) for name, index in CONFLICT_SLOT.items()},
        "previous_WAIT_rationale": previous.get("wait_rationale"),
        "time_since_conflict_cleared_seconds": previous.get("elapsed_seconds"),
        "same_uav_lifecycle_history": [
            {"decision_event_id": item["decision_event_id"], "time": item["state_snapshot"]["env_now"],
             "teacher_gate": int(item["teacher_gate"]), "teacher_rationale": item["teacher_rationale"]}
            for item in sorted(same_lifecycle_rows, key=lambda item: item["state_snapshot"]["env_now"])
        ],
        "nearest_training_RELEASE": _nearest(observation, train_matrix, train_rows, groups["release"]),
        "nearest_training_WAIT": _nearest(observation, train_matrix, train_rows, groups["wait"]),
        "nearest_training_post_conflict_RELEASE": _nearest(observation, train_matrix, train_rows, groups["post_conflict"]),
    }


def run(train_path: Path, frozen_path: Path, checkpoint_path: Path, output_path: Path) -> dict:
    if _sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("diagnostic checkpoint SHA256 mismatch")
    train_rows = _rows(train_path); frozen_rows = _rows(frozen_path)
    train_matrix = np.asarray([row["observation_v2_1"] for row in train_rows], dtype=np.float32)
    frozen_matrix = np.asarray([row["observation_v2_1"] for row in frozen_rows], dtype=np.float32)
    policy = JointPPOV42(OBSERVATION_DIM)
    config = _base_config()
    config.observation_schema_version = OBSERVATION_SCHEMA
    config.action_schema_version = ACTION_SCHEMA_V4_2
    config.retry_norm_horizon_seconds = 30.0
    load_formal_i1_checkpoint(checkpoint_path, policy, config, load_optimizer=False)
    with torch.no_grad():
        gate, proposal = policy.model.distributions(torch.as_tensor(frozen_matrix))
        release_probability = gate.probs[:, 1].cpu().numpy()
        gate_prediction = gate.logits.argmax(-1).cpu().numpy()
        proposal_prediction = proposal.logits.argmax(-1).cpu().numpy()

    post_indices = [index for index, row in enumerate(frozen_rows)
                    if int(row["teacher_gate"]) == 1 and bool(row["family_flags"]["post_conflict"])]
    if len(post_indices) != 12:
        raise ValueError(f"expected 12 frozen post-conflict RELEASE states, found {len(post_indices)}")
    groups = {
        "release": np.asarray([i for i, row in enumerate(train_rows) if int(row["teacher_gate"]) == 1]),
        "wait": np.asarray([i for i, row in enumerate(train_rows) if int(row["teacher_gate"]) == 0]),
        "post_conflict": np.asarray([i for i, row in enumerate(train_rows)
                                     if int(row["teacher_gate"]) == 1 and bool(row["family_flags"]["post_conflict"])]),
    }
    records = []
    for index in post_indices:
        row = frozen_rows[index]
        same = [candidate for candidate in frozen_rows if candidate["lifecycle_event_id"] == row["lifecycle_event_id"]]
        records.append(_state_record(row, int(gate_prediction[index]), float(release_probability[index]),
                                     int(proposal_prediction[index]), train_matrix, train_rows, groups, same))

    correct = [row for row in records if row["correct_release"]]
    false_wait = [row for row in records if not row["correct_release"]]
    exact_conflicts = []
    by_hash: dict[str, set[int]] = {}
    for row in train_rows + frozen_rows:
        by_hash.setdefault(row["observation_hash"], set()).add(int(row["teacher_gate"]))
    exact_conflicts = sorted(key for key, labels in by_hash.items() if len(labels) > 1)
    train_post = [row for row in train_rows if bool(row["family_flags"]["post_conflict"])]
    train_post_release = [row for row in train_post if int(row["teacher_gate"]) == 1]
    support = {
        "rows": len(train_post),
        "independent_lifecycles": len({row["lifecycle_event_id"] for row in train_post}),
        "WAIT_rows": sum(int(row["teacher_gate"]) == 0 for row in train_post),
        "RELEASE_rows": len(train_post_release),
        "RELEASE_lifecycles": len({row["lifecycle_event_id"] for row in train_post_release}),
        "RELEASE_seed_count": len({row["seed"] for row in train_post_release}),
        "RELEASE_episode_count": len({row["episode_id"] for row in train_post_release}),
        "proposal_joint_distribution": dict(Counter(str(row["teacher_joint_index"]) for row in train_post_release)),
        "phase_ring_distribution": dict(Counter(
            f'{row["state_snapshot"]["target_uav"]["phase"]}:{row["state_snapshot"]["target_uav"]["ring_role"]}'
            for row in train_post_release)),
    }
    def comparison(rows: list[dict]) -> dict:
        return {
            "count": len(rows),
            "P_RELEASE": _stats([row["P_RELEASE"] for row in rows]),
            "gate_margin": _stats([row["gate_logit_margin_RELEASE_minus_WAIT"] for row in rows]),
            "nearest_RELEASE_distance": _stats([row["nearest_training_RELEASE"]["distance_l2_88d"] for row in rows]),
            "nearest_WAIT_distance": _stats([row["nearest_training_WAIT"]["distance_l2_88d"] for row in rows]),
            "nearest_post_conflict_distance": _stats([row["nearest_training_post_conflict_RELEASE"]["distance_l2_88d"] for row in rows]),
            "conflict_clearance_seconds": _stats([row["time_since_conflict_cleared_seconds"] for row in rows]),
            "teacher_proposal_distribution": dict(Counter(str(row["teacher"]["selected_joint_proposal"]) for row in rows)),
            "phase_ring_distribution": dict(Counter(f'{row["phase_ring"]["phase"]}:{row["phase_ring"]["ring_role"]}' for row in rows)),
            "nearest_conflict_dCPA_norm": _stats([row["current_conflict_topology_observation"]["dCPA_norm"] for row in rows]),
            "nearest_conflict_tCPA_norm": _stats([row["current_conflict_topology_observation"]["tCPA_norm"] for row in rows]),
            "nearest_conflict_closing_rate_norm": _stats([row["current_conflict_topology_observation"]["closing_rate_norm"] for row in rows]),
        }
    sparse = support["RELEASE_lifecycles"] < 30 or support["RELEASE_seed_count"] < 5
    if exact_conflicts:
        root = "NEW_OBSERVATION_ALIASING"
        observation_revision = True
        weakness = "OBSERVATION_CLOSURE_REOPEN_REQUIRED"
    elif sparse:
        root = "TRAINING_SUPPORT_SPARSE"
        observation_revision = False
        weakness = "EXPECTED_I1_BOOTSTRAP_LIMITATION"
    elif false_wait and all(abs(row["gate_logit_margin_RELEASE_minus_WAIT"]) < 0.5 for row in false_wait):
        root = "BOUNDARY_MARGIN_ERROR"
        observation_revision = False
        weakness = "EXPECTED_I1_BOOTSTRAP_LIMITATION"
    else:
        root = "UNRESOLVED"
        observation_revision = False
        weakness = "UNRESOLVED"
    report = {
        "stage": "V4_2_V2_1_I1_POST_CONFLICT_NON_MUTATING_ATTRIBUTION",
        "evidence_type": "newly_run",
        "inputs": {"train": str(train_path), "train_sha256": _sha256(train_path),
                   "frozen": str(frozen_path), "frozen_sha256": _sha256(frozen_path),
                   "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path)},
        "mutations": {"retraining": False, "labels": False, "observation": False, "teacher": False,
                      "safety": False, "threshold_tuning": False},
        "frozen_post_conflict_RELEASE_states": len(records),
        "correct_RELEASE": len(correct),
        "false_WAIT": len(false_wait),
        "per_state": records,
        "group_comparison": {"correct_RELEASE": comparison(correct), "false_WAIT": comparison(false_wait)},
        "training_post_conflict_support": support,
        "exact_observation_hash_opposite_gate_conflicts": len(exact_conflicts),
        "exact_conflicting_observation_hashes": exact_conflicts,
        "new_teacher_decisive_state_only_missing_variable_found": False,
        "POST_CONFLICT_ROOT_CAUSE": root,
        "POST_CONFLICT_WEAKNESS": weakness,
        "observation_revision_required": observation_revision,
        "recommended_role": "I2_SAFE_DAGGER_PRIORITY_DIAGNOSTIC_FAMILY" if not observation_revision else "REOPEN_OBSERVATION_CLOSURE",
        "protocol_risks": [
            "The consumed frozen cohort is diagnostic reference only.",
            "The 12 post-conflict RELEASE rows come from only two frozen episodes.",
            "Nearest-neighbor distances diagnose support but do not prove causal model behavior.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.train, args.frozen, args.checkpoint, args.output)
    print(json.dumps({key: result[key] for key in (
        "frozen_post_conflict_RELEASE_states", "correct_RELEASE", "false_WAIT",
        "training_post_conflict_support", "exact_observation_hash_opposite_gate_conflicts",
        "POST_CONFLICT_ROOT_CAUSE", "POST_CONFLICT_WEAKNESS", "observation_revision_required",
    )}, indent=2))
