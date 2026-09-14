"""Read-only diagnosis of the V4.2 DAgger-R1 frozen RELEASE-retention failure.

This module never rewrites D1, Teacher labels, checkpoints used by the formal
qualification, or a qualification threshold.  Its newly written checkpoints
are diagnostic snapshots only and cannot promote Round 1 or authorize Round 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from ..algo.joint_ppo_v4_2 import JointPPOV42, hierarchical_bc_loss
from ..core.checkpoints import load_checkpoint, save_checkpoint
from ..core.low_level_joint_action_v4_2 import RELEASE_GATE
from .run_v4_2_dagger_round1 import _config_for_steps, _rates, _validate_closed_loop
from .run_v4_2_i1_hierarchical_bc import _arrays, _metrics, _rows


BOUNDARY_LOW = 0.40
BOUNDARY_HIGH = 0.60


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_policy(checkpoint: str | Path, *, learning_rate: float) -> JointPPOV42:
    policy = JointPPOV42(87, learning_rate=learning_rate)
    load_checkpoint(checkpoint, policy, _config_for_steps(10), load_optimizer=True)
    policy.model.eval()  # Match formal R1 before its CE-only fine-tuning.
    return policy


def _prediction_details(policy: JointPPOV42, x: torch.Tensor) -> dict[str, np.ndarray]:
    with torch.no_grad():
        gate, proposal = policy.model.distributions(x)
        probabilities = gate.probs[:, RELEASE_GATE].cpu().numpy()
        margin = (gate.logits[:, RELEASE_GATE] - gate.logits[:, 0]).cpu().numpy()
        predicted_gate = gate.logits.argmax(-1).cpu().numpy()
        predicted_proposal = proposal.logits.argmax(-1).cpu().numpy()
    predicted_joint = np.where(predicted_gate == RELEASE_GATE, predicted_proposal + 1, 0)
    return {"release_probability": probabilities, "gate_margin": margin, "gate": predicted_gate,
            "proposal": predicted_proposal, "joint": predicted_joint}


def _d1_disagreement(policy: JointPPOV42, x: torch.Tensor, gate: torch.Tensor, proposal: torch.Tensor, joint: torch.Tensor) -> dict:
    prediction = _prediction_details(policy, x)
    targets = joint.cpu().numpy()
    gate_targets = gate.cpu().numpy()
    release = gate_targets == RELEASE_GATE
    return {
        "gate_disagreement": float(np.mean(prediction["gate"] != gate_targets)),
        "decoded_action_disagreement": float(np.mean(prediction["joint"] != targets)),
        "proposal_disagreement_on_teacher_RELEASE": float(np.mean(prediction["proposal"][release] != proposal.cpu().numpy()[release])) if release.any() else 0.0,
    }


def _snapshot_context(row: dict) -> dict:
    snapshot = row.get("state_snapshot", {})
    uid = int(row.get("uav_id", -1))
    uav = next((item for item in snapshot.get("uavs", []) if int(item.get("uid", -2)) == uid), {})
    return {
        "phase": row.get("phase"), "lifecycle": row.get("lifecycle"), "uav_count": len(snapshot.get("uavs", [])),
        "target_uav": {key: uav.get(key) for key in ("uid", "ring_role", "ring_progress", "speed", "committed", "reservation_ids", "last_release_decision", "last_safety_decision")},
        "release_state": snapshot.get("release_state", {}), "reservations": snapshot.get("reservations", []),
        "future_release_reservations": snapshot.get("future_release_reservations", {}),
    }


def _nearest_d1(row: dict, d1_rows: list[dict], d1_observations: np.ndarray, *, count: int = 5) -> list[dict]:
    observation = np.asarray(row["observation"], dtype=np.float32)
    distances = np.linalg.norm(d1_observations - observation, axis=1) / np.sqrt(observation.size)
    nearest = np.argsort(distances)[:count]
    return [{"distance": float(distances[index]), "decision_event_id": d1_rows[index].get("decision_event_id"),
             "source": d1_rows[index].get("source"), "seed": d1_rows[index].get("seed"),
             "state_hash": d1_rows[index].get("state_hash"), "teacher_gate": d1_rows[index].get("gate_target"),
             "teacher_proposal": d1_rows[index].get("proposal_target"),
             "sampling_reason": d1_rows[index].get("semantic_sampling_reason")} for index in nearest]


def _frozen_transitions(i1: JointPPOV42, r1: JointPPOV42, rows: list[dict], d1_rows: list[dict]) -> dict:
    x, gate_target, _, _ = _arrays(rows)
    first, second = _prediction_details(i1, x), _prediction_details(r1, x)
    release = gate_target.cpu().numpy() == RELEASE_GATE
    i1_release, r1_release = first["gate"] == RELEASE_GATE, second["gate"] == RELEASE_GATE
    selected = release & i1_release & ~r1_release
    d1_observations = np.asarray([row["observation"] for row in d1_rows], dtype=np.float32)
    flips = []
    for index in np.where(selected)[0]:
        row = rows[int(index)]
        boundary = bool(BOUNDARY_LOW <= first["release_probability"][index] <= BOUNDARY_HIGH or
                        BOUNDARY_LOW <= second["release_probability"][index] <= BOUNDARY_HIGH)
        flips.append({
            "frozen_row": int(index), "decision_event_id": row.get("decision_event_id"), "state_hash": row.get("state_hash"),
            "source": row.get("source"), "scenario": row.get("scenario_class"), "seed": row.get("seed"),
            "teacher_gate": int(gate_target[index]), "teacher_proposal": row.get("proposal_target"),
            "teacher_joint": row.get("teacher_semantic_joint_index_v4_1"), "I1_P_RELEASE": float(first["release_probability"][index]),
            "R1_P_RELEASE": float(second["release_probability"][index]), "I1_gate_margin": float(first["gate_margin"][index]),
            "R1_gate_margin": float(second["gate_margin"][index]), "boundary_by_probability_0_4_to_0_6": boundary,
            "topology": _snapshot_context(row), "nearest_D1_events": _nearest_d1(row, d1_rows, d1_observations),
        })
    return {
        "teacher_RELEASE_rows": int(release.sum()),
        "confusion_transition_on_teacher_RELEASE": {
            "I1_correct_RELEASE_to_R1_incorrect_WAIT": int((release & i1_release & ~r1_release).sum()),
            "I1_correct_RELEASE_to_R1_correct_RELEASE": int((release & i1_release & r1_release).sum()),
            "I1_incorrect_WAIT_to_R1_correct_RELEASE": int((release & ~i1_release & r1_release).sum()),
            "I1_incorrect_WAIT_to_R1_incorrect_WAIT": int((release & ~i1_release & ~r1_release).sum()),
        },
        "flipped_release_states": flips,
        "flipped_release_state_count": len(flips),
        "boundary_definition": f"either checkpoint P(RELEASE) in [{BOUNDARY_LOW:.2f}, {BOUNDARY_HIGH:.2f}]",
        "boundary_state_fraction": float(sum(item["boundary_by_probability_0_4_to_0_6"] for item in flips) / max(1, len(flips))),
    }


def _diagnostic_metrics(policy: JointPPOV42, frozen: tuple[torch.Tensor, ...], d1: tuple[torch.Tensor, ...]) -> dict:
    # Sampling during the label-only closed-loop evaluator must not perturb the
    # subsequent CE minibatch trajectory.  Snapshot both RNGs around every
    # intermediate observation so an epoch-N checkpoint has exactly the same
    # training history as a standalone N-epoch retraining with this seed.
    torch_rng, numpy_rng = torch.get_rng_state(), np.random.get_state()
    try:
        fixed = _metrics(policy, *frozen)
        d1_disagreement = _d1_disagreement(policy, *d1)
        torch.manual_seed(7101); np.random.seed(7101)
        student, safety, label_only = _validate_closed_loop(policy, _config_for_steps(8))
        return {"frozen_cohort": fixed, "D1_teacher_disagreement": d1_disagreement,
                "student_state_validation": {"diagnostics": student, "rates": _rates(student), "safety": safety, "teacher_label_only": label_only}}
    finally:
        torch.set_rng_state(torch_rng); np.random.set_state(numpy_rng)


def _train_with_snapshots(checkpoint: str | Path, d1: tuple[torch.Tensor, ...], frozen: tuple[torch.Tensor, ...], output: Path, *, epochs: int, batch_size: int, learning_rate: float, seed: int) -> list[dict]:
    policy = _load_policy(checkpoint, learning_rate=learning_rate)
    x, gate, proposal, _ = d1
    torch.manual_seed(seed); np.random.seed(seed)
    checkpoints = sorted(set((0, 2, max(1, round(epochs * .25)), max(1, round(epochs * .50)), max(1, round(epochs * .75)), epochs)))
    result = []
    checkpoint_dir = output / "trajectory_checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs + 1):
        if epoch in checkpoints:
            path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(path, policy, _config_for_steps(10), stage="V4_2_DAGGER_R1_DIAGNOSTIC_ONLY", ppo_called=False,
                            diagnostic_only=True, parent_checkpoint_sha256=_sha256(checkpoint))
            result.append({"epoch": epoch, "checkpoint": str(path), **_diagnostic_metrics(policy, frozen, d1)})
        if epoch == epochs:
            break
        for ids in (np.random.permutation(len(x))[offset:offset + batch_size] for offset in range(0, len(x), batch_size)):
            gate_dist, proposal_dist = policy.model.distributions(x[ids])
            loss, _ = hierarchical_bc_loss(gate_dist.logits, proposal_dist.logits, gate[ids], proposal[ids])
            policy.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0); policy.optimizer.step()
    return result


def _seed_audit(checkpoint: str | Path, d1: tuple[torch.Tensor, ...], frozen: tuple[torch.Tensor, ...], *, epochs: int, batch_size: int, learning_rate: float, seeds: tuple[int, ...]) -> list[dict]:
    x, gate, proposal, _ = d1
    result = []
    for seed in seeds:
        policy = _load_policy(checkpoint, learning_rate=learning_rate)
        torch.manual_seed(seed); np.random.seed(seed)
        for _ in range(epochs):
            for ids in (np.random.permutation(len(x))[offset:offset + batch_size] for offset in range(0, len(x), batch_size)):
                gate_dist, proposal_dist = policy.model.distributions(x[ids])
                loss, _ = hierarchical_bc_loss(gate_dist.logits, proposal_dist.logits, gate[ids], proposal[ids])
                policy.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0); policy.optimizer.step()
        metric = _diagnostic_metrics(policy, frozen, d1)
        fixed = metric["frozen_cohort"]
        result.append({"seed": seed, "WAIT_recall": fixed["gate"]["WAIT_recall"], "RELEASE_recall": fixed["gate"]["RELEASE_recall"],
                       "macro_F1": fixed["gate"]["macro_F1"], "decoded_accuracy": fixed["decoded_executable_action_accuracy"],
                       "D1_teacher_disagreement": metric["D1_teacher_disagreement"],
                       "student_state_disagreement": metric["student_state_validation"]["rates"],
                       "proposal": fixed["proposal"]})
    return result


def _aggregate_seed_audit(rows: list[dict]) -> dict:
    names = ("WAIT_recall", "RELEASE_recall", "macro_F1", "decoded_accuracy")
    return {name: {"mean": float(np.mean([row[name] for row in rows])), "std": float(np.std([row[name] for row in rows])),
                   "min": float(np.min([row[name] for row in rows])), "max": float(np.max([row[name] for row in rows]))} for name in names}


def _root_cause(trajectory: list[dict], seed_rows: list[dict], *, i1_release_recall: float) -> tuple[str, dict]:
    retention_minimum = i1_release_recall - .05
    full = next(row for row in trajectory if row["epoch"] == max(item["epoch"] for item in trajectory))
    initial = next(row for row in trajectory if row["epoch"] == 0)
    early = [row for row in trajectory if 0 < row["epoch"] < full["epoch"]]
    seed_passes = sum(row["RELEASE_recall"] >= retention_minimum for row in seed_rows)
    early_pass_with_lower_d1_disagreement = any(
        row["frozen_cohort"]["gate"]["RELEASE_recall"] >= retention_minimum and
        row["D1_teacher_disagreement"]["decoded_action_disagreement"] < initial["D1_teacher_disagreement"]["decoded_action_disagreement"]
        for row in early)
    full_fails = full["frozen_cohort"]["gate"]["RELEASE_recall"] < retention_minimum
    all_nonzero_drop = all(row["frozen_cohort"]["gate"]["RELEASE_recall"] < i1_release_recall for row in trajectory if row["epoch"] > 0)
    all_seed_fail = seed_passes == 0
    if seed_passes >= 4 and full_fails:
        root = "RETRAINING_VARIANCE"
    elif early_pass_with_lower_d1_disagreement and full_fails:
        root = "OVER_FINETUNING"
    elif all_nonzero_drop and all_seed_fail:
        root = "D1_VS_ANCHOR_DISTRIBUTION_CONFLICT"
    else:
        root = "UNRESOLVED"
    return root, {"retention_minimum": retention_minimum, "seed_pass_count": seed_passes, "seed_count": len(seed_rows),
                  "early_pass_with_lower_D1_disagreement": early_pass_with_lower_d1_disagreement,
                  "full_trajectory_fails_retention": full_fails, "all_nonzero_trajectory_points_below_I1_release_recall": all_nonzero_drop,
                  "all_seed_runs_fail_retention": all_seed_fail}


def run(i1_checkpoint: str | Path, r1_checkpoint: str | Path, d1_path: str | Path, frozen_cohort: str | Path, output_dir: str | Path, *, epochs: int = 35, batch_size: int = 64, learning_rate: float = .0003, seeds: tuple[int, ...] = (4301, 4302, 4303, 4304, 4305)) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    d1_rows, frozen_rows = _rows(d1_path), _rows(frozen_cohort)
    d1, frozen = _arrays(d1_rows), _arrays(frozen_rows)
    i1, r1 = _load_policy(i1_checkpoint, learning_rate=learning_rate), _load_policy(r1_checkpoint, learning_rate=learning_rate)
    transitions = _frozen_transitions(i1, r1, frozen_rows, d1_rows)
    trajectory = _train_with_snapshots(i1_checkpoint, d1, frozen, output, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seeds[0])
    seed_rows = _seed_audit(i1_checkpoint, d1, frozen, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seeds=seeds)
    i1_frozen = _metrics(i1, *frozen)
    root, evidence = _root_cause(trajectory, seed_rows, i1_release_recall=i1_frozen["gate"]["RELEASE_recall"])
    result = {
        "stage": "V4_2_DAGGER_R1_RELEASE_RETENTION_REGRESSION_DIAGNOSTIC", "evidence_type": "newly_run",
        "inputs": {"i1_checkpoint": str(i1_checkpoint), "i1_checkpoint_sha256": _sha256(i1_checkpoint), "r1_checkpoint": str(r1_checkpoint),
                   "r1_checkpoint_sha256": _sha256(r1_checkpoint), "D1": str(d1_path), "D1_sha256": _sha256(d1_path),
                   "frozen_cohort": str(frozen_cohort), "frozen_cohort_sha256": _sha256(frozen_cohort)},
        "rows": {"D1": len(d1_rows), "frozen_cohort": len(frozen_rows)},
        "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "retraining_seeds": list(seeds)},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
        "formal_round1_comparison": {"I1_frozen": i1_frozen, "R1_frozen": _metrics(r1, *frozen), "transitions": transitions},
        "trajectory": trajectory, "seed_audit": {"runs": seed_rows, "aggregate": _aggregate_seed_audit(seed_rows)},
        "ROOT_CAUSE": root, "root_cause_evidence": evidence,
        "round1_protocol_revision_required": True, "round1_rerun_allowed": False,
        "V4_2_DAGGER_R1": "FAIL", "I2_DAGGER_ROUND2_ALLOWED": False, "PPO_ALLOWED": False,
        "forbidden_operations_called": [],
    }
    (output / "v4_2_dagger_r1_release_retention_diagnostic.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i1-checkpoint", required=True); parser.add_argument("--r1-checkpoint", required=True)
    parser.add_argument("--d1", required=True); parser.add_argument("--frozen-cohort", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=35); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--seeds", default="4301,4302,4303,4304,4305")
    arguments = parser.parse_args(); values = tuple(int(value) for value in arguments.seeds.split(",") if value)
    print(json.dumps(run(arguments.i1_checkpoint, arguments.r1_checkpoint, arguments.d1, arguments.frozen_cohort, arguments.output_dir,
                         epochs=arguments.epochs, batch_size=arguments.batch_size, learning_rate=arguments.learning_rate, seeds=values), indent=2))
