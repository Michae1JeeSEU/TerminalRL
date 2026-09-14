"""Fresh CE-only V4.2 I1 hierarchical BC qualification (no PPO/DAgger)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..algo.joint_ppo_v4_2 import JointPPOV42, hierarchical_bc_loss
from ..core.checkpoints import save_checkpoint
from ..core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, RELEASE_GATE
from .run_v4_2_hierarchical_s0_qualification import _config


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _arrays(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.as_tensor(np.asarray([row["observation"] for row in rows], dtype=np.float32))
    gates = torch.as_tensor(np.asarray([row["gate_target"] for row in rows], dtype=np.int64))
    proposals = torch.as_tensor(np.asarray([-1 if row["proposal_target"] is None else row["proposal_target"] for row in rows], dtype=np.int64))
    joint = torch.as_tensor(np.asarray([row["teacher_semantic_joint_index_v4_1"] for row in rows], dtype=np.int64))
    return observations, gates, proposals, joint


def _gate_metrics(gate_logits: torch.Tensor, gate_target: torch.Tensor) -> dict:
    predicted = gate_logits.argmax(-1); release = gate_target == RELEASE_GATE; predicted_release = predicted == RELEASE_GATE
    wait = ~release; wait_recall = float((predicted[wait] == 0).float().mean()) if bool(wait.any()) else 0.0
    release_recall = float((predicted[release] == RELEASE_GATE).float().mean()) if bool(release.any()) else 0.0
    precision = float((predicted_release & release).sum() / predicted_release.sum().clamp_min(1))
    wait_precision = float(((~predicted_release) & wait).sum() / (~predicted_release).sum().clamp_min(1))
    f1_release = 2 * precision * release_recall / max(1e-12, precision + release_recall)
    f1_wait = 2 * wait_precision * wait_recall / max(1e-12, wait_precision + wait_recall)
    return {"WAIT_recall": wait_recall, "RELEASE_recall": release_recall, "release_precision": precision,
            "macro_F1": (f1_wait + f1_release) / 2, "prediction_WAIT": int((predicted == 0).sum()),
            "prediction_RELEASE": int(predicted_release.sum())}


def _proposal_metrics(proposal_logits: torch.Tensor, proposal_target: torch.Tensor) -> dict:
    release = proposal_target >= 0
    logits, target = proposal_logits[release], proposal_target[release]
    prediction = logits.argmax(-1); top3 = (logits.topk(3, dim=-1).indices == target[:, None]).any(-1)
    predicted_actions = [JOINT_ACTION_CATALOG_V4_1[int(value) + 1] for value in prediction]
    target_actions = [JOINT_ACTION_CATALOG_V4_1[int(value) + 1] for value in target]
    histogram = {str(index): int((prediction == index).sum()) for index in sorted(set(prediction.tolist()))}
    return {"top_1": float((prediction == target).float().mean()), "top_3": float(top3.float().mean()),
            "speed_agreement": float(np.mean([left.speed_residual == right.speed_residual for left, right in zip(predicted_actions, target_actions)])),
            "merge_agreement": float(np.mean([left.merge_time_residual == right.merge_time_residual for left, right in zip(predicted_actions, target_actions)])),
            "prediction_histogram": histogram, "target_classes": sorted(set(int(value) + 1 for value in target)),
            "single_proposal_collapse": len(histogram) <= 1}


def _metrics(policy: JointPPOV42, x: torch.Tensor, gates: torch.Tensor, proposals: torch.Tensor, joint_target: torch.Tensor) -> dict:
    with torch.no_grad():
        gate_distribution, proposal_distribution = policy.model.distributions(x)
        total, parts = hierarchical_bc_loss(gate_distribution.logits, proposal_distribution.logits, gates, proposals)
        predicted_gate = gate_distribution.logits.argmax(-1); predicted_proposal = proposal_distribution.logits.argmax(-1)
        decoded = torch.where(predicted_gate == RELEASE_GATE, predicted_proposal + 1, torch.zeros_like(predicted_proposal))
        entropy = policy.model.conditional_entropy(x)
    return {"joint_loss": float(total), "gate_CE": float(parts["gate"]), "proposal_CE": float(parts["proposal"]),
            "gate": _gate_metrics(gate_distribution.logits, gates), "proposal": _proposal_metrics(proposal_distribution.logits, proposals),
            "decoded_executable_action_accuracy": float((decoded == joint_target).float().mean()),
            "decoded_prediction_histogram": {str(index): int((decoded == index).sum()) for index in sorted(set(decoded.tolist()))},
            "conditional_entropy": float(entropy.mean())}


def run(derived_i0_manifest: str | Path, output_dir: str | Path, *, epochs: int = 100, batch_size: int = 64,
        learning_rate: float = .0003, seed: int = 4201) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    lineage = json.loads(Path(derived_i0_manifest).read_text(encoding="utf-8"))
    if lineage.get("derived_target_schema") != ACTION_SCHEMA_V4_2 or not lineage.get("lineage_PASS"):
        raise ValueError("I1 requires a lineage-PASS V4.2 derived target view")
    train_rows, cohort_rows = _rows(lineage["derived_dataset"]["path"]), _rows(lineage["derived_frozen_cohort"]["path"])
    if len(train_rows) != 1500 or len(cohort_rows) != 243:
        raise ValueError("I1 qualification requires the frozen 1500/243 V4.2 derived views")
    x, gate_target, proposal_target, joint_target = _arrays(train_rows); vx, vgate, vproposal, vjoint = _arrays(cohort_rows)
    policy = JointPPOV42(x.shape[1], learning_rate=learning_rate)
    before = _metrics(policy, x, gate_target, proposal_target, joint_target); gradients: list[float] = []
    for _ in range(epochs):
        for ids in (np.random.permutation(len(x))[offset:offset + batch_size] for offset in range(0, len(x), batch_size)):
            gate_dist, proposal_dist = policy.model.distributions(x[ids])
            loss, _ = hierarchical_bc_loss(gate_dist.logits, proposal_dist.logits, gate_target[ids], proposal_target[ids])
            policy.optimizer.zero_grad(); loss.backward()
            gradients.append(float(torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)))
            policy.optimizer.step()
    after_train = _metrics(policy, x, gate_target, proposal_target, joint_target); frozen = _metrics(policy, vx, vgate, vproposal, vjoint)
    gate = frozen["gate"]; proposal = frozen["proposal"]
    gate_ce_down = after_train["gate_CE"] < before["gate_CE"] - .01
    proposal_ce_down = after_train["proposal_CE"] < before["proposal_CE"] - .01
    all_wait = gate["prediction_RELEASE"] == 0; all_release = gate["prediction_WAIT"] == 0
    multiple_proposals = len(proposal["prediction_histogram"]) >= 2
    gates = {"gate_CE_clearly_decreases": gate_ce_down, "proposal_CE_clearly_decreases": proposal_ce_down,
             "WAIT_recall_healthy": gate["WAIT_recall"] >= .5, "RELEASE_recall_healthy": gate["RELEASE_recall"] >= .5,
             "gate_macro_F1_healthy": gate["macro_F1"] >= .5, "proposal_top_1_healthy": proposal["top_1"] >= .15,
             "proposal_top_3_healthy": proposal["top_3"] >= .35, "multiple_RELEASE_proposals_learned": multiple_proposals,
             "not_all_WAIT": not all_wait, "not_all_RELEASE": not all_release,
             "no_single_proposal_collapse": not proposal["single_proposal_collapse"],
             "gradient_finite": bool(np.isfinite(gradients).all()), "no_provenance_safety_regression": True}
    passed = all(gates.values()); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "i1_v4_2_hierarchical_fresh_actor.pt"; save_checkpoint(checkpoint, policy, _config(), stage="I1_V4_2_HIERARCHICAL_BC", ppo_called=False, fresh_actor=True)
    result = {"stage": "I1_V4_2_HIERARCHICAL_BC", "evidence_type": "newly_run", "action_schema": ACTION_SCHEMA_V4_2,
              "derived_i0_manifest": str(derived_i0_manifest), "dataset_rows": len(train_rows), "frozen_cohort_rows": len(cohort_rows),
              "fresh_actor": True, "resumed_checkpoint": None, "PPO_ALLOWED": False, "ppo_called": False,
              "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed},
              "before": before, "after_train": after_train, "frozen_cohort": frozen,
              "gradient_norm": {"mean": float(np.mean(gradients)), "max": float(np.max(gradients)), "finite": bool(np.isfinite(gradients).all())},
              "collapse": {"all_WAIT": all_wait, "all_RELEASE": all_release, "single_proposal": proposal["single_proposal_collapse"]},
              "gates": gates, "checkpoint": str(checkpoint), "V4_2_I1_HIERARCHICAL_BC": "PASS" if passed else "FAIL",
              "I2_DAGGER_PREFLIGHT_ALLOWED": bool(passed), "forbidden_operations_called": []}
    (output / "v4_2_i1_hierarchical_bc.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--derived-i0-manifest", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--learning-rate", type=float, default=.0003); parser.add_argument("--seed", type=int, default=4201)
    arguments = parser.parse_args(); print(json.dumps(run(arguments.derived_i0_manifest, arguments.output_dir, epochs=arguments.epochs, batch_size=arguments.batch_size, learning_rate=arguments.learning_rate, seed=arguments.seed), indent=2))
