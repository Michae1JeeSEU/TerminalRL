"""Offline V4.1 Flat-26 versus hierarchical learnability diagnostic.

This does not alter the official action schema, Teacher, observations, I0
dataset, BC loss, class weighting, Safety, reward, PPO, or deployment code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..algo.joint_ppo_v4_1 import JointActorCentralCriticV41, JointPPOV41
from ..algo.shared_ppo import mlp
from ..core.checkpoints import load_checkpoint
from ..core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from .low_v4_1_imitation import _config


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _unique_lifecycle(rows: list[dict]) -> list[dict]:
    """One deterministic, real state per semantic lifecycle event."""
    groups: dict[str, dict] = {}
    for row in rows:
        key = row["lifecycle_event_id"]
        if key not in groups or (row.get("semantic_sampling_reason") == "event_start" and groups[key].get("semantic_sampling_reason") != "event_start"):
            groups[key] = row
    return [groups[key] for key in sorted(groups)]


def _arrays(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.as_tensor(np.asarray([row["observation"] for row in rows], dtype=np.float32)),
            torch.as_tensor(np.asarray([row["teacher_joint_index"] for row in rows], dtype=np.int64)))


def _binary_metrics(prob_release: np.ndarray, target_release: np.ndarray) -> dict:
    prediction = prob_release >= .5
    wait = ~target_release
    def recall(mask, expected): return float((prediction[mask] == expected).mean()) if np.any(mask) else 0.0
    wait_recall, release_recall = recall(wait, False), recall(target_release, True)
    precision = float(np.logical_and(prediction, target_release).sum() / max(1, prediction.sum()))
    wait_precision = float(np.logical_and(~prediction, wait).sum() / max(1, (~prediction).sum()))
    f1r = 2 * precision * release_recall / max(1e-12, precision + release_recall)
    f1w = 2 * wait_precision * wait_recall / max(1e-12, wait_precision + wait_recall)
    # Mann-Whitney rank AUC without a sklearn dependency.
    ranks = np.argsort(np.argsort(prob_release)).astype(float) + 1.0
    positive = int(target_release.sum()); negative = len(target_release) - positive
    auc = float((ranks[target_release].sum() - positive * (positive + 1) / 2) / max(1, positive * negative))
    return {"WAIT_recall": wait_recall, "RELEASE_recall": release_recall, "release_precision": precision,
            "macro_F1": (f1r + f1w) / 2, "ROC_AUC_rank": auc,
            "P_RELEASE_given_Teacher_WAIT": float(prob_release[wait].mean()) if np.any(wait) else 0.0,
            "P_RELEASE_given_Teacher_RELEASE": float(prob_release[target_release].mean()) if np.any(target_release) else 0.0,
            "predicted_WAIT": int((~prediction).sum()), "predicted_RELEASE": int(prediction.sum())}


def _train(model: nn.Module, x: torch.Tensor, target: torch.Tensor, loss_fn, *, epochs: int = 36, batch: int = 64) -> tuple[float, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    with torch.no_grad(): before = float(loss_fn(model, x, target))
    for _ in range(epochs):
        for ids in (np.random.permutation(len(x))[start:start + batch] for start in range(0, len(x), batch)):
            loss = loss_fn(model, x[ids], target[ids]); optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    with torch.no_grad(): after = float(loss_fn(model, x, target))
    return before, after


class BinaryGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__(); self.encoder = mlp(dim, 96, 96); self.logits = nn.Linear(96, 2)
    def forward(self, x): return self.logits(self.encoder(x))


class ReleaseProposal(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__(); self.encoder = mlp(dim, 96, 96); self.logits = nn.Linear(96, 25)
    def forward(self, x): return self.logits(self.encoder(x))


class HierarchicalPrototype(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__(); self.encoder = mlp(dim, 96, 96); self.gate = nn.Linear(96, 2); self.proposal = nn.Linear(96, 25)
    def forward(self, x):
        encoded = self.encoder(x); return self.gate(encoded), self.proposal(encoded)


def _proposal_metrics(logits: torch.Tensor, target_joint: torch.Tensor) -> dict:
    target = target_joint - 1; prediction = logits.argmax(-1)
    top3 = float((logits.topk(3, dim=-1).indices == target[:, None]).any(-1).float().mean())
    predicted_catalogue = [JOINT_ACTION_CATALOG_V4_1[int(value) + 1] for value in prediction]
    target_catalogue = [JOINT_ACTION_CATALOG_V4_1[int(value) + 1] for value in target]
    return {"top_1": float((prediction == target).float().mean()), "top_3": top3,
            "speed_agreement": float(np.mean([a.speed_residual == b.speed_residual for a, b in zip(predicted_catalogue, target_catalogue)])),
            "merge_agreement": float(np.mean([a.merge_time_residual == b.merge_time_residual for a, b in zip(predicted_catalogue, target_catalogue)])),
            "classes_represented": sorted(set(int(value) + 1 for value in target))}


def _flat_metrics(model, x, target) -> dict:
    with torch.no_grad():
        logits = model.actor_distribution(x).logits; probs = logits.softmax(-1); prediction = logits.argmax(-1)
    release_total = probs[:, 1:].sum(-1).cpu().numpy(); wait = probs[:, 0].cpu().numpy(); truth_release = (target > 0).cpu().numpy()
    binary = _binary_metrics(release_total, truth_release)
    return {"joint_top_1": float((prediction == target).float().mean()), "joint_top_3": float((logits.topk(3, dim=-1).indices == target[:, None]).any(-1).float().mean()),
            "binary": binary, "P_RELEASE_TOTAL_mean": float(release_total.mean()), "P_WAIT_mean": float(wait.mean()),
            "prediction_histogram": {str(index): int((prediction == index).sum()) for index in sorted(set(prediction.cpu().tolist()))},
            "single_class_collapse": len(set(prediction.cpu().tolist())) == 1}


def _flat_probability_audit(checkpoint: str | Path, dim: int, cohort: list[dict]) -> dict:
    policy = JointPPOV41(dim); load_checkpoint(checkpoint, policy, _config(__import__("terminal_marl.training.low_v4_1_imitation", fromlist=["SOURCES"]).SOURCES[0]), load_optimizer=False)
    x, target = _arrays(cohort)
    with torch.no_grad(): probs = policy.model.actor_distribution(x).probs
    wait = probs[:, 0].numpy(); release = probs[:, 1:].sum(-1).numpy(); best, best_class = probs[:, 1:].max(-1)
    result = {}
    for name, mask in {"Teacher_WAIT": target.numpy() == 0, "Teacher_RELEASE": target.numpy() > 0}.items():
        result[name] = {"P_WAIT_mean": float(wait[mask].mean()), "P_RELEASE_TOTAL_mean": float(release[mask].mean()),
                        "release_margin_mean": float((release[mask] - wait[mask]).mean()),
                        "best_RELEASE_probability_mean": float(best.numpy()[mask].mean()),
                        "best_RELEASE_class_histogram": {str(int(k) + 1): int(v) for k, v in zip(*np.unique(best_class.numpy()[mask], return_counts=True))}}
    return result


def _separability(rows: list[dict]) -> dict:
    vectors = np.asarray([row["observation"] for row in rows], dtype=np.float32); labels = np.asarray([row["teacher_joint_index"] > 0 for row in rows])
    exact: dict[str, set[bool]] = defaultdict(set); states: dict[str, set[bool]] = defaultdict(set)
    for row, label in zip(rows, labels):
        exact[hashlib.sha256(np.asarray(row["observation"], dtype=np.float32).tobytes()).hexdigest()].add(bool(label)); states[row["state_hash"]].add(bool(label))
    left, right = vectors[~labels], vectors[labels]
    distances = np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).sum(-1)) if len(left) and len(right) else np.asarray([])
    nearest = np.concatenate((distances.min(1), distances.min(0))) if distances.size else np.asarray([float("inf")])
    return {"binary_exact_hash_conflicts": sum(len(value) > 1 for value in exact.values()), "same_full_state_opposite_labels": sum(len(value) > 1 for value in states.values()),
            "nearest_opposite_label_distance": {"min": float(nearest.min()), "p50": float(np.percentile(nearest, 50)), "p95": float(np.percentile(nearest, 95))}}


def run(*, i0_manifest: str | Path, failed_i1_checkpoint: str | Path, output_dir: str | Path) -> dict:
    torch.manual_seed(6101); np.random.seed(6101); manifest = json.loads(Path(i0_manifest).read_text())
    train, cohort = _unique_lifecycle(_rows(manifest["dataset_path"])), _unique_lifecycle(_rows(manifest["fixed_decision_state_cohort_path"]))
    x, target = _arrays(train); vx, vy = _arrays(cohort); dim = x.shape[1]
    wait, release = [row for row in train if row["teacher_joint_index"] == 0], [row for row in train if row["teacher_joint_index"] > 0]
    balanced = sorted(wait, key=lambda row: row["lifecycle_event_id"])[:len(release)] + release
    bx, by_joint = _arrays(balanced); by_gate = (by_joint > 0).long(); gate_target = (target > 0).long()
    separability = _separability(train + cohort); flat_failed = _flat_probability_audit(failed_i1_checkpoint, dim, cohort)
    # A1 / A2 binary gate.
    natural = BinaryGate(dim); a1_before, a1_after = _train(natural, x, gate_target, lambda m,a,b: F.cross_entropy(m(a), b));
    balanced_gate = BinaryGate(dim); a2_before, a2_after = _train(balanced_gate, bx, by_gate, lambda m,a,b: F.cross_entropy(m(a), b))
    with torch.no_grad(): a1 = _binary_metrics(natural(vx).softmax(-1)[:, 1].numpy(), (vy > 0).numpy()); a2 = _binary_metrics(balanced_gate(vx).softmax(-1)[:, 1].numpy(), (vy > 0).numpy())
    # B release conditional classifier.
    rx, ry = _arrays(release); vr_mask = vy > 0; proposal = ReleaseProposal(dim); b_before, b_after = _train(proposal, rx, ry - 1, lambda m,a,b: F.cross_entropy(m(a), b))
    with torch.no_grad(): b_metrics = _proposal_metrics(proposal(vx[vr_mask]), vy[vr_mask])
    # C flat-26 on the same balanced unique-event states as A2.
    flat = JointActorCentralCriticV41(dim); c_before, c_after = _train(flat, bx, by_joint, lambda m,a,b: F.cross_entropy(m.actor_distribution(a).logits, b))
    c_metrics = _flat_metrics(flat, vx, vy)
    # D hierarchical prototype, conditional proposal CE only on real RELEASE rows.
    hierarchical = HierarchicalPrototype(dim)
    def d_loss(m, values, labels):
        gate, proposal_logits = m(values); release_mask = labels > 0
        gate_loss = F.cross_entropy(gate, release_mask.long())
        proposal_loss = F.cross_entropy(proposal_logits[release_mask], labels[release_mask] - 1) if bool(release_mask.any()) else torch.zeros((), device=values.device)
        return gate_loss + proposal_loss
    d_before, d_after = _train(hierarchical, bx, by_joint, d_loss)
    with torch.no_grad():
        gate_logits, proposal_logits = hierarchical(vx); gate_prob = gate_logits.softmax(-1)[:, 1].numpy(); d_binary = _binary_metrics(gate_prob, (vy > 0).numpy()); d_proposal = _proposal_metrics(proposal_logits[vr_mask], vy[vr_mask]); gate_prediction = gate_logits.argmax(-1); proposal_prediction = proposal_logits.argmax(-1) + 1; decoded = torch.where(gate_prediction > 0, proposal_prediction, torch.zeros_like(proposal_prediction))
    d_metrics = {"binary": d_binary, "proposal": d_proposal, "joint_decoded_accuracy": float((decoded == vy).float().mean()), "prediction_histogram": {str(index): int((decoded == index).sum()) for index in sorted(set(decoded.tolist()))}}
    binary_ok = a2["WAIT_recall"] > .5 and a2["RELEASE_recall"] > .5 and a2["macro_F1"] > .5
    proposal_ok = b_metrics["top_1"] > .15 and b_metrics["top_3"] > .35
    flat_ok = c_metrics["binary"]["WAIT_recall"] > .5 and c_metrics["binary"]["RELEASE_recall"] > .5 and not c_metrics["single_class_collapse"]
    hierarchical_ok = d_binary["WAIT_recall"] > .5 and d_binary["RELEASE_recall"] > .5 and d_proposal["top_3"] > .35
    if not binary_ok: root, recommendation = "OBSERVATION", "undecided"
    elif proposal_ok and hierarchical_ok and not flat_ok: root, recommendation = "FLAT_26_FACTORIZATION", "V4.2_hierarchical"
    elif flat_ok: root, recommendation = "DATA_DISTRIBUTION", "V4.1_flat26"
    else: root, recommendation = "DATA_DISTRIBUTION", "undecided"
    ppo_preflight = {"scope": "semantic design pre-audit only; no V4.2 code migration or PPO run was performed",
                     "applies_if_recommended": recommendation == "V4.2_hierarchical",
                     "WAIT_requested_action_logprob": "log P(WAIT | observation)",
                     "RELEASE_requested_action_logprob": "log P(RELEASE | observation) + log P(release_proposal | observation, RELEASE)",
                     "ratio_provenance": "Store and reuse the sampled requested action and its joint log-prob only; do not use teacher, execution, applied, or shield actions in the PPO ratio.",
                     "execution_boundary": "The scheduler receives exactly one decoded requested proposal; it must not search the release catalogue.",
                     "status": "PASS (design semantics only)"}
    result = {"evidence_type": "newly_run", "diagnostic_only": True, "official_data_modified": False, "i0_manifest": str(i0_manifest), "failed_i1_checkpoint": str(failed_i1_checkpoint),
              "split": {"train": "one deterministic state per I0-R lifecycle event", "validation": "one deterministic state per frozen cohort lifecycle event", "train_events": len(train), "validation_events": len(cohort), "balanced_events": len(balanced)},
              "flat26_failed_checkpoint_probability": flat_failed, "observation_separability": separability,
              "probe_A_binary": {"A1_natural": {"CE_before": a1_before, "CE_after": a1_after, "validation": a1}, "A2_event_balanced": {"CE_before": a2_before, "CE_after": a2_after, "validation": a2}},
              "probe_B_release_proposal": {"CE_before": b_before, "CE_after": b_after, "validation": b_metrics},
              "probe_C_flat26_balanced": {"CE_before": c_before, "CE_after": c_after, "validation": c_metrics},
              "probe_D_hierarchical_offline": {"CE_before": d_before, "CE_after": d_after, "validation": d_metrics},
              "binary_release_separable": binary_ok, "release_proposal_learnable": proposal_ok, "flat26_learnable": flat_ok, "hierarchical_probe_learnable": hierarchical_ok,
              "ROOT_CAUSE": root, "recommended_action_schema": recommendation, "I0_recollection_allowed": False, "I1_allowed": False, "PPO_ALLOWED": False,
              "ppo_provenance_preflight": ppo_preflight,
              "protocol_risks": ["single deterministic seed per offline probe", "frozen cohort is internal validation, not an external test", "event-balanced A2/C are diagnostic only and are not formal I0 training data"]}
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True); (destination / "v4_1_flat26_vs_hierarchical_learnability_diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--i0-manifest", required=True); parser.add_argument("--failed-i1-checkpoint", required=True); parser.add_argument("--output-dir", required=True); args = parser.parse_args()
    print(json.dumps(run(i0_manifest=args.i0_manifest, failed_i1_checkpoint=args.failed_i1_checkpoint, output_dir=args.output_dir), indent=2))
