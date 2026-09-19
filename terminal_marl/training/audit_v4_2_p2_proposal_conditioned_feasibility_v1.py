"""Offline action-conditioned feasibility audit on frozen flow12 proposal sets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from terminal_marl.core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from terminal_marl.training import audit_v4_2_p2_gate_proposal_set_learnability_v1 as set_audit
from terminal_marl.training import diagnose_v4_2_p2_gate_proposal_set_failure_v1 as failure


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_proposal_conditioned_feasibility_audit_v1.yaml")
PROTOCOL_SHA = "68694cefa2e8907b3d4beb77ca3abc2daf569bc9b5ffab6b46b1d70b9888a896"
STAGE = "V4_2_P2_PROPOSAL_CONDITIONED_FEASIBILITY_AUDIT_V1"
FAILURE_MANIFEST_SHA = "4a1d6d9d9b594c1d680203e8c8b728e08101ec1fdfc26aa64028be9f800c5e10"
PROBE_SEEDS = (799101, 799102, 799103)
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 1.0
THRESHOLDS = {
    "macro_f1_min": 0.80, "roc_auc_min": 0.90,
    "feasible_recall_min": 0.85, "infeasible_recall_min": 0.75,
    "top1_feasible_min": 0.90, "partial_top1_min": 0.80,
    "interface_top1_min": 0.85,
}
PROPOSAL_FEATURES = np.asarray([
    [float(item.speed_residual), float(item.merge_time_residual)]
    for item in JOINT_ACTION_CATALOG_V4_1 if item.index > 0
], dtype=np.float32)
if PROPOSAL_FEATURES.shape != (25, 2):
    raise RuntimeError("V4.1 RELEASE catalogue must expose 25 proposal feature pairs")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def protocol() -> dict[str, Any]:
    _require(PROTOCOL.is_file(), f"protocol missing: {PROTOCOL}")
    _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "conditioned feasibility protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict) and value.get("protocol") == STAGE, "protocol id changed")
    bindings = value.get("bindings", {})
    for key, expected in {
        "failure_attribution_manifest_sha256": FAILURE_MANIFEST_SHA,
        "failure_attribution_protocol_sha256": failure.PROTOCOL_SHA,
        "failure_classification": "observation_or_set_target_learnability_limit_supported",
        "support_manifest_sha256": set_audit.SUPPORT_MANIFEST_SHA,
        "support_train_sha256": set_audit.SUPPORT_TRAIN_SHA,
        "support_dev_sha256": set_audit.SUPPORT_DEV_SHA,
        "observation_dimension": 88, "proposal_feature_dimension": 2,
        "proposal_features": ["speed_residual", "merge_time_residual"],
        "proposal_classes": 25, "no_environment": True, "no_teacher": True,
        "no_policy_update": True,
    }.items():
        _require(bindings.get(key) == expected, f"binding changed: {key}")
    probe = value.get("probe", {})
    for key, expected in {
        "seeds": list(PROBE_SEEDS), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "objective": "train_split_balanced_binary_cross_entropy",
        "normalization": "train_split_mean_std_only",
        "state_grouped_train_dev_split": True, "final_epoch_only": True,
        "early_stopping": False, "threshold_tuning": False, "dev_gradients": False,
    }.items():
        actual = probe.get(key)
        _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                 f"probe changed: {key}")
    evaluation = value.get("evaluation", {})
    binary, state = evaluation.get("binary", {}), evaluation.get("state_ranking", {})
    configured = {
        "macro_f1_min": binary.get("macro_f1_min"), "roc_auc_min": binary.get("roc_auc_min"),
        "feasible_recall_min": binary.get("feasible_recall_min"),
        "infeasible_recall_min": binary.get("infeasible_recall_min"),
        "top1_feasible_min": state.get("top1_feasible_min"),
        "partial_top1_min": state.get("partial_set_top1_feasible_min"),
        "interface_top1_min": state.get("every_interface_top1_feasible_min"),
    }
    for key, expected in THRESHOLDS.items():
        _require(float(configured[key]) == expected, f"threshold changed: {key}")
    _require(evaluation.get("all_three_mlp_seeds_required") is True, "seed rule changed")
    return value


def validate_failure_manifest(path: str | Path) -> dict[str, Any]:
    _require(_sha256(path) == FAILURE_MANIFEST_SHA, "failure manifest SHA mismatch")
    value = _read_json(path)
    for key, expected in {
        "stage": failure.STAGE, "status": "COMPLETE",
        "classification": "observation_or_set_target_learnability_limit_supported",
        "standalone_mlp_pass": False,
        "candidate_checkpoint_sha256": set_audit.CANDIDATE_SHA,
        "PROPOSAL_REPAIR_ALLOWED": False, "PPO_ALLOWED": False,
    }.items():
        _require(value.get(key) == expected, f"failure manifest changed: {key}")
    return value


def expand_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    observations, target_mask = set_audit.set_tensors(rows)
    repeated = observations[:, None, :].expand(-1, 25, -1)
    proposal = torch.as_tensor(PROPOSAL_FEATURES)[None, :, :].expand(len(rows), -1, -1)
    features = torch.cat((repeated, proposal), dim=-1).reshape(-1, 90)
    return features, target_mask.reshape(-1).float()


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    _require(logits.shape == targets.shape and len(logits) > 0, "binary metrics shape mismatch")
    prediction = logits >= 0.0
    positive = targets == 1
    negative = ~positive
    tp = int((prediction & positive).sum())
    tn = int((~prediction & negative).sum())
    fp = int((prediction & negative).sum())
    fn = int((~prediction & positive).sum())
    positive_precision = tp / max(1, tp + fp)
    negative_precision = tn / max(1, tn + fn)
    positive_recall = tp / max(1, tp + fn)
    negative_recall = tn / max(1, tn + fp)
    positive_f1 = 2 * positive_precision * positive_recall / max(1e-12, positive_precision + positive_recall)
    negative_f1 = 2 * negative_precision * negative_recall / max(1e-12, negative_precision + negative_recall)
    order = torch.argsort(logits)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(logits) + 1, dtype=torch.float64)
    positives, negatives = int(positive.sum()), int(negative.sum())
    auc = float((ranks[positive].sum() - positives * (positives + 1) / 2) / max(1, positives * negatives))
    return {
        "rows": int(len(targets)), "positive": positives, "negative": negatives,
        "accuracy": (tp + tn) / len(targets), "macro_f1": (positive_f1 + negative_f1) / 2,
        "feasible_recall": positive_recall, "infeasible_recall": negative_recall,
        "roc_auc": auc, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def state_ranking_metrics(logits: torch.Tensor, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(logits.shape == (len(rows), 25), "state ranking shape mismatch")
    _, target_mask = set_audit.set_tensors(rows)
    top1 = logits.argmax(dim=-1)
    hit = target_mask.gather(1, top1[:, None]).squeeze(1)
    partial = target_mask.sum(dim=-1) < 25
    _require(bool(partial.any()), "partial proposal sets required")
    result: dict[str, Any] = {
        "states": len(rows), "top1_feasible": float(hit.float().mean()),
        "partial_states": int(partial.sum()),
        "partial_top1_feasible": float(hit[partial].float().mean()),
        "interface": {},
    }
    for name in set_audit.support.INTERFACES:
        indices = torch.as_tensor([index for index, row in enumerate(rows) if row.get("interface") == name])
        _require(bool(len(indices)), f"dev missing interface {name}")
        result["interface"][name] = {
            "states": int(len(indices)), "top1_feasible": float(hit[indices].float().mean()),
        }
    return result


class LinearProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(90, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(90, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


def run_probe(
    model_type: str, train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]], *, seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_x, train_y = expand_rows(train_rows)
    dev_x, dev_y = expand_rows(dev_rows)
    mean, scale = train_x.mean(dim=0), train_x.std(dim=0).clamp_min(1e-6)
    train_x, dev_x = (train_x - mean) / scale, (dev_x - mean) / scale
    model: nn.Module = LinearProbe() if model_type == "linear" else MLPProbe()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    positive_weight = float((train_y == 0).sum() / (train_y == 1).sum().clamp_min(1))
    optimizer_steps = 0
    final_loss = None
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(train_y))
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            logits = model(train_x[index])
            loss = F.binary_cross_entropy_with_logits(
                logits, train_y[index], pos_weight=logits.new_tensor(positive_weight),
            )
            _require(bool(torch.isfinite(loss)), "non-finite conditioned feasibility loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            _require(bool(torch.isfinite(norm)), "non-finite conditioned feasibility gradient")
            optimizer.step()
            optimizer_steps += 1
            final_loss = float(loss.detach())
    model.eval()
    with torch.no_grad():
        dev_logits = model(dev_x)
    binary = binary_metrics(dev_logits, dev_y)
    state = state_ranking_metrics(dev_logits.reshape(len(dev_rows), 25), dev_rows)
    interface = {
        name: float(state["interface"][name]["top1_feasible"]) >= THRESHOLDS["interface_top1_min"]
        for name in set_audit.support.INTERFACES
    }
    checks = {
        "macro_f1": float(binary["macro_f1"]) >= THRESHOLDS["macro_f1_min"],
        "roc_auc": float(binary["roc_auc"]) >= THRESHOLDS["roc_auc_min"],
        "feasible_recall": float(binary["feasible_recall"]) >= THRESHOLDS["feasible_recall_min"],
        "infeasible_recall": float(binary["infeasible_recall"]) >= THRESHOLDS["infeasible_recall_min"],
        "top1_feasible": float(state["top1_feasible"]) >= THRESHOLDS["top1_feasible_min"],
        "partial_top1_feasible": float(state["partial_top1_feasible"]) >= THRESHOLDS["partial_top1_min"],
        "interface": interface, "every_interface_top1": all(interface.values()),
    }
    checks["all"] = all(bool(value) for key, value in checks.items() if key != "interface")
    return {
        "model": model_type, "seed": int(seed), "epochs": EPOCHS,
        "optimizer_steps": optimizer_steps, "final_loss": final_loss,
        "train_positive_weight": positive_weight, "binary_dev": binary,
        "state_ranking_dev": state, "checks": checks, "checkpoint_saved": False,
    }


def run(
    output: str | Path, failure_manifest: str | Path,
    support_train: str | Path, support_dev: str | Path,
) -> dict[str, Any]:
    protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"output already exists: {output_path}")
    validate_failure_manifest(failure_manifest)
    _require(_sha256(support_train) == set_audit.SUPPORT_TRAIN_SHA, "support train SHA mismatch")
    _require(_sha256(support_dev) == set_audit.SUPPORT_DEV_SHA, "support dev SHA mismatch")
    output_path.mkdir(parents=True, exist_ok=False)
    train_rows = set_audit.normalize_set_rows(set_audit._read_jsonl(support_train), split="train")
    dev_rows = set_audit.normalize_set_rows(set_audit._read_jsonl(support_dev), split="dev")
    linear_runs, mlp_runs = [], []
    for seed in PROBE_SEEDS:
        linear_runs.append(run_probe("linear", train_rows, dev_rows, seed=seed))
        mlp_runs.append(run_probe("mlp", train_rows, dev_rows, seed=seed))
        _write_json(output_path / "progress.json", {
            "stage": STAGE, "completed_seeds": [item["seed"] for item in mlp_runs],
            "planned_seeds": list(PROBE_SEEDS), "last_mlp_checks": mlp_runs[-1]["checks"],
        })
    complete = len(linear_runs) == len(mlp_runs) == len(PROBE_SEEDS)
    passed = bool(complete and all(item["checks"]["all"] for item in mlp_runs))
    result = {
        "stage": STAGE, "status": "PASS" if passed else "FAIL", "evidence_type": "newly_run",
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA,
        "failure_manifest": str(failure_manifest), "failure_manifest_sha256": FAILURE_MANIFEST_SHA,
        "support_train": str(support_train), "support_train_sha256": set_audit.SUPPORT_TRAIN_SHA,
        "support_dev": str(support_dev), "support_dev_sha256": set_audit.SUPPORT_DEV_SHA,
        "rows": {"train_states": len(train_rows), "dev_states": len(dev_rows),
                 "train_state_proposals": len(train_rows) * 25, "dev_state_proposals": len(dev_rows) * 25},
        "proposal_features": PROPOSAL_FEATURES, "linear_runs": linear_runs, "mlp_runs": mlp_runs,
        "thresholds": dict(THRESHOLDS), "AUDIT_COMPLETE": complete,
        "proposal_conditioned_feasibility_learnable": passed,
        "factorized_proposal_architecture_preregistration_allowed": passed,
        "environment_called": False, "teacher_called": False,
        "policy_training_called": False, "policy_optimizer_called": False,
        "diagnostic_probe_optimizer_called": True, "checkpoint_saved": False,
        "dev_gradients": False, "threshold_tuning": False,
        "PROPOSAL_REPAIR_ALLOWED": False, "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False, "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False, "checkpoint_promotable": False,
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    manifest_path = output_path / "manifest.json"
    result["manifest"] = str(manifest_path)
    _write_json(manifest_path, result)
    print(json.dumps({
        "stage": STAGE, "status": result["status"], "AUDIT_COMPLETE": complete,
        "proposal_conditioned_feasibility_learnable": passed,
        "factorized_proposal_architecture_preregistration_allowed": passed,
        "PROPOSAL_REPAIR_ALLOWED": False, "PPO_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-manifest", type=Path, required=True)
    parser.add_argument("--support-train", type=Path, required=True)
    parser.add_argument("--support-dev", type=Path, required=True)
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    _main()
