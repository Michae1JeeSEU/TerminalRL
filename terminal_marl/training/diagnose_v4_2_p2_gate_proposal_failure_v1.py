"""Offline attribution for the failed anchored proposal learnability audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from terminal_marl.training import audit_v4_2_p2_gate_proposal_learnability_v1 as audit
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_proposal_failure_attribution_v1.yaml")
PROTOCOL_SHA = "2b39758e62af4aed58c7e5dadc0172e96aa920523d29db2819e0ebc5f6dd8f43"
STAGE = "V4_2_P2_GATE_PROPOSAL_FAILURE_ATTRIBUTION_V1"
AUDIT_MANIFEST_SHA = "87bdf9b5ce0b9d9cbf967d858669a6f17679c0cf271d83c8c39e7f4fa4e52e95"
PROBE_SEEDS = (793101, 793102, 793103)
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 1.0
TOP1_MIN = 0.60
TOP3_MIN = 0.80
INTERFACE_TOP1_MIN = 0.50


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
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "failure-attribution protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict) and value.get("protocol") == STAGE, "protocol id changed")
    bindings = value.get("bindings", {})
    for key, expected in {
        "audit_manifest_sha256": AUDIT_MANIFEST_SHA,
        "audit_protocol_sha256": audit.PROTOCOL_SHA,
        "audit_status": "FAIL",
        "support_manifest_sha256": audit.SUPPORT_MANIFEST_SHA,
        "support_train_sha256": audit.SUPPORT_TRAIN_SHA,
        "support_dev_sha256": audit.SUPPORT_DEV_SHA,
        "candidate_checkpoint_sha256": audit.CANDIDATE_SHA,
        "d1_dataset_sha256": audit.D1_DATASET_SHA,
        "observation_dimension": audit.OBSERVATION_DIM,
        "proposal_classes": audit.N_PROPOSALS,
    }.items():
        _require(bindings.get(key) == expected, f"binding changed: {key}")
    probes = value.get("standalone_probes", {})
    for key, expected in {
        "seeds": list(PROBE_SEEDS), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "loss": "unweighted_cross_entropy", "final_epoch_only": True,
        "early_stopping": False, "threshold_tuning": False, "dev_gradients": False,
    }.items():
        actual = probes.get(key)
        _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                 f"probe binding changed: {key}")
    classification = value.get("classification", {})
    _require(float(classification.get("standalone_top1_min")) == TOP1_MIN, "top1 threshold changed")
    _require(float(classification.get("standalone_top3_min")) == TOP3_MIN, "top3 threshold changed")
    _require(float(classification.get("every_interface_top1_min")) == INTERFACE_TOP1_MIN,
             "interface threshold changed")
    _require(classification.get("all_three_mlp_seeds_required") is True, "seed rule changed")
    return value


def validate_audit_manifest(path: str | Path) -> dict[str, Any]:
    _require(_sha256(path) == AUDIT_MANIFEST_SHA, "audit manifest SHA mismatch")
    value = _read_json(path)
    for key, expected in {
        "stage": audit.STAGE, "status": "FAIL", "AUDIT_COMPLETE": True,
        "proposal_learnability_pass": False,
        "proposal_repair_preregistration_allowed": False,
        "PROPOSAL_REPAIR_ALLOWED": False, "PPO_ALLOWED": False,
        "candidate_checkpoint_sha256": audit.CANDIDATE_SHA,
        "support_train_sha256": audit.SUPPORT_TRAIN_SHA,
        "support_dev_sha256": audit.SUPPORT_DEV_SHA,
    }.items():
        _require(value.get(key) == expected, f"audit manifest changed: {key}")
    return value


class LinearProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(audit.OBSERVATION_DIM, audit.N_PROPOSALS)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class MLPProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audit.OBSERVATION_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, audit.N_PROPOSALS),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def evaluate_probe(
    model: nn.Module, rows: Sequence[Mapping[str, Any]], *, by_interface: bool = True,
) -> dict[str, Any]:
    observations, targets = audit.tensors(rows)
    model.eval()
    with torch.no_grad():
        logits = model(observations)
    result: dict[str, Any] = audit.metrics(logits, targets)
    if by_interface:
        result["interface"] = {}
        for name in audit.support.INTERFACES:
            indices = [index for index, row in enumerate(rows) if row.get("interface") == name]
            _require(bool(indices), f"dev missing interface {name}")
            index = torch.as_tensor(indices, dtype=torch.long)
            result["interface"][name] = audit.metrics(logits[index], targets[index])
    return result


def train_probe(
    model_type: str, train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]], *, seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model: nn.Module = LinearProbe() if model_type == "linear" else MLPProbe()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    observations, targets = audit.tensors(train_rows)
    final_loss = None
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(targets))
        for start in range(0, len(targets), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            loss = F.cross_entropy(model(observations[index]), targets[index])
            _require(bool(torch.isfinite(loss)), "non-finite standalone probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            _require(bool(torch.isfinite(norm)), "non-finite standalone probe gradient")
            optimizer.step()
            final_loss = float(loss.detach())
    metrics = evaluate_probe(model, dev_rows)
    checks = {
        "top1": float(metrics["top1"]) >= TOP1_MIN,
        "top3": float(metrics["top3"]) >= TOP3_MIN,
        "interface": {
            name: float(metrics["interface"][name]["top1"]) >= INTERFACE_TOP1_MIN
            for name in audit.support.INTERFACES
        },
    }
    checks["every_interface_top1"] = all(checks["interface"].values())
    checks["all"] = checks["top1"] and checks["top3"] and checks["every_interface_top1"]
    return {
        "model": model_type, "seed": int(seed), "epochs": EPOCHS,
        "optimizer_steps": EPOCHS * math_ceil_div(len(targets), BATCH_SIZE),
        "final_loss": final_loss, "dev": metrics, "checks": checks,
        "checkpoint_saved": False,
    }


def math_ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


def majority_baseline(train_rows: Sequence[Mapping[str, Any]], dev_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(int(row["proposal_target"]) for row in train_rows)
    majority = counts.most_common(1)[0][0]
    targets = [int(row["proposal_target"]) for row in dev_rows]
    return {
        "train_majority_proposal": int(majority),
        "train_majority_fraction": float(counts[majority] / len(train_rows)),
        "dev_accuracy": float(sum(target == majority for target in targets) / len(targets)),
    }


def exact_hash_conflicts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        groups[str(row.get("observation_hash"))].add(int(row["proposal_target"]))
    conflicts = {key: sorted(values) for key, values in groups.items() if len(values) > 1}
    return {
        "unique_hashes": len(groups), "conflicting_hashes": len(conflicts),
        "rows_in_conflicting_hashes": sum(
            sum(str(row.get("observation_hash")) == key for row in rows) for key in conflicts
        ),
    }


def candidate_margin(policy: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observations, targets = audit.tensors(rows)
    logits = audit.base_logits(policy, observations)
    target_logits = logits.gather(1, targets[:, None]).squeeze(1)
    masked = logits.clone()
    masked[torch.arange(len(targets)), targets] = -torch.inf
    margins = (target_logits - masked.max(dim=1).values).numpy()
    quantiles = np.quantile(margins, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        "quantiles": {name: float(value) for name, value in zip(("q0", "q10", "q50", "q90", "q100"), quantiles)},
        "target_is_top1_fraction": float(np.mean(margins >= 0.0)),
        "negative_margin_fraction": float(np.mean(margins < 0.0)),
    }


def nearest_neighbor(
    reference_rows: Sequence[Mapping[str, Any]], query_rows: Sequence[Mapping[str, Any]],
    *, statistics_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_x, reference_y = audit.tensors(reference_rows)
    query_x, query_y = audit.tensors(query_rows)
    stats_x, _ = audit.tensors(statistics_rows)
    mean = stats_x.mean(dim=0)
    scale = stats_x.std(dim=0).clamp_min(1e-6)
    reference = (reference_x - mean) / scale
    query = (query_x - mean) / scale
    best_distances, best_targets = [], []
    for start in range(0, len(query), 128):
        batch = query[start:start + 128]
        distances = torch.cdist(batch, reference)
        values, indices = distances.min(dim=1)
        best_distances.append(values)
        best_targets.append(reference_y[indices])
    distance = torch.cat(best_distances)
    predicted = torch.cat(best_targets)
    return {
        "queries": int(len(query_y)),
        "target_agreement": float((predicted == query_y).float().mean()),
        "distance_quantiles": {
            name: float(value) for name, value in zip(
                ("q0", "q10", "q50", "q90", "q100"),
                torch.quantile(distance, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])).tolist(),
            )
        },
    }


def run(
    output: str | Path, audit_manifest: str | Path,
    support_train: str | Path, support_dev: str | Path,
    candidate_checkpoint: str | Path, d1_dataset: str | Path,
) -> dict[str, Any]:
    protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"output already exists: {output_path}")
    validate_audit_manifest(audit_manifest)
    _require(_sha256(support_train) == audit.SUPPORT_TRAIN_SHA, "support train SHA mismatch")
    _require(_sha256(support_dev) == audit.SUPPORT_DEV_SHA, "support dev SHA mismatch")
    _require(_sha256(d1_dataset) == audit.D1_DATASET_SHA, "D1 dataset SHA mismatch")
    policy, _payload = effectiveness.load_candidate_checkpoint(candidate_checkpoint)
    output_path.mkdir(parents=True, exist_ok=False)
    train_rows = audit._normalize_rows(audit._read_jsonl(support_train), split="train")
    dev_rows = audit._normalize_rows(audit._read_jsonl(support_dev), split="dev")
    d1_rows = audit.load_d1_proposal_rows(d1_dataset)
    state_before = audit._state_digest(policy)
    linear_runs, mlp_runs = [], []
    for seed in PROBE_SEEDS:
        linear_runs.append(train_probe("linear", train_rows, dev_rows, seed=seed))
        mlp_runs.append(train_probe("mlp", train_rows, dev_rows, seed=seed))
        _write_json(output_path / "progress.json", {
            "stage": STAGE, "completed_seeds": [run["seed"] for run in mlp_runs],
            "planned_seeds": list(PROBE_SEEDS), "last_mlp_checks": mlp_runs[-1]["checks"],
        })
    mlp_pass = all(run["checks"]["all"] for run in mlp_runs)
    classification = (
        "base_logit_or_D1_anchor_constraint_supported" if mlp_pass
        else "observation_or_label_learnability_limit_supported"
    )
    result = {
        "stage": STAGE, "status": "COMPLETE", "evidence_type": "newly_run",
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "audit_manifest": str(audit_manifest), "audit_manifest_sha256": AUDIT_MANIFEST_SHA,
        "support_train": str(support_train), "support_train_sha256": audit.SUPPORT_TRAIN_SHA,
        "support_dev": str(support_dev), "support_dev_sha256": audit.SUPPORT_DEV_SHA,
        "candidate_checkpoint": str(candidate_checkpoint), "candidate_checkpoint_sha256": audit.CANDIDATE_SHA,
        "d1_dataset": str(d1_dataset), "d1_dataset_sha256": audit.D1_DATASET_SHA,
        "rows": {"train": len(train_rows), "dev": len(dev_rows), "d1_release": len(d1_rows)},
        "majority_baseline": majority_baseline(train_rows, dev_rows),
        "exact_hash_conflicts": {
            "train": exact_hash_conflicts(train_rows), "dev": exact_hash_conflicts(dev_rows),
        },
        "candidate_target_margin": candidate_margin(policy, dev_rows),
        "nearest_neighbor": {
            "dev_to_train": nearest_neighbor(train_rows, dev_rows, statistics_rows=train_rows),
            "dev_to_D1": nearest_neighbor(d1_rows, dev_rows, statistics_rows=train_rows),
        },
        "linear_runs": linear_runs, "mlp_runs": mlp_runs,
        "classification": classification,
        "standalone_mlp_pass": mlp_pass,
        "candidate_state_before": state_before,
        "candidate_state_after": audit._state_digest(policy),
        "candidate_state_unchanged": state_before == audit._state_digest(policy),
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
        "stage": STAGE, "status": "COMPLETE", "classification": classification,
        "standalone_mlp_pass": mlp_pass, "PROPOSAL_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--support-train", type=Path, required=True)
    parser.add_argument("--support-dev", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--d1-dataset", type=Path, required=True)
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    _main()
