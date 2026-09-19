"""Offline attribution for the failed proposal-set learnability audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from terminal_marl.training import audit_v4_2_p2_gate_proposal_set_learnability_v1 as audit
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_proposal_set_failure_attribution_v1.yaml")
PROTOCOL_SHA = "f2e145734c267742ca13093c39df4e1534c6f7855413e07fff5d94ba35bbaa26"
STAGE = "V4_2_P2_GATE_PROPOSAL_SET_FAILURE_ATTRIBUTION_V1"
AUDIT_MANIFEST_SHA = "a1b76fa16427cecdc0c70e0d38c6247d8e6e6d943063aeeddd00a8cede28e937"
PROBE_SEEDS = (798101, 798102, 798103)
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 1.0
TOP1_MIN = 0.80
TOP3_MIN = 0.93
PARTIAL_TOP1_MIN = 0.70
INTERFACE_TOP1_MIN = 0.70


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
    _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "set failure-attribution protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict) and value.get("protocol") == STAGE, "protocol id changed")
    bindings = value.get("bindings", {})
    for key, expected in {
        "audit_manifest_sha256": AUDIT_MANIFEST_SHA,
        "audit_protocol_sha256": audit.PROTOCOL_SHA, "audit_status": "FAIL",
        "support_manifest_sha256": audit.SUPPORT_MANIFEST_SHA,
        "support_train_sha256": audit.SUPPORT_TRAIN_SHA,
        "support_dev_sha256": audit.SUPPORT_DEV_SHA,
        "candidate_checkpoint_sha256": audit.CANDIDATE_SHA,
        "observation_dimension": audit.OBSERVATION_DIM,
        "proposal_classes": audit.N_PROPOSALS,
    }.items():
        _require(bindings.get(key) == expected, f"binding changed: {key}")
    probes = value.get("standalone_probes", {})
    for key, expected in {
        "seeds": list(PROBE_SEEDS), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "loss": "proposal_set_negative_log_mass", "final_epoch_only": True,
        "early_stopping": False, "threshold_tuning": False, "dev_gradients": False,
    }.items():
        actual = probes.get(key)
        _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                 f"probe binding changed: {key}")
    classification = value.get("classification", {})
    for key, expected in {
        "standalone_top1_feasible_min": TOP1_MIN,
        "standalone_top3_feasible_min": TOP3_MIN,
        "standalone_partial_top1_feasible_min": PARTIAL_TOP1_MIN,
        "every_interface_top1_feasible_min": INTERFACE_TOP1_MIN,
    }.items():
        _require(float(classification.get(key)) == expected, f"classification changed: {key}")
    _require(classification.get("all_three_mlp_seeds_required") is True, "seed rule changed")
    return value


def validate_audit_manifest(path: str | Path) -> dict[str, Any]:
    _require(_sha256(path) == AUDIT_MANIFEST_SHA, "audit manifest SHA mismatch")
    value = _read_json(path)
    for key, expected in {
        "stage": audit.STAGE, "status": "FAIL", "AUDIT_COMPLETE": True,
        "proposal_set_learnability_pass": False,
        "proposal_set_repair_preregistration_allowed": False,
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
    observations, target_mask = audit.set_tensors(rows)
    model.eval()
    with torch.no_grad():
        logits = model(observations)
    result: dict[str, Any] = audit.set_metrics(logits, target_mask)
    if by_interface:
        result["interface"] = {}
        for name in audit.support.INTERFACES:
            indices = [index for index, row in enumerate(rows) if row.get("interface") == name]
            _require(bool(indices), f"dev missing interface {name}")
            selected = torch.as_tensor(indices, dtype=torch.long)
            result["interface"][name] = audit.set_metrics(logits[selected], target_mask[selected])
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
    observations, target_mask = audit.set_tensors(train_rows)
    final_loss = None
    optimizer_steps = 0
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(observations))
        for start in range(0, len(observations), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            loss = audit.set_nll(model(observations[index]), target_mask[index])
            _require(bool(torch.isfinite(loss)), "non-finite standalone set probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            _require(bool(torch.isfinite(norm)), "non-finite standalone set probe gradient")
            optimizer.step()
            optimizer_steps += 1
            final_loss = float(loss.detach())
    metrics = evaluate_probe(model, dev_rows)
    checks = {
        "top1": float(metrics["top1_feasible"]) >= TOP1_MIN,
        "top3": float(metrics["top3_feasible_hit"]) >= TOP3_MIN,
        "partial_top1": float(metrics["partial_top1_feasible"]) >= PARTIAL_TOP1_MIN,
        "interface": {
            name: float(metrics["interface"][name]["top1_feasible"]) >= INTERFACE_TOP1_MIN
            for name in audit.support.INTERFACES
        },
    }
    checks["every_interface_top1"] = all(checks["interface"].values())
    checks["all"] = checks["top1"] and checks["top3"] and checks["partial_top1"] and checks["every_interface_top1"]
    return {
        "model": model_type, "seed": int(seed), "epochs": EPOCHS,
        "optimizer_steps": optimizer_steps, "final_loss": final_loss,
        "dev": metrics, "checks": checks, "checkpoint_saved": False,
    }


def exact_hash_set_conflicts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        key = str(row.get("observation_hash"))
        groups[key].add(tuple(int(value) for value in row["proposal_set_target"]))
        counts[key] += 1
    conflicts = {key: values for key, values in groups.items() if len(values) > 1}
    disjoint = {
        key for key, values in conflicts.items()
        if any(not set(left) & set(right) for left in values for right in values if left != right)
    }
    return {
        "unique_hashes": len(groups), "conflicting_hashes": len(conflicts),
        "disjoint_conflicting_hashes": len(disjoint),
        "rows_in_conflicting_hashes": sum(counts[key] for key in conflicts),
    }


def candidate_feasible_rank(policy: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observations, target_mask = audit.set_tensors(rows)
    logits = audit.exact_audit.base_logits(policy, observations)
    ranks = logits.argsort(dim=-1, descending=True)
    ordered_feasible = target_mask.gather(1, ranks)
    best = ordered_feasible.float().argmax(dim=1) + 1
    logp = torch.log_softmax(logits.double(), dim=-1)
    mass = torch.logsumexp(logp.masked_fill(~target_mask, float("-inf")), dim=-1).exp()
    quantiles = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        "best_feasible_rank_quantiles": {
            name: float(value) for name, value in zip(
                ("q0", "q10", "q50", "q90", "q100"), torch.quantile(best.float(), quantiles).tolist(),
            )
        },
        "feasible_mass_quantiles": {
            name: float(value) for name, value in zip(
                ("q0", "q10", "q50", "q90", "q100"), torch.quantile(mass.float(), quantiles).tolist(),
            )
        },
    }


def nearest_neighbor_set_overlap(
    reference_rows: Sequence[Mapping[str, Any]], query_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_x, reference_mask = audit.set_tensors(reference_rows)
    query_x, query_mask = audit.set_tensors(query_rows)
    mean = reference_x.mean(dim=0)
    scale = reference_x.std(dim=0).clamp_min(1e-6)
    reference = (reference_x - mean) / scale
    query = (query_x - mean) / scale
    best_distances, best_indices = [], []
    for start in range(0, len(query), 128):
        distances = torch.cdist(query[start:start + 128], reference)
        values, indices = distances.min(dim=1)
        best_distances.append(values)
        best_indices.append(indices)
    distance = torch.cat(best_distances)
    neighbor_mask = reference_mask[torch.cat(best_indices)]
    intersection = (neighbor_mask & query_mask).sum(dim=-1).float()
    union = (neighbor_mask | query_mask).sum(dim=-1).float()
    jaccard = intersection / union.clamp_min(1.0)
    quantiles = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        "queries": int(len(query)),
        "any_intersection": float((intersection > 0).float().mean()),
        "jaccard_mean": float(jaccard.mean()),
        "jaccard_quantiles": {
            name: float(value) for name, value in zip(
                ("q0", "q10", "q50", "q90", "q100"), torch.quantile(jaccard, quantiles).tolist(),
            )
        },
        "distance_quantiles": {
            name: float(value) for name, value in zip(
                ("q0", "q10", "q50", "q90", "q100"), torch.quantile(distance, quantiles).tolist(),
            )
        },
    }


def run(
    output: str | Path, audit_manifest: str | Path,
    support_train: str | Path, support_dev: str | Path,
    candidate_checkpoint: str | Path,
) -> dict[str, Any]:
    protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"output already exists: {output_path}")
    validate_audit_manifest(audit_manifest)
    _require(_sha256(support_train) == audit.SUPPORT_TRAIN_SHA, "support train SHA mismatch")
    _require(_sha256(support_dev) == audit.SUPPORT_DEV_SHA, "support dev SHA mismatch")
    policy, _payload = effectiveness.load_candidate_checkpoint(candidate_checkpoint)
    output_path.mkdir(parents=True, exist_ok=False)
    train_rows = audit.normalize_set_rows(audit._read_jsonl(support_train), split="train")
    dev_rows = audit.normalize_set_rows(audit._read_jsonl(support_dev), split="dev")
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
        "D1_anchor_or_residual_adapter_constraint_supported" if mlp_pass
        else "observation_or_set_target_learnability_limit_supported"
    )
    state_after = audit._state_digest(policy)
    result = {
        "stage": STAGE, "status": "COMPLETE", "evidence_type": "newly_run",
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA,
        "audit_manifest": str(audit_manifest), "audit_manifest_sha256": AUDIT_MANIFEST_SHA,
        "support_train": str(support_train), "support_train_sha256": audit.SUPPORT_TRAIN_SHA,
        "support_dev": str(support_dev), "support_dev_sha256": audit.SUPPORT_DEV_SHA,
        "candidate_checkpoint": str(candidate_checkpoint), "candidate_checkpoint_sha256": audit.CANDIDATE_SHA,
        "rows": {"train": len(train_rows), "dev": len(dev_rows)},
        "exact_hash_set_conflicts": {
            "train": exact_hash_set_conflicts(train_rows), "dev": exact_hash_set_conflicts(dev_rows),
        },
        "candidate_feasible_rank": candidate_feasible_rank(policy, dev_rows),
        "nearest_neighbor": {"dev_to_train": nearest_neighbor_set_overlap(train_rows, dev_rows)},
        "linear_runs": linear_runs, "mlp_runs": mlp_runs,
        "classification": classification, "standalone_mlp_pass": mlp_pass,
        "candidate_state_before": state_before, "candidate_state_after": state_after,
        "candidate_state_unchanged": state_before == state_after,
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
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    _main()
