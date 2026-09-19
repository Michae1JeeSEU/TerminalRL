"""Offline learnability audit for set-valued P2 proposal targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from terminal_marl.training import audit_v4_2_p2_gate_proposal_learnability_v1 as exact_audit
from terminal_marl.training import collect_v4_2_p2_gate_proposal_set_support_v1 as support
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_proposal_set_learnability_audit_v1.yaml")
PROTOCOL_SHA = "9dcb54d0b23dc81ad5fabacfb97762e127b3169f7908736540b9b8275e28868f"
STAGE = "V4_2_P2_GATE_PROPOSAL_SET_LEARNABILITY_AUDIT_V1"
SUPPORT_MANIFEST_SHA = "d057d0f39a3a99f6641c331eecd1a127a2b639b5046bf78d39b55d1b8caf7f37"
SUPPORT_TRAIN_SHA = "f5511da554f21614e399b6384609bac93a41e9fdef94b8735eb2ecd6cb5d3f14"
SUPPORT_DEV_SHA = "dd07e8f2b877bd7f08db113aa0fc8988ed8d106223d799284b3e1f8b6324d17a"
CANDIDATE_SHA = effectiveness.REPAIRED_CHECKPOINT_SHA
D1_DATASET_SHA = exact_audit.D1_DATASET_SHA
D1_MANIFEST_SHA = exact_audit.D1_MANIFEST_SHA
OBSERVATION_DIM = 88
N_PROPOSALS = 25
PROBE_SEEDS = (797101, 797102, 797103)
EPOCHS = 50
SUPPORT_BATCH_SIZE = 128
D1_BATCH_SIZE = 128
LEARNING_RATE = 3e-4
GRADIENT_CLIP_NORM = 1.0
ANCHOR_KL_WEIGHT = 1.0
THRESHOLDS = {
    "dev_top1_min": 0.80,
    "dev_top1_gain_min": 0.07,
    "dev_top3_min": 0.93,
    "dev_top3_gain_min": 0.05,
    "partial_top1_min": 0.70,
    "interface_top1_min": 0.70,
    "partial_nll_reduction_min": 0.20,
    "d1_top1_drop_max": 0.03,
    "d1_top3_drop_max": 0.02,
    "d1_nll_increase_max": 0.10,
}


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


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, dict), f"row is not object: {path}:{line_number}")
            rows.append(value)
    _require(bool(rows), f"dataset empty: {path}")
    return rows


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
    _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "proposal-set audit protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict) and value.get("protocol") == STAGE, "protocol id changed")
    bindings = value.get("bindings", {})
    for key, expected in {
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "support_train_sha256": SUPPORT_TRAIN_SHA,
        "support_dev_sha256": SUPPORT_DEV_SHA,
        "support_stage": support.STAGE,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "candidate_architecture_id": effectiveness.ARCHITECTURE_ID,
        "candidate_temperature": effectiveness.TEMPERATURE,
        "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest_sha256": D1_MANIFEST_SHA,
        "observation_schema": effectiveness.OBSERVATION_SCHEMA,
        "observation_dimension": OBSERVATION_DIM,
        "action_schema": effectiveness.ACTION_SCHEMA,
    }.items():
        _require(bindings.get(key) == expected, f"binding changed: {key}")
    requirements = value.get("support_requirements", {})
    for key, expected in {
        "PROPOSAL_SET_SUPPORT_COLLECTION": "PASS",
        "PROPOSAL_SET_SUPPORT_FROZEN": True,
        "PROPOSAL_SET_LEARNABILITY_AUDIT_ALLOWED": True,
        "train_rows": 463, "dev_rows": 372,
        "train_all_gates": True, "dev_all_gates": True,
        "train_dev_seed_disjoint": True, "train_dev_event_disjoint": True,
        "training_called": False, "policy_update": False,
    }.items():
        _require(requirements.get(key) == expected, f"support requirement changed: {key}")
    probe = value.get("probe", {})
    for key, expected in {
        "seeds": list(PROBE_SEEDS), "epochs": EPOCHS,
        "architecture": "frozen_candidate_proposal_logits_plus_residual_MLP_88x64x25",
        "initialization": "zero_output_layer",
        "trainable_scope": "diagnostic_residual_adapter_only",
        "support_batch_size": SUPPORT_BATCH_SIZE, "d1_batch_size": D1_BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "anchor_kl_weight": ANCHOR_KL_WEIGHT,
        "objective": "0.5_support_set_NLL_plus_0.5_D1_exact_CE_plus_1.0_D1_anchor_KL",
        "checkpoint_selection": "final_epoch_only", "early_stopping": False,
        "threshold_tuning": False, "dev_gradients": False, "d1_selection": False,
    }.items():
        actual = probe.get(key)
        _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                 f"probe binding changed: {key}")
    evaluation = value.get("evaluation", {})
    fresh, anchor = evaluation.get("fresh_dev", {}), evaluation.get("d1_anchor", {})
    configured = {
        "dev_top1_min": fresh.get("top1_feasible_min"),
        "dev_top1_gain_min": fresh.get("top1_feasible_gain_min"),
        "dev_top3_min": fresh.get("top3_feasible_min"),
        "dev_top3_gain_min": fresh.get("top3_feasible_gain_min"),
        "partial_top1_min": fresh.get("partial_set_top1_feasible_min"),
        "interface_top1_min": fresh.get("every_interface_top1_feasible_min"),
        "partial_nll_reduction_min": fresh.get("partial_set_nll_reduction_min"),
        "d1_top1_drop_max": anchor.get("top1_drop_max"),
        "d1_top3_drop_max": anchor.get("top3_drop_max"),
        "d1_nll_increase_max": anchor.get("nll_increase_max"),
    }
    for key, expected in THRESHOLDS.items():
        _require(float(configured[key]) == expected, f"threshold changed: {key}")
    _require(evaluation.get("all_three_seeds_required") is True, "seed rule changed")
    return value


def validate_support_manifest(path: str | Path) -> dict[str, Any]:
    _require(_sha256(path) == SUPPORT_MANIFEST_SHA, "support manifest SHA mismatch")
    value = _read_json(path)
    for key, expected in {
        "stage": support.STAGE, "status": "PASS",
        "PROPOSAL_SET_SUPPORT_COLLECTION": "PASS", "PROPOSAL_SET_SUPPORT_FROZEN": True,
        "PROPOSAL_SET_LEARNABILITY_AUDIT_ALLOWED": True,
        "PROPOSAL_REPAIR_ALLOWED": False, "PPO_ALLOWED": False,
        "training_called": False, "optimizer_called": False, "policy_update": False,
    }.items():
        _require(value.get(key) == expected, f"support manifest changed: {key}")
    _require(value.get("train", {}).get("sha256") == SUPPORT_TRAIN_SHA, "support train binding changed")
    _require(value.get("dev", {}).get("sha256") == SUPPORT_DEV_SHA, "support dev binding changed")
    _require(value.get("train", {}).get("gates", {}).get("all") is True
             and value.get("dev", {}).get("gates", {}).get("all") is True,
             "support split gates are not passing")
    _require(all(value.get("cross_split_checks", {}).values()), "support split overlap")
    return value


def normalize_set_rows(rows: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, Any]]:
    result = []
    for index, source in enumerate(rows):
        observation = np.asarray(source.get("observation_v2_1"), dtype=np.float32)
        _require(observation.shape == (OBSERVATION_DIM,) and np.isfinite(observation).all(),
                 f"{split} row {index} observation invalid")
        target = sorted({int(value) for value in source.get("proposal_set_target", ())})
        _require(bool(target) and all(0 <= value < N_PROPOSALS for value in target),
                 f"{split} row {index} proposal set invalid")
        _require(int(source.get("teacher_proposal", -1)) in target,
                 f"{split} row {index} Teacher proposal outside set")
        row = dict(source)
        row["observation_v2_1"] = observation
        row["proposal_set_target"] = target
        result.append(row)
    return result


def set_tensors(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.as_tensor(
        np.asarray([row["observation_v2_1"] for row in rows], dtype=np.float32)
    )
    mask = torch.zeros((len(rows), N_PROPOSALS), dtype=torch.bool)
    for index, row in enumerate(rows):
        mask[index, torch.as_tensor(row["proposal_set_target"], dtype=torch.long)] = True
    return observations, mask


def set_nll(logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    _require(logits.shape == target_mask.shape and target_mask.dtype == torch.bool,
             "set loss shape/dtype mismatch")
    _require(bool(target_mask.any(dim=-1).all()), "set loss received empty target")
    logp = F.log_softmax(logits.double(), dim=-1)
    feasible_log_mass = torch.logsumexp(logp.masked_fill(~target_mask, float("-inf")), dim=-1)
    return -feasible_log_mass.mean().float()


def set_metrics(logits: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float | int | None]:
    _require(len(logits) > 0, "set metrics require rows")
    top = logits.topk(3, dim=-1).indices
    top1 = target_mask.gather(1, top[:, :1]).squeeze(1)
    top3 = target_mask.gather(1, top).any(dim=-1)
    sizes = target_mask.sum(dim=-1)
    partial = sizes < logits.shape[-1]
    result: dict[str, float | int | None] = {
        "rows": int(len(logits)),
        "top1_feasible": float(top1.float().mean()),
        "top3_feasible_hit": float(top3.float().mean()),
        "set_nll": float(set_nll(logits, target_mask)),
        "partial_rows": int(partial.sum()),
    }
    result.update({
        "partial_top1_feasible": float(top1[partial].float().mean()) if bool(partial.any()) else None,
        "partial_top3_feasible_hit": float(top3[partial].float().mean()) if bool(partial.any()) else None,
        "partial_set_nll": float(set_nll(logits[partial], target_mask[partial])) if bool(partial.any()) else None,
    })
    return result


def evaluate_set(
    policy: Any, adapter: exact_audit.ProposalResidualProbe | None,
    rows: Sequence[Mapping[str, Any]], *, by_interface: bool = False,
) -> dict[str, Any]:
    observations, target_mask = set_tensors(rows)
    logits = exact_audit.base_logits(policy, observations)
    if adapter is not None:
        adapter.eval()
        with torch.no_grad():
            logits = logits + adapter(observations)
    result: dict[str, Any] = set_metrics(logits, target_mask)
    if by_interface:
        result["interface"] = {}
        for name in support.INTERFACES:
            indices = [index for index, row in enumerate(rows) if row.get("interface") == name]
            _require(bool(indices), f"fresh dev missing interface {name}")
            selected = torch.as_tensor(indices, dtype=torch.long)
            result["interface"][name] = set_metrics(logits[selected], target_mask[selected])
    return result


def _cyclic_batch(permutation: torch.Tensor, step: int, size: int) -> torch.Tensor:
    start = (step * size) % len(permutation)
    end = start + size
    if end <= len(permutation):
        return permutation[start:end]
    return torch.cat((permutation[start:], permutation[:end - len(permutation)]))


def run_probe(
    policy: Any, train_rows: Sequence[Mapping[str, Any]], dev_rows: Sequence[Mapping[str, Any]],
    d1_rows: Sequence[Mapping[str, Any]], *, seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    adapter = exact_audit.ProposalResidualProbe()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=LEARNING_RATE)
    support_x, support_mask = set_tensors(train_rows)
    d1_x, d1_y = exact_audit.tensors(d1_rows)
    d1_base = exact_audit.base_logits(policy, d1_x)
    steps = max(math.ceil(len(support_x) / SUPPORT_BATCH_SIZE), math.ceil(len(d1_y) / D1_BATCH_SIZE))
    final_components: dict[str, float] = {}
    for _epoch in range(EPOCHS):
        support_order = torch.randperm(len(support_x))
        d1_order = torch.randperm(len(d1_y))
        for step in range(steps):
            support_index = _cyclic_batch(support_order, step, SUPPORT_BATCH_SIZE)
            d1_index = _cyclic_batch(d1_order, step, D1_BATCH_SIZE)
            support_base = exact_audit.base_logits(policy, support_x[support_index])
            d1_base_batch = d1_base[d1_index]
            support_logits = support_base + adapter(support_x[support_index])
            d1_logits = d1_base_batch + adapter(d1_x[d1_index])
            support_loss = set_nll(support_logits, support_mask[support_index])
            d1_ce = F.cross_entropy(d1_logits, d1_y[d1_index])
            base_logp = F.log_softmax(d1_base_batch.double(), dim=-1)
            repaired_logp = F.log_softmax(d1_logits.double(), dim=-1)
            anchor_kl = torch.sum(base_logp.exp() * (base_logp - repaired_logp), dim=-1).mean()
            loss = 0.5 * support_loss + 0.5 * d1_ce + ANCHOR_KL_WEIGHT * anchor_kl.float()
            _require(bool(torch.isfinite(loss)), "non-finite proposal-set probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), GRADIENT_CLIP_NORM)
            _require(bool(torch.isfinite(norm)), "non-finite proposal-set probe gradient")
            optimizer.step()
            final_components = {
                "loss": float(loss.detach()), "support_set_nll": float(support_loss.detach()),
                "d1_exact_ce": float(d1_ce.detach()), "d1_anchor_kl": float(anchor_kl.detach()),
            }
    return {
        "seed": int(seed), "epochs": EPOCHS, "optimizer_steps": EPOCHS * steps,
        "final_components": final_components,
        "fresh_dev": evaluate_set(policy, adapter, dev_rows, by_interface=True),
        "d1_anchor": exact_audit.evaluate(policy, adapter, d1_rows),
        "adapter_parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "final_epoch_only": True, "checkpoint_saved": False,
    }


def probe_checks(
    result: Mapping[str, Any], candidate_dev: Mapping[str, Any], candidate_d1: Mapping[str, Any],
) -> dict[str, Any]:
    dev, d1 = result["fresh_dev"], result["d1_anchor"]
    partial_nll_reduction = (
        float(candidate_dev["partial_set_nll"]) - float(dev["partial_set_nll"])
    ) / max(float(candidate_dev["partial_set_nll"]), 1e-12)
    interface = {
        name: float(dev["interface"][name]["top1_feasible"]) >= THRESHOLDS["interface_top1_min"]
        for name in support.INTERFACES
    }
    checks: dict[str, Any] = {
        "dev_top1": float(dev["top1_feasible"]) >= THRESHOLDS["dev_top1_min"],
        "dev_top1_gain": float(dev["top1_feasible"]) - float(candidate_dev["top1_feasible"])
        >= THRESHOLDS["dev_top1_gain_min"],
        "dev_top3": float(dev["top3_feasible_hit"]) >= THRESHOLDS["dev_top3_min"],
        "dev_top3_gain": float(dev["top3_feasible_hit"]) - float(candidate_dev["top3_feasible_hit"])
        >= THRESHOLDS["dev_top3_gain_min"],
        "partial_top1": float(dev["partial_top1_feasible"]) >= THRESHOLDS["partial_top1_min"],
        "partial_nll_reduction": partial_nll_reduction >= THRESHOLDS["partial_nll_reduction_min"],
        "every_interface_top1": all(interface.values()),
        "interface": interface,
        "partial_nll_reduction_value": float(partial_nll_reduction),
        "d1_top1_retained": float(candidate_d1["top1"]) - float(d1["top1"])
        <= THRESHOLDS["d1_top1_drop_max"],
        "d1_top3_retained": float(candidate_d1["top3"]) - float(d1["top3"])
        <= THRESHOLDS["d1_top3_drop_max"],
        "d1_nll_retained": float(d1["nll"]) - float(candidate_d1["nll"])
        <= THRESHOLDS["d1_nll_increase_max"],
    }
    checks["all"] = all(
        bool(value) for key, value in checks.items()
        if key not in {"interface", "partial_nll_reduction_value"}
    )
    return checks


def _state_digest(policy: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(
    output: str | Path, support_manifest: str | Path, support_train: str | Path,
    support_dev: str | Path, candidate_checkpoint: str | Path,
    d1_dataset: str | Path, d1_manifest: str | Path,
) -> dict[str, Any]:
    protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"output already exists: {output_path}")
    validate_support_manifest(support_manifest)
    _require(_sha256(support_train) == SUPPORT_TRAIN_SHA, "support train SHA mismatch")
    _require(_sha256(support_dev) == SUPPORT_DEV_SHA, "support dev SHA mismatch")
    exact_audit.validate_d1_manifest(d1_manifest, d1_dataset)
    policy, payload = effectiveness.load_candidate_checkpoint(candidate_checkpoint)
    output_path.mkdir(parents=True, exist_ok=False)
    train_rows = normalize_set_rows(_read_jsonl(support_train), split="train")
    dev_rows = normalize_set_rows(_read_jsonl(support_dev), split="dev")
    d1_rows = exact_audit.load_d1_proposal_rows(d1_dataset)
    _require(len(train_rows) == 463 and len(dev_rows) == 372, "support cardinality changed")
    candidate_state_before = _state_digest(policy)
    candidate_dev = evaluate_set(policy, None, dev_rows, by_interface=True)
    candidate_d1 = exact_audit.evaluate(policy, None, d1_rows)
    runs = []
    for seed in PROBE_SEEDS:
        probe = run_probe(policy, train_rows, dev_rows, d1_rows, seed=seed)
        probe["checks"] = probe_checks(probe, candidate_dev, candidate_d1)
        runs.append(probe)
        _write_json(output_path / "progress.json", {
            "stage": STAGE, "completed_seeds": [run["seed"] for run in runs],
            "planned_seeds": list(PROBE_SEEDS), "last_checks": probe["checks"],
        })
    candidate_state_after = _state_digest(policy)
    all_pass = all(run["checks"]["all"] for run in runs)
    audit_complete = len(runs) == len(PROBE_SEEDS) and candidate_state_before == candidate_state_after
    passed = bool(audit_complete and all_pass)
    result = {
        "stage": STAGE, "status": "PASS" if passed else "FAIL", "evidence_type": "newly_run",
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA,
        "support_manifest": str(support_manifest), "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "support_train": str(support_train), "support_train_sha256": SUPPORT_TRAIN_SHA,
        "support_dev": str(support_dev), "support_dev_sha256": SUPPORT_DEV_SHA,
        "candidate_checkpoint": str(candidate_checkpoint), "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "candidate_payload": {"stage": payload.get("stage"), "architecture_id": payload.get("architecture_id"),
                              "temperature": payload.get("temperature"), "optimizer_loaded": False},
        "d1_dataset": str(d1_dataset), "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest": str(d1_manifest), "d1_manifest_sha256": D1_MANIFEST_SHA,
        "rows": {"train": len(train_rows), "dev": len(dev_rows), "d1_release": len(d1_rows)},
        "candidate_frozen": {"fresh_dev": candidate_dev, "d1_anchor": candidate_d1},
        "runs": runs, "thresholds": dict(THRESHOLDS),
        "candidate_state_before": candidate_state_before,
        "candidate_state_after": candidate_state_after,
        "candidate_state_unchanged": candidate_state_before == candidate_state_after,
        "AUDIT_COMPLETE": audit_complete,
        "proposal_set_learnability_pass": passed,
        "proposal_set_repair_preregistration_allowed": passed,
        "environment_called": False, "teacher_called": False,
        "policy_training_called": False, "policy_optimizer_called": False,
        "probe_optimizer_called": True, "checkpoint_saved": False,
        "dev_gradients": False, "d1_selection": False, "threshold_tuning": False,
        "PROPOSAL_REPAIR_ALLOWED": False, "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False, "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False, "checkpoint_promotable": False,
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    manifest_path = output_path / "manifest.json"
    result["manifest"] = str(manifest_path)
    _write_json(manifest_path, result)
    print(json.dumps({
        "stage": STAGE, "status": result["status"], "AUDIT_COMPLETE": audit_complete,
        "proposal_set_learnability_pass": passed,
        "proposal_set_repair_preregistration_allowed": passed, "PPO_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--support-train", type=Path, required=True)
    parser.add_argument("--support-dev", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--d1-dataset", type=Path, required=True)
    parser.add_argument("--d1-manifest", type=Path, required=True)
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    _main()
