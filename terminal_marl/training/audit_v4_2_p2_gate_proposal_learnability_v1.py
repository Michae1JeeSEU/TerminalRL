"""Offline learnability audit for fresh P2 conditional-proposal support."""
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
from torch import nn
from torch.nn import functional as F

from terminal_marl.training import collect_v4_2_p2_gate_proposal_support_v1 as support
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_proposal_learnability_audit_v1.yaml")
PROTOCOL_SHA = "fd05c7b7928474bd649520b6bec7261df997e7d04ec14972955ae3335e99d196"
STAGE = "V4_2_P2_GATE_PROPOSAL_LEARNABILITY_AUDIT_V1"
SUPPORT_MANIFEST_SHA = "48c0b1cd0b92bd746bd52f18090bbffd8facf59457eb82d503802eb4fc2c512b"
SUPPORT_TRAIN_SHA = "7aa8adaf805b22fd58aaaf258111e1b457cf5159241135d8ca262d32a5d11fa2"
SUPPORT_DEV_SHA = "2ea5bbfe66d6a688a486039a6c12461356ea51abce3c778cc85e8b6b05589415"
CANDIDATE_SHA = effectiveness.REPAIRED_CHECKPOINT_SHA
D1_DATASET_SHA = "270e54ef6ed0c36f535e2a0f25cef7cf9fdd6c91016ee295e7e336fb800cacb0"
D1_MANIFEST_SHA = "a6319c291e5c91e155fd1c7fb20ae95dcdd4973733abfe83fc10c5e6006f472c"
OBSERVATION_DIM = 88
N_PROPOSALS = 25
PROBE_SEEDS = (792101, 792102, 792103)
EPOCHS = 50
SUPPORT_BATCH_SIZE = 128
D1_BATCH_SIZE = 128
LEARNING_RATE = 3e-4
GRADIENT_CLIP_NORM = 1.0
ANCHOR_KL_WEIGHT = 1.0
THRESHOLDS = {
    "dev_top1_min": 0.60,
    "dev_top3_min": 0.80,
    "dev_nll_max": 1.30,
    "interface_top1_min": 0.50,
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
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "proposal audit protocol changed")
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
        "PROPOSAL_SUPPORT_COLLECTION": "PASS", "PROPOSAL_SUPPORT_FROZEN": True,
        "PROPOSAL_LEARNABILITY_AUDIT_ALLOWED": True,
        "train_rows": 540, "dev_rows": 455,
        "train_all_gates": True, "dev_all_gates": True,
        "train_dev_seed_disjoint": True, "train_dev_event_disjoint": True,
        "training_called": False, "policy_update": False,
    }.items():
        _require(requirements.get(key) == expected, f"support requirement changed: {key}")
    probe = value.get("probe", {})
    for key, expected in {
        "seeds": list(PROBE_SEEDS), "epochs": EPOCHS,
        "support_batch_size": SUPPORT_BATCH_SIZE, "d1_batch_size": D1_BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "anchor_kl_weight": ANCHOR_KL_WEIGHT,
        "checkpoint_selection": "final_epoch_only",
        "early_stopping": False, "threshold_tuning": False,
        "dev_gradients": False, "d1_selection": False,
    }.items():
        actual = probe.get(key)
        _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                 f"probe binding changed: {key}")
    evaluation = value.get("evaluation", {})
    fresh, anchor = evaluation.get("fresh_dev", {}), evaluation.get("d1_anchor", {})
    configured = {
        "dev_top1_min": fresh.get("top1_min"),
        "dev_top3_min": fresh.get("top3_min"),
        "dev_nll_max": fresh.get("nll_max"),
        "interface_top1_min": fresh.get("every_interface_top1_min"),
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
        "PROPOSAL_SUPPORT_COLLECTION": "PASS", "PROPOSAL_SUPPORT_FROZEN": True,
        "PROPOSAL_LEARNABILITY_AUDIT_ALLOWED": True,
        "PROPOSAL_REPAIR_ALLOWED": False, "PPO_ALLOWED": False,
        "training_called": False, "optimizer_called": False, "policy_update": False,
    }.items():
        _require(value.get(key) == expected, f"support manifest changed: {key}")
    _require(value.get("train", {}).get("sha256") == SUPPORT_TRAIN_SHA,
             "support train binding changed")
    _require(value.get("dev", {}).get("sha256") == SUPPORT_DEV_SHA,
             "support dev binding changed")
    _require(value.get("train", {}).get("gates", {}).get("all") is True
             and value.get("dev", {}).get("gates", {}).get("all") is True,
             "support split gates are not passing")
    _require(all(value.get("cross_split_checks", {}).values()), "support split overlap")
    return value


def validate_d1_manifest(path: str | Path, dataset: str | Path) -> dict[str, Any]:
    _require(_sha256(path) == D1_MANIFEST_SHA, "D1 manifest SHA mismatch")
    _require(_sha256(dataset) == D1_DATASET_SHA, "D1 dataset SHA mismatch")
    value = _read_json(path)
    _require(value.get("dataset_sha256") == D1_DATASET_SHA, "D1 manifest dataset binding changed")
    _require(value.get("smoke") is False and int(value.get("aggregate_rows", 0)) > 0,
             "D1 manifest is not formal aggregate")
    return value


def _normalize_rows(rows: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, Any]]:
    result = []
    for index, source in enumerate(rows):
        observation = np.asarray(source.get("observation_v2_1"), dtype=np.float32)
        _require(observation.shape == (OBSERVATION_DIM,) and np.isfinite(observation).all(),
                 f"{split} row {index} observation invalid")
        target = source.get("proposal_target", source.get("teacher_proposal"))
        _require(target is not None and 0 <= int(target) < N_PROPOSALS,
                 f"{split} row {index} proposal invalid")
        row = dict(source)
        row["observation_v2_1"] = observation
        row["proposal_target"] = int(target)
        result.append(row)
    return result


def load_d1_proposal_rows(path: str | Path) -> list[dict[str, Any]]:
    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if int(value.get("teacher_gate", -1)) != 1 or value.get("teacher_proposal") is None:
                continue
            observation = np.asarray(value.get("observation_v2_1"), dtype=np.float32)
            _require(observation.shape == (OBSERVATION_DIM,) and np.isfinite(observation).all(),
                     f"D1 row {line_number} observation invalid")
            target = int(value["teacher_proposal"])
            _require(0 <= target < N_PROPOSALS, f"D1 row {line_number} proposal invalid")
            result.append({"observation_v2_1": observation, "proposal_target": target})
    _require(bool(result), "D1 has no RELEASE proposal rows")
    return result


def tensors(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.as_tensor(
        np.asarray([row["observation_v2_1"] for row in rows], dtype=np.float32)
    )
    targets = torch.as_tensor([int(row["proposal_target"]) for row in rows], dtype=torch.long)
    return observations, targets


class ProposalResidualProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, 64), nn.Tanh(), nn.Linear(64, N_PROPOSALS),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def base_logits(policy: Any, observations: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, proposal = policy.model.distributions(observations.to(policy.device))
    return proposal.logits.detach().cpu()


def metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    _require(len(targets) > 0, "metrics require rows")
    top = logits.topk(3, dim=-1).indices
    return {
        "rows": int(len(targets)),
        "top1": float((top[:, 0] == targets).float().mean()),
        "top3": float((top == targets[:, None]).any(dim=-1).float().mean()),
        "nll": float(F.cross_entropy(logits, targets)),
    }


def evaluate(
    policy: Any, adapter: ProposalResidualProbe | None,
    rows: Sequence[Mapping[str, Any]], *, by_interface: bool = False,
) -> dict[str, Any]:
    observations, targets = tensors(rows)
    logits = base_logits(policy, observations)
    if adapter is not None:
        adapter.eval()
        with torch.no_grad():
            logits = logits + adapter(observations)
    result: dict[str, Any] = metrics(logits, targets)
    if by_interface:
        result["interface"] = {}
        for name in support.INTERFACES:
            indices = [index for index, row in enumerate(rows) if row.get("interface") == name]
            _require(bool(indices), f"fresh dev missing interface {name}")
            index = torch.as_tensor(indices, dtype=torch.long)
            result["interface"][name] = metrics(logits[index], targets[index])
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
    adapter = ProposalResidualProbe()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=LEARNING_RATE)
    support_x, support_y = tensors(train_rows)
    d1_x, d1_y = tensors(d1_rows)
    d1_base = base_logits(policy, d1_x)
    steps = max(math.ceil(len(support_y) / SUPPORT_BATCH_SIZE), math.ceil(len(d1_y) / D1_BATCH_SIZE))
    final_components: dict[str, float] = {}
    for _epoch in range(EPOCHS):
        support_order = torch.randperm(len(support_y))
        d1_order = torch.randperm(len(d1_y))
        for step in range(steps):
            support_index = _cyclic_batch(support_order, step, SUPPORT_BATCH_SIZE)
            d1_index = _cyclic_batch(d1_order, step, D1_BATCH_SIZE)
            support_base = base_logits(policy, support_x[support_index])
            d1_base_batch = d1_base[d1_index]
            support_logits = support_base + adapter(support_x[support_index])
            d1_logits = d1_base_batch + adapter(d1_x[d1_index])
            support_ce = F.cross_entropy(support_logits, support_y[support_index])
            d1_ce = F.cross_entropy(d1_logits, d1_y[d1_index])
            base_logp = F.log_softmax(d1_base_batch.double(), dim=-1)
            repaired_logp = F.log_softmax(d1_logits.double(), dim=-1)
            anchor_kl = torch.sum(base_logp.exp() * (base_logp - repaired_logp), dim=-1).mean()
            loss = 0.5 * support_ce + 0.5 * d1_ce + ANCHOR_KL_WEIGHT * anchor_kl.float()
            _require(bool(torch.isfinite(loss)), "non-finite proposal probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), GRADIENT_CLIP_NORM)
            _require(bool(torch.isfinite(norm)), "non-finite proposal probe gradient")
            optimizer.step()
            final_components = {
                "loss": float(loss.detach()), "support_ce": float(support_ce.detach()),
                "d1_ce": float(d1_ce.detach()), "d1_anchor_kl": float(anchor_kl.detach()),
            }
    dev = evaluate(policy, adapter, dev_rows, by_interface=True)
    d1 = evaluate(policy, adapter, d1_rows)
    return {
        "seed": int(seed), "epochs": EPOCHS, "optimizer_steps": EPOCHS * steps,
        "final_components": final_components, "fresh_dev": dev, "d1_anchor": d1,
        "adapter_parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "final_epoch_only": True, "checkpoint_saved": False,
    }


def probe_checks(
    result: Mapping[str, Any], candidate_d1: Mapping[str, Any],
) -> dict[str, Any]:
    dev, d1 = result["fresh_dev"], result["d1_anchor"]
    interface = {
        name: float(dev["interface"][name]["top1"]) >= THRESHOLDS["interface_top1_min"]
        for name in support.INTERFACES
    }
    checks: dict[str, Any] = {
        "dev_top1": float(dev["top1"]) >= THRESHOLDS["dev_top1_min"],
        "dev_top3": float(dev["top3"]) >= THRESHOLDS["dev_top3_min"],
        "dev_nll": float(dev["nll"]) <= THRESHOLDS["dev_nll_max"],
        "every_interface_top1": all(interface.values()),
        "interface": interface,
        "d1_top1_retained": float(candidate_d1["top1"]) - float(d1["top1"]) <= THRESHOLDS["d1_top1_drop_max"],
        "d1_top3_retained": float(candidate_d1["top3"]) - float(d1["top3"]) <= THRESHOLDS["d1_top3_drop_max"],
        "d1_nll_retained": float(d1["nll"]) - float(candidate_d1["nll"]) <= THRESHOLDS["d1_nll_increase_max"],
    }
    checks["all"] = all(bool(value) for key, value in checks.items() if key != "interface")
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
    validate_d1_manifest(d1_manifest, d1_dataset)
    policy, payload = effectiveness.load_candidate_checkpoint(candidate_checkpoint)
    output_path.mkdir(parents=True, exist_ok=False)
    train_rows = _normalize_rows(_read_jsonl(support_train), split="train")
    dev_rows = _normalize_rows(_read_jsonl(support_dev), split="dev")
    d1_rows = load_d1_proposal_rows(d1_dataset)
    _require(len(train_rows) == 540 and len(dev_rows) == 455, "support cardinality changed")
    candidate_state_before = _state_digest(policy)
    candidate_dev = evaluate(policy, None, dev_rows, by_interface=True)
    candidate_d1 = evaluate(policy, None, d1_rows)
    runs = []
    for seed in PROBE_SEEDS:
        probe = run_probe(policy, train_rows, dev_rows, d1_rows, seed=seed)
        probe["checks"] = probe_checks(probe, candidate_d1)
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
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
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
        "proposal_learnability_pass": passed,
        "proposal_repair_preregistration_allowed": passed,
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
        "proposal_learnability_pass": passed,
        "proposal_repair_preregistration_allowed": passed, "PPO_ALLOWED": False,
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
