"""Exploratory 25-logit feasibility-head audit on frozen V2.2 V3 rows.

This audit changes only the diagnostic prediction head.  It does not call the
environment or Teacher, update a policy, save a checkpoint, or alter the
one-intent/one-validation runtime contract.
"""
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

from terminal_marl.training import run_v4_2_observation_v2_2_closure_pilot as closure
from terminal_marl.training.audit_v4_2_p2_proposal_conditioned_feasibility_v1 import binary_metrics


STAGE = "V4_2_OBSERVATION_V2_2_JOINT_FEASIBILITY_HEAD_AUDIT_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/observation_v2_2_joint_feasibility_head_audit_v1.yaml")
PROTOCOL_SHA = "3440e206303673e6d55c0fafd43d2d209c97f7fa3648362fcfa32f6446f3c4cb"
CLOSURE_STAGE = "V4_2_OBSERVATION_V2_2_CLOSURE_PILOT_V3"
CLOSURE_PROTOCOL_SHA = "227dfa7e252613f5c3755b04fd143c9c42d53d141451cef9748d83cb42926c4a"
CLOSURE_MANIFEST_SHA = "20cb9d6ac2377aa5bb3e20c87707405f4aa553eecadae0581ac64a0631cc8211"
TRAIN_ROWS_SHA = "57e376cc164e42af56d2ead967eca988c7a90bf9a31d60b204d7ec0f0c2c5199"
DEV_ROWS_SHA = "6f7232887be3d9f07c12514a7ba3693d4d17fb06154344050695f0366c8f2d34"
PROBE_SEEDS = (801101, 801102, 801103)
OBS_DIM = 94
N_PROPOSALS = 25
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 1.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def protocol() -> dict[str, Any]:
    _require(PROTOCOL_SHA and _sha256(PROTOCOL) == PROTOCOL_SHA, "joint-feasibility protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    _require(value.get("protocol") == STAGE, "joint-feasibility protocol id changed")
    inputs = value["inputs"]
    expected_inputs = {
        "closure_stage": CLOSURE_STAGE,
        "closure_protocol_sha256": CLOSURE_PROTOCOL_SHA,
        "closure_manifest_sha256": CLOSURE_MANIFEST_SHA,
        "train_rows_sha256": TRAIN_ROWS_SHA,
        "dev_rows_sha256": DEV_ROWS_SHA,
        "train_rows": 247,
        "dev_rows": 106,
        "no_new_environment_seeds": True,
    }
    for key, expected in expected_inputs.items():
        _require(inputs.get(key) == expected, f"joint-feasibility input changed: {key}")
    probe = value["probe"]
    _require(probe["seeds"] == list(PROBE_SEEDS), "probe seeds changed")
    _require(probe["epochs"] == EPOCHS and probe["batch_size_states"] == BATCH_SIZE,
             "probe budget changed")
    _require(float(probe["learning_rate"]) == LEARNING_RATE, "learning rate changed")
    _require(float(probe["gradient_clip_norm"]) == GRADIENT_CLIP_NORM, "gradient clip changed")
    _require(probe["threshold_tuning"] is False and probe["dev_gradients"] is False,
             "dev isolation changed")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_inputs(manifest_path: Path, train_path: Path, dev_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _require(_sha256(manifest_path) == CLOSURE_MANIFEST_SHA, "closure manifest SHA mismatch")
    _require(_sha256(train_path) == TRAIN_ROWS_SHA, "train rows SHA mismatch")
    _require(_sha256(dev_path) == DEV_ROWS_SHA, "dev rows SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("stage") == CLOSURE_STAGE and manifest.get("status") == "COMPLETE",
             "closure manifest is not complete V3 evidence")
    _require(manifest.get("protocol_sha256") == CLOSURE_PROTOCOL_SHA, "closure protocol mismatch")
    _require(manifest.get("train_rows_sha256") == TRAIN_ROWS_SHA, "manifest train rows mismatch")
    _require(manifest.get("dev_rows_sha256") == DEV_ROWS_SHA, "manifest dev rows mismatch")
    _require(manifest.get("collection_checks", {}).get("all_episodes_pass") is True,
             "closure collection gates failed")
    train_rows, dev_rows = _read_jsonl(train_path), _read_jsonl(dev_path)
    _require(len(train_rows) == 247 and len(dev_rows) == 106, "closure row counts changed")
    return manifest, train_rows, dev_rows


def tensors(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.as_tensor(
        np.asarray([row["observation_v2_2"] for row in rows], dtype=np.float32)
    )
    targets = closure._target_mask(rows).float()
    _require(observations.shape == (len(rows), OBS_DIM), "V2.2 observation shape changed")
    _require(targets.shape == (len(rows), N_PROPOSALS), "feasibility target shape changed")
    return observations, targets


class JointFeasibilityHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, N_PROPOSALS),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def run_probe(train_rows: Sequence[Mapping[str, Any]], dev_rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_x, train_y = tensors(train_rows)
    dev_x, dev_y = tensors(dev_rows)
    mean, scale = train_x.mean(dim=0), train_x.std(dim=0).clamp_min(1e-6)
    train_x, dev_x = (train_x - mean) / scale, (dev_x - mean) / scale
    model = JointFeasibilityHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    positive_weight = float((train_y == 0).sum() / (train_y == 1).sum().clamp_min(1))
    steps = 0
    for _ in range(EPOCHS):
        order = torch.randperm(len(train_y))
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            logits = model(train_x[index])
            loss = F.binary_cross_entropy_with_logits(
                logits, train_y[index], pos_weight=logits.new_tensor(positive_weight)
            )
            _require(bool(torch.isfinite(loss)), "non-finite joint-feasibility loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            _require(bool(torch.isfinite(norm)), "non-finite joint-feasibility gradient")
            optimizer.step(); steps += 1
    with torch.no_grad():
        dev_logits = model(dev_x)
    return {
        "seed": seed,
        "optimizer_steps": steps,
        "binary_dev": binary_metrics(dev_logits.reshape(-1), dev_y.reshape(-1)),
        "state_dev": closure._state_metrics(dev_logits, dev_rows),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_saved": False,
    }


def run(*, manifest: Path, train_rows: Path, dev_rows: Path, output: Path) -> dict[str, Any]:
    protocol()
    _require(not output.exists(), f"output already exists: {output}")
    closure_manifest, train, dev = validate_inputs(manifest, train_rows, dev_rows)
    output.mkdir(parents=True, exist_ok=False)
    baseline = {int(item["seed"]): item["v2_2_94d"] for item in closure_manifest["paired_probes"]}
    probes = []
    for seed in PROBE_SEEDS:
        head = run_probe(train, dev, seed=seed)
        conditioned = baseline[seed]
        delta = {
            key: float(head["binary_dev"][key]) - float(conditioned["binary_dev"][key])
            for key in ("feasible_recall", "infeasible_recall", "macro_f1", "roc_auc")
        }
        delta["state_top1_feasible"] = (
            float(head["state_dev"]["top1_feasible"])
            - float(conditioned["state_dev"]["top1_feasible"])
        )
        probes.append({"seed": seed, "conditioned_shared_baseline": conditioned,
                       "joint_feasibility_head": head, "delta_head_minus_conditioned": delta})
    result = {
        "stage": STAGE,
        "status": "COMPLETE",
        "evidence_type": "newly_run",
        "input_evidence_type": "preexisting_artifact",
        "protocol": str(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA,
        "source_manifest": str(manifest),
        "source_manifest_sha256": CLOSURE_MANIFEST_SHA,
        "source_train_rows_sha256": TRAIN_ROWS_SHA,
        "source_dev_rows_sha256": DEV_ROWS_SHA,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "probes": probes,
        "environment_called": False,
        "teacher_called": False,
        "policy_updated": False,
        "checkpoint_saved": False,
        "threshold_tuning": False,
        "dev_gradients": False,
        "architecture_freeze_allowed": False,
        "support_collection_allowed": False,
        "policy_repair_allowed": False,
        "student_qualification_allowed": False,
        "effectiveness_evaluation_allowed": False,
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
    }
    output_path = output / "manifest.json"
    result["manifest"] = str(output_path)
    output_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"stage": STAGE, "status": "COMPLETE", "probes": len(probes),
                      "PPO_ALLOWED": False}, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-rows", type=Path, required=True)
    parser.add_argument("--dev-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    _main()
