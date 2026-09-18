"""Offline gate-calibration comparison on frozen WAIT-diagnostic observations."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r3 import _write
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import _sha


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_forgetting_audit.yaml")
PROTOCOL_SHA = "a0dd9b6db4e56263cddde2a8d726150c56a7ddbbf63c817c900eef3a45cbac1a"
INPUT_MANIFEST_SHA = "b290e2d5f69749e1109bff2c5f348458732a22fbe8c0ce34c15ce98880eefc93"
CHECKPOINTS = {
    "i3_dagger": "a97e0284b07b18404f60cd666c72806cbf5e261bf807c7f9f3a1d897c69b1584",
    "checkpoint24": "6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f",
    "p2_v4_checkpoint8": "5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f",
}
STATE_SETS = ("candidate_visited", "baseline_visited", "combined")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def protocol() -> dict[str, Any]:
    _require(_sha(PROTOCOL) == PROTOCOL_SHA, "gate forgetting protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "gate forgetting protocol must be a mapping")
    return value


def gate_metrics(rows: list[Mapping[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    """Score one checkpoint against immutable Teacher gate labels."""
    _require(len(rows) == len(probabilities) and bool(rows), "rows/probabilities must align and be nonempty")
    labels = np.asarray([int(row["teacher_gate"]) for row in rows], dtype=np.int64)
    predicted = np.asarray(probabilities >= 0.5, dtype=np.int64)
    stable = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    teacher_release = labels == 1
    teacher_wait = ~teacher_release
    false_wait = (predicted == 0) & teacher_release
    false_release = (predicted == 1) & teacher_wait

    def stats(values: np.ndarray) -> dict[str, float | int | None]:
        if not values.size:
            return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
        return {
            "n": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
        }

    interfaces = {}
    for interface in ("o2m", "m2i", "i2f"):
        mask = np.asarray([str(row["interface"]) == interface for row in rows], dtype=bool)
        release_count = int((teacher_release & mask).sum())
        interfaces[interface] = {
            "samples": int(mask.sum()),
            "teacher_RELEASE": release_count,
            "false_WAIT": int((false_wait & mask).sum()),
            "false_wait_rate": None if not release_count else float((false_wait & mask).sum() / release_count),
        }
    nll = -(labels * np.log(stable) + (1 - labels) * np.log(1.0 - stable))
    return {
        "samples": len(rows),
        "teacher": {"WAIT": int(teacher_wait.sum()), "RELEASE": int(teacher_release.sum())},
        "predicted": {"WAIT": int((predicted == 0).sum()), "RELEASE": int((predicted == 1).sum())},
        "false_WAIT": int(false_wait.sum()),
        "false_RELEASE": int(false_release.sum()),
        "false_wait_rate": float(false_wait.sum() / max(1, teacher_release.sum())),
        "false_release_rate": float(false_release.sum() / max(1, teacher_wait.sum())),
        "brier": float(np.mean((stable - labels) ** 2)),
        "binary_nll": float(nll.mean()),
        "release_probability": {
            "teacher_WAIT": stats(stable[teacher_wait]),
            "teacher_RELEASE": stats(stable[teacher_release]),
        },
        "interfaces": interfaces,
    }


def _read_labels(root: Path, manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    sets: dict[str, list[dict[str, Any]]] = {name: [] for name in STATE_SETS}
    for record in manifest["records"]:
        controller = str(record["controller"])
        seed = int(record["result"]["seed"])
        path = root / f"{controller}_{seed}" / "labels.jsonl"
        _require(path.is_file(), f"label sidecar missing: {path}")
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        rows = [row for row in rows if bool(row.get("actor_mask"))]
        target = f"{controller}_visited"
        _require(target in sets, f"unexpected controller: {controller}")
        for row in rows:
            row["interface"] = {"ring_outer": "o2m", "ring_middle": "m2i", "ring_inner": "i2f"}.get(row["phase"])
        _require(all(row["interface"] is not None for row in rows), "actor row outside ring interface")
        sets[target].extend(rows)
        sets["combined"].extend(rows)
    return sets


def _load_policy(path: Path, expected_sha: str) -> StudentPPOV42:
    _require(path.is_file() and _sha(path) == expected_sha, f"checkpoint binding mismatch: {path}")
    policy = StudentPPOV42(88, learning_rate=0.00003)
    load_checkpoint(path, policy, p2_config(6), load_optimizer=False)
    policy.model.eval()
    return policy


def _probabilities(policy: StudentPPOV42, rows: list[Mapping[str, Any]], batch_size: int = 2048) -> np.ndarray:
    values = np.asarray([row["observation_v2_1"] for row in rows], dtype=np.float32)
    output = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            gate, _ = policy.model.distributions(torch.as_tensor(values[start:start + batch_size], device=policy.device))
            output.append(gate.probs[:, 1].cpu().numpy())
    return np.concatenate(output)


def run(
    output: Path, diagnostic_root: Path, i3_checkpoint: Path,
    baseline: Path, candidate: Path,
) -> dict[str, Any]:
    p = protocol()
    manifest_path = diagnostic_root / "manifest.json"
    _require(manifest_path.is_file() and _sha(manifest_path) == INPUT_MANIFEST_SHA, "input manifest binding mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("DIAGNOSTIC_COMPLETE") is False and len(manifest.get("records", [])) == 12,
             "input failure status/shape must be preserved")
    paths = {"i3_dagger": i3_checkpoint, "checkpoint24": baseline, "p2_v4_checkpoint8": candidate}
    _require(p["input"]["wait_diagnostic_manifest_sha256"] == INPUT_MANIFEST_SHA, "protocol input changed")
    _require({name: value["sha256"] for name, value in p["checkpoints"].items()} == CHECKPOINTS,
             "protocol checkpoints changed")
    rows = _read_labels(diagnostic_root, manifest)
    policies = {name: _load_policy(paths[name], expected) for name, expected in CHECKPOINTS.items()}
    scores = {
        name: {
            state_set: gate_metrics(state_rows, _probabilities(policy, state_rows))
            for state_set, state_rows in rows.items()
        }
        for name, policy in policies.items()
    }
    combined = {name: score["combined"] for name, score in scores.items()}
    i3 = combined["i3_dagger"]
    ppo = [combined["checkpoint24"], combined["p2_v4_checkpoint8"]]
    supports_forgetting = all(
        i3["false_wait_rate"] < item["false_wait_rate"]
        and i3["brier"] < item["brier"]
        and i3["binary_nll"] < item["binary_nll"]
        for item in ppo
    )
    result = {
        "stage": "V4_2_P2_GATE_FORGETTING_AUDIT",
        "AUDIT_COMPLETE": True,
        "supports_PPO_gate_forgetting_hypothesis": bool(supports_forgetting),
        "scores": scores,
        "protocol_sha256": PROTOCOL_SHA,
        "input_manifest_sha256": INPUT_MANIFEST_SHA,
        "checkpoint_sha256": CHECKPOINTS,
        "source_hashes": {"runner": _sha(Path(__file__)), "protocol": _sha(PROTOCOL)},
        "training_called": False,
        "environment_called": False,
        "teacher_called": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, result)
    print(json.dumps({"AUDIT_COMPLETE": True, "supports_PPO_gate_forgetting_hypothesis": supports_forgetting}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("output", "diagnostic-root", "i3-checkpoint", "baseline", "candidate"):
        parser.add_argument("--" + name, type=Path, required=True)
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))
