"""Bounded inference-only anchor evaluation for the coverage fit.

This temporary runner compares the fixed E04 checkpoint with the fixed
coverage-fit checkpoint on the two registered development scenarios.  It
builds the canonical-WAIT execution configuration, performs stochastic
student rollouts without labels or updates, and writes only per-case JSON
artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.training import (
    evaluate_random_soc_gate_repair_v1 as base,
    run_v4_2_p3_controlled_emergency_admission_v5 as admission,
    student_rollout_v4_2 as student_rollout,
)


OBSERVATION_DIM = 88
CHECKPOINTS = ("e04", "coverage")
SCENARIOS = {"flow6": 846201, "flow12": 846202}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(scenario: str):
    config = admission._build_config(scenario, base._load_yaml_protocol())
    config = replace(config, v4_2_rejected_release_hold_mode="canonical_wait")
    if str(config.v4_2_rejected_release_hold_mode) != "canonical_wait":
        raise RuntimeError("canonical_wait configuration was not applied")
    if str(config.action_schema_version) != "low_action_schema_v4_2_hierarchical":
        raise RuntimeError("unexpected action schema")
    if str(config.observation_schema_version) != "phaseA_low_obs_v2_1_retry_state":
        raise RuntimeError("unexpected observation schema")
    if bool(config.teacher_enabled) or str(config.scheduler_mode) != "execution":
        raise RuntimeError("student-only execution configuration was not preserved")
    if not bool(config.shield_enabled):
        raise RuntimeError("safety shield is disabled")
    return config


def _load_policy(checkpoint: Path, config: Any):
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        temperature=CALIBRATION_TEMPERATURE,
    )
    payload = load_checkpoint(checkpoint, policy, config, load_optimizer=False)
    if len(policy.optimizer.state) != 0:
        raise RuntimeError("optimizer state was restored unexpectedly")
    policy.model.eval()
    return policy, payload


def _outcomes(summary: Mapping[str, Any], requests: int) -> dict[str, Any]:
    raw = summary.get("outcomes") or {}
    raw = raw if isinstance(raw, Mapping) else {}
    values = {str(uid): str(getattr(value, "value", value)) for uid, value in raw.items()}
    counts = Counter(values.values())
    unresolved = int(counts.get("unresolved", 0)) + max(0, int(requests) - len(values))
    known = {"completed", "emergency_landed", "failed_energy", "unresolved"}
    other = {key: int(value) for key, value in counts.items() if key not in known}
    normal = int(counts.get("completed", 0))
    emergency = int(counts.get("emergency_landed", 0))
    failed_energy = int(counts.get("failed_energy", 0))
    return {
        "requests": int(requests),
        "normal": normal,
        "emergency": emergency,
        "failed_energy": failed_energy,
        "unresolved": unresolved,
        "other_failures": other,
        "resolved": normal + emergency,
        "terminal": bool(summary.get("terminal", False)),
        "full_resolution": bool(
            summary.get("terminal", False)
            and normal + emergency == int(requests)
            and failed_energy == 0
            and unresolved == 0
            and not other
        ),
        "raw_outcomes": values,
    }


def run_case(*, checkpoint_label: str, checkpoint: Path, scenario: str,
             seed: int, output: Path, max_steps: int) -> dict[str, Any]:
    if checkpoint_label not in CHECKPOINTS:
        raise ValueError(f"checkpoint label must be one of {CHECKPOINTS}")
    if scenario not in SCENARIOS or int(seed) != int(SCENARIOS[scenario]):
        raise ValueError("scenario/seed is outside the fixed development pair")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha_before = _sha256(checkpoint)
    config = _load_config(scenario)
    policy, payload = _load_policy(checkpoint, config)
    output.mkdir(parents=True, exist_ok=False)
    rollout_output = output / "rollout"
    steps, summary = student_rollout.rollout(
        policy,
        config,
        seed=int(seed),
        max_steps=int(max_steps),
        output=rollout_output,
        deterministic=False,
        collect_labels=False,
    )
    checkpoint_sha_after = _sha256(checkpoint)
    requests = int(config.total_requested_uavs)
    result = {
        "checkpoint_label": checkpoint_label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_unchanged": bool(checkpoint_sha_before == checkpoint_sha_after),
        "checkpoint_payload_stage": payload.get("stage"),
        "scenario": scenario,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "deterministic": False,
        "collect_labels": False,
        "PPO_ALLOWED": False,
        "config": {
            "v4_2_rejected_release_hold_mode": str(config.v4_2_rejected_release_hold_mode),
            "action_schema_version": str(config.action_schema_version),
            "observation_schema_version": str(config.observation_schema_version),
            "scheduler_mode": str(config.scheduler_mode),
            "teacher_enabled": bool(config.teacher_enabled),
            "shield_enabled": bool(config.shield_enabled),
            "max_simulation_seconds": float(config.max_simulation_seconds),
            "total_requested_uavs": requests,
        },
        "steps": int(len(steps)),
        "summary": _jsonable(summary),
        "outcomes": _outcomes(summary, requests),
        "safety": _jsonable(summary.get("safety", {})),
        "safety_error": summary.get("safety_error"),
        "rollout_output": str(rollout_output),
    }
    (output / "summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-label", choices=CHECKPOINTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args()
    result = run_case(
        checkpoint_label=args.checkpoint_label,
        checkpoint=args.checkpoint,
        scenario=args.scenario,
        seed=args.seed,
        output=args.output,
        max_steps=args.max_steps,
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
