"""Inference-only bounded-gate diagnostic for the three failed P3 runs.

The diagnostic restores each run's last available calibrated checkpoint, adds
the probability-floor adapter, and replays only the corresponding failed
episode seed with the canonical rejected-release execution mode.  It does not
perform PPO updates, consume held-out seeds, or authorize a qualification
campaign.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from terminal_marl.algo.gate_bounded_calibrated_v4_2 import (
    BOUNDED_CALIBRATED_ARCHITECTURE_ID,
    DEFAULT_MINIMUM_GATE_PROBABILITY,
    BoundedTemperatureCalibratedGateResidualAdapterStudentPPOV42,
    migrate_calibrated_policy_to_bounded,
)
from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint, save_checkpoint
from terminal_marl.training import (
    evaluate_random_soc_gate_repair_v1 as base,
    run_v4_2_p3_controlled_emergency_admission_v5 as admission,
    student_rollout_v4_2 as student_rollout,
)


STAGE = "P3_BOUNDED_GATE_DIAGNOSTIC_V1"
SOURCE_STAGE = "P3_FORMAL_CONTROLLED_EMERGENCY_V5_RESTART1"
SOURCE_OUTPUT_NAME = "p3_controlled_emergency_formal_v5_restart1_20260925"
OBSERVATION_DIM = 88
MINIMUM_GATE_PROBABILITY = float(DEFAULT_MINIMUM_GATE_PROBABILITY)
SCENARIO = "dense24"
DEFAULT_MAX_STEPS = 1800
RUN_INDICES = (0, 1, 2)
SOURCE_ROOT_CANDIDATES = (
    Path("/home/user/training_outputs/terminal") / SOURCE_OUTPUT_NAME,
    Path(r"D:\7_codespace\training_outputs\terminal") / SOURCE_OUTPUT_NAME,
)


class BoundedGateDiagnosticError(ValueError):
    """Raised when a diagnostic input or output contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundedGateDiagnosticError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    return admission._jsonable(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        _require(isinstance(value, torch.Tensor), f"model state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right))
    return left == right


def _source_roots(source_root: str | Path | None = None) -> tuple[Path, ...]:
    if source_root is not None:
        return (Path(source_root),)
    configured = os.environ.get("P3_BOUNDED_GATE_SOURCE_ROOT")
    roots = []
    if configured:
        roots.append(Path(configured))
    roots.extend(SOURCE_ROOT_CANDIDATES)
    return tuple(dict.fromkeys(roots))


def _manifest_for_run(run_index: int, *, source_root: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    if int(run_index) not in RUN_INDICES:
        raise BoundedGateDiagnosticError(f"run_index must be one of {RUN_INDICES}")
    for root in _source_roots(source_root):
        manifest_path = root / f"run{int(run_index)}" / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BoundedGateDiagnosticError(f"cannot read source manifest: {manifest_path}") from exc
        _require(isinstance(manifest, Mapping), "source manifest must be a JSON object")
        return manifest_path, dict(manifest)
    searched = ", ".join(str(root) for root in _source_roots(source_root))
    raise BoundedGateDiagnosticError(f"source manifest missing for run {run_index}; searched {searched}")


def source_binding(run_index: int, *, source_root: str | Path | None = None) -> dict[str, Any]:
    """Resolve the observed final checkpoint and the next recorded failure seed."""

    manifest_path, manifest = _manifest_for_run(run_index, source_root=source_root)
    _require(manifest.get("run_index") == int(run_index), "source run index mismatch")
    _require(manifest.get("status") == "FAILED", "bounded diagnostic requires a failed source run")
    final_raw = manifest.get("final_checkpoint")
    _require(isinstance(final_raw, str) and final_raw, "source final_checkpoint is missing")
    run_root = manifest_path.parent
    source_path = Path(final_raw)
    if not source_path.is_file():
        source_path = run_root / "checkpoints" / Path(final_raw).name
    _require(source_path.is_file(), f"source final checkpoint missing: {source_path}")
    accepted = int(manifest.get("accepted_episodes", -1))
    recorded = int(manifest.get("episodes_recorded", -1))
    seeds = manifest.get("train_seeds")
    _require(isinstance(seeds, list), "source train_seeds is missing")
    _require(accepted >= 0 and recorded == accepted + 1, "source failure episode binding changed")
    _require(accepted < len(seeds), "source failure seed is missing")
    return {
        "run_index": int(run_index),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_stage": str(manifest.get("stage", SOURCE_STAGE)),
        "source_collection": SOURCE_OUTPUT_NAME,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _sha256(source_path),
        "source_checkpoint_name": source_path.name,
        "accepted_episodes_before_failure": accepted,
        "failure_episode_number": recorded,
        "failure_seed": int(seeds[accepted]),
        "source_status": str(manifest.get("status")),
    }


def build_config() -> Any:
    """Build the original dense24 execution config with canonical WAIT gating."""

    protocol = base._load_yaml_protocol()
    config = admission._build_config(SCENARIO, protocol)
    _require(hasattr(config, "v4_2_rejected_release_hold_mode"),
             "current config lacks v4_2_rejected_release_hold_mode")
    config = replace(config, v4_2_rejected_release_hold_mode="canonical_wait")
    _require(config.v4_2_rejected_release_hold_mode == "canonical_wait",
             "bounded diagnostic config did not apply canonical_wait")
    _require(str(config.action_schema_version) == "low_action_schema_v4_2_hierarchical",
             "bounded diagnostic requires V4.2 actions")
    _require(str(config.observation_schema_version) == "phaseA_low_obs_v2_1_retry_state",
             "bounded diagnostic requires V2.1 observations")
    _require(str(config.scheduler_mode) == "execution", "bounded diagnostic requires execution scheduler")
    _require(not bool(config.teacher_enabled), "bounded diagnostic requires Teacher disabled")
    _require(bool(config.shield_enabled), "bounded diagnostic requires safety shield")
    return config


def _load_calibrated_source(source_checkpoint: Path, config: Any):
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        temperature=CALIBRATION_TEMPERATURE,
    )
    payload = load_checkpoint(
        source_checkpoint,
        policy,
        config,
        load_optimizer=False,
    )
    _require(len(policy.optimizer.state) == 0, "source optimizer state was restored")
    policy.model.eval()
    return policy, payload


def _migrate_source(policy: Any):
    source_state = {
        name: value.detach().cpu().clone()
        for name, value in policy.model.state_dict().items()
    }
    source_digest = _state_digest(source_state)
    bounded = migrate_calibrated_policy_to_bounded(
        policy,
        minimum_gate_probability=MINIMUM_GATE_PROBABILITY,
    )
    bounded_state = bounded.model.state_dict()
    common = set(source_state) & set(bounded_state)
    old_weights_unchanged = all(
        _state_equal(source_state[name], bounded_state[name]) for name in common
    ) and set(source_state) == common
    _require(old_weights_unchanged, "calibrated source weights changed during bounded migration")
    bounded.model.eval()
    return bounded, {
        "source_weight_digest": source_digest,
        "bounded_common_weight_digest": _state_digest({name: bounded_state[name] for name in sorted(common)}),
        "common_state_names": sorted(common),
        "old_weights_unchanged": bool(old_weights_unchanged),
        "source_optimizer_state_copied": False,
    }


def _outcome_summary(summary: Mapping[str, Any], requests: int) -> dict[str, Any]:
    raw = summary.get("outcomes") or {}
    raw = raw if isinstance(raw, Mapping) else {}
    raw_outcomes = {str(key): str(getattr(value, "value", value)) for key, value in raw.items()}
    counts = Counter(raw_outcomes.values())
    unresolved = int(counts.get("unresolved", 0)) + max(0, int(requests) - len(raw_outcomes))
    normal = int(counts.get("completed", 0))
    emergency = int(counts.get("emergency_landed", 0))
    failed_energy = int(counts.get("failed_energy", 0))
    recognized = {"completed", "emergency_landed", "failed_energy", "unresolved"}
    other = {key: int(value) for key, value in counts.items() if key not in recognized}
    return {
        "normal": normal,
        "emergency": emergency,
        "failed_energy": failed_energy,
        "unresolved": unresolved,
        "other_failures": other,
        "raw_outcomes": raw_outcomes,
        "requests": int(requests),
        "resolved": normal + emergency,
        "full_resolution": bool(
            summary.get("terminal", False)
            and normal + emergency == int(requests)
            and failed_energy == 0
            and unresolved == 0
            and not other
        ),
    }


def _roundtrip_candidate(candidate: Any, path: Path, config: Any) -> dict[str, Any]:
    restored = BoundedTemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        temperature=CALIBRATION_TEMPERATURE,
        minimum_gate_probability=MINIMUM_GATE_PROBABILITY,
    )
    load_checkpoint(path, restored, config, load_optimizer=True)
    model_exact = _state_equal(candidate.model.state_dict(), restored.model.state_dict())
    optimizer_exact = _state_equal(candidate.optimizer.state_dict(), restored.optimizer.state_dict())
    _require(model_exact and optimizer_exact, "bounded candidate checkpoint restore mismatch")
    return {
        "model_exact": bool(model_exact),
        "optimizer_exact": bool(optimizer_exact),
        "restored_architecture_id": str(restored.architecture_id),
        "restored_floor": float(restored.model.minimum_gate_probability.detach().cpu()),
        "restored_optimizer_state_entries": len(restored.optimizer.state),
    }


def run(*, run_index: int, output: str | Path, max_steps: int = DEFAULT_MAX_STEPS,
        source_root: str | Path | None = None) -> dict[str, Any]:
    if isinstance(max_steps, bool) or int(max_steps) < 1:
        raise BoundedGateDiagnosticError("max_steps must be a positive integer")
    config = build_config()
    binding = source_binding(run_index, source_root=source_root)
    source_checkpoint = Path(binding["source_checkpoint"])
    calibrated, source_payload = _load_calibrated_source(source_checkpoint, config)
    bounded, migration = _migrate_source(calibrated)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=False)
    rollout_path = output_path / "rollout"
    steps, summary = student_rollout.rollout(
        bounded,
        config,
        seed=int(binding["failure_seed"]),
        max_steps=int(max_steps),
        output=rollout_path,
        deterministic=False,
        collect_labels=False,
    )
    checkpoint_path = output_path / "bounded_gate_candidate.pt"
    save_checkpoint(
        checkpoint_path,
        bounded,
        config,
        stage=STAGE,
        diagnostic_only=True,
        formal_qualification=False,
        checkpoint_promotable=False,
        ppo_called=False,
        training_called=False,
        heldout_consumed=False,
        source_stage=str(binding["source_stage"]),
        source_run_index=int(run_index),
        source_failure_seed=int(binding["failure_seed"]),
        source_checkpoint_sha256=str(binding["source_checkpoint_sha256"]),
        source_checkpoint_path=str(source_checkpoint),
        bounded_gate_architecture_id=BOUNDED_CALIBRATED_ARCHITECTURE_ID,
        minimum_gate_probability=MINIMUM_GATE_PROBABILITY,
        v4_2_rejected_release_hold_mode="canonical_wait",
        old_weights_unchanged=True,
        source_optimizer_state_copied=False,
    )
    restore = _roundtrip_candidate(bounded, checkpoint_path, config)
    result = {
        "stage": STAGE,
        "formal_qualification": False,
        "PPO_ALLOWED": False,
        "ppo_called": False,
        "training_called": False,
        "heldout_consumed": False,
        "run_index": int(run_index),
        "scenario": SCENARIO,
        "max_steps": int(max_steps),
        "seed": int(binding["failure_seed"]),
        "config": {
            "v4_2_rejected_release_hold_mode": str(config.v4_2_rejected_release_hold_mode),
            "action_schema_version": str(config.action_schema_version),
            "observation_schema_version": str(config.observation_schema_version),
            "max_simulation_seconds": float(config.max_simulation_seconds),
            "total_requested_uavs": int(config.total_requested_uavs),
        },
        "source": binding,
        "source_payload_stage": source_payload.get("stage"),
        "source_optimizer_restored": False,
        "floor": {
            "minimum_gate_probability": MINIMUM_GATE_PROBABILITY,
            "temperature": CALIBRATION_TEMPERATURE,
        },
        "new_architecture": BOUNDED_CALIBRATED_ARCHITECTURE_ID,
        "migration": migration,
        "summary": _jsonable(summary),
        "steps": int(len(steps)),
        "full_outcomes": _outcome_summary(summary, int(config.total_requested_uavs)),
        "safety": _jsonable(summary.get("safety", {})),
        "safety_error": summary.get("safety_error"),
        "candidate_checkpoint": str(checkpoint_path),
        "candidate_checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_restore": restore,
    }
    _write_json(output_path / "summary.json", result)
    return result


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, choices=RUN_INDICES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args()
    result = run(run_index=args.run_index, output=args.output, max_steps=args.max_steps)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
