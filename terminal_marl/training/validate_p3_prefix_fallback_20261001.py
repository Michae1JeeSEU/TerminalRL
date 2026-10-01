"""Fresh six-scene validation of the frozen prefix-fallback interaction.

This module owns a new, development-only validation panel.  It calls the
existing native runtime and prefix-fallback arm functions directly with a new
fixed six-cell grid; it does not alter those functions, the environment, or a
policy.  The four arms are BEFORE/AFTER x legacy/prefix-fallback.  No
optimizer, Teacher, held-out data, or PPO admission is involved.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import run_p3_before_prefix_fallback_20261001 as before_prefix
from terminal_marl.training import run_p3_prefix_fallback_20261001 as prefix
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as support


STAGE = "P3_PREFIX_FALLBACK_FRESH_VALIDATION_20261001_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
OFF_POLICY = True
NO_TRAINING = True
OBSERVATION_SCHEMA = support.OBSERVATION_SCHEMA
ACTION_SCHEMA = support.ACTION_SCHEMA
PUBLIC_K = 3
CONTROLLER_MAX = 25
REQUESTS_PER_CELL = 24
MAX_STEPS = 1800
OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/prefix_fallback_fresh_validation_20261001"
)
PROTOCOL = Path(__file__).resolve().parents[2] / (
    "docs/evidence/v4_2/p3_prefix_fallback_fresh_validation_protocol_20261001.md"
)
BEFORE_CHECKPOINT = support.BEFORE_CHECKPOINT
BEFORE_CHECKPOINT_SHA256 = support.BEFORE_CHECKPOINT_SHA256
AFTER_CHECKPOINT = support.AFTER_CHECKPOINT
AFTER_CHECKPOINT_SHA256 = support.AFTER_CHECKPOINT_SHA256
WARMUP_CHECKPOINT = support.WARMUP_CHECKPOINT
WARMUP_CHECKPOINT_SHA256 = support.WARMUP_CHECKPOINT_SHA256
ARMS = ("before_legacy", "after_legacy", "before_prefix", "after_prefix")
CELL_SPECS = tuple(
    {
        "cell_id": index,
        "scenario": "dense24",
        "env_seed": 886100 + index,
        "policy_seed": 886200 + index,
        "split": "development_fresh_prefix_validation",
    }
    for index in range(1, 7)
)

# These names are deliberately explicit.  The new runner is an independent
# source binding and does not inherit an old CELL_SPECS or baseline guard.
DEPENDENCY_FILES = {
    "runtime_rank": Path(rank_eval.__file__),
    "proposal_support": Path(support.__file__),
    "prefix_runner": Path(prefix.__file__),
    "before_prefix_wrapper": Path(before_prefix.__file__),
    "prefix_control": prefix.HELPER_PATH,
    "fresh_validator": Path(__file__).resolve(),
    "protocol": PROTOCOL,
}


class FreshValidationError(ValueError):
    """Raised when the fresh validation contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreshValidationError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
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
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable(value.value)
        except Exception:
            pass
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshValidationError(f"invalid JSON input: {target}") from exc
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise FreshValidationError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_path(path: str | Path) -> Path:
    return support._sync_path(path)


def _resolve_checkpoint(path: Path, expected_sha: str) -> Path:
    resolved = _sync_path(path)
    _require(resolved.is_file(), f"checkpoint missing: {path}")
    actual = _sha256(resolved)
    _require(actual.lower() == expected_sha.lower(), f"checkpoint SHA mismatch for {path}: {actual}")
    return resolved


def _rng_snapshot() -> Any:
    return support._rng_snapshot()


def _rng_equal(left: Any, right: Any) -> bool:
    return support._rng_equal(left, right)


def _restore_rng(snapshot: Any) -> None:
    support._restore_rng(snapshot)


def _config_snapshot(config: Any) -> dict[str, Any]:
    return support._config_snapshot(config)


def _build_inputs() -> tuple[Any, dict[str, Any]]:
    config, configs = support._build_inputs()
    _require(config.action_schema_version == ACTION_SCHEMA, "action schema changed")
    _require(config.observation_schema_version == OBSERVATION_SCHEMA, "observation schema changed")
    _require(int(config.runtime_proposal_shortlist_k) == PUBLIC_K, "public shortlist K changed")
    _require(bool(config.teacher_enabled) is False, "Teacher must be disabled")
    _require(str(config.scheduler_mode) == "execution", "scheduler mode changed")
    _require(float(config.dt) == 1.0 and float(config.physics_dt) == 0.2, "time step changed")
    _require(float(config.max_simulation_seconds) == float(MAX_STEPS), "simulation cap changed")
    _require(bool(getattr(config, "task_priority_enabled", False)) is False, "task priority changed")
    _require(float(getattr(config, "low_level_priority_time_weight", 0.0)) == 0.0,
             "task priority reward weight changed")
    _require(bool(getattr(config, "rule_flow_coordination_enabled", False)) is False,
             "rule flow coordination changed")
    return config, configs


def _runtime_dependency_shas() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in DEPENDENCY_FILES.items():
        resolved = _sync_path(path)
        _require(resolved.is_file(), f"runtime dependency missing: {path}")
        result[name] = _sha256(resolved)
    return result


def _source_bindings(config: Any) -> dict[str, Any]:
    before = support._policy_binding(BEFORE_CHECKPOINT, BEFORE_CHECKPOINT_SHA256, config, "before")
    after = support._policy_binding(AFTER_CHECKPOINT, AFTER_CHECKPOINT_SHA256, config, "after")
    freeze = support._qualified_frozen_binding(config)
    warmup = _resolve_checkpoint(WARMUP_CHECKPOINT, WARMUP_CHECKPOINT_SHA256)
    return {
        "before": before,
        "after": after,
        "qualified_frozen_groups": freeze,
        "warmup": {
            "declared": str(WARMUP_CHECKPOINT),
            "resolved": str(warmup),
            "sha256": _sha256(warmup),
            "role": "historical_input_binding_only",
            "loaded_for_current_evaluation": False,
        },
    }


def _schema_contract() -> dict[str, Any]:
    return {
        "required_arm_fields": [
            "cell", "arm", "checkpoint_sha256", "config", "status", "summary",
            "lifecycle", "runtime_capture", "metrics", "caller_rng_unchanged",
            "model_unchanged", "weights_unchanged", "teacher_called",
            "training_or_update_called", "PPO_ALLOWED",
        ],
        "required_metric_fields": [
            "generated", "admitted", "successful", "resolved", "normal", "emergency",
            "failed", "diverted", "unresolved", "request_identity", "request_identity_exact_24",
            "wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed",
            "raw_reward_excluding_rejection_penalty", "release_rejection_penalty",
            "hard_safety_clear", "technical_ok", "complete_scene_metrics_known",
        ],
        "denominator_note": "six scenes x four frozen arms; not 576 independent samples",
    }


def _build_preflight() -> dict[str, Any]:
    caller_before = _rng_snapshot()
    caller_after = None
    try:
        config, _configs = _build_inputs()
        _require(PROTOCOL.is_file(), f"protocol missing: {PROTOCOL}")
        bindings = _source_bindings(config)
        dependencies = _runtime_dependency_shas()
        config_snapshot = _config_snapshot(config)
        caller_after = _rng_snapshot()
    finally:
        if caller_after is not None:
            caller_unchanged = _rng_equal(caller_before, caller_after)
        else:
            caller_unchanged = False
        _restore_rng(caller_before)
    return {
        "stage": STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "environment_created": False,
        "PPO_ALLOWED": PPO_ALLOWED,
        "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED,
        "optimizer_used": OPTIMIZER_USED,
        "off_policy": OFF_POLICY,
        "no_training": NO_TRAINING,
        "cells": [dict(item) for item in CELL_SPECS],
        "arms": list(ARMS),
        "requests_per_cell": REQUESTS_PER_CELL,
        "max_steps": MAX_STEPS,
        "schema": {
            "observation": OBSERVATION_SCHEMA,
            "observation_dim": 88,
            "action": ACTION_SCHEMA,
            "public_shortlist_k": PUBLIC_K,
            "controller_max_proposals": CONTROLLER_MAX,
            "controller_temperature": 2.5,
        },
        "runtime": {
            "config_snapshot": config_snapshot,
            "native_rollout": "evaluate_p3_runtime_rank_repair._runtime_record",
            "prefix_after": "run_p3_prefix_fallback_20261001._arm_record",
            "prefix_before_scope": "run_p3_before_prefix_fallback_20261001._before_base_scope",
            "deterministic": False,
            "warmup_loaded": False,
        },
        "config": config_snapshot,
        "source_bindings": bindings,
        "schema_contract": _schema_contract(),
        "runtime_dependency_shas": dependencies,
        "protocol_sha256": _sha256(PROTOCOL),
        "runner_sha256": _sha256(__file__),
        "caller_rng_unchanged_before_restore": caller_unchanged,
        "caller_rng_restored": True,
        "qualification": "not_a_qualification; six-scene fixed development validation",
    }


def _validate_preflight(payload: Mapping[str, Any]) -> None:
    caller_before = _rng_snapshot()
    caller_after = None
    try:
        _validate_preflight_body(payload)
        caller_after = _rng_snapshot()
    finally:
        if caller_after is None:
            caller_after = _rng_snapshot()
        _restore_rng(caller_before)


def _validate_preflight_body(payload: Mapping[str, Any]) -> None:
    _require(payload.get("stage") == STAGE and payload.get("status") == "PREFLIGHT_OK",
             "fresh preflight is not successful")
    for key, expected in (
        ("environment_created", False), ("PPO_ALLOWED", False), ("heldout_used", False),
        ("teacher_used", False), ("optimizer_used", False), ("off_policy", True),
        ("no_training", True), ("caller_rng_restored", True),
    ):
        _require(payload.get(key) is expected, f"preflight flag {key} changed")
    _require(payload.get("runner_sha256") == _sha256(__file__), "validator changed after preflight")
    _require(payload.get("protocol_sha256") == _sha256(PROTOCOL), "protocol changed after preflight")
    _require(payload.get("runtime_dependency_shas") == _runtime_dependency_shas(),
             "runtime dependency changed after preflight")
    _require(payload.get("cells") == [dict(item) for item in CELL_SPECS], "fresh cell grid changed")
    _require(payload.get("arms") == list(ARMS), "arm grid changed")
    _require(int(payload.get("requests_per_cell", -1)) == REQUESTS_PER_CELL, "request count changed")
    _require(int(payload.get("max_steps", -1)) == MAX_STEPS, "step cap changed")
    schema = payload.get("schema")
    _require(isinstance(schema, Mapping), "schema binding missing")
    for key, expected in (
        ("observation", OBSERVATION_SCHEMA), ("observation_dim", 88),
        ("action", ACTION_SCHEMA), ("public_shortlist_k", PUBLIC_K),
        ("controller_max_proposals", CONTROLLER_MAX), ("controller_temperature", 2.5),
    ):
        _require(schema.get(key) == expected, f"schema binding {key} changed")
    config, _configs = _build_inputs()
    live_config = _config_snapshot(config)
    _require(payload.get("config") == live_config, "config changed after preflight")
    runtime = payload.get("runtime")
    _require(isinstance(runtime, Mapping) and runtime.get("config_snapshot") == live_config,
             "runtime config binding changed")
    _require(runtime.get("warmup_loaded") is False, "warmup role changed")
    _require(payload.get("source_bindings") == _source_bindings(config),
             "source/checkpoint binding changed")
    _require(payload.get("schema_contract") == _schema_contract(), "output schema contract changed")


def _write_progress(output: Path, payload: Mapping[str, Any]) -> None:
    _write_json(output / "progress.json", dict(payload))


def _steps_observed(record: Mapping[str, Any]) -> int | None:
    summary = record.get("summary")
    if isinstance(summary, Mapping):
        value = summary.get("steps")
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _record_common(
    record: Mapping[str, Any], *, arm: str, cell: Mapping[str, Any], checkpoint: Path,
    expected_sha: str, config: Any, caller_rng_unchanged: bool,
    runtime_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(record)
    result.update({
        "validation_stage": STAGE,
        "arm": arm,
        "cell": dict(cell),
        "checkpoint_declared": str(checkpoint),
        "checkpoint_sha256_expected": expected_sha,
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
        "config": _config_snapshot(config),
        "caller_rng_unchanged": bool(caller_rng_unchanged),
        "teacher_called": False,
        "training_or_update_called": False,
        "PPO_ALLOWED": False,
    })
    if runtime_snapshot is not None:
        result["runtime_capture"] = dict(runtime_snapshot)
    result.setdefault("model_unchanged", result.get("weights_unchanged"))
    result.setdefault("weights_unchanged", result.get("model_unchanged"))
    result["sampler_intervention"] = {
        "name": "native_legacy_student_request_details" if arm.endswith("legacy") else "sampled_first_prefix_fallback",
        "public_shortlist_k": PUBLIC_K,
        "controller_max_proposals": CONTROLLER_MAX if arm.endswith("prefix") else None,
        "off_policy": bool(arm.endswith("prefix")),
        "training": False,
    }
    try:
        result["metrics"] = support._record_metrics(result)
    except BaseException as exc:
        result["metrics"] = {
            "status": result.get("status"),
            "technical_ok": False,
            "complete_scene_metrics_known": False,
            "censored": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return result


def _native_arm_record(*, config: Any, cell: Mapping[str, Any], arm: str, output: Path) -> dict[str, Any]:
    _require(arm in ("before_legacy", "after_legacy"), f"invalid native arm {arm}")
    checkpoint, expected_sha = (
        (BEFORE_CHECKPOINT, BEFORE_CHECKPOINT_SHA256)
        if arm == "before_legacy" else (AFTER_CHECKPOINT, AFTER_CHECKPOINT_SHA256)
    )
    caller_before = _rng_snapshot()
    caller_rng_restore_error = None
    policy = None
    before_digest = None
    resolved = checkpoint
    capture: dict[str, Any] = {"replay_factory_calls": 0}
    record: dict[str, Any]
    record_output = output / "rollouts" / f"cell_{int(cell['cell_id']):02d}_{arm}"
    wall_start = time.monotonic()
    original_capture_class = support._CapturedTerminalEnv

    class _ProgressCapturedTerminalEnv(original_capture_class):
        """Passively expose sparse native-step progress for startup monitoring."""

        def step(self, *args: Any, **kwargs: Any):
            result = super().step(*args, **kwargs)
            observed = int(capture.get("observed_step_count", 0) or 0) + 1
            capture["observed_step_count"] = observed
            capture["first_step_observed"] = True
            absolute_step = int(getattr(self, "step_count", observed) or observed)
            if observed == 1 or observed % 50 == 0:
                active = getattr(self, "active", {})
                active_count = len(active) if isinstance(active, Mapping) else None
                captured_rows = getattr(self, "_captured_request_rows", []) or []
                _write_progress(output, {
                    "stage": STAGE,
                    "phase": "native_step",
                    "status": "RUNNING",
                    "cell_id": int(cell["cell_id"]),
                    "arm": arm,
                    "absolute_step": absolute_step,
                    "observed_steps": observed,
                    "generated_requests": len(captured_rows),
                    "active_uavs": active_count,
                })
            return result

    # The existing passive factory resolves this module-level class at call
    # time.  A scoped subclass adds progress observation while preserving the
    # native constructor, step arguments, and returned transition unchanged.
    support._CapturedTerminalEnv = _ProgressCapturedTerminalEnv
    original_replay_factory = rank_eval.terminal_lifecycle_metrics.TerminalParallelEnv

    def _replay_factory(*args: Any, **kwargs: Any) -> Any:
        capture["replay_factory_calls"] = int(capture.get("replay_factory_calls", 0) or 0) + 1
        env = original_replay_factory(*args, **kwargs)
        capture["replay_env"] = env
        return env

    rank_eval.terminal_lifecycle_metrics.TerminalParallelEnv = _replay_factory
    try:
        resolved = _resolve_checkpoint(checkpoint, expected_sha)
        policy, _payload = rank_eval._load_policy(resolved, config)
        before_digest = rank_eval._model_digest(policy)
        with support._capture_runtime_env(capture):
            record = rank_eval._runtime_record(
                policy=policy,
                checkpoint=resolved,
                config=config,
                scenario=str(cell["scenario"]),
                seed=int(cell["env_seed"]),
                policy_seed=int(cell["policy_seed"]),
                output=record_output,
                smoke=False,
            )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        record = rank_eval._failure_record(
            scenario=str(cell["scenario"]), seed=int(cell["env_seed"]), checkpoint=resolved,
            requests=REQUESTS_PER_CELL, error=exc, weights_before=before_digest,
            weights_unchanged=(
                None if policy is None or before_digest is None
                else before_digest == rank_eval._model_digest(policy)
            ), policy_seed=int(cell["policy_seed"]),
        )
    finally:
        support._CapturedTerminalEnv = original_capture_class
        rank_eval.terminal_lifecycle_metrics.TerminalParallelEnv = original_replay_factory
        capture["aliases_restored"] = bool(
            support._CapturedTerminalEnv is original_capture_class
            and rank_eval.terminal_lifecycle_metrics.TerminalParallelEnv is original_replay_factory
        )
        caller_after_before_restore = _rng_snapshot()
        caller_rng_unchanged_before_restore = _rng_equal(caller_before, caller_after_before_restore)
        try:
            _restore_rng(caller_before)
            caller_after_restore = _rng_snapshot()
            caller_rng_restored = _rng_equal(caller_before, caller_after_restore)
        except Exception as exc:
            caller_rng_restored = False
            caller_rng_restore_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        caller_unchanged = caller_rng_restored
    record = support._reclassify_incomplete_lifecycle(record)
    after_digest = None if policy is None else rank_eval._model_digest(policy)
    try:
        runtime_snapshot = support._live_runtime_snapshot(capture, config)
    except BaseException as exc:
        runtime_snapshot = {
            "available": False,
            "factory_calls": int(capture.get("factory_calls", 0) or 0),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    runtime_snapshot["replay_factory_calls"] = int(capture.get("replay_factory_calls", 0) or 0)
    runtime_snapshot["aliases_restored"] = bool(capture.get("aliases_restored", False))
    runtime_snapshot["native_step_observer"] = {
        "available": True,
        "first_step_observed": bool(capture.get("first_step_observed", False)),
        "observed_steps": int(capture.get("observed_step_count", 0) or 0),
    }
    record.update({
        "label": "runtime_rank_fresh_validation",
        "scenario": str(cell["scenario"]),
        "seed": int(cell["env_seed"]),
        "policy_seed": int(cell["policy_seed"]),
        "model_digest_before": before_digest,
        "model_digest_after": after_digest,
        "model_unchanged": bool(before_digest is not None and before_digest == after_digest),
        "weights_unchanged": bool(before_digest is not None and before_digest == after_digest),
        "runtime_dependency_shas": _runtime_dependency_shas(),
        "wall_time_seconds": float(time.monotonic() - wall_start),
        "safety_nonzero": bool(record.get("safety_nonzero", False)),
        "caller_rng_unchanged_before_restore": bool(caller_rng_unchanged_before_restore),
        "caller_rng_restored": bool(caller_rng_restored),
        "caller_rng_restore_error": caller_rng_restore_error,
    })
    return _record_common(
        record, arm=arm, cell=cell, checkpoint=resolved, expected_sha=expected_sha,
        config=config, caller_rng_unchanged=caller_unchanged,
        runtime_snapshot=runtime_snapshot,
    )


def _prefix_arm_record(*, config: Any, cell: Mapping[str, Any], arm: str, output: Path) -> dict[str, Any]:
    _require(arm in ("before_prefix", "after_prefix"), f"invalid prefix arm {arm}")
    caller_before = _rng_snapshot()
    caller_after = None
    try:
        if arm == "after_prefix":
            record = prefix._arm_record(config=config, cell=cell, output=output)
        else:
            with before_prefix._before_base_scope():
                record = prefix._arm_record(config=config, cell=cell, output=output)
    except (KeyboardInterrupt, SystemExit):
        raise
    finally:
        caller_after = _rng_snapshot()
        caller_unchanged = _rng_equal(caller_before, caller_after)
        _restore_rng(caller_before)
    result = dict(record)
    result["validation_stage"] = STAGE
    result["arm"] = arm
    result["cell"] = dict(cell)
    result["caller_rng_unchanged"] = bool(caller_unchanged and result.get("caller_rng_unchanged") is True)
    result["fresh_runtime_dependency_shas"] = _runtime_dependency_shas()
    result["fresh_protocol_sha256"] = _sha256(PROTOCOL)
    result["sampler_intervention"] = {
        "name": "sampled_first_prefix_fallback",
        "public_shortlist_k": PUBLIC_K,
        "controller_max_proposals": CONTROLLER_MAX,
        "off_policy": True,
        "training": False,
    }
    # The existing arm function already records source/checkpoint/config,
    # lifecycle, safety, and prefix controller metadata.  Recompute metrics
    # after adding the fresh caller binding, without replacing unknown values.
    try:
        result["metrics"] = support._record_metrics(result)
    except BaseException as exc:
        result["metrics"] = {
            "status": result.get("status"),
            "technical_ok": False,
            "complete_scene_metrics_known": False,
            "censored": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return result


def _arm_record(*, config: Any, cell: Mapping[str, Any], arm: str, output: Path) -> dict[str, Any]:
    if arm.endswith("legacy"):
        return _native_arm_record(config=config, cell=cell, arm=arm, output=output)
    return _prefix_arm_record(config=config, cell=cell, arm=arm, output=output)


def _write_arm(output: Path, cell: Mapping[str, Any], arm: str, record: Mapping[str, Any]) -> Path:
    path = output / "arms" / f"cell_{int(cell['cell_id']):02d}_{arm}.json"
    _write_json(path, record, refuse_existing=True)
    return path


def _metric(record: Mapping[str, Any], key: str) -> Any:
    metrics = record.get("metrics")
    return metrics.get(key) if isinstance(metrics, Mapping) else None


def _ids(record: Mapping[str, Any], kinds: set[str]) -> list[int] | None:
    try:
        values = support._issue_ids(record, kinds)
    except BaseException:
        return None
    return None if values is None else sorted(int(value) for value in values)


def _delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "generated", "admitted", "successful", "resolved", "normal", "emergency",
        "failed", "diverted", "unresolved", "wait_seconds",
        "request_elapsed_total_seconds", "energy_total_consumed",
        "raw_reward_excluding_rejection_penalty", "release_rejection_penalty",
    )
    values: dict[str, Any] = {}
    for key in keys:
        lv, rv = left.get(key), right.get(key)
        values[key] = None if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in (lv, rv)) else float(lv) - float(rv)
    return values


def _compact_arm(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    return {
        "arm": record.get("arm"),
        "status": record.get("status"),
        "checkpoint_sha256": record.get("checkpoint_sha256"),
        "model_unchanged": record.get("model_unchanged"),
        "weights_unchanged": record.get("weights_unchanged"),
        "caller_rng_unchanged": record.get("caller_rng_unchanged"),
        "observer_rng_unchanged": record.get("observer_rng_unchanged"),
        "teacher_called": record.get("teacher_called"),
        "training_or_update_called": record.get("training_or_update_called"),
        "safety_nonzero": record.get("safety_nonzero"),
        "metrics": {
            key: metrics.get(key)
            for key in (
                "status", "technical_ok", "natural_terminal", "censored", "complete_scene_metrics_known",
                "generated", "admitted", "successful", "resolved", "normal", "emergency", "failed",
                "diverted", "unresolved", "request_identity_ids", "request_identity_exact_24",
                "wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed",
                "raw_reward_excluding_rejection_penalty", "release_rejection_penalty", "hard_safety_clear",
                "error",
            )
            if key in metrics
        },
        "arm_file": record.get("_arm_file"),
    }


def _pair_summary(
    cell: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], arm_paths: Mapping[str, Path],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage": STAGE,
        "cell": dict(cell),
        "arms": {arm: _compact_arm(records[arm]) for arm in records},
        "arm_paths": {arm: str(path) for arm, path in arm_paths.items()},
        "arm_order": list(records),
    }
    technical_error = None
    for arm, record in records.items():
        reason = _technical_failure_arm(record)
        if reason:
            technical_error = f"{arm}: {reason}"
            break
    if technical_error:
        result["status"] = "PAIR_TECHNICAL_ERROR"
        result["technical_error"] = technical_error
        result["technical_ok"] = False
        return result
    if set(records) != set(ARMS):
        result["status"] = "PAIR_INCOMPLETE"
        result["technical_ok"] = True
        result["technical_error"] = None
        return result
    result["status"] = "PAIR_OK"
    result["technical_ok"] = all(_metric(record, "technical_ok") is True for record in records.values())
    result["all_four_natural"] = all(_metric(record, "complete_scene_metrics_known") is True for record in records.values())
    result["all_request_identity_exact_24"] = all(_metric(record, "request_identity_exact_24") is True for record in records.values())
    identities = [_metric(record, "request_identity_ids") for record in records.values()]
    result["request_identity_match"] = bool(identities and all(value == identities[0] for value in identities)) if all(value is not None for value in identities) else None
    result["hard_safety_clear"] = all(_metric(record, "hard_safety_clear") is True for record in records.values())
    for left_name, right_name, delta_name in (
        ("after_legacy", "before_legacy", "after_legacy_minus_before_legacy"),
        ("before_prefix", "before_legacy", "before_prefix_minus_before_legacy"),
        ("after_prefix", "after_legacy", "after_prefix_minus_after_legacy"),
        ("after_prefix", "before_prefix", "after_prefix_minus_before_prefix"),
    ):
        result[delta_name] = _delta(records[left_name].get("metrics", {}), records[right_name].get("metrics", {}))
    for target, refs in (("after_legacy", ("before_legacy",)), ("before_prefix", ("before_legacy",)), ("after_prefix", ("after_legacy", "before_prefix"))):
        for reference in refs:
            target_record, ref_record = records[target], records[reference]
            for kind in ("failed", "diverted", "unresolved"):
                target_ids = _ids(target_record, {kind})
                ref_ids = _ids(ref_record, {kind})
                result[f"new_{kind}_arrival_ids_{target}_vs_{reference}"] = (
                    None if target_ids is None or ref_ids is None else sorted(set(target_ids) - set(ref_ids))
                )
    result["request_identity_exact_24"] = result["all_request_identity_exact_24"] and result["request_identity_match"] is True
    result["technical_ok"] = bool(result["technical_ok"] and result["hard_safety_clear"])
    return result


def _technical_config_snapshot(record: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    """Resolve the serialized config field used by native and prefix arms.

    Native arms serialize ``config`` while the prefix runner serializes the
    before/after snapshots.  Compare every field that is present through the
    canonical JSON representation and keep the original record unchanged.
    """

    snapshots: dict[str, tuple[Mapping[str, Any], str]] = {}
    for field in ("config", "config_snapshot", "config_snapshot_before"):
        if field not in record:
            continue
        value = record.get(field)
        if not isinstance(value, Mapping):
            return None, f"{field} snapshot is missing or invalid"
        try:
            canonical = json.dumps(
                _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            return None, f"{field} snapshot is not canonical JSON: {exc}"
        snapshots[field] = (value, canonical)

    if not snapshots:
        return None, "config snapshot is missing"
    if "config" in snapshots and "config_snapshot" in snapshots:
        if snapshots["config"][1] != snapshots["config_snapshot"][1]:
            return None, "config snapshots differ"
    after = snapshots.get("config_snapshot") or snapshots.get("config")
    before = snapshots.get("config_snapshot_before")
    if before is not None and after is not None and before[1] != after[1]:
        return None, "config snapshots differ"
    return after[0] if after is not None else None, None


def _technical_failure_arm(record: Mapping[str, Any]) -> str | None:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    if record.get("error") or metrics.get("error"):
        return str(record.get("error") or metrics.get("error"))
    if record.get("status") in {"failed", "technical_error", "safety_violation"}:
        return f"record status={record.get('status')}"
    for key in ("model_unchanged", "weights_unchanged", "caller_rng_unchanged"):
        if record.get(key) is not True:
            return f"{key} is not true"
    if record.get("observer_rng_unchanged") is False:
        return "observer_rng_unchanged is false"
    if record.get("teacher_called") is not False or record.get("training_or_update_called") is not False:
        return "training/Teacher flag changed"
    if record.get("safety_nonzero") is True or metrics.get("hard_safety_clear") is not True:
        return "hard-safety check is not clear"
    expected_checkpoint = (
        BEFORE_CHECKPOINT_SHA256 if str(record.get("arm", "")).startswith("before")
        else AFTER_CHECKPOINT_SHA256
    )
    if record.get("checkpoint_sha256") != expected_checkpoint:
        return "checkpoint binding changed"
    config, config_error = _technical_config_snapshot(record)
    if config_error is not None:
        return config_error
    if config is None:
        return "config snapshot is missing"
    if config.get("action_schema_version") != ACTION_SCHEMA or config.get("observation_schema_version") != OBSERVATION_SCHEMA:
        return "schema binding changed"
    runtime = record.get("runtime_capture")
    if not isinstance(runtime, Mapping) or runtime.get("available") is not True:
        return "runtime capture is unavailable"
    if runtime.get("factory_calls") != 1 or runtime.get("replay_factory_calls") != 1:
        return "environment factory counts are incomplete"
    if str(record.get("arm", "")).endswith("prefix"):
        control = record.get("prefix_fallback_control")
        if not isinstance(control, Mapping):
            return "prefix controller metadata is missing"
        for key in ("source_distributions_immutable", "sampler_aliases_restored", "runtime_aliases_restored"):
            if control.get(key) is not True:
                return f"prefix controller binding {key} is not true"
    if metrics.get("technical_ok") is not True:
        if metrics.get("censored") is True and metrics.get("error") is None:
            return None
        return "arm technical fields are incomplete"
    return None


def _run_cell(output: Path, config: Any, cell: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for arm in ARMS:
        _write_progress(output, {
            "stage": STAGE, "phase": "arm", "status": "RUNNING",
            "cell_id": int(cell["cell_id"]), "arm": arm, "steps_observed": None,
        })
        try:
            record = _arm_record(config=config, cell=cell, arm=arm, output=output)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            checkpoint, expected_sha = (
                (BEFORE_CHECKPOINT, BEFORE_CHECKPOINT_SHA256)
                if arm.startswith("before") else (AFTER_CHECKPOINT, AFTER_CHECKPOINT_SHA256)
            )
            resolved = _sync_path(checkpoint)
            record = {
                "validation_stage": STAGE,
                "arm": arm,
                "cell": dict(cell),
                "checkpoint_declared": str(checkpoint),
                "checkpoint_sha256_expected": expected_sha,
                "checkpoint_sha256": _sha256(resolved) if resolved.is_file() else None,
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "model_unchanged": None,
                "weights_unchanged": None,
                "caller_rng_unchanged": False,
                "teacher_called": False,
                "training_or_update_called": False,
                "PPO_ALLOWED": False,
                "metrics": {
                    "status": "failed", "technical_ok": False,
                    "complete_scene_metrics_known": False, "censored": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            }
        record = dict(record)
        record["_arm_file"] = str(output / "arms" / f"cell_{int(cell['cell_id']):02d}_{arm}.json")
        path = _write_arm(output, cell, arm, record)
        paths[arm] = path
        records[arm] = record
        _write_progress(output, {
            "stage": STAGE, "phase": "arm", "status": "ARM_WRITTEN",
            "cell_id": int(cell["cell_id"]), "arm": arm,
            "steps_observed": _steps_observed(record),
            "generated_requests": _metric(record, "generated"),
        })
        if _technical_failure_arm(record):
            summary = _pair_summary(cell, records, paths)
            summary["technical_error"] = _technical_failure_arm(record)
            _write_json(output / "pairs" / f"cell_{int(cell['cell_id']):02d}.json", summary, refuse_existing=True)
            return summary
    summary = _pair_summary(cell, records, paths)
    _write_json(output / "pairs" / f"cell_{int(cell['cell_id']):02d}.json", summary, refuse_existing=True)
    _write_progress(output, {
        "stage": STAGE, "phase": "pair", "status": summary.get("status"),
        "cell_id": int(cell["cell_id"]), "arms_completed": list(records),
    })
    return summary


def _aggregate(summaries: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    keys = (
        "generated", "admitted", "successful", "resolved", "normal", "emergency",
        "failed", "diverted", "unresolved", "wait_seconds",
        "request_elapsed_total_seconds", "energy_total_consumed",
        "raw_reward_excluding_rejection_penalty", "release_rejection_penalty",
    )
    result: dict[str, Any] = {"scenes": len(summaries), "arm": arm}
    for key in keys:
        values = []
        for summary in summaries:
            row = summary.get("arms", {}).get(arm, {}).get("metrics", {})
            value = row.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                values = []
                break
            values.append(float(value))
        result[key] = float(sum(values)) if values and len(values) == len(summaries) else None
    result["known_complete"] = all(
        summary.get("arms", {}).get(arm, {}).get("metrics", {}).get("complete_scene_metrics_known") is True
        for summary in summaries
    ) if summaries else False
    return result


def _aggregate_ids(summaries: Sequence[Mapping[str, Any]], arm: str, kind: str) -> list[int] | None:
    values: set[int] = set()
    for summary in summaries:
        row = summary.get("arms", {}).get(arm)
        if not isinstance(row, Mapping):
            return None
        path = row.get("arm_file")
        if not path or not Path(path).is_file():
            return None
        record = _read_json(path)
        ids = _ids(record, {kind})
        if ids is None:
            return None
        values.update(ids)
    return sorted(values)


def _business_review(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(summaries)
    aggregates = {arm: _aggregate(rows, arm) for arm in ARMS}
    natural = len(rows) == len(CELL_SPECS) and all(
        summary.get("status") == "PAIR_OK" and summary.get("all_four_natural") is True
        for summary in rows
    )
    technical = len(rows) == len(CELL_SPECS) and all(
        summary.get("technical_ok") is True for summary in rows
    )
    exact = len(rows) == len(CELL_SPECS) and all(
        summary.get("request_identity_exact_24") is True for summary in rows
    )
    no_new = {}
    for target, refs in (("after_prefix", ("after_legacy", "before_prefix")),):
        for ref in refs:
            keys = [f"new_{kind}_arrival_ids_{target}_vs_{ref}" for kind in ("failed", "diverted", "unresolved")]
            no_new[f"{target}_vs_{ref}"] = (
                all(isinstance(summary.get(key), list) and not summary.get(key) for summary in rows for key in keys)
                if rows else False
            )
    diagnostic_no_new = {}
    for kind in ("failed", "diverted", "unresolved"):
        key = f"new_{kind}_arrival_ids_before_prefix_vs_before_legacy"
        diagnostic_no_new[key] = all(
            isinstance(summary.get(key), list) and not summary.get(key) for summary in rows
        ) if rows else False
    refs = {}
    for target, reference in (("after_prefix", "after_legacy"), ("after_prefix", "before_prefix")):
        candidate, baseline = aggregates[target], aggregates[reference]
        complete = bool(candidate.get("known_complete") and baseline.get("known_complete"))
        refs[f"{target}_vs_{reference}"] = {
            "normal_not_lower": None if not complete else candidate["normal"] >= baseline["normal"],
            "successful_not_lower": None if not complete else candidate["successful"] >= baseline["successful"],
            "wait_elapsed_energy_not_higher": None if not complete else all(
                candidate[key] <= baseline[key] + 1e-8
                for key in ("wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed")
            ),
        }
    after, before = aggregates["after_legacy"], aggregates["before_legacy"]
    strict_gain = None if not (after.get("known_complete") and aggregates["after_prefix"].get("known_complete")) else (
        aggregates["after_prefix"]["normal"] > after["normal"] or aggregates["after_prefix"]["failed"] < after["failed"]
    )
    criterion = bool(
        natural and technical and exact and all(no_new.values()) and
        all(value["normal_not_lower"] is True and value["successful_not_lower"] is True and value["wait_elapsed_energy_not_higher"] is True for value in refs.values()) and
        strict_gain is True
    )
    return {
        "pair_count": len(rows),
        "all_four_natural": natural,
        "all_technical_ok": technical,
        "all_request_identity_exact_24": exact,
        "no_new_failed_diverted_unresolved": no_new,
        "aggregates": aggregates,
        "reference_comparisons": refs,
        "diagnostic_before_prefix_vs_before_legacy_no_new": diagnostic_no_new,
        "after_prefix_vs_after_legacy_strict_gain": strict_gain,
        "criterion_pass": criterion,
        "development_candidate": criterion,
        "PPO_ALLOWED": False,
        "qualification": "not_a_qualification",
    }


def _pilot_artifacts(output: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    pair = output / "pairs" / "cell_01.json"
    arm_paths = {
        arm: output / "arms" / f"cell_01_{arm}.json"
        for arm in ARMS
        if (output / "arms" / f"cell_01_{arm}.json").is_file()
    }
    return {
        "pair_path": str(pair),
        "pair_sha256": _sha256(pair) if pair.is_file() else None,
        "arm_paths": {arm: str(path) for arm, path in arm_paths.items()},
        "arm_sha256": {arm: _sha256(path) for arm, path in arm_paths.items()},
        "summary_status": summary.get("status"),
    }


def _validate_pilot_reuse(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    pilot = _read_json(output / "pilot.json")
    _require(pilot.get("preflight_sha256") == _sha256(output / "preflight.json"), "pilot preflight binding changed")
    _require(pilot.get("preflight_runner_sha256") == preflight.get("runner_sha256"), "pilot validator binding changed")
    _require(pilot.get("status") in {"PILOT_OK", "PILOT_INCOMPLETE"}, "pilot has technical failure")
    artifacts = pilot.get("pilot_artifacts")
    _require(isinstance(artifacts, Mapping), "pilot artifacts missing")
    pair_path = Path(artifacts.get("pair_path", ""))
    _require(pair_path.is_file() and _sha256(pair_path) == artifacts.get("pair_sha256"), "pilot pair changed")
    arm_sha = artifacts.get("arm_sha256")
    _require(isinstance(arm_sha, Mapping) and set(arm_sha) == set(ARMS), "pilot arm set changed")
    for arm in ARMS:
        path = Path(artifacts.get("arm_paths", {}).get(arm, ""))
        _require(path.is_file() and _sha256(path) == arm_sha[arm], f"pilot arm {arm} changed")
    pair = _read_json(pair_path)
    _require(pair.get("cell", {}).get("cell_id") == 1, "pilot cell changed")
    _require(set(pair.get("arms", {})) == set(ARMS), "pilot arm records incomplete")
    _require(pair.get("technical_ok") is True, "pilot technical checks are incomplete")
    return pilot


def run_preflight(output: Path) -> dict[str, Any]:
    _require(not (output / "preflight.json").exists(), "preflight artifact already exists")
    payload = _build_preflight()
    _write_json(output / "preflight.json", payload, refuse_existing=True)
    _write_progress(output, {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK", "environment_created": False,
    })
    return payload


def _phase_inputs(output: Path) -> tuple[dict[str, Any], Any]:
    preflight = _read_json(output / "preflight.json")
    _validate_preflight(preflight)
    config, _configs = _build_inputs()
    return preflight, config


def run_pilot(output: Path) -> dict[str, Any]:
    preflight, config = _phase_inputs(output)
    _require(not (output / "pilot.json").exists(), "pilot artifact already exists")
    summary = _run_cell(output, config, CELL_SPECS[0])
    status = "PILOT_TECHNICAL_ERROR" if summary.get("status") == "PAIR_TECHNICAL_ERROR" else (
        "PILOT_OK" if summary.get("status") == "PAIR_OK" else "PILOT_INCOMPLETE"
    )
    payload = {
        "stage": STAGE, "phase": "pilot", "status": status,
        "PPO_ALLOWED": False, "preflight_sha256": _sha256(output / "preflight.json"),
        "preflight_runner_sha256": preflight["runner_sha256"], "pair": summary,
        "pilot_artifacts": _pilot_artifacts(output, summary),
    }
    _write_json(output / "pilot.json", payload, refuse_existing=True)
    return payload


def _write_phase_error(output: Path, phase: str, exc: BaseException) -> None:
    _write_json(output / "error.json", {
        "stage": STAGE, "phase": phase, "status": "ERROR",
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }, refuse_existing=True)


def run_full(output: Path) -> dict[str, Any]:
    preflight, config = _phase_inputs(output)
    _require(not (output / "full.json").exists(), "full artifact already exists")
    pilot = _validate_pilot_reuse(output, preflight)
    summaries = [pilot["pair"]]
    for cell in CELL_SPECS[1:]:
        summary = _run_cell(output, config, cell)
        summaries.append(summary)
        if summary.get("status") == "PAIR_TECHNICAL_ERROR":
            payload = {
                "stage": STAGE, "phase": "full", "status": "FULL_TECHNICAL_ERROR",
                "PPO_ALLOWED": False, "preflight_sha256": _sha256(output / "preflight.json"),
                "pilot_sha256": _sha256(output / "pilot.json"), "pair_summaries": summaries,
                "business_review": _business_review(summaries),
            }
            _write_json(output / "full.json", payload, refuse_existing=True)
            return payload
    review = _business_review(summaries)
    payload = {
        "stage": STAGE, "phase": "full", "status": "FULL_OK",
        "PPO_ALLOWED": False, "preflight_sha256": _sha256(output / "preflight.json"),
        "pilot_sha256": _sha256(output / "pilot.json"),
        "pair_count": len(summaries), "pair_summaries": summaries,
        "business_review": review,
        "qualification": "not_a_qualification; fixed six-scene development validation",
    }
    _write_json(output / "full.json", payload, refuse_existing=True)
    _write_progress(output, {"stage": STAGE, "phase": "full", "status": "FULL_OK", "pair_count": len(summaries)})
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir)
    try:
        if args.phase == "preflight":
            run_preflight(output)
        elif args.phase == "pilot":
            run_pilot(output)
        else:
            run_full(output)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        try:
            _write_phase_error(output, args.phase, exc)
        except BaseException:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
