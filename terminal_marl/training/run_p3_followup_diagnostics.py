"""Run the pre-registered, source-bound P3 follow-up diagnostics.

The phases are intentionally explicit:

``preflight``
    Read the six fixed source records/tapes and checkpoint metadata.  This
    phase creates no environment and performs no replay.
``replay``
    Replay all six source tapes with the frozen ``3fe7+a82`` policy, collect
    lightweight pre-step candidate metadata, and register the deterministic
    24-occurrence sample.
``pilot``
    Re-run the source tapes exactly, deep-copy only the first selected state,
    and evaluate one continuation-seed paired WAIT/RELEASE diagnostic.
``diagnose``
    Re-run the source tapes exactly, deep-copy only the 24 selected states,
    and evaluate all four continuation seeds for every selected state.

No phase trains a model, invokes an optimizer or teacher, consumes held-out
data, or changes the source reward, mask, observation, or proposal contract.
The paired diagnostic keeps the source first-step requests for every other
UID fixed to the recorded tape row.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import enum
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.core.low_level_joint_action_v4_2 import (
    WAIT_GATE,
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.training import diagnose_p3_ppo_credit_signal as credit
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import paired_gate_diagnostic
from terminal_marl.training import run_p3_task_neutral_pilot as pilot


STAGE = "P3_TASK_NEUTRAL_FOLLOWUP_DIAGNOSTICS_V1"
PPO_ALLOWED = False
OBSERVATION_DIM = 88
EXPECTED_EPISODES = 6
EXPECTED_REQUESTS_PER_EPISODE = 24
EXPECTED_TOTAL_REQUESTS = EXPECTED_EPISODES * EXPECTED_REQUESTS_PER_EPISODE
EXPECTED_STEPS = (725, 864, 699, 729, 724, 756)
EXPECTED_TOTAL_STEPS = sum(EXPECTED_STEPS)
EXPECTED_ENV_SEEDS = (877101, 877102, 877103, 877104, 877105, 877106)
EXPECTED_POLICY_SEEDS = (878201, 878202, 878203, 878204, 878205, 878206)
EXPECTED_REQUESTS_PER_EPISODE = 24
ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_episodic_learning_20260929"
)
TRAINING_ROOT = ROOT / "full" / "training" / "batch_01" / "training"
MAX_REPLAY_STEPS = 1000
MAX_TOTAL_REPLAY_STEPS = 10800
GAMMA = 1.0
GAE_LAMBDA = 1.0
REWARD_SCALE = 0.01
SELECTION_SEED = 9303001
SELECTION_SIZE = 24
MAX_SHARD_COUNT = 4
CONTINUATION_SEEDS = (9303101, 9303102, 9303103, 9303104)
PILOT_MAX_STEPS = 1800

SOURCE_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt"
)
SOURCE_CHECKPOINT_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
WARMUP_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_episodic_learning_20260929/warmup_scaled/warmup.pt"
)
WARMUP_CHECKPOINT_SHA256 = "a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710"
SOURCE_WARMUP_MODEL_DIGEST = "a8a133863bc03627452814535aa938ed062fe0496b35678d0fe9333524503f07"

# This is the actual after checkpoint produced by the six source episodic
# cells.  Its producer summary is part of the binding and is read during
# preflight; it is not a new continuation checkpoint.
AFTER_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_episodic_learning_20260929/full/training/batch_01/policy.pt"
)
AFTER_CHECKPOINT_SHA256 = "a7bcf3d1e6f4bda82d0d051ec25966a18f267309f1d6030689b52d90e63aa366"
AFTER_MODEL_DIGEST = "c47ffa5855779b3455ee3b0ef243365780389c6000903c72e5ba1b5c865b2f3e"
AFTER_SUMMARY = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_episodic_learning_20260929/full/training/batch_01/summary.json"
)


class FollowupDiagnosticError(ValueError):
    """Raised when a fixed follow-up protocol contract is not met."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FollowupDiagnosticError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return _jsonable(value.value)
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"JSON object required: {path}")
    return dict(value)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    declared = Path(path)
    return Path(neutral_eval._synced_path(declared))


def _config_snapshot(config: Any) -> dict[str, Any]:
    snapshot = neutral_eval._config_snapshot(config)
    _require(isinstance(snapshot, Mapping), "neutral config snapshot is not a mapping")
    return dict(snapshot)


def _runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_version_string": sys.version,
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }


def _loaded_code_manifest() -> dict[str, Any]:
    """Hash the Python files actually loaded into this runtime."""
    package_root = Path(__file__).resolve().parents[1]
    seen: dict[str, dict[str, Any]] = {}
    for module_name, module in sorted(sys.modules.items()):
        if module_name != "__main__" and not str(module_name).startswith("terminal_marl"):
            continue
        module_path = getattr(module, "__file__", None)
        if not module_path or not str(module_path).endswith(".py"):
            continue
        path = Path(module_path).resolve()
        if not path.is_file() or package_root not in path.parents:
            continue
        key = str(path)
        seen.setdefault(key, {
            "module": str(module_name),
            "path": key,
            "sha256": _sha256(path),
        })
    files = [seen[key] for key in sorted(seen)]
    return {"package_root": str(package_root), "count": len(files), "files": files}


def _artifact(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    declared = Path(path)
    resolved = _resolve(declared)
    _require(resolved.is_file(), f"artifact missing: {declared}")
    actual = _sha256(resolved)
    if expected_sha256 is not None:
        _require(actual.lower() == expected_sha256.lower(),
                 f"artifact SHA mismatch: {declared}: expected {expected_sha256}, got {actual}")
    return {
        "declared": str(declared),
        "resolved": str(resolved),
        "sha256": actual,
        "expected_sha256": expected_sha256,
        "size_bytes": int(resolved.stat().st_size),
    }


def _protocol_metadata(config_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observation_schema": str(config_snapshot.get("observation_schema_version", "phaseA_low_obs_v2_1_retry_state")),
        "observation_dimension": OBSERVATION_DIM,
        "action_schema": str(config_snapshot.get("action_schema_version", "low_action_schema_v4_2_hierarchical")),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "warmup_checkpoint_sha256": WARMUP_CHECKPOINT_SHA256,
        "source_warmup_model_digest": SOURCE_WARMUP_MODEL_DIGEST,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "reward_scale": REWARD_SCALE,
        "selection_seed": SELECTION_SEED,
        "selection_size": SELECTION_SIZE,
        "selection_without_replacement": True,
        "continuation_seeds": list(CONTINUATION_SEEDS),
        "pilot_max_steps": PILOT_MAX_STEPS,
        "max_replay_steps_per_episode": MAX_REPLAY_STEPS,
        "max_total_replay_steps": MAX_TOTAL_REPLAY_STEPS,
        "PPO_ALLOWED": PPO_ALLOWED,
        "heldout_used": False,
        "optimizer_used": False,
        "teacher_used": False,
        "diagnostic_performance_claim": False,
        "conditional_estimate": "other_uids_first_step_requests_fixed_to_recorded_tape",
        "after_alignment": {
            "checkpoint_sha256": AFTER_CHECKPOINT_SHA256,
            "model_digest": AFTER_MODEL_DIGEST,
            "summary": str(AFTER_SUMMARY),
            "d449_endpoint_used": False,
        },
    }


def _compact_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    tape = Path(str(cell["tape_path"]))
    record = Path(str(cell["record_path"]))
    return {
        "index": int(cell["index"]),
        "scenario": "dense24",
        "env_seed": int(cell["env_seed"]),
        "policy_seed": int(cell["policy_seed"]),
        "steps": int(cell["steps"]),
        "requests": int(cell["requests"]),
        "record_path": str(record),
        "record_sha256": _sha256(record),
        "tape_path": str(tape),
        "tape_sha256": _sha256(tape),
    }


def _episode_path(index: int) -> Path:
    return TRAINING_ROOT / f"episode_{int(index):02d}" / "record.json"


def _preflight_cells() -> dict[str, Any]:
    """Read and validate the six registered records/tapes without replay."""
    cells: list[dict[str, Any]] = []
    total_steps = 0
    for index, (expected_steps, env_seed, policy_seed) in enumerate(
        zip(EXPECTED_STEPS, EXPECTED_ENV_SEEDS, EXPECTED_POLICY_SEEDS, strict=True),
        start=1,
    ):
        declared_record = _episode_path(index)
        record_path = _resolve(declared_record)
        record = _read_json(record_path)
        _require(int(record.get("seed", -1)) == int(env_seed),
                 f"episode {index} env seed mismatch")
        _require(int(record.get("policy_seed", -1)) == int(policy_seed),
                 f"episode {index} policy seed mismatch")
        _require(str(record.get("scenario")) == "dense24",
                 f"episode {index} scenario mismatch")
        _require(int(record.get("requests", -1)) == EXPECTED_REQUESTS_PER_EPISODE,
                 f"episode {index} request count mismatch")
        _require(record.get("technical_ok") is True,
                 f"episode {index} technical flag is incomplete")
        _require(record.get("weights_unchanged") is True,
                 f"episode {index} weight flag is incomplete")
        _require(record.get("lifecycle_valid") is True,
                 f"episode {index} lifecycle flag is incomplete")
        _require(str(record.get("checkpoint_sha256", "")).lower() == WARMUP_CHECKPOINT_SHA256,
                 f"episode {index} warmup checkpoint SHA mismatch")
        declared_tape = Path(str(record.get("action_tape", "")))
        _require(str(record.get("action_tape", "")),
                 f"episode {index} action tape is missing")
        tape_path = _resolve(declared_tape)
        tape_payload, tape_rows = credit._load_tape(tape_path)
        _require(len(tape_rows) == int(expected_steps),
                 f"episode {index} tape step count mismatch")
        _require(len(tape_rows) <= MAX_REPLAY_STEPS,
                 f"episode {index} tape exceeds replay limit")
        rollout_summary = record.get("rollout_summary", {})
        rollout_steps = int(rollout_summary.get("steps", expected_steps)) \
            if isinstance(rollout_summary, Mapping) else int(expected_steps)
        _require(rollout_steps == int(expected_steps),
                 f"episode {index} record step count mismatch")
        cells.append({
            "index": int(index),
            "record_path": str(record_path),
            "tape_path": str(tape_path),
            "declared_tape_path": str(declared_tape),
            "record": record,
            "tape_payload": tape_payload,
            "tape_rows": tape_rows,
            "steps": int(expected_steps),
            "requests": EXPECTED_REQUESTS_PER_EPISODE,
            "env_seed": int(env_seed),
            "policy_seed": int(policy_seed),
        })
        total_steps += int(expected_steps)
    _require(len(cells) == EXPECTED_EPISODES, "exactly six episode inputs are required")
    _require(total_steps == EXPECTED_TOTAL_STEPS, "fixed total step count mismatch")
    return {
        "root": str(ROOT),
        "training_root": str(TRAINING_ROOT),
        "cells": cells,
        "episodes": len(cells),
        "total_steps": int(total_steps),
        "total_requests": len(cells) * EXPECTED_REQUESTS_PER_EPISODE,
        "max_replay_steps": MAX_TOTAL_REPLAY_STEPS,
    }


def _validate_after_summary() -> dict[str, Any]:
    summary_path = _resolve(AFTER_SUMMARY)
    summary = _read_json(summary_path)
    _require(str(summary.get("before_model_digest")) == SOURCE_WARMUP_MODEL_DIGEST,
             "after summary before_model_digest does not match source 3fe7+a82")
    roundtrip = summary.get("checkpoint_roundtrip")
    _require(isinstance(roundtrip, Mapping), "after summary checkpoint_roundtrip is missing")
    _require(str(roundtrip.get("checkpoint_sha256", "")).lower() == AFTER_CHECKPOINT_SHA256,
             "after summary checkpoint SHA does not match registered after checkpoint")
    _require(str(roundtrip.get("model_digest", "")) == AFTER_MODEL_DIGEST,
             "after summary model digest does not match registered after checkpoint")
    return {
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "before_model_digest": str(summary["before_model_digest"]),
        "after_model_digest": str(roundtrip["model_digest"]),
        "checkpoint_roundtrip": dict(roundtrip),
        "actual_after_checkpoint": str(_resolve(AFTER_CHECKPOINT)),
    }


def _build_preflight() -> dict[str, Any]:
    """Read fixed inputs and metadata without creating an environment."""
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    _require("dense24" in neutral_configs, "dense24 neutral config is missing")
    config = neutral_configs["dense24"]
    config_snapshot = _config_snapshot(config)
    preflight = _preflight_cells()
    _require(int(preflight["episodes"]) == EXPECTED_EPISODES, "source episode count changed")
    _require(int(preflight["total_steps"]) == EXPECTED_TOTAL_STEPS, "source total step count changed")
    _require(int(preflight["total_requests"]) == EXPECTED_TOTAL_REQUESTS, "source request count changed")
    cells = [_compact_cell(cell) for cell in preflight["cells"]]
    _require(tuple(item["steps"] for item in cells) == EXPECTED_STEPS, "source episode step contract changed")
    artifacts = {
        "source_checkpoint": _artifact(SOURCE_CHECKPOINT, expected_sha256=SOURCE_CHECKPOINT_SHA256),
        "warmup_checkpoint": _artifact(WARMUP_CHECKPOINT, expected_sha256=WARMUP_CHECKPOINT_SHA256),
        "after_checkpoint": _artifact(AFTER_CHECKPOINT, expected_sha256=AFTER_CHECKPOINT_SHA256),
        "after_summary": _artifact(AFTER_SUMMARY),
        "tapes": [
            {"episode": int(item["index"]), "path": item["tape_path"], "sha256": item["tape_sha256"]}
            for item in cells
        ],
    }
    after_binding = _validate_after_summary()
    return {
        "stage": STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": PPO_ALLOWED,
        "protocol": _protocol_metadata(config_snapshot),
        "runtime": {
            "versions": _runtime_versions(),
            "loaded_code_manifest": _loaded_code_manifest(),
            "config_snapshot": config_snapshot,
        },
        "artifacts": artifacts,
        "after_alignment_binding": after_binding,
        "inputs": {
            "root": str(ROOT),
            "training_root": str(TRAINING_ROOT),
            "episodes": len(cells),
            "total_steps": int(preflight["total_steps"]),
            "total_requests": int(preflight["total_requests"]),
            "cells": cells,
        },
        "selection": {
            "status": "PENDING_REPLAY",
            "eligible_occurrences": None,
            "selected": None,
            "mechanism_labels": "candidate observer fields are collected only during replay",
        },
        "scope": {
            "environment_created": False,
            "replay_calls": 0,
            "optimizer_calls": 0,
            "heldout_used": False,
            "teacher_used": False,
        },
    }


def _uid_value(mapping: Mapping[Any, Any] | None, uid: int, default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    if uid in mapping:
        return mapping[uid]
    if str(uid) in mapping:
        return mapping[str(uid)]
    return default


def _enum_or_value(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    value = getattr(value, "value", value)
    if isinstance(value, enum.Enum):
        return value.value
    return value


def _scalar_or_none(value: Any) -> Any:
    value = _enum_or_value(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value)
    return None


def _energy_diagnostic(env: Any, uid: int) -> dict[str, Any] | None:
    diagnostics = getattr(env, "energy_reachability_diagnostics", None)
    value = _uid_value(diagnostics, uid)
    if not isinstance(value, Mapping):
        return None
    keys = (
        "candidate_energy", "fallback_route_energy", "pad_wait_seconds",
        "required_energy", "slack_energy", "selected_lz_id",
    )
    return {key: _scalar_or_none(value.get(key)) for key in keys if key in value}


def _light_state_observer(env: Any) -> dict[str, Any]:
    active = getattr(env, "active", {})
    actors: dict[str, Any] = {}
    for uid, uav in sorted(active.items(), key=lambda item: int(item[0])):
        assigned = getattr(uav, "assigned_lz", None)
        assigned = getattr(assigned, "lz_id", assigned)
        actors[str(int(uid))] = {
            "uid": int(uid),
            "arrival_id": _scalar_or_none(getattr(uav, "arrival_id", None)),
            "phase": _scalar_or_none(getattr(uav, "phase", None)),
            "ring_role": _scalar_or_none(getattr(uav, "ring_role", None)),
            "soc": _scalar_or_none(getattr(uav, "soc", None)),
            "energy_remaining": _scalar_or_none(getattr(uav, "energy_remaining", None)),
            "energy_capacity": _scalar_or_none(getattr(uav, "energy_capacity", None)),
            "ring_progress": _scalar_or_none(getattr(uav, "ring_progress", None)),
            "emergency_retry_after": _scalar_or_none(getattr(uav, "emergency_retry_after", None)),
            "committed": bool(getattr(uav, "committed_plan", None) is not None),
            "assigned_lz": _scalar_or_none(assigned),
            "energy_reachability": _energy_diagnostic(env, int(uid)),
            "last_release_decision": _scalar_or_none(_uid_value(getattr(env, "last_release_decisions", None), int(uid))),
            "last_safety_decision": _scalar_or_none(_uid_value(getattr(env, "last_safety_decisions", None), int(uid))),
        }

    zones: list[dict[str, Any]] = []
    manager = getattr(env, "lz_manager", None)
    for zone in list(getattr(manager, "zones", ()) or ()):
        zones.append({
            "lz_id": _scalar_or_none(getattr(zone, "lz_id", None)),
            "status": _scalar_or_none(getattr(zone, "status", None)),
            "reserved_by": _scalar_or_none(getattr(zone, "reserved_by", None)),
            "occupied_by": _scalar_or_none(getattr(zone, "occupied_by", None)),
            "service_remaining": _scalar_or_none(getattr(zone, "service_remaining", None)),
            "expected_release_at": _scalar_or_none(getattr(zone, "expected_release_at", None)),
        })
    arrivals = getattr(env, "arrivals", None)
    exhausted = getattr(arrivals, "exhausted", None)
    if callable(exhausted):
        exhausted = exhausted()
    queue = getattr(getattr(getattr(env, "admission", None), "queue", None), "items", ())
    telemetry = getattr(env, "telemetry", {})
    return {
        "time": _scalar_or_none(getattr(env, "now", None)),
        "step": _scalar_or_none(getattr(env, "step_count", None)),
        "actors": actors,
        "queue_count": int(len(queue or ())),
        "service_count": int(len(getattr(env, "service_uavs", {}) or {})),
        "landing_zones": zones,
        "arrivals_generated": _scalar_or_none(getattr(arrivals, "generated", None)),
        "arrivals_exhausted": None if exhausted is None else bool(exhausted),
        "telemetry": {str(key): _scalar_or_none(value) for key, value in dict(telemetry or {}).items()},
    }


def _action_gate(row: Mapping[str, Any], uid: int) -> int | None:
    gate = _uid_value(row.get("requested_gate"), uid)
    if gate is not None:
        return int(gate)
    raw = _uid_value(row.get("a_requested"), uid)
    if raw is None:
        return None
    return int(hierarchical_from_normalized_v4_2(np.asarray(raw, dtype=np.float32)).gate)


def _action_proposal(row: Mapping[str, Any], uid: int) -> int | None:
    value = _uid_value(row.get("requested_proposal"), uid)
    if value is None:
        raw = _uid_value(row.get("a_requested"), uid)
        if raw is None:
            return None
        value = hierarchical_from_normalized_v4_2(np.asarray(raw, dtype=np.float32)).proposal
    return None if value is None or int(value) < 0 else int(value)


def _mechanism_labels(
    context: Mapping[str, Any],
    row: Mapping[str, Any],
    uid: int,
    wait_streak_before: int,
    config: Any,
) -> dict[str, Any]:
    actor = _uid_value(context.get("actors"), uid, {})
    actor = actor if isinstance(actor, Mapping) else {}
    soc = actor.get("soc")
    diag = actor.get("energy_reachability")
    diag = diag if isinstance(diag, Mapping) else {}
    slack = diag.get("slack_energy")
    zones = context.get("landing_zones")
    zones_available = isinstance(zones, Sequence)
    busy = None
    if zones_available:
        busy = any(
            str(zone.get("status")) in {"occupied", "reserved"}
            or (zone.get("service_remaining") is not None and float(zone.get("service_remaining")) > 0.0)
            for zone in zones if isinstance(zone, Mapping)
        )
    shortlist = _uid_value(row.get("runtime_shortlist_result"), uid)
    shortlist_available = isinstance(shortlist, Mapping)
    checked = None if not shortlist_available else int(shortlist.get("candidates_checked", 0))
    rank = None if not shortlist_available else shortlist.get("rank")
    return {
        "soc_or_slack": {
            "available": bool(soc is not None or slack is not None),
            "soc": None if soc is None else float(soc),
            "slack_energy": None if slack is None else float(slack),
            "soc_below_warning": None if soc is None else bool(float(soc) <= float(config.soc_warning)),
            "slack_nonpositive": None if slack is None else bool(float(slack) <= 0.0),
        },
        "wait_persistence": {
            "available": True,
            "prior_wait_streak": int(wait_streak_before),
            "current_action_wait": bool(_action_gate(row, uid) == WAIT_GATE),
            "persistent": bool(wait_streak_before >= 2),
        },
        "lz_busy": {
            "available": bool(zones_available),
            "busy": busy,
            "busy_zone_count": None if not zones_available else int(sum(
                str(zone.get("status")) in {"occupied", "reserved"}
                or (zone.get("service_remaining") is not None and float(zone.get("service_remaining")) > 0.0)
                for zone in zones if isinstance(zone, Mapping)
            )),
        },
        "k3_exhaustion": {
            "available": bool(shortlist_available),
            "candidates_checked": checked,
            "runtime_rank": rank,
            "exhausted": None if not shortlist_available else bool(checked >= 3 and rank is None),
        },
    }


def _candidate_rows(
    captures: Sequence[Mapping[str, Any]],
    tape_rows: Sequence[Mapping[str, Any]],
    *,
    episode_index: int,
    config: Any,
) -> list[dict[str, Any]]:
    wait_streaks: dict[int, int] = {}
    candidates: list[dict[str, Any]] = []
    for capture, row in zip(captures, tape_rows, strict=True):
        context = capture.get("state_context", {})
        context = context if isinstance(context, Mapping) else {}
        uids = [int(uid) for uid in capture.get("uids", ())]
        mask = np.asarray(capture.get("actor_mask", ()), dtype=bool)
        world = np.asarray(capture.get("world"), dtype=np.float32)
        for index, uid in enumerate(uids):
            if index >= mask.size or not bool(mask[index]):
                continue
            actor = _uid_value(context.get("actors"), uid, {})
            actor = actor if isinstance(actor, Mapping) else {}
            candidates.append({
                "episode": int(episode_index),
                "step": int(capture["step"]),
                "uid": int(uid),
                "arrival_id": _uid_value(capture.get("arrival_by_uid"), uid),
                "phase": _uid_value(capture.get("phase_by_uid"), uid),
                "world_index": int(index),
                "observation_hash": str(row.get("observation_hash", "")),
                "source_gate": _action_gate(row, uid),
                "source_proposal": _action_proposal(row, uid),
                "light_state": dict(actor),
                "mechanism_labels": _mechanism_labels(context, row, uid, wait_streaks.get(uid, 0), config),
            })
        for uid in uids:
            if _action_gate(row, uid) == WAIT_GATE:
                wait_streaks[uid] = wait_streaks.get(uid, 0) + 1
            else:
                wait_streaks[uid] = 0
        _require(world.ndim == 2 and world.shape[0] == len(uids), "captured world/UID rows are misaligned")
    return candidates


def _select_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        (dict(item) for item in candidates),
        key=lambda item: (int(item["episode"]), int(item["step"]), int(item["uid"])),
    )
    _require(len(ordered) >= SELECTION_SIZE,
             f"eligible actor occurrences are insufficient: {len(ordered)} < {SELECTION_SIZE}")
    keys = [(int(item["episode"]), int(item["step"]), int(item["uid"])) for item in ordered]
    _require(len(set(keys)) == len(keys), "eligible actor occurrences are not unique")
    rng = np.random.default_rng(SELECTION_SEED)
    selected_indices = sorted(int(index) for index in rng.choice(len(ordered), size=SELECTION_SIZE, replace=False))
    selected: list[dict[str, Any]] = []
    for order, index in enumerate(selected_indices, start=1):
        item = dict(ordered[index])
        item["selection_order"] = int(order)
        item["eligible_rank"] = int(index)
        selected.append(item)
    by_episode: dict[str, int] = {}
    for item in ordered:
        key = str(int(item["episode"]))
        by_episode[key] = by_episode.get(key, 0) + 1
    return {
        "status": "SELECTED",
        "seed": SELECTION_SEED,
        "sample_size": SELECTION_SIZE,
        "without_replacement": True,
        "sort_key": ["episode", "step", "uid"],
        "eligible_total": len(ordered),
        "eligible_by_episode": by_episode,
        "selected": selected,
    }


def _row_inputs(row: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    actions = {
        str(uid): np.asarray(value, dtype=np.float32).copy()
        for uid, value in (row.get("a_requested") or {}).items()
    }
    shortlists = {
        str(uid): tuple(np.asarray(value, dtype=np.float32).copy() for value in values)
        for uid, values in (row.get("runtime_proposal_shortlists") or {}).items()
    }
    return actions, shortlists


def _validate_replay(cell: Mapping[str, Any], replay: Mapping[str, Any]) -> None:
    exact = replay.get("replay", {}).get("exact")
    _require(exact is True, f"episode {cell['index']} action tape replay was not exact")
    _require(int(replay.get("replay", {}).get("steps_replayed", -1)) == int(cell["steps"]),
             f"episode {cell['index']} replay step count mismatch")
    lifecycle = replay.get("lifecycle", {})
    _require(lifecycle.get("terminal") is True, f"episode {cell['index']} replay did not terminate")
    _require(lifecycle.get("all_requests_observed") is True,
             f"episode {cell['index']} request lifecycle incomplete")
    _require(lifecycle.get("ledger_alignment") is True,
             f"episode {cell['index']} request ledger is not aligned")
    _require(len(lifecycle.get("request_rows", ())) == EXPECTED_REQUESTS_PER_EPISODE,
             f"episode {cell['index']} request ledger count mismatch")
    _require(len(replay.get("captures", ())) == int(cell["steps"]),
             f"episode {cell['index']} capture count mismatch")


def _load_source_runtime(config: Any) -> dict[str, Any]:
    resolved, policy, binding = pilot._load_binding(config)
    candidate_digest = pilot._model_digest(policy)
    warmup_path = _resolve(WARMUP_CHECKPOINT)
    warmup_payload = torch.load(warmup_path, map_location=policy.device, weights_only=False)
    _require(isinstance(warmup_payload, Mapping), "warmup checkpoint payload must be a mapping")
    fit_parameters = warmup_payload.get("fit_parameters")
    _require(isinstance(fit_parameters, Mapping), "warmup fit_parameters are missing")
    _require(math.isclose(float(fit_parameters.get("gamma")), GAMMA, rel_tol=0.0, abs_tol=1e-12),
             "warmup gamma does not match the registered value")
    _require(math.isclose(float(fit_parameters.get("reward_scale")), REWARD_SCALE,
                          rel_tol=0.0, abs_tol=1e-12),
             "warmup reward_scale does not match the registered value")
    warmup_resolved, warmup_check = pilot._load_initial_warmup(
        WARMUP_CHECKPOINT,
        policy,
        config,
    )
    source_digest = pilot._model_digest(policy)
    _require(source_digest == SOURCE_WARMUP_MODEL_DIGEST,
             f"source 3fe7+a82 model digest mismatch: {source_digest}")
    policy.model.eval()
    return {
        "policy": policy,
        "resolved_checkpoint": str(resolved),
        "candidate_digest_before_warmup": candidate_digest,
        "source_model_digest": source_digest,
        "binding": _jsonable(binding),
        "warmup": _jsonable(warmup_check),
        "warmup_checkpoint": str(warmup_resolved),
    }


def _public_source_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return serializable source binding fields without the live policy object."""
    return {
        key: value for key, value in source.items() if key != "policy"
    }


def _load_preflight(output: Path) -> dict[str, Any]:
    path = output / "preflight.json"
    if not path.is_file():
        payload = _build_preflight()
        _atomic_write_json(path, payload)
        return payload
    payload = _read_json(path)
    _require(payload.get("status") == "PREFLIGHT_OK", "preflight artifact is not successful")
    return payload


def _load_cells() -> dict[str, Any]:
    return _preflight_cells()


def _progress(
    output: Path,
    *,
    phase: str,
    status: str,
    processed: int = 0,
    expected: int | None = None,
    processed_steps: int = 0,
    expected_steps: int | None = None,
    eta_seconds: float | None = None,
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "stage": STAGE,
        "phase": phase,
        "status": status,
        "processed": int(processed),
        "expected": None if expected is None else int(expected),
        "processed_steps": int(processed_steps),
        "expected_steps": None if expected_steps is None else int(expected_steps),
        "eta_seconds": None if eta_seconds is None else float(eta_seconds),
        **extra,
    }
    _atomic_write_json(output / "progress.json", payload)


def _paired_branch_callback(
    output: Path,
    *,
    selection_order: int,
    total_branches: int,
    completed_states: int,
    expected_states: int,
    started: float,
    registered_selection_size: int = SELECTION_SIZE,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Callable[[Mapping[str, Any], int, int], None]:
    """Persist each completed raw branch and expose bounded live progress."""
    branch_dir = output / "branches" / f"state_{int(selection_order):04d}"

    def callback(report: Mapping[str, Any], completed: int, total: int) -> None:
        completed_index = int(completed)
        total_count = int(total)
        _require(total_count == int(total_branches),
                 f"paired branch total changed: {total_count} != {total_branches}")
        _require(1 <= completed_index <= total_count,
                 f"paired branch completion index is invalid: {completed_index}/{total_count}")
        _require(isinstance(report, Mapping), "paired branch callback report must be a mapping")
        artifact = branch_dir / f"branch_{completed_index:04d}.json"
        _atomic_write_json(artifact, report)
        classification = report.get("termination_classification")
        if classification is None:
            classification = report.get("classification")
        if classification is None:
            classification = "error" if report.get("error") else "unknown"
        branch_steps = report.get("steps")
        try:
            branch_steps_value = int(branch_steps)
        except (TypeError, ValueError):
            branch_steps_value = 0
        _progress(
            output,
            phase="paired_running",
            status="RUNNING",
            processed=int(completed_states),
            expected=int(expected_states),
            processed_steps=branch_steps_value,
            expected_steps=PILOT_MAX_STEPS,
            selection_order=int(selection_order),
            current_branch=completed_index,
            total_branches=total_count,
            branch_steps=branch_steps_value,
            last_classification=str(classification),
            elapsed_seconds=max(0.0, float(time.monotonic() - started)),
            branch_artifact=str(artifact),
            registered_selection_size=int(registered_selection_size),
            processed_selection_size=int(expected_states),
            shard_index=int(shard_index),
            shard_count=int(shard_count),
        )

    return callback


def _run_replay(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    _progress(output, phase="replay", status="STARTING", expected=EXPECTED_EPISODES,
              expected_steps=EXPECTED_TOTAL_STEPS)
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    config = neutral_configs["dense24"]
    _require(_config_snapshot(config) == preflight["runtime"]["config_snapshot"],
             "resolved config differs from preflight config")
    source = _load_source_runtime(config)
    inputs = _load_cells()
    candidates: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    processed_steps = 0
    for cell in inputs["cells"]:
        index = int(cell["index"])

        def replay_progress(
            step_count: int,
            *,
            _index: int = index,
            _processed_steps: int = processed_steps,
        ) -> None:
            _progress(
                output,
                phase="replay",
                status="REPLAYING",
                processed=max(0, _index - 1),
                expected=EXPECTED_EPISODES,
                processed_steps=_processed_steps + int(step_count),
                expected_steps=EXPECTED_TOTAL_STEPS,
            )

        replay = credit._capture_replay(
            source["policy"],
            config,
            Path(cell["tape_path"]),
            cell["tape_rows"],
            progress_callback=replay_progress,
            state_observer=_light_state_observer,
        )
        _validate_replay(cell, replay)
        candidates.extend(_candidate_rows(
            replay["captures"], cell["tape_rows"], episode_index=index, config=config,
        ))
        processed_steps += int(cell["steps"])
        _progress(output, phase="replay", status="REPLAYING", processed=index,
                  expected=EXPECTED_EPISODES, processed_steps=processed_steps,
                  expected_steps=EXPECTED_TOTAL_STEPS)
        episode_reports.append({
            "episode": index,
            "env_seed": int(cell["env_seed"]),
            "policy_seed": int(cell["policy_seed"]),
            "steps": int(cell["steps"]),
            "requests": EXPECTED_REQUESTS_PER_EPISODE,
            "replay_exact": True,
            "terminal": True,
            "ledger_alignment": True,
            "model_digest": source["source_model_digest"],
        })
    digest_after = pilot._model_digest(source["policy"])
    _require(digest_after == source["source_model_digest"], "source model changed during fixed replay")
    selection = _select_candidates(candidates)
    selection_payload = {
        "stage": STAGE,
        "phase": "replay",
        "status": "SELECTED",
        "protocol": preflight["protocol"],
        "selection": selection,
        "candidate_observer": {
            "state_context": "light pre-step fields only",
            "deepcopy_count": 0,
            "business_state_removed": False,
        },
    }
    _atomic_write_json(output / "selection.json", selection_payload)
    report = {
        "stage": STAGE,
        "phase": "replay",
        "status": "COMPLETED",
        "PPO_ALLOWED": PPO_ALLOWED,
        "source": {
            "binding": source["binding"],
            "warmup": source["warmup"],
            "model_digest_before_replay": source["source_model_digest"],
            "model_digest_after_replay": digest_after,
            "model_unchanged": True,
        },
        "episodes": episode_reports,
        "steps": processed_steps,
        "requests": EXPECTED_TOTAL_REQUESTS,
        "replay_exact": True,
        "all_requests_observed": True,
        "ledger_alignment": True,
        "selection": selection_payload,
        "scope": {
            "replay_calls": EXPECTED_EPISODES,
            "environment_resets": EXPECTED_EPISODES,
            "optimizer_calls": 0,
            "training_called": False,
            "teacher_used": False,
            "heldout_used": False,
        },
        "artifacts": {
            "preflight": str(output / "preflight.json"),
            "selection": str(output / "selection.json"),
            "report": str(output / "replay.json"),
            "progress": str(output / "progress.json"),
        },
    }
    _atomic_write_json(output / "replay.json", report)
    _atomic_write_json(output / "report.json", report)
    _atomic_write_json(output / "summary.json", {
        "stage": STAGE, "phase": "replay", "status": "COMPLETED",
        "steps": processed_steps, "requests": EXPECTED_TOTAL_REQUESTS,
        "replay_exact": True, "selection_size": SELECTION_SIZE,
    })
    _progress(output, phase="replay", status="COMPLETED", processed=EXPECTED_EPISODES,
              expected=EXPECTED_EPISODES, processed_steps=processed_steps,
              expected_steps=EXPECTED_TOTAL_STEPS)
    return report


def _load_selection(output: Path) -> dict[str, Any]:
    payload = _read_json(output / "selection.json")
    selection = payload.get("selection")
    _require(isinstance(selection, Mapping), "selection artifact is missing selection mapping")
    selected = selection.get("selected")
    _require(isinstance(selected, list) and len(selected) == SELECTION_SIZE,
             "selection artifact does not contain exactly 24 selected states")
    return dict(payload)


def _validate_shard_args(phase: str, shard_index: int, shard_count: int) -> tuple[int, int]:
    """Validate the bounded modulo sharding contract."""
    _require(not isinstance(shard_index, bool) and not isinstance(shard_count, bool),
             "shard index/count must be integers")
    try:
        index = int(shard_index)
        count = int(shard_count)
    except (TypeError, ValueError) as exc:
        raise FollowupDiagnosticError("shard index/count must be integers") from exc
    _require(count in range(1, MAX_SHARD_COUNT + 1),
             f"shard_count must be in 1..{MAX_SHARD_COUNT}")
    _require(0 <= index < count,
             "shard_index must satisfy 0 <= shard_index < shard_count")
    if str(phase) != "diagnose":
        _require(index == 0 and count == 1,
                 "non-default sharding is allowed only for diagnose")
    return index, count


def _shard_selection(
    selected: Sequence[Mapping[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    """Filter the registered selection without changing its order or contents."""
    index, count = _validate_shard_args("diagnose", shard_index, shard_count)
    result = [
        dict(item) for item in selected
        if (int(item["selection_order"]) - 1) % count == index
    ]
    _require(result, f"shard {index}/{count} has no registered selected states")
    return result


def _capture_selected_states(
    output: Path,
    preflight: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[int, int, int], dict[str, Any]]]:
    """Exact replay with deep copies only at selected pre-step positions."""
    _progress(output, phase="selected_replay", status="STARTING", expected=EXPECTED_EPISODES,
              expected_steps=EXPECTED_TOTAL_STEPS, selected_states=len(selected), deepcopy_count=0)
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    config = neutral_configs["dense24"]
    _require(_config_snapshot(config) == preflight["runtime"]["config_snapshot"],
             "resolved config differs from preflight config")
    source = _load_source_runtime(config)
    inputs = _load_cells()
    by_episode: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for item in selected:
        by_episode.setdefault(int(item["episode"]), {}).setdefault(int(item["step"]), []).append(dict(item))
    states: dict[tuple[int, int, int], dict[str, Any]] = {}
    processed_steps = 0
    for cell in inputs["cells"]:
        episode = int(cell["index"])
        targets = by_episode.get(episode, {})

        def replay_progress(
            step_count: int,
            *,
            _episode: int = episode,
            _processed_steps: int = processed_steps,
        ) -> None:
            _progress(
                output,
                phase="selected_replay",
                status="REPLAYING",
                processed=max(0, _episode - 1),
                expected=EXPECTED_EPISODES,
                processed_steps=_processed_steps + int(step_count),
                expected_steps=EXPECTED_TOTAL_STEPS,
                selected_states=len(selected),
                deepcopy_count=len(states),
            )

        def observe(env: Any, *, _episode: int = episode, _targets: Mapping[int, Sequence[Mapping[str, Any]]] = targets) -> dict[str, Any]:
            step = int(getattr(env, "step_count", -1))
            if step in _targets:
                snapshot = copy.deepcopy(env)
                restore_snapshot = getattr(
                    paired_gate_diagnostic, "restore_diagnostic_snapshot", None
                )
                _require(callable(restore_snapshot),
                         "paired diagnostic snapshot restoration helper is unavailable")
                restore_snapshot(snapshot)
                bound_step = getattr(snapshot, "step", None)
                _require(getattr(bound_step, "__self__", None) is snapshot,
                         "deep-copied environment retained observed_step closure")
                for method_name in (
                    "_resolve", "_release", "_try_emergency_landing",
                    "_candidate_energy_guard_reason",
                ):
                    method = getattr(snapshot, method_name, None)
                    _require(getattr(method, "__self__", None) is snapshot,
                             f"deep-copied environment method is not bound: {method_name}")
                requests_until = getattr(getattr(snapshot, "arrivals", None), "requests_until", None)
                _require(
                    getattr(requests_until, "__self__", None) is getattr(snapshot, "arrivals", None),
                    "deep-copied arrivals.requests_until retained source closure",
                )
                safety = getattr(snapshot, "safety", None)
                filter_plan = getattr(safety, "filter_plan", None)
                _require(
                    getattr(filter_plan, "__self__", None) is safety,
                    "deep-copied safety.filter_plan retained source closure",
                )
                for candidate in _targets[step]:
                    key = (_episode, step, int(candidate["uid"]))
                    states[key] = {
                        "snapshot": snapshot,
                        "shadow_methods_restored": True,
                    }
            return _light_state_observer(env)

        replay = credit._capture_replay(
            source["policy"], config, Path(cell["tape_path"]), cell["tape_rows"],
            progress_callback=replay_progress,
            state_observer=observe,
        )
        _validate_replay(cell, replay)
        for capture, row in zip(replay["captures"], cell["tape_rows"], strict=True):
            step = int(capture["step"])
            for candidate in targets.get(step, ()):
                uid = int(candidate["uid"])
                key = (episode, step, uid)
                _require(key in states, f"selected state was not deep-copied: {key}")
                actions, shortlists = _row_inputs(row)
                uid_index = next((i for i, value in enumerate(capture["uids"]) if int(value) == uid), None)
                _require(uid_index is not None, f"selected UID is absent from capture: {key}")
                _require(bool(np.asarray(capture["actor_mask"], dtype=bool)[int(uid_index)]),
                         f"selected UID is not actor-eligible at capture: {key}")
                snapshot = states[key]["snapshot"]
                _require(uid in getattr(snapshot, "active", {}),
                         f"selected UID is absent from deep-copied environment: {key}")
                world = np.asarray(capture["world"], dtype=np.float32).copy()
                target_world = world[int(uid_index):int(uid_index) + 1].copy()
                states[key].update({
                    "candidate": dict(candidate),
                    "episode": episode,
                    "step": step,
                    "uid": uid,
                    "world": world,
                    "target_world": target_world,
                    "target_world_index": int(uid_index),
                    "first_actions": actions,
                    "first_shortlists": shortlists,
                    "source_observation_hash": str(row.get("observation_hash", "")),
                    "source_action_tape_index": row.get("action_tape_index"),
                })
        processed_steps += int(cell["steps"])
        _progress(output, phase="selected_replay", status="REPLAYING", processed=episode,
                  expected=EXPECTED_EPISODES, processed_steps=processed_steps,
                  expected_steps=EXPECTED_TOTAL_STEPS, selected_states=len(selected),
                  deepcopy_count=len(states))
    _require(len(states) == len(selected), f"selected state count mismatch: {len(states)} != {len(selected)}")
    digest_after = pilot._model_digest(source["policy"])
    _require(digest_after == source["source_model_digest"], "source model changed during selected replay")
    return source, states


def _alignment_report(
    config: Any,
    source_policy: Any,
    target_world: np.ndarray,
    *,
    target_uid: int,
    delta_q: float | None = None,
) -> dict[str, Any]:
    resolved = _resolve(AFTER_CHECKPOINT)
    _require(_sha256(resolved).lower() == AFTER_CHECKPOINT_SHA256, "after checkpoint SHA mismatch")
    after_policy, payload = rank_eval._load_policy(resolved, config)
    before_digest = pilot._model_digest(source_policy)
    after_digest = pilot._model_digest(after_policy)
    _require(before_digest == SOURCE_WARMUP_MODEL_DIGEST, "alignment source model digest mismatch")
    _require(after_digest == AFTER_MODEL_DIGEST, "alignment after model digest mismatch")
    alignment = paired_gate_diagnostic.gate_update_alignment(
        source_policy,
        after_policy,
        np.asarray(target_world, dtype=np.float32),
        delta_q=delta_q,
    )
    return {
        "protocol": "source_3fe7_plus_a82_to_actual_episodic_batch01_after",
        "source_model_digest": before_digest,
        "after_model_digest": after_digest,
        "after_checkpoint": _artifact(AFTER_CHECKPOINT, expected_sha256=AFTER_CHECKPOINT_SHA256),
        "after_payload_keys": sorted(payload) if isinstance(payload, Mapping) else [],
        "alignment_scope": {
            "target_uid": int(target_uid),
            "target_world_only": True,
            "target_world_shape": list(np.asarray(target_world).shape),
            "delta_q": None if delta_q is None else float(delta_q),
        },
        "alignment": alignment,
        "d449_endpoint_used": False,
    }


def _state_report(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selection_order": int(state["candidate"]["selection_order"]),
        "episode": int(state["episode"]),
        "step": int(state["step"]),
        "uid": int(state["uid"]),
        "candidate": state["candidate"],
        "source_observation_hash": state["source_observation_hash"],
        "source_action_tape_index": state["source_action_tape_index"],
        "target_world": {
            "shape": list(np.asarray(state["target_world"]).shape),
            "row_index": int(state["target_world_index"]),
        },
        "deepcopy": {
            "selected_pre_step_only": True,
            "shadow_methods_restored": bool(state.get("shadow_methods_restored")),
            "business_state_removed": False,
        },
    }


def _run_pilot(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    selection_payload = _load_selection(output)
    selected = list(selection_payload["selection"]["selected"])
    source, states = _capture_selected_states(output, preflight, selected[:1])
    state = states[(int(selected[0]["episode"]), int(selected[0]["step"]), int(selected[0]["uid"]))]
    before_digest = pilot._model_digest(source["policy"])
    paired_started = time.monotonic()
    branch_callback = _paired_branch_callback(
        output,
        selection_order=int(state["candidate"]["selection_order"]),
        total_branches=26,
        completed_states=0,
        expected_states=1,
        started=paired_started,
    )
    paired = paired_gate_diagnostic.paired_gate_value(
        source["policy"], state["snapshot"], state["first_actions"],
        state["first_shortlists"], state["uid"], CONTINUATION_SEEDS[:1], PILOT_MAX_STEPS,
        branch_callback=branch_callback,
    )
    after_digest = pilot._model_digest(source["policy"])
    _require(before_digest == after_digest, "source model changed during pilot paired diagnostic")
    _require(len(paired.get("branches", ())) == 26, "pilot did not return WAIT plus 25 RELEASE branches")
    config = neutral_eval._build_neutral_inputs()[1]["dense24"]
    alignment = _alignment_report(
        config,
        source["policy"],
        state["target_world"],
        target_uid=state["uid"],
        delta_q=paired.get("qualified_delta_Q"),
    )
    report = {
        "stage": STAGE,
        "phase": "pilot",
        "status": "COMPLETED",
        "PPO_ALLOWED": PPO_ALLOWED,
        "source": _public_source_binding(source),
        "state": _state_report(state),
        "paired_gate": paired,
        "alignment": alignment,
        "scope": {
            "source_continuation": True,
            "continuation_seeds": list(CONTINUATION_SEEDS[:1]),
            "max_steps": PILOT_MAX_STEPS,
            "branch_count": 26,
            "optimizer_calls": 0,
            "heldout_used": False,
            "teacher_used": False,
            "performance_claim": False,
        },
        "artifacts": {
            "preflight": str(output / "preflight.json"),
            "selection": str(output / "selection.json"),
            "report": str(output / "pilot.json"),
            "progress": str(output / "progress.json"),
        },
    }
    _atomic_write_json(output / "pilot.json", report)
    _atomic_write_json(output / "report.json", report)
    _atomic_write_json(output / "summary.json", {
        "stage": STAGE, "phase": "pilot", "status": "COMPLETED",
        "branch_count": 26, "qualified_delta_Q": paired.get("qualified_delta_Q"),
    })
    _progress(output, phase="pilot", status="COMPLETED", processed=1, expected=1,
              processed_steps=0, expected_steps=0, branch_count=26)
    return report


def _run_diagnose(
    output: Path,
    preflight: Mapping[str, Any],
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    selection_payload = _load_selection(output)
    registered_selected = list(selection_payload["selection"]["selected"])
    registered_selection_size = len(registered_selected)
    _require(registered_selection_size == SELECTION_SIZE,
             "registered selection size changed before diagnose")
    shard_index, shard_count = _validate_shard_args("diagnose", shard_index, shard_count)
    selected = _shard_selection(
        registered_selected, shard_index=shard_index, shard_count=shard_count,
    )
    global_selection_orders = [int(item["selection_order"]) for item in selected]
    source, states = _capture_selected_states(output, preflight, selected)
    config = neutral_eval._build_neutral_inputs()[1]["dense24"]
    state_dir = output / "diagnose"
    state_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source_digest_before = pilot._model_digest(source["policy"])
    state_paths: list[str] = []
    first_alignment: dict[str, Any] | None = None
    for processed, candidate in enumerate(selected, start=1):
        key = (int(candidate["episode"]), int(candidate["step"]), int(candidate["uid"]))
        state = states[key]
        paired_started = time.monotonic()
        branch_callback = _paired_branch_callback(
            output,
            selection_order=int(state["candidate"]["selection_order"]),
            total_branches=len(CONTINUATION_SEEDS) * 26,
            completed_states=processed - 1,
            expected_states=len(selected),
            started=paired_started,
            registered_selection_size=registered_selection_size,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        paired = paired_gate_diagnostic.paired_gate_value(
            source["policy"], state["snapshot"], state["first_actions"],
            state["first_shortlists"], state["uid"], CONTINUATION_SEEDS, PILOT_MAX_STEPS,
            branch_callback=branch_callback,
        )
        _require(len(paired.get("branches", ())) == len(CONTINUATION_SEEDS) * 26,
                 f"selected state {key} branch count mismatch")
        alignment = _alignment_report(
            config,
            source["policy"],
            state["target_world"],
            target_uid=state["uid"],
            delta_q=paired.get("qualified_delta_Q"),
        )
        if first_alignment is None:
            first_alignment = alignment
        state_report = {
            "stage": STAGE,
            "phase": "diagnose",
            "status": "COMPLETED",
            "state": _state_report(state),
            "paired_gate": paired,
            "alignment": alignment,
            "scope": {
                "continuation_seeds": list(CONTINUATION_SEEDS),
                "max_steps": PILOT_MAX_STEPS,
                "branch_count": len(CONTINUATION_SEEDS) * 26,
                "other_uids_first_step_fixed": True,
                "performance_claim": False,
            },
        }
        global_selection_order = int(candidate["selection_order"])
        path = state_dir / f"state_{global_selection_order:04d}.json"
        _atomic_write_json(path, state_report)
        state_paths.append(str(path))
        elapsed = time.monotonic() - started
        eta = (elapsed / processed) * (len(selected) - processed) if processed else None
        _progress(output, phase="diagnose", status="RUNNING", processed=processed,
                  expected=len(selected), processed_steps=0, expected_steps=0,
                  eta_seconds=eta, branch_count=processed * len(CONTINUATION_SEEDS) * 26,
                  registered_selection_size=registered_selection_size,
                  processed_selection_size=len(selected),
                  shard_index=shard_index, shard_count=shard_count,
                  global_selection_orders=global_selection_orders)
    source_digest_after = pilot._model_digest(source["policy"])
    _require(source_digest_after == source_digest_before,
             "source model changed during diagnose paired diagnostics")
    _require(first_alignment is not None, "diagnose produced no per-state alignment")
    report = {
        "stage": STAGE,
        "phase": "diagnose",
        "status": "COMPLETED",
        "PPO_ALLOWED": PPO_ALLOWED,
        "source": _public_source_binding(source),
        "selection_size": len(selected),
        "registered_selection_size": registered_selection_size,
        "processed_selection_size": len(selected),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "global_selection_orders": global_selection_orders,
        "state_reports": state_paths,
        "alignment": first_alignment,
        "scope": {
            "selected_states": len(selected),
            "registered_selection_size": registered_selection_size,
            "processed_selection_size": len(selected),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "global_selection_orders": global_selection_orders,
            "continuation_seeds": list(CONTINUATION_SEEDS),
            "paired_state_seed_groups": len(selected) * len(CONTINUATION_SEEDS),
            "branches": len(selected) * len(CONTINUATION_SEEDS) * 26,
            "optimizer_calls": 0,
            "heldout_used": False,
            "teacher_used": False,
            "performance_claim": False,
            "source_model_unchanged": True,
        },
        "artifacts": {
            "preflight": str(output / "preflight.json"),
            "selection": str(output / "selection.json"),
            "state_directory": str(state_dir),
            "report": str(output / "diagnose.json"),
            "progress": str(output / "progress.json"),
        },
    }
    _atomic_write_json(output / "diagnose.json", report)
    _atomic_write_json(output / "report.json", report)
    _atomic_write_json(output / "summary.json", {
        "stage": STAGE, "phase": "diagnose", "status": "COMPLETED",
        "selected_states": len(selected),
        "registered_selection_size": registered_selection_size,
        "processed_selection_size": len(selected),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "global_selection_orders": global_selection_orders,
        "branches": report["scope"]["branches"],
    })
    _progress(output, phase="diagnose", status="COMPLETED", processed=len(selected),
              expected=len(selected), processed_steps=0, expected_steps=0,
              eta_seconds=0.0, branch_count=report["scope"]["branches"],
              registered_selection_size=registered_selection_size,
              processed_selection_size=len(selected),
              shard_index=shard_index, shard_count=shard_count,
              global_selection_orders=global_selection_orders)
    return report


def _write_failure(output: Path, phase: str, error: BaseException) -> dict[str, Any]:
    report = {
        "stage": STAGE,
        "phase": phase,
        "status": "FAILED",
        "PPO_ALLOWED": PPO_ALLOWED,
        "error": {"type": type(error).__name__, "message": str(error)},
        "scope": {
            "optimizer_calls": 0,
            "training_called": False,
            "heldout_used": False,
            "teacher_used": False,
        },
        "artifacts": {
            "report": str(output / "report.json"),
            "progress": str(output / "progress.json"),
            "summary": str(output / "summary.json"),
        },
    }
    _atomic_write_json(output / "report.json", report)
    _atomic_write_json(output / "summary.json", report)
    _progress(output, phase=phase, status="FAILED", error=report["error"])
    return report


def run(
    output_dir: str | Path,
    phase: str,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phase = str(phase)
    _require(phase in {"preflight", "replay", "pilot", "diagnose"}, f"unsupported phase: {phase}")
    shard_index, shard_count = _validate_shard_args(phase, shard_index, shard_count)
    try:
        if phase == "preflight":
            report = _build_preflight()
            _atomic_write_json(output / "preflight.json", report)
            _atomic_write_json(output / "report.json", report)
            _atomic_write_json(output / "summary.json", {
                "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
                "replay_calls": 0, "environment_created": False,
            })
            _progress(output, phase="preflight", status="PREFLIGHT_OK", processed=0,
                      expected=EXPECTED_EPISODES, processed_steps=0, expected_steps=EXPECTED_TOTAL_STEPS)
            return report
        preflight = _load_preflight(output)
        if phase == "replay":
            return _run_replay(output, preflight)
        if phase == "pilot":
            return _run_pilot(output, preflight)
        return _run_diagnose(
            output,
            preflight,
            shard_index=shard_index,
            shard_count=shard_count,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        return _write_failure(output, phase, exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "replay", "pilot", "diagnose"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        args.output_dir,
        args.phase,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(json.dumps(_jsonable({
        "stage": STAGE,
        "phase": args.phase,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "status": result.get("status"),
        "output_dir": str(args.output_dir),
    }), sort_keys=True), flush=True)
    return 0 if result.get("status") in {"PREFLIGHT_OK", "COMPLETED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
