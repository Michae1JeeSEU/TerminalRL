"""Collect full actor-state anchors from four immutable BEFORE-prefix tapes.

This is an off-policy, replay-only preparation artifact.  It records the
native 88D actor observations before every source tape step, including WAIT,
RELEASE, masked, and empty-world steps.  It does not load a policy, perform a
forward pass, sample an action, or change a replay action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.training import diagnose_p3_matched_gate_shift_20261001 as matched
from terminal_marl.training import diagnose_p3_prefix_gate_events_20261001 as base
from terminal_marl.training import diagnose_p3_prefix_feasible_gate_20261001 as feasible
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as support


STAGE = "P3_GATE_ANCHOR_STATES_20261001_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
NO_TRAINING = True
OFF_POLICY = True
OBSERVATION_DIM = 88
PROPOSAL_COUNT = 25
PUBLIC_K = 3
MAX_STEPS = 1800
REQUESTS_PER_CELL = 24
CELLS = (1, 2, 3, 4)
EXPECTED_STEPS = {1: 708, 2: 674, 3: 644, 4: 723}
TOTAL_STEPS = sum(EXPECTED_STEPS.values())
SOURCE_FULL_SHA256 = "30c28c03d26466eabeab2a35c9fc117424648f1a3ae17be6f8d69b6b92cb2195"
EVENT_FULL_SHA256 = "fb89f62339755f7ebcd2bb0f9672438000fe8b754067cd2cdb28bf8407feb3e6"
BEFORE_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
AFTER_SHA256 = "d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f"
PROTOCOL_SHA256 = "15057283b5adabee1e37079956eff0858e786739a4df07665676a01bda200060"
OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/gate_anchor_states_20261001"
)
PROTOCOL = Path(__file__).resolve().parents[2] / "docs/evidence/v4_2/p3_gate_anchor_states_protocol_20261001.md"


class AnchorCollectionError(ValueError):
    """Raised when the immutable replay or anchor serialization contract fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnchorCollectionError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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
        raise AnchorCollectionError(f"invalid JSON input: {target}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise AnchorCollectionError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    if target.is_file():
        return target
    resolved = base._resolve(target)
    if resolved.is_file():
        return resolved
    return target


def _rng_snapshot() -> Any:
    return base._rng_snapshot()


def _restore_rng(value: Any) -> None:
    base._restore_rng(value)


def _rng_equal(left: Any, right: Any) -> bool:
    return base._rng_equal(left, right)


def _mapping_value(mapping: Mapping[Any, Any] | None, key: int, default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def _binary_code(value: Any) -> tuple[int, bool | None]:
    """Encode a source boolean without turning a missing value into false."""
    if isinstance(value, bool):
        return (1 if value else 0), value
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        result = bool(int(value))
        return int(value), result
    return -1, None


def _proposal_code(value: Any) -> int:
    if value is None:
        return -2
    try:
        number = int(value)
    except (TypeError, ValueError):
        return -2
    return number if number == -1 or 0 <= number < PROPOSAL_COUNT else -2


def _safe_context(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _jsonable(value)
        for key, value in context.items()
        if key not in {"world", "full_world", "index"}
    }


def _cell_binding(item: Mapping[str, Any]) -> dict[str, Any]:
    cell_id = int(item["cell_id"])
    return {
        "cell_id": cell_id,
        "arm": "before_prefix",
        "full_sha256": str(item["full_sha256"]),
        "arm_sha256": str(item["arm_sha256"]),
        "tape_sha256": str(item["tape_sha256"]),
        "sidecar_sha256": str(item["sidecar_sha256"]),
        "expected_steps": int(EXPECTED_STEPS[cell_id]),
    }


def _load_inputs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for cell_id in CELLS:
        item = base._load_cell_inputs(int(cell_id))
        _require(str(item.get("full_sha256")) == SOURCE_FULL_SHA256,
                 f"cell {cell_id} source full binding changed")
        rows = item.get("rows")
        _require(isinstance(rows, list), f"cell {cell_id} tape rows missing")
        _require(len(rows) == EXPECTED_STEPS[cell_id],
                 f"cell {cell_id} expected {EXPECTED_STEPS[cell_id]} steps, got {len(rows)}")
        _require(len({int(row.get("step", -1)) for row in rows}) == len(rows),
                 f"cell {cell_id} tape step keys are not unique")
        items.append(dict(item))
    return items


def _config_snapshot(config: Any) -> dict[str, Any]:
    return base._config_snapshot(config)


def _source_bindings(config: Any) -> dict[str, Any]:
    bindings = base._source_bindings(config)
    _require(bindings.get("before", {}).get("sha256") == BEFORE_SHA256,
             "BEFORE source binding changed")
    _require(bindings.get("after", {}).get("sha256") == AFTER_SHA256,
             "AFTER source binding changed")
    return _jsonable(bindings)


def _event_full_binding() -> dict[str, Any]:
    event_full = _resolve(base.OUTPUT_DIR / "full.json")
    _require(event_full.is_file(), f"event full artifact missing: {event_full}")
    actual = _sha256(event_full)
    _require(actual == EVENT_FULL_SHA256, f"event full artifact SHA mismatch: {actual}")
    return {"path": str(event_full), "sha256": actual}


def _build_preflight() -> dict[str, Any]:
    caller = _rng_snapshot()
    try:
        config, _ = support._build_inputs()
        _require(PROTOCOL.is_file() and _sha256(PROTOCOL) == PROTOCOL_SHA256,
                 f"protocol binding changed: {PROTOCOL}")
        event_full = _event_full_binding()
        items = _load_inputs()
        source = _source_bindings(config)
    finally:
        _restore_rng(caller)
    return {
        "stage": STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "environment_created": False,
        "model_forward_used": False,
        "gradient_operations": 0,
        "optimizer_operations": 0,
        "teacher_calls": 0,
        "PPO_ALLOWED": PPO_ALLOWED,
        "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED,
        "optimizer_used": OPTIMIZER_USED,
        "off_policy": OFF_POLICY,
        "no_training": NO_TRAINING,
        "schema": {
            "observation_schema": "phaseA_low_obs_v2_1_retry_state",
            "observation_dimension": OBSERVATION_DIM,
            "world_dtype": "float32",
            "world_row_order": "matched._world_from_observations returned UID string order",
            "public_shortlist_k": PUBLIC_K,
            "proposal_count": PROPOSAL_COUNT,
        },
        "input_bindings": {
            "source_full_sha256": SOURCE_FULL_SHA256,
            "event_full": event_full,
            "cells": [_cell_binding(item) for item in items],
            "expected_steps_total": TOTAL_STEPS,
        },
        "source_bindings": source,
        "config": _config_snapshot(config),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": _sha256(Path(__file__)),
        "selection": {
            "all_actor_rows": True,
            "requested_gate_filter": False,
            "actor_mask_filter": False,
            "outcome_filter": False,
            "empty_world_ticks_retained": True,
            "deduplicated_worlds": False,
        },
        "safety_provenance": {
            "source_certification": "preserved only from immutable arm metadata when present",
            "missing_fields_are_null": True,
        },
        "qualification": {
            "status": "preparation_only",
            "performance_sample": False,
            "PPO_ALLOWED": False,
            "heldout_evaluation": False,
        },
    }


def _validate_preflight(payload: Mapping[str, Any]) -> None:
    _require(payload.get("stage") == STAGE and payload.get("status") == "PREFLIGHT_OK",
             "anchor preflight is not successful")
    expected_flags = {
        "environment_created": False,
        "model_forward_used": False,
        "gradient_operations": 0,
        "optimizer_operations": 0,
        "teacher_calls": 0,
        "PPO_ALLOWED": False,
        "heldout_used": False,
        "teacher_used": False,
        "optimizer_used": False,
        "off_policy": True,
        "no_training": True,
    }
    for key, expected in expected_flags.items():
        _require(payload.get(key) == expected, f"preflight flag {key} changed: {key}")
    _require(payload.get("runner_sha256") == _sha256(Path(__file__)),
             "runner changed after preflight")
    _require(payload.get("protocol_sha256") == PROTOCOL_SHA256 and _sha256(PROTOCOL) == PROTOCOL_SHA256,
             "protocol changed after preflight")
    config, _ = support._build_inputs()
    _require(payload.get("config") == _config_snapshot(config),
             "config changed after preflight")
    _require(payload.get("source_bindings") == _source_bindings(config),
             "source binding changed after preflight")
    event_full = _event_full_binding()
    inputs = payload.get("input_bindings")
    _require(isinstance(inputs, Mapping), "input bindings missing")
    _require(inputs.get("source_full_sha256") == SOURCE_FULL_SHA256,
             "source full binding changed")
    stored_event = inputs.get("event_full")
    _require(isinstance(stored_event, Mapping) and stored_event.get("sha256") == event_full["sha256"] == EVENT_FULL_SHA256,
             "event full binding changed")
    current = _load_inputs()
    _require(inputs.get("cells") == [_cell_binding(item) for item in current],
             "cell/tape/sidecar input binding changed")
    _require(int(inputs.get("expected_steps_total", -1)) == TOTAL_STEPS,
             "expected total replay steps changed")


def _save_npz(path: Path, *, world: np.ndarray, steps: np.ndarray, uids: np.ndarray,
              arrivals: np.ndarray, actor_mask: np.ndarray, requested_gate: np.ndarray,
              requested_proposal: np.ndarray, metadata: Mapping[str, Any]) -> str:
    _require(world.dtype == np.float32 and world.ndim == 2 and world.shape[1] == OBSERVATION_DIM,
             "anchor world must be a finite float32 rows x 88 array")
    _require(bool(np.isfinite(world).all()), "anchor world contains non-finite values")
    count = int(world.shape[0])
    for array in (steps, uids, arrivals, actor_mask, requested_gate, requested_proposal):
        _require(len(array) == count, "anchor arrays have different lengths")
    _require(not path.exists(), f"refusing to overwrite existing anchor NPZ: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            world=world,
            step=steps,
            uid=uids,
            arrival_id=arrivals,
            row_index=np.arange(count, dtype=np.int64),
            actor_mask=actor_mask,
            requested_gate=requested_gate,
            requested_proposal=requested_proposal,
            metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
        )
    os.replace(temporary, path)
    return _sha256(path)


def _source_safety_provenance(item: Mapping[str, Any]) -> dict[str, Any]:
    arm = item.get("arm")
    if not isinstance(arm, Mapping):
        return {"safety_nonzero": None, "binding_checks": None, "lifecycle_valid": None}
    return {
        "safety_nonzero": _jsonable(arm.get("safety_nonzero")),
        "binding_checks": _jsonable(arm.get("binding_checks")),
        "lifecycle_valid": _jsonable(arm.get("lifecycle_valid")),
        "source_status": _jsonable(arm.get("status")),
    }


def _collect_cell(item: Mapping[str, Any], config: Any, output: Path) -> dict[str, Any]:
    cell_id = int(item["cell_id"])
    rows = list(item["rows"])
    row_by_step = {int(row["step"]): row for row in rows}
    caller_before = _rng_snapshot()
    env: Any = None
    original_step: Any = None
    controller_restored = False
    row_metadata: list[dict[str, Any]] = []
    step_metadata: list[dict[str, Any]] = []
    worlds: list[np.ndarray] = []
    steps: list[int] = []
    uids: list[int] = []
    arrivals: list[int] = []
    actor_mask_codes: list[int] = []
    gate_codes: list[int] = []
    proposal_codes: list[int] = []
    seen_keys: set[str] = set()
    counters = {
        "requested_gate": {"true": 0, "false": 0, "unknown": 0},
        "actor_mask": {"true": 0, "false": 0, "unknown": 0},
    }
    current_step: int | None = None
    replay: Mapping[str, Any] | None = None
    rng_before_restore: Any = None
    constructor_rng_consumed = False
    result: dict[str, Any] | None = None

    def _progress(step: int, active_count: int) -> None:
        if step == 0 or (step + 1) % 50 == 0:
            _write_json(output / "progress.json", {
                "stage": STAGE,
                "phase": "arm",
                "status": "RUNNING",
                "cell_id": cell_id,
                "step": step,
                "steps_expected": len(rows),
                "rows_recorded": len(row_metadata),
                "active_actor_count": int(active_count),
                "empty_world_ticks": sum(1 for record in step_metadata if not record["uids"]),
            })

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal current_step
        current_step = int(getattr(env, "step_count", -1))
        source_row = row_by_step.get(current_step)
        _require(source_row is not None, f"cell {cell_id} replay step {current_step} missing from tape")
        observations = env._observations()
        _require(isinstance(observations, Mapping), "environment observations must be a mapping")
        source_hash = source_row.get("observation_hash")
        _require(isinstance(source_hash, str) and source_hash,
                 f"cell {cell_id} source observation hash missing at step {current_step}")
        observed_hash = _digest({key: value for key, value in sorted(observations.items(), key=lambda pair: str(pair[0]))})
        _require(observed_hash == source_hash,
                 f"cell {cell_id} observation hash mismatch at step {current_step}")
        ordered_uids, world = matched._world_from_observations(observations)
        world = np.asarray(world, dtype=np.float32)
        _require(world.ndim == 2 and world.shape[1] == OBSERVATION_DIM,
                 f"cell {cell_id} world shape changed at step {current_step}: {world.shape}")
        _require(bool(np.isfinite(world).all()), f"cell {cell_id} world contains non-finite values")
        start_index = len(row_metadata)
        for row_index, uid_text in enumerate(ordered_uids):
            uid = int(uid_text)
            key = f"{cell_id}|{current_step}|{uid}"
            _require(key not in seen_keys, f"duplicate anchor key: {key}")
            seen_keys.add(key)
            context = feasible._actor_context(env, uid, ordered_uids, world)
            gate_raw = _mapping_value(source_row.get("requested_gate"), uid, None)
            proposal_raw = _mapping_value(source_row.get("requested_proposal"), uid, None)
            gate_code, gate_value = _binary_code(gate_raw)
            mask_code, mask_value = _binary_code(context.get("actor_mask"))
            gate_label = "true" if gate_value is True else "false" if gate_value is False else "unknown"
            mask_label = "true" if mask_value is True else "false" if mask_value is False else "unknown"
            counters["requested_gate"][gate_label] += 1
            counters["actor_mask"][mask_label] += 1
            row_metadata.append({
                "row_index": len(row_metadata),
                "cell_id": cell_id,
                "step": current_step,
                "uid": uid,
                "arrival_id": _jsonable(context.get("arrival_id")),
                "observation_hash": source_hash,
                "world_npz_field": "world",
                "world_row_index": len(row_metadata),
                "world_shape": [OBSERVATION_DIM],
                "world_dtype": "float32",
                "actor_mask": _jsonable(context.get("actor_mask")),
                "requested_gate": _jsonable(gate_raw),
                "requested_proposal": _jsonable(proposal_raw),
                "context": _safe_context(context),
            })
            worlds.append(world[row_index].copy())
            steps.append(current_step)
            uids.append(uid)
            arrivals.append(int(context["arrival_id"]))
            actor_mask_codes.append(mask_code)
            gate_codes.append(gate_code)
            proposal_codes.append(_proposal_code(proposal_raw))
        step_metadata.append({
            "step": current_step,
            "observation_hash": source_hash,
            "uids": [int(value) for value in ordered_uids],
            "row_start": start_index,
            "row_end": len(row_metadata),
            "world_shape": [int(value) for value in world.shape],
        })
        _progress(current_step, len(ordered_uids))
        return original_step(*args, **kwargs)

    try:
        env = frozen._CapturedTerminalEnv(config)
        constructor_after = _rng_snapshot()
        constructor_rng_consumed = not _rng_equal(caller_before, constructor_after)
        _restore_rng(caller_before)
        original_step = env.step
        env.step = observed_step  # type: ignore[method-assign]
        with feasible._controller_scope(env, item["sidecar"]):
            replay = SafetyTraceRecorder.replay_action_tape(env, item["tape_path"])
        controller_restored = True
        _require(isinstance(replay, Mapping), "replay result is not a mapping")
        _require(replay.get("exact") is True, f"cell {cell_id} action replay is not exact")
        _require(int(replay.get("steps_replayed", -1)) == len(rows),
                 f"cell {cell_id} replay step count changed")
        for check_name in (
            "config_match", "reset_recipe_match", "reset_state_match",
            "base_reset_state_match", "action_match", "reward_match", "outcomes_match",
        ):
            if check_name in replay:
                _require(replay.get(check_name) is True,
                         f"cell {cell_id} replay check failed: {check_name}")
        rng_before_restore = _rng_snapshot()
        caller_unchanged = _rng_equal(caller_before, rng_before_restore)
        _require(caller_unchanged, f"cell {cell_id} caller RNG changed before restoration")
        _require(len(step_metadata) == len(rows),
                 f"cell {cell_id} observed {len(step_metadata)} steps, expected {len(rows)}")
        metadata = {
            "stage": STAGE,
            "cell_id": cell_id,
            "observation_dimension": OBSERVATION_DIM,
            "world_dtype": "float32",
            "world_row_order": "base.matched._world_from_observations returned UID string order",
            "steps_expected": len(rows),
            "steps_replayed": int(replay.get("steps_replayed", -1)),
            "rows": len(row_metadata),
            "step_metadata": step_metadata,
        }
        npz_path = output / "anchors" / f"cell_{cell_id:02d}.npz"
        world_array = np.asarray(worlds, dtype=np.float32).reshape((-1, OBSERVATION_DIM))
        npz_sha = _save_npz(
            npz_path,
            world=world_array,
            steps=np.asarray(steps, dtype=np.int64),
            uids=np.asarray(uids, dtype=np.int64),
            arrivals=np.asarray(arrivals, dtype=np.int64),
            actor_mask=np.asarray(actor_mask_codes, dtype=np.int8),
            requested_gate=np.asarray(gate_codes, dtype=np.int8),
            requested_proposal=np.asarray(proposal_codes, dtype=np.int8),
            metadata=metadata,
        )
        replay_summary = {str(key): _jsonable(value) for key, value in replay.items() if key not in {"steps", "action_tape"}}
        result = {
            "stage": STAGE,
            "phase": "arm",
            "status": "COMPLETED",
            "cell_id": cell_id,
            "arm": "before_prefix",
            "input_binding": _cell_binding(item),
            "source_provenance": _source_safety_provenance(item),
            "replay": replay_summary,
            "replay_exact": True,
            "steps_expected": len(rows),
            "steps_replayed": len(step_metadata),
            "rows": row_metadata,
            "step_metadata": step_metadata,
            "empty_world_ticks": [row for row in step_metadata if not row["uids"]],
            "counts": counters,
            "anchors_npz": {
                "path": str(npz_path),
                "sha256": npz_sha,
                "world_field": "world",
                "row_index_field": "row_index",
                "step_field": "step",
                "uid_field": "uid",
                "shape": [int(world_array.shape[0]), OBSERVATION_DIM],
                "dtype": "float32",
            },
            "controller_installed_and_restored": controller_restored,
            "constructor_rng_consumed": constructor_rng_consumed,
            "caller_rng_unchanged_before_restore": caller_unchanged,
            "caller_rng_unchanged_after_restore": None,
            "model_loaded": False,
            "model_forward_used": False,
        }
    except BaseException as exc:
        partial = {
            "stage": STAGE,
            "phase": "arm",
            "status": "PARTIAL",
            "cell_id": cell_id,
            "rows_recorded": len(row_metadata),
            "steps_recorded": len(step_metadata),
            "current_step": current_step,
            "counts": counters,
            "error": {"error_type": type(exc).__name__, "message": str(exc)},
        }
        try:
            _write_json(output / f"cell_{cell_id:02d}_partial.json", partial, refuse_existing=True)
        except BaseException:
            pass
        raise
    finally:
        if env is not None and original_step is not None:
            env.step = original_step  # type: ignore[method-assign]
        _restore_rng(caller_before)
        caller_after_restore = _rng_snapshot()
        restored = _rng_equal(caller_before, caller_after_restore)
        if result is not None:
            result["caller_rng_unchanged_after_restore"] = restored
        # Restoration is checked for both successful and failed replays.  The
        # pre-restore snapshot is only a success-reporting field; a failed
        # replay must still prove that its caller state was restored.
        _require(restored, f"cell {cell_id} caller RNG was not restored")
    _require(result is not None, f"cell {cell_id} did not produce a result")
    return result


def _validate_pilot_reuse(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    pilot_path = output / "pilot.json"
    pilot = _read_json(pilot_path)
    _require(pilot.get("status") == "PILOT_OK" and pilot.get("cell_ids") == [1],
             "pilot is not accepted")
    preflight_path = output / "preflight.json"
    _require(pilot.get("preflight_sha256") == _sha256(preflight_path),
             "pilot preflight binding changed")
    cell_path = output / "cells" / "cell_01.json"
    _require(cell_path.is_file(), "pilot cell artifact missing")
    _require(pilot.get("cell_sha256") == _sha256(cell_path), "pilot cell binding changed")
    cell = _read_json(cell_path)
    _require(cell.get("status") == "COMPLETED" and cell.get("replay_exact") is True,
             "pilot cell is not complete")
    npz_binding = cell.get("anchors_npz")
    _require(isinstance(npz_binding, Mapping), "pilot anchor NPZ binding is missing")
    npz_path = Path(str(npz_binding.get("path", "")))
    npz_path = _resolve(npz_path)
    _require(npz_path.is_file(), "pilot anchor NPZ is missing")
    _require(_sha256(npz_path) == npz_binding.get("sha256"), "pilot anchor NPZ binding changed")
    _require(preflight.get("status") == "PREFLIGHT_OK", "preflight is not accepted")
    return pilot


def _guard_phase_artifacts(output: Path, phase: str, cell_ids: Sequence[int]) -> None:
    for name in ("error.json", "partial.json"):
        _require(not (output / name).exists(), f"refusing phase after existing {name}")
    for cell_id in cell_ids:
        cell_path = output / "cells" / f"cell_{int(cell_id):02d}.json"
        npz_path = output / "anchors" / f"cell_{int(cell_id):02d}.npz"
        _require(not cell_path.exists(), f"refusing phase after existing cell artifact: {cell_path}")
        _require(not npz_path.exists(), f"refusing phase after existing anchor artifact: {npz_path}")


def _cell_summary(output: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    cell_id = int(result["cell_id"])
    cell_path = output / "cells" / f"cell_{cell_id:02d}.json"
    npz_binding = result.get("anchors_npz")
    _require(isinstance(npz_binding, Mapping), "cell anchor NPZ binding is missing")
    return {
        "cell_id": cell_id,
        "status": result["status"],
        "rows": len(result.get("rows", [])),
        "steps": int(result["steps_replayed"]),
        "cell_json_path": str(cell_path),
        "cell_json_sha256": _sha256(cell_path),
        "anchors_npz_path": str(npz_binding["path"]),
        "anchors_npz_sha256": str(npz_binding["sha256"]),
    }


def _run_phase(phase: str, output: Path) -> dict[str, Any]:
    if phase == "preflight":
        _require(not output.exists(), f"refusing to overwrite existing output: {output}")
        output.mkdir(parents=True, exist_ok=False)
        payload = _build_preflight()
        _write_json(output / "preflight.json", payload, refuse_existing=True)
        _write_json(output / "progress.json", {
            "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
            "environment_created": False, "cells": list(CELLS),
        }, refuse_existing=True)
        return payload
    _require(output.is_dir(), f"output directory missing: {output}")
    preflight_path = output / "preflight.json"
    _require(preflight_path.is_file(), "preflight.json is required")
    preflight = _read_json(preflight_path)
    _validate_preflight(preflight)
    config, _ = support._build_inputs()
    items = _load_inputs()
    cells_dir = output / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    if phase == "pilot":
        _require(not (output / "pilot.json").exists(), "pilot already exists")
        _require(not (output / "full.json").exists(), "full already exists")
        _guard_phase_artifacts(output, phase, [1])
        result = _collect_cell(items[0], config, output)
        cell_path = cells_dir / "cell_01.json"
        _write_json(cell_path, result, refuse_existing=True)
        summary = {
            "stage": STAGE, "phase": "pilot", "status": "PILOT_OK", "cell_ids": [1],
            "cells": [_cell_summary(output, result)],
            "preflight_sha256": _sha256(preflight_path),
            "cell_sha256": _sha256(cell_path),
            "environment_created": True,
            "performance_sample": False,
        }
        _write_json(output / "pilot.json", summary, refuse_existing=True)
        return summary
    _require(phase == "full", f"unknown phase: {phase}")
    pilot = _validate_pilot_reuse(output, preflight)
    results = [_read_json(cells_dir / "cell_01.json")]
    _guard_phase_artifacts(output, phase, CELLS[1:])
    for item, cell_id in zip(items[1:], CELLS[1:]):
        _require(not (cells_dir / f"cell_{cell_id:02d}.json").exists(),
                 f"cell {cell_id} already exists")
        result = _collect_cell(item, config, output)
        _write_json(cells_dir / f"cell_{cell_id:02d}.json", result, refuse_existing=True)
        results.append(result)
    summary = {
        "stage": STAGE, "phase": "full", "status": "FULL_OK", "cell_ids": list(CELLS),
        "cells": [
            _cell_summary(output, results[0]),
            *[_cell_summary(output, row) for row in results[1:]],
        ],
        "preflight_sha256": _sha256(preflight_path),
        "pilot_sha256": _sha256(output / "pilot.json"),
        "source_full_sha256": SOURCE_FULL_SHA256,
        "event_full_sha256": EVENT_FULL_SHA256,
        "performance_sample": False,
    }
    _write_json(output / "full.json", summary, refuse_existing=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir)
    try:
        result = _run_phase(args.phase, output)
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        error = {
            "stage": STAGE,
            "phase": args.phase,
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            _write_json(output / "error.json", error, refuse_existing=True)
            _write_json(output / "partial.json", {"stage": STAGE, "phase": args.phase,
                                                    "status": "PARTIAL", "error": error},
                        refuse_existing=True)
        except BaseException:
            pass
        print(json.dumps(_jsonable(error), sort_keys=True))
        return 2
    print(json.dumps(_jsonable(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
