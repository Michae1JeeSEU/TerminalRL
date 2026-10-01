"""Frozen full-cycle gate-value diagnostic on four normal-inner tape states.

The runner is an off-policy diagnostic.  It replays one existing BEFORE-prefix
tape, then evaluates WAIT and all 25 RELEASE proposals from one captured
pre-state under the existing prefix controller.  It does not train, sample a
new episode, or change the native K=3 public interface.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from terminal_marl.core.low_level_joint_action_v4_2 import (
    HierarchicalActionV42,
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.training import diagnose_p3_matched_gate_shift_20261001 as matched
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_followup_diagnostics as followup
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_prefix_fallback_20261001 as prefix_base
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as support
from terminal_marl.training import diagnose_p3_trained_gate_value_20261001 as trained_gate


STAGE = "P3_PREFIX_NORMAL_GATE_VALUE_20261001_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
OFF_POLICY = True
NO_TRAINING = True
SCENARIO = "dense24"
OBSERVATION_DIM = 88
PROPOSAL_COUNT = 25
PUBLIC_K = 3
MAX_STEPS = 1800
CONTINUATION_SEEDS = (9303401, 9303402, 9303403, 9303404)
CELLS = (1, 2, 3, 4)
REQUESTS_PER_CELL = 24
BRANCHES_PER_SEED = 1 + PROPOSAL_COUNT
BRANCHES_PER_STATE = BRANCHES_PER_SEED * len(CONTINUATION_SEEDS)
TOTAL_BRANCHES = BRANCHES_PER_STATE * len(CELLS)
PILOT_BRANCHES = BRANCHES_PER_STATE
FULL_NEW_BRANCHES = BRANCHES_PER_STATE * (len(CELLS) - 1)
EVENT_FULL_SHA256 = "fb89f62339755f7ebcd2bb0f9672438000fe8b754067cd2cdb28bf8407feb3e6"
FRESH_FULL_SHA256 = "30c28c03d26466eabeab2a35c9fc117424648f1a3ae17be6f8d69b6b92cb2195"
BEFORE_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
AFTER_SHA256 = "d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f"
BEFORE_MODEL_DIGEST = "3434378719efdb02bf3d41febb52e66797a0d3928756fc0cbbfde056ddd337b0"
AFTER_MODEL_DIGEST = "5261fc7d0da7deb15e6433f743162e79e8fc13b86205bf90e5ed8525e44530a5"
PROTOCOL_SHA256 = "086f6e5cc42941cf2e564d7cfcbb44c057d3cdeba0e2936340d6ece80dcd6be6"
EVENT_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/prefix_gate_events_20261001"
)
FRESH_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/prefix_fallback_fresh_validation_20261001_v4"
)
OUTPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/prefix_normal_gate_value_20261001"
)
PROTOCOL = Path(__file__).resolve().parents[2] / (
    "docs/evidence/v4_2/p3_prefix_normal_gate_value_protocol_20261001.md"
)

STATE_LOCKS = {
    1: {"key": "307|15", "step": 307, "uid": 15, "arrival_id": 15,
        "soc": 0.7262173226131179},
    2: {"key": "321|25", "step": 321, "uid": 25, "arrival_id": 15,
        "soc": 0.6731115455491892},
    3: {"key": "303|20", "step": 303, "uid": 20, "arrival_id": 15,
        "soc": 0.6404768921709051},
    4: {"key": "365|46", "step": 365, "uid": 46, "arrival_id": 15,
        "soc": 0.47479269296572224},
}


class NormalGateValueError(ValueError):
    """Raised when a frozen normal-inner diagnostic contract changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NormalGateValueError(message)


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
    value = json.loads(target.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise NormalGateValueError(f"refusing existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    try:
        synced = support._sync_path(target)
    except AttributeError:
        synced = target
    if synced.is_file():
        return synced
    text = str(target).replace("\\", "/")
    prefix = "/home/user/training_outputs/terminal/"
    if text.startswith(prefix):
        mirror = Path(__file__).resolve().parents[3] / "training_outputs" / "terminal" / text[len(prefix):]
        if mirror.is_file():
            return mirror
    return synced


def _rng_snapshot() -> Any:
    return support._rng_snapshot()


def _restore_rng(value: Any) -> None:
    support._restore_rng(value)


def _rng_equal(left: Any, right: Any) -> bool:
    return bool(support._rng_equal(left, right))


def _config_snapshot(config: Any) -> dict[str, Any]:
    return support._config_snapshot(config)


def _cell_arm_path(cell_id: int) -> Path:
    return _resolve(FRESH_ROOT / "arms" / f"cell_{int(cell_id):02d}_before_prefix.json")


def _event_group_path(cell_id: int) -> Path:
    return _resolve(EVENT_ROOT / "cells" / f"cell_{int(cell_id):02d}.json")


def _select_event(group: Mapping[str, Any], cell_id: int) -> dict[str, Any]:
    replay = group.get("replay")
    events = replay.get("events") if isinstance(replay, Mapping) else None
    _require(isinstance(events, Sequence), f"cell {cell_id} event list missing")
    candidates: list[dict[str, Any]] = []
    for raw in events:
        if not isinstance(raw, Mapping):
            continue
        context = raw.get("context")
        if not isinstance(context, Mapping):
            continue
        if (raw.get("status") == "NORMAL_ELIGIBLE"
                and context.get("phase") == "ring_inner"
                and int(context.get("arrival_id", -1)) >= 13):
            candidates.append(dict(raw))
    _require(candidates, f"cell {cell_id} has no normal-inner arrival>=13 event")
    candidates.sort(key=lambda row: (int(row["step"]), int(row["uid"])))
    selected = candidates[0]
    lock = STATE_LOCKS[int(cell_id)]
    context = selected.get("context", {})
    _require(str(selected.get("registration_key")) == lock["key"], f"cell {cell_id} selection key changed")
    _require((int(selected["step"]), int(selected["uid"])) == (lock["step"], lock["uid"]),
             f"cell {cell_id} selection step/UID changed")
    _require(int(context.get("arrival_id", -1)) == lock["arrival_id"], f"cell {cell_id} arrival changed")
    _require(math.isclose(float(context.get("soc")), lock["soc"], rel_tol=0.0, abs_tol=1e-12),
             f"cell {cell_id} SOC changed")
    comparison = selected.get("comparison")
    _require(isinstance(comparison, Mapping), f"cell {cell_id} selected comparison missing")
    _require(comparison.get("D") is None and comparison.get("local_m") is None,
             f"cell {cell_id} source event unexpectedly contains value estimate")
    return selected


def _sidecar_from_arm(arm: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    control = arm.get("prefix_fallback_control")
    _require(isinstance(control, Mapping), "prefix fallback control metadata missing")
    raw_path = control.get("orders_sidecar")
    sidecar_path = _resolve(str(raw_path))
    sidecar = _read_json(sidecar_path)
    declared = control.get("orders_sidecar_sha256")
    _require(isinstance(declared, str) and _sha256(sidecar_path) == declared,
             "orders sidecar SHA mismatch")
    registrations = sidecar.get("registrations")
    _require(isinstance(registrations, Mapping), "orders sidecar registrations missing")
    return sidecar_path, sidecar


def _load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_full_path = _resolve(EVENT_ROOT / "full.json")
    fresh_full_path = _resolve(FRESH_ROOT / "full.json")
    _require(_sha256(event_full_path) == EVENT_FULL_SHA256, "event full SHA changed")
    _require(_sha256(fresh_full_path) == FRESH_FULL_SHA256, "fresh full SHA changed")
    event_full = _read_json(event_full_path)
    fresh_full = _read_json(fresh_full_path)
    _require(event_full.get("status") == "FULL_OK", "event full is not complete")
    _require(fresh_full.get("status") == "FULL_OK", "fresh full is not complete")
    cells: list[dict[str, Any]] = []
    for cell_id in CELLS:
        group_path = _event_group_path(cell_id)
        arm_path = _cell_arm_path(cell_id)
        group = _read_json(group_path)
        arm = _read_json(arm_path)
        _require(group.get("status") == "COMPLETED", f"event group {cell_id} incomplete")
        _require(arm.get("status") == "completed", f"fresh arm {cell_id} incomplete")
        _require(arm.get("arm") == "before_prefix", f"cell {cell_id} arm changed")
        _require(int(arm.get("cell", {}).get("cell_id", -1)) == cell_id, f"cell {cell_id} arm binding changed")
        _require(str(arm.get("checkpoint_sha256")) == BEFORE_SHA256, f"cell {cell_id} checkpoint changed")
        tape_path = _resolve(str(arm.get("action_tape")))
        sidecar_path, sidecar = _sidecar_from_arm(arm)
        _require(tape_path.is_file(), f"cell {cell_id} action tape missing")
        selected = _select_event(group, cell_id)
        cells.append({
            "cell_id": cell_id,
            "group_path": group_path,
            "group_sha256": _sha256(group_path),
            "group": group,
            "selected_event": selected,
            "arm_path": arm_path,
            "arm_sha256": _sha256(arm_path),
            "arm": arm,
            "tape_path": tape_path,
            "tape_sha256": _sha256(tape_path),
            "sidecar_path": sidecar_path,
            "sidecar_sha256": _sha256(sidecar_path),
            "sidecar": sidecar,
        })
    return {
        "event_full_path": event_full_path,
        "event_full_sha256": EVENT_FULL_SHA256,
        "fresh_full_path": fresh_full_path,
        "fresh_full_sha256": FRESH_FULL_SHA256,
        "event_full": event_full,
        "fresh_full": fresh_full,
    }, cells


def _source_bindings() -> dict[str, Any]:
    before = _resolve(support.BEFORE_CHECKPOINT)
    after = _resolve(support.AFTER_CHECKPOINT)
    _require(_sha256(before) == BEFORE_SHA256, "BEFORE checkpoint SHA changed")
    _require(_sha256(after) == AFTER_SHA256, "AFTER checkpoint SHA changed")
    return {
        "before": {"path": str(support.BEFORE_CHECKPOINT), "sha256": BEFORE_SHA256},
        "after": {"path": str(support.AFTER_CHECKPOINT), "sha256": AFTER_SHA256},
    }


def _cell_binding(item: Mapping[str, Any]) -> dict[str, Any]:
    selection = item["selected_event"]
    context = selection["context"]
    return {
        "cell_id": int(item["cell_id"]),
        "event_group_sha256": str(item["group_sha256"]),
        "arm_sha256": str(item["arm_sha256"]),
        "tape_sha256": str(item["tape_sha256"]),
        "sidecar_sha256": str(item["sidecar_sha256"]),
        "selected_key": str(selection["registration_key"]),
        "step": int(selection["step"]),
        "uid": int(selection["uid"]),
        "arrival_id": int(context["arrival_id"]),
        "soc": float(context["soc"]),
    }


def _build_preflight() -> dict[str, Any]:
    caller = _rng_snapshot()
    try:
        _require(PROTOCOL.is_file() and _sha256(PROTOCOL) == PROTOCOL_SHA256,
                 "normal gate-value protocol SHA changed")
        config, _ = support._build_inputs()
        source = _source_bindings()
        inputs, cells = _load_inputs()
        config_snapshot = _config_snapshot(config)
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
        "PPO_ALLOWED": False,
        "heldout_used": False,
        "teacher_used": False,
        "optimizer_used": False,
        "off_policy": True,
        "no_training": True,
        "event_full_sha256": inputs["event_full_sha256"],
        "fresh_full_sha256": inputs["fresh_full_sha256"],
        "source_bindings": source,
        "config_snapshot": config_snapshot,
        "cells": [_cell_binding(item) for item in cells],
        "continuation_seeds": list(CONTINUATION_SEEDS),
        "max_steps": MAX_STEPS,
        "branches_per_seed": BRANCHES_PER_SEED,
        "branches_per_state": BRANCHES_PER_STATE,
        "total_branches": TOTAL_BRANCHES,
        "pilot_branches": PILOT_BRANCHES,
        "full_new_branches": FULL_NEW_BRANCHES,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": _sha256(Path(__file__)),
        "output_schema": "normal_gate_value_v1",
    }


def _validate_preflight(payload: Mapping[str, Any]) -> None:
    _require(payload.get("stage") == STAGE, "preflight stage changed")
    _require(payload.get("phase") == "preflight", "preflight phase changed")
    _require(payload.get("status") == "PREFLIGHT_OK", "preflight status changed")
    expected_flags = {
        "environment_created": False, "model_forward_used": False,
        "gradient_operations": 0, "optimizer_operations": 0, "teacher_calls": 0,
        "PPO_ALLOWED": False, "heldout_used": False, "teacher_used": False,
        "optimizer_used": False, "off_policy": True, "no_training": True,
    }
    for key, expected in expected_flags.items():
        _require(payload.get(key) == expected, f"preflight flag changed: {key}")
    _require(payload.get("protocol_sha256") == PROTOCOL_SHA256, "preflight protocol changed")
    _require(payload.get("runner_sha256") == _sha256(Path(__file__)), "preflight runner changed")
    _require(payload.get("event_full_sha256") == EVENT_FULL_SHA256, "preflight event input changed")
    _require(payload.get("fresh_full_sha256") == FRESH_FULL_SHA256, "preflight fresh input changed")
    _require(payload.get("continuation_seeds") == list(CONTINUATION_SEEDS),
             "preflight continuation seeds changed")
    _require(payload.get("max_steps") == MAX_STEPS, "preflight max steps changed")
    for key, expected in {
        "branches_per_seed": BRANCHES_PER_SEED,
        "branches_per_state": BRANCHES_PER_STATE,
        "total_branches": TOTAL_BRANCHES,
        "pilot_branches": PILOT_BRANCHES,
        "full_new_branches": FULL_NEW_BRANCHES,
    }.items():
        _require(payload.get(key) == expected, f"preflight branch budget changed: {key}")
    caller = _rng_snapshot()
    try:
        config, _ = support._build_inputs()
        inputs, cells = _load_inputs()
        _require(payload.get("config_snapshot") == _config_snapshot(config), "preflight config changed")
        _require(payload.get("source_bindings") == _source_bindings(), "preflight sources changed")
        _require(payload.get("cells") == [_cell_binding(item) for item in cells], "preflight cells changed")
        _require(inputs["event_full_sha256"] == EVENT_FULL_SHA256, "event full input changed")
    finally:
        _restore_rng(caller)


def _load_policy_restoring_rng(path: Path, config: Any) -> tuple[Any, bool]:
    caller = _rng_snapshot()
    consumed = False
    try:
        policy, _payload = rank_eval._load_policy(path, config)
        after = _rng_snapshot()
        consumed = not _rng_equal(caller, after)
        return policy, consumed
    finally:
        _restore_rng(caller)


def _sidecar_registration(item: Mapping[str, Any], env: Any, uid: int) -> Mapping[str, Any]:
    registrations = item["sidecar"].get("registrations", {})
    step = int(getattr(env, "step_count", -1))
    direct = registrations.get(f"{step}|{int(uid)}")
    if isinstance(direct, Mapping):
        return direct
    now = float(getattr(env, "now", float("nan")))
    options = [
        row for row in registrations.values()
        if isinstance(row, Mapping) and int(row.get("uid", -1)) == int(uid)
        and math.isfinite(now) and math.isclose(float(row.get("env_now")), now, rel_tol=0.0, abs_tol=1e-9)
    ]
    _require(len(options) == 1, f"missing unique sidecar registration step={step} uid={uid}")
    return options[0]


def _metadata_for_order(sampled: int, logits: Sequence[float]) -> dict[str, Any]:
    values = [float(item) for item in logits]
    _require(len(values) == PROPOSAL_COUNT and all(math.isfinite(item) for item in values),
             "proposal logits are not finite 25-way values")
    return {"sampled_proposal": int(sampled), "source_logits": values,
            "requested_k": PUBLIC_K, "source_forward_id": "normal_gate_value"}


def _stable_order(sampled: int, logits: Sequence[float]) -> list[int]:
    ranked = sorted(range(PROPOSAL_COUNT), key=lambda index: (-float(logits[index]), index))
    return [int(sampled)] + [int(index) for index in ranked if int(index) != int(sampled)]


def _controller_scope(env: Any, orders: Mapping[int, Sequence[int]], metadata: Mapping[int, Mapping[str, Any]]):
    def order_provider(_env: Any, _uav: Any, uid: int):
        _require(int(uid) in orders, f"prefix order missing for uid={uid}")
        return orders[int(uid)]

    def metadata_provider(_env: Any, _uav: Any, uid: int):
        _require(int(uid) in metadata, f"prefix metadata missing for uid={uid}")
        return metadata[int(uid)]

    return prefix_base.prefix_fallback_controller(
        env, order_provider, source_metadata=metadata_provider,
    )


def _rebind_native_snapshot_methods(snapshot: Any) -> None:
    """Drop deepcopy-retained step/controller closures before branch use."""
    for name in ("step", "_runtime_shortlist_release"):
        native_method = getattr(type(snapshot), name, None)
        if callable(native_method):
            setattr(snapshot, name, native_method.__get__(snapshot, type(snapshot)))
        else:
            snapshot.__dict__.pop(name, None)


def _details_context_once(policy: Any, env: Any, *, details_fn: Any = None) -> dict[str, Any]:
    """Run canonical details construction with one shared model forward."""
    model = paired._policy_model(policy)
    original = getattr(model, "distributions", None)
    _require(callable(original), "policy model lacks distributions")
    model_dict = getattr(model, "__dict__", None)
    _require(isinstance(model_dict, dict), "policy model lacks scoped alias storage")
    had_alias = "distributions" in model_dict
    previous_alias = model_dict.get("distributions")
    cached_input: torch.Tensor | None = None
    cached_result: Any = None

    def cached_distributions(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal cached_input, cached_result
        value = tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
        if cached_input is not None:
            _require(torch.equal(cached_input, value), "details context requested a different distribution tensor")
            return cached_result
        result = original(tensor, *args, **kwargs)
        if cached_input is None:
            cached_input = value.clone()
            cached_result = result
        return result

    model_dict["distributions"] = cached_distributions
    try:
        fn = paired._details_context if details_fn is None else details_fn
        return fn(policy, env)
    finally:
        if had_alias:
            model_dict["distributions"] = previous_alias
        else:
            model_dict.pop("distributions", None)


def _action_proposal(action: Any) -> int | None:
    values = np.asarray(action, dtype=np.float32)
    decoded = hierarchical_from_normalized_v4_2(values)
    return None if decoded.proposal is None else int(decoded.proposal)


def _target_index(details: Mapping[str, Any], target_uid: int) -> int:
    for index, uid in enumerate(details.get("uids", ())):
        if str(uid) == str(target_uid):
            return int(index)
    raise NormalGateValueError(f"target UID {target_uid} absent from details")


def _target_context(
    env: Any,
    target_uid: int,
    uids: Sequence[str],
    world: np.ndarray,
    *,
    actor_mask: bool,
) -> dict[str, Any]:
    actor = getattr(env, "active", {}).get(int(target_uid))
    if actor is None:
        actor = getattr(env, "active", {}).get(str(target_uid))
    _require(actor is not None, f"target UID {target_uid} is not active")
    index = [str(uid) for uid in uids].index(str(target_uid))
    phase = getattr(getattr(actor, "phase", None), "value", getattr(actor, "phase", None))
    now = float(getattr(env, "now", 0.0))
    retry_after = float(getattr(actor, "safety_retry_after", 0.0))
    return {
        "uid": int(target_uid), "arrival_id": int(getattr(actor, "arrival_id")),
        "phase": str(phase), "soc": float(getattr(actor, "soc")),
        "safety_retry_after": retry_after, "retry_free": bool(now >= retry_after),
        "committed_plan": bool(getattr(actor, "committed_plan", None) is not None),
        "now": now, "actor_mask": bool(actor_mask),
        "world": np.asarray(world[index], dtype=np.float32).copy(), "index": int(index),
    }


def _capture_state(item: Mapping[str, Any], config: Any, before_policy: Any) -> dict[str, Any]:
    selected = item["selected_event"]
    step = int(selected["step"])
    tape_payload = _read_json(item["tape_path"])
    tape_rows = list(tape_payload.get("action_tape", ()))
    _require(step < len(tape_rows), f"selected step outside tape: {step}")
    row_by_step = {int(row["step"]): row for row in tape_rows}
    _require(step in row_by_step, f"selected tape row missing: {step}")
    caller_before = _rng_snapshot()
    env = frozen._CapturedTerminalEnv(config)
    constructor_after = _rng_snapshot()
    constructor_consumed = not _rng_equal(caller_before, constructor_after)
    _restore_rng(caller_before)
    original_step = env.step
    captured: dict[str, Any] = {}
    replay_result: Mapping[str, Any] | None = None
    replay_error: BaseException | None = None

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        current = int(getattr(env, "step_count", -1))
        source_row = row_by_step.get(current)
        _require(source_row is not None, f"tape row missing at step {current}")
        observations = env._observations()
        _require(_digest({key: value for key, value in sorted(observations.items())})
                 == source_row.get("observation_hash"), f"observation hash mismatch at {current}")
        if current == step:
            uids, world = matched._world_from_observations(observations)
            _require(str(source_row.get("observation_hash")) == str(selected.get("pre_observation_hash")),
                     "selected event/tape observation hash differs")
            expected_matrix = np.asarray(selected.get("pre_world_matrix_88d"), dtype=np.float32)
            _require(expected_matrix.shape == world.shape and np.array_equal(expected_matrix, world),
                     "selected event/tape world differs")
            details = _details_context_once(before_policy, env)
            _require([str(uid) for uid in details["uids"]] == [str(uid) for uid in uids],
                     "selected UID order differs")
            _require(np.array_equal(np.asarray(details["world"], dtype=np.float32), world),
                     "selected policy world differs")
            event_context = selected.get("context", {})
            target_index = _target_index(details, int(selected["uid"]))
            live_context = _target_context(
                env, int(selected["uid"]), uids, world,
                actor_mask=bool(details["actor_mask"][target_index]),
            )
            _require(live_context["actor_mask"] is True, "selected source actor mask is false")
            for key in ("arrival_id", "phase", "soc", "retry_free", "committed_plan"):
                if key == "soc":
                    _require(math.isclose(float(live_context[key]), float(event_context[key]), rel_tol=0.0, abs_tol=1e-12),
                             f"selected context {key} differs")
                else:
                    _require(live_context[key] == event_context[key], f"selected context {key} differs")
            snapshot = copy.deepcopy(env)
            paired.restore_diagnostic_snapshot(snapshot)
            # A deepcopy made while the source controller is installed can
            # retain a closure bound to the replay environment.  Rebind the
            # native class seam before any branch-local controller is entered.
            _rebind_native_snapshot_methods(snapshot)
            actions, shortlists = followup._row_inputs(source_row)
            target_index = _target_index(details, int(selected["uid"]))
            captured.update({
                "snapshot": snapshot, "uids": list(uids), "world": np.asarray(world, dtype=np.float32).copy(),
                "details": details, "actions": actions, "shortlists": shortlists,
                "source_row": dict(source_row), "target_index": target_index,
                "event": dict(selected), "constructor_rng_consumed": constructor_consumed,
            })
        result = original_step(*args, **kwargs)
        if current == step:
            info = result[4] if isinstance(result, tuple) and len(result) >= 5 and isinstance(result[4], Mapping) else {}
            decisions = info.get("release_decisions", {}) if isinstance(info, Mapping) else {}
            uid = int(selected["uid"])
            decision = decisions.get(uid, decisions.get(str(uid))) if isinstance(decisions, Mapping) else None
            last = getattr(env, "last_release_decisions", {})
            last_decision = last.get(uid, last.get(str(uid))) if isinstance(last, Mapping) else None
            _require(str(decision) == "approved" and str(last_decision) == "approved",
                     "selected source event is not currently approved")
            telemetry_map = info.get("release_intent_telemetry", {}) if isinstance(info, Mapping) else {}
            telemetry = telemetry_map.get(uid, telemetry_map.get(str(uid))) if isinstance(telemetry_map, Mapping) else None
            actual_rank = telemetry.get("runtime_shortlist_rank") if isinstance(telemetry, Mapping) else None
            expected_rank = selected.get("actual", {}).get("runtime_shortlist_rank")
            _require(actual_rank == expected_rank, "selected current shortlist rank differs")
            captured["actual"] = {"decision": decision, "last_decision": last_decision,
                                   "telemetry": telemetry}
        return result

    # A local sidecar scope is used for the one replay; no factory aliases are
    # changed.  Its providers resolve the exact current step and UID.
    def sidecar_order(_env: Any, _uav: Any, uid: int):
        return _sidecar_registration(item, _env, int(uid))["order"]

    def sidecar_metadata(_env: Any, _uav: Any, uid: int):
        return _sidecar_registration(item, _env, int(uid))["metadata"]

    caller_before_replay = _rng_snapshot()
    try:
        env.step = observed_step  # type: ignore[method-assign]
        with prefix_base.prefix_fallback_controller(env, sidecar_order, source_metadata=sidecar_metadata):
            replay_result = SafetyTraceRecorder.replay_action_tape(env, item["tape_path"])
    except BaseException as exc:
        replay_error = exc
    finally:
        env.step = original_step  # type: ignore[method-assign]
        caller_after_replay = _rng_snapshot()
        _restore_rng(caller_before_replay)
    _require(_rng_equal(caller_before_replay, caller_after_replay), "source replay changed caller RNG")
    if replay_error is not None:
        raise replay_error
    _require(isinstance(replay_result, Mapping) and replay_result.get("exact") is True,
             "source action-tape replay was not exact")
    _require(int(replay_result.get("steps_replayed", -1)) == len(tape_rows),
             "source replay step count changed")
    _require(captured and "actual" in captured, "selected approved source event was not captured")
    captured["replay"] = {key: value for key, value in replay_result.items() if key != "steps"}
    captured["caller_rng_unchanged"] = _rng_equal(caller_before_replay, caller_after_replay)
    return captured


def _continuation_request(policy: Any, env: Any, seed: int) -> paired.KeyedRequest:
    caller = _rng_snapshot()
    try:
        context = paired._details_context(policy, env)
        step = int(context.get("step", getattr(env, "step_count", 0)))
        actions: dict[str, np.ndarray] = {}
        shortlists: dict[str, tuple[np.ndarray, ...]] = {}
        orders: dict[int, list[int]] = {}
        metadata: dict[int, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        for index, uid_raw in enumerate(context["uids"]):
            uid = str(uid_raw)
            arrival_id = int(frozen._arrival_id(env, uid_raw))
            actor_mask = bool(context["actor_mask"][index])
            gate = paired.WAIT_GATE
            proposal: int | None = None
            gate_u = proposal_u = None
            if actor_mask:
                gate_u = paired.keyed_uniform(int(seed), step, arrival_id, "gate")
                gate = paired._cdf_sample(context["gate_probabilities"][index], gate_u)
                if gate == paired.RELEASE_GATE:
                    proposal_u = paired.keyed_uniform(int(seed), step, arrival_id, "proposal")
                    proposal = paired._cdf_sample(context["proposal_probabilities"][index], proposal_u)
            if gate == paired.RELEASE_GATE:
                action = HierarchicalActionV42(paired.RELEASE_GATE, proposal).normalized.copy()
                shortlist = paired._shortlist_from_logits(int(proposal), context["proposal_logits"][index])
                shortlists[uid] = shortlist
                logits = [float(item) for item in context["proposal_logits"][index]]
                orders[int(uid_raw)] = _stable_order(int(proposal), logits)
                metadata[int(uid_raw)] = _metadata_for_order(int(proposal), logits)
            else:
                action = HierarchicalActionV42(paired.WAIT_GATE, None).normalized.copy()
            actions[uid] = action
            rows.append({"uid": uid, "arrival_id": arrival_id, "actor_mask": actor_mask,
                         "gate": int(gate), "proposal": proposal, "gate_uniform": gate_u,
                         "proposal_uniform": proposal_u})
        details = {"step": step, "seed": int(seed), "rows": rows,
                   "prefix_orders": orders, "prefix_metadata": metadata,
                   "keying": "arrival_id|absolute_tick|stream"}
        request = paired.KeyedRequest(actions, shortlists, details)
        setattr(env, "_normal_gate_provider", {"orders": orders, "metadata": metadata})
        return request
    finally:
        _restore_rng(caller)


@contextmanager
def _branch_hooks(source_logits: Sequence[float], target_uid: int):
    original_step = paired._call_step
    original_request = paired.keyed_request
    original_details = paired._details_context

    def hooked_request(policy: Any, env: Any):
        seed = getattr(env, "_paired_seed", None)
        _require(seed is not None, "continuation seed missing")
        return _continuation_request(policy, env, int(seed))

    def hooked_details(policy: Any, env: Any):
        return _details_context_once(policy, env, details_fn=original_details)

    def hooked_step(env: Any, actions: Mapping[Any, Any], shortlists: Mapping[Any, Any]):
        provider = getattr(env, "_normal_gate_provider", None)
        if not isinstance(provider, Mapping):
            orders: dict[int, Sequence[int]] = {}
            metadata: dict[int, Mapping[str, Any]] = {}
            step = int(getattr(env, "step_count", -1))
            for raw_uid in shortlists:
                uid = int(raw_uid)
                if uid == int(target_uid):
                    proposal = _action_proposal(actions.get(str(uid), actions.get(uid)))
                    _require(proposal is not None, "target RELEASE proposal missing")
                    orders[uid] = _stable_order(int(proposal), source_logits)
                    metadata[uid] = _metadata_for_order(int(proposal), source_logits)
                else:
                    # Native singleton RELEASE requests are intentionally
                    # outside the K3 prefix registration scope.
                    if len(shortlists[raw_uid]) == 1:
                        continue
                    registrations = getattr(env, "_normal_gate_sidecar", {}).get("registrations", {})
                    registration = registrations.get(f"{step}|{uid}")
                    _require(isinstance(registration, Mapping), f"non-target sidecar registration missing {step}|{uid}")
                    orders[uid] = registration["order"]
                    metadata[uid] = registration["metadata"]
        else:
            orders = provider.get("orders", {})
            metadata = provider.get("metadata", {})
        with _controller_scope(env, orders, metadata):
            return original_step(env, actions, shortlists)

    paired.keyed_request = hooked_request
    paired._call_step = hooked_step
    paired._details_context = hooked_details
    try:
        yield
    finally:
        paired.keyed_request = original_request
        paired._call_step = original_step
        paired._details_context = original_details


def _branch_callback(output: Path, cell_id: int, state_step: int, branch_start: float, phase: str):
    def callback(report: Mapping[str, Any], completed: int, total: int) -> None:
        branch_dir = output / "branches"
        branch_dir.mkdir(parents=True, exist_ok=True)
        branch_index = int(completed) - 1
        seed_index, proposal_index = divmod(branch_index, BRANCHES_PER_SEED)
        continuation_seed = int(CONTINUATION_SEEDS[seed_index])
        proposal = None if proposal_index == 0 else int(proposal_index - 1)
        branch_payload = {
            "stage": STAGE, "phase": str(phase), "cell_id": int(cell_id),
            "branch_index": int(completed), "branches_expected": int(total),
            "continuation_seed": continuation_seed, "proposal": proposal,
            "start_step": int(state_step), "elapsed_wall_seconds": time.perf_counter() - branch_start,
            "report": dict(report), "PPO_ALLOWED": False,
        }
        _write_json(branch_dir / f"cell_{int(cell_id):02d}_{int(completed):03d}.json", branch_payload,
                    refuse_existing=True)
        _write_json(output / "progress.json", {
            "stage": STAGE, "phase": str(phase), "status": "RUNNING", "cell_id": int(cell_id),
            "branches_completed": int(completed), "branches_expected": int(total),
            "continuation_seed": continuation_seed, "proposal": proposal,
            "start_step": int(state_step), "elapsed_wall_seconds": time.perf_counter() - branch_start,
            "PPO_ALLOWED": False, "D_known": False,
        })
        if report.get("error") or report.get("safety_violation") or report.get("safety_invariant_error"):
            raise NormalGateValueError(f"technical branch failure at cell {cell_id} branch {completed}")
    return callback


def _run_state(item: Mapping[str, Any], config: Any, output: Path, phase: str) -> dict[str, Any]:
    caller = _rng_snapshot()
    before, before_load_rng = _load_policy_restoring_rng(_resolve(support.BEFORE_CHECKPOINT), config)
    after, after_load_rng = _load_policy_restoring_rng(_resolve(support.AFTER_CHECKPOINT), config)
    try:
        before_digest = rank_eval._model_digest(before)
        after_digest = rank_eval._model_digest(after)
        _require(before_digest == BEFORE_MODEL_DIGEST, "BEFORE model digest golden changed")
        _require(after_digest == AFTER_MODEL_DIGEST, "AFTER model digest golden changed")
        captured = _capture_state(item, config, before)
        details = captured["details"]
        target_uid = int(item["selected_event"]["uid"])
        index = _target_index(details, target_uid)
        world = np.asarray(captured["world"], dtype=np.float32)
        alignment = paired.gate_update_alignment(before, after, world, delta_q=None)
        _require(alignment.get("proposal_probability_unchanged") is True, "BEFORE/AFTER proposal q changed")
        _require(alignment.get("model_unchanged") is True and alignment.get("rng_unchanged") is True,
                 "gate alignment changed model or RNG")
        q = np.asarray(details["proposal_probabilities"][index], dtype=np.float64)
        logits = np.asarray(details["proposal_logits"][index], dtype=np.float64)
        _require(q.shape == (PROPOSAL_COUNT,) and np.isfinite(q).all() and np.isclose(float(q.sum()), 1.0),
                 "source proposal q invalid")
        source_comparison = item["selected_event"]["comparison"]
        _require(source_comparison.get("q_bitwise_equal") is True,
                 "source event q equality golden is not true")
        _require(source_comparison.get("source_logits_match_before") is True,
                 "source event proposal-logit golden is not true")
        source_logprob = source_comparison.get("source_gate_logprob_check")
        _require(isinstance(source_logprob, Mapping) and source_logprob.get("match") is True,
                 "source event gate-logprob golden is not true")
        _require(np.array_equal(q, np.asarray(source_comparison["q_before"], dtype=np.float64)),
                 "source event q differs")
        q_after_alignment = np.asarray(alignment["proposal_probabilities_after"], dtype=np.float64)[index]
        _require(np.array_equal(q_after_alignment, np.asarray(source_comparison["q_after"], dtype=np.float64)),
                 "source event AFTER q differs")
        _require(np.allclose(logits, np.asarray(source_comparison["source_proposal_logits"], dtype=np.float64), atol=1e-4, rtol=1e-6),
                 "source event logits differ")
        p_before = float(np.asarray(alignment["before_gate_release_probability"], dtype=np.float64)[index])
        p_after = float(np.asarray(alignment["after_gate_release_probability"], dtype=np.float64)[index])
        _require(math.isfinite(p_before) and 0.0 <= p_before <= 1.0 and math.isfinite(p_after) and 0.0 <= p_after <= 1.0,
                 "gate probability invalid")
        _require(math.isclose(p_before, float(source_comparison["before_gate_p"]), rel_tol=0.0, abs_tol=1e-7),
                 "source BEFORE gate probability differs")
        _require(math.isclose(p_after, float(source_comparison["after_gate_p"]), rel_tol=0.0, abs_tol=1e-7),
                 "source AFTER gate probability differs")
        callback = _branch_callback(
            output, int(item["cell_id"]), int(item["selected_event"]["step"]),
            time.perf_counter(), phase,
        )
        setattr(captured["snapshot"], "_normal_gate_sidecar", item["sidecar"])
        branch_start = time.perf_counter()
        with _branch_hooks(logits, target_uid):
            paired_report = paired.paired_gate_value(
                before, captured["snapshot"], captured["actions"], captured["shortlists"],
                target_uid, CONTINUATION_SEEDS, MAX_STEPS,
                branch_callback=callback,
            )
        summary = trained_gate._summarize_paired(paired_report)
        _require(paired_report.get("model_unchanged") is True and paired_report.get("env_snapshot_unchanged") is True
                 and paired_report.get("rng_unchanged") is True, "paired integrity check failed")
        _require(trained_gate._first_actions_unchanged(
            paired_report, captured["actions"], captured["shortlists"], target_uid,
        ), "non-target first action or shortlist changed")
        _require(before_digest == rank_eval._model_digest(before) and after_digest == rank_eval._model_digest(after),
                 "model digest changed")
        delta_p = p_after - p_before
        return {
            "stage": STAGE, "phase": phase, "status": "COMPLETED", "cell_id": int(item["cell_id"]),
            "selection": {key: item["selected_event"].get(key) for key in ("registration_key", "step", "uid", "context")},
            "input_binding": {"event_group_sha256": item["group_sha256"], "arm_sha256": item["arm_sha256"],
                               "tape_sha256": item["tape_sha256"], "sidecar_sha256": item["sidecar_sha256"]},
            "replay": captured["replay"], "source_actual": captured["actual"],
            "p_before": p_before, "p_after": p_after, "delta_p": delta_p,
            "proposal_q_before": q.tolist(), "proposal_logits_before": logits.tolist(),
            "C": summary.get("C"), "D": summary.get("D"),
            "local_m": None if summary.get("D") is None else float(delta_p * float(summary["D"])),
            "per_seed": summary.get("per_seed"), "branch_metrics": summary.get("branch_metrics"),
            "alternative_branch_outcome_counts": summary.get("alternative_branch_outcome_counts"),
            "all_branches_natural": summary.get("all_branches_natural"),
            "qualified_D": summary.get("qualified"),
            "source_policy": {"model_digest": before_digest, "model_unchanged": True},
            "after_policy": {"model_digest": after_digest, "model_unchanged": True},
            "model_load_rng_consumed": {"before": before_load_rng, "after": after_load_rng},
            "caller_rng_unchanged": _rng_equal(caller, _rng_snapshot()),
            "invariants": {"public_k": PUBLIC_K, "controller_max": PROPOSAL_COUNT,
                           "continuation_seeds": list(CONTINUATION_SEEDS), "max_steps": MAX_STEPS,
                           "prefix_sampling": True, "native_policy_sampler_used": False,
                           "optimizer_used": False, "teacher_used": False, "PPO_ALLOWED": False,
                           "D_includes_first_reward": True, "local_m_is_state_local": True},
            "elapsed_wall_seconds": time.perf_counter() - branch_start,
        }
    except BaseException as exc:
        _write_json(output / f"cell_{int(item['cell_id']):02d}_partial.json", {
            "stage": STAGE, "phase": phase, "status": "PARTIAL", "cell_id": int(item["cell_id"]),
            "selection": item["selected_event"], "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        raise
    finally:
        _restore_rng(caller)


def _validate_pilot_reuse(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    pilot_path = output / "pilot.json"
    pilot = _read_json(pilot_path)
    _require(pilot.get("status") == "PILOT_OK", "pilot is not complete")
    _require(pilot.get("phase") == "pilot", "pilot phase changed")
    _require(pilot.get("preflight_sha256") == _sha256(output / "preflight.json"), "pilot preflight binding changed")
    _require(pilot.get("preflight_payload", {}).get("cells") == preflight.get("cells"), "pilot cells changed")
    cell_path = output / "cell_01.json"
    _require(cell_path.is_file(), "immutable pilot cell artifact missing")
    cell_payload = _read_json(cell_path)
    _require(pilot.get("cell_01_sha256") == _sha256(cell_path), "pilot cell artifact hash changed")
    _require(pilot.get("cell") == cell_payload, "pilot embedded cell differs from immutable cell artifact")
    _require(cell_payload.get("status") == "COMPLETED" and int(cell_payload.get("cell_id", -1)) == 1,
             "pilot cell artifact is not the completed cell 1 result")
    return pilot


def run_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = _build_preflight()
    payload["runner_sha256"] = _sha256(Path(__file__))
    payload["protocol_sha256"] = PROTOCOL_SHA256
    _write_json(output / "preflight.json", payload, refuse_existing=True)
    return payload


def run_pilot(output: Path) -> dict[str, Any]:
    preflight_path = output / "preflight.json"
    preflight = _read_json(preflight_path)
    _validate_preflight(preflight)
    config, _ = support._build_inputs()
    _inputs, cells = _load_inputs()
    item = cells[0]
    result = _run_state(item, config, output, "pilot")
    cell_path = output / "cell_01.json"
    _write_json(cell_path, result, refuse_existing=True)
    pilot = {"stage": STAGE, "phase": "pilot", "status": "PILOT_OK",
             "preflight_sha256": _sha256(preflight_path), "preflight_payload": preflight,
             "cell": result, "cell_01_sha256": _sha256(cell_path), "PPO_ALLOWED": False}
    _write_json(output / "pilot.json", pilot, refuse_existing=True)
    return pilot


def run_full(output: Path) -> dict[str, Any]:
    preflight = _read_json(output / "preflight.json")
    _validate_preflight(preflight)
    _validate_pilot_reuse(output, preflight)
    config, _ = support._build_inputs()
    _inputs, cells = _load_inputs()
    # cell_01.json is the immutable result emitted by run_pilot; pilot.json
    # wraps that result again under ``cell``.  Reuse the result file directly
    # so full does not depend on a second, mutable copy.
    results = [_read_json(output / "cell_01.json")]
    for item in cells[1:]:
        result = _run_state(item, config, output, "full")
        _write_json(output / f"cell_{int(item['cell_id']):02d}.json", result, refuse_existing=True)
        results.append(result)
    full = {"stage": STAGE, "phase": "full", "status": "FULL_OK", "cells": results,
            "preflight_sha256": _sha256(output / "preflight.json"),
            "pilot_sha256": _sha256(output / "pilot.json"), "PPO_ALLOWED": False,
            "D": None, "local_m": None}
    _write_json(output / "full.json", full, refuse_existing=True)
    return full


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_dir
    if args.phase == "preflight":
        run_preflight(output)
    elif args.phase == "pilot":
        run_pilot(output)
    else:
        run_full(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
