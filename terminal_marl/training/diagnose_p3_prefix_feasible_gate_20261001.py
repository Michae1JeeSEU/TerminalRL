"""Replay-only P3 diagnosis for a native rank-greater-than-three release.

The runner consumes the immutable BEFORE-prefix tapes and their registration
sidecars.  It never samples a policy action or changes an environment action.
The first potential registration is selected from the tape/sidecar contract;
the actual replay step must then report ``approved`` before the state is used
for the two frozen-policy distribution comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.training import diagnose_p3_matched_gate_shift_20261001 as matched
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_before_prefix_fallback_20261001 as before_prefix
from terminal_marl.training import run_p3_prefix_fallback_20261001 as prefix_base
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as support
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen


STAGE = "P3_PREFIX_FEASIBLE_GATE_20261001_V1"
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
FULL_SHA256 = "30c28c03d26466eabeab2a35c9fc117424648f1a3ae17be6f8d69b6b92cb2195"
BEFORE_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
AFTER_SHA256 = "d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f"
PROTOCOL_SHA256 = "0f76d68574256e8e90a15036ccd01181534f21b665fc16d0662b350e489754b9"
OUTPUT_DIR = Path("/home/user/training_outputs/terminal/p3_followup_20260930/prefix_feasible_gate_20261001")
FRESH_ROOT = Path("/home/user/training_outputs/terminal/p3_followup_20260930/prefix_fallback_fresh_validation_20261001_v4")
INVENTORY = Path(__file__).resolve().parents[2] / ".tmp/p3_prefix_feasible_state_candidates.json"  # legacy reference only
PROTOCOL = Path(__file__).resolve().parents[2] / "docs/evidence/v4_2/p3_prefix_feasible_gate_protocol_20261001.md"


class FeasibleGateError(ValueError):
    """Raised when an immutable replay contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeasibleGateError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
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
        raise FeasibleGateError(f"invalid JSON input: {target}") from exc
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise FeasibleGateError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
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
    resolved = support._sync_path(target)
    if resolved.is_file():
        return resolved
    text = str(target).replace("\\", "/")
    prefix = "/home/user/training_outputs/terminal/"
    if text.startswith(prefix):
        mirror = Path(__file__).resolve().parents[3] / "training_outputs" / "terminal" / text[len(prefix):]
        if mirror.is_file():
            return mirror
    return resolved


def _rng_snapshot() -> Any:
    return support._rng_snapshot()


def _restore_rng(value: Any) -> None:
    support._restore_rng(value)


def _rng_equal(left: Any, right: Any) -> bool:
    return support._rng_equal(left, right)


def _mapping_value(mapping: Mapping[Any, Any] | None, key: int, default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def _numeric_key(row: Mapping[str, Any]) -> tuple[int, int]:
    try:
        step = int(row["step"])
        uid = int(row["uid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeasibleGateError("candidate step and UID must be numeric") from exc
    return step, uid


def _validate_order(order: Any) -> list[int]:
    _require(isinstance(order, Sequence) and not isinstance(order, (str, bytes)), "sidecar order is missing")
    values = [int(item) for item in order]
    _require(len(values) == PROPOSAL_COUNT and set(values) == set(range(PROPOSAL_COUNT)),
             "sidecar proposal order is not a complete 25-entry catalogue")
    return values


def _potential_candidate(row: Mapping[str, Any], registration: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a pre-replay candidate without consulting p, reward, or outcomes."""
    step, uid = _numeric_key(row)
    runtime = _mapping_value(row.get("runtime_shortlist_result"), uid)
    requested_gate = _mapping_value(row.get("requested_gate"), uid)
    if not isinstance(runtime, Mapping) or runtime.get("used") is not True:
        return None
    try:
        rank = int(runtime.get("rank"))
    except (TypeError, ValueError):
        return None
    if rank <= PUBLIC_K or requested_gate != 1:
        return None
    key = f"{step}|{uid}"
    _validate_order(registration.get("order"))
    metadata = registration.get("metadata")
    _require(isinstance(metadata, Mapping), f"sidecar metadata missing for {key}")
    logits = metadata.get("source_logits")
    _require(isinstance(logits, Sequence) and len(logits) == PROPOSAL_COUNT,
             f"sidecar source logits missing for {key}")
    return {
        "step": step,
        "uid": uid,
        "registration_key": key,
        "rank": rank,
        "runtime_candidates_checked": int(runtime.get("candidates_checked", rank)),
        "requested_gate": int(requested_gate),
        "requested_proposal": _mapping_value(row.get("requested_proposal"), uid),
        "observation_hash": row.get("observation_hash"),
        "sidecar_order": [int(item) for item in registration["order"]],
        "sidecar_source_logits": [float(item) for item in logits],
    }


def select_first_potential(rows: Sequence[Mapping[str, Any]], sidecar: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select potential rows in stable ``(step, numeric UID)`` order.

    This function deliberately does not read p, reward, terminal outcome, or
    ``decision_history``.  Actual approval is checked only after the native
    replay step.
    """
    registrations = sidecar.get("registrations")
    _require(isinstance(registrations, Mapping), "sidecar registrations missing")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        runtime = row.get("runtime_shortlist_result")
        row_variants: list[Mapping[str, Any]]
        if "uid" in row:
            row_variants = [row]
        elif isinstance(runtime, Mapping):
            # ActionTape rows contain per-UID maps rather than a scalar UID.
            row_variants = [{**row, "uid": raw_uid} for raw_uid in runtime]
        else:
            row_variants = []
        for row_variant in row_variants:
            try:
                step, uid = _numeric_key(row_variant)
            except FeasibleGateError:
                continue
            key = f"{step}|{uid}"
            registration = registrations.get(key)
            if not isinstance(registration, Mapping):
                continue
            candidate = _potential_candidate(row_variant, registration)
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=lambda item: (int(item["step"]), int(item["uid"])))
    return candidates


def select_first_actual_approved(candidates: Sequence[Mapping[str, Any]], *, actual: Mapping[int, str]) -> dict[str, Any] | None:
    """Apply the post-step approval gate to already ordered candidates."""
    for candidate in sorted(candidates, key=lambda item: _numeric_key(item)):
        if str(_mapping_value(actual, int(candidate["uid"]), "")) == "approved":
            return dict(candidate)
    return None


def physical_context_eligible(context: Mapping[str, Any]) -> bool:
    """Apply the pre-registered physical-window predicates to a saved row."""
    try:
        soc = float(context["soc"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        str(context.get("phase")) == "ring_inner"
        and math.isfinite(soc) and 0.4 < soc <= 0.65
        and bool(context.get("actor_mask"))
        and bool(context.get("retry_free"))
        and not bool(context.get("committed_plan"))
    )


def _validate_inventory(inventory: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    scope = inventory.get("scope")
    _require(isinstance(scope, Mapping) and scope.get("cells") == list(CELLS), "candidate inventory cell scope changed")
    _require(scope.get("arm") == "before_prefix", "candidate inventory arm changed")
    cells = inventory.get("cells")
    _require(isinstance(cells, Mapping), "candidate inventory cells missing")
    result: dict[int, dict[str, Any]] = {}
    for cell_id in CELLS:
        item = cells.get(str(cell_id), cells.get(cell_id))
        _require(isinstance(item, Mapping), f"candidate inventory cell {cell_id} missing")
        for key in ("arm_path", "tape_path", "sidecar_path", "tape_sha256", "sidecar_sha256"):
            _require(isinstance(item.get(key), str) and item[key], f"candidate inventory {cell_id} {key} missing")
        result[cell_id] = dict(item)
    return result


def _load_cell_inputs(cell_id: int, item: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load the fresh V4 arm and its own tape/sidecar declarations.

    The earlier compact candidate JSON is historical evidence only.  Runtime
    selection is recomputed from the immutable tape and sidecar below.
    """
    arm_path = _resolve(FRESH_ROOT / "arms" / f"cell_{int(cell_id):02d}_before_prefix.json")
    _require(arm_path.is_file(), f"cell {cell_id} arm input missing")
    arm = _read_json(arm_path)
    _require(arm.get("status") == "completed", f"cell {cell_id} arm is not completed")
    _require(str(arm.get("checkpoint_sha256")) == BEFORE_SHA256, f"cell {cell_id} checkpoint is not BEFORE")
    control = arm.get("prefix_fallback_control")
    _require(isinstance(control, Mapping), f"cell {cell_id} prefix control binding missing")
    tape_declared = arm.get("action_tape")
    sidecar_declared = control.get("orders_sidecar")
    sidecar_declared_sha = control.get("orders_sidecar_sha256")
    _require(isinstance(tape_declared, str) and tape_declared, f"cell {cell_id} action tape declaration missing")
    _require(isinstance(sidecar_declared, str) and sidecar_declared, f"cell {cell_id} sidecar declaration missing")
    _require(isinstance(sidecar_declared_sha, str) and sidecar_declared_sha, f"cell {cell_id} sidecar SHA declaration missing")
    tape_path = _resolve(tape_declared)
    sidecar_path = _resolve(sidecar_declared)
    _require(tape_path.is_file() and sidecar_path.is_file(), f"cell {cell_id} tape/sidecar input missing")
    _require(_sha256(sidecar_path) == str(sidecar_declared_sha), f"cell {cell_id} sidecar SHA changed")
    tape = _read_json(tape_path)
    rows = tape.get("action_tape", tape.get("records", []))
    _require(isinstance(rows, list) and rows, f"cell {cell_id} tape rows missing")
    sidecar = _read_json(sidecar_path)
    _require(sidecar.get("arm") == "before_prefix_fallback", f"cell {cell_id} sidecar arm changed")
    potential = select_first_potential(rows, sidecar)
    full_path = arm_path.parent.parent / "full.json"
    _require(full_path.is_file() and _sha256(full_path) == FULL_SHA256, f"cell {cell_id} full binding changed")
    return {
        "cell_id": int(cell_id), "arm_path": arm_path, "tape_path": tape_path,
        "sidecar_path": sidecar_path, "arm": arm, "tape": tape, "rows": rows,
        "sidecar": sidecar, "potential": potential, "full_path": full_path,
        "arm_sha256": _sha256(arm_path), "tape_sha256": _sha256(tape_path),
        "sidecar_sha256": _sha256(sidecar_path), "full_sha256": _sha256(full_path),
    }


def _config_snapshot(config: Any) -> dict[str, Any]:
    return support._config_snapshot(config)


def _source_bindings(config: Any) -> dict[str, Any]:
    before = _resolve(support.BEFORE_CHECKPOINT)
    after = _resolve(support.AFTER_CHECKPOINT)
    _require(_sha256(before) == BEFORE_SHA256, "BEFORE checkpoint SHA changed")
    _require(_sha256(after) == AFTER_SHA256, "AFTER checkpoint SHA changed")
    return {
        "before": {"declared": str(support.BEFORE_CHECKPOINT), "resolved": str(before), "sha256": BEFORE_SHA256, "critic_role": "BEFORE own critic"},
        "after": {"declared": str(support.AFTER_CHECKPOINT), "resolved": str(after), "sha256": AFTER_SHA256, "critic_role": "AFTER own critic"},
        "warmup_loaded": False,
        "model_forward_used": False,
        "config_snapshot": _config_snapshot(config),
    }


def _build_preflight() -> dict[str, Any]:
    caller = _rng_snapshot()
    try:
        config, _ = support._build_inputs()
        _require(PROTOCOL.is_file() and _sha256(PROTOCOL) == PROTOCOL_SHA256, "feasible-gate protocol binding changed")
        cells = [_load_cell_inputs(cell_id) for cell_id in CELLS]
        source = _source_bindings(config)
    finally:
        _restore_rng(caller)
    return {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
        "environment_created": False, "model_forward_used": False,
        "gradient_operations": 0, "optimizer_operations": 0, "teacher_calls": 0,
        "PPO_ALLOWED": PPO_ALLOWED, "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED, "optimizer_used": OPTIMIZER_USED,
        "off_policy": OFF_POLICY, "no_training": NO_TRAINING,
        "cells": [{"cell_id": int(item["cell_id"]), "arm": "before_prefix", "full_sha256": item["full_sha256"],
                   "arm_sha256": item["arm_sha256"], "tape_sha256": item["tape_sha256"],
                   "sidecar_sha256": item["sidecar_sha256"], "potential_candidates": item["potential"]}
                  for item in cells],
        "requests_per_cell": REQUESTS_PER_CELL, "max_steps": MAX_STEPS,
        "schema": {"observation": str(config.observation_schema_version), "observation_dim": OBSERVATION_DIM,
                    "action": str(config.action_schema_version), "public_shortlist_k": PUBLIC_K,
                    "proposal_count": PROPOSAL_COUNT, "selection_actor_mask": "verified only during replay"},
        "config": _config_snapshot(config), "source_bindings": source,
        "input_inventory": {"path": str(FRESH_ROOT), "full_sha256": FULL_SHA256,
                            "source": "fresh V4 full plus arm-declared tape/sidecar hashes"},
        "protocol_sha256": PROTOCOL_SHA256, "runner_sha256": _sha256(Path(__file__)),
        "selection_rule": "first (step,numeric_uid) potential tape/sidecar row; actual post-step info release_decision approved required",
        "value_outputs": {"D": None, "local_m": None, "interpretation": "one approved RELEASE is not a long-horizon value or root-cause result"},
    }


def _validate_preflight(payload: Mapping[str, Any]) -> None:
    _require(payload.get("stage") == STAGE and payload.get("status") == "PREFLIGHT_OK", "preflight is not successful")
    for key, expected in (("environment_created", False), ("model_forward_used", False),
                          ("gradient_operations", 0), ("optimizer_operations", 0), ("teacher_calls", 0),
                          ("PPO_ALLOWED", False),
                          ("heldout_used", False), ("teacher_used", False), ("optimizer_used", False),
                          ("off_policy", True), ("no_training", True)):
        _require(payload.get(key) is expected, f"preflight flag {key} changed")
    _require(payload.get("runner_sha256") == _sha256(Path(__file__)), "runner changed after preflight")
    _require(payload.get("protocol_sha256") == PROTOCOL_SHA256, "protocol changed after preflight")
    _require(payload.get("input_inventory", {}).get("full_sha256") == FULL_SHA256, "fresh full binding changed")
    config, _ = support._build_inputs()
    _require(payload.get("config") == _config_snapshot(config), "config changed after preflight")
    _require(payload.get("source_bindings") == _source_bindings(config), "source binding changed after preflight")
    current = _build_preflight()
    _require(payload.get("cells") == current.get("cells"), "immutable cell input binding changed")


def _native_actor_mask(env: Any, actor: Any) -> bool:
    uid = int(getattr(actor, "uid"))
    phase = str(getattr(getattr(actor, "phase", None), "value", getattr(actor, "phase", None)))
    eligible = bool(
        getattr(actor, "committed_plan", None) is None
        and phase in {"ring_outer", "ring_middle", "ring_inner"}
        and uid not in getattr(env, "future_release_reservations", set())
    )
    if eligible and str(getattr(env.config, "student_decision_mask_mode", "legacy")) == "pre_action_forced_wait":
        affected = set(env.airspace.affected_interfaces())
        role = str(getattr(getattr(actor, "ring_role", None), "value", getattr(actor, "ring_role", None)))
        interface = {"outer": "o2m", "middle": "m2i", "inner": "i2f"}.get(role)
        eligible = bool(interface is not None and float(getattr(actor, "soc")) > 0.0
                        and float(getattr(env, "now", 0.0)) >= float(getattr(actor, "safety_retry_after", 0.0))
                        and interface not in affected)
    return eligible


def _actor_context(env: Any, uid: int, uids: Sequence[str], world: np.ndarray) -> dict[str, Any]:
    actor = getattr(env, "active", {}).get(uid)
    if actor is None:
        actor = getattr(env, "active", {}).get(str(uid))
    if actor is None:
        return {"uid": int(uid), "actor_mask": False, "missing_actor": True}
    index = list(uids).index(str(uid))
    now = float(getattr(env, "now", 0.0))
    retry_after = float(getattr(actor, "safety_retry_after", 0.0))
    phase = str(getattr(getattr(actor, "phase", None), "value", getattr(actor, "phase", None)))
    return {
        "uid": int(uid), "arrival_id": int(getattr(actor, "arrival_id")), "phase": phase,
        "soc": float(getattr(actor, "soc")), "safety_retry_after": retry_after,
        "retry_free": bool(now >= retry_after), "committed_plan": bool(getattr(actor, "committed_plan", None) is not None),
        "now": now, "actor_mask": bool(_native_actor_mask(env, actor)),
        "world": np.asarray(world[index], dtype=np.float32).copy(), "index": int(index),
    }


def _distribution_snapshot(policy: Any, world: np.ndarray) -> dict[str, np.ndarray]:
    before = _rng_snapshot()
    after = None
    try:
        result = matched._distribution_snapshot(policy, world)
        after = _rng_snapshot()
        _require(_rng_equal(before, after), "distribution observation changed RNG")
        return result
    finally:
        _restore_rng(before)


def _load_policy_restoring_rng(path: Path, config: Any) -> tuple[Any, Any, bool]:
    """Load a frozen policy without leaking constructor RNG into replay."""
    before = _rng_snapshot()
    after = None
    try:
        policy, payload = rank_eval._load_policy(path, config)
        after = _rng_snapshot()
        return policy, payload, not _rng_equal(before, after)
    finally:
        if after is None:
            after = _rng_snapshot()
        _restore_rng(before)


def _controller_scope(env: Any, sidecar: Mapping[str, Any]):
    registrations = sidecar.get("registrations")
    _require(isinstance(registrations, Mapping), "sidecar registrations missing")
    audit = {"registrations": dict(registrations)}

    def order_provider(current_env: Any, _uav: Any, uid: int):
        return prefix_base._registration_lookup(audit, current_env, uid)["order"]

    def metadata_provider(current_env: Any, _uav: Any, uid: int):
        return prefix_base._registration_lookup(audit, current_env, uid)["metadata"]

    return prefix_base.prefix_fallback_controller(
        env, order_provider, source_metadata=metadata_provider,
    )


def _replay_pair(item: Mapping[str, Any], before_policy: Any, after_policy: Any,
                 config: Any, output: Path) -> dict[str, Any]:
    env_rng_before = _rng_snapshot()
    env = frozen._CapturedTerminalEnv(config)
    env_rng_after = _rng_snapshot()
    env_constructor_rng_consumed = not _rng_equal(env_rng_before, env_rng_after)
    _restore_rng(env_rng_before)
    original_step = env.step
    rows = list(item["rows"])
    row_by_step = {int(row["step"]): row for row in rows}
    potential = list(item["potential"])
    selected: dict[str, Any] | None = None
    excluded: dict[str, int] = {"physical_window": 0, "not_approved": 0, "rank_unknown_or_le3": 0}
    progress = output / "progress.json"
    before_digest = rank_eval._model_digest(before_policy)
    after_digest = rank_eval._model_digest(after_policy)

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal selected
        step = int(getattr(env, "step_count", -1))
        source_row = row_by_step.get(step)
        _require(source_row is not None, f"replay step {step} missing from tape")
        observations = env._observations()
        _require(_digest({key: value for key, value in sorted(observations.items())}) == source_row.get("observation_hash"),
                 f"observation hash mismatch at step {step}")
        candidates: list[dict[str, Any]] = []
        if selected is None:
            uids, world = matched._world_from_observations(observations)
            for candidate in potential:
                if int(candidate["step"]) != step:
                    continue
                context = _actor_context(env, int(candidate["uid"]), uids, world)
                if physical_context_eligible(context):
                    candidates.append({**candidate, "context": context, "world": world.copy()})
                else:
                    excluded["physical_window"] += 1
        result = original_step(*args, **kwargs)
        info = result[4] if isinstance(result, tuple) and len(result) >= 5 and isinstance(result[4], Mapping) else {}
        decisions = info.get("release_decisions", {}) if isinstance(info, Mapping) else {}
        for candidate in sorted(candidates, key=lambda x: (int(x["step"]), int(x["uid"]))):
            if selected is not None:
                break
            uid = int(candidate["uid"])
            decision = str(_mapping_value(decisions, uid, ""))
            last_decision = str(_mapping_value(getattr(env, "last_release_decisions", {}), uid, ""))
            if decision != "approved" or last_decision != "approved":
                excluded["not_approved"] += 1
                continue
            telemetry = _mapping_value(info.get("release_intent_telemetry"), uid, None)
            if not isinstance(telemetry, Mapping):
                telemetry = _mapping_value(getattr(env, "last_release_telemetry", {}), uid, None)
            try:
                actual_rank = int(telemetry.get("runtime_shortlist_rank")) if isinstance(telemetry, Mapping) else None
            except (TypeError, ValueError):
                actual_rank = None
            if actual_rank is None or actual_rank <= PUBLIC_K:
                excluded["rank_unknown_or_le3"] += 1
                continue
            context = candidate["context"]
            index = int(context["index"])
            registration = item["sidecar"]["registrations"][str(candidate["registration_key"])]
            before_dist = _distribution_snapshot(before_policy, candidate["world"])
            after_dist = _distribution_snapshot(after_policy, candidate["world"])
            before_logits = np.asarray(before_dist["proposal_logits"][index], dtype=np.float64)
            _require(np.allclose(before_logits, np.asarray(candidate["sidecar_source_logits"], dtype=np.float64), atol=1e-4, rtol=1e-6),
                     f"sidecar source logits mismatch at {candidate['registration_key']}")
            _require(np.array_equal(np.asarray(before_dist["proposal_q"][index]), np.asarray(after_dist["proposal_q"][index])),
                     "BEFORE/AFTER proposal q is not bitwise equal")
            tape_gate = _mapping_value(source_row.get("gate_logprob"), uid, None)
            before_gate_p = float(np.asarray(before_dist["gate_probs"], dtype=np.float64)[index, 1])
            if tape_gate is not None and math.isfinite(float(tape_gate)):
                _require(abs(float(tape_gate) - math.log(max(before_gate_p, np.finfo(float).tiny))) <= 1e-4,
                         f"source RELEASE gate logprob mismatch at {candidate['registration_key']}")
            selected = {
                **candidate, "decision": "approved", "actual_rank": actual_rank,
                "pre_observation_hash": source_row.get("observation_hash"),
                "post_state_hash": _digest(SafetyTraceRecorder._state(env)),
                "before_distribution": before_dist, "after_distribution": after_dist,
                "before_gate_p": before_gate_p,
                "before_q": np.asarray(before_dist["proposal_q"][index], dtype=np.float64).copy(),
                "after_gate_p": float(np.asarray(after_dist["gate_probs"], dtype=np.float64)[index, 1]),
                "after_q": np.asarray(after_dist["proposal_q"][index], dtype=np.float64).copy(),
                "source_proposal_logits": before_logits.copy(), "registration": dict(registration),
            }
        if step == 1 or (step + 1) % 50 == 0:
            _write_json(progress, {"stage": STAGE, "phase": "arm", "status": "RUNNING", "cell_id": int(item["cell_id"]),
                                   "step": step, "potential_candidates_seen": len(potential),
                                   "selected": None if selected is None else {"step": selected["step"], "uid": selected["uid"]}})
        return result

    try:
        env.step = observed_step  # type: ignore[method-assign]
        with _controller_scope(env, item["sidecar"]):
            replay = SafetyTraceRecorder.replay_action_tape(env, item["tape_path"])
    finally:
        env.step = original_step  # type: ignore[method-assign]
    _require(replay.get("exact") is True and int(replay.get("steps_replayed", -1)) == len(rows),
             "action tape replay was not exact")
    _require(rank_eval._model_digest(before_policy) == before_digest, "BEFORE model digest changed during replay")
    _require(rank_eval._model_digest(after_policy) == after_digest, "AFTER model digest changed during replay")
    return {"replay": {key: value for key, value in replay.items() if key != "steps"},
            "selected": selected, "excluded": excluded,
            "model_digest_before": before_digest, "model_digest_after": after_digest,
            "controller_installed_and_restored": True,
            "environment_constructor_rng_consumed": env_constructor_rng_consumed}


def _save_selected_npz(path: Path, selected: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    context = selected["context"]
    index = int(context["index"])
    np.savez_compressed(
        path, world=np.asarray(selected["world"][index], dtype=np.float32),
        uid=np.asarray([int(selected["uid"])], dtype=np.int64),
        arrival_id=np.asarray([int(context["arrival_id"])], dtype=np.int64),
        actor_mask=np.asarray([bool(context["actor_mask"])], dtype=bool),
        before_gate_p=np.asarray([float(selected["before_gate_p"])], dtype=np.float64),
        after_gate_p=np.asarray([float(selected["after_gate_p"])], dtype=np.float64),
        before_q=np.asarray(selected["before_q"], dtype=np.float64), after_q=np.asarray(selected["after_q"], dtype=np.float64),
        metadata_json=np.asarray(json.dumps({"D": None, "local_m": None, "step": int(selected["step"])}, sort_keys=True)),
    )
    return _sha256(path)


def _run_cell(item: Mapping[str, Any], config: Any, output: Path) -> dict[str, Any]:
    caller_before = _rng_snapshot()
    caller_after = None
    try:
        before_policy, _, before_load_rng_consumed = _load_policy_restoring_rng(
            _resolve(support.BEFORE_CHECKPOINT), config
        )
        after_policy, _, after_load_rng_consumed = _load_policy_restoring_rng(
            _resolve(support.AFTER_CHECKPOINT), config
        )
        replay = _replay_pair(item, before_policy, after_policy, config, output)
        selected = replay["selected"]
        comparison: dict[str, Any] = {"D": None, "local_m": None, "complete_value_known": False}
        npz_sha = None
        if selected is not None:
            comparison.update({"selected_step": int(selected["step"]), "selected_uid": int(selected["uid"]),
                               "arrival_id": int(selected["context"]["arrival_id"]),
                               "before_gate_p": float(selected["before_gate_p"]), "after_gate_p": float(selected["after_gate_p"]),
                               "q_bitwise_equal": True, "decision": "approved", "actual_rank": int(selected["actual_rank"])})
            npz_sha = _save_selected_npz(output / "selected" / f"cell_{int(item['cell_id']):02d}_selected.npz", selected)
        caller_after = _rng_snapshot()
        caller_unchanged = _rng_equal(caller_before, caller_after)
        _require(caller_unchanged, "caller RNG changed before restoration")
        result = {"stage": STAGE, "phase": "arm", "status": "COMPLETED" if selected is not None else "COMPLETED_MISSING",
                  "cell_id": int(item["cell_id"]), "arm": "before_prefix", "tape_sha256": item["tape_sha256"],
                  "sidecar_sha256": item["sidecar_sha256"], "potential_candidates": item["potential"],
                  "replay": replay, "comparison": comparison, "selected_npz_sha256": npz_sha,
                  "caller_rng_unchanged_before_restore": caller_unchanged,
                  "model_load_rng_consumed": {"before": bool(before_load_rng_consumed),
                                               "after": bool(after_load_rng_consumed)}}
        return result
    finally:
        if caller_after is None:
            caller_after = _rng_snapshot()
        _restore_rng(caller_before)


def _run_phase(phase: str, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    preflight_path = output / "preflight.json"
    if phase == "preflight":
        payload = _build_preflight()
        _write_json(preflight_path, payload, refuse_existing=True)
        return payload
    _require(preflight_path.is_file(), "preflight.json is required")
    preflight = _read_json(preflight_path)
    _validate_preflight(preflight)
    config, _ = support._build_inputs()
    cells = [_load_cell_inputs(cell_id) for cell_id in CELLS]
    cells_dir = output / "cells"
    if phase == "pilot":
        result = _run_cell(cells[0], config, output)
        _write_json(cells_dir / "cell_01.json", result, refuse_existing=True)
        summary = {"stage": STAGE, "phase": "pilot", "status": "PILOT_OK", "cell_ids": [1],
                   "cells": [{"cell_id": 1, "status": result["status"], "selected": result["comparison"].get("selected_step") is not None},
                             ], "preflight_sha256": _sha256(preflight_path),
                   "cell_sha256": _sha256(cells_dir / "cell_01.json")}
        _write_json(output / "pilot.json", summary, refuse_existing=True)
        return summary
    _require(phase == "full", f"unknown phase: {phase}")
    pilot = _read_json(output / "pilot.json")
    _require(pilot.get("status") == "PILOT_OK" and pilot.get("cell_ids") == [1], "pilot reuse binding changed")
    _require(pilot.get("preflight_sha256") == _sha256(preflight_path), "pilot preflight binding changed")
    pilot_cell = cells_dir / "cell_01.json"
    _require(pilot_cell.is_file(), "pilot cell artifact missing")
    _require(pilot.get("cell_sha256") == _sha256(pilot_cell), "pilot cell binding changed")
    _require(_read_json(pilot_cell).get("status") in {"COMPLETED", "COMPLETED_MISSING"}, "pilot cell is not complete")
    results = [_read_json(pilot_cell)]
    for item in cells[1:]:
        result = _run_cell(item, config, output)
        _write_json(cells_dir / f"cell_{int(item['cell_id']):02d}.json", result, refuse_existing=True)
        results.append(result)
    summary = {"stage": STAGE, "phase": "full", "status": "FULL_OK", "cell_ids": CELLS,
               "cells": [{"cell_id": int(result["cell_id"]), "status": result["status"],
                          "selected": result["comparison"].get("selected_step") is not None} for result in results],
               "preflight_sha256": _sha256(preflight_path), "pilot_sha256": _sha256(output / "pilot.json"),
               "D": None, "local_m": None}
    _write_json(output / "full.json", summary, refuse_existing=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        result = _run_phase(args.phase, args.output_dir)
    except Exception as exc:
        error = {"stage": STAGE, "phase": args.phase, "status": "ERROR",
                 "error": {"type": type(exc).__name__, "message": str(exc)}}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "error.json", error, refuse_existing=False)
        _write_json(args.output_dir / "partial.json", {**error, "retained": True}, refuse_existing=False)
        print(json.dumps(error, sort_keys=True))
        return 1
    print(json.dumps(_jsonable(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
