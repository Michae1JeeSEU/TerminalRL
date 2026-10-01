"""Enumerate all registered native RELEASE gate events by exact replay.

This module is intentionally separate from the earlier first-feasible-event
diagnostic.  It replays each immutable BEFORE-prefix tape once, records every
source potential registration (including native K=3 entries), and evaluates
the two frozen policies only for current, normal approved events.  It never
produces a return or an action value estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.training import diagnose_p3_prefix_feasible_gate_20261001 as base
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as support
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval


STAGE = "P3_PREFIX_GATE_EVENTS_20261001_V1"
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
PROTOCOL_SHA256 = "59d812d5ff1a53d18b7407dedf25a2b7d107902087796e4bd3a97f41910ff6e5"
OUTPUT_DIR = Path("/home/user/training_outputs/terminal/p3_followup_20260930/prefix_gate_events_20261001")
PROTOCOL = Path(__file__).resolve().parents[2] / "docs/evidence/v4_2/p3_prefix_gate_events_protocol_20261001.md"


class GateEventsError(ValueError):
    """Raised when the immutable event-replay contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateEventsError(message)


def _jsonable(value: Any) -> Any:
    return base._jsonable(value)


def _read_json(path: str | Path) -> dict[str, Any]:
    return base._read_json(path)


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    base._write_json(path, value, refuse_existing=refuse_existing)


def _sha256(path: str | Path) -> str:
    return base._sha256(path)


def _resolve(path: str | Path) -> Path:
    return base._resolve(path)


def _rng_snapshot() -> Any:
    return base._rng_snapshot()


def _restore_rng(value: Any) -> None:
    base._restore_rng(value)


def _rng_equal(left: Any, right: Any) -> bool:
    return base._rng_equal(left, right)


def _mapping_value(mapping: Mapping[Any, Any] | None, key: int, default: Any = None) -> Any:
    return base._mapping_value(mapping, key, default)


def _numeric_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return base._numeric_key(row)


def select_all_potential(rows: Sequence[Mapping[str, Any]], sidecar: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enumerate every registered native requested RELEASE event.

    The native shortlist rank is descriptive here.  Rank 1--3 and rank>3
    registrations are both retained; the current replay rank is checked later
    against the tape value when it is available.
    """
    registrations = sidecar.get("registrations")
    _require(isinstance(registrations, Mapping), "sidecar registrations missing")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        requested = row.get("requested_gate")
        runtime = row.get("runtime_shortlist_result")
        requested_uids = list(requested) if isinstance(requested, Mapping) else []
        if "uid" in row:
            variants = [row]
        else:
            # requested_gate is the native call catalogue.  A rejected or
            # unattempted request may have no runtime shortlist row; it is
            # retained as rank-unknown when a matching sidecar registration
            # exists.
            variants = [dict(row, uid=raw_uid) for raw_uid in requested_uids]
        for variant in variants:
            try:
                step, uid = _numeric_key(variant)
            except GateEventsError:
                continue
            requested_gate = _mapping_value(variant.get("requested_gate"), uid, None)
            runtime_row = _mapping_value(variant.get("runtime_shortlist_result"), uid, None)
            if requested_gate != 1:
                continue
            key = f"{step}|{uid}"
            registration = registrations.get(key)
            if not isinstance(registration, Mapping):
                # Canonical/system requests without a prefix registration are
                # outside this event catalogue and are retained by the native
                # tape, but are not a prefix event.
                continue
            order = registration.get("order")
            metadata = registration.get("metadata")
            _require(isinstance(order, Sequence) and not isinstance(order, (str, bytes))
                     and len(order) == PROPOSAL_COUNT and set(int(x) for x in order) == set(range(PROPOSAL_COUNT)),
                     f"sidecar proposal order is invalid: {key}")
            logits = metadata.get("source_logits") if isinstance(metadata, Mapping) else None
            _require(isinstance(logits, Sequence) and len(logits) == PROPOSAL_COUNT,
                     f"sidecar source logits missing: {key}")
            raw_rank = runtime_row.get("rank") if isinstance(runtime_row, Mapping) else None
            try:
                rank = int(raw_rank) if raw_rank is not None and int(raw_rank) > 0 else None
            except (TypeError, ValueError):
                rank = None
            if rank is not None:
                _require(1 <= rank <= PROPOSAL_COUNT, f"native shortlist rank out of range: {key}")
            candidates.append({
                "step": step, "uid": uid, "registration_key": key,
                "rank": rank, "runtime_candidates_checked": runtime_row.get("candidates_checked")
                if isinstance(runtime_row, Mapping) else None,
                "runtime_used": runtime_row.get("used") if isinstance(runtime_row, Mapping) else None,
                "requested_gate": 1, "requested_proposal": _mapping_value(variant.get("requested_proposal"), uid),
                "observation_hash": variant.get("observation_hash"),
                "sidecar_order": [int(x) for x in order],
                "sidecar_source_logits": [float(x) for x in logits],
            })
    candidates.sort(key=lambda item: (int(item["step"]), int(item["uid"])))
    return candidates


def _cell_binding(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": int(item["cell_id"]),
        "arm": "before_prefix",
        "full_sha256": str(item["full_sha256"]),
        "arm_sha256": str(item["arm_sha256"]),
        "tape_sha256": str(item["tape_sha256"]),
        "sidecar_sha256": str(item["sidecar_sha256"]),
        "potential": [
            {"step": int(row["step"]), "uid": int(row["uid"]),
             "rank": None if row.get("rank") is None else int(row["rank"]),
             "runtime_used": row.get("runtime_used"),
             "registration_key": str(row["registration_key"])}
            for row in item["potential"]
        ],
    }


def _load_cell_inputs(cell_id: int) -> dict[str, Any]:
    """Load the same fresh V4 input contract used by the prior replay."""
    item = base._load_cell_inputs(int(cell_id))
    item["potential"] = select_all_potential(item["rows"], item["sidecar"])
    return item


def _config_snapshot(config: Any) -> dict[str, Any]:
    return base._config_snapshot(config)


def _source_bindings(config: Any) -> dict[str, Any]:
    bindings = base._source_bindings(config)
    _require(bindings.get("before", {}).get("sha256") == BEFORE_SHA256, "BEFORE source binding changed")
    _require(bindings.get("after", {}).get("sha256") == AFTER_SHA256, "AFTER source binding changed")
    return bindings


def _build_preflight() -> dict[str, Any]:
    caller = _rng_snapshot()
    try:
        config, _ = support._build_inputs()
        _require(PROTOCOL.is_file() and _sha256(PROTOCOL) == PROTOCOL_SHA256,
                 "gate-events protocol binding changed")
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
        "cells": [_cell_binding(item) for item in cells],
        "requests_per_cell": REQUESTS_PER_CELL, "max_steps": MAX_STEPS,
        "input_full_sha256": FULL_SHA256,
        "source_bindings": source, "config": _config_snapshot(config),
        "protocol_sha256": PROTOCOL_SHA256, "runner_sha256": _sha256(Path(__file__)),
        "selection": {
            "all_registered_requested_gate_events": True, "rank_filter": False,
            "all_source_potentials": True, "first_event_selection": False,
            "upper_soc_filter": False, "decision_history_used": False,
            "physical_normal_predicates": [
                "current_info_approved", "current_last_release_approved",
                "actor_mask_true", "ring_outer_middle_inner", "finite_soc_gt_0.4",
                "retry_free", "no_committed_plan",
            ],
            "rank_descriptor": "known 1..25 validated against current telemetry; unknown retained; no rank filter",
        },
        "value_outputs": {"D": None, "local_m": None},
    }


def _validate_preflight(payload: Mapping[str, Any]) -> None:
    _require(payload.get("stage") == STAGE and payload.get("status") == "PREFLIGHT_OK",
             "gate-events preflight is not successful")
    expected_flags = {
        "environment_created": False, "model_forward_used": False,
        "gradient_operations": 0, "optimizer_operations": 0, "teacher_calls": 0,
        "PPO_ALLOWED": False, "heldout_used": False, "teacher_used": False,
        "optimizer_used": False, "off_policy": True, "no_training": True,
    }
    for key, expected in expected_flags.items():
        _require(payload.get(key) == expected, f"preflight flag {key} changed")
    _require(payload.get("runner_sha256") == _sha256(Path(__file__)), "runner changed after preflight")
    _require(payload.get("protocol_sha256") == PROTOCOL_SHA256, "protocol changed after preflight")
    _require(payload.get("input_full_sha256") == FULL_SHA256, "input full binding changed")
    config, _ = support._build_inputs()
    _require(payload.get("config") == _config_snapshot(config), "config changed after preflight")
    _require(payload.get("source_bindings") == _source_bindings(config), "source binding changed")
    current = [_cell_binding(_load_cell_inputs(cell_id)) for cell_id in CELLS]
    _require(payload.get("cells") == current, "cell/tape/sidecar binding changed")


def _actor_context(env: Any, uid: int, uids: Sequence[str], world: np.ndarray) -> dict[str, Any]:
    return base._actor_context(env, uid, uids, world)


def _predicate_state(
    context: Mapping[str, Any], *, decision: Any, last_decision: Any, actual_rank: Any,
) -> dict[str, Any]:
    missing_actor = bool(context.get("missing_actor"))
    phase = context.get("phase") if not missing_actor else None
    soc_raw = context.get("soc") if not missing_actor else None
    try:
        soc = float(soc_raw)
    except (TypeError, ValueError):
        soc = None
    soc_finite: bool | None = None if soc is None else bool(math.isfinite(soc))
    rank: int | None
    try:
        rank = int(actual_rank)
    except (TypeError, ValueError):
        rank = None
    predicates: dict[str, bool | None] = {
        "current_info_approved": None if decision is None else str(decision) == "approved",
        "current_last_release_approved": None if last_decision is None else str(last_decision) == "approved",
        "actor_mask_true": None if missing_actor else bool(context.get("actor_mask")),
        "ring_phase": None if phase is None else phase in {"ring_outer", "ring_middle", "ring_inner"},
        "soc_finite": soc_finite,
        "soc_gt_0_4": None if soc_finite is not True else soc > 0.4,
        "retry_free": None if missing_actor else bool(context.get("retry_free")),
        "no_committed_plan": None if missing_actor else not bool(context.get("committed_plan")),
    }
    unknown = sorted(key for key, value in predicates.items() if value is None)
    false = sorted(key for key, value in predicates.items() if value is False)
    normal = not unknown and not false
    auto_emergency = bool(soc_finite is True and soc is not None and soc <= 0.4)
    reasons = list(unknown) + list(false)
    if auto_emergency:
        reasons.append("soc_le_0_4_autoemergency")
    return {
        "predicates": predicates, "unknown_predicates": unknown,
        "false_predicates": false, "reasons": sorted(set(reasons)),
        "normal_eligible": normal, "autoemergency_excluded": auto_emergency,
        "phase": phase, "soc": soc, "actual_rank": rank,
        "rank_class": None if rank is None else ("first3" if rank <= PUBLIC_K else "late_prefix_gt3"),
    }


def _safe_context(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(value) for key, value in context.items()
        if key not in {"world", "full_world", "index"}
    }


def _actual_telemetry(info: Mapping[str, Any], env: Any, uid: int) -> Mapping[str, Any] | None:
    info_telemetry = info.get("release_intent_telemetry")
    telemetry = _mapping_value(info_telemetry, uid, None) if isinstance(info_telemetry, Mapping) else None
    if isinstance(telemetry, Mapping):
        return telemetry
    last = _mapping_value(getattr(env, "last_release_telemetry", {}), uid, None)
    return last if isinstance(last, Mapping) else None


def _event_comparison(
    candidate: Mapping[str, Any], context: Mapping[str, Any], row: Mapping[str, Any],
    before_policy: Any, after_policy: Any,
) -> dict[str, Any]:
    index = int(context["index"])
    world = np.asarray(context["full_world"], dtype=np.float32)
    before_dist = base._distribution_snapshot(before_policy, world)
    after_dist = base._distribution_snapshot(after_policy, world)
    before_q = np.asarray(before_dist["proposal_q"], dtype=np.float64)[index]
    after_q = np.asarray(after_dist["proposal_q"], dtype=np.float64)[index]
    _require(before_q.shape == (PROPOSAL_COUNT,) and after_q.shape == (PROPOSAL_COUNT,),
             "proposal q does not have 25 entries")
    q_equal = bool(np.array_equal(before_q, after_q))
    _require(q_equal, f"proposal q mismatch at {candidate['registration_key']}")
    before_logits = np.asarray(before_dist["proposal_logits"], dtype=np.float64)[index]
    source_logits = np.asarray(candidate["sidecar_source_logits"], dtype=np.float64)
    _require(source_logits.shape == (PROPOSAL_COUNT,), "source logits are not 25-way")
    source_logits_match = bool(np.allclose(before_logits, source_logits, atol=1e-4, rtol=1e-6))
    _require(source_logits_match, f"BEFORE source logits mismatch at {candidate['registration_key']}")
    gate_p_before = float(np.asarray(before_dist["gate_probs"], dtype=np.float64)[index, 1])
    gate_p_after = float(np.asarray(after_dist["gate_probs"], dtype=np.float64)[index, 1])
    _require(math.isfinite(gate_p_before) and 0.0 <= gate_p_before <= 1.0,
             f"BEFORE gate probability is invalid at {candidate['registration_key']}")
    _require(math.isfinite(gate_p_after) and 0.0 <= gate_p_after <= 1.0,
             f"AFTER gate probability is invalid at {candidate['registration_key']}")
    stored_gate = _mapping_value(row.get("gate_logprob"), int(candidate["uid"]), None)
    source_lp = {
        "stored_available": stored_gate is not None,
        "stored_value": None if stored_gate is None else float(stored_gate),
        "before_gate_logprob": float(math.log(max(gate_p_before, np.finfo(float).tiny))),
        "match": None,
        "checked_policy": "BEFORE",
    }
    if stored_gate is not None and math.isfinite(float(stored_gate)):
        source_lp["match"] = bool(abs(float(stored_gate) - source_lp["before_gate_logprob"]) <= 1e-4)
        _require(source_lp["match"], f"BEFORE source gate logprob mismatch at {candidate['registration_key']}")
    return {
        "before_gate_p": gate_p_before, "after_gate_p": gate_p_after,
        "delta_p": gate_p_after - gate_p_before,
        "q_before": before_q.tolist(), "q_after": after_q.tolist(),
        "source_proposal_logits": source_logits.tolist(),
        "source_logits_match_before": source_logits_match,
        "q_bitwise_equal": q_equal, "source_gate_logprob_check": source_lp,
        "D": None, "local_m": None, "complete_value_known": False,
    }


def _replay_events(
    item: Mapping[str, Any], before_policy: Any, after_policy: Any,
    config: Any, output: Path,
) -> dict[str, Any]:
    env_rng_before = _rng_snapshot()
    env = base.frozen._CapturedTerminalEnv(config)
    env_rng_after = _rng_snapshot()
    env_constructor_rng_consumed = not _rng_equal(env_rng_before, env_rng_after)
    _restore_rng(env_rng_before)
    original_step = env.step
    rows = list(item["rows"])
    row_by_step = {int(row["step"]): row for row in rows}
    by_step: dict[int, list[dict[str, Any]]] = {}
    for candidate in item["potential"]:
        by_step.setdefault(int(candidate["step"]), []).append(dict(candidate))
    events: list[dict[str, Any]] = []
    current_step: int | None = None
    exclusion_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    before_digest = rank_eval._model_digest(before_policy)
    after_digest = rank_eval._model_digest(after_policy)

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal current_step
        step = int(getattr(env, "step_count", -1))
        current_step = step
        source_row = row_by_step.get(step)
        _require(source_row is not None, f"replay step {step} missing from tape")
        observations = env._observations()
        _require(_digest({key: value for key, value in sorted(observations.items())}) == source_row.get("observation_hash"),
                 f"observation hash mismatch at step {step}")
        uids, world = base.matched._world_from_observations(observations)
        pre = []
        for candidate in by_step.get(step, []):
            context = _actor_context(env, int(candidate["uid"]), uids, world)
            context["full_world"] = np.asarray(world, dtype=np.float32).copy()
            pre.append((candidate, context))
        result = original_step(*args, **kwargs)
        info = result[4] if isinstance(result, tuple) and len(result) >= 5 and isinstance(result[4], Mapping) else {}
        decisions = info.get("release_decisions") if isinstance(info, Mapping) else None
        for candidate, context in pre:
            uid = int(candidate["uid"])
            decision = _mapping_value(decisions, uid, None)
            last_decision = _mapping_value(getattr(env, "last_release_decisions", {}), uid, None)
            telemetry = _actual_telemetry(info, env, uid)
            actual_rank = telemetry.get("runtime_shortlist_rank") if telemetry is not None else None
            try:
                if actual_rank is not None and int(actual_rank) <= 0:
                    actual_rank = None
            except (TypeError, ValueError):
                actual_rank = None
            state = _predicate_state(context, decision=decision, last_decision=last_decision, actual_rank=actual_rank)
            candidate_rank = candidate.get("rank")
            actual_rank_value = state.get("actual_rank")
            if actual_rank_value is not None:
                _require(1 <= int(actual_rank_value) <= PROPOSAL_COUNT,
                         f"current shortlist rank out of range at {candidate['registration_key']}")
            rank_binding_match = None
            if candidate_rank is not None and actual_rank_value is not None:
                rank_binding_match = int(candidate_rank) == int(actual_rank_value)
                _require(rank_binding_match, f"tape/current shortlist rank mismatch at {candidate['registration_key']}")
            state["tape_rank"] = candidate_rank
            state["rank_binding_match"] = rank_binding_match
            event: dict[str, Any] = {
                "registration_key": str(candidate["registration_key"]),
                "step": int(candidate["step"]), "uid": uid,
                "candidate_rank": None if candidate.get("rank") is None else int(candidate["rank"]),
                "runtime_used": _jsonable(candidate.get("runtime_used")),
                "requested_gate": int(candidate["requested_gate"]),
                "requested_proposal": _jsonable(candidate.get("requested_proposal")),
                "pre_observation_hash": source_row.get("observation_hash"),
                "pre_world_88d": np.asarray(context.get("world"), dtype=np.float32).tolist()
                if "world" in context else None,
                "pre_world_matrix_88d": np.asarray(context.get("full_world"), dtype=np.float32).tolist()
                if "full_world" in context else None,
                "context": _safe_context(context),
                "actual": {
                    "info_release_decision": _jsonable(decision),
                    "env_last_release_decision": _jsonable(last_decision),
                    "runtime_shortlist_rank": _jsonable(actual_rank),
                    "release_intent_telemetry": _jsonable(telemetry),
                },
                "predicate_state": state,
                "status": "NORMAL_ELIGIBLE" if state["normal_eligible"] else (
                    "EXCLUDED_AUTOEMERGENCY" if state["autoemergency_excluded"] else "EXCLUDED"
                ),
            }
            for reason in state["reasons"]:
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            phase = state.get("phase") or "unknown"
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            if state["normal_eligible"]:
                event["comparison"] = _event_comparison(candidate, context, source_row, before_policy, after_policy)
            events.append(event)
        if step == 1 or (step + 1) % 50 == 0:
            _write_json(output / "progress.json", {
                "stage": STAGE, "phase": "arm", "status": "RUNNING",
                "cell_id": int(item["cell_id"]), "step": step,
                "potential_candidates_seen": len(item["potential"]),
                "events_recorded": len(events), "normal_events": sum(1 for row in events if row["status"] == "NORMAL_ELIGIBLE"),
            })
        return result

    replay: Mapping[str, Any] = {}
    try:
        env.step = observed_step  # type: ignore[method-assign]
        with base._controller_scope(env, item["sidecar"]):
            replay = SafetyTraceRecorder.replay_action_tape(env, item["tape_path"])
    except BaseException as exc:
        partial = {
            "stage": STAGE, "phase": "arm", "status": "PARTIAL",
            "cell_id": int(item["cell_id"]), "arm": "before_prefix",
            "current_step": current_step, "events_recorded": len(events),
            "normal_event_count": sum(row.get("status") == "NORMAL_ELIGIBLE" for row in events),
            "events": events, "exclusion_counts": exclusion_counts,
            "phase_counts": phase_counts,
            "error": {"error_type": type(exc).__name__, "message": str(exc)},
        }
        try:
            _write_json(output / f"cell_{int(item['cell_id']):02d}_partial.json", partial)
        except BaseException:
            pass
        raise
    finally:
        env.step = original_step  # type: ignore[method-assign]
    _require(isinstance(replay, Mapping), "replay result is not a mapping")
    _require(replay.get("exact") is True and int(replay.get("steps_replayed", -1)) == len(rows),
             "action tape replay was not exact")
    _require(rank_eval._model_digest(before_policy) == before_digest, "BEFORE model digest changed during replay")
    _require(rank_eval._model_digest(after_policy) == after_digest, "AFTER model digest changed during replay")
    return {
        "replay": {key: value for key, value in replay.items() if key != "steps"},
        "events": events, "potential_count": len(item["potential"]),
        "normal_event_count": sum(1 for row in events if row["status"] == "NORMAL_ELIGIBLE"),
        "exclusion_counts": exclusion_counts, "phase_counts": phase_counts,
        "model_digest_before": before_digest, "model_digest_after": after_digest,
        "controller_installed_and_restored": True,
        "environment_constructor_rng_consumed": env_constructor_rng_consumed,
    }


def _run_cell(item: Mapping[str, Any], config: Any, output: Path) -> dict[str, Any]:
    caller_before = _rng_snapshot()
    caller_before_restore: Any = None
    result: dict[str, Any] | None = None
    before_policy = after_policy = None
    before_load_rng = after_load_rng = False
    try:
        before_policy, _, before_load_rng = base._load_policy_restoring_rng(_resolve(support.BEFORE_CHECKPOINT), config)
        after_policy, _, after_load_rng = base._load_policy_restoring_rng(_resolve(support.AFTER_CHECKPOINT), config)
        replay = _replay_events(item, before_policy, after_policy, config, output)
        caller_before_restore = _rng_snapshot()
        _require(_rng_equal(caller_before, caller_before_restore), "caller RNG changed before restoration")
        result = {
            "stage": STAGE, "phase": "arm",
            "status": "COMPLETED", "cell_id": int(item["cell_id"]), "arm": "before_prefix",
            "cell_binding": _cell_binding(item), "replay": replay,
            "comparison": {"D": None, "local_m": None, "complete_value_known": False},
            "caller_rng_unchanged_before_restore": _rng_equal(caller_before, caller_before_restore),
            "model_load_rng_consumed": {"before": bool(before_load_rng), "after": bool(after_load_rng)},
        }
    finally:
        _restore_rng(caller_before)
        caller_after_restore = _rng_snapshot()
        caller_restored = _rng_equal(caller_before, caller_after_restore)
        if result is not None:
            result["caller_rng_unchanged_after_restore"] = caller_restored
        _require(caller_restored, "caller RNG was not restored")
    _require(result is not None, "cell replay produced no result")
    return result


def _validate_pilot_reuse(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    pilot = _read_json(output / "pilot.json")
    _require(pilot.get("status") == "PILOT_OK" and pilot.get("cell_ids") == [1], "pilot is not accepted")
    _require(pilot.get("preflight_sha256") == _sha256(output / "preflight.json"), "pilot preflight binding changed")
    pilot_cell = output / "cells" / "cell_01.json"
    _require(pilot_cell.is_file(), "pilot cell artifact missing")
    _require(pilot.get("cell_sha256") == _sha256(pilot_cell), "pilot cell binding changed")
    stored = _read_json(pilot_cell)
    _require(stored.get("status") == "COMPLETED", "pilot cell is not completed")
    _require(preflight.get("status") == "PREFLIGHT_OK", "preflight is not accepted")
    return pilot


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
    cells_dir.mkdir(parents=True, exist_ok=True)
    if phase == "pilot":
        result = _run_cell(cells[0], config, output)
        _write_json(cells_dir / "cell_01.json", result, refuse_existing=True)
        summary = {
            "stage": STAGE, "phase": "pilot", "status": "PILOT_OK", "cell_ids": [1],
            "cells": [{"cell_id": 1, "status": result["status"], "normal_event_count": result["replay"]["normal_event_count"]}],
            "preflight_sha256": _sha256(preflight_path), "cell_sha256": _sha256(cells_dir / "cell_01.json"),
        }
        _write_json(output / "pilot.json", summary, refuse_existing=True)
        return summary
    _require(phase == "full", f"unknown phase: {phase}")
    pilot = _validate_pilot_reuse(output, preflight)
    results = [_read_json(cells_dir / "cell_01.json")]
    for item in cells[1:]:
        result = _run_cell(item, config, output)
        _write_json(cells_dir / f"cell_{int(item['cell_id']):02d}.json", result, refuse_existing=True)
        results.append(result)
    summary = {
        "stage": STAGE, "phase": "full", "status": "FULL_OK", "cell_ids": CELLS,
        "cells": [{"cell_id": int(row["cell_id"]), "status": row["status"],
                   "normal_event_count": row["replay"]["normal_event_count"]} for row in results],
        "preflight_sha256": _sha256(preflight_path), "pilot_sha256": _sha256(output / "pilot.json"),
        "D": None, "local_m": None,
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
        error = {"stage": STAGE, "phase": args.phase, "status": "ERROR",
                 "error_type": type(exc).__name__, "message": str(exc)}
        try:
            _write_json(output / "error.json", error, refuse_existing=True)
            _write_json(output / "partial.json", {"stage": STAGE, "phase": args.phase, "status": "PARTIAL",
                                                    "error": error}, refuse_existing=True)
        except BaseException:
            pass
        print(json.dumps(_jsonable(error), sort_keys=True))
        return 2
    print(json.dumps(_jsonable(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
