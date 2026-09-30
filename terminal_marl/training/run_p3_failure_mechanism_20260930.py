"""Bounded replay and mechanism diagnostic for the refresh energy failure.

The phases are deliberately separate.  ``preflight`` reads the fixed record,
tape, checkpoint metadata, and neutral configuration without creating an
environment.  ``replay`` performs one exact source-tape replay and registers
three deterministic eligible states.  ``diagnose`` replays the tape again,
deep-copies only those states, and runs the existing paired WAIT/all RELEASE
diagnostic.  Optional ``blockers`` uses the registered states for one-step
telemetry and a separate exact replay.  This module does not train, sample an
optimizer, use a teacher, or consume held-out data.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.training import diagnose_p3_ppo_credit_signal as credit
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import paired_gate_diagnostic
from terminal_marl.training import run_p3_followup_diagnostics as runner_utils


STAGE = "P3_FAILURE_MECHANISM_20260930_V1"
PPO_ALLOWED = False
ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_critic_refresh_comparison_20260929/full/refresh/training"
)
RECORD = ROOT / "batch_02/training/episode_24/record.json"
TAPE = ROOT / "batch_02/training/episode_24/rollout/action_tapes/safety_trace_seed_881148_step_983.json"
SOURCE_CHECKPOINT = ROOT / "batch_01/policy.pt"
AFTER_CHECKPOINT = ROOT / "batch_02/policy.pt"
AFTER_SUMMARY = ROOT / "batch_02/summary.json"
FAILURE_OUTPUT_DIR = Path("/home/user/training_outputs/terminal/p3_followup_20260930/failure_mechanism")
BLOCKERS_OUTPUT_DIR = Path("/home/user/training_outputs/terminal/p3_followup_20260930/failure_blockers")
SOURCE_SHA256 = "1637a15d73184617a1e386cd29cbcdbe29c0679b1414f5fb413ec7a9fbef3805"
TAPE_SHA256 = "7bf68b8159e9d6038744735618067f0344a5a64218798bea2b93260c955212e3"
RECORD_SHA256 = "34e3bff462dba2e28571804fbcb530961d805740aa1e6b90fc39508bed3d60e1"
AFTER_SHA256 = "370344b6523f250bd6816bbfd6ef51bd065c4aea1352b21d8910adc1275e942c"
SOURCE_MODEL_DIGEST = "861cdc5f86e5e782a92b9a7cff9874836d3c3069397b7bcf8dddfbe0b8283e5c"
TARGET_UID = 57
TARGET_ARRIVAL_ID = 14
TARGET_FAILURE_STEP = 892
EXPECTED_STEPS = 984
EXPECTED_REQUESTS = 24
CONTINUATION_SEEDS = (9303201, 9303202, 9303203, 9303204)
BRANCH_MAX_STEPS = 1800
QUANTILES = (0.25, 0.50, 0.75)
REGISTERED_SELECTED_STEPS = (280, 445, 726)


class FailureMechanismError(ValueError):
    """Raised when a fixed failure-mechanism input/output contract is unmet."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FailureMechanismError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"JSON object required: {path}")
    return dict(value)


def _resolve(path: Path) -> Path:
    return Path(neutral_eval._synced_path(path))


def _hash_artifact(path: Path, expected: str | None = None) -> dict[str, Any]:
    item = runner_utils._artifact(path, expected_sha256=expected)
    return dict(item)


def _failure_row(record: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = record.get("lifecycle", {})
    rows = lifecycle.get("rows", []) if isinstance(lifecycle, Mapping) else []
    _require(isinstance(rows, list), "record lifecycle rows are missing")
    matches = [row for row in rows if int(row.get("arrival_id", -1)) == TARGET_ARRIVAL_ID]
    _require(len(matches) == 1, "target failure arrival row is not unique")
    row = dict(matches[0])
    _require(int(row.get("uid", -1)) == TARGET_UID, "target failure UID mismatch")
    _require(str(row.get("outcome")) == "failed_energy", "target failure outcome mismatch")
    _require(int(float(row.get("terminal_time", -1))) == TARGET_FAILURE_STEP,
             "target failure terminal step mismatch")
    return row


def _strict_config() -> tuple[Any, dict[str, Any]]:
    # Keep this exact source of the neutral dense24 configuration in the
    # protocol; it prevents accidental use of a legacy or warmup config.
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    _require("dense24" in neutral_configs, "neutral dense24 config is missing")
    config = neutral_configs["dense24"]
    _require(int(getattr(config, "total_requested_uavs", -1)) == EXPECTED_REQUESTS,
             "dense24 config request count mismatch")
    _require(getattr(config, "task_priority_enabled", False) is False,
             "dense24 config is not task-neutral")
    _require(math.isclose(float(getattr(config, "low_level_priority_time_weight", 1.0)), 0.0,
                          rel_tol=0.0, abs_tol=1e-12),
             "dense24 priority weight is not zero")
    return config, runner_utils._config_snapshot(config)


def _load_fixed_inputs() -> dict[str, Any]:
    record_path = _resolve(RECORD)
    tape_path = _resolve(TAPE)
    source_path = _resolve(SOURCE_CHECKPOINT)
    after_path = _resolve(AFTER_CHECKPOINT)
    summary_path = _resolve(AFTER_SUMMARY)
    record = _read_json(record_path)
    _require(int(record.get("seed", -1)) == 881148, "refresh env seed mismatch")
    _require(int(record.get("policy_seed", -1)) == 881248, "refresh policy seed mismatch")
    _require(str(record.get("scenario")) == "dense24", "refresh scenario mismatch")
    _require(int(record.get("requests", -1)) == EXPECTED_REQUESTS, "refresh request count mismatch")
    _require(int(record.get("rollout_summary", {}).get("steps", -1)) == EXPECTED_STEPS,
             "refresh rollout step count mismatch")
    _require(str(record.get("checkpoint_sha256", "")).lower() == SOURCE_SHA256,
             "record producer checkpoint SHA mismatch")
    failure = _failure_row(record)
    tape_payload, tape_rows = credit._load_tape(tape_path)
    _require(len(tape_rows) == EXPECTED_STEPS, "refresh tape step count mismatch")
    _require(str(record.get("action_tape", "")).endswith("safety_trace_seed_881148_step_983.json"),
             "record action tape binding mismatch")
    summary = _read_json(summary_path)
    roundtrip = summary.get("checkpoint_roundtrip")
    _require(isinstance(roundtrip, Mapping), "after summary checkpoint_roundtrip is missing")
    _require(str(roundtrip.get("checkpoint_sha256", "")).lower() == AFTER_SHA256,
             "after summary checkpoint SHA mismatch")
    after_digest = str(roundtrip.get("model_digest", ""))
    _require(after_digest, "after summary model digest is missing")
    return {
        "record_path": record_path,
        "record": record,
        "failure": failure,
        "tape_path": tape_path,
        "tape_payload": tape_payload,
        "tape_rows": tape_rows,
        "source_path": source_path,
        "after_path": after_path,
        "after_summary_path": summary_path,
        "after_digest": after_digest,
    }


def _source_policy(config: Any, source_path: Path) -> tuple[Any, str, dict[str, Any]]:
    _require(source_path == _resolve(SOURCE_CHECKPOINT), "source checkpoint path is not batch_01")
    policy, payload = rank_eval._load_policy(source_path, config)
    digest = str(rank_eval._model_digest(policy))
    _require(digest == SOURCE_MODEL_DIGEST, "source full-checkpoint model digest mismatch")
    return policy, digest, dict(payload) if isinstance(payload, Mapping) else {}


def _public_candidate(capture: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    context = capture.get("state_context", {})
    marker = context.get("failure_target", {}) if isinstance(context, Mapping) else {}
    _require(isinstance(marker, Mapping), "failure observer marker is missing")
    actions, shortlists = runner_utils._row_inputs(row)
    _require(str(TARGET_UID) in actions, "target action is missing from eligible tape row")
    return {
        "episode": 24,
        "step": int(capture["step"]),
        "uid": TARGET_UID,
        "arrival_id": TARGET_ARRIVAL_ID,
        "phase": marker.get("phase"),
        "soc": marker.get("soc"),
        "energy_remaining": marker.get("energy_remaining"),
        "world_index": int(marker.get("world_index", -1)),
        "observation_hash": str(row.get("observation_hash", "")),
        "source_action_tape_index": row.get("action_tape_index"),
        "first_actions": runner_utils._jsonable(actions),
        "runtime_proposal_shortlists": runner_utils._jsonable(shortlists),
        "target_actor_mask": True,
        "selection_rule": "sorted_episode_step_uid_quantile_floor",
    }


def _eligible_points(captures: Sequence[Mapping[str, Any]], tape_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for capture, row in zip(captures, tape_rows, strict=True):
        context = capture.get("state_context", {})
        marker = context.get("failure_target", {}) if isinstance(context, Mapping) else {}
        if not isinstance(marker, Mapping):
            continue
        if not bool(marker.get("active")) or not bool(marker.get("actor_mask")):
            continue
        if int(marker.get("arrival_id", -1)) != TARGET_ARRIVAL_ID:
            continue
        if int(marker.get("uid", -1)) != TARGET_UID:
            continue
        points.append(_public_candidate(capture, row))
    ordered = sorted(points, key=lambda item: (int(item["episode"]), int(item["step"]), int(item["uid"])))
    keys = [(int(item["episode"]), int(item["step"]), int(item["uid"])) for item in ordered]
    _require(len(keys) == len(set(keys)), "eligible failure occurrences are not unique")
    return ordered


def _select_points(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = [dict(item) for item in sorted(points, key=lambda item: (int(item["episode"]), int(item["step"]), int(item["uid"]))) ]
    _require(len(ordered) >= 3, f"eligible failure occurrences are insufficient: {len(ordered)}")
    selected_indices = [int(math.floor((len(ordered) - 1) * q)) for q in QUANTILES]
    _require(len(set(selected_indices)) == 3, "registered quantile indices are not unique")
    selected = []
    for order, (quantile, index) in enumerate(zip(QUANTILES, selected_indices, strict=True), start=1):
        item = dict(ordered[index])
        item.update({"selection_order": order, "eligible_index": index, "quantile": quantile})
        selected.append(item)
    return {
        "selection_rule": "sorted_episode_step_uid_quantile_floor",
        "quantiles": list(QUANTILES),
        "eligible_total": len(ordered),
        "selected": selected,
    }


def _validate_lifecycle(lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    _require(bool(lifecycle.get("terminal")), "source replay did not terminate")
    _require(bool(lifecycle.get("all_requests_observed")), "source replay requests incomplete")
    _require(bool(lifecycle.get("ledger_alignment")), "source replay ledger is not aligned")
    request_rows = lifecycle.get("request_rows", [])
    _require(isinstance(request_rows, list) and len(request_rows) == EXPECTED_REQUESTS,
             "source replay must retain all 24 request inventory rows")
    request_ids = {int(row.get("arrival_id", -1)) for row in request_rows}
    _require(request_ids == set(range(EXPECTED_REQUESTS)),
             "source replay request inventory IDs are incomplete")
    rows = lifecycle.get("rows", [])
    _require(isinstance(rows, list) and len(rows) == EXPECTED_REQUESTS,
             "source replay must retain all 24 terminal rows")
    terminal_ids = {int(row.get("arrival_id", -1)) for row in rows}
    _require(terminal_ids == set(range(EXPECTED_REQUESTS)),
             "source replay terminal IDs are incomplete")
    target = [row for row in rows if int(row.get("arrival_id", -1)) == TARGET_ARRIVAL_ID]
    _require(len(target) == 1 and str(target[0].get("outcome")) == "failed_energy",
             "source replay natural failed_energy row was not retained")
    _require(int(float(target[0].get("terminal_time", -1))) == TARGET_FAILURE_STEP,
             "source replay failure terminal step mismatch")
    return {
        "request_count": len(request_rows),
        "request_rows": [dict(row) for row in request_rows],
        "terminal_rows": [dict(row) for row in rows],
        "failed_energy_rows": [dict(row) for row in rows if str(row.get("outcome")) == "failed_energy"],
        "all_rows_retained": True,
    }


def _failure_observer(policy: Any):
    def observe(env: Any) -> dict[str, Any]:
        light = runner_utils._light_state_observer(env)
        context = paired_gate_diagnostic._details_context(policy, env)
        target_index = next((i for i, uid in enumerate(context["uids"]) if int(uid) == TARGET_UID), None)
        actor = light.get("actors", {}).get(str(TARGET_UID), {})
        marker = {
            "uid": TARGET_UID,
            "arrival_id": actor.get("arrival_id"),
            "active": target_index is not None and str(TARGET_UID) in light.get("actors", {}),
            "actor_mask": False if target_index is None else bool(context["actor_mask"][target_index]),
            "phase": actor.get("phase"),
            "soc": actor.get("soc"),
            "energy_remaining": actor.get("energy_remaining"),
            "world_index": None if target_index is None else int(target_index),
        }
        light["failure_target"] = marker
        return light
    return observe


def _progress(output: Path, phase: str, status: str, processed: int = 0, **extra: Any) -> None:
    payload = {
        "stage": STAGE,
        "phase": phase,
        "status": status,
        "processed": int(processed),
        "expected": EXPECTED_STEPS,
        "processed_steps": int(processed),
        "expected_steps": EXPECTED_STEPS,
        **extra,
    }
    runner_utils._atomic_write_json(output / "progress.json", payload)


def _write_failed_phase(output: Path, phase: str, error: BaseException) -> None:
    report = {
        "stage": STAGE,
        "phase": phase,
        "status": "FAILED",
        "PPO_ALLOWED": PPO_ALLOWED,
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    runner_utils._atomic_write_json(output / "report.json", report)
    _progress(output, phase, "FAILED", error=report["error"])


def _build_preflight() -> dict[str, Any]:
    config, config_snapshot = _strict_config()
    inputs = _load_fixed_inputs()
    source_policy, source_digest, source_payload = _source_policy(config, inputs["source_path"])
    del source_policy
    return {
        "stage": STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": PPO_ALLOWED,
        "runtime_versions": runner_utils._runtime_versions(),
        "config_snapshot": config_snapshot,
        "loaded_sourcefiles": runner_utils._loaded_code_manifest(),
        "source": {
            "checkpoint": _hash_artifact(inputs["source_path"], SOURCE_SHA256),
            "model_digest": source_digest,
            "loader": "evaluate_p3_runtime_rank_repair._load_policy(full_checkpoint)",
            "payload_keys": sorted(source_payload),
        },
        "after_alignment": {
            "checkpoint": _hash_artifact(inputs["after_path"], AFTER_SHA256),
            "summary": _hash_artifact(inputs["after_summary_path"]),
            "model_digest_from_summary": inputs["after_digest"],
            "causal_scope": "alignment_only; after batch_02 update is not claimed to cause the earlier batch_02 rollout failure",
        },
        "failure_input": {
            "record": _hash_artifact(inputs["record_path"], RECORD_SHA256),
            "tape": _hash_artifact(inputs["tape_path"], TAPE_SHA256),
            "env_seed": 881148,
            "policy_seed": 881248,
            "steps": EXPECTED_STEPS,
            "requests": EXPECTED_REQUESTS,
            "failure": inputs["failure"],
        },
        "scope": {
            "warmup_loaded": False,
            "after_continuation_used": False,
            "optimizer_calls": 0,
            "teacher_used": False,
            "heldout_used": False,
            "population_estimate": False,
        },
    }


def _load_preflight(output: Path) -> dict[str, Any]:
    value = _read_json(output / "preflight.json")
    _require(value.get("status") == "PREFLIGHT_OK", "preflight artifact is not complete")
    return value


def _run_replay(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    config, config_snapshot = _strict_config()
    _require(dict(preflight.get("config_snapshot", {})) == config_snapshot, "resolved config differs from preflight")
    inputs = _load_fixed_inputs()
    policy, before_digest, _payload = _source_policy(config, inputs["source_path"])
    _progress(output, "replay", "RUNNING", 0)
    replay = credit._capture_replay(
        policy, config, inputs["tape_path"], inputs["tape_rows"],
        progress_callback=lambda count: _progress(output, "replay", "RUNNING", int(count)),
        state_observer=_failure_observer(policy),
    )
    runner_utils._atomic_write_json(
        output / "replay_raw_lifecycle.json",
        {"stage": STAGE, "phase": "replay", "lifecycle": replay.get("lifecycle", {})},
    )
    _require(bool(replay.get("replay", {}).get("exact")), "refresh replay was not exact")
    _require(int(replay.get("replay", {}).get("steps_replayed", -1)) == EXPECTED_STEPS,
             "refresh replay step count mismatch")
    lifecycle = _validate_lifecycle(replay.get("lifecycle", {}))
    points = _eligible_points(replay["captures"], inputs["tape_rows"])
    selection = _select_points(points)
    after_digest = str(rank_eval._model_digest(policy))
    _require(after_digest == before_digest, "source model changed during replay")
    report = {
        "stage": STAGE,
        "phase": "replay",
        "status": "COMPLETED",
        "PPO_ALLOWED": PPO_ALLOWED,
        "source_model_digest_before": before_digest,
        "source_model_digest_after": after_digest,
        "source_model_unchanged": True,
        "replay": {"exact": True, "steps": EXPECTED_STEPS, "requests": EXPECTED_REQUESTS},
        "lifecycle": lifecycle,
        "eligible": {"count": len(points), "failure_target_only": True},
        "scope": {"natural_failure_retained": True, "all_24_requests_retained": True,
                  "optimizer_calls": 0, "teacher_used": False, "heldout_used": False},
    }
    runner_utils._atomic_write_json(output / "selected.json", {"stage": STAGE, "phase": "replay", "selection": selection})
    runner_utils._atomic_write_json(output / "replay.json", report)
    _progress(output, "replay", "COMPLETED", EXPECTED_STEPS, selected=3)
    return report


def _capture_selected_states(policy: Any, config: Any, inputs: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], output: Path) -> dict[int, dict[str, Any]]:
    wanted = {int(item["step"]): dict(item) for item in selected}
    states: dict[int, dict[str, Any]] = {}

    def observe(env: Any) -> dict[str, Any]:
        step = int(getattr(env, "step_count", -1))
        if step in wanted:
            snapshot = copy.deepcopy(env)
            paired_gate_diagnostic.restore_diagnostic_snapshot(snapshot)
            states[step] = {"snapshot": snapshot, "candidate": wanted[step]}
        return _failure_observer(policy)(env)

    replay = credit._capture_replay(
        policy, config, inputs["tape_path"], inputs["tape_rows"],
        progress_callback=lambda count: _progress(output, "selected_replay", "RUNNING", int(count)),
        state_observer=observe,
    )
    _require(bool(replay.get("replay", {}).get("exact")), "selected-state replay was not exact")
    _require(len(states) == len(selected), "selected state deep-copy count mismatch")
    rows_by_step = {int(row.get("step", index)): row for index, row in enumerate(inputs["tape_rows"])}
    captures_by_step = {int(row["step"]): row for row in replay["captures"]}
    for step, state in states.items():
        row = rows_by_step[step]
        actions, shortlists = runner_utils._row_inputs(row)
        capture = captures_by_step[step]
        uid_index = [int(uid) for uid in capture["uids"]].index(TARGET_UID)
        state.update({
            "first_actions": actions,
            "first_shortlists": shortlists,
            "target_world": np.asarray(capture["world"], dtype=np.float32).copy(),
            "target_world_row": int(uid_index),
            "source_observation_hash": str(row.get("observation_hash", "")),
        })
    return states


def _zone_context(env: Any, uav: Any | None = None) -> list[dict[str, Any]]:
    now = float(getattr(env, "now", 0.0))
    mission = getattr(uav, "mission_type", None) if uav is not None else None
    rows: list[dict[str, Any]] = []
    for zone in list(getattr(getattr(env, "lz_manager", None), "zones", ()) or ()):
        compatible = sorted(
            str(getattr(item, "value", item)) for item in getattr(zone, "compatible_missions", ())
        )
        try:
            eta = float(env.lz_manager.earliest_release_time(zone.lz_id, now))
        except (AttributeError, TypeError, ValueError):
            eta = None
        rows.append({
            "lz_id": int(zone.lz_id),
            "status": str(getattr(getattr(zone, "status", None), "value", getattr(zone, "status", None))),
            "reserved_by": getattr(zone, "reserved_by", None),
            "occupied_by": getattr(zone, "occupied_by", None),
            "service_remaining": float(getattr(zone, "service_remaining", 0.0)),
            "expected_release_at": getattr(zone, "expected_release_at", None),
            "eta": eta,
            "compatible_missions": compatible,
            "compatible_with_target": mission is not None and mission in getattr(zone, "compatible_missions", ()),
        })
    return rows


def _target_blocker_context(env: Any) -> dict[str, Any]:
    actor = getattr(env, "active", {}).get(TARGET_UID)
    if actor is None:
        return {"uid": TARGET_UID, "active": False}
    config = getattr(env, "config", None)
    try:
        available = list(env.lz_manager.rank_available_zones(actor, float(env.now)))
    except (AttributeError, TypeError, ValueError):
        available = []
    return {
        "uid": TARGET_UID,
        "arrival_id": int(getattr(actor, "arrival_id", -1)),
        "active": True,
        "soc": float(getattr(actor, "soc", 0.0)),
        "energy_remaining": float(getattr(actor, "energy_remaining", 0.0)),
        "energy_consumed": float(getattr(actor, "energy_consumed", 0.0)),
        "phase": str(getattr(getattr(actor, "phase", None), "value", getattr(actor, "phase", None))),
        "ring_role": str(getattr(getattr(actor, "ring_role", None), "value", getattr(actor, "ring_role", None))),
        "safety_retry_after": float(getattr(actor, "safety_retry_after", 0.0)),
        "emergency_retry_after": float(getattr(actor, "emergency_retry_after", 0.0)),
        "retry_after": float(
            getattr(actor, "emergency_retry_after", 0.0)
            if bool(getattr(config, "emergency_priority_guard_enabled", False))
            else getattr(actor, "safety_retry_after", 0.0)
        ),
        "committed": bool(getattr(actor, "committed_plan", None) is not None),
        "emergency_landing": bool(getattr(actor, "emergency_landing", False)),
        "available_compatible_lz_ids": [int(value) for value in available],
        "landing_zones": _zone_context(env, actor),
    }


def _blocker_context(env: Any) -> dict[str, Any]:
    return {
        "light_state": runner_utils._light_state_observer(env),
        "target": _target_blocker_context(env),
    }


def _first_step_payload(
    *,
    proposal: int | None,
    proposal_probability: float | None = None,
    env: Any,
    before_target: Mapping[str, Any],
    after_target: Mapping[str, Any],
    before_telemetry: Mapping[str, float],
    after_telemetry: Mapping[str, float],
    info: Mapping[str, Any] | None,
    reward: float | None,
    terminated: Any = None,
    truncated: Any = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    info = info if isinstance(info, Mapping) else {}
    release_telemetry = paired_gate_diagnostic._uid_lookup(
        info.get("release_intent_telemetry", {}), TARGET_UID, {}
    )
    if not isinstance(release_telemetry, Mapping):
        release_telemetry = {}
    safety_reason = getattr(env, "last_safety_decisions", {}).get(TARGET_UID)
    diagnostics = getattr(env, "energy_reachability_diagnostics", {}).get(TARGET_UID)
    emergency_delta = {
        key: value for key, value in paired_gate_diagnostic._telemetry_delta(
            before_telemetry, after_telemetry
        ).items() if "emergency" in key
    }
    result = {
        "proposal": None if proposal is None else int(proposal),
        "proposal_probability": (
            None if proposal_probability is None else float(proposal_probability)
        ),
        "reward": reward,
        "decision": runner_utils._jsonable(paired_gate_diagnostic._decision_for(info, TARGET_UID)),
        "release_intent_telemetry": runner_utils._jsonable(dict(release_telemetry)),
        "last_safety_decision": runner_utils._jsonable(safety_reason),
        "energy_reachability_diagnostics": runner_utils._jsonable(diagnostics),
        "emergency_telemetry_delta": emergency_delta,
        "before_target_context": runner_utils._jsonable(before_target),
        "post_target_context": runner_utils._jsonable(after_target),
        "terminated": runner_utils._jsonable(terminated),
        "truncated": runner_utils._jsonable(truncated),
        "error": None,
    }
    if error is not None:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def _first_step_branch(
    policy: Any,
    state: Mapping[str, Any],
    proposal: int | None,
    logits: Sequence[float],
    proposal_probability: float | None = None,
) -> dict[str, Any]:
    branch_env = copy.deepcopy(state["snapshot"])
    paired_gate_diagnostic.restore_diagnostic_snapshot(branch_env)
    actions, shortlists = paired_gate_diagnostic.p3_action_credit.branch_inputs(
        state["first_actions"], state["first_shortlists"], TARGET_UID, proposal, logits, shortlist_k=3,
    )
    before_target = _target_blocker_context(branch_env)
    before_telemetry = paired_gate_diagnostic._telemetry_snapshot(branch_env)
    try:
        _following, rewards, terminated, truncated, info = paired_gate_diagnostic._call_step(
            branch_env, actions, shortlists
        )
        reward = paired_gate_diagnostic._step_reward(rewards)
        return _first_step_payload(
            proposal=proposal, proposal_probability=proposal_probability,
            env=branch_env, before_target=before_target,
            after_target=_target_blocker_context(branch_env),
            before_telemetry=before_telemetry,
            after_telemetry=paired_gate_diagnostic._telemetry_snapshot(branch_env),
            info=info, reward=reward, terminated=terminated, truncated=truncated,
        )
    except Exception as error:
        return _first_step_payload(
            proposal=proposal, proposal_probability=proposal_probability,
            env=branch_env, before_target=before_target,
            after_target=_target_blocker_context(branch_env),
            before_telemetry=before_telemetry,
            after_telemetry=paired_gate_diagnostic._telemetry_snapshot(branch_env),
            info=None, reward=None, error=error,
        )


def _install_emergency_trace(env: Any, calls: list[dict[str, Any]]) -> None:
    if bool(getattr(env, "_failure_mechanism_emergency_trace_installed", False)):
        return
    original = env._try_emergency_landing

    def audited(uav: Any, *args: Any, **kwargs: Any) -> bool:
        if int(getattr(uav, "uid", -1)) != TARGET_UID:
            return bool(original(uav, *args, **kwargs))
        before = _target_blocker_context(env)
        before_telemetry = paired_gate_diagnostic._telemetry_snapshot(env)
        result: bool | None = None
        error: BaseException | None = None
        try:
            result = bool(original(uav, *args, **kwargs))
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            after = _target_blocker_context(env)
            delta = paired_gate_diagnostic._telemetry_delta(before_telemetry, paired_gate_diagnostic._telemetry_snapshot(env))
            calls.append({
                "time": float(getattr(env, "now", 0.0)),
                "uid": TARGET_UID,
                "force": bool(kwargs.get("force", False)),
                "before": before,
                "after": after,
                "before_energy": {key: before.get(key) for key in ("soc", "energy_remaining", "energy_consumed")},
                "after_energy": {key: after.get(key) for key in ("soc", "energy_remaining", "energy_consumed")},
                "emergency_telemetry_delta": {key: value for key, value in delta.items() if "emergency" in key},
                "last_safety_decision": runner_utils._jsonable(getattr(env, "last_safety_decisions", {}).get(TARGET_UID)),
                "available_compatible_lz_ids_before": before.get("available_compatible_lz_ids", []),
                "available_compatible_lz_ids_after": after.get("available_compatible_lz_ids", []),
                "returned": result,
                "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
            })

    env._try_emergency_landing = audited
    env._failure_mechanism_emergency_trace_installed = True


def _fixed_selected(output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_path = _resolve(output / "selected.json")
    selected_payload = _read_json(selected_path)
    selected = list(selected_payload.get("selection", {}).get("selected", []))
    _require(len(selected) == 3, "blockers requires exactly three registered states")
    _require(tuple(int(item.get("step", -1)) for item in selected) == REGISTERED_SELECTED_STEPS,
             "blockers selection changed; no reselection is permitted")
    for item in selected:
        _require(int(item.get("episode", -1)) == 24, "blockers selection episode binding changed")
        _require(int(item.get("uid", -1)) == TARGET_UID, "blockers selection UID binding changed")
        _require(int(item.get("arrival_id", -1)) == TARGET_ARRIVAL_ID,
                 "blockers selection arrival binding changed")
    return [dict(item) for item in selected], _hash_artifact(selected_path)


def _run_blockers(output: Path) -> dict[str, Any]:
    config, config_snapshot = _strict_config()
    inputs = _load_fixed_inputs()
    fixed_preflight_path = _resolve(FAILURE_OUTPUT_DIR / "preflight.json")
    fixed_preflight = _read_json(fixed_preflight_path)
    _require(dict(fixed_preflight.get("config_snapshot", {})) == config_snapshot,
             "blocker config differs from registered failure preflight")
    selected, selected_artifact = _fixed_selected(_resolve(FAILURE_OUTPUT_DIR))
    record_artifact = _hash_artifact(inputs["record_path"], RECORD_SHA256)
    policy, source_digest, _payload = _source_policy(config, inputs["source_path"])
    before_digest = source_digest
    _progress(output, "blockers_selected_replay", "RUNNING", 0, selected=3)
    states = _capture_selected_states(policy, config, inputs, selected, output)
    first_step_reports: list[dict[str, Any]] = []
    for order, item in enumerate(selected, start=1):
        state = states[int(item["step"])]
        context = paired_gate_diagnostic._details_context(policy, state["snapshot"])
        index, _key = paired_gate_diagnostic._context_target(context, TARGET_UID)
        logits = np.asarray(context["proposal_logits"][index], dtype=np.float64).copy()
        probabilities = np.asarray(context["proposal_probabilities"][index], dtype=np.float64).copy()
        branches = [_first_step_branch(policy, state, None, logits, None)]
        branches.extend(
            _first_step_branch(policy, state, proposal, logits, float(probabilities[proposal]))
            for proposal in range(25)
        )
        error_count = sum(1 for branch in branches if branch.get("error") is not None)
        first_step_reports.append({
            "selection_order": order,
            "state": item,
            "before_context": _blocker_context(state["snapshot"]),
            "proposal_logit_source": "paired_gate_diagnostic._details_context frozen source logits",
            "shortlist_k": 3,
            "branches": branches,
            "branch_counts": {
                "expected": len(branches), "successful": len(branches) - error_count,
                "errors": error_count,
            },
            "full_cycle_diagnostic": "omitted_first_step_only",
        })
        _progress(output, "blockers_first_step", "RUNNING", order, selected=3, branches=order * 26)
    after_selected_digest = str(rank_eval._model_digest(policy))
    _require(after_selected_digest == before_digest, "source model changed during clean blocker snapshots")

    branch_errors = sum(
        int(item["branch_counts"]["errors"]) for item in first_step_reports
    )
    branch_total = sum(
        int(item["branch_counts"]["expected"]) for item in first_step_reports
    )
    first_step_document = {
        "stage": STAGE, "phase": "blockers", "status": "COMPLETED",
        "selected_states": first_step_reports, "source_model_unchanged": True,
        "branch_counts": {
            "expected": branch_total, "successful": branch_total - branch_errors,
            "errors": branch_errors,
        },
        "original_replay_status": "PENDING",
        "scope": {"first_step_only": True, "full_cycle_diagnostic": "omitted_first_step_only", "optimizer_calls": 0,
                   "teacher_used": False, "heldout_used": False},
    }
    runner_utils._atomic_write_json(output / "firststep_blockers.json", first_step_document)

    emergency_calls: list[dict[str, Any]] = []
    installed = False

    def observe(env: Any) -> dict[str, Any]:
        nonlocal installed
        if not installed:
            _install_emergency_trace(env, emergency_calls)
            installed = True
        return _failure_observer(policy)(env)

    _progress(output, "blockers_original_replay", "RUNNING", 0)
    replay: Mapping[str, Any] = {}
    lifecycle: dict[str, Any] | None = None
    after_replay_digest: str | None = None
    replay_error: BaseException | None = None
    try:
        replay = credit._capture_replay(
            policy, config, inputs["tape_path"], inputs["tape_rows"],
            progress_callback=lambda count: _progress(output, "blockers_original_replay", "RUNNING", int(count)),
            state_observer=observe,
        )
        replay_meta = replay.get("replay", {}) if isinstance(replay, Mapping) else {}
        _require(bool(replay_meta.get("exact")), "blocker original replay was not exact")
        _require(int(replay_meta.get("steps_replayed", -1)) == EXPECTED_STEPS,
                 "blocker original replay step count mismatch")
        lifecycle = _validate_lifecycle(replay.get("lifecycle", {}))
        after_replay_digest = str(rank_eval._model_digest(policy))
        _require(after_replay_digest == before_digest, "source model changed during blocker replay")
    except Exception as error:
        replay_error = error
    finally:
        replay_meta = replay.get("replay", {}) if isinstance(replay, Mapping) else {}
        trace_status = "COMPLETED" if replay_error is None else "FAILED"
        runner_utils._atomic_write_json(output / "original_auto_emergency_trace.json", {
            "stage": STAGE, "phase": "blockers", "status": trace_status,
            "replay": {
                "exact": bool(replay_meta.get("exact", False)),
                "steps": replay_meta.get("steps_replayed"), "requests": EXPECTED_REQUESTS,
            },
            "lifecycle": lifecycle, "target_uid": TARGET_UID, "calls": emergency_calls,
            "source_model_digest_before": before_digest,
            "source_model_digest_after": after_replay_digest,
            "source_model_unchanged": after_replay_digest == before_digest if after_replay_digest is not None else None,
            "error": None if replay_error is None else {
                "type": type(replay_error).__name__, "message": str(replay_error),
            },
        })
    if replay_error is not None:
        raise replay_error
    first_step_document["original_replay_status"] = "COMPLETED"
    runner_utils._atomic_write_json(output / "firststep_blockers.json", first_step_document)
    manifest = {
        "stage": STAGE, "phase": "blockers", "status": "COMPLETED", "PPO_ALLOWED": PPO_ALLOWED,
        "registered_preflight": {"path": str(fixed_preflight_path), "sha256": _hash_artifact(fixed_preflight_path)["sha256"]},
        "record": record_artifact, "selected": selected_artifact,
        "config_snapshot": config_snapshot, "loaded_sourcefiles": runner_utils._loaded_code_manifest(),
        "source_model_digest": before_digest,
        "selected_steps": list(REGISTERED_SELECTED_STEPS),
        "branch_count": branch_total,
        "branch_counts": {
            "expected": branch_total, "successful": branch_total - branch_errors,
            "errors": branch_errors,
        },
        "original_replay": {"exact": True, "steps": EXPECTED_STEPS, "requests": EXPECTED_REQUESTS},
        "scope": {"no_continuation": True, "no_optimizer": True, "no_teacher": True, "no_heldout": True},
    }
    runner_utils._atomic_write_json(output / "manifest.json", manifest)
    _progress(output, "blockers", "COMPLETED", EXPECTED_STEPS, selected=3, branches=3 * 26)
    return manifest


def _run_diagnose(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    config, _snapshot = _strict_config()
    inputs = _load_fixed_inputs()
    selected_payload = _read_json(output / "selected.json")
    selected = list(selected_payload.get("selection", {}).get("selected", []))
    _require(len(selected) == 3, "diagnose requires exactly three registered states")
    policy, source_digest, _payload = _source_policy(config, inputs["source_path"])
    states = _capture_selected_states(policy, config, inputs, selected, output)
    after_policy, after_payload = rank_eval._load_policy(inputs["after_path"], config)
    after_digest = str(rank_eval._model_digest(after_policy))
    _require(after_digest == inputs["after_digest"], "after checkpoint digest differs from summary")
    reports: list[str] = []
    for index, item in enumerate(selected, start=1):
        step = int(item["step"])
        state = states[step]
        started = time.monotonic()
        callback = runner_utils._paired_branch_callback(
            output, selection_order=index, total_branches=26 * len(CONTINUATION_SEEDS),
            completed_states=index - 1, expected_states=len(selected), started=started,
            registered_selection_size=3,
        )
        paired = paired_gate_diagnostic.paired_gate_value(
            policy, state["snapshot"], state["first_actions"], state["first_shortlists"],
            TARGET_UID, CONTINUATION_SEEDS, BRANCH_MAX_STEPS, branch_callback=callback,
        )
        _require(bool(paired.get("model_unchanged")), "source model changed in paired diagnostic")
        _require(bool(paired.get("env_snapshot_unchanged")), "source environment snapshot changed in paired diagnostic")
        _require(bool(paired.get("rng_unchanged")), "caller RNG changed in paired diagnostic")
        delta_q = paired.get("qualified_delta_Q")
        target_world = np.asarray(state["target_world"], dtype=np.float32)[int(state["target_world_row"]):int(state["target_world_row"]) + 1]
        alignment = paired_gate_diagnostic.gate_update_alignment(
            policy, after_policy, target_world, delta_q=delta_q,
        )
        state_report = {
            "stage": STAGE, "phase": "diagnose", "status": "COMPLETED",
            "state": item, "paired": paired,
            "mechanism_metrics": {"D": delta_q, "C": paired.get("C_i"),
                                   "p": paired.get("p_release"), "m": alignment.get("local_m")},
            "alignment": {"source_model_digest": source_digest, "after_model_digest": after_digest,
                          "report": alignment,
                          "scope": "target-row probability alignment only; no causal claim"},
            "scope": {"continuation_seeds": list(CONTINUATION_SEEDS),
                      "branch_count": 26 * len(CONTINUATION_SEEDS),
                      "max_steps": BRANCH_MAX_STEPS,
                      "model_unchanged": bool(paired.get("model_unchanged")),
                      "env_snapshot_unchanged": bool(paired.get("env_snapshot_unchanged")),
                      "rng_unchanged": bool(paired.get("rng_unchanged")),
                      "retain_all_failed_censored_error": True,
                      "optimizer_calls": 0, "teacher_used": False, "heldout_used": False,
                      "population_estimate": False},
        }
        path = output / "diagnose" / f"state_{index:02d}.json"
        runner_utils._atomic_write_json(path, state_report)
        reports.append(str(path))
    after_source_digest = str(rank_eval._model_digest(policy))
    _require(after_source_digest == source_digest, "source model changed during diagnose")
    report = {
        "stage": STAGE, "phase": "diagnose", "status": "COMPLETED", "PPO_ALLOWED": PPO_ALLOWED,
        "selected_states": 3, "state_reports": reports,
        "source_model_unchanged": True,
        "after_alignment_model_digest": after_digest,
        "scope": {"independent_failure_mechanism_sample": True,
                   "not_merged_with_24_random_actor_samples": True,
                   "after_update_not_causal_for_batch_02_failure": True,
                   "optimizer_calls": 0, "teacher_used": False, "heldout_used": False},
    }
    runner_utils._atomic_write_json(output / "diagnose.json", report)
    _progress(output, "diagnose", "COMPLETED", 3)
    return report


def run_phase(phase: str, output_dir: Path) -> dict[str, Any]:
    output = Path(output_dir)
    _require(phase in {"preflight", "replay", "diagnose", "blockers"}, f"unknown phase: {phase}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        if phase == "preflight":
            report = _build_preflight()
            runner_utils._atomic_write_json(output / "preflight.json", report)
            runner_utils._atomic_write_json(output / "summary.json", {"stage": STAGE, "phase": phase, "status": report["status"]})
            return report
        if phase == "blockers":
            return _run_blockers(output)
        preflight = _load_preflight(output)
        if phase == "replay":
            return _run_replay(output, preflight)
        return _run_diagnose(output, preflight)
    except Exception as error:
        _write_failed_phase(output, phase, error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "replay", "diagnose", "blockers"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_phase(args.phase, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
