"""Passive frozen-policy executable-opportunity coverage diagnostic for P3.

The diagnostic reuses the completed forced-WAIT legacy arms as read-only
references.  It runs one legacy arm per registered scene, records the first
eligible state in each absolute-tick window, and evaluates WAIT plus the 25
proposal interventions from restored copies of that state.  No training or
qualification is performed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import RELEASE_GATE, WAIT_GATE
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import p3_action_credit
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_followup_diagnostics as followup
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen


STAGE = "P3_OPPORTUNITY_COVERAGE_20260930_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
EXECUTION_MODEL_DIGEST_ALGORITHM = "paired_gate_diagnostic._model_digest(policy)"
CANONICAL_SOURCE_DIGEST_ALGORITHM = "run_p3_task_neutral_pilot._model_digest(policy)"
MAX_STEPS = 1800
WINDOWS = ((100, 149), (300, 349), (500, 549))
PROPOSAL_COUNT = 25
BRANCH_COUNT = 26
REFERENCE_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_followup_20260930/forced_wait_frozen"
)
OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/p3_followup_20260930/opportunity_coverage"
)
PAIR_SPECS = (
    {"pair_id": 1, "env_seed": 877101, "keyed_policy_seed": 9303101},
    {"pair_id": 5, "env_seed": 877102, "keyed_policy_seed": 9303101},
    {"pair_id": 9, "env_seed": 877103, "keyed_policy_seed": 9303101},
    {"pair_id": 13, "env_seed": 877104, "keyed_policy_seed": 9303101},
    {"pair_id": 17, "env_seed": 877105, "keyed_policy_seed": 9303101},
    {"pair_id": 21, "env_seed": 877106, "keyed_policy_seed": 9303101},
    {"pair_id": 14, "env_seed": 877104, "keyed_policy_seed": 9303102},
    {"pair_id": 15, "env_seed": 877104, "keyed_policy_seed": 9303103},
    {"pair_id": 16, "env_seed": 877104, "keyed_policy_seed": 9303104},
)
PAIR_BY_ID = {int(item["pair_id"]): item for item in PAIR_SPECS}


class OpportunityProtocolError(ValueError):
    """Raised when a registered opportunity diagnostic contract fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OpportunityProtocolError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_path(path: str | Path) -> Path:
    target = Path(path)
    if target.exists():
        return target
    value = str(target).replace("\\", "/")
    prefix = "/home/user/training_outputs/terminal/"
    if value.startswith(prefix):
        local = Path(__file__).resolve().parents[3] / "training_outputs" / "terminal" / value[len(prefix):]
        if local.exists():
            return local
    return target


def _read_json(path: str | Path) -> dict[str, Any]:
    target = _sync_path(path)
    _require(target.is_file(), f"JSON artifact missing: {path}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpportunityProtocolError(f"invalid JSON artifact: {path}") from exc
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise OpportunityProtocolError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _config_snapshot(config: Any) -> dict[str, Any]:
    # Keep the exact existing frozen protocol serialization, including enum
    # and nested dataclass handling (not a second config schema).
    return dict(frozen._config_snapshot(config))


def _digest_metadata(source: Mapping[str, Any]) -> dict[str, str]:
    """Name the two existing model-digest algorithms in pair summaries."""

    return {
        "source_model_digest_algorithm": EXECUTION_MODEL_DIGEST_ALGORITHM,
        "canonical_source_model_digest": str(source.get("source_model_digest", "")),
        "canonical_source_model_digest_algorithm": CANONICAL_SOURCE_DIGEST_ALGORITHM,
    }


def _manifest_by_relative(manifest: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in manifest.get("files", ()) if isinstance(manifest, Mapping) else ():
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path", "")).replace("\\", "/")
        marker = "/terminal_marl/"
        if marker in path:
            relative = path.split(marker, 1)[1]
        else:
            relative = path
        if relative.endswith("run_p3_opportunity_coverage_20260930.py"):
            continue
        # Compare file hashes only.  The same old runner is ``__main__`` in
        # the reference process and an imported module in this process.
        result[relative] = str(item.get("sha256", ""))
    return result


def _logical_binding_value(value: Any) -> str:
    text = str(value).replace("\\", "/")
    marker = "/training_outputs/terminal/"
    if marker in text:
        return text.split(marker, 1)[1]
    return text


def _binding_equal(left: Any, right: Any, key: str) -> bool:
    if key in {"resolved_checkpoint", "warmup_checkpoint"}:
        return _logical_binding_value(left) == _logical_binding_value(right)
    return left == right


def _reference_layout(reference_root: Path) -> dict[str, Any]:
    root = _sync_path(reference_root)
    full_path = root / "full.json"
    full = _read_json(full_path)
    _require(full.get("status") == "FULL_OK", "forced-WAIT reference full.json is not FULL_OK")
    _require(int(full.get("pair_count", -1)) == 24, "forced-WAIT reference pair count is not 24")
    _require([int(value) for value in full.get("pair_ids", ())] == list(range(1, 25)),
             "forced-WAIT reference pair IDs are not 1..24")
    arms: dict[str, dict[str, Any]] = {}
    for spec in PAIR_SPECS:
        pair_id = int(spec["pair_id"])
        arm_path = root / "arms" / f"pair_{pair_id:04d}_legacy.json"
        arm = _read_json(arm_path)
        _require(arm.get("status") == "completed" and arm.get("censored") is False,
                 f"reference legacy arm {pair_id} is not natural and complete")
        _require(int(arm.get("env_seed", -1)) == int(spec["env_seed"]),
                 f"reference environment seed mismatch for pair {pair_id}")
        _require(int(arm.get("keyed_policy_seed", -1)) == int(spec["keyed_policy_seed"]),
                 f"reference keyed seed mismatch for pair {pair_id}")
        pair_status = (full.get("pair_statuses") or {}).get(str(pair_id), {})
        _require(pair_status.get("legacy") == "completed",
                 f"reference pair status mismatch for pair {pair_id}")
        arms[str(pair_id)] = {
            "declared_path": str(REFERENCE_ROOT / "arms" / f"pair_{pair_id:04d}_legacy.json"),
            "path": str(arm_path),
            "sha256": _sha256(arm_path),
            "env_seed": int(spec["env_seed"]),
            "keyed_policy_seed": int(spec["keyed_policy_seed"]),
            "arm": arm,
        }
    return {
        "declared_root": str(REFERENCE_ROOT),
        "resolved_root": str(root),
        "full_path": str(full_path),
        "full_sha256": _sha256(full_path),
        "pair_count": 24,
        "selected_pair_ids": [int(item["pair_id"]) for item in PAIR_SPECS],
        "arms": {key: {k: v for k, v in value.items() if k != "arm"} for key, value in arms.items()},
        "_arms": arms,
    }


def _validate_source_runtime(config: Any) -> dict[str, Any]:
    frozen._validate_config(config)
    _require(str(getattr(config, "student_decision_mask_mode", "legacy")) == "legacy",
             "opportunity diagnostic requires legacy action mask")
    source = frozen._runtime_source(config)
    _require(str(source.get("source_model_digest")) == followup.SOURCE_WARMUP_MODEL_DIGEST,
             "source model digest does not match 3fe7+a82")
    return source


def _build_preflight(reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    _configs, neutral_configs, runtime_protocol = neutral_eval._build_neutral_inputs()
    _require("dense24" in neutral_configs, "dense24 neutral config is missing")
    config = neutral_configs["dense24"]
    source = _validate_source_runtime(config)
    reference = _reference_layout(reference_root)
    old_preflight_path = _sync_path(reference_root) / "preflight.json"
    old_preflight = _read_json(old_preflight_path)
    old_config = old_preflight.get("runtime", {}).get("config_snapshot")
    _require(old_config == _config_snapshot(config),
             "dense24 config snapshot differs from frozen reference preflight")
    old_binding = old_preflight.get("source_binding", {})
    live_binding = frozen._source_binding(source)
    for key in ("source_model_digest", "candidate_digest_before_warmup", "resolved_checkpoint", "warmup_checkpoint"):
        _require(_binding_equal(live_binding.get(key), old_binding.get(key), key),
                 f"source binding differs from frozen reference for {key}")
    old_manifest = _manifest_by_relative(old_preflight.get("runtime", {}).get("code_manifest", {}))
    current_manifest = _manifest_by_relative(followup._loaded_code_manifest())
    _require(all(current_manifest.get(path) == digest for path, digest in old_manifest.items()),
             "an existing frozen runtime file hash differs from the reference")
    return {
        "stage": STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": PPO_ALLOWED,
        "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED,
        "optimizer_used": OPTIMIZER_USED,
        "source_binding": frozen._source_binding(source),
        "runtime": {
            "versions": followup._runtime_versions(),
            "config_snapshot": _config_snapshot(config),
            "runtime_protocol": _jsonable(runtime_protocol),
            "existing_code_manifest": followup._loaded_code_manifest(),
            "existing_code_manifest_excluding_runner": current_manifest,
        },
        "runner_sha256": _sha256(__file__),
        "protocol": {
            "observation_schema": frozen.OBSERVATION_SCHEMA,
            "observation_dimension": frozen.OBSERVATION_DIM,
            "action_schema": frozen.ACTION_SCHEMA,
            "runtime_proposal_shortlist_k": 3,
            "windows": [list(window) for window in WINDOWS],
            "branch_count": BRANCH_COUNT,
            "proposal_count": PROPOSAL_COUNT,
            "max_steps": MAX_STEPS,
            "reference_pair_ids": [int(item["pair_id"]) for item in PAIR_SPECS],
            "target_rule": "arrival13 only in env877104; max internal wait then min arrival ID otherwise",
            "full_cycle_values": False,
            "PPO_ALLOWED": False,
        },
        "reference": {key: value for key, value in reference.items() if key != "_arms"},
        "pair_grid": [dict(item) for item in PAIR_SPECS],
        "expected_max_states": len(PAIR_SPECS) * len(WINDOWS),
        "qualification": "not_a_qualification; passive one-step coverage only",
    }


def _validate_preflight(payload: Mapping[str, Any], reference_root: Path) -> None:
    _require(payload.get("stage") == STAGE, "opportunity preflight stage mismatch")
    _require(payload.get("status") == "PREFLIGHT_OK", "opportunity preflight is not successful")
    _require(payload.get("PPO_ALLOWED") is False, "PPO must remain disabled")
    _require(payload.get("pair_grid") == [dict(item) for item in PAIR_SPECS],
             "opportunity pair grid changed")
    _require(payload.get("protocol", {}).get("windows") == [list(window) for window in WINDOWS],
             "opportunity windows changed")
    _require(payload.get("runner_sha256") == _sha256(__file__),
             "opportunity runner changed after preflight")
    reference = _reference_layout(reference_root)
    stored = payload.get("reference", {})
    _require(stored.get("full_sha256") == reference.get("full_sha256"),
             "reference full.json changed after preflight")
    _require(stored.get("selected_pair_ids") == reference.get("selected_pair_ids"),
             "reference selected pairs changed after preflight")
    stored_arms = stored.get("arms", {})
    for pair_id, metadata in reference.get("arms", {}).items():
        old = stored_arms.get(str(pair_id), {})
        for key in ("sha256", "env_seed", "keyed_policy_seed"):
            _require(old.get(key) == metadata.get(key),
                     f"reference arm {pair_id} binding changed for {key}")


def _ensure_preflight(output: Path, reference_root: Path) -> dict[str, Any]:
    path = output / "preflight.json"
    if path.is_file():
        payload = _read_json(path)
        _validate_preflight(payload, reference_root)
        return payload
    payload = _build_preflight(reference_root)
    _write_json(path, payload, refuse_existing=True)
    return payload


def _actor(env: Any, uid: Any) -> Any | None:
    active = getattr(env, "active", {})
    return active.get(int(uid), active.get(str(uid)))


def _phase(actor: Any) -> str | None:
    value = getattr(getattr(actor, "phase", None), "value", getattr(actor, "phase", None))
    return None if value is None else str(value)


def _ring_role(actor: Any) -> str | None:
    value = getattr(getattr(actor, "ring_role", None), "value", getattr(actor, "ring_role", None))
    return None if value is None else str(value)


def _interface(actor: Any) -> str | None:
    return {"outer": "o2m", "middle": "m2i", "inner": "i2f"}.get(_ring_role(actor))


def _target_by_arrival(env: Any, arrival_id: int) -> tuple[Any, Any] | None:
    for uid, actor in getattr(env, "active", {}).items():
        if int(getattr(actor, "arrival_id", -1)) == int(arrival_id):
            return uid, actor
    return None


def _resources(env: Any, actor: Any | None) -> dict[str, Any]:
    queue = getattr(getattr(getattr(env, "admission", None), "queue", None), "items", ())
    zones: list[dict[str, Any]] = []
    for zone in getattr(getattr(env, "lz_manager", None), "zones", ()) or ():
        zones.append({
            "lz_id": int(getattr(zone, "lz_id", -1)),
            "status": _jsonable(getattr(getattr(zone, "status", None), "value", getattr(zone, "status", None))),
            "reserved_by": getattr(zone, "reserved_by", None),
            "occupied_by": getattr(zone, "occupied_by", None),
            "service_remaining": float(getattr(zone, "service_remaining", 0.0)),
        })
    compatible: list[int] | None = None
    if actor is not None:
        try:
            compatible = [int(value) for value in env.lz_manager.rank_available_zones(actor, float(env.now))]
        except (AttributeError, TypeError, ValueError):
            compatible = None
    return {
        "active_count": len(getattr(env, "active", {})),
        "service_count": len(getattr(env, "service_uavs", {})),
        "external_queue_count": len(queue),
        "final_active_count": sum(1 for item in getattr(env, "active", {}).values()
                                   if _phase(item) == "final"),
        "affected_interfaces": sorted(str(value) for value in getattr(getattr(env, "airspace", None), "affected_interfaces", lambda: set())()),
        "landing_zones": zones,
        "compatible_available_lz_ids": compatible,
    }


def _target_context(env: Any, uid: Any | None) -> dict[str, Any]:
    actor = None if uid is None else _actor(env, uid)
    if actor is None:
        return {"active": False, "uid": None if uid is None else int(uid)}
    return {
        "active": True,
        "uid": int(uid),
        "arrival_id": int(getattr(actor, "arrival_id", -1)),
        "phase": _phase(actor),
        "ring_role": _ring_role(actor),
        "soc": float(getattr(actor, "soc", 0.0)),
        "energy_remaining": float(getattr(actor, "energy_remaining", 0.0)),
        "energy_consumed": float(getattr(actor, "energy_consumed", 0.0)),
        "energy_capacity": float(getattr(actor, "energy_capacity", 0.0)),
        "internal_wait_time": float(getattr(actor, "internal_wait_time", 0.0)),
        "external_wait_time": float(getattr(actor, "external_wait_time", 0.0)),
        "safety_retry_after": float(getattr(actor, "safety_retry_after", 0.0)),
        "emergency_retry_after": float(getattr(actor, "emergency_retry_after", 0.0)),
        "committed": getattr(actor, "committed_plan", None) is not None,
        "interface": _interface(actor),
        "resources": _resources(env, actor),
    }


def _eligibility(env: Any, context: Mapping[str, Any], index: int, uid: Any) -> dict[str, Any]:
    actor = _actor(env, uid)
    phase = None if actor is None else _phase(actor)
    interface = None if actor is None else _interface(actor)
    affected = set(getattr(getattr(env, "airspace", None), "affected_interfaces", lambda: set())())
    actor_mask = bool(context["actor_mask"][index])
    reasons: list[str] = []
    if actor is None:
        reasons.append("inactive")
    if phase not in {"ring_outer", "ring_middle", "ring_inner"}:
        reasons.append("non_ring")
    if actor is not None and getattr(actor, "committed_plan", None) is not None:
        reasons.append("committed_plan")
    if not actor_mask:
        reasons.append("legacy_actor_mask_false")
    if actor is not None and float(getattr(actor, "soc", 0.0)) <= 0.0:
        reasons.append("nonpositive_soc")
    if actor is not None and float(getattr(env, "now", 0.0)) < float(getattr(actor, "safety_retry_after", 0.0)):
        reasons.append("safety_retry")
    if interface is None:
        reasons.append("missing_interface")
    elif interface in affected:
        reasons.append("geometry_affected")
    return {
        "uid": int(uid),
        "arrival_id": None if actor is None else int(getattr(actor, "arrival_id", -1)),
        "phase": phase,
        "interface": interface,
        "actor_mask": actor_mask,
        "affected_interfaces": sorted(str(value) for value in affected),
        "geometry_precheck": interface is not None and interface not in affected,
        "eligible": not reasons,
        "reasons": reasons,
    }


def _select_target(env: Any, context: Mapping[str, Any], env_seed: int) -> tuple[int | None, dict[str, Any]]:
    candidates = [
        _eligibility(env, context, index, uid)
        for index, uid in enumerate(context.get("uids", ()))
    ]
    eligible = [item for item in candidates if item["eligible"]]
    if int(env_seed) == 877104:
        eligible = [item for item in eligible if int(item["arrival_id"]) == 13]
    if not eligible:
        return None, {"candidates": candidates, "eligible": [], "selection": "missing"}
    if int(env_seed) == 877104:
        chosen = min(eligible, key=lambda item: int(item["uid"]))
    else:
        chosen = min(
            eligible,
            key=lambda item: (-float(getattr(_actor(env, item["uid"]), "internal_wait_time", 0.0)),
                              int(item["arrival_id"])),
        )
    return int(chosen["uid"]), {
        "candidates": candidates,
        "eligible": eligible,
        "selection": dict(chosen),
    }


def _env_fingerprint(env: Any) -> str:
    # The existing paired diagnostic fingerprints the complete SafetyTrace
    # state plus ledgers, queues, request times, and energy outcomes.  Reuse
    # that contract so position/velocity and other safety state cannot be
    # omitted by this runner.
    return str(paired._state_fingerprint(env)[0])


def _serial_actions(actions: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): np.asarray(value, dtype=np.float32).tolist() for key, value in actions.items()}


def _serial_shortlists(shortlists: Mapping[Any, Sequence[Any]]) -> dict[str, Any]:
    return {str(key): [np.asarray(value, dtype=np.float32).tolist() for value in values]
            for key, values in shortlists.items()}


def _trace_pre(env: Any, context: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    target = _target_by_arrival(env, 13)
    target_uid = None if target is None else target[0]
    index = None
    if target_uid is not None:
        for position, uid in enumerate(context.get("uids", ())):
            if str(uid) == str(target_uid):
                index = position
                break
    p_release = None
    q_max = None
    if index is not None:
        p_release = float(context["gate_probabilities"][index][RELEASE_GATE])
        q_max = float(np.max(context["proposal_probabilities"][index]))
    return {
        "tick": int(context.get("step", getattr(env, "step_count", 0))),
        "now": float(getattr(env, "now", 0.0)),
        "target_arrival_id": 13,
        "target_uid": None if target_uid is None else int(target_uid),
        "target": _target_context(env, target_uid),
        "p_release": p_release,
        "q_max": q_max,
        "resources": _resources(env, None if target is None else target[1]),
        "real_actions": _serial_actions(request["actions"]),
        "real_shortlists": _serial_shortlists(request["shortlists"]),
        "actual_decision": None,
        "actual_reward": None,
        "actual_error": None,
    }


def _branch_one(
    policy: Any,
    snapshot: Any,
    live_env: Any,
    actions: Mapping[Any, Any],
    shortlists: Mapping[Any, Any],
    target_uid: int,
    proposal: int | None,
    proposal_probability: float | None,
    wait_reward: float | None,
    original_step: Any,
    logits: Sequence[float],
) -> dict[str, Any]:
    caller_rng = paired._rng_snapshot()
    before_live = _env_fingerprint(live_env)
    model_before = paired._model_digest(policy)
    branch_env: Any = None
    before_target: dict[str, Any] = {}
    after_target: dict[str, Any] = {}
    before_telemetry: dict[str, float] = {}
    reward: float | None = None
    decision: str | None = None
    error: dict[str, str] | None = None
    terminated = None
    truncated = None
    info: Any = None
    safety: dict[str, Any] = {"safety_violation": False, "safety_violation_status": "unavailable"}
    release_intent_telemetry: dict[str, Any] = {}
    native_blocker_reasons: list[Any] = []
    live_unchanged = False
    caller_unchanged = False
    model_after = model_before
    try:
        branch_env = copy.deepcopy(snapshot)
        paired.restore_diagnostic_snapshot(branch_env)
        branch_actions, branch_shortlists = p3_action_credit.branch_inputs(
            actions, shortlists, target_uid, proposal, logits, shortlist_k=3,
        )
        before_telemetry = paired._telemetry_snapshot(branch_env)
        before_target = _target_context(branch_env, target_uid)
        _observations, rewards, terminated, truncated, info = original_step(
            branch_env, branch_actions, branch_shortlists
        )
        reward = paired._step_reward(rewards)
        decision = paired._decision_name(paired._decision_for(info, target_uid))
        raw_telemetry = paired._uid_lookup(
            info.get("release_intent_telemetry", {}) if isinstance(info, Mapping) else {},
            target_uid,
            {},
        )
        if isinstance(raw_telemetry, Mapping):
            release_intent_telemetry = _jsonable(dict(raw_telemetry))
            for key in ("primary_blocker", "secondary_constraints", "decision_history"):
                if key in raw_telemetry and raw_telemetry[key] is not None:
                    value = raw_telemetry[key]
                    native_blocker_reasons.extend(value if isinstance(value, (list, tuple)) else [value])
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if branch_env is not None:
            try:
                after_telemetry = paired._telemetry_snapshot(branch_env)
                safety = paired._safety_violation_details(
                    paired._telemetry_delta(before_telemetry, after_telemetry),
                    list(getattr(branch_env, "conflict_events", []) or []),
                )
                after_target = _target_context(branch_env, target_uid)
            except Exception as exc:
                if error is None:
                    error = {"type": type(exc).__name__, "message": str(exc)}
        try:
            model_after = paired._model_digest(policy)
            live_unchanged = before_live == _env_fingerprint(live_env)
            caller_unchanged = paired._rng_equal(caller_rng, paired._rng_snapshot())
        except Exception as exc:
            if error is None:
                error = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            # This restoration encloses deepcopy, branch construction, step,
            # serialization checks, and all post-checks.
            paired._restore_rng(caller_rng)
    if error is None and not live_unchanged:
        error = {"type": "LiveEnvironmentChanged", "message": "shadow branch changed live fingerprint"}
    if error is None and not caller_unchanged:
        error = {"type": "CallerRNGChanged", "message": "shadow branch changed caller RNG"}
    if error is None and model_before != model_after:
        error = {"type": "ModelChanged", "message": "shadow branch changed frozen model"}
    if bool(safety.get("safety_violation")) and error is None:
        error = {"type": "HardSafetyViolation", "message": "shadow branch reported hard safety violation"}
    return {
        "proposal": None if proposal is None else int(proposal),
        "proposal_probability": None if proposal_probability is None else float(proposal_probability),
        "status": "ok" if error is None else "error",
        "reward": reward,
        "reward_diff_vs_wait": None if reward is None or wait_reward is None else float(reward - wait_reward),
        "decision": decision,
        "approved": decision == "approved",
        "release_intent_telemetry": release_intent_telemetry,
        "native_blocker_reasons": _jsonable(native_blocker_reasons),
        "terminated": _jsonable(terminated),
        "truncated": _jsonable(truncated),
        "before_target": before_target,
        "after_target": after_target,
        "safety": _jsonable(safety),
        "model_unchanged": bool(model_before == model_after),
        "caller_rng_unchanged": bool(caller_unchanged),
        "live_environment_unchanged": bool(live_unchanged),
        "error": error,
    }


def _shadow_branches(policy: Any, env: Any, context: Mapping[str, Any], request: Mapping[str, Any],
                     target_uid: int, snapshot: Any, original_step: Any) -> dict[str, Any]:
    index = next(index for index, uid in enumerate(context["uids"]) if str(uid) == str(target_uid))
    logits = np.asarray(context["proposal_logits"][index], dtype=np.float64)
    probabilities = np.asarray(context["proposal_probabilities"][index], dtype=np.float64)
    live_before = _env_fingerprint(env)
    branches: list[dict[str, Any]] = []
    wait = _branch_one(policy, snapshot, env, request["actions"], request["shortlists"], target_uid,
                       None, None, None, original_step, logits)
    branches.append(wait)
    wait_reward = wait.get("reward") if wait.get("status") == "ok" else None
    for proposal in range(PROPOSAL_COUNT):
        branch = _branch_one(
            policy, snapshot, env, request["actions"], request["shortlists"], target_uid,
            proposal, float(probabilities[proposal]), wait_reward, original_step, logits,
        )
        branches.append(branch)
    errors = [item for item in branches if item.get("status") != "ok"]
    live_after = _env_fingerprint(env)
    approved = [int(item["proposal"]) for item in branches[1:]
                if item.get("status") == "ok" and item.get("decision") == "approved"]
    coverage = None if errors else float(sum(float(probabilities[index]) for index in approved))
    p_release = float(context["gate_probabilities"][index][RELEASE_GATE])
    return {
        "branch_count": len(branches),
        "branches": branches,
        "wait_reward": wait_reward,
        "approved_proposals": approved,
        "C": coverage,
        "pC": None if coverage is None else float(p_release * coverage),
        "p_release": p_release,
        "q_max": float(np.max(probabilities)),
        "branch_status_counts": dict(Counter(str(item.get("status")) for item in branches)),
        "live_environment_unchanged": bool(live_before == live_after),
        "C_unknown_reason": None if not errors else "branch_error_or_hard_safety_or_integrity",
        "contains_full_cycle_value": False,
    }


def _numeric_close(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_numeric_close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_numeric_close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def _compare_reference(arm: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    differences: list[str] = []
    for name in ("request_identity", "ticks", "safety", "model_unchanged", "caller_rng_unchanged"):
        left, right = arm.get(name), reference.get(name)
        if left != right:
            differences.append(name)
    if arm.get("ledger", {}).get("outcomes") != reference.get("ledger", {}).get("outcomes"):
        differences.append("ledger_outcomes")
    if arm.get("ledger", {}).get("terminal_rows") != reference.get("ledger", {}).get("terminal_rows"):
        differences.append("ledger_terminal_rows")
    for name in ("wait_metrics", "energy_ledger"):
        if not _numeric_close(arm.get(name), reference.get(name)):
            differences.append(name)
    for name in ("reward_total", "raw_reward_excluding_rejection_penalty", "release_rejection_penalty"):
        try:
            close = math.isclose(float(arm.get(name, float("nan"))),
                                 float(reference.get(name, float("nan"))),
                                 rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            close = False
        if not close:
            differences.append(name)
    return {
        "exact": not differences,
        "differences": differences,
        "reward_tolerance": 1e-9,
        "reference_status": reference.get("status"),
        "observed_status": arm.get("status"),
    }


def _attach_actual(trace: Mapping[str, Any], env: Any, target_uid: int | None,
                   rewards: Any, info: Any, error: BaseException | None = None) -> None:
    trace["actual_decision"] = None if target_uid is None else paired._decision_name(
        paired._decision_for(info, target_uid)
    )
    trace["actual_reward"] = None if error is not None else paired._step_reward(rewards)
    trace["actual_target_after"] = _target_context(env, target_uid)
    if error is not None:
        trace["actual_error"] = {"type": type(error).__name__, "message": str(error)}


def _run_opportunity_arm(source: Mapping[str, Any], config: Any, spec: Mapping[str, Any],
                         reference: Mapping[str, Any], progress: Any = None,
                         output: Path | None = None) -> dict[str, Any]:
    policy = copy.deepcopy(source["policy"])
    collector: dict[str, Any] = {"selected_states": [], "target13_trace": [], "technical_error": None}
    selected_windows: set[int] = set()
    original_keyed = frozen._keyed_request
    original_step = frozen.paired._call_step
    pending: dict[int, dict[str, Any]] = {}

    def wrapped_keyed(current_policy: Any, env: Any, keyed_seed: int) -> dict[str, Any]:
        context = paired._details_context(current_policy, env)
        request = original_keyed(current_policy, env, keyed_seed)
        tick = int(context.get("step", getattr(env, "step_count", 0)))
        trace = _trace_pre(env, context, request)
        collector["target13_trace"].append(trace)
        pending[id(env)] = {"trace": trace, "target_uid": trace.get("target_uid")}
        for window_index, (lo, hi) in enumerate(WINDOWS):
            if window_index in selected_windows or not lo <= tick <= hi:
                continue
            target_uid, selection = _select_target(env, context, int(spec["env_seed"]))
            if target_uid is None:
                continue
            snapshot = copy.deepcopy(env)
            paired.restore_diagnostic_snapshot(snapshot)
            target_index = next(index for index, uid in enumerate(context["uids"])
                                if str(uid) == str(target_uid))
            shadow = _shadow_branches(current_policy, env, context, request, target_uid, snapshot, original_step)
            state = {
                "window_index": int(window_index),
                "window": [int(lo), int(hi)],
                "absolute_tick": tick,
                "target_uid": int(target_uid),
                "target_arrival_id": int(getattr(_actor(env, target_uid), "arrival_id", -1)),
                "selection": selection,
                "target_before": _target_context(env, target_uid),
                "world_observation": np.asarray(context["world"][target_index], dtype=np.float32).tolist(),
                "world_observation_dimension": int(np.asarray(context["world"][target_index]).size),
                "uids": [str(uid) for uid in context["uids"]],
                "actor_mask": np.asarray(context["actor_mask"], dtype=bool).tolist(),
                "gate_logits": np.asarray(context["gate_logits"], dtype=np.float64).tolist(),
                "proposal_logits": np.asarray(context["proposal_logits"], dtype=np.float64).tolist(),
                "gate_probabilities": np.asarray(context["gate_probabilities"], dtype=np.float64).tolist(),
                "proposal_probabilities": np.asarray(context["proposal_probabilities"], dtype=np.float64).tolist(),
                "real_actions": _serial_actions(request["actions"]),
                "real_shortlists": _serial_shortlists(request["shortlists"]),
                "shadow": shadow,
                "branch_count": BRANCH_COUNT,
                "full_cycle_value_present": False,
                "m_present": False,
            }
            collector["selected_states"].append(state)
            if output is not None:
                _write_json(
                    output / "states" / f"pair_{int(spec['pair_id']):04d}_window_{window_index + 1:02d}.json",
                    state,
                    refuse_existing=True,
                )
                _write_json(output / "progress.json", {
                    "stage": STAGE,
                    "phase": "shadow",
                    "status": "RUNNING",
                    "pair_id": int(spec["pair_id"]),
                    "window_index": int(window_index),
                    "absolute_tick": tick,
                    "branch_count": int(shadow["branch_count"]),
                    "selected_state_count": len(collector["selected_states"]),
                }, refuse_existing=False)
            selected_windows.add(window_index)
            if not shadow["live_environment_unchanged"] or any(item.get("status") != "ok" for item in shadow["branches"]):
                collector["technical_error"] = {
                    "type": "ShadowIntegrityError",
                    "message": "shadow branch integrity check failed",
                    "window_index": window_index,
                }
                raise OpportunityProtocolError("shadow branch integrity check failed")
        return request

    def wrapped_step(env: Any, actions: Mapping[Any, Any], shortlists: Mapping[Any, Any]):
        try:
            result = original_step(env, actions, shortlists)
        except Exception as exc:
            meta = pending.pop(id(env), None)
            if meta is not None:
                _attach_actual(meta["trace"], env, meta.get("target_uid"), None, None, exc)
            raise
        meta = pending.pop(id(env), None)
        if meta is not None:
            _observations, rewards, _terminated, _truncated, info = result
            _attach_actual(meta["trace"], env, meta.get("target_uid"), rewards, info)
        return result

    frozen._keyed_request = wrapped_keyed
    frozen.paired._call_step = wrapped_step
    try:
        arm = frozen._run_arm(
            policy, config, env_seed=int(spec["env_seed"]), keyed_seed=int(spec["keyed_policy_seed"]),
            mode="legacy", progress=progress, pair_id=int(spec["pair_id"]),
        )
    finally:
        frozen._keyed_request = original_keyed
        frozen.paired._call_step = original_step
    arm["opportunity_diagnostic"] = {
        "selected_states": collector["selected_states"],
        "target13_trace": collector["target13_trace"],
        "selected_state_count": len(collector["selected_states"]),
        "missing_window_indices": [index for index in range(len(WINDOWS)) if index not in selected_windows],
        "max_selected_states": len(WINDOWS),
        "technical_error": collector["technical_error"],
    }
    ref_arm = reference["arm"]
    comparison = _compare_reference(arm, ref_arm) if arm.get("status") == "completed" else {
        "exact": False,
        "differences": ["observed_arm_not_natural_complete"],
        "reward_tolerance": 1e-9,
    }
    arm["reference_exact"] = comparison
    if arm.get("status") == "completed" and comparison.get("exact") is not True:
        arm["status"] = "technical_error"
        arm["error"] = {
            "type": "ReferenceReplayMismatch",
            "message": "new legacy replay differs from frozen reference",
            "differences": comparison.get("differences", []),
        }
    elif arm.get("status") != "completed":
        arm["reference_exact"] = {
            "exact": None,
            "known": False,
            "differences": ["observed_arm_not_natural_complete"],
            "unknown_reason": "censored_or_error_or_safety_violation",
            "reward_tolerance": 1e-9,
        }
    return arm


def _run_pair(source: Mapping[str, Any], config: Any, spec: Mapping[str, Any],
              reference: Mapping[str, Any], output: Path) -> dict[str, Any]:
    pair_id = int(spec["pair_id"])
    def progress(row: Mapping[str, Any]) -> None:
        _write_json(output / "progress.json", {
            "stage": STAGE, "phase": "arm", "status": "RUNNING", **dict(row),
        })
    try:
        arm = _run_opportunity_arm(source, config, spec, reference, progress, output)
    except Exception as exc:
        arm = {
            "mode": "legacy", "env_seed": int(spec["env_seed"]),
            "keyed_policy_seed": int(spec["keyed_policy_seed"]),
            "status": "technical_error", "censored": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "reference_exact": {"exact": False, "differences": ["runner_exception"]},
        }
    _write_json(output / "arms" / f"pair_{pair_id:04d}_legacy.json", arm, refuse_existing=True)
    pair = {
        "stage": STAGE,
        "pair": dict(spec),
        "status": "PAIR_OK" if arm.get("status") == "completed" and arm.get("reference_exact", {}).get("exact") else "PAIR_TECHNICAL_ERROR",
        "arm_path": str(output / "arms" / f"pair_{pair_id:04d}_legacy.json"),
        "reference": dict(reference["metadata"]),
        "selected_state_count": int(arm.get("opportunity_diagnostic", {}).get("selected_state_count", 0)),
        "missing_window_indices": arm.get("opportunity_diagnostic", {}).get("missing_window_indices", list(range(len(WINDOWS)))),
        "reference_exact": arm.get("reference_exact"),
        "source_model_digest": arm.get("model_digest_before"),
        "config_snapshot": _config_snapshot(config),
    }
    pair.update(_digest_metadata(source))
    _write_json(output / "pairs" / f"pair_{pair_id:04d}.json", pair, refuse_existing=True)
    return {"summary": pair, "arm": arm}


def _phase_config(output: Path, reference_root: Path) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    preflight = _ensure_preflight(output, reference_root)
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    config = neutral_configs["dense24"]
    _require(_config_snapshot(config) == preflight.get("runtime", {}).get("config_snapshot"),
             "live dense24 config differs from stored opportunity preflight")
    source = _validate_source_runtime(config)
    stored_binding = preflight.get("source_binding", {})
    live_binding = frozen._source_binding(source)
    for key in ("source_model_digest", "candidate_digest_before_warmup", "resolved_checkpoint", "warmup_checkpoint"):
        _require(_binding_equal(live_binding.get(key), stored_binding.get(key), key),
                 f"live source differs from stored opportunity preflight for {key}")
    current_manifest = _manifest_by_relative(followup._loaded_code_manifest())
    stored_manifest = preflight.get("runtime", {}).get("existing_code_manifest_excluding_runner", {})
    _require(all(current_manifest.get(path) == digest for path, digest in stored_manifest.items()),
             "live existing runtime hash differs from opportunity preflight")
    _require(preflight.get("runner_sha256") == _sha256(__file__),
             "opportunity runner changed after preflight")
    layout = _reference_layout(reference_root)
    return preflight, config, source, layout


def run_preflight(output: Path, reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = _build_preflight(reference_root)
    _write_json(output / "preflight.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
        "environment_created": False,
    }, refuse_existing=False)
    return payload


def run_pilot(output: Path, reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    preflight, config, source, layout = _phase_config(output, reference_root)
    spec = PAIR_SPECS[0]
    reference = layout["_arms"][str(spec["pair_id"])]
    reference["metadata"] = layout["arms"][str(spec["pair_id"])]
    result = _run_pair(source, config, spec, reference, output)
    status = "PILOT_OK" if result["summary"]["status"] == "PAIR_OK" else "PILOT_TECHNICAL_ERROR"
    payload = {
        "stage": STAGE, "phase": "pilot", "status": status, "PPO_ALLOWED": False,
        "pair": result["summary"], "selected_state_count": result["summary"]["selected_state_count"],
        "preflight_reference_sha256": preflight["reference"]["full_sha256"],
        "qualification": "not_a_qualification",
    }
    _write_json(output / "pilot.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", payload, refuse_existing=False)
    if status != "PILOT_OK":
        raise OpportunityProtocolError("pilot stopped after technical or reference failure")
    return payload


def _validate_pilot_reuse(pilot: Mapping[str, Any], preflight: Mapping[str, Any],
                          config: Any, source: Mapping[str, Any], output: Path) -> None:
    """Bind full phase to the exact successful pilot artifact and inputs."""

    _require(pilot.get("status") == "PILOT_OK", "full phase requires a successful pilot")
    expected = dict(PAIR_SPECS[0])
    pilot_pair = pilot.get("pair", {})
    _require(pilot_pair.get("pair") == expected, "pilot pair does not match registered pair 1")
    _require(pilot_pair.get("status") == "PAIR_OK", "pilot summary pair is not PAIR_OK")
    _require(pilot_pair.get("config_snapshot") == _config_snapshot(config),
             "pilot config snapshot differs from live full-phase config")
    source_policy = source.get("policy")
    _require(source_policy is not None, "live source policy is missing for execution digest check")
    execution_digest = paired._model_digest(source_policy)
    _require(pilot_pair.get("source_model_digest") == execution_digest,
             "pilot execution model digest differs from paired diagnostic digest")
    stored_binding = preflight.get("source_binding", {})
    _require(source.get("source_model_digest") == stored_binding.get("source_model_digest"),
             "live canonical source digest differs from frozen preflight binding")
    _require(pilot.get("preflight_reference_sha256") == preflight.get("reference", {}).get("full_sha256"),
             "pilot preflight reference binding differs from current preflight")

    pair_path = output / "pairs" / "pair_0001.json"
    arm_path = output / "arms" / "pair_0001_legacy.json"
    stored_pair = _read_json(pair_path)
    arm = _read_json(arm_path)
    _require(stored_pair.get("status") == "PAIR_OK", "stored pilot pair 1 is not PAIR_OK")
    _require(stored_pair.get("pair") == expected, "stored pilot pair 1 identity changed")
    for key in ("selected_state_count", "reference_exact", "source_model_digest", "config_snapshot"):
        _require(stored_pair.get(key) == pilot_pair.get(key),
                 f"stored pilot pair differs from pilot summary for {key}")
    for key, value in _digest_metadata(source).items():
        if key in pilot_pair or key in stored_pair:
            _require(stored_pair.get(key) == pilot_pair.get(key) == value,
                     f"stored pilot pair differs from pilot digest metadata for {key}")
    _require(arm.get("status") == "completed" and arm.get("censored") is False,
             "stored pilot arm is not natural and complete")
    _require(arm.get("model_unchanged") is True and arm.get("caller_rng_unchanged") is True,
             "stored pilot arm model/RNG integrity is incomplete")
    _require(arm.get("reference_exact", {}).get("exact") is True,
             "stored pilot arm reference replay is not exact")
    _require(pilot_pair.get("reference_exact") == arm.get("reference_exact"),
             "pilot summary reference result differs from stored pilot arm")


def run_full(output: Path, reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    pilot = _read_json(output / "pilot.json")
    preflight, config, source, layout = _phase_config(output, reference_root)
    _validate_pilot_reuse(pilot, preflight, config, source, output)
    pilot_summary = dict(pilot["pair"])
    pilot_summary.update(_digest_metadata(source))
    summaries = [pilot_summary]
    for spec in PAIR_SPECS[1:]:
        reference = layout["_arms"][str(spec["pair_id"])]
        reference["metadata"] = layout["arms"][str(spec["pair_id"])]
        result = _run_pair(source, config, spec, reference, output)
        summaries.append(result["summary"])
        if result["summary"]["status"] != "PAIR_OK":
            payload = {
                "stage": STAGE, "phase": "full", "status": "FULL_TECHNICAL_ERROR",
                "PPO_ALLOWED": False, "pair_summaries": summaries,
                "qualification": "not_a_qualification",
            }
            _write_json(output / "full.json", payload, refuse_existing=True)
            _write_json(output / "progress.json", payload, refuse_existing=False)
            raise OpportunityProtocolError("full phase stopped after technical or reference failure")
    payload = {
        "stage": STAGE, "phase": "full", "status": "FULL_OK", "PPO_ALLOWED": False,
        "pair_count": len(summaries), "pair_summaries": summaries,
        "selected_state_count": sum(int(item.get("selected_state_count", 0)) for item in summaries),
        "max_selected_states": len(PAIR_SPECS) * len(WINDOWS),
        "preflight_reference_sha256": preflight["reference"]["full_sha256"],
        "qualification": "not_a_qualification; known fixed profiles, one-step shadow coverage only",
    }
    _write_json(output / "full.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", payload, refuse_existing=False)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.phase == "preflight":
            run_preflight(args.output_dir, args.reference_root)
        elif args.phase == "pilot":
            run_pilot(args.output_dir, args.reference_root)
        else:
            run_full(args.output_dir, args.reference_root)
    except Exception as exc:
        _write_json(args.output_dir / "progress.json", {
            "stage": STAGE, "phase": args.phase, "status": "FAILED",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }, refuse_existing=False)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
