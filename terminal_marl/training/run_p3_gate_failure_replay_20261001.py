"""Passive exact replay and per-tick trace for the registered gate failure.

The replay uses the already registered pair-15 legacy arm and the fixed
always-RELEASE gate control.  It observes the existing executor through
temporary hooks, validates the resulting arm against its stored artifact, and
does not change a policy, action rule, safety rule, or optimizer state.
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import (
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_followup_diagnostics as followup
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_gate_release_control_20260930 as gate


STAGE = "P3_GATE_FAILURE_REPLAY_20261001_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
PAIR = {"pair_id": 15, "env_seed": 877104, "keyed_policy_seed": 9303103}
TARGET_ARRIVAL_ID = 21
OPTIONAL_ARRIVAL_ID = 13
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
OBSERVATION_DIM = 88
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"
SHORTLIST_K = 3
MAX_STEPS = 1800
FLOAT_ABS_TOL = 1e-8
FLOAT_REL_TOL = 1e-10
OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/gate_failure_replay_20261001"
)
GATE_OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/gate_release_control"
)
REFERENCE_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/forced_wait_frozen"
)


class FailureReplayError(ValueError):
    """Raised when the registered passive replay contract is not met."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FailureReplayError(message)


_jsonable = gate._jsonable
_read_json = gate._read_json
_write_json = gate._write_json
_sha256 = gate._sha256
_config_snapshot = gate._config_snapshot


def _manifest_signature(manifest: Any) -> list[tuple[str, str]]:
    rows = manifest.get("files", ()) if isinstance(manifest, Mapping) else ()
    runner_name = Path(__file__).name
    return sorted(
        (str(row.get("module")), str(row.get("sha256")))
        for row in rows
        if isinstance(row, Mapping) and Path(str(row.get("path", ""))).name != runner_name
    )


def _state_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return _jsonable(value)


def _mapping_lookup(mapping: Any, uid: Any, default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    for key in (uid, str(uid), int(uid) if str(uid).lstrip("-").isdigit() else uid):
        if key in mapping:
            return mapping[key]
    return default


def _telemetry_snapshot(env: Any) -> dict[str, float]:
    values = getattr(env, "telemetry", {})
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            result[str(key)] = float(value)
    return result


def _telemetry_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        try:
            before_value = float(before.get(key, 0.0))
            after_value = float(after.get(key, 0.0))
        except (TypeError, ValueError):
            result[key] = None
            continue
        if after_value == before_value:
            continue
        if not math.isfinite(before_value) or not math.isfinite(after_value):
            result[key] = None
            continue
        delta = after_value - before_value
        if delta != 0.0:
            result[key] = float(delta)
    return result


def _actor_for_arrival(env: Any, arrival_id: int) -> Any | None:
    for actor in getattr(env, "active", {}).values():
        try:
            if int(getattr(actor, "arrival_id")) == int(arrival_id):
                return actor
        except (AttributeError, TypeError, ValueError):
            continue
    return None


def _actor_uid(actor: Any) -> int | None:
    try:
        return int(getattr(actor, "uid"))
    except (AttributeError, TypeError, ValueError):
        return None


def _airspace_snapshot(env: Any) -> dict[str, Any]:
    airspace = getattr(env, "airspace", None)
    state_vector = getattr(airspace, "state_vector", None)
    state_vector_error = None
    try:
        geometry = _state_value(state_vector() if callable(state_vector) else None)
    except Exception as exc:
        geometry = None
        state_vector_error = {"type": type(exc).__name__, "message": str(exc)}
    headways = getattr(airspace, "headways", {})
    return {
        "state_vector": geometry,
        "headways": _jsonable(headways) if isinstance(headways, Mapping) else None,
        "transition_active": getattr(airspace, "transition", None) is not None,
        "state_vector_error": state_vector_error,
    }


def _lz_snapshot(env: Any) -> dict[str, Any]:
    manager = getattr(env, "lz_manager", None)
    zones = getattr(manager, "zones", [])
    rows = []
    for zone in zones:
        status = getattr(getattr(zone, "status", None), "value", getattr(zone, "status", None))
        compatible = getattr(zone, "compatible_missions", None)
        rows.append(
            {
                "lz_id": _jsonable(getattr(zone, "lz_id", None)),
                "status": _jsonable(status),
                "reserved_by": _jsonable(getattr(zone, "reserved_by", None)),
                "occupied_by": _jsonable(getattr(zone, "occupied_by", None)),
                "reservation_start": _jsonable(getattr(zone, "reservation_start", None)),
                "expected_release_at": _jsonable(getattr(zone, "expected_release_at", None)),
                "service_remaining": _jsonable(getattr(zone, "service_remaining", None)),
                "compatible_missions": sorted(
                    str(getattr(item, "value", item)) for item in (compatible or ())
                ),
            }
        )
    return {"zone_count": len(rows), "zones": rows}


def _target_state(env: Any, arrival_id: int) -> dict[str, Any] | None:
    actor = _actor_for_arrival(env, arrival_id)
    if actor is None:
        return None
    trace_error = None
    try:
        trace = env._uav_trace(actor) if callable(getattr(env, "_uav_trace", None)) else {}
    except Exception as exc:
        trace = {}
        trace_error = {"type": type(exc).__name__, "message": str(exc)}
    phase = getattr(getattr(actor, "phase", None), "value", getattr(actor, "phase", None))
    ring_role = getattr(getattr(actor, "ring_role", None), "value", getattr(actor, "ring_role", None))
    return {
        "uid": _actor_uid(actor),
        "arrival_id": int(arrival_id),
        "phase": _jsonable(phase),
        "ring_role": _jsonable(ring_role),
        "position": _state_value(getattr(actor, "position", None)),
        "energy_remaining": _jsonable(getattr(actor, "energy_remaining", None)),
        "energy_consumed": _jsonable(getattr(actor, "energy_consumed", None)),
        "soc": _jsonable(getattr(actor, "soc", None)),
        "safety_retry_after": _jsonable(getattr(actor, "safety_retry_after", None)),
        "emergency_retry_after": _jsonable(getattr(actor, "emergency_retry_after", None)),
        "emergency_landing": _jsonable(getattr(actor, "emergency_landing", None)),
        "committed": getattr(actor, "committed_plan", None) is not None,
        "assigned_lz": _jsonable(getattr(actor, "assigned_lz", None)),
        "trace": _jsonable(trace),
        "trace_error": trace_error,
    }


def _action_for_uid(actions: Any, uid: int | None) -> Any:
    if uid is None or not isinstance(actions, Mapping):
        return None
    return _state_value(_mapping_lookup(actions, uid))


def _shortlist_for_uid(shortlists: Any, uid: int | None) -> Any:
    if uid is None or not isinstance(shortlists, Mapping):
        return None
    return _state_value(_mapping_lookup(shortlists, uid))


def _action_fields(action: Any) -> dict[str, Any]:
    if action is None:
        return {"gate": None, "proposal_index": None, "joint_index_v4_1": None}
    try:
        decoded = hierarchical_from_normalized_v4_2(np.asarray(action, dtype=float))
        return {
            "gate": int(decoded.gate),
            "proposal_index": None if decoded.proposal is None else int(decoded.proposal),
            "joint_index_v4_1": int(decoded.joint_index_v4_1),
        }
    except (TypeError, ValueError, OverflowError):
        return {"gate": None, "proposal_index": None, "joint_index_v4_1": None}


def _tick_context(env: Any, actions: Any, shortlists: Any, info: Any, targets: Sequence[int]) -> dict[str, Any]:
    before_telemetry = _telemetry_snapshot(env)
    return {
        "now": _jsonable(getattr(env, "now", None)),
        "step": _jsonable(getattr(env, "step_count", None)),
        "active_count": len(getattr(env, "active", {}) or {}),
        "queue_count": len(getattr(getattr(getattr(env, "admission", None), "queue", None), "items", []) or []),
        "service_count": len(getattr(env, "service_uavs", {}) or {}),
        "targets": {
            str(arrival_id): {
                "state": _target_state(env, int(arrival_id)),
                "action": _action_for_uid(actions, _actor_uid(_actor_for_arrival(env, int(arrival_id)))),
                "action_fields": _action_fields(
                    _action_for_uid(actions, _actor_uid(_actor_for_arrival(env, int(arrival_id))))
                ),
                "shortlist": _shortlist_for_uid(
                    shortlists, _actor_uid(_actor_for_arrival(env, int(arrival_id)))
                ),
                "release_decision": _jsonable(
                    _mapping_lookup(info.get("release_decisions", {}),
                                   _actor_uid(_actor_for_arrival(env, int(arrival_id))))
                    if isinstance(info, Mapping) else None
                ),
                "release_intent_telemetry": _jsonable(
                    _mapping_lookup(info.get("release_intent_telemetry", {}),
                                   _actor_uid(_actor_for_arrival(env, int(arrival_id))))
                    if isinstance(info, Mapping) else None
                ),
                "last_safety_decision": _jsonable(
                    _mapping_lookup(getattr(env, "last_safety_decisions", {}),
                                   _actor_uid(_actor_for_arrival(env, int(arrival_id))))
                ),
                "energy_reachability_diagnostic": _jsonable(
                    _mapping_lookup(getattr(env, "energy_reachability_diagnostics", {}),
                                   _actor_uid(_actor_for_arrival(env, int(arrival_id))))
                ),
            }
            for arrival_id in targets
        },
        "lz": _lz_snapshot(env),
        "airspace": _airspace_snapshot(env),
        "emergency_telemetry": {
            key: value for key, value in before_telemetry.items() if key.startswith("emergency_")
        },
        "telemetry": before_telemetry,
    }


class _TraceCollector:
    def __init__(self, targets: Sequence[int]) -> None:
        self.targets = tuple(int(value) for value in targets)
        self.steps: list[dict[str, Any]] = []
        self.emergency_calls: list[dict[str, Any]] = []

    def payload(self, *, arm_mode: str, env_seed: int, keyed_seed: int) -> dict[str, Any]:
        relevant = [
            row for row in self.steps
            if any(item.get("before", {}).get("state") is not None
                   or item.get("after", {}).get("state") is not None
                   for item in row.get("targets", {}).values())
        ]
        first_step = relevant[0].get("step_before") if relevant else None
        last_step = relevant[-1].get("step_after") if relevant else None
        return {
            "schema": "p3_gate_failure_replay_trace_v1",
            "arm_mode": arm_mode,
            "env_seed": int(env_seed),
            "keyed_policy_seed": int(keyed_seed),
            "target_arrival_ids": list(self.targets),
            "steps": self.steps,
            "emergency_calls": self.emergency_calls,
            "summary": {
                "step_records": len(self.steps),
                "relevant_step_records": len(relevant),
                "first_relevant_step": first_step,
                "last_relevant_step": last_step,
                "emergency_call_records": len(self.emergency_calls),
                "native_lz_rejection_detail": None,
            },
        }


def _make_observed_env_class(trace: _TraceCollector) -> type:
    base = frozen._CapturedTerminalEnv

    class ObservedTerminalEnv(base):
        def __init__(self, config: Any) -> None:
            self._failure_replay_trace = trace
            super().__init__(config)

        def _try_emergency_landing(self, uav: Any, *, force: bool = False) -> bool:
            arrival_id = getattr(uav, "arrival_id", None)
            if arrival_id is None or int(arrival_id) not in trace.targets:
                return super()._try_emergency_landing(uav, force=force)
            before_telemetry = _telemetry_snapshot(self)
            before_state = _target_state(self, int(arrival_id))
            before_lz = _lz_snapshot(self)
            try:
                result = super()._try_emergency_landing(uav, force=force)
            except Exception as exc:
                trace.emergency_calls.append(
                    {
                        "arrival_id": int(arrival_id),
                        "uid": _actor_uid(uav),
                        "now": _jsonable(getattr(self, "now", None)),
                        "force": bool(force),
                        "result": None,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "before": before_state,
                        "after": _target_state(self, int(arrival_id)),
                        "telemetry_delta": _telemetry_delta(before_telemetry, _telemetry_snapshot(self)),
                        "lz_before": before_lz,
                        "lz_after": _lz_snapshot(self),
                        "lz_rejection_detail": None,
                    }
                )
                raise
            trace.emergency_calls.append(
                {
                    "arrival_id": int(arrival_id),
                    "uid": _actor_uid(uav),
                    "now": _jsonable(getattr(self, "now", None)),
                    "force": bool(force),
                    "result": bool(result),
                    "before": before_state,
                    "after": _target_state(self, int(arrival_id)),
                    "telemetry_delta": _telemetry_delta(before_telemetry, _telemetry_snapshot(self)),
                    "lz_before": before_lz,
                    "lz_after": _lz_snapshot(self),
                    "lz_rejection_detail": None,
                }
            )
            return result

    ObservedTerminalEnv.__name__ = "ObservedTerminalEnv"
    return ObservedTerminalEnv


def _comparison_view(arm: Mapping[str, Any], *, control: bool) -> dict[str, Any]:
    fields = (
        "mode", "env_seed", "keyed_policy_seed", "status", "censored", "stop_reason",
        "budget_cap_steps", "budget_cap_is_not_natural_terminal", "ticks", "simulation_seconds",
        "generated_requests", "generated_process_count", "request_identity", "request_identity_count",
        "request_identity_unique", "request_groups", "ledger", "terminal_summary", "outcome_counts",
        "request_capture_integrity", "wait_integrity", "admitted_count", "admitted_arrival_ids",
        "telemetry_arrivals_admitted", "normal_count", "emergency_count", "resolved_count",
        "completed_count", "reward_total", "reward_components", "raw_reward_excluding_rejection_penalty",
        "release_rejection_penalty", "wait_metrics", "energy_ledger", "mask_counts", "telemetry",
        "telemetry_delta", "safety", "model_digest_before", "model_digest_after", "model_unchanged",
        "caller_rng_unchanged", "keying", "error",
    )
    result = {key: arm.get(key) for key in fields}
    if control:
        result["control"] = arm.get("control")
        result["control_metrics"] = arm.get("control_metrics")
        result["reference_control_checks"] = arm.get("reference_control_checks")
    return _jsonable(result)


def _compare_values(expected: Any, actual: Any, path: str, errors: list[dict[str, Any]]) -> None:
    if len(errors) >= 40:
        return
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            errors.append({"path": path, "expected_keys": sorted(map(str, expected)), "actual_keys": sorted(map(str, actual))})
            return
        for key in expected:
            _compare_values(expected[key], actual[key], f"{path}.{key}", errors)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            errors.append({"path": path, "expected_length": len(expected), "actual_length": len(actual)})
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare_values(left, right, f"{path}[{index}]", errors)
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(expected, bool) and not isinstance(actual, bool):
        if not math.isclose(float(expected), float(actual), rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL):
            errors.append({"path": path, "expected": expected, "actual": actual})
        return
    if expected != actual:
        errors.append({"path": path, "expected": expected, "actual": actual})


def _validate_arm_replay(actual: Mapping[str, Any], expected: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    for field in ("request_identity", "request_groups", "ticks", "outcome_counts"):
        if actual.get(field) != expected.get(field):
            errors.append({
                "path": field,
                "expected": expected.get(field),
                "actual": actual.get(field),
                "comparison": "exact",
            })
    if actual.get("model_unchanged") is not True:
        errors.append({"path": "model_unchanged", "expected": True, "actual": actual.get("model_unchanged")})
    if actual.get("caller_rng_unchanged") is not True:
        errors.append({"path": "caller_rng_unchanged", "expected": True, "actual": actual.get("caller_rng_unchanged")})
    safety = actual.get("safety", {})
    if safety.get("safety_violation") is not False:
        errors.append({"path": "safety.safety_violation", "expected": False, "actual": safety.get("safety_violation")})
    checks = actual.get("reference_control_checks", {})
    if mode == "control" and checks.get("hard_safety_clear") is not True:
        errors.append({"path": "reference_control_checks.hard_safety_clear", "expected": True,
                       "actual": checks.get("hard_safety_clear")})
    _compare_values(
        _comparison_view(expected, control=mode == "control"),
        _comparison_view(actual, control=mode == "control"),
        "arm",
        errors,
    )
    expected_target = next(
        (row for row in expected.get("ledger", {}).get("terminal_rows", []) if row.get("arrival_id") == TARGET_ARRIVAL_ID),
        None,
    )
    actual_target = next(
        (row for row in actual.get("ledger", {}).get("terminal_rows", []) if row.get("arrival_id") == TARGET_ARRIVAL_ID),
        None,
    )
    target_errors: list[dict[str, Any]] = []
    _compare_values(expected_target, actual_target, "target_terminal_row", target_errors)
    if target_errors:
        errors.extend(target_errors)
    return {
        "match": not errors,
        "mode": mode,
        "float_abs_tol": FLOAT_ABS_TOL,
        "float_rel_tol": FLOAT_REL_TOL,
        "request_identity_exact": actual.get("request_identity") == expected.get("request_identity"),
        "ticks_exact": actual.get("ticks") == expected.get("ticks"),
        "outcome_counts_exact": actual.get("outcome_counts") == expected.get("outcome_counts"),
        "model_unchanged": actual.get("model_unchanged") is True,
        "caller_rng_unchanged": actual.get("caller_rng_unchanged") is True,
        "hard_safety_clear": safety.get("safety_violation") is False,
        "terminal_rows_checked": True,
        "target_terminal_row_match": not target_errors,
        "mismatches": errors,
    }


def _load_live_inputs() -> tuple[Any, dict[str, Any]]:
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    _require("dense24" in neutral_configs, "dense24 neutral config is missing")
    config = neutral_configs["dense24"]
    frozen._validate_config(config)
    _require(str(getattr(config, "observation_schema_version", "")) == OBSERVATION_SCHEMA, "observation schema mismatch")
    _require(int(getattr(config, "runtime_proposal_shortlist_k", -1)) == SHORTLIST_K, "shortlist K mismatch")
    _require(float(getattr(config, "max_simulation_seconds", 0.0)) == 1800.0, "simulation cap mismatch")
    source = frozen._runtime_source(config)
    _require(str(source.get("source_model_digest")) == followup.SOURCE_WARMUP_MODEL_DIGEST, "source model digest mismatch")
    return config, source


def _reference_paths() -> dict[str, Path]:
    return {
        "legacy": REFERENCE_ROOT / "arms" / "pair_0015_legacy.json",
        "control": GATE_OUTPUT_DIR / "arms" / "pair_0015_control.json",
        "gate_preflight": GATE_OUTPUT_DIR / "preflight.json",
    }


def _validate_reference_arm(arm: Mapping[str, Any], label: str) -> None:
    _require(arm.get("status") == "completed" and arm.get("censored") is False, f"registered {label} arm is not natural")
    _require(int(arm.get("env_seed", -1)) == PAIR["env_seed"], f"{label} environment seed mismatch")
    _require(int(arm.get("keyed_policy_seed", -1)) == PAIR["keyed_policy_seed"], f"{label} policy seed mismatch")
    _require(int(arm.get("request_identity_count", -1)) == 24, f"{label} request count is not 24")
    _require(arm.get("request_identity_unique") is True, f"{label} request identities are not unique")
    _require(len(arm.get("request_identity", ())) == 24, f"{label} request identity rows are incomplete")
    _require(arm.get("model_unchanged") is True, f"{label} model-unchanged flag is false")
    _require(arm.get("caller_rng_unchanged") is True, f"{label} caller-RNG flag is false")
    safety = arm.get("safety", {})
    _require(safety.get("safety_violation") is False, f"{label} safety violation flag is set")
    checks = arm.get("reference_control_checks", {})
    if checks:
        _require(checks.get("hard_safety_clear") is True, f"{label} hard-safety check is not true")
        _require(checks.get("technical_ok") is True, f"{label} technical check is not true")
    terminal_rows = arm.get("ledger", {}).get("terminal_rows", ())
    _require(any(int(row.get("arrival_id", -1)) == TARGET_ARRIVAL_ID for row in terminal_rows),
             f"{label} target arrival terminal row is missing")


def _build_preflight() -> dict[str, Any]:
    config, source = _load_live_inputs()
    paths = _reference_paths()
    gate_preflight = _read_json(paths["gate_preflight"])
    _require(gate_preflight.get("status") == "PREFLIGHT_OK", "registered gate preflight is not successful")
    gate_runtime = gate_preflight.get("runtime", {})
    gate_config = gate_runtime.get("config_snapshot")
    _require(gate_config == _config_snapshot(config), "live dense24 config differs from gate preflight")
    binding = frozen._source_binding(source)
    stored_binding = gate_runtime.get("source_binding", {})
    for key in ("source_model_digest", "candidate_digest_before_warmup", "resolved_checkpoint", "warmup_checkpoint"):
        _require(gate._binding_equal(binding.get(key), stored_binding.get(key), key), f"source binding differs for {key}")
    expected_legacy = _read_json(paths["legacy"])
    expected_control = _read_json(paths["control"])
    for label, arm in (("legacy", expected_legacy), ("control", expected_control)):
        _validate_reference_arm(arm, label)
    return {
        "stage": STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "environment_created": False,
        "PPO_ALLOWED": PPO_ALLOWED,
        "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED,
        "optimizer_used": OPTIMIZER_USED,
        "protocol": {
            "pair": dict(PAIR),
            "target_arrival_id": TARGET_ARRIVAL_ID,
            "optional_arrival_id": OPTIONAL_ARRIVAL_ID,
            "observation_schema": OBSERVATION_SCHEMA,
            "observation_dimension": OBSERVATION_DIM,
            "action_schema": ACTION_SCHEMA,
            "runtime_proposal_shortlist_k": SHORTLIST_K,
            "max_steps": MAX_STEPS,
            "float_abs_tol": FLOAT_ABS_TOL,
            "float_rel_tol": FLOAT_REL_TOL,
            "passive_only": True,
        },
        "source_binding": binding,
        "runtime": {
            "config_snapshot": _config_snapshot(config),
            "code_manifest": followup._loaded_code_manifest(),
            "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        },
        "registered_artifacts": {
            label: {"path": str(path), "sha256": _sha256(gate._sync_path(path))}
            for label, path in paths.items()
        },
        "qualification": "not_a_qualification; exact passive failure replay only",
    }


def _validate_preflight(payload: Mapping[str, Any]) -> None:
    _require(payload.get("stage") == STAGE, "failure replay preflight stage mismatch")
    _require(payload.get("status") == "PREFLIGHT_OK", "failure replay preflight is not successful")
    _require(payload.get("environment_created") is False, "preflight created an environment")
    _require(payload.get("PPO_ALLOWED") is False, "PPO must remain disabled")
    protocol = payload.get("protocol", {})
    _require(protocol.get("pair") == PAIR, "registered pair changed")
    _require(int(protocol.get("target_arrival_id", -1)) == TARGET_ARRIVAL_ID, "target arrival changed")
    _require(int(protocol.get("runtime_proposal_shortlist_k", -1)) == SHORTLIST_K, "K=3 binding changed")
    _require(int(protocol.get("max_steps", -1)) == MAX_STEPS, "max step cap changed")
    config, source = _load_live_inputs()
    runtime = payload.get("runtime", {})
    _require(runtime.get("config_snapshot") == _config_snapshot(config), "live dense24 config differs from saved preflight")
    _require(
        _manifest_signature(runtime.get("code_manifest")) ==
        _manifest_signature(followup._loaded_code_manifest()),
        "live runtime code manifest differs from saved preflight",
    )
    saved_binding = payload.get("source_binding", {})
    live_binding = frozen._source_binding(source)
    for key in ("source_model_digest", "candidate_digest_before_warmup", "resolved_checkpoint", "warmup_checkpoint"):
        _require(gate._binding_equal(live_binding.get(key), saved_binding.get(key), key), f"live source binding differs for {key}")
    saved_runner = runtime.get("runner", {})
    _require(saved_runner.get("sha256") == _sha256(Path(__file__)), "failure replay runner changed")
    paths = _reference_paths()
    registered = payload.get("registered_artifacts", {})
    for label, path in paths.items():
        expected = registered.get(label, {}).get("sha256")
        _require(expected == _sha256(gate._sync_path(path)), f"registered artifact changed: {label}")


def run_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = _build_preflight()
    _write_json(output / "preflight.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK", "environment_created": False,
    }, refuse_existing=True)
    return payload


def run_check(output: Path) -> dict[str, Any]:
    payload = _build_preflight()
    preflight_path = output / "preflight.json"
    if preflight_path.exists():
        stored = _read_json(preflight_path)
        _validate_preflight(stored)
        payload = stored
    else:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(preflight_path, payload, refuse_existing=True)
    check = {
        "stage": STAGE, "phase": "check", "status": "CHECK_OK",
        "environment_created": False, "preflight_status": payload.get("status"),
        "registered_artifacts": payload.get("registered_artifacts"),
    }
    _write_json(output / "check.json", check, refuse_existing=True)
    _write_json(output / "progress.json", check)
    return check


def _make_step_observer(trace: _TraceCollector, original: Callable[..., Any]) -> Callable[..., Any]:
    def observed_step(env: Any, actions: Mapping[Any, Any], shortlists: Mapping[Any, Any]):
        before = _tick_context(env, actions, shortlists, {}, trace.targets)
        try:
            result = original(env, actions, shortlists)
        except Exception as exc:
            trace.steps.append({
                "step_before": before.get("step"), "step_after": None,
                "targets": {
                    str(key): {"before": before["targets"].get(str(key)), "after": None}
                    for key in trace.targets
                },
                "before": before, "after": None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })
            raise
        info = result[4] if isinstance(result, tuple) and len(result) >= 5 else {}
        after = _tick_context(env, {}, {}, info, trace.targets)
        trace.steps.append({
            "step_before": before.get("step"), "step_after": after.get("step"),
            "targets": {
                str(key): {
                    "before": before["targets"].get(str(key)),
                    "after": after["targets"].get(str(key)),
                }
                for key in trace.targets
            },
            "before": before, "after": after, "error": None,
        })
        return result
    return observed_step


@contextmanager
def _instrument_runtime(trace: _TraceCollector):
    original_step = paired._call_step
    original_env_class = frozen._CapturedTerminalEnv
    frozen._CapturedTerminalEnv = _make_observed_env_class(trace)
    paired._call_step = _make_step_observer(trace, original_step)
    try:
        yield
    finally:
        paired._call_step = original_step
        frozen._CapturedTerminalEnv = original_env_class


def _run_legacy(source: Mapping[str, Any], config: Any, trace: _TraceCollector) -> dict[str, Any]:
    with _instrument_runtime(trace):
        return frozen._run_arm(
            copy.deepcopy(source["policy"]), config,
            env_seed=PAIR["env_seed"], keyed_seed=PAIR["keyed_policy_seed"],
            mode="legacy", progress=None, pair_id=PAIR["pair_id"],
        )


def _run_control(source: Mapping[str, Any], config: Any, expected: Mapping[str, Any], trace: _TraceCollector) -> dict[str, Any]:
    reference = {"arm": dict(expected), "metadata": {"sha256": "registered"}}
    with _instrument_runtime(trace):
        return gate._run_control_arm(
            source, config, PAIR, reference, progress=None,
            output=Path("/tmp/p3_gate_failure_replay_unused"),
        )


def _trace_path(output: Path, mode: str) -> Path:
    return output / "traces" / f"pair_0015_{mode}_trace.json"


def _arm_path(output: Path, mode: str) -> Path:
    return output / "arms" / f"pair_0015_{mode}_replay.json"


def _validate_trace(trace: _TraceCollector, arm: Mapping[str, Any]) -> dict[str, Any]:
    target_seen = any(
        item.get("targets", {}).get(str(TARGET_ARRIVAL_ID), {}).get("before", {}).get("state") is not None
        or item.get("targets", {}).get(str(TARGET_ARRIVAL_ID), {}).get("after", {}).get("state") is not None
        for item in trace.steps
    )
    errors = []
    if len(trace.steps) != int(arm.get("ticks", -1)):
        errors.append({"field": "step_records", "expected": arm.get("ticks"), "actual": len(trace.steps)})
    if not target_seen:
        errors.append({"field": "target_state", "expected": "arrival 21 observed", "actual": "missing"})
    return {"match": not errors, "step_records": len(trace.steps), "target_state_observed": target_seen, "errors": errors}


def run_replay(output: Path, *, include_arrival13: bool = False) -> dict[str, Any]:
    preflight = _read_json(output / "preflight.json")
    _validate_preflight(preflight)
    config, source = _load_live_inputs()
    _require(include_arrival13, "registered replay requires target arrivals 21 and 13")
    targets = (TARGET_ARRIVAL_ID, OPTIONAL_ARRIVAL_ID)
    expected_legacy = _read_json(_reference_paths()["legacy"])
    expected_control = _read_json(_reference_paths()["control"])
    results: dict[str, Any] = {}
    _write_json(output / "progress.json", {
        "stage": STAGE, "phase": "run", "status": "RUNNING", "pair": dict(PAIR),
        "target_arrival_ids": list(targets),
    })
    for mode, expected, runner in (
        ("legacy", expected_legacy, _run_legacy),
        ("control", expected_control, _run_control),
    ):
        trace = _TraceCollector(targets)
        try:
            actual = runner(source, config, trace) if mode == "legacy" else runner(source, config, expected, trace)
        except Exception as exc:
            trace_payload = trace.payload(arm_mode=mode, env_seed=PAIR["env_seed"], keyed_seed=PAIR["keyed_policy_seed"])
            trace_payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
            _write_json(_trace_path(output, mode), trace_payload, refuse_existing=True)
            failure = {
                "stage": STAGE, "phase": "run", "status": "REPLAY_ERROR", "pair": dict(PAIR),
                "failed_mode": mode, "error": trace_payload["error"],
                "trace_paths": {name: str(_trace_path(output, name)) for name in results} | {mode: str(_trace_path(output, mode))},
            }
            _write_json(output / "failure_replay.json", failure, refuse_existing=True)
            _write_json(output / "progress.json", failure)
            return failure
        trace_payload = trace.payload(arm_mode=mode, env_seed=PAIR["env_seed"], keyed_seed=PAIR["keyed_policy_seed"])
        validation = _validate_arm_replay(actual, expected, mode=mode)
        validation["trace"] = _validate_trace(trace, actual)
        validation["match"] = bool(validation["match"] and validation["trace"]["match"])
        arm_output = dict(actual)
        _write_json(_arm_path(output, mode), arm_output, refuse_existing=True)
        _write_json(_trace_path(output, mode), trace_payload, refuse_existing=True)
        results[mode] = {
            "arm_path": str(_arm_path(output, mode)),
            "trace_path": str(_trace_path(output, mode)),
            "validation": validation,
            "target_terminal_row": next(
                (row for row in actual.get("ledger", {}).get("terminal_rows", []) if row.get("arrival_id") == TARGET_ARRIVAL_ID),
                None,
            ),
        }
        if validation.get("match") is not True:
            failure = {
                "stage": STAGE, "phase": "run", "status": "REPLAY_MISMATCH", "pair": dict(PAIR),
                "failed_mode": mode, "arms": dict(results),
                "error": {"type": "ReplayMismatch", "message": f"{mode} arm differs from registered artifact"},
            }
            _write_json(output / "failure_replay.json", failure, refuse_existing=True)
            _write_json(output / "progress.json", failure)
            return failure
    final = {
        "stage": STAGE, "phase": "run", "status": "REPLAY_OK", "pair": dict(PAIR),
        "PPO_ALLOWED": PPO_ALLOWED, "off_policy": True,
        "target_arrival_ids": list(targets), "arms": results,
        "source_model_digest": source.get("source_model_digest"),
        "qualification": "not_a_qualification; passive exact replay only",
    }
    _write_json(output / "failure_replay.json", final, refuse_existing=True)
    _write_json(output / "progress.json", final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "check", "run"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--include-arrival13", action="store_true", default=True)
    args = parser.parse_args(argv)
    try:
        if args.phase == "preflight":
            run_preflight(args.output_dir)
        elif args.phase == "check":
            run_check(args.output_dir)
        else:
            result = run_replay(args.output_dir, include_arrival13=args.include_arrival13)
            return 0 if result.get("status") == "REPLAY_OK" else 2
    except Exception as exc:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "stage": STAGE, "phase": args.phase, "status": "FAILED",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            _write_json(args.output_dir / "progress.json", failure)
        except Exception:
            pass
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
