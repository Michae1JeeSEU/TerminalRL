"""Bounded, frozen P3 foundation repair comparison.

This module registers and executes the small development-only comparison
approved for the P3 foundation review.  It is deliberately separate from
the existing P3 evaluators: no optimizer, Teacher, held-out cell, checkpoint
write, or PPO admission path is reachable from this entry point.

The ``--check-only`` path validates the fixed source checkpoint and both
observation schemas without constructing an environment.  ``--smoke`` runs
one arm for two ticks only to validate serialization fields; it is always
reported as censored.  ``--run`` is the fixed eight-arm development matrix.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from terminal_marl.algo.service_context_runtime_v4_2 import (
    initialize_service_context_policy,
)
from terminal_marl.core.low_level_joint_action_v4_2 import (
    ACTION_SCHEMA_V4_2,
    HierarchicalActionV42,
    RELEASE_GATE,
    WAIT_GATE,
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.training import evaluate_p3_proposal_head_probe_v1 as probe
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as forced
from terminal_marl.training import terminal_lifecycle_metrics as lifecycle


STAGE = "P3_FOUNDATION_REPAIR_COMPARISON_V3_R2_20261002"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
CHECKPOINT_WRITES = False

SOURCE_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt"
)
SOURCE_CHECKPOINT_SHA256 = (
    "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
)
OUTPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_foundation_repair_v3_r2_20261002"
)

TRAJECTORY_SCHEMA_V2 = "executable_trajectory_v2"
TRAJECTORY_SCHEMA_V3 = "executable_trajectory_v3"
FROZEN_INFERENCE_TRANSFER = "frozen_v2_source_checkpoint_to_v3_runtime"
UNCERTIFIED_COUNTER_SOURCE = "joint_safety_fail_closed"

OBSERVATION_SCHEMA_88 = "phaseA_low_obs_v2_1_retry_state"
OBSERVATION_SCHEMA_90 = "phaseA_low_obs_v2_4_service_context_90d"
OBSERVATION_DIM_88 = 88
OBSERVATION_DIM_90 = 90
SHORTLIST_K = 3
REQUEST_LIMIT = 24
MAX_STEPS = 1800

PAIR_SPECS: tuple[dict[str, int | str], ...] = (
    {"cell_id": "cell_1", "env_seed": 877101, "policy_seed": 878201},
    {"cell_id": "cell_2", "env_seed": 877102, "policy_seed": 878202},
)

SOURCE_ARM_SPECS: tuple[dict[str, str], ...] = (
    {"arm_id": "legacy88", "kind": "source", "schema": "88", "execution": "legacy"},
    {"arm_id": "repaired88", "kind": "source", "schema": "88", "execution": "speed_timing_v1"},
    {
        "arm_id": "repaired90_zero_initialized",
        "kind": "source",
        "schema": "90",
        "execution": "speed_timing_v1",
    },
)
RULE_ARM_SPECS: tuple[dict[str, str], ...] = (
    {"arm_id": "nominal_rule_legacy88", "kind": "nominal_rule", "schema": "88", "execution": "legacy"},
    {"arm_id": "nominal_rule_repaired90", "kind": "nominal_rule", "schema": "90", "execution": "speed_timing_v1"},
)


class FoundationRepairProtocolError(ValueError):
    """Raised when the fixed comparison input or output contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FoundationRepairProtocolError(message)


def _jsonable(value: Any) -> Any:
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


def _local_checkpoint_path() -> Path:
    """Resolve the documented Syncthing mirror when the server path is absent."""

    if SOURCE_CHECKPOINT.is_file():
        return SOURCE_CHECKPOINT
    local_root = Path(__file__).resolve().parents[3] / "training_outputs" / "terminal"
    relative = "p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt"
    return local_root / relative


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise FoundationRepairProtocolError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _config_snapshot(config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(config):
        raw = dataclasses.asdict(config)
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raw = dict(vars(config))
    value = _jsonable(raw)
    _require(isinstance(value, dict), "config snapshot must be a mapping")
    return value


def _base_config() -> Any:
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    _require("dense24" in neutral_configs, "neutral dense24 config is missing")
    config = neutral_configs["dense24"]
    try:
        config = dataclasses.replace(config, trajectory_schema_version=TRAJECTORY_SCHEMA_V3)
    except TypeError:
        config = copy.copy(config)
        setattr(config, "trajectory_schema_version", TRAJECTORY_SCHEMA_V3)
    _require(str(getattr(config, "trajectory_schema_version", "")) == TRAJECTORY_SCHEMA_V3,
             "base config is not common executable trajectory V3")
    _require(str(getattr(config, "vehicle_execution_mode", "")) == "performance_constrained",
             "V3 comparison requires performance-constrained execution")
    _require(str(getattr(config, "action_schema_version", "")) == ACTION_SCHEMA_V4_2,
             "base config is not V4.2 hierarchical")
    _require(str(getattr(config, "student_decision_mask_mode", "legacy")) == "legacy",
             "base config student mask mode is not legacy")
    _require(getattr(config, "teacher_enabled", False) is False, "Teacher must be disabled")
    _require(getattr(config, "rule_flow_coordination_enabled", False) is False,
             "rule flow coordination must be disabled")
    _require(getattr(config, "high_level_enabled", False) is False,
             "high-level control must be disabled")
    return config


def _variant_config(base: Any, *, schema: str, execution: str) -> Any:
    _require(schema in {"88", "90"}, f"unknown observation schema arm: {schema}")
    _require(execution in {"legacy", "speed_timing_v1"}, f"unknown execution mode: {execution}")
    observation_schema = OBSERVATION_SCHEMA_88 if schema == "88" else OBSERVATION_SCHEMA_90
    try:
        config = dataclasses.replace(
            base,
            observation_schema_version=observation_schema,
            final_approach_execution_mode=execution,
        )
    except TypeError:
        config = copy.copy(base)
        setattr(config, "observation_schema_version", observation_schema)
        setattr(config, "final_approach_execution_mode", execution)
    _require(str(getattr(config, "observation_schema_version", "")) == observation_schema,
             "observation schema override was not applied")
    _require(str(getattr(config, "final_approach_execution_mode", "")) == execution,
             "final execution override was not applied")
    _require(str(getattr(config, "trajectory_schema_version", "")) == TRAJECTORY_SCHEMA_V3,
             "arm does not preserve common executable trajectory V3")
    return config


def _arm_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for pair in PAIR_SPECS:
        for arm in SOURCE_ARM_SPECS:
            specs.append({**dict(pair), **dict(arm)})
        if pair["cell_id"] == "cell_1":
            for arm in RULE_ARM_SPECS:
                specs.append({**dict(pair), **dict(arm)})
    _require(len(specs) == 8, f"fixed arm count changed: {len(specs)}")
    return specs


def _validate_source_metadata(metadata: Mapping[str, Any]) -> None:
    expected = {
        "observation_schema_version": OBSERVATION_SCHEMA_88,
        "observation_dimension": OBSERVATION_DIM_88,
        "action_schema_version": ACTION_SCHEMA_V4_2,
        "low_level_joint_action_count": 26,
        "low_level_policy_distribution": "hierarchical_gate_2_conditional_proposal_25",
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    _require(not mismatches, f"source checkpoint metadata mismatch: {mismatches}")


def _load_source_policy(config: Any) -> tuple[Any, dict[str, Any], Path]:
    _require(
        str(getattr(config, "trajectory_schema_version", "")) == TRAJECTORY_SCHEMA_V3,
        "source policy runtime config must target executable trajectory V3",
    )
    try:
        checkpoint_load_config = dataclasses.replace(
            config,
            trajectory_schema_version=TRAJECTORY_SCHEMA_V2,
        )
    except TypeError:
        checkpoint_load_config = copy.copy(config)
        setattr(checkpoint_load_config, "trajectory_schema_version", TRAJECTORY_SCHEMA_V2)
    _require(
        str(getattr(checkpoint_load_config, "trajectory_schema_version", ""))
        == TRAJECTORY_SCHEMA_V2,
        "source checkpoint load config is not executable trajectory V2",
    )
    checkpoint = _local_checkpoint_path()
    _require(checkpoint.is_file(), f"source checkpoint missing: {SOURCE_CHECKPOINT}")
    observed_sha = _sha256(checkpoint)
    _require(observed_sha.lower() == SOURCE_CHECKPOINT_SHA256.lower(),
             f"source checkpoint SHA mismatch: {observed_sha}")
    policy, payload = probe._load_policy(checkpoint, checkpoint_load_config)
    payload = dict(payload)
    metadata = payload.get("metadata")
    _require(isinstance(metadata, Mapping), "source checkpoint metadata missing")
    _validate_source_metadata(metadata)
    _require(
        metadata.get("trajectory_schema_version") == TRAJECTORY_SCHEMA_V2,
        "source checkpoint trajectory schema must be executable_trajectory_v2",
    )
    _require(int(getattr(policy, "observation_dim", -1)) == OBSERVATION_DIM_88,
             "loaded source policy is not 88D")
    return policy, payload, checkpoint


def _load_policy_for_arm(config: Any, schema: str) -> tuple[Any, dict[str, Any], Path, dict[str, Any]]:
    source_config = _variant_config(config, schema="88", execution="legacy")
    source_policy, payload, checkpoint = _load_source_policy(source_config)
    metadata = payload.get("metadata")
    _require(isinstance(metadata, Mapping), "source metadata missing after load")
    source_binding = {
        "source_checkpoint_trajectory_schema_version": TRAJECTORY_SCHEMA_V2,
        "source_load_trajectory_schema_version": TRAJECTORY_SCHEMA_V2,
        "runtime_environment_trajectory_schema_version": str(
            getattr(config, "trajectory_schema_version", "")
        ),
        "inference_transfer": FROZEN_INFERENCE_TRANSFER,
    }
    if schema == "88":
        return source_policy, payload, checkpoint, {
            "schema": OBSERVATION_SCHEMA_88,
            "dimension": OBSERVATION_DIM_88,
            "migration": None,
            **source_binding,
        }
    migrated, migration = initialize_service_context_policy(
        source_policy,
        source_checkpoint_metadata=metadata,
    )
    _require(int(getattr(migrated, "observation_dim", -1)) == OBSERVATION_DIM_90,
             "migrated policy is not 90D")
    return migrated, payload, checkpoint, {
        "schema": OBSERVATION_SCHEMA_90,
        "dimension": OBSERVATION_DIM_90,
        "migration": _jsonable(migration),
        **source_binding,
    }


def _observations(env: Any) -> dict[Any, np.ndarray]:
    method = getattr(env, "_observations", None)
    value = method() if callable(method) else getattr(env, "observations", {})
    _require(isinstance(value, Mapping), "environment observations must be a mapping")
    return {key: np.asarray(item, dtype=np.float32).copy() for key, item in value.items()}


def _uid_sort_key(uid: Any) -> tuple[int, str]:
    try:
        return (0, f"{int(uid):020d}")
    except (TypeError, ValueError):
        return (1, str(uid))


def _tiny_context(policy: Any, env: Any) -> dict[str, Any]:
    """Construct only the policy world/mask context; no hardcoded 88D helper."""

    paired._ensure_cpu(policy)
    observations = _observations(env)
    uids = sorted(observations, key=_uid_sort_key)
    dimension = int(getattr(policy, "observation_dim", 0))
    _require(dimension in {OBSERVATION_DIM_88, OBSERVATION_DIM_90},
             f"unsupported policy observation dimension: {dimension}")
    if uids:
        world = np.asarray([observations[uid] for uid in uids], dtype=np.float32).reshape(-1, dimension)
    else:
        world = np.empty((0, dimension), dtype=np.float32)
    actor_mask = np.asarray([forced._legacy_actor(env, uid) for uid in uids], dtype=bool)
    if not uids:
        return {
            "uids": [],
            "world": world,
            "actor_mask": actor_mask,
            "gate_probabilities": np.empty((0, 2), dtype=np.float64),
            "proposal_probabilities": np.empty((0, paired.PROPOSAL_COUNT), dtype=np.float64),
            "proposal_logits": np.empty((0, paired.PROPOSAL_COUNT), dtype=np.float64),
            "step": int(getattr(env, "step_count", 0)),
        }
    tensor = torch.as_tensor(world, dtype=torch.float32, device=getattr(policy, "device", "cpu"))
    model = getattr(policy, "model", policy)
    distributions = getattr(model, "distributions", None)
    _require(callable(distributions), "policy model must expose distributions(observations)")
    with torch.no_grad():
        gate_dist, proposal_dist = distributions(tensor)
        gate_probabilities = gate_dist.probs.detach().cpu().numpy().astype(np.float64, copy=True)
        proposal_probabilities = proposal_dist.probs.detach().cpu().numpy().astype(np.float64, copy=True)
        proposal_logits = proposal_dist.logits.detach().cpu().numpy().astype(np.float64, copy=True)
    _require(gate_probabilities.shape == (len(uids), 2), "gate probability shape mismatch")
    _require(proposal_probabilities.shape == (len(uids), paired.PROPOSAL_COUNT),
             "proposal probability shape mismatch")
    _require(proposal_logits.shape == (len(uids), paired.PROPOSAL_COUNT),
             "proposal logit shape mismatch")
    return {
        "uids": [str(uid) for uid in uids],
        "world": world,
        "actor_mask": actor_mask,
        "gate_probabilities": gate_probabilities,
        "proposal_probabilities": proposal_probabilities,
        "proposal_logits": proposal_logits,
        "step": int(getattr(env, "step_count", 0)),
    }


def _source_request(policy: Any, env: Any, policy_seed: int) -> dict[str, Any]:
    caller_rng = paired._rng_snapshot()
    try:
        context = _tiny_context(policy, env)
        actions: dict[str, np.ndarray] = {}
        shortlists: dict[str, tuple[np.ndarray, ...]] = {}
        rows: list[dict[str, Any]] = []
        step = int(context["step"])
        for index, uid in enumerate(context["uids"]):
            arrival_id = forced._arrival_id(env, uid)
            actor = bool(context["actor_mask"][index])
            gate = WAIT_GATE
            proposal: int | None = None
            gate_uniform = None
            proposal_uniform = None
            if actor:
                gate_uniform = paired.keyed_uniform(policy_seed, step, arrival_id, "gate")
                gate = paired._cdf_sample(context["gate_probabilities"][index], gate_uniform)
                if gate == RELEASE_GATE:
                    proposal_uniform = paired.keyed_uniform(policy_seed, step, arrival_id, "proposal")
                    proposal = paired._cdf_sample(context["proposal_probabilities"][index], proposal_uniform)
            if gate == RELEASE_GATE:
                action = HierarchicalActionV42(RELEASE_GATE, proposal).normalized.copy()
                shortlist = paired._shortlist_from_logits(proposal, context["proposal_logits"][index])
                shortlists[str(uid)] = shortlist
                shortlist_value = shortlist
            else:
                action = HierarchicalActionV42(WAIT_GATE, None).normalized.copy()
                shortlist_value = ()
            actions[str(uid)] = action
            rows.append({
                "uid": str(uid),
                "arrival_id": int(arrival_id),
                "actor_mask": actor,
                "gate": int(gate),
                "proposal": None if proposal is None else int(proposal),
                "gate_uniform": gate_uniform,
                "proposal_uniform": proposal_uniform,
                "action": action.tolist(),
                "shortlist": [np.asarray(item, dtype=np.float32).tolist() for item in shortlist_value],
                "policy_seed": int(policy_seed),
                "absolute_tick": step,
            })
        return {
            "actions": actions,
            "shortlists": shortlists,
            "rows": rows,
            "step": step,
            "keying": "policy_seed|absolute_tick|arrival_id|stream",
            "observation_dimension": int(getattr(policy, "observation_dim", -1)),
        }
    finally:
        paired._restore_rng(caller_rng)


_RULE_SHORTLIST = tuple(
    np.asarray(item, dtype=np.float32)
    for item in ((1.0, 0.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0))
)
_RULE_REQUESTED = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)


def _rule_request(env: Any) -> dict[str, Any]:
    observations = _observations(env)
    uids = sorted(observations, key=_uid_sort_key)
    actions: dict[str, np.ndarray] = {}
    shortlists: dict[str, tuple[np.ndarray, ...]] = {}
    rows: list[dict[str, Any]] = []
    step = int(getattr(env, "step_count", 0))
    requested_proposal = int(hierarchical_from_normalized_v4_2(_RULE_REQUESTED).proposal)
    for uid in uids:
        actor = bool(forced._legacy_actor(env, uid))
        arrival_id = forced._arrival_id(env, uid)
        if actor:
            action = _RULE_REQUESTED.copy()
            shortlist = tuple(item.copy() for item in _RULE_SHORTLIST)
            shortlists[str(uid)] = shortlist
            gate = RELEASE_GATE
            proposal = requested_proposal
        else:
            action = HierarchicalActionV42(WAIT_GATE, None).normalized.copy()
            shortlist = ()
            gate = WAIT_GATE
            proposal = None
        actions[str(uid)] = action
        rows.append({
            "uid": str(uid),
            "arrival_id": int(arrival_id),
            "actor_mask": actor,
            "gate": int(gate),
            "proposal": proposal,
            "gate_uniform": None,
            "proposal_uniform": None,
            "action": action.tolist(),
            "shortlist": [item.tolist() for item in shortlist],
            "policy_seed": None,
            "absolute_tick": step,
        })
    return {
        "actions": actions,
        "shortlists": shortlists,
        "rows": rows,
        "step": step,
        "keying": "fixed_nominal_rule",
        "observation_dimension": None,
    }


def _queue_size(env: Any) -> int:
    queue = getattr(getattr(env, "admission", None), "queue", None)
    return len(getattr(queue, "items", ()))


def _safe_plan_snapshot(plan: Any, uav: Any, release_telemetry: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = getattr(plan, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    selected_metadata = {
        str(key): _jsonable(value)
        for key, value in metadata.items()
        if str(key) in {
            "requested_speed_residual", "requested_merge_time_residual",
            "requested_target_speed_mps", "accepted_target_speed_mps",
            "requested_reservation_departure", "accepted_reservation_departure",
            "actual_entry_speed_mps", "timing_clamped", "timing_clamp_reason",
            "timing_constraints", "speed_clamp_reason", "scheduled_departure",
            "effective_delay_seconds", "scheduled_delay", "speed_residual_mps",
            "final_approach_execution_mode",
            "execution_mode", "trajectory_schema_version", "target_phase", "geometry_mode",
        }
    }
    return {
        "plan_id": str(getattr(plan, "plan_id", "")),
        "uid": getattr(uav, "uid", None),
        "arrival_id": getattr(uav, "arrival_id", None),
        "mode": getattr(plan, "mode", None),
        "start_time": getattr(plan, "start_time", None),
        "end_time": getattr(plan, "end_time", None),
        "trajectory_hash": getattr(plan, "trajectory_hash", None),
        "requested_accepted_actual_metadata": selected_metadata,
        "release_intent_telemetry": _jsonable(release_telemetry or {}),
    }


def _collect_new_final_plans(
    env: Any,
    seen_plan_ids: set[str],
    trace: list[dict[str, Any]],
) -> None:
    telemetry_by_plan: dict[str, Mapping[str, Any]] = {}
    for row in getattr(env, "last_release_telemetry", {}).values():
        if isinstance(row, Mapping) and row.get("trajectory_plan_id") is not None:
            telemetry_by_plan[str(row["trajectory_plan_id"])] = row
    objects = list(getattr(env, "active", {}).values())
    objects += list(getattr(env, "service_uavs", {}).values())
    objects += list(getattr(env, "completed_uavs", ()))
    for uav in objects:
        plan = getattr(uav, "committed_plan", None)
        if plan is None:
            continue
        if str(getattr(plan, "mode", "")) != "final_approach":
            continue
        plan_id = str(getattr(plan, "plan_id", ""))
        if not plan_id or plan_id in seen_plan_ids:
            continue
        seen_plan_ids.add(plan_id)
        trace.append(_safe_plan_snapshot(plan, uav, telemetry_by_plan.get(plan_id)))


def _telemetry_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value) - float(before.get(key, 0.0))
        for key, value in after.items()
        if isinstance(value, (int, float)) and isinstance(before.get(key, 0.0), (int, float))
    }


def _safety_report(env: Any, before: Mapping[str, Any]) -> dict[str, Any]:
    after = dict(getattr(env, "telemetry", {}))
    delta = _telemetry_delta(before, after)
    events = list(getattr(env, "conflict_events", ()) or ())
    try:
        report = dict(paired._safety_violation_details(delta, events))
        report["uncertified"] = (
            int(delta[UNCERTIFIED_COUNTER_SOURCE])
            if UNCERTIFIED_COUNTER_SOURCE in delta
            else None
        )
        report["uncertified_counter_source"] = UNCERTIFIED_COUNTER_SOURCE
        report["SafetyInvariantError"] = 0
        report["telemetry_delta"] = delta
        report["conflict_events"] = _jsonable(events)
        return report
    except Exception as exc:
        return {
            "safety_violation_status": "unavailable",
            "safety_violation": None,
            "uncertified": None,
            "uncertified_counter_source": UNCERTIFIED_COUNTER_SOURCE,
            "SafetyInvariantError": 0,
            "telemetry_delta": delta,
            "conflict_events": _jsonable(events),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _classify_outcomes(
    outcomes: Mapping[Any, Any],
    unresolved_ids: Sequence[int],
    requested_ids: Sequence[int],
) -> dict[str, int]:
    """Classify each requested ID once; missing rows remain unresolved."""

    counts: Counter[str] = Counter({
        "normal": 0,
        "emergency": 0,
        "failed": 0,
        "diverted": 0,
        "unresolved": 0,
    })
    unresolved = {int(value) for value in unresolved_ids}
    for arrival in {int(value) for value in requested_ids}:
        if arrival in unresolved:
            counts["unresolved"] += 1
            continue
        outcome = outcomes.get(str(arrival), outcomes.get(arrival))
        if outcome is None:
            counts["unresolved"] += 1
            continue
        counts[forced._terminal_category(outcome)] += 1
    return dict(counts)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _total_elapsed_seconds(
    wait: Mapping[str, Any],
    requested_ids: Sequence[int],
) -> tuple[float | None, str | None]:
    values = wait.get("request_elapsed_seconds_by_arrival")
    if not isinstance(values, Mapping):
        return None, "total_elapsed_seconds missing mapping"
    expected = {str(int(value)) for value in requested_ids}
    if expected - {str(key) for key in values}:
        return None, "total_elapsed_seconds missing arrival values"
    if not values and expected:
        return None, "total_elapsed_seconds empty for requested IDs"
    numbers: list[float] = []
    for key in sorted(values, key=str):
        value = values[key]
        if not _finite_number(value):
            return None, f"total_elapsed_seconds nonfinite value for arrival {key}"
        numbers.append(float(value))
    return float(sum(numbers)), None


def _total_energy_consumed(
    energy: Mapping[str, Any],
    requested_ids: Sequence[int],
) -> tuple[float | None, str | None]:
    rows = energy.get("rows")
    if not isinstance(rows, Mapping):
        return None, "total_energy_consumed missing energy rows"
    expected = {str(int(value)) for value in requested_ids}
    observed = {str(key) for key in rows}
    if expected - observed:
        return None, "total_energy_consumed missing requested energy rows"
    if not rows and expected:
        return None, "total_energy_consumed empty for requested IDs"
    numbers: list[float] = []
    for key in sorted(rows, key=str):
        row = rows[key]
        if not isinstance(row, Mapping) or not _finite_number(row.get("energy_consumed")):
            return None, f"total_energy_consumed missing/nonfinite value for arrival {key}"
        numbers.append(float(row["energy_consumed"]))
    return float(sum(numbers)), None


def _metrics_integrity(
    *,
    requested_ids: Sequence[int],
    outcomes: Mapping[Any, Any],
    unresolved_ids: Sequence[int],
    terminal_rows: Sequence[Mapping[str, Any]],
    energy: Mapping[str, Any],
    wait: Mapping[str, Any],
    status: str,
    observed_request_count: int | None = None,
    lifecycle_alignment: Mapping[str, Any] | None = None,
    total_elapsed_seconds: float | None = None,
    total_energy_consumed: float | None = None,
    elapsed_error: str | None = None,
    energy_error: str | None = None,
    lifecycle_error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    requested = {int(value) for value in requested_ids}
    resolved = {int(key) for key in outcomes}
    unresolved = {int(value) for value in unresolved_ids}
    terminal_ids = [int(row["arrival_id"]) for row in terminal_rows if row.get("arrival_id") is not None]
    duplicate_terminal_ids = sorted({value for value in terminal_ids if terminal_ids.count(value) > 1})
    errors: list[str] = []
    if duplicate_terminal_ids:
        errors.append("duplicate terminal rows")
    if requested != resolved | unresolved or not resolved.isdisjoint(unresolved):
        errors.append("ledger conservation mismatch")
    if not energy.get("coverage", {}).get("valid", False):
        errors.append("energy coverage mismatch")
    if lifecycle_alignment is not None and lifecycle_alignment.get("valid") is not True:
        errors.append("terminal rows do not align with ledger outcomes")
    if lifecycle_error is not None:
        errors.append("terminal row merge unavailable")
    if requested and total_elapsed_seconds is None:
        errors.append(elapsed_error or "total_elapsed_seconds unavailable")
    if requested and total_energy_consumed is None:
        errors.append(energy_error or "total_energy_consumed unavailable")
    if status == "completed":
        if set(terminal_ids) != resolved:
            errors.append("terminal rows do not match resolved ledger IDs")
        if not all(
            isinstance(wait.get(name), (int, float)) and math.isfinite(float(wait[name]))
            for name in (
                "all_request_cumulative_wait_seconds",
                "external_queue_integral_seconds",
                "internal_holding_sum_seconds",
            )
        ):
            errors.append("wait metrics incomplete")
        if observed_request_count is not None and int(observed_request_count) != REQUEST_LIMIT:
            errors.append(
                f"completed request count {int(observed_request_count)} != {REQUEST_LIMIT}"
            )
    return {
        "status": "valid" if not errors else "invalid",
        "technical_errors": errors,
        "requested_count": len(requested),
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "terminal_row_count": len(terminal_rows),
        "duplicate_terminal_ids": duplicate_terminal_ids,
        "lifecycle_alignment_valid": (
            None if lifecycle_alignment is None else lifecycle_alignment.get("valid")
        ),
        "total_elapsed_seconds": total_elapsed_seconds,
        "total_energy_consumed": total_energy_consumed,
        "business_metrics_independent_from_validity": True,
    }


def _tolerant_compare(left: Any, right: Any, *, tolerance: float = 1e-6) -> tuple[bool, float | None]:
    """Compare JSON-like values without treating tiny numeric drift as a mismatch."""

    if _finite_number(left) and _finite_number(right):
        difference = abs(float(left) - float(right))
        return difference <= float(tolerance), difference
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False, None
        equal = True
        maximum = 0.0
        for key in left:
            same, difference = _tolerant_compare(left[key], right[key], tolerance=tolerance)
            equal = equal and same
            if difference is None:
                if left[key] != right[key]:
                    maximum = math.inf
            else:
                maximum = max(maximum, difference)
        return equal, maximum
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False, None
        equal = True
        maximum = 0.0
        for left_item, right_item in zip(left, right):
            same, difference = _tolerant_compare(left_item, right_item, tolerance=tolerance)
            equal = equal and same
            if difference is None:
                if left_item != right_item:
                    maximum = math.inf
            else:
                maximum = max(maximum, difference)
        return equal, maximum
    return left == right, 0.0 if left == right else None


def _config_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    return {
        key: {"left": left.get(key), "right": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    }


def _action_tape_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    equal, maximum = _tolerant_compare(
        left.get("requested_action_groups", []),
        right.get("requested_action_groups", []),
        tolerance=1e-6,
    )
    return {"equal": equal, "max_abs_diff": maximum, "tolerance": 1e-6}


def _wait_elapsed_energy_view(arm: Mapping[str, Any]) -> dict[str, Any]:
    wait = arm.get("wait_metrics", {})
    energy = arm.get("energy_ledger", {})
    rows = energy.get("rows", {}) if isinstance(energy, Mapping) else {}
    energy_view = {
        str(key): {
            "energy_consumed": value.get("energy_consumed") if isinstance(value, Mapping) else None,
            "energy_remaining": value.get("energy_remaining") if isinstance(value, Mapping) else None,
            "energy_capacity": value.get("energy_capacity") if isinstance(value, Mapping) else None,
        }
        for key, value in rows.items()
    } if isinstance(rows, Mapping) else None
    return {
        "wait": {
            "all_request_cumulative_wait_seconds": wait.get("all_request_cumulative_wait_seconds"),
            "external_queue_integral_seconds": wait.get("external_queue_integral_seconds"),
            "internal_holding_sum_seconds": wait.get("internal_holding_sum_seconds"),
        } if isinstance(wait, Mapping) else None,
        "elapsed": wait.get("request_elapsed_seconds_by_arrival") if isinstance(wait, Mapping) else None,
        "energy": energy_view,
    }


def _final_trajectory_hashes(arm: Mapping[str, Any]) -> list[str]:
    trace = arm.get("final_requested_accepted_actual_trace", [])
    if not isinstance(trace, Sequence):
        return []
    return sorted(
        str(item.get("trajectory_hash"))
        for item in trace
        if isinstance(item, Mapping) and item.get("trajectory_hash") is not None
    )


def _business_scalar_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_summary = left.get("summary", {})
    right_summary = right.get("summary", {})
    names = (
        "normal", "emergency", "failed", "diverted", "unresolved",
        "total_wait_seconds", "total_elapsed_seconds", "total_energy_consumed",
    )
    delta: dict[str, Any] = {}
    for name in names:
        left_value = left_summary.get(name)
        right_value = right_summary.get(name)
        if _finite_number(left_value) and _finite_number(right_value):
            delta[name] = float(right_value) - float(left_value)
        else:
            delta[name] = None
    return delta


def _compare_zero_initialized_arms(
    repaired88: Mapping[str, Any],
    repaired90: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_config_fields = {"observation_schema_version", "final_approach_execution_mode"}
    config_left = repaired88.get("config", {})
    config_right = repaired90.get("config", {})
    config_diff = _config_differences(config_left, config_right)
    outcomes_left = repaired88.get("ledger", {}).get("outcomes", {})
    outcomes_right = repaired90.get("ledger", {}).get("outcomes", {})
    outcomes_equal = outcomes_left == outcomes_right
    metrics_equal, metrics_max_abs_diff = _tolerant_compare(
        _wait_elapsed_energy_view(repaired88),
        _wait_elapsed_energy_view(repaired90),
        tolerance=1e-6,
    )
    action_tape = _action_tape_difference(repaired88, repaired90)
    hashes_left = _final_trajectory_hashes(repaired88)
    hashes_right = _final_trajectory_hashes(repaired90)
    hashes_equal = hashes_left == hashes_right
    return {
        "left_arm": repaired88.get("arm_id"),
        "right_arm": repaired90.get("arm_id"),
        "request_identity_equal": repaired88.get("request_identity") == repaired90.get("request_identity"),
        "config_differences": config_diff,
        "config_differences_only_allowed_fields": set(config_diff).issubset(allowed_config_fields),
        "outcomes": {"equal": outcomes_equal},
        "wait_elapsed_energy": {
            "equal_with_tolerance": metrics_equal,
            "max_abs_diff": metrics_max_abs_diff,
            "tolerance": 1e-6,
        },
        "requested_action_tape": action_tape,
        "final_trajectory_hashes": {
            "equal": hashes_equal,
            "left": hashes_left,
            "right": hashes_right,
            "compared_independently": True,
        },
        # UUID plan IDs are intentionally excluded.  Hash equality is kept as
        # an independent observation and does not veto zero-init equivalence.
        "equivalent_excluding_hash": bool(
            outcomes_equal
            and metrics_equal
            and action_tape["equal"]
            and repaired88.get("request_identity") == repaired90.get("request_identity")
            and set(config_diff).issubset(allowed_config_fields)
        ),
    }


def _hard_safety_issue(arm: Mapping[str, Any]) -> bool:
    safety = arm.get("safety")
    if not isinstance(safety, Mapping):
        return True
    if safety.get("safety_violation_status") in {"violation", "unavailable"}:
        return True
    if safety.get("SafetyInvariantError") is None or int(safety.get("SafetyInvariantError", 0)) != 0:
        return True
    if safety.get("uncertified") is None or int(safety.get("uncertified", 0)) != 0:
        return True
    for key in ("collision_events", "nmac_events", "protected_floor_events", "hard_separation_events"):
        if safety.get(key):
            return True
    return False


def _hard_safety_violation(arm: Mapping[str, Any]) -> bool:
    """Return true only for an observed hard-safety violation.

    Missing metrics are handled as a matrix issue by ``_hard_safety_issue``;
    they do not stop a live arm before a violation has actually been observed.
    """

    safety = arm.get("safety")
    if not isinstance(safety, Mapping):
        return False
    if safety.get("safety_violation_status") == "violation":
        return True
    if _finite_number(safety.get("SafetyInvariantError")) and int(safety.get("SafetyInvariantError", 0)) != 0:
        return True
    if _finite_number(safety.get("uncertified")) and int(safety.get("uncertified", 0)) != 0:
        return True
    return any(
        bool(safety.get(key))
        for key in ("collision_events", "nmac_events", "protected_floor_events", "hard_separation_events")
    )


def _arm_has_issue(arm: Mapping[str, Any]) -> bool:
    validity = arm.get("validity")
    return bool(
        str(arm.get("status")) in {"error", "technical_error", "safety_violation", "not_run"}
        or arm.get("censored") is True
        or not isinstance(validity, Mapping)
        or validity.get("status") != "valid"
        or _hard_safety_issue(arm)
    )


def _not_run_arm(spec: Mapping[str, Any], base_config: Any, reason: str) -> dict[str, Any]:
    config = _variant_config(base_config, schema=str(spec["schema"]), execution=str(spec["execution"]))
    return {
        "stage": STAGE,
        "arm_id": str(spec["arm_id"]),
        "cell_id": str(spec["cell_id"]),
        "kind": str(spec["kind"]),
        "env_seed": int(spec["env_seed"]),
        "policy_seed": int(spec["policy_seed"]),
        "observation_schema": getattr(config, "observation_schema_version", None),
        "observation_dimension": OBSERVATION_DIM_88 if str(spec["schema"]) == "88" else OBSERVATION_DIM_90,
        "trajectory_schema_version": getattr(config, "trajectory_schema_version", None),
        "source_checkpoint_trajectory_schema_version": None,
        "source_load_trajectory_schema_version": None,
        "runtime_environment_trajectory_schema_version": getattr(
            config, "trajectory_schema_version", None
        ),
        "inference_transfer": "not_run",
        "final_approach_execution_mode": str(spec["execution"]),
        "status": "not_run",
        "stop_reason": reason,
        "censored": None,
        "progress": [],
        "summary": {},
        "request_identity": [],
        "requested_action_groups": [],
        "config": _config_snapshot(config),
        "ledger": {"outcomes": {}, "terminal_rows": []},
        "validity": {"status": "unavailable", "technical_errors": [reason]},
        "safety": {"safety_violation_status": "unavailable", "uncertified": None},
        "error": {"type": "NotRun", "message": reason},
        "frozen_flags": {
            "PPO_ALLOWED": PPO_ALLOWED,
            "heldout_used": HELDOUT_USED,
            "teacher_used": TEACHER_USED,
            "optimizer_used": OPTIMIZER_USED,
            "checkpoint_writes": CHECKPOINT_WRITES,
        },
    }


def _run_arm(
    spec: Mapping[str, Any],
    base_config: Any,
    *,
    max_steps: int = MAX_STEPS,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    arm_id = str(spec["arm_id"])
    schema = str(spec["schema"])
    execution = str(spec["execution"])
    config = _variant_config(base_config, schema=schema, execution=execution)
    policy = None
    payload: dict[str, Any] = {}
    checkpoint = None
    binding: dict[str, Any] = {
        "schema": OBSERVATION_SCHEMA_88 if schema == "88" else OBSERVATION_SCHEMA_90,
        "dimension": OBSERVATION_DIM_88 if schema == "88" else OBSERVATION_DIM_90,
        "migration": None,
        "source_checkpoint_trajectory_schema_version": None,
        "source_load_trajectory_schema_version": None,
        "runtime_environment_trajectory_schema_version": str(
            getattr(config, "trajectory_schema_version", "")
        ),
        "inference_transfer": "rule_or_unloaded_arm",
    }
    if str(spec["kind"]) == "source":
        policy, payload, checkpoint, binding = _load_policy_for_arm(config, schema)
    env = forced._CapturedTerminalEnv(config)
    env.reset(int(spec["env_seed"]))
    captured, first_seen = lifecycle._capture_terminal_rows(env)
    telemetry_before = dict(getattr(env, "telemetry", {}))
    progress: list[dict[str, Any]] = []
    request_groups: list[list[dict[str, Any]]] = []
    requested_action_groups: list[list[dict[str, Any]]] = []
    final_trace: list[dict[str, Any]] = []
    seen_plan_ids: set[str] = set()
    reward_total = 0.0
    reward_components: dict[str, float] = {}
    status = "running"
    stop_reason: str | None = None
    error: dict[str, Any] | None = None
    safety_exception_count = 0
    started = time.monotonic()
    group_index = len(getattr(env, "_captured_request_groups", ()))
    try:
        for tick in range(int(max_steps)):
            request = _rule_request(env) if str(spec["kind"]) == "nominal_rule" else _source_request(
                policy, env, int(spec["policy_seed"])
            )
            requested_action_groups.append(copy.deepcopy(request["rows"]))
            before_groups = len(getattr(env, "_captured_request_groups", ()))
            observations, rewards, _terminated, truncated, info = paired._call_step(
                env, request["actions"], request["shortlists"]
            )
            del observations
            new_groups = getattr(env, "_captured_request_groups", ())[before_groups:]
            request_groups.extend(copy.deepcopy(new_groups))
            if len(getattr(env, "_captured_request_rows", ())) > REQUEST_LIMIT:
                raise FoundationRepairProtocolError(
                    f"arrival process exceeded fixed request limit {REQUEST_LIMIT}"
                )
            reward_total += float(sum(rewards.values()) if isinstance(rewards, Mapping) else rewards)
            raw_components = info.get("reward_components", {}) if isinstance(info, Mapping) else {}
            if isinstance(raw_components, Mapping):
                for components in raw_components.values():
                    if isinstance(components, Mapping):
                        for name, value in components.items():
                            if name != "total":
                                reward_components[str(name)] = reward_components.get(str(name), 0.0) + float(value)
            _collect_new_final_plans(env, seen_plan_ids, final_trace)
            step_safety = _safety_report(env, telemetry_before)
            hard_safety_violation = _hard_safety_violation({"safety": step_safety})
            progress_row = {
                "tick": int(tick),
                "simulation_seconds": float(getattr(env, "now", 0.0)),
                "generated_requests": len(getattr(env, "_captured_request_rows", ())),
                "active": len(getattr(env, "active", {})),
                "queue": _queue_size(env),
                "service": len(getattr(env, "service_uavs", {})),
                "reward_total": float(reward_total),
                "safety_violation_status": step_safety.get("safety_violation_status"),
            }
            progress.append(progress_row)
            if progress_callback is not None:
                progress_callback(progress_row)
            if hard_safety_violation:
                status = "safety_violation"
                stop_reason = "hard_safety_violation"
                error = {"type": "HardSafetyViolation", "details": step_safety}
                break
            reason, natural = paired._continuation_stop_reason(env, truncated, tick + 1, max_steps)
            if reason is not None:
                stop_reason = str(reason)
                status = "completed" if natural else "censored"
                break
        if status == "running":
            status, stop_reason = "censored", "max_steps"
    except SafetyInvariantError as exc:
        safety_exception_count += 1
        status = "error"
        stop_reason = "SafetyInvariantError"
        error = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        status = "error"
        stop_reason = "exception"
        error = {"type": type(exc).__name__, "message": str(exc)}

    telemetry_after = dict(getattr(env, "telemetry", {}))
    safety = _safety_report(env, telemetry_before)
    safety["SafetyInvariantError"] = int(safety_exception_count)
    requested_ids = sorted(int(value) for value in getattr(env.ledger, "requested", set()))
    outcomes = {
        str(int(key)): getattr(value, "value", value)
        for key, value in getattr(env.ledger, "outcomes", {}).items()
    }
    unresolved_ids = sorted(int(value) for value in getattr(env.ledger, "unresolved", set()))
    lifecycle_error: dict[str, Any] | None = None
    try:
        raw_rows = lifecycle._merge_terminal_rows(captured, lifecycle._completed_rows(env, first_seen))
        terminal_rows = forced._merge_energy_terminal_rows(
            raw_rows,
            getattr(env, "energy_outcomes", ()),
            getattr(env, "_captured_request_rows", ()),
        )
        terminal_summary = lifecycle.summarize_terminal_rows(
            terminal_rows,
            low_soc_threshold=float(getattr(config, "soc_warning", 0.35)),
            requested_count=len(requested_ids),
            missing_arrival_ids=sorted(set(requested_ids) - {int(row["arrival_id"]) for row in terminal_rows}),
            enforce_unique=True,
        )
    except Exception as exc:
        lifecycle_error = {"type": type(exc).__name__, "message": str(exc)}
        # Preserve the raw rows and ledger IDs so duplicate/missing metrics
        # stay visible as validity errors rather than becoming failed rows.
        terminal_rows = [
            dict(row)
            for row in list(captured) + list(lifecycle._completed_rows(env, first_seen))
        ]
        terminal_summary = {"status": "unavailable", "error": lifecycle_error, "rows": terminal_rows}
        if error is None:
            error = lifecycle_error
    energy = forced._energy_ledger(env, requested_ids)
    wait = forced._wait_metrics(env, getattr(env, "_captured_request_rows", ()), terminal_rows, requested_ids)
    alignment = forced._lifecycle_alignment(
        requested_ids,
        outcomes,
        terminal_rows,
        unresolved_ids,
    )
    total_elapsed_seconds, elapsed_error = _total_elapsed_seconds(wait, requested_ids)
    total_energy_consumed, energy_error = _total_energy_consumed(energy, requested_ids)
    outcome_counts = _classify_outcomes(outcomes, unresolved_ids, requested_ids)
    integrity = _metrics_integrity(
        requested_ids=requested_ids,
        outcomes=outcomes,
        unresolved_ids=unresolved_ids,
        terminal_rows=terminal_rows,
        energy=energy,
        wait=wait,
        status=status,
        observed_request_count=len(getattr(env, "_captured_request_rows", ())),
        lifecycle_alignment=alignment,
        total_elapsed_seconds=total_elapsed_seconds,
        total_energy_consumed=total_energy_consumed,
        elapsed_error=elapsed_error,
        energy_error=energy_error,
        lifecycle_error=lifecycle_error,
    )
    result = {
        "stage": STAGE,
        "arm_id": arm_id,
        "cell_id": str(spec["cell_id"]),
        "kind": str(spec["kind"]),
        "env_seed": int(spec["env_seed"]),
        "policy_seed": None if spec.get("policy_seed") is None else int(spec["policy_seed"]),
        "observation_schema": binding.get("schema"),
        "observation_dimension": binding.get("dimension"),
        "trajectory_schema_version": getattr(config, "trajectory_schema_version", None),
        "source_checkpoint_trajectory_schema_version": binding.get(
            "source_checkpoint_trajectory_schema_version"
        ),
        "source_load_trajectory_schema_version": binding.get(
            "source_load_trajectory_schema_version"
        ),
        "runtime_environment_trajectory_schema_version": binding.get(
            "runtime_environment_trajectory_schema_version"
        ),
        "inference_transfer": binding.get("inference_transfer"),
        "final_approach_execution_mode": execution,
        "status": status,
        "stop_reason": stop_reason,
        "censored": True if status == "censored" else False if status == "completed" else None,
        "max_steps": int(max_steps),
        "request_limit": REQUEST_LIMIT,
        "generated_requests": len(getattr(env, "_captured_request_rows", ())),
        "request_count_contract": {
            "expected": REQUEST_LIMIT,
            "observed": len(getattr(env, "_captured_request_rows", ())),
            "exceeded": len(getattr(env, "_captured_request_rows", ())) > REQUEST_LIMIT,
            "completion_requires_exact_count": True,
        },
        "progress": progress,
        "summary": {
            "normal": int(outcome_counts["normal"]),
            "emergency": int(outcome_counts["emergency"]),
            "failed": int(outcome_counts["failed"]),
            "diverted": int(outcome_counts["diverted"]),
            "unresolved": int(outcome_counts["unresolved"]),
            "total_wait_seconds": wait.get("all_request_cumulative_wait_seconds"),
            "elapsed_seconds_by_arrival": wait.get("request_elapsed_seconds_by_arrival", {}),
            "total_elapsed_seconds": total_elapsed_seconds,
            "total_energy_consumed": total_energy_consumed,
            "energy": energy,
        },
        "reward_total": float(reward_total),
        "reward_components": reward_components,
        "request_identity": list(getattr(env, "_captured_request_rows", ())),
        "request_groups": request_groups,
        "requested_action_groups": requested_action_groups,
        "config": _config_snapshot(config),
        "ledger": {
            "requested_ids": requested_ids,
            "outcomes": outcomes,
            "unresolved_ids": unresolved_ids,
            "terminal_rows": terminal_rows,
            "terminal_alignment": alignment,
        },
        "terminal_summary": terminal_summary,
        "wait_metrics": wait,
        "energy_ledger": energy,
        "safety": safety,
        "final_requested_accepted_actual_trace": final_trace,
        "validity": integrity,
        "error": error,
        "lifecycle_error": lifecycle_error,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": None if checkpoint is None else _sha256(checkpoint),
        "checkpoint_metadata": _jsonable(payload.get("metadata", {})) if payload else None,
        "migration": binding.get("migration"),
        "frozen_flags": {
            "PPO_ALLOWED": PPO_ALLOWED,
            "heldout_used": HELDOUT_USED,
            "teacher_used": TEACHER_USED,
            "optimizer_used": OPTIMIZER_USED,
            "checkpoint_writes": CHECKPOINT_WRITES,
        },
        "elapsed_wall_seconds": float(time.monotonic() - started),
    }
    return _jsonable(result)


def _check_only() -> dict[str, Any]:
    base = _base_config()
    config88 = _variant_config(base, schema="88", execution="legacy")
    source_policy, payload, checkpoint, source_binding = _load_policy_for_arm(config88, "88")
    policy90, payload90, checkpoint90, migration = _load_policy_for_arm(
        _variant_config(base, schema="90", execution="speed_timing_v1"), "90"
    )
    del source_policy, policy90
    _require(checkpoint == checkpoint90, "schema arms resolved different checkpoint paths")
    _require(payload.get("metadata") == payload90.get("metadata"), "schema arms loaded different metadata")
    return {
        "stage": STAGE,
        "status": "CHECK_ONLY_OK",
        "trajectory_schema_version": TRAJECTORY_SCHEMA_V3,
        "common_physics_basis": {
            "trajectory_schema_version": TRAJECTORY_SCHEMA_V3,
            "vehicle_execution_mode": getattr(base, "vehicle_execution_mode", None),
            "vehicle_dynamics_schema_version": getattr(base, "vehicle_dynamics_schema_version", None),
            "physics_dt": getattr(base, "physics_dt", None),
            "safety_sampling_dt": getattr(base, "safety_sampling_dt", None),
        },
        "checkpoint_inference_binding": {
            "source_checkpoint_trajectory_schema_version": TRAJECTORY_SCHEMA_V2,
            "source_load_trajectory_schema_version": TRAJECTORY_SCHEMA_V2,
            "runtime_environment_trajectory_schema_version": TRAJECTORY_SCHEMA_V3,
            "inference_transfer": FROZEN_INFERENCE_TRANSFER,
            "source_checkpoint_is_v3_trained": False,
        },
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_metadata": _jsonable(payload.get("metadata", {})),
        "source_binding_88": source_binding,
        "migration_90": migration,
        "base_config": _config_snapshot(base),
        "config_88_legacy": _config_snapshot(config88),
        "config_88_repaired": _config_snapshot(_variant_config(base, schema="88", execution="speed_timing_v1")),
        "config_90_repaired": _config_snapshot(_variant_config(base, schema="90", execution="speed_timing_v1")),
        "arm_count": len(_arm_specs()),
        "frozen_flags": {
            "PPO_ALLOWED": PPO_ALLOWED,
            "heldout_used": HELDOUT_USED,
            "teacher_used": TEACHER_USED,
            "optimizer_used": OPTIMIZER_USED,
            "checkpoint_writes": CHECKPOINT_WRITES,
        },
    }


def _smoke() -> dict[str, Any]:
    base = _base_config()
    specs = [spec for spec in _arm_specs() if spec["cell_id"] == "cell_1"][:3]
    arms = [_run_arm(spec, base, max_steps=2) for spec in specs]
    for arm in arms:
        _require(arm.get("status") == "censored", "two-tick smoke arm must be censored")
        _require(arm.get("censored") is True, "two-tick smoke must be censored")
        _require(arm.get("error") is None, "two-tick smoke arm has an error")
        _require(arm.get("lifecycle_error") is None,
                 "two-tick smoke arm has a lifecycle error")
        validity = arm.get("validity")
        _require(
            isinstance(validity, Mapping) and validity.get("status") == "valid",
            "two-tick smoke arm validity is not valid",
        )
        if _hard_safety_issue(arm):
            raise FoundationRepairProtocolError(
                "two-tick smoke arm has a hard safety issue: "
                f"arm_id={arm.get('arm_id')}; "
                f"safety={json.dumps(_jsonable(arm.get('safety')), sort_keys=True)}"
            )
        _require(int(arm.get("observation_dimension", -1)) in {OBSERVATION_DIM_88, OBSERVATION_DIM_90},
                 "smoke observation dimension is missing")
        _require(isinstance(arm.get("progress"), list) and len(arm["progress"]) == 2,
                 "smoke must contain exactly two progress ticks")
        _require(_finite_number(arm.get("elapsed_wall_seconds")),
                 "smoke elapsed wall time is missing")
        _require(arm.get("trajectory_schema_version") == TRAJECTORY_SCHEMA_V3,
                 "smoke runtime trajectory schema is missing")
        _require(
            arm.get("source_checkpoint_trajectory_schema_version") == TRAJECTORY_SCHEMA_V2
            and arm.get("source_load_trajectory_schema_version") == TRAJECTORY_SCHEMA_V2,
            "smoke source/load trajectory schema binding is missing",
        )
        _require(
            arm.get("runtime_environment_trajectory_schema_version") == TRAJECTORY_SCHEMA_V3,
            "smoke runtime trajectory schema binding is missing",
        )
        _require(arm.get("inference_transfer") == FROZEN_INFERENCE_TRANSFER,
                 "smoke frozen inference transfer binding is missing")
        _require(arm.get("checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
                 "smoke source checkpoint SHA is missing or changed")
        checkpoint_metadata = arm.get("checkpoint_metadata")
        _require(
            isinstance(checkpoint_metadata, Mapping)
            and checkpoint_metadata.get("trajectory_schema_version") == TRAJECTORY_SCHEMA_V2,
            "smoke source checkpoint metadata schema is missing",
        )
        _require(isinstance(arm.get("energy_ledger"), Mapping), "smoke energy ledger is missing")
        coverage = arm["energy_ledger"].get("coverage")
        _require(isinstance(coverage, Mapping) and {
            "requested_ids", "missing_requested_ids", "unexpected_energy_ids", "valid"
        }.issubset(coverage),
                 "smoke energy coverage is missing")
        _require(coverage.get("valid") is True, "smoke energy coverage is invalid")
        summary = arm.get("summary")
        _require(
            isinstance(summary, Mapping)
            and {"total_wait_seconds", "total_elapsed_seconds", "total_energy_consumed"}.issubset(summary),
            "smoke wait/elapsed/energy summary is missing",
        )
        trace = arm.get("final_requested_accepted_actual_trace")
        _require(isinstance(trace, list),
                 "smoke final-plan trace is missing")
        for item in trace:
            _require(isinstance(item, Mapping) and {
                "plan_id", "mode", "trajectory_hash", "requested_accepted_actual_metadata",
                "release_intent_telemetry",
            }.issubset(item), "smoke final-plan trace fields are incomplete")
        json.dumps(_jsonable(arm), allow_nan=False)
    repaired88 = next(arm for arm in arms if arm["arm_id"] == "repaired88")
    repaired90 = next(arm for arm in arms if arm["arm_id"] == "repaired90_zero_initialized")
    equivalence = _compare_zero_initialized_arms(repaired88, repaired90)
    _require(equivalence["requested_action_tape"]["equal"],
             "zero-initialized 90D smoke action tape differs from repaired88")
    # Smoke is a serialization check; callers must not read it as a metric.
    return {
        "stage": STAGE,
        "status": "SMOKE_OK",
        "trajectory_schema_version": TRAJECTORY_SCHEMA_V3,
        "common_physics_basis": {
            "trajectory_schema_version": TRAJECTORY_SCHEMA_V3,
            "vehicle_execution_mode": getattr(base, "vehicle_execution_mode", None),
            "vehicle_dynamics_schema_version": getattr(base, "vehicle_dynamics_schema_version", None),
            "physics_dt": getattr(base, "physics_dt", None),
            "safety_sampling_dt": getattr(base, "safety_sampling_dt", None),
        },
        "smoke_only": True,
        "performance_interpretation": "not_available",
        "arm_count": len(arms),
        "arms": [
            {
                "arm_id": arm.get("arm_id"),
                "status": arm.get("status"),
                "censored": arm.get("censored"),
                "observation_schema": arm.get("observation_schema"),
                "observation_dimension": arm.get("observation_dimension"),
                "trajectory_schema_version": arm.get("trajectory_schema_version"),
                "source_checkpoint_trajectory_schema_version": arm.get(
                    "source_checkpoint_trajectory_schema_version"
                ),
                "source_load_trajectory_schema_version": arm.get(
                    "source_load_trajectory_schema_version"
                ),
                "runtime_environment_trajectory_schema_version": arm.get(
                    "runtime_environment_trajectory_schema_version"
                ),
                "inference_transfer": arm.get("inference_transfer"),
                "checkpoint_sha256": arm.get("checkpoint_sha256"),
                "ticks": len(arm.get("progress", [])),
                "energy_coverage": arm.get("energy_ledger", {}).get("coverage"),
                "safety": arm.get("safety"),
                "validity": arm.get("validity"),
                "error": arm.get("error"),
                "lifecycle_error": arm.get("lifecycle_error"),
                "elapsed_wall_seconds": arm.get("elapsed_wall_seconds"),
                "wait_elapsed_energy": {
                    "total_wait_seconds": arm.get("summary", {}).get("total_wait_seconds"),
                    "total_elapsed_seconds": arm.get("summary", {}).get("total_elapsed_seconds"),
                    "total_energy_consumed": arm.get("summary", {}).get("total_energy_consumed"),
                },
                "final_trace": arm.get("final_requested_accepted_actual_trace"),
                "final_trace_count": len(arm.get("final_requested_accepted_actual_trace", [])),
            }
            for arm in arms
        ],
        "repaired88_to_repaired90_zero_init": equivalence,
        "frozen_flags": {
            "PPO_ALLOWED": PPO_ALLOWED,
            "heldout_used": HELDOUT_USED,
            "teacher_used": TEACHER_USED,
            "optimizer_used": OPTIMIZER_USED,
            "checkpoint_writes": CHECKPOINT_WRITES,
        },
    }


def _run_matrix() -> dict[str, Any]:
    base = _base_config()
    arms: list[dict[str, Any]] = []
    root = OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    specs = _arm_specs()
    matrix_started = time.monotonic()
    stopped_early = False
    _write_json(root / "progress.json", {
        "stage": STAGE,
        "status": "running",
        "current_cell": None,
        "current_arm": None,
        "ticks": 0,
        "wall_seconds": 0.0,
        "completed_arms": 0,
        "total_arms": len(specs),
        "stopped_early": False,
    })
    for spec_index, spec in enumerate(specs):
        arm_dir = root / f"{spec['cell_id']}_{spec['arm_id']}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_started = time.monotonic()
        _write_json(arm_dir / "progress.json", {
            "stage": STAGE,
            "status": "running",
            "cell_id": spec["cell_id"],
            "arm_id": spec["arm_id"],
            "ticks": 0,
            "wall_seconds": 0.0,
        })

        def progress_callback(row: Mapping[str, Any], *, _spec=spec, _arm_dir=arm_dir,
                              _arm_started=arm_started) -> None:
            tick = int(row.get("tick", 0))
            if tick != 0 and (tick + 1) % 25 != 0:
                return
            elapsed = float(time.monotonic() - _arm_started)
            _write_json(_arm_dir / "progress.json", {
                "stage": STAGE,
                "status": "running",
                "cell_id": _spec["cell_id"],
                "arm_id": _spec["arm_id"],
                "ticks": tick + 1,
                "wall_seconds": elapsed,
                "progress": dict(row),
            })
            _write_json(root / "progress.json", {
                "stage": STAGE,
                "status": "running",
                "current_cell": _spec["cell_id"],
                "current_arm": _spec["arm_id"],
                "ticks": tick + 1,
                "wall_seconds": float(time.monotonic() - matrix_started),
                "completed_arms": len(arms),
                "total_arms": len(specs),
                "stopped_early": False,
            })

        try:
            arm = _run_arm(spec, base, progress_callback=progress_callback)
        except Exception as exc:
            arm = _not_run_arm(spec, base, f"runner exception: {type(exc).__name__}: {exc}")
        arms.append(arm)
        _write_json(
            arm_dir / "result.json",
            arm,
            refuse_existing=True,
        )
        _write_json(arm_dir / "progress.json", {
            "stage": STAGE,
            "status": arm.get("status"),
            "cell_id": spec["cell_id"],
            "arm_id": spec["arm_id"],
            "ticks": len(arm.get("progress", [])),
            "wall_seconds": float(time.monotonic() - arm_started),
            "stop_reason": arm.get("stop_reason"),
        })
        _write_json(root / "progress.json", {
            "stage": STAGE,
            "status": arm.get("status"),
            "current_cell": spec["cell_id"],
            "current_arm": spec["arm_id"],
            "ticks": len(arm.get("progress", [])),
            "wall_seconds": float(time.monotonic() - matrix_started),
            "completed_arms": len(arms),
            "total_arms": len(specs),
            "stopped_early": False,
        })
        if _hard_safety_violation(arm):
            stopped_early = True
            break

    if stopped_early:
        reason = "hard safety issue in completed arm"
        for spec in specs[len(arms):]:
            not_run = _not_run_arm(spec, base, reason)
            arms.append(not_run)
            _write_json(
                root / f"{spec['cell_id']}_{spec['arm_id']}" / "result.json",
                not_run,
                refuse_existing=True,
            )

    arm_by_key = {(str(arm.get("cell_id")), str(arm.get("arm_id"))): arm for arm in arms}
    comparisons: dict[str, Any] = {}
    for pair in PAIR_SPECS:
        cell_id = str(pair["cell_id"])
        legacy = arm_by_key.get((cell_id, "legacy88"))
        repaired88 = arm_by_key.get((cell_id, "repaired88"))
        repaired90 = arm_by_key.get((cell_id, "repaired90_zero_initialized"))
        cell_result: dict[str, Any] = {}
        if legacy is not None and repaired88 is not None:
            legacy_config_diff = _config_differences(
                legacy.get("config", {}), repaired88.get("config", {})
            )
            cell_result["legacy88_to_repaired88_business_scalar_delta"] = _business_scalar_delta(
                legacy, repaired88
            )
            cell_result["legacy88_to_repaired88_request_identity_equal"] = (
                legacy.get("request_identity") == repaired88.get("request_identity")
            )
            cell_result["legacy88_to_repaired88_config_differences"] = legacy_config_diff
            cell_result["legacy88_to_repaired88_config_differences_only_allowed_fields"] = set(
                legacy_config_diff
            ).issubset({"observation_schema_version", "final_approach_execution_mode"})
        else:
            cell_result["legacy88_to_repaired88_business_scalar_delta"] = None
        if repaired88 is not None and repaired90 is not None:
            cell_result["repaired88_to_repaired90_zero_init"] = _compare_zero_initialized_arms(
                repaired88, repaired90
            )
        else:
            cell_result["repaired88_to_repaired90_zero_init"] = None
        comparisons[cell_id] = cell_result

    hard_issue = any(_hard_safety_issue(arm) for arm in arms if arm.get("status") != "not_run")
    any_issue = any(_arm_has_issue(arm) for arm in arms) or stopped_early
    matrix_status = "COMPLETED" if not any_issue and not hard_issue and len(arms) == len(specs) else "COMPLETED_WITH_ISSUES"
    comparison = {
        "stage": STAGE,
        "status": matrix_status,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_V3,
        "common_physics_basis": {
            "trajectory_schema_version": TRAJECTORY_SCHEMA_V3,
            "vehicle_execution_mode": getattr(base, "vehicle_execution_mode", None),
            "vehicle_dynamics_schema_version": getattr(base, "vehicle_dynamics_schema_version", None),
            "physics_dt": getattr(base, "physics_dt", None),
            "safety_sampling_dt": getattr(base, "safety_sampling_dt", None),
        },
        "arm_count": len(arms),
        "executed_arm_count": sum(arm.get("status") != "not_run" for arm in arms),
        "stopped_early": stopped_early,
        "output_root": str(root),
        "comparisons_by_cell": comparisons,
        "comparison": [
            {
                "arm_id": arm.get("arm_id"),
                "cell_id": arm.get("cell_id"),
                "kind": arm.get("kind"),
                "status": arm.get("status"),
                "censored": arm.get("censored"),
                "summary": arm.get("summary"),
                "validity": arm.get("validity"),
                "safety": arm.get("safety"),
            }
            for arm in arms
        ],
        "arms": arms,
        "frozen_flags": {
            "PPO_ALLOWED": PPO_ALLOWED,
            "heldout_used": HELDOUT_USED,
            "teacher_used": TEACHER_USED,
            "optimizer_used": OPTIMIZER_USED,
            "checkpoint_writes": CHECKPOINT_WRITES,
        },
    }
    _write_json(root / "progress.json", {
        "stage": STAGE,
        "status": matrix_status,
        "current_cell": None,
        "current_arm": None,
        "ticks": 0,
        "wall_seconds": float(time.monotonic() - matrix_started),
        "completed_arms": sum(arm.get("status") != "not_run" for arm in arms),
        "total_arms": len(specs),
        "stopped_early": stopped_early,
    })
    _write_json(root / "comparison.json", comparison, refuse_existing=True)
    return comparison


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check_only:
            output = _check_only()
        elif args.smoke:
            output = _smoke()
        else:
            output = _run_matrix()
        if args.run:
            compact = {
                "stage": output.get("stage"),
                "status": output.get("status"),
                "arm_count": output.get("arm_count"),
                "executed_arm_count": output.get("executed_arm_count"),
                "stopped_early": output.get("stopped_early"),
                "output_root": output.get("output_root"),
                "comparisons_by_cell": output.get("comparisons_by_cell"),
            }
            print(json.dumps(_jsonable(compact), indent=2, sort_keys=True, allow_nan=False))
            return 0 if output.get("status") == "COMPLETED" else 2
        print(json.dumps(_jsonable(output), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (FoundationRepairProtocolError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": {
            "type": type(exc).__name__, "message": str(exc),
        }}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
