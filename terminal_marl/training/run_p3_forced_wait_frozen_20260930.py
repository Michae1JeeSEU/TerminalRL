"""Frozen, paired protocol for the opt-in V4.2 pre-action WAIT mask.

This entry point is a protocol-effect diagnostic.  It compares the historical
``legacy`` mask with ``pre_action_forced_wait`` under one frozen source model,
one neutral dense24 scenario, and keyed common random numbers.  It does not
train, load held-out data, call a Teacher, or update an optimizer.
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
from typing import Any

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import (
    HierarchicalActionV42,
    RELEASE_GATE,
    WAIT_GATE,
)
from terminal_marl.core.uav import MISSION_SERVICE_SECONDS, MissionType
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_followup_diagnostics as followup
from terminal_marl.training import terminal_lifecycle_metrics as lifecycle


STAGE = "P3_FORCED_WAIT_FROZEN_20260930_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
HISTORICAL_TAPE_REPLAY = False
OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/p3_followup_20260930/forced_wait_frozen"
)
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
OBSERVATION_DIM = 88
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"
SHORTLIST_K = 3
MAX_STEPS = 1800
ARMS = ("legacy", "pre_action_forced_wait")
ENV_SEEDS = tuple(int(value) for value in followup.EXPECTED_ENV_SEEDS)
POLICY_SEEDS = tuple(int(value) for value in (9303101, 9303102, 9303103, 9303104))
PAIR_SPECS = tuple(
    {
        "pair_id": int(index),
        "env_seed": int(env_seed),
        "keyed_policy_seed": int(policy_seed),
    }
    for index, (env_seed, policy_seed) in enumerate(
        ((env, policy) for env in ENV_SEEDS for policy in POLICY_SEEDS),
        start=1,
    )
)


class ForcedWaitProtocolError(ValueError):
    """Raised when the registered frozen protocol cannot be executed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ForcedWaitProtocolError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
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


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise ForcedWaitProtocolError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON artifact missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForcedWaitProtocolError(f"invalid JSON artifact: {path}") from exc
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _config_snapshot(config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(config):
        raw = dataclasses.asdict(config)
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raw = dict(vars(config))
    raw["student_decision_mask_mode"] = str(
        getattr(config, "student_decision_mask_mode", "legacy")
    )
    value = _jsonable(raw)
    _require(isinstance(value, dict), "config snapshot must be a mapping")
    return value


def _mask_config(config: Any, mode: str) -> Any:
    _require(mode in ARMS, f"unknown mask mode: {mode}")
    _require(hasattr(config, "student_decision_mask_mode"), "config mask mode is unavailable")
    try:
        value = dataclasses.replace(config, student_decision_mask_mode=mode)
    except TypeError:
        value = copy.copy(config)
        setattr(value, "student_decision_mask_mode", mode)
    _require(getattr(value, "student_decision_mask_mode") == mode, "mask mode copy failed")
    return value


def _request_row(request: Any) -> dict[str, Any]:
    """Capture fields from the actual ArrivalRequest object."""
    mission = getattr(request, "mission_type", None)
    mission_value = getattr(mission, "value", mission)
    try:
        service = float(MISSION_SERVICE_SECONDS[MissionType(str(mission_value))])
    except (KeyError, TypeError, ValueError):
        service = None
    return {
        "arrival_id": int(getattr(request, "arrival_id")),
        "request_time": float(getattr(request, "request_time")),
        "gate_id": int(getattr(request, "gate_id")),
        "initial_soc": float(getattr(request, "initial_soc")),
        "energy_capacity": float(getattr(request, "energy_capacity")),
        "base_priority": int(getattr(request, "base_priority")),
        "mission": None if mission_value is None else str(mission_value),
        "service_seconds": service,
        "service_seconds_source": "MISSION_SERVICE_SECONDS[mission_type]",
    }


class _CapturedTerminalEnv(TerminalParallelEnv):
    """Passive request capture that survives reset and warm-start creation."""

    def __init__(self, config: Any) -> None:
        self._captured_request_rows: list[dict[str, Any]] = []
        self._captured_request_groups: list[list[dict[str, Any]]] = []
        super().__init__(config)

    def reset(self, seed: int | None = None):  # type: ignore[override]
        self._captured_request_rows = []
        self._captured_request_groups = []
        result = super().reset(seed)
        self._install_request_capture()
        return result

    def _install_request_capture(self) -> None:
        original = self.arrivals.requests_until
        if getattr(original, "_forced_wait_capture", False):
            return

        def captured_requests(now: float):
            requests = original(now)
            rows = [_request_row(request) for request in requests]
            self._captured_request_groups.append(rows)
            self._captured_request_rows.extend(rows)
            return requests

        setattr(captured_requests, "_forced_wait_capture", True)
        self.arrivals.requests_until = captured_requests  # type: ignore[method-assign]

    def _warm_start(self) -> None:  # type: ignore[override]
        self._install_request_capture()
        return super()._warm_start()


def _validate_config(config: Any) -> None:
    _require(
        str(getattr(config, "observation_schema_version", "")) == OBSERVATION_SCHEMA,
        "protocol requires V2.1/88D observations",
    )
    _require(
        str(getattr(config, "action_schema_version", "")) == ACTION_SCHEMA,
        "protocol requires the V4.2 hierarchical action schema",
    )
    _require(int(getattr(config, "runtime_proposal_shortlist_k", -1)) == SHORTLIST_K,
             "protocol requires runtime proposal shortlist K=3")
    _require(getattr(config, "teacher_enabled", False) is False,
             "protocol forbids Teacher execution")
    _require(str(getattr(config, "scheduler_mode", "")) == "execution",
             "protocol requires execution scheduler mode")
    _require(getattr(config, "teacher_release_label_enabled", False) is False,
             "protocol forbids Teacher release labels")
    _require(getattr(config, "high_level_enabled", False) is False,
             "protocol requires fixed low-level environment")
    _require(getattr(config, "rule_flow_coordination_enabled", False) is False,
             "protocol forbids rule-flow substitution")
    _require(float(getattr(config, "max_simulation_seconds", 0.0)) > 0.0,
             "simulation horizon must be positive")
    _require(float(getattr(config, "dt", 0.0)) > 0.0, "dt must be positive")


def _source_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    binding = followup._public_source_binding(source)
    digest = str(binding.get("source_model_digest", ""))
    _require(digest == followup.SOURCE_WARMUP_MODEL_DIGEST,
             "source 3fe7+a82 model digest does not match the registered digest")
    return _jsonable(binding)


def _validate_preflight_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("stage") == STAGE, "preflight stage mismatch")
    _require(payload.get("status") == "PREFLIGHT_OK", "preflight status is not successful")
    _require(payload.get("PPO_ALLOWED") is False, "PPO must remain disabled")
    protocol = payload.get("protocol")
    _require(isinstance(protocol, Mapping), "preflight protocol metadata is missing")
    _require(str(protocol.get("source_checkpoint_sha256", "")).lower() == followup.SOURCE_CHECKPOINT_SHA256.lower(),
             "source checkpoint binding mismatch")
    _require(str(protocol.get("warmup_checkpoint_sha256", "")).lower() == followup.WARMUP_CHECKPOINT_SHA256.lower(),
             "warmup checkpoint binding mismatch")
    _require(str(protocol.get("source_warmup_model_digest", "")) == followup.SOURCE_WARMUP_MODEL_DIGEST,
             "source model digest binding mismatch")
    binding = payload.get("source_binding")
    _require(isinstance(binding, Mapping), "source binding is missing")
    _require(str(binding.get("source_model_digest", "")) == followup.SOURCE_WARMUP_MODEL_DIGEST,
             "source binding model digest mismatch")
    _require(str(binding.get("resolved_checkpoint", "")), "resolved source checkpoint is missing")
    _require(
        str(binding.get("warmup_checkpoint", "")),
        "resolved warmup checkpoint is missing",
    )
    _require(int(protocol.get("observation_dimension", -1)) == OBSERVATION_DIM,
             "observation dimension binding mismatch")
    runtime = payload.get("runtime")
    _require(isinstance(runtime, Mapping), "preflight runtime metadata is missing")
    snapshot = runtime.get("config_snapshot")
    _require(isinstance(snapshot, Mapping), "preflight config snapshot is missing")
    _require(str(snapshot.get("observation_schema_version", "")) == OBSERVATION_SCHEMA,
             "preflight config schema mismatch")
    _require(str(snapshot.get("action_schema_version", "")) == ACTION_SCHEMA,
             "preflight action schema mismatch")
    _require(int(snapshot.get("runtime_proposal_shortlist_k", -1)) == SHORTLIST_K,
             "preflight shortlist K mismatch")
    _require(tuple(int(value) for value in payload.get("fixed_env_seeds", ())) == ENV_SEEDS,
             "registered environment seed list changed")
    _require(tuple(int(value) for value in payload.get("fixed_keyed_policy_seeds", ())) == POLICY_SEEDS,
             "registered keyed policy seed list changed")
    _require(int(payload.get("expected_pairs", -1)) == len(PAIR_SPECS),
             "registered pair count changed")
    _require(payload.get("pair_grid") == [dict(item) for item in PAIR_SPECS],
             "registered pair grid changed")
    _require(payload.get("expected_arms") == len(PAIR_SPECS) * len(ARMS),
             "registered arm count changed")
    _require(protocol.get("mask_arms") in (None, list(ARMS)), "registered mask arms changed")
    _require(protocol.get("mask_arms") == list(ARMS), "registered mask arms changed")
    _require(int(protocol.get("max_steps", -1)) == MAX_STEPS,
             "registered max-step budget changed")


def _build_preflight() -> dict[str, Any]:
    """Build protocol metadata without creating an environment."""
    source_preflight = followup._build_preflight()
    _configs, neutral_configs, runtime_protocol = neutral_eval._build_neutral_inputs()
    _require("dense24" in neutral_configs, "dense24 neutral config is missing")
    config = neutral_configs["dense24"]
    _validate_config(config)
    _require(getattr(config, "student_decision_mask_mode", "legacy") == "legacy",
             "neutral dense24 base mode must remain legacy")
    source = followup._load_source_runtime(config)
    binding = _source_binding(source)
    legacy = _mask_config(config, "legacy")
    forced = _mask_config(config, "pre_action_forced_wait")
    legacy_snapshot = _config_snapshot(legacy)
    forced_snapshot = _config_snapshot(forced)
    changed = {
        key for key in set(legacy_snapshot) | set(forced_snapshot)
        if legacy_snapshot.get(key) != forced_snapshot.get(key)
    }
    _require(changed == {"student_decision_mask_mode"},
             f"mask arms changed unexpected config fields: {sorted(changed)}")
    return {
        "stage": STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": PPO_ALLOWED,
        "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED,
        "optimizer_used": OPTIMIZER_USED,
        "historical_tape_replay": HISTORICAL_TAPE_REPLAY,
        "protocol": {
            "observation_schema": OBSERVATION_SCHEMA,
            "observation_dimension": OBSERVATION_DIM,
            "action_schema": ACTION_SCHEMA,
            "runtime_proposal_shortlist_k": SHORTLIST_K,
            "source_checkpoint_sha256": followup.SOURCE_CHECKPOINT_SHA256,
            "warmup_checkpoint_sha256": followup.WARMUP_CHECKPOINT_SHA256,
            "source_warmup_model_digest": followup.SOURCE_WARMUP_MODEL_DIGEST,
            "gamma": followup.GAMMA,
            "gae_lambda": followup.GAE_LAMBDA,
            "reward_scale": followup.REWARD_SCALE,
            "crn_key": "stable_arrival_id|absolute_tick|stream",
            "crn_streams": ["gate", "proposal"],
            "mask_arms": list(ARMS),
            "max_steps": MAX_STEPS,
            "natural_terminal_or_budget": MAX_STEPS,
            "budget_cap_is_not_natural_terminal": True,
            "protocol_effect_only": True,
        },
        "source_binding": binding,
        "source_preflight": {
            "status": source_preflight.get("status"),
            "inputs": source_preflight.get("inputs"),
            "artifacts": source_preflight.get("artifacts"),
        },
        "runtime": {
            "versions": followup._runtime_versions(),
            "config_snapshot": _config_snapshot(config),
            "legacy_config_snapshot": legacy_snapshot,
            "forced_config_snapshot": forced_snapshot,
            "runtime_protocol": _jsonable(runtime_protocol),
            "code_manifest": followup._loaded_code_manifest(),
        },
        "fixed_env_seeds": list(ENV_SEEDS),
        "fixed_keyed_policy_seeds": list(POLICY_SEEDS),
        "pair_grid": [dict(item) for item in PAIR_SPECS],
        "expected_pairs": len(PAIR_SPECS),
        "expected_arms": len(PAIR_SPECS) * len(ARMS),
        "qualification": {
            "status": "not_a_qualification",
            "heldout_evaluation": False,
            "PPO_ALLOWED": False,
            "performance_promotion": False,
        },
    }


def _ensure_preflight(output: Path) -> dict[str, Any]:
    path = output / "preflight.json"
    if path.is_file():
        payload = _read_json(path)
        _validate_preflight_payload(payload)
        return payload
    payload = _build_preflight()
    _write_json(path, payload, refuse_existing=True)
    return payload


def _arrival_id(env: Any, uid: Any) -> int:
    actor = getattr(env, "active", {}).get(int(uid))
    if actor is None:
        actor = getattr(env, "active", {}).get(str(uid))
    _require(actor is not None, f"active actor missing for UID {uid}")
    value = getattr(actor, "arrival_id", None)
    _require(value is not None, f"arrival_id missing for active UID {uid}")
    return int(value)


def _legacy_actor(env: Any, uid: Any) -> bool:
    actor = getattr(env, "active", {}).get(int(uid))
    if actor is None:
        actor = getattr(env, "active", {}).get(str(uid))
    if actor is None:
        return False
    phase = getattr(getattr(actor, "phase", None), "value", getattr(actor, "phase", None))
    return bool(
        getattr(actor, "committed_plan", None) is None
        and phase in {"ring_outer", "ring_middle", "ring_inner"}
        and int(uid) not in getattr(env, "future_release_reservations", {})
    )


def _interface_for(actor: Any) -> str | None:
    role = getattr(getattr(actor, "ring_role", None), "value", getattr(actor, "ring_role", None))
    return {"outer": "o2m", "middle": "m2i", "inner": "i2f"}.get(role)


def _mask_counts(env: Any, uids: Sequence[Any], actor_mask: Sequence[bool]) -> dict[str, int]:
    affected = set()
    airspace = getattr(env, "airspace", None)
    method = getattr(airspace, "affected_interfaces", None)
    if callable(method):
        affected = set(method())
    counts = Counter(
        {
            "legacy_eligible": 0,
            "current_actor": 0,
            "forced_wait": 0,
            "retry_blocked": 0,
            "interface_blocked": 0,
            "soc_blocked": 0,
            "future_reservation_or_nonring": 0,
        }
    )
    for index, uid in enumerate(uids):
        base = _legacy_actor(env, uid)
        current = bool(actor_mask[index])
        if base:
            counts["legacy_eligible"] += 1
        if current:
            counts["current_actor"] += 1
        if base and not current:
            counts["forced_wait"] += 1
            actor = getattr(env, "active", {}).get(int(uid))
            if actor is None:
                continue
            if float(getattr(env, "now", 0.0)) < float(getattr(actor, "safety_retry_after", 0.0)):
                counts["retry_blocked"] += 1
            if _interface_for(actor) in affected:
                counts["interface_blocked"] += 1
            if float(getattr(actor, "soc", 0.0)) <= 0.0:
                counts["soc_blocked"] += 1
        elif not base:
            counts["future_reservation_or_nonring"] += 1
    return {str(key): int(value) for key, value in counts.items()}


def _keyed_request(policy: Any, env: Any, keyed_seed: int) -> dict[str, Any]:
    """Build one request using arrival-id keyed CDF draws only."""
    caller_rng = paired._rng_snapshot()
    try:
        context = paired._details_context(policy, env)
        actions: dict[str, np.ndarray] = {}
        shortlists: dict[str, tuple[np.ndarray, ...]] = {}
        rows: list[dict[str, Any]] = []
        step = int(context.get("step", getattr(env, "step_count", 0)))
        for index, uid in enumerate(context["uids"]):
            arrival_id = _arrival_id(env, uid)
            actor_mask = bool(context["actor_mask"][index])
            gate = WAIT_GATE
            proposal: int | None = None
            gate_u = None
            proposal_u = None
            if actor_mask:
                gate_u = paired.keyed_uniform(int(keyed_seed), step, arrival_id, "gate")
                gate = paired._cdf_sample(context["gate_probabilities"][index], gate_u)
                if gate == RELEASE_GATE:
                    proposal_u = paired.keyed_uniform(int(keyed_seed), step, arrival_id, "proposal")
                    proposal = paired._cdf_sample(context["proposal_probabilities"][index], proposal_u)
            if gate == RELEASE_GATE:
                action = HierarchicalActionV42(RELEASE_GATE, proposal).normalized.copy()
                shortlist = paired._shortlist_from_logits(
                    int(proposal), context["proposal_logits"][index]
                )
                shortlists[str(uid)] = shortlist
                shortlist_value = shortlist
            else:
                action = HierarchicalActionV42(WAIT_GATE, None).normalized.copy()
                shortlist_value = ()
            actions[str(uid)] = action
            rows.append({
                "uid": str(uid),
                "arrival_id": int(arrival_id),
                "actor_mask": actor_mask,
                "gate": int(gate),
                "proposal": None if proposal is None else int(proposal),
                "gate_uniform": gate_u,
                "proposal_uniform": proposal_u,
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "shortlist": [np.asarray(item, dtype=np.float32).tolist() for item in shortlist_value],
            })
        return {
            "actions": actions,
            "shortlists": shortlists,
            "rows": rows,
            "step": step,
            "keyed_seed": int(keyed_seed),
            "keying": "arrival_id|absolute_tick|stream",
        }
    finally:
        paired._restore_rng(caller_rng)


def _action_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("gate", WAIT_GATE)),
        None if row.get("proposal") is None else int(row["proposal"]),
        tuple(tuple(float(v) for v in item) for item in row.get("shortlist", ())),
    )


def _terminal_category(outcome: Any) -> str:
    value = getattr(outcome, "value", outcome)
    text = str(value)
    if text == "completed":
        return "normal"
    if text == "emergency_landed":
        return "emergency"
    if text.startswith("failed") or text == "invalid":
        return "failed"
    if text.startswith("diverted"):
        return "diverted"
    return "unresolved"


def _energy_ledger(env: Any, requested_ids: Sequence[int]) -> dict[str, Any]:
    """Combine permanent energy outcomes with every live/queue state once."""
    rows: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[str]] = {}

    def add(arrival: Any, source: str, **values: Any) -> None:
        if arrival is None:
            return
        key = str(int(arrival))
        row = rows.setdefault(key, {"arrival_id": int(arrival)})
        sources.setdefault(key, []).append(source)
        for name, value in values.items():
            if value is not None and name not in row:
                try:
                    row[name] = float(value)
                except (TypeError, ValueError):
                    row[name] = value

    for item in getattr(env, "energy_outcomes", ()):
        if isinstance(item, Mapping):
            add(item.get("arrival_id"), "energy_outcome", **dict(item))
    objects = list(getattr(env, "active", {}).values())
    objects += list(getattr(env, "service_uavs", {}).values())
    objects += list(getattr(env, "completed_uavs", ()))
    for item in objects:
        add(
            getattr(item, "arrival_id", None),
            "active_service_or_completed",
            energy_consumed=getattr(item, "energy_consumed", None),
            energy_remaining=getattr(item, "energy_remaining", None),
            energy_capacity=getattr(item, "energy_capacity", None),
            outcome=getattr(getattr(item, "terminal_outcome", None), "value", None),
        )
    for item in getattr(getattr(getattr(env, "admission", None), "queue", None), "items", ()):
        request = getattr(item, "request", item)
        add(
            getattr(request, "arrival_id", getattr(item, "arrival_id", None)),
            "external_queue",
            energy_consumed=getattr(item, "energy_consumed", None),
            energy_remaining=getattr(item, "energy_remaining", None),
            energy_capacity=getattr(request, "energy_capacity", None),
        )
    requested = {int(value) for value in requested_ids}
    observed = {int(value) for value in rows}
    return {
        "rows": rows,
        "sources_by_arrival": sources,
        "unique_arrival_ids": sorted(observed),
        "coverage": {
            "requested_ids": sorted(requested),
            "missing_requested_ids": sorted(requested - observed),
            "unexpected_energy_ids": sorted(observed - requested),
            "unique_arrival_ids": len(observed),
            "valid": not (requested - observed) and not (observed - requested),
        },
    }


def _merge_energy_terminal_rows(
    rows: Sequence[Mapping[str, Any]],
    energy_outcomes: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add only authoritative direct queue/service outcomes absent from hooks."""
    merged = {int(row["arrival_id"]): dict(row) for row in rows}
    request_time = {int(row["arrival_id"]): row.get("request_time") for row in request_rows}
    for source in energy_outcomes:
        if not isinstance(source, Mapping) or source.get("arrival_id") is None:
            continue
        arrival = int(source["arrival_id"])
        if arrival in merged:
            continue
        merged[arrival] = {
            "arrival_id": arrival,
            "uid": source.get("uid"),
            "request_time": request_time.get(arrival),
            "terminal_time": source.get("time"),
            "outcome": source.get("outcome"),
            "reason": source.get("reason"),
            "energy_remaining": source.get("energy_remaining"),
            "energy_consumed": source.get("energy_consumed"),
            "energy_capacity": source.get("energy_capacity"),
            "external_wait_time": None,
            "internal_wait_time": 0.0,
            "wait_seconds": None,
            "source": "energy_outcomes",
        }
    return [merged[key] for key in sorted(merged)]


def _wait_metrics(
    env: Any,
    request_rows: Sequence[Mapping[str, Any]],
    terminal_rows: Sequence[Mapping[str, Any]],
    requested_ids: Sequence[int],
) -> dict[str, Any]:
    """Use the permanent external integral plus one internal wait per request."""
    internal: dict[int, float] = {}
    for row in terminal_rows:
        value = row.get("internal_wait_time")
        if value is not None:
            try:
                internal[int(row["arrival_id"])] = float(value)
            except (TypeError, ValueError):
                pass
    objects = list(getattr(env, "active", {}).values())
    objects += list(getattr(env, "service_uavs", {}).values())
    objects += list(getattr(env, "completed_uavs", ()))
    for item in objects:
        arrival = getattr(item, "arrival_id", None)
        if arrival is not None:
            try:
                internal.setdefault(int(arrival), float(getattr(item, "internal_wait_time", 0.0)))
            except (TypeError, ValueError):
                pass
    for item in getattr(getattr(getattr(env, "admission", None), "queue", None), "items", ()):
        request = getattr(item, "request", item)
        arrival = getattr(request, "arrival_id", getattr(item, "arrival_id", None))
        if arrival is not None:
            internal.setdefault(int(arrival), 0.0)
    requested = {int(value) for value in requested_ids}
    missing_internal = sorted(requested - set(internal))
    external = float(getattr(env, "external_queue_uav_seconds", 0.0))
    internal_total = None if missing_internal else float(sum(internal.get(key, 0.0) for key in requested))
    times: dict[str, float | None] = {}
    for row in request_rows:
        arrival = int(row["arrival_id"])
        start = row.get("request_time")
        end = getattr(env, "resolution_times", {}).get(arrival, getattr(env, "now", None))
        try:
            times[str(arrival)] = float(end) - float(start)
        except (TypeError, ValueError):
            times[str(arrival)] = None
    return {
        "all_request_cumulative_wait_seconds": None if internal_total is None else external + internal_total,
        "external_queue_integral_seconds": external,
        "internal_holding_sum_seconds": internal_total,
        "time_scope": "all_requested_ids; failure/diversion-inclusive; external integral + internal holding",
        "missing_internal_wait_ids": missing_internal,
        "request_elapsed_seconds_by_arrival": times,
        "completed_only_condition": "terminal successful rows with observed wait fields",
    }


def _lifecycle_alignment(
    requested_ids: Sequence[int],
    outcomes: Mapping[str, Any],
    terminal_rows: Sequence[Mapping[str, Any]],
    unresolved_ids: Sequence[int],
) -> dict[str, Any]:
    requested = {int(value) for value in requested_ids}
    resolved = {int(key) for key in outcomes}
    unresolved = {int(value) for value in unresolved_ids}
    row_outcomes = {int(row["arrival_id"]): str(row.get("outcome")) for row in terminal_rows}
    mismatches = sorted(
        arrival for arrival in resolved
        if arrival in row_outcomes and str(outcomes[str(arrival)]) != row_outcomes[arrival]
    )
    return {
        "requested_ids": sorted(requested),
        "resolved_ids": sorted(resolved),
        "unresolved_ids": sorted(unresolved),
        "terminal_row_ids": sorted(row_outcomes),
        "terminal_row_outcome_mismatches": mismatches,
        "valid": (
            requested == resolved | unresolved
            and resolved.isdisjoint(unresolved)
            and set(row_outcomes) == resolved
            and not mismatches
        ),
    }


def _run_arm(
    policy: Any,
    config: Any,
    *,
    env_seed: int,
    keyed_seed: int,
    mode: str,
    progress: Any = None,
    pair_id: int | None = None,
) -> dict[str, Any]:
    _require(mode in ARMS, f"unknown arm mode: {mode}")
    caller_rng_before = paired._rng_snapshot()
    env = _CapturedTerminalEnv(config)
    env.reset(int(env_seed))
    captured, first_seen = lifecycle._capture_terminal_rows(env)
    reward_total = 0.0
    reward_components: dict[str, float] = {}
    tick_request_rows: list[list[dict[str, Any]]] = []
    mask_total: Counter[str] = Counter()
    telemetry_before = dict(getattr(env, "telemetry", {}))
    digest_before = paired._model_digest(policy)
    status, stop_reason, error = "running", None, None
    started = time.monotonic()
    group_index = len(env._captured_request_groups)
    try:
        for tick in range(MAX_STEPS):
            request = _keyed_request(policy, env, int(keyed_seed))
            rows = request["rows"]
            mask_total.update(_mask_counts(env, [row["uid"] for row in rows], [row["actor_mask"] for row in rows]))
            if progress is not None and (tick % 25 == 0 or tick == MAX_STEPS - 1):
                progress({"pair_id": pair_id, "mode": mode, "tick": tick, "now": float(env.now),
                          "request_count": len(env._captured_request_rows)})
            _observations, rewards, _terminated, truncated, info = paired._call_step(
                env, request["actions"], request["shortlists"]
            )
            new_groups = env._captured_request_groups[group_index:]
            tick_request_rows.extend(copy.deepcopy(new_groups))
            group_index = len(env._captured_request_groups)
            reward_total += float(sum(rewards.values()) if isinstance(rewards, Mapping) else rewards)
            raw_components = info.get("reward_components", {}) if isinstance(info, Mapping) else {}
            for components in raw_components.values() if isinstance(raw_components, Mapping) else ():
                if isinstance(components, Mapping):
                    for name, value in components.items():
                        if name != "total":
                            reward_components[str(name)] = reward_components.get(str(name), 0.0) + float(value)
            reason, natural = paired._continuation_stop_reason(env, truncated, tick + 1, MAX_STEPS)
            if reason is not None:
                stop_reason, status = str(reason), ("completed" if natural else "censored")
                break
        if status == "running":
            stop_reason, status = "max_steps", "censored"
    except Exception as exc:
        status, error = "error", {"type": type(exc).__name__, "message": str(exc)}

    digest_after = paired._model_digest(policy)
    caller_rng_after = paired._rng_snapshot()
    caller_rng_unchanged = paired._rng_equal(caller_rng_before, caller_rng_after)
    requests = [dict(row) for row in env._captured_request_rows]
    request_ids = [int(row["arrival_id"]) for row in requests]
    duplicate_request_ids = sorted({value for value in request_ids if request_ids.count(value) > 1})
    telemetry_after = dict(getattr(env, "telemetry", {}))
    telemetry_delta = {
        str(key): float(value) - float(telemetry_before.get(key, 0.0))
        for key, value in telemetry_after.items()
        if isinstance(value, (int, float)) and isinstance(telemetry_before.get(key, 0.0), (int, float))
    }
    conflict_events = list(getattr(env, "conflict_events", []) or [])
    safety_error = None
    try:
        safety = paired._safety_violation_details(telemetry_delta, conflict_events)
        _require(isinstance(safety, Mapping), "safety detail helper returned a non-mapping")
        _require({"safety_violation_status", "safety_violation", "collision_events", "nmac_events",
                  "protected_floor_events", "hard_separation_events"} <= set(safety),
                 "safety detail helper omitted required fields")
        safety = dict(safety)
    except Exception as exc:
        safety_error = {"type": type(exc).__name__, "message": str(exc)}
        safety = {"status": "unavailable", "error": safety_error}

    try:
        raw_rows = lifecycle._merge_terminal_rows(captured, lifecycle._completed_rows(env, first_seen))
        requested_ids = sorted(int(value) for value in getattr(env.ledger, "requested", set()))
        outcomes = {str(int(key)): getattr(value, "value", value)
                    for key, value in getattr(env.ledger, "outcomes", {}).items()}
        unresolved_ids = sorted(int(value) for value in getattr(env.ledger, "unresolved", set()))
        terminal_rows = _merge_energy_terminal_rows(raw_rows, getattr(env, "energy_outcomes", ()), requests)
        terminal_summary = lifecycle.summarize_terminal_rows(
            terminal_rows, low_soc_threshold=float(getattr(config, "soc_warning", 0.35)),
            requested_count=len(requested_ids),
            missing_arrival_ids=sorted(set(requested_ids) - {int(row["arrival_id"]) for row in terminal_rows}),
            enforce_unique=True,
        )
        conservation = _conservation(requested_ids=requested_ids, outcomes=outcomes, unresolved_ids=unresolved_ids)
        alignment = _lifecycle_alignment(requested_ids, outcomes, terminal_rows, unresolved_ids)
    except Exception as exc:
        requested_ids, outcomes, unresolved_ids = [], {}, []
        terminal_rows, terminal_summary = [], {"status": "unavailable", "error": str(exc), "rows": []}
        conservation, alignment = {"known": False, "error": str(exc)}, {"valid": False, "error": str(exc)}
    energy = _energy_ledger(env, requested_ids)
    wait = _wait_metrics(env, requests, terminal_rows, requested_ids)
    captured_request_ids = sorted(set(request_ids))
    ledger_request_ids = sorted(int(value) for value in requested_ids)
    arrivals_generated = getattr(getattr(env, "arrivals", None), "generated", None)
    request_capture_integrity = {
        "captured_ids": captured_request_ids,
        "ledger_requested_ids": ledger_request_ids,
        "captured_ids_equal_ledger": captured_request_ids == ledger_request_ids,
        "captured_count": len(requests),
        "arrivals_generated": None if arrivals_generated is None else int(arrivals_generated),
        "arrivals_generated_equals_captured": (
            arrivals_generated is not None and int(arrivals_generated) == len(requests)
        ),
    }
    elapsed_by_arrival = wait.get("request_elapsed_seconds_by_arrival", {})
    elapsed_integrity = bool(
        isinstance(elapsed_by_arrival, Mapping)
        and set(str(value) for value in requested_ids) == set(elapsed_by_arrival)
        and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in elapsed_by_arrival.values()
        )
    )
    wait_total = wait.get("all_request_cumulative_wait_seconds")
    wait_integrity = {
        "all_request_wait_observed_finite": (
            isinstance(wait_total, (int, float)) and math.isfinite(float(wait_total))
        ),
        "external_queue_integral_observed_finite": (
            isinstance(wait.get("external_queue_integral_seconds"), (int, float))
            and math.isfinite(float(wait["external_queue_integral_seconds"]))
        ),
        "internal_holding_observed_finite": (
            isinstance(wait.get("internal_holding_sum_seconds"), (int, float))
            and math.isfinite(float(wait["internal_holding_sum_seconds"]))
        ),
        "elapsed_by_requested_id_observed_finite": elapsed_integrity,
    }
    rejection_penalty = -float(telemetry_after.get("release_rejected", 0.0)) * float(
        getattr(config, "low_level_release_rejected_penalty", 0.0))
    outcome_counts = Counter({"normal": 0, "emergency": 0, "resolved": 0, "failed": 0, "diverted": 0,
                              "unresolved": len(unresolved_ids)})
    for value in outcomes.values():
        outcome_counts[_terminal_category(value)] += 1
    outcome_counts["resolved"] = len(outcomes)
    technical_reasons = []
    if duplicate_request_ids:
        technical_reasons.append(f"duplicate request IDs: {duplicate_request_ids}")
    if not conservation.get("requested_equals_resolved_union_unresolved", False):
        technical_reasons.append("ledger conservation failed")
    if not alignment.get("valid", False):
        technical_reasons.append("terminal rows do not align with ledger outcomes")
    if not energy.get("coverage", {}).get("valid", False):
        technical_reasons.append("energy ledger coverage failed")
    if status == "completed" and not request_capture_integrity["captured_ids_equal_ledger"]:
        technical_reasons.append("captured request IDs do not match ledger requested IDs")
    if status == "completed" and not request_capture_integrity["arrivals_generated_equals_captured"]:
        technical_reasons.append("arrival process generated count does not match captured requests")
    if status == "completed" and not all(wait_integrity.values()):
        technical_reasons.append("natural terminal wait or elapsed fields are incomplete or nonfinite")
    if safety_error:
        technical_reasons.append("safety detail helper unavailable")
    if digest_before != digest_after:
        technical_reasons.append("frozen model digest changed")
    if not caller_rng_unchanged:
        technical_reasons.append("caller RNG state changed")
    if technical_reasons and status not in {"error", "technical_error"}:
        status, error = "technical_error", {"type": "ProtocolIntegrityError", "reasons": technical_reasons}
    elif safety.get("safety_violation") and status not in {"error", "technical_error"}:
        status, error = "safety_violation", {"type": "HardSafetyViolation", "details": safety}
    admitted_ids: set[int] = set()
    for row in terminal_rows:
        if row.get("uid") is not None and row.get("arrival_id") is not None:
            admitted_ids.add(int(row["arrival_id"]))
    live_objects = list(getattr(env, "active", {}).values())
    live_objects += list(getattr(env, "service_uavs", {}).values())
    live_objects += list(getattr(env, "completed_uavs", ()))
    for item in live_objects:
        if getattr(item, "arrival_id", None) is not None:
            admitted_ids.add(int(item.arrival_id))
    telemetry_admitted_count = int(telemetry_after.get("arrivals_admitted", 0))
    return {
        "mode": mode, "env_seed": int(env_seed), "keyed_policy_seed": int(keyed_seed),
        "status": status, "stop_reason": stop_reason,
        "censored": None if status in {"error", "technical_error", "safety_violation"} else bool(status == "censored"),
        "budget_cap_steps": MAX_STEPS, "budget_cap_is_not_natural_terminal": True,
        "error": error, "elapsed_seconds": float(time.monotonic() - started),
        "ticks": int(getattr(env, "step_count", 0)), "simulation_seconds": float(getattr(env, "now", 0.0)),
        "generated_requests": len(requests), "generated_process_count": int(getattr(env.arrivals, "generated", 0)),
        "request_identity": requests, "request_identity_count": len(requests),
        "request_identity_unique": len(request_ids) == len(set(request_ids)),
        "request_groups": tick_request_rows,
        "ledger": {"requested_ids": requested_ids, "outcomes": outcomes, "unresolved_ids": unresolved_ids,
                   "conservation": conservation, "terminal_alignment": alignment,
                   "terminal_rows": terminal_rows},
        "terminal_summary": terminal_summary, "outcome_counts": dict(outcome_counts),
        "request_capture_integrity": request_capture_integrity,
        "wait_integrity": wait_integrity,
        "admitted_count": len(admitted_ids),
        "admitted_arrival_ids": sorted(admitted_ids),
        "telemetry_arrivals_admitted": telemetry_admitted_count,
        "normal_count": int(outcome_counts["normal"]), "emergency_count": int(outcome_counts["emergency"]),
        "resolved_count": int(len(outcomes)),
        "completed_count": int(outcome_counts["normal"] + outcome_counts["emergency"]),
        "reward_total": float(reward_total), "reward_components": reward_components,
        "raw_reward_excluding_rejection_penalty": float(reward_total - rejection_penalty),
        "release_rejection_penalty": float(rejection_penalty), "wait_metrics": wait,
        "energy_ledger": energy, "mask_counts": dict(mask_total), "telemetry": telemetry_after,
        "telemetry_delta": telemetry_delta, "safety": safety,
        "model_digest_before": digest_before, "model_digest_after": digest_after,
        "model_unchanged": bool(digest_before == digest_after),
        "caller_rng_unchanged": bool(caller_rng_unchanged),
        "keying": "stable arrival_id|absolute_tick|stream; UID only action-map key",
    }


def _conservation(*, requested_ids: Sequence[int], outcomes: Mapping[str, Any], unresolved_ids: Sequence[int]) -> dict[str, Any]:
    requested = {int(value) for value in requested_ids}
    resolved = {int(key) for key in outcomes}
    unresolved = {int(value) for value in unresolved_ids}
    return {
        "known": True,
        "requested_equals_resolved_union_unresolved": requested == resolved | unresolved,
        "resolved_unresolved_disjoint": resolved.isdisjoint(unresolved),
        "requested_count": len(requested),
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
    }


def _paired_difference(legacy_arm: Mapping[str, Any], forced_arm: Mapping[str, Any]) -> dict[str, Any]:
    request_match = legacy_arm.get("request_identity") == forced_arm.get("request_identity")
    left_conservation = legacy_arm.get("ledger", {}).get("conservation", {})
    right_conservation = forced_arm.get("ledger", {}).get("conservation", {})
    conservation_match = bool(
        left_conservation.get("known", False)
        and right_conservation.get("known", False)
        and left_conservation.get("requested_equals_resolved_union_unresolved") is True
        and right_conservation.get("requested_equals_resolved_union_unresolved") is True
        and left_conservation.get("resolved_unresolved_disjoint") is True
        and right_conservation.get("resolved_unresolved_disjoint") is True
        and legacy_arm.get("ledger", {}).get("requested_ids")
        == forced_arm.get("ledger", {}).get("requested_ids")
    )
    complete = bool(
        request_match
        and conservation_match
        and legacy_arm.get("status") == "completed"
        and forced_arm.get("status") == "completed"
        and legacy_arm.get("model_unchanged") is True
        and forced_arm.get("model_unchanged") is True
    )
    metrics = ("reward_total", "raw_reward_excluding_rejection_penalty", "release_rejection_penalty",
               "simulation_seconds", "ticks", "generated_requests", "admitted_count",
               "resolved_count", "completed_count")
    deltas = {
        key: (float(forced_arm[key]) - float(legacy_arm[key])) if complete else None
        for key in metrics
        if key in legacy_arm and key in forced_arm
    }
    return {
        "request_identity_match": request_match,
        "lifecycle_conservation_match": conservation_match,
        "both_natural_terminal_and_complete": complete,
        "unknown_difference_reason": None if complete else "censored_or_error_or_identity_or_conservation_mismatch",
        "delta_forced_minus_legacy": deltas,
    }


def _error_arm(pair: Mapping[str, Any], mode: str, exc: Exception) -> dict[str, Any]:
    return {
        "mode": mode, "env_seed": int(pair["env_seed"]),
        "keyed_policy_seed": int(pair["keyed_policy_seed"]),
        "status": "technical_error", "censored": None,
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "model_unchanged": None, "caller_rng_unchanged": None,
        "request_identity": [], "ledger": {"conservation": {"known": False}},
    }


def _pair_status(arms: Mapping[str, Mapping[str, Any]], difference: Mapping[str, Any]) -> str:
    statuses = {str(arm.get("status")) for arm in arms.values()}
    if statuses & {"error", "technical_error", "safety_violation", "not_run"}:
        return "PAIR_TECHNICAL_ERROR"
    if difference.get("request_identity_match") is not True or difference.get("lifecycle_conservation_match") is not True:
        return "PAIR_TECHNICAL_ERROR"
    if difference.get("both_natural_terminal_and_complete") is True:
        return "PAIR_OK"
    return "PAIR_INCOMPLETE"


def _run_pair(
    source: Mapping[str, Any],
    config: Any,
    pair: Mapping[str, Any],
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    source_policy = source["policy"]
    arms: dict[str, dict[str, Any]] = {}
    def progress(row: Mapping[str, Any]) -> None:
        if output is not None:
            _write_json(output / "progress.json", {
                "stage": STAGE, "phase": "arm", "status": "RUNNING", **dict(row),
            })
    for mode in ARMS:
        try:
            arm_policy = copy.deepcopy(source_policy)
            arms[mode] = _run_arm(
                arm_policy, _mask_config(config, mode),
                env_seed=int(pair["env_seed"]), keyed_seed=int(pair["keyed_policy_seed"]),
                mode=mode, progress=progress, pair_id=int(pair["pair_id"]),
            )
        except Exception as exc:
            arms[mode] = _error_arm(pair, mode, exc)
        if output is not None:
            _write_json(output / "arms" / f"pair_{int(pair['pair_id']):04d}_{mode}.json", arms[mode], refuse_existing=True)
        if arms[mode].get("status") in {"error", "technical_error", "safety_violation"}:
            break
    for mode in ARMS:
        arms.setdefault(mode, _error_arm(pair, mode, ForcedWaitProtocolError("arm not run after prior technical failure")))
        if output is not None and not (output / "arms" / f"pair_{int(pair['pair_id']):04d}_{mode}.json").exists():
            _write_json(output / "arms" / f"pair_{int(pair['pair_id']):04d}_{mode}.json", arms[mode], refuse_existing=True)
    result = {
        "stage": STAGE,
        "pair": dict(pair),
        "config_snapshot": _config_snapshot(config),
        "runtime_manifest": followup._loaded_code_manifest(),
        "arms": dict(arms),
        "source_model_digest": str(source.get("source_model_digest", "")),
        "PPO_ALLOWED": PPO_ALLOWED,
        "qualification": "protocol_effect_only; no promotion or performance claim",
    }
    result["paired_difference"] = _paired_difference(arms["legacy"], arms["pre_action_forced_wait"])
    result["pair_status"] = _pair_status(arms, result["paired_difference"])
    expected_requests = getattr(config, "total_requested_uavs", None)
    result["request_count_checks"] = {
        mode: {
            "expected": None if expected_requests is None else int(expected_requests),
            "observed": int(arms[mode].get("generated_requests", 0)),
            "required_when_natural": arms[mode].get("status") == "completed",
            "pass": expected_requests is None or arms[mode].get("status") != "completed"
            or int(arms[mode].get("generated_requests", -1)) == int(expected_requests),
        }
        for mode in ARMS
    }
    if not all(item["pass"] for item in result["request_count_checks"].values()):
        result["pair_status"] = "PAIR_TECHNICAL_ERROR"
        result["integrity_error"] = "natural arm did not capture the exact configured request count"
    return result


def _pair_path(output: Path, pair_id: int) -> Path:
    return output / "pairs" / f"pair_{int(pair_id):04d}.json"


def _runtime_source(config: Any) -> dict[str, Any]:
    source = followup._load_source_runtime(config)
    _source_binding(source)
    return source


def _validate_live_contract(preflight: Mapping[str, Any], config: Any, source: Mapping[str, Any]) -> None:
    """Reject reuse when live config/source/runtime differs from preflight."""
    _validate_config(config)
    stored = preflight.get("runtime", {})
    _require(_config_snapshot(config) == stored.get("config_snapshot"),
             "live config snapshot differs from stored preflight")
    live_binding = _source_binding(source)
    stored_binding = preflight.get("source_binding")
    _require(isinstance(stored_binding, Mapping), "stored source binding missing")
    for key in ("source_model_digest", "candidate_digest_before_warmup", "resolved_checkpoint", "warmup_checkpoint"):
        _require(live_binding.get(key) == stored_binding.get(key),
                 f"live source binding differs for {key}")
    stored_manifest = stored.get("code_manifest")
    live_manifest = followup._loaded_code_manifest()
    _require(stored_manifest == live_manifest, "live runtime code manifest differs from preflight")
    _require(tuple(preflight.get("fixed_env_seeds", ())) == ENV_SEEDS, "stored environment seeds changed")
    _require(tuple(preflight.get("fixed_keyed_policy_seeds", ())) == POLICY_SEEDS, "stored keyed seeds changed")
    _require(preflight.get("pair_grid") == [dict(item) for item in PAIR_SPECS], "stored pair grid changed")


def _manifest(output: Path, preflight: Mapping[str, Any], *, phase: str, pairs: Sequence[Mapping[str, Any]], status: str) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "phase": phase,
        "status": status,
        "PPO_ALLOWED": PPO_ALLOWED,
        "source_binding": preflight.get("source_binding"),
        "config_snapshot": preflight.get("runtime", {}).get("config_snapshot"),
        "mask_arms": list(ARMS),
        "pair_count": len(pairs),
        "pair_artifacts": [str(_pair_path(output, int(item["pair"]["pair_id"]))) for item in pairs],
        "qualification": "not_a_qualification",
    }


def run_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = _build_preflight()
    _write_json(output / "preflight.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK", "environment_created": False,
    })
    return payload


def run_pilot(output: Path) -> dict[str, Any]:
    preflight = _ensure_preflight(output)
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    config = neutral_configs["dense24"]
    _validate_config(config)
    source = _runtime_source(config)
    _validate_live_contract(preflight, config, source)
    pair = dict(PAIR_SPECS[0])
    path = _pair_path(output, int(pair["pair_id"]))
    _write_json(output / "progress.json", {"stage": STAGE, "phase": "pilot", "status": "RUNNING", "pair_id": 1})
    result = _run_pair(source, config, pair, output=output)
    _write_json(path, result, refuse_existing=True)
    pair_status = str(result.get("pair_status"))
    pilot_status = "PILOT_OK" if pair_status == "PAIR_OK" else (
        "PILOT_CENSORED" if pair_status == "PAIR_INCOMPLETE" else "PILOT_TECHNICAL_ERROR"
    )
    pilot = {"stage": STAGE, "phase": "pilot", "status": pilot_status, "pair": result}
    _write_json(output / "pilot.json", pilot, refuse_existing=True)
    _write_json(output / "manifest.json", _manifest(output, preflight, phase="pilot", pairs=[result], status=pilot_status))
    _write_json(output / "progress.json", {"stage": STAGE, "phase": "pilot", "status": pilot_status, "pair_id": 1})
    if pair_status == "PAIR_TECHNICAL_ERROR":
        raise ForcedWaitProtocolError("pilot stopped after a technical, safety, identity, or ledger error")
    return pilot


def run_full(output: Path) -> dict[str, Any]:
    preflight = _ensure_preflight(output)
    pilot_path = output / "pilot.json"
    pilot = _read_json(pilot_path)
    _require(pilot.get("status") == "PILOT_OK", "full phase requires an uncensored successful pilot")
    pilot_pair = pilot.get("pair")
    _require(isinstance(pilot_pair, Mapping), "pilot pair artifact is missing")
    _require(dict(pilot_pair.get("pair", {})) == dict(PAIR_SPECS[0]),
             "pilot is not the registered pair 0001")
    pilot_pair_path = _pair_path(output, 1)
    _require(pilot_pair_path.is_file(), "pilot pair artifact is missing")
    _require(_read_json(pilot_pair_path) == dict(pilot_pair),
             "pilot metadata and pair artifact differ")
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    config = neutral_configs["dense24"]
    _validate_config(config)
    source = _runtime_source(config)
    _validate_live_contract(preflight, config, source)
    _require(pilot_pair.get("source_model_digest") == source.get("source_model_digest"),
             "pilot source digest differs from live source")
    _require(pilot_pair.get("config_snapshot") == _config_snapshot(config),
             "pilot config snapshot differs from live config")
    _require(pilot_pair.get("runtime_manifest") == followup._loaded_code_manifest(),
             "pilot runtime manifest differs from live runtime")
    results: list[dict[str, Any]] = [dict(pilot_pair)]
    _write_json(output / "progress.json", {"stage": STAGE, "phase": "full", "status": "RUNNING", "processed": 1, "expected": len(PAIR_SPECS)})
    for pair in PAIR_SPECS[1:]:
        path = _pair_path(output, int(pair["pair_id"]))
        result = _run_pair(source, config, pair, output=output)
        _write_json(path, result, refuse_existing=True)
        results.append(result)
        if result.get("pair_status") == "PAIR_TECHNICAL_ERROR":
            _write_json(output / "progress.json", {
                "stage": STAGE, "phase": "full", "status": "TECHNICAL_ERROR",
                "processed": len(results), "expected": len(PAIR_SPECS),
                "pair_id": int(pair["pair_id"]),
            })
            raise ForcedWaitProtocolError("full phase stopped after a technical, safety, identity, or ledger error")
        _write_json(output / "progress.json", {
            "stage": STAGE, "phase": "full", "status": "RUNNING",
            "processed": len(results), "expected": len(PAIR_SPECS),
        })
    full = {
        "stage": STAGE,
        "phase": "full",
        "status": "FULL_OK" if all(item.get("pair_status") == "PAIR_OK" for item in results) else "FULL_INCOMPLETE",
        "PPO_ALLOWED": PPO_ALLOWED,
        "pilot_reused": True,
        "pair_count": len(results),
        "pair_ids": [int(item["pair"]["pair_id"]) for item in results],
        "paired_differences": [item.get("paired_difference") for item in results],
        "pair_statuses": {
            str(item["pair"]["pair_id"]): {
                mode: arm.get("status") for mode, arm in item.get("arms", {}).items()
            }
            for item in results
        },
        "qualification": "protocol_effect_only; no promotion or performance claim",
    }
    _write_json(output / "full.json", full, refuse_existing=True)
    _write_json(output / "manifest.json", _manifest(output, preflight, phase="full", pairs=results, status=full["status"]))
    _write_json(output / "progress.json", {
        "stage": STAGE, "phase": "full", "status": full["status"],
        "processed": len(results), "expected": len(PAIR_SPECS),
    })
    return full


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("preflight", "pilot", "full"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.phase == "preflight":
        run_preflight(args.output_dir)
    elif args.phase == "pilot":
        run_pilot(args.output_dir)
    else:
        run_full(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
