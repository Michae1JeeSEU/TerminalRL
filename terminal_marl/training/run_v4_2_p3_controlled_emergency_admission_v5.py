"""Bounded P3 admission pilot with controlled emergency outcomes.

This runner is an admission check for the next random-SOC PPO block.  It
keeps the V4.2 requested gate/proposal interface, accepts a certified
``emergency_landed`` outcome as a reportable performance result, and treats
energy depletion or an unrecognised terminal outcome as a technical failure.
It never authorizes P3 qualification or checkpoint promotion by itself.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.training import (
    diagnose_trained_dense30_v2 as diagnostic,
)
from terminal_marl.training import (
    diagnose_v4_2_p3_mixed_priority_soc_pilot_v2 as lifecycle,
)
from terminal_marl.training import (
    evaluate_random_soc_gate_repair_v1 as base,
)
from terminal_marl.training import (
    run_v4_2_p3_mixed_priority_soc_training_v3 as p3_legacy,
)
from terminal_marl.training import (
    run_v4_2_p3_mixed_priority_soc_training_v4 as p3_reference,
)
from terminal_marl.training import (
    train_random_soc_gate_repair_v1 as repair,
)
from terminal_marl.training.student_rollout_v4_2 import rollout


STAGE = "V4_2_P3_CONTROLLED_EMERGENCY_ADMISSION_V5"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/p3_controlled_emergency_admission_v5.yaml")
OBSERVATION_DIM = 88
CANONICAL_INITIAL_CHECKPOINT_SHA256 = (
    "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
)
REPAIR_CANDIDATE_CHECKPOINT_SHA256 = (
    "e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353"
)
TRAINABLE_PREFIXES = ("gate_adapter.", "critic_encoder.", "critic_head.")
FROZEN_PREFIXES = ("actor_encoder.", "gate_head.", "proposal_head.")
ALLOWED_OUTCOMES = frozenset({"completed", "emergency_landed", "failed_energy", "unresolved"})
SMOKE_SEED = 849000
ENTRY_SEEDS = (849101, 849102, 849103)
ENTRY_SCENARIOS = ("flow6", "flow12", "dense24")
PILOT_SEEDS = (849201, 849202, 849203, 849204, 849205, 849206)
PILOT_SCENARIOS = ("flow6", "dense24", "flow12", "dense24", "flow6", "dense24")
DEV_SEEDS = (849301, 849302, 849303)
DEV_SCENARIOS = ("flow6", "flow12", "dense24")
SMOKE_MAX_STEPS = 2


class V5InputError(ValueError):
    """Raised when a V5 input or report violates its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V5InputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return _jsonable(value.value)
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
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def protocol() -> dict[str, Any]:
    _require(PROTOCOL.is_file(), f"V5 protocol missing: {PROTOCOL}")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, Mapping), "V5 protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "V5 protocol id changed")
    _require(value.get("status") == "PREREGISTERED_BOUNDED_ADMISSION", "V5 protocol status changed")
    bindings = value.get("bindings")
    _require(isinstance(bindings, Mapping), "V5 bindings are missing")
    expected_bindings = {
        "canonical_initial_checkpoint_sha256": CANONICAL_INITIAL_CHECKPOINT_SHA256,
        "repair_candidate_checkpoint_sha256": REPAIR_CANDIDATE_CHECKPOINT_SHA256,
        "action_schema_version": "low_action_schema_v4_2_hierarchical",
        "observation_schema_version": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": OBSERVATION_DIM,
        "vehicle_profile_version": "eh216s_research_v2",
        "no_heldout_seeds": True,
        "no_historical_source_hashes": True,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"V5 binding changed: {key}")
    reward = value.get("reward")
    _require(isinstance(reward, Mapping), "V5 reward section is missing")
    for key, expected in {
        "low_level_priority_time_weight": 1.0,
        "low_level_soc_urgency_time_weight": 2.0,
        "low_level_soc_urgency_threshold": 0.65,
    }.items():
        _require(float(reward.get(key)) == expected, f"V5 reward binding changed: {key}")
    for key, expected in {
        "seed": SMOKE_SEED,
        "scenario": "dense24",
        "steps": SMOKE_MAX_STEPS,
        "full_horizon_seconds": 1800,
    }.items():
        _require(value.get("smoke", {}).get(key) == expected, f"V5 smoke binding changed: {key}")
    _require(tuple(value.get("entry", {}).get("seeds", ())) == ENTRY_SEEDS, "V5 entry seeds changed")
    _require(tuple(value.get("entry", {}).get("scenarios", ())) == ENTRY_SCENARIOS, "V5 entry scenarios changed")
    _require(tuple(value.get("pilot", {}).get("seeds", ())) == PILOT_SEEDS, "V5 pilot seeds changed")
    _require(tuple(value.get("pilot", {}).get("scenarios", ())) == PILOT_SCENARIOS, "V5 pilot scenarios changed")
    _require(tuple(value.get("dev", {}).get("seeds", ())) == DEV_SEEDS, "V5 dev seeds changed")
    _require(tuple(value.get("dev", {}).get("scenarios", ())) == DEV_SCENARIOS, "V5 dev scenarios changed")
    optimizer = value.get("optimizer")
    _require(isinstance(optimizer, Mapping), "V5 optimizer section is missing")
    for key, expected in {
        "learning_rate": 3e-5,
        "update_epochs": 2,
        "minibatch_size": 32,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.1,
        "value_coef": 0.5,
        "entropy_coef": 0.001,
        "target_kl": 0.01,
        "maximum_approx_kl": 0.05,
        "minimum_accepted_updates": 1,
        "maximum_rejected_updates": 1,
    }.items():
        _require(float(optimizer.get(key)) == float(expected), f"V5 optimizer binding changed: {key}")
    _require(value.get("reservation", {}).get("heldout") == [], "V5 must not reserve heldout seeds")
    authorization = value.get("authorization")
    _require(isinstance(authorization, Mapping), "V5 authorization section is missing")
    _require(authorization.get("FORMAL_P3_ALLOWED") is False, "V5 formal authorization changed")
    _require(authorization.get("P3_QUALIFIED") is False, "V5 qualification authorization changed")
    return dict(value)


def _checkpoint_sha(checkpoint_kind: str) -> str:
    value = str(checkpoint_kind)
    if value == "canonical":
        return CANONICAL_INITIAL_CHECKPOINT_SHA256
    if value == "repair":
        return REPAIR_CANDIDATE_CHECKPOINT_SHA256
    raise V5InputError("checkpoint_kind must be canonical or repair")


def _actual_checkpoint_sha(checkpoint: str | Path, checkpoint_kind: str) -> str:
    """Return the bytes on disk when available, otherwise the bound input SHA."""
    path = Path(checkpoint)
    return _sha256(path) if path.is_file() else _checkpoint_sha(checkpoint_kind)


def _protocol_digest(p: Mapping[str, Any]) -> str:
    payload = json.dumps(_jsonable(p), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _schema_binding(config: Any) -> dict[str, Any]:
    values = {
        "action_schema_version": getattr(config, "action_schema_version", None),
        "observation_schema_version": getattr(config, "observation_schema_version", None),
        "observation_dimension": OBSERVATION_DIM,
        "vehicle_profile_version": getattr(getattr(config, "vehicle_profile", None), "profile_version", None),
        "scheduler_mode": getattr(config, "scheduler_mode", None),
        "teacher_enabled": bool(getattr(config, "teacher_enabled", False)),
        "high_level_enabled": bool(getattr(config, "high_level_enabled", False)),
        "shield_enabled": bool(getattr(config, "shield_enabled", False)),
    }
    return values


def _schema_digest(config: Any) -> str:
    payload = json.dumps(_jsonable(_schema_binding(config)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_config(scenario: str, base_protocol: Mapping[str, Any]) -> Any:
    config = base._build_config(str(scenario), base_protocol)
    _require(str(getattr(config, "action_schema_version", "")) == "low_action_schema_v4_2_hierarchical",
             "V5 requires V4.2 hierarchical actions")
    _require(str(getattr(config, "observation_schema_version", "")) == "phaseA_low_obs_v2_1_retry_state",
             "V5 requires V2.1 observations")
    _require(not bool(getattr(config, "teacher_enabled", False)), "V5 requires Teacher disabled")
    _require(str(getattr(config, "scheduler_mode", "")) == "execution", "V5 requires execution scheduler")
    _require(bool(getattr(config, "shield_enabled", False)), "V5 requires the safety shield")
    _require(not bool(getattr(config, "high_level_enabled", False)), "V5 requires fixed geometry")
    return config


def _load_policy(checkpoint: Path, checkpoint_kind: str, config: Any, p: Mapping[str, Any]):
    expected = _checkpoint_sha(checkpoint_kind)
    if checkpoint_kind == "canonical":
        policy, metadata = diagnostic.load_canonical_initial_checkpoint(checkpoint, config)
    else:
        policy, metadata = repair.load_repaired_checkpoint(
            checkpoint, config, expected_sha256=REPAIR_CANDIDATE_CHECKPOINT_SHA256,
        )
    trainable: list[str] = []
    for name, parameter in policy.model.named_parameters():
        enabled = name.startswith(TRAINABLE_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    _require(bool(trainable), "V5 trainable parameter scope is empty")
    optimizer = p["optimizer"]
    policy.optimizer = torch.optim.Adam(
        [parameter for parameter in policy.model.parameters() if parameter.requires_grad],
        lr=float(optimizer["learning_rate"]),
    )
    _require(len(policy.optimizer.state) == 0, "V5 initial optimizer must be fresh")
    return policy, metadata, tuple(trainable), expected


def _policy_weight_digest(policy: Any) -> str:
    return diagnostic._policy_weight_digest(policy)


def _state_equal(left: Any, right: Any) -> bool:
    return p3_reference.state_equal(left, right)


def _frozen_state_equal(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    frozen = {key for key in before if not key.startswith(TRAINABLE_PREFIXES)}
    return frozen == {key for key in after if not key.startswith(TRAINABLE_PREFIXES)} and all(
        _state_equal(before[key], after[key]) for key in frozen
    )


def _outcome_data(summary: Mapping[str, Any], requests: int) -> dict[str, Any]:
    raw = summary.get("outcomes") or {}
    raw = raw if isinstance(raw, Mapping) else {}
    values = [str(getattr(value, "value", value)) for value in raw.values()]
    counts = Counter(values)
    unresolved = int(counts.get("unresolved", 0)) + max(0, int(requests) - len(raw))
    other = {
        key: int(value) for key, value in counts.items() if key not in ALLOWED_OUTCOMES
    }
    normal = int(counts.get("completed", 0))
    emergency = int(counts.get("emergency_landed", 0))
    failed_energy = int(counts.get("failed_energy", 0))
    resolved = normal + emergency
    return {
        "normal": normal,
        "emergency": emergency,
        "failed_energy": failed_energy,
        "unresolved": int(unresolved),
        "resolved": int(resolved),
        "other_failures": other,
        "other_failure_count": int(sum(other.values())),
        "recognized_outcomes": not other,
        "zero_failed_energy": failed_energy == 0,
        "zero_other_failures": not other,
        "full_resolution": bool(
            summary.get("terminal", False)
            and resolved == int(requests)
            and unresolved == 0
            and failed_energy == 0
            and not other
        ),
        "raw_outcomes": {str(key): str(getattr(value, "value", value)) for key, value in raw.items()},
    }


def _priority_outcomes(lifecycle_data: Mapping[str, Any] | None,
                       summary: Mapping[str, Any]) -> dict[str, Any]:
    """Join already-computed lifecycle rows to raw outcome keys by exact UID.

    The current lifecycle helper exposes priority and UID for completed rows,
    while the rollout summary exposes the raw outcome mapping.  No second
    environment replay is performed here; missing joins remain explicit.
    """

    result: dict[str, dict[str, int]] = {str(priority): {} for priority in (1, 3, 5)}
    raw = summary.get("outcomes") or {}
    raw = raw if isinstance(raw, Mapping) else {}
    raw_by_uid = {str(key): str(getattr(value, "value", value)) for key, value in raw.items()}
    rows = lifecycle_data.get("rows") if isinstance(lifecycle_data, Mapping) else None
    missing: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping) or row.get("uid") is None:
                continue
            uid = str(row["uid"])
            outcome = raw_by_uid.get(uid)
            if outcome is None:
                missing.append(uid)
                continue
            priority = str(int(row.get("base_priority", 0)))
            result.setdefault(priority, {})[outcome] = result.setdefault(priority, {}).get(outcome, 0) + 1
    else:
        missing = sorted(raw_by_uid)
    return {
        "by_priority": result,
        "priority_unavailable": bool(missing) or not isinstance(rows, list),
        "missing_uids": sorted(set(missing)),
        "known_raw_uids": sorted(raw_by_uid),
    }


def _outcome_gate_data(outcomes: Mapping[str, Any], *, require_full_resolution: bool) -> dict[str, bool]:
    values = {
        "recognized_outcomes": bool(outcomes.get("recognized_outcomes")),
        "zero_failed_energy": bool(outcomes.get("zero_failed_energy")),
        "zero_other_failures": bool(outcomes.get("zero_other_failures")),
    }
    if require_full_resolution:
        values["full_resolution"] = bool(outcomes.get("full_resolution"))
    return values


def _rollout_gates(policy: Any, config: Any, steps: list[Any], summary: Mapping[str, Any],
                   p: Mapping[str, Any], *, deterministic: bool, scenario: str,
                   require_full_resolution: bool) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    gates, diagnostics, shortlist, completion = p3_reference._rollout_gates(
        policy, config, steps, summary, p, deterministic=deterministic, scenario=scenario,
    )
    # The historical helper encodes the old all-normal performance criterion.
    # V5 replaces that single field with explicit outcome accounting.
    gates.pop("no_failed_outcomes", None)
    outcomes = _outcome_data(summary, int(config.total_requested_uavs))
    gates.update(_outcome_gate_data(outcomes, require_full_resolution=require_full_resolution))
    diagnostics = dict(diagnostics)
    diagnostics.update({
        "outcomes": outcomes,
        "full_horizon_seconds": float(config.max_simulation_seconds),
        "full_horizon_steps": int(round(float(config.max_simulation_seconds) / float(config.dt))),
    })
    return gates, diagnostics, shortlist, completion, outcomes


def _new_record(label: str, scenario: str, seed: int, checkpoint: Path, config: Any) -> dict[str, Any]:
    return {
        "label": label,
        "scenario": str(scenario),
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "status": "pending",
        "requests": int(config.total_requested_uavs),
        "completed": 0,
        "normal": 0,
        "emergency": 0,
        "failed_energy": 0,
        "unresolved": 0,
        "other_failures": {},
        "summary": None,
        "action_tape": None,
        "lifecycle": None,
        "gates": None,
        "diagnostics": None,
        "update": None,
        "update_gates": None,
        "checkpoint_after": None,
        "checkpoint_after_sha256": None,
        "weight_digest_before": None,
        "weight_digest_after": None,
        "weights_unchanged": None,
        "teacher_enabled": bool(getattr(config, "teacher_enabled", False)),
        "label_rows": 0,
        "PPO_ALLOWED": False,
        "ppo_called": False,
        "full_horizon_requested": True,
        "development_only": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
    }


def _record_rollout(
    *, policy: Any, config: Any, p: Mapping[str, Any], scenario: str, seed: int,
    output: Path, label: str, deterministic: bool, max_steps: int,
    require_full_resolution: bool,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    record = _new_record(label, scenario, seed, checkpoint or Path(output), config)
    weight_before = _policy_weight_digest(policy)
    record["weight_digest_before"] = weight_before
    try:
        steps, summary = rollout(
            policy, config, seed=int(seed), max_steps=int(max_steps), output=output,
            deterministic=bool(deterministic), collect_labels=False,
        )
        if not isinstance(summary, Mapping):
            raise V5InputError("student rollout summary must be a mapping")
        record["summary"] = dict(summary)
        reward = _reward_telemetry(summary)
        record["reward_telemetry"] = reward
        record["reward_component_totals"] = reward["component_totals"]
        record["reward_total"] = reward["total"]
        record["reward_components_consistent"] = reward["components_consistent"]
        if not reward["available"]:
            raise V5InputError("student rollout reward totals are missing or nonfinite")
        record["action_tape"] = summary.get("action_tape")
        record["completed"] = int(summary.get("completed", 0))
        record["label_rows"] = int(summary.get("label_rows", 0))
        record["PPO_ALLOWED"] = bool(summary.get("PPO_ALLOWED", False))
        gates, diagnostics, shortlist, completion, outcomes = _rollout_gates(
            policy, config, steps, summary, p, deterministic=bool(deterministic),
            scenario=scenario, require_full_resolution=require_full_resolution,
        )
        record["gates"] = gates
        record["diagnostics"] = diagnostics
        record["runtime_shortlist_checks"] = shortlist
        record["completion_fraction"] = completion
        record.update({
            "normal": outcomes["normal"],
            "emergency": outcomes["emergency"],
            "failed_energy": outcomes["failed_energy"],
            "unresolved": outcomes["unresolved"],
            "other_failures": outcomes["other_failures"],
            "outcomes": outcomes,
        })
        tape = summary.get("action_tape")
        if not isinstance(tape, str) or not tape:
            raise V5InputError("student rollout returned no action tape")
        try:
            record["lifecycle"] = lifecycle.lifecycle_metrics(
                config, tape, low_soc_threshold=float(p["attributes"]["low_soc_analysis_threshold"])
                if isinstance(p.get("attributes"), Mapping) else 0.65,
            )
        except Exception as exc:
            if require_full_resolution:
                raise
            record["lifecycle"] = None
            record["lifecycle_error"] = {"type": type(exc).__name__, "message": str(exc)}
        record["priority_outcomes"] = _priority_outcomes(record.get("lifecycle"), summary)
        record["full_horizon"] = int(max_steps) == int(round(float(config.max_simulation_seconds) / float(config.dt)))
        record["full_resolution"] = bool(outcomes["full_resolution"] and record["full_horizon"])
        record["_steps"] = steps
        record["status"] = "completed"
    except SafetyInvariantError as exc:
        record.update({
            "status": "failed",
            "safety_invariant_error": True,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
    except Exception as exc:
        record.update({
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
    finally:
        try:
            record["weight_digest_after"] = _policy_weight_digest(policy)
            record["weights_unchanged"] = bool(record["weight_digest_before"] == record["weight_digest_after"])
        except Exception as exc:
            record["weights_unchanged"] = False
            record.setdefault("error", {"type": type(exc).__name__, "message": str(exc)})
            record["status"] = "failed"
    hard_keys = tuple(key for key in (record.get("gates") or {}) if key not in {"full_resolution"})
    record["hard_gate_pass"] = bool(
        record["status"] == "completed"
        and record.get("weights_unchanged") is True
        and not bool(record.get("teacher_enabled"))
        and int(record.get("label_rows", 0)) == 0
        and not bool(record.get("PPO_ALLOWED"))
        and bool(record.get("gates"))
        and all(bool(record["gates"][key]) for key in hard_keys)
    )
    return record


def _reward_telemetry(summary: Mapping[str, Any]) -> dict[str, Any]:
    raw_totals = summary.get("reward_component_totals")
    totals = raw_totals if isinstance(raw_totals, Mapping) else {}
    finite_components = bool(totals) and all(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and bool(np.isfinite(float(value)))
        for value in totals.values()
    )
    raw_total = summary.get("reward_total")
    finite_total = (
        isinstance(raw_total, (int, float, np.integer, np.floating))
        and not isinstance(raw_total, bool)
        and bool(np.isfinite(float(raw_total)))
    )
    return {
        "component_totals": {str(key): float(value) for key, value in totals.items()}
        if finite_components else {},
        "total": float(raw_total) if finite_total else None,
        "components_finite": finite_components,
        "total_finite": finite_total,
        "components_consistent": bool(summary.get("reward_components_consistent") is True),
        "available": bool(finite_components and finite_total),
    }


def _smoke_contract(record: Mapping[str, Any], steps: list[Any], *, expected_steps: int) -> dict[str, bool]:
    worlds_finite_88 = True
    stochastic_steps = True
    for step in steps:
        world = np.asarray(getattr(step, "world", np.asarray([])))
        worlds_finite_88 = worlds_finite_88 and bool(
            world.ndim >= 2 and world.shape[-1] == OBSERVATION_DIM and np.isfinite(world).all()
        )
        stochastic_steps = stochastic_steps and bool(getattr(step, "stochastic_behavior", False))
    tape_rows = 0
    tape_path = record.get("action_tape")
    try:
        payload = json.loads(Path(str(tape_path)).read_text(encoding="utf-8"))
        rows = payload.get("action_tape", payload.get("records", []))
        tape_rows = len(rows) if isinstance(rows, list) else 0
    except Exception:
        tape_rows = 0
    summary = record.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    safety = summary.get("safety")
    safety_zero = isinstance(safety, Mapping) and all(
        isinstance(value, (int, float, np.integer, np.floating)) and float(value) == 0.0
        for value in safety.values()
    )
    reward = record.get("reward_telemetry")
    reward = reward if isinstance(reward, Mapping) else {}
    return {
        "exact_step_count": len(steps) == int(expected_steps),
        "finite_88d_observations": bool(worlds_finite_88),
        "stochastic_behavior": bool(stochastic_steps),
        "action_tape_rows": tape_rows == int(expected_steps),
        "teacher_disabled": record.get("teacher_enabled") is False,
        "no_label_rows": int(record.get("label_rows", 0)) == 0,
        "rollout_PPO_disabled": record.get("PPO_ALLOWED") is False,
        "safety_counters_zero": bool(safety_zero and not record.get("safety_invariant_error", False)),
        "weights_unchanged": record.get("weights_unchanged") is True,
        "reward_totals_finite": bool(reward.get("components_finite") and reward.get("total_finite")),
        "reward_components_consistent": reward.get("components_consistent") is True,
    }


def _entry_report_binding(config: Any, checkpoint_kind: str, checkpoint_sha: str, *,
                          actual_config_snapshots: Mapping[str, Any] | None = None,
                          p: Mapping[str, Any] | None = None) -> dict[str, Any]:
    binding = {
        "checkpoint_kind": str(checkpoint_kind),
        "selected_checkpoint_sha256": str(checkpoint_sha),
        "schema_binding": _schema_binding(config),
        "schema_digest": _schema_digest(config),
    }
    if actual_config_snapshots is not None:
        binding["actual_config_snapshots"] = _jsonable(actual_config_snapshots)
    if p is not None:
        binding["protocol_sha256"] = _protocol_digest(p)
        binding["optimizer_boundary"] = _jsonable(p.get("optimizer", {}))
    return binding


def validate_entry_report(report: Mapping[str, Any], *, config: Any,
                          checkpoint_kind: str, checkpoint_sha: str,
                          p: Mapping[str, Any] | None = None,
                          expected_config_snapshots: Mapping[str, Any] | None = None) -> None:
    _require(report.get("stage") == "V5_ENTRY", "entry report stage is not V5_ENTRY")
    _require(report.get("PASS") is True, "entry report did not pass")
    _require(report.get("selected_checkpoint_sha256") == checkpoint_sha,
             "entry report selected checkpoint SHA mismatch")
    _require(report.get("selected_checkpoint_kind") == checkpoint_kind,
             "entry report checkpoint kind mismatch")
    _require(report.get("schema_digest") == _schema_digest(config),
             "entry report configuration/schema binding mismatch")
    _require(report.get("schema_binding") == _schema_binding(config),
             "entry report schema fields changed")
    if p is not None:
        _require(report.get("PPO_ENTRY_ALLOWED") is True,
                 "entry report does not authorize the bounded pilot")
        _require(report.get("bounded_pilot_training_authorized") is True,
                 "entry report pilot authorization binding is missing")
        _require(report.get("protocol_sha256") == _protocol_digest(p),
                 "entry report protocol binding mismatch")
        _require(report.get("optimizer_boundary") == _jsonable(p.get("optimizer", {})),
                 "entry report optimizer boundary mismatch")
    if expected_config_snapshots is not None:
        _require(report.get("actual_config_snapshots") == _jsonable(expected_config_snapshots),
                 "entry report actual configuration snapshots changed")


def _persist_progress(output: Path, stage: str, records: list[Mapping[str, Any]], *, stopped: bool = False,
                      reason: str | None = None) -> None:
    _write_json(output / "progress.json", {
        "stage": stage,
        "completed_records": len(records),
        "records": records,
        "stopped": bool(stopped),
        "stop_reason": reason,
    })


def _load_source(checkpoint: Path, checkpoint_kind: str, config: Any, p: Mapping[str, Any]):
    return _load_policy(checkpoint, checkpoint_kind, config, p)


def _entry_stage(*, output: Path, checkpoint: Path, checkpoint_kind: str,
                 p: Mapping[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    base_protocol = base._load_yaml_protocol()
    entry_configs = {
        scenario: _build_config(scenario, base_protocol) for scenario in ENTRY_SCENARIOS
    }
    actual_config_snapshots = {
        scenario: base._config_snapshot(config) for scenario, config in entry_configs.items()
    }
    first_config = entry_configs[ENTRY_SCENARIOS[0]]
    policy, metadata, trainable, checkpoint_sha = _load_source(checkpoint, checkpoint_kind, first_config, p)
    records: list[dict[str, Any]] = []
    stopped = False
    stop_reason = None
    for index, (seed, scenario) in enumerate(zip(ENTRY_SEEDS, ENTRY_SCENARIOS)):
        config = entry_configs[scenario]
        max_steps = int(round(float(config.max_simulation_seconds) / float(config.dt)))
        record = _record_rollout(
            policy=policy, config=config, p=p, scenario=scenario, seed=seed,
            output=output / f"{scenario}_{seed}", label="entry",
            deterministic=False, max_steps=max_steps, checkpoint=checkpoint,
            require_full_resolution=True,
        )
        record.pop("_steps", None)
        records.append(record)
        _persist_progress(output, "V5_ENTRY", records)
        if not record.get("hard_gate_pass") or not record.get("full_resolution"):
            stopped = True
            stop_reason = "entry hard gate or full-resolution failure"
            break
    passed = bool(
        len(records) == len(ENTRY_SEEDS)
        and all(record.get("hard_gate_pass") and record.get("full_resolution") for record in records)
    )
    result = {
        "stage": "V5_ENTRY", "PASS": passed, "status": "COMPLETED" if passed else "FAILED",
        "records": records, "stopped": stopped, "stop_reason": stop_reason,
        "selected_checkpoint": str(checkpoint), "selected_checkpoint_kind": checkpoint_kind,
        "selected_checkpoint_sha256": checkpoint_sha, "loader": str(metadata.get("checkpoint_identity", metadata.get("stage", checkpoint_kind))) if isinstance(metadata, Mapping) else checkpoint_kind,
        "trainable_parameters": list(trainable),
        **_entry_report_binding(
            first_config, checkpoint_kind, checkpoint_sha,
            actual_config_snapshots=actual_config_snapshots, p=p,
        ),
        "entry_updates_discarded": True, "fresh_optimizer": True,
        "P3_QUALIFIED": False, "P4_ALLOWED": False,
        "PPO_ALLOWED": False, "ppo_called": False,
        "PPO_ENTRY_ALLOWED": passed, "bounded_pilot_training_authorized": passed,
        "checkpoint_promotable": False, "FORMAL_P3_ALLOWED": False,
    }
    _persist_progress(output, "V5_ENTRY", records, stopped=stopped, reason=stop_reason)
    _write_json(output / "manifest.json", result)
    return result


def _smoke_stage(*, output: Path, checkpoint: Path, checkpoint_kind: str,
                 p: Mapping[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    base_protocol = base._load_yaml_protocol()
    config = _build_config("dense24", base_protocol)
    policy, metadata, trainable, checkpoint_sha = _load_source(checkpoint, checkpoint_kind, config, p)
    record = _record_rollout(
        policy=policy, config=config, p=p, scenario="dense24", seed=SMOKE_SEED,
        output=output / "dense24_849000", label="smoke", deterministic=False,
        max_steps=SMOKE_MAX_STEPS, checkpoint=checkpoint, require_full_resolution=False,
    )
    steps = record.pop("_steps", [])
    record["smoke_partial"] = True
    smoke_checks = _smoke_contract(record, steps, expected_steps=SMOKE_MAX_STEPS)
    record["smoke_checks"] = smoke_checks
    record["hard_gate_pass"] = bool(
        record.get("hard_gate_pass") is True and all(smoke_checks.values())
    )
    smoke_pass = bool(record["hard_gate_pass"])
    result = {
        "stage": "V5_SMOKE", "PASS": smoke_pass,
        "status": "SMOKE_COMPLETED" if smoke_pass else "SMOKE_FAILED",
        "smoke_partial": True, "records": [record],
        "selected_checkpoint": str(checkpoint), "selected_checkpoint_kind": checkpoint_kind,
        "selected_checkpoint_sha256": checkpoint_sha,
        "trainable_parameters": list(trainable), "fresh_optimizer": True,
        "config": base._config_snapshot(config),
        "horizon_steps": int(round(float(config.max_simulation_seconds) / float(config.dt))),
        "smoke_max_steps": SMOKE_MAX_STEPS,
        "loader": str(metadata.get("checkpoint_identity", metadata.get("stage", checkpoint_kind))) if isinstance(metadata, Mapping) else checkpoint_kind,
        "P3_QUALIFIED": False, "P4_ALLOWED": False,
        "PPO_ALLOWED": False, "ppo_called": False,
        "checkpoint_promotable": False, "FORMAL_P3_ALLOWED": False,
    }
    _persist_progress(output, "V5_SMOKE", [record])
    _write_json(output / "manifest.json", result)
    return result


def _dev_stage(*, output: Path, policy: Any, checkpoint: Path, checkpoint_kind: str,
               p: Mapping[str, Any], label: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    base_protocol = base._load_yaml_protocol()
    records: list[dict[str, Any]] = []
    for seed, scenario in zip(DEV_SEEDS, DEV_SCENARIOS):
        config = _build_config(scenario, base_protocol)
        max_steps = int(round(float(config.max_simulation_seconds) / float(config.dt)))
        record = _record_rollout(
            policy=policy, config=config, p=p, scenario=scenario, seed=seed,
            output=output / f"{scenario}_{seed}", label=label,
            deterministic=True, max_steps=max_steps, checkpoint=checkpoint,
            require_full_resolution=True,
        )
        record.pop("_steps", None)
        records.append(record)
        _persist_progress(output, label, records)
        if not record.get("hard_gate_pass") or not record.get("full_resolution"):
            break
    passed = bool(
        len(records) == len(DEV_SEEDS)
        and all(record.get("hard_gate_pass") and record.get("full_resolution") for record in records)
    )
    return {
        "stage": label, "PASS": passed, "records": records,
        "same_seed_schedule": list(DEV_SEEDS), "scenarios": list(DEV_SCENARIOS),
        "deterministic": True, "updates": False,
        "full_resolution": bool(passed),
        "checkpoint": str(checkpoint), "checkpoint_kind": checkpoint_kind,
        "checkpoint_sha256": _actual_checkpoint_sha(checkpoint, checkpoint_kind),
    }


def _lifecycle_aggregate_for_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Use the existing aggregate helper and keep unavailable joins explicit."""
    try:
        aggregate = p3_reference._lifecycle_aggregate([record])
    except Exception as exc:
        return {
            "available": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {
        "available": True,
        "completed": int(aggregate.get("completed", 0)),
        "requests": int(aggregate.get("requests", 0)),
        "priority_weighted_wait_seconds": aggregate.get("priority_weighted_wait_seconds"),
        "low_soc_mean_wait_seconds": aggregate.get("low_soc_mean_wait_seconds"),
        "low_soc_count": int(aggregate.get("low_soc_count", 0)),
    }


def _pooled_emergency_report(rows: list[Mapping[str, Any]], scenarios: set[str]) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("scenario")) in scenarios]
    before = int(sum(int(row.get("before_emergency", 0) or 0) for row in selected))
    after = int(sum(int(row.get("after_emergency", 0) or 0) for row in selected))
    return {
        "scenarios": sorted(scenarios),
        "paired_cells": len(selected),
        "before_emergency": before,
        "after_emergency": after,
        "delta": after - before,
        "regression": bool(after > before),
        "report_only": True,
    }


def _performance_comparison(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for before_record, after_record in zip(before.get("records", ()), after.get("records", ())):
        before_metrics = _lifecycle_aggregate_for_record(before_record)
        after_metrics = _lifecycle_aggregate_for_record(after_record)
        before_wait = before_metrics.get("priority_weighted_wait_seconds")
        after_wait = after_metrics.get("priority_weighted_wait_seconds")
        before_low_wait = before_metrics.get("low_soc_mean_wait_seconds")
        after_low_wait = after_metrics.get("low_soc_mean_wait_seconds")
        rows.append({
            "seed": before_record.get("seed"),
            "scenario": before_record.get("scenario"),
            "before_emergency": before_record.get("emergency"),
            "after_emergency": after_record.get("emergency"),
            "before_failed_energy": before_record.get("failed_energy"),
            "after_failed_energy": after_record.get("failed_energy"),
            "before_lifecycle": before_metrics,
            "after_lifecycle": after_metrics,
            "before_priority_weighted_wait_seconds": before_wait,
            "after_priority_weighted_wait_seconds": after_wait,
            "priority_weighted_wait_delta_seconds": (
                None if before_wait is None or after_wait is None
                else float(after_wait) - float(before_wait)
            ),
            "priority_wait_regression": (
                None if before_wait is None or after_wait is None
                else bool(float(after_wait) > float(before_wait))
            ),
            "before_low_soc_mean_wait_seconds": before_low_wait,
            "after_low_soc_mean_wait_seconds": after_low_wait,
            "low_soc_mean_wait_delta_seconds": (
                None if before_low_wait is None or after_low_wait is None
                else float(after_low_wait) - float(before_low_wait)
            ),
            "low_soc_wait_regression": (
                None if before_low_wait is None or after_low_wait is None
                else bool(float(after_low_wait) > float(before_low_wait))
            ),
            "before_completed": before_metrics.get("completed"),
            "after_completed": after_metrics.get("completed"),
            "before_low_soc_count": before_metrics.get("low_soc_count"),
            "after_low_soc_count": after_metrics.get("low_soc_count"),
            "before_full_resolution": before_record.get("full_resolution"),
            "after_full_resolution": after_record.get("full_resolution"),
        })
    return {
        "paired_records": rows,
        "pooled_emergency": {
            "sparse": _pooled_emergency_report(rows, {"flow6", "flow12"}),
            "dense": _pooled_emergency_report(rows, {"dense24"}),
        },
        "report_only": True,
        "review_required": True,
    }


def _pilot_stage(*, output: Path, checkpoint: Path, checkpoint_kind: str,
                 entry_report: Mapping[str, Any], p: Mapping[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    base_protocol = base._load_yaml_protocol()
    entry_configs = {
        scenario: _build_config(scenario, base_protocol) for scenario in ENTRY_SCENARIOS
    }
    expected_config_snapshots = {
        scenario: base._config_snapshot(config) for scenario, config in entry_configs.items()
    }
    first_config = _build_config(PILOT_SCENARIOS[0], base_protocol)
    checkpoint_sha = _actual_checkpoint_sha(checkpoint, checkpoint_kind)
    validate_entry_report(
        entry_report, config=first_config, checkpoint_kind=checkpoint_kind,
        checkpoint_sha=checkpoint_sha, p=p,
        expected_config_snapshots=expected_config_snapshots,
    )
    policy, metadata, trainable, loaded_sha = _load_source(checkpoint, checkpoint_kind, first_config, p)
    before_dev = _dev_stage(
        output=output / "dev_before", policy=policy, checkpoint=checkpoint,
        checkpoint_kind=checkpoint_kind, p=p, label="V5_DEV_BEFORE",
    )
    if not before_dev.get("PASS"):
        result = {
            "stage": "V5_PILOT", "PASS": False, "status": "FAILED",
            "records": [], "stopped": True,
            "stop_reason": "before-development gate failure; no pilot update attempted",
            "entry_report_stage": entry_report.get("stage"),
            "entry_report_sha256": entry_report.get("selected_checkpoint_sha256"),
            "selected_checkpoint": str(checkpoint),
            "selected_checkpoint_kind": checkpoint_kind,
            "selected_checkpoint_sha256": checkpoint_sha,
            "final_checkpoint": None, "final_checkpoint_sha256": None,
            "trainable_parameters": list(trainable), "fresh_optimizer": True,
            "entry_updates_discarded": True,
            "dev_before": before_dev, "dev_after": None,
            "performance_comparison": _performance_comparison(before_dev, {}),
            "performance_review_required": True,
            "pilot_updates_ok": False, "dev_full_resolution_ok": False,
            "PILOT_TECHNICAL_PASS": False, "FORMAL_P3_ALLOWED": False,
            "P3_QUALIFIED": False, "P4_ALLOWED": False,
            "checkpoint_promotable": False, "PPO_ALLOWED": False,
            "ppo_called": False, "bounded_pilot_training_authorized": True,
        }
        _persist_progress(output, "V5_PILOT", [], stopped=True,
                          reason="before-development gate failure; no pilot update attempted")
        _write_json(output / "manifest.json", result)
        return result
    records: list[dict[str, Any]] = []
    stopped = False
    stop_reason: str | None = None
    current_checkpoint = checkpoint
    current_sha = loaded_sha
    for index, (seed, scenario) in enumerate(zip(PILOT_SEEDS, PILOT_SCENARIOS)):
        config = _build_config(scenario, base_protocol)
        max_steps = int(round(float(config.max_simulation_seconds) / float(config.dt)))
        record = _record_rollout(
            policy=policy, config=config, p=p, scenario=scenario, seed=seed,
            output=output / f"pilot_{scenario}_{seed}", label="pilot",
            deterministic=False, max_steps=max_steps, checkpoint=current_checkpoint,
            require_full_resolution=False,
        )
        steps = record.pop("_steps", [])
        record["incomplete_is_report_only"] = not bool(record.get("full_resolution"))
        technical_rollout_ok = bool(
            record.get("hard_gate_pass")
            and record.get("outcomes", {}).get("zero_failed_energy")
            and record.get("outcomes", {}).get("zero_other_failures")
        )
        if technical_rollout_ok:
            record["ppo_called"] = True
            model_before = copy.deepcopy(policy.model.state_dict())
            optimizer_before = copy.deepcopy(policy.optimizer.state_dict())
            try:
                update, update_gates = p3_legacy._transactional_update(
                    policy, steps,
                    config, str(record.get("action_tape")), p,
                )
            except Exception as exc:
                update = {"exception": type(exc).__name__, "message": str(exc)}
                update_gates = {"update_transaction": False}
                policy.model.load_state_dict(model_before)
                policy.optimizer.load_state_dict(optimizer_before)
            record["update"] = update
            record["update_gates"] = update_gates
            if not all(bool(value) for value in update_gates.values()):
                stopped = True
                stop_reason = "pilot PPO update gate failure"
            else:
                checkpoint_after = output / f"checkpoint_{index + 1}.pt"
                try:
                    save_checkpoint(
                        checkpoint_after, policy, config,
                        stage=STAGE, formal=False, ppo_called=True,
                        protocol_sha256=_sha256(PROTOCOL),
                        protocol_identity=_protocol_digest(p),
                        config_identity=_schema_digest(config),
                        config_snapshot=base._config_snapshot(config),
                        initial_checkpoint_sha256=checkpoint_sha,
                        selected_checkpoint_sha256=checkpoint_sha,
                        checkpoint_kind=checkpoint_kind,
                        checkpoint_promotable=False,
                        P3_QUALIFIED=False, P4_ALLOWED=False,
                        PPO_ALLOWED=False,
                        urgency_reward_config=dict(p["reward"]),
                        trainable_parameters=list(trainable),
                    )
                    restored = p3_legacy._restored_policy(checkpoint_after, config, p)
                    restore_exact = _state_equal(policy.model.state_dict(), restored.model.state_dict()) and _state_equal(
                        policy.optimizer.state_dict(), restored.optimizer.state_dict()
                    )
                    record["checkpoint_restore_exact"] = bool(restore_exact)
                    if not restore_exact:
                        raise V5InputError("checkpoint restore state mismatch")
                    policy = restored
                    current_checkpoint = checkpoint_after
                    current_sha = _sha256(checkpoint_after)
                    record["checkpoint_after"] = str(checkpoint_after)
                    record["checkpoint_after_sha256"] = current_sha
                except Exception as exc:
                    record["checkpoint_restore_exact"] = False
                    record["checkpoint_error"] = {"type": type(exc).__name__, "message": str(exc)}
                    stopped = True
                    stop_reason = "pilot checkpoint restore failure"
        else:
            stopped = True
            stop_reason = "pilot safety, energy, or outcome gate failure"
        records.append(record)
        _persist_progress(output, "V5_PILOT", records, stopped=stopped, reason=stop_reason)
        if stopped:
            break
    after_dev = None
    if not stopped and records:
        after_dev = _dev_stage(
            output=output / "dev_after", policy=policy, checkpoint=current_checkpoint,
            checkpoint_kind=checkpoint_kind, p=p, label="V5_DEV_AFTER",
        )
    pilot_updates_ok = bool(
        len(records) == len(PILOT_SEEDS)
        and all(record.get("hard_gate_pass") and record.get("update_gates")
                and all(bool(value) for value in record["update_gates"].values())
                and record.get("checkpoint_restore_exact") is True for record in records)
    )
    dev_ok = bool(
        before_dev.get("PASS")
        and after_dev is not None
        and after_dev.get("PASS")
        and all(record.get("failed_energy", 0) == 0 for record in before_dev.get("records", ()))
        and all(record.get("failed_energy", 0) == 0 for record in after_dev.get("records", ()))
    )
    technical_pass = bool(pilot_updates_ok and dev_ok)
    result = {
        "stage": "V5_PILOT", "PASS": technical_pass,
        "status": "COMPLETED" if technical_pass else "FAILED",
        "records": records, "stopped": stopped, "stop_reason": stop_reason,
        "entry_report_stage": entry_report.get("stage"),
        "entry_report_sha256": entry_report.get("selected_checkpoint_sha256"),
        "selected_checkpoint": str(checkpoint), "selected_checkpoint_kind": checkpoint_kind,
        "selected_checkpoint_sha256": checkpoint_sha,
        "final_checkpoint": str(current_checkpoint) if records and not stopped else None,
        "final_checkpoint_sha256": current_sha if records and not stopped else None,
        "trainable_parameters": list(trainable), "fresh_optimizer": True,
        "entry_updates_discarded": True,
        "dev_before": before_dev, "dev_after": after_dev,
        "performance_comparison": _performance_comparison(before_dev, after_dev or {}),
        "performance_review_required": True,
        "pilot_updates_ok": pilot_updates_ok, "dev_full_resolution_ok": dev_ok,
        "PILOT_TECHNICAL_PASS": technical_pass,
        "FORMAL_P3_ALLOWED": False,
        "P3_QUALIFIED": False, "P4_ALLOWED": False,
        "checkpoint_promotable": False, "PPO_ALLOWED": False,
        "ppo_called": bool(any(record.get("ppo_called") is True for record in records)),
        "bounded_pilot_training_authorized": True,
    }
    _persist_progress(output, "V5_PILOT", records, stopped=stopped, reason=stop_reason)
    _write_json(output / "manifest.json", result)
    return result


def run(*, stage: str, checkpoint: str | Path, checkpoint_kind: str,
        output: str | Path, entry_report: str | Path | None = None) -> dict[str, Any]:
    p = protocol()
    stage = str(stage)
    if stage not in {"smoke", "entry", "pilot"}:
        raise ValueError("stage must be smoke, entry, or pilot")
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"V5 output must be fresh: {output_path}")
    checkpoint_path = Path(checkpoint)
    output_path.mkdir(parents=True, exist_ok=False)
    try:
        if stage == "smoke":
            result = _smoke_stage(output=output_path, checkpoint=checkpoint_path,
                                  checkpoint_kind=checkpoint_kind, p=p)
        elif stage == "entry":
            result = _entry_stage(output=output_path, checkpoint=checkpoint_path,
                                  checkpoint_kind=checkpoint_kind, p=p)
        else:
            _require(entry_report is not None, "pilot requires --entry-report")
            report_path = Path(entry_report)
            _require(report_path.is_file(), f"entry report missing: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            _require(isinstance(report, Mapping), "entry report must be a mapping")
            result = _pilot_stage(output=output_path, checkpoint=checkpoint_path,
                                  checkpoint_kind=checkpoint_kind, entry_report=report, p=p)
        _write_json(output_path / "manifest.json", result)
        return result
    except Exception as exc:
        progress_path = output_path / "progress.json"
        failure_reason = f"{type(exc).__name__}: {exc}"
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                if isinstance(progress, Mapping):
                    progress = dict(progress)
                    progress["stopped"] = True
                    progress["stop_reason"] = failure_reason
                    _write_json(progress_path, progress)
            except Exception:
                pass
        else:
            _persist_progress(
                output_path, f"V5_{stage.upper()}", [], stopped=True,
                reason=failure_reason,
            )
        failure = {
            "stage": f"V5_{stage.upper()}", "PASS": False, "status": "FAILED",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "checkpoint": str(checkpoint_path), "checkpoint_kind": checkpoint_kind,
            "P3_QUALIFIED": False, "P4_ALLOWED": False,
            "PPO_ALLOWED": False, "FORMAL_P3_ALLOWED": False,
        }
        _write_json(output_path / "manifest.json", failure)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "entry", "pilot"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", choices=("canonical", "repair"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry-report", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = run(
        stage=args.stage, checkpoint=args.checkpoint,
        checkpoint_kind=args.checkpoint_kind, output=args.output,
        entry_report=args.entry_report,
    )
    print(json.dumps(_jsonable({
        "stage": result.get("stage"), "PASS": result.get("PASS"),
        "PILOT_TECHNICAL_PASS": result.get("PILOT_TECHNICAL_PASS"),
        "FORMAL_P3_ALLOWED": result.get("FORMAL_P3_ALLOWED", False),
    }), sort_keys=True), flush=True)
    return 0 if result.get("status") in {"COMPLETED", "SMOKE_COMPLETED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
