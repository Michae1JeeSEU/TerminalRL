"""Bounded diagnostic replay of the frozen V4.2 K=3 learned policy.

This module is deliberately inference-only.  It adapts the frozen P2 trained
checkpoint to the current dense30 V2.1 environment by validating metadata and
recording the declared profile-version migration.  It never restores an
optimizer, invokes a Teacher, performs an update, or promotes a checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import checkpoint_metadata
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.diagnose_energy_reachability_guard_smoke import (
    build_config as build_rule_config,
)
from terminal_marl.training.evaluate_v4_2_p2_runtime_shortlist_heldout_v1 import (
    protocol as p2_heldout_protocol,
)
from terminal_marl.training.student_rollout_v4_2 import (
    _student_request_details,
    validate_student_config,
)


STAGE = "V4_2_TRAINED_DENSE30_V2_DIAGNOSTIC"
OBSERVATION_DIM = 88
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"
TRAINED_CHECKPOINT_SHA256 = (
    "714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e"
)
SOURCE_OBSERVATION_FEATURE_HASH = (
    "8cde32fb4d7481649aad5d01fae0e97c3bb7671c0379d1c4847915493d7beb98"
)
SOURCE_VEHICLE_PROFILE_VERSION = "eh216s_research_v1"
TARGET_VEHICLE_PROFILE_VERSION = "eh216s_research_v2"
MAX_STEPS = 1800
RUNTIME_SHORTLIST_K = 3
SAFETY_DECISION_ACCEPTED = {
    "approved",
    "emergency_redirected",
    "future_reserved",
    "future_reservation_wait",
}
FAILURE_OUTCOMES = {
    "failed_collision",
    "failed_energy",
    "diverted_low_soc",
    "diverted_capacity",
    "invalid",
    "unresolved",
}


class DiagnosticInputError(ValueError):
    """Raised when a frozen diagnostic input is not compatible."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise DiagnosticInputError(f"model state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_dense30_config():
    """Build the current dense30 learned-policy V2.1 diagnostic config."""

    source = build_rule_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=True,
        max_simulation_seconds=1800.0,
        dense_warm_start_uavs=30,
        final_approach_geometry_mode="continuous_inward_v1",
        release_arbitration_mode="downstream_priority",
        reconcile_late_merge_slots_enabled=False,
    )
    config = replace(
        source,
        observation_schema_version=OBSERVATION_SCHEMA,
        neighbor_comm_range_m=None,
        runtime_proposal_shortlist_k=RUNTIME_SHORTLIST_K,
        experiment_mode="fixed_low_student",
        rule_flow_coordination_enabled=False,
        teacher_enabled=False,
        teacher_release_label_enabled=False,
        scheduler_mode="execution",
    )
    validate_student_config(config)
    return config


def _checkpoint_bindings() -> dict[str, Any]:
    protocol = p2_heldout_protocol()
    bindings = protocol.get("bindings")
    if not isinstance(bindings, Mapping):
        raise DiagnosticInputError("P2 heldout bindings are missing")
    expected = {
        "trained_checkpoint_sha256": TRAINED_CHECKPOINT_SHA256,
        "observation_dimension": OBSERVATION_DIM,
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "runtime_proposal_shortlist_k": RUNTIME_SHORTLIST_K,
        "deterministic": True,
        "one_policy_forward": True,
        "no_teacher": True,
        "no_labels": True,
        "no_updates": True,
    }
    for key, value in expected.items():
        if bindings.get(key) != value:
            raise DiagnosticInputError(f"P2 heldout binding changed: {key}")
    return dict(bindings)


def validate_checkpoint_metadata_for_diagnostic(
    metadata: Mapping[str, Any],
    config,
) -> dict[str, Any]:
    """Compare all checkpoint metadata, allowing only declared field deltas."""

    if not isinstance(metadata, Mapping):
        raise DiagnosticInputError("checkpoint metadata must be a mapping")
    expected = checkpoint_metadata(config)
    deltas = {
        "vehicle_profile_version": {
            "source": SOURCE_VEHICLE_PROFILE_VERSION,
            "target": TARGET_VEHICLE_PROFILE_VERSION,
        },
    }
    if set(metadata) != set(expected):
        missing = sorted(set(expected) - set(metadata))
        extra = sorted(set(metadata) - set(expected))
        raise DiagnosticInputError(
            f"checkpoint metadata keys differ: missing={missing}, extra={extra}"
        )
    changes: list[dict[str, Any]] = []
    for key, target in expected.items():
        source = metadata[key]
        if source == target:
            continue
        delta = deltas.get(key)
        if not isinstance(delta, Mapping) or source != delta.get("source") or target != delta.get("target"):
            raise DiagnosticInputError(
                f"checkpoint metadata mismatch for {key}: source={source!r}, target={target!r}"
            )
        changes.append({"field": key, "source": source, "target": target})
    return {
        "allowed": True,
        "source_metadata": _jsonable(dict(metadata)),
        "target_metadata": _jsonable(expected),
        "allowed_deltas": _jsonable(deltas),
        "applied_deltas": changes,
        "source_unchanged": True,
    }


def _validate_payload_bindings(payload: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "stage": "V4_2_P2_RUNTIME_SHORTLIST_FORMAL_V1",
        "formal": True,
        "ppo_called": True,
        "protocol_sha256": bindings.get("formal_protocol_sha256"),
        "initial_checkpoint_sha256": bindings.get("initial_checkpoint_sha256"),
        "checkpoint_promotable": False,
    }
    mismatches = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise DiagnosticInputError(f"trained checkpoint payload binding changed: {mismatches}")
    return {key: payload.get(key) for key in expected}


def load_trained_checkpoint(checkpoint: str | Path, config):
    """Load the frozen trained policy after strict diagnostic-only checks."""

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise DiagnosticInputError(f"checkpoint not found: {checkpoint}")
    bindings = _checkpoint_bindings()
    actual_sha = _sha256(checkpoint)
    if actual_sha != bindings["trained_checkpoint_sha256"]:
        raise DiagnosticInputError(
            f"trained checkpoint SHA mismatch: {actual_sha}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise DiagnosticInputError("trained checkpoint payload must be a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise DiagnosticInputError("trained checkpoint metadata is missing")
    migration = validate_checkpoint_metadata_for_diagnostic(metadata, config)
    if metadata.get("observation_feature_hash") != SOURCE_OBSERVATION_FEATURE_HASH:
        raise DiagnosticInputError("trained checkpoint observation feature hash changed")
    payload_bindings = _validate_payload_bindings(payload, bindings)
    model_state = payload.get("model")
    if not isinstance(model_state, Mapping):
        raise DiagnosticInputError("trained checkpoint model state is missing")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        temperature=CALIBRATION_TEMPERATURE,
    )
    before = _state_digest(policy.model.state_dict())
    policy.model.load_state_dict(model_state, strict=True)
    after = _state_digest(policy.model.state_dict())
    policy.model.eval()
    return policy, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "payload_bindings": payload_bindings,
        "migration": migration,
        "weight_digest_before": before,
        "weight_digest_after": after,
        "weight_digest_loaded_from_payload": _state_digest(model_state),
        "weight_digest_matches_payload": after == _state_digest(model_state),
        "optimizer_restored": False,
        "source_payload_unchanged": True,
    }


def _terminal(env: TerminalParallelEnv) -> bool:
    return bool(
        env.arrivals.exhausted()
        and not env.active
        and not env.admission.queue.items
        and not env.service_uavs
    )


def _config_payload(config) -> dict[str, Any]:
    values = dict(vars(config))
    values.update({
        "vehicle_profile_id": config.vehicle_profile_id,
        "vehicle_profile_version": config.vehicle_profile.profile_version,
        "min_landing_soc": config.min_landing_soc,
        "soc_reserve": config.soc_reserve,
        "safe_distance_m": config.safe_distance,
        "protected_floor_m": config.nmac_distance,
        "collision_distance_m": config.collision_distance,
    })
    return _jsonable(values)


def _release_veto(decision: Any, trace: Mapping[str, Any], release_telemetry: Mapping[str, Any]) -> dict[str, Any]:
    requested_release = bool(
        isinstance(trace.get("requested_hierarchical_action"), Mapping)
        and trace["requested_hierarchical_action"].get("gate_index") == 1
    )
    if not requested_release:
        return {"requested_release": False, "vetoed": False, "reason": None}
    reason = release_telemetry.get("safety_reason") or release_telemetry.get("primary_blocker")
    vetoed = bool(
        decision is not None
        and str(decision) not in SAFETY_DECISION_ACCEPTED
    )
    if trace.get("shield_used"):
        vetoed = True
        reason = reason or trace.get("shield_reason") or "shield_intervention"
    return {
        "requested_release": True,
        "vetoed": vetoed,
        "reason": None if reason is None else str(reason),
    }


def _policy_weight_digest(policy) -> str:
    model = getattr(policy, "model", None)
    if model is None or not hasattr(model, "state_dict"):
        raise DiagnosticInputError("diagnostic policy must expose model state")
    state = model.state_dict()
    if not isinstance(state, Mapping):
        raise DiagnosticInputError("diagnostic policy model state must be a mapping")
    return _state_digest(state)


def _terminal_identity_records(env: TerminalParallelEnv, outcomes: Mapping[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-arrival terminal and unresolved identity records."""

    completed_by_arrival = {
        int(uav.arrival_id): uav for uav in env.completed_uavs
    }
    active_by_arrival = {
        int(uav.arrival_id): uav
        for uav in list(env.active.values()) + list(env.service_uavs.values())
    }
    energy_by_arrival: dict[int, dict[str, Any]] = {}
    for row in env.energy_outcomes:
        arrival_id = row.get("arrival_id")
        if arrival_id is not None:
            energy_by_arrival.setdefault(int(arrival_id), row)

    terminal: list[dict[str, Any]] = []
    for arrival_id_text, outcome in sorted(outcomes.items(), key=lambda item: int(item[0])):
        arrival_id = int(arrival_id_text)
        row = energy_by_arrival.get(arrival_id, {})
        uav = completed_by_arrival.get(arrival_id) or active_by_arrival.get(arrival_id)
        uid = row.get("uid")
        if uid is None and uav is not None:
            uid = uav.uid
        reason = row.get("reason")
        if reason is None and uav is not None:
            reason = uav.terminal_reason
        phase = row.get("phase")
        if phase is None and uav is not None:
            phase = uav.phase.value
        terminal_time = row.get("time")
        if terminal_time is None and uav is not None:
            terminal_time = uav.completion_time
        terminal.append({
            "arrival_id": arrival_id,
            "uid": None if uid is None else int(uid),
            "outcome": str(outcome),
            "reason": None if reason is None else str(reason),
            "phase": None if phase is None else str(phase),
            "time": None if terminal_time is None else float(terminal_time),
        })

    unresolved: list[dict[str, Any]] = []
    for arrival_id in sorted(int(value) for value in env.ledger.unresolved):
        uav = active_by_arrival.get(arrival_id)
        unresolved.append({
            "arrival_id": arrival_id,
            "uid": None if uav is None else int(uav.uid),
            "phase": None if uav is None else uav.phase.value,
            "status": "unresolved",
        })
    return terminal, unresolved


def _progress_payload(env, *, step: int, wall_elapsed: float, safety_error: str | None,
                      decision_counts: Mapping[str, int]) -> dict[str, Any]:
    return {
        "step": int(step),
        "sim_time": float(env.now),
        "wall_elapsed": float(wall_elapsed),
        "active": int(len(env.active)),
        "resolved": int(len(env.ledger.outcomes)),
        "unresolved": int(len(env.ledger.unresolved)),
        "decision_counts": dict(decision_counts),
        "safety_error": safety_error,
        "terminal": _terminal(env),
        "status": "safety_error" if safety_error else "terminal" if _terminal(env) else "running",
    }


def run_rollout(
    policy,
    config,
    *,
    output: str | Path,
    seed: int,
    max_steps: int,
    checkpoint_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a deterministic, inference-only diagnostic with a supplied policy."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if isinstance(max_steps, bool) or not 1 <= int(max_steps) <= MAX_STEPS:
        raise ValueError(f"max_steps must be in [1, {MAX_STEPS}]")
    max_steps = int(max_steps)
    output.mkdir(parents=True, exist_ok=False)
    validate_student_config(config)
    if int(getattr(policy, "observation_dim", -1)) != OBSERVATION_DIM:
        raise DiagnosticInputError("diagnostic policy must have 88D observations")
    weight_digest_before = _policy_weight_digest(policy)
    started = time.perf_counter()
    _write_json_atomic(output / "config.json", {
        "diagnostic_only": True,
        "config": _config_payload(config),
    })
    _write_json_atomic(output / "metadata.json", {
        "stage": STAGE,
        "diagnostic_only": True,
        "ppo_called": False,
        "training_called": False,
        "PPO_ALLOWED": False,
        "checkpoint_promotable": False,
        "teacher_enabled": bool(config.teacher_enabled),
        "scheduler_mode": config.scheduler_mode,
        "checkpoint": checkpoint_info,
    })
    _write_json_atomic(output / "migration.json", {
        "diagnostic_migration": True,
        "source_payload_unchanged": True,
        "checkpoint": checkpoint_info,
    })
    steps_path = output / "actor_steps.jsonl"
    decision_counts: Counter[str] = Counter()
    actor_decisions = 0
    wait_decisions = 0
    release_decisions = 0
    release_vetoes = 0
    safety_error: str | None = None
    steps = 0
    terminal = False
    env = TerminalParallelEnv(config)
    observations: dict[str, np.ndarray] = {}
    try:
        observations, _ = env.reset(seed=int(seed))
    except SafetyInvariantError as exc:
        safety_error = str(exc)
    progress_path = output / "progress.json"
    _write_json_atomic(progress_path, _progress_payload(
        env, step=0, wall_elapsed=time.perf_counter() - started,
        safety_error=safety_error, decision_counts=decision_counts,
    ))
    with steps_path.open("w", encoding="utf-8") as handle:
        while safety_error is None and steps < max_steps and not _terminal(env):
            details = _student_request_details(
                policy,
                env,
                observations,
                deterministic=True,
                shortlist_k=RUNTIME_SHORTLIST_K,
            )
            uids, _, actor_mask, requested, gates, proposals = details[:6]
            shortlists = details[-1]
            before = {
                int(uid): {
                    "uid": int(uid),
                    "time": float(env.now),
                    "phase": env.active[int(uid)].phase.value,
                    "soc": float(env.active[int(uid)].soc),
                    "actor_mask": bool(actor_mask[index]),
                    "gate": int(gates[index]),
                    "proposal": None if int(proposals[index]) < 0 else int(proposals[index]),
                    "requested_action": _jsonable(requested.get(uid, requested.get(str(uid), {}))),
                }
                for index, uid in enumerate(uids)
            }
            for index, uid in enumerate(uids):
                if bool(actor_mask[index]):
                    actor_decisions += 1
                    if int(gates[index]) == 0:
                        wait_decisions += 1
                        decision_counts["WAIT"] += 1
                    else:
                        release_decisions += 1
                        decision_counts["RELEASE"] += 1
            try:
                following, _, _, _, info = env.step(requested, shortlists)
            except SafetyInvariantError as exc:
                safety_error = str(exc)
                steps += 1
                for row in before.values():
                    row.update({
                        "step": steps - 1,
                        "action_trace": None,
                        "release_decision": None,
                        "release_telemetry": None,
                        "current_telemetry": _jsonable(env.telemetry),
                        "veto": None,
                        "status": "safety_invariant_error",
                    })
                    handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                break
            traces = info.get("action_traces", {})
            decisions = info.get("release_decisions", {})
            release_telemetry = info.get("release_intent_telemetry", {})
            current_telemetry = info.get("telemetry", dict(env.telemetry))
            for uid, row in before.items():
                trace = traces.get(str(uid), traces.get(uid, {}))
                decision = decisions.get(str(uid), decisions.get(uid))
                telemetry = release_telemetry.get(str(uid), release_telemetry.get(uid, {}))
                if trace.get("teacher_used"):
                    raise DiagnosticInputError("diagnostic trace unexpectedly used Teacher")
                trace_requested = trace.get("requested_low_action")
                if trace_requested is not None and not np.array_equal(
                    np.asarray(row["requested_action"], dtype=float),
                    np.asarray(trace_requested, dtype=float),
                ):
                    raise DiagnosticInputError(
                        f"requested action trace mismatch for uid={uid}"
                    )
                veto = _release_veto(decision, trace, telemetry)
                release_vetoes += int(veto["vetoed"])
                row.update({
                    "step": steps,
                    "action_trace": trace,
                    "release_decision": decision,
                    "release_telemetry": telemetry,
                    "current_telemetry": current_telemetry,
                    "veto": veto,
                    "status": "observed",
                })
                handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
            handle.flush()
            steps += 1
            observations = following
            terminal = _terminal(env)
            if steps % 25 == 0 or terminal:
                _write_json_atomic(progress_path, _progress_payload(
                    env, step=steps, wall_elapsed=time.perf_counter() - started,
                    safety_error=safety_error, decision_counts=decision_counts,
                ))
    terminal = _terminal(env)
    _write_json_atomic(progress_path, _progress_payload(
        env, step=steps, wall_elapsed=time.perf_counter() - started,
        safety_error=safety_error, decision_counts=decision_counts,
    ))
    outcomes = {
        str(arrival_id): outcome.value
        for arrival_id, outcome in env.ledger.outcomes.items()
    }
    outcome_counts = Counter(outcomes.values())
    terminal_identities, unresolved_identities = _terminal_identity_records(env, outcomes)
    telemetry = dict(env.telemetry)
    summary = {
        "stage": STAGE,
        "diagnostic_only": True,
        "ppo_called": False,
        "training_called": False,
        "teacher_used": False,
        "PPO_ALLOWED": False,
        "checkpoint_promotable": False,
        "seed": int(seed),
        "max_steps": max_steps,
        "steps_executed": int(steps),
        "sim_time_seconds": float(env.now),
        "terminal": bool(terminal),
        "horizon_reached": bool(not terminal and steps >= max_steps),
        "safety_error": safety_error,
        "normal": int(outcome_counts.get("completed", 0)),
        "emergency": int(outcome_counts.get("emergency_landed", 0)),
        "failed_energy": int(outcome_counts.get("failed_energy", 0)),
        "unresolved": int(len(env.ledger.unresolved)),
        "outcomes": outcomes,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "terminal_identities": terminal_identities,
        "unresolved_identities": unresolved_identities,
        "min_actual_separation_m": telemetry.get("minimum_actual_separation"),
        "min_swept_separation_m": telemetry.get("minimum_swept_separation"),
        "env_telemetry": telemetry,
        "decision_counts": dict(decision_counts),
        "actor_decisions": int(actor_decisions),
        "wait_decisions": int(wait_decisions),
        "release_decisions": int(release_decisions),
        "release_vetoes": int(release_vetoes),
        "action_rows": 0,
        "config": _config_payload(config),
        "checkpoint": checkpoint_info,
        "progress_path": str(progress_path),
        "actor_steps_path": str(steps_path),
        "simulation_wall_time_seconds": float(time.perf_counter() - started),
    }
    weight_digest_after = _policy_weight_digest(policy)
    summary.update({
        "weight_digest_before": weight_digest_before,
        "weight_digest_after": weight_digest_after,
        "weight_digest_unchanged": weight_digest_before == weight_digest_after,
    })
    with steps_path.open("r", encoding="utf-8") as read_handle:
        summary["action_rows"] = sum(1 for _ in read_handle)
    _write_json_atomic(output / "summary.json", summary)
    return summary


def run(*, checkpoint: str | Path, output: str | Path, seed: int, max_steps: int) -> dict[str, Any]:
    config = build_dense30_config()
    policy, checkpoint_info = load_trained_checkpoint(checkpoint, config)
    return run_rollout(
        policy,
        config,
        output=output,
        seed=seed,
        max_steps=max_steps,
        checkpoint_info=checkpoint_info,
    )


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args()
    result = run(
        checkpoint=args.checkpoint,
        output=args.output,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
