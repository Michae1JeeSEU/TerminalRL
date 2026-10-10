"""Bounded exact-replay diagnosis for coverage run2 UID 134 / arrival 16.

This script observes one existing coverage ActionTape only.  It keeps the
environment's original release, emergency, and safety guards, and runs
state-only Teacher queries on deep copies at sparse actor decision points.
No rollout sampling, environment Teacher substitution, or policy update is
performed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATED_ARCHITECTURE_ID,
    CALIBRATION_TEMPERATURE,
    OBSERVATION_DIM,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.core.low_level_joint_action_v4_2 import (
    hierarchical_from_joint_index_v4_1,
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import diagnose_p3_bounded_gate_v1 as bounded_diagnostic
from terminal_marl.training.v4_1_joint_teacher import (
    teacher_joint_expert_v4_1,
    teacher_joint_label_v4_1,
)


TARGET_UID = 134
TARGET_ARRIVAL_ID = 16
RUN_INDEX = 2
EXPECTED_SEED = 852090
SAMPLE_STRIDE_STEPS = 40
STAGE = "P3_REMAINING_OUTER_DIAGNOSTIC_V1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _telemetry_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(set(before) | set(after), key=str):
        left, right = before.get(key, 0), after.get(key, 0)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            result[str(key)] = right - left
    return result


def _find_uav(env: TerminalParallelEnv, uid: int) -> Any | None:
    if uid in env.active:
        return env.active[uid]
    if uid in env.service_uavs:
        return env.service_uavs[uid]
    for uav in env.completed_uavs:
        if int(uav.uid) == uid:
            return uav
    return None


def _uav_state(env: TerminalParallelEnv, uid: int) -> dict[str, Any]:
    uav = _find_uav(env, uid)
    if uav is None:
        return {"uid": int(uid), "present": False}
    return {
        "uid": int(uid),
        "present": True,
        "arrival_id": int(uav.arrival_id),
        "phase": uav.phase.value,
        "ring_role": None if uav.ring_role is None else uav.ring_role.value,
        "soc": float(uav.soc),
        "energy_remaining": float(uav.energy_remaining),
        "committed": bool(uav.committed_plan is not None),
        "assigned_lz": None if uav.assigned_lz is None else int(uav.assigned_lz),
        "emergency_landing": bool(uav.emergency_landing),
        "terminal_outcome": None if uav.terminal_outcome is None else uav.terminal_outcome.value,
        "terminal_reason": uav.terminal_reason,
    }


def _requested(actions: Any, uid: int) -> dict[str, Any]:
    if not isinstance(actions, Mapping):
        return {"gate": None, "proposal": None}
    raw = actions.get(str(uid), actions.get(uid))
    if raw is None:
        return {"gate": None, "proposal": None}
    try:
        action = hierarchical_from_normalized_v4_2(np.asarray(raw, dtype=np.float32))
        return {"gate": int(action.gate),
                "proposal": None if action.proposal is None else int(action.proposal)}
    except (TypeError, ValueError) as exc:
        return {"gate": None, "proposal": None,
                "decode_error": f"{type(exc).__name__}: {exc}"}


def _intent_payload(intent: Any) -> dict[str, float]:
    return {
        "release_request": float(intent.release_request),
        "speed_residual": float(intent.speed_residual),
        "merge_time_residual": float(intent.merge_time_residual),
    }


def _shortlist_payload(shortlist: Any) -> list[dict[str, float]] | None:
    if shortlist is None:
        return None
    return [_intent_payload(intent) for intent in tuple(shortlist)]


def _energy_diagnostic(env: TerminalParallelEnv) -> dict[str, Any]:
    diagnostic = getattr(env, "energy_reachability_diagnostics", {}).get(TARGET_UID)
    result = {} if diagnostic is None else dict(diagnostic)
    uav = _find_uav(env, TARGET_UID)
    result["current_energy"] = None if uav is None else float(uav.energy_remaining)
    result["current_soc"] = None if uav is None else float(uav.soc)
    return _copy_json(result)


def _restore_original_methods(shadow: TerminalParallelEnv) -> None:
    cls = type(shadow)
    for name in ("step", "_release", "_try_emergency_landing", "_candidate_energy_guard_reason"):
        setattr(shadow, name, getattr(cls, name).__get__(shadow, cls))


def _state_only_label(env: TerminalParallelEnv, observation: np.ndarray) -> dict[str, Any]:
    uav = env.active[TARGET_UID]
    before_hash = _digest(SafetyTraceRecorder._state(env))
    shadow = copy.deepcopy(env)
    _restore_original_methods(shadow)
    shadow_uav = shadow.active[TARGET_UID]
    label = teacher_joint_label_v4_1(shadow, shadow_uav, state_only=True)
    _, audit = teacher_joint_expert_v4_1(
        shadow, shadow_uav, audit_catalogue=True, state_only=True,
    )
    after_hash = _digest(SafetyTraceRecorder._state(env))
    _require(before_hash == after_hash, "state-only label query changed replay environment")
    target = hierarchical_from_joint_index_v4_1(label.index)
    certified = [
        {
            "rank": int(row.get("rank", -1)),
            "joint_index": int(row.get("joint_index", -1)),
            "decoded": _copy_json(row.get("decoded", {})),
            "reason": row.get("reason"),
        }
        for row in audit
        if bool(row.get("feasible"))
        and float(row.get("decoded", {}).get("release_request", 0.0)) > 0.0
    ]
    return {
        "status": "evaluated",
        "observation_dimension": int(np.asarray(observation).size),
        "selected_label": {
            "gate": int(target.gate),
            "proposal": None if target.proposal is None else int(target.proposal),
            "joint_index_v4_1": int(label.index),
        },
        "certified_release_exists": bool(certified),
        "certified_release_candidates": certified,
        "audit_count": int(len(audit)),
    }


def _load_policy(checkpoint: Path, config: Any) -> TemperatureCalibratedGateResidualAdapterStudentPPOV42:
    _require(checkpoint.is_file(), f"coverage checkpoint missing: {checkpoint}")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM, device="cpu", temperature=CALIBRATION_TEMPERATURE,
    )
    payload = load_checkpoint(checkpoint, policy, config, load_optimizer=False)
    _require(payload.get("policy_architecture_migrated") is False,
             "coverage checkpoint architecture migration occurred")
    _require(getattr(policy.model, "architecture_id", None) == CALIBRATED_ARCHITECTURE_ID,
             "coverage checkpoint architecture mismatch")
    _require(len(policy.optimizer.state) == 0, "coverage optimizer state was restored")
    policy.model.eval()
    return policy


def _p_release(policy: TemperatureCalibratedGateResidualAdapterStudentPPOV42,
               observation: np.ndarray) -> float:
    import torch

    values = torch.as_tensor(observation, dtype=torch.float32, device=policy.device).reshape(1, -1)
    with torch.no_grad():
        encoded = policy.model.actor_encoder(values)
        logits = (policy.model.gate_head(encoded) + policy.model.gate_adapter(values)) / policy.model.gate_temperature
        return float(torch.softmax(logits, dim=-1)[0, 1].detach().cpu().item())


def replay_diagnostic(*, tape: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    _require(tape.is_file(), f"ActionTape missing: {tape}")
    _require(checkpoint.is_file(), f"checkpoint missing: {checkpoint}")
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    config = bounded_diagnostic.build_config()
    _require(config.v4_2_rejected_release_hold_mode == "canonical_wait",
             "diagnostic config is not canonical_wait")
    policy = _load_policy(checkpoint, config)
    payload = json.loads(tape.read_text(encoding="utf-8"))
    _require(int(payload["episode_seed"]) == EXPECTED_SEED,
             "coverage tape seed does not match run2")
    env = TerminalParallelEnv(config)

    samples: list[dict[str, Any]] = []
    releases: list[dict[str, Any]] = []
    emergencies: list[dict[str, Any]] = []
    identity_checks: list[dict[str, Any]] = []
    outer_entry_step: int | None = None
    next_sample_step: int | None = None
    first_present_step: int | None = None
    last_state: dict[str, Any] | None = None

    original_step = env.step
    original_release = env._release
    original_emergency = env._try_emergency_landing

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal outer_entry_step, next_sample_step, first_present_step, last_state
        actions = args[0] if args else kwargs.get("actions")
        uav = env.active.get(TARGET_UID)
        state = _uav_state(env, TARGET_UID)
        requested = _requested(actions, TARGET_UID)
        if state.get("present"):
            if first_present_step is None:
                first_present_step = int(env.step_count)
            identity_checks.append({
                "step": int(env.step_count), "time": float(env.now),
                "uid": TARGET_UID, "arrival_id": state.get("arrival_id"),
            })
        actor_eligible = bool(
            uav is not None
            and uav.committed_plan is None
            and uav.phase.value in {"ring_outer", "ring_middle", "ring_inner"}
            and TARGET_UID not in env.future_release_reservations
            and not uav.emergency_landing
        )
        if uav is not None and uav.phase.value == "ring_outer" and outer_entry_step is None:
            outer_entry_step = int(env.step_count)
            next_sample_step = int(env.step_count)
        if (
            actor_eligible
            and next_sample_step is not None
            and int(env.step_count) >= next_sample_step
        ):
            observation = env._observations().get(str(TARGET_UID))
            _require(observation is not None, "target observation missing while actor eligible")
            label = _state_only_label(env, observation)
            samples.append({
                "step": int(env.step_count), "time": float(env.now),
                "soc": float(uav.soc), "phase": uav.phase.value,
                "requested": requested, "actor_eligible": True,
                "coverage_pRelease": _p_release(policy, observation),
                "state_only_teacher": label,
            })
            next_sample_step += SAMPLE_STRIDE_STEPS
        result = original_step(*args, **kwargs)
        last_state = _uav_state(env, TARGET_UID)
        return result

    def audited_release(uav: Any, intent: Any, shortlist: Any = None) -> Any:
        if int(uav.uid) != TARGET_UID:
            return original_release(uav, intent, shortlist)
        before_telemetry = dict(env.telemetry)
        before = {
            "step": int(env.step_count), "time": float(env.now),
            "state": _uav_state(env, TARGET_UID),
            "intent": _intent_payload(intent),
            "shortlist": _shortlist_payload(shortlist),
            "requested": None,
            "energy": _energy_diagnostic(env),
        }
        result: Any = None
        error: str | None = None
        try:
            result = original_release(uav, intent, shortlist)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            releases.append({
                "step": int(before["step"]), "time": float(before["time"]),
                "returned": result, "error": error, "before": before,
                "after": {
                    "state": _uav_state(env, TARGET_UID),
                    "energy": _energy_diagnostic(env),
                    "telemetry_delta": _telemetry_delta(before_telemetry, env.telemetry),
                },
            })

    def audited_emergency(uav: Any, *, force: bool = False) -> bool:
        if int(uav.uid) != TARGET_UID:
            return original_emergency(uav, force=force)
        before_telemetry = dict(env.telemetry)
        before = {
            "step": int(env.step_count), "time": float(env.now),
            "state": _uav_state(env, TARGET_UID), "force": bool(force),
            "energy": _energy_diagnostic(env),
        }
        result: bool | None = None
        error: str | None = None
        try:
            result = bool(original_emergency(uav, force=force))
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            emergencies.append({
                "step": int(before["step"]), "time": float(before["time"]),
                "returned": result, "error": error, "before": before,
                "after": {
                    "state": _uav_state(env, TARGET_UID),
                    "energy": _energy_diagnostic(env),
                    "telemetry_delta": _telemetry_delta(before_telemetry, env.telemetry),
                },
            })

    env.step = observed_step  # type: ignore[method-assign]
    env._release = audited_release  # type: ignore[method-assign]
    env._try_emergency_landing = audited_emergency  # type: ignore[method-assign]
    try:
        replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    finally:
        env.step = original_step  # type: ignore[method-assign]
        env._release = original_release  # type: ignore[method-assign]
        env._try_emergency_landing = original_emergency  # type: ignore[method-assign]
    _require(replay.get("exact") is True, f"coverage tape replay was not exact: {replay}")
    _require(first_present_step is not None, "UID 134 was never active in replay")
    identity_ok = all(int(row["arrival_id"]) == TARGET_ARRIVAL_ID for row in identity_checks)
    _require(identity_ok, f"UID 134 arrival identity mismatch: {identity_checks[:3]}")
    report = {
        "stage": STAGE,
        "run_index": RUN_INDEX,
        "target_uid": TARGET_UID,
        "target_arrival_id": TARGET_ARRIVAL_ID,
        "seed": EXPECTED_SEED,
        "scenario": "dense24",
        "config_v4_2_rejected_release_hold_mode": "canonical_wait",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "action_tape": str(tape),
        "action_tape_sha256": _sha256(tape),
        "replay": {key: value for key, value in replay.items() if key != "steps"},
        "exact": True,
        "identity": {
            "first_present_step": first_present_step,
            "arrival_id_all_observed_16": identity_ok,
            "observed_rows": len(identity_checks),
        },
        "outer_entry_step": outer_entry_step,
        "sample_stride_steps": SAMPLE_STRIDE_STEPS,
        "samples": samples,
        "release_attempts": releases,
        "emergency_attempts": emergencies,
        "final_target_state": last_state,
        "evaluation": {
            "policy_sampling": False,
            "environment_teacher_substitution": False,
            "state_only_deepcopy_teacher": True,
            "policy_update": False,
            "guards_modified": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay_diagnostic(tape=args.tape, checkpoint=args.checkpoint, output=args.output)
    print(json.dumps({
        "stage": report["stage"], "exact": report["exact"],
        "target_uid": report["target_uid"], "target_arrival_id": report["target_arrival_id"],
        "outer_entry_step": report["outer_entry_step"],
        "samples": len(report["samples"]),
        "release_attempts": len(report["release_attempts"]),
        "emergency_attempts": len(report["emergency_attempts"]),
        "output": str(args.output),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
