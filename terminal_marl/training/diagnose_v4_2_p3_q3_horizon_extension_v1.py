"""Diagnostic-only horizon extension for the exposed Q3 dense24 episode.

The experiment keeps the frozen V3 dense24 configuration unchanged except for
the simulation cap.  It evaluates checkpoint 5, the original K=3 checkpoint,
and the rule controller at 1800 seconds.  A safety exception stops only the
current controller; the remaining controllers still run.  The result is
descriptive evidence about the cap and is never a qualification or promotion
decision.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import (
    evaluate_v4_2_p3_v3_early_stop_q3_v1 as q3,
)
from terminal_marl.training import (
    run_v4_2_p3_mixed_priority_soc_training_v3 as v3,
)
from terminal_marl.training import student_rollout_v4_2
from terminal_marl.training.run_v4_2_curriculum_pilot import (
    episode as rule_episode,
)


STAGE = "V4_2_P3_Q3_HORIZON_EXTENSION_V1"
STATUS = "PREREGISTERED_EXPOSED_SEED_DIAGNOSTIC"
SEED = 826006
SCENARIO = "dense24"
CONTROLLERS = ("trained", "initial", "rule")
BASE_HORIZON_SECONDS = 1080.0
EXTENDED_HORIZON_SECONDS = 1800.0
EXPECTED_REQUESTS = 24
TRAINED_CHECKPOINT_SHA = (
    "19c31195dd881e7e4fde91530aff00375f2787754569491749474699b7f06b2d"
)
INITIAL_CHECKPOINT_SHA = (
    "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
)
TRAINING_PROTOCOL_SHA = (
    "c69dc4b308e97652936c8da5bb6a47ef61a73a4f3cb599de8f39cb703f3c4d25"
)
Q3_PROTOCOL_SHA = q3.PROTOCOL_SHA
LOW_SOC_THRESHOLD = 0.65
REWARD_CONFIG = {
    "low_level_priority_time_weight": 1.0,
    "low_level_soc_urgency_time_weight": 2.0,
    "low_level_soc_urgency_threshold": 0.65,
}


class HorizonExtensionInputError(ValueError):
    """Raised when a frozen diagnostic input is not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HorizonExtensionInputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(path: str | Path) -> str:
    return _sha256(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def _payload_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep lineage metadata without serializing model tensors into JSON."""
    keys = (
        "stage",
        "formal",
        "ppo_called",
        "protocol_sha256",
        "initial_checkpoint_sha256",
        "checkpoint_promotable",
        "urgency_reward_config",
        "policy_architecture_migrated",
        "metadata",
    )
    return {
        key: _jsonable(payload[key])
        for key in keys
        if key in payload
    }


def protocol() -> dict[str, Any]:
    """Return the immutable diagnostic contract carried by this runner."""
    return {
        "protocol": STAGE,
        "status": STATUS,
        "scope": "exposed_q3_dense24_cap_extension",
        "bindings": {
            "seed": SEED,
            "scenario": SCENARIO,
            "controllers": list(CONTROLLERS),
            "base_horizon_seconds": BASE_HORIZON_SECONDS,
            "extended_horizon_seconds": EXTENDED_HORIZON_SECONDS,
            "expected_requests": EXPECTED_REQUESTS,
            "trained_checkpoint_sha256": TRAINED_CHECKPOINT_SHA,
            "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
            "training_protocol_sha256": TRAINING_PROTOCOL_SHA,
            "q3_protocol_sha256": Q3_PROTOCOL_SHA,
        },
        "scenario": {
            "deterministic": True,
            "dt": 1.0,
            "total_requested_uavs": EXPECTED_REQUESTS,
            "same_config_except_max_simulation_seconds": True,
            "no_teacher": True,
            "no_labels": True,
            "no_updates": True,
            "no_optimizer": True,
            "reward": dict(REWARD_CONFIG),
        },
        "measurement": {
            "completed_only_waits": True,
            "censored_when_incomplete": True,
            "safety_exception_stops_current_controller_only": True,
            "first_last_touchdown": True,
            "final_active_and_queue": True,
        },
        "authorization": {
            "diagnostic_only": True,
            "exposed_seed_diagnostic_only": True,
            "PASS_means_run_integrity_only": True,
            "P3_QUALIFIED": False,
            "checkpoint_promotable": False,
            "P4_ALLOWED": False,
            "no_seed_reservation": True,
            "no_seed_consumption": True,
        },
    }


def _validate_q3_lineage(q3_protocol: Mapping[str, Any]) -> None:
    _require(q3_protocol.get("protocol") == q3.STAGE, "Q3 protocol stage changed")
    bindings = q3_protocol.get("bindings")
    _require(isinstance(bindings, Mapping), "Q3 protocol bindings missing")
    _require(
        bindings.get("trained_checkpoint_sha256") == TRAINED_CHECKPOINT_SHA,
        "Q3 trained checkpoint binding changed",
    )
    _require(
        bindings.get("initial_checkpoint_sha256") == INITIAL_CHECKPOINT_SHA,
        "Q3 initial checkpoint binding changed",
    )
    _require(
        bindings.get("training_protocol_sha256") == TRAINING_PROTOCOL_SHA,
        "Q3 training protocol binding changed",
    )


def _extended_config() -> tuple[Any, Any]:
    training = v3.protocol()
    base = q3.scenario_config(SCENARIO, training)
    _require(
        float(base.max_simulation_seconds) == BASE_HORIZON_SECONDS,
        "base dense24 horizon changed",
    )
    extended = replace(base, max_simulation_seconds=EXTENDED_HORIZON_SECONDS)
    for field in fields(base):
        if field.name == "max_simulation_seconds":
            continue
        _require(
            getattr(base, field.name) == getattr(extended, field.name),
            f"extended config changed frozen field: {field.name}",
        )
    for key, expected in {
        "dt": 1.0,
        "total_requested_uavs": EXPECTED_REQUESTS,
        "teacher_enabled": False,
        "teacher_release_label_enabled": False,
        "low_level_priority_time_weight": 1.0,
        "low_level_soc_urgency_time_weight": 2.0,
        "low_level_soc_urgency_threshold": LOW_SOC_THRESHOLD,
    }.items():
        _require(getattr(extended, key) == expected, f"extended config changed: {key}")
    _require(
        float(extended.max_simulation_seconds) == EXTENDED_HORIZON_SECONDS,
        "extended horizon was not applied",
    )
    return base, extended


def _validate_checkpoint_inputs(
    trained_checkpoint: Path, initial_checkpoint: Path, output: Path
) -> None:
    _require(not output.exists(), f"diagnostic output already exists: {output}")
    for path, expected, label in (
        (trained_checkpoint, TRAINED_CHECKPOINT_SHA, "trained checkpoint"),
        (initial_checkpoint, INITIAL_CHECKPOINT_SHA, "initial checkpoint"),
    ):
        _require(path.is_file(), f"{label} missing: {path}")
        _require(_sha(path) == expected, f"{label} SHA mismatch")


def _load_policies(
    trained_checkpoint: Path, initial_checkpoint: Path
) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any]]:
    q3_protocol = q3.protocol()
    _validate_q3_lineage(q3_protocol)
    trained, trained_payload = q3.load_trained_checkpoint(
        trained_checkpoint, q3_protocol
    )
    initial, initial_payload = q3.load_initial_checkpoint(
        initial_checkpoint, q3_protocol
    )
    return trained, initial, trained_payload, initial_payload


def _completed_waits(completed: list[Any]) -> dict[str, Any]:
    waits = [
        float(uav.external_wait_time + uav.internal_wait_time)
        for uav in completed
    ]
    request_to_touchdown = [
        float(uav.touchdown_time - uav.request_time)
        for uav in completed
    ]
    return {
        "completed_count": len(completed),
        "completed_only": True,
        "censored": len(completed) < EXPECTED_REQUESTS,
        "mean_wait_seconds": None if not waits else float(np.mean(waits)),
        "mean_request_to_touchdown_seconds": (
            None if not request_to_touchdown else float(np.mean(request_to_touchdown))
        ),
    }


def _lifecycle_snapshot(config: Any, tape_path: str | Path) -> dict[str, Any]:
    """Replay one produced tape only to collect terminal/lifecycle metrics."""
    env = TerminalParallelEnv(config)
    replay_error = None
    replay = {"exact": False, "steps_replayed": 0, "steps_expected": None}
    if not tape_path:
        replay_error = "missing action tape path"
    else:
        try:
            replay = SafetyTraceRecorder.replay_action_tape(env, tape_path)
        except Exception as exc:  # preserve raw diagnostic, do not qualify on it
            replay_error = f"{type(exc).__name__}: {exc}"
    completed = list(env.completed_uavs)
    touchdowns = [
        float(uav.touchdown_time)
        for uav in completed
        if uav.touchdown_time is not None and math.isfinite(float(uav.touchdown_time))
    ]
    return {
        "replay_exact": bool(replay.get("exact", False)),
        "replay_steps_replayed": int(replay.get("steps_replayed", 0)),
        "replay_steps_expected": replay.get("steps_expected"),
        "replay_error": replay_error,
        "first_touchdown_seconds": None if not touchdowns else min(touchdowns),
        "last_touchdown_seconds": None if not touchdowns else max(touchdowns),
        "final_active": len(env.active),
        "final_queue": len(env.admission.queue.items),
        "arrivals_generated": int(env.arrivals.generated),
        "completed_after_replay": len(completed),
        "waits": _completed_waits(completed),
    }


def _summary_record(
    *,
    controller: str,
    checkpoint_sha256: str | None,
    summary: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    completed = int(summary.get("completed", lifecycle.get("completed_after_replay", 0)))
    requests = int(summary.get("requests", EXPECTED_REQUESTS))
    tape_path = summary.get("action_tape")
    tape_exists = bool(tape_path and Path(str(tape_path)).is_file())
    safety = dict(summary.get("safety", {}))
    return {
        "controller": controller,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": SEED,
        "scenario": SCENARIO,
        "horizon_seconds": EXTENDED_HORIZON_SECONDS,
        "requests": requests,
        "completed": completed,
        "completion_fraction": completed / max(1, requests),
        "terminal": bool(summary.get("terminal", False)),
        "seconds": float(summary.get("seconds", 0.0)),
        "safety": safety,
        "safety_error": summary.get("safety_error"),
        "first_touchdown_seconds": lifecycle.get("first_touchdown_seconds"),
        "last_touchdown_seconds": lifecycle.get("last_touchdown_seconds"),
        "final_active": lifecycle.get("final_active", summary.get("remaining_active")),
        "final_queue": lifecycle.get("final_queue", summary.get("remaining_queue")),
        "arrivals_generated": lifecycle.get("arrivals_generated"),
        "waits": lifecycle.get("waits"),
        "replay_exact": lifecycle.get("replay_exact", False),
        "replay_steps_replayed": lifecycle.get("replay_steps_replayed", 0),
        "replay_steps_expected": lifecycle.get("replay_steps_expected"),
        "replay_error": lifecycle.get("replay_error"),
        "action_tape": None if tape_path is None else str(tape_path),
        "summary_path": str(output / "summary.json"),
        "action_tape_exists": tape_exists,
        "deterministic": True,
        "teacher_enabled": False,
        "no_labels": True,
        "no_updates": True,
        "no_optimizer": True,
        "diagnostic_only": True,
        "exposed_seed_diagnostic_only": True,
        "run_integrity": bool(tape_exists and "safety" in summary
                              and lifecycle.get("replay_exact") is True),
        "qualification": False,
        "checkpoint_promotable": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
    }


def _run_controller(
    *,
    controller: str,
    policy: Any | None,
    config: Any,
    output: Path,
    checkpoint_sha256: str | None,
) -> dict[str, Any]:
    max_steps = int(round(EXTENDED_HORIZON_SECONDS / float(config.dt)))
    try:
        if controller == "rule":
            summary = dict(rule_episode(config, SEED, output, max_steps=max_steps))
        else:
            _, summary = student_rollout_v4_2.rollout(
                policy,
                config,
                seed=SEED,
                max_steps=max_steps,
                output=output,
                deterministic=True,
                collect_labels=False,
            )
    except SafetyInvariantError as exc:
        # Existing controllers normally catch this internally.  Keep a raw
        # escaped error as diagnostic evidence and continue the other cases.
        return {
            "controller": controller,
            "checkpoint_sha256": checkpoint_sha256,
            "seed": SEED,
            "scenario": SCENARIO,
            "horizon_seconds": EXTENDED_HORIZON_SECONDS,
            "safety": {"SafetyInvariantError": 1},
            "safety_error": str(exc),
            "exception_type": type(exc).__name__,
            "diagnostic_only": True,
            "exposed_seed_diagnostic_only": True,
            "run_integrity": False,
            "qualification": False,
            "checkpoint_promotable": False,
            "P3_QUALIFIED": False,
            "P4_ALLOWED": False,
        }
    lifecycle = _lifecycle_snapshot(config, summary.get("action_tape"))
    return _summary_record(
        controller=controller,
        checkpoint_sha256=checkpoint_sha256,
        summary=summary,
        lifecycle=lifecycle,
        output=output,
    )


def run(
    *, trained_checkpoint: Path, initial_checkpoint: Path, output: Path
) -> dict[str, Any]:
    """Run the three diagnostic controllers and write descriptive evidence."""
    p = protocol()
    _validate_checkpoint_inputs(trained_checkpoint, initial_checkpoint, output)
    trained, initial, trained_payload, initial_payload = _load_policies(
        trained_checkpoint, initial_checkpoint
    )
    base_config, config = _extended_config()
    output.mkdir(parents=True, exist_ok=False)
    specs = (
        ("trained", trained, TRAINED_CHECKPOINT_SHA),
        ("initial", initial, INITIAL_CHECKPOINT_SHA),
        ("rule", None, None),
    )
    records: list[dict[str, Any]] = []
    for controller, policy, checkpoint_sha in specs:
        folder = output / controller
        record = _run_controller(
            controller=controller,
            policy=policy,
            config=config,
            output=folder,
            checkpoint_sha256=checkpoint_sha,
        )
        record["base_horizon_seconds"] = BASE_HORIZON_SECONDS
        record["configuration_changed_only_horizon"] = True
        _write(folder / "horizon_evaluation.json", record)
        records.append(record)
    run_integrity = len(records) == len(CONTROLLERS) and all(
        bool(record.get("run_integrity")) for record in records
    )
    manifest = {
        **p,
        "PASS": bool(run_integrity),
        "run_integrity": bool(run_integrity),
        "diagnostic_only": True,
        "exposed_seed_diagnostic_only": True,
        "seed": SEED,
        "scenario": SCENARIO,
        "controllers": list(CONTROLLERS),
        "base_horizon_seconds": BASE_HORIZON_SECONDS,
        "extended_horizon_seconds": EXTENDED_HORIZON_SECONDS,
        "trained_checkpoint_sha256": TRAINED_CHECKPOINT_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "training_protocol_sha256": TRAINING_PROTOCOL_SHA,
        "base_config_max_simulation_seconds": float(base_config.max_simulation_seconds),
        "extended_config_max_simulation_seconds": float(config.max_simulation_seconds),
        "trained_checkpoint_payload": _payload_evidence(trained_payload),
        "initial_checkpoint_payload": _payload_evidence(initial_payload),
        "records": records,
        "training_called": False,
        "optimizer_loaded": False,
        "seed_reserved": False,
        "seed_consumed": False,
        "qualification": False,
        "checkpoint_promotable": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
    }
    _write(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        trained_checkpoint=args.trained_checkpoint,
        initial_checkpoint=args.initial_checkpoint,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "stage": result["protocol"],
                "PASS": result["PASS"],
                "diagnostic_only": result["diagnostic_only"],
                "records": len(result["records"]),
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
