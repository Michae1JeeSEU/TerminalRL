"""Descriptive P3 dense24 safety-distance/energy diagnostic.

This module is intentionally separate from the frozen P3 training and
qualification runners.  It constructs a small paired matrix over
``safe_distance`` and controller, executes each case without updates, and
records the runtime separation/energy evidence that is available from the
canonical action tape replay.

The diagnostic uses the V2 dense24 scenario builder and changes only the
requested engineering sensitivity.  It is not a training protocol and never
authorizes PPO or checkpoint promotion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import (
    diagnose_v4_2_p3_mixed_priority_soc_pilot_v2 as p3_v2,
)
from terminal_marl.training import (
    evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness,
)
from terminal_marl.training import (
    run_v4_2_curriculum_pilot as curriculum,
)
from terminal_marl.training import student_rollout_v4_2


STAGE = "V4_2_P3_SAFETY_ENERGY_SENSITIVITY_DIAGNOSTIC_V1"
SCENARIO = "dense24"
SCENARIO_PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v4.yaml"
)
DEFAULT_SAFE_DISTANCES = (60.0, 80.0, 100.0)
DEFAULT_CONTROLLERS = ("rule", "original_k3")
PROTECTED_FLOOR_M = 40.0
COLLISION_DISTANCE_M = 10.0
ENERGY_CAPACITY = 1500.0
FULL_HORIZON_SECONDS = 1800.0
ORIGINAL_K3_SHA256 = "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
CONTROLLERS = frozenset(DEFAULT_CONTROLLERS)


class DiagnosticInputError(ValueError):
    """Raised for invalid descriptive-diagnostic inputs."""


def _jsonable(value: Any) -> Any:
    """Convert common simulator values into strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _jsonable(value.value)
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write progress/manifest JSON without exposing a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_scenario_protocol(path: str | Path = SCENARIO_PROTOCOL) -> dict[str, Any]:
    """Load the existing P3 V4 YAML template without changing it."""
    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DiagnosticInputError("scenario protocol must be a mapping")
    if SCENARIO not in value.get("scenarios", {}):
        raise DiagnosticInputError("dense24 scenario is missing from the template")
    return value


def scenario_config(
    safe_distance: float,
    *,
    raw_protocol: Mapping[str, Any] | None = None,
):
    """Build one dense24 config with the diagnostic safety/energy constants."""
    distance = float(safe_distance)
    if not math.isfinite(distance) or distance <= PROTECTED_FLOOR_M:
        raise DiagnosticInputError(
            "safe_distance must be finite and greater than the 40 m protected floor"
        )
    raw = dict(raw_protocol) if raw_protocol is not None else load_scenario_protocol()
    config = p3_v2.scenario_config(SCENARIO, raw)
    # The sensitivity matrix changes only the operating separation.  Keep the
    # 40 m protected floor, 10 m collision threshold, EH216-S reference energy
    # budget, and full dense24 drain horizon fixed for every paired case.
    return replace(
        config,
        safe_distance=distance,
        nmac_distance=PROTECTED_FLOOR_M,
        collision_distance=COLLISION_DISTANCE_M,
        energy_capacity=ENERGY_CAPACITY,
        max_simulation_seconds=FULL_HORIZON_SECONDS,
    )


def _normalise_controllers(controllers: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(item) for item in controllers)
    if not values:
        raise DiagnosticInputError("at least one controller is required")
    unknown = sorted(set(values) - CONTROLLERS)
    if unknown:
        raise DiagnosticInputError(f"unsupported controller(s): {unknown}")
    # Preserve the caller's order while avoiding duplicate simulations.
    return tuple(dict.fromkeys(values))


def build_case_matrix(
    seeds: Sequence[int],
    safe_distances: Sequence[float] = DEFAULT_SAFE_DISTANCES,
    controllers: Sequence[str] = DEFAULT_CONTROLLERS,
) -> list[dict[str, Any]]:
    """Return deterministic seed × distance × controller case descriptors."""
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise DiagnosticInputError("at least one diagnostic seed is required")
    if any(seed < 0 for seed in seed_values):
        raise DiagnosticInputError("diagnostic seeds must be non-negative")
    if len(set(seed_values)) != len(seed_values):
        raise DiagnosticInputError("diagnostic seeds must be unique")
    distance_values = tuple(float(value) for value in safe_distances)
    if not distance_values:
        raise DiagnosticInputError("at least one safe distance is required")
    if len(set(distance_values)) != len(distance_values):
        raise DiagnosticInputError("safe distances must be unique")
    # Validate before returning any partially valid matrix.
    for distance in distance_values:
        scenario_config(distance)
    controller_values = _normalise_controllers(controllers)
    cases: list[dict[str, Any]] = []
    for seed in seed_values:
        for distance in distance_values:
            for controller in controller_values:
                distance_tag = f"{distance:g}".replace(".", "p")
                cases.append({
                    "case_id": f"dense24_seed{seed}_safe{distance_tag}_{controller}",
                    "scenario": SCENARIO,
                    "seed": seed,
                    "safe_distance": distance,
                    "controller": controller,
                })
    return cases


# Stable aliases for small callers/tests that prefer a noun phrase.
matrix_cases = build_case_matrix
construct_case_matrix = build_case_matrix


def _outcome_counts(outcomes: Mapping[Any, Any] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_key, raw_value in (outcomes or {}).items():
        key = str(raw_key)
        value = getattr(raw_value, "value", raw_value)
        name = str(value)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _completed_only_metrics(env: TerminalParallelEnv) -> dict[str, Any]:
    """Compute delay metrics only from aircraft with completed touchdown."""
    rows = list(env.completed_uavs)
    waits: list[float] = []
    request_to_touchdown: list[float] = []
    for uav in rows:
        try:
            waits.append(float(uav.external_wait_time + uav.internal_wait_time))
            request_to_touchdown.append(float(uav.touchdown_time - uav.request_time))
        except (TypeError, ValueError):
            # A malformed lifecycle row is not silently converted to zero.
            continue
    return {
        "completed_only_count": len(rows),
        "completed_only_metrics_defined": bool(rows) and len(waits) == len(rows),
        "mean_wait_seconds_completed_only": None if not waits else float(np.mean(waits)),
        "mean_request_to_touchdown_seconds_completed_only": (
            None if not request_to_touchdown else float(np.mean(request_to_touchdown))
        ),
        "condition": "completed_uavs_only; null when no completed UAVs",
    }


def _telemetry_defaults() -> dict[str, Any]:
    return {
        "minimum_actual_separation": None,
        "minimum_swept_separation": None,
        "operational_buffer_entries": 0,
        "operational_buffer_exposure_steps": 0,
        "hard_separation_violations": 0,
        "hard_separation_exposure_steps": 0,
        "nmac": 0,
        "nmac_exposure_steps": 0,
        "collisions": 0,
        "collision_exposure_steps": 0,
    }


def _replay_telemetry(
    config: Any,
    tape_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """Replay one tape to expose authoritative separation and lifecycle facts."""
    telemetry = _telemetry_defaults()
    if not tape_path:
        return telemetry, None, "action tape path missing", None
    env = TerminalParallelEnv(config)
    try:
        replay = SafetyTraceRecorder.replay_action_tape(env, tape_path)
    except Exception as exc:  # retain the original run and report replay separately
        return telemetry, None, f"{type(exc).__name__}: {exc}", None
    for key in telemetry:
        value = env.telemetry.get(key)
        if value is not None:
            numeric = _jsonable(value)
            if key.startswith("minimum_") and isinstance(numeric, (int, float)) and not math.isfinite(float(numeric)):
                numeric = None
            telemetry[key] = numeric
    return (
        telemetry,
        {"exact": bool(replay.get("exact")),
         "steps_replayed": int(replay.get("steps_replayed", 0)),
         "steps_expected": int(replay.get("steps_expected", 0))},
        None,
        _completed_only_metrics(env),
    )


def _base_record(case: Mapping[str, Any], *, case_dir: Path) -> dict[str, Any]:
    return {
        **dict(case),
        "status": "failed",
        "requests": None,
        "completed": None,
        "terminal": False,
        "outcomes": {},
        "unresolved": None,
        "low_soc_diversions": 0,
        "minimum_actual_separation": None,
        "minimum_swept_separation": None,
        "operational_buffer_entries": 0,
        "operational_buffer_exposure_steps": 0,
        "shield_uses": 0,
        "shield_actor_uses": None,
        "hard_separation_violations": 0,
        "nmac": 0,
        "collisions": 0,
        "safety_invariant_errors": 0,
        "uncertified_execution": 0,
        "exception": None,
        "elapsed_sim_time_seconds": None,
        "mean_wait_seconds_completed_only": None,
        "mean_request_to_touchdown_seconds_completed_only": None,
        "completed_only_count": 0,
        "completed_only_metrics_defined": False,
        "completed_only_condition": "completed_uavs_only; null when no completed UAVs",
        "replay": None,
        "replay_error": None,
        "artifacts": {
            "case_directory": str(case_dir),
            "summary": str(case_dir / "summary.json"),
            "action_tape": None,
        },
    }


def _merge_summary_record(
    record: dict[str, Any],
    summary: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    replay: Mapping[str, Any] | None,
    replay_error: str | None,
) -> dict[str, Any]:
    outcomes = _outcome_counts(summary.get("outcomes"))
    safety = summary.get("safety") if isinstance(summary.get("safety"), Mapping) else {}
    requests = int(summary.get("requests", record["requests"] or 0))
    record.update({
        "status": "completed",
        "requests": requests,
        "completed": int(summary.get("completed", 0)),
        "terminal": bool(summary.get("terminal", False)),
        "outcomes": outcomes,
        "unresolved": max(0, requests - sum(outcomes.values())),
        "low_soc_diversions": int(outcomes.get("diverted_low_soc", 0)),
        "shield_uses": int(summary.get("shield", (summary.get("counts") or {}).get("shield", 0))),
        "shield_actor_uses": (
            int(summary.get("shield_actor", (summary.get("counts") or {}).get("shield_actor", 0)))
            if "shield_actor" in summary or "counts" in summary else None
        ),
        "exception": summary.get("safety_error"),
        "elapsed_sim_time_seconds": float(summary.get("seconds", 0.0)),
        "replay": None if replay is None else dict(replay),
        "replay_error": replay_error,
    })
    for key in (
        "minimum_actual_separation", "minimum_swept_separation",
        "operational_buffer_entries", "operational_buffer_exposure_steps",
        "hard_separation_violations", "nmac", "collisions",
    ):
        record[key] = telemetry.get(key)
    record["safety_invariant_errors"] = int(
        safety.get("SafetyInvariantError", 0)
    )
    record["uncertified_execution"] = int(safety.get("uncertified_execution", 0))
    record["hard_separation_violations"] = int(
        telemetry.get("hard_separation_violations", safety.get("hard_separation_violations", 0)) or 0
    )
    record["nmac"] = int(telemetry.get("nmac", safety.get("nmac", 0)) or 0)
    record["collisions"] = int(telemetry.get("collisions", safety.get("collisions", 0)) or 0)
    return record


def run_case(
    case: Mapping[str, Any],
    *,
    output: Path,
    policy: Any | None = None,
    raw_protocol: Mapping[str, Any] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run one case and always return a serializable record."""
    case_dir = output / str(case["case_id"])
    record = _base_record(case, case_dir=case_dir)
    try:
        config = scenario_config(float(case["safe_distance"]), raw_protocol=raw_protocol)
        record["requests"] = int(config.total_requested_uavs)
        if case_dir.exists():
            raise FileExistsError(f"case output already exists: {case_dir}")
        max_steps_value = (
            int(max_steps)
            if max_steps is not None
            else int(round(FULL_HORIZON_SECONDS / float(config.dt)))
        )
        if max_steps_value < 1:
            raise DiagnosticInputError("max_steps must be positive")
        if str(case["controller"]) == "original_k3":
            if policy is None:
                raise DiagnosticInputError("original_k3 requires a loaded checkpoint")
            _, summary = student_rollout_v4_2.rollout(
                policy,
                config,
                seed=int(case["seed"]),
                max_steps=max_steps_value,
                output=case_dir,
                deterministic=True,
                collect_labels=False,
            )
        elif str(case["controller"]) == "rule":
            summary = curriculum.episode(
                config,
                int(case["seed"]),
                case_dir,
                policy=None,
                max_steps=max_steps_value,
            )
        else:
            raise DiagnosticInputError(f"unsupported controller: {case['controller']}")
        tape_path = summary.get("action_tape")
        record["artifacts"]["action_tape"] = None if tape_path is None else str(tape_path)
        telemetry, replay, replay_error, lifecycle = _replay_telemetry(config, tape_path)
        record = _merge_summary_record(record, summary, telemetry, replay, replay_error)
        # Completed-only lifecycle metrics come from the same replay that
        # supplied the authoritative separation telemetry.  Missing replay
        # evidence remains null rather than being converted to zero.
        if lifecycle is not None:
            record.update(lifecycle)
        return record
    except Exception as exc:
        record["exception"] = f"{type(exc).__name__}: {exc}"
        return record


def run_matrix(
    *,
    output: str | Path,
    checkpoint: str | Path | None,
    seeds: Sequence[int],
    safe_distances: Sequence[float] = DEFAULT_SAFE_DISTANCES,
    controllers: Sequence[str] = DEFAULT_CONTROLLERS,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Execute the diagnostic matrix and write progress/manifest JSON."""
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"diagnostic output already exists: {output_path}")
    cases = build_case_matrix(seeds, safe_distances, controllers)
    controller_values = _normalise_controllers(controllers)
    policy = None
    checkpoint_sha256 = None
    if "original_k3" in controller_values:
        if checkpoint is None:
            raise DiagnosticInputError("--checkpoint is required for original_k3")
        checkpoint_path = Path(checkpoint)
        checkpoint_sha256 = effectiveness._sha256(checkpoint_path)
        if checkpoint_sha256 != ORIGINAL_K3_SHA256:
            raise DiagnosticInputError("checkpoint is not the frozen original K3 candidate")
        policy, _ = effectiveness.load_candidate_checkpoint(checkpoint_path)
    raw_protocol = load_scenario_protocol()
    output_path.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for case in cases:
        record = run_case(
            case,
            output=output_path,
            policy=policy,
            raw_protocol=raw_protocol,
            max_steps=max_steps,
        )
        records.append(record)
        _write_json_atomic(output_path / "progress.json", {
            "stage": STAGE,
            "diagnostic_only": True,
            "completed_records": len(records),
            "expected_records": len(cases),
            "last_case": record,
        })
    manifest = {
        "stage": STAGE,
        "diagnostic_only": True,
        "PPO_ALLOWED": False,
        "qualification": False,
        "scenario": SCENARIO,
        "scenario_protocol": str(SCENARIO_PROTOCOL),
        "constants": {
            "max_simulation_seconds": FULL_HORIZON_SECONDS,
            "energy_capacity": ENERGY_CAPACITY,
            "protected_floor_m": PROTECTED_FLOOR_M,
            "collision_distance_m": COLLISION_DISTANCE_M,
        },
        "matrix": {
            "seeds": [int(value) for value in seeds],
            "safe_distances": [float(value) for value in safe_distances],
            "controllers": list(controller_values),
            "expected_records": len(cases),
        },
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "records": records,
        "completed_records": len(records),
        "failed_records": sum(row.get("status") != "completed" for row in records),
    }
    _write_json_atomic(output_path / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--safe-distances", type=float, nargs="+", default=list(DEFAULT_SAFE_DISTANCES)
    )
    parser.add_argument(
        "--controllers", nargs="+", choices=sorted(CONTROLLERS),
        default=list(DEFAULT_CONTROLLERS),
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="optional bounded smoke horizon; omit for the 1800 s dense24 horizon",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    result = run_matrix(
        output=args.output,
        checkpoint=args.checkpoint,
        seeds=args.seeds,
        safe_distances=args.safe_distances,
        controllers=args.controllers,
        max_steps=args.max_steps,
    )
    print(json.dumps({
        "stage": result["stage"],
        "output": str(args.output),
        "records": result["completed_records"],
        "failed_records": result["failed_records"],
        "PPO_ALLOWED": result["PPO_ALLOWED"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
