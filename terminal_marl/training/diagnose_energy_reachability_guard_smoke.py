"""Bounded, rule-only smoke diagnostic for the route-aware energy guard.

The entry point intentionally constructs the frozen V4.2 / Phase-A scenario
directly in code.  It does not load a checkpoint, invoke a Teacher, train, or
modify the simulator.  The command writes one JSON file and refuses to
overwrite an existing path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from enum import Enum
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from terminal_marl.core.config import SimConfig
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.envs.parallel_env import TerminalParallelEnv


ARRIVAL_SCHEDULE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
SUCCESS_OUTCOMES = {"completed", "emergency_landed"}
FAILURE_OUTCOMES = {
    "failed_collision", "failed_energy", "diverted_low_soc",
    "diverted_capacity", "invalid", "unresolved",
}


def _jsonable(value: Any) -> Any:
    """Convert simulator values to strict JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _terminal(env: TerminalParallelEnv) -> bool:
    return bool(
        env.arrivals.exhausted()
        and not env.active
        and not env.admission.queue.items
        and not env.service_uavs
    )


def _progress_payload(
    env: TerminalParallelEnv,
    *,
    step: int,
    wall_elapsed: float,
    safety_error: str | None,
    terminal: bool,
    horizon_reached: bool,
) -> dict[str, Any]:
    """Return the small progress record written during a bounded run."""

    failed_energy = sum(
        str(row.get("outcome")) == "failed_energy"
        for row in env.energy_outcomes
    )
    return {
        "step": int(step),
        "sim_time": float(env.now),
        "wall_elapsed": float(wall_elapsed),
        "active": int(len(env.active)),
        "resolved": int(len(env.ledger.outcomes)),
        "failed_energy": int(failed_energy),
        "safety_error": safety_error,
        "terminal": bool(terminal),
        "horizon_reached": bool(horizon_reached),
        "status": (
            "terminal" if terminal else
            "safety_error" if safety_error is not None else
            "horizon" if horizon_reached else "running"
        ),
    }


def _write_progress_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace the run-owned progress file and emit one line."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(json.dumps(_jsonable(payload), sort_keys=True), flush=True)


def build_config(
    *,
    initial_soc_min: float,
    initial_soc_max: float,
    route_guard_enabled: bool,
    max_simulation_seconds: float,
) -> SimConfig:
    """Construct the fixed V4.2 / V2.3 smoke scenario."""

    return SimConfig(
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version="phaseA_low_obs_v2_3_source_bound_88d",
        neighbor_comm_range_m=320.0,
        experiment_mode="fixed_rule",
        high_level_enabled=False,
        reset_mode="warm_start_rings",
        warm_start_uavs=6,
        warm_start_steady_motion=True,
        warm_start_energy_reachability_check_enabled=True,
        total_requested_uavs=12,
        arrival_model="finite_cohort",
        arrival_rate_per_second=0.0,
        fixed_arrival_schedule=ARRIVAL_SCHEDULE,
        initial_soc_mode="uniform",
        initial_soc_min=float(initial_soc_min),
        initial_soc_max=float(initial_soc_max),
        max_simulation_seconds=float(max_simulation_seconds),
        energy_model_mode="phase_motion_v2",
        emergency_landing_enabled=True,
        emergency_priority_guard_enabled=True,
        energy_reachability_guard_enabled=bool(route_guard_enabled),
        emergency_trigger_soc=0.40,
        emergency_energy_margin_seconds=15.0,
        lz_assignment_mode="nearest_available",
        release_arbitration_mode="downstream_priority",
        final_queue_capacity=3,
        safe_distance=80.0,
        nmac_distance=40.0,
        collision_distance=10.0,
        teacher_enabled=False,
        teacher_mode="teacher_full",
        teacher_release_label_enabled=False,
        scheduler_mode="execution",
        shield_enabled=True,
        runtime_proposal_shortlist_k=1,
    )


def _config_summary(config: SimConfig, *, seed: int, guard_on: bool) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "schema_version": config.schema_version,
        "action_schema_version": config.action_schema_version,
        "observation_schema_version": config.observation_schema_version,
        "neighbor_comm_range_m": float(config.neighbor_comm_range_m),
        "energy_model_mode": config.energy_model_mode,
        "reset_mode": config.reset_mode,
        "warm_start_uavs": int(config.warm_start_uavs),
        "warm_start_steady_motion": bool(config.warm_start_steady_motion),
        "warm_start_energy_reachability_check_enabled": bool(
            config.warm_start_energy_reachability_check_enabled
        ),
        "fixed_arrival_schedule": list(ARRIVAL_SCHEDULE),
        "initial_soc_mode": config.initial_soc_mode,
        "initial_soc_min": float(config.initial_soc_min),
        "initial_soc_max": float(config.initial_soc_max),
        "energy_reachability_guard_enabled": bool(guard_on),
        "emergency_priority_guard_enabled": bool(config.emergency_priority_guard_enabled),
        "emergency_landing_enabled": bool(config.emergency_landing_enabled),
        "emergency_trigger_soc": float(config.emergency_trigger_soc),
        "emergency_energy_margin_seconds": float(config.emergency_energy_margin_seconds),
        "safe_distance_m": float(config.safe_distance),
        "protected_floor_m": float(config.nmac_distance),
        "collision_distance_m": float(config.collision_distance),
        "teacher_enabled": bool(config.teacher_enabled),
        "scheduler_mode": config.scheduler_mode,
        "experiment_mode": config.experiment_mode,
    }


def run_smoke(
    *,
    seed: int,
    guard_on: bool,
    initial_soc_min: float,
    initial_soc_max: float,
    max_steps: int,
    progress_every_steps: int,
    output: Path,
) -> dict[str, Any]:
    """Run a bounded environment smoke and return one JSON record."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if progress_every_steps <= 0:
        raise ValueError("progress_every_steps must be positive")
    if not 0.0 < initial_soc_min <= initial_soc_max <= 1.0:
        raise ValueError("initial SOC bounds must satisfy 0 < min <= max <= 1")

    progress_path = Path(f"{output}.progress.json")
    if progress_path.exists():
        raise FileExistsError(f"refusing to overwrite existing progress output: {progress_path}")

    config = build_config(
        initial_soc_min=initial_soc_min,
        initial_soc_max=initial_soc_max,
        route_guard_enabled=guard_on,
        max_simulation_seconds=max(1.0, float(max_steps)),
    )
    started = time.perf_counter()
    env = TerminalParallelEnv(config)
    env.reset(seed=int(seed))
    safety_error: str | None = None
    steps = 0
    terminal = _terminal(env)
    _write_progress_atomic(
        progress_path,
        _progress_payload(
            env,
            step=0,
            wall_elapsed=time.perf_counter() - started,
            safety_error=safety_error,
            terminal=terminal,
            horizon_reached=False,
        ),
    )
    while steps < max_steps and not terminal:
        # [1, 0, 0] is the explicit rule-only release request used by this
        # diagnostic. Newly admitted UAVs use the environment's same default
        # when they enter during this step.
        actions = {
            str(uid): np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            for uid in env.active
        }
        try:
            env.step(actions)
        except SafetyInvariantError as exc:
            safety_error = str(exc)
            steps += 1
            break
        steps += 1
        terminal = _terminal(env)
        if steps % progress_every_steps == 0 and not terminal and steps < max_steps:
            _write_progress_atomic(
                progress_path,
                _progress_payload(
                    env,
                    step=steps,
                    wall_elapsed=time.perf_counter() - started,
                    safety_error=safety_error,
                    terminal=terminal,
                    horizon_reached=False,
                ),
            )

    horizon_reached = bool(not terminal and steps >= max_steps)
    _write_progress_atomic(
        progress_path,
        _progress_payload(
            env,
            step=steps,
            wall_elapsed=time.perf_counter() - started,
            safety_error=safety_error,
            terminal=terminal,
            horizon_reached=horizon_reached,
        ),
    )

    outcome_rows = {
        str(arrival_id): outcome.value
        for arrival_id, outcome in env.ledger.outcomes.items()
    }
    outcome_counts = Counter(outcome_rows.values())
    requested_ids = {int(value) for value in env.ledger.requested}
    unresolved_ids = sorted(int(value) for value in env.ledger.unresolved)
    resolved_ids = {int(value) for value in env.ledger.outcomes}
    accounting_identity = (
        requested_ids == resolved_ids | set(unresolved_ids)
        and resolved_ids.isdisjoint(set(unresolved_ids))
        and len(requested_ids) == len(resolved_ids) + len(unresolved_ids)
    )
    failure_rows = [
        dict(row) for row in env.energy_outcomes
        if str(row.get("outcome")) in FAILURE_OUTCOMES
    ]
    telemetry = dict(env.telemetry)
    wall_seconds = time.perf_counter() - started
    result = {
        "diagnostic_only": True,
        "training_called": False,
        "teacher_used": False,
        "PPO_ALLOWED": False,
        "config": _config_summary(config, seed=seed, guard_on=guard_on),
        "step_count": int(env.step_count),
        "steps_executed": int(steps),
        "sim_time_seconds": float(env.now),
        "terminal": bool(terminal),
        "horizon_reached": bool(not terminal and steps >= max_steps),
        "safety_error": safety_error,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "outcomes": outcome_rows,
        "request_accounting": {
            "requested": len(requested_ids),
            "requested_ids": sorted(requested_ids),
            "resolved": len(resolved_ids),
            "unresolved": len(unresolved_ids),
            "unresolved_ids": unresolved_ids,
            "accounting_identity": bool(accounting_identity),
        },
        "safety_telemetry": telemetry,
        "failed_energy": int(outcome_counts.get("failed_energy", 0)),
        "emergency_committed": int(telemetry.get("emergency_committed", 0)),
        "release_approved": int(telemetry.get("release_approved", 0)),
        "energy_reachability_checks": int(telemetry.get("energy_reachability_checks", 0)),
        "failed_energy_outcomes": failure_rows,
        "energy_outcomes": list(env.energy_outcomes),
        "simulation_wall_time_seconds": float(wall_seconds),
        "progress_path": str(progress_path),
        "progress_every_steps": int(progress_every_steps),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--guard-on", action="store_true", dest="guard_on")
    parser.add_argument("--initial-soc-min", type=float, default=0.41)
    parser.add_argument("--initial-soc-max", type=float, default=0.55)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--progress-every-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_smoke(
        seed=args.seed,
        guard_on=bool(args.guard_on),
        initial_soc_min=args.initial_soc_min,
        initial_soc_max=args.initial_soc_max,
        max_steps=args.max_steps,
        progress_every_steps=args.progress_every_steps,
        output=args.output,
    )
    print(json.dumps({
        "output": str(args.output),
        "step_count": result["step_count"],
        "terminal": result["terminal"],
        "outcome_counts": result["outcome_counts"],
        "failed_energy": result["failed_energy"],
        "emergency_committed": result["emergency_committed"],
        "release_approved": result["release_approved"],
        "energy_reachability_checks": result["energy_reachability_checks"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
