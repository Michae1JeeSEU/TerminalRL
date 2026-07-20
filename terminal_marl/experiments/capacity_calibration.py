"""Rule-only traffic capacity calibration for dry-run server validation.

This module deliberately performs simulation-only rollouts.  It does not train
PPO and does not connect to ROS2, Crazyswarm2, or Crazyflie hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import load_experiment_config
from ..envs.parallel_env import TerminalParallelEnv


DEFAULT_INTERVAL_STEPS = (1, 2, 3, 4, 5, 6, 8, 10)


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _calibrated_config(
    config_name: str,
    *,
    arrival_interval_high_level_steps: int,
    initial_uavs: int,
    high_level_steps: int,
):
    config = load_experiment_config(config_name)
    config.initial_aircraft_count = int(initial_uavs)
    config.formation_rows = max(1, int(initial_uavs))
    config.formation_cols = 1
    config.max_active_uavs = max(config.max_active_uavs, int(initial_uavs))
    config.max_steps = int(high_level_steps) * int(config.high_level_interval_steps)
    seconds_between_arrivals = (
        float(arrival_interval_high_level_steps)
        * float(config.high_level_interval_steps)
        * float(config.dt)
    )
    config.arrival_model = "deterministic_interval"
    config.arrival_rate = 0.0 if seconds_between_arrivals <= 0.0 else 1.0 / seconds_between_arrivals
    # ArrivalProcess validates this label against the canonical demand names.
    # The swept interval itself is reported as an explicit metric instead of
    # being encoded as a non-standard demand level.
    config.traffic_demand_level = "medium"
    return config


def run_capacity_case(
    *,
    config_name: str = "fixed_rule",
    arrival_interval_high_level_steps: int,
    seed: int = 0,
    initial_uavs: int = 6,
    high_level_steps: int = 30,
) -> dict[str, float | str]:
    """Run one rule-only calibration case and return capacity metrics."""

    config = _calibrated_config(
        config_name,
        arrival_interval_high_level_steps=arrival_interval_high_level_steps,
        initial_uavs=initial_uavs,
        high_level_steps=high_level_steps,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=seed)
    active_trace: list[float] = []
    approach_trace: list[float] = []
    final_trace: list[float] = []
    while env.agents and env.step_count < config.max_steps:
        actions = {
            agent: np.zeros(env.action_spec(agent).shape[0], dtype=np.float32)
            for agent in env.agents
        }
        env.step(actions)
        metrics = env.metrics()
        active_trace.append(float(metrics["active_count"]))
        approach_trace.append(float(metrics["approach_waiting_time"]))
        final_trace.append(float(metrics.get("queue_0", 0)))

    metrics = env.metrics()
    elapsed = max(float(env.time), float(config.dt))
    active_slope = (
        (active_trace[-1] - active_trace[0]) / elapsed if len(active_trace) >= 2 else 0.0
    )
    spawned = float(metrics["total_spawned_count"])
    landed = float(metrics["landed_count"])
    landed_ratio = landed / max(1.0, spawned)
    stable = (
        landed_ratio >= 0.80
        and active_slope <= 0.02
        and float(metrics["active_count"]) < float(config.max_active_uavs)
    )
    return {
        "config_name": config_name,
        "seed": float(seed),
        "arrival_interval_high_level_steps": float(arrival_interval_high_level_steps),
        "arrival_rate": float(config.arrival_rate),
        "initial_uavs": float(initial_uavs),
        "high_level_steps": float(high_level_steps),
        "elapsed_time": elapsed,
        "spawned_count": spawned,
        "landed_count": landed,
        "landed_ratio": landed_ratio,
        "throughput": landed / elapsed,
        "active_end": float(metrics["active_count"]),
        "active_mean": float(np.mean(active_trace)) if active_trace else 0.0,
        "active_max": float(np.max(active_trace)) if active_trace else 0.0,
        "active_slope": float(active_slope),
        "final_queue_time": float(metrics["final_queue_time"]),
        "final_queue_mean": float(np.mean(final_trace)) if final_trace else 0.0,
        "approach_waiting_time": float(metrics["approach_waiting_time"]),
        "normalized_waiting": float(metrics["normalized_waiting"]),
        "normalized_congestion": float(metrics["normalized_congestion"]),
        "raw_pair_conflict_samples": float(metrics["raw_pair_conflict_samples"]),
        "pair_opportunities": float(metrics["pair_opportunities"]),
        "normalized_conflict_rate": float(metrics["normalized_conflict_rate"]),
        "safety_override_count": float(metrics["safety_override_count"]),
        "capacity_stable": float(stable),
    }


def _recommend_demand_levels(rows: list[dict[str, float | str]]) -> dict[str, float | None]:
    by_interval: dict[float, list[dict[str, float | str]]] = {}
    for row in rows:
        by_interval.setdefault(float(row["arrival_interval_high_level_steps"]), []).append(row)
    aggregates: list[tuple[float, float, float]] = []
    for interval, interval_rows in sorted(by_interval.items()):
        stable_rate = float(np.mean([float(row["capacity_stable"]) for row in interval_rows]))
        landed_ratio = float(np.mean([float(row["landed_ratio"]) for row in interval_rows]))
        aggregates.append((interval, stable_rate, landed_ratio))

    stable_intervals = [interval for interval, stable_rate, _ in aggregates if stable_rate >= 0.5]
    unstable_intervals = [interval for interval, stable_rate, _ in aggregates if stable_rate < 0.5]
    lowest_stable = min(stable_intervals) if stable_intervals else None
    return {
        "low_interval_high_level_steps": max(stable_intervals) if stable_intervals else None,
        "medium_interval_high_level_steps": lowest_stable,
        "high_interval_high_level_steps": (
            max(1.0, lowest_stable - 1.0) if lowest_stable is not None else None
        ),
        "overload_interval_high_level_steps": min(unstable_intervals) if unstable_intervals else None,
    }


def run_capacity_calibration(
    *,
    config_name: str = "fixed_rule",
    interval_steps: Iterable[int] = DEFAULT_INTERVAL_STEPS,
    seeds: Iterable[int] = (0,),
    initial_uavs: int = 6,
    high_level_steps: int = 30,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Run a rule-only arrival interval sweep and optionally persist outputs."""

    rows = [
        run_capacity_case(
            config_name=config_name,
            arrival_interval_high_level_steps=int(interval),
            seed=int(seed),
            initial_uavs=initial_uavs,
            high_level_steps=high_level_steps,
        )
        for interval in interval_steps
        for seed in seeds
    ]
    recommendations = _recommend_demand_levels(rows)
    result: dict[str, object] = {
        "config_name": config_name,
        "initial_uavs": initial_uavs,
        "high_level_steps": high_level_steps,
        "rows": rows,
        "recommended_demand_intervals": recommendations,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in rows for key in row})
        with (output / "capacity_calibration.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (output / "capacity_calibration.json").write_text(
            json.dumps(_jsonable(result), indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="fixed_rule")
    parser.add_argument(
        "--interval-steps",
        nargs="+",
        type=int,
        default=list(DEFAULT_INTERVAL_STEPS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--initial-uavs", type=int, default=6)
    parser.add_argument("--high-level-steps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/capacity_calibration"))
    args = parser.parse_args()
    result = run_capacity_calibration(
        config_name=args.config,
        interval_steps=args.interval_steps,
        seeds=args.seeds,
        initial_uavs=args.initial_uavs,
        high_level_steps=args.high_level_steps,
        output_dir=args.output_dir,
    )
    print(json.dumps(_jsonable(result), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
