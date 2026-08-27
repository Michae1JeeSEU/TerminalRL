"""Non-training D7 dynamics qualification and performance benchmark runner."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ..core.config import SimConfig
from ..envs.parallel_env import TerminalParallelEnv


@dataclass(frozen=True)
class BenchmarkResult:
    warm_start_uavs: int
    physics_dt: float
    simulated_seconds: float
    wall_seconds: float
    environment_steps_per_second: float
    active_uavs: int
    hard_separation_violations: int
    nmac: int
    collisions: int


def benchmark(*, warm_start_uavs: int, physics_dt: float, simulated_seconds: float, seed: int = 0) -> BenchmarkResult:
    """Measure holder-execution cost only; this never calls PPO or updates a policy."""
    schedule = (0.0,) * warm_start_uavs
    config = SimConfig(
        experiment_mode="fixed_rule", reset_mode="warm_start_rings", warm_start_uavs=warm_start_uavs,
        fixed_arrival_schedule=schedule, total_requested_uavs=warm_start_uavs,
        physics_dt=physics_dt, safety_sampling_dt=physics_dt,
        max_simulation_seconds=simulated_seconds,
    )
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(seed)
    hold_actions = {uid: np.asarray((0.0, 0.0, 0.0), dtype=np.float32) for uid in observations}
    step_count = int(simulated_seconds / config.dt)
    started = perf_counter()
    for _ in range(step_count):
        env.step(hold_actions)
    elapsed = perf_counter() - started
    return BenchmarkResult(
        warm_start_uavs=warm_start_uavs, physics_dt=physics_dt, simulated_seconds=simulated_seconds,
        wall_seconds=elapsed, environment_steps_per_second=step_count / max(elapsed, 1e-9),
        active_uavs=len(env.active), hard_separation_violations=int(env.telemetry["hard_separation_violations"]),
        nmac=int(env.telemetry["nmac"]), collisions=int(env.telemetry["collisions"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D7 dynamics performance benchmarks without PPO training.")
    parser.add_argument("--uavs", type=int, nargs="+", default=(10, 30, 60))
    parser.add_argument("--physics-dts", type=float, nargs="+", default=(.2, .1))
    parser.add_argument("--simulated-seconds", type=float, default=360.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [benchmark(warm_start_uavs=count, physics_dt=dt, simulated_seconds=args.simulated_seconds, seed=args.seed)
               for count in args.uavs for dt in args.physics_dts]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
