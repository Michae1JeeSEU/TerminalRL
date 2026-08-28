"""Repeatable P0 simulation-performance benchmark.

This is deliberately a qualification utility, not a training entry point.  It
creates deterministic traffic witnesses and writes all profiles/reports to an
explicit output directory supplied by the caller.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
import json
from pathlib import Path
import pstats
import time

import numpy as np

from terminal_marl.core.config import SimConfig
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _config(name: str, count: int) -> SimConfig:
    schedule = tuple(0.0 for _ in range(count))
    base = SimConfig(
        experiment_mode="fixed_teacher_heavy" if name == "full_teacher" else "fixed_rule",
        reset_mode="warm_start_rings", warm_start_uavs=count,
        fixed_arrival_schedule=schedule, total_requested_uavs=count,
        max_active_uavs=max(60, count), max_simulation_seconds=240.0,
        teacher_enabled=name == "full_teacher",
        scheduler_mode="teacher_full" if name == "full_teacher" else "execution",
    )
    if name == "inbound_heavy":
        return replace(base, reset_mode="flow_from_empty", warm_start_uavs=0,
                       arrival_model="finite_cohort", fixed_arrival_schedule=tuple(float(i % 4) * 20.0 for i in range(count)))
    return base


def _actions(env: TerminalParallelEnv, name: str) -> dict[str, np.ndarray]:
    # Holder/observation witnesses remain in their rings.  Teacher and mixed
    # witnesses issue normal requests, exercising candidate generation.
    release = 0.0 if name in {"holder_only", "observation_v2"} else 1.0
    return {str(uid): np.asarray((release, 0.0, 0.0), dtype=np.float32) for uid in env.active}


def run_case(name: str, count: int, steps: int, output: Path) -> dict:
    config = _config(name, count)
    env = TerminalParallelEnv(config)
    env.reset(seed=17)
    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    for _ in range(steps):
        env.step(_actions(env, name))
    profiler.disable()
    wall = time.perf_counter() - wall_start
    profile_path = output / f"{name}_n{count}.prof"
    profiler.dump_stats(profile_path)
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    rows = []
    for (file, line, function), values in list(stats.stats.items()):
        primitive, total, tottime, cumulative, callers = values
        rows.append({"function": f"{Path(file).name}:{line}({function})", "calls": total,
                     "self_s": round(tottime, 6), "cumulative_s": round(cumulative, 6)})
    rows.sort(key=lambda row: row["cumulative_s"], reverse=True)
    return {"scenario": name, "n": count, "steps": steps, "wall_seconds": wall,
            "simulated_seconds": steps * config.dt, "steps_per_second": steps / max(wall, 1e-9),
            "wall_per_simulated_second": wall / max(steps * config.dt, 1e-9),
            "profile": str(profile_path), "top_cumulative": rows[:20], "telemetry": dict(env.telemetry)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("holder_only", n) for n in (10, 30, 60)]
    cases += [("inbound_heavy", 30), ("mixed", 30), ("full_teacher", 30)]
    cases += [("observation_v2", n) for n in (10, 30, 60)]
    report = {"label": args.label, "cases": [run_case(name, count, args.steps, args.output) for name, count in cases]}
    (args.output / f"performance_{args.label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
