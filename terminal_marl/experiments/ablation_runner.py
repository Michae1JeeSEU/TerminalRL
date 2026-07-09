"""Run rule, random-policy, or dry-run terminal-airspace ablations."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import load_experiment_config, load_sim_config
from ..core.config import SimConfig
from ..core.geometry import distance_3d
from ..envs.parallel_env import TerminalParallelEnv


ABLATION_METRIC_KEYS = (
    "landed_count",
    "average_delay",
    "total_completion_time",
    "conflict_count",
    "min_pairwise_distance",
    "safety_override_count",
    "final_queue_time",
    "throughput",
)


def _resolve_config(config: SimConfig | str | Path) -> SimConfig:
    if isinstance(config, SimConfig):
        return copy.deepcopy(config)
    path = Path(config)
    if path.suffix == ".yaml" or path.is_file():
        return load_sim_config(path)
    return load_experiment_config(str(config))


def _current_minimum_distance(env: TerminalParallelEnv) -> float:
    active = [aircraft for aircraft in env.aircraft if not aircraft.finished]
    minimum = math.inf
    for i, first in enumerate(active):
        for second in active[i + 1 :]:
            minimum = min(minimum, distance_3d(first.position, second.position))
    return minimum


def run_rollout(
    config: SimConfig | str | Path,
    *,
    policy: str = "rule-only",
    seed: int = 0,
    max_steps: int | None = None,
) -> dict[str, float | str]:
    """Run one simulation-only rollout and return publication-facing metrics."""

    if policy not in {"rule-only", "random-policy", "dry-run"}:
        raise ValueError("policy must be rule-only, random-policy, or dry-run")
    sim_config = _resolve_config(config)
    if not sim_config.simulation_only:
        raise ValueError("ablation runner accepts simulation_only configurations")
    if max_steps is not None:
        sim_config.max_steps = min(sim_config.max_steps, int(max_steps))
    env = TerminalParallelEnv(sim_config)
    rng = np.random.default_rng(seed)
    observations, _ = env.reset(seed=seed)
    observed_minimum = _current_minimum_distance(env)

    while env.agents and env.step_count < sim_config.max_steps:
        action_size = env.action_spec(env.agents[0]).shape[0]
        if policy == "random-policy":
            actions = {
                agent: rng.uniform(-1.0, 1.0, size=action_size).astype(np.float32)
                for agent in env.agents
            }
            high_level_action = (
                rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
                if env.should_update_high_level()
                else None
            )
        else:
            actions = {
                agent: np.zeros(action_size, dtype=np.float32)
                for agent in env.agents
            }
            high_level_action = (
                np.zeros(8, dtype=np.float32)
                if env.should_update_high_level()
                else None
            )

        observations, _rewards, _terminations, _truncations, _infos = env.step(
            actions,
            high_level_action=high_level_action,
        )
        observed_minimum = min(observed_minimum, _current_minimum_distance(env))

    per_aircraft = env.per_aircraft_metrics()
    completion_times = [
        float(metrics["completion_time"])
        for metrics in per_aircraft.values()
        if metrics["completion_time"] is not None
    ]
    landed_count = len(completion_times)
    elapsed = float(env.time)
    metrics: dict[str, float | str] = {
        "config_name": sim_config.config_name,
        "experiment_mode": env.experiment_mode.value,
        "policy": policy,
        "landed_count": float(landed_count),
        "average_delay": float(np.mean(completion_times)) if completion_times else elapsed,
        "total_completion_time": max(completion_times) if completion_times else elapsed,
        "conflict_count": float(env.nmac_count + env.collision_count),
        "min_pairwise_distance": (
            0.0 if math.isinf(observed_minimum) else float(observed_minimum)
        ),
        "safety_override_count": float(env.safety_override_count),
        "final_queue_time": float(env.final_queue_time),
        "throughput": float(landed_count / elapsed) if elapsed > 0.0 else 0.0,
    }
    return metrics


def run_ablation(
    configs: Iterable[str] = (
        "fixed_rule",
        "fixed_residual",
        "dynamic_rule",
        "dynamic_hrl",
    ),
    *,
    policy: str = "rule-only",
    seed: int = 0,
    max_steps: int | None = None,
) -> dict[str, dict[str, float | str]]:
    """Run a common policy over multiple YAML experiment presets."""

    return {
        name: run_rollout(
            name,
            policy=policy,
            seed=seed,
            max_steps=max_steps,
        )
        for name in configs
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["fixed_rule", "fixed_residual", "dynamic_rule", "dynamic_hrl"],
    )
    parser.add_argument(
        "--policy",
        choices=["rule-only", "random-policy", "dry-run"],
        default="rule-only",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = run_ablation(
        args.configs,
        policy=args.policy,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
