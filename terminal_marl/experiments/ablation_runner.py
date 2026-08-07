"""Run rule, random-policy, or dry-run terminal-airspace ablations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import load_experiment_config, load_sim_config
from ..core.config import SimConfig
from ..core.geometry import distance_3d
from ..envs.parallel_env import TerminalParallelEnv
from ..metrics.nominal_schedule import NominalScheduleEstimator
from ..policies import DynamicRuleHighLevelPolicy
from .logger import ExperimentLogger
from .aggregate import aggregate_statistics


ABLATION_METRIC_KEYS = (
    "landed_count",
    "average_delay",
    "average_completion_time",
    "delay_p50",
    "delay_p90",
    "max_delay",
    "total_completion_time",
    "conflict_count",
    "raw_pair_conflict_samples",
    "pair_opportunities",
    "normalized_conflict_rate",
    "min_pairwise_distance",
    "safety_override_count",
    "total_waiting_time",
    "normalized_waiting",
    "normalized_congestion",
    "final_queue_time",
    "throughput",
    "safety_enabled",
)

POLICIES = (
    "rule-only",
    "random-policy",
    "dry-run",
    "high-level-random",
    "high-level-static",
    "dynamic-rule",
    "event-rule",
    "high-level-ppo",
)

DEMAND_PROFILES = {
    "low": {"initial_aircraft_count": 4, "formation_rows": 2, "formation_cols": 2, "arrival_rate": 0.0},
    # Recalibrated after server capacity checks: interval=8 high-level steps
    # (10 low-level steps each) gives 0.00625 arrivals/s and avoids treating
    # the old overload flow as "medium" during PPO pilot training.
    "medium": {"initial_aircraft_count": 6, "formation_rows": 3, "formation_cols": 2, "arrival_rate": 0.00625},
    "high": {"initial_aircraft_count": 8, "formation_rows": 4, "formation_cols": 2, "arrival_rate": 0.10},
    "bursty": {"initial_aircraft_count": 10, "formation_rows": 5, "formation_cols": 2, "arrival_rate": 0.25},
}


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


def apply_demand_profile(config: SimConfig, level: str) -> SimConfig:
    """Apply an arrival-flow profile to a copied experiment configuration."""

    if level not in DEMAND_PROFILES:
        raise ValueError(f"unknown demand level: {level}")
    updated = copy.deepcopy(config)
    for key, value in DEMAND_PROFILES[level].items():
        setattr(updated, key, value)
    updated.traffic_demand_level = level
    return updated


def run_rollout(
    config: SimConfig | str | Path,
    *,
    policy: str = "rule-only",
    seed: int = 0,
    max_steps: int | None = None,
    traffic_demand_level: str = "medium",
    output_dir: str | Path | None = None,
    episode: int = 0,
    model_path: str | Path | None = None,
    nominal_delay_mode: str = "analytic",
) -> dict[str, float | str]:
    """Run one simulation-only rollout and return publication-facing metrics."""

    if policy not in POLICIES:
        raise ValueError(f"policy must be one of: {', '.join(POLICIES)}")
    sim_config = apply_demand_profile(
        _resolve_config(config),
        traffic_demand_level,
    )
    if not sim_config.simulation_only:
        raise ValueError("ablation runner accepts simulation_only configurations")
    if max_steps is not None:
        sim_config.max_steps = min(sim_config.max_steps, int(max_steps))
    env = TerminalParallelEnv(sim_config)
    ppo_model = None
    if policy == "high-level-ppo":
        if model_path is None:
            raise ValueError("high-level-ppo requires model_path")
        from ..training.train_high_level_ppo import require_sb3

        PPO = require_sb3()[0]
        ppo_model = PPO.load(model_path)
    dynamic_rule_policy = DynamicRuleHighLevelPolicy()
    rng = np.random.default_rng(seed)
    observations, _ = env.reset(seed=seed)
    if nominal_delay_mode == "rule-only-reference":
        reference = {
            aircraft.uid: float(aircraft.planned_nominal_landing_time)
            for aircraft in env.aircraft
            if aircraft.planned_nominal_landing_time is not None
        }
        estimator = NominalScheduleEstimator(
            sim_config,
            mode="rule-only-reference",
            rule_only_reference=reference,
        )
        for aircraft in env.aircraft:
            aircraft.planned_nominal_landing_time = estimator.estimate(
                aircraft.position,
                spawn_time=aircraft.spawn_time,
                sequence_index=aircraft.uid,
            )
    elif nominal_delay_mode != "analytic":
        raise ValueError("unknown nominal_delay_mode")
    observed_minimum = _current_minimum_distance(env)
    logger = (
        ExperimentLogger(
            output_dir,
            config_name=sim_config.config_name,
            experiment_mode=env.experiment_mode.value,
            policy=policy,
            seed=seed,
            traffic_demand_level=traffic_demand_level,
        )
        if output_dir is not None
        else None
    )
    if logger is not None:
        logger.record_step(env)

    while env.agents and env.step_count < sim_config.max_steps:
        action_size = env.action_spec(env.agents[0]).shape[0]
        if policy == "random-policy":
            actions = {
                agent: rng.uniform(-1.0, 1.0, size=action_size).astype(np.float32)
                for agent in env.agents
            }
        else:
            actions = {
                agent: np.zeros(action_size, dtype=np.float32)
                for agent in env.agents
            }
        if env.should_update_high_level():
            if policy == "high-level-random":
                high_level_action = rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
            elif policy in {"dynamic-rule", "event-rule"}:
                high_level_action = dynamic_rule_policy.act(env)
            elif policy == "high-level-ppo":
                high_level_action, _ = ppo_model.predict(
                    env.get_high_level_observation(), deterministic=True
                )
            else:
                high_level_action = np.zeros(8, dtype=np.float32)
        else:
            high_level_action = None

        observations, _rewards, _terminations, _truncations, _infos = env.step(
            actions,
            high_level_action=high_level_action,
        )
        observed_minimum = min(observed_minimum, _current_minimum_distance(env))
        if logger is not None:
            logger.record_step(env)

    per_aircraft = env.per_aircraft_metrics()
    completion_times = [
        float(metrics["completion_time"])
        for metrics in per_aircraft.values()
        if metrics["completion_time"] is not None
    ]
    delays = [
        float(metrics["landing_delay"])
        for metrics in per_aircraft.values()
        if metrics["landing_delay"] is not None
    ]
    landed_count = len(completion_times)
    elapsed = float(env.time)
    env_metrics = env.metrics()
    metrics: dict[str, float | str] = {
        "config_name": sim_config.config_name,
        "experiment_mode": env.experiment_mode.value,
        "policy": policy,
        "seed": float(seed),
        "episode": float(episode),
        "traffic_demand_level": traffic_demand_level,
        "nominal_delay_mode": nominal_delay_mode,
        "landed_count": float(landed_count),
        "average_delay": float(np.mean(delays)) if delays else 0.0,
        "average_completion_time": (
            float(np.mean(completion_times)) if completion_times else elapsed
        ),
        "delay_p50": float(np.percentile(delays, 50)) if delays else 0.0,
        "delay_p90": float(np.percentile(delays, 90)) if delays else 0.0,
        "max_delay": max(delays) if delays else 0.0,
        "total_completion_time": max(completion_times) if completion_times else elapsed,
        "conflict_count": float(env.nmac_count + env.collision_count),
        "raw_pair_conflict_samples": env_metrics["raw_pair_conflict_samples"],
        "pair_opportunities": env_metrics["pair_opportunities"],
        "normalized_conflict_rate": env_metrics["normalized_conflict_rate"],
        "min_pairwise_distance": float(observed_minimum),
        "safety_override_count": float(env.safety_override_count),
        "total_waiting_time": env_metrics["total_waiting_time"],
        "normalized_waiting": env_metrics["normalized_waiting"],
        "normalized_congestion": env_metrics["normalized_congestion"],
        "final_queue_time": float(env.final_queue_time),
        "throughput": float(landed_count / elapsed) if elapsed > 0.0 else 0.0,
        "safety_enabled": float(env.safety_enabled),
    }
    if logger is not None:
        logger.finalize(metrics)
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
    traffic_demand_level: str = "medium",
    model_path: str | Path | None = None,
    nominal_delay_mode: str = "analytic",
) -> dict[str, dict[str, float | str]]:
    """Run a common policy over multiple YAML experiment presets."""

    return {
        name: run_rollout(
            name,
            policy=policy,
            seed=seed,
            max_steps=max_steps,
            traffic_demand_level=traffic_demand_level,
            model_path=model_path,
            nominal_delay_mode=nominal_delay_mode,
        )
        for name in configs
    }


def run_sweep(
    configs: Iterable[str],
    *,
    policies: Iterable[str],
    seeds: Iterable[int],
    demand_levels: Iterable[str],
    episodes: int,
    output_dir: str | Path,
    max_steps: int | None = None,
    model_path: str | Path | None = None,
    nominal_delay_mode: str = "analytic",
) -> list[dict[str, float | str]]:
    """Run every config/policy/seed/demand/episode combination with logs."""

    root = Path(output_dir)
    results: list[dict[str, float | str]] = []
    for config_name in configs:
        for policy in policies:
            for demand in demand_levels:
                for seed in seeds:
                    for episode in range(episodes):
                        run_dir = (
                            root
                            / config_name
                            / policy
                            / demand
                            / f"seed_{seed}"
                            / f"episode_{episode:03d}"
                        )
                        results.append(
                            run_rollout(
                                config_name,
                                policy=policy,
                                seed=int(seed),
                                max_steps=max_steps,
                                traffic_demand_level=demand,
                                output_dir=run_dir,
                                episode=episode,
                                model_path=model_path,
                                nominal_delay_mode=nominal_delay_mode,
                            )
                        )
    aggregate_path = root / "aggregate_results.csv"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in results for key in row})
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    aggregate_statistics(aggregate_path, root)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["fixed_rule", "fixed_residual", "dynamic_rule", "dynamic_hrl"],
    )
    parser.add_argument(
        "--policy",
        nargs="+",
        choices=POLICIES,
        default=["rule-only"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--demand-levels",
        nargs="+",
        choices=tuple(DEMAND_PROFILES),
        default=["medium"],
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ablation"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--nominal-delay-mode",
        choices=["analytic", "rule-only-reference"],
        default="analytic",
    )
    args = parser.parse_args()

    results = run_sweep(
        args.configs,
        policies=args.policy,
        seeds=args.seeds,
        demand_levels=args.demand_levels,
        episodes=args.episodes,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        model_path=args.model_path,
        nominal_delay_mode=args.nominal_delay_mode,
    )
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    print(serialized)


if __name__ == "__main__":
    main()
