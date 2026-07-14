"""Dry-run high-level reward component diagnostics before server training."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import load_experiment_config
from ..envs.high_level_wrapper import HighLevelHRLWrapper
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import DynamicRuleHighLevelPolicy
from .ablation_runner import apply_demand_profile


DIAGNOSTIC_POLICIES = (
    "rule-only",
    "dynamic-rule",
    "high-level-random",
    "untrained-ppo",
)


def _component_rows(
    values: dict[str, list[float]],
    *,
    config_name: str,
    policy: str,
    demand_level: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for component, component_values in sorted(values.items()):
        array = np.asarray(component_values, dtype=float)
        rows.append(
            {
                "config_name": config_name,
                "policy": policy,
                "traffic_demand_level": demand_level,
                "component": component,
                "mean": float(np.mean(array)) if array.size else 0.0,
                "variance": float(np.var(array)) if array.size else 0.0,
                "mean_abs_magnitude": float(np.mean(np.abs(array))) if array.size else 0.0,
                "max_abs_magnitude": float(np.max(np.abs(array))) if array.size else 0.0,
                "n": int(array.size),
            }
        )
    return rows


def run_reward_diagnostics(
    *,
    config_name: str = "dynamic_hrl",
    policies: Iterable[str] = DIAGNOSTIC_POLICIES,
    demand_levels: Iterable[str] = ("medium",),
    seeds: Iterable[int] = (0,),
    episodes: int = 1,
    high_level_steps: int = 3,
    output_dir: str | Path | None = None,
) -> list[dict[str, float | str]]:
    """Collect reward component means/variances from short dry-runs."""

    rng = np.random.default_rng(1234)
    dynamic_rule = DynamicRuleHighLevelPolicy()
    rows: list[dict[str, float | str]] = []
    ppo_available = importlib.util.find_spec("stable_baselines3") is not None
    for policy in policies:
        if policy not in DIAGNOSTIC_POLICIES:
            raise ValueError(f"unsupported diagnostic policy: {policy}")
        for demand in demand_levels:
            component_values: dict[str, list[float]] = {}
            for seed in seeds:
                for episode in range(episodes):
                    config = apply_demand_profile(
                        load_experiment_config(config_name),
                        demand,
                    )
                    env = HighLevelHRLWrapper(
                        TerminalParallelEnv(config),
                        low_level_policy="rule-only",
                        seed=int(seed) + episode,
                    )
                    observation, _ = env.reset(seed=int(seed) + episode)
                    untrained_model = None
                    if policy == "untrained-ppo" and ppo_available:
                        from stable_baselines3 import PPO

                        untrained_model = PPO(
                            "MlpPolicy",
                            env,
                            seed=int(seed),
                            n_steps=8,
                            batch_size=8,
                            verbose=0,
                            device="cpu",
                        )
                    for _ in range(high_level_steps):
                        if policy == "high-level-random":
                            action = rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
                        elif policy == "dynamic-rule":
                            action = dynamic_rule.act(env.env)
                        elif policy == "untrained-ppo" and untrained_model is not None:
                            action, _ = untrained_model.predict(
                                observation, deterministic=False
                            )
                        else:
                            action = np.zeros(8, dtype=np.float32)
                        observation, _reward, terminated, truncated, info = env.step(action)
                        diagnostics = info.get("reward_diagnostics", {})
                        for key, value in diagnostics.items():
                            if key.startswith("component_") or key == "total_reward":
                                component_values.setdefault(key, []).append(float(value))
                        if terminated or truncated:
                            break
            rows.extend(
                _component_rows(
                    component_values,
                    config_name=config_name,
                    policy=policy,
                    demand_level=demand,
                )
            )
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        fields = [
            "config_name",
            "policy",
            "traffic_demand_level",
            "component",
            "mean",
            "variance",
            "mean_abs_magnitude",
            "max_abs_magnitude",
            "n",
        ]
        with (output / "reward_diagnostics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (output / "reward_diagnostics.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dynamic_hrl")
    parser.add_argument("--policies", nargs="+", default=list(DIAGNOSTIC_POLICIES))
    parser.add_argument("--demand-level")
    parser.add_argument("--demand-levels", nargs="+", default=["medium"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--high-level-steps", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reward_diagnostics"))
    args = parser.parse_args()
    demand_levels = [args.demand_level] if args.demand_level else args.demand_levels
    rows = run_reward_diagnostics(
        config_name=args.config,
        policies=args.policies,
        demand_levels=demand_levels,
        seeds=args.seeds,
        episodes=args.episodes,
        high_level_steps=args.high_level_steps,
        output_dir=args.output_dir,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
