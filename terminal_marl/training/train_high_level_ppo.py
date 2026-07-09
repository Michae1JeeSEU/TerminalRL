"""Train a high-level PPO policy on dynamic terminal airspace management."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

from ..config import load_experiment_config
from ..envs.high_level_wrapper import HighLevelHRLWrapper
from ..envs.parallel_env import TerminalParallelEnv
from ..experiments import apply_demand_profile


def require_sb3():
    if importlib.util.find_spec("stable_baselines3") is None:
        raise RuntimeError(
            "stable-baselines3 is required for high-level PPO; install it to train"
        )
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    return PPO, BaseCallback


def train_high_level_ppo(
    *,
    config_name: str,
    seed: int,
    total_timesteps: int,
    output_dir: str | Path,
    demand_level: str,
    low_level_policy: str,
) -> dict:
    PPO, BaseCallback = require_sb3()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = apply_demand_profile(load_experiment_config(config_name), demand_level)
    env = HighLevelHRLWrapper(
        TerminalParallelEnv(config),
        low_level_policy=low_level_policy,
        seed=seed,
    )
    progress: list[dict] = []

    class ProgressCallback(BaseCallback):
        def _on_step(self) -> bool:
            progress.append(
                {
                    "timesteps": self.num_timesteps,
                    "reward": float(self.locals["rewards"][0]),
                }
            )
            return True

    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        n_steps=max(8, min(64, total_timesteps)),
        batch_size=8,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps, callback=ProgressCallback())
    model.save(output / "model")
    training_config = {
        "config": config_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "demand_level": demand_level,
        "low_level_policy": low_level_policy,
    }
    (output / "training_config.json").write_text(
        json.dumps(training_config, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "training_progress.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timesteps", "reward"))
        writer.writeheader()
        writer.writerows(progress)
    observation, _ = env.reset(seed=seed + 1)
    total_reward = 0.0
    for _ in range(2):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    summary = {"eval_steps": 2, "eval_total_reward": total_reward}
    (output / "eval_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dynamic_hrl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--demand-level", default="medium")
    parser.add_argument(
        "--low-level-policy",
        choices=["rule-only", "random-policy", "intention-residual-placeholder"],
        default="rule-only",
    )
    args = parser.parse_args()
    train_high_level_ppo(
        config_name=args.config,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=args.output_dir,
        demand_level=args.demand_level,
        low_level_policy=args.low_level_policy,
    )


if __name__ == "__main__":
    main()
