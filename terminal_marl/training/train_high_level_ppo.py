"""Train a high-level PPO policy on dynamic terminal airspace management.

The entry point is server-ready but remains safe for local development:
Stable-Baselines3 is imported lazily, defaults are CPU-oriented, and tests use
only tiny smoke runs.  The policy controls high-level airspace/flow decisions,
not free heading, altitude, or velocity-vector flight commands.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable

from ..config import load_experiment_config
from ..envs.high_level_wrapper import HighLevelHRLWrapper
from ..envs.parallel_env import TerminalParallelEnv
from ..experiments import apply_demand_profile
from .training_config import (
    PPOTrainingConfig,
    load_training_config,
    merge_training_overrides,
)


def require_sb3():
    if importlib.util.find_spec("stable_baselines3") is None:
        raise RuntimeError(
            "stable-baselines3 is required for high-level PPO training/evaluation. "
            "Install stable-baselines3, or skip PPO-specific smoke tests."
        )
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    return PPO, BaseCallback, CallbackList, CheckpointCallback, DummyVecEnv, SubprocVecEnv


def _make_high_level_env(
    *,
    config_name: str,
    demand_level: str,
    low_level_policy: str,
    seed: int,
    high_level_interval: int | None = None,
    max_steps: int | None = None,
) -> HighLevelHRLWrapper:
    config = apply_demand_profile(load_experiment_config(config_name), demand_level)
    if high_level_interval is not None:
        config.high_level_interval_steps = int(high_level_interval)
    if max_steps is not None:
        config.max_steps = min(config.max_steps, int(max_steps))
    if not config.simulation_only:
        raise ValueError("high-level PPO trainer accepts simulation_only configs only")
    return HighLevelHRLWrapper(
        TerminalParallelEnv(config),
        low_level_policy=low_level_policy,
        seed=seed,
    )


def _make_env_factory(
    *,
    config_name: str,
    demand_level: str,
    low_level_policy: str,
    seed: int,
    rank: int,
    high_level_interval: int | None,
    max_steps: int | None,
) -> Callable[[], HighLevelHRLWrapper]:
    def _factory() -> HighLevelHRLWrapper:
        return _make_high_level_env(
            config_name=config_name,
            demand_level=demand_level,
            low_level_policy=low_level_policy,
            seed=seed + rank,
            high_level_interval=high_level_interval,
            max_steps=max_steps,
        )

    return _factory


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_model(
    model: Any,
    *,
    config_name: str,
    demand_level: str,
    low_level_policy: str,
    seed: int,
    high_level_interval: int | None,
    max_steps: int | None,
    episodes: int = 1,
    save_trace_path: Path | None = None,
) -> dict[str, Any]:
    rewards: list[float] = []
    lengths: list[int] = []
    trace_rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        env = _make_high_level_env(
            config_name=config_name,
            demand_level=demand_level,
            low_level_policy=low_level_policy,
            seed=seed + 10_000 + episode,
            high_level_interval=high_level_interval,
            max_steps=max_steps,
        )
        observation, _ = env.reset(seed=seed + 10_000 + episode)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated) and steps < 8:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if save_trace_path is not None:
                trace_rows.append(
                    {
                        "episode": episode,
                        "high_level_step": steps,
                        "reward": float(reward),
                        "landed_count": info["metrics"].get("landed_count", 0),
                        "active_count": info["metrics"].get("active_count", 0),
                        "safety_override_count": info["metrics"].get(
                            "safety_override_count", 0
                        ),
                    }
                )
        rewards.append(total_reward)
        lengths.append(steps)
    if save_trace_path is not None:
        _write_csv(
            save_trace_path,
            trace_rows,
            (
                "episode",
                "high_level_step",
                "reward",
                "landed_count",
                "active_count",
                "safety_override_count",
            ),
        )
    return {
        "eval_episodes": episodes,
        "eval_mean_reward": float(sum(rewards) / max(1, len(rewards))),
        "eval_rewards": rewards,
        "eval_lengths": lengths,
    }


def train_high_level_ppo(
    *,
    config_name: str,
    seed: int,
    total_timesteps: int,
    output_dir: str | Path,
    demand_level: str,
    low_level_policy: str,
    n_envs: int = 1,
    vec_env: str = "dummy",
    device: str = "cpu",
    num_threads: int = 1,
    learning_rate: float = 3e-4,
    n_steps: int = 64,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    high_level_interval: int | None = None,
    checkpoint_freq: int = 0,
    eval_freq: int = 0,
    save_replay_traces: bool = False,
    resume_from: str | Path | None = None,
    nominal_delay_mode: str = "analytic",
    max_steps: int | None = None,
) -> dict:
    PPO, BaseCallback, CallbackList, CheckpointCallback, DummyVecEnv, SubprocVecEnv = require_sb3()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.set_num_threads(num_threads)
    if vec_env not in {"dummy", "subproc"}:
        raise ValueError("vec_env must be 'dummy' or 'subproc'")
    env_factories = [
        _make_env_factory(
            config_name=config_name,
            demand_level=demand_level,
            low_level_policy=low_level_policy,
            seed=seed,
            rank=rank,
            high_level_interval=high_level_interval,
            max_steps=max_steps,
        )
        for rank in range(n_envs)
    ]
    vec_cls = DummyVecEnv if vec_env == "dummy" else SubprocVecEnv
    env = vec_cls(env_factories)
    progress: list[dict] = []
    eval_rows: list[dict] = []

    class ProgressCallback(BaseCallback):
        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards", [0.0])
            progress.append(
                {
                    "timesteps": self.num_timesteps,
                    "mean_step_reward": float(sum(rewards) / max(1, len(rewards))),
                    "n_envs": n_envs,
                }
            )
            if eval_freq and self.num_timesteps % eval_freq == 0:
                summary = _evaluate_model(
                    self.model,
                    config_name=config_name,
                    demand_level=demand_level,
                    low_level_policy=low_level_policy,
                    seed=seed,
                    high_level_interval=high_level_interval,
                    max_steps=max_steps,
                    episodes=1,
                    save_trace_path=(
                        output / "replay_traces" / f"eval_{self.num_timesteps}.csv"
                        if save_replay_traces
                        else None
                    ),
                )
                eval_rows.append({"timesteps": self.num_timesteps, **summary})
            return True

    callbacks: list[Any] = [ProgressCallback()]
    if checkpoint_freq and checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, checkpoint_freq // max(1, n_envs)),
                save_path=str(output / "checkpoints"),
                name_prefix="high_level_ppo",
            )
        )
    callback = CallbackList(callbacks)
    effective_n_steps = max(8, min(int(n_steps), max(8, int(total_timesteps))))
    effective_batch_size = max(1, min(int(batch_size), effective_n_steps * max(1, n_envs)))
    if resume_from is not None:
        model = PPO.load(
            resume_from,
            env=env,
            device=device,
            seed=seed,
            learning_rate=learning_rate,
            n_steps=effective_n_steps,
            batch_size=effective_batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            n_steps=effective_n_steps,
            batch_size=effective_batch_size,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            device=device,
            verbose=0,
        )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(output / "model")
    training_config = {
        "config": config_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "demand_level": demand_level,
        "low_level_policy": low_level_policy,
        "n_envs": n_envs,
        "vec_env": vec_env,
        "device": device,
        "num_threads": num_threads,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "effective_n_steps": effective_n_steps,
        "effective_batch_size": effective_batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "high_level_interval": high_level_interval,
        "checkpoint_freq": checkpoint_freq,
        "eval_freq": eval_freq,
        "save_replay_traces": save_replay_traces,
        "resume_from": None if resume_from is None else str(resume_from),
        "nominal_delay_mode": nominal_delay_mode,
        "simulation_only": True,
    }
    (output / "training_config.json").write_text(
        json.dumps(training_config, indent=2) + "\n", encoding="utf-8"
    )
    (output / "config_snapshot.json").write_text(
        json.dumps(training_config, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        output / "training_progress.csv",
        progress,
        ("timesteps", "mean_step_reward", "n_envs"),
    )
    summary = _evaluate_model(
        model,
        config_name=config_name,
        demand_level=demand_level,
        low_level_policy=low_level_policy,
        seed=seed + 1,
        high_level_interval=high_level_interval,
        max_steps=max_steps,
        episodes=1,
        save_trace_path=output / "replay_traces" / "final_eval.csv"
        if save_replay_traces
        else None,
    )
    summary["periodic_evaluations"] = eval_rows
    (output / "eval_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    env.close()
    return summary


def train_from_config(
    training_config: PPOTrainingConfig,
    *,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> dict:
    """Run one seed from a loaded ``PPOTrainingConfig``."""

    selected_seed = int(training_config.seed_list[0] if seed is None else seed)
    return train_high_level_ppo(
        config_name=training_config.config,
        seed=selected_seed,
        total_timesteps=training_config.total_timesteps,
        output_dir=output_dir or training_config.output_dir,
        demand_level=training_config.demand_level,
        low_level_policy=training_config.low_level_policy,
        n_envs=training_config.n_envs,
        vec_env=training_config.vec_env,
        device=training_config.device,
        num_threads=training_config.num_threads,
        learning_rate=training_config.learning_rate,
        n_steps=training_config.n_steps,
        batch_size=training_config.batch_size,
        gamma=training_config.gamma,
        gae_lambda=training_config.gae_lambda,
        clip_range=training_config.clip_range,
        ent_coef=training_config.ent_coef,
        vf_coef=training_config.vf_coef,
        high_level_interval=training_config.high_level_interval,
        checkpoint_freq=training_config.checkpoint_freq,
        eval_freq=training_config.eval_freq,
        save_replay_traces=training_config.save_replay_traces,
        resume_from=resume_from,
        nominal_delay_mode=training_config.nominal_delay_mode,
        max_steps=training_config.max_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--demand-level")
    parser.add_argument(
        "--low-level-policy",
        choices=["rule-only", "random-policy", "intention-residual-placeholder"],
    )
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"])
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--clip-range", type=float)
    parser.add_argument("--ent-coef", type=float)
    parser.add_argument("--vf-coef", type=float)
    parser.add_argument("--high-level-interval", type=int)
    parser.add_argument("--checkpoint-freq", type=int)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--save-replay-traces", action="store_true", default=None)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--nominal-delay-mode",
        choices=["analytic", "rule-only-reference"],
    )
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    base = (
        load_training_config(args.training_config)
        if args.training_config is not None
        else PPOTrainingConfig()
    )
    merged = merge_training_overrides(
        base,
        {
            "config": args.config,
            "demand_level": args.demand_level,
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "vec_env": args.vec_env,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "high_level_interval": args.high_level_interval,
            "low_level_policy": args.low_level_policy,
            "nominal_delay_mode": args.nominal_delay_mode,
            "output_dir": None if args.output_dir is None else str(args.output_dir),
            "device": args.device,
            "num_threads": args.num_threads,
            "checkpoint_freq": args.checkpoint_freq,
            "eval_freq": args.eval_freq,
            "save_replay_traces": args.save_replay_traces,
            "max_steps": args.max_steps,
        },
    )
    train_from_config(
        merged,
        seed=args.seed,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
