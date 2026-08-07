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
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..config import load_experiment_config, load_sim_config
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
    base_config = _load_base_sim_config(config_name)
    config = apply_demand_profile(base_config, demand_level)
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


def _load_base_sim_config(config_name: str):
    config_path = Path(config_name)
    return (
        load_sim_config(config_path)
        if config_path.suffix == ".yaml" or config_path.is_file()
        else load_experiment_config(config_name)
    )


class ScenarioMixtureHighLevelWrapper(HighLevelHRLWrapper):
    """High-level wrapper that samples a demand scenario at every reset.

    The PPO observation/action spaces remain identical across demand levels; only
    the arrival-flow parameters of the internal simulator are rebuilt.  This
    gives the new event-gated MDP a training distribution of medium/high/bursty
    traffic without turning the high-level problem into a different API.
    """

    def __init__(
        self,
        *,
        config_name: str,
        demand_mixture: Mapping[str, float],
        low_level_policy: str,
        seed: int,
        high_level_interval: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.config_name = config_name
        self.demand_mixture = _normalise_demand_mixture(demand_mixture)
        self._mixture_names = tuple(self.demand_mixture)
        self._mixture_probabilities = np.asarray(
            [self.demand_mixture[name] for name in self._mixture_names],
            dtype=np.float64,
        )
        self.high_level_interval = high_level_interval
        self.max_steps = max_steps
        self.current_demand_level = self._mixture_names[0]
        self.scenario_counts = {name: 0 for name in self._mixture_names}
        super().__init__(
            self._build_env(self.current_demand_level),
            low_level_policy=low_level_policy,
            seed=seed,
        )

    def _sample_demand(self) -> str:
        index = int(
            self.rng.choice(
                len(self._mixture_names),
                p=self._mixture_probabilities,
            )
        )
        return self._mixture_names[index]

    def _build_env(self, demand_level: str) -> TerminalParallelEnv:
        base_config = _load_base_sim_config(self.config_name)
        config = apply_demand_profile(base_config, demand_level)
        if self.high_level_interval is not None:
            config.high_level_interval_steps = int(self.high_level_interval)
        if self.max_steps is not None:
            config.max_steps = min(config.max_steps, int(self.max_steps))
        if not config.simulation_only:
            raise ValueError("scenario-mixture PPO accepts simulation_only configs only")
        return TerminalParallelEnv(config)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_demand_level = self._sample_demand()
        self.scenario_counts[self.current_demand_level] += 1
        self.env = self._build_env(self.current_demand_level)
        observation, info = super().reset(seed=seed, options=options)
        info["demand_level"] = self.current_demand_level
        info["demand_mixture"] = dict(self.demand_mixture)
        return observation, info


def _normalise_demand_mixture(mixture: Mapping[str, float]) -> dict[str, float]:
    allowed = {"low", "medium", "high", "bursty"}
    clean: dict[str, float] = {}
    for name, raw_weight in mixture.items():
        if name not in allowed:
            raise ValueError(f"unknown demand mixture level: {name}")
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError("demand mixture weights must be non-negative")
        if weight > 0.0:
            clean[name] = weight
    if not clean:
        raise ValueError("demand mixture must include at least one positive weight")
    total = sum(clean.values())
    return {name: weight / total for name, weight in clean.items()}


def _make_env_factory(
    *,
    config_name: str,
    demand_level: str,
    demand_mixture: Mapping[str, float] | None = None,
    low_level_policy: str,
    seed: int,
    rank: int,
    high_level_interval: int | None,
    max_steps: int | None,
) -> Callable[[], HighLevelHRLWrapper]:
    def _factory() -> HighLevelHRLWrapper:
        if demand_mixture is not None:
            return ScenarioMixtureHighLevelWrapper(
                config_name=config_name,
                demand_mixture=demand_mixture,
                low_level_policy=low_level_policy,
                seed=seed + rank,
                high_level_interval=high_level_interval,
                max_steps=max_steps,
            )
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


def _select_exact_rollout_n_steps(
    *,
    requested_n_steps: int,
    total_timesteps: int,
    n_envs: int,
) -> int:
    """Choose the largest n_steps not exceeding the request that avoids SB3 overshoot."""

    if total_timesteps % n_envs != 0:
        raise ValueError(
            "total_timesteps must be divisible by n_envs to avoid PPO rollout "
            "overshooting the requested budget"
        )
    per_env_budget = total_timesteps // n_envs
    upper = max(1, min(int(requested_n_steps), int(per_env_budget)))
    for candidate in range(upper, 0, -1):
        if per_env_budget % candidate == 0:
            if candidate * n_envs <= 1:
                raise ValueError("PPO rollout size must be greater than one")
            return candidate
    raise ValueError("could not find an exact PPO rollout length")


def _count_fields_from_info(info: dict[str, Any]) -> dict[str, float]:
    metrics = info.get("metrics", {})
    diagnostics = info.get("reward_diagnostics", {})
    return {
        "active_count_start": float(diagnostics.get("active_count_start", 0.0)),
        "active_count_mean": float(diagnostics.get("active_count_mean", 0.0)),
        "active_count_max": float(diagnostics.get("active_count_max", 0.0)),
        "active_count_end": float(diagnostics.get("active_count_end", metrics.get("active_count", 0.0))),
        "active_count": float(diagnostics.get("active_count", metrics.get("active_count", 0.0))),
        "spawned_count_delta": float(diagnostics.get("spawned_count_delta", 0.0)),
        "landed_count_delta": float(diagnostics.get("landed_count_delta", 0.0)),
        "total_spawned_count": float(
            diagnostics.get("total_spawned_count", metrics.get("total_spawned_count", 0.0))
        ),
        "total_landed_count": float(
            diagnostics.get("total_landed_count", metrics.get("total_landed_count", 0.0))
        ),
        "safety_override_delta": float(diagnostics.get("safety_override_delta", 0.0)),
        "total_safety_override_count": float(
            diagnostics.get(
                "total_safety_override_count",
                metrics.get("total_safety_override_count", metrics.get("safety_override_count", 0.0)),
            )
        ),
    }


def run_pretraining_traffic_check(
    *,
    config_name: str,
    demand_level: str,
    demand_mixture: Mapping[str, float] | None = None,
    low_level_policy: str,
    seed: int,
    high_level_interval: int | None,
    max_steps: int | None,
    traffic_check_steps: int,
) -> dict[str, Any]:
    """Run a short high-level rollout before PPO to verify traffic is present."""

    env = (
        ScenarioMixtureHighLevelWrapper(
            config_name=config_name,
            demand_mixture=demand_mixture,
            low_level_policy=low_level_policy,
            seed=seed,
            high_level_interval=high_level_interval,
            max_steps=max_steps,
        )
        if demand_mixture is not None
        else _make_high_level_env(
            config_name=config_name,
            demand_level=demand_level,
            low_level_policy=low_level_policy,
            seed=seed,
            high_level_interval=high_level_interval,
            max_steps=max_steps,
        )
    )
    observation, _ = env.reset(seed=seed)
    active_count_max = float(sum(not aircraft.finished for aircraft in env.env.aircraft))
    total_spawned_count = float(len(env.env.aircraft))
    total_landed_count = float(sum(aircraft.finished for aircraft in env.env.aircraft))
    total_safety_override_count = float(env.env.safety_override_count)
    rewards: list[float] = []
    for _ in range(max(0, int(traffic_check_steps))):
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(8, dtype=np.float32)
        )
        rewards.append(float(reward))
        counts = _count_fields_from_info(info)
        active_count_max = max(active_count_max, counts["active_count_max"])
        total_spawned_count = max(total_spawned_count, counts["total_spawned_count"])
        total_landed_count = max(total_landed_count, counts["total_landed_count"])
        total_safety_override_count = max(
            total_safety_override_count,
            counts["total_safety_override_count"],
        )
        if terminated or truncated:
            break
    return {
        "config": config_name,
        "demand_level": demand_level,
        "demand_mixture": None if demand_mixture is None else dict(demand_mixture),
        "sampled_demand_level": getattr(env, "current_demand_level", demand_level),
        "low_level_policy": low_level_policy,
        "seed": int(seed),
        "traffic_check_steps": int(traffic_check_steps),
        "active_count_max": active_count_max,
        "total_spawned_count": total_spawned_count,
        "total_landed_count": total_landed_count,
        "total_safety_override_count": total_safety_override_count,
        "mean_reward": float(sum(rewards) / max(1, len(rewards))),
    }


def _evaluate_model(
    model: Any,
    *,
    config_name: str,
    demand_level: str,
    demand_mixture: Mapping[str, float] | None = None,
    low_level_policy: str,
    seed: int,
    high_level_interval: int | None,
    max_steps: int | None,
    episodes: int = 1,
    eval_high_level_steps: int = 20,
    save_trace_path: Path | None = None,
) -> dict[str, Any]:
    rewards: list[float] = []
    lengths: list[int] = []
    trace_rows: list[dict[str, Any]] = []
    max_active_count = 0.0
    total_spawned_count = 0.0
    total_landed_count = 0.0
    total_safety_override_count = 0.0
    for episode in range(episodes):
        env = (
            ScenarioMixtureHighLevelWrapper(
                config_name=config_name,
                demand_mixture=demand_mixture,
                low_level_policy=low_level_policy,
                seed=seed + 10_000 + episode,
                high_level_interval=high_level_interval,
                max_steps=max_steps,
            )
            if demand_mixture is not None
            else _make_high_level_env(
                config_name=config_name,
                demand_level=demand_level,
                low_level_policy=low_level_policy,
                seed=seed + 10_000 + episode,
                high_level_interval=high_level_interval,
                max_steps=max_steps,
            )
        )
        observation, _ = env.reset(seed=seed + 10_000 + episode)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        horizon = max(20, int(eval_high_level_steps))
        while not (terminated or truncated) and steps < horizon:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            counts = _count_fields_from_info(info)
            max_active_count = max(max_active_count, counts["active_count_max"])
            total_spawned_count = max(total_spawned_count, counts["total_spawned_count"])
            total_landed_count = max(total_landed_count, counts["total_landed_count"])
            total_safety_override_count = max(
                total_safety_override_count,
                counts["total_safety_override_count"],
            )
            if save_trace_path is not None:
                trace_rows.append(
                    {
                        "episode": episode,
                        "high_level_step": steps,
                        "reward": float(reward),
                        **counts,
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
                "active_count_start",
                "active_count_mean",
                "active_count_max",
                "active_count_end",
                "active_count",
                "spawned_count_delta",
                "landed_count_delta",
                "total_spawned_count",
                "total_landed_count",
                "safety_override_delta",
                "total_safety_override_count",
            ),
        )
    return {
        "eval_episodes": episodes,
        "eval_high_level_steps": max(20, int(eval_high_level_steps)),
        "eval_mean_reward": float(sum(rewards) / max(1, len(rewards))),
        "eval_rewards": rewards,
        "eval_lengths": lengths,
        "max_active_count": max_active_count,
        "total_spawned_count": total_spawned_count,
        "total_landed_count": total_landed_count,
        "total_safety_override_count": total_safety_override_count,
    }


def train_high_level_ppo(
    *,
    config_name: str,
    seed: int,
    total_timesteps: int,
    output_dir: str | Path,
    demand_level: str,
    demand_mixture: Mapping[str, float] | None = None,
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
    fail_if_no_traffic: bool = False,
    min_spawned_before_training_check: int = 1,
    traffic_check_steps: int = 20,
    sanity_check_baseline: bool = False,
    eval_high_level_steps: int = 20,
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
    traffic_check_summary = None
    if fail_if_no_traffic:
        traffic_check_summary = run_pretraining_traffic_check(
            config_name=config_name,
            demand_level=demand_level,
            demand_mixture=demand_mixture,
            low_level_policy=low_level_policy,
            seed=seed,
            high_level_interval=high_level_interval,
            max_steps=max_steps,
            traffic_check_steps=traffic_check_steps,
        )
        (output / "traffic_check_summary.json").write_text(
            json.dumps(traffic_check_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        if (
            traffic_check_summary["total_spawned_count"]
            < float(min_spawned_before_training_check)
            or traffic_check_summary["active_count_max"] <= 0.0
        ):
            raise RuntimeError(
                "Insufficient traffic detected before PPO training. "
                "Check arrival_process/config/demand overrides."
            )
    sanity_summary = None
    if sanity_check_baseline:
        sanity_summary = run_pretraining_traffic_check(
            config_name=config_name,
            demand_level=demand_level,
            demand_mixture=demand_mixture,
            low_level_policy="rule-only",
            seed=seed + 123,
            high_level_interval=high_level_interval,
            max_steps=max_steps,
            traffic_check_steps=traffic_check_steps,
        )
        (output / "sanity_check_summary.json").write_text(
            json.dumps(sanity_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        if sanity_summary["total_spawned_count"] <= 0.0 and sanity_summary["active_count_max"] <= 0.0:
            raise RuntimeError(
                "Rule-only sanity check found no traffic. Arrival process/config is not active."
            )
        if sanity_summary["total_landed_count"] <= 0.0:
            warnings.warn(
                "Rule-only sanity check observed traffic but no landings in the short rollout.",
                RuntimeWarning,
                stacklevel=2,
            )
    env_factories = [
        _make_env_factory(
            config_name=config_name,
            demand_level=demand_level,
            demand_mixture=demand_mixture,
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
                    demand_mixture=demand_mixture,
                    low_level_policy=low_level_policy,
                    seed=seed,
                    high_level_interval=high_level_interval,
                    max_steps=max_steps,
                    episodes=1,
                    eval_high_level_steps=eval_high_level_steps,
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
    effective_n_steps = _select_exact_rollout_n_steps(
        requested_n_steps=max(1, int(n_steps)),
        total_timesteps=int(total_timesteps),
        n_envs=max(1, int(n_envs)),
    )
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
        "demand_mixture": None if demand_mixture is None else dict(demand_mixture),
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
        "rollout_size": effective_n_steps * max(1, n_envs),
        "exact_timestep_budget": True,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "high_level_interval": high_level_interval,
        "checkpoint_freq": checkpoint_freq,
        "eval_freq": eval_freq,
        "save_replay_traces": save_replay_traces,
        "fail_if_no_traffic": fail_if_no_traffic,
        "min_spawned_before_training_check": min_spawned_before_training_check,
        "traffic_check_steps": traffic_check_steps,
        "sanity_check_baseline": sanity_check_baseline,
        "eval_high_level_steps": max(20, int(eval_high_level_steps)),
        "max_steps": None if max_steps is None else int(max_steps),
        "resume_from": None if resume_from is None else str(resume_from),
        "nominal_delay_mode": nominal_delay_mode,
        "simulation_only": True,
    }
    resolved_sim_config = _resolved_sim_training_snapshot(
        config_name=config_name,
        demand_level=demand_level,
        high_level_interval=high_level_interval,
        max_steps=max_steps,
    )
    resolved_training_config = {
        **training_config,
        "resolved_sim_config": resolved_sim_config,
    }
    (output / "training_config.json").write_text(
        json.dumps(training_config, indent=2) + "\n", encoding="utf-8"
    )
    (output / "config_snapshot.json").write_text(
        json.dumps(training_config, indent=2) + "\n", encoding="utf-8"
    )
    (output / "resolved_training_config.json").write_text(
        json.dumps(resolved_training_config, indent=2) + "\n", encoding="utf-8"
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
        demand_mixture=demand_mixture,
        low_level_policy=low_level_policy,
        seed=seed + 1,
        high_level_interval=high_level_interval,
        max_steps=max_steps,
        episodes=1,
        eval_high_level_steps=eval_high_level_steps,
        save_trace_path=output / "final_eval.csv",
    )
    summary["periodic_evaluations"] = eval_rows
    summary["traffic_check_summary"] = traffic_check_summary
    summary["sanity_check_summary"] = sanity_summary
    (output / "eval_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    env.close()
    return summary


def _resolved_sim_training_snapshot(
    *,
    config_name: str,
    demand_level: str,
    high_level_interval: int | None,
    max_steps: int | None,
) -> dict[str, Any]:
    config = apply_demand_profile(_load_base_sim_config(config_name), demand_level)
    if high_level_interval is not None:
        config.high_level_interval_steps = int(high_level_interval)
    if max_steps is not None:
        config.max_steps = min(config.max_steps, int(max_steps))
    return {
        "simulation_only": bool(config.simulation_only),
        "dt": float(config.dt),
        "control_interval": float(getattr(config, "control_interval", config.dt)),
        "max_steps": int(config.max_steps),
        "episode_duration_seconds": float(config.max_steps * config.dt),
        "high_level_interval_steps": int(config.high_level_interval_steps),
        "metrics_sampling_interval_seconds": float(config.metrics_sampling_interval_seconds),
        "high_level_decision_interval_seconds": float(config.decision_interval_seconds),
        "geometry_min_dwell_seconds": float(config.reconfiguration_dwell_seconds),
        "geometry_gate_mode": str(config.geometry_gate_mode),
        "traffic_demand_level": str(config.traffic_demand_level),
        "arrival_rate": float(config.arrival_rate),
        "arrival_model": str(config.arrival_model),
        "reward_scaling_warning": False,
    }


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
        demand_mixture=training_config.demand_mixture,
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
        fail_if_no_traffic=training_config.fail_if_no_traffic,
        min_spawned_before_training_check=training_config.min_spawned_before_training_check,
        traffic_check_steps=training_config.traffic_check_steps,
        sanity_check_baseline=training_config.sanity_check_baseline,
        eval_high_level_steps=training_config.eval_high_level_steps,
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
    parser.add_argument("--fail-if-no-traffic", action="store_true", default=None)
    parser.add_argument("--min-spawned-before-training-check", type=int)
    parser.add_argument("--traffic-check-steps", type=int)
    parser.add_argument("--sanity-check-baseline", action="store_true", default=None)
    parser.add_argument("--eval-high-level-steps", type=int)
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
            "fail_if_no_traffic": args.fail_if_no_traffic,
            "min_spawned_before_training_check": args.min_spawned_before_training_check,
            "traffic_check_steps": args.traffic_check_steps,
            "sanity_check_baseline": args.sanity_check_baseline,
            "max_steps": args.max_steps,
            "eval_high_level_steps": args.eval_high_level_steps,
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
