"""PPO training for multi-UAV terminal area landing (Phase 4)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import EnvConfig
from src.env.terminal_area_env import TerminalAreaEnv
from src.env.wrappers import DiscreteActionWrapper
from src.baselines.heuristic_policy import HeuristicPolicy
from src.train import evaluate_policy


def make_env(config: EnvConfig):
    def _init():
        env = TerminalAreaEnv(config)
        env = DiscreteActionWrapper(env)
        return Monitor(env)
    return _init


def train_ppo(config: EnvConfig | None = None, total_timesteps: int = 500_000):
    if config is None:
        config = EnvConfig()
        config.num_uavs = 5
        config.sim.max_steps = 400

    env = DummyVecEnv([make_env(config)])
    eval_env = DummyVecEnv([make_env(config)])

    eval_callback = EvalCallback(
        eval_env, best_model_save_path="logs/best_model_multi",
        log_path="logs/", eval_freq=10000,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy", env,
        learning_rate=1e-3,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        verbose=1,
    )

    print(f"Training PPO for {total_timesteps} timesteps with {config.num_uavs} UAVs...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print(f"Training completed in {time.time() - t0:.1f}s")

    model.save("logs/ppo_multi_uav")
    return model, config


def evaluate_ppo_wrapper(env: TerminalAreaEnv, model, num_episodes: int = 20) -> dict:
    """Evaluate a trained PPO model using DiscreteActionWrapper."""
    wrapper = DiscreteActionWrapper(env)
    all_landed, all_failed, all_wait, all_energy, all_conflicts, ep_lens = \
        [], [], [], [], [], []

    for ep in range(num_episodes):
        obs, _ = wrapper.reset(seed=42 + ep)
        ep_conflicts = 0
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapper.step(action)
            ep_conflicts += info.get("conflicts", 0)
            step += 1
            if terminated or truncated:
                break

        ep_metrics = env.get_episode_metrics()
        all_landed.append(env.uav_manager.landed_count)
        all_failed.append(env.uav_manager.failed_count)
        all_wait.append(ep_metrics.avg_wait_time)
        all_energy.append(ep_metrics.total_energy)
        all_conflicts.append(ep_conflicts)
        ep_lens.append(step)

    return {
        "avg_landed": float(np.mean(all_landed)),
        "avg_failed": float(np.mean(all_failed)),
        "avg_wait_time": float(np.mean(all_wait)),
        "avg_energy": float(np.mean(all_energy)),
        "avg_conflicts": float(np.mean(all_conflicts)),
        "avg_episode_length": float(np.mean(ep_lens)),
    }


def compare_ppo_vs_heuristic(model, config: EnvConfig, num_episodes: int = 20):
    print("\n=== Evaluating Trained PPO ===")
    env1 = TerminalAreaEnv(config)
    t0 = time.time()
    ppo_metrics = evaluate_ppo_wrapper(env1, model, num_episodes)
    ppo_metrics["eval_time_s"] = time.time() - t0
    _print_metrics("PPO", ppo_metrics)

    print("\n=== Evaluating Heuristic ===")
    env2 = TerminalAreaEnv(config)
    heuristic = HeuristicPolicy(config)
    t0 = time.time()
    heur_metrics = evaluate_policy(env2, heuristic, num_episodes)
    heur_metrics["eval_time_s"] = time.time() - t0
    _print_metrics("Heuristic", heur_metrics)

    Path("results").mkdir(exist_ok=True)
    with open("results/ppo_vs_heuristic_multi.json", "w") as f:
        json.dump({"ppo": ppo_metrics, "heuristic": heur_metrics}, f, indent=2)
    print("\nResults saved to results/ppo_vs_heuristic_multi.json")


def _print_metrics(name: str, m: dict) -> None:
    print(f"  {name}: landed={m['avg_landed']:.1f}, failed={m['avg_failed']:.1f}, "
          f"wait={m['avg_wait_time']:.1f}s, energy={m['avg_energy']:.1f}, "
          f"conflicts={m.get('avg_conflicts', 0):.1f}, "
          f"ep_len={m['avg_episode_length']:.1f}, time={m['eval_time_s']:.1f}s")


if __name__ == "__main__":
    model, config = train_ppo(total_timesteps=500_000)
    compare_ppo_vs_heuristic(model, config)
