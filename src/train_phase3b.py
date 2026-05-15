"""Phase 3b: All UAVs controlled, clear the airspace — per-UAV shared policy.

Scenario: N UAVs distributed on ring orbits (random angle/battery/priority/wait).
Agent controls all N UAVs independently via a shared policy network.
Episode ends when all UAVs have landed or failed. No inflow.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn
import torch

from src.config import EnvConfig
from src.env.terminal_area_env import TerminalAreaEnv
from src.env.uav import UAVStatus
from src.env.wrappers import PerUAVDiscreteWrapper
from src.baselines.heuristic_policy import HeuristicPolicy


# ---- Phase 3b Scenario Wrapper ----

class Phase3bWrapper(gym.Wrapper):
    """All UAVs controlled, terminate when all land or fail.

    Gives all UAVs ample battery so the focus is on navigation strategy.
    """

    def __init__(self, env: TerminalAreaEnv, n_uavs: int = 10):
        super().__init__(env)
        self.n_uavs = n_uavs

    @property
    def max_n(self) -> int:
        return self.env.max_n

    @property
    def config(self):
        return self.env.config

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for u in self.unwrapped.uav_manager.uavs:
            u.is_controlled = True
            u.battery = self.unwrapped.config.uav.max_battery * 5.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if len(self.unwrapped.uav_manager.uavs) == 0:
            terminated = True
        return obs, reward, terminated, truncated, info


# ---- Per-UAV Shared Feature Extractor ----

class PerUAVFeatureExtractor(BaseFeaturesExtractor):
    """Shared per-UAV encoder + global aggregator → flat feature vector.

    Input: (max_n, local_obs_dim) matrix.
    Each UAV row is independently encoded with a shared MLP, then all UAV
    embeddings are concatenated and aggregated into a global feature vector.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_uavs, obs_dim = observation_space.shape
        self.n_uavs = n_uavs
        self.obs_dim = obs_dim

        self.uav_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.global_net = nn.Sequential(
            nn.Linear(64 * n_uavs, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        flat = observations.view(batch_size * self.n_uavs, self.obs_dim)
        uav_features = self.uav_encoder(flat)
        uav_features = uav_features.view(batch_size, self.n_uavs * 64)
        return self.global_net(uav_features)


# ---- Environment Factory ----

def make_env_phase3b(config: EnvConfig, n_uavs: int = 10):
    def _init():
        env = TerminalAreaEnv(config)
        env = Phase3bWrapper(env, n_uavs=n_uavs)
        env = PerUAVDiscreteWrapper(env)
        return Monitor(env)
    return _init


# ---- Evaluation ----

def evaluate_phase3b(policy, config: EnvConfig, n_uavs: int = 10,
                     num_episodes: int = 20, seed: int = 42) -> dict:
    """Evaluate a per-UAV shared policy in the Phase 3b scenario."""
    all_landed = []
    all_failed = []
    all_steps = []
    all_conflicts = []
    all_near_misses = []
    all_energy = []

    for ep in range(num_episodes):
        env = TerminalAreaEnv(config)
        wrapper = Phase3bWrapper(env, n_uavs=n_uavs)
        disc = PerUAVDiscreteWrapper(wrapper)
        obs, _ = disc.reset(seed=seed + ep)
        step = 0
        ep_conflicts = 0
        ep_near_misses = 0

        while True:
            action = policy.predict(obs)
            obs, _, terminated, truncated, info = disc.step(action)
            ep_conflicts += info.get("conflicts", 0)
            ep_near_misses += info.get("near_misses", 0)
            step += 1
            if terminated or truncated:
                break

        all_landed.append(info.get("landed", 0))
        all_failed.append(info.get("failed", 0))
        all_steps.append(step)
        all_conflicts.append(ep_conflicts)
        all_near_misses.append(ep_near_misses)
        all_energy.append(env.get_episode_metrics().total_energy)

    return {
        "avg_landed": float(np.mean(all_landed)),
        "avg_failed": float(np.mean(all_failed)),
        "landing_rate": float(np.mean([l / max(l + f, 1) for l, f in zip(all_landed, all_failed)])),
        "avg_steps": float(np.mean(all_steps)),
        "avg_conflicts": float(np.mean(all_conflicts)),
        "avg_near_misses": float(np.mean(all_near_misses)),
        "avg_energy": float(np.mean(all_energy)),
    }


# ---- PPO Training ----

def train_phase3b(config: EnvConfig | None = None, n_uavs: int = 10,
                  total_timesteps: int = 500_000):
    """Train PPO with per-UAV shared policy for Phase 3b."""
    if config is None:
        config = EnvConfig()
        config.num_uavs = n_uavs
        config.max_uavs = n_uavs + 5
        config.sim.max_steps = 800
        config.uav_inflow_rate = 0.0

    env = DummyVecEnv([make_env_phase3b(config, n_uavs)])
    eval_env = DummyVecEnv([make_env_phase3b(config, n_uavs)])

    eval_callback = EvalCallback(
        eval_env, best_model_save_path="logs/best_model_phase3b",
        log_path="logs/", eval_freq=10000,
        deterministic=True,
    )

    policy_kwargs = dict(
        features_extractor_class=PerUAVFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        ent_coef=0.1,
        verbose=1,
    )

    print(f"Training Phase 3b PPO: {n_uavs} UAVs, {total_timesteps} steps...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print(f"Training completed in {time.time() - t0:.1f}s")

    model.save("logs/ppo_phase3b")
    return model, config


# ---- Compare PPO vs Heuristic ----

def compare_ppo_vs_heuristic_phase3b(model, config: EnvConfig, n_uavs: int = 10,
                                     num_episodes: int = 20):
    """Compare trained PPO vs HeuristicPolicy in Phase 3b."""
    from src.env.wrappers import DiscreteActionWrapper

    heuristic = HeuristicPolicy(config)

    print(f"\n=== Phase 3b: {n_uavs} UAVs, all controlled ===")

    # --- PPO (PerUAVDiscreteWrapper) ---
    print("--- PPO (per-UAV shared policy) ---")
    ppo_landed = []
    ppo_steps = []
    ppo_collisions = []
    ppo_near = []
    for ep in range(num_episodes):
        env = TerminalAreaEnv(config)
        wrapper = Phase3bWrapper(env, n_uavs=n_uavs)
        disc = PerUAVDiscreteWrapper(wrapper)
        obs, _ = disc.reset(seed=42 + ep)
        step, collisions, nears = 0, 0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = disc.step(action)
            collisions += info.get("conflicts", 0)
            nears += info.get("near_misses", 0)
            step += 1
            if terminated or truncated:
                break
        ppo_landed.append(info.get("landed", 0))
        ppo_steps.append(step)
        ppo_collisions.append(collisions)
        ppo_near.append(nears)

    ppo_l = np.mean(ppo_landed)
    ppo_s = np.mean(ppo_steps)
    ppo_c = np.mean(ppo_collisions)
    ppo_n = np.mean(ppo_near)
    print(f"  Landed: {ppo_l:.1f}, Steps: {ppo_s:.0f}, Collisions: {ppo_c:.2f}, Near-misses: {ppo_n:.1f}")

    # --- Heuristic ---
    print("--- Heuristic (per-UAV greedy) ---")
    heur_landed = []
    heur_steps = []
    heur_collisions = []
    heur_near = []
    for ep in range(num_episodes):
        env = TerminalAreaEnv(config)
        wrapper = Phase3bWrapper(env, n_uavs=n_uavs)
        obs, _ = wrapper.reset(seed=42 + ep)
        step, collisions, nears = 0, 0, 0
        while True:
            action = heuristic.predict(obs)
            obs, _, terminated, truncated, info = wrapper.step(action)
            collisions += info.get("conflicts", 0)
            nears += info.get("near_misses", 0)
            step += 1
            if terminated or truncated:
                break
        heur_landed.append(info.get("landed", 0))
        heur_steps.append(step)
        heur_collisions.append(collisions)
        heur_near.append(nears)

    heur_l = np.mean(heur_landed)
    heur_s = np.mean(heur_steps)
    heur_c = np.mean(heur_collisions)
    heur_n = np.mean(heur_near)
    print(f"  Landed: {heur_l:.1f}, Steps: {heur_s:.0f}, Collisions: {heur_c:.2f}, Near-misses: {heur_n:.1f}")

    return {
        "n_uavs": n_uavs,
        "safe_separation_m": config.uav.safe_separation,
        "near_miss_threshold_m": config.uav.near_miss_threshold,
        "ppo": {"avg_landed": float(ppo_l), "avg_steps": float(ppo_s),
                "avg_collisions": float(ppo_c), "avg_near_misses": float(ppo_n)},
        "heuristic": {"avg_landed": float(heur_l), "avg_steps": float(heur_s),
                      "avg_collisions": float(heur_c), "avg_near_misses": float(heur_n)},
    }


# ---- Quick Heuristic Smoke Test ----

def run_heuristic_baseline(n_uavs_list: list[int] | None = None,
                           num_episodes: int = 10):
    """Quick heuristic baseline for Phase 3b across different UAV counts."""
    if n_uavs_list is None:
        n_uavs_list = [5, 10, 15]

    print("=== Phase 3b Heuristic Baseline ===\n")
    for n in n_uavs_list:
        config = EnvConfig()
        config.num_uavs = n
        config.max_uavs = n + 5
        config.sim.max_steps = 800
        config.uav_inflow_rate = 0.0
        heuristic = HeuristicPolicy(config)
        all_landed = []
        all_steps = []
        all_collisions = []
        all_near = []

        for ep in range(num_episodes):
            env = TerminalAreaEnv(config)
            wrapper = Phase3bWrapper(env, n_uavs=n)
            obs, _ = wrapper.reset(seed=ep)
            step, collisions, nears = 0, 0, 0
            while True:
                action = heuristic.predict(obs)
                obs, _, terminated, truncated, info = wrapper.step(action)
                collisions += info.get("conflicts", 0)
                nears += info.get("near_misses", 0)
                step += 1
                if terminated or truncated:
                    break
            all_landed.append(info.get("landed", 0))
            all_steps.append(step)
            all_collisions.append(collisions)
            all_near.append(nears)

        collision_eps = sum(1 for c in all_collisions if c > 0)
        print(f"  N={n}: landed={np.mean(all_landed):.1f}/{n}, "
              f"steps={np.mean(all_steps):.0f}, collisions={np.mean(all_collisions):.3f}, "
              f"near_misses={np.mean(all_near):.1f} ({collision_eps}/{num_episodes} eps with collision)")

    return True


if __name__ == "__main__":
    import sys

    if "--heuristic" in sys.argv:
        run_heuristic_baseline()
    elif "--evaluate" in sys.argv:
        config = EnvConfig()
        config.num_uavs = 10
        config.max_uavs = 15
        config.sim.max_steps = 800
        config.uav_inflow_rate = 0.0

        import os
        model_path = "logs/ppo_phase3b.zip"
        if os.path.exists(model_path):
            model = PPO.load(model_path)
            result = compare_ppo_vs_heuristic_phase3b(model, config, n_uavs=10)
            Path("results").mkdir(exist_ok=True)
            json.dump(result, open("results/phase3b_comparison.json", "w"), indent=2)
            print("Results saved to results/phase3b_comparison.json")
        else:
            print(f"Model not found at {model_path}. Train first with: python -m src.train_phase3b")
    else:
        model, config = train_phase3b(n_uavs=10, total_timesteps=500_000)
        result = compare_ppo_vs_heuristic_phase3b(model, config, n_uavs=10)
        Path("results").mkdir(exist_ok=True)
        json.dump(result, open("results/phase3b_comparison.json", "w"), indent=2)
        print("Results saved to results/phase3b_comparison.json")
