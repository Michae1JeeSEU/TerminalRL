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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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
            u.battery = self.unwrapped.config.uav.max_battery * 100.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if len(self.unwrapped.uav_manager.uavs) == 0:
            terminated = True
        return obs, reward, terminated, truncated, info


# ---- Per-UAV Shared Feature Extractor ----

class PerUAVFeatureExtractor(BaseFeaturesExtractor):
    """Shared per-UAV encoder + mean-pooled aggregator → per-UAV feature vector.

    Input: (max_n, local_obs_dim) matrix.
    Each UAV row is independently encoded with a shared MLP, then all active UAV
    embeddings are mean-pooled into a global context. Each UAV's final feature
    combines [global_context | local_embedding] through a shared combiner.

    The encoder and combiner weights are independent of max_n — only the final
    flatten dimension (max_n * per_uav_dim) scales with max_n, which is inherent
    to SB3's fixed output requirement.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_uavs, obs_dim = observation_space.shape
        self.n_uavs = n_uavs
        self.obs_dim = obs_dim
        self.per_uav_dim = 8

        self.uav_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.per_uav_combiner = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.per_uav_dim),
            nn.ReLU(),
        )

        self._features_dim = n_uavs * self.per_uav_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B, N, D = observations.shape

        flat = observations.view(B * N, D)
        uav_embeds = self.uav_encoder(flat).view(B, N, 64)

        # Active mask: rows with non-zero observation are active UAVs
        active_mask = (observations.abs().sum(dim=-1) > 1e-6).float()  # (B, N)
        n_active = active_mask.sum(dim=1, keepdim=True).clamp(min=1)   # (B, 1)

        # Mean pool active UAV embeddings → global context
        masked = uav_embeds * active_mask.unsqueeze(-1)               # (B, N, 64)
        global_ctx = masked.sum(dim=1) / n_active                      # (B, 64)

        # Combine global context with each UAV's local embedding
        global_exp = global_ctx.unsqueeze(1).expand(-1, N, -1)        # (B, N, 64)
        combined = torch.cat([global_exp, uav_embeds], dim=-1)        # (B, N, 128)

        # Shared per-UAV combiner
        combined_flat = combined.view(B * N, 128)
        per_uav = self.per_uav_combiner(combined_flat)                 # (B*N, per_uav_dim)
        return per_uav.view(B, N * self.per_uav_dim)                   # (B, features_dim)


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

def train_phase3b(config: EnvConfig | None = None, n_uavs: int = 20,
                  total_timesteps: int = 500_000, n_envs: int = 4):
    """Train PPO with per-UAV shared policy for Phase 3b.

    SafetyController is enabled during training. The low-level policy only
    requests CIRCLE/DESCEND and circling speed; the environment meters descent
    releases and generates rule-based descent trajectories.

    Uses SubprocVecEnv for parallel training across multiple env instances.
    """
    if config is None:
        config = EnvConfig()
        config.num_uavs = n_uavs
        config.max_uavs = 100
        config.sim.max_steps = 1200
        config.uav_inflow_rate = 0.0
        config.safety_enabled = True

    # Parallel envs for training
    env_factories = [make_env_phase3b(config, n_uavs) for _ in range(n_envs)]
    if n_envs > 1:
        env = SubprocVecEnv(env_factories)
    else:
        env = DummyVecEnv(env_factories)

    eval_env = DummyVecEnv([make_env_phase3b(config, n_uavs)])

    eval_callback = EvalCallback(
        eval_env, best_model_save_path="logs/best_model_phase3b",
        log_path="logs/", eval_freq=max(10000 // n_envs, 2000),
        deterministic=True,
    )

    policy_kwargs = dict(
        features_extractor_class=PerUAVFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    # Scale batch_size and n_steps with n_envs
    n_steps = 2048 // n_envs  # keep total steps per rollout similar
    batch_size = max(64, 256 // n_envs)

    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=5,
        gamma=0.99,
        ent_coef=0.1,
        verbose=1,
    )

    print(f"Training Phase 3b PPO: {n_uavs} UAVs, {total_timesteps} steps, "
          f"{n_envs} parallel envs...")
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
        n_uavs_list = [5, 10, 15, 20, 25, 30]

    print("=== Phase 3b Heuristic Baseline ===\n")
    for n in n_uavs_list:
        config = EnvConfig()
        config.num_uavs = n
        config.max_uavs = 100
        config.sim.max_steps = 1200
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


def eval_scale(model, base_config: EnvConfig, n_list: list[int] | None = None,
               num_episodes: int = 10):
    """Evaluate PPO vs Heuristic across multiple UAV counts to test generalization."""
    if n_list is None:
        n_list = [5, 10, 15, 20, 25, 30]

    print(f"\n{'='*60}")
    print(f"Multi-Scale Evaluation: PPO vs Heuristic")
    print(f"{'='*60}\n")

    results = {}
    for n in n_list:
        config = EnvConfig()
        config.num_uavs = n
        config.max_uavs = base_config.max_uavs
        config.sim.max_steps = base_config.sim.max_steps
        config.uav_inflow_rate = 0.0

        result = compare_ppo_vs_heuristic_phase3b(model, config, n_uavs=n, num_episodes=num_episodes)
        results[str(n)] = result

        ppo_l = result["ppo"]["avg_landed"]
        heur_l = result["heuristic"]["avg_landed"]
        print(f"  N={n:2d}: PPO {ppo_l:.1f}/{n} ({ppo_l/n*100:.0f}%) | "
              f"Heuristic {heur_l:.1f}/{n} ({heur_l/n*100:.0f}%) | "
              f"PPO collisions={result['ppo']['avg_collisions']:.2f}")

    Path("results").mkdir(exist_ok=True)
    json.dump(results, open("results/phase3b_scale_comparison.json", "w"), indent=2)
    print(f"\nResults saved to results/phase3b_scale_comparison.json")

    # Summary table
    print(f"\n{'N':<6} {'PPO Landed':<12} {'PPO Rate':<10} {'Heur Landed':<12} {'Heur Rate':<10}")
    print("-" * 50)
    for n in n_list:
        r = results[str(n)]
        print(f"{n:<6} {r['ppo']['avg_landed']:<12.1f} {r['ppo']['avg_landed']/n*100:<10.0f}% "
              f"{r['heuristic']['avg_landed']:<12.1f} {r['heuristic']['avg_landed']/n*100:<10.0f}%")

    return results


if __name__ == "__main__":
    import sys

    base_config = EnvConfig()
    base_config.max_uavs = 100
    base_config.sim.max_steps = 1200
    base_config.uav_inflow_rate = 0.0

    if "--heuristic" in sys.argv:
        run_heuristic_baseline()
    elif "--eval-scale" in sys.argv:
        import os
        model_path = "logs/best_model_phase3b/best_model.zip"
        if not os.path.exists(model_path):
            model_path = "logs/ppo_phase3b.zip"
        if os.path.exists(model_path):
            model = PPO.load(model_path)
            eval_scale(model, base_config)
        else:
            print(f"No model found. Train first with: python -m src.train_phase3b")
    elif "--evaluate" in sys.argv:
        import os
        model_path = "logs/best_model_phase3b/best_model.zip"
        if not os.path.exists(model_path):
            model_path = "logs/ppo_phase3b.zip"
        if os.path.exists(model_path):
            model = PPO.load(model_path)
            result = compare_ppo_vs_heuristic_phase3b(model, base_config, n_uavs=10)
            Path("results").mkdir(exist_ok=True)
            json.dump(result, open("results/phase3b_comparison.json", "w"), indent=2)
            print("Results saved to results/phase3b_comparison.json")
        else:
            print(f"No model found. Train first with: python -m src.train_phase3b")
    else:
        n_envs = 4  # parallel environments for faster training
        model, train_cfg = train_phase3b(n_uavs=10, total_timesteps=500_000, n_envs=n_envs)
        eval_cfg = EnvConfig()
        eval_cfg.num_uavs = 10
        eval_cfg.max_uavs = train_cfg.max_uavs
        eval_cfg.sim.max_steps = train_cfg.sim.max_steps
        eval_cfg.uav_inflow_rate = 0.0
        eval_cfg.safety_enabled = True
        result = compare_ppo_vs_heuristic_phase3b(model, eval_cfg, n_uavs=10)
        Path("results").mkdir(exist_ok=True)
        json.dump(result, open("results/phase3b_comparison.json", "w"), indent=2)
        print("Results saved to results/phase3b_comparison.json")
