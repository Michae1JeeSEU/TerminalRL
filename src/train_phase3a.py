"""Phase 3a: Single UAV landing with background circling UAVs as interference."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import EnvConfig
from src.env.terminal_area_env import TerminalAreaEnv
from src.env.wrappers import DiscreteActionWrapper


class Phase3aWrapper(gym.Wrapper):
    """Phase 3a scenario: N background UAVs circling, 1 controlled UAV landing.

    On reset:
    - N_bg background UAVs are spawned on ring orbits (all CRUISING, only CIRCLE)
    - 1 additional UAV spawns at a random entry point (ring 1)
    - Only this entry UAV is controlled by the agent
    """

    def __init__(self, env: TerminalAreaEnv, n_background: int = 10):
        super().__init__(env)
        self.n_background = n_background
        self.controlled_idx: int = n_background  # the entry UAV's list index
        self._entry_spawned = False

    def reset(self, *, seed=None, options=None):
        # Spawn background UAVs only, then add 1 entry UAV
        opts = options or {}
        opts["initial_uavs"] = self.n_background
        obs, info = self.env.reset(seed=seed, options=opts)

        # Mark all background UAVs as not controlled
        for u in self.unwrapped.uav_manager.uavs:
            u.is_controlled = False

        # Spawn the controlled entry UAV with full battery
        entry_uav = self.unwrapped.uav_manager._spawn_one(
            ring_id=1, battery=self.unwrapped.config.uav.max_battery
        )
        entry_uav.is_controlled = True
        self.controlled_idx = len(self.unwrapped.uav_manager.uavs) - 1
        self._entry_spawned = True

        return self.env._get_obs(), self.env._get_info()

    def step(self, action):
        return self.env.step(action)


def make_env_phase3a(config: EnvConfig, n_background: int = 10):
    def _init():
        env = TerminalAreaEnv(config)
        env = Phase3aWrapper(env, n_background=n_background)
        env = DiscreteActionWrapper(env)
        return Monitor(env)
    return _init


def evaluate_phase3a(env: TerminalAreaEnv, policy, num_episodes: int = 20,
                     seed: int = 42) -> dict:
    """Evaluate a single-UAV policy in the Phase 3a scenario."""
    all_landed = []
    all_steps = []
    all_conflicts = []
    all_energy = []

    for ep in range(num_episodes):
        test_env = TerminalAreaEnv(env.config)
        wrapper = Phase3aWrapper(test_env, n_background=n_background)
        obs, _ = wrapper.reset(seed=seed + ep)
        step = 0
        ep_conflicts = 0

        while True:
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = wrapper.step(action)
            ep_conflicts += info.get("conflicts", 0)
            step += 1
            if terminated or truncated:
                break

        all_landed.append(1 if wrapper.unwrapped.uav_manager.landed_count > 0 else 0)

        ep_metrics = wrapper.unwrapped.get_episode_metrics()
        all_steps.append(step)
        all_conflicts.append(ep_conflicts)
        all_energy.append(ep_metrics.total_energy)

    return {
        "landing_rate": float(np.mean(all_landed)),
        "avg_steps": float(np.mean(all_steps)),
        "avg_conflicts": float(np.mean(all_conflicts)),
        "avg_energy": float(np.mean(all_energy)),
    }


def train_phase3a(config: EnvConfig | None = None, n_background: int = 10,
                  total_timesteps: int = 200_000):
    """Train PPO for single-UAV landing with background interference."""
    if config is None:
        config = EnvConfig()
        config.num_uavs = 0   # no initial UAVs, Phase3aWrapper handles spawning
        config.max_uavs = 20
        config.sim.max_steps = 400
        config.uav_inflow_rate = 0.0

    env = DummyVecEnv([make_env_phase3a(config, n_background)])
    eval_env = DummyVecEnv([make_env_phase3a(config, n_background)])

    eval_callback = EvalCallback(
        eval_env, best_model_save_path="logs/best_model_phase3a",
        log_path="logs/", eval_freq=5000,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy", env,
        learning_rate=1e-3,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        verbose=1,
    )

    print(f"Training Phase 3a PPO: {n_background} background UAVs, {total_timesteps} steps...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print(f"Training completed in {time.time() - t0:.1f}s")

    model.save("logs/ppo_phase3a")
    return model, config


def evaluate_ppo_phase3a(model, config: EnvConfig, n_background: int = 10,
                         num_episodes: int = 20):
    """Evaluate trained PPO vs heuristic in Phase 3a scenario."""
    from src.baselines.heuristic_policy import HeuristicPolicy

    print(f"\n=== Phase 3a: {n_background} background UAVs ===")

    # PPO evaluation
    print("--- PPO ---")
    all_landed, all_steps, all_conflicts = [], [], []
    for ep in range(num_episodes):
        env = TerminalAreaEnv(config)
        wrapper = Phase3aWrapper(env, n_background=n_background)
        disc = DiscreteActionWrapper(wrapper)
        obs, _ = disc.reset(seed=42 + ep)
        step, conflicts = 0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = disc.step(action)
            conflicts += info.get("conflicts", 0)
            step += 1
            if terminated or truncated:
                break
        all_landed.append(1 if env.uav_manager.landed_count > 0 else 0)
        all_steps.append(step)
        all_conflicts.append(conflicts)

    ppo_rate = float(np.mean(all_landed))
    print(f"  Landing rate: {ppo_rate:.1%}, avg steps: {np.mean(all_steps):.0f}, "
          f"conflicts: {np.mean(all_conflicts):.1f}")

    # Heuristic evaluation
    print("--- Heuristic ---")
    heuristic = HeuristicPolicy(config)
    all_landed_h, all_steps_h, all_conflicts_h = [], [], []
    for ep in range(num_episodes):
        env = TerminalAreaEnv(config)
        wrapper = Phase3aWrapper(env, n_background=n_background)
        obs, _ = wrapper.reset(seed=42 + ep)
        step, conflicts = 0, 0
        while True:
            action = heuristic.predict(obs)
            obs, _, terminated, truncated, info = wrapper.step(action)
            conflicts += info.get("conflicts", 0)
            step += 1
            if terminated or truncated:
                break
        all_landed_h.append(1 if wrapper.unwrapped.uav_manager.landed_count > 0 else 0)
        all_steps_h.append(step)
        all_conflicts_h.append(conflicts)

    heur_rate = float(np.mean(all_landed_h))
    print(f"  Landing rate: {heur_rate:.1%}, avg steps: {np.mean(all_steps_h):.0f}, "
          f"conflicts: {np.mean(all_conflicts_h):.1f}")

    return {"ppo_landing_rate": ppo_rate, "heuristic_landing_rate": heur_rate}


if __name__ == "__main__":
    from src.baselines.heuristic_policy import HeuristicPolicy

    config = EnvConfig()
    config.num_uavs = 0
    config.max_uavs = 25  # > max n_background (20) + 1 entry UAV
    config.sim.max_steps = 400

    for n_bg in [5, 10, 20]:
        hp = HeuristicPolicy(config)
        all_ok = 0
        total_conflicts = 0
        total_steps = 0
        for ep in range(10):
            env = TerminalAreaEnv(config)
            wrapper = Phase3aWrapper(env, n_background=n_bg)
            obs, _ = wrapper.reset(seed=ep)
            step = 0
            ep_conflicts = 0
            while True:
                action = hp.predict(obs)
                obs, _, terminated, truncated, info = wrapper.step(action)
                ep_conflicts += info.get("conflicts", 0)
                step += 1
                if info.get("landed", 0) > 0 or terminated or truncated:
                    break
            if wrapper.unwrapped.uav_manager.landed_count > 0:
                all_ok += 1
            total_conflicts += ep_conflicts
            total_steps += step
        print(f"  {n_bg} bg UAVs: {all_ok}/10 landed ({all_ok*10}%), "
              f"avg conflicts={(total_conflicts/10):.1f}, avg steps={(total_steps/10):.0f}")
