"""Terminal area HRL training and evaluation script."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.config import EnvConfig
from src.env.terminal_area_env import TerminalAreaEnv
from src.env.wrappers import FixedAirspaceWrapper
from src.baselines.random_policy import RandomPolicy
from src.baselines.heuristic_policy import HeuristicPolicy


def evaluate_policy(env: TerminalAreaEnv, policy, num_episodes: int = 10,
                    seed: int = 42) -> dict:
    """Evaluate a policy over multiple episodes and return aggregated metrics."""
    all_landed = []
    all_failed = []
    all_wait = []
    all_energy = []
    all_conflicts = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_landed = 0
        ep_failed = 0
        ep_conflicts = 0
        ep_reward = 0.0
        step = 0

        while True:
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
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
        episode_lengths.append(step)

    return {
        "avg_landed": float(np.mean(all_landed)),
        "avg_failed": float(np.mean(all_failed)),
        "avg_wait_time": float(np.mean(all_wait)),
        "avg_energy": float(np.mean(all_energy)),
        "avg_conflicts_per_ep": float(np.mean(all_conflicts)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "std_landed": float(np.std(all_landed)),
        "std_wait": float(np.std(all_wait)),
    }


def run_baseline_comparison(config: EnvConfig | None = None,
                            num_episodes: int = 20) -> dict[str, dict]:
    """Compare random baseline vs heuristic baseline."""
    if config is None:
        config = EnvConfig()
        config.num_uavs = 10
        config.sim.max_steps = 800

    rng = np.random.default_rng(42)
    random_policy = RandomPolicy(config, rng)
    heuristic_policy = HeuristicPolicy(config)

    results = {}
    for name, policy in [("random", random_policy), ("heuristic", heuristic_policy)]:
        print(f"\n=== Evaluating {name} policy ({num_episodes} episodes) ===")
        env = TerminalAreaEnv(config)
        t0 = time.time()
        metrics = evaluate_policy(env, policy, num_episodes=num_episodes)
        elapsed = time.time() - t0
        metrics["eval_time_s"] = elapsed
        results[name] = metrics
        _print_metrics(name, metrics)

    # Save results
    out_path = Path("results/baseline_comparison.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def _print_metrics(name: str, m: dict) -> None:
    print(f"  {name}: landed={m['avg_landed']:.1f}±{m.get('std_landed', 0):.1f}, "
          f"failed={m['avg_failed']:.1f}, "
          f"wait={m['avg_wait_time']:.1f}s, "
          f"energy={m['avg_energy']:.1f}, "
          f"conflicts={m['avg_conflicts_per_ep']:.1f}, "
          f"ep_len={m['avg_episode_length']:.1f}, "
          f"time={m['eval_time_s']:.1f}s")


if __name__ == "__main__":
    run_baseline_comparison()
