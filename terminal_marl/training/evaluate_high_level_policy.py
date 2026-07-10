"""Evaluate a trained high-level PPO policy across seeds and demand levels."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Iterable

from ..experiments import run_sweep
from ..experiments.plot_results import plot_summary
from .train_high_level_ppo import require_sb3


def evaluate_high_level_policy(
    *,
    model_path: str | Path,
    configs: Iterable[str] = ("dynamic_hrl",),
    seeds: Iterable[int] = (0,),
    demand_levels: Iterable[str] = ("medium",),
    episodes: int = 1,
    output_dir: str | Path = "runs/high_level_ppo_eval",
    max_steps: int | None = None,
    nominal_delay_mode: str = "analytic",
) -> dict:
    """Run simulation-only PPO evaluation and create aggregate plots."""

    if importlib.util.find_spec("stable_baselines3") is None:
        require_sb3()
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(model)
    root = Path(output_dir)
    results = run_sweep(
        configs,
        policies=("high-level-ppo",),
        seeds=seeds,
        demand_levels=demand_levels,
        episodes=episodes,
        output_dir=root,
        max_steps=max_steps,
        model_path=model,
        nominal_delay_mode=nominal_delay_mode,
    )
    figures_dir = root / "figures"
    summary_csv = root / "aggregate_summary.csv"
    figures = plot_summary(summary_csv, figures_dir) if summary_csv.is_file() else []
    summary = {
        "model_path": str(model),
        "configs": list(configs),
        "seeds": [int(seed) for seed in seeds],
        "demand_levels": list(demand_levels),
        "episodes": int(episodes),
        "result_count": len(results),
        "aggregate_results": str(root / "aggregate_results.csv"),
        "aggregate_summary": str(summary_csv),
        "figures": [str(path) for path in figures],
    }
    (root / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=["dynamic_hrl"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--demand-levels",
        nargs="+",
        choices=["low", "medium", "high", "bursty"],
        default=["medium"],
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/high_level_ppo_eval"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--nominal-delay-mode",
        choices=["analytic", "rule-only-reference"],
        default="analytic",
    )
    args = parser.parse_args()
    summary = evaluate_high_level_policy(
        model_path=args.model_path,
        configs=args.configs,
        seeds=args.seeds,
        demand_levels=args.demand_levels,
        episodes=args.episodes,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        nominal_delay_mode=args.nominal_delay_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

