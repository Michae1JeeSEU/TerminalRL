import importlib.util

import pytest

from terminal_marl.training.evaluate_high_level_policy import evaluate_high_level_policy
from terminal_marl.training.train_high_level_ppo import train_high_level_ppo


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 unavailable")
def test_evaluate_high_level_policy_writes_aggregate_and_figures(tmp_path) -> None:
    train_dir = tmp_path / "train"
    train_high_level_ppo(
        config_name="dynamic_hrl",
        seed=3,
        total_timesteps=8,
        output_dir=train_dir,
        demand_level="low",
        low_level_policy="rule-only",
        n_envs=1,
        n_steps=8,
        batch_size=8,
        max_steps=1,
    )
    eval_dir = tmp_path / "eval"
    summary = evaluate_high_level_policy(
        model_path=train_dir / "model.zip",
        seeds=[0],
        demand_levels=["low"],
        episodes=1,
        output_dir=eval_dir,
        max_steps=1,
    )
    assert summary["result_count"] == 1
    assert (eval_dir / "aggregate_results.csv").is_file()
    assert (eval_dir / "aggregate_summary.csv").is_file()
    assert (eval_dir / "evaluation_summary.json").is_file()

