import importlib.util

import pytest

from terminal_marl.experiments import run_rollout
from terminal_marl.training.train_high_level_ppo import train_high_level_ppo


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 unavailable")
def test_ablation_runner_loads_trained_high_level_policy(tmp_path) -> None:
    train_high_level_ppo(
        config_name="dynamic_hrl", seed=1, total_timesteps=8,
        output_dir=tmp_path, demand_level="low", low_level_policy="rule-only",
    )
    metrics = run_rollout(
        "dynamic_hrl", policy="high-level-ppo", model_path=tmp_path / "model.zip",
        max_steps=1, traffic_demand_level="low",
    )
    assert metrics["policy"] == "high-level-ppo"


def test_ppo_policy_requires_model_path() -> None:
    with pytest.raises(ValueError, match="model_path"):
        run_rollout("dynamic_hrl", policy="high-level-ppo", max_steps=1)
