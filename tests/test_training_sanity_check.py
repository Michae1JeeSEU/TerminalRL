import importlib.util
import json

import pytest

from terminal_marl.training.train_high_level_ppo import train_high_level_ppo


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 unavailable")
def test_training_sanity_check_writes_summary(tmp_path) -> None:
    train_high_level_ppo(
        config_name="dynamic_hrl",
        seed=2,
        total_timesteps=8,
        output_dir=tmp_path,
        demand_level="medium",
        low_level_policy="rule-only",
        n_envs=1,
        n_steps=8,
        batch_size=8,
        sanity_check_baseline=True,
        traffic_check_steps=1,
        max_steps=2,
    )
    summary = json.loads((tmp_path / "sanity_check_summary.json").read_text(encoding="utf-8"))
    assert summary["total_spawned_count"] > 0.0
    assert summary["active_count_max"] > 0.0

