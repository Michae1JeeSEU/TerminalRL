import importlib.util

import pytest

from terminal_marl.training.train_high_level_ppo import train_high_level_ppo


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 unavailable")
def test_training_fail_fast_rejects_insufficient_traffic(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Insufficient traffic"):
        train_high_level_ppo(
            config_name="dynamic_hrl",
            seed=0,
            total_timesteps=8,
            output_dir=tmp_path,
            demand_level="medium",
            low_level_policy="rule-only",
            n_envs=1,
            n_steps=8,
            batch_size=8,
            fail_if_no_traffic=True,
            min_spawned_before_training_check=999,
            traffic_check_steps=1,
        )
    assert (tmp_path / "traffic_check_summary.json").is_file()

