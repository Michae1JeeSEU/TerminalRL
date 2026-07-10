from pathlib import Path

import pytest

from terminal_marl.training.training_config import (
    PPOTrainingConfig,
    load_training_config,
    merge_training_overrides,
)


def test_load_all_server_training_presets() -> None:
    for name in [
        "ppo_smoke",
        "ppo_pilot_medium",
        "ppo_main_medium",
        "ppo_main_high",
        "ppo_main_bursty",
        "ppo_reward_sweep",
    ]:
        config = load_training_config(name)
        assert config.config == "dynamic_hrl"
        assert config.simulation_only is True
        assert config.device == "cpu"
        assert config.n_envs >= 1
        assert config.seed_list


def test_cli_overrides_yaml_values() -> None:
    base = PPOTrainingConfig(total_timesteps=100, n_envs=2, demand_level="medium")
    merged = merge_training_overrides(
        base,
        {"total_timesteps": 32, "n_envs": None, "demand_level": "high"},
    )
    assert merged.total_timesteps == 32
    assert merged.n_envs == 2
    assert merged.demand_level == "high"


def test_training_config_rejects_hardware_mode(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "config: crazyflie_lab\n"
        "demand_level: medium\n"
        "total_timesteps: 8\n"
        "n_envs: 1\n"
        "vec_env: dummy\n"
        "seed_list: [0]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="crazyflie_lab"):
        load_training_config(path)

