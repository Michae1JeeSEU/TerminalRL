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
        "ppo_debug_local",
    ]:
        config = load_training_config(name)
        assert config.config == "dynamic_hrl"
        assert config.simulation_only is True
        assert config.device == "cpu"
        assert config.n_envs >= 1
        assert config.seed_list


def test_load_event_gated_training_presets_with_demand_mixture() -> None:
    for name, expected_config in [
        ("event_gated_smoke", "event_gated_hrl"),
        ("event_gated_stage05_5k", "event_gated_hrl"),
        ("periodic_stage05_5k", "periodic_hrl"),
        ("event_gated_stage1_20k", "event_gated_hrl"),
        ("periodic_stage1_20k", "periodic_hrl"),
    ]:
        config = load_training_config(name)
        assert config.config == expected_config
        assert config.demand_level == "medium"
        assert config.demand_mixture == {
            "medium": pytest.approx(0.4),
            "high": pytest.approx(0.3),
            "bursty": pytest.approx(0.3),
        }
        assert config.high_level_interval == 180
        assert config.max_steps == 3600
        assert config.traffic_check_steps >= 20
        assert config.eval_high_level_steps >= 20


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


def test_training_config_rejects_short_high_level_horizons(tmp_path: Path) -> None:
    path = tmp_path / "short_eval.yaml"
    path.write_text(
        "config: dynamic_hrl\n"
        "demand_level: medium\n"
        "total_timesteps: 8\n"
        "n_envs: 1\n"
        "vec_env: dummy\n"
        "seed_list: [0]\n"
        "traffic_check_steps: 8\n"
        "eval_high_level_steps: 8\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="traffic_check_steps"):
        load_training_config(path)
