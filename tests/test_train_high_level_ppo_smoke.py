import importlib.util
import json

import pytest

from terminal_marl.training.train_high_level_ppo import (
    ScenarioMixtureHighLevelWrapper,
    _select_exact_rollout_n_steps,
    train_high_level_ppo,
)


def test_select_exact_rollout_steps_prevents_budget_overshoot() -> None:
    assert (
        _select_exact_rollout_n_steps(
            requested_n_steps=128,
            total_timesteps=20_000,
            n_envs=10,
        )
        == 125
    )


def test_select_exact_rollout_steps_rejects_non_divisible_budget() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _select_exact_rollout_n_steps(
            requested_n_steps=128,
            total_timesteps=20_001,
            n_envs=10,
        )


def test_scenario_mixture_wrapper_samples_episode_demand_levels() -> None:
    wrapper = ScenarioMixtureHighLevelWrapper(
        config_name="event_gated_hrl",
        demand_mixture={"medium": 0.4, "high": 0.3, "bursty": 0.3},
        low_level_policy="rule-only",
        seed=2,
        high_level_interval=180,
        max_steps=3600,
    )

    seen = set()
    for episode_seed in range(12):
        observation, info = wrapper.reset(seed=episode_seed)
        seen.add(info["demand_level"])
        assert observation.shape == wrapper.observation_space.shape
        assert info["demand_mixture"] == {
            "medium": pytest.approx(0.4),
            "high": pytest.approx(0.3),
            "bursty": pytest.approx(0.3),
        }

    assert seen <= {"medium", "high", "bursty"}
    assert len(seen) >= 2
    assert sum(wrapper.scenario_counts.values()) == 12


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 unavailable")
def test_tiny_high_level_ppo_training_writes_artifacts(tmp_path) -> None:
    train_high_level_ppo(
        config_name="dynamic_hrl",
        seed=0,
        total_timesteps=8,
        output_dir=tmp_path,
        demand_level="low",
        low_level_policy="rule-only",
    )
    for name in [
        "model.zip",
        "training_config.json",
        "training_progress.csv",
        "eval_summary.json",
        "config_snapshot.json",
    ]:
        assert (tmp_path / name).is_file()
    summary = json.loads((tmp_path / "eval_summary.json").read_text(encoding="utf-8"))
    assert summary["eval_high_level_steps"] >= 20
    assert max(summary["eval_lengths"]) >= 20
    config = json.loads((tmp_path / "training_config.json").read_text(encoding="utf-8"))
    assert config["exact_timestep_budget"] is True
    assert 8 % config["rollout_size"] == 0
