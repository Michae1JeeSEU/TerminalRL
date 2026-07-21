import importlib.util
import json

import pytest

from terminal_marl.training.train_high_level_ppo import train_high_level_ppo


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
