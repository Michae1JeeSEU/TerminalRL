import csv
import importlib.util
import json

import pytest

from terminal_marl.training.train_high_level_ppo import train_high_level_ppo


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 unavailable")
def test_final_eval_logs_interval_count_fields(tmp_path) -> None:
    train_high_level_ppo(
        config_name="dynamic_hrl",
        seed=4,
        total_timesteps=8,
        output_dir=tmp_path,
        demand_level="medium",
        low_level_policy="rule-only",
        n_envs=1,
        n_steps=8,
        batch_size=8,
        max_steps=2,
    )
    with (tmp_path / "final_eval.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    required = {
        "active_count_start",
        "active_count_mean",
        "active_count_max",
        "active_count_end",
        "active_count",
        "spawned_count_delta",
        "landed_count_delta",
        "total_spawned_count",
        "total_landed_count",
        "safety_override_delta",
        "total_safety_override_count",
    }
    assert required <= set(rows[0])
    assert max(float(row["active_count_max"]) for row in rows) > 0.0
    summary = json.loads((tmp_path / "eval_summary.json").read_text(encoding="utf-8"))
    assert summary["max_active_count"] > 0.0
    assert summary["total_spawned_count"] > 0.0

