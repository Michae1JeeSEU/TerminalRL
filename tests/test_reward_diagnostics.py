from terminal_marl.experiments.reward_diagnostics import run_reward_diagnostics


def test_reward_diagnostics_outputs_component_statistics(tmp_path) -> None:
    rows = run_reward_diagnostics(
        policies=["rule-only", "dynamic-rule", "high-level-random", "untrained-ppo"],
        demand_levels=["low"],
        seeds=[0],
        episodes=1,
        high_level_steps=1,
        output_dir=tmp_path,
    )
    assert rows
    assert {row["policy"] for row in rows} >= {
        "rule-only",
        "dynamic-rule",
        "high-level-random",
        "untrained-ppo",
    }
    assert any(row["component"] == "total_reward" for row in rows)
    assert (tmp_path / "reward_diagnostics.csv").is_file()
    assert (tmp_path / "reward_diagnostics.json").is_file()

