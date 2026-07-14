import sys

from terminal_marl.experiments.reward_diagnostics import main, run_reward_diagnostics


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


def test_reward_diagnostics_accepts_config_and_single_demand_level(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reward_diagnostics",
            "--config",
            "dynamic_hrl",
            "--demand-level",
            "medium",
            "--policies",
            "rule-only",
            "--episodes",
            "1",
            "--high-level-steps",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()
    text = (tmp_path / "reward_diagnostics.csv").read_text(encoding="utf-8")
    assert "dynamic_hrl" in text
    assert "medium" in text


def test_reward_diagnostics_accepts_multiple_demand_levels(tmp_path) -> None:
    rows = run_reward_diagnostics(
        config_name="dynamic_hrl",
        policies=["rule-only"],
        demand_levels=["medium", "high"],
        seeds=[0],
        episodes=1,
        high_level_steps=1,
        output_dir=tmp_path,
    )
    assert {row["traffic_demand_level"] for row in rows} == {"medium", "high"}
