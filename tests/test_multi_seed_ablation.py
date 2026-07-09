"""Tests for multi-seed, multi-demand sweep output."""

import csv

from terminal_marl.experiments import run_sweep


def test_sweep_runs_every_combination_and_writes_aggregate(tmp_path) -> None:
    results = run_sweep(
        ["fixed_rule"],
        policies=["dry-run"],
        seeds=[0, 1],
        demand_levels=["low", "high"],
        episodes=2,
        output_dir=tmp_path,
        max_steps=1,
    )
    assert len(results) == 8
    aggregate = tmp_path / "aggregate_results.csv"
    assert aggregate.is_file()
    with aggregate.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {row["traffic_demand_level"] for row in rows} == {"low", "high"}


def test_each_combination_has_separate_episode_log(tmp_path) -> None:
    run_sweep(
        ["dynamic_rule"],
        policies=["high-level-static"],
        seeds=[3],
        demand_levels=["medium"],
        episodes=2,
        output_dir=tmp_path,
        max_steps=1,
    )
    summaries = list(tmp_path.rglob("episode_summary.json"))
    assert len(summaries) == 2
    assert summaries[0].parent != summaries[1].parent
