from terminal_marl.experiments.debug_arrivals import run_debug_arrivals


def test_debug_arrivals_writes_counts(tmp_path) -> None:
    rows = run_debug_arrivals(
        config_name="dynamic_hrl",
        demand_level="medium",
        policy="rule-only",
        high_level_steps=2,
        output_dir=tmp_path,
    )
    assert rows
    assert max(row["total_spawned_count"] for row in rows) > 0
    assert max(row["active_count_max"] for row in rows) > 0
    assert (tmp_path / "debug_arrivals.csv").is_file()

