from terminal_marl.experiments.capacity_calibration import (
    run_capacity_calibration,
    run_capacity_case,
)


def test_capacity_case_reports_rule_only_stability_metrics() -> None:
    row = run_capacity_case(
        config_name="fixed_rule",
        arrival_interval_high_level_steps=10,
        seed=0,
        initial_uavs=2,
        high_level_steps=3,
    )

    assert row["arrival_interval_high_level_steps"] == 10.0
    assert row["spawned_count"] >= 2.0
    assert row["pair_opportunities"] >= 0.0
    assert row["normalized_conflict_rate"] >= 0.0
    assert row["capacity_stable"] in {0.0, 1.0}


def test_capacity_calibration_writes_csv_and_json(tmp_path) -> None:
    result = run_capacity_calibration(
        config_name="fixed_rule",
        interval_steps=[8, 10],
        seeds=[0],
        initial_uavs=2,
        high_level_steps=2,
        output_dir=tmp_path,
    )

    assert len(result["rows"]) == 2
    assert "recommended_demand_intervals" in result
    assert (tmp_path / "capacity_calibration.csv").is_file()
    assert (tmp_path / "capacity_calibration.json").is_file()
