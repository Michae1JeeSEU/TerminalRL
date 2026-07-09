import csv
import json

from terminal_marl.experiments.aggregate import aggregate_statistics


def test_aggregate_outputs_mean_std_ci_and_seed_count(tmp_path) -> None:
    raw = tmp_path / "aggregate_results.csv"
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_name", "policy", "traffic_demand_level", "seed", "episode", "throughput"])
        writer.writeheader()
        writer.writerows([
            {"config_name": "dynamic_hrl", "policy": "rule-only", "traffic_demand_level": "high", "seed": 0, "episode": 0, "throughput": 1},
            {"config_name": "dynamic_hrl", "policy": "rule-only", "traffic_demand_level": "high", "seed": 1, "episode": 0, "throughput": 3},
        ])
    rows = aggregate_statistics(raw, tmp_path)
    row = next(item for item in rows if item["metric"] == "throughput")
    assert row["mean"] == 2
    assert row["std"] > 0
    assert row["ci95_low"] < row["mean"] < row["ci95_high"]
    assert row["number_of_seeds"] == 2
    assert (tmp_path / "aggregate_summary.csv").is_file()
    assert json.loads((tmp_path / "aggregate_summary.json").read_text())
