import csv

from terminal_marl.experiments.plot_results import plot_rollout_traces, plot_summary


def test_plot_pipeline_generates_individual_png_files(tmp_path) -> None:
    summary = tmp_path / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_name", "policy", "traffic_demand_level", "metric", "mean"])
        writer.writeheader()
        for metric in ["throughput", "average_delay", "conflict_count", "safety_override_count"]:
            writer.writerow({"config_name": "dynamic_hrl", "policy": "rule-only", "traffic_demand_level": "low", "metric": metric, "mean": 1})
    figures = plot_summary(summary, tmp_path / "figures")
    assert len(figures) == 4 and all(path.is_file() for path in figures)

    airspace = tmp_path / "airspace.csv"
    step = tmp_path / "step.csv"
    airspace.write_text("time,outer_radius,inner_radius,outer_height,inner_height\n0,4,2,4,2\n", encoding="utf-8")
    step.write_text("time,landed_count\n0,0\n1,1\n", encoding="utf-8")
    traces = plot_rollout_traces(airspace, step, tmp_path / "figures")
    assert len(traces) == 2 and all(path.is_file() for path in traces)
