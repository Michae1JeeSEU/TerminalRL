"""Generate one-file-per-figure paper experiment plot skeletons."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_METRICS = (
    "throughput",
    "average_delay",
    "conflict_count",
    "safety_override_count",
)


def _rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_summary(summary_csv: str | Path, output_dir: str | Path) -> list[Path]:
    rows = _rows(summary_csv)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    demand_order = ["low", "medium", "high", "bursty"]
    for metric in SUMMARY_METRICS:
        selected = [row for row in rows if row["metric"] == metric]
        if not selected:
            continue
        plt.figure()
        for label in sorted({f"{row['config_name']}:{row['policy']}" for row in selected}):
            subset = [
                row for row in selected
                if f"{row['config_name']}:{row['policy']}" == label
            ]
            subset.sort(key=lambda row: demand_order.index(row["traffic_demand_level"]))
            plt.plot(
                [row["traffic_demand_level"] for row in subset],
                [float(row["mean"]) for row in subset],
                marker="o",
                label=label,
            )
        plt.xlabel("traffic demand")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        path = output / f"{metric}_vs_demand.png"
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    return paths


def plot_rollout_traces(
    airspace_csv: str | Path,
    step_csv: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    airspace = _rows(airspace_csv)
    steps = _rows(step_csv)
    paths: list[Path] = []
    if airspace:
        plt.figure()
        for field in ("outer_radius", "inner_radius", "outer_height", "inner_height"):
            plt.plot([float(row["time"]) for row in airspace], [float(row[field]) for row in airspace], label=field)
        plt.xlabel("time")
        plt.ylabel("airspace parameter")
        plt.legend()
        plt.tight_layout()
        path = output / "airspace_trace.png"
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    if steps:
        plt.figure()
        plt.plot([float(row["time"]) for row in steps], [float(row["landed_count"]) for row in steps])
        plt.xlabel("time")
        plt.ylabel("landed count")
        plt.tight_layout()
        path = output / "landed_count_over_time.png"
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    return paths
