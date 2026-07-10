"""Aggregate multi-seed ablation metrics with normal-approximation 95% CIs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import argparse

import numpy as np


GROUP_FIELDS = ("config_name", "policy", "traffic_demand_level")
NON_METRICS = {*GROUP_FIELDS, "seed", "episode", "experiment_mode", "nominal_delay_mode"}


def aggregate_statistics(
    results_csv: str | Path,
    output_dir: str | Path,
) -> list[dict]:
    with Path(results_csv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)
    summary: list[dict] = []
    for key, group in groups.items():
        metrics = sorted(set(group[0]) - NON_METRICS)
        for metric in metrics:
            try:
                values = np.asarray([float(row[metric]) for row in group], dtype=float)
            except (ValueError, TypeError):
                continue
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            margin = 1.96 * std / math.sqrt(len(values)) if values.size else 0.0
            summary.append(
                {
                    **dict(zip(GROUP_FIELDS, key)),
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": std,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "ci95_low": float(np.mean(values) - margin),
                    "ci95_high": float(np.mean(values) + margin),
                    "n": len(values),
                    "number_of_seeds": len({row["seed"] for row in group}),
                }
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(summary[0]) if summary else [
        *GROUP_FIELDS, "metric", "mean", "std", "min", "max", "median",
        "ci95_low", "ci95_high", "n", "number_of_seeds",
    ]
    with (output / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    (output / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    aggregate_statistics(args.results_csv, args.output_dir)


if __name__ == "__main__":
    main()
