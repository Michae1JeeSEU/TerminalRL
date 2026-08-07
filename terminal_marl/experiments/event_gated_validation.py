"""Baseline and gate-behavior validation for event-gated HRL."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
from statistics import mean

from .evaluation_protocol import WindowedEvaluationProtocol, run_windowed_evaluation


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def config_for_policy(policy: str) -> str:
    if policy == "rule-only":
        return "fixed_rule"
    if policy == "event-rule":
        return "event_rule"
    if policy == "periodic-hrl":
        return "periodic_hrl"
    return "event_gated_hrl"


def run_validation(
    *,
    output_dir: str | Path,
    demands: tuple[str, ...] = ("low", "medium", "high", "bursty"),
    seeds: tuple[int, ...] = (10, 11, 12),
    policies: tuple[str, ...] = ("rule-only", "event-rule"),
    warm_up_seconds: float = 600.0,
    measurement_seconds: float = 1800.0,
    max_drain_seconds: float = 1800.0,
    max_workers: int = 1,
) -> dict:
    root = Path(output_dir)
    protocol = WindowedEvaluationProtocol(
        warm_up_seconds=warm_up_seconds,
        measurement_seconds=measurement_seconds,
        max_drain_seconds=max_drain_seconds,
        metrics_sampling_interval_seconds=20.0,
        decision_interval_seconds=180.0,
        reconfiguration_dwell_seconds=600.0,
        geometry_transition_seconds=20.0,
    )
    summaries: list[dict] = []
    trace_rows: list[dict] = []
    jobs = [
        {
            "policy": policy,
            "demand": demand,
            "seed": seed,
            "protocol": protocol,
        }
        for policy in policies
        for demand in demands
        for seed in seeds
    ]
    results: list[dict]
    if max_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(max_workers), len(jobs))) as pool:
            results = list(pool.map(_run_validation_job, jobs))
    else:
        results = [_run_validation_job(job) for job in jobs]
    for result in results:
        row = dict(result["summary"])
        row["policy"] = result["policy"]
        summaries.append(row)
        for sample in result["samples"]:
            trace = dict(sample)
            trace["policy"] = result["policy"]
            trace_rows.append(trace)
    _write_csv(root / "baseline_validation.csv", summaries)
    _write_csv(root / "gate_event_trace.csv", trace_rows)
    behavior_rows = summarize_gate_behavior(trace_rows, summaries)
    _write_csv(root / "gate_behavior_by_demand.csv", behavior_rows)
    behavior = {"rows": behavior_rows}
    (root / "gate_behavior_summary.json").write_text(
        json.dumps(behavior, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "gate_behavior_analysis.md").write_text(
        build_behavior_markdown(behavior_rows),
        encoding="utf-8",
    )
    return behavior


def _run_validation_job(job: dict) -> dict:
    policy = job["policy"]
    result = run_windowed_evaluation(
        config_for_policy(policy),
        protocol=job["protocol"],
        policy="event-rule" if policy == "event-rule" else "rule-only",
        demand_level=job["demand"],
        seed=job["seed"],
    )
    return {"policy": policy, **result}


def summarize_gate_behavior(trace_rows: list[dict], summaries: list[dict]) -> list[dict]:
    rows: list[dict] = []
    keys = sorted({(row["policy"], row["traffic_demand_level"]) for row in trace_rows})
    for policy, demand in keys:
        traces = [row for row in trace_rows if row["policy"] == policy and row["traffic_demand_level"] == demand and row["window"] == "measurement"]
        summary_group = [row for row in summaries if row["policy"] == policy and row["traffic_demand_level"] == demand]
        active = [float(row.get("geometry_gate_active", 0.0)) for row in traces]
        opportunities = [float(row.get("geometry_opportunity_count", 0.0)) for row in traces]
        changes = [float(row.get("applied_geometry_change_count", 0.0)) for row in traces]
        trigger_count = count_edges(active)
        measurement_hours = max(1e-9, (max(float(row["time"]) for row in traces) - min(float(row["time"]) for row in traces)) / 3600.0) if traces else 1.0
        rows.append(
            {
                "policy": policy,
                "traffic_demand_level": demand,
                "trigger_count": trigger_count,
                "triggers_per_hour": trigger_count / measurement_hours,
                "gate_open_fraction": mean(active) if active else 0.0,
                "mean_trigger_duration_seconds": mean_trigger_duration(active) * 20.0 if active else 0.0,
                "geometry_opportunity_count": max(opportunities) if opportunities else 0.0,
                "geometry_opportunities_per_hour": (max(opportunities) if opportunities else 0.0) / measurement_hours,
                "actual_geometry_changes_per_hour": (max(changes) if changes else 0.0) / measurement_hours,
                "nominal_recovery_count": max(float(row.get("nominal_recovery_count", 0.0)) for row in traces) if traces else 0.0,
                "completion": mean(float(row["final_completion_ratio"]) for row in summary_group) if summary_group else 0.0,
                "throughput": mean(float(row["throughput"]) for row in summary_group) if summary_group else 0.0,
                "delay": mean(float(row["mean_delay"]) for row in summary_group) if summary_group else 0.0,
                "backlog": mean(float(row["mean_backlog"]) for row in summary_group) if summary_group else 0.0,
                "conflict": mean(float(row["normalized_conflict_rate"]) for row in summary_group) if summary_group else 0.0,
            }
        )
    return rows


def count_edges(values: list[float]) -> int:
    count = 0
    previous = 0.0
    for value in values:
        if value > 0.5 and previous <= 0.5:
            count += 1
        previous = value
    return count


def mean_trigger_duration(values: list[float]) -> float:
    durations: list[int] = []
    current = 0
    for value in values:
        if value > 0.5:
            current += 1
        elif current:
            durations.append(current)
            current = 0
    if current:
        durations.append(current)
    return mean(durations) if durations else 0.0


def build_behavior_markdown(rows: list[dict]) -> str:
    lines = ["# Gate behavior analysis", ""]
    for row in rows:
        lines.append(
            f"- {row['policy']} / {row['traffic_demand_level']}: "
            f"gate_open_fraction={row['gate_open_fraction']:.3f}, "
            f"geometry_changes/hour={row['actual_geometry_changes_per_hour']:.3f}, "
            f"completion={row['completion']:.3f}, conflict={row['conflict']:.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12])
    parser.add_argument("--warm-up-seconds", type=float, default=600.0)
    parser.add_argument("--measurement-seconds", type=float, default=1800.0)
    parser.add_argument("--max-drain-seconds", type=float, default=1800.0)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            run_validation(
                output_dir=args.output_dir,
                seeds=tuple(args.seeds),
                warm_up_seconds=args.warm_up_seconds,
                measurement_seconds=args.measurement_seconds,
                max_drain_seconds=args.max_drain_seconds,
                max_workers=args.max_workers,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
