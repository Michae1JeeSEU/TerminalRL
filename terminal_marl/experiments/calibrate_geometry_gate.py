"""Calibrate event-gate trigger thresholds from fixed-geometry rule rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from ..config import load_experiment_config
from ..control.geometry_event_gate import GateConfig, GateMode, GeometryEventGate
from ..envs.parallel_env import TerminalParallelEnv
from .ablation_runner import apply_demand_profile
from .evaluation_protocol import set_runtime_arrivals_enabled


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_pressure_trace(
    *,
    demand: str,
    seed: int,
    warm_up_seconds: float,
    measurement_seconds: float,
    max_drain_seconds: float,
    sample_seconds: float = 20.0,
) -> list[dict]:
    config = apply_demand_profile(load_experiment_config("fixed_rule"), demand)
    config.max_steps = int(warm_up_seconds + measurement_seconds + max_drain_seconds + 5)
    env = TerminalParallelEnv(config)
    env.reset(seed=seed)
    probe = GeometryEventGate(GateConfig(metrics_sampling_interval_seconds=sample_seconds), mode=GateMode.EVENT)
    rows: list[dict] = []
    next_sample = 0.0
    measurement_end = warm_up_seconds + measurement_seconds
    while env.time < warm_up_seconds + measurement_seconds + max_drain_seconds:
        if env.time >= measurement_end:
            set_runtime_arrivals_enabled(env, False)
        env.step({})
        if env.time + 1e-9 >= next_sample:
            gate_metrics = env._gate_metrics()
            score = probe.trigger_score(gate_metrics)
            window = "warm_up" if env.time < warm_up_seconds else ("measurement" if env.time < measurement_end else "drain")
            rows.append(
                {
                    "demand": demand,
                    "seed": seed,
                    "time": float(env.time),
                    "window": window,
                    "trigger_score": score,
                    "normalized_congestion": gate_metrics.normalized_congestion,
                    "approach_backlog_ratio": gate_metrics.approach_backlog_ratio,
                    "occupancy_pressure": gate_metrics.occupancy_pressure,
                    "backlog_growth": gate_metrics.backlog_growth,
                    "normalized_conflict_rate": gate_metrics.normalized_conflict_rate,
                    "recent_arrivals": gate_metrics.recent_arrivals,
                    "recent_landings": gate_metrics.recent_landings,
                    "arrival_service_imbalance": gate_metrics.arrival_service_imbalance,
                    "active_capacity_ratio": gate_metrics.active_capacity_ratio,
                }
            )
            next_sample += sample_seconds
        if env.time >= measurement_end and not any(not aircraft.finished for aircraft in env.aircraft):
            break
    return rows


def evaluate_candidate(rows: list[dict], *, theta_on: float, theta_off: float, k_on: int, k_off: int) -> dict:
    by_run: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        if row["window"] == "measurement":
            by_run.setdefault((row["demand"], int(row["seed"])), []).append(row)
    trigger_counts: list[int] = []
    gate_open_fractions: list[float] = []
    detection_delays: list[float] = []
    false_triggers = 0
    missed_sustained = 0
    for (demand, _seed), trace in by_run.items():
        gate = GeometryEventGate(
            GateConfig(theta_on=theta_on, theta_off=theta_off, k_on=k_on, k_off=k_off),
            mode=GateMode.EVENT,
        )
        run_trigger_count = 0
        open_samples = 0
        first_pressure_time = None
        first_trigger_time = None
        for row in trace:
            score = float(row["trigger_score"])
            if score >= theta_on and first_pressure_time is None:
                first_pressure_time = float(row["time"])
            fake_metrics = {
                "normalized_congestion": score,
                "approach_backlog_ratio": score,
                "occupancy_pressure": score,
                "backlog_growth": 0.0,
                "normalized_conflict_rate": 0.0,
                "recent_arrivals": 0.0,
                "recent_landings": 0.0,
                "arrival_service_imbalance": score,
                "active_capacity_ratio": score,
            }
            before = gate.trigger_active
            gate.update(gate_metrics_from_score(fake_metrics), time_seconds=float(row["time"]), force_sample=True)
            if gate.trigger_active:
                open_samples += 1
                if not before:
                    run_trigger_count += 1
                    first_trigger_time = first_trigger_time or float(row["time"])
        trigger_counts.append(run_trigger_count)
        gate_open_fractions.append(open_samples / max(1, len(trace)))
        if first_pressure_time is not None and first_trigger_time is not None:
            detection_delays.append(first_trigger_time - first_pressure_time)
        if demand in {"low", "medium"}:
            false_triggers += run_trigger_count
        if demand in {"high", "bursty"} and first_pressure_time is not None and first_trigger_time is None:
            missed_sustained += 1
    return {
        "theta_on": theta_on,
        "theta_off": theta_off,
        "k_on": k_on,
        "k_off": k_off,
        "false_trigger_rate": false_triggers / max(1, len([k for k in by_run if k[0] in {"low", "medium"}])),
        "missed_sustained_pressure_events": missed_sustained,
        "mean_detection_delay": mean(detection_delays) if detection_delays else 0.0,
        "trigger_count_per_run": mean(trigger_counts) if trigger_counts else 0.0,
        "gate_open_fraction": mean(gate_open_fractions) if gate_open_fractions else 0.0,
        "geometry_opportunity_count_per_hour": (mean(gate_open_fractions) if gate_open_fractions else 0.0) * 180.0,
    }


def gate_metrics_from_score(values: dict) -> object:
    from ..control.geometry_event_gate import GateMetrics

    return GateMetrics.from_mapping(values)


def calibrate(
    *,
    output_dir: str | Path,
    warm_up_seconds: float = 600.0,
    measurement_seconds: float = 1800.0,
    max_drain_seconds: float = 1800.0,
    seeds: tuple[int, ...] = (0, 1, 2),
    demands: tuple[str, ...] = ("low", "medium", "high", "bursty"),
) -> dict:
    root = Path(output_dir)
    rows: list[dict] = []
    for demand in demands:
        for seed in seeds:
            rows.extend(
                collect_pressure_trace(
                    demand=demand,
                    seed=seed,
                    warm_up_seconds=warm_up_seconds,
                    measurement_seconds=measurement_seconds,
                    max_drain_seconds=max_drain_seconds,
                )
            )
    _write_csv(root / "gate_calibration_raw.csv", rows)
    scores = np.asarray([float(row["trigger_score"]) for row in rows if row["window"] == "measurement"], dtype=float)
    if scores.size == 0:
        candidates_on = [0.55]
    else:
        candidates_on = sorted(set(float(np.quantile(scores, q)) for q in (0.60, 0.70, 0.80)))
    summary: list[dict] = []
    for theta_on in candidates_on:
        theta_off = max(0.05, theta_on * 0.65)
        for k_on in (3, 4, 5):
            for k_off in (6, 9, 15):
                summary.append(evaluate_candidate(rows, theta_on=theta_on, theta_off=theta_off, k_on=k_on, k_off=k_off))
    summary.sort(key=lambda row: (row["false_trigger_rate"], row["missed_sustained_pressure_events"], row["mean_detection_delay"]))
    _write_csv(root / "gate_calibration_summary.csv", summary)
    best = summary[0] if summary else {
        "theta_on": 0.55,
        "theta_off": 0.35,
        "k_on": 3,
        "k_off": 9,
    }
    recommendation = {
        "theta_on": float(best["theta_on"]),
        "theta_off": float(best["theta_off"]),
        "k_on": int(best["k_on"]),
        "k_off": int(best["k_off"]),
        "recovery_k": int(best["k_off"]),
        "source": "fixed-geometry rule-only pressure distribution",
        "score_quantiles": {
            "p50": float(np.quantile(scores, 0.50)) if scores.size else 0.0,
            "p70": float(np.quantile(scores, 0.70)) if scores.size else 0.0,
            "p90": float(np.quantile(scores, 0.90)) if scores.size else 0.0,
        },
        "best_candidate": best,
    }
    (root / "gate_threshold_recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "gate_threshold_recommendation.md").write_text(
        "\n".join(
            [
                "# Geometry gate threshold recommendation",
                "",
                f"- theta_on: {recommendation['theta_on']:.4f}",
                f"- theta_off: {recommendation['theta_off']:.4f}",
                f"- K_on: {recommendation['k_on']}",
                f"- K_off: {recommendation['k_off']}",
                "- Thresholds are generated from observed pressure-score quantiles; not hand-picked constants.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warm-up-seconds", type=float, default=600.0)
    parser.add_argument("--measurement-seconds", type=float, default=1800.0)
    parser.add_argument("--max-drain-seconds", type=float, default=1800.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    print(
        json.dumps(
            calibrate(
                output_dir=args.output_dir,
                warm_up_seconds=args.warm_up_seconds,
                measurement_seconds=args.measurement_seconds,
                max_drain_seconds=args.max_drain_seconds,
                seeds=tuple(args.seeds),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
