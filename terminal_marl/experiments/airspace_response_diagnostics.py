"""One-shot dynamic-airspace response diagnostics.

This script probes whether ring radius / speed / headway changes have a
measurable terminal-flow response within 20, 60, 100, 120, or 180 seconds.
It is simulation-only and never starts PPO training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.dynamic_airspace import DynamicAirspace
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import DynamicRuleHighLevelPolicy
from .evaluation_protocol import (
    WindowedEvaluationProtocol,
    collect_sample,
    configure_high_level_timing,
    count_phase_progression,
    occupancy_snapshot,
    phase_snapshot,
    resolve_config,
)
from .ablation_runner import apply_demand_profile


ACTION_LIBRARY: dict[str, np.ndarray] = {
    "no_change": np.zeros(DynamicAirspace.ACTION_SIZE, dtype=np.float32),
    # The normalized dimensions are bounded increments:
    # [outer_r, inner_r, outer_h, inner_h, outer_v, inner_v, ring_headway, final_headway]
    "outer_radius_plus_5pct": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "inner_radius_plus_5pct": np.asarray([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "ring_speed_plus_5pct": np.asarray([0.0, 0.0, 0.0, 0.0, 0.7, 0.5, 0.0, 0.0], dtype=np.float32),
    "ring_headway_plus_10pct": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0], dtype=np.float32),
    "final_headway_minus_10pct": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    "safe_combo": np.asarray([0.35, 0.15, 0.0, 0.0, 0.35, 0.25, -0.35, -0.30], dtype=np.float32),
}


def _response_signal(row: dict[str, float | str]) -> float:
    """Compact traffic-flow response signal used only for delay estimation."""

    return float(
        row["landed_delta_cumulative"]
        + 0.5 * row["approach_to_outer"]
        + 0.75 * row["outer_to_inner"]
        + row["inner_to_final"]
        + 0.25 * row["release_approvals"]
        - 0.20 * row["approach_backlog"]
        - 0.50 * row["normalized_conflict_rate"]
    )


def _summarize_response(
    action_name: str,
    demand_level: str,
    seed: int,
    rows: list[dict[str, float | str]],
    control_rows: list[dict[str, float | str]],
    action_time: float,
) -> dict[str, float | str]:
    by_time = {float(row["seconds_after_action"]): row for row in rows}
    control_by_time = {float(row["seconds_after_action"]): row for row in control_rows}
    response_series: list[tuple[float, float]] = []
    for seconds, row in sorted(by_time.items()):
        if seconds < 0.0 or seconds not in control_by_time:
            continue
        response = _response_signal(row) - _response_signal(control_by_time[seconds])
        response_series.append((seconds, response))
    baseline_eps = 0.25
    first = next((seconds for seconds, value in response_series if abs(value) > baseline_eps), None)
    if response_series:
        peak_seconds, peak_value = max(response_series, key=lambda item: abs(item[1]))
    else:
        peak_seconds, peak_value = 0.0, 0.0
    settling = 0.0
    if response_series:
        trailing_threshold = max(0.25, 0.10 * abs(peak_value))
        for i, (seconds, _value) in enumerate(response_series):
            tail = [abs(value) for _, value in response_series[i:]]
            if tail and max(tail) <= trailing_threshold:
                settling = seconds
                break
        if settling == 0.0:
            settling = response_series[-1][0]
    recommended_bucket = 20.0
    if first is None or settling > 120.0:
        recommended_bucket = 180.0
    elif settling > 100.0:
        recommended_bucket = 120.0
    elif settling > 60.0:
        recommended_bucket = 100.0
    elif settling > 20.0:
        recommended_bucket = 60.0
    return {
        "action_name": action_name,
        "traffic_demand_level": demand_level,
        "seed": float(seed),
        "action_time": float(action_time),
        "first_response_seconds": float(first if first is not None else 0.0),
        "peak_response_seconds": float(peak_seconds),
        "peak_response_signal": float(peak_value),
        "settling_time_seconds": float(settling),
        "reliable_within_20s": float(first is not None and settling <= 20.0),
        "recommended_decision_interval_seconds": float(recommended_bucket),
    }


def run_response_diagnostic(
    *,
    config: str = "dynamic_hrl",
    demand_level: str = "medium",
    seed: int = 0,
    warm_up_seconds: float = 600.0,
    response_seconds: float = 600.0,
    sampling_interval_seconds: float = 20.0,
) -> dict[str, Any]:
    """Run all one-shot probes for one demand/seed combination."""

    protocol = WindowedEvaluationProtocol(
        warm_up_seconds=warm_up_seconds,
        measurement_seconds=response_seconds,
        max_drain_seconds=0.0,
        metrics_sampling_interval_seconds=sampling_interval_seconds,
        decision_interval_seconds=sampling_interval_seconds,
        reconfiguration_dwell_seconds=response_seconds,
        geometry_transition_seconds=20.0,
    )

    traces: list[dict[str, float | str]] = []
    summaries: list[dict[str, float | str]] = []
    control_rows: list[dict[str, float | str]] = []
    for action_name, one_shot_action in ACTION_LIBRARY.items():
        config_obj = configure_high_level_timing(
            apply_demand_profile(resolve_config(config), demand_level),
            protocol,
        )
        config_obj.max_steps = int(warm_up_seconds + response_seconds + 5)
        env = TerminalParallelEnv(config_obj)
        env.reset(seed=seed)
        policy = DynamicRuleHighLevelPolicy()
        transitions = {
            "approach_to_outer": 0,
            "outer_to_inner": 0,
            "inner_to_final": 0,
            "final_to_land": 0,
        }
        sampling_steps = max(1, int(round(sampling_interval_seconds / env.config.dt)))
        next_sample = 0
        action_applied = False
        start_landed = 0.0
        while env.time < warm_up_seconds + response_seconds:
            high_level_action = None
            if env.should_update_high_level():
                if env.time >= warm_up_seconds and not action_applied:
                    high_level_action = one_shot_action.copy()
                    action_applied = True
                    start_landed = float(env.metrics()["landed_count"])
                elif env.time < warm_up_seconds:
                    high_level_action = policy.act(env)
                else:
                    high_level_action = np.zeros(DynamicAirspace.ACTION_SIZE, dtype=np.float32)

            previous = phase_snapshot(env)
            env.step({}, high_level_action=high_level_action)
            step_counts = count_phase_progression(previous, env)
            for key, value in step_counts.items():
                transitions[key] += value

            if env.step_count >= next_sample:
                row = collect_sample(
                    env,
                    policy="rule-only-one-shot-airspace",
                    demand_level=demand_level,
                    seed=seed,
                    window="response" if env.time >= warm_up_seconds else "warm_up",
                    decision_interval_seconds=sampling_interval_seconds,
                    action=one_shot_action,
                    action_applied=bool(action_applied and env.time <= warm_up_seconds + env.config.dt),
                    transition_counts=transitions,
                    measurement_start={"time": warm_up_seconds, "landed_count": start_landed},
                )
                row["action_name"] = action_name
                row["action_time"] = float(warm_up_seconds)
                row["seconds_after_action"] = float(env.time - warm_up_seconds)
                row["landed_delta_cumulative"] = float(env.metrics()["landed_count"] - start_landed)
                row.update(occupancy_snapshot(env))
                traces.append(row)
                next_sample += sampling_steps

        rows = [row for row in traces if row["action_name"] == action_name]
        if action_name == "no_change":
            control_rows = rows
        summaries.append(
            _summarize_response(
                action_name,
                demand_level,
                seed,
                rows,
                control_rows if control_rows else rows,
                warm_up_seconds,
            )
        )
    return {"trace": traces, "summary": summaries}


def write_response_outputs(result: dict[str, Any], output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    trace_path = root / "response_trace.csv"
    summary_path = root / "response_summary.csv"
    json_path = root / "response_summary.json"
    trace = result["trace"]
    summaries = result["summary"]
    if trace:
        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in trace for key in row}))
            writer.writeheader()
            writer.writerows(trace)
    if summaries:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summaries for key in row}))
            writer.writeheader()
            writer.writerows(summaries)
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dynamic_hrl")
    parser.add_argument("--demand-levels", nargs="+", default=["medium", "high"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warm-up-seconds", type=float, default=600.0)
    parser.add_argument("--response-seconds", type=float, default=600.0)
    parser.add_argument("--sampling-interval-seconds", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    combined = {"trace": [], "summary": []}
    for demand in args.demand_levels:
        result = run_response_diagnostic(
            config=args.config,
            demand_level=demand,
            seed=args.seed,
            warm_up_seconds=args.warm_up_seconds,
            response_seconds=args.response_seconds,
            sampling_interval_seconds=args.sampling_interval_seconds,
        )
        combined["trace"].extend(result["trace"])
        combined["summary"].extend(result["summary"])
    write_response_outputs(combined, args.output_dir)


if __name__ == "__main__":
    main()
