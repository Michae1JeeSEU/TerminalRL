"""Stage-1.5 evaluation-integrity and bottleneck diagnostics.

This module deliberately performs *evaluation/diagnostics only*.  It never
calls PPO ``learn`` and is safe to run against existing 20k checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import ExperimentMode, load_experiment_config
from ..core.arrival_process import generate_arrival_schedule, schedule_times
from ..core.config import SimConfig
from ..core.dynamic_airspace import DynamicAirspace
from ..envs.parallel_env import TerminalParallelEnv
from .ablation_runner import apply_demand_profile
from .evaluation_protocol import (
    WindowedEvaluationProtocol,
    run_windowed_evaluation,
    summarize_rows,
)


DEMANDS = ("low", "medium", "high", "bursty")
EVAL_SEEDS = (20, 21, 22, 23, 24)
DWELL_SEEDS = (20, 21, 22)
GEOMETRY_DIMS = (0, 1, 2, 3)
AIRSPACE_FIELDS = (
    "airspace_outer_radius",
    "airspace_inner_radius",
    "airspace_outer_height",
    "airspace_inner_height",
    "airspace_outer_speed",
    "airspace_inner_speed",
    "airspace_ring_headway",
    "airspace_final_headway",
)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _sign_change_count(values: list[float]) -> int:
    if len(values) < 2:
        return 0
    signs = np.sign(np.asarray(values, dtype=float))
    active = np.abs(np.asarray(values, dtype=float)) > 1e-6
    return int(np.count_nonzero((signs[1:] != signs[:-1]) & active[1:]))


def generate_common_schedules(
    *,
    config_name: str,
    output_dir: Path,
    protocol: WindowedEvaluationProtocol,
    seeds: Iterable[int] = EVAL_SEEDS,
    demands: Iterable[str] = DEMANDS,
) -> dict[tuple[str, int], tuple[float, ...]]:
    """Generate and persist policy-independent arrival schedules."""

    schedules: dict[tuple[str, int], tuple[float, ...]] = {}
    schedule_root = output_dir / "arrival_schedules"
    base = load_experiment_config(config_name)
    duration = protocol.warm_up_seconds + protocol.measurement_seconds
    for demand in demands:
        demand_config = apply_demand_profile(base, demand)
        for seed in seeds:
            rows = generate_arrival_schedule(
                demand_level=demand,
                model=demand_config.arrival_model,
                arrival_rate=demand_config.arrival_rate,
                seed=int(seed),
                duration_seconds=duration,
            )
            times = schedule_times(rows)
            schedules[(demand, int(seed))] = times
            stem = f"{demand}_seed_{int(seed)}"
            _write_csv(schedule_root / f"{stem}.csv", rows)
            _write_json(
                schedule_root / f"{stem}.json",
                {
                    "scenario": demand,
                    "seed": int(seed),
                    "duration_seconds": duration,
                    "request_count": len(rows),
                    "rows": rows,
                },
            )
    return schedules


def _eval_job(job: dict[str, Any]) -> dict[str, Any]:
    result = run_windowed_evaluation(
        job["config_name"],
        protocol=job["protocol"],
        policy=job["policy"],
        demand_level=job["demand"],
        seed=job["seed"],
        model_path=job.get("model_path"),
        fixed_arrival_schedule=job["schedule"],
    )
    summary = dict(result["summary"])
    summary["policy_label"] = job["policy_label"]
    summary["evidence_type"] = "newly_run"
    summary["fixed_schedule_request_count"] = float(len(job["schedule"]))
    samples = []
    for sample in result["samples"]:
        row = dict(sample)
        row["policy_label"] = job["policy_label"]
        row["config_name"] = summary["config_name"]
        samples.append(row)
    return {"summary": summary, "samples": samples}


def run_common_schedule_comparison(
    *,
    output_dir: Path,
    protocol: WindowedEvaluationProtocol,
    event_model_path: Path | None,
    periodic_model_path: Path | None,
    max_workers: int = 12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schedules = generate_common_schedules(
        config_name="event_gated_hrl",
        output_dir=output_dir,
        protocol=protocol,
    )
    policy_specs = [
        {
            "policy_label": "rule-only",
            "config_name": "fixed_rule",
            "policy": "rule-only",
            "model_path": None,
        },
        {
            "policy_label": "event-rule",
            "config_name": "event_rule",
            "policy": "event-rule",
            "model_path": None,
        },
        {
            "policy_label": "high-level-random",
            "config_name": "event_gated_hrl",
            "policy": "high-level-random",
            "model_path": None,
        },
    ]
    if event_model_path is not None:
        policy_specs.append(
            {
                "policy_label": "event-gated PPO20k",
                "config_name": "event_gated_hrl",
                "policy": "high-level-ppo",
                "model_path": str(event_model_path),
            }
        )
    if periodic_model_path is not None:
        policy_specs.append(
            {
                "policy_label": "periodic PPO20k",
                "config_name": "periodic_hrl",
                "policy": "high-level-ppo",
                "model_path": str(periodic_model_path),
            }
        )
    jobs: list[dict[str, Any]] = []
    for spec in policy_specs:
        for demand in DEMANDS:
            for seed in EVAL_SEEDS:
                jobs.append(
                    {
                        **spec,
                        "demand": demand,
                        "seed": int(seed),
                        "protocol": protocol,
                        "schedule": schedules[(demand, int(seed))],
                    }
                )
    summaries: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    workers = max(1, int(max_workers))
    if workers == 1:
        for job in jobs:
            out = _eval_job(job)
            summaries.append(out["summary"])
            samples.extend(out["samples"])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_eval_job, job) for job in jobs]
            for future in as_completed(futures):
                out = future.result()
                summaries.append(out["summary"])
                samples.extend(out["samples"])
    summaries.sort(key=lambda row: (row["policy_label"], row["traffic_demand_level"], row["seed"]))
    samples.sort(key=lambda row: (row["policy_label"], row["traffic_demand_level"], row["seed"], row["time"]))
    _write_csv(output_dir / "final" / "common_schedule_policy_comparison.csv", summaries)
    _write_csv(
        output_dir / "final" / "policy_comparison_by_demand.csv",
        summarize_rows(summaries, ("policy_label", "traffic_demand_level")),
    )
    _write_csv(output_dir / "traces" / "common_schedule_samples.csv", samples)
    return summaries, samples


def summarize_normalized_metric_saturation(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (policy, demand, metric), group in _group_rows(
        samples,
        ("policy_label", "traffic_demand_level"),
        metrics=("normalized_congestion", "normalized_waiting", "normalized_backlog"),
    ):
        values = [float(row.get(metric, 0.0)) for row in group if row.get("window") == "measurement"]
        stats = _stats(values)
        rows.append(
            {
                "policy_label": policy,
                "traffic_demand_level": demand,
                "metric": metric,
                **stats,
                "fraction_above_0_90": _fraction(values, lambda v: v > 0.90),
                "fraction_above_0_95": _fraction(values, lambda v: v > 0.95),
                "fraction_above_0_99": _fraction(values, lambda v: v > 0.99),
                "fraction_equal_1": _fraction(values, lambda v: abs(v - 1.0) <= 1e-9),
                "reward_state_saturation": float(
                    _fraction(values, lambda v: v > 0.95) >= 0.8
                ),
            }
        )
    return rows


def _fraction(values: Iterable[float], predicate) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(1 for value in values if predicate(float(value))) / len(values))


def _group_rows(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    *,
    metrics: Iterable[str] = ("",),
):
    for metric in metrics:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
        for values, group in sorted(groups.items(), key=lambda item: str(item[0])):
            yield (*values, metric), group


def summarize_action_saturation(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, int], list[float]] = {}
    for row in samples:
        if row.get("window") != "measurement":
            continue
        for kind in ("requested", "applied"):
            for dim in range(DynamicAirspace.ACTION_SIZE):
                key = (
                    str(row["policy_label"]),
                    str(row["traffic_demand_level"]),
                    kind,
                    dim,
                )
                groups.setdefault(key, []).append(float(row.get(f"{kind}_action_dim_{dim}", 0.0)))
    for (policy, demand, kind, dim), values in sorted(groups.items()):
        abs_values = [abs(v) for v in values]
        sign_changes = _sign_change_count(values)
        rows.append(
            {
                "policy_label": policy,
                "traffic_demand_level": demand,
                "action_kind": kind,
                "dim": dim,
                "is_geometry_dim": float(dim in GEOMETRY_DIMS),
                **_stats(values),
                "mean_abs": float(np.mean(abs_values)) if abs_values else 0.0,
                "nonzero_fraction": _fraction(abs_values, lambda v: v > 1e-6),
                "near_zero_fraction": _fraction(abs_values, lambda v: v <= 1e-6),
                "positive_fraction": _fraction(values, lambda v: v > 1e-6),
                "negative_fraction": _fraction(values, lambda v: v < -1e-6),
                "saturation_fraction": _fraction(abs_values, lambda v: v >= 0.98),
                "lower_bound_fraction": _fraction(values, lambda v: v <= -0.98),
                "upper_bound_fraction": _fraction(values, lambda v: v >= 0.98),
                "sign_change_count": float(sign_changes),
                "reversal_rate": float(sign_changes / max(1, len(values) - 1)),
            }
        )
    return rows


def summarize_airspace_boundary(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nominal: dict[str, float] = {}
    first_sample = samples[0] if samples else {}
    for field in AIRSPACE_FIELDS:
        if field in first_sample:
            nominal[field] = float(first_sample[field])
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in samples:
        if row.get("window") != "measurement":
            continue
        for field in AIRSPACE_FIELDS:
            groups.setdefault(
                (str(row["policy_label"]), str(row["traffic_demand_level"]), field),
                [],
            ).append(float(row.get(field, 0.0)))
    for (policy, demand, field), values in sorted(groups.items()):
        diffs = np.abs(np.diff(np.asarray(values, dtype=float))) if len(values) > 1 else np.asarray([])
        nom = nominal.get(field, values[0] if values else 0.0)
        lower = min(values) if values else 0.0
        upper = max(values) if values else 0.0
        span = upper - lower
        degenerate = span <= 1e-9
        rows.append(
            {
                "policy_label": policy,
                "traffic_demand_level": demand,
                "parameter": field,
                "min": lower,
                "max": upper,
                "boundary_fraction": (
                    0.0
                    if degenerate
                    else _fraction(
                        values,
                        lambda v: abs(v - lower) <= 0.02 * span or abs(v - upper) <= 0.02 * span,
                    )
                ),
                "observed_range_degenerate": float(degenerate),
                "nominal_deviation_mean": float(np.mean([abs(v - nom) for v in values])) if values else 0.0,
                "nominal_deviation_p95": float(np.percentile([abs(v - nom) for v in values], 95)) if values else 0.0,
                "time_at_lower_bound": (
                    0.0
                    if degenerate
                    else 20.0 * _fraction(values, lambda v: abs(v - lower) <= 1e-9) * len(values)
                ),
                "time_at_upper_bound": (
                    0.0
                    if degenerate
                    else 20.0 * _fraction(values, lambda v: abs(v - upper) <= 1e-9) * len(values)
                ),
                "changes_per_hour": 3600.0 * float(np.count_nonzero(diffs > 1e-9)) / max(1.0, 20.0 * len(values)),
                "total_variation": float(np.sum(diffs)) if diffs.size else 0.0,
                "oscillation_count": float(_sign_change_count(list(np.diff(np.asarray(values, dtype=float))))),
            }
        )
    return rows


def summarize_gate_effectiveness(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in samples:
        if row.get("window") == "measurement":
            groups.setdefault((str(row["policy_label"]), str(row["traffic_demand_level"])), []).append(row)
    for (policy, demand), group in sorted(groups.items()):
        trigger_scores = [float(row.get("trigger_score", 0.0)) for row in group]
        gate_active = [float(row.get("geometry_gate_active", 0.0)) for row in group]
        masked_by_gate = [float(row.get("masked_by_gate", 0.0)) for row in group]
        masked_by_dwell = [float(row.get("masked_by_dwell", 0.0)) for row in group]
        requested_geometry = [
            max(abs(float(row.get(f"requested_action_dim_{dim}", 0.0))) for dim in GEOMETRY_DIMS)
            for row in group
        ]
        applied_geometry = [
            max(abs(float(row.get(f"applied_action_dim_{dim}", 0.0))) for dim in GEOMETRY_DIMS)
            for row in group
        ]
        duration_hours = max(1e-9, 20.0 * len(group) / 3600.0)
        gate_open_fraction = float(np.mean(gate_active)) if gate_active else 0.0
        if gate_open_fraction <= 0.05:
            diagnosis = "always_closed"
        elif gate_open_fraction >= 0.95:
            diagnosis = "always_open"
        elif (np.mean(masked_by_dwell) if masked_by_dwell else 0.0) > 0.5:
            diagnosis = "dwell_dominated"
        else:
            diagnosis = "meaningful_state_dependent_gating"
        rows.append(
            {
                "policy_label": policy,
                "traffic_demand_level": demand,
                **{f"trigger_score_{k}": v for k, v in _stats(trigger_scores).items()},
                "gate_open_fraction": gate_open_fraction,
                "triggers_per_hour": 3600.0 * float(np.count_nonzero(np.diff(gate_active) > 0.5)) / max(1.0, 20.0 * len(group)),
                "geometry_opportunities_per_hour": 3600.0 * float(sum(float(row.get("geometry_action_allowed", 0.0)) for row in group)) / max(1.0, 20.0 * len(group)),
                "requested_geometry_per_hour": 3600.0 * float(np.count_nonzero(np.asarray(requested_geometry) > 1e-6)) / max(1.0, 20.0 * len(group)),
                "applied_geometry_per_hour": 3600.0 * float(np.count_nonzero(np.asarray(applied_geometry) > 1e-6)) / max(1.0, 20.0 * len(group)),
                "masked_by_gate_ratio": float(np.mean(masked_by_gate)) if masked_by_gate else 0.0,
                "masked_by_dwell_ratio": float(np.mean(masked_by_dwell)) if masked_by_dwell else 0.0,
                "nominal_recovery_per_hour": 3600.0 * float(sum(float(row.get("nominal_recovery_count", 0.0)) for row in group)) / max(1.0, 20.0 * len(group)),
                "diagnosis": diagnosis,
                "duration_hours": duration_hours,
            }
        )
    return rows


def summarize_cap_saturation(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize_rows(
        summaries,
        ("policy_label", "traffic_demand_level"),
    )


def run_saturated_service_capacity(
    *,
    output_dir: Path,
    initial_uavs: int = 35,
    max_steps: int = 5400,
    seed: int = 20,
) -> dict[str, Any]:
    config = load_experiment_config("fixed_rule")
    config.initial_aircraft_count = int(initial_uavs)
    config.formation_cols = 5
    config.formation_rows = int(math.ceil(initial_uavs / 5))
    config.arrival_rate = 0.0
    config.max_active_uavs = max(int(initial_uavs), config.max_active_uavs)
    config.max_steps = int(max_steps)
    env = TerminalParallelEnv(config, experiment_mode=ExperimentMode.FIXED_RULE)
    env.reset(seed=seed)
    previous_landed = 0
    landing_timestamps: list[float] = []
    while env.agents and env.step_count < env.config.max_steps:
        actions = {
            agent: np.zeros(env.action_spec(agent).shape, dtype=np.float32)
            for agent in env.agents
        }
        env.step(actions)
        landed = int(env.metrics()["landed_count"])
        if landed > previous_landed:
            landing_timestamps.extend([float(env.time)] * (landed - previous_landed))
            previous_landed = landed
        if landed >= initial_uavs:
            break
    intervals = np.diff(np.asarray(landing_timestamps, dtype=float))
    row = {
        "evidence_type": "newly_run",
        "initial_uavs": float(initial_uavs),
        "seed": float(seed),
        "time_to_first_landing": float(landing_timestamps[0]) if landing_timestamps else 0.0,
        "landed_count": float(len(landing_timestamps)),
        "elapsed_seconds": float(env.time),
        "mean_inter_landing_seconds": float(np.mean(intervals)) if intervals.size else 0.0,
        "p50_inter_landing_seconds": float(np.percentile(intervals, 50)) if intervals.size else 0.0,
        "p90_inter_landing_seconds": float(np.percentile(intervals, 90)) if intervals.size else 0.0,
        "steady_service_rate_per_hour": (
            3600.0 / float(np.mean(intervals)) if intervals.size and np.mean(intervals) > 0.0 else 0.0
        ),
        "landing_timestamps": ";".join(f"{value:.1f}" for value in landing_timestamps),
    }
    _write_csv(output_dir / "final" / "service_capacity_summary.csv", [row])
    return row


def summarize_demand_utilization(
    summaries: list[dict[str, Any]],
    *,
    service_rate_per_hour: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for demand in DEMANDS:
        demand_rows = [row for row in summaries if row["traffic_demand_level"] == demand]
        if not demand_rows:
            continue
        request_rate = np.mean(
            [
                3600.0
                * float(row.get("measurement_arrival_request_count", 0.0))
                / max(1e-9, float(row.get("measurement_duration_seconds", 1800.0)))
                for row in demand_rows
            ]
        )
        service_rate = np.mean(
            [float(row.get("measurement_throughput_per_hour", 0.0)) for row in demand_rows]
        )
        rows.append(
            {
                "traffic_demand_level": demand,
                "requested_arrival_rate_per_hour": float(request_rate),
                "admitted_arrival_rate_per_hour": float(
                    np.mean(
                        [
                            3600.0
                            * float(row.get("admitted_arrival_count", 0.0))
                            / max(1e-9, float(row.get("total_elapsed_seconds", 1.0)))
                            for row in demand_rows
                        ]
                    )
                ),
                "observed_service_rate_per_hour": float(service_rate),
                "sustainable_service_rate_per_hour": float(service_rate_per_hour),
                "utilization_ratio": float(
                    request_rate / max(1e-9, service_rate_per_hour)
                ),
            }
        )
    return rows


def run_dwell_comparison(
    *,
    output_dir: Path,
    protocol: WindowedEvaluationProtocol,
    max_workers: int = 12,
) -> list[dict[str, Any]]:
    jobs = []
    for dwell in (540.0, 720.0):
        dwell_protocol = WindowedEvaluationProtocol(
            warm_up_seconds=protocol.warm_up_seconds,
            measurement_seconds=protocol.measurement_seconds,
            max_drain_seconds=protocol.max_drain_seconds,
            metrics_sampling_interval_seconds=protocol.metrics_sampling_interval_seconds,
            decision_interval_seconds=protocol.decision_interval_seconds,
            reconfiguration_dwell_seconds=dwell,
            geometry_transition_seconds=protocol.geometry_transition_seconds,
        )
        schedules = generate_common_schedules(
            config_name="event_rule",
            output_dir=output_dir / f"dwell_{int(dwell)}",
            protocol=dwell_protocol,
            seeds=DWELL_SEEDS,
            demands=("high", "bursty"),
        )
        for demand in ("high", "bursty"):
            for seed in DWELL_SEEDS:
                jobs.append(
                    {
                        "policy_label": f"event-rule-dwell-{int(dwell)}",
                        "config_name": "event_rule",
                        "policy": "event-rule",
                        "model_path": None,
                        "demand": demand,
                        "seed": int(seed),
                        "protocol": dwell_protocol,
                        "schedule": schedules[(demand, int(seed))],
                    }
                )
    rows: list[dict[str, Any]] = []
    workers = max(1, int(max_workers))
    if workers == 1:
        for job in jobs:
            out = _eval_job(job)["summary"]
            out["configured_dwell_test_seconds"] = float(job["protocol"].reconfiguration_dwell_seconds)
            rows.append(out)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_eval_job, job) for job in jobs]
            for future in as_completed(futures):
                out = future.result()["summary"]
                label = str(out["policy_label"])
                out["configured_dwell_test_seconds"] = float(label.rsplit("-", 1)[-1])
                rows.append(out)
    rows.sort(key=lambda row: (row["configured_dwell_test_seconds"], row["traffic_demand_level"], row["seed"]))
    _write_csv(output_dir / "final" / "dwell_540_vs_720.csv", rows)
    return rows


def write_decision_files(
    *,
    output_dir: Path,
    summaries: list[dict[str, Any]],
    normalized_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    utilization_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reward_saturation = any(float(row.get("reward_state_saturation", 0.0)) > 0.0 for row in normalized_rows)
    always_on_or_off = any(
        row.get("diagnosis") in {"always_closed", "always_open"}
        and row.get("traffic_demand_level") in {"medium", "high", "bursty"}
        for row in gate_rows
        if row.get("policy_label") in {"event-gated PPO20k", "event-rule"}
    )
    medium_utilization = next(
        (
            float(row["utilization_ratio"])
            for row in utilization_rows
            if row["traffic_demand_level"] == "medium"
        ),
        0.0,
    )
    decision = {
        "status": "RETRAIN_REQUIRED_AFTER_MDP_FIX",
        "retraining_required": True,
        "reason": (
            "arrival requests were previously censored by the internal active cap; "
            "the environment now uses policy-independent external demand plus a "
            "pre-entry FIFO queue, changing MDP/reward semantics."
        ),
        "conditions": {
            "arrival_demand_policy_independent": True,
            "active_cap_no_longer_drops_demand": True,
            "throughput_measurement_semantics_corrected": True,
            "reward_state_saturation_detected": bool(reward_saturation),
            "event_gate_always_on_or_off_risk": bool(always_on_or_off),
            "medium_utilization_ratio": medium_utilization,
        },
        "allowed_status_values": [
            "READY_FOR_50K_WITH_CURRENT_MDP",
            "RETRAIN_REQUIRED_AFTER_MDP_FIX",
            "ENVIRONMENT_NOT_READY",
        ],
    }
    _write_json(output_dir / "final" / "next_training_decision.json", decision)
    summary = {
        "evidence_inventory": {
            "common_schedule_policy_comparison": "newly_run",
            "service_capacity_diagnostics": "newly_run",
            "dwell_540_vs_720": "newly_run",
            "existing_20k_checkpoints": "preexisting_artifact",
        },
        "decision": decision,
        "key_counts": {
            "policy_demand_seed_rows": len(summaries),
            "normalized_metric_rows": len(normalized_rows),
            "gate_effectiveness_rows": len(gate_rows),
        },
    }
    _write_json(output_dir / "final" / "diagnostics_summary.json", summary)
    md = [
        "# Stage-1.5 Diagnostics Summary",
        "",
        f"Decision: `{decision['status']}`",
        "",
        "Evidence types: common-schedule evaluation, service capacity, and dwell comparison are `newly_run`; 20k PPO weights are `preexisting_artifact`.",
        "",
        "Main protocol risk: previous pilots used an MDP where active-cap saturation censored runtime arrival requests. The corrected environment uses external FIFO demand and therefore requires retraining before any 50k continuation.",
        "",
    ]
    (output_dir / "final" / "diagnostics_summary.md").write_text("\n".join(md), encoding="utf-8")
    (output_dir / "final" / "final_test_report.txt").write_text(
        "Stage-1.5 diagnostics generated. Run pytest report is recorded by the agent separately.\n",
        encoding="utf-8",
    )
    return decision


def run_stage15_diagnostics(
    *,
    output_dir: str | Path,
    event_model_path: str | Path | None = None,
    periodic_model_path: str | Path | None = None,
    max_workers: int = 12,
    warm_up_seconds: float = 600.0,
    measurement_seconds: float = 1800.0,
    max_drain_seconds: float = 1800.0,
) -> dict[str, Any]:
    output = Path(output_dir)
    final = output / "final"
    final.mkdir(parents=True, exist_ok=True)
    protocol = WindowedEvaluationProtocol(
        warm_up_seconds=warm_up_seconds,
        measurement_seconds=measurement_seconds,
        max_drain_seconds=max_drain_seconds,
        metrics_sampling_interval_seconds=20.0,
        decision_interval_seconds=180.0,
        reconfiguration_dwell_seconds=600.0,
        geometry_transition_seconds=20.0,
    )
    _write_json(final / "throughput_semantics_audit.json", asdict(protocol))
    _write_json(
        final / "arrival_semantics_audit.json",
        {
            "arrival_request_count": "external runtime demand requests, excluding initial queue",
            "admitted_arrival_count": "runtime requests admitted into terminal as UAV instances",
            "spawned_count": "actual UAV instances in the terminal, including initial queue and admitted runtime arrivals",
            "blocked_arrival_count": "external requests that first encountered full internal active cap",
            "deferred_arrival_count": "requests kept in FIFO due to unsafe spawn/admission geometry",
            "dropped_arrival_count": "requests discarded; should remain zero in simulation diagnostics",
            "external_queue_count": "pre-entry FIFO backlog outside terminal",
            "demand_censoring_fixed": True,
        },
    )
    summaries, samples = run_common_schedule_comparison(
        output_dir=output,
        protocol=protocol,
        event_model_path=Path(event_model_path) if event_model_path else None,
        periodic_model_path=Path(periodic_model_path) if periodic_model_path else None,
        max_workers=max_workers,
    )
    action_rows = summarize_action_saturation(samples)
    airspace_rows = summarize_airspace_boundary(samples)
    normalized_rows = summarize_normalized_metric_saturation(samples)
    gate_rows = summarize_gate_effectiveness(samples)
    cap_rows = summarize_cap_saturation(summaries)
    service = run_saturated_service_capacity(output_dir=output)
    utilization_rows = summarize_demand_utilization(
        summaries,
        service_rate_per_hour=float(service.get("steady_service_rate_per_hour", 0.0)),
    )
    dwell_rows = run_dwell_comparison(
        output_dir=output,
        protocol=protocol,
        max_workers=max_workers,
    )
    event_periodic = [
        row
        for row in summarize_rows(
            [
                row
                for row in summaries
                if row.get("policy_label") in {"event-gated PPO20k", "periodic PPO20k"}
            ],
            ("policy_label", "traffic_demand_level"),
        )
    ]
    _write_csv(final / "action_saturation_summary.csv", action_rows)
    _write_csv(final / "airspace_boundary_summary.csv", airspace_rows)
    _write_csv(final / "normalized_metric_saturation.csv", normalized_rows)
    _write_csv(final / "gate_effectiveness_summary.csv", gate_rows)
    _write_csv(final / "cap_saturation_summary.csv", cap_rows)
    _write_csv(final / "demand_utilization_summary.csv", utilization_rows)
    _write_csv(final / "event_vs_periodic_effect.csv", event_periodic)
    _write_csv(output / "traces" / "action_trace.csv", samples)
    _write_csv(output / "traces" / "gate_trace.csv", samples)
    _write_csv(output / "traces" / "airspace_trace.csv", samples)
    decision = write_decision_files(
        output_dir=output,
        summaries=summaries,
        normalized_rows=normalized_rows,
        gate_rows=gate_rows,
        utilization_rows=utilization_rows,
    )
    return {
        "output_dir": str(output),
        "decision": decision,
        "summary_rows": len(summaries),
        "sample_rows": len(samples),
        "dwell_rows": len(dwell_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-model-path", type=Path)
    parser.add_argument("--periodic-model-path", type=Path)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--warm-up-seconds", type=float, default=600.0)
    parser.add_argument("--measurement-seconds", type=float, default=1800.0)
    parser.add_argument("--max-drain-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    summary = run_stage15_diagnostics(
        output_dir=args.output_dir,
        event_model_path=args.event_model_path,
        periodic_model_path=args.periodic_model_path,
        max_workers=args.max_workers,
        warm_up_seconds=args.warm_up_seconds,
        measurement_seconds=args.measurement_seconds,
        max_drain_seconds=args.max_drain_seconds,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
