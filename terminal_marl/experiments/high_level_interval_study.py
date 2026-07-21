"""High-level timing, horizon, and evaluation-protocol study.

This module is deliberately evaluation-only.  It may load existing PPO
checkpoints for deterministic rollouts, but it never calls ``learn()`` or
starts a new PPO training run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_experiment_config
from ..core.dynamic_airspace import DynamicAirspace
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import DynamicRuleHighLevelPolicy
from .ablation_runner import DEMAND_PROFILES
from .airspace_response_diagnostics import run_response_diagnostic, write_response_outputs
from .evaluation_protocol import (
    WindowedEvaluationProtocol,
    collect_sample,
    configure_high_level_timing,
    count_phase_progression,
    load_existing_ppo_model,
    phase_snapshot,
    run_windowed_evaluation,
    select_high_level_action,
    set_runtime_arrivals_enabled,
    summarize_rows,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def write_timing_audit(output_root: Path, *, ppo_training_config: Path | None = None) -> dict[str, Any]:
    config = load_experiment_config("dynamic_hrl")
    env = TerminalParallelEnv(config)
    env.reset(seed=0)
    pilot_interval = None
    if ppo_training_config is not None and ppo_training_config.exists():
        text = ppo_training_config.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip().startswith("high_level_interval:"):
                pilot_interval = float(line.split(":", 1)[1].strip())
                break
    full_flight_reference_seconds = 220.0
    decision_seconds = config.high_level_interval_steps * config.dt
    audit = {
        "git_commit": _git_commit(),
        "dynamic_hrl_preset_high_level_interval_seconds": decision_seconds,
        "pilot_training_high_level_interval_steps": pilot_interval,
        "pilot_training_high_level_interval_seconds_assuming_dt_1": pilot_interval,
        "is_airspace_action_applied_every_20s_in_dynamic_hrl_preset": bool(math.isclose(decision_seconds, 20.0)),
        "action_semantics": "incremental bounded update over 8 airspace dimensions",
        "action_dimensions": [
            "outer_radius_increment",
            "inner_radius_increment",
            "outer_height_increment",
            "inner_height_increment",
            "outer_speed_increment",
            "inner_speed_increment",
            "ring_headway_increment",
            "final_headway_increment",
        ],
        "immediate_effect": "applied immediately at high-level boundary; non-boundary calls are cached",
        "in_ring_adaptation": "existing UAV guidance reads updated ring radius/height/speed after env._sync_airspace_to_config()",
        "vtac_slot_trajectory_recompute": "VTAC release checks and candidate trajectory approvals are recomputed every low-level step; airspace changes therefore affect subsequent checks immediately",
        "minimum_hold_time_seconds": "no separate reconfiguration dwell beyond high_level_interval_steps in current environment",
        "metrics_sampling_interval_seconds_current_default": config.metrics_sampling_interval_seconds,
        "observation_interval_seconds_current": decision_seconds,
        "decision_interval_seconds_current": decision_seconds,
        "reconfiguration_interval_seconds_current": decision_seconds,
        "estimated_reconfigurations_per_220s_flight": full_flight_reference_seconds / max(1e-9, decision_seconds),
        "recommended_decoupled_protocol": {
            "metrics_sampling_interval_seconds": 20.0,
            "decision_interval_seconds": 60.0,
            "reconfiguration_dwell_seconds": 60.0,
            "geometry_transition_seconds": 20.0,
        },
    }
    audit_dir = output_root / "audit"
    _write_json(audit_dir / "timing_audit.json", audit)
    md = "\n".join(
        [
            "# High-level timing audit",
            "",
            f"- dynamic_hrl preset applies high-level actions every {decision_seconds:.1f}s.",
            f"- 20k pilot training config override: {pilot_interval if pilot_interval is not None else 'not found'} low-level steps.",
            "- Airspace action semantics: incremental bounded changes, not absolute geometry.",
            "- Reconfiguration is immediate at the high-level boundary; non-boundary calls are cached.",
            "- VTAC/release/trajectory safety are recomputed on subsequent low-level steps.",
            "- There is currently no independent minimum dwell beyond the high-level interval.",
        ]
    )
    (audit_dir / "timing_audit.md").write_text(md, encoding="utf-8")
    return audit


def run_interval_sweep(
    output_root: Path,
    *,
    intervals: list[float],
    demands: list[str],
    seeds: list[int],
    warm_up_seconds: float,
    measurement_seconds: float,
    max_drain_seconds: float,
    config: str = "dynamic_hrl",
    policy: str = "dynamic-rule",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for demand in demands:
        for interval in intervals:
            for seed in seeds:
                protocol = WindowedEvaluationProtocol(
                    warm_up_seconds=warm_up_seconds,
                    measurement_seconds=measurement_seconds,
                    max_drain_seconds=max_drain_seconds,
                    metrics_sampling_interval_seconds=20.0,
                    decision_interval_seconds=float(interval),
                    reconfiguration_dwell_seconds=float(interval),
                    geometry_transition_seconds=min(20.0, float(interval)),
                )
                result = run_windowed_evaluation(
                    config,
                    protocol=protocol,
                    policy=policy,
                    demand_level=demand,
                    seed=seed,
                )
                rows.append(result["summary"])
    _write_csv(output_root / "interval_sweep_raw.csv", rows)
    summaries = summarize_rows(
        rows,
        group_keys=("policy", "traffic_demand_level", "decision_interval_seconds"),
    )
    _write_csv(output_root / "interval_sweep_summary.csv", summaries)
    _write_json(output_root / "interval_sweep_summary.json", summaries)
    return rows


def rank_intervals(rows: list[dict[str, Any]]) -> list[float]:
    scores: dict[float, list[float]] = {}
    for row in rows:
        interval = float(row["decision_interval_seconds"])
        score = (
            4.0 * float(row.get("throughput", 0.0))
            + 2.0 * float(row.get("final_completion_ratio", 0.0))
            - 1.5 * float(row.get("normalized_conflict_rate", 0.0))
            - 0.8 * float(row.get("boundary_time_fraction", 0.0))
            - 0.02 * float(row.get("geometry_oscillation_count", 0.0))
            - 0.01 * float(row.get("mean_delay", 0.0))
        )
        scores.setdefault(interval, []).append(score)
    return [
        interval
        for interval, _score in sorted(
            ((interval, float(np.mean(values))) for interval, values in scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def run_horizon_censoring(
    output_root: Path,
    *,
    horizons: list[int],
    policies: list[str],
    seed: int,
    demand_level: str,
    max_drain_seconds: float = 1800.0,
    model_paths: dict[str, Path] | None = None,
    config: str = "dynamic_hrl",
) -> list[dict[str, Any]]:
    model_paths = model_paths or {}
    raw_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    base_protocol = WindowedEvaluationProtocol(
        warm_up_seconds=0.0,
        measurement_seconds=0.0,
        max_drain_seconds=0.0,
        metrics_sampling_interval_seconds=20.0,
        decision_interval_seconds=20.0,
        reconfiguration_dwell_seconds=20.0,
        geometry_transition_seconds=20.0,
    )
    for policy in policies:
        ppo_model = None
        if policy.startswith("ppo_"):
            ppo_model = load_existing_ppo_model(model_paths.get(policy))
            effective_policy = "high-level-ppo"
        else:
            effective_policy = policy
        for horizon in horizons:
            sim_config = configure_high_level_timing(
                load_experiment_config(config),
                base_protocol,
            )
            from .ablation_runner import apply_demand_profile

            sim_config = apply_demand_profile(sim_config, demand_level)
            sim_config.max_steps = max(sim_config.max_steps, horizon * sim_config.high_level_interval_steps + 2000)
            env = TerminalParallelEnv(sim_config)
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            dynamic_rule_policy = DynamicRuleHighLevelPolicy()
            transitions = {
                "approach_to_outer": 0,
                "outer_to_inner": 0,
                "inner_to_final": 0,
                "final_to_land": 0,
            }
            for _hl in range(horizon):
                action = select_high_level_action(
                    env,
                    policy=effective_policy,
                    rng=rng,
                    dynamic_rule_policy=dynamic_rule_policy,
                    ppo_model=ppo_model,
                )
                for _ in range(env.config.high_level_interval_steps):
                    previous = phase_snapshot(env)
                    env.step({}, high_level_action=action if env.should_update_high_level() else None)
                    action = None
                    step_counts = count_phase_progression(previous, env)
                    for key, value in step_counts.items():
                        transitions[key] += value
                curve_rows.append(
                    {
                        "policy": policy,
                        "horizon_steps": float(horizon),
                        "time": float(env.time),
                        "spawned_count": float(len(env.aircraft)),
                        "landed_count": float(env.metrics()["landed_count"]),
                        "active_count": float(env.metrics()["active_count"]),
                        "normalized_conflict_rate": float(env.metrics()["normalized_conflict_rate"]),
                        **{key: float(value) for key, value in transitions.items()},
                    }
                )
            spawned_at_horizon = float(len(env.aircraft))
            landed_at_horizon = float(env.metrics()["landed_count"])
            set_runtime_arrivals_enabled(env, False)
            drain_limit = env.time + float(max_drain_seconds)
            while env.time < drain_limit and any(not aircraft.finished for aircraft in env.aircraft):
                previous = phase_snapshot(env)
                env.step({})
                step_counts = count_phase_progression(previous, env)
                for key, value in step_counts.items():
                    transitions[key] += value
            raw_rows.append(
                {
                    "policy": policy,
                    "seed": float(seed),
                    "traffic_demand_level": demand_level,
                    "horizon_steps": float(horizon),
                    "horizon_seconds": float(horizon * 20.0),
                    "spawned_at_horizon": spawned_at_horizon,
                    "landed_at_horizon": landed_at_horizon,
                    "landed_by_horizon_ratio": landed_at_horizon / max(1.0, spawned_at_horizon),
                    "spawned_after_drain": float(len(env.aircraft)),
                    "landed_after_drain": float(env.metrics()["landed_count"]),
                    "final_completion_ratio": float(env.metrics()["landed_count"]) / max(1.0, float(len(env.aircraft))),
                    "active_count_after_drain": float(env.metrics()["active_count"]),
                    "normalized_conflict_rate": float(env.metrics()["normalized_conflict_rate"]),
                    **{key: float(value) for key, value in transitions.items()},
                }
            )
    _write_csv(output_root / "horizon_censoring_raw.csv", raw_rows)
    _write_csv(output_root / "cumulative_landing_curves.csv", curve_rows)
    summary = summarize_rows(raw_rows, group_keys=("policy", "horizon_steps"))
    _write_csv(output_root / "horizon_censoring_summary.csv", summary)
    return raw_rows


def run_action_effectiveness(
    output_root: Path,
    *,
    policies: list[str],
    demand_level: str,
    seed: int,
    decision_interval_seconds: float,
    warm_up_seconds: float = 600.0,
    measurement_seconds: float = 1200.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    protocol = WindowedEvaluationProtocol(
        warm_up_seconds=warm_up_seconds,
        measurement_seconds=measurement_seconds,
        max_drain_seconds=0.0,
        metrics_sampling_interval_seconds=20.0,
        decision_interval_seconds=decision_interval_seconds,
        reconfiguration_dwell_seconds=decision_interval_seconds,
        geometry_transition_seconds=min(20.0, decision_interval_seconds),
    )
    for policy in policies:
        result = run_windowed_evaluation(
            "dynamic_hrl",
            protocol=protocol,
            policy=policy,
            demand_level=demand_level,
            seed=seed,
        )
        row = dict(result["summary"])
        row["airspace_parameter_deviation_l1_final"] = float(
            result["samples"][-1]["airspace_deviation_l1"] if result["samples"] else 0.0
        )
        rows.append(row)
    _write_csv(output_root / "action_effectiveness_summary.csv", rows)
    lines = ["# Action effectiveness", ""]
    for row in rows:
        lines.append(
            f"- {row['policy']}: mean_action_l1={row['mean_action_l1']:.3f}, "
            f"nonzero_ratio={row['action_nonzero_ratio']:.3f}, "
            f"boundary_time={row['boundary_time_fraction']:.3f}, "
            f"final_airspace_deviation_l1={row['airspace_parameter_deviation_l1_final']:.3f}"
        )
    (output_root / "action_effectiveness_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def run_demand_scenarios(
    output_root: Path,
    *,
    demands: list[str],
    policies: list[str],
    seed: int,
    decision_interval_seconds: float,
    warm_up_seconds: float = 600.0,
    measurement_seconds: float = 1200.0,
    max_drain_seconds: float = 1200.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    protocol = WindowedEvaluationProtocol(
        warm_up_seconds=warm_up_seconds,
        measurement_seconds=measurement_seconds,
        max_drain_seconds=max_drain_seconds,
        metrics_sampling_interval_seconds=20.0,
        decision_interval_seconds=decision_interval_seconds,
        reconfiguration_dwell_seconds=decision_interval_seconds,
        geometry_transition_seconds=min(20.0, decision_interval_seconds),
    )
    for demand in demands:
        for policy in policies:
            rows.append(
                run_windowed_evaluation(
                    "dynamic_hrl" if policy != "rule-only" else "fixed_rule",
                    protocol=protocol,
                    policy=policy,
                    demand_level=demand,
                    seed=seed,
                )["summary"]
            )
    _write_csv(output_root / "demand_scenario_summary.csv", rows)
    return rows


def write_evaluation_protocol(output_root: Path, interval: float) -> None:
    text = "\n".join(
        [
            "simulation_only: true",
            "high_level:",
            "  metrics_sampling_interval_seconds: 20",
            f"  decision_interval_seconds: {interval:g}",
            f"  reconfiguration_dwell_seconds: {interval:g}",
            f"  geometry_transition_seconds: {min(20.0, interval):g}",
            "evaluation:",
            "  warm_up_seconds: 600",
            "  measurement_seconds: 1800",
            "  max_drain_seconds: 1800",
            "  count_warm_up_in_metrics: false",
            "  drain_arrivals_enabled: false",
            "  compare_by_physical_duration: true",
        ]
    )
    (output_root / "evaluation_protocol.yaml").write_text(text + "\n", encoding="utf-8")


def build_final_recommendation(
    output_root: Path,
    *,
    audit: dict[str, Any],
    response_summaries: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    demand_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = rank_intervals(interval_rows)
    recommended = ranked[0] if ranked else 60.0
    response_settling = [
        float(row.get("settling_time_seconds", 0.0))
        for row in response_summaries
        if row.get("action_name") != "no_change"
    ]
    median_settling = float(np.median(response_settling)) if response_settling else 0.0
    horizon20 = [
        row for row in horizon_rows if float(row["horizon_steps"]) == 20.0
    ]
    horizon100 = [
        row for row in horizon_rows if float(row["horizon_steps"]) == 100.0
    ]
    censoring_detected = (
        float(np.mean([row["landed_by_horizon_ratio"] for row in horizon20])) + 0.05
        < float(np.mean([row["landed_by_horizon_ratio"] for row in horizon100]))
        if horizon20 and horizon100
        else False
    )
    dynamic_rule_action = next(
        (row for row in action_rows if row.get("policy") == "dynamic-rule"),
        {},
    )
    action_near_noop = float(dynamic_rule_action.get("mean_action_abs", 0.0)) < 0.01
    demand_by_level = {
        demand: [
            row
            for row in demand_rows
            if row.get("traffic_demand_level") == demand
        ]
        for demand in DEMAND_PROFILES
    }
    separability = {
        demand: float(np.mean([row.get("measurement_spawned", 0.0) for row in rows]))
        for demand, rows in demand_by_level.items()
        if rows
    }
    no_baseline_fail = all(float(row.get("landed_total", 0.0)) >= 0.0 for row in demand_rows)
    if not no_baseline_fail:
        status = "NOT_READY_FOR_RETRAINING"
    elif recommended != 20.0 or median_settling > 20.0:
        status = "CHANGE_INTERVAL_AND_RETRAIN_FROM_SCRATCH"
    elif action_near_noop or censoring_detected:
        status = "NOT_READY_FOR_RETRAINING"
    else:
        status = "KEEP_20S_AND_CONTINUE_50K"
    recommendation = {
        "status": status,
        "recommended_decision_interval_seconds": float(recommended),
        "recommended_min_dwell_seconds": float(max(recommended, median_settling)),
        "keep_metrics_sampling_interval_seconds": 20.0,
        "median_response_settling_seconds": median_settling,
        "horizon_censoring_detected": bool(censoring_detected),
        "dynamic_rule_action_near_noop": bool(action_near_noop),
        "demand_separability_spawned_by_level": separability,
        "old_20k_ppo_can_resume": bool(status == "KEEP_20S_AND_CONTINUE_50K"),
        "requires_retrain_from_scratch": bool(status == "CHANGE_INTERVAL_AND_RETRAIN_FROM_SCRATCH"),
        "notes": [
            "Existing 20k PPO is only directly comparable under its native 20s/high-level-step evaluation protocol.",
            "If interval changes, do not resume the old checkpoint; run a new 512 smoke and 20k pilot from scratch.",
        ],
        "audit": audit,
    }
    final_dir = output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "response_summary.csv",
        "interval_sweep_summary.csv",
        "horizon_censoring_summary.csv",
        "demand_scenario_summary.csv",
        "action_effectiveness_summary.csv",
    ):
        src = output_root / filename
        if src.exists():
            (final_dir / filename).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    _write_json(final_dir / "high_level_interval_recommendation.json", recommendation)
    _write_json(final_dir / "retraining_decision.json", {
        "status": status,
        "recommended_decision_interval_seconds": float(recommended),
        "requires_retrain_from_scratch": bool(status == "CHANGE_INTERVAL_AND_RETRAIN_FROM_SCRATCH"),
    })
    write_evaluation_protocol(final_dir, float(recommended))
    md = [
        "# High-level interval recommendation",
        "",
        f"- Status: `{status}`",
        f"- Recommended decision interval: {recommended:.0f}s",
        f"- Recommended minimum dwell: {max(recommended, median_settling):.0f}s",
        "- Keep metrics sampling at 20s for plots and comparable traces.",
        f"- Median one-shot response settling time: {median_settling:.1f}s",
        f"- Horizon censoring detected: {censoring_detected}",
        f"- Dynamic-rule near no-op: {action_near_noop}",
    ]
    (final_dir / "high_level_interval_recommendation.md").write_text("\n".join(md), encoding="utf-8")
    (final_dir / "final_test_report.txt").write_text(
        "Generated by high_level_interval_study; run local pytest for code-level validation.\n",
        encoding="utf-8",
    )
    return recommendation


def discover_stage1_models(root: Path) -> dict[str, Path]:
    model_root = root / "ppo_stage1_pilot_20k_seed0"
    candidates = {
        "ppo_final": model_root / "model.zip",
        "ppo_5k": model_root / "checkpoints" / "ppo_checkpoint_5000_steps.zip",
        "ppo_10k": model_root / "checkpoints" / "ppo_checkpoint_10000_steps.zip",
        "ppo_15k": model_root / "checkpoints" / "ppo_checkpoint_15000_steps.zip",
        "ppo_20k": model_root / "checkpoints" / "ppo_checkpoint_20000_steps.zip",
    }
    return {name: path for name, path in candidates.items() if path.exists()}


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    training_root = Path(args.training_root)
    audit = write_timing_audit(
        output_root,
        ppo_training_config=Path("terminal_marl/configs/training/ppo_pilot_medium.yaml"),
    )

    combined_response = {"trace": [], "summary": []}
    for demand in ["medium", "high"]:
        result = run_response_diagnostic(
            demand_level=demand,
            seed=0,
            warm_up_seconds=args.warm_up_seconds,
            response_seconds=args.response_seconds,
            sampling_interval_seconds=20.0,
        )
        combined_response["trace"].extend(result["trace"])
        combined_response["summary"].extend(result["summary"])
    write_response_outputs(combined_response, output_root)

    first_round = run_interval_sweep(
        output_root,
        intervals=[float(value) for value in args.intervals],
        demands=["medium", "high"],
        seeds=[0],
        warm_up_seconds=args.warm_up_seconds,
        measurement_seconds=args.measurement_seconds,
        max_drain_seconds=args.max_drain_seconds,
    )
    selected = sorted(set([20.0, *rank_intervals(first_round)[:2]]))
    second_round = run_interval_sweep(
        output_root,
        intervals=selected,
        demands=["medium", "high", "bursty"],
        seeds=[0, 1, 2],
        warm_up_seconds=args.warm_up_seconds,
        measurement_seconds=args.measurement_seconds,
        max_drain_seconds=args.max_drain_seconds,
    )
    interval_rows = [*first_round, *second_round]
    _write_csv(output_root / "interval_sweep_raw.csv", interval_rows)
    interval_summary = summarize_rows(
        interval_rows,
        group_keys=("policy", "traffic_demand_level", "decision_interval_seconds"),
    )
    _write_csv(output_root / "interval_sweep_summary.csv", interval_summary)
    _write_json(output_root / "interval_sweep_summary.json", interval_summary)
    (output_root / "interval_recommendation.md").write_text(
        f"Selected second-round intervals: {selected}\n",
        encoding="utf-8",
    )

    model_paths = discover_stage1_models(training_root)
    horizon_policies = ["rule-only", "dynamic-rule", "high-level-random", *model_paths.keys()]
    horizon_rows = run_horizon_censoring(
        output_root,
        horizons=[int(value) for value in args.horizons],
        policies=horizon_policies,
        seed=0,
        demand_level="medium",
        max_drain_seconds=args.max_drain_seconds,
        model_paths=model_paths,
    )
    action_rows = run_action_effectiveness(
        output_root,
        policies=["rule-only", "dynamic-rule", "high-level-random"],
        demand_level="medium",
        seed=0,
        decision_interval_seconds=20.0,
        warm_up_seconds=args.warm_up_seconds,
        measurement_seconds=args.measurement_seconds,
    )
    demand_rows = run_demand_scenarios(
        output_root,
        demands=["low", "medium", "high", "bursty"],
        policies=["rule-only", "dynamic-rule"],
        seed=0,
        decision_interval_seconds=20.0,
        warm_up_seconds=args.warm_up_seconds,
        measurement_seconds=args.measurement_seconds,
        max_drain_seconds=args.max_drain_seconds,
    )
    recommendation = build_final_recommendation(
        output_root,
        audit=audit,
        response_summaries=combined_response["summary"],
        interval_rows=interval_rows,
        horizon_rows=horizon_rows,
        action_rows=action_rows,
        demand_rows=demand_rows,
    )
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=Path.home() / "training_outputs" / "terminal")
    parser.add_argument("--warm-up-seconds", type=float, default=600.0)
    parser.add_argument("--measurement-seconds", type=float, default=1200.0)
    parser.add_argument("--max-drain-seconds", type=float, default=1200.0)
    parser.add_argument("--response-seconds", type=float, default=600.0)
    parser.add_argument("--intervals", nargs="+", type=float, default=[20.0, 40.0, 60.0, 100.0, 180.0])
    parser.add_argument("--horizons", nargs="+", type=int, default=[20, 50, 100])
    args = parser.parse_args()
    recommendation = run_study(args)
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
