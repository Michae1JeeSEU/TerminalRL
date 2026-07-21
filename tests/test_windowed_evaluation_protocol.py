"""Regression tests for warm-up / measurement / drain evaluation."""

from __future__ import annotations

import math

import pytest

from terminal_marl.experiments.ablation_runner import apply_demand_profile
from terminal_marl.config import load_experiment_config
from terminal_marl.experiments.evaluation_protocol import (
    WindowedEvaluationProtocol,
    configure_high_level_timing,
    run_windowed_evaluation,
)


def short_protocol(decision_interval: float) -> WindowedEvaluationProtocol:
    return WindowedEvaluationProtocol(
        warm_up_seconds=5.0,
        measurement_seconds=10.0,
        max_drain_seconds=10.0,
        metrics_sampling_interval_seconds=2.0,
        decision_interval_seconds=decision_interval,
        reconfiguration_dwell_seconds=decision_interval,
        geometry_transition_seconds=min(2.0, decision_interval),
    )


def test_drain_disables_new_arrivals() -> None:
    result = run_windowed_evaluation(
        "dynamic_hrl",
        protocol=short_protocol(2.0),
        policy="dynamic-rule",
        demand_level="high",
        seed=2,
    )

    drain_samples = [row for row in result["samples"] if row["window"] == "drain"]
    assert drain_samples
    spawned = [row["spawned_count"] for row in drain_samples]
    assert spawned == sorted(spawned)
    assert spawned[-1] == pytest.approx(spawned[0])


def test_arrival_rate_is_not_changed_by_decision_interval() -> None:
    base = apply_demand_profile(load_experiment_config("dynamic_hrl"), "medium")
    cfg20 = configure_high_level_timing(base, short_protocol(2.0))
    cfg60 = configure_high_level_timing(base, short_protocol(6.0))

    assert cfg20.arrival_rate == pytest.approx(cfg60.arrival_rate)
    assert cfg20.high_level_interval_steps != cfg60.high_level_interval_steps


def test_measurement_stats_do_not_include_warmup_start() -> None:
    result = run_windowed_evaluation(
        "dynamic_hrl",
        protocol=short_protocol(2.0),
        policy="rule-only",
        demand_level="medium",
        seed=3,
    )

    summary = result["summary"]
    assert summary["measurement_start_time"] >= 5.0
    assert summary["warm_up_seconds"] == pytest.approx(5.0)
    assert summary["measurement_seconds"] == pytest.approx(10.0)


def test_final_completion_ratio_uses_spawned_after_drain() -> None:
    result = run_windowed_evaluation(
        "dynamic_hrl",
        protocol=short_protocol(2.0),
        policy="rule-only",
        demand_level="low",
        seed=4,
    )

    summary = result["summary"]
    expected = summary["landed_total"] / max(1.0, summary["spawned_total"])
    assert summary["final_completion_ratio"] == pytest.approx(expected)


def test_physical_duration_is_comparable_across_intervals() -> None:
    protocol_a = short_protocol(2.0)
    protocol_b = short_protocol(5.0)
    result_a = run_windowed_evaluation(
        "dynamic_hrl",
        protocol=protocol_a,
        policy="dynamic-rule",
        demand_level="medium",
        seed=5,
    )
    result_b = run_windowed_evaluation(
        "dynamic_hrl",
        protocol=protocol_b,
        policy="dynamic-rule",
        demand_level="medium",
        seed=5,
    )

    assert math.isclose(
        result_a["summary"]["warm_up_seconds"] + result_a["summary"]["measurement_seconds"],
        result_b["summary"]["warm_up_seconds"] + result_b["summary"]["measurement_seconds"],
    )
