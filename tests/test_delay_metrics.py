"""Tests for nominal schedule and strict landing-delay statistics."""

import pytest

from terminal_marl import SimConfig
from terminal_marl.core.scenario import build_initial_2x4_queue
from terminal_marl.experiments import ABLATION_METRIC_KEYS, run_rollout


def test_scenario_assigns_nominal_landing_schedule() -> None:
    aircraft = build_initial_2x4_queue(SimConfig())
    planned = [item.planned_nominal_landing_time for item in aircraft]
    assert all(value is not None and value > 0.0 for value in planned)
    assert planned == sorted(planned)


def test_landing_delay_is_actual_minus_planned() -> None:
    aircraft = build_initial_2x4_queue(SimConfig())[0]
    planned = aircraft.planned_nominal_landing_time
    aircraft.mark_finished(step=100, time=planned + 7.5)
    metrics = aircraft.metrics_dict((0.0, 0.0, 0.0))
    assert aircraft.actual_landing_time == pytest.approx(planned + 7.5)
    assert aircraft.landing_delay == pytest.approx(7.5)
    assert metrics["landing_delay"] == pytest.approx(7.5)


def test_ablation_keeps_completion_and_delay_metrics_separate() -> None:
    metrics = run_rollout("fixed_rule", max_steps=1)
    assert {
        "average_delay",
        "average_completion_time",
        "delay_p50",
        "delay_p90",
        "max_delay",
        "total_completion_time",
    } <= set(metrics)
    assert set(ABLATION_METRIC_KEYS) <= set(metrics)
