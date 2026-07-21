"""Smoke tests for one-shot airspace response diagnostics."""

from terminal_marl.experiments.airspace_response_diagnostics import (
    ACTION_LIBRARY,
    run_response_diagnostic,
)


def test_response_diagnostic_outputs_trace_and_summary() -> None:
    result = run_response_diagnostic(
        demand_level="medium",
        seed=0,
        warm_up_seconds=2.0,
        response_seconds=4.0,
        sampling_interval_seconds=2.0,
    )

    assert result["trace"]
    assert result["summary"]
    assert {row["action_name"] for row in result["summary"]} == set(ACTION_LIBRARY)
    for row in result["summary"]:
        assert "first_response_seconds" in row
        assert "settling_time_seconds" in row
        assert "recommended_decision_interval_seconds" in row
