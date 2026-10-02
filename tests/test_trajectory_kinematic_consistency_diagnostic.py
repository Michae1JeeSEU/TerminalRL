"""Regression tests for the trajectory kinematic consistency diagnostic."""

from __future__ import annotations

import json

import numpy as np

from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL
from terminal_marl.scripts.diagnose_trajectory_kinematic_consistency_20261002 import (
    _v3_analytic_consistency,
    main,
    make_linear_plan,
    run_diagnostic,
    diagnose_plan,
)


def test_constant_velocity_linear_plan_is_consistent() -> None:
    row = diagnose_plan(
        make_linear_plan(),
        profile=RESEARCH_EVTOL,
        physics_dt=0.2,
        mode="synthetic",
        reservation_slot=0.0,
    )

    assert row["consistent"] is True
    assert row["findings"] == []
    assert row["max_chord_minus_average_velocity"]["norm"] == 0.0
    assert row["max_displacement_integral_residual"]["norm"] == 0.0
    assert row["first_segment_midpoint_difference"]["central_derivative_minus_sample_velocity_norm"] < 1.0e-12
    assert row["execution_sample_metrics"]["max_horizontal_acceleration"]["value"] == 0.0
    assert row["execution_sample_metrics"]["max_vertical_acceleration_abs"]["value"] == 0.0


def test_reported_velocity_mismatch_is_detected() -> None:
    row = diagnose_plan(
        make_linear_plan(reported_velocity=(2.0, 0.0, 0.0)),
        profile=RESEARCH_EVTOL,
        physics_dt=0.2,
        mode="synthetic",
        reservation_slot=0.0,
    )

    assert row["consistent"] is False
    assert "node_chord_velocity_mismatch" in row["findings"]
    assert "node_displacement_integral_residual" in row["findings"]
    assert "first_segment_midpoint_velocity_mismatch" in row["findings"]
    assert row["max_chord_minus_average_velocity"]["norm"] == 1.0
    assert row["max_displacement_integral_residual"]["norm"] == 1.0
    assert row["execution_sample_metrics"]["max_horizontal_speed"]["value"] == 1.0


def test_v3_uses_analytic_derivative_and_integral_checks() -> None:
    row = diagnose_plan(
        make_linear_plan(trajectory_schema="executable_trajectory_v3"),
        profile=RESEARCH_EVTOL,
        physics_dt=0.2,
        mode="synthetic",
        reservation_slot=0.0,
        trajectory_schema="executable_trajectory_v3",
    )

    assert row["consistent"] is True
    assert row["v3_analytic_consistency"]["consistent"] is True
    assert row["legacy_node_diagnostics"]["applicability"] == "v2_only_nonbinding"
    assert row["findings"] == []


def test_v3_analytic_altitude_bounds_aggregate_across_segments() -> None:
    points = tuple(
        TrajectoryPoint(
            float(index),
            np.asarray(position, dtype=float),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
        )
        for index, position in enumerate(((0.0, 0.0, 5.0), (1.0, 0.0, 10.0), (2.0, 0.0, 2.0)))
    )
    plan = TrajectoryPlan(
        "v3-altitude-aggregate",
        1,
        "diagnostic",
        points,
        0.0,
        2.0,
        metadata={"trajectory_schema_version": "executable_trajectory_v3"},
    )

    analytic = _v3_analytic_consistency(plan)
    np.testing.assert_allclose(analytic["analytic_bounds"]["min_altitude"], 2.0, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(analytic["analytic_bounds"]["max_altitude"], 10.0, atol=1.0e-12, rtol=0.0)


def test_four_fixture_report_serializes_checks_and_sampled_results(tmp_path) -> None:
    output = tmp_path / "kinematic-consistency.json"
    payload = run_diagnostic(output)

    assert payload["stage"] == "trajectory_kinematic_consistency"
    assert payload["status"] == "KINEMATIC_CONSISTENCY_FAIL"
    assert payload["consistent"] is False
    assert payload["technical_error_count"] == 0
    assert len(payload["plans"]) == 4
    assert payload["checks"]["validate_kinematics_results_included"] is True
    assert payload["checks"]["execution_sample_results_included"] is True

    by_key = {(row["mode"], row["reservation_slot"]): row for row in payload["plans"]}
    assert set(by_key) == {
        ("legacy", 0.0),
        ("legacy", 30.0),
        ("speed_timing_v1", 0.0),
        ("speed_timing_v1", 30.0),
    }
    for row in payload["plans"]:
        assert row["validate_kinematics"] == []
        assert row["execution_sample_metrics"]["physics_dt"] == 0.2
        assert row["execution_sample_metrics"]["max_horizontal_acceleration"] is not None
        assert row["execution_sample_metrics"]["max_vertical_acceleration_abs"] is not None
        assert row["execution_sample_metrics"]["last_interval_dt"] > 0.0
        assert row["execution_sample_metrics"]["last_interval_is_short"] is True
        assert len(row["first_three_segments"]) == 3

    assert by_key[("legacy", 0.0)]["trajectory_hash"] == by_key[("legacy", 30.0)]["trajectory_hash"]
    assert by_key[("speed_timing_v1", 0.0)]["trajectory_hash"] != by_key[("speed_timing_v1", 30.0)]["trajectory_hash"]
    assert by_key[("legacy", 0.0)]["execution_sample_metrics"]["max_horizontal_acceleration"]["value"] > 1.5

    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["status"] == payload["status"]
    assert np.isfinite(parsed["plans"][0]["duration"])


def test_cli_reports_findings_with_success_exit_code(tmp_path, capsys) -> None:
    output = tmp_path / "cli-kinematic-consistency.json"
    exit_code = main(["--output", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"status": "KINEMATIC_CONSISTENCY_FAIL"' in captured.out
    assert output.exists()
