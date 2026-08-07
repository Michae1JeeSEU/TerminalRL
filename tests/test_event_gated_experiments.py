"""Smoke tests for event-gated audit/calibration/validation tools."""

from pathlib import Path

from terminal_marl.experiments.calibrate_geometry_gate import calibrate
from terminal_marl.experiments.event_gated_audit import write_audit
from terminal_marl.experiments.event_gated_validation import run_validation


def test_event_gated_audit_writes_required_files(tmp_path: Path) -> None:
    audit = write_audit(tmp_path / "audit")

    assert (tmp_path / "audit" / "action_semantics.json").exists()
    assert (tmp_path / "audit" / "timing_semantics.json").exists()
    assert (tmp_path / "audit" / "current_mdp_audit.md").exists()
    assert audit["action_semantics"]["structural_geometry_indices"] == [0, 1, 2, 3]


def test_gate_calibration_writes_threshold_recommendation(tmp_path: Path) -> None:
    recommendation = calibrate(
        output_dir=tmp_path,
        warm_up_seconds=1.0,
        measurement_seconds=2.0,
        max_drain_seconds=1.0,
        seeds=(0,),
        demands=("low", "high"),
    )

    assert (tmp_path / "gate_calibration_raw.csv").exists()
    assert (tmp_path / "gate_calibration_summary.csv").exists()
    assert (tmp_path / "gate_threshold_recommendation.json").exists()
    assert recommendation["theta_off"] < recommendation["theta_on"]


def test_event_gated_validation_writes_baseline_and_gate_behavior(tmp_path: Path) -> None:
    behavior = run_validation(
        output_dir=tmp_path,
        demands=("medium",),
        seeds=(10,),
        policies=("rule-only", "event-rule"),
        warm_up_seconds=1.0,
        measurement_seconds=2.0,
        max_drain_seconds=1.0,
    )

    assert (tmp_path / "baseline_validation.csv").exists()
    assert (tmp_path / "gate_event_trace.csv").exists()
    assert (tmp_path / "gate_behavior_by_demand.csv").exists()
    assert behavior["rows"]
