"""Unit tests for high-level interval study helpers."""

from pathlib import Path

from terminal_marl.experiments.high_level_interval_study import (
    build_final_recommendation,
    rank_intervals,
    write_timing_audit,
)


def test_rank_intervals_prefers_better_throughput_and_safety() -> None:
    rows = [
        {
            "decision_interval_seconds": 20.0,
            "throughput": 0.01,
            "final_completion_ratio": 0.5,
            "normalized_conflict_rate": 0.2,
            "boundary_time_fraction": 0.4,
            "geometry_oscillation_count": 5.0,
            "mean_delay": 40.0,
        },
        {
            "decision_interval_seconds": 60.0,
            "throughput": 0.03,
            "final_completion_ratio": 0.8,
            "normalized_conflict_rate": 0.01,
            "boundary_time_fraction": 0.0,
            "geometry_oscillation_count": 1.0,
            "mean_delay": 5.0,
        },
    ]

    assert rank_intervals(rows)[0] == 60.0


def test_timing_audit_writes_json_and_markdown(tmp_path: Path) -> None:
    audit = write_timing_audit(tmp_path)

    assert (tmp_path / "audit" / "timing_audit.json").exists()
    assert (tmp_path / "audit" / "timing_audit.md").exists()
    assert audit["action_semantics"].startswith("incremental")


def test_final_recommendation_writes_required_files(tmp_path: Path) -> None:
    recommendation = build_final_recommendation(
        tmp_path,
        audit={"dynamic_hrl_preset_high_level_interval_seconds": 20.0},
        response_summaries=[
            {"action_name": "safe_combo", "settling_time_seconds": 80.0},
        ],
        interval_rows=[
            {
                "decision_interval_seconds": 60.0,
                "throughput": 0.03,
                "final_completion_ratio": 0.8,
                "normalized_conflict_rate": 0.01,
                "boundary_time_fraction": 0.0,
                "geometry_oscillation_count": 0.0,
                "mean_delay": 5.0,
            }
        ],
        horizon_rows=[
            {"horizon_steps": 20.0, "landed_by_horizon_ratio": 0.1},
            {"horizon_steps": 100.0, "landed_by_horizon_ratio": 0.6},
        ],
        action_rows=[
            {"policy": "dynamic-rule", "mean_action_abs": 0.05},
        ],
        demand_rows=[
            {"traffic_demand_level": "medium", "measurement_spawned": 2.0, "landed_total": 1.0},
        ],
    )

    assert recommendation["status"] == "CHANGE_INTERVAL_AND_RETRAIN_FROM_SCRATCH"
    assert (tmp_path / "final" / "high_level_interval_recommendation.json").exists()
    assert (tmp_path / "final" / "evaluation_protocol.yaml").exists()
    assert (tmp_path / "final" / "retraining_decision.json").exists()
