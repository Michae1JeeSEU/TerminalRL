"""Pure construction/schema checks for the P3 safety-energy diagnostic."""
from __future__ import annotations

from pathlib import Path

import pytest

from terminal_marl.training.diagnose_v4_2_p3_safety_energy_sensitivity_v1 import (
    COLLISION_DISTANCE_M,
    ENERGY_CAPACITY,
    FULL_HORIZON_SECONDS,
    PROTECTED_FLOOR_M,
    _base_record,
    _merge_summary_record,
    build_case_matrix,
    scenario_config,
)


def test_case_matrix_is_seed_distance_controller_cartesian_product() -> None:
    cases = build_case_matrix(
        [101, 102], [60.0, 80.0], ["rule", "original_k3"]
    )

    assert len(cases) == 8
    assert cases[0] == {
        "case_id": "dense24_seed101_safe60_rule",
        "scenario": "dense24",
        "seed": 101,
        "safe_distance": 60.0,
        "controller": "rule",
    }
    assert cases[-1]["case_id"] == "dense24_seed102_safe80_original_k3"
    assert {(row["seed"], row["safe_distance"], row["controller"]) for row in cases} == {
        (seed, distance, controller)
        for seed in (101, 102)
        for distance in (60.0, 80.0)
        for controller in ("rule", "original_k3")
    }


def test_matrix_rejects_a_safe_distance_at_or_below_protected_floor() -> None:
    with pytest.raises(ValueError, match="protected floor"):
        build_case_matrix([101], [PROTECTED_FLOOR_M], ["rule"])


def test_dense24_config_freezes_diagnostic_constants() -> None:
    config = scenario_config(80.0)

    assert config.total_requested_uavs == 24
    assert config.warm_start_uavs == 12
    assert config.safe_distance == 80.0
    assert config.nmac_distance == PROTECTED_FLOOR_M
    assert config.collision_distance == COLLISION_DISTANCE_M
    assert config.energy_capacity == ENERGY_CAPACITY
    assert config.max_simulation_seconds == FULL_HORIZON_SECONDS


def test_base_record_exposes_required_output_schema(tmp_path: Path) -> None:
    case = {
        "case_id": "dense24_seed101_safe80_rule",
        "scenario": "dense24",
        "seed": 101,
        "safe_distance": 80.0,
        "controller": "rule",
    }
    record = _base_record(case, case_dir=tmp_path / case["case_id"])

    required = {
        "requests",
        "completed",
        "terminal",
        "outcomes",
        "low_soc_diversions",
        "minimum_swept_separation",
        "operational_buffer_entries",
        "operational_buffer_exposure_steps",
        "shield_uses",
        "hard_separation_violations",
        "nmac",
        "collisions",
        "safety_invariant_errors",
        "elapsed_sim_time_seconds",
        "mean_wait_seconds_completed_only",
        "mean_request_to_touchdown_seconds_completed_only",
        "completed_only_condition",
        "exception",
    }
    assert required <= set(record)
    assert record["status"] == "failed"
    assert record["completed_only_condition"] == (
        "completed_uavs_only; null when no completed UAVs"
    )


def test_student_summary_without_requests_retains_scenario_count(tmp_path: Path) -> None:
    case = {
        "case_id": "dense24_seed101_safe80_original_k3",
        "scenario": "dense24",
        "seed": 101,
        "safe_distance": 80.0,
        "controller": "original_k3",
    }
    record = _base_record(case, case_dir=tmp_path / case["case_id"])
    record["requests"] = 24
    result = _merge_summary_record(
        record,
        {"completed": 12, "outcomes": {}, "counts": {"shield": 0}, "seconds": 300.0},
        {},
        None,
        None,
    )
    assert result["requests"] == 24
    assert result["unresolved"] == 24
