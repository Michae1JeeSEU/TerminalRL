"""Focused construction and serialization checks for the solo SOC diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from terminal_marl.training import diagnose_v4_2_p3_solo_soc_floor_v1 as diagnostic


def _synthetic_record(
    case_id: str,
    required_soc: float,
    *,
    full_touchdown: bool = True,
    floor_touchdown: bool = True,
    safety_violation: bool = False,
) -> dict:
    safety = {"collisions": 1} if safety_violation else {"collisions": 0}
    return {
        "case_id": case_id,
        "required_initial_soc": required_soc,
        "pass": True,
        "full_soc_1p0": {
            "touchdown": full_touchdown,
            "safety_violation": safety_violation,
            "safety_counters": safety,
        },
        "fixed_soc_0p55": {
            "touchdown": floor_touchdown,
            "safety_violation": safety_violation,
            "safety_counters": safety,
        },
    }


def test_case_matrix_enumerates_gate_mission_cartesian_product() -> None:
    cases = diagnostic.build_case_matrix()

    assert len(cases) == 16
    assert (cases[0]["gate_index"], cases[0]["mission_type"]) == (0, "standard")
    assert (cases[-1]["gate_index"], cases[-1]["mission_type"]) == (3, "surveillance")
    assert {
        (row["gate_index"], row["mission_index"])
        for row in cases
    } == {(gate, mission) for gate in range(4) for mission in range(4)}


def test_solo_config_uses_fixed_arrival_soc_and_p3r_resources() -> None:
    raw = diagnostic.load_p3_v4_protocol()
    config = diagnostic.scenario_config(2, 1, initial_soc=0.55, raw_protocol=raw)

    assert config.reset_mode == "flow_from_empty"
    assert config.total_requested_uavs == 1
    assert config.fixed_arrival_schedule == (10.0,)
    assert config.initial_soc_mode == "fixed"
    assert config.initial_soc == pytest.approx(0.55)
    assert config.gate_probabilities == (0.0, 0.0, 1.0, 0.0)
    assert config.mission_probabilities == (0.0, 1.0, 0.0, 0.0)
    assert config.safe_distance == 80.0
    assert config.nmac_distance == 40.0
    assert config.collision_distance == 10.0
    assert config.energy_capacity == 1500.0
    assert config.lz_assignment_mode == "nearest_available"
    assert config.release_arbitration_mode == "downstream_priority"
    assert config.final_queue_capacity == 3


def test_aggregate_reports_max_required_soc_and_margin() -> None:
    result = diagnostic.aggregate_records(
        [
            _synthetic_record("a", 0.41),
            _synthetic_record("b", 0.52),
        ],
        expected_case_count=2,
    )

    assert result["max_required_initial_soc"] == pytest.approx(0.52)
    assert result["margin_above_max_required_soc"] == pytest.approx(0.03)
    assert result["gates"]["configured_floor_sufficient"] is True
    assert result["PASS"] is True


def test_aggregate_requires_every_case_pass_flag() -> None:
    record = _synthetic_record("bad_pass", 0.50)
    record["pass"] = False

    result = diagnostic.aggregate_records([record], expected_case_count=1)

    assert result["gates"]["all_case_pass_flags"] is False
    assert result["PASS"] is False


def test_aggregate_fails_when_expected_case_is_missing() -> None:
    result = diagnostic.aggregate_records(
        [_synthetic_record("one", 0.50)], expected_case_count=2
    )

    assert result["gates"]["expected_case_count"] is False
    assert result["PASS"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"full_touchdown": False},
        {"floor_touchdown": False},
        {"safety_violation": True},
    ],
)
def test_aggregate_fails_closed_for_touchdown_or_safety_failure(kwargs: dict) -> None:
    result = diagnostic.aggregate_records([_synthetic_record("bad", 0.50, **kwargs)])

    assert result["PASS"] is False


def test_run_writes_manifest_and_progress_with_synthetic_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_case(case, *, raw_protocol, output_dir):
        del raw_protocol, output_dir
        return _synthetic_record(str(case["case_id"]), 0.50)

    monkeypatch.setattr(diagnostic, "_run_rule_case", fake_run_case)
    output = tmp_path / "solo_soc"
    manifest = diagnostic.run(output_dir=output, max_cases=2, seed=12)

    assert manifest["PASS"] is True
    assert manifest["aggregate"]["case_count"] == 2
    assert (output / "progress.json").is_file()
    manifest_path = output / "manifest.json"
    assert manifest_path.is_file()
    serialized = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert serialized["stage"] == diagnostic.STAGE
    assert serialized["max_cases"] == 2
    assert serialized["aggregate"]["max_required_initial_soc"] == pytest.approx(0.50)
