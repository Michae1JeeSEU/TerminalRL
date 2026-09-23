from pathlib import Path

import pytest

from terminal_marl.training.diagnose_emergency_energy_v2_pilot import (
    PROTOCOL_PATH,
    REPAIR_PROTOCOL_PATH,
    ROUTE_REPAIR_PROTOCOL_PATH,
    ROUTE_REPAIR_PROTOCOL_SHA256,
    _sha256,
    aggregate_records,
    build_config,
    case_matrix,
    load_protocol,
    run_matrix,
)


def test_protocol_and_config_bindings_are_frozen():
    protocol = load_protocol(PROTOCOL_PATH)
    scenario = protocol["scenario"]

    assert tuple(protocol["matrix"]["seeds"]) == (840001, 840002)
    assert tuple(scenario["emergency_trigger_soc_values"]) == (0.35, 0.40)
    assert scenario["fixed_arrival_schedule"] == [0, 0, 0, 0, 0, 0, 15, 30, 45, 60, 75, 90]
    assert len(case_matrix(protocol)) == 4

    config = build_config(protocol, 0.40)
    assert config.total_requested_uavs == 12
    assert config.warm_start_uavs == 6
    assert config.fixed_arrival_schedule == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
    assert config.max_simulation_seconds == 1200.0
    assert config.initial_soc_mode == "uniform"
    assert (config.initial_soc_min, config.initial_soc_max) == (0.35, 0.55)
    assert config.energy_model_mode == "phase_motion_v2"
    assert config.emergency_landing_enabled is True
    assert config.emergency_trigger_soc == 0.40
    assert config.lz_assignment_mode == "nearest_available"
    assert config.release_arbitration_mode == "downstream_priority"
    assert config.final_queue_capacity == 3
    assert (config.safe_distance, config.nmac_distance, config.collision_distance) == (80.0, 40.0, 10.0)
    assert config.high_level_enabled is False
    assert config.teacher_enabled is False


def test_short_smoke_uses_non_dev_seed_and_emits_json_records(tmp_path: Path):
    output = tmp_path / "smoke"
    manifest = run_matrix(output, seeds=(840000,), triggers=(0.35, 0.40), max_steps=2)

    assert {record["seed"] for record in manifest["records"]} == {840000}
    assert {record["emergency_trigger_soc"] for record in manifest["records"]} == {0.35, 0.40}
    assert manifest["PPO_ALLOWED"] is False
    assert manifest["training_called"] is False
    assert manifest["teacher_used"] is False
    assert manifest["checkpoint_used"] is False
    assert len(manifest["records"]) == 2
    assert (output / "manifest.json").is_file()
    assert all((output / record["case_id"] / "summary.json").is_file() for record in manifest["records"])


def test_runner_refuses_to_overwrite_existing_output(tmp_path: Path):
    output = tmp_path / "already_exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_matrix(output, seeds=(840000,), triggers=(0.35,), max_steps=1)


def test_repair_protocol_binds_fresh_seeds_and_explicit_energy_gate(tmp_path: Path):
    protocol = load_protocol(REPAIR_PROTOCOL_PATH)
    assert protocol["matrix"]["seeds"] == [840101, 840102]
    assert protocol["scenario"]["emergency_trigger_soc_values"] == [0.40]
    config = build_config(protocol, 0.40)
    assert config.phase_motion_nominal_solo_energy == pytest.approx(240.9)

    manifest = run_matrix(tmp_path / "repair_smoke", protocol_path=REPAIR_PROTOCOL_PATH,
                          seeds=(840100,), triggers=(0.40,), max_steps=2)
    assert manifest["records"][0]["seed"] == 840100
    assert "zero_in_terminal_low_soc_diversion" in manifest["aggregate"]["hard_gates"]
    assert manifest["source_sha256"]


def test_route_repair_protocol_binds_fresh_seeds_and_route_repair_fields(tmp_path: Path):
    assert _sha256(ROUTE_REPAIR_PROTOCOL_PATH) == ROUTE_REPAIR_PROTOCOL_SHA256
    protocol = load_protocol(ROUTE_REPAIR_PROTOCOL_PATH)
    assert protocol["protocol"] == "EMERGENCY_ENERGY_V2_ROUTE_REPAIR_PILOT"
    assert protocol["status"] == "PREREGISTERED_ROUTE_REPAIR_DIAGNOSTIC"
    assert protocol["repair"]["predecessor_protocols"] == [
        "EMERGENCY_ENERGY_V2_PILOT", "EMERGENCY_ENERGY_V2_REPAIR_PILOT"
    ]
    assert protocol["repair"]["predecessor_seeds_consumed"] == [840001, 840002, 840101, 840102]
    assert protocol["repair"]["phase_motion_nominal_solo_energy"] == pytest.approx(240.9)
    assert protocol["repair"]["touchdown_reserve_soc"] == pytest.approx(0.25)
    assert protocol["repair"]["admission_soc_floor_at_capacity_1500"] == pytest.approx(0.4106)
    assert protocol["repair"]["active_low_soc_outcome_must_be_explicit"] is True
    assert protocol["repair"]["continuous_tangential_emergency_departure"] is True
    assert protocol["scenario"]["emergency_trigger_soc_values"] == [0.40]
    assert protocol["matrix"]["seeds"] == [840201, 840202]
    assert protocol["matrix"]["expected_records"] == 2
    assert len(case_matrix(protocol)) == 2

    config = build_config(protocol, 0.40)
    assert config.phase_motion_nominal_solo_energy == pytest.approx(240.9)
    assert config.energy_model_mode == "phase_motion_v2"
    assert config.emergency_landing_enabled is True
    assert config.high_level_enabled is False
    assert config.teacher_enabled is False

    # This smoke uses a non-formal seed and a one-step bound; formal route-repair
    # seeds are not consumed by the test.
    manifest = run_matrix(tmp_path / "route_repair_smoke", protocol_path=ROUTE_REPAIR_PROTOCOL_PATH,
                          seeds=(840200,), triggers=(0.40,), max_steps=1)
    assert manifest["protocol"] == "EMERGENCY_ENERGY_V2_ROUTE_REPAIR_PILOT"
    assert manifest["records"][0]["seed"] == 840200
    assert "all_records_terminal" in manifest["aggregate"]["hard_gates"]


def test_route_repair_keeps_legacy_protocols_loadable():
    assert load_protocol(PROTOCOL_PATH)["protocol"] == "EMERGENCY_ENERGY_V2_PILOT"
    assert load_protocol(REPAIR_PROTOCOL_PATH)["protocol"] == "EMERGENCY_ENERGY_V2_REPAIR_PILOT"


def test_aggregate_hard_gates_include_emergency_landing_and_failed_energy():
    safe = {
        "exact_request_accounting": True,
        "outcome_counts": {"completed": 11, "emergency_landed": 1},
        "safety_counters": {"collisions": 0, "nmac": 0, "hard_separation_violations": 0, "SafetyInvariantError": 0},
    }
    passed = aggregate_records([safe], expected_records=1)
    assert passed["PASS"] is True
    assert passed["hard_gates"]["at_least_one_emergency_landed"] is True

    failed_energy = dict(safe, outcome_counts={"completed": 10, "emergency_landed": 1, "failed_energy": 1})
    rejected = aggregate_records([failed_energy], expected_records=1)
    assert rejected["PASS"] is False
    assert rejected["hard_gates"]["zero_failed_energy"] is False
