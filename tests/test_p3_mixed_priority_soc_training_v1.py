import inspect
import json

import pytest

from terminal_marl.training import (
    run_v4_2_p3_mixed_priority_soc_training_v1 as training,
)


def test_training_protocol_freezes_pilot_binding_and_fresh_seed_schedule():
    value = training.protocol()

    assert value["protocol"] == training.STAGE
    assert value["bindings"]["pilot_stage"] == "V4_2_P3_MIXED_PRIORITY_SOC_PILOT_V4"
    assert value["bindings"]["pilot_manifest_sha256"] == (
        "c094528791475d8748174c2b3c0408d22dc53d0f8e0a1b64029d54e8bb012d0e"
    )
    assert value["entry"]["seeds"] == [817001, 817002]
    assert value["entry"]["scenarios"] == ["flow6", "dense24"]
    assert value["formal"]["seeds"] == [
        818001, 818002, 818003, 818004,
        818005, 818006, 818007, 818008,
    ]
    assert value["formal"]["scenarios"] == [
        "flow6", "flow12", "dense24", "dense24",
        "flow6", "flow12", "dense24", "dense24",
    ]
    assert value["heldout"]["seeds"] == [819001, 819002, 819003, 819004, 819005, 819006]
    assert value["heldout"]["untouched"]


def test_training_scenarios_reuse_exact_v2_sparse_and_dense_semantics():
    value = training.protocol()
    flow6 = training.scenario_config("flow6", value)
    flow12 = training.scenario_config("flow12", value)
    dense = training.scenario_config("dense24", value)

    assert flow6.reset_mode == "flow_from_empty"
    assert flow6.fixed_arrival_schedule == tuple(10.0 + 60.0 * i for i in range(6))
    assert flow6.max_simulation_seconds == 1210.0
    assert flow12.fixed_arrival_schedule[-1] == 670.0
    assert flow12.max_simulation_seconds == 1570.0
    assert dense.reset_mode == "warm_start_rings"
    assert dense.warm_start_uavs == 12
    assert dense.fixed_arrival_schedule[:12] == (0.0,) * 12
    assert dense.fixed_arrival_schedule[12:] == tuple(15.0 * i for i in range(1, 13))
    assert dense.lz_assignment_mode == "nearest_available"
    assert dense.release_arbitration_mode == "downstream_priority"
    assert dense.final_queue_capacity == 3


def test_pilot_manifest_validation_is_fail_closed(tmp_path, monkeypatch):
    value = training.protocol()
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({"stage": training.PILOT_STAGE, "PASS": True}), encoding="utf-8")
    monkeypatch.setattr(
        training,
        "_sha256",
        lambda _: value["bindings"]["pilot_manifest_sha256"],
    )
    with pytest.raises(PermissionError):
        training.validate_pilot_manifest(path, value)


def test_dense_formal_horizon_and_completion_floors_are_preregistered():
    value = training.protocol()
    dense = training.scenario_config("dense24", value)
    configured, steps = training._scenario_horizon(
        dense, "dense24", stage="formal", index=2, p=value,
    )

    assert configured.max_simulation_seconds == 720.0
    assert steps == int(720.0 / configured.dt)
    assert training.completion_floor("flow6", value) == 1.0
    assert training.completion_floor("flow12", value) == 1.0
    assert training.completion_floor("dense24", value) == 0.75
    assert "all_requested_complete" not in value["hard_gates"]


def test_entry_and_formal_start_from_original_with_fresh_optimizer():
    value = training.protocol()
    entry = training.stage_initialization_contract("entry", value)
    formal = training.stage_initialization_contract("formal", value)

    assert entry["checkpoint_sha256"] == value["bindings"]["candidate_checkpoint_sha256"]
    assert formal["checkpoint_sha256"] == value["bindings"]["candidate_checkpoint_sha256"]
    assert entry["fresh_optimizer"] and formal["fresh_optimizer"]
    assert entry["entry_updates_discarded"]
    assert formal["formal_initialization_is_original"]
    assert value["bindings"]["entry_checkpoints_formal_initialization"] == "forbidden"


def test_source_hashes_are_repository_relative_and_bind_scheduler():
    hashes = training.source_hashes()
    assert "terminal_marl/core/merge_scheduler.py" in hashes
    assert "terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v1.py" in hashes
    assert all(not key.startswith(("/", "\\")) and ":" not in key for key in hashes)
    assert all(len(value) == 64 for value in hashes.values())


def test_formal_rollout_has_completion_floor_but_no_dense_full_completion_gate():
    source = inspect.getsource(training._rollout_gates)
    assert "completion_floor" in source
    assert "all_requested_complete" not in source
    assert training.protocol()["hard_gates"]["completion_floor_by_scenario"]["dense24"] == 0.75


def test_terminal_outcomes_fail_closed_but_incomplete_aircraft_are_allowed():
    assert training._no_failed_outcomes({"outcomes": {"1": "completed"}})
    assert training._no_failed_outcomes({"outcomes": {}})
    assert not training._no_failed_outcomes({"outcomes": {"1": "diverted_low_soc"}})
    assert not training._no_failed_outcomes({"outcomes": {"1": "future_failure_kind"}})
