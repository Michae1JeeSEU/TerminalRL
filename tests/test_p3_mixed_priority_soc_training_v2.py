import inspect
import json

import pytest

from terminal_marl.training import (
    run_v4_2_p3_mixed_priority_soc_training_v2 as training,
)


def test_v2_protocol_freezes_repair_binding_and_fresh_schedule():
    value = training.protocol()

    assert value["protocol"] == training.STAGE
    assert value["status"] == "PREREGISTERED_TRAINING_REPAIR"
    assert value["supersedes"] == training.V1_STAGE
    bindings = value["bindings"]
    assert bindings["superseded_training_manifest_sha256"] == training.V1_FAILED_MANIFEST_SHA
    assert bindings["superseded_training_failure"] == {
        "kind": "completion_floor",
        "seed": 818003,
        "scenario": "dense24",
        "formal_episode_index": 3,
        "formal_episode_count": 8,
        "only_failed_gate": "completion_floor",
        "safety_and_integrity_gates_passed": True,
        "update_attempted": False,
        "observed_completion_fraction": 0.5,
        "required_completion_fraction": 0.75,
        "heldout_seeds_already_reserved": [819001, 819002, 819003, 819004, 819005, 819006],
    }
    assert value["entry"]["seeds"] == [820001, 820002]
    assert value["entry"]["scenarios"] == ["dense24", "flow6"]
    assert value["entry"]["steps"] == [256, 160]
    assert value["formal"]["seeds"] == [
        821001, 821002, 821003, 821004,
        821005, 821006, 821007, 821008,
    ]
    assert value["formal"]["scenarios"] == [
        "dense24", "flow6", "dense24", "flow12",
        "dense24", "flow6", "dense24", "flow12",
    ]
    assert value["heldout"]["seeds"] == [822001, 822002, 822003, 822004, 822005, 822006]
    assert value["heldout"]["untouched"]


def test_v2_reuses_exact_sparse_and_dense_scenario_semantics():
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
    configured, steps = training._scenario_horizon(
        dense, "dense24", stage="formal", index=0, p=value,
    )
    assert configured.max_simulation_seconds == 720.0
    assert steps == int(720.0 / configured.dt)


def test_formal_coverage_recomputes_from_lifecycle_rows():
    value = training.protocol()
    attributes = [
        {
            "priorities": [99],
            "initial_soc": [0.99],
            "rows": [
                {"base_priority": 1, "initial_soc": 0.55},
                {"base_priority": 3, "initial_soc": 0.72},
                {"base_priority": 5, "initial_soc": 0.90},
            ],
        }
    ]

    coverage = training._coverage(attributes, value)
    assert coverage["priorities_observed"] == [1, 3, 5]
    assert coverage["priority_coverage"]
    assert coverage["soc_count"] == 3
    assert coverage["soc_min"] == 0.55
    assert coverage["soc_max"] == 0.90
    assert coverage["soc_in_bounds"]
    assert coverage["soc_range_observed"]


def test_formal_coverage_fails_closed_for_missing_or_nonfinite_row_fields():
    value = training.protocol()
    with pytest.raises(ValueError):
        training._coverage([{"rows": [{"base_priority": 1}]}], value)
    with pytest.raises(ValueError):
        training._coverage(
            [{"rows": [{"base_priority": 1, "initial_soc": float("nan")}]}],
            value,
        )


def test_v1_failed_manifest_validation_is_exact_and_fail_closed(tmp_path, monkeypatch):
    value = training.protocol()
    path = tmp_path / "v1.json"
    valid = {
        "stage": training.V1_STAGE,
        "PASS": False,
        "HELDOUT_EVALUATION_ALLOWED": False,
        "heldout_untouched": True,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "formal_training_called": True,
        "entry": {"PASS": True},
        "formal": {
            "PASS": False,
            "final_checkpoint": None,
            "final_checkpoint_sha256": None,
            "gates": {"all_episodes": False, "all_episode_gates": False},
        },
    }
    path.write_text(json.dumps(valid), encoding="utf-8")
    monkeypatch.setattr(training, "_sha256", lambda _: training.V1_FAILED_MANIFEST_SHA)
    assert training.validate_superseded_v1_manifest(path, value)["PASS"] is False

    valid["entry"]["PASS"] = False
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(PermissionError):
        training.validate_superseded_v1_manifest(path, value)


def test_completion_is_report_only_and_absent_from_training_gates():
    value = training.protocol()
    assert "completion_floor_by_scenario" not in value["hard_gates"]
    assert value["completion_reporting"]["mode"] == "report_only"
    assert value["completion_reporting"]["training_rollout_completion_gate"] is False
    source = inspect.getsource(training._rollout_gates)
    assert "completion_floor" not in source
    assert "all_requested_complete" not in source
    assert "completion_gate" in source


def test_dense_completion_is_retained_as_diagnostic_only():
    value = training.protocol()
    assert value["scenarios"]["dense24"]["training_horizon_seconds"] == 720
    source = inspect.getsource(training._training_stage)
    assert '"completion_fraction": completion' in source
    assert '"completion_report_only": True' in source
    assert "if not all(bool(value) for value in gates.values())" in source


def test_entry_and_formal_independently_start_original_with_fresh_optimizer():
    value = training.protocol()
    entry = training.stage_initialization_contract("entry", value)
    formal = training.stage_initialization_contract("formal", value)

    assert entry["checkpoint_sha256"] == value["bindings"]["candidate_checkpoint_sha256"]
    assert formal["checkpoint_sha256"] == value["bindings"]["candidate_checkpoint_sha256"]
    assert entry["fresh_optimizer"] and formal["fresh_optimizer"]
    assert entry["entry_updates_discarded"]
    assert formal["formal_initialization_is_original"]
    assert value["bindings"]["entry_checkpoints_formal_initialization"] == "forbidden"


def test_source_hashes_bind_v2_runner_and_frozen_scheduler():
    hashes = training.source_hashes()
    assert "terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v2.py" in hashes
    assert hashes["terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v2.yaml"] == training.PROTOCOL_SHA
    assert hashes["terminal_marl/core/merge_scheduler.py"] == training.SCHEDULER_SOURCE_SHA
    assert all(not key.startswith(("/", "\\")) and ":" not in key for key in hashes)
    assert all(len(value) == 64 for value in hashes.values())


def test_action_tape_adapter_accepts_student_and_rule_result_shapes():
    assert training._action_tape_path({"summary": {"action_tape": "student.json"}}) == "student.json"
    assert training._action_tape_path({"action_tape": "rule.json"}) == "rule.json"


def test_root_manifest_records_entry_ppo_even_if_formal_is_not_admitted():
    source = inspect.getsource(training.run)
    assert '"PPO_called": True' in source
    assert '"formal_training_called": formal is not None' in source
