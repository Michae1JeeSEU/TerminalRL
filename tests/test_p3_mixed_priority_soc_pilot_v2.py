import pytest

from terminal_marl.training import diagnose_v4_2_p3_mixed_priority_soc_pilot_v2 as p3
from terminal_marl.training import diagnose_v4_2_p3_mixed_priority_soc_pilot_v3 as p3v3
from terminal_marl.training import diagnose_v4_2_p3_mixed_priority_soc_pilot_v4 as p3v4


def _protocol_without_hash_check():
    import yaml
    return yaml.safe_load(p3.PROTOCOL.read_text(encoding="utf-8"))


def test_sparse_regression_scenarios_start_empty_with_frozen_arrival_flow():
    value = _protocol_without_hash_check()
    flow6 = p3.scenario_config("flow6", value)
    flow12 = p3.scenario_config("flow12", value)

    assert flow6.reset_mode == "flow_from_empty"
    assert flow6.total_requested_uavs == 6
    assert flow6.fixed_arrival_schedule == tuple(10.0 + 60.0 * index for index in range(6))
    assert flow6.max_simulation_seconds == 1210.0
    assert flow12.total_requested_uavs == 12
    assert flow12.fixed_arrival_schedule[-1] == 670.0
    assert flow12.max_simulation_seconds == 1570.0


def test_dense24_starts_twelve_uavs_and_uses_p2r_resource_semantics():
    value = _protocol_without_hash_check()
    config = p3.scenario_config("dense24", value)

    assert config.reset_mode == "warm_start_rings"
    assert config.warm_start_uavs == 12
    assert config.warm_start_steady_motion
    assert config.total_requested_uavs == 24
    assert config.fixed_arrival_schedule[:12] == (0.0,) * 12
    assert config.fixed_arrival_schedule[12:] == tuple(15.0 * index for index in range(1, 13))
    assert config.max_simulation_seconds == 1080.0
    assert config.lz_assignment_mode == "nearest_available"
    assert config.release_arbitration_mode == "downstream_priority"
    assert config.final_queue_capacity == 3


def test_all_p3_v2_scenarios_enable_priority_and_soc_heterogeneity():
    value = _protocol_without_hash_check()
    for scenario in ("flow6", "flow12", "dense24"):
        config = p3.scenario_config(scenario, value)
        assert config.priority_mode == "random_135"
        assert config.initial_priority is None
        assert config.initial_soc_mode == "uniform"
        assert config.initial_soc_min == 0.55
        assert config.initial_soc_max == 0.90
        assert config.runtime_proposal_shortlist_k == 3
        assert not config.teacher_enabled


def test_lifecycle_summary_preserves_priority_and_low_soc_metrics():
    rows = [
        {"uid": 0, "base_priority": 1, "initial_soc": 0.60, "landing_soc": 0.45,
         "wait_seconds": 30.0, "request_to_touchdown_seconds": 130.0},
        {"uid": 1, "base_priority": 3, "initial_soc": 0.70, "landing_soc": 0.55,
         "wait_seconds": 20.0, "request_to_touchdown_seconds": 120.0},
        {"uid": 2, "base_priority": 5, "initial_soc": 0.90, "landing_soc": 0.75,
         "wait_seconds": 10.0, "request_to_touchdown_seconds": 110.0},
    ]

    result = p3.summarize_lifecycle_rows(rows, low_soc_threshold=0.65)

    assert result["priorities_observed"] == [1, 3, 5]
    assert result["by_priority"]["1"]["mean_wait_seconds"] == 30.0
    assert result["by_priority"]["5"]["mean_request_to_touchdown_seconds"] == 110.0
    assert result["priority_weighted_wait_seconds"] == 140.0 / 9.0
    assert result["low_soc"] == {
        "threshold": 0.65,
        "count": 1,
        "mean_wait_seconds": 30.0,
        "mean_landing_soc": 0.45,
        "minimum_landing_soc": 0.45,
    }


def test_v3_action_tape_adapter_handles_student_and_rule_result_shapes():
    assert p3v3.action_tape_path({"summary": {"action_tape": "student.json"}}) == "student.json"
    assert p3v3.action_tape_path({"action_tape": "rule.json"}) == "rule.json"
    assert p3v3.action_tape_path({"summary": {}, "action_tape": "rule.json"}) == "rule.json"
    with pytest.raises(KeyError, match="no action_tape"):
        p3v3.action_tape_path({"summary": {}})


def test_v3_protocol_is_frozen_and_uses_fresh_seeds():
    value = p3v3.protocol()

    assert value["protocol"] == p3v3.STAGE
    assert value["pilot"]["seeds"] == [815001, 815002, 815003]
    assert value["pilot"]["expected_records"] == 6


def test_v4_protocol_binds_repaired_scheduler_and_fresh_seeds():
    value = p3v4.protocol()

    assert value["protocol"] == p3v4.STAGE
    assert value["status"] == "PREREGISTERED_REQUALIFICATION"
    assert value["pilot"]["seeds"] == [816001, 816002, 816003]
    assert value["pilot"]["reservation_marker"] == "p3_mixed_priority_soc_pilot_v4_consumed.json"
    assert value["bindings"]["scheduler_source"] == "terminal_marl/core/merge_scheduler.py"
    assert value["bindings"]["scheduler_source_sha256"] == p3v4.SCHEDULER_SOURCE_SHA
    assert value["authorization"]["PPO_ALLOWED"] is False


def test_v4_integrity_helper_excludes_completion_and_all_but_detects_safety_failure():
    result = {"gates": {
        key: True for key in set(p3v4.CANDIDATE_INTEGRITY_KEYS)
    } | {"complete": False, "all": False, "all_requests_complete": False}}
    shortlist = {"all": True}
    values = p3v4.integrity_gates(
        result, controller="candidate_k3", runtime_shortlist=shortlist
    )

    assert "complete" not in values
    assert "all" not in values
    assert "all_requests_complete" not in values
    assert p3v4.integrity_passes(values)
    result["gates"]["safety"] = False
    failed = p3v4.integrity_gates(
        result, controller="candidate_k3", runtime_shortlist=shortlist
    )
    assert failed["safety"] is False
    assert not p3v4.integrity_passes(failed)


def test_v4_run_has_no_fail_fast_break_path():
    import inspect

    assert "break" not in inspect.getsource(p3v4.run)


def test_v4_optional_comparison_does_not_crash_on_zero_completion():
    assert p3v4._optional_difference(None, 1.0) is None
    assert p3v4._optional_difference(3.0, 1.0) == 2.0
