from terminal_marl.training.run_v4_2_v2_1_formal_i0_r2 import (
    ANCHOR_MINIMA,
    CONTEXT_EPISODE_MINIMA,
    CONTEXT_MINIMA,
    STRATA,
    _make_transition_proof,
    _verify_protocol,
)


def test_frozen_protocol_matches_collector_constants() -> None:
    _, result = _verify_protocol()
    assert result["PASS"]
    assert [item.name for item in STRATA] == [
        "post_wait_release", "retry_expired_transition_release",
        "conflict_recovery_release", "high_traffic_proposal11",
        "middle_interface_proposal14", "post_wait_proposal14",
        "post_conflict_proposal14", "nominal_mixed", "clear_outer",
        "resource_pressure",
    ]
    assert CONTEXT_MINIMA["post_WAIT_RELEASE"] == 60
    assert CONTEXT_EPISODE_MINIMA["post_conflict_proposal_14"] == 8
    assert ANCHOR_MINIMA["proposal_13_RELEASE"] == 160


def test_retry_wait_release_proof_requires_same_lifecycle() -> None:
    row = {
        "uav_id": 7, "lifecycle_event_id": "life-1", "teacher_gate": 1,
        "full_state_hash": "current", "headway_state": {"interface": "o2m"},
    }
    previous = {
        "uid": 7, "lifecycle_id": "life-1", "teacher_gate": 0,
        "wait_rationale": "retry", "state_hash": "previous", "time": 10.0,
        "retry_active": True, "interface_open": True,
        "conflict_state": False, "headway_state": {"interface": "o2m"},
    }
    current = {
        "time": 11.0, "retry_active": False, "interface_open": True,
        "conflict_state": False,
    }
    proof = _make_transition_proof(row, previous, current)
    assert proof["wait_release_transition_kind"] == "retry_WAIT_to_RELEASE"
    assert proof["same_uav_id"] and proof["same_lifecycle_event_id"]
    assert proof["previous_wait_state_hash"] == "previous"

    row["lifecycle_event_id"] = "life-2"
    broken = _make_transition_proof(row, previous, current)
    assert broken["wait_release_transition_kind"] is None
    assert not broken["same_lifecycle_event_id"]


def test_interface_clear_is_counted_as_transition_not_generic_conflict() -> None:
    row = {
        "uav_id": 4, "lifecycle_event_id": "life-4", "teacher_gate": 1,
        "full_state_hash": "released", "headway_state": {"interface": "m2i"},
    }
    previous = {
        "uid": 4, "lifecycle_id": "life-4", "teacher_gate": 0,
        "wait_rationale": "geometry_conflict", "state_hash": "waiting",
        "time": 20.0, "retry_active": False, "interface_open": False,
        "conflict_state": True, "headway_state": {"interface": "m2i"},
    }
    current = {"time": 21.0, "retry_active": False,
               "interface_open": True, "conflict_state": False}
    proof = _make_transition_proof(row, previous, current)
    assert proof["wait_release_transition_kind"] == "transition_WAIT_to_RELEASE"
