from __future__ import annotations

import numpy as np

from terminal_marl.training import run_v4_2_v2_1_formal_i0_r3 as r3


def _row(*, joint: int, n: int, phase: str, interface: str, families=()):
    return {
        "teacher_gate": int(joint > 0),
        "teacher_joint_index": joint,
        "teacher_rationale": "certified_release" if joint else "other_wait",
        "retry_active": False,
        "state_snapshot": {"active_uavs": [{}] * n, "target_uav": {"phase": phase}},
        "headway_state": {"interface": interface},
        "family_flags": {name: name in families for name in ("clear_outer", "middle_interface", "post_wait", "post_conflict")},
        "collection_stratum": "test",
        "transition_ledger": {"wait_to_release_proof": {}},
        "lifecycle_event_id": "life",
        "episode_id": "episode",
    }


def test_r3_protocol_hash_and_schedule_are_frozen() -> None:
    protocol, check = r3._verify_protocol()
    assert check["PASS"]
    assert protocol["collection"]["strata"] == [item.name for item in r3.STRATA]
    assert len(r3.STRATA) == 20


def test_r3_context_cells_use_actual_state_and_teacher_outcome() -> None:
    predicates = r3._cell_predicates()
    assert predicates["C01"](_row(joint=11, n=12, phase="ring_outer", interface="o2m", families=("clear_outer",)))
    assert predicates["C05"](_row(joint=14, n=5, phase="ring_outer", interface="o2m", families=("clear_outer",)))
    assert predicates["C09"](_row(joint=13, n=12, phase="ring_middle", interface="m2i", families=("middle_interface",)))
    assert not predicates["C05"](_row(joint=13, n=5, phase="ring_outer", interface="o2m", families=("clear_outer",)))


def test_radius_coverage_enforces_candidate_semantics() -> None:
    reference = [{**_row(joint=14, n=5, phase="ring_outer", interface="o2m", families=("clear_outer",)), "observation_v2_1": [0.0] * 88}]
    wrong = {**_row(joint=13, n=5, phase="ring_outer", interface="o2m", families=("clear_outer",)), "observation_v2_1": [0.0] * 88}
    right = {**_row(joint=14, n=5, phase="ring_outer", interface="o2m", families=("clear_outer",)), "observation_v2_1": (np.ones(88) * 0.01).tolist()}
    result = r3._radius_coverage(reference, [wrong, right], candidate_mode="same_proposal")
    assert result["coverage_0.25"] == 1.0
    assert result["PASS"]


def test_r3_lock_constants_keep_future_stages_disabled() -> None:
    assert r3.TRAIN_RANGE == (410001, 419999)
    assert r3.AUDIT_RANGE == (420001, 429999)
    assert r3.Q2_RESERVED == (430001, 439999)
    assert r3.TRAIN_CEILING == 4200
    assert r3.AUDIT_CEILING == 1400
