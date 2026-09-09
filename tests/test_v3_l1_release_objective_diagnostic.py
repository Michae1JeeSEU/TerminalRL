import numpy as np

from terminal_marl.training.run_v3_l1_release_objective_diagnostic import _aliasing_audit, _bc_distribution, _events, _streaks


def _row(tick, *, requested=1, target=0, used=True, eligible=True):
    return {"update": 1, "episode": 0, "uid": "7", "tick": tick, "phase": 1,
            "observation": np.full(87, tick, dtype=float).tolist(), "observation_hash": str(tick),
            "requested": requested, "native_teacher": target, "all_event_teacher": target,
            "teacher_used": used, "teacher_action_present": True, "native_release_corrected": used and requested != target,
            "eligible": eligible}


def test_streak_audit_counts_persistent_r2w_as_one_event():
    rows = [_row(1), _row(2), _row(3), _row(6), _row(7)]
    streaks, audit = _streaks(rows, (1, 0))
    assert [len(value) for value in streaks] == [3, 2]
    assert audit["raw_tick_corrections"] == 5
    assert audit["unique_decision_events"] == 2
    assert audit["duplication_factor"] == 2.5


def test_current_bc_gate_and_event_deduplication_are_explicit():
    rows = [_row(1), _row(2), _row(3), _row(4, requested=0, target=1),
            _row(5, requested=0, target=0, used=True)]
    bc = _bc_distribution(rows)
    assert bc["current_release_CE_gate"] == "teacher_used && teacher_action is not None"
    assert bc["labels"] == {"RELEASE_to_WAIT": 3, "WAIT_to_RELEASE": 1, "correct_WAIT": 1, "correct_RELEASE": 0}
    assert not bc["CORRECTION_ONLY_BINARY_SUPERVISION"]
    assert len(_events(rows, deduplicate=False)) == 5
    assert len(_events(rows, deduplicate=True)) == 3


def test_aliasing_audit_reports_exact_opposite_label_hashes():
    left, right = _row(1), _row(2, requested=0, target=1)
    left.update(label=0, observation_hash="shared")
    right.update(label=1, observation_hash="shared")
    audit = _aliasing_audit([left, right])
    assert audit["opposite_label_exact_hashes"] == 1
    assert audit["nearest_opposite_label_l2"]["count"] == 2
