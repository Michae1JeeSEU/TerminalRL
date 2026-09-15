import pytest

from terminal_marl.training.run_v4_2_v2_1_i0_r2_support_closure_attribution import (
    _entropy,
    _groups,
    _tv,
)


def test_total_variation_uses_normalized_categorical_counts() -> None:
    assert _tv({"12": 63}, {"12": 17, "5": 2, "6": 3}) == pytest.approx(5 / 22)


def test_gate_entropy_is_one_at_balanced_boundary() -> None:
    assert _entropy(0.5) == 1.0
    assert _entropy(0.0) == 0.0
    assert _entropy(1.0) == 0.0


def test_group_membership_keeps_family_and_proposal_views() -> None:
    row = {
        "teacher_joint_index": 14,
        "collection_stratum": "transition_retry",
        "family_flags": {"post_wait": True, "post_conflict": False},
    }
    assert {"all_RELEASE", "post_WAIT", "transition_retry", "proposal14"} <= _groups(row)
