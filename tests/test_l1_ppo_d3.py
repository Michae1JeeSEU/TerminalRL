"""Pure acceptance-rule coverage for the bounded-distribution A/B report."""
from __future__ import annotations

from terminal_marl.training.l1_ppo_d3_bounded_distribution import _comparison


def _variant(*, exact: float, near: float) -> dict:
    return {
        "saturation": {
            head: {"exact_bound_rate_by_update": [exact] * 3, "near_boundary_rate_by_update": [near] * 3}
            for head in ("release_request", "speed_residual", "merge_time_residual")
        },
        "teacher_burden_overall": 5.0,
        "correction_p95": .3,
        "shield_burden": .1,
        "ppo_health": {"healthy": True},
        "safety_passed": True,
        "provenance_passed": True,
    }


def test_d3_allows_log_std_entropy_only_after_exact_pile_is_repaired() -> None:
    result = _comparison(_variant(exact=.2, near=.3), _variant(exact=0.0, near=.2))
    assert result["B_passed"]
    assert result["allow_log_std_entropy_ablation"]


def test_d3_rejects_a_bounded_arm_with_worse_teacher_burden() -> None:
    control = _variant(exact=.2, near=.3)
    bounded = _variant(exact=0.0, near=.05)
    bounded["shield_burden"] = .2
    result = _comparison(control, bounded)
    assert not result["B_passed"]
    assert not result["allow_log_std_entropy_ablation"]
