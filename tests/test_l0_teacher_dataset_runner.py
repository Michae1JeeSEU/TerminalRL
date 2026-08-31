from __future__ import annotations

import json

import pytest

from terminal_marl.training.run_l0_teacher_dataset import (
    L0_EMPTY_EPISODES,
    L0_WARM_EPISODES,
    _balanced_rows,
    l0_cohorts,
    l0_minor_correction_probe,
    load_frozen_mu,
    validate_l0_warm_starts,
)


def test_l0_matrix_is_fixed_teacher_nominal_and_65_percent_warm_start():
    cohorts = l0_cohorts(0.003)
    assert sum(L0_WARM_EPISODES) == 13 and sum(L0_EMPTY_EPISODES) == 7
    assert sum(cohort.episodes for cohort in cohorts) == 20
    assert sum(cohort.episodes for cohort in cohorts if cohort.config.reset_mode == "warm_start_rings") / 20 == pytest.approx(0.65)
    for cohort in cohorts:
        config = cohort.config
        assert config.experiment_mode == "fixed_teacher_heavy"
        assert config.scheduler_mode == config.teacher_mode == "teacher_full"
        assert config.teacher_enabled and config.shield_enabled and not config.high_level_enabled
        assert config.mission_probabilities == (1.0, 0.0, 0.0, 0.0)
        assert 5 <= config.total_requested_uavs <= 15


def test_l0_warm_starts_are_safe_and_frozen_c0_is_required(tmp_path):
    report = tmp_path / "c0.json"
    report.write_text(json.dumps({"frozen_capacity": {"frozen": True, "mu_hat": 0.003}}), encoding="utf-8")
    assert load_frozen_mu(report) == pytest.approx(0.003)
    checks = validate_l0_warm_starts(l0_cohorts(0.003), seed=1001)
    assert len(checks) == 3 and all(row["passed"] for row in checks)
    report.write_text(json.dumps({"frozen_capacity": {"frozen": False, "mu_hat": None}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="C0"):
        load_frozen_mu(report)


def test_l0_balancing_retains_required_correction_and_release_coverage():
    def row(*, corrected, magnitude=0.0, kind=None, release="plan_active"):
        return {"teacher_used": corrected, "correction_magnitude": magnitude, "teacher_kind": kind, "release_decision": release}
    rows = [
        row(corrected=True, magnitude=0.1, kind="speed_repair"),
        row(corrected=True, magnitude=1.0, kind="merge_time_repair"),
        row(corrected=False, release="approved"),
        row(corrected=False, release="safety_wait"),
        *[row(corrected=False) for _ in range(40)],
    ]
    balanced, coverage = _balanced_rows(rows, target_fraction=0.2, seed=7)
    assert len(balanced) == 10 and coverage["minor_corrections"] == coverage["major_corrections"] == 1
    assert coverage["release_decision_counts"]["approved"] == coverage["release_decision_counts"]["safety_wait"] == 1
    assert l0_minor_correction_probe(0.003).max_steps == 10
