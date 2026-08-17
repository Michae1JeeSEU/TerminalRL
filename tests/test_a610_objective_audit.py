from terminal_marl.training.a610_objective_audit import (
    MATERIALITY_VERSION,
    classify_delta,
    common_unresolved_cohort,
    _local_worker_count,
    ppo_authorization,
)


def _calibration(abs_threshold=10., relative_threshold=.1):
    return {"thresholds": {"unresolved_age_p95": {"epsilon_abs": abs_threshold, "epsilon_rel": relative_threshold}}}


def test_sign_regression_not_automatically_material():
    labels, material = classify_delta({"completed": 1, "unresolved_age_p95": 4.}, _calibration(), {"unresolved_age_p95": 100.})
    assert labels["unresolved_age_p95"] == "SIGN_REGRESSION"
    assert not material


def test_material_countermetric_blocks_admissible_gain():
    labels, material = classify_delta({"completed": 1, "unresolved_age_p95": 20.}, _calibration(), {"unresolved_age_p95": 100.})
    assert labels["unresolved_age_p95"] == "MATERIAL_REGRESSION"
    assert material


def test_material_threshold_is_versioned():
    assert MATERIALITY_VERSION.startswith("a610-v")


def test_completed_is_primary_gain():
    labels, material = classify_delta({"completed": 1}, {"thresholds": {}}, {})
    assert labels["completed"] == "IMPROVED" and not material


def test_common_unresolved_cohort_metrics():
    rows = common_unresolved_cohort({"noop": {"a": 10., "b": 30.}, "inner": {"a": 12., "b": 32., "c": 5.}})
    assert {row["common_unresolved_count"] for row in rows} == {2}
    assert {row["p95"] for row in rows} == {29., 31.}


def test_objective_classification_is_deterministic():
    delta = {"completed": 1, "unresolved_age_p95": 4.}
    first = classify_delta(delta, _calibration(), {"unresolved_age_p95": 100.})
    assert first == classify_delta(delta, _calibration(), {"unresolved_age_p95": 100.})


def test_a610_cannot_authorize_ppo():
    assert ppo_authorization() == "STOP"


def test_local_worker_cap_is_explicit_and_deterministic(monkeypatch):
    monkeypatch.setenv("A610_MAX_WORKERS", "6")
    assert _local_worker_count(18) == 6


def test_no_scalar_reward_introduced():
    source = open("terminal_marl/training/a610_objective_audit.py", encoding="utf-8").read().lower()
    assert "ppo" not in source or "never trains ppo" in source
    assert "reward weights" not in source
