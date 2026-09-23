from __future__ import annotations

from types import SimpleNamespace

from terminal_marl.core.config import SimConfig
from terminal_marl.core.runtime_traffic_safety import (
    JointSafetyAssessment,
    PredictiveSafetyShield,
)


def _key(trial: dict[int, float]) -> tuple[tuple[int, float], ...]:
    return tuple((uid, float(trial[uid])) for uid in sorted(trial))


def test_backup_dedup_preserves_old_choice_and_minimum(monkeypatch) -> None:
    config = SimConfig()
    shield = PredictiveSafetyShield(config)
    env = SimpleNamespace()
    residuals = {1: 0.2, 2: -0.3}
    pairs = ((1, 2),)
    baseline = {1: -0.4, 2: -0.4}
    candidates = [
        {1: 0.0, 2: 0.0},
        {2: 0.0, 1: 0.0},
        {1: 0.4, 2: 0.0},
        {1: 0.4, 2: 0.0},
    ]
    minimums = {
        _key(baseline): 75.0,
        _key(candidates[0]): 82.0,
        _key(candidates[2]): 91.0,
    }
    calls: list[tuple[tuple[tuple[int, float], ...], dict[str, object]]] = []

    monkeypatch.setattr(
        shield,
        "_headway_backup_request",
        lambda _env, _residuals, _pairs: dict(baseline),
    )
    monkeypatch.setattr(
        shield,
        "_ring_profile_backup_candidates",
        lambda _env, _residuals, _pairs: [dict(item) for item in candidates],
    )
    monkeypatch.setattr(shield, "_components", lambda _edges: ())

    def assess(_env, trial, **kwargs):
        trial_key = _key(trial)
        calls.append((trial_key, dict(kwargs)))
        return JointSafetyAssessment(True, minimums[trial_key], ())

    monkeypatch.setattr(shield, "_joint_assessment", assess)

    choice, assessment = shield._backup_safe(env, residuals, pairs, threshold=79.0)

    expected_certified = []
    for trial in [*candidates, baseline]:
        verdict = JointSafetyAssessment(True, minimums[_key(trial)], ())
        correction = sum(
            abs(trial.get(uid, residuals.get(uid, 0.0)) - residuals.get(uid, 0.0))
            for uid in sorted(residuals)
        )
        action_key = tuple(
            round(trial.get(uid, 0.0) / config.max_speed_residual_mps, 8)
            for uid in sorted(residuals)
        )
        expected_certified.append((
            (-verdict.minimum_separation, correction, action_key),
            trial,
            verdict,
        ))
    expected_certified.sort(key=lambda item: item[0])
    _, expected_choice, expected_assessment = expected_certified[0]

    assert choice == expected_choice
    assert assessment.minimum_separation == expected_assessment.minimum_separation
    assert [item[0] for item in calls] == [
        _key(baseline), _key(candidates[0]), _key(candidates[2]),
    ]
    assert len(calls) == 3
    assert all(kwargs["threshold"] == 79.0 for _, kwargs in calls)


def test_backup_dedup_baseline_rejected_preserves_old_choice_and_minimum(monkeypatch) -> None:
    config = SimConfig()
    shield = PredictiveSafetyShield(config)
    env = SimpleNamespace()
    residuals = {1: 0.2, 2: -0.3}
    pairs = ((1, 2),)
    baseline = {1: -0.4, 2: -0.4}
    candidates = [
        {1: 0.0, 2: 0.0},
        {2: 0.0, 1: 0.0},
        {1: 0.4, 2: 0.0},
        {1: 0.4, 2: 0.0},
    ]
    minimums = {
        _key(baseline): 75.0,
        _key(candidates[0]): 82.0,
        _key(candidates[2]): 91.0,
    }
    baseline_key = _key(baseline)
    calls: list[tuple[tuple[tuple[int, float], ...], dict[str, object]]] = []

    monkeypatch.setattr(
        shield,
        "_headway_backup_request",
        lambda _env, _residuals, _pairs: dict(baseline),
    )
    monkeypatch.setattr(
        shield,
        "_ring_profile_backup_candidates",
        lambda _env, _residuals, _pairs: [dict(item) for item in candidates],
    )
    monkeypatch.setattr(shield, "_components", lambda _edges: ())

    def assess(_env, trial, **kwargs):
        trial_key = _key(trial)
        calls.append((trial_key, dict(kwargs)))
        approved = trial_key != baseline_key
        return JointSafetyAssessment(approved, minimums[trial_key], ())

    monkeypatch.setattr(shield, "_joint_assessment", assess)

    choice, assessment = shield._backup_safe(env, residuals, pairs, threshold=79.0)

    expected_certified = []
    for trial in [*candidates, baseline]:
        verdict = JointSafetyAssessment(
            _key(trial) != baseline_key, minimums[_key(trial)], (),
        )
        if not verdict.approved:
            continue
        correction = sum(
            abs(trial.get(uid, residuals.get(uid, 0.0)) - residuals.get(uid, 0.0))
            for uid in sorted(residuals)
        )
        action_key = tuple(
            round(trial.get(uid, 0.0) / config.max_speed_residual_mps, 8)
            for uid in sorted(residuals)
        )
        expected_certified.append((
            (-verdict.minimum_separation, correction, action_key),
            trial,
            verdict,
        ))
    expected_certified.sort(key=lambda item: item[0])
    _, expected_choice, expected_assessment = expected_certified[0]

    assert choice == expected_choice
    assert assessment.minimum_separation == expected_assessment.minimum_separation
    assert [item[0] for item in calls] == [
        _key(baseline), _key(candidates[0]), _key(candidates[2]),
    ]
    assert len(calls) == 3
    assert all(kwargs["threshold"] == 79.0 for _, kwargs in calls)
