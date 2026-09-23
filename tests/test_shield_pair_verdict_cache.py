from __future__ import annotations

import pytest

from terminal_marl.core.config import SimConfig
from terminal_marl.core.runtime_traffic_safety import PredictiveSafetyShield
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _env() -> TerminalParallelEnv:
    config = SimConfig(
        experiment_mode="fixed_low_student",
        reset_mode="warm_start_rings",
        warm_start_uavs=2,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0, 0.0),
        total_requested_uavs=2,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=840605)
    return env


def _pair(env: TerminalParallelEnv) -> tuple[int, int]:
    pair = tuple(sorted(env.active))
    assert len(pair) == 2
    return pair


def test_repeated_assessment_reuses_exact_pair_verdict_within_protect(monkeypatch) -> None:
    env = _env()
    shield = PredictiveSafetyShield(env.config)
    pair = _pair(env)
    residuals = {pair[0]: 0.0, pair[1]: 0.0}
    calls = []
    original = shield._assess_position_pair

    def counted(left, right, threshold=None):
        calls.append((left, right, threshold))
        return original(left, right, threshold)

    monkeypatch.setattr(shield, "_assess_position_pair", counted)
    assessments = []

    def body(call_env, intents, traces=None):
        assessments.append(shield._joint_assessment(call_env, residuals, pairs=(pair,)))
        assessments.append(shield._joint_assessment(call_env, residuals, pairs=(pair,)))
        assert len(shield._joint_pair_verdict_cache) == 1
        assert len(shield._joint_pair_snapshot_refs) == 1
        return intents

    monkeypatch.setattr(shield, "_protect_impl", body)
    shield.protect(env, {})

    assert len(calls) == 1
    assert assessments[0] == assessments[1]
    assert shield._joint_pair_verdict_cache is None
    assert shield._joint_pair_snapshot_refs is None


def test_pair_verdict_key_changes_with_endpoint_residuals(monkeypatch) -> None:
    env = _env()
    shield = PredictiveSafetyShield(env.config)
    pair = _pair(env)
    calls = []
    original = shield._assess_position_pair

    def counted(left, right, threshold=None):
        calls.append((left, right, threshold))
        return original(left, right, threshold)

    monkeypatch.setattr(shield, "_assess_position_pair", counted)

    def body(call_env, intents, traces=None):
        shield._joint_assessment(
            call_env, {pair[0]: 0.0, pair[1]: 0.0}, pairs=(pair,),
        )
        shield._joint_assessment(
            call_env, {pair[0]: 0.25, pair[1]: 0.0}, pairs=(pair,),
        )
        return intents

    monkeypatch.setattr(shield, "_protect_impl", body)
    shield.protect(env, {})

    assert len(calls) == 2


def test_pair_verdict_key_misses_for_different_snapshot_and_keeps_snapshot_alive(monkeypatch) -> None:
    env = _env()
    other = _env()
    shield = PredictiveSafetyShield(env.config)
    pair = _pair(env)
    other_pair = _pair(other)
    assert pair == other_pair
    calls = []
    original = shield._assess_position_pair

    def counted(left, right, threshold=None):
        calls.append((left, right, threshold))
        return original(left, right, threshold)

    monkeypatch.setattr(shield, "_assess_position_pair", counted)

    def body(call_env, intents, traces=None):
        residuals = {pair[0]: 0.0, pair[1]: 0.0}
        shield._joint_assessment(call_env, residuals, pairs=(pair,))
        shield._joint_assessment(other, residuals, pairs=(other_pair,))
        assert len(shield._joint_pair_snapshot_refs) == 2
        assert shield._joint_pair_snapshot_refs[id(call_env)] is call_env
        assert shield._joint_pair_snapshot_refs[id(other)] is other
        return intents

    monkeypatch.setattr(shield, "_protect_impl", body)
    shield.protect(env, {})

    assert len(calls) == 2
    assert shield._joint_pair_verdict_cache is None
    assert shield._joint_pair_snapshot_refs is None


def test_direct_joint_assessments_do_not_share_pair_verdict_cache(monkeypatch) -> None:
    env = _env()
    shield = PredictiveSafetyShield(env.config)
    pair = _pair(env)
    calls = []
    original = shield._assess_position_pair

    def counted(left, right, threshold=None):
        calls.append((left, right, threshold))
        return original(left, right, threshold)

    monkeypatch.setattr(shield, "_assess_position_pair", counted)
    residuals = {pair[0]: 0.0, pair[1]: 0.0}
    first = shield._joint_assessment(env, residuals, pairs=(pair,))
    second = shield._joint_assessment(env, residuals, pairs=(pair,))

    assert first == second
    assert len(calls) == 2
    assert shield._joint_pair_verdict_cache is None


def test_protect_clears_pair_cache_on_exception(monkeypatch) -> None:
    env = _env()
    shield = PredictiveSafetyShield(env.config)

    def body(call_env, intents, traces=None):
        shield._joint_pair_snapshot_refs[id(call_env)] = call_env
        raise RuntimeError("test protect exception")

    monkeypatch.setattr(shield, "_protect_impl", body)
    with pytest.raises(RuntimeError, match="test protect exception"):
        shield.protect(env, {})

    assert shield._joint_pair_verdict_cache is None
    assert shield._joint_pair_snapshot_refs is None


def test_cached_and_cache_disabled_assessments_have_same_decision_and_telemetry(monkeypatch) -> None:
    env_uncached = _env()
    env_cached = _env()
    pair = _pair(env_uncached)
    cached_pair = _pair(env_cached)
    assert pair == cached_pair
    residuals = {pair[0]: -0.4, pair[1]: 0.6}

    uncached = PredictiveSafetyShield(env_uncached.config)
    uncached_results = [
        uncached._joint_assessment(env_uncached, residuals, pairs=(pair,), horizon=12.0, threshold=65.0)
        for _ in range(2)
    ]

    cached = PredictiveSafetyShield(env_cached.config)
    cached_results = []

    def body(call_env, intents, traces=None):
        cached_results.extend([
            cached._joint_assessment(
                call_env, residuals, pairs=(cached_pair,), horizon=12.0, threshold=65.0,
            ),
            cached._joint_assessment(
                call_env, residuals, pairs=(cached_pair,), horizon=12.0, threshold=65.0,
            ),
        ])
        return intents

    monkeypatch.setattr(cached, "_protect_impl", body)
    cached.protect(env_cached, {})

    assert cached_results == uncached_results
    assert cached.telemetry["joint_safety_checks"] == uncached.telemetry["joint_safety_checks"] == 2
