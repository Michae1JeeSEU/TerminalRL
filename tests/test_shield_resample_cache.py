from __future__ import annotations

import numpy as np

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


def _count_resamples(shield: PredictiveSafetyShield, monkeypatch):
    calls = []
    original = shield._resample_position_track

    def counted(source_times, source_positions, sample_times):
        calls.append((len(source_times), len(sample_times)))
        return original(source_times, source_positions, sample_times)

    monkeypatch.setattr(shield, "_resample_position_track", counted)
    return calls


def test_same_residual_and_horizon_resample_once_per_track(monkeypatch) -> None:
    env = _env()
    shield = PredictiveSafetyShield(env.config)
    calls = _count_resamples(shield, monkeypatch)
    uids = set(env.active)
    residuals = {uid: 0.0 for uid in uids}

    samples_first, tracks_first = shield._joint_position_tracks(env, residuals, uids)
    samples_second, tracks_second = shield._joint_position_tracks(env, residuals, uids)

    assert len(calls) == len(uids)
    assert np.array_equal(samples_first, samples_second)
    for uid in uids:
        assert np.array_equal(tracks_first[uid], tracks_second[uid])
    assert len(shield._joint_sampled_track_cache) == len(uids)


def test_resample_cache_key_changes_with_residual_and_horizon(monkeypatch) -> None:
    env = _env()
    shield = PredictiveSafetyShield(env.config)
    calls = _count_resamples(shield, monkeypatch)
    uid = min(env.active)

    shield._joint_position_tracks(env, {uid: 0.0}, {uid}, horizon=30.0)
    shield._joint_position_tracks(env, {uid: 1.0}, {uid}, horizon=30.0)
    shield._joint_position_tracks(env, {uid: 1.0}, {uid}, horizon=15.0)
    shield.config.safety_sampling_dt = 0.5
    shield._joint_position_tracks(env, {uid: 1.0}, {uid}, horizon=15.0)

    assert len(calls) == 4
    assert len(shield._joint_sampled_track_cache) == 4


def test_protect_clears_sampled_tracks_before_next_state(monkeypatch) -> None:
    env = _env()
    shield = PredictiveSafetyShield(env.config)
    calls = _count_resamples(shield, monkeypatch)
    uid = min(env.active)

    _, first = shield._joint_position_tracks(env, {uid: 0.0}, {uid})
    assert shield._joint_sampled_track_cache
    first_position = first[uid].copy()

    env.now += env.config.dt
    env.active[uid].position = env.active[uid].position + np.asarray((1.0, 0.0, 0.0))
    shield.protect(env, {})
    assert shield._joint_sampled_track_cache == {}
    _, second = shield._joint_position_tracks(env, {uid: 0.0}, {uid})

    assert len(calls) == 2
    assert not np.array_equal(first_position, second[uid])
