from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.runtime_traffic_safety import PredictiveSafetyShield


def _env(
    config: SimConfig,
    holders: tuple[tuple[int, float, float, RingRole], ...],
) -> SimpleNamespace:
    airspace = DynamicAirspace(config)
    active = {}
    for uid, theta, speed, role in holders:
        ring = airspace.actual_ring(role)
        active[uid] = SimpleNamespace(
            uid=uid,
            committed_plan=None,
            ring_role=role,
            ring_progress=theta,
            position=np.asarray((
                ring.radius * np.cos(theta),
                ring.radius * np.sin(theta),
                ring.height,
            ), dtype=float),
            speed=speed,
        )
    return SimpleNamespace(now=0.0, active=active, airspace=airspace)


def _scalar_track(shield, env, uid: int, times: np.ndarray, residual: float) -> np.ndarray:
    uav = env.active[uid]
    ring = env.airspace.actual_ring(uav.ring_role)
    theta = float(uav.ring_progress)
    speed = max(0.0, float(uav.speed))
    target_speed = float(np.clip(
        ring.nominal_speed + env.airspace.speed_profile_bias + residual,
        shield.config.speed_min,
        shield.config.max_speed,
    ))
    values = [np.asarray(uav.position, dtype=float).copy()]
    for left, right in zip(times, times[1:]):
        motion = shield.dynamics.ring_motion_step(
            ring, theta, speed, target_speed, float(right - left),
        )
        theta, speed = motion.theta, motion.speed
        values.append(np.asarray(motion.position, dtype=float))
    return np.asarray(values, dtype=float)


@pytest.mark.parametrize(
    "speed,target_speed,role",
    (
        (8.0, 20.0, RingRole.OUTER),
        (31.0, 8.0, RingRole.MIDDLE),
        (-2.0, 100.0, RingRole.INNER),
    ),
)
def test_shared_holder_speed_sequence_matches_scalar_ring_motion(
    speed: float, target_speed: float, role: RingRole,
) -> None:
    config = SimConfig(physics_dt=0.2, shield_horizon_seconds=1.1)
    env = _env(config, ((1, 0.17, speed, role),))
    shield = PredictiveSafetyShield(config)
    residual = target_speed - env.airspace.actual_ring(role).nominal_speed

    samples, tracks = shield._joint_position_tracks(env, {1: residual}, {1})
    source_times = np.arange(
        env.now,
        env.now + config.shield_horizon_seconds + 1e-9,
        config.physics_dt,
    )
    if source_times[-1] < env.now + config.shield_horizon_seconds:
        source_times = np.append(source_times, env.now + config.shield_horizon_seconds)

    expected = shield._resample_position_track(
        source_times,
        _scalar_track(shield, env, 1, source_times, residual),
        samples,
    )
    assert np.array_equal(tracks[1], expected)
    assert len(shield._joint_speed_sequence_cache) == 1
    assert len(source_times) == 7  # 5 full steps followed by the 0.1 s fractional step


def test_speed_sequence_reused_for_holders_with_different_theta(monkeypatch) -> None:
    config = SimConfig(physics_dt=0.2, shield_horizon_seconds=1.1)
    env = _env(config, (
        (1, 0.17, 12.0, RingRole.OUTER),
        (2, 1.23, 12.0, RingRole.OUTER),
    ))
    shield = PredictiveSafetyShield(config)
    calls = []
    original = shield.dynamics.limited_tangential_speed

    def counted(current_speed, target_speed, radius, dt):
        calls.append((current_speed, target_speed, radius, dt))
        return original(current_speed, target_speed, radius, dt)

    monkeypatch.setattr(shield.dynamics, "limited_tangential_speed", counted)
    _, tracks = shield._joint_position_tracks(env, {1: 0.0, 2: 0.0}, {1, 2})

    assert len(calls) == 6
    assert len(shield._joint_speed_sequence_cache) == 1
    assert not np.array_equal(tracks[1], tracks[2])


def test_speed_sequence_cache_is_cleared_at_protect_exit() -> None:
    config = SimConfig()
    env = _env(config, ((1, 0.17, 12.0, RingRole.OUTER),))
    shield = PredictiveSafetyShield(config)
    shield._joint_position_tracks(env, {1: 0.0}, {1}, horizon=1.0)
    assert shield._joint_speed_sequence_cache

    shield.protect(env, {})

    assert shield._joint_speed_sequence_cache == {}
