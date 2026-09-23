"""Parity and lookup-path checks for the cached holder reachable tube."""

from __future__ import annotations

import time

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def _case(role: RingRole = RingRole.MIDDLE, speed: float = 0.0):
    config = SimConfig(
        experiment_mode="fixed_low_student",
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
        max_simulation_seconds=30.0,
    )
    env = TerminalParallelEnv(config)
    env.reset(20260924)
    uav = next(iter(env.active.values()))
    ring = env.airspace.actual_ring(role)
    theta = 0.37
    uav.ring_role = role
    uav.ring_progress = theta
    uav.speed = speed
    uav.position = np.asarray((ring.radius * np.cos(theta), ring.radius * np.sin(theta), ring.height), dtype=float)
    return config, uav, WorldSnapshot(env.now, env.active, env.airspace)


def _scalar_reference(safety: SafetyFilter, uav, snapshot: WorldSnapshot, times: np.ndarray):
    """Reference implementation matching the pre-vectorized lookup path."""
    ring = snapshot.airspace.actual_ring(uav.ring_role)
    theta = uav.ring_progress if uav.ring_progress is not None else float(np.arctan2(uav.position[1], uav.position[0]))
    target_low = float(np.clip(
        ring.nominal_speed + snapshot.airspace.speed_profile_bias - safety.config.max_speed_residual_mps,
        safety.config.speed_min,
        safety.config.max_speed,
    ))
    target_high = float(np.clip(
        ring.nominal_speed + snapshot.airspace.speed_profile_bias + safety.config.max_speed_residual_mps,
        safety.config.speed_min,
        safety.config.max_speed,
    ))
    angles_low = [float(theta)]
    speeds_low = [max(0.0, float(uav.speed))]
    angles_high = [float(theta)]
    speeds_high = [max(0.0, float(uav.speed))]
    max_offset = max(0.0, float(np.max(times)) - float(snapshot.now))
    needed = int(np.floor(max_offset / safety.config.physics_dt + 1e-9))
    while len(angles_low) - 1 < needed:
        low = safety.dynamics.ring_motion_step(
            ring, angles_low[-1], speeds_low[-1], target_low, safety.config.physics_dt
        )
        high = safety.dynamics.ring_motion_step(
            ring, angles_high[-1], speeds_high[-1], target_high, safety.config.physics_dt
        )
        angles_low.append(low.theta)
        speeds_low.append(low.speed)
        angles_high.append(high.theta)
        speeds_high.append(high.speed)

    lower = []
    upper = []
    for time in times:
        offset = max(0.0, float(time) - float(snapshot.now))
        index = int(np.floor(offset / safety.config.physics_dt + 1e-9))
        fraction = offset - index * safety.config.physics_dt
        if fraction <= 1e-8:
            lower.append(angles_low[index])
            upper.append(angles_high[index])
        else:
            low = safety.dynamics.ring_motion_step(
                ring, angles_low[index], speeds_low[index], target_low, fraction
            )
            high = safety.dynamics.ring_motion_step(
                ring, angles_high[index], speeds_high[index], target_high, fraction
            )
            lower.append(low.theta)
            upper.append(high.theta)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def test_cached_tube_lookup_matches_scalar_reference_for_grid_and_fractional_times():
    config, uav, snapshot = _case(RingRole.MIDDLE, speed=7.0)
    safety = SafetyFilter(config)
    dt = config.physics_dt
    times = np.asarray((snapshot.now - dt, snapshot.now, snapshot.now + dt,
                        snapshot.now + dt + 1e-9,
                        snapshot.now + dt + 2e-8,
                        snapshot.now + 0.35, snapshot.now + 2.0), dtype=float)

    expected = _scalar_reference(safety, uav, snapshot, times)
    actual = safety.holder_reachable_tube(uav, snapshot, times)

    assert actual[0].shape == times.shape
    assert actual[1].shape == times.shape
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_grid_aligned_lookup_uses_cached_arrays_without_extra_integrations(monkeypatch):
    config, uav, snapshot = _case(RingRole.OUTER, speed=11.0)
    safety = SafetyFilter(config)
    dt = config.physics_dt
    warm = np.arange(snapshot.now, snapshot.now + 8.0 * dt + 1e-9, dt)
    safety.holder_reachable_tube(uav, snapshot, warm)

    calls = 0
    original = safety.dynamics.ring_motion_step

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(safety.dynamics, "ring_motion_step", counted)
    grid = np.asarray((snapshot.now, snapshot.now + dt, snapshot.now + 3.0 * dt,
                       snapshot.now + 8.0 * dt), dtype=float)
    safety.holder_reachable_tube(uav, snapshot, grid)
    assert calls == 0


def test_fractional_lookup_only_integrates_fractional_offsets(monkeypatch):
    config, uav, snapshot = _case(RingRole.INNER, speed=5.0)
    safety = SafetyFilter(config)
    dt = config.physics_dt
    warm = np.arange(snapshot.now, snapshot.now + 8.0 * dt + 1e-9, dt)
    safety.holder_reachable_tube(uav, snapshot, warm)

    calls = 0
    original = safety.dynamics.ring_motion_step

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(safety.dynamics, "ring_motion_step", counted)
    mixed = np.asarray((snapshot.now, snapshot.now + dt,
                        snapshot.now + 1.35 * dt,
                        snapshot.now + 3.0 * dt,
                        snapshot.now + 3.25 * dt), dtype=float)
    safety.holder_reachable_tube(uav, snapshot, mixed)
    assert calls == 4


@pytest.mark.parametrize("role", tuple(RingRole))
@pytest.mark.parametrize("speed_fraction", (0.0, 0.5, 1.0))
def test_cache_extension_preserves_tube_for_each_ring_and_initial_speed(role, speed_fraction):
    config, uav, snapshot = _case(role, speed=0.0)
    ring = snapshot.airspace.actual_ring(role)
    uav.speed = speed_fraction * ring.nominal_speed
    safety = SafetyFilter(config)
    short = np.asarray((snapshot.now, snapshot.now + config.physics_dt, snapshot.now + 0.45), dtype=float)
    long = np.asarray((snapshot.now, snapshot.now + 0.2, snapshot.now + 1.1,
                       snapshot.now + 5.0 + 3e-8), dtype=float)

    safety.holder_reachable_tube(uav, snapshot, short)
    actual = safety.holder_reachable_tube(uav, snapshot, long)
    expected = _scalar_reference(safety, uav, snapshot, long)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_reachable_tube_preserves_empty_and_unordered_time_errors():
    config, uav, snapshot = _case()
    safety = SafetyFilter(config)
    with pytest.raises(ValueError, match="non-empty and ordered"):
        safety.holder_reachable_tube(uav, snapshot, np.asarray([], dtype=float))
    with pytest.raises(ValueError, match="non-empty and ordered"):
        safety.holder_reachable_tube(uav, snapshot, np.asarray((1.0, 0.0), dtype=float))


def test_holder_reachable_tube_lookup_microbenchmark():
    """Report cached scalar-loop versus NumPy gather timings."""
    config, uav, snapshot = _case(RingRole.MIDDLE, speed=7.0)
    safety = SafetyFilter(config)
    ring = snapshot.airspace.actual_ring(uav.ring_role)
    target_low = float(np.clip(
        ring.nominal_speed + snapshot.airspace.speed_profile_bias - safety.config.max_speed_residual_mps,
        safety.config.speed_min,
        safety.config.max_speed,
    ))
    target_high = float(np.clip(
        ring.nominal_speed + snapshot.airspace.speed_profile_bias + safety.config.max_speed_residual_mps,
        safety.config.speed_min,
        safety.config.max_speed,
    ))

    for count in (300, 1500):
        times = snapshot.now + np.arange(count, dtype=float) * config.physics_dt
        safety.holder_reachable_tube(uav, snapshot, times)
        cached = next(iter(safety._holder_tube_cache.values()))
        angles_low = np.asarray(cached["angles_low"], dtype=float)
        angles_high = np.asarray(cached["angles_high"], dtype=float)

        def legacy_lookup():
            lower = []
            upper = []
            for time_value in times:
                offset = max(0.0, float(time_value) - float(snapshot.now))
                index = int(np.floor(offset / config.physics_dt + 1e-9))
                fraction = offset - index * config.physics_dt
                if fraction <= 1e-8:
                    lower.append(cached["angles_low"][index])
                    upper.append(cached["angles_high"][index])
                else:
                    low = safety.dynamics.ring_motion_step(
                        ring, angles_low[index], cached["speeds_low"][index], target_low, fraction
                    )
                    high = safety.dynamics.ring_motion_step(
                        ring, angles_high[index], cached["speeds_high"][index], target_high, fraction
                    )
                    lower.append(low.theta)
                    upper.append(high.theta)
            return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

        expected = legacy_lookup()
        actual = safety.holder_reachable_tube(uav, snapshot, times)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

        for _ in range(3):
            legacy_lookup()
            safety.holder_reachable_tube(uav, snapshot, times)
        legacy_start = time.perf_counter()
        for _ in range(12):
            legacy_lookup()
        legacy_seconds = (time.perf_counter() - legacy_start) / 12.0
        vector_start = time.perf_counter()
        for _ in range(24):
            safety.holder_reachable_tube(uav, snapshot, times)
        vector_seconds = (time.perf_counter() - vector_start) / 24.0
        print(
            f"holder_reachable_tube_lookup n={count}: "
            f"scalar={legacy_seconds:.6e}s vector={vector_seconds:.6e}s "
            f"speedup={legacy_seconds / vector_seconds:.2f}x"
        )
