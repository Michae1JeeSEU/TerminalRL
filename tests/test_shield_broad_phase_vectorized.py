from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.runtime_traffic_safety import (
    PredictionCache,
    PredictiveSafetyShield,
)
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint


@dataclass
class _Holder:
    ring_role: RingRole


def _plan(points: np.ndarray, times: np.ndarray, plan_id: str = "broad-phase") -> TrajectoryPlan:
    return TrajectoryPlan(
        plan_id,
        1,
        "broad_phase_test",
        tuple(
            TrajectoryPoint(float(time), np.asarray(position, dtype=float), np.zeros(3))
            for time, position in zip(times, points)
        ),
        float(times[0]),
        float(times[-1]),
    )


def _old_envelope(plan, start: float, end: float):
    segments = []
    for left, right in zip(plan.points, plan.points[1:]):
        segment_start = max(start, left.t)
        segment_end = min(end, right.t)
        if segment_end < segment_start:
            continue
        segments.append((plan.sample_position(segment_start), plan.sample_position(segment_end)))
    if not segments:
        return None
    radial_min = float("inf")
    radial_max = 0.0
    z_min = float("inf")
    z_max = -float("inf")
    for left, right in segments:
        left_xy = np.asarray(left[:2], dtype=float)
        delta = np.asarray(right[:2], dtype=float) - left_xy
        denominator = float(np.dot(delta, delta))
        projection = 0.0 if denominator <= 1e-12 else float(
            np.clip(-np.dot(left_xy, delta) / denominator, 0.0, 1.0)
        )
        radial_min = min(radial_min, float(np.linalg.norm(left_xy + projection * delta)))
        radial_max = max(
            radial_max,
            float(np.linalg.norm(left[:2])),
            float(np.linalg.norm(right[:2])),
        )
        z_min = min(z_min, float(left[2]), float(right[2]))
        z_max = max(z_max, float(left[2]), float(right[2]))
    return radial_min, radial_max, z_min, z_max


def _run(shield, plan, *, now: float, cache: PredictionCache | None = None):
    config = shield.config
    airspace = DynamicAirspace(config)
    holder = _Holder(RingRole.OUTER)
    result = shield._broad_phase_reject(holder, plan, airspace, now, cache)
    start = max(now, plan.start_time)
    end = min(plan.end_time, now + config.shield_horizon_seconds)
    key = (plan.plan_id, plan.trajectory_hash, round(start, 8), round(end, 8))
    return result, None if cache is None else cache.plan_envelopes.get(key), airspace, holder


@pytest.mark.parametrize(
    "points,times,now",
    (
        (
            np.asarray(((900.0, 0.0, 100.0), (900.0, 0.0, 100.0))),
            np.asarray((0.0, 10.0)),
            2.5,
        ),
        (
            np.asarray(((700.0, 100.0, 100.0), (500.0, 300.0, 100.0),
                        (300.0, 500.0, 80.0), (100.0, 700.0, 120.0))),
            np.asarray((0.0, 4.0, 8.0, 12.0)),
            4.0,
        ),
        (
            np.asarray(((840.0, 0.0, 100.0), (650.0, 220.0, 90.0),
                        (420.0, 420.0, 110.0), (220.0, 650.0, 100.0),
                        (0.0, 840.0, 100.0))),
            np.asarray((0.0, 1.0, 2.0, 3.0, 4.0)),
            1.0,
        ),
    ),
)
def test_vectorized_envelope_matches_reference_for_aligned_and_single_segments(
    points: np.ndarray, times: np.ndarray, now: float,
) -> None:
    config = SimConfig(shield_horizon_seconds=2.5)
    shield = PredictiveSafetyShield(config)
    plan = _plan(points, times, plan_id=f"case-{now}")
    cache = PredictionCache()
    result, envelope, airspace, holder = _run(shield, plan, now=now, cache=cache)

    start = max(now, plan.start_time)
    end = min(plan.end_time, now + config.shield_horizon_seconds)
    expected = _old_envelope(plan, start, end)
    assert expected is not None
    np.testing.assert_allclose(envelope, expected, rtol=0.0, atol=1e-12)
    radial_min, radial_max, z_min, z_max = expected
    ring = airspace.actual_ring(holder.ring_role)
    lower_bound = np.hypot(
        max(radial_min - ring.radius, ring.radius - radial_max, 0.0),
        max(z_min - ring.height, ring.height - z_max, 0.0),
    )
    assert result == (
        lower_bound > config.safe_distance + config.shield_broad_phase_margin
    )


def test_long_plan_interpolates_only_overlap_boundaries() -> None:
    times = np.linspace(0.0, 1000.0, 10001)
    angles = np.linspace(0.0, 3.0 * np.pi, len(times))
    points = np.column_stack((
        700.0 * np.cos(angles),
        700.0 * np.sin(angles),
        100.0 + 5.0 * np.sin(angles),
    ))
    base = _plan(points, times, plan_id="long-plan")

    class CountingPlan:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.plan_id = wrapped.plan_id
            self.trajectory_hash = wrapped.trajectory_hash
            self.start_time = wrapped.start_time
            self.end_time = wrapped.end_time
            self.points = wrapped.points
            self.sample_calls = []

        def sample_position(self, moment):
            self.sample_calls.append(float(moment))
            return self._wrapped.sample_position(moment)

    plan = CountingPlan(base)
    shield = PredictiveSafetyShield(SimConfig(shield_horizon_seconds=30.0))
    _run(shield, plan, now=475.25, cache=None)
    assert plan.sample_calls == [475.25, 505.25]


def test_randomized_envelopes_and_rejections_match_reference() -> None:
    rng = np.random.default_rng(20260924)
    config = SimConfig(shield_horizon_seconds=3.0)
    shield = PredictiveSafetyShield(config)
    airspace = DynamicAirspace(config)
    holder = _Holder(RingRole.OUTER)
    for trial in range(32):
        count = int(rng.integers(2, 40))
        times = np.concatenate(([0.0], np.cumsum(rng.uniform(0.05, 2.0, count - 1))))
        points = rng.normal(0.0, 1.0, (count, 3))
        points[:, :2] *= rng.uniform(200.0, 1000.0, (count, 1))
        points[:, 2] = rng.uniform(0.0, 140.0, count)
        plan = _plan(points, times, plan_id=f"random-{trial}")
        now = float(rng.uniform(times[0], times[-1]))
        cache = PredictionCache()
        actual = shield._broad_phase_reject(holder, plan, airspace, now, cache)
        start = max(now, plan.start_time)
        end = min(plan.end_time, now + config.shield_horizon_seconds)
        expected_envelope = _old_envelope(plan, start, end)
        key = (plan.plan_id, plan.trajectory_hash, round(start, 8), round(end, 8))
        expected = None if expected_envelope is None else (
            max(expected_envelope[0] - airspace.actual_ring(holder.ring_role).radius,
                airspace.actual_ring(holder.ring_role).radius - expected_envelope[1], 0.0),
            max(expected_envelope[2] - airspace.actual_ring(holder.ring_role).height,
                airspace.actual_ring(holder.ring_role).height - expected_envelope[3], 0.0),
        )
        if expected is None:
            assert actual is True
            continue
        np.testing.assert_allclose(cache.plan_envelopes[key], expected_envelope, rtol=0.0, atol=1e-12)
        expected_rejection = np.hypot(*expected) > config.safe_distance + config.shield_broad_phase_margin
        assert actual == expected_rejection


@pytest.mark.parametrize("delta,expected", ((0.0, False), (1e-9, True)))
def test_broad_phase_boundary_keeps_strict_greater_than_rule(delta: float, expected: bool) -> None:
    config = SimConfig(shield_horizon_seconds=1.0)
    threshold = config.safe_distance + config.shield_broad_phase_margin
    radius = config.rings[RingRole.OUTER].radius + threshold + delta
    plan = _plan(
        np.asarray(((radius, 0.0, config.rings[RingRole.OUTER].height),
                    (radius, 0.0, config.rings[RingRole.OUTER].height))),
        np.asarray((0.0, 1.0)),
        plan_id=f"boundary-{delta}",
    )
    shield = PredictiveSafetyShield(config)
    result, envelope, _, _ = _run(shield, plan, now=0.0, cache=PredictionCache())
    reference = _old_envelope(plan, 0.0, 1.0)
    np.testing.assert_allclose(envelope, reference, rtol=0.0, atol=1e-12)
    assert result == expected
