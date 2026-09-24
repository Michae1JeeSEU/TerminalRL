from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeScheduler
from terminal_marl.core.uav import FlightPhase
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def _snapshot(active: dict[int, object], *, now: float = 10.0):
    env = TerminalParallelEnv(SimConfig())
    return env, SimpleNamespace(now=now, active=active, airspace=env.airspace)


def _holder(uid: int, angle: float) -> SimpleNamespace:
    return SimpleNamespace(uid=uid, ring_role=RingRole.MIDDLE, ring_progress=angle)


def _requester(angle: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        ring_role=RingRole.OUTER,
        ring_progress=angle,
        phase=FlightPhase.RING_OUTER,
        position=np.asarray((600.0, 0.0, 100.0)),
    )


def test_single_holder_gap_returns_antipode_after_wrapped_time_prediction():
    holder_angle = 2.0 * math.pi - 0.15
    now = 10.0
    merge_time = 10.0 + 137.5
    speed = 20.0
    env, snapshot = _snapshot({7: _holder(7, holder_angle)}, now=now)
    scheduler = MergeScheduler(env.config)

    predicted = (holder_angle + speed * (merge_time - now) /
                 snapshot.airspace.actual_ring(RingRole.MIDDLE).radius) % (2.0 * math.pi)
    actual = scheduler._gap_center(_requester(), RingRole.MIDDLE, merge_time, snapshot, speed)

    assert actual == pytest.approx((predicted + math.pi) % (2.0 * math.pi))


def test_empty_target_gap_keeps_requester_phase_prediction():
    now = 10.0
    merge_time = 95.0
    requester = _requester(angle=1.25)
    env, snapshot = _snapshot({}, now=now)
    scheduler = MergeScheduler(env.config)
    source = env.airspace.actual_ring(RingRole.OUTER)
    expected = (requester.ring_progress + source.nominal_speed * (merge_time - now) / source.radius) % (2.0 * math.pi)

    actual = scheduler._gap_center(requester, RingRole.MIDDLE, merge_time, snapshot, 20.0)

    assert actual == pytest.approx(expected)


def test_multiple_holder_gap_preserves_cyclic_midpoint_behavior():
    now = 10.0
    merge_time = 70.0
    speed = 20.0
    angles = (0.2, 2.1, 5.4)
    env, snapshot = _snapshot({uid: _holder(uid, angle) for uid, angle in enumerate(angles)})
    scheduler = MergeScheduler(env.config)
    radius = env.airspace.actual_ring(RingRole.MIDDLE).radius
    predicted = sorted((angle + speed * (merge_time - now) / radius) % (2.0 * math.pi) for angle in angles)
    gaps = [
        ((predicted[(index + 1) % len(predicted)] - angle) % (2.0 * math.pi), angle)
        for index, angle in enumerate(predicted)
    ]
    gap, start = max(gaps)

    actual = scheduler._gap_center(_requester(), RingRole.MIDDLE, merge_time, snapshot, speed)

    assert actual == pytest.approx((start + gap / 2.0) % (2.0 * math.pi))


def test_candidate_path_still_calls_full_safety_filter_with_single_target_holder(monkeypatch):
    config = SimConfig(
        reset_mode="warm_start_rings",
        warm_start_uavs=2,
        fixed_arrival_schedule=(0.0, 0.0),
        total_requested_uavs=2,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=841301)
    requester, holder = tuple(env.active.values())
    requester.phase = FlightPhase.RING_OUTER
    requester.ring_role = RingRole.OUTER
    requester.ring_progress = 0.0
    outer = env.airspace.actual_ring(RingRole.OUTER)
    requester.position = np.asarray((outer.radius, 0.0, outer.height), dtype=float)
    requester.speed = outer.nominal_speed
    requester.velocity = np.asarray((0.0, outer.nominal_speed, 0.0), dtype=float)
    holder.phase = FlightPhase.RING_MIDDLE
    holder.ring_role = RingRole.MIDDLE
    holder.ring_progress = 0.2
    middle = env.airspace.actual_ring(RingRole.MIDDLE)
    holder.position = np.asarray((middle.radius * math.cos(0.2), middle.radius * math.sin(0.2), middle.height), dtype=float)
    holder.speed = middle.nominal_speed
    holder.velocity = np.asarray((-middle.nominal_speed * math.sin(0.2), middle.nominal_speed * math.cos(0.2), 0.0), dtype=float)
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    candidate = type("Candidate", (), {"intent": LowLevelIntent(1.0, 0.0, 0.0), "delay_seconds": 0.0})()
    calls = []
    original = env.safety.filter_plan

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(env.safety, "filter_plan", counted)
    env._candidate_plan(requester, candidate, RingRole.MIDDLE, snapshot, None)

    assert calls == [1]


def test_legacy_diagnostic_wrapper_restores_scheduler_method(tmp_path):
    from terminal_marl.core.merge_scheduler import MergeScheduler
    from terminal_marl.training.diagnose_single_holder_gap import run_variant

    original = MergeScheduler._gap_center
    output = tmp_path / "missing" / "nested" / "seed841301_legacy_pilot.json"
    result = run_variant(
        seed=841301,
        legacy_single_holder_gap=True,
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        max_steps=1,
        progress_every_steps=1,
        output=output,
        dense_warm_start_uavs=2,
        final_approach_geometry_mode="continuous_inward_v1",
        reconcile_late_merge_slots_enabled=False,
    )

    assert MergeScheduler._gap_center is original
    assert result["single_holder_gap_variant"] == "legacy_same_angle"
    assert output.exists()
    assert output.with_name(f"{output.name}.progress.json").exists()
