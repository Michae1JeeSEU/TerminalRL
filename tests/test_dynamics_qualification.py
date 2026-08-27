"""D7 non-training qualification gates for the performance-constrained model."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def _integrate_ring(dt: float, duration: float = 40.0):
    ring = SimpleNamespace(radius=200.0, height=40.0)
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    angle = speed = 0.0
    for _ in range(round(duration / dt)):
        motion = dynamics.ring_motion_step(ring, angle, speed, 16.0, dt)
        angle, speed = motion.theta, motion.speed
    return motion


def test_d7_ring_holder_dt_converges_between_frozen_physics_steps():
    coarse, fine = _integrate_ring(.2), _integrate_ring(.1)
    assert coarse.speed == fine.speed == 16.0
    assert np.linalg.norm(coarse.position - fine.position) < .25
    assert abs(coarse.theta - fine.theta) < .002


def test_d7_checked_plan_hash_is_the_hash_of_every_executed_sample():
    config = SimConfig(
        experiment_mode="fixed_rule", reset_mode="warm_start_rings", warm_start_uavs=1,
        fixed_arrival_schedule=(0.0,), total_requested_uavs=1, max_simulation_seconds=120.0,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=3)
    uid = env.agents[0]
    for _ in range(5):
        env.step({uid: np.asarray((1.0, 0.0, 0.0), dtype=np.float32)})
        uav = env.active[int(uid)]
        plan = uav.committed_plan
        assert plan is not None
        assert env.safety.checked_trajectory_hashes[plan.plan_id] == plan.trajectory_hash
        expected = plan.sample(env.now)
        assert np.allclose(uav.position, expected.position)
        assert np.allclose(uav.velocity, expected.velocity)
        assert np.allclose(uav.acceleration, expected.acceleration)
    assert env.telemetry["checked_executed_plan_samples"] >= 5 * int(config.dt / config.physics_dt)
    assert env.telemetry["checked_executed_plan_hash_mismatches"] == 0


def test_d7_holder_prediction_is_linear_in_the_number_of_safety_samples(monkeypatch):
    config = SimConfig()
    ring = SimpleNamespace(radius=600.0, height=100.0, nominal_speed=22.0)
    holder = SimpleNamespace(
        uid=1, ring_role=RingRole.OUTER, ring_progress=0.0,
        position=np.asarray((600.0, 0.0, 100.0)), speed=0.0, committed_plan=None,
    )
    candidate = TrajectoryPlan("candidate", 2, "qualification", [
        TrajectoryPoint(0.0, np.zeros(3)), TrajectoryPoint(100.0, np.zeros(3)),
    ], 0.0, 100.0)
    snapshot = WorldSnapshot(0.0, {1: holder}, SimpleNamespace(
        actual_ring=lambda role: ring, speed_profile_bias=0.0,
    ))
    safety = SafetyFilter(config)
    original = safety.dynamics.ring_motion_step
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(safety.dynamics, "ring_motion_step", counted)
    assert safety.filter_plan(candidate, {}, snapshot).approved
    # 100 seconds at 0.2s needs 500 integrations.  A generous margin allows
    # endpoints while rejecting the previous O(samples^2) replay behaviour.
    assert calls <= 510
