from __future__ import annotations

import numpy as np

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2
from terminal_marl.core.safety_filter import TrajectorySafetyResult
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import ArrivalRequest, FlightPhase, MissionType, QueuedArrivalState, UAVState
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _env_and_uav() -> tuple[TerminalParallelEnv, UAVState]:
    config = SimConfig(
        arrival_model="finite_cohort",
        total_requested_uavs=0,
        arrival_rate_per_second=0.0,
        action_schema_version=ACTION_SCHEMA_V4_2,
        trajectory_schema_version="executable_trajectory_v3",
        vehicle_execution_mode="performance_constrained",
        max_simulation_seconds=120.0,
    )
    env = TerminalParallelEnv(config)
    request = ArrivalRequest(1, 0.0, 0, MissionType.STANDARD, 1, config.energy_capacity, 1.0)
    queued = QueuedArrivalState(request, request.energy_capacity)
    uav = UAVState.from_request(0, request, queued, np.zeros(3), env.now)
    uav.phase = FlightPhase.RING_INNER
    uav.ring_role = RingRole.INNER
    ring = env.airspace.actual_ring(RingRole.INNER)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    uav.acceleration = np.asarray((-(ring.nominal_speed ** 2) / ring.radius, 0.0, 0.0), dtype=float)
    uav.speed = ring.nominal_speed
    env.active[uav.uid] = uav
    return env, uav


def _plan_from_live_state(env: TerminalParallelEnv, uav: UAVState, *, end_position=None, target_phase="landing"):
    end_position = np.asarray(
        uav.position + np.asarray((0.0, 1.0, 0.0), dtype=float)
        if end_position is None else end_position,
        dtype=float,
    )
    points = (
        TrajectoryPoint(env.now, uav.position, uav.velocity, uav.acceleration),
        TrajectoryPoint(env.now + 2.0, end_position, uav.velocity, uav.acceleration),
    )
    return TrajectoryPlan(
        "v3-execution-test",
        uav.uid,
        "test",
        points,
        env.now,
        env.now + 2.0,
        metadata={
            "trajectory_schema_version": "executable_trajectory_v3",
            "target_phase": target_phase,
        },
    )


def _approve_with_recording(env: TerminalParallelEnv, calls: list[str]) -> None:
    def _approve(plan, _committed, _snapshot):
        calls.append(plan.plan_id)
        env.safety.checked_trajectory_hashes[plan.plan_id] = plan.trajectory_hash
        return TrajectorySafetyResult(
            True,
            "approved",
            checked_plan_id=plan.plan_id,
            checked_trajectory_hash=plan.trajectory_hash,
        )

    env.safety.filter_plan = _approve


def test_v3_initial_boundary_mismatch_is_rejected_before_safety_filter() -> None:
    env, uav = _env_and_uav()
    plan = _plan_from_live_state(env, uav)
    mismatched = TrajectoryPlan(
        "v3-mismatch",
        uav.uid,
        "test",
        (
            TrajectoryPoint(env.now, uav.position, uav.velocity + np.asarray((1.0, 0.0, 0.0)), uav.acceleration),
            plan.points[1],
        ),
        env.now,
        env.now + 2.0,
        metadata=dict(plan.metadata),
    )
    calls: list[str] = []
    _approve_with_recording(env, calls)

    assert env._commit(uav, mismatched) is False
    assert calls == []
    assert env.last_safety_decisions[uav.uid] == "v3_initial_state_velocity_mismatch"
    assert env.telemetry["safety_interventions"] == 1
    assert uav.committed_plan is None


def test_v3_valid_boundary_executes_authoritative_sample_without_startup_teleport() -> None:
    env, uav = _env_and_uav()
    plan = _plan_from_live_state(env, uav)
    calls: list[str] = []
    _approve_with_recording(env, calls)
    initial_position = uav.position.copy()

    assert env._commit(uav, plan) is True
    assert calls == [plan.plan_id]
    assert np.array_equal(uav.position, initial_position)

    env._advance_plan(uav)
    expected = plan.sample(env.now + env.config.dt).position
    np.testing.assert_allclose(uav.position, expected, atol=1e-12, rtol=0.0)
    assert not np.array_equal(uav.position, initial_position)


def test_v3_ring_endpoint_requires_tangent_velocity_and_centripetal_acceleration() -> None:
    env, uav = _env_and_uav()
    ring = env.airspace.actual_ring(RingRole.MIDDLE)
    endpoint = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    bad_endpoint = TrajectoryPoint(
        env.now + 2.0,
        endpoint,
        np.asarray((ring.nominal_speed, 0.0, 0.0), dtype=float),
        np.zeros(3, dtype=float),
    )
    plan = TrajectoryPlan(
        "v3-ring-endpoint-mismatch",
        uav.uid,
        "ring_transfer",
        (
            TrajectoryPoint(env.now, uav.position, uav.velocity, uav.acceleration),
            bad_endpoint,
        ),
        env.now,
        env.now + 2.0,
        metadata={
            "trajectory_schema_version": "executable_trajectory_v3",
            "target_phase": "ring_middle",
        },
    )
    calls: list[str] = []
    _approve_with_recording(env, calls)

    assert env._commit(uav, plan) is False
    assert calls == []
    assert env.last_safety_decisions[uav.uid] == "v3_ring_endpoint_velocity_mismatch"
