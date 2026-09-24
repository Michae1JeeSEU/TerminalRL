"""SafetyFilter certificates for the fixed-rule nominal holding contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    UAVState,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def _holder_env(*, max_speed_residual_mps: float) -> tuple[TerminalParallelEnv, UAVState]:
    config = SimConfig(
        max_speed_residual_mps=max_speed_residual_mps,
        physics_dt=0.2,
        safety_sampling_dt=1.0,
    )
    env = TerminalParallelEnv(config)
    request = ArrivalRequest(
        21,
        0.0,
        0,
        MissionType.STANDARD,
        1,
        config.energy_capacity,
        0.8,
    )
    queued = QueuedArrivalState(request, request.energy_capacity * request.initial_soc)
    holder = UAVState.from_request(21, request, queued, np.zeros(3), env.now)
    ring = env.airspace.actual_ring(RingRole.INNER)
    holder.phase = FlightPhase.RING_INNER
    holder.ring_role = RingRole.INNER
    holder.ring_progress = 0.0
    holder.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    holder.speed = ring.nominal_speed
    holder.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    holder.acceleration = np.asarray(
        (-(ring.nominal_speed**2) / ring.radius, 0.0, 0.0),
        dtype=float,
    )
    env.active[holder.uid] = holder
    env.ledger.request(request.arrival_id)
    return env, holder


def _ring_arc_candidate(
    env: TerminalParallelEnv,
    *,
    radial_offset: float,
    vertical_offset: float,
    target_phase: str = "ring_middle",
) -> TrajectoryPlan:
    ring = env.airspace.actual_ring(RingRole.INNER)
    radius = ring.radius + radial_offset
    omega = ring.nominal_speed / ring.radius
    times = np.arange(0.0, 30.0 + 1e-9, env.config.physics_dt)
    theta = omega * times
    positions = np.column_stack(
        (
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.full_like(times, ring.height + vertical_offset),
        )
    )
    speed = radius * omega
    velocities = np.column_stack(
        (-speed * np.sin(theta), speed * np.cos(theta), np.zeros_like(times))
    )
    accelerations = np.column_stack(
        (-radius * omega**2 * np.cos(theta), -radius * omega**2 * np.sin(theta), np.zeros_like(times))
    )
    points = tuple(
        TrajectoryPoint(float(t), position, velocity, acceleration)
        for t, position, velocity, acceleration in zip(times, positions, velocities, accelerations)
    )
    return TrajectoryPlan(
        "nominal-holder-arc",
        99,
        "ring_transfer",
        points,
        0.0,
        30.0,
        metadata={"target_phase": target_phase},
    )


def _post_handoff_candidate(env: TerminalParallelEnv, holder: UAVState) -> TrajectoryPlan:
    ring = env.airspace.actual_ring(RingRole.INNER)
    omega = ring.nominal_speed / ring.radius
    chord = 80.02
    separation_angle = 2.0 * math.asin(chord / (2.0 * ring.radius))
    end_time = 10.0
    endpoint_angle = omega * end_time + separation_angle
    times = np.arange(0.0, end_time + 1e-9, env.config.physics_dt)
    positions = []
    velocities = []
    accelerations = []
    for moment in times:
        if moment < end_time - 1e-9:
            positions.append(
                np.asarray(
                    (
                        ring.radius * math.cos(endpoint_angle),
                        ring.radius * math.sin(endpoint_angle),
                        ring.height + 100.0,
                    ),
                    dtype=float,
                )
            )
            velocities.append(np.zeros(3, dtype=float))
            accelerations.append(np.zeros(3, dtype=float))
        else:
            positions.append(
                np.asarray(
                    (
                        ring.radius * math.cos(endpoint_angle),
                        ring.radius * math.sin(endpoint_angle),
                        ring.height,
                    ),
                    dtype=float,
                )
            )
            velocities.append(
                np.asarray(
                    (-ring.nominal_speed * math.sin(endpoint_angle),
                     ring.nominal_speed * math.cos(endpoint_angle), 0.0),
                    dtype=float,
                )
            )
            accelerations.append(
                np.asarray(
                    (-ring.nominal_speed**2 * math.cos(endpoint_angle) / ring.radius,
                     -ring.nominal_speed**2 * math.sin(endpoint_angle) / ring.radius,
                     0.0),
                    dtype=float,
                )
            )
    points = tuple(
        TrajectoryPoint(float(t), position, velocity, acceleration)
        for t, position, velocity, acceleration in zip(times, positions, velocities, accelerations)
    )
    return TrajectoryPlan(
        "nominal-holder-post-handoff",
        99,
        "ring_transfer",
        points,
        0.0,
        end_time,
        metadata={"target_phase": "ring_inner"},
    )


def _filter(env: TerminalParallelEnv, plan: TrajectoryPlan):
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    return env.safety.filter_plan(plan, {}, snapshot)


def _witness_env(*, max_speed_residual_mps: float) -> tuple[TerminalParallelEnv, UAVState, UAVState]:
    config = SimConfig(
        action_schema_version="low_action_schema_v4_2_hierarchical",
        energy_model_mode="phase_motion_v2",
        emergency_landing_enabled=True,
        emergency_priority_guard_enabled=True,
        energy_reachability_guard_enabled=True,
        rule_flow_coordination_enabled=max_speed_residual_mps == 0.0,
        final_approach_geometry_mode="continuous_inward_v1",
        final_queue_capacity=3,
        lz_assignment_mode="nearest_available",
        release_arbitration_mode="downstream_priority",
        max_speed_residual_mps=max_speed_residual_mps,
        neighbor_comm_range_m=320.0,
        total_requested_uavs=0,
        arrival_rate_per_second=0.0,
    )
    env = TerminalParallelEnv(config)
    env.now = 182.0
    states = {
        19: (
            (-166.31066667493454, 111.08898302774588, 40.0),
            (-8.887118642745, -13.304853333995, 0.0),
            0.6883894748596898,
        ),
        21: (
            (-51.13299519848311, 193.35308842124013, 40.0),
            (-15.46824707369921, -4.090639615878649, 0.0),
            0.5068242817052832,
        ),
    }
    result = []
    for uid, (position, velocity, soc) in states.items():
        request = ArrivalRequest(
            uid,
            0.0,
            0,
            MissionType.STANDARD,
            1,
            config.energy_capacity,
            soc,
        )
        queued = QueuedArrivalState(request, request.energy_capacity * soc)
        uav = UAVState.from_request(uid, request, queued, np.asarray(position), env.now)
        uav.phase = FlightPhase.RING_INNER
        uav.ring_role = RingRole.INNER
        uav.ring_progress = float(math.atan2(position[1], position[0]))
        uav.position = np.asarray(position, dtype=float)
        uav.velocity = np.asarray(velocity, dtype=float)
        uav.speed = float(np.linalg.norm(uav.velocity[:2]))
        uav.acceleration = np.asarray(
            (-uav.position[0] * uav.speed**2 / 200.0**2,
             -uav.position[1] * uav.speed**2 / 200.0**2,
             0.0),
            dtype=float,
        )
        env.active[uid] = uav
        env.ledger.request(request.arrival_id)
        result.append(uav)
    return env, result[0], result[1]


def _witness_plan(env: TerminalParallelEnv, uav: UAVState) -> TrajectoryPlan:
    return env.flow_coordinator.trajectory.final_approach(
        uav,
        env.lz_manager.zones[2],
        env.airspace,
        env.now,
        48.0,
        emergency=False,
    )


def test_uid19_uid21_witness_rejects_nominal_holder_swept_separation() -> None:
    env, candidate_uav, _holder = _witness_env(max_speed_residual_mps=0.0)
    plan = _witness_plan(env, candidate_uav)
    result = _filter(env, plan)

    assert not result.approved
    assert result.reason == "nominal_holder_swept_separation"
    assert result.checked_trajectory_hash is None


def test_zero_scale_rejects_point_safe_but_swept_unsafe_nominal_holder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _holder = _holder_env(max_speed_residual_mps=0.0)
    monkeypatch.setattr(env.safety.dynamics, "validate_plan", lambda _points: [])
    radial_offset = 79.0
    vertical_offset = math.sqrt(80.02**2 - radial_offset**2)
    result = _filter(
        env,
        _ring_arc_candidate(
            env,
            radial_offset=radial_offset,
            vertical_offset=vertical_offset,
        ),
    )

    assert not result.approved
    assert result.reason == "nominal_holder_swept_separation"
    assert result.checked_trajectory_hash is None


def test_zero_scale_safely_separated_nominal_holder_is_approved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _holder = _holder_env(max_speed_residual_mps=0.0)
    monkeypatch.setattr(env.safety.dynamics, "validate_plan", lambda _points: [])
    plan = _ring_arc_candidate(env, radial_offset=79.0, vertical_offset=14.0)
    result = _filter(env, plan)

    assert result.approved
    assert result.checked_plan_id == plan.plan_id
    assert result.checked_trajectory_hash == plan.trajectory_hash


def test_nonzero_residual_configuration_keeps_existing_holder_certificate_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _holder = _holder_env(max_speed_residual_mps=2.0)
    monkeypatch.setattr(env.safety.dynamics, "validate_plan", lambda _points: [])
    plan = _ring_arc_candidate(
        env,
        radial_offset=79.0,
        vertical_offset=math.sqrt(80.02**2 - 79.0**2),
    )
    result = _filter(env, plan)

    assert result.approved
    assert result.checked_trajectory_hash == plan.trajectory_hash


def test_zero_scale_checks_post_handoff_nominal_swept_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, holder = _holder_env(max_speed_residual_mps=0.0)
    monkeypatch.setattr(env.safety.dynamics, "validate_plan", lambda _points: [])
    result = _filter(env, _post_handoff_candidate(env, holder))

    assert not result.approved
    assert result.reason == "nominal_post_plan_holder_swept_separation"
