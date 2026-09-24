"""Fixed-rule flow nominal holding and runtime safety contracts."""

from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    UAVState,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _config(**overrides) -> SimConfig:
    values = dict(
        action_schema_version="low_action_schema_v4_2_hierarchical",
        energy_model_mode="phase_motion_v2",
        emergency_landing_enabled=True,
        emergency_priority_guard_enabled=True,
        energy_reachability_guard_enabled=True,
        rule_flow_coordination_enabled=True,
        arrival_model="finite_cohort",
        total_requested_uavs=0,
        arrival_rate_per_second=0.0,
        max_simulation_seconds=120.0,
    )
    values.update(overrides)
    return SimConfig(**values)


def _ring_uav(
    env: TerminalParallelEnv,
    role: RingRole,
    *,
    soc: float,
    uid: int = 0,
    angle: float = 0.0,
) -> UAVState:
    request = ArrivalRequest(
        uid,
        0.0,
        0,
        MissionType.STANDARD,
        1,
        env.config.energy_capacity,
        soc,
    )
    queued = QueuedArrivalState(request, request.energy_capacity * soc)
    uav = UAVState.from_request(uid, request, queued, np.zeros(3), env.now)
    uav.phase = {
        RingRole.OUTER: FlightPhase.RING_OUTER,
        RingRole.MIDDLE: FlightPhase.RING_MIDDLE,
        RingRole.INNER: FlightPhase.RING_INNER,
    }[role]
    uav.ring_role = role
    uav.ring_progress = float(angle)
    ring = env.airspace.actual_ring(role)
    uav.position = np.asarray(
        (ring.radius * np.cos(angle), ring.radius * np.sin(angle), ring.height),
        dtype=float,
    )
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray(
        (-np.sin(angle) * ring.nominal_speed, np.cos(angle) * ring.nominal_speed, 0.0),
        dtype=float,
    )
    uav.acceleration = np.asarray(
        (
            -np.cos(angle) * (ring.nominal_speed**2) / ring.radius,
            -np.sin(angle) * (ring.nominal_speed**2) / ring.radius,
            0.0,
        ),
        dtype=float,
    )
    env.active[uav.uid] = uav
    env.ledger.request(request.arrival_id)
    return uav


def test_nominal_holding_contract_is_derived_only_for_opt_in_flow() -> None:
    assert SimConfig().max_speed_residual_mps == pytest.approx(2.0)
    assert _config(max_speed_residual_mps=1.25).max_speed_residual_mps == pytest.approx(0.0)


def test_flow_step_rejects_nonzero_speed_residual_action() -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.OUTER, soc=0.90)

    with pytest.raises(ValueError, match="zero speed_residual"):
        env.step({str(uav.uid): np.asarray((1.0, 0.5, 0.0), dtype=np.float32)})


def test_flow_step_rejects_nonzero_speed_residual_shortlist() -> None:
    env = TerminalParallelEnv(_config(runtime_proposal_shortlist_k=3))
    uav = _ring_uav(env, RingRole.OUTER, soc=0.90)
    requested = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    nonzero_speed = np.asarray((1.0, 0.5, 0.0), dtype=np.float32)

    with pytest.raises(ValueError, match="zero speed_residual"):
        env.step(
            {str(uav.uid): requested},
            {str(uav.uid): (requested, nonzero_speed)},
        )


def test_zero_scale_backup_and_protect_do_not_divide_by_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.OUTER, soc=0.90)
    shield = env.runtime_traffic

    monkeypatch.setattr(
        shield,
        "_ring_profile_backup_candidates",
        lambda *_args, **_kwargs: pytest.fail("zero-scale backup entered profile search"),
    )
    choice, assessment = shield._backup_safe(
        env, {uav.uid: 0.0}, (), threshold=env.config.safe_distance,
    )
    assert choice == {uav.uid: 0.0}
    assert assessment.approved

    repaired = shield.protect(
        env, {uav.uid: LowLevelIntent(0.0, 0.0, 0.0)},
    )
    assert repaired[uav.uid].speed_residual == pytest.approx(0.0)


def test_nominal_conflict_fails_closed_without_soft_buffer_or_speed_search() -> None:
    env = TerminalParallelEnv(_config())
    first = _ring_uav(env, RingRole.OUTER, soc=0.90, uid=0, angle=0.0)
    second = _ring_uav(env, RingRole.OUTER, soc=0.90, uid=1, angle=0.10)
    actual_distance = float(np.linalg.norm(first.position - second.position))
    assert 40.0 < actual_distance < 80.0
    shield = env.runtime_traffic

    with pytest.raises(SafetyInvariantError, match="Nominal fixed-rule holding") as exc_info:
        shield.protect(
            env,
            {
                first.uid: LowLevelIntent(0.0, 0.0, 0.0),
                1: LowLevelIntent(0.0, 0.0, 0.0),
            },
        )

    assert shield.telemetry["shield_fail_closed"] == 1
    assert shield.telemetry["joint_safety_fail_closed"] == 1
    assert shield.telemetry["joint_search_candidates"] == 0
    assert shield.telemetry["operational_buffer_takeovers"] == 0
    assert exc_info.value.diagnostic["operational_threshold"] == pytest.approx(80.0)


def test_normal_flow_candidate_remains_checked_by_exact_trajectory_hash() -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.INNER, soc=0.90)

    env.step({str(uav.uid): np.asarray((1.0, 0.0, 0.0), dtype=np.float32)})

    assert uav.committed_plan is not None
    assert env.safety.checked_trajectory_hashes[uav.committed_plan.plan_id] == uav.committed_plan.trajectory_hash
