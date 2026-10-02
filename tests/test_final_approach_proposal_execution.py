"""Contracts for the opt-in final-approach speed/timing execution mode."""

from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.merge_scheduler import (
    LowLevelIntent,
    MergeProposal,
    MergeReservation,
    RepairCandidate,
)
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    UAVState,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def _env(*, execution_mode: str = "legacy", geometry_mode: str = "continuous_inward_v1"):
    action_schema = (
        "low_action_schema_v4_2_hierarchical"
        if execution_mode == "speed_timing_v1"
        else "low_action_schema_v4_joint_discrete"
    )
    config = SimConfig(
        arrival_model="finite_cohort",
        total_requested_uavs=0,
        arrival_rate_per_second=0.0,
        action_schema_version=action_schema,
        final_approach_geometry_mode=geometry_mode,
        final_approach_execution_mode=execution_mode,
        max_simulation_seconds=120.0,
    )
    env = TerminalParallelEnv(config)
    request = ArrivalRequest(1, 0.0, 0, MissionType.STANDARD, 1, config.energy_capacity, 1.0)
    queued = QueuedArrivalState(request, request.energy_capacity)
    uav = UAVState.from_request(0, request, queued, np.zeros(3), env.now)
    uav.phase = FlightPhase.RING_INNER
    uav.ring_role = RingRole.INNER
    uav.ring_progress = 0.0
    ring = env.airspace.actual_ring(RingRole.INNER)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    uav.acceleration = np.asarray((-(ring.nominal_speed ** 2) / ring.radius, 0.0, 0.0), dtype=float)
    env.active[uav.uid] = uav
    lz_id = env.lz_manager.choose_rule_zone(uav, env.now)
    assert lz_id is not None
    return env, uav, lz_id


def _reservation(uav: UAVState, merge_time: float) -> MergeReservation:
    return MergeReservation(
        f"final-{merge_time}", uav.uid, "ring_inner", "final",
        merge_time, None, None, None,
    )


def test_legacy_default_explicit_and_emergency_hashes_remain_unchanged() -> None:
    implicit, implicit_uav, implicit_lz = _env()
    explicit, explicit_uav, explicit_lz = _env(execution_mode="legacy")
    left = implicit.generator.final_approach(
        implicit_uav, implicit.lz_manager.zones[implicit_lz], _reservation(implicit_uav, 0.0),
        0.0, scheduled_delay=12.0, speed_residual_mps=0.5, airspace=implicit.airspace,
    )
    right = explicit.generator.final_approach(
        explicit_uav, explicit.lz_manager.zones[explicit_lz], _reservation(explicit_uav, 0.0),
        0.0, scheduled_delay=12.0, speed_residual_mps=0.5, airspace=explicit.airspace,
    )
    assert left.metadata == right.metadata
    assert left.trajectory_hash == right.trajectory_hash

    legacy, legacy_uav, legacy_lz = _env()
    timed, timed_uav, timed_lz = _env(execution_mode="speed_timing_v1")
    emergency_left = legacy.generator.emergency_landing(
        legacy_uav, legacy.lz_manager.zones[legacy_lz], legacy.airspace, 0.0,
    )
    emergency_right = timed.generator.emergency_landing(
        timed_uav, timed.lz_manager.zones[timed_lz], timed.airspace, 0.0,
    )
    assert emergency_left.metadata == emergency_right.metadata
    assert emergency_left.trajectory_hash == emergency_right.trajectory_hash


def test_speed_timing_uses_target_speed_and_reservation_departure() -> None:
    plans = []
    for residual, expected in ((-2.0, 8.0), (0.0, 10.0), (2.0, 12.0)):
        env, uav, lz_id = _env(execution_mode="speed_timing_v1")
        plan = env.generator.final_approach(
            uav, env.lz_manager.zones[lz_id], _reservation(uav, 30.0), 0.0,
            speed_residual_mps=residual, airspace=env.airspace,
        )
        assert plan.metadata["requested_target_speed_mps"] == pytest.approx(expected)
        assert plan.metadata["accepted_target_speed_mps"] == pytest.approx(expected)
        assert plan.metadata["requested_reservation_departure"] == pytest.approx(30.0)
        assert plan.metadata["scheduled_departure"] == pytest.approx(30.0)
        assert plan.metadata["actual_entry_speed_mps"] == pytest.approx(expected, abs=1e-5)
        assert plan.metadata["final_approach_execution_mode"] == "speed_timing_v1"
        assert plan.metadata["speed_clamp_reason"] == "none"
        assert env.generator.validate_kinematics(plan) == []
        assert np.allclose(plan.points[0].position, uav.position)
        assert np.allclose(plan.points[0].velocity, uav.velocity)
        assert np.allclose(plan.points[0].acceleration, uav.acceleration)
        assert np.allclose(plan.points[-1].position, env.lz_manager.zones[lz_id].position)
        plans.append(plan)
    assert len({plan.trajectory_hash for plan in plans}) == 3
    assert len({round(plan.end_time, 8) for plan in plans}) == 3


def test_same_speed_late_slots_change_actual_departure_and_hash() -> None:
    plans = []
    for slot in (20.0, 25.0, 30.0):
        env, uav, lz_id = _env(execution_mode="speed_timing_v1")
        plan = env.generator.final_approach(
            uav, env.lz_manager.zones[lz_id], _reservation(uav, slot), 0.0,
            speed_residual_mps=0.0, airspace=env.airspace,
        )
        assert plan.metadata["scheduled_departure"] == pytest.approx(slot)
        assert plan.metadata["accepted_reservation_departure"] == pytest.approx(slot)
        assert plan.metadata["timing_clamped"] is False
        assert plan.metadata["timing_clamp_reason"] == "none"
        assert env.generator.validate_kinematics(plan) == []
        plans.append(plan)
    assert len({plan.trajectory_hash for plan in plans}) == 3


def test_early_reservation_is_clamped_by_finite_ring_braking() -> None:
    env, uav, lz_id = _env(execution_mode="speed_timing_v1")
    position = uav.position.copy()
    velocity = uav.velocity.copy()
    plan = env.generator.final_approach(
        uav, env.lz_manager.zones[lz_id], _reservation(uav, 1.0), 0.0,
        speed_residual_mps=0.0, airspace=env.airspace,
    )
    assert plan.metadata["scheduled_departure"] > 1.0
    assert plan.metadata["accepted_reservation_departure"] == pytest.approx(
        plan.metadata["scheduled_departure"]
    )
    assert plan.metadata["timing_clamped"] is True
    assert "finite_ring_braking" in plan.metadata["timing_clamp_reason"]
    assert plan.metadata["required_braking_duration_seconds"] > 0.0
    assert np.allclose(plan.points[0].position, position)
    assert np.allclose(plan.points[0].velocity, velocity)
    assert np.allclose(plan.points[0].acceleration, uav.acceleration)
    assert env.generator.validate_kinematics(plan) == []


def test_candidate_plan_reconciles_early_final_slot_to_checked_departure() -> None:
    env, uav, lz_id = _env(execution_mode="speed_timing_v1")
    env.scheduler.propose = lambda *_args, **_kwargs: MergeProposal(
        "final", 1.0, None, env.config.headway_i2f,
    )
    candidate = RepairCandidate("requested", LowLevelIntent(1.0, 0.0, 0.0))
    proposal, plan, reason = env._candidate_plan(
        uav,
        candidate,
        "final",
        WorldSnapshot(env.now, env.active, env.airspace),
        lz_id,
    )
    assert reason is None
    assert proposal is not None and plan is not None
    assert proposal.merge_time == pytest.approx(plan.metadata["scheduled_departure"])
    assert proposal.merge_time > 1.0
    assert plan.metadata["timing_clamped"] is True
    assert env.generator.validate_kinematics(plan) == []
    assert env.safety.checked_trajectory_hashes.get(plan.plan_id) == plan.trajectory_hash


def test_low_initial_speed_does_not_force_ring_acceleration() -> None:
    env, uav, lz_id = _env(execution_mode="speed_timing_v1")
    uav.speed = 8.0
    uav.velocity = np.asarray((0.0, 8.0, 0.0), dtype=float)
    plan = env.generator.final_approach(
        uav, env.lz_manager.zones[lz_id], _reservation(uav, 0.0), 0.0,
        speed_residual_mps=2.0, airspace=env.airspace,
    )
    assert plan.metadata["accepted_target_speed_mps"] == pytest.approx(12.0)
    assert plan.metadata["actual_entry_speed_mps"] == pytest.approx(8.0)
    assert plan.metadata["required_braking_duration_seconds"] == pytest.approx(0.0)
    assert env.generator.validate_kinematics(plan) == []


def test_speed_timing_missing_conditions_fail_closed() -> None:
    env, uav, lz_id = _env(execution_mode="speed_timing_v1")
    with pytest.raises(RuntimeError, match="inner-ring airspace context"):
        env.generator.final_approach(
            uav, env.lz_manager.zones[lz_id], _reservation(uav, 0.0), 0.0,
            airspace=None,
        )
    uav.ring_role = RingRole.OUTER
    uav.phase = FlightPhase.RING_OUTER
    with pytest.raises(RuntimeError, match="inner-ring airspace context"):
        env.generator.final_approach(
            uav, env.lz_manager.zones[lz_id], _reservation(uav, 0.0), 0.0,
            airspace=env.airspace,
        )


def test_speed_timing_config_requires_supported_combination() -> None:
    with pytest.raises(ValueError, match="final_approach_execution_mode"):
        SimConfig(final_approach_execution_mode="speed_timing_v1")
    with pytest.raises(ValueError, match="final_approach_execution_mode"):
        SimConfig(
            final_approach_execution_mode="speed_timing_v1",
            action_schema_version="low_action_schema_v4_2_hierarchical",
        )
    with pytest.raises(ValueError, match="final_approach_execution_mode"):
        SimConfig(
            final_approach_execution_mode="speed_timing_v1",
            action_schema_version="low_action_schema_v4_2_hierarchical",
            final_approach_geometry_mode="continuous_inward_v1",
            vehicle_execution_mode="legacy_kinematic",
        )
