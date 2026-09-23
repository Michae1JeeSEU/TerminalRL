"""Contracts for the opt-in continuous inward normal final approach."""

from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.merge_scheduler import MergeReservation
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    UAVState,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _env(*, geometry_mode: str = "brake_bezier_v1") -> tuple[TerminalParallelEnv, UAVState, int]:
    config = SimConfig(
        arrival_model="finite_cohort",
        total_requested_uavs=0,
        arrival_rate_per_second=0.0,
        final_approach_geometry_mode=geometry_mode,
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
    reservation = MergeReservation("final-test", uav.uid, "ring_inner", "final", env.now, None, None, None)
    lz_id = env.lz_manager.choose_rule_zone(uav, env.now)
    assert lz_id is not None
    return env, uav, lz_id


def test_default_final_approach_is_byte_compatible_with_explicit_default_mode() -> None:
    implicit, uav_implicit, lz_implicit = _env()
    explicit, uav_explicit, lz_explicit = _env(geometry_mode="brake_bezier_v1")
    reservation_implicit = MergeReservation("final-test", uav_implicit.uid, "ring_inner", "final", 0.0, None, None, None)
    reservation_explicit = MergeReservation("final-test", uav_explicit.uid, "ring_inner", "final", 0.0, None, None, None)

    left = implicit.generator.final_approach(
        uav_implicit, implicit.lz_manager.zones[lz_implicit], reservation_implicit,
        0.0, scheduled_delay=12.0, speed_residual_mps=0.5, airspace=implicit.airspace,
    )
    right = explicit.generator.final_approach(
        uav_explicit, explicit.lz_manager.zones[lz_explicit], reservation_explicit,
        0.0, scheduled_delay=12.0, speed_residual_mps=0.5, airspace=explicit.airspace,
    )

    assert left.mode == right.mode == "final_approach"
    assert left.metadata == right.metadata
    assert left.trajectory_hash == right.trajectory_hash
    assert len(left.points) == len(right.points)
    for first, second in zip(left.points, right.points):
        assert first.t == second.t
        np.testing.assert_array_equal(first.position, second.position)
        np.testing.assert_array_equal(first.velocity, second.velocity)
        np.testing.assert_array_equal(first.acceleration, second.acceleration)


def test_continuous_inner_final_approach_reuses_corridor_with_normal_semantics() -> None:
    env, uav, lz_id = _env(geometry_mode="continuous_inward_v1")
    reservation = MergeReservation("final-test", uav.uid, "ring_inner", "final", 0.0, None, None, None)
    plan = env.generator.final_approach(
        uav, env.lz_manager.zones[lz_id], reservation, 0.0,
        scheduled_delay=12.0, speed_residual_mps=-0.5, airspace=env.airspace,
    )

    assert plan.mode == "final_approach"
    assert plan.metadata["target_phase"] == "landing"
    assert plan.metadata["final_approach_geometry"] == "continuous_inward_v1"
    assert plan.metadata["scheduled_delay"] == pytest.approx(12.0)
    assert plan.metadata["scheduled_departure"] == pytest.approx(12.0)
    assert plan.metadata["speed_residual_mps"] == pytest.approx(-0.5)
    assert plan.metadata["lz_id"] == lz_id
    assert plan.metadata["source_ring_role"] == RingRole.INNER.value
    assert plan.reservation_ids == (reservation.reservation_id,)
    assert plan.metadata["vehicle_profile_id"] == env.config.vehicle_profile_id
    assert plan.metadata["vehicle_profile_version"] == env.config.vehicle_profile.profile_version
    assert "emergency_landing" not in plan.metadata.values()
    assert uav.emergency_landing is False
    assert env.generator.validate_kinematics(plan) == []
    assert np.allclose(plan.points[0].position, uav.position)
    assert np.allclose(plan.points[-1].position, env.lz_manager.zones[lz_id].position)

    departure_index = next(
        index for index, point in enumerate(plan.points)
        if point.t == pytest.approx(plan.metadata["scheduled_departure"])
    )
    ring = env.airspace.actual_ring(RingRole.INNER)
    assert np.linalg.norm(plan.points[departure_index].position[:2]) == pytest.approx(ring.radius)
    assert np.linalg.norm(plan.points[departure_index].velocity[:2]) == pytest.approx(uav.speed)
    assert plan.points[departure_index].t > plan.points[departure_index - 1].t


def test_continuous_mode_falls_back_without_inner_airspace_context() -> None:
    env, uav, lz_id = _env(geometry_mode="continuous_inward_v1")
    reservation = MergeReservation("final-test", uav.uid, "ring_inner", "final", 0.0, None, None, None)
    fallback = env.generator.final_approach(
        uav, env.lz_manager.zones[lz_id], reservation, 0.0,
        scheduled_delay=0.0, airspace=None,
    )

    assert fallback.mode == "final_approach"
    assert fallback.metadata["target_phase"] == "landing"
    assert "final_approach_geometry" not in fallback.metadata
    assert env.generator.validate_kinematics(fallback) == []


def test_continuous_mode_retains_default_for_non_inner_source() -> None:
    env, uav, lz_id = _env(geometry_mode="continuous_inward_v1")
    uav.phase = FlightPhase.RING_OUTER
    uav.ring_role = RingRole.OUTER
    uav.ring_progress = 0.0
    ring = env.airspace.actual_ring(RingRole.OUTER)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    reservation = MergeReservation("final-test", uav.uid, "ring_outer", "final", 0.0, None, None, None)

    plan = env.generator.final_approach(
        uav, env.lz_manager.zones[lz_id], reservation, 0.0,
        scheduled_delay=0.0, airspace=env.airspace,
    )

    assert plan.mode == "final_approach"
    assert plan.metadata["target_phase"] == "landing"
    assert "final_approach_geometry" not in plan.metadata
    assert env.generator.validate_kinematics(plan) == []


def test_invalid_final_approach_geometry_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="final_approach_geometry_mode"):
        SimConfig(final_approach_geometry_mode="unknown")
