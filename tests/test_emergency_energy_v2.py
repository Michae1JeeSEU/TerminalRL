"""Focused contracts for emergency landing and phase/motion energy V2."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.energy import EnergyModel
from terminal_marl.core.arrival_process import ArrivalProcess
from terminal_marl.core.landing_zone import LZStatus
from terminal_marl.core.observations import LOW_OBS_V2, build_low_observation
from terminal_marl.core.priority import PriorityEngine
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    TerminalOutcome,
    UAVState,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def _config(**overrides) -> SimConfig:
    values = {
        "arrival_model": "finite_cohort",
        "total_requested_uavs": 0,
        "arrival_rate_per_second": 0.0,
        "energy_model_mode": "phase_motion_v2",
        "emergency_landing_enabled": True,
        "emergency_trigger_soc": 0.35,
        "max_simulation_seconds": 600.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _ring_uav(env: TerminalParallelEnv, role: RingRole, *, soc: float, arrival_id: int = 0) -> UAVState:
    request = ArrivalRequest(
        arrival_id,
        0.0,
        0,
        MissionType.STANDARD,
        1,
        env.config.energy_capacity,
        soc,
    )
    queued = QueuedArrivalState(request, request.energy_capacity * soc)
    uav = UAVState.from_request(arrival_id, request, queued, np.zeros(3), env.now)
    uav.phase = {
        RingRole.OUTER: FlightPhase.RING_OUTER,
        RingRole.MIDDLE: FlightPhase.RING_MIDDLE,
        RingRole.INNER: FlightPhase.RING_INNER,
    }[role]
    uav.ring_role = role
    uav.ring_progress = 0.0
    ring = env.airspace.actual_ring(role)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    uav.acceleration = np.asarray(
        (-(ring.nominal_speed**2) / ring.radius, 0.0, 0.0), dtype=float
    )
    env.active[uav.uid] = uav
    env.ledger.request(request.arrival_id)
    return uav


def _motion_plan() -> TrajectoryPlan:
    return TrajectoryPlan(
        "energy-test-plan",
        0,
        "energy_test",
        (
            TrajectoryPoint(0.0, np.asarray((0.0, 0.0, 0.0)), np.asarray((10.0, 0.0, 0.0))),
            TrajectoryPoint(1.0, np.asarray((10.0, 0.0, 0.0)), np.asarray((10.0, 0.0, 0.0))),
            TrajectoryPoint(2.0, np.asarray((20.0, 0.0, -1.0)), np.asarray((10.0, 0.0, -1.0))),
        ),
        0.0,
        2.0,
    )


def test_phase_motion_v2_integrates_motion_power_and_legacy_mode_remains_compatible() -> None:
    legacy_config = SimConfig(energy_model_mode="legacy_phase")
    legacy = EnergyModel(legacy_config)
    assert legacy_config.energy_model_mode == "legacy_phase"
    assert legacy.phase_power(FlightPhase.TRANSFER_O2M) == pytest.approx(
        legacy_config.power_reference * legacy_config.power_ring_transfer
    )
    assert legacy.motion_power(np.asarray((10.0, 0.0, -1.0))) == pytest.approx(
        legacy_config.power_reference
    )
    assert legacy.estimate_plan_energy(_motion_plan()) == pytest.approx(
        2.0 * legacy_config.power_reference
    )

    v2_config = SimConfig(energy_model_mode="phase_motion_v2")
    v2 = EnergyModel(v2_config)
    assert v2.phase_power(FlightPhase.EXTERNAL_QUEUE) == pytest.approx(42.0 / 36.0)
    assert v2.phase_power(FlightPhase.TRANSFER_O2M) == pytest.approx(28.0 / 36.0)
    assert v2.phase_power(FlightPhase.EMERGENCY_LANDING) == pytest.approx(42.0 / 36.0)
    assert v2.motion_power(np.asarray((10.0, 0.0, 0.0))) == pytest.approx(1.0)
    assert v2.motion_power(np.asarray((10.0, 0.0, -1.0))) == pytest.approx(28.0 / 36.0)
    assert v2.motion_power(np.asarray((0.0, 0.0, 0.0))) == pytest.approx(42.0 / 36.0)
    assert v2.estimate_plan_energy(_motion_plan()) == pytest.approx(1.0 + 28.0 / 36.0)
    request = ArrivalRequest(20, 0.0, 0, MissionType.STANDARD, 1, v2_config.energy_capacity, 0.50)
    queued = QueuedArrivalState(request, request.energy_capacity * request.initial_soc)
    assert v2.admission_required_energy(queued) == pytest.approx(240.9 + 0.10 * 1500.0)
    assert legacy.admission_required_energy(queued) == pytest.approx(120.0)


def test_phase_motion_v2_default_uniform_soc_respects_solo_landing_floor() -> None:
    count = 256
    config = SimConfig(
        energy_model_mode="phase_motion_v2",
        total_requested_uavs=count,
        fixed_arrival_schedule=(0.0,) * count,
    )
    process = ArrivalProcess(config, seed=840000)
    lower, upper = process._uniform_soc_bounds()
    expected_lower = max(
        config.soc_warning,
        config.soc_reserve,
        config.min_landing_soc + config.phase_motion_nominal_solo_energy / config.energy_capacity,
    )
    requests = process.requests_until(0.0)

    assert lower == pytest.approx(expected_lower)
    assert lower == pytest.approx(0.35)
    assert upper == pytest.approx(1.0)
    assert len(requests) == count
    assert all(expected_lower <= request.initial_soc <= 1.0 for request in requests)


def test_explicit_phase_motion_v2_soc_range_remains_low_soc_stress_range() -> None:
    count = 256
    config = SimConfig(
        energy_model_mode="phase_motion_v2",
        initial_soc_min=0.20,
        initial_soc_max=0.55,
        total_requested_uavs=count,
        fixed_arrival_schedule=(0.0,) * count,
    )
    process = ArrivalProcess(config, seed=840000)
    lower, upper = process._uniform_soc_bounds()
    requests = process.requests_until(0.0)
    solo_floor = config.min_landing_soc + config.phase_motion_nominal_solo_energy / config.energy_capacity

    assert (lower, upper) == pytest.approx((0.20, 0.55))
    assert min(request.initial_soc for request in requests) < solo_floor


def test_legacy_default_uniform_soc_range_remains_point_three_five_to_one() -> None:
    count = 256
    config = SimConfig(
        energy_model_mode="legacy_phase",
        total_requested_uavs=count,
        fixed_arrival_schedule=(0.0,) * count,
    )
    process = ArrivalProcess(config, seed=840000)
    lower, upper = process._uniform_soc_bounds()
    requests = process.requests_until(0.0)

    assert (lower, upper) == pytest.approx((0.35, 1.0))
    assert all(0.35 <= request.initial_soc <= 1.0 for request in requests)


def test_phase_motion_v2_default_soc_floor_above_one_is_rejected() -> None:
    reference = SimConfig(energy_model_mode="phase_motion_v2")
    # Route energy equal to capacity, plus the landing reserve, exceeds 1.0.
    # The former 1200 / 1500 + .1 fixture was only .9 and was feasible.
    config = replace(reference, phase_motion_nominal_solo_energy=reference.energy_capacity)
    with pytest.raises(ValueError, match="default initial SOC lower bound exceeds 1.0"):
        ArrivalProcess(config, seed=840000)


@pytest.mark.parametrize("role", tuple(RingRole))
def test_emergency_route_from_each_ring_is_kinematically_feasible_and_safety_approved(role: RingRole) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, role, soc=0.30)
    zone = env.lz_manager.zones[0]

    plan = env.generator.emergency_landing(uav, zone, env.airspace, env.now)
    result = env.safety.filter_plan(plan, {}, WorldSnapshot(env.now, env.active, env.airspace))

    assert plan.mode == "emergency_landing"
    assert plan.metadata["target_phase"] == "emergency_landing"
    assert np.allclose(plan.points[0].position, uav.position)
    assert np.allclose(plan.points[0].velocity, uav.velocity)
    assert plan.metadata["source_ring_role"] == role.value
    staging = np.asarray(plan.metadata["staging_point"], dtype=float)
    assert np.allclose(plan.points[-17].position, staging)
    assert np.allclose(plan.points[-1].position, zone.position)
    source_radius = float(np.linalg.norm(uav.position[:2]))
    assert float(np.linalg.norm(staging[:2])) < source_radius
    assert any(float(np.linalg.norm(point.position[:2])) < source_radius - 1.0 for point in plan.points[1:])
    assert plan.end_time > plan.start_time
    assert env.generator.validate_kinematics(plan) == []
    assert result.approved
    assert result.reason == ""


def test_insufficient_energy_does_not_commit_emergency_route_or_reserve_pad() -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.MIDDLE, soc=0.5 * env.config.min_landing_soc)

    assert not env._try_emergency_landing(uav)
    assert uav.committed_plan is None
    assert uav.assigned_lz is None
    assert not uav.emergency_landing
    assert all(zone.status is LZStatus.FREE for zone in env.lz_manager.zones)
    assert env.telemetry["emergency_attempts"] == 1
    assert env.telemetry["emergency_blocked_energy"] >= 1
    assert env.telemetry["emergency_committed"] == 0


def test_queue_diversion_records_actual_energy_snapshot() -> None:
    env = TerminalParallelEnv(_config())
    request = ArrivalRequest(7, 0.0, 0, MissionType.STANDARD, 1,
                             env.config.energy_capacity, 0.24)
    env.ledger.request(request.arrival_id)
    env.admission.enqueue(request)

    _, _, _, _, info = env.step({})
    row = info["energy_outcomes"][0]
    assert row["arrival_id"] == request.arrival_id
    assert row["outcome"] == TerminalOutcome.DIVERTED_LOW_SOC.value
    assert row["phase"] == FlightPhase.EXTERNAL_QUEUE.value
    assert row["soc"] is not None and row["soc"] < request.initial_soc
    assert row["energy_remaining"] is not None
    assert row["energy_consumed"] is not None and row["energy_consumed"] > 0.0


def test_emergency_mode_keeps_under_reserve_aircraft_active_until_explicit_energy_outcome() -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.INNER, soc=0.5 * env.config.min_landing_soc)
    env.step({str(uav.uid): np.zeros(3, dtype=float)})
    assert uav.uid in env.active
    assert uav.arrival_id not in env.ledger.outcomes
    assert env.telemetry["emergency_blocked_energy"] > 0

    uav.energy_remaining = 0.1
    env.step({str(uav.uid): np.zeros(3, dtype=float)})
    assert env.ledger.outcomes[uav.arrival_id] is TerminalOutcome.FAILED_ENERGY
    assert uav.terminal_reason == "energy_depleted"


def test_emergency_curriculum_prioritizes_lower_soc_with_shared_priority_rule() -> None:
    config = _config()
    engine = PriorityEngine(config)
    request_low = ArrivalRequest(1, 0.0, 0, MissionType.STANDARD, 1, config.energy_capacity, 0.30)
    request_high = ArrivalRequest(2, 0.0, 0, MissionType.STANDARD, 1, config.energy_capacity, 0.80)
    low = QueuedArrivalState(request_low, request_low.energy_capacity * request_low.initial_soc)
    high = QueuedArrivalState(request_high, request_high.energy_capacity * request_high.initial_soc)

    assert engine.compute(low) > engine.compute(high)
    assert engine.rank_key(low) < engine.rank_key(high)

    env = TerminalParallelEnv(config)
    high_uav = _ring_uav(env, RingRole.OUTER, soc=0.34, arrival_id=2)
    low_uav = _ring_uav(env, RingRole.MIDDLE, soc=0.30, arrival_id=1)
    assert env._emergency_processing_order() == [low_uav, high_uav]


def test_successful_emergency_landing_uses_service_pad_penalizes_and_keeps_episode_open() -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.INNER, soc=0.34)
    uid = str(uav.uid)
    assert env._try_emergency_landing(uav)
    emergency_observation = build_low_observation(uav, env)
    phase_slice = LOW_OBS_V2.slices["phase"]
    assert emergency_observation[phase_slice].sum() == pytest.approx(1.0)
    assert emergency_observation[phase_slice][6] == pytest.approx(1.0)  # existing FINAL slot

    touchdown_info = None
    touchdown_rewards = None
    touchdown_terminated = None
    for _ in range(220):
        _, rewards, terminated, _, info = env.step({uid: np.zeros(3, dtype=float)})
        if env.telemetry["emergency_touchdowns"]:
            touchdown_info = info
            touchdown_rewards = rewards
            touchdown_terminated = terminated
            break

    assert touchdown_info is not None
    assert touchdown_rewards is not None
    assert touchdown_terminated == {uid: True}
    assert uav.terminal_outcome is None
    assert uav.uid in env.service_uavs
    assert not env.active
    assert not env.ledger.closes()
    assert len(env.lz_manager.zones) == 6
    assigned_zone = env.lz_manager.zones[uav.assigned_lz]
    assert assigned_zone.status is LZStatus.OCCUPIED
    assert assigned_zone.occupied_by == uav.uid
    assert touchdown_info["reward_components"][uid]["emergency"] == pytest.approx(
        -env.config.low_level_emergency_landing_penalty
    )

    # A later simulator step advances ground service after the UAV leaves the
    # airborne agent set; this demonstrates that emergency touchdown is not an
    # episode-wide stop condition.
    env.step({})
    assert uav.uid in env.service_uavs
    assert env.ledger.outcomes == {}

    for _ in range(200):
        env.step({})
        if env.ledger.outcomes:
            break
    assert env.ledger.outcomes[uav.arrival_id] is TerminalOutcome.EMERGENCY_LANDED
    assert not env.service_uavs
    assert env.completed_uavs[-1].uid == uav.uid
