"""Focused tests for the standalone candidate energy/LZ budget calculator."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.energy import EnergyModel
from terminal_marl.core.energy_reachability import (
    compute_candidate_energy_budget,
    minimum_immediate_emergency_soc,
)
from terminal_marl.core.landing_zone import LZStatus, LandingZoneManager
from terminal_marl.core.merge_scheduler import MergeReservation
from terminal_marl.core.trajectory_generator import TrajectoryGenerator, TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    UAVState,
)


def _components() -> tuple[SimConfig, DynamicAirspace, LandingZoneManager, TrajectoryGenerator, EnergyModel]:
    config = SimConfig(
        arrival_model="finite_cohort",
        total_requested_uavs=0,
        arrival_rate_per_second=0.0,
        energy_model_mode="phase_motion_v2",
        emergency_landing_enabled=True,
    )
    return (
        config,
        DynamicAirspace(config),
        LandingZoneManager(),
        TrajectoryGenerator(config),
        EnergyModel(config),
    )


def _ring_uav(
    config: SimConfig,
    airspace: DynamicAirspace,
    role: RingRole = RingRole.OUTER,
    *,
    uid: int = 1,
    mission: MissionType = MissionType.STANDARD,
    soc: float = 0.80,
) -> UAVState:
    request = ArrivalRequest(
        uid,
        0.0,
        0,
        mission,
        1,
        config.energy_capacity,
        soc,
    )
    queued = QueuedArrivalState(request, request.energy_capacity * soc)
    uav = UAVState.from_request(uid, request, queued, np.zeros(3), 0.0)
    uav.phase = {
        RingRole.OUTER: FlightPhase.RING_OUTER,
        RingRole.MIDDLE: FlightPhase.RING_MIDDLE,
        RingRole.INNER: FlightPhase.RING_INNER,
    }[role]
    uav.ring_role = role
    uav.ring_progress = 0.0
    ring = airspace.actual_ring(role)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    uav.acceleration = np.asarray(
        (-(ring.nominal_speed**2) / ring.radius, 0.0, 0.0),
        dtype=float,
    )
    return uav


def _transfer_candidate(
    uav: UAVState,
    airspace: DynamicAirspace,
    generator: TrajectoryGenerator,
    *,
    target: RingRole = RingRole.MIDDLE,
) -> TrajectoryPlan:
    reservation = MergeReservation(
        "candidate-transfer",
        uav.uid,
        uav.phase.value,
        target.value,
        220.0,
        0.0,
        None,
        None,
    )
    return generator.ring_transfer(
        uav,
        uav.ring_role or RingRole.OUTER,
        target,
        reservation,
        airspace,
        0.0,
    )


def _short_transfer_candidate(
    uav: UAVState,
    airspace: DynamicAirspace,
    target: RingRole = RingRole.MIDDLE,
) -> TrajectoryPlan:
    target_ring = airspace.actual_ring(target)
    endpoint = np.asarray((target_ring.radius, 0.0, target_ring.height), dtype=float)
    return TrajectoryPlan(
        "short-transfer",
        uav.uid,
        "ring_transfer",
        (
            TrajectoryPoint(0.0, np.asarray(uav.position, dtype=float), uav.velocity),
            TrajectoryPoint(
                1.0,
                endpoint,
                np.asarray((0.0, target_ring.nominal_speed, 0.0), dtype=float),
            ),
        ),
        0.0,
        1.0,
        metadata={"target_phase": "ring_middle"},
    )


def _final_candidate(
    uav: UAVState,
    zone,
    airspace: DynamicAirspace,
    generator: TrajectoryGenerator,
) -> TrajectoryPlan:
    reservation = MergeReservation(
        "candidate-final",
        uav.uid,
        FlightPhase.RING_INNER.value,
        "final",
        0.0,
        0.0,
        None,
        None,
    )
    return generator.final_approach(uav, zone, reservation, 0.0, airspace=airspace)


def _manager_snapshot(manager: LandingZoneManager):
    return tuple(
        (
            zone.status,
            zone.reserved_by,
            zone.reservation_start,
            zone.expected_release_at,
            zone.occupied_by,
            zone.service_remaining,
        )
        for zone in manager.zones
    )


def test_final_budget_accepts_free_or_owned_pad_without_wait() -> None:
    config, airspace, manager, generator, energy = _components()
    uav = _ring_uav(config, airspace, RingRole.INNER)
    zone = manager.zones[0]
    candidate = _final_candidate(uav, zone, airspace, generator)
    before = _manager_snapshot(manager)

    result = compute_candidate_energy_budget(
        uav,
        candidate,
        "final",
        0.0,
        airspace,
        manager,
        generator,
        energy,
        config,
    )

    expected = (
        result.candidate_energy
        + energy.min_landing_soc * uav.energy_capacity
        + config.emergency_energy_margin_seconds * config.power_reference
    )
    assert result.selected_lz_id == zone.lz_id
    assert result.pad_wait_seconds == 0.0
    assert result.fallback_route_energy == 0.0
    assert result.required_energy == pytest.approx(expected)
    assert result.slack_energy == pytest.approx(uav.energy_remaining - expected)
    assert _manager_snapshot(manager) == before

    manager.reserve(zone.lz_id, 99, 0.0)
    blocked = compute_candidate_energy_budget(
        uav,
        candidate,
        "final",
        0.0,
        airspace,
        manager,
        generator,
        energy,
        config,
    )
    assert blocked.selected_lz_id is None
    assert math.isinf(blocked.required_energy)
    assert blocked.slack_energy == -math.inf

    owned_manager = LandingZoneManager()
    owned_manager.reserve(zone.lz_id, uav.uid, 0.0)
    owned = compute_candidate_energy_budget(
        uav,
        candidate,
        "final",
        0.0,
        airspace,
        owned_manager,
        generator,
        energy,
        config,
    )
    assert owned.selected_lz_id == zone.lz_id
    assert owned.pad_wait_seconds == 0.0


def test_transfer_budget_uses_free_lz_and_keeps_inputs_unchanged() -> None:
    config, airspace, manager, generator, energy = _components()
    uav = _ring_uav(config, airspace, RingRole.OUTER)
    candidate = _transfer_candidate(uav, airspace, generator)
    position_before = uav.position.copy()
    velocity_before = uav.velocity.copy()
    before = _manager_snapshot(manager)

    result = compute_candidate_energy_budget(
        uav,
        candidate,
        RingRole.MIDDLE,
        0.0,
        airspace,
        manager,
        generator,
        energy,
        config,
    )

    assert result.selected_lz_id in range(config.num_landing_zones)
    assert result.pad_wait_seconds == 0.0
    assert result.fallback_route_energy > 0.0
    assert result.required_energy >= result.candidate_energy
    assert result.slack_energy == pytest.approx(uav.energy_remaining - result.required_energy)
    assert np.array_equal(uav.position, position_before)
    assert np.array_equal(uav.velocity, velocity_before)
    assert _manager_snapshot(manager) == before

    minimum_soc = minimum_immediate_emergency_soc(
        uav,
        RingRole.OUTER,
        0.0,
        airspace,
        manager,
        generator,
        energy,
        config,
    )
    assert math.isfinite(minimum_soc)
    assert 0.0 < minimum_soc < 1.0


def test_first_feasible_mode_matches_default_feasibility() -> None:
    config, airspace, manager, generator, energy = _components()
    uav = _ring_uav(config, airspace, RingRole.OUTER, soc=0.80)
    candidate = _transfer_candidate(uav, airspace, generator)

    default = compute_candidate_energy_budget(
        uav, candidate, RingRole.MIDDLE, 0.0, airspace, manager, generator, energy, config
    )
    first = compute_candidate_energy_budget(
        uav,
        candidate,
        RingRole.MIDDLE,
        0.0,
        airspace,
        manager,
        generator,
        energy,
        config,
        stop_at_first_feasible=True,
    )

    assert default.selected_lz_id is not None
    assert first.selected_lz_id is not None
    assert (default.slack_energy >= -1e-12) == (first.slack_energy >= -1e-12)
    assert first.slack_energy >= -1e-12


def test_first_feasible_mode_returns_same_infeasible_minimum() -> None:
    config, airspace, manager, generator, energy = _components()
    uav = _ring_uav(config, airspace, RingRole.OUTER, soc=0.37)
    candidate = _transfer_candidate(uav, airspace, generator)

    default = compute_candidate_energy_budget(
        uav, candidate, RingRole.MIDDLE, 0.0, airspace, manager, generator, energy, config
    )
    first = compute_candidate_energy_budget(
        uav,
        candidate,
        RingRole.MIDDLE,
        0.0,
        airspace,
        manager,
        generator,
        energy,
        config,
        stop_at_first_feasible=True,
    )

    assert default.slack_energy < -1e-12
    assert first == default


def test_first_feasible_mode_generates_fewer_fallback_routes(monkeypatch) -> None:
    config, airspace, manager, generator, energy = _components()
    uav = _ring_uav(config, airspace, RingRole.OUTER, soc=0.80)
    candidate = _transfer_candidate(uav, airspace, generator)
    original = generator.emergency_landing
    calls: list[int] = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(generator, "emergency_landing", counted)
    default = compute_candidate_energy_budget(
        uav, candidate, RingRole.MIDDLE, 0.0, airspace, manager, generator, energy, config
    )
    default_calls = len(calls)
    calls.clear()
    first = compute_candidate_energy_budget(
        uav,
        candidate,
        RingRole.MIDDLE,
        0.0,
        airspace,
        manager,
        generator,
        energy,
        config,
        stop_at_first_feasible=True,
    )

    assert default.selected_lz_id is not None
    assert first.selected_lz_id is not None
    assert first.slack_energy >= -1e-12
    assert default_calls == config.num_landing_zones
    assert 0 < len(calls) < default_calls


def test_reserved_known_release_is_used_and_unknown_slots_are_skipped() -> None:
    config, airspace, manager, generator, energy = _components()
    uav = _ring_uav(config, airspace, RingRole.OUTER)
    candidate = _transfer_candidate(uav, airspace, generator)
    expected_eta = candidate.end_time + 25.0

    for zone_id in range(1, config.num_landing_zones):
        assert manager.reserve(zone_id, 99, 0.0).approved
    assert manager.reserve(0, 99, 0.0).approved
    assert manager.note_committed_landing(0, 99, expected_eta)

    result = compute_candidate_energy_budget(
        uav,
        candidate,
        RingRole.MIDDLE,
        0.0,
        airspace,
        manager,
        generator,
        energy,
        config,
    )
    assert result.selected_lz_id == 0
    assert result.pad_wait_seconds == pytest.approx(25.0)
    assert math.isfinite(result.required_energy)

    unknown_manager = LandingZoneManager()
    for zone_id in range(config.num_landing_zones):
        assert unknown_manager.reserve(zone_id, 99, 0.0).approved
    blocked = compute_candidate_energy_budget(
        uav,
        candidate,
        RingRole.MIDDLE,
        0.0,
        airspace,
        unknown_manager,
        generator,
        energy,
        config,
    )
    assert blocked.selected_lz_id is None
    assert blocked.required_energy == math.inf
    assert blocked.slack_energy == -math.inf


def test_occupied_lz_wait_uses_service_release_time() -> None:
    config, airspace, manager, generator, energy = _components()
    uav = _ring_uav(config, airspace, RingRole.OUTER, mission=MissionType.CARGO)
    candidate = _short_transfer_candidate(uav, airspace)

    for zone_id in range(1, config.num_landing_zones):
        assert manager.reserve(zone_id, 99, 0.0).approved
    assert manager.reserve(0, 99, 0.0).approved
    manager.touchdown(0, 99, MissionType.CARGO)
    expected_wait = max(0.0, manager.earliest_release_time(0, 0.0) - candidate.end_time)

    result = compute_candidate_energy_budget(
        uav,
        candidate,
        RingRole.MIDDLE,
        0.0,
        airspace,
        manager,
        generator,
        energy,
        config,
    )
    assert result.selected_lz_id == 0
    assert result.pad_wait_seconds == pytest.approx(expected_wait)
    assert result.required_energy >= result.candidate_energy


def test_energy_and_slack_are_monotonic_with_remaining_energy() -> None:
    config, airspace, manager, generator, energy = _components()
    uav = _ring_uav(config, airspace, RingRole.OUTER, soc=0.60)
    candidate = _transfer_candidate(uav, airspace, generator)
    base = compute_candidate_energy_budget(
        uav,
        candidate,
        RingRole.MIDDLE,
        0.0,
        airspace,
        manager,
        generator,
        energy,
        config,
    )
    more_energy = replace(uav, energy_remaining=uav.energy_remaining + 100.0)
    higher = compute_candidate_energy_budget(
        more_energy,
        candidate,
        RingRole.MIDDLE,
        0.0,
        airspace,
        manager,
        generator,
        energy,
        config,
    )
    assert higher.required_energy == pytest.approx(base.required_energy)
    assert higher.slack_energy == pytest.approx(base.slack_energy + 100.0)
    assert base.required_energy >= base.candidate_energy
