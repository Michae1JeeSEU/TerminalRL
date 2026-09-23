from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.admission import AdmissionDecision, AdmissionManager
from terminal_marl.core.config import SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.energy import EnergyModel
from terminal_marl.core.landing_zone import LandingZoneManager
from terminal_marl.core.priority import PriorityEngine
from terminal_marl.core.uav import ArrivalRequest, MissionType


def _config(**overrides) -> SimConfig:
    values = {
        "action_schema_version": "low_action_schema_v4_2_hierarchical",
        "energy_model_mode": "phase_motion_v2",
        "emergency_landing_enabled": True,
        "emergency_priority_guard_enabled": True,
        "energy_reachability_guard_enabled": True,
        "arrival_model": "finite_cohort",
        "total_requested_uavs": 1,
        "arrival_rate_per_second": 0.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _request(config: SimConfig, soc: float = 0.5, mission: MissionType = MissionType.STANDARD) -> ArrivalRequest:
    return ArrivalRequest(1, 0.0, 0, mission, 1, config.energy_capacity, soc)


def _run(config: SimConfig, request: ArrivalRequest, lz_manager: LandingZoneManager):
    manager = AdmissionManager(config)
    manager.enqueue(request)
    energy = EnergyModel(config)
    priority = PriorityEngine(config)
    admitted, records = manager.step(
        0.0,
        {},
        DynamicAirspace(config),
        energy,
        priority,
        lz_manager=lz_manager,
    )
    return admitted, records, manager


def _required_free_energy(config: SimConfig, request: ArrivalRequest) -> float:
    manager = AdmissionManager(config)
    manager.enqueue(request)
    queued = manager.queue.items[0]
    energy = EnergyModel(config)
    return energy.admission_required_energy(queued) + config.emergency_energy_margin_seconds * config.power_reference


def _minimum_travel_time(config: SimConfig, lz_id: int) -> float:
    gate = AdmissionManager(config).gates[0]
    zone = LandingZoneManager().zones[lz_id]
    distance = float(np.linalg.norm(zone.position[:2] - gate.spawn_point[:2]))
    return distance / config.max_speed


def test_free_compatible_lz_has_no_extra_wait_and_admits() -> None:
    config = _config()
    request = _request(config, soc=0.5)
    admitted, records, manager = _run(config, request, LandingZoneManager())
    assert len(admitted) == 1
    assert records[0].decision is AdmissionDecision.ADMIT
    assert records[0].reason_code == "ok"
    assert not manager.queue.items


@pytest.mark.parametrize("status", ("reserved", "occupied"))
def test_late_reserved_or_occupied_eta_diverts_at_marginal_soc(status: str) -> None:
    config = _config()
    minimum_travel_time = _minimum_travel_time(config, 0)
    base_request = _request(config, soc=0.5)
    marginal_soc = (
        _required_free_energy(config, base_request) + 1.0
    ) / config.energy_capacity
    request = _request(config, soc=marginal_soc)
    lz_manager = LandingZoneManager()
    for zone in lz_manager.zones[1:]:
        zone.compatible_missions = frozenset({MissionType.CARGO})
    if status == "reserved":
        assert lz_manager.reserve(0, 99, 0.0).approved
        assert lz_manager.note_committed_landing(0, 99, minimum_travel_time + 10.0)
    else:
        assert lz_manager.reserve(0, 99, 0.0).approved
        lz_manager.touchdown(0, 99, MissionType.STANDARD)
        lz_manager.zones[0].service_remaining = minimum_travel_time + 10.0

    admitted, records, manager = _run(config, request, lz_manager)
    assert not admitted
    assert records[0].decision is AdmissionDecision.DIVERT
    assert records[0].reason_code == "low_soc_slot"
    assert records[0].request == request
    assert manager.queue.items == []


def test_known_near_eta_has_no_extra_wait_and_admits_at_marginal_soc() -> None:
    config = _config()
    base_request = _request(config, soc=0.5)
    marginal_soc = (_required_free_energy(config, base_request) + 1.0) / config.energy_capacity
    request = _request(config, soc=marginal_soc)
    lz_manager = LandingZoneManager()
    for zone in lz_manager.zones[1:]:
        zone.compatible_missions = frozenset({MissionType.CARGO})
    assert lz_manager.reserve(0, 99, 0.0).approved
    assert lz_manager.note_committed_landing(0, 99, _minimum_travel_time(config, 0) + 1.0)

    admitted, records, manager = _run(config, request, lz_manager)
    assert len(admitted) == 1
    assert records[0].decision is AdmissionDecision.ADMIT
    assert records[0].reason_code == "ok"
    assert not manager.queue.items


def test_all_unknown_reserved_slots_wait_and_remain_queued() -> None:
    config = _config()
    request = _request(config, soc=0.5)
    lz_manager = LandingZoneManager()
    for lz_id in range(len(lz_manager.zones)):
        assert lz_manager.reserve(lz_id, 100 + lz_id, 0.0).approved

    admitted, records, manager = _run(config, request, lz_manager)
    assert not admitted
    assert records[0].decision is AdmissionDecision.WAIT
    assert records[0].reason_code == "lz_slot_unknown"
    assert len(manager.queue.items) == 1


def test_external_wait_timeout_precedes_unknown_slot_wait() -> None:
    config = _config(max_external_wait_seconds=0.0)
    request = _request(config, soc=0.5)
    lz_manager = LandingZoneManager()
    for lz_id in range(len(lz_manager.zones)):
        assert lz_manager.reserve(lz_id, 100 + lz_id, 0.0).approved

    admitted, records, manager = _run(config, request, lz_manager)
    assert not admitted
    assert records[0].decision is AdmissionDecision.DIVERT
    assert records[0].reason_code == "external_wait_limit"
    assert not manager.queue.items


def test_incompatible_pad_is_skipped_when_compatible_pad_is_free() -> None:
    config = _config()
    request = _request(config, soc=0.5, mission=MissionType.STANDARD)
    lz_manager = LandingZoneManager()
    lz_manager.zones[0].compatible_missions = frozenset({MissionType.CARGO})
    assert lz_manager.reserve(0, 99, 0.0).approved

    admitted, records, manager = _run(config, request, lz_manager)
    assert len(admitted) == 1
    assert records[0].decision is AdmissionDecision.ADMIT
    assert not manager.queue.items


def test_missing_lz_manager_is_rejected_when_guard_is_enabled() -> None:
    config = _config()
    manager = AdmissionManager(config)
    manager.enqueue(_request(config))
    with pytest.raises(ValueError, match="requires lz_manager"):
        manager.step(0.0, {}, DynamicAirspace(config), EnergyModel(config), PriorityEngine(config))


def test_flag_off_ignores_lz_manager_and_preserves_sample_record() -> None:
    config = _config(energy_reachability_guard_enabled=False)
    request = _request(config, soc=0.5)
    left = AdmissionManager(config)
    right = AdmissionManager(config)
    left.enqueue(request)
    right.enqueue(request)
    energy = EnergyModel(config)
    priority = PriorityEngine(config)
    airspace = DynamicAirspace(config)
    lz_manager = LandingZoneManager()
    for lz_id in range(len(lz_manager.zones)):
        assert lz_manager.reserve(lz_id, 100 + lz_id, 0.0).approved
    left_admitted, left_records = left.step(0.0, {}, airspace, energy, priority)
    right_admitted, right_records = right.step(0.0, {}, DynamicAirspace(config), EnergyModel(config), PriorityEngine(config), lz_manager=lz_manager)
    assert len(left_admitted) == len(right_admitted) == 1
    assert left_records == right_records
    assert not left.queue.items and not right.queue.items
