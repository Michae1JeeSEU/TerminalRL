"""End-to-end environment checks for energy-aware admission slot handling."""

from __future__ import annotations

from terminal_marl.core.config import SimConfig
from terminal_marl.core.energy import EnergyModel
from terminal_marl.core.landing_zone import LandingZoneManager
from terminal_marl.core.uav import (
    ArrivalRequest,
    MissionType,
    QueuedArrivalState,
    TerminalOutcome,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _config(**overrides) -> SimConfig:
    values = {
        "action_schema_version": "low_action_schema_v4_2_hierarchical",
        "observation_schema_version": "phaseA_low_obs_v2_1_retry_state",
        "experiment_mode": "fixed_low_student",
        "teacher_enabled": False,
        "scheduler_mode": "execution",
        "runtime_proposal_shortlist_k": 3,
        "energy_model_mode": "phase_motion_v2",
        "emergency_landing_enabled": True,
        "emergency_priority_guard_enabled": True,
        "energy_reachability_guard_enabled": True,
        "arrival_model": "finite_cohort",
        "total_requested_uavs": 0,
        "arrival_rate_per_second": 0.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _request(config: SimConfig, soc: float, arrival_id: int = 1) -> ArrivalRequest:
    return ArrivalRequest(
        arrival_id,
        0.0,
        0,
        MissionType.STANDARD,
        1,
        config.energy_capacity,
        soc,
    )


def _enqueue(env: TerminalParallelEnv, request: ArrivalRequest) -> None:
    env.ledger.request(request.arrival_id)
    env.admission.enqueue(request)


def _late_known_lz_release(manager: LandingZoneManager, release_at: float) -> None:
    for zone in manager.zones:
        assert manager.reserve(zone.lz_id, 100 + zone.lz_id, 0.0).approved
        assert manager.note_committed_landing(zone.lz_id, 100 + zone.lz_id, release_at)


def test_late_known_lz_slots_divert_marginal_request_as_low_soc() -> None:
    config = _config()
    env = TerminalParallelEnv(config)
    energy = EnergyModel(config)
    queued = QueuedArrivalState(_request(config, 0.5), config.energy_capacity * 0.5)
    free_slot_requirement = (
        energy.admission_required_energy(queued)
        + config.emergency_energy_margin_seconds * config.power_reference
    )
    marginal_soc = (free_slot_requirement + 2.0) / config.energy_capacity
    request = _request(config, marginal_soc)
    _enqueue(env, request)
    nominal_solo_time = config.phase_motion_nominal_solo_energy / config.power_reference
    _late_known_lz_release(env.lz_manager, nominal_solo_time + 100.0)

    env.step({})

    assert env.active == {}
    assert env.ledger.outcomes[request.arrival_id] is TerminalOutcome.DIVERTED_LOW_SOC
    row = next(item for item in env.energy_outcomes if item["arrival_id"] == request.arrival_id)
    assert row["outcome"] == TerminalOutcome.DIVERTED_LOW_SOC.value
    assert row["reason"] == "low_soc_slot"
    assert row["uid"] is None
    assert env.admission.queue.items == []


def test_energy_guard_admits_normal_request_when_compatible_pads_are_free() -> None:
    config = _config()
    env = TerminalParallelEnv(config)
    request = _request(config, 0.50)
    _enqueue(env, request)

    env.step({})

    assert any(uav.arrival_id == request.arrival_id for uav in env.active.values())
    assert request.arrival_id not in env.ledger.outcomes
    assert not any(item["reason"] == "low_soc_slot" for item in env.energy_outcomes)


def test_flag_off_preserves_admission_when_known_lz_slots_are_late() -> None:
    config = _config(energy_reachability_guard_enabled=False)
    env = TerminalParallelEnv(config)
    request = _request(config, 0.50)
    _enqueue(env, request)
    _late_known_lz_release(env.lz_manager, 2000.0)

    env.step({})

    assert any(uav.arrival_id == request.arrival_id for uav in env.active.values())
    assert request.arrival_id not in env.ledger.outcomes
    assert not any(item["reason"] == "low_soc_slot" for item in env.energy_outcomes)
