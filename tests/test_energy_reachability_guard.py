"""Focused tests for the opt-in route-aware candidate energy guard."""

from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeReservation
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    UAVState,
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
        "emergency_trigger_soc": 0.40,
        "arrival_model": "finite_cohort",
        "total_requested_uavs": 0,
        "arrival_rate_per_second": 0.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _ring_uav(env: TerminalParallelEnv, soc: float, uid: int = 1) -> UAVState:
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
    uav.phase = FlightPhase.RING_OUTER
    uav.ring_role = RingRole.OUTER
    uav.ring_progress = 0.0
    ring = env.airspace.actual_ring(RingRole.OUTER)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    uav.acceleration = np.asarray(
        (-(ring.nominal_speed**2) / ring.radius, 0.0, 0.0),
        dtype=float,
    )
    return uav


def _transfer_plan(env: TerminalParallelEnv, uav: UAVState):
    reservation = MergeReservation(
        "guard-candidate",
        uav.uid,
        "ring_outer",
        "middle",
        160.0,
        0.0,
        None,
        None,
    )
    return env.generator.ring_transfer(
        uav,
        RingRole.OUTER,
        RingRole.MIDDLE,
        reservation,
        env.airspace,
        env.now,
    )


def test_route_guard_validation_and_default_off() -> None:
    assert SimConfig().energy_reachability_guard_enabled is False
    with pytest.raises(ValueError, match="energy_reachability_guard_enabled requires"):
        SimConfig(energy_reachability_guard_enabled=True)
    with pytest.raises(ValueError, match="energy_reachability_guard_enabled requires"):
        SimConfig(
            energy_reachability_guard_enabled=True,
            emergency_priority_guard_enabled=True,
            emergency_landing_enabled=True,
            energy_model_mode="phase_motion_v2",
        )
    assert _config().energy_reachability_guard_enabled is True


def test_route_aware_guard_allows_feasible_point_four_three_and_records_budget() -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, 0.43)
    plan = _transfer_plan(env, uav)

    old_plan_energy = env.energy.estimate_plan_energy(plan, env.now, plan.end_time)
    old_projected_soc = (uav.energy_remaining - old_plan_energy) / uav.energy_capacity
    old_reserve_soc = (
        env.config.min_landing_soc * uav.energy_capacity
        + env.config.emergency_energy_margin_seconds * env.config.power_reference
    ) / uav.energy_capacity

    reason = env._candidate_energy_guard_reason(uav, plan, RingRole.MIDDLE)

    assert old_projected_soc > old_reserve_soc
    assert reason is None
    diagnostic = env.energy_reachability_diagnostics[uav.uid]
    assert diagnostic["candidate_energy"] > 0.0
    assert diagnostic["required_energy"] <= uav.energy_remaining
    assert diagnostic["slack_energy"] >= 0.0
    assert diagnostic["selected_lz_id"] is not None
    assert env.telemetry["energy_reachability_checks"] == 1
    assert env.telemetry["energy_reachability_negative_budgets"] == 0


def test_route_guard_requests_first_feasible_budget_shortcut(monkeypatch) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, 0.43)
    plan = _transfer_plan(env, uav)
    observed = {}

    import terminal_marl.envs.parallel_env as parallel_env_module

    original = parallel_env_module.compute_candidate_energy_budget

    def wrapped(*args, **kwargs):
        observed["stop_at_first_feasible"] = kwargs.get("stop_at_first_feasible")
        return original(*args, **kwargs)

    monkeypatch.setattr(parallel_env_module, "compute_candidate_energy_budget", wrapped)
    assert env._candidate_energy_guard_reason(uav, plan, RingRole.MIDDLE) is None
    assert observed["stop_at_first_feasible"] is True


def test_route_aware_guard_rejects_negative_budget_without_resource_side_effects() -> None:
    env = TerminalParallelEnv(_config())
    # Valid negative transfer budget with an immediately feasible emergency witness.
    uav = _ring_uav(env, env.config.min_landing_soc + 0.15)
    plan = _transfer_plan(env, uav)
    lz_before = tuple(
        (zone.status, zone.reserved_by, zone.expected_release_at)
        for zone in env.lz_manager.zones
    )

    reason = env._candidate_energy_guard_reason(uav, plan, RingRole.MIDDLE)

    assert reason == "emergency_energy_guard"
    diagnostic = env.energy_reachability_diagnostics[uav.uid]
    assert diagnostic["slack_energy"] < 0.0
    assert diagnostic["selected_lz_id"] is not None
    assert env.telemetry["energy_reachability_checks"] == 1
    assert env.telemetry["energy_reachability_negative_budgets"] == 1
    assert uav.committed_plan is None
    assert not env.scheduler.reservations
    assert tuple(
        (zone.status, zone.reserved_by, zone.expected_release_at)
        for zone in env.lz_manager.zones
    ) == lz_before


def test_known_reserved_eta_changes_recorded_slack() -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, 0.50)
    plan = _transfer_plan(env, uav)
    for lz_id in range(1, len(env.lz_manager.zones)):
        assert env.lz_manager.reserve(lz_id, 99, env.now).approved

    assert env._candidate_energy_guard_reason(uav, plan, RingRole.MIDDLE) is None
    free_diagnostic = dict(env.energy_reachability_diagnostics[uav.uid])

    assert env.lz_manager.reserve(0, 98, env.now).approved
    assert env.lz_manager.note_committed_landing(0, 98, plan.end_time + 20.0)
    assert env._candidate_energy_guard_reason(uav, plan, RingRole.MIDDLE) is None
    delayed_diagnostic = env.energy_reachability_diagnostics[uav.uid]

    assert free_diagnostic["selected_lz_id"] == 0
    assert delayed_diagnostic["selected_lz_id"] == 0
    assert free_diagnostic["pad_wait_seconds"] == pytest.approx(0.0)
    assert delayed_diagnostic["pad_wait_seconds"] == pytest.approx(20.0)
    assert delayed_diagnostic["slack_energy"] < free_diagnostic["slack_energy"]
    assert env.telemetry["energy_reachability_checks"] == 2
    assert env.telemetry["energy_reachability_negative_budgets"] == 0


def test_legacy_guard_formula_is_unchanged_when_new_flag_is_off() -> None:
    env = TerminalParallelEnv(_config(energy_reachability_guard_enabled=False))
    uav = _ring_uav(env, 0.37)
    plan = _transfer_plan(env, uav)

    assert env._candidate_energy_guard_reason(uav, plan, RingRole.MIDDLE) is None
    assert env.energy_reachability_diagnostics == {}
    assert env.telemetry["energy_reachability_checks"] == 0


def _patch_route_blocked_candidate(monkeypatch, env: TerminalParallelEnv, plan):
    calls: list[int] = []

    def fake_candidate_plan(uav, _candidate, target, _snapshot, _lz_id):
        calls.append(uav.uid)
        proposal = env.scheduler.propose(
            uav,
            _candidate.intent,
            _snapshot,
            target,
            delay_seconds=_candidate.delay_seconds,
        )
        return proposal, plan, env._candidate_energy_guard_reason(uav, plan, target)

    monkeypatch.setattr(env, "_candidate_plan", fake_candidate_plan)
    return calls


def test_negative_transfer_budget_redirects_to_certified_emergency_route(monkeypatch) -> None:
    env = TerminalParallelEnv(_config())
    # Valid negative transfer budget with an immediately feasible emergency witness.
    uav = _ring_uav(env, env.config.min_landing_soc + 0.15)
    env.active[uav.uid] = uav
    plan = _transfer_plan(env, uav)
    calls = _patch_route_blocked_candidate(monkeypatch, env, plan)
    intent = LowLevelIntent(1.0, 0.0, 0.0)

    decision = env._runtime_shortlist_release(uav, intent, (intent,))

    assert decision == "emergency_redirected"
    assert calls == [uav.uid]
    assert uav.committed_plan is not None
    assert uav.committed_plan.mode == "emergency_landing"
    assert uav.emergency_landing is True
    assert uav.assigned_lz is not None
    assert env.lz_manager.zones[uav.assigned_lz].status.value == "reserved"
    assert env.lz_manager.zones[uav.assigned_lz].expected_release_at is not None
    assert not env.scheduler.reservations
    assert env.telemetry["emergency_attempts"] == 1
    assert env.telemetry["emergency_committed"] == 1
    assert env.telemetry["emergency_energy_guard_rejections"] == 0
    assert env.last_release_telemetry[uav.uid]["decision_history"][-1] == "emergency_redirected"
    assert uav.uid not in env._active_release_intents


def test_failed_forced_redirect_sets_normal_cooldown_and_skips_next_candidate(monkeypatch) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, 0.37)
    env.active[uav.uid] = uav
    plan = _transfer_plan(env, uav)
    for lz_id in range(len(env.lz_manager.zones)):
        assert env.lz_manager.reserve(lz_id, 100 + lz_id, env.now).approved
    calls = _patch_route_blocked_candidate(monkeypatch, env, plan)
    intent = LowLevelIntent(1.0, 0.0, 0.0)

    first = env._runtime_shortlist_release(uav, intent, (intent,))
    assert first == "emergency_energy_guard"
    assert calls == [uav.uid]
    assert uav.committed_plan is None
    assert uav.safety_retry_after == pytest.approx(5.0)
    assert uav.emergency_retry_after == pytest.approx(5.0)
    assert not env.scheduler.reservations
    assert all(zone.status.value == "reserved" for zone in env.lz_manager.zones)
    assert env.telemetry["emergency_blocked_lz"] == 1

    env.now = 1.0
    second = env._runtime_shortlist_release(uav, intent, (intent,))
    assert second == "safety_wait"
    assert calls == [uav.uid]
    assert env.telemetry["energy_reachability_checks"] == 1
