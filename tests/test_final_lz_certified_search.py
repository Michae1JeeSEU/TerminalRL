"""Contracts for scheduler-owned LZ search in continuous final approach mode."""

from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.landing_zone import ReservationResult
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    UAVState,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _dense_config(*, geometry_mode: str) -> SimConfig:
    return SimConfig(
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version="phaseA_low_obs_v2_3_source_bound_88d",
        neighbor_comm_range_m=320.0,
        experiment_mode="fixed_rule",
        high_level_enabled=False,
        reset_mode="warm_start_rings",
        warm_start_uavs=30,
        warm_start_steady_motion=True,
        warm_start_energy_reachability_check_enabled=True,
        total_requested_uavs=30,
        arrival_model="finite_cohort",
        arrival_rate_per_second=0.0,
        fixed_arrival_schedule=(0.0,) * 30,
        initial_soc_mode="uniform",
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        max_simulation_seconds=240.0,
        energy_model_mode="phase_motion_v2",
        emergency_landing_enabled=True,
        emergency_priority_guard_enabled=True,
        energy_reachability_guard_enabled=True,
        emergency_trigger_soc=0.40,
        emergency_energy_margin_seconds=15.0,
        lz_assignment_mode="nearest_available",
        release_arbitration_mode="downstream_priority",
        final_queue_capacity=3,
        safe_distance=80.0,
        nmac_distance=40.0,
        collision_distance=10.0,
        teacher_enabled=False,
        teacher_mode="teacher_full",
        teacher_release_label_enabled=False,
        scheduler_mode="execution",
        shield_enabled=True,
        runtime_proposal_shortlist_k=1,
        final_approach_geometry_mode=geometry_mode,
    )


def _single_inner_env(*, geometry_mode: str) -> tuple[TerminalParallelEnv, UAVState, int]:
    config = SimConfig(
        action_schema_version="low_action_schema_v4_2_hierarchical",
        experiment_mode="fixed_rule",
        total_requested_uavs=0,
        arrival_rate_per_second=0.0,
        final_approach_geometry_mode=geometry_mode,
        lz_assignment_mode="nearest_available",
        runtime_proposal_shortlist_k=3,
        max_simulation_seconds=120.0,
    )
    env = TerminalParallelEnv(config)
    request = ArrivalRequest(
        1, 0.0, 0, MissionType.STANDARD, 1, config.energy_capacity, 1.0
    )
    queued = QueuedArrivalState(request, request.energy_capacity)
    uav = UAVState.from_request(0, request, queued, np.zeros(3), env.now)
    uav.phase = FlightPhase.RING_INNER
    uav.ring_role = RingRole.INNER
    uav.ring_progress = 0.0
    ring = env.airspace.actual_ring(RingRole.INNER)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    uav.acceleration = np.asarray(
        (-(ring.nominal_speed**2) / ring.radius, 0.0, 0.0), dtype=float
    )
    env.active[uav.uid] = uav
    env.ledger.request(request.arrival_id)
    chosen = env.lz_manager.choose_rule_zone(uav, env.now)
    assert chosen is not None
    return env, uav, chosen


def test_default_brake_mode_keeps_one_rule_lz_and_legacy_telemetry() -> None:
    env, uav, rule_lz_id = _single_inner_env(geometry_mode="brake_bezier_v1")
    intent = LowLevelIntent(1.0, 0.0, 0.0)

    decision = env._runtime_shortlist_release(uav, intent, (intent,))

    assert decision == "approved"
    assert uav.phase is FlightPhase.FINAL
    assert uav.assigned_lz == rule_lz_id
    assert env.lz_manager.zones[rule_lz_id].reserved_by == uav.uid
    row = env.last_release_telemetry[uav.uid]
    assert "final_lz_pairs_checked" not in row
    assert "final_lz_selected_id" not in row


def test_continuous_dense30_step_finds_a_certified_normal_final_release() -> None:
    env = TerminalParallelEnv(_dense_config(geometry_mode="continuous_inward_v1"))
    observations, _ = env.reset(840626)
    action = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    actions = {uid: action.copy() for uid in observations}

    _, _, _, _, info = env.step(actions)

    final_uavs = [uav for uav in env.active.values() if uav.phase is FlightPhase.FINAL]
    assert final_uavs
    assert env.telemetry["emergency_committed"] == 0
    assert env.config.safe_distance == 80.0
    assert env.config.nmac_distance == 40.0
    assert env.config.collision_distance == 10.0
    assert info["release_intent_telemetry"]

    for uav in final_uavs:
        assert not uav.emergency_landing
        assert uav.assigned_lz is not None
        assert uav.committed_plan is not None
        assert uav.committed_plan.metadata["target_phase"] == "landing"
        assert uav.committed_plan.metadata["final_approach_geometry"] == "continuous_inward_v1"
        zone = env.lz_manager.zones[uav.assigned_lz]
        assert zone.reserved_by == uav.uid
        row = env.last_release_telemetry[uav.uid]
        assert row["final_lz_selected_id"] == uav.assigned_lz
        assert any(
            pair["status"] == "approved"
            and pair["lz_id"] == uav.assigned_lz
            for pair in row["final_lz_pairs_checked"]
        )


def test_continuous_lz_search_records_only_certified_selected_pairs() -> None:
    env = TerminalParallelEnv(_dense_config(geometry_mode="continuous_inward_v1"))
    observations, _ = env.reset(840626)
    action = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    env.step({uid: action.copy() for uid in observations})

    for uid, row in env.last_release_telemetry.items():
        selected = row.get("final_lz_selected_id")
        pairs = row.get("final_lz_pairs_checked")
        if selected is None:
            continue
        approved = [pair for pair in pairs if pair["status"] == "approved"]
        assert len(approved) == 1
        assert approved[0]["lz_id"] == selected
        uav = env.active[int(uid)]
        assert uav.committed_plan is not None
        assert env.last_safety_decisions[int(uid)] == "approved"


def test_default_lz_reservation_reject_stops_the_whole_k3_shortlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, uav, _ = _single_inner_env(geometry_mode="brake_bezier_v1")
    first = LowLevelIntent(1.0, 0.0, 0.0)
    second = LowLevelIntent(1.0, 0.0, 1.0)
    checked: list[LowLevelIntent] = []

    monkeypatch.setattr(
        env,
        "_candidate_plan",
        lambda _uav, candidate, _target, _snapshot, _lz_id: (
            checked.append(candidate.intent) or (object(), object(), None)
        ),
    )
    monkeypatch.setattr(
        env.lz_manager,
        "reserve",
        lambda _lz_id, _uid, _now: ReservationResult(False, None, "forced_reject"),
    )

    decision = env._runtime_shortlist_release(uav, first, (first, second))

    assert decision == "safety_reject"
    assert checked == [first]
    assert env.last_release_telemetry[uav.uid]["runtime_shortlist_candidates_checked"] == 1


def test_continuous_energy_guard_preserves_emergency_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, uav, _ = _single_inner_env(geometry_mode="continuous_inward_v1")
    env.config.energy_reachability_guard_enabled = True
    intent = LowLevelIntent(1.0, 0.0, 0.0)
    emergency_calls: list[bool] = []

    monkeypatch.setattr(
        env,
        "_candidate_plan",
        lambda *_args, **_kwargs: (None, None, "emergency_energy_guard"),
    )

    def emergency_redirect(_uav, *, force: bool = False):
        emergency_calls.append(force)
        return True

    monkeypatch.setattr(env, "_try_emergency_landing", emergency_redirect)
    decision = env._runtime_shortlist_release(uav, intent, (intent,))

    assert decision == "emergency_redirected"
    assert emergency_calls == [True]
    row = env.last_release_telemetry[uav.uid]
    assert len(row["final_lz_pairs_checked"]) == len(env.lz_manager.zones)
    assert row["decision_history"][-1] == "emergency_redirected"
