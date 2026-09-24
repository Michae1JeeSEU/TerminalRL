"""Contracts for the opt-in finite-candidate landing-flow coordinator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.flow_trajectories import FlowTrajectoryPreview
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.merge_scheduler import LowLevelIntent
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


def _inner_plan(
    env: TerminalParallelEnv,
    *,
    uid: int,
    end_time: float,
    angle: float = 0.0,
    target_phase: str = "ring_inner",
    plan_id: str | None = None,
) -> TrajectoryPlan:
    ring = env.airspace.actual_ring(RingRole.INNER)
    source = env.airspace.actual_ring(RingRole.MIDDLE)
    endpoint = np.asarray(
        (ring.radius * np.cos(angle), ring.radius * np.sin(angle), ring.height),
        dtype=float,
    )
    return TrajectoryPlan(
        plan_id or f"v5-inner-{uid}-{end_time}-{angle:.6f}",
        uid,
        "ring_transfer",
        (
            TrajectoryPoint(
                0.0,
                np.asarray((source.radius, 0.0, source.height), dtype=float),
            ),
            TrajectoryPoint(float(end_time), endpoint),
        ),
        0.0,
        float(end_time),
        metadata={"target_phase": target_phase},
    )


def _set_inner_holder_angle(env: TerminalParallelEnv, uav: UAVState, angle: float) -> None:
    ring = env.airspace.actual_ring(RingRole.INNER)
    uav.ring_progress = float(angle)
    uav.position = np.asarray(
        (ring.radius * np.cos(angle), ring.radius * np.sin(angle), ring.height),
        dtype=float,
    )
    uav.velocity = np.asarray(
        (-ring.nominal_speed * np.sin(angle), ring.nominal_speed * np.cos(angle), 0.0),
        dtype=float,
    )
    uav.acceleration = np.asarray(
        (
            -(ring.nominal_speed**2) * np.cos(angle) / ring.radius,
            -(ring.nominal_speed**2) * np.sin(angle) / ring.radius,
            0.0,
        ),
        dtype=float,
    )


def _first_pass_spacing_rejections():
    calls = {"count": 0}

    def guard(_uav, _plan):
        calls["count"] += 1
        return "inner_departure_headway" if calls["count"] <= 12 else None

    return calls, guard


def test_flow_flag_is_strictly_opt_in_and_validated() -> None:
    assert TerminalParallelEnv(_config(rule_flow_coordination_enabled=False)).flow_coordinator is None
    with pytest.raises(ValueError, match="rule_flow_coordination_enabled"):
        SimConfig(rule_flow_coordination_enabled=True)
    with pytest.raises(ValueError, match="rule_flow_coordination_enabled"):
        _config(experiment_mode="dynamic_rule", high_level_enabled=True)


def test_wait_does_not_start_a_flow_intent_and_active_plan_is_reported() -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.OUTER, soc=0.90)
    wait = LowLevelIntent(0.0, 0.0, 0.0)

    assert env.flow_coordinator.release(uav, wait) == "not_requested"
    assert not env.release_intents
    uav.committed_plan = object()
    assert env.flow_coordinator.release(uav, LowLevelIntent(1.0, 0.0, 0.0)) == "plan_active"


def test_flow_order_uses_downstream_rank_before_energy_slack() -> None:
    env = TerminalParallelEnv(_config())
    outer = _ring_uav(env, RingRole.OUTER, soc=0.50, uid=0)
    inner = _ring_uav(env, RingRole.INNER, soc=0.80, uid=1)

    ordered = [uid for uid, _ in env.flow_coordinator.processing_order()]
    assert ordered[:2] == [inner.uid, outer.uid]


def test_flow_order_uses_energy_slack_within_downstream_ring() -> None:
    env = TerminalParallelEnv(_config())
    high_slack = _ring_uav(env, RingRole.INNER, soc=0.80, uid=0)
    low_slack = _ring_uav(env, RingRole.INNER, soc=0.50, uid=1)

    ordered = [uid for uid, _ in env.flow_coordinator.processing_order()]
    assert ordered[:2] == [low_slack.uid, high_slack.uid]


def test_flow_order_uses_mission_priority_then_uid_within_equal_slack(
) -> None:
    env = TerminalParallelEnv(_config())
    first = _ring_uav(env, RingRole.INNER, soc=0.80, uid=0)
    second = _ring_uav(env, RingRole.INNER, soc=0.80, uid=1)
    first.base_priority = 1
    second.base_priority = 2

    ordered = [uid for uid, _ in env.flow_coordinator.processing_order()]
    assert ordered[:2] == [second.uid, first.uid]

    second.base_priority = first.base_priority
    ordered = [uid for uid, _ in env.flow_coordinator.processing_order()]
    assert ordered[:2] == [first.uid, second.uid]


@pytest.mark.parametrize(
    ("source", "target", "durations"),
    (
        (RingRole.OUTER, RingRole.MIDDLE, (48.0, 60.0, 72.0, 84.0)),
        (RingRole.MIDDLE, RingRole.INNER, (60.0, 72.0, 84.0, 96.0)),
    ),
)
def test_ring_flow_duration_groups_have_four_dynamics_valid_candidates(
    source: RingRole,
    target: RingRole,
    durations: tuple[float, ...],
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, source, soc=0.90)

    assert env.flow_coordinator.RING_DURATIONS[target] == durations
    for duration in durations:
        plan = env.flow_coordinator.trajectory.ring_transfer(
            uav, target, env.airspace, env.now, duration
        )
        assert plan.end_time == pytest.approx(duration)
        assert env.generator.dynamics.validate_plan(plan.points) == []


def test_energy_fallback_ignores_only_known_geometry_failures() -> None:
    flow = TerminalParallelEnv(_config()).flow_coordinator

    assert flow._energy_guard_only(
        ["trajectory_infeasible", "emergency_energy_guard"]
    )
    assert not flow._energy_guard_only(["trajectory_infeasible"])
    assert not flow._energy_guard_only(
        ["trajectory_infeasible", "headway_conflict", "emergency_energy_guard"]
    )


def test_normal_rejection_deadline_expires_after_two_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.OUTER, soc=0.90)
    monkeypatch.setattr(
        env.flow_coordinator,
        "_candidate_plan",
        lambda *_args, **_kwargs: (None, "predicted_separation"),
    )
    intent = LowLevelIntent(1.0, 0.0, 0.0)

    assert env.flow_coordinator.release(uav, intent) == "safety_reject"
    assert uav.safety_retry_after == pytest.approx(2.0)
    env.now = 1.0
    assert env.flow_coordinator.release(uav, intent) == "safety_wait"
    assert uav.safety_retry_after == pytest.approx(2.0)
    env.now = 2.0
    assert env.flow_coordinator.release(uav, intent) == "safety_reject"
    assert uav.safety_retry_after == pytest.approx(4.0)


def test_emergency_slack_gate_rejects_nonurgent_takeover(monkeypatch: pytest.MonkeyPatch) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.INNER, soc=0.34)
    monkeypatch.setattr(
        type(env.energy),
        "emergency_required_energy",
        lambda _self, *_args, **_kwargs: 300.0,
    )
    monkeypatch.setattr(
        env.safety,
        "filter_plan",
        lambda plan, *_args: type("Result", (), {"approved": True})(),
    )

    assert not env.flow_coordinator.try_emergency(uav)
    assert env.telemetry["emergency_blocked_slack"] > 0
    assert uav.committed_plan is None
    assert uav.assigned_lz is None


def test_coordinated_emergency_uses_route_slack_above_soc_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config(emergency_trigger_soc=0.40))
    uav = _ring_uav(env, RingRole.INNER, soc=0.45)
    monkeypatch.setattr(
        type(env.energy),
        "emergency_required_energy",
        lambda _self, *_args, **_kwargs: 300.0,
    )
    monkeypatch.setattr(
        env.safety,
        "filter_plan",
        lambda plan, *_args: type("Result", (), {"approved": True})(),
    )

    assert not env.flow_coordinator.try_emergency(uav)
    assert env.telemetry["emergency_attempts"] > 0
    assert env.telemetry["emergency_blocked_slack"] > 0
    assert uav.committed_plan is None
    assert uav.assigned_lz is None


def test_coordinated_step_checks_emergency_above_soc_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config(emergency_trigger_soc=0.40))
    uav = _ring_uav(env, RingRole.INNER, soc=0.45)
    attempted: list[int] = []

    def record_attempt(candidate: UAVState) -> bool:
        attempted.append(candidate.uid)
        return False

    monkeypatch.setattr(env.flow_coordinator, "try_emergency", record_attempt)
    env.step({str(uav.uid): np.asarray((0.0, 0.0, 0.0), dtype=np.float32)})

    assert attempted == [uav.uid]


def test_legacy_emergency_path_keeps_soc_trigger_gate() -> None:
    env = TerminalParallelEnv(
        _config(rule_flow_coordination_enabled=False, emergency_trigger_soc=0.40)
    )
    uav = _ring_uav(env, RingRole.INNER, soc=0.45)

    assert env.flow_coordinator is None
    assert not env._try_emergency_landing(uav)
    assert env.telemetry["emergency_attempts"] == 0
    assert uav.committed_plan is None
    assert uav.assigned_lz is None


def test_emergency_redirect_closes_existing_release_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.INNER, soc=0.34)
    intent = LowLevelIntent(1.0, 0.0, 0.0)
    row = env._start_release_intent(uav, intent, "i2f")
    row.update(
        {
            "flow_coordination": True,
            "flow_candidate_rejection_reasons": ["emergency_energy_guard"],
            "flow_candidates_checked": 1,
            "flow_candidate_durations": [],
        }
    )
    env.last_release_telemetry[uav.uid] = dict(row)
    monkeypatch.setattr(
        type(env.energy),
        "emergency_required_energy",
        lambda _self, *_args, **_kwargs: uav.energy_remaining - 1.0,
    )
    monkeypatch.setattr(
        env,
        "_commit",
        lambda _uav, plan: True,
    )

    assert env.flow_coordinator.try_emergency(uav)
    assert uav.emergency_landing
    assert uav.phase is FlightPhase.EMERGENCY_LANDING
    assert uav.uid not in env._active_release_intents
    assert env.last_release_decisions[uav.uid] == "emergency_redirected"
    assert env.last_release_telemetry[uav.uid]["decision_history"][-1] == "emergency_redirected"
    assert env.last_release_telemetry[uav.uid]["execution_selection_source"] == "flow_emergency"
    assert env.last_release_telemetry[uav.uid]["flow_emergency_wait_budget_seconds"] == pytest.approx(
        2.0 * np.pi * 200.0 / 16.0
    )
    assert env.last_release_telemetry[uav.uid]["flow_emergency_slack_energy"] == pytest.approx(1.0)
    assert env.last_release_telemetry[uav.uid]["flow_emergency_slack_seconds"] == pytest.approx(1.0)


def test_emergency_candidate_rollback_releases_pad(monkeypatch: pytest.MonkeyPatch) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.INNER, soc=0.34)
    monkeypatch.setattr(
        type(env.energy),
        "emergency_required_energy",
        lambda _self, *_args, **_kwargs: uav.energy_remaining - 1.0,
    )
    monkeypatch.setattr(env, "_commit", lambda _uav, _plan: False)

    assert not env.flow_coordinator.try_emergency(uav)
    assert all(zone.reserved_by is None for zone in env.lz_manager.zones)
    assert env.telemetry["emergency_blocked_safety"] > 0


def test_headway_rejection_short_circuits_flow_trajectory_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.OUTER, soc=0.90)
    monkeypatch.setattr(env.scheduler, "can_reserve_at", lambda *_args, **_kwargs: False)

    def fail_if_generated(*_args, **_kwargs):
        raise AssertionError("headway rejection must precede trajectory generation")

    monkeypatch.setattr(env.flow_coordinator.trajectory, "ring_transfer", fail_if_generated)
    plan, reason = env.flow_coordinator._candidate_plan(
        uav, RingRole.MIDDLE, None, 32.0
    )
    assert plan is None
    assert reason == "headway_conflict"


def test_certified_normal_inner_release_is_not_taken_over_by_emergency() -> None:
    env = TerminalParallelEnv(_config(emergency_trigger_soc=0.40))
    uav = _ring_uav(env, RingRole.INNER, soc=0.40)

    env.step({str(uav.uid): np.asarray((1.0, 0.0, 0.0), dtype=np.float32)})

    assert uav.committed_plan is not None
    assert uav.phase is FlightPhase.FINAL
    assert uav.emergency_landing is False
    assert env.telemetry["emergency_committed"] == 0


def test_flow_does_not_apply_phantom_emergency_pad_lock_to_inner_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config(emergency_trigger_soc=0.40))
    _ring_uav(env, RingRole.OUTER, soc=0.34, uid=0)
    inner = _ring_uav(env, RingRole.INNER, soc=0.80, uid=1)
    for lz_id in range(1, 6):
        assert env.lz_manager.reserve(lz_id, 100 + lz_id, env.now).approved

    assert env._emergency_lz_priority_blocks(inner) is True
    checked: list[tuple[str, int | None, float]] = []
    monkeypatch.setattr(
        env.flow_coordinator,
        "_candidate_plan",
        lambda _uav, target, lz_id, duration: (
            checked.append((str(target), lz_id, duration))
            or (None, "predicted_separation")
        ),
    )

    decision = env.flow_coordinator.release(
        inner, LowLevelIntent(1.0, 0.0, 0.0)
    )

    assert checked
    assert all(lz_id == 0 for _, lz_id, _ in checked)
    assert decision == "safety_reject"
    assert env.last_release_telemetry[inner.uid]["flow_candidates_checked"] == len(checked)


def test_normal_commit_rejection_rolls_back_scheduler_and_lz_reservations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.INNER, soc=0.80)
    plan = env.flow_coordinator.trajectory.final_approach(
        uav,
        env.lz_manager.zones[0],
        env.airspace,
        env.now,
        48.0,
        emergency=False,
    )
    monkeypatch.setattr(
        env.flow_coordinator,
        "_candidate_plan",
        lambda *_args, **_kwargs: (plan, None),
    )
    monkeypatch.setattr(env, "_commit", lambda *_args, **_kwargs: False)

    assert env.flow_coordinator.release(
        uav, LowLevelIntent(1.0, 0.0, 0.0)
    ) == "safety_reject"
    assert not env.scheduler.reservations
    assert all(zone.reserved_by is None for zone in env.lz_manager.zones)


def test_normal_commit_exception_rolls_back_then_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.INNER, soc=0.80)
    plan = env.flow_coordinator.trajectory.final_approach(
        uav,
        env.lz_manager.zones[0],
        env.airspace,
        env.now,
        48.0,
        emergency=False,
    )
    monkeypatch.setattr(
        env.flow_coordinator,
        "_candidate_plan",
        lambda *_args, **_kwargs: (plan, None),
    )

    def raise_invariant(*_args, **_kwargs):
        raise RuntimeError("checked plan/hash mismatch")

    monkeypatch.setattr(env, "_commit", raise_invariant)
    with pytest.raises(RuntimeError, match="checked plan/hash mismatch"):
        env.flow_coordinator.release(uav, LowLevelIntent(1.0, 0.0, 0.0))
    assert not env.scheduler.reservations
    assert all(zone.reserved_by is None for zone in env.lz_manager.zones)


def test_emergency_commit_exception_releases_pad_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.INNER, soc=0.34)
    monkeypatch.setattr(
        type(env.energy),
        "emergency_required_energy",
        lambda _self, *_args, **_kwargs: uav.energy_remaining - 1.0,
    )

    def raise_invariant(*_args, **_kwargs):
        raise RuntimeError("checked plan/hash mismatch")

    monkeypatch.setattr(env, "_commit", raise_invariant)
    with pytest.raises(RuntimeError, match="checked plan/hash mismatch"):
        env.flow_coordinator.try_emergency(uav)
    assert not env.scheduler.reservations
    assert all(zone.reserved_by is None for zone in env.lz_manager.zones)


def test_inner_departure_capacity_is_derived_from_i2f_headway() -> None:
    env = TerminalParallelEnv(_config())
    flow = env.flow_coordinator
    angle = env.airspace.headways["i2f"] * 16.0 / 200.0

    assert flow._inner_departure_gap_angle() == pytest.approx(angle)
    assert flow._inner_departure_capacity_limit() == 6
    assert 2.0 * 200.0 * np.sin(angle / 2.0) == pytest.approx(184.71, abs=0.01)


def test_inner_departure_capacity_counts_occupied_and_committed_incoming() -> None:
    env = TerminalParallelEnv(_config())
    for uid in range(5):
        _ring_uav(env, RingRole.INNER, soc=0.90, uid=uid)
    incoming = _ring_uav(env, RingRole.MIDDLE, soc=0.90, uid=5)
    incoming.committed_plan = _inner_plan(env, uid=5, end_time=20.0, plan_id="incoming")
    requester = _ring_uav(env, RingRole.MIDDLE, soc=0.90, uid=6)

    occupied, limit = env.flow_coordinator._inner_departure_occupancy()
    assert (occupied, limit) == (6, 6)
    assert env.flow_coordinator.release(
        requester, LowLevelIntent(1.0, 0.0, 0.0)
    ) == "capacity"
    row = env.last_release_telemetry[requester.uid]
    assert row["flow_candidate_rejection_reasons"] == ["inner_departure_capacity"]


@pytest.mark.parametrize(
    ("chord", "expected"),
    ((120.0, "inner_departure_headway"), (200.0, None)),
)
def test_inner_departure_headway_uses_projected_endpoint_spacing(
    chord: float, expected: str | None
) -> None:
    env = TerminalParallelEnv(_config())
    requester = _ring_uav(env, RingRole.MIDDLE, soc=0.90, uid=0)
    candidate = _inner_plan(env, uid=requester.uid, end_time=10.0)
    required_gap = 2.0 * np.arcsin(
        chord / (2.0 * env.airspace.actual_ring(RingRole.INNER).radius)
    )
    holder = _ring_uav(env, RingRole.INNER, soc=0.90, uid=1)
    ring = env.airspace.actual_ring(RingRole.INNER)
    _set_inner_holder_angle(
        env,
        holder,
        float(required_gap) - ring.nominal_speed * candidate.end_time / ring.radius,
    )

    assert env.flow_coordinator._inner_departure_headway_reason(requester, candidate) == expected


def test_inner_departure_headway_checks_a_later_committed_incoming() -> None:
    env = TerminalParallelEnv(_config())
    requester = _ring_uav(env, RingRole.MIDDLE, soc=0.90, uid=0)
    candidate = _inner_plan(env, uid=requester.uid, end_time=10.0)
    ring = env.airspace.actual_ring(RingRole.INNER)
    required_gap = 2.0 * np.arcsin(120.0 / (2.0 * ring.radius))
    snapshot = env.flow_coordinator._snapshot()
    candidate_later_position = env.safety._post_plan_positions(
        candidate, snapshot, np.asarray([20.0], dtype=float)
    )[0]
    candidate_at_later_time = float(
        np.arctan2(candidate_later_position[1], candidate_later_position[0])
    )
    incoming = _ring_uav(env, RingRole.MIDDLE, soc=0.90, uid=1)
    incoming.committed_plan = _inner_plan(
        env,
        uid=incoming.uid,
        end_time=20.0,
        angle=float(candidate_at_later_time + required_gap),
        plan_id="later-incoming",
    )

    assert env.flow_coordinator._inner_departure_headway_reason(requester, candidate) == (
        "inner_departure_headway"
    )


def test_committed_final_departure_does_not_consume_inner_departure_capacity() -> None:
    env = TerminalParallelEnv(_config())
    for uid in range(5):
        _ring_uav(env, RingRole.INNER, soc=0.90, uid=uid)
    departing = _ring_uav(env, RingRole.INNER, soc=0.90, uid=5)
    departing.phase = FlightPhase.FINAL
    departing.ring_role = None
    departing.committed_plan = _inner_plan(
        env,
        uid=departing.uid,
        end_time=20.0,
        target_phase="landing",
        plan_id="departing-final",
    )

    assert env.flow_coordinator._inner_departure_occupancy() == (5, 6)


def test_inner_bias_candidates_are_duration_outer_then_bias_inner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.MIDDLE, soc=0.90)
    attempted: list[tuple[float, float]] = []

    def reject_candidate(
        _uav,
        _target,
        _lz_id,
        duration,
        tangential_speed_bias_mps: float = 0.0,
    ):
        attempted.append((float(duration), float(tangential_speed_bias_mps)))
        return None, "predicted_separation"

    monkeypatch.setattr(env.flow_coordinator, "_candidate_plan", reject_candidate)
    assert env.flow_coordinator.release(
        uav, LowLevelIntent(1.0, 0.0, 0.0)
    ) == "safety_reject"

    expected = [
        (duration, bias)
        for duration in env.flow_coordinator.RING_DURATIONS[RingRole.INNER]
        for bias in env.flow_coordinator.INNER_TANGENTIAL_SPEED_BIASES_MPS
    ]
    assert attempted == expected
    rows = env.last_release_telemetry[uav.uid]["flow_candidate_durations"]
    assert [
        (row["duration"], row["tangential_speed_bias_mps"]) for row in rows
    ] == expected


def test_biased_inner_candidate_commits_only_after_normal_certification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.MIDDLE, soc=0.95)
    safety_biases: list[float] = []

    def filter_plan(plan, *_args):
        bias = float(plan.metadata.get("flow_tangential_speed_bias_mps", 0.0))
        safety_biases.append(bias)
        if bias == 0.0:
            return SimpleNamespace(
                approved=False,
                reason="synthetic_safety_reject",
                checked_plan_id=None,
                checked_trajectory_hash=None,
            )
        return SimpleNamespace(
            approved=True,
            reason=None,
            checked_plan_id=plan.plan_id,
            checked_trajectory_hash=plan.trajectory_hash,
        )

    monkeypatch.setattr(env.safety, "filter_plan", filter_plan)
    assert env.flow_coordinator.release(
        uav, LowLevelIntent(1.0, 0.0, 0.0)
    ) == "approved"
    assert uav.committed_plan is not None
    assert uav.committed_plan.metadata["flow_tangential_speed_bias_mps"] == pytest.approx(-4.0)
    assert safety_biases[:2] == [0.0, -4.0]
    assert safety_biases.count(-4.0) == 2  # candidate filter and _commit hash check
    row = env.last_release_telemetry[uav.uid]
    assert row["flow_selected_tangential_speed_bias_mps"] == pytest.approx(-4.0)
    assert row["flow_candidate_durations"][-1]["reason"] == "approved"
    assert env.telemetry["checked_executed_plan_matches"] == 1


def test_tangential_bias_repairs_are_scoped_to_m2i(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.OUTER, soc=0.90)
    seen: list[float] = []
    original = env.flow_coordinator.trajectory.ring_transfer

    def record_bias(*args, **kwargs):
        seen.append(float(kwargs.get("tangential_speed_bias_mps", 0.0)))
        return original(*args, **kwargs)

    monkeypatch.setattr(env.flow_coordinator.trajectory, "ring_transfer", record_bias)
    env.flow_coordinator._candidate_plan(uav, RingRole.MIDDLE, None, 48.0)
    assert seen == [0.0]


def test_emergency_middle_ring_wait_budget_allows_v5_witness_slack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.MIDDLE, soc=0.3775, uid=3)
    witness_slack = 60.28
    monkeypatch.setattr(
        type(env.energy),
        "emergency_required_energy",
        lambda _self, state, _plan: state.energy_remaining - witness_slack,
    )

    required, slack, wait_seconds, limit = env.flow_coordinator._emergency_slack_budget(
        uav, object()
    )
    assert required == pytest.approx(uav.energy_remaining - witness_slack)
    assert slack == pytest.approx(witness_slack)
    assert wait_seconds == pytest.approx(2.0 * np.pi * 400.0 / 20.0)
    assert wait_seconds < 126.0
    assert limit == pytest.approx(wait_seconds * env.energy.phase_power(uav.phase))
    assert env.flow_coordinator._emergency_slack_allowed(uav, object())


def test_emergency_inner_ring_slack_above_one_circuit_is_not_urgent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config(emergency_trigger_soc=0.40))
    uav = _ring_uav(env, RingRole.INNER, soc=0.39)
    slack = 79.0
    monkeypatch.setattr(
        type(env.energy),
        "emergency_required_energy",
        lambda _self, state, _plan: state.energy_remaining - slack,
    )

    assert not env.flow_coordinator.try_emergency(uav)
    assert uav.committed_plan is None
    assert env.telemetry["emergency_blocked_slack"] > 0
    assert env.flow_coordinator._emergency_slack_budget(uav, object())[3] < slack


def test_emergency_energy_below_route_requirement_is_always_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.MIDDLE, soc=0.3775)
    monkeypatch.setattr(
        type(env.energy),
        "emergency_required_energy",
        lambda _self, state, _plan: state.energy_remaining + 1.0,
    )

    assert not env.flow_coordinator._emergency_slack_allowed(uav, object())
    assert env.telemetry["emergency_blocked_energy"] > 0


def test_inner_departure_headway_rejection_precedes_energy_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    requester = _ring_uav(env, RingRole.MIDDLE, soc=0.90, uid=0)
    candidate = _inner_plan(env, uid=requester.uid, end_time=10.0)
    holder = _ring_uav(env, RingRole.INNER, soc=0.90, uid=1)
    ring = env.airspace.actual_ring(RingRole.INNER)
    gap = 2.0 * np.arcsin(120.0 / (2.0 * ring.radius))
    _set_inner_holder_angle(
        env,
        holder,
        float(gap) - ring.nominal_speed * candidate.end_time / ring.radius,
    )
    monkeypatch.setattr(
        env.flow_coordinator.trajectory,
        "_ring_transfer_context",
        lambda *_args, **_kwargs: (
            float(candidate.end_time),
            0.0,
            RingRole.MIDDLE,
            RingRole.INNER,
            env.airspace.actual_ring(RingRole.INNER),
            env.now,
        ),
    )
    monkeypatch.setattr(
        env.flow_coordinator.trajectory,
        "_prepare_ring_transfer",
        lambda *_args, **_kwargs: object(),
    )
    preview = FlowTrajectoryPreview(
        candidate.end_time,
        dict(candidate.metadata),
        (candidate.points[-1],),
    )
    monkeypatch.setattr(
        env.flow_coordinator.trajectory,
        "_ring_transfer_preview",
        lambda *_args, **_kwargs: preview,
    )
    monkeypatch.setattr(
        env.flow_coordinator.trajectory,
        "_materialize_ring_transfer",
        lambda *_args, **_kwargs: pytest.fail("headway rejection must not materialize"),
    )
    energy_called = False

    def fail_energy(*_args, **_kwargs):
        nonlocal energy_called
        energy_called = True
        raise AssertionError("queue gap must precede energy budget")

    monkeypatch.setattr(env, "_candidate_energy_guard_reason", fail_energy)
    plan, reason = env.flow_coordinator._candidate_plan(
        requester, RingRole.INNER, None, candidate.end_time
    )
    assert plan is None
    assert reason == "inner_departure_headway"
    assert energy_called is False


def test_inner_deadline_fallback_commits_generated_safe_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.MIDDLE, soc=0.42)
    calls, guard = _first_pass_spacing_rejections()
    monkeypatch.setattr(env.flow_coordinator, "_inner_departure_headway_reason", guard)

    decision = env.flow_coordinator.release(uav, LowLevelIntent(1.0, 0.0, 0.0))

    assert decision == "approved"
    assert calls["count"] == 13
    assert uav.committed_plan is not None
    assert uav.phase is FlightPhase.TRANSFER_M2I
    row = env.last_release_telemetry[uav.uid]
    assert row["flow_deadline_fallback"] is True
    assert row["flow_selected_duration"] == pytest.approx(60.0)
    assert row["flow_selected_tangential_speed_bias_mps"] == pytest.approx(0.0)
    assert env.telemetry["checked_executed_plan_matches"] == 1


def test_inner_deadline_fallback_keeps_high_slack_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.MIDDLE, soc=0.90)
    monkeypatch.setattr(
        env.flow_coordinator,
        "_inner_departure_headway_reason",
        lambda *_args, **_kwargs: "inner_departure_headway",
    )

    decision = env.flow_coordinator.release(uav, LowLevelIntent(1.0, 0.0, 0.0))

    assert decision == "safety_reject"
    assert uav.committed_plan is None
    row = env.last_release_telemetry[uav.uid]
    assert row["flow_deadline_fallback"] is False
    assert row["flow_candidate_rejection_reasons"]
    assert set(row["flow_candidate_rejection_reasons"]) == {
        "inner_departure_headway"
    }
    assert env.flow_coordinator._urgent_spacing_energy_limit(uav) == pytest.approx(
        2.0 * np.pi * 400.0 / 20.0
    )


def test_inner_deadline_fallback_requires_fresh_energy_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.MIDDLE, soc=0.42)
    monkeypatch.setattr(
        env.flow_coordinator,
        "_inner_departure_headway_reason",
        lambda *_args, **_kwargs: "inner_departure_headway",
    )
    monkeypatch.setattr(
        env,
        "_candidate_energy_guard_reason",
        lambda *_args, **_kwargs: None,
    )

    assert env.flow_coordinator.release(
        uav, LowLevelIntent(1.0, 0.0, 0.0)
    ) == "safety_reject"
    assert uav.committed_plan is None
    assert env.last_release_telemetry[uav.uid]["flow_deadline_fallback"] is False


def test_inner_deadline_fallback_rejects_negative_energy_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.MIDDLE, soc=0.42)
    monkeypatch.setattr(
        env.flow_coordinator,
        "_inner_departure_headway_reason",
        lambda *_args, **_kwargs: "inner_departure_headway",
    )

    def negative_budget(*_args, **_kwargs):
        env.energy_reachability_diagnostics[uav.uid] = {
            "slack_energy": -1.0,
            "selected_lz_id": 0,
        }
        return "emergency_energy_guard"

    monkeypatch.setattr(env, "_candidate_energy_guard_reason", negative_budget)

    assert env.flow_coordinator.release(
        uav, LowLevelIntent(1.0, 0.0, 0.0)
    ) == "safety_reject"
    assert uav.committed_plan is None
    assert env.last_release_telemetry[uav.uid]["flow_deadline_fallback"] is False


def test_inner_deadline_fallback_rechecks_capacity_before_commit() -> None:
    env = TerminalParallelEnv(_config())
    for uid in range(1, 7):
        _ring_uav(env, RingRole.INNER, soc=0.90, uid=uid)
    requester = _ring_uav(env, RingRole.MIDDLE, soc=0.42, uid=0)

    assert env.flow_coordinator.release(
        requester, LowLevelIntent(1.0, 0.0, 0.0)
    ) == "capacity"
    assert requester.committed_plan is None
    assert env.last_release_telemetry[requester.uid]["flow_deadline_fallback"] is False


def test_inner_deadline_fallback_does_not_bypass_scheduler_precheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.MIDDLE, soc=0.42)
    monkeypatch.setattr(
        env.scheduler,
        "can_reserve_at",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        env.flow_coordinator,
        "_inner_departure_headway_reason",
        lambda *_args, **_kwargs: "inner_departure_headway",
    )

    assert env.flow_coordinator.release(
        uav, LowLevelIntent(1.0, 0.0, 0.0)
    ) == "safety_reject"
    assert uav.committed_plan is None
    assert env.last_release_telemetry[uav.uid]["flow_deadline_fallback"] is False


def test_inner_physics_prescreen_rejection_precedes_headway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config())
    requester = _ring_uav(env, RingRole.MIDDLE, soc=0.90, uid=0)
    monkeypatch.setattr(
        env.flow_coordinator.trajectory.dynamics,
        "samples_are_feasible",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        env.flow_coordinator,
        "_inner_departure_headway_reason",
        lambda *_args, **_kwargs: pytest.fail(
            "physical prescreen rejection must precede headway"
        ),
    )
    monkeypatch.setattr(
        env,
        "_candidate_energy_guard_reason",
        lambda *_args, **_kwargs: pytest.fail(
            "physical prescreen rejection must precede energy"
        ),
    )

    plan, reason = env.flow_coordinator._candidate_plan(
        requester, RingRole.INNER, None, 60.0,
    )

    assert plan is None
    assert reason == "trajectory_infeasible"


def test_normal_final_release_tries_90_seconds_after_shorter_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TerminalParallelEnv(_config(emergency_trigger_soc=0.40))
    uav = _ring_uav(env, RingRole.INNER, soc=0.80)
    long_plan = env.flow_coordinator.trajectory.final_approach(
        uav,
        env.lz_manager.zones[0],
        env.airspace,
        env.now,
        90.0,
        emergency=False,
    )
    attempted: list[tuple[float, int | None]] = []

    def reject_shorter(_uav, _target, lz_id, duration):
        attempted.append((float(duration), lz_id))
        if float(duration) < 90.0:
            return None, "synthetic_reject"
        return long_plan, None

    monkeypatch.setattr(env.flow_coordinator, "_available_lz_ids", lambda _uav: (0,))
    monkeypatch.setattr(env.flow_coordinator, "_candidate_plan", reject_shorter)
    assert env.flow_coordinator.release(
        uav, LowLevelIntent(1.0, 0.0, 0.0)
    ) == "approved"
    assert [duration for duration, _lz_id in attempted] == [
        20.0,
        28.0,
        36.0,
        48.0,
        60.0,
        90.0,
    ]
    assert uav.phase is FlightPhase.FINAL
    assert uav.emergency_landing is False
    row = env.last_release_telemetry[uav.uid]
    assert row["flow_selected_duration"] == pytest.approx(90.0)
    assert row["flow_candidate_durations"][-1]["reason"] == "approved"
