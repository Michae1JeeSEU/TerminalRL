from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeProposal, MergeScheduler, RepairCandidate
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import FlightPhase, MissionType, UAVState
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def _holder(uid: int = 1) -> UAVState:
    return UAVState(
        uid, uid, 0.0, MissionType.STANDARD, 1, 0, 10_000.0,
        FlightPhase.RING_OUTER, np.asarray((600.0, 0.0, 100.0)),
        np.asarray((0.0, 22.0, 0.0)), 22.0, 0.0, 0.0, 10_000.0,
        ring_role=RingRole.OUTER, ring_progress=0.0,
    )


def _proposal(target=RingRole.MIDDLE, merge_time=30.0, headway=9.0):
    return MergeProposal(target, merge_time, 1.25, headway)


def test_early_plan_endpoint_reconciles_without_changing_geometry_fields():
    scheduler = MergeScheduler(SimConfig())
    proposal = _proposal()
    reconciled = scheduler.reconcile_proposal(proposal, 24.5)
    assert reconciled.merge_time == pytest.approx(24.5)
    assert reconciled.target == proposal.target
    assert reconciled.merge_angle == proposal.merge_angle
    assert reconciled.headway == proposal.headway


def test_reconcile_checks_executed_and_outstanding_future_slots():
    config = SimConfig()
    scheduler = MergeScheduler(config)
    uav = _holder()
    scheduler.last_executed_release["o2m"] = 20.0
    with pytest.raises(ValueError, match="merge_slot_retime_conflict"):
        scheduler.reconcile_proposal(_proposal(merge_time=40.0), 25.0)

    scheduler = MergeScheduler(config)
    first = scheduler.reserve(uav, _proposal(merge_time=30.0))
    second = scheduler.reserve(uav, _proposal(merge_time=50.0))
    assert first.merge_time == 30.0 and second.merge_time == 50.0
    with pytest.raises(ValueError, match="merge_slot_retime_conflict"):
        scheduler.retime_reservation(second.reservation_id, 35.0)
    assert scheduler.reservations[second.reservation_id].merge_time == 50.0


def test_future_reservation_retime_is_atomic_and_matches_plan_endpoint():
    env = TerminalParallelEnv(SimConfig())
    uav = _holder()
    proposal = _proposal(merge_time=40.0)
    reservation = env.scheduler.reserve(uav, proposal, now=env.now)
    plan = TrajectoryPlan(
        "future-plan", uav.uid, "ring_transfer",
        (TrajectoryPoint(0.0, np.asarray((600.0, 0.0, 100.0))),
         TrajectoryPoint(31.0, np.asarray((400.0, 0.0, 70.0)))),
        0.0, 31.0, (reservation.reservation_id,),
        metadata={"target_phase": "ring_middle"},
    )
    assert env._retime_reservation_to_plan(reservation, plan)
    assert reservation.merge_time == pytest.approx(plan.end_time)
    assert abs(reservation.merge_time - plan.end_time) <= 1e-6
    assert plan.metadata["requested_merge_time"] == pytest.approx(40.0)
    assert plan.metadata["merge_time"] == pytest.approx(plan.end_time)


def test_future_reservation_retime_conflict_leaves_original_slot_unchanged():
    env = TerminalParallelEnv(SimConfig())
    uav = _holder()
    first = env.scheduler.reserve(uav, _proposal(merge_time=30.0), now=env.now)
    second = env.scheduler.reserve(uav, _proposal(merge_time=50.0), now=env.now)
    plan = TrajectoryPlan(
        "conflicting-future-plan", uav.uid, "ring_transfer",
        (TrajectoryPoint(0.0, np.asarray((600.0, 0.0, 100.0))),
         TrajectoryPoint(35.0, np.asarray((400.0, 0.0, 70.0)))),
        0.0, 35.0, (second.reservation_id,),
        metadata={"target_phase": "ring_middle"},
    )
    assert not env._retime_reservation_to_plan(second, plan)
    assert second.merge_time == pytest.approx(50.0)
    assert first.merge_time == pytest.approx(30.0)


def test_final_reservation_uses_departure_event_not_touchdown_endpoint():
    env = TerminalParallelEnv(SimConfig())
    uav = _holder()
    proposal = MergeProposal("final", 40.0, None, 12.0)
    reservation = env.scheduler.reserve(uav, proposal, now=env.now)
    plan = TrajectoryPlan(
        "final-plan", uav.uid, "final_approach",
        (TrajectoryPoint(0.0, np.asarray((200.0, 0.0, 40.0))),
         TrajectoryPoint(20.0, np.asarray((0.0, 0.0, 0.0)))),
        0.0, 20.0, (reservation.reservation_id,),
        metadata={"target_phase": "landing", "scheduled_departure": 6.0},
    )
    assert env._retime_reservation_to_plan(reservation, plan)
    assert reservation.merge_time == pytest.approx(6.0)
    assert reservation.merge_time != pytest.approx(plan.end_time)


def test_final_reservation_rejects_departure_outside_executable_plan():
    env = TerminalParallelEnv(SimConfig())
    uav = _holder()
    reservation = env.scheduler.reserve(
        uav, MergeProposal("final", 40.0, None, 12.0), now=env.now,
    )
    plan = TrajectoryPlan(
        "invalid-final-plan", uav.uid, "final_approach",
        (TrajectoryPoint(0.0, np.asarray((200.0, 0.0, 40.0))),
         TrajectoryPoint(20.0, np.asarray((0.0, 0.0, 0.0)))),
        0.0, 20.0, (reservation.reservation_id,),
        metadata={"target_phase": "landing", "scheduled_departure": 25.0},
    )
    assert not env._retime_reservation_to_plan(reservation, plan)
    assert reservation.merge_time == pytest.approx(40.0)


def test_legacy_release_retimes_shared_execution_scheduler_reservation():
    config = SimConfig(
        action_schema_version="low_intent_v3_factorized_discrete",
        experiment_mode="fixed_low_rl",
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
        max_simulation_seconds=20.0,
    )
    env = TerminalParallelEnv(config)
    env.reset(75)
    uav = next(iter(env.active.values()))
    assert env._release_legacy(uav, LowLevelIntent(1.0, 0.0, 0.0)) == "approved"
    assert env.execution_scheduler.reservations is env.scheduler.reservations
    reservation = env.scheduler.reservations[uav.merge_reservation_id]
    assert reservation.merge_time == pytest.approx(uav.committed_plan.end_time)


def _late_ring_transfer_plan(uav: UAVState, *, end_time: float = 35.0) -> TrajectoryPlan:
    return TrajectoryPlan(
        "late-ring-transfer",
        uav.uid,
        "ring_transfer",
        (
            TrajectoryPoint(
                0.0,
                np.asarray((600.0, 0.0, 100.0)),
                np.zeros(3),
                np.zeros(3),
            ),
            TrajectoryPoint(
                end_time,
                np.asarray((400.0, 0.0, 70.0)),
                np.zeros(3),
                np.zeros(3),
            ),
        ),
        0.0,
        end_time,
        metadata={
            "target_phase": "ring_middle",
            "dynamic_unreachable_merge_time": True,
        },
    )


def _late_candidate_env(*, reconcile: bool) -> tuple[TerminalParallelEnv, UAVState, RepairCandidate, WorldSnapshot]:
    config = SimConfig(reconcile_late_merge_slots_enabled=reconcile)
    env = TerminalParallelEnv(config)
    uav = _holder()
    env.active[uav.uid] = uav
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    candidate = RepairCandidate("requested", LowLevelIntent(1.0, 0.0, 0.0))
    env.scheduler.propose = lambda *args, **kwargs: _proposal(merge_time=30.0)
    # The initial requested slot is deliberately accepted; the canonical
    # actual slot is checked by reconcile_proposal below.
    env.scheduler.can_reserve = lambda proposal: True
    env.generator.ring_transfer = lambda *args, **kwargs: _late_ring_transfer_plan(uav)
    env._candidate_energy_guard_reason = lambda *args, **kwargs: None
    return env, uav, candidate, snapshot


def test_default_late_ring_transfer_remains_rejected_by_safety(monkeypatch):
    env, uav, candidate, snapshot = _late_candidate_env(reconcile=False)
    proposal, plan, reason = env._candidate_plan(
        uav, candidate, RingRole.MIDDLE, snapshot, None,
    )
    assert proposal is not None and plan is not None
    assert reason == "dynamics_unreachable_merge_time"
    assert plan.metadata["dynamic_unreachable_merge_time"] is True


def test_opt_in_late_slot_reconciliation_runs_energy_and_safety_on_actual_slot(monkeypatch):
    env, uav, candidate, snapshot = _late_candidate_env(reconcile=True)
    observed: list[tuple[str, dict]] = []

    def energy_guard(_uav, plan, _target):
        observed.append(("energy", dict(plan.metadata)))
        assert "dynamic_unreachable_merge_time" not in plan.metadata
        return None

    def safety_filter(plan, _committed, _snapshot):
        observed.append(("safety", dict(plan.metadata)))
        assert "dynamic_unreachable_merge_time" not in plan.metadata
        return SimpleNamespace(approved=True, reason=None)

    monkeypatch.setattr(env, "_candidate_energy_guard_reason", energy_guard)
    monkeypatch.setattr(env.safety, "filter_plan", safety_filter)
    proposal, plan, reason = env._candidate_plan(
        uav, candidate, RingRole.MIDDLE, snapshot, None,
    )
    assert reason is None
    assert proposal is not None and proposal.merge_time == pytest.approx(35.0)
    assert plan is not None
    assert plan.metadata["requested_merge_time"] == pytest.approx(30.0)
    assert plan.metadata["merge_time"] == pytest.approx(35.0)
    assert plan.metadata["requested_merge_late_seconds"] == pytest.approx(5.0)
    assert "dynamic_unreachable_merge_time" not in plan.metadata
    assert [name for name, _ in observed] == ["energy", "safety"]


def test_opt_in_late_slot_conflict_keeps_unsafe_marker_and_scheduler_state(monkeypatch):
    env, uav, candidate, snapshot = _late_candidate_env(reconcile=True)
    existing = env.scheduler.reserve(_holder(uid=2), _proposal(merge_time=35.0), now=env.now)
    before_reservations = {key: vars(value).copy() for key, value in env.scheduler.reservations.items()}
    before_last_release = dict(env.scheduler.last_release)
    assert env.telemetry["repair_candidates_checked"] == 0
    monkeypatch.setattr(
        env,
        "_candidate_energy_guard_reason",
        lambda *args, **kwargs: pytest.fail("energy guard must not run after slot conflict"),
    )
    monkeypatch.setattr(
        env.safety,
        "filter_plan",
        lambda *args, **kwargs: pytest.fail("safety filter must not run after slot conflict"),
    )
    proposal, plan, reason = env._candidate_plan(
        uav, candidate, RingRole.MIDDLE, snapshot, None,
    )
    assert reason == "merge_slot_retime_conflict"
    assert proposal is not None and proposal.merge_time == pytest.approx(30.0)
    assert plan is not None and plan.metadata["dynamic_unreachable_merge_time"] is True
    assert {key: vars(value).copy() for key, value in env.scheduler.reservations.items()} == before_reservations
    assert env.scheduler.last_release == before_last_release
    assert env.scheduler.reservations[existing.reservation_id].merge_time == pytest.approx(35.0)
    assert env.telemetry["repair_candidates_checked"] == 1


def test_opt_in_safety_rejection_is_not_overridden_after_reconciliation(monkeypatch):
    env, uav, candidate, snapshot = _late_candidate_env(reconcile=True)
    monkeypatch.setattr(
        env.safety,
        "filter_plan",
        lambda *args, **kwargs: SimpleNamespace(approved=False, reason="predicted_separation"),
    )
    proposal, plan, reason = env._candidate_plan(
        uav, candidate, RingRole.MIDDLE, snapshot, None,
    )
    assert reason == "predicted_separation"
    assert proposal is not None and proposal.merge_time == pytest.approx(35.0)
    assert plan is not None
    assert plan.metadata["requested_merge_time"] == pytest.approx(30.0)
    assert plan.metadata["merge_time"] == pytest.approx(35.0)
    assert plan.metadata["requested_merge_late_seconds"] == pytest.approx(5.0)
    assert "dynamic_unreachable_merge_time" not in plan.metadata


def test_opt_in_energy_rejection_is_not_overridden_after_reconciliation(monkeypatch):
    env, uav, candidate, snapshot = _late_candidate_env(reconcile=True)
    monkeypatch.setattr(
        env,
        "_candidate_energy_guard_reason",
        lambda *args, **kwargs: "emergency_energy_guard",
    )
    monkeypatch.setattr(
        env.safety,
        "filter_plan",
        lambda *args, **kwargs: pytest.fail("safety filter must not run after energy rejection"),
    )
    proposal, plan, reason = env._candidate_plan(
        uav, candidate, RingRole.MIDDLE, snapshot, None,
    )
    assert reason == "emergency_energy_guard"
    assert proposal is not None and proposal.merge_time == pytest.approx(35.0)
    assert plan is not None
    assert plan.metadata["requested_merge_time"] == pytest.approx(30.0)
    assert plan.metadata["merge_time"] == pytest.approx(35.0)
    assert plan.metadata["requested_merge_late_seconds"] == pytest.approx(5.0)
    assert "dynamic_unreachable_merge_time" not in plan.metadata
