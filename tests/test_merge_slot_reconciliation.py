import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeProposal, MergeScheduler
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import FlightPhase, MissionType, UAVState
from terminal_marl.envs.parallel_env import TerminalParallelEnv


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
