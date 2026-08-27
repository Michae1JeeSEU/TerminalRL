import math
import numpy as np

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeScheduler
from terminal_marl.core.trajectory_generator import TrajectoryGenerator
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import ArrivalRequest, FlightPhase, MissionType, UAVState
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from terminal_marl.training.a67_controllability_audit import _classify_reason, _interface_snapshot


def _uav(uid: int, phase: FlightPhase) -> UAVState:
    request = ArrivalRequest(uid, 0., 0, MissionType.STANDARD, 1, 10_000., 1.)
    return UAVState(uid, uid, 0., MissionType.STANDARD, 1, 0, 10_000., phase, np.zeros(3), np.zeros(3), 0., 0., 0., 10_000.)


def test_a67_consumed_and_expired_reservations_do_not_accumulate_or_poison_headway():
    env = TerminalParallelEnv(SimConfig()); env.reset(seed=0)
    uav = _uav(1, FlightPhase.RING_OUTER); env.active[1] = uav
    scheduler = MergeScheduler(env.config); snapshot = WorldSnapshot(0., env.active, env.airspace)
    reservation = scheduler.reserve(uav, scheduler.propose(uav, LowLevelIntent(1., 0., 0.), snapshot, RingRole.MIDDLE))
    assert len(scheduler.reservations) == 1
    scheduler.consume(reservation.reservation_id)
    assert not scheduler.reservations
    assert scheduler.last_executed_release["o2m"] == reservation.merge_time
    orphan = scheduler.reserve(uav, scheduler.propose(uav, LowLevelIntent(1., 0., 1.), snapshot, RingRole.MIDDLE))
    assert scheduler.expire(orphan.expires_at + 1., set()) == [orphan.reservation_id]
    assert not scheduler.reservations


def test_a67_interface_trace_counts_only_its_source_phase_and_reasons():
    env = TerminalParallelEnv(SimConfig()); env.reset(seed=0)
    env.active[1] = _uav(1, FlightPhase.RING_OUTER)
    row = _interface_snapshot(env, "o2m")
    assert row["eligible"] == 1
    _classify_reason(row, "geometry_frozen")
    _classify_reason(row, "safety_reject")
    _classify_reason(row, "final_capacity")
    assert row["interface_frozen"] == row["safety_rejected"] == row["resource_blocked"] == 1


def test_a67_empty_target_ring_merges_through_source_gap_not_fixed_zero_angle():
    env = TerminalParallelEnv(SimConfig()); env.reset(seed=0)
    uav = _uav(1, FlightPhase.RING_OUTER); uav.ring_role = RingRole.OUTER; uav.ring_progress = 1.25
    env.active[1] = uav
    blocker = _uav(2, FlightPhase.RING_OUTER); blocker.ring_role = RingRole.OUTER; blocker.ring_progress = 0.
    env.active[2] = blocker
    proposal = MergeScheduler(env.config).propose(uav, LowLevelIntent(1., 0., 0.), WorldSnapshot(0., env.active, env.airspace), RingRole.MIDDLE)
    assert not math.isclose(proposal.merge_angle, 0.)
    assert not math.isclose(proposal.merge_angle, 1.25)


def test_a67_transfer_wait_is_explicit_source_ring_arc_in_canonical_plan():
    env = TerminalParallelEnv(SimConfig()); env.reset(seed=0)
    ring = env.airspace.actual_ring(RingRole.OUTER)
    uav = _uav(1, FlightPhase.RING_OUTER); uav.ring_role = RingRole.OUTER; uav.ring_progress = .4
    uav.position = np.array((ring.radius * math.cos(.4), ring.radius * math.sin(.4), ring.height))
    env.active[1] = uav
    snapshot = WorldSnapshot(0., env.active, env.airspace)
    scheduler = MergeScheduler(env.config)
    proposal = scheduler.propose(uav, LowLevelIntent(1., 0., 0.), snapshot, RingRole.MIDDLE)
    reservation = scheduler.reserve(uav, proposal)
    plan = TrajectoryGenerator(env.config).ring_transfer(uav, RingRole.OUTER, RingRole.MIDDLE, reservation, env.airspace, 0.)
    departure = plan.metadata["scheduled_departure"]
    assert departure > 0.
    held = plan.sample(min(1., departure / 2.)).position
    assert math.isclose(float(np.linalg.norm(held[:2])), ring.radius, rel_tol=0., abs_tol=1e-3)
    assert not TrajectoryGenerator(env.config).validate_kinematics(plan)


def test_a67_inbound_handoff_is_checked_against_continuing_transfer():
    env = TerminalParallelEnv(SimConfig()); env.reset(seed=0)
    candidate = TrajectoryPlan("candidate", 1, "inbound", [
            TrajectoryPoint(0., np.asarray((600., 0., 100.))),
        TrajectoryPoint(1., np.asarray((600., 0., 100.))),
    ], 0., 1., metadata={"target_phase": "ring_outer"})
    # The committed transfer is clear during [0, 1], but intersects the
    # holder that candidate becomes after t=1.
    other = TrajectoryPlan("other", 2, "ring_transfer", [
        TrajectoryPoint(0., np.asarray((0., 300., 70.))),
        TrajectoryPoint(2., np.asarray((600., 0., 100.))),
    ], 0., 2., metadata={"target_phase": "ring_middle"})
    snapshot = WorldSnapshot(0., {}, env.airspace)
    verdict = SafetyFilter(env.config).filter_plan(candidate, {2: other}, snapshot)
    assert not verdict.approved
    assert verdict.reason.startswith("predicted_post_plan_holder")


def test_a67_candidate_checks_committed_plan_after_its_endpoint():
    env = TerminalParallelEnv(SimConfig()); env.reset(seed=0)
    candidate = TrajectoryPlan("candidate", 1, "ring_transfer", [
        TrajectoryPoint(0., np.asarray((0., 300., 350.))),
        TrajectoryPoint(3., np.asarray((600., 0., 500.))),
    ], 0., 3., metadata={"target_phase": "ring_middle"})
    other = TrajectoryPlan("other", 2, "ring_transfer", [
        TrajectoryPoint(0., np.asarray((600., 0., 500.))),
        TrajectoryPoint(1., np.asarray((400., 0., 350.))),
    ], 0., 1., metadata={"target_phase": "ring_outer"})
    verdict = SafetyFilter(env.config).filter_plan(candidate, {2: other}, WorldSnapshot(0., {}, env.airspace))
    assert not verdict.approved
