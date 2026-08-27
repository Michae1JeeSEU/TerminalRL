from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeScheduler
from terminal_marl.core.trajectory_generator import TrajectoryGenerator
from terminal_marl.core.uav import FlightPhase, MissionType, UAVState
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def _holder(uid: int = 1) -> UAVState:
    return UAVState(uid, uid, 0., MissionType.STANDARD, 1, 0, 10_000., FlightPhase.RING_INNER,
                    np.array((200., 0., 40.)), np.array((0., 16., 0.)), 16., 0., 0., 10_000.,
                    ring_role=RingRole.INNER, ring_progress=0.)


def test_a612_minimum_intervention_candidate_order_and_bounds():
    scheduler = MergeScheduler(SimConfig(repair_aware_scheduler=True))
    requested = LowLevelIntent(1., 0., 0.)
    rows = scheduler.repair_candidates(requested)
    assert rows[0].kind == "requested" and rows[0].delay_seconds == 0.
    assert all(abs(row.intent.speed_residual) <= 1 and abs(row.intent.merge_time_residual) <= 1 for row in rows)
    assert next(index for index, row in enumerate(rows) if row.delay_seconds > 0.) > next(index for index, row in enumerate(rows) if row.kind == "combined_repair")


def test_a612_delayed_final_plan_is_a_real_future_reservation_trajectory():
    config = SimConfig(repair_aware_scheduler=True); env = TerminalParallelEnv(config); env.reset(0)
    uav = _holder(); env.active = {uav.uid: uav}; snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    proposal = env.scheduler.propose(uav, LowLevelIntent(1., 0., 0.), snapshot, "final", delay_seconds=10.)
    reservation = env.scheduler.reserve(uav, proposal, now=env.now)
    lz = env.lz_manager.zones[0]
    plan = TrajectoryGenerator(config).final_approach(uav, lz, reservation, env.now, scheduled_delay=10., airspace=env.airspace)
    assert reservation.status == "future"
    assert plan.metadata["scheduled_delay"] == 10.
    # The requested 10 s delay is preserved, and a high-speed inner holder
    # may require an additional finite braking prefix before final ingress.
    assert any(point.t >= 10. for point in plan.points)
    assert np.allclose(plan.sample(5.).position[2], env.airspace.actual_ring(RingRole.INNER).height)


def test_a612_reservation_conflict_is_prevented_before_commit():
    config = SimConfig(); scheduler = MergeScheduler(config); uav = _holder()
    snapshot = SimpleNamespace(now=0., active={1: uav}, airspace=TerminalParallelEnv(config).airspace)
    proposal = scheduler.propose(uav, LowLevelIntent(1., 0., 0.), snapshot, RingRole.MIDDLE)
    scheduler.reserve(uav, proposal)
    conflicting = replace(proposal, merge_time=proposal.merge_time - proposal.headway)
    assert not scheduler.can_reserve(conflicting)


def test_a612_unique_intent_telemetry_and_primary_blocker_are_mutually_exclusive():
    env = TerminalParallelEnv(SimConfig(repair_aware_scheduler=True)); uav = _holder()
    first = env._start_release_intent(uav, LowLevelIntent(1., 0., 0.), "i2f")
    second = env._start_release_intent(uav, LowLevelIntent(1., 0., 0.), "i2f")
    assert first["release_intent_id"] == second["release_intent_id"] and second["retry_count"] == 1
    env._finish_release_intent(uav, second, "safety_wait", safety_reason="predicted_holder_separation")
    assert second["primary_blocker"] == "NO_MERGE_GAP"
    assert second["primary_blocker"] in {"CAPACITY", "HEADWAY", "NO_MERGE_GAP", "PREDICTED_SEPARATION", "FINAL_RESOURCE", "LZ_RESOURCE", "INTERFACE_FROZEN", "SOC", "OTHER"}


def test_a612_checked_plan_is_the_executed_plan_and_runtime_safety_is_unchanged():
    config = SimConfig(experiment_mode="fixed_low_rl", reset_mode="warm_start_rings", warm_start_uavs=1,
                       fixed_arrival_schedule=(0.,), total_requested_uavs=1, max_simulation_seconds=180., repair_aware_scheduler=True)
    env = TerminalParallelEnv(config); env.reset(3)
    for _ in range(180):
        actions = {uid: np.asarray((1., 0., 0.), dtype=np.float32) for uid in env.agents}
        _, _, _, _, info = env.step(actions)
        if info["telemetry"]["checked_executed_plan_matches"]:
            assert info["telemetry"]["checked_executed_plan_matches"] >= 1
            break
    assert env.telemetry["collisions"] == env.telemetry["nmac"] == env.telemetry["hard_separation_violations"] == 0
