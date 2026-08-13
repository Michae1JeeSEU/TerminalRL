"""A6.8 pre-commit timeline regression tests (no PPO)."""
import math
import numpy as np

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.reconfiguration_safety import ReconfigurationSafetyCoordinator
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import ArrivalRequest, FlightPhase, MissionType, UAVState
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.policies import DynamicRulePolicy
from terminal_marl.training.a68_reconfiguration_safety_audit import HISTORICAL_T720_486_735, t720_geometry_conflict_regression


def _uav(uid, phase):
    request = ArrivalRequest(uid, 0., 0, MissionType.STANDARD, 1, 10_000., 1.)
    return UAVState(uid, uid, 0., MissionType.STANDARD, 1, 0, 10_000., phase,
                    np.zeros(3), np.zeros(3), 0., 0., 0., 10_000.)


def _env():
    env = TerminalParallelEnv(SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True))
    env.reset(0)
    return env


def test_a68_holder_prediction_uses_actual_transition_geometry_and_atomic_commit():
    env = _env(); ring = env.airspace.actual_ring(RingRole.OUTER)
    holder = _uav(1, FlightPhase.RING_OUTER); holder.ring_role = RingRole.OUTER; holder.ring_progress = 0.
    holder.position = np.asarray((ring.radius, 0., ring.height)); env.active[1] = holder
    coordinator = ReconfigurationSafetyCoordinator(env)
    action = np.asarray((.60, 0, 0, 0, 0, 0, 0, 0, 0, 0), np.float32)
    update, verdict = coordinator._candidate_safety(action, {})
    assert verdict.approved and coordinator._candidate_airspace is not None
    tracks = coordinator._tracks(coordinator._candidate_airspace, {}, coordinator._protected_horizon(coordinator._candidate_airspace))
    # Runtime moves the first interval on R(t=0)=600; by t=2 the second
    # interval has used the interpolated actual geometry, R(t=1)=600.6.
    assert math.isclose(float(np.linalg.norm(tracks[1].sample(1.).position[:2])), 600., abs_tol=1e-5)
    assert math.isclose(float(np.linalg.norm(tracks[1].sample(2.).position[:2])), 600.6, abs_tol=1e-5)
    coordinator._atomic_commit(action, update)
    assert env.airspace.transition is not None
    assert math.isclose(env.airspace.target_ring(RingRole.OUTER).radius, 612., abs_tol=1e-5)


def test_a68_committed_transfer_gets_target_ring_post_handoff_continuation():
    env = _env(); outer = env.airspace.actual_ring(RingRole.OUTER); middle = env.airspace.actual_ring(RingRole.MIDDLE)
    transfer = _uav(2, FlightPhase.TRANSFER_O2M); transfer.ring_role = None
    transfer.position = np.asarray((outer.radius, 0., outer.height))
    transfer.committed_plan = TrajectoryPlan("o2m", 2, "ring_transfer", [
        TrajectoryPoint(0., transfer.position.copy()), TrajectoryPoint(1., np.asarray((middle.radius, 0., middle.height))),
    ], 0., 1., metadata={"target_phase": "ring_middle"})
    env.active[2] = transfer; coordinator = ReconfigurationSafetyCoordinator(env)
    tracks = coordinator._tracks(env.airspace, {}, 31.)
    assert math.isclose(float(np.linalg.norm(tracks[2].sample(2.).position[:2])), middle.radius, abs_tol=1e-5)
    assert tracks[2].metadata["active_end"] == 31.


def test_a68_missing_active_trajectory_fails_closed_without_commit():
    env = _env(); env.active[3] = _uav(3, FlightPhase.INBOUND)
    coordinator = ReconfigurationSafetyCoordinator(env)
    update, verdict = coordinator._candidate_safety(np.asarray((.60, 0, 0, 0, 0, 0, 0, 0, 0, 0), np.float32), {})
    assert update.structural_applied and not verdict.approved
    assert coordinator._candidate_airspace is None and env.airspace.transition is None


def test_a68_swept_verifier_records_authoritative_minimum():
    env = _env(); safety = SafetyFilter(env.config)
    tracks = {
        1: TrajectoryPlan("a", 1, "test", [TrajectoryPoint(0., np.zeros(3)), TrajectoryPoint(1., np.zeros(3))], 0., 1., metadata={}),
        2: TrajectoryPlan("b", 2, "test", [TrajectoryPoint(0., np.asarray((10., 0., 0.))), TrajectoryPoint(1., np.asarray((10., 0., 0.)))], 0., 1., metadata={}),
    }
    verdict = safety.verify_predicted_tracks(tracks, start=0., end=1.)
    assert not verdict.approved
    assert safety.last_predicted_track_audit["minimum_separation"] == 10.
    assert safety.last_predicted_track_audit["checked_pairs"] >= 1


def test_a68_historical_t720_486_735_geometry_fixture_is_hard_safe():
    """The pre-A6.8 TRANSFER-holder failure stays an executable fixture."""
    # The historical encounter is at t=985, so a 300 s branch from t=720
    # still crosses it while keeping this executable pytest fixture bounded.
    result = t720_geometry_conflict_regression(horizon=300.)
    assert result["fixture"] == HISTORICAL_T720_486_735
    current = result["current"]
    assert current["safe"]
    assert current["safety_delta"]["hard_separation_violations"] == 0
    assert current["first_conflict"]["pair"] is None


def test_a68_dynamic_rule_requests_outer_capacity_threshold_under_real_upstream_pressure():
    action = DynamicRulePolicy().act(np.zeros(29, dtype=np.float32), {
        "external_queue": 8, "occupancy_ratio": [.20, .05, 0.],
        "final_pressure": 0., "lz_utilization": 0., "transition_active": False,
    })
    assert np.isclose(action[0], .60)
