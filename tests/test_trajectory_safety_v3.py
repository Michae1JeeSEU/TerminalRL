"""Focused V3 safety-margin and source-provenance checks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.runtime_traffic_safety import PredictiveSafetyShield
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.envs.parallel_env import TerminalParallelEnv


V3 = "executable_trajectory_v3"


def _v3_config(**kwargs) -> SimConfig:
    return SimConfig(trajectory_schema_version=V3, vehicle_execution_mode="performance_constrained", **kwargs)


def _plan(*, start: float = 0.0, end: float = 1.0, offset: float = 0.0) -> TrajectoryPlan:
    points = (
        TrajectoryPoint(start, np.asarray((offset, 0.0, 0.0)), np.zeros(3), np.zeros(3)),
        TrajectoryPoint(end, np.asarray((offset + 1.0, 0.0, 0.0)), np.zeros(3), np.zeros(3)),
    )
    return TrajectoryPlan(
        "v3-safety-plan", 1, "transfer", points, start, end,
        metadata={"trajectory_schema_version": V3, "target_phase": "transfer"},
    )


def test_v3_between_sample_curve_budget_rejects_naive_safe_chord() -> None:
    config = _v3_config(safety_sampling_dt=1.0)
    safety = SafetyFilter(config)
    left = TrajectoryPlan(
        "stationary-left", 1, "transfer", (
            TrajectoryPoint(0.0, np.asarray((0.0, 0.0, 40.0)), np.zeros(3), np.zeros(3)),
            TrajectoryPoint(1.0, np.asarray((0.0, 0.0, 40.0)), np.zeros(3), np.zeros(3)),
        ), 0.0, 1.0,
        metadata={"trajectory_schema_version": V3, "target_phase": "transfer"},
    )
    right = TrajectoryPlan(
        "inward-bulge-right", 2, "transfer", (
            TrajectoryPoint(0.0, np.asarray((0.0, 80.05, 40.0)), np.asarray((0.0, -0.4, 0.0)), np.zeros(3)),
            TrajectoryPoint(1.0, np.asarray((0.0, 80.05, 40.0)), np.asarray((0.0, 0.4, 0.0)), np.zeros(3)),
        ), 0.0, 1.0,
        metadata={"trajectory_schema_version": V3, "target_phase": "transfer", "committed": True},
    )

    midpoint = right.sample_position(0.5)
    assert midpoint[1] < config.safe_distance
    assert right.polynomial_segments[0].bounds()["max_horizontal_accel"] <= config.vehicle_profile.max_horizontal_accel
    endpoint_gap = np.linalg.norm(right.sample_positions((0.0, 1.0)) - left.sample_positions((0.0, 1.0)), axis=1)
    assert np.all(endpoint_gap > config.safe_distance)
    snapshot = SimpleNamespace(now=0.0, active={}, airspace=None)
    assert safety.filter_plan(right, {}, snapshot).approved
    admission = safety.filter_plan(left, {right.uav_id: right}, snapshot)
    assert not admission.approved
    assert admission.reason == "predicted_swept_separation"

    verdict = safety.verify_predicted_tracks({1: left, 2: right}, start=0.0, end=1.0)
    assert not verdict.approved
    assert verdict.reason.startswith("repair_swept_separation:")


def test_v3_separated_pair_passes_after_curve_budget() -> None:
    safety = SafetyFilter(_v3_config())
    left = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    right = np.asarray(((100.0, 0.0, 0.0), (100.0, 0.0, 0.0)))
    _, swept = safety._track_distances_with_margin(
        left, right, np.asarray((0.0, 1.0)), "flight_v3", "flight_v3",
    )
    assert swept[0] > safety.config.safe_distance


def test_v3_separated_flight_and_direct_holder_proxy_pass() -> None:
    safety = SafetyFilter(_v3_config())
    left = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    right = np.asarray(((100.0, 0.0, 0.0), (100.0, 0.0, 0.0)))
    _, swept = safety._track_distances_with_margin(
        left, right, np.asarray((0.0, 1.0)), "flight_v3", "holder_proxy",
    )
    assert swept[0] > safety.config.safe_distance


def test_v3_unqualified_interior_acceleration_is_rejected_before_separation() -> None:
    config = _v3_config()
    safety = SafetyFilter(config)
    plan = _plan()
    snapshot = SimpleNamespace(now=0.0, active={}, airspace=None)

    verdict = safety.filter_plan(plan, {}, snapshot)
    assert not verdict.approved
    assert verdict.reason == "dynamics_horizontal_acceleration_limit"


def test_v3_shorter_final_interval_uses_its_actual_time_budget() -> None:
    safety = SafetyFilter(_v3_config())
    left = np.zeros((3, 3), dtype=float)
    right = np.full((3, 3), 100.0, dtype=float)
    times = np.asarray((0.0, 0.2, 0.7))
    _, swept = safety._track_distances_with_margin(left, right, times, "flight_v3", "flight_v3")
    assert (100.0 - swept[0]) < (100.0 - swept[1])


def test_v3_broad_phase_does_not_exclude_a_pair_without_curve_proof() -> None:
    config = _v3_config(reset_mode="warm_start_rings", warm_start_uavs=1, fixed_arrival_schedule=(0.0,))
    env = TerminalParallelEnv(config)
    env.reset(11)
    holder = next(iter(env.active.values()))
    safety = SafetyFilter(config)
    shield = PredictiveSafetyShield(config)
    assert safety._holder_broad_phase_safe(holder, SimpleNamespace(airspace=env.airspace), np.zeros((2, 3))) is False
    assert shield._broad_phase_reject(holder, _plan(), env.airspace, env.now) is False


def test_v3_joint_tracks_sample_committed_curve_at_safety_times() -> None:
    config = _v3_config(
        reset_mode="warm_start_rings", warm_start_uavs=1, fixed_arrival_schedule=(0.0,),
        safety_sampling_dt=1.0, physics_dt=0.2,
    )
    env = TerminalParallelEnv(config)
    env.reset(13)
    uid, uav = next(iter(env.active.items()))
    start = env.now
    points = (
        TrajectoryPoint(start, uav.position.copy(), np.asarray((0.0, 0.0, 0.0)), np.zeros(3)),
        TrajectoryPoint(start + 1.0, uav.position + np.asarray((1.0, 0.0, 0.0)), np.zeros(3), np.zeros(3)),
    )
    plan = TrajectoryPlan(
        "v3-joint-sample", uid, "transfer", points, start, start + 1.0,
        metadata={"trajectory_schema_version": V3, "target_phase": "transfer", "committed": True},
    )
    uav.committed_plan = plan
    shield = PredictiveSafetyShield(config)
    samples, tracks = shield._joint_position_tracks(env, {}, {uid}, horizon=1.0)

    np.testing.assert_allclose(tracks[uid], plan.sample_positions(samples), atol=1e-12, rtol=0.0)
    assert shield._joint_track_info[uid]["kind"] == "flight_v3"


def test_v3_unknown_source_fails_closed_in_predicted_track_verifier() -> None:
    config = _v3_config()
    safety = SafetyFilter(config)
    legacy = TrajectoryPlan(
        "unknown-v3-source", 1, "repair", (
            TrajectoryPoint(0.0, np.zeros(3)),
            TrajectoryPoint(1.0, np.ones(3)),
        ), 0.0, 1.0, metadata={"committed": False},
    )
    verdict = safety.verify_predicted_tracks({1: legacy}, start=0.0, end=1.0)
    assert not verdict.approved
    assert verdict.reason == "trajectory_uncertified"


def test_v3_committed_plan_without_prior_certificate_still_validates() -> None:
    config = _v3_config()
    safety = SafetyFilter(config)
    bad = TrajectoryPlan(
        "unqualified-committed-v3", 1, "repair", (
            TrajectoryPoint(0.0, np.zeros(3), np.asarray((20.0, 0.0, 0.0)), np.zeros(3)),
            TrajectoryPoint(1.0, np.ones(3)),
        ), 0.0, 1.0,
        metadata={"trajectory_schema_version": V3, "committed": True},
    )
    verdict = safety.verify_predicted_tracks({1: bad}, start=0.0, end=1.0)
    assert not verdict.approved
    assert verdict.reason == "repair_dynamics"


def test_v3_tube_interval_margin_catches_interior_shortcut() -> None:
    config = _v3_config()
    safety = SafetyFilter(config)
    ring = SimpleNamespace(radius=200.0, height=40.0)
    points = np.asarray(((280.1, 0.0, 40.0), (280.1, 0.0, 40.0)))
    lower = np.asarray((0.0, 0.0))
    upper = np.asarray((0.0, 0.0))
    distances = safety._tube_distances_with_margin(
        points, ring, lower, upper, np.asarray((0.0, 1.0)), "flight_v3",
    )
    assert safety._tube_point_distances(points, ring, lower, upper)[0] > config.safe_distance
    assert np.any(distances < config.safe_distance)


def test_v2_pair_margin_and_broad_phase_remain_unchanged() -> None:
    config = SimConfig()
    safety = SafetyFilter(config)
    left = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    right = np.asarray(((80.1, 0.0, 0.0), (80.1, 0.0, 0.0)))
    expected = safety._track_distances(left, right)
    actual = safety._track_distances_with_margin(left, right, np.asarray((0.0, 1.0)), "legacy", "legacy")
    for first, second in zip(expected, actual):
        np.testing.assert_array_equal(first, second)
