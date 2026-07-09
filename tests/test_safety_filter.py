"""Tests for deterministic safety checks and override behavior."""

from dataclasses import dataclass

import numpy as np

from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint


@dataclass
class State:
    uid: int
    position: np.ndarray


def make_filter() -> SafetyFilter:
    return SafetyFilter(
        safe_distance=0.5,
        boundary=(-2.0, 2.0, -2.0, 2.0, 0.0, 2.0),
        speed_max=1.0,
        acceleration_max=2.0,
    )


def test_airspace_action_is_clipped() -> None:
    safety = make_filter()
    airspace = DynamicAirspace()
    filtered, info = safety.filter_airspace_action(np.full(8, 3.0), airspace)

    assert np.all(filtered == 1.0)
    assert info["modified"]
    assert "action_bounds" in info["violations"]


def test_fixed_airspace_action_is_overridden() -> None:
    safety = make_filter()
    airspace = DynamicAirspace(dynamic_enabled=False)
    filtered, info = safety.filter_airspace_action(np.ones(8), airspace)

    assert np.all(filtered == 0.0)
    assert "fixed_airspace_override" in info["violations"]


def test_intention_action_is_clipped_and_release_can_be_blocked() -> None:
    safety = make_filter()
    filtered, info = safety.filter_intention_action(
        np.array([2.0, -2.0, 0.5, 0.25]),
        release_allowed=False,
    )

    assert np.allclose(filtered, [-1.0, -1.0, 0.5, 0.25])
    assert "action_bounds" in info["violations"]
    assert "release_override" in info["violations"]


def test_pairwise_distance_reports_only_close_pairs() -> None:
    safety = make_filter()
    states = [
        State(1, np.array([0.0, 0.0, 1.0])),
        State(2, np.array([0.3, 0.0, 1.0])),
        State(3, np.array([1.5, 0.0, 1.0])),
    ]
    violations = safety.check_pairwise_distance(states)

    assert len(violations) == 1
    assert {violations[0]["first"], violations[0]["second"]} == {"1", "2"}


def test_boundary_and_kinematic_checks() -> None:
    safety = make_filter()
    plan = TrajectoryPlan(
        uav_id="1",
        mode="test",
        points=[
            TrajectoryPoint(0.0, 0.0, 1.0, 0.0, 0.0),
            TrajectoryPoint(2.5, 0.0, 1.0, 0.0, 1.0),
            TrajectoryPoint(2.5, 0.0, 1.0, 0.0, 2.0),
        ],
        metadata={},
    )

    assert safety.check_boundary(plan) == ["boundary_violation"]
    assert "speed_limit" in safety.check_speed_acceleration(plan)
    assert "acceleration_limit" in safety.check_speed_acceleration(plan)


def test_unsafe_trajectory_is_overridden_with_hover() -> None:
    safety = make_filter()
    plan = TrajectoryPlan(
        uav_id="1",
        mode="landing",
        points=[
            TrajectoryPoint(3.0, 0.0, 1.0, 0.0, 0.0),
            TrajectoryPoint(0.0, 0.0, 1.0, 0.0, 1.0),
        ],
        metadata={},
    )

    filtered, info = safety.filter_trajectory_plan(plan)

    assert filtered.mode == "safety_override_hover"
    assert info["modified"]
    assert not info["approved"]
    positions = {(point.x, point.y, point.z) for point in filtered.points}
    assert positions == {(2.0, 0.0, 1.0)}
