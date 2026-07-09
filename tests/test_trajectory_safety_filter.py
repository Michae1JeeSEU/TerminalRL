"""Detailed trajectory-level safety checks and bounded-hover fallback tests."""

import numpy as np

from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint


def make_filter() -> SafetyFilter:
    return SafetyFilter(
        safe_distance=0.5,
        boundary=(-2.0, 2.0, -2.0, 2.0, 0.1, 2.0),
        speed_max=1.0,
        acceleration_max=1.0,
        min_segment_duration=0.2,
    )


def plan(points: list[TrajectoryPoint]) -> TrajectoryPlan:
    return TrajectoryPlan("uav_1", "test", points, {})


def test_safe_plan_returns_diagnostics_without_modification() -> None:
    original = plan(
        [
            TrajectoryPoint(0.0, 0.0, 1.0, 0.0, 0.0),
            TrajectoryPoint(0.2, 0.0, 1.0, 0.0, 0.5),
            TrajectoryPoint(0.4, 0.0, 1.0, 0.0, 1.0),
        ]
    )
    filtered, info = make_filter().filter_trajectory_plan(original)

    assert filtered is original
    assert info["is_safe"]
    assert info["reason"] is None
    assert set(info["diagnostics"]) == {"boundary", "height", "kinematics"}


def test_time_and_minimum_segment_duration_are_checked() -> None:
    non_monotonic = plan(
        [
            TrajectoryPoint(0.0, 0.0, 1.0, 0.0, 1.0),
            TrajectoryPoint(0.0, 0.0, 1.0, 0.0, 0.5),
        ]
    )
    short = plan(
        [
            TrajectoryPoint(0.0, 0.0, 1.0, 0.0, 0.0),
            TrajectoryPoint(0.01, 0.0, 1.0, 0.0, 0.1),
        ]
    )

    assert make_filter().check_speed_acceleration(non_monotonic) == ["non_monotonic_time"]
    assert "minimum_segment_duration" in make_filter().check_speed_acceleration(short)


def test_speed_acceleration_boundary_and_height_are_checked() -> None:
    unsafe = plan(
        [
            TrajectoryPoint(0.0, 0.0, 1.0, 0.0, 0.0),
            TrajectoryPoint(1.5, 0.0, 1.0, 0.0, 0.5),
            TrajectoryPoint(1.5, 0.0, 2.5, 0.0, 1.0),
        ]
    )
    safety = make_filter()

    assert "speed_limit" in safety.check_speed_acceleration(unsafe)
    assert "acceleration_limit" in safety.check_speed_acceleration(unsafe)
    assert safety.check_boundary(unsafe) == ["boundary_violation"]
    assert safety.check_height_range(unsafe) == ["height_range"]


def test_unsafe_plan_becomes_safe_bounded_hover() -> None:
    unsafe = plan(
        [
            TrajectoryPoint(5.0, 0.0, -1.0, 0.0, 1.0),
            TrajectoryPoint(6.0, 0.0, -1.0, 0.0, 0.5),
        ]
    )
    safety = make_filter()
    filtered, info = safety.filter_trajectory_plan(unsafe)

    assert not info["is_safe"]
    assert info["modified"]
    assert filtered.mode == "safety_override_hover"
    assert not safety.check_boundary(filtered)
    assert not safety.check_height_range(filtered)
    assert not safety.check_speed_acceleration(filtered)
    positions = np.array([[point.x, point.y, point.z] for point in filtered.points])
    assert np.allclose(positions, positions[0])
