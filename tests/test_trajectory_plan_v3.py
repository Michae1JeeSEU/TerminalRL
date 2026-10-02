"""Tests for TrajectoryPlan V3 polynomial execution semantics."""

from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint


V3 = {"trajectory_schema_version": "executable_trajectory_v3"}


def _points(*, yaw_shift: float = 0.0, velocity_shift: float = 0.0) -> tuple[TrajectoryPoint, ...]:
    return (
        TrajectoryPoint(
            0.0,
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((1.0 + velocity_shift, 0.0, 0.0)),
            np.asarray((0.0, 0.5, 0.0)),
            0.1 + yaw_shift,
            0.01,
        ),
        TrajectoryPoint(
            1.0,
            np.asarray((1.5, 0.5, 0.25)),
            np.asarray((0.5, 1.0, 0.2)),
            np.asarray((-0.1, 0.0, 0.1)),
            0.5 + yaw_shift,
            0.02,
        ),
        TrajectoryPoint(
            2.0,
            np.asarray((2.0, 2.0, 0.5)),
            np.asarray((0.0, 0.5, 0.0)),
            np.asarray((0.2, -0.1, 0.0)),
            0.9 + yaw_shift,
            0.03,
        ),
    )


def _plan(*, metadata: dict | None = None, points: tuple[TrajectoryPoint, ...] | None = None) -> TrajectoryPlan:
    details = dict(V3 if metadata is None else metadata)
    return TrajectoryPlan(
        "plan-v3",
        7,
        "final_approach",
        _points() if points is None else points,
        0.0,
        2.0,
        metadata=details,
    )


def test_v3_scalar_batch_sampling_and_derivatives_match() -> None:
    plan = _plan()
    times = np.asarray(((2.5, -1.0), (1.25, 0.0)), dtype=float)
    batch = plan.sample_positions(times)
    assert batch.shape == (2, 2, 3)
    for index in np.ndindex(times.shape):
        scalar = plan.sample(float(times[index]))
        np.testing.assert_allclose(batch[index], scalar.position, atol=1e-11)
        np.testing.assert_allclose(plan.sample_position(float(times[index])), scalar.position, atol=1e-11)

    interior = 0.73
    position, velocity, acceleration = plan.sample(interior).position, plan.sample(interior).velocity, plan.sample(interior).acceleration
    epsilon = 1.0e-5
    before = plan.sample(interior - epsilon)
    after = plan.sample(interior + epsilon)
    np.testing.assert_allclose((after.position - before.position) / (2.0 * epsilon), velocity, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose((after.velocity - before.velocity) / (2.0 * epsilon), acceleration, atol=1e-6, rtol=1e-6)

    np.testing.assert_allclose(plan.sample(-10.0).position, plan.points[0].position, atol=1e-12)
    np.testing.assert_allclose(plan.sample(10.0).position, plan.points[-1].position, atol=1e-12)
    np.testing.assert_allclose(plan.sample_position(-10.0), plan.points[0].position, atol=1e-12)
    np.testing.assert_allclose(plan.sample_position(10.0), plan.points[-1].position, atol=1e-12)


def test_v3_segments_are_c2_and_read_only() -> None:
    plan = _plan()
    assert plan.metadata["trajectory_schema_version"] == "executable_trajectory_v3"
    assert isinstance(plan.polynomial_segments, tuple)
    assert len(plan.polynomial_segments) == 2
    assert all(segment.coefficients.flags.writeable is False for segment in plan.polynomial_segments)
    for left, right in zip(plan.polynomial_segments, plan.polynomial_segments[1:]):
        left_state = left.sample(left.end_time)
        right_state = right.sample(right.start_time)
        for left_value, right_value in zip(left_state, right_state):
            np.testing.assert_allclose(left_value, right_value, atol=1e-11)


def test_segment_bounds_cache_is_private_and_return_is_a_copy() -> None:
    segment = _plan().polynomial_segments[0]
    first = segment.bounds()
    assert isinstance(segment._bounds_cache, tuple)
    first["max_horizontal_speed"] = -1.0
    second = segment.bounds()
    assert second["max_horizontal_speed"] >= 0.0
    assert second["max_horizontal_speed"] != -1.0


def test_v3_hash_binds_schema_coefficients_and_yaw_nodes() -> None:
    base = _plan()
    changed_yaw = _plan(points=_points(yaw_shift=0.001))
    changed_velocity = _plan(points=_points(velocity_shift=0.001))
    v2 = TrajectoryPlan("plan-v2", 7, "final_approach", _points(), 0.0, 2.0)

    assert base.verify_trajectory_integrity() is True
    assert changed_yaw.trajectory_hash != base.trajectory_hash
    assert changed_velocity.trajectory_hash != base.trajectory_hash
    assert v2.trajectory_hash != base.trajectory_hash
    assert v2.polynomial_segments == ()

    with_reservation = base.with_reservation_ids(("reservation-1",))
    assert with_reservation.trajectory_hash == base.trajectory_hash
    assert with_reservation.metadata["trajectory_schema_version"] == "executable_trajectory_v3"
    assert with_reservation.polynomial_segments is not base.polynomial_segments
    for first, second in zip(base.polynomial_segments, with_reservation.polynomial_segments):
        assert first.start_time == second.start_time
        assert first.end_time == second.end_time
        np.testing.assert_array_equal(first.coefficients, second.coefficients)


def test_v2_default_keeps_linear_sampling_branch() -> None:
    plan = TrajectoryPlan("plan-v2", 1, "test", _points(), 0.0, 2.0)
    assert plan.metadata["trajectory_schema_version"] == "executable_trajectory_v2"
    assert plan.polynomial_segments == ()
    expected = plan.points[0].position + 0.5 * (plan.points[1].position - plan.points[0].position)
    np.testing.assert_allclose(plan.sample(0.5).position, expected, atol=1e-12)
    np.testing.assert_allclose(plan.sample_positions([0.5]), [expected], atol=1e-12)
    assert plan.verify_trajectory_integrity() is True


def test_v3_rejects_unknown_schema_and_invalid_point_contracts() -> None:
    with pytest.raises(ValueError, match="unsupported trajectory schema"):
        TrajectoryPlan("unknown", 1, "test", _points(), 0.0, 2.0, metadata={"trajectory_schema_version": "future"})
    with pytest.raises(ValueError, match="match first and last"):
        TrajectoryPlan("bad-time", 1, "test", _points(), 0.0, 2.1, metadata=dict(V3))

    bad_shape = list(_points())
    bad_shape[1] = TrajectoryPoint(1.0, np.zeros(2), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        TrajectoryPlan("bad-shape", 1, "test", tuple(bad_shape), 0.0, 2.0, metadata=dict(V3))

    bad_time = list(_points())
    bad_time[1] = TrajectoryPoint(np.nan, np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="time must be finite"):
        TrajectoryPlan("bad-point-time", 1, "test", tuple(bad_time), 0.0, 2.0, metadata=dict(V3))
