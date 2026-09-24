from __future__ import annotations

import numpy as np

from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint


def _plan() -> TrajectoryPlan:
    times = (2.0, 3.5, 7.0, 11.0, 16.0)
    positions = (
        (0.0, 0.0, 0.0),
        (4.0, -1.0, 2.0),
        (12.0, 3.0, 4.0),
        (8.0, 9.0, 7.0),
        (1.0, 5.0, 10.0),
    )
    return TrajectoryPlan(
        "batch-sampling", 1, "test",
        tuple(TrajectoryPoint(time, position) for time, position in zip(times, positions)),
        times[0], times[-1],
    )


def test_batch_position_sampling_matches_scalar_at_endpoints_knots_and_near_knots():
    plan = _plan()
    epsilon = 1e-10
    query_times = np.asarray((
        plan.start_time - 1.0,
        plan.start_time,
        plan.start_time + epsilon,
        3.5 - epsilon,
        3.5,
        3.5 + epsilon,
        7.0 - epsilon,
        7.0,
        7.0 + epsilon,
        11.0 - epsilon,
        11.0,
        11.0 + epsilon,
        plan.end_time,
        plan.end_time + 1.0,
    ))

    expected = np.asarray([plan.sample_position(float(time)) for time in query_times])
    actual = plan.sample_positions(query_times)

    assert actual.shape == (len(query_times), 3)
    np.testing.assert_array_equal(actual, expected)


def test_batch_position_sampling_matches_scalar_for_random_irregular_times():
    plan = _plan()
    rng = np.random.default_rng(20260924)
    query_times = rng.uniform(plan.start_time - 5.0, plan.end_time + 5.0, size=1000)

    expected = np.asarray([plan.sample_position(float(time)) for time in query_times])
    actual = plan.sample_positions(query_times)

    np.testing.assert_array_equal(actual, expected)


def test_batch_position_sampling_preserves_scalar_shape_for_single_time_and_empty_batches():
    plan = _plan()

    assert plan.sample_positions(5.0).shape == (3,)
    np.testing.assert_array_equal(plan.sample_positions(5.0), plan.sample_position(5.0))
    assert plan.sample_positions(np.empty(0, dtype=float)).shape == (0, 3)
