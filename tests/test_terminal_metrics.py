import math

import numpy as np

from terminal_marl import ExperimentMode, SimConfig, TerminalParallelEnv


def test_no_pair_metrics_do_not_report_false_zero_distance_or_conflict() -> None:
    env = TerminalParallelEnv(
        SimConfig(initial_aircraft_count=1, formation_rows=1, formation_cols=1),
        experiment_mode=ExperimentMode.FIXED_RULE,
    )
    env.reset(seed=3)

    metrics = env.metrics()

    assert metrics["has_pair"] == 0.0
    assert metrics["current_pair_count"] == 0.0
    assert metrics["pair_opportunities"] == 0.0
    assert metrics["normalized_conflict_rate"] == 0.0
    assert math.isinf(metrics["min_pairwise_distance"])


def test_pairwise_conflict_metrics_are_normalized_by_pair_opportunities() -> None:
    env = TerminalParallelEnv(
        SimConfig(
            initial_aircraft_count=2,
            formation_rows=1,
            formation_cols=2,
            col_spacing=100.0,
            safe_distance=80.0,
        ),
        experiment_mode=ExperimentMode.FIXED_RULE,
    )
    env.reset(seed=4)
    env.aircraft[0].position = np.array([0.0, 0.0, 100.0])
    env.aircraft[1].position = np.array([50.0, 0.0, 100.0])

    env._track_conflicts()
    metrics = env.metrics()

    assert metrics["has_pair"] == 1.0
    assert metrics["pair_opportunities"] == 1.0
    assert metrics["raw_pair_conflict_samples"] == 1.0
    assert metrics["unique_conflict_events"] == 1.0
    assert metrics["normalized_conflict_rate"] == 1.0
    assert metrics["min_pairwise_distance"] == 50.0


def test_waiting_and_congestion_include_approach_backlog_and_active_limit() -> None:
    env = TerminalParallelEnv(
        SimConfig(
            initial_aircraft_count=2,
            formation_rows=1,
            formation_cols=2,
            max_active_uavs=2,
        ),
        experiment_mode=ExperimentMode.FIXED_RULE,
    )
    env.reset(seed=5)

    env._track_waiting_and_congestion()
    metrics = env.metrics()

    assert metrics["approach_waiting_time"] > 0.0
    assert metrics["total_waiting_time"] > 0.0
    assert metrics["congestion_pressure_time"] > 0.0
    assert metrics["normalized_congestion"] > 0.0
