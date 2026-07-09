import numpy as np
import pytest

from terminal_marl import SimConfig
from terminal_marl.metrics import NominalScheduleEstimator


def test_analytic_schedule_includes_headway() -> None:
    estimator = NominalScheduleEstimator(SimConfig())
    pos = np.array([1000.0, 0.0, 500.0])
    first = estimator.estimate(pos, spawn_time=0.0, sequence_index=1)
    second = estimator.estimate(pos, spawn_time=0.0, sequence_index=2)
    assert second - first == pytest.approx(SimConfig().final_min_time_gap_s)


def test_rule_reference_uses_reference_times() -> None:
    estimator = NominalScheduleEstimator(
        SimConfig(), mode="rule-only-reference", rule_only_reference={1: 42.0}
    )
    assert estimator.estimate(np.zeros(3), spawn_time=5.0, sequence_index=1) == 47.0
