import numpy as np

from terminal_marl.training.run_v4_2_v2_1_i1_failure_attribution import (
    _macro_f1,
    _stats,
    _total_variation,
)


def test_stats_reports_requested_summary() -> None:
    result = _stats([1.0, 2.0, 3.0, float("nan")])
    assert result["count"] == 3
    assert result["mean"] == 2.0
    assert result["median"] == 2.0
    assert result["p90"] == 2.8


def test_total_variation_detects_context_shift() -> None:
    assert _total_variation({"low": 5}, {"low": 7}) == 0.0
    assert _total_variation({"low": 5}, {"high": 7}) == 1.0
    assert _total_variation({"low": 1, "high": 1}, {"low": 1}) == 0.5


def test_macro_f1_perfect_binary_predictions() -> None:
    target = np.asarray([0, 0, 1, 1])
    assert _macro_f1(target, target.copy()) == 1.0
