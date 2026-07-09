"""Tests for simulation-only ablation rollout orchestration."""

import pytest

from terminal_marl.experiments import (
    ABLATION_METRIC_KEYS,
    run_ablation,
    run_rollout,
)


@pytest.mark.parametrize("policy", ["rule-only", "random-policy", "dry-run"])
def test_runner_supports_requested_policy_types(policy: str) -> None:
    metrics = run_rollout("fixed_rule", policy=policy, seed=7, max_steps=2)

    assert set(ABLATION_METRIC_KEYS) <= set(metrics)
    assert metrics["experiment_mode"] == "fixed_rule"
    assert metrics["policy"] == policy
    assert metrics["total_completion_time"] == pytest.approx(2.0)
    assert metrics["min_pairwise_distance"] >= 0.0


def test_ablation_runs_all_core_modes_with_common_schema() -> None:
    results = run_ablation(
        ["fixed_rule", "fixed_residual", "dynamic_rule", "dynamic_hrl"],
        policy="dry-run",
        seed=8,
        max_steps=1,
    )

    assert set(results) == {
        "fixed_rule",
        "fixed_residual",
        "dynamic_rule",
        "dynamic_hrl",
    }
    for metrics in results.values():
        assert set(ABLATION_METRIC_KEYS) <= set(metrics)


def test_random_policy_is_seed_reproducible() -> None:
    first = run_rollout("dynamic_hrl", policy="random-policy", seed=9, max_steps=3)
    second = run_rollout("dynamic_hrl", policy="random-policy", seed=9, max_steps=3)
    assert first == second


def test_invalid_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="policy"):
        run_rollout("fixed_rule", policy="free-flight", max_steps=1)
