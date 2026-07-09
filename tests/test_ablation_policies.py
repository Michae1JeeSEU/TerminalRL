"""Smoke tests for every non-PPO ablation policy."""

import pytest

from terminal_marl.experiments import POLICIES, run_rollout


@pytest.mark.parametrize("policy", POLICIES)
def test_every_ablation_policy_runs_one_episode_step(policy: str) -> None:
    metrics = run_rollout(
        "dynamic_hrl",
        policy=policy,
        seed=4,
        max_steps=1,
    )
    assert metrics["policy"] == policy
    assert metrics["total_completion_time"] == pytest.approx(1.0)


def test_high_level_random_changes_dynamic_airspace() -> None:
    random_metrics = run_rollout(
        "dynamic_hrl",
        policy="high-level-random",
        seed=5,
        max_steps=1,
    )
    static_metrics = run_rollout(
        "dynamic_hrl",
        policy="high-level-static",
        seed=5,
        max_steps=1,
    )
    assert random_metrics["policy"] != static_metrics["policy"]
