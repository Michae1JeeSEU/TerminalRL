"""Regression coverage for the low-level bounded PPO action distribution."""
from __future__ import annotations

import numpy as np
import torch

from terminal_marl.algo import RolloutBuffer, SharedPPO, Transition
from terminal_marl.core.observations import LOW_OBS_V2


def _observations(count: int = 256) -> np.ndarray:
    return np.zeros((count, LOW_OBS_V2.dimension), dtype=np.float32)


def test_tanh_policy_actions_are_bounded_without_an_exact_endpoint_pile() -> None:
    torch.manual_seed(7)
    policy = SharedPPO(LOW_OBS_V2.dimension)
    actions, _, _ = policy.act(_observations(20_000), _observations(20_000))
    assert np.all(actions <= 1.0) and np.all(actions >= -1.0)
    assert not np.any(np.abs(actions) == 1.0)


def test_bounded_sample_log_prob_and_ppo_ratio_use_the_same_requested_action() -> None:
    torch.manual_seed(9)
    policy = SharedPPO(LOW_OBS_V2.dimension)
    observation = _observations(2)
    action, old_log_prob, value = policy.act(observation, observation)
    with torch.no_grad():
        distribution = policy.model.actor_distribution(torch.as_tensor(observation), bounded=True)
        recomputed = distribution.log_prob(torch.as_tensor(action)).sum(-1).cpu().numpy()
    assert np.allclose(old_log_prob, recomputed, atol=2e-6)

    buffer = RolloutBuffer()
    for index in range(2):
        buffer.add(Transition(f"uav-{index}", observation[index], observation, action[index], float(old_log_prob[index]),
                              value, 1.0, True, applied_action=np.full(3, -1.0, dtype=np.float32)))
    snapshot = policy.diagnostic_snapshot(buffer, teacher_bc_weight=0.0)
    assert snapshot["ppo"]["approx_kl"] == pytest_approx_zero()
    # The action in the buffer is the bounded request, never a later applied action.
    assert np.array_equal(buffer.by_agent["uav-0"][0].action, action[0])


def pytest_approx_zero():
    # Keep this tiny helper local to avoid importing pytest into runtime code.
    import pytest
    return pytest.approx(0.0, abs=2e-6)


def test_deterministic_action_is_tanh_of_latent_mean_and_teacher_endpoints_are_finite() -> None:
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05)
    observation = _observations(1)
    first, _, _ = policy.act(observation, observation, deterministic=True)
    second, _, _ = policy.act(observation, observation, deterministic=True)
    assert np.array_equal(first, second)
    assert np.all(np.abs(first) < 1.0)

    buffer = RolloutBuffer()
    buffer.add(Transition("uav", observation[0], observation, first[0], 0.0, 0.0, 1.0, True,
                          teacher_action=np.asarray((1.0, -1.0, 1.0), dtype=np.float32),
                          teacher_used=True))
    metrics = policy.update(buffer, epochs=1, minibatch_size=1)
    assert all(np.isfinite(value) for value in metrics.values())


def test_historical_control_exposes_its_clamp_pile_for_d3_comparison() -> None:
    torch.manual_seed(11)
    policy = SharedPPO(LOW_OBS_V2.dimension, policy_distribution="gaussian_hard_clamp")
    actions, _, _ = policy.act(_observations(10_000), _observations(10_000))
    assert np.mean(np.abs(actions[:, 0]) == 1.0) > .10
