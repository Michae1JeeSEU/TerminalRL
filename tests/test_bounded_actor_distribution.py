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


def test_squashed_entropy_has_finite_gradients_for_boundary_seeking_logits() -> None:
    policy = SharedPPO(LOW_OBS_V2.dimension)
    with torch.no_grad():
        policy.model.actor_mean.bias.fill_(20.0)
    distribution = policy.model.actor_distribution(torch.as_tensor(_observations(4)), bounded=True)
    loss = -distribution.entropy().sum()
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in policy.model.parameters())


def test_mixed_release_sample_and_joint_log_prob_are_consistent() -> None:
    torch.manual_seed(19)
    policy = SharedPPO(LOW_OBS_V2.dimension, policy_distribution="mixed_bernoulli_tanh")
    observation = _observations(512)
    actions, log_prob, _, by_head = policy.act(observation, observation, return_head_log_probs=True)
    assert set(np.unique(actions[:, 0])).issubset({-1.0, 1.0})
    assert np.all(np.abs(actions[:, 1:]) < 1.0)
    assert np.allclose(log_prob, by_head.sum(axis=-1), atol=2e-6)
    with torch.no_grad():
        recomputed = policy.distribution(torch.as_tensor(observation)).log_prob(torch.as_tensor(actions)).sum(-1)
    assert np.allclose(log_prob, recomputed.cpu().numpy(), atol=2e-6)


def test_mixed_release_deterministic_mapping_binary_teacher_and_entropy() -> None:
    policy = SharedPPO(LOW_OBS_V2.dimension, policy_distribution="mixed_bernoulli_tanh", teacher_bc_weight=.05)
    observation = _observations(2)
    with torch.no_grad():
        policy.model.actor_mean.bias[0] = -0.1
    wait, _, _ = policy.act(observation[:1], observation[:1], deterministic=True)
    with torch.no_grad():
        policy.model.actor_mean.bias[0] = 0.1
    release, _, _ = policy.act(observation[:1], observation[:1], deterministic=True)
    assert wait[0, 0] == -1.0 and release[0, 0] == 1.0

    distribution = policy.distribution(torch.as_tensor(observation))
    targets = torch.as_tensor(((-1.0, -0.2, 0.3), (1.0, 0.2, -0.3)))
    losses = policy._teacher_loss_per_head(distribution, distribution.mean, targets)
    assert losses.shape == (2, 3) and torch.isfinite(losses).all()
    assert torch.isfinite(distribution.entropy()).all()


def test_mixed_policy_keeps_speed_and_merge_tanh_distribution_unchanged() -> None:
    continuous = SharedPPO(LOW_OBS_V2.dimension, policy_distribution="tanh_squashed_gaussian")
    mixed = SharedPPO(LOW_OBS_V2.dimension, policy_distribution="mixed_bernoulli_tanh")
    mixed.model.load_state_dict(continuous.model.state_dict())
    observation = torch.as_tensor(_observations(4))
    continuous_distribution = continuous.distribution(observation)
    mixed_distribution = mixed.distribution(observation)
    assert torch.equal(continuous_distribution.mean[:, 1:], mixed_distribution.mean[:, 1:])
    probe = torch.as_tensor(np.tile((1.0, .2, -.3), (4, 1)), dtype=torch.float32)
    assert torch.allclose(continuous_distribution.log_prob(probe)[:, 1:], mixed_distribution.log_prob(probe)[:, 1:])


def test_autoregressive_release_sample_log_prob_and_residual_conditioning_are_consistent() -> None:
    torch.manual_seed(29)
    source = SharedPPO(LOW_OBS_V2.dimension, policy_distribution="mixed_bernoulli_tanh")
    policy = SharedPPO(LOW_OBS_V2.dimension, policy_distribution="autoregressive_release_tanh")
    assert policy.load_model_state(source.model.state_dict(), allow_architecture_migration=True)
    observation = _observations(128)
    actions, log_prob, _, by_head = policy.act(observation, observation, return_head_log_probs=True)
    assert set(np.unique(actions[:, 0])).issubset({-1.0, 1.0})
    assert np.all(np.abs(actions[:, 1:]) < 1.0)
    distribution = policy.distribution(torch.as_tensor(observation))
    recomputed = distribution.log_prob(torch.as_tensor(actions))
    assert np.allclose(by_head, recomputed.detach().numpy(), atol=2e-6)
    assert np.allclose(log_prob, by_head.sum(axis=-1), atol=2e-6)

    first, last = policy.model.release_conditioner[0], policy.model.release_conditioner[-1]
    with torch.no_grad():
        first.weight.zero_(); first.bias.zero_(); first.weight[0, -1] = 1.
        last.weight.zero_(); last.bias.zero_(); last.weight[0, 0] = 2.
    hidden = policy.model.actor_body(torch.as_tensor(observation[:1]))
    logits = policy.model.actor_mean(hidden)
    conditioned = policy.model.actor_distribution(torch.as_tensor(observation[:1]), conditional_release=True)
    low = conditioned.release_logits_for(torch.as_tensor(((0., -.8),)))
    high = conditioned.release_logits_for(torch.as_tensor(((0., .8),)))
    assert low < logits[:, 0] < high


def test_autoregressive_release_teacher_loss_deterministic_mode_and_entropy_are_finite() -> None:
    policy = SharedPPO(LOW_OBS_V2.dimension, policy_distribution="autoregressive_release_tanh",
                       teacher_bc_weight=.05)
    observation = _observations(2)
    first, _, _ = policy.act(observation, observation, deterministic=True)
    second, _, _ = policy.act(observation, observation, deterministic=True)
    assert np.array_equal(first, second)
    distribution = policy.distribution(torch.as_tensor(observation))
    targets = torch.as_tensor(((-1., -.2, .3), (1., .2, -.3)))
    losses = policy._teacher_loss_per_head(distribution, distribution.mean, targets)
    assert losses.shape == (2, 3) and torch.isfinite(losses).all()
    entropy = distribution.entropy()
    assert entropy.shape == (2, 3) and torch.isfinite(entropy).all()
