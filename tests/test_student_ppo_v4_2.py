from dataclasses import replace

import numpy as np
import pytest
import torch

from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42, StudentStepV42, team_gae


def step(policy, *, reward=1., n=2, terminated=True, episode_end=True, next_value=0.):
    world = np.random.default_rng(n).normal(size=(n, 88)).astype(np.float32)
    batch = policy.act_details(world, world)
    return StudentStepV42(world, batch.requested_gate,
        np.asarray([-1 if p is None else p for p in batch.requested_proposal]),
        batch.gate_logprob, np.asarray([np.nan if p is None else p for p in batch.proposal_logprob]),
        np.ones(n, dtype=bool), batch.value, next_value, reward, terminated, episode_end)


def test_ppo_is_locked_without_admission():
    policy = StudentPPOV42(88)
    with pytest.raises(PermissionError):
        policy.update([step(policy)])


def test_deterministic_eval_cannot_be_used_as_ppo_behavior():
    policy = StudentPPOV42(88)
    with pytest.raises(ValueError, match="deterministic"):
        policy.update([replace(step(policy), stochastic_behavior=False)], training_authorized=True)


def test_time_limit_bootstraps_but_does_not_leak_next_episode():
    policy = StudentPPOV42(88)
    first = replace(step(policy, terminated=False, next_value=4.), value=2.)
    second = replace(step(policy, reward=100.), value=3.)
    adv, returns = team_gae([first, second], .5, 1.)
    np.testing.assert_allclose(adv, [1., 97.])
    np.testing.assert_allclose(returns, [3., 100.])


def test_teacher_replacement_and_wait_proposal_are_rejected():
    policy = StudentPPOV42(88)
    row = step(policy)
    with pytest.raises(ValueError, match="Teacher"):
        replace(row, teacher_used=True).validate(88)
    with pytest.raises(ValueError, match="sentinel"):
        replace(row, requested_gate=np.zeros(2, dtype=int), requested_proposal=np.zeros(2, dtype=int)).validate(88)


def test_variable_n_update_has_finite_gradients_and_unit_initial_ratio():
    torch.manual_seed(17)
    policy = StudentPPOV42(88)
    rows = [step(policy, reward=1., n=2), step(policy, reward=3., n=5)]
    before = policy.model.gate_head.weight.detach().clone()
    result = policy.update(rows, training_authorized=True, epochs=1, minibatch_size=2)
    assert all(np.isfinite(v) for v in result.values())
    assert result["approx_kl"] == pytest.approx(0., abs=1e-7)
    assert result["actor_decisions"] == 7
    assert not torch.equal(before, policy.model.gate_head.weight)


def test_committed_worlds_update_critic_without_changing_actor():
    policy = StudentPPOV42(88)
    row = replace(step(policy), actor_mask=np.zeros(2, dtype=bool))
    before = {k: v.detach().clone() for k, v in policy.model.state_dict().items() if "critic" not in k}
    result = policy.update([row], training_authorized=True, epochs=1)
    assert result["actor_decisions"] == 0
    assert result["entropy"] == 0
    assert all(torch.equal(v, policy.model.state_dict()[k]) for k, v in before.items())
