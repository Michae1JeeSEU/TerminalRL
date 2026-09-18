import copy
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


def _state_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right))
    return left == right


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


def test_target_kl_accepts_normal_update_and_keeps_original_metrics():
    torch.manual_seed(31)
    policy = StudentPPOV42(88, learning_rate=.00003)
    rows = [step(policy, reward=reward, n=2 + index) for index, reward in enumerate((1., 2., 3., 4.))]
    result = policy.update(rows, training_authorized=True, epochs=1, minibatch_size=4, target_kl=.01)
    assert result["accepted_updates"] == result["updates"] == 1
    assert result["attempted_updates"] == 1
    assert result["rejected_updates"] == 0
    assert result["stopped"] is False
    assert 0. <= result["final_rollout_kl"] <= .01
    assert result["max_accepted_rollout_kl"] == result["final_rollout_kl"]
    assert np.isfinite(result["approx_kl"])
    assert np.isfinite(result["final_rollout_kl"])


def test_target_kl_rejection_restores_model_and_adam_state(monkeypatch):
    torch.manual_seed(32)
    policy = StudentPPOV42(88, learning_rate=.0003)
    rows = [step(policy, reward=1., n=2), step(policy, reward=3., n=3)]
    accepted = policy.update(rows, training_authorized=True, epochs=1, minibatch_size=2, target_kl=.01)
    assert accepted["accepted_updates"] == accepted["updates"] == 1
    assert policy.optimizer.state_dict()["state"]
    model_before = copy.deepcopy(policy.model.state_dict())
    optimizer_before = copy.deepcopy(policy.optimizer.state_dict())
    monkeypatch.setattr(
        "terminal_marl.algo.student_ppo_v4_2._full_rollout_sampled_approx_kl",
        lambda _policy, _steps: .02,
    )
    result = policy.update(rows, training_authorized=True, epochs=2, minibatch_size=2, target_kl=.01)
    assert result["accepted_updates"] == result["updates"] == 0
    assert result["attempted_updates"] == result["rejected_updates"] == 1
    assert result["stopped"] is True
    assert result["final_rollout_kl"] == 0.
    assert result["max_accepted_rollout_kl"] == 0.
    assert all(torch.equal(model_before[key], policy.model.state_dict()[key]) for key in model_before)
    assert _state_equal(optimizer_before, policy.optimizer.state_dict())


def test_target_kl_nonfinite_model_after_step_restores_model_and_optimizer(monkeypatch):
    torch.manual_seed(35)
    policy = StudentPPOV42(88, learning_rate=.0003)
    rows = [step(policy, reward=1., n=2), step(policy, reward=3., n=3)]
    accepted = policy.update(rows, training_authorized=True, epochs=1, minibatch_size=2, target_kl=.01)
    assert accepted["accepted_updates"] == accepted["updates"] == 1
    assert policy.optimizer.state_dict()["state"]
    model_before = copy.deepcopy(policy.model.state_dict())
    optimizer_before = copy.deepcopy(policy.optimizer.state_dict())
    original_step = policy.optimizer.step

    def nonfinite_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            next(policy.model.parameters()).fill_(float("nan"))
        return result

    monkeypatch.setattr(policy.optimizer, "step", nonfinite_step)
    result = policy.update(rows, training_authorized=True, epochs=2, minibatch_size=2, target_kl=.01)
    assert result["accepted_updates"] == result["updates"] == 0
    assert result["attempted_updates"] == result["rejected_updates"] == 1
    assert result["stopped"] is True
    assert all(torch.isfinite(value).all() for value in policy.model.state_dict().values())
    assert all(torch.isfinite(value).all() for value in policy.optimizer.state_dict()["state"].values()
               for value in value.values() if isinstance(value, torch.Tensor))
    assert all(torch.equal(model_before[key], policy.model.state_dict()[key]) for key in model_before)
    assert _state_equal(optimizer_before, policy.optimizer.state_dict())


@pytest.mark.parametrize("target_kl", [0., -1., np.nan, np.inf, -np.inf])
def test_target_kl_requires_finite_positive_value(target_kl):
    policy = StudentPPOV42(88)
    with pytest.raises(ValueError, match="target_kl"):
        policy.update([step(policy)], training_authorized=True, target_kl=target_kl)


def test_target_kl_no_actor_path_is_finite_and_does_not_divide_by_zero():
    torch.manual_seed(33)
    policy = StudentPPOV42(88)
    row = replace(step(policy, reward=2.), actor_mask=np.zeros(2, dtype=bool))
    actor_before = {key: value.detach().clone() for key, value in policy.model.state_dict().items()
                    if not key.startswith("critic")}
    result = policy.update([row], training_authorized=True, epochs=2, minibatch_size=1, target_kl=.01)
    assert result["accepted_updates"] == result["updates"] == 2
    assert result["attempted_updates"] == 2
    assert result["rejected_updates"] == 0
    assert result["stopped"] is False
    assert result["final_rollout_kl"] == result["max_accepted_rollout_kl"] == 0.
    assert result["approx_kl"] == result["entropy"] == 0.
    assert all(torch.equal(value, policy.model.state_dict()[key]) for key, value in actor_before.items())


def test_target_kl_keeps_teacher_rejection_and_disabled_output_compatibility():
    torch.manual_seed(34)
    policy = StudentPPOV42(88)
    row = step(policy)
    with pytest.raises(ValueError, match="Teacher"):
        policy.update([replace(row, teacher_used=True)], training_authorized=True, target_kl=.01)

    result = policy.update([row], training_authorized=True, epochs=1, minibatch_size=1, target_kl=None)
    assert set(result) == {"loss", "actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                           "grad_norm", "updates", "team_steps", "actor_decisions"}
    assert "accepted_updates" not in result
    assert "final_rollout_kl" not in result
