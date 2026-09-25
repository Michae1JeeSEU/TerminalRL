import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_reference_v4_2 import GateReference
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42, StudentStepV42


def _step(policy: StudentPPOV42, *, reward: float = 1.0, n: int = 2) -> StudentStepV42:
    world = np.random.default_rng(n).normal(size=(n, 88)).astype(np.float32)
    batch = policy.act_details(world, world)
    return StudentStepV42(
        world,
        batch.requested_gate,
        np.asarray([-1 if value is None else value for value in batch.requested_proposal]),
        batch.gate_logprob,
        np.asarray([np.nan if value is None else value for value in batch.proposal_logprob]),
        np.ones(n, dtype=bool),
        batch.value,
        0.0,
        reward,
        True,
        True,
    )


def _state_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right))
    return left == right


def _reference(policy: StudentPPOV42, *, max_kl: float = .05) -> GateReference:
    observations = np.random.default_rng(91).normal(size=(5, 88)).astype(np.float32)
    weights = np.asarray([1., 2., 3., 4., 5.], dtype=np.float32)
    return GateReference.from_policy(
        policy, observations, weights, coefficient=.1, max_kl=max_kl,
    )


def test_reference_snapshot_is_detached_and_gate_kl_has_no_proposal_gradient():
    torch.manual_seed(101)
    policy = StudentPPOV42(88)
    observations = np.random.default_rng(11).normal(size=(4, 88)).astype(np.float32)
    weights = np.ones(4, dtype=np.float32)
    reference = GateReference.from_policy(policy, observations, weights)
    observations[0, 0] = 99.0
    weights[0] = 99.0
    assert reference.observations[0, 0].item() != 99.0
    assert np.isclose(float(reference.weights.sum()), 1.0)
    assert not reference.q_logits.requires_grad
    assert not reference.q_probs.requires_grad

    with torch.no_grad():
        policy.model.gate_head.bias[0].add_(.2)
    policy.model.zero_grad(set_to_none=True)
    loss = reference.weighted_kl(policy.model)
    assert loss.requires_grad
    loss.backward()
    assert policy.model.gate_head.weight.grad is not None
    assert float(policy.model.gate_head.weight.grad.abs().sum()) > 0.0
    assert policy.model.proposal_head.weight.grad is None


def test_reference_allows_float32_probability_underflow_but_requires_consistent_logits():
    policy = StudentPPOV42(88)
    observations = torch.zeros((2, 88))
    reference = GateReference(
        observations=observations,
        reference_logits=torch.tensor([[-1000.0, 0.0], [0.0, -1000.0]]),
        reference_probs=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        weights=torch.ones(2),
    )
    metrics = reference.metrics(policy.model)
    assert np.isfinite(metrics["reference_kl"])
    with pytest.raises(ValueError, match="disagree|normalized"):
        GateReference(
            observations=observations,
            reference_logits=torch.zeros((2, 2)),
            reference_probs=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            weights=torch.ones(2),
        )
    with pytest.raises(ValueError, match="positive"):
        GateReference.from_policy(policy, observations, torch.tensor([1.0, 0.0]))


def test_reference_requires_target_kl_and_initial_excess_is_rejected_without_rebinding():
    torch.manual_seed(102)
    policy = StudentPPOV42(88, learning_rate=.0003)
    rows = [_step(policy, n=2), _step(policy, n=3)]
    reference = _reference(policy, max_kl=1e-12)
    policy.gate_reference = reference
    with pytest.raises(ValueError, match="target_kl"):
        policy.update(rows, training_authorized=True, epochs=1)
    with torch.no_grad():
        policy.model.gate_head.bias[0].add_(.5)
    model_before = copy.deepcopy(policy.model.state_dict())
    optimizer_before = copy.deepcopy(policy.optimizer.state_dict())
    result = policy.update(rows, training_authorized=True, epochs=1, target_kl=.1)
    assert result["updates"] == 0
    assert result["reference_rejected_updates"] == 1
    assert result["rejection_reason"] == "reference_kl_before_exceeds_max"
    assert result["reference_kl_final"] == pytest.approx(result["reference_kl_before"])
    assert all(torch.equal(model_before[key], policy.model.state_dict()[key]) for key in model_before)
    assert _state_equal(optimizer_before, policy.optimizer.state_dict())
    assert torch.equal(reference.q_logits, reference.reference_logits)


def test_poststep_reference_kl_rejection_restores_model_and_real_adam_state(monkeypatch):
    torch.manual_seed(103)
    policy = StudentPPOV42(88, learning_rate=.0003)
    rows = [_step(policy, reward=1.0, n=2), _step(policy, reward=3.0, n=3)]
    policy.gate_reference = _reference(policy, max_kl=1.0)
    accepted = policy.update(
        rows, training_authorized=True, epochs=1, minibatch_size=2, target_kl=.5,
    )
    assert accepted["updates"] == 1
    assert policy.optimizer.state_dict()["state"]
    model_before = copy.deepcopy(policy.model.state_dict())
    optimizer_before = copy.deepcopy(policy.optimizer.state_dict())
    original_metrics = GateReference.metrics
    calls = {"count": 0}

    def force_poststep_excess(self, model):
        result = original_metrics(self, model)
        calls["count"] += 1
        if calls["count"] >= 2:
            result["reference_kl"] = self.max_kl + 1.0
            result["kl"] = result["reference_kl"]
        return result

    monkeypatch.setattr(GateReference, "metrics", force_poststep_excess)
    result = policy.update(
        rows, training_authorized=True, epochs=1, minibatch_size=2, target_kl=.5,
    )
    assert calls["count"] >= 2
    assert result["updates"] == 0
    assert result["reference_rejected_updates"] == 1
    assert result["rejection_reason"] == "reference_kl_exceeds_max"
    assert all(torch.equal(model_before[key], policy.model.state_dict()[key]) for key in model_before)
    assert _state_equal(optimizer_before, policy.optimizer.state_dict())


def test_no_reference_preserves_existing_update_output_shape():
    policy = StudentPPOV42(88)
    result = policy.update([_step(policy)], training_authorized=True, epochs=1, target_kl=None)
    assert set(result) == {
        "loss", "actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
        "grad_norm", "updates", "team_steps", "actor_decisions",
    }
