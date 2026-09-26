from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.algo.gate_reference_v4_2 import GateReference
from terminal_marl.algo.student_ppo_v4_2 import StudentStepV42


def _fixed_reference(**kwargs) -> GateReference:
    probabilities = torch.tensor([[.75, .25], [.25, .75]], dtype=torch.float32)
    return GateReference(
        observations=torch.zeros((2, 88), dtype=torch.float32),
        reference_logits=torch.log(probabilities),
        reference_probs=probabilities,
        weights=torch.ones(2),
        **kwargs,
    )


def test_soft_penalty_has_old_gradient_at_or_below_target_and_extra_gradient_above():
    reference = _fixed_reference(coefficient=.1, max_kl=.05, soft_kl_target=.02, soft_kl_gain=400.)

    below = torch.tensor(.02, requires_grad=True)
    below_loss = reference.regularization_loss(below)
    below_gradient = torch.autograd.grad(below_loss, below)[0]
    assert below_loss.item() == pytest.approx(.002)
    assert below_gradient.item() == pytest.approx(.1)

    above = torch.tensor(.03, requires_grad=True)
    above_loss = reference.regularization_loss(above)
    above_gradient = torch.autograd.grad(above_loss, above)[0]
    assert above_loss.item() == pytest.approx(.043)
    assert above_gradient.item() == pytest.approx(8.1)


def test_default_reference_penalty_is_exact_old_form_and_metrics_are_compatible():
    reference = _fixed_reference()
    weighted_kl = torch.tensor(.03, requires_grad=True)
    expected = .1 * weighted_kl
    torch.testing.assert_close(reference.regularization_loss(weighted_kl), expected)
    assert reference.soft_kl_target is None
    assert reference.soft_kl_gain == 0.0

    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    metrics = reference.metrics(policy.model)
    assert metrics["effective_kl_coefficient"] == pytest.approx(.1)
    assert metrics["soft_kl_target"] is None
    assert metrics["soft_kl_gain"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "target,gain",
    [(None, 400.), (.05, 400.), (float("nan"), 400.), (-1., 400.)],
)
def test_soft_penalty_requires_target_inside_hard_cap(target, gain):
    with pytest.raises(ValueError, match="soft KL"):
        _fixed_reference(max_kl=.05, soft_kl_target=target, soft_kl_gain=gain)


@pytest.mark.parametrize("gain", [-1., float("nan"), float("inf")])
def test_soft_penalty_gain_must_be_finite_and_nonnegative(gain):
    with pytest.raises(ValueError, match="soft KL gain"):
        _fixed_reference(soft_kl_gain=gain)


def _step(policy, *, seed: int) -> StudentStepV42:
    world = np.random.default_rng(seed).normal(size=(2, 88)).astype(np.float32)
    batch = policy.act_details(world, world)
    return StudentStepV42(
        world,
        batch.requested_gate,
        np.asarray([-1 if value is None else value for value in batch.requested_proposal]),
        batch.gate_logprob,
        np.asarray([np.nan if value is None else value for value in batch.proposal_logprob]),
        np.ones(2, dtype=bool),
        batch.value,
        0.0,
        1.0 + seed * .01,
        True,
        True,
    )


def test_real_calibrated_gate_update_learns_with_soft_penalty_and_keeps_reference_frozen():
    torch.manual_seed(707)
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88, learning_rate=1e-5)
    reference_observations = np.random.default_rng(17).normal(size=(8, 88)).astype(np.float32)
    reference = GateReference.from_policy(
        policy,
        reference_observations,
        np.ones(8, dtype=np.float32),
        coefficient=.1,
        max_kl=.05,
        soft_kl_target=.02,
        soft_kl_gain=400.,
    )
    with torch.no_grad():
        policy.model.gate_adapter[-1].bias.zero_()
        policy.model.gate_adapter[-1].bias[1] = 1.2
    initial_reference_logits = reference.reference_logits.clone()
    initial_kl = float(reference.weighted_kl(policy.model).detach())
    assert .02 < initial_kl < .05
    before = copy.deepcopy(policy.model.state_dict())
    policy.gate_reference = reference

    result = policy.update(
        [_step(policy, seed=1), _step(policy, seed=2), _step(policy, seed=3), _step(policy, seed=4)],
        training_authorized=True,
        epochs=1,
        minibatch_size=4,
        target_kl=.01,
    )

    assert result["updates"] == 1
    assert policy.optimizer.state
    assert any(not torch.equal(before[key], policy.model.state_dict()[key]) for key in before)
    assert torch.equal(initial_reference_logits, reference.reference_logits)
