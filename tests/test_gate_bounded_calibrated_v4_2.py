from __future__ import annotations

import pytest
import torch

from terminal_marl.algo.gate_bounded_calibrated_v4_2 import (
    BOUNDED_CALIBRATED_ARCHITECTURE_ID,
    DEFAULT_MINIMUM_GATE_PROBABILITY,
    MINIMUM_GATE_PROBABILITY_BUFFER,
    BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
    BoundedTemperatureCalibratedGateResidualAdapterStudentPPOV42,
    migrate_calibrated_policy_to_bounded,
)
from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATED_ARCHITECTURE_ID,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42


def _observations() -> torch.Tensor:
    torch.manual_seed(701)
    return torch.randn(7, 88)


def test_extreme_gate_logits_respect_probability_floor_and_are_finite() -> None:
    model = BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42(
        88, minimum_gate_probability=0.05
    )
    with torch.no_grad():
        model.gate_head.weight.zero_()
        model.gate_head.bias.copy_(torch.tensor([1000.0, -1000.0]))
    gate, proposal = model.distributions(_observations())
    assert torch.isfinite(gate.probs).all()
    assert torch.isfinite(gate.logits).all()
    assert torch.isfinite(proposal.logits).all()
    assert bool((gate.probs >= 0.05 - 1e-6).all())
    assert bool((gate.probs <= 0.95 + 1e-6).all())
    assert torch.allclose(gate.probs[:, 0], torch.full((7,), 0.95), atol=1e-6)
    assert torch.allclose(gate.probs[:, 1], torch.full((7,), 0.05), atol=1e-6)


def test_zero_floor_is_identity_and_proposal_distribution_is_unchanged() -> None:
    torch.manual_seed(702)
    source = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    bounded = migrate_calibrated_policy_to_bounded(
        source, minimum_gate_probability=0.0
    )
    observations = _observations()
    source_gate, source_proposal = source.model.distributions(observations)
    bounded_gate, bounded_proposal = bounded.model.distributions(observations)
    assert torch.allclose(source_gate.probs, bounded_gate.probs, atol=1e-7, rtol=1e-7)
    assert torch.equal(source_proposal.logits, bounded_proposal.logits)
    assert torch.equal(source_proposal.probs, bounded_proposal.probs)
    assert torch.equal(
        source.model.gate_temperature, bounded.model.gate_temperature
    )


def test_migration_copies_old_state_bitwise_and_adds_only_floor_buffer() -> None:
    torch.manual_seed(703)
    source = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    bounded = migrate_calibrated_policy_to_bounded(source)
    source_state = source.model.state_dict()
    bounded_state = bounded.model.state_dict()
    assert bounded.architecture_id == BOUNDED_CALIBRATED_ARCHITECTURE_ID
    assert bounded.architecture_id != CALIBRATED_ARCHITECTURE_ID
    assert isinstance(bounded, StudentPPOV42)
    assert set(bounded_state) - set(source_state) == {MINIMUM_GATE_PROBABILITY_BUFFER}
    for name, value in source_state.items():
        assert torch.equal(value, bounded_state[name]), name
    assert torch.equal(
        bounded_state[MINIMUM_GATE_PROBABILITY_BUFFER],
        torch.tensor(DEFAULT_MINIMUM_GATE_PROBABILITY),
    )
    assert bounded.migration["source_optimizer_state_copied"] is False
    assert bounded.migration["training_called"] is False
    assert bounded.optimizer.state == {}


def test_categorical_sampling_logprob_matches_probability_and_gradient_is_finite() -> None:
    torch.manual_seed(704)
    policy = BoundedTemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, minimum_gate_probability=0.05
    )
    observations = _observations().requires_grad_(True)
    gate, proposal = policy.model.distributions(observations)
    sampled = gate.sample()
    direct_logprob = torch.log(gate.probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1))
    assert sampled.shape == (7,)
    assert bool(torch.isfinite(gate.log_prob(sampled)).all())
    assert torch.allclose(gate.log_prob(sampled), direct_logprob, atol=1e-6)
    loss = -gate.log_prob(sampled).mean() + proposal.entropy().mean()
    loss.backward()
    assert observations.grad is not None
    assert torch.isfinite(observations.grad).all()
    gradients = [parameter.grad for parameter in policy.model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


@pytest.mark.parametrize("value", [-1e-6, 0.5, 0.500001, float("nan"), float("inf"), True])
def test_invalid_probability_floor_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="minimum_gate_probability"):
        BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42(
            88, minimum_gate_probability=value
        )
