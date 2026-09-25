"""Probability-floored gate calibration for the V4.2 residual adapter.

The bounded calibration changes only the categorical gate distribution.  It
keeps the calibrated actor encoder, gate residual, proposal head, critic, and
temperature state intact, then mixes the gate probabilities with a symmetric
floor so both WAIT and RELEASE retain nonzero stochastic support.  This is an
inference/training-distribution candidate; it does not modify old loaders or
existing checkpoints.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .gate_adapter_v4_2 import OBSERVATION_DIM
from .gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TEMPERATURE_BUFFER,
    TemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from .student_ppo_v4_2 import StudentPPOV42


MINIMUM_GATE_PROBABILITY_BUFFER = "minimum_gate_probability"
DEFAULT_MINIMUM_GATE_PROBABILITY = 0.05
BOUNDED_CALIBRATED_ARCHITECTURE_ID = (
    "v4_2_hierarchical_gate_temperature_bounded_probability_residual_adapter_v4"
)


def _validate_minimum_gate_probability(value: float | torch.Tensor) -> float:
    if isinstance(value, bool):
        raise ValueError("minimum_gate_probability must be finite and in [0, .5)")
    probability = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not torch.isfinite(torch.tensor(probability)) or not 0.0 <= probability < 0.5:
        raise ValueError("minimum_gate_probability must be finite and in [0, .5)")
    return probability


class BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42(
    TemperatureCalibratedGateResidualAdapterActorCentralCriticV42
):
    """Calibrated residual adapter with a symmetric stochastic gate floor."""

    architecture_id = BOUNDED_CALIBRATED_ARCHITECTURE_ID

    def __init__(self, observation_dim: int = OBSERVATION_DIM,
                 *, temperature: float = CALIBRATION_TEMPERATURE,
                 minimum_gate_probability: float = DEFAULT_MINIMUM_GATE_PROBABILITY) -> None:
        minimum_probability = _validate_minimum_gate_probability(minimum_gate_probability)
        super().__init__(observation_dim, temperature=temperature)
        self.register_buffer(
            MINIMUM_GATE_PROBABILITY_BUFFER,
            torch.tensor(minimum_probability, dtype=torch.float32),
            persistent=True,
        )

    def distributions(self, observations: torch.Tensor) -> tuple[Categorical, Categorical]:
        encoded = self.actor_encoder(observations)
        base_gate_logits = self.gate_head(encoded)
        calibrated_gate_logits = (
            base_gate_logits + self.gate_adapter(observations)
        ) / self.gate_temperature
        calibrated_gate_probabilities = torch.softmax(calibrated_gate_logits, dim=-1)
        floor = self.minimum_gate_probability
        gate_probabilities = floor + (1.0 - 2.0 * floor) * calibrated_gate_probabilities
        gate = Categorical(probs=gate_probabilities)
        proposal = Categorical(logits=self.proposal_head(encoded))
        return gate, proposal


class BoundedTemperatureCalibratedGateResidualAdapterStudentPPOV42(StudentPPOV42):
    """Student PPO container for the bounded calibrated actor."""

    architecture_id = BOUNDED_CALIBRATED_ARCHITECTURE_ID

    def __init__(self, observation_dim: int = OBSERVATION_DIM, *, device: str = "cpu",
                 temperature: float = CALIBRATION_TEMPERATURE,
                 minimum_gate_probability: float = DEFAULT_MINIMUM_GATE_PROBABILITY,
                 learning_rate: float = 3e-4) -> None:
        self.device = torch.device(device)
        self.model = BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42(
            observation_dim,
            temperature=temperature,
            minimum_gate_probability=minimum_gate_probability,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.observation_dim = int(observation_dim)


def migrate_calibrated_policy_to_bounded(
    source: TemperatureCalibratedGateResidualAdapterStudentPPOV42,
    *,
    minimum_gate_probability: float = DEFAULT_MINIMUM_GATE_PROBABILITY,
) -> BoundedTemperatureCalibratedGateResidualAdapterStudentPPOV42:
    """Strictly copy a calibrated policy and add only the probability-floor buffer."""

    if not isinstance(source, TemperatureCalibratedGateResidualAdapterStudentPPOV42):
        raise TypeError("source must be a calibrated residual-adapter StudentPPO policy")
    if isinstance(source, BoundedTemperatureCalibratedGateResidualAdapterStudentPPOV42):
        raise TypeError("source must be the unbounded calibrated StudentPPO policy")
    if int(source.observation_dim) != OBSERVATION_DIM:
        raise ValueError(f"expected {OBSERVATION_DIM}D calibrated policy")
    source_temperature = float(source.model.gate_temperature.detach().cpu())
    if source_temperature != float(CALIBRATION_TEMPERATURE):
        raise ValueError("calibrated source gate_temperature must remain 2.5")
    minimum_probability = _validate_minimum_gate_probability(minimum_gate_probability)
    target = BoundedTemperatureCalibratedGateResidualAdapterStudentPPOV42(
        source.observation_dim,
        device=str(source.device),
        temperature=source_temperature,
        minimum_gate_probability=minimum_probability,
    )
    source_state = source.model.state_dict()
    target_state = target.model.state_dict()
    target_only = {MINIMUM_GATE_PROBABILITY_BUFFER}
    source_keys = set(source_state)
    target_keys = set(target_state)
    common_keys = target_keys - target_only
    missing = sorted(common_keys - source_keys)
    unexpected = sorted(source_keys - common_keys)
    if missing:
        raise ValueError(f"calibrated source state missing keys: {missing}")
    if unexpected:
        raise ValueError(f"calibrated source state has unexpected keys: {unexpected}")
    for key in sorted(common_keys):
        if tuple(source_state[key].shape) != tuple(target_state[key].shape):
            raise ValueError(f"calibrated state shape mismatch for {key}")
        target_state[key] = source_state[key].detach().clone().to(target_state[key].device)
    target.model.load_state_dict(target_state, strict=True)
    for key in sorted(common_keys):
        if not torch.equal(
            source_state[key].detach().cpu(),
            target.model.state_dict()[key].detach().cpu(),
        ):
            raise ValueError(f"calibrated state copy was not exact for {key}")
    target.migration = {
        "copied_common_state_names": sorted(common_keys),
        "new_buffer_names": sorted(target_only),
        "missing_common_keys": missing,
        "unexpected_source_keys": unexpected,
        "source_optimizer_state_copied": False,
        "training_called": False,
        "gate_temperature": source_temperature,
        "minimum_gate_probability": minimum_probability,
    }
    return target


migrate_calibrated_to_bounded = migrate_calibrated_policy_to_bounded


def bounded_calibration_buffer_names(model: nn.Module) -> tuple[str, ...]:
    """Return the persistent state introduced by bounded calibration."""

    return tuple(
        name for name, _ in model.named_buffers()
        if name == MINIMUM_GATE_PROBABILITY_BUFFER
    )
