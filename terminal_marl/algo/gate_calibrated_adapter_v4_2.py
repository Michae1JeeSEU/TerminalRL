"""Fixed-temperature calibration wrapper for the V4.2 gate adapter.

The calibration layer changes only the gate distribution temperature.  The
residual-adapter model, proposal head, and centralized critic are copied from
the validated V2 repair checkpoint without modification.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .gate_adapter_v4_2 import (
    OBSERVATION_DIM,
    GateResidualAdapterActorCentralCriticV42,
    GateResidualAdapterStudentPPOV42,
)


CALIBRATION_TEMPERATURE = 2.5
CALIBRATED_ARCHITECTURE_ID = "v4_2_hierarchical_gate_temperature_calibrated_residual_adapter_v3"
TEMPERATURE_BUFFER = "gate_temperature"


def _validate_temperature(value: float | torch.Tensor) -> float:
    temperature = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0.0:
        raise ValueError("gate temperature must be finite and positive")
    return temperature


class TemperatureCalibratedGateResidualAdapterActorCentralCriticV42(
    GateResidualAdapterActorCentralCriticV42
):
    """Residual-adapter actor whose gate logits are divided by fixed T."""

    architecture_id = CALIBRATED_ARCHITECTURE_ID

    def __init__(self, observation_dim: int = OBSERVATION_DIM,
                 *, temperature: float = CALIBRATION_TEMPERATURE) -> None:
        temperature_value = _validate_temperature(temperature)
        super().__init__(observation_dim)
        self.register_buffer(
            TEMPERATURE_BUFFER,
            torch.tensor(temperature_value, dtype=torch.float32),
            persistent=True,
        )

    def distributions(self, observations: torch.Tensor) -> tuple[Categorical, Categorical]:
        encoded = self.actor_encoder(observations)
        base_gate_logits = self.gate_head(encoded)
        gate_logits = (base_gate_logits + self.gate_adapter(observations)) / self.gate_temperature
        gate = Categorical(logits=gate_logits)
        proposal = Categorical(logits=self.proposal_head(encoded))
        return gate, proposal


class TemperatureCalibratedGateResidualAdapterStudentPPOV42(
    GateResidualAdapterStudentPPOV42
):
    """Student-compatible policy container for calibrated act/rollout calls."""

    architecture_id = CALIBRATED_ARCHITECTURE_ID

    def __init__(self, observation_dim: int = OBSERVATION_DIM, *, device: str = "cpu",
                 temperature: float = CALIBRATION_TEMPERATURE,
                 learning_rate: float = 3e-4) -> None:
        self.device = torch.device(device)
        self.model = TemperatureCalibratedGateResidualAdapterActorCentralCriticV42(
            observation_dim,
            temperature=temperature,
        ).to(self.device)
        # The calibration phase never calls an optimizer.  A fresh optimizer
        # keeps this container compatible with the StudentPPO interface while
        # avoiding any source optimizer-state migration.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.observation_dim = int(observation_dim)


def migrate_v2_policy_to_calibrated(
    source: GateResidualAdapterStudentPPOV42,
    *,
    temperature: float = CALIBRATION_TEMPERATURE,
) -> TemperatureCalibratedGateResidualAdapterStudentPPOV42:
    """Strictly copy a V2 adapter policy and introduce only the temperature buffer."""
    if not isinstance(source, GateResidualAdapterStudentPPOV42) or isinstance(
        source, TemperatureCalibratedGateResidualAdapterStudentPPOV42
    ):
        raise TypeError("source must be a V2 GateResidualAdapterStudentPPOV42")
    if int(source.observation_dim) != OBSERVATION_DIM:
        raise ValueError(f"expected {OBSERVATION_DIM}D V2 adapter policy")
    temperature_value = _validate_temperature(temperature)
    if temperature_value != CALIBRATION_TEMPERATURE:
        raise ValueError("V3 calibration temperature is fixed at 2.5")
    target = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        source.observation_dim,
        device=str(source.device),
        temperature=temperature_value,
    )
    source_state = source.model.state_dict()
    target_state = target.model.state_dict()
    target_only = {TEMPERATURE_BUFFER}
    source_keys = set(source_state)
    target_keys = set(target_state)
    common_keys = target_keys - target_only
    missing = sorted(common_keys - source_keys)
    unexpected = sorted(source_keys - common_keys)
    if missing:
        raise ValueError(f"V2 adapter state missing common keys: {missing}")
    if unexpected:
        raise ValueError(f"V2 adapter state has unexpected keys: {unexpected}")
    for key in sorted(common_keys):
        if tuple(source_state[key].shape) != tuple(target_state[key].shape):
            raise ValueError(f"V2 adapter state shape mismatch for {key}")
        target_state[key] = source_state[key].detach().clone().to(target_state[key].device)
    target.model.load_state_dict(target_state, strict=True)
    for key in sorted(common_keys):
        if not torch.equal(
            source_state[key].detach().cpu(),
            target.model.state_dict()[key].detach().cpu(),
        ):
            raise ValueError(f"V2 adapter state copy was not exact for {key}")
    target.migration = {
        "copied_common_state_names": sorted(common_keys),
        "new_buffer_names": sorted(target_only),
        "missing_common_keys": missing,
        "unexpected_source_keys": unexpected,
        "source_optimizer_state_copied": False,
        "training_called": False,
        "temperature": temperature_value,
    }
    return target


migrate_v2_to_calibrated = migrate_v2_policy_to_calibrated


def calibrated_buffer_names(model: nn.Module) -> tuple[str, ...]:
    """Return the persistent state introduced by V3 calibration."""
    return tuple(name for name, _ in model.named_buffers() if name == TEMPERATURE_BUFFER)

