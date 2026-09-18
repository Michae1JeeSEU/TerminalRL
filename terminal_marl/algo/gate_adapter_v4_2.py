"""Residual gate adapter for the V4.2 P2 support-repair qualification.

The adapter is initialized as an exact identity over a validated hierarchical
V4.2 actor.  Only the new observation-to-gate residual is intended to move;
the base actor encoder, proposal head, and centralized critic remain copied
and can be frozen by the repair runner.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .joint_ppo_v4_2 import HierarchicalActorCentralCriticV42
from .shared_ppo import mlp
from .student_ppo_v4_2 import StudentPPOV42


OBSERVATION_DIM = 88
GATE_DIM = 2
ADAPTER_HIDDEN = 96
ADAPTER_PREFIX = "gate_adapter"


class GateResidualAdapterActorCentralCriticV42(HierarchicalActorCentralCriticV42):
    """V4.2 actor with an initially-zero residual gate-logit adapter."""

    architecture_id = "v4_2_hierarchical_gate_residual_adapter_v1"

    def __init__(self, observation_dim: int, hidden: int = ADAPTER_HIDDEN) -> None:
        super().__init__(observation_dim, hidden=hidden)
        self.gate_adapter = mlp(observation_dim, GATE_DIM, hidden)
        final = self.gate_adapter[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("gate adapter final layer must be Linear")
        # Only the final adapter layer is zero-initialized.  Its preceding
        # layers retain their normal initialization but are masked at the
        # identity start because the final affine map is exactly zero.
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def distributions(self, observations: torch.Tensor) -> tuple[Categorical, Categorical]:
        encoded = self.actor_encoder(observations)
        base_gate_logits = self.gate_head(encoded)
        gate_logits = base_gate_logits + self.gate_adapter(observations)
        gate = Categorical(logits=gate_logits)
        proposal = Categorical(logits=self.proposal_head(encoded))
        return gate, proposal


class GateResidualAdapterStudentPPOV42(StudentPPOV42):
    """Student PPO container using the residual-adapter actor architecture."""

    architecture_id = GateResidualAdapterActorCentralCriticV42.architecture_id

    def __init__(self, observation_dim: int = OBSERVATION_DIM, *, device: str = "cpu",
                 learning_rate: float = 3e-4) -> None:
        self.device = torch.device(device)
        self.model = GateResidualAdapterActorCentralCriticV42(observation_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.observation_dim = int(observation_dim)


def _copy_base_model_state(
    target: GateResidualAdapterStudentPPOV42,
    source: StudentPPOV42,
) -> dict[str, Any]:
    """Copy every base model tensor and explicitly account for adapter keys."""
    source_state = source.model.state_dict()
    target_state = target.model.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    adapter_keys = {key for key in target_keys if key == ADAPTER_PREFIX or key.startswith(ADAPTER_PREFIX + ".")}
    expected_base_keys = target_keys - adapter_keys
    missing = sorted(expected_base_keys - source_keys)
    unexpected = sorted(source_keys - expected_base_keys)
    if missing:
        raise ValueError(f"base model state missing keys: {missing}")
    if unexpected:
        raise ValueError(f"base model state has unexpected keys: {unexpected}")
    for key in sorted(expected_base_keys):
        if tuple(source_state[key].shape) != tuple(target_state[key].shape):
            raise ValueError(f"base model state shape mismatch for {key}")
        target_state[key] = source_state[key].detach().clone().to(target_state[key].device)
    # strict=True now proves that the only state not supplied by the base is
    # the intentionally new adapter state.
    target.model.load_state_dict(target_state, strict=True)
    return {
        "copied_parameter_names": sorted(expected_base_keys),
        "new_adapter_parameter_names": sorted(adapter_keys),
        "missing_base_keys": missing,
        "unexpected_base_keys": unexpected,
    }


def migrate_base_policy_to_adapter(
    source: StudentPPOV42,
    *,
    learning_rate: float = 3e-4,
) -> GateResidualAdapterStudentPPOV42:
    """Migrate a validated base StudentPPOV42 into an identity adapter policy.

    The source must be the ordinary V4.2 StudentPPOV42 container.  Existing
    model tensors are copied with explicit missing/unexpected-key checks; the
    only target-only keys are the new residual adapter parameters.
    """
    if not isinstance(source, StudentPPOV42) or isinstance(source, GateResidualAdapterStudentPPOV42):
        raise TypeError("source must be a validated base StudentPPOV42")
    if int(source.observation_dim) != OBSERVATION_DIM:
        raise ValueError(f"expected {OBSERVATION_DIM}D base policy")
    target = GateResidualAdapterStudentPPOV42(
        source.observation_dim,
        device=str(source.device),
        learning_rate=float(learning_rate),
    )
    model_copy = _copy_base_model_state(target, source)
    target.migration = {
        **model_copy,
        "base_optimizer_state_copied": False,
        "adapter_optimizer_state_initialized": False,
    }
    return target


# Explicit alias for callers that use the shorter migration spelling.
migrate_base_to_adapter = migrate_base_policy_to_adapter


def adapter_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Return the exact trainable residual parameter names."""
    return tuple(name for name, _ in model.named_parameters()
                 if name == ADAPTER_PREFIX or name.startswith(ADAPTER_PREFIX + "."))
