"""Fresh V4.2 hierarchical requested-action policy; PPO updates remain locked."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..core.low_level_joint_action_v4_2 import (N_GATE_ACTIONS_V4_2,
    N_RELEASE_PROPOSALS_V4_2, RELEASE_GATE, WAIT_GATE, decode_hierarchical_actions_v4_2)
from .shared_ppo import mlp


class HierarchicalActorCentralCriticV42(nn.Module):
    def __init__(self, observation_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.actor_encoder = mlp(observation_dim, hidden, hidden)
        self.gate_head = nn.Linear(hidden, N_GATE_ACTIONS_V4_2)
        self.proposal_head = nn.Linear(hidden, N_RELEASE_PROPOSALS_V4_2)
        self.critic_encoder = mlp(observation_dim, hidden, hidden)
        self.critic_head = mlp(2 * hidden, 1, hidden)

    def distributions(self, observations: torch.Tensor) -> tuple[Categorical, Categorical]:
        encoded = self.actor_encoder(observations)
        return Categorical(logits=self.gate_head(encoded)), Categorical(logits=self.proposal_head(encoded))

    def conditional_entropy(self, observations: torch.Tensor) -> torch.Tensor:
        gate, proposal = self.distributions(observations)
        return gate.entropy() + gate.probs[:, RELEASE_GATE] * proposal.entropy()

    def critic(self, agent_sets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.critic_encoder(agent_sets); valid = mask.unsqueeze(-1)
        mean = (encoded * valid).sum(1) / valid.sum(1).clamp_min(1)
        maximum = encoded.masked_fill(~valid, float("-inf")).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.critic_head(torch.cat((mean, maximum), -1)).squeeze(-1)


def hierarchical_joint_logprob(gate_distribution: Categorical, proposal_distribution: Categorical,
                                gates: torch.Tensor, proposals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gate, conditional proposal, and requested-action joint log-probs."""
    if gates.shape != proposals.shape:
        raise ValueError("gate and proposal tensors must have the same shape")
    release = gates == RELEASE_GATE
    if torch.any((gates != WAIT_GATE) & ~release):
        raise ValueError("gate must be 0 or 1")
    if torch.any(release & ((proposals < 0) | (proposals >= N_RELEASE_PROPOSALS_V4_2))):
        raise ValueError("RELEASE proposal must be in [0, 25)")
    if torch.any(~release & (proposals != -1)):
        raise ValueError("WAIT proposal sentinel must be -1")
    gate_logprob = gate_distribution.log_prob(gates)
    proposal_logprob = torch.full_like(gate_logprob, float("nan"))
    # Categorical keeps the full batch shape.  Evaluate valid placeholders
    # for WAIT rows, then expose proposal likelihood only for RELEASE rows.
    proposal_values = torch.where(release, proposals, torch.zeros_like(proposals))
    all_proposal_logprob = proposal_distribution.log_prob(proposal_values)
    proposal_logprob[release] = all_proposal_logprob[release]
    joint = gate_logprob + torch.where(release, proposal_logprob, torch.zeros_like(gate_logprob))
    return gate_logprob, proposal_logprob, joint


def hierarchical_bc_loss(gate_logits: torch.Tensor, proposal_logits: torch.Tensor,
                         gate_targets: torch.Tensor, proposal_targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Unweighted CE: proposal CE is averaged only over real RELEASE labels."""
    gate_loss = nn.functional.cross_entropy(gate_logits, gate_targets)
    release = gate_targets == RELEASE_GATE
    if torch.any(release):
        proposal_loss = nn.functional.cross_entropy(proposal_logits[release], proposal_targets[release])
    else:
        proposal_loss = gate_loss.new_zeros(())
    return gate_loss + proposal_loss, {"gate": gate_loss, "proposal": proposal_loss, "release_mask": release}


@dataclass(frozen=True)
class HierarchicalPolicyBatchV42:
    requested_gate: np.ndarray
    requested_proposal: tuple[int | None, ...]
    decoded: np.ndarray
    gate_logprob: np.ndarray
    proposal_logprob: tuple[float | None, ...]
    joint_logprob: np.ndarray
    conditional_entropy: np.ndarray
    value: float


class JointPPOV42:
    """V4.2 actor container. It deliberately exposes no PPO update method."""
    action_schema_version = "low_action_schema_v4_2_hierarchical"

    def __init__(self, observation_dim: int, *, device: str = "cpu", learning_rate: float = .0003) -> None:
        self.device = torch.device(device); self.model = HierarchicalActorCentralCriticV42(observation_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate); self.observation_dim = observation_dim

    def act_details(self, observations: np.ndarray, world: np.ndarray, *, deterministic: bool = False) -> HierarchicalPolicyBatchV42:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        mask = torch.ones((1, len(world)), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            gate_dist, proposal_dist = self.model.distributions(obs)
            gates = gate_dist.probs.argmax(-1) if deterministic else gate_dist.sample()
            # Do not draw a behaviour proposal for WAIT rows.  Its proposal
            # is structurally absent, rather than sampled then discarded.
            proposals = torch.full_like(gates, -1)
            release = gates == RELEASE_GATE
            if torch.any(release):
                release_logits = proposal_dist.logits[release]
                release_proposals = release_logits.argmax(-1) if deterministic else Categorical(logits=release_logits).sample()
                proposals[release] = release_proposals
            gate_lp, proposal_lp, joint_lp = hierarchical_joint_logprob(gate_dist, proposal_dist, gates, proposals)
            entropy = self.model.conditional_entropy(obs); value = self.model.critic(torch.as_tensor(world[None], dtype=torch.float32, device=self.device), mask)[0]
        gate_np, proposal_np = gates.cpu().numpy(), proposals.cpu().numpy()
        return HierarchicalPolicyBatchV42(gate_np, tuple(None if value == -1 else int(value) for value in proposal_np),
            decode_hierarchical_actions_v4_2(gate_np, np.asarray([None if value == -1 else int(value) for value in proposal_np], dtype=object)),
            gate_lp.cpu().numpy(), tuple(None if value == -1 else float(logprob) for value, logprob in zip(proposal_np, proposal_lp.cpu().numpy())),
            joint_lp.cpu().numpy(), entropy.cpu().numpy(), float(value.cpu()))
