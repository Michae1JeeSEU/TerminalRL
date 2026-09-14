"""V4.1 joint-categorical actor: 26 requested scheduling actions.

This module deliberately has no checkpoint migration from V4's 30-way head.
PPO provenance is always the sampled/requested joint index and its log-prob.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..core.low_level_joint_action_v4_1 import N_JOINT_ACTIONS_V4_1, decode_joint_actions_v4_1
from .shared_ppo import mlp


class JointActorCentralCriticV41(nn.Module):
    def __init__(self, observation_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.actor_body = mlp(observation_dim, hidden, hidden)
        self.joint_action_logits = nn.Linear(hidden, N_JOINT_ACTIONS_V4_1)
        nn.init.zeros_(self.joint_action_logits.weight)
        nn.init.zeros_(self.joint_action_logits.bias)
        self.joint_action_logits.bias.data[13] = 0.5  # RELEASE, nominal speed/time
        self.critic_encoder = mlp(observation_dim, hidden, hidden)
        self.critic_head = mlp(2 * hidden, 1, hidden)

    def actor_distribution(self, observations: torch.Tensor) -> Categorical:
        return Categorical(logits=self.joint_action_logits(self.actor_body(observations)))

    def critic(self, agent_sets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.critic_encoder(agent_sets)
        valid = mask.unsqueeze(-1)
        mean = (encoded * valid).sum(1) / valid.sum(1).clamp_min(1)
        maximum = encoded.masked_fill(~valid, float("-inf")).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.critic_head(torch.cat((mean, maximum), -1)).squeeze(-1)


@dataclass(frozen=True)
class JointPolicyBatchV41:
    joint_action_index: np.ndarray
    decoded: np.ndarray
    joint_log_prob: np.ndarray
    value: float


class JointPPOV41:
    """Fresh V4.1 actor interface; qualification never invokes ``update``."""
    action_schema_version = "low_action_schema_v4_1_joint_discrete_26"

    def __init__(self, observation_dim: int, *, device: str = "cpu", learning_rate: float = .0003) -> None:
        self.device = torch.device(device)
        self.model = JointActorCentralCriticV41(observation_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.observation_dim = observation_dim

    def act_details(self, observations: np.ndarray, world: np.ndarray, *, deterministic: bool = False) -> JointPolicyBatchV41:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        mask = torch.ones((1, len(world)), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            distribution = self.model.actor_distribution(obs)
            action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            value = self.model.critic(torch.as_tensor(world[None], dtype=torch.float32, device=self.device), mask)[0]
        indices = action.cpu().numpy().astype(np.int64)
        return JointPolicyBatchV41(indices, decode_joint_actions_v4_1(indices),
                                  distribution.log_prob(action).cpu().numpy(), float(value.cpu()))
