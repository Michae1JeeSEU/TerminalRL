"""Factorized categorical PPO for Low-Level Action V3.

The actor learns scheduling requests, not vehicle controls. It samples three
independent categories (release, speed residual, merge-time residual); their
joint PPO density is the sum of the three categorical log-probabilities.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..core.low_level_action import HEAD_CARDINALITIES, indices_to_normalized, normalized_to_indices


def mlp(input_dim: int, output_dim: int, hidden: int = 96) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))


class FactorizedLowLevelCategorical:
    """Three categorical distributions with an explicit joint density."""

    def __init__(self, logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        self.logits = logits
        self.heads = tuple(Categorical(logits=value) for value in logits)

    def sample(self) -> torch.Tensor:
        return torch.stack([head.sample() for head in self.heads], dim=-1)

    def mode(self) -> torch.Tensor:
        return torch.stack([value.argmax(dim=-1) for value in self.logits], dim=-1)

    def log_prob_per_head(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[-1] != 3:
            raise ValueError("categorical PPO actions must have shape (batch, 3)")
        return torch.stack([head.log_prob(actions[:, index].long()) for index, head in enumerate(self.heads)], dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.log_prob_per_head(actions).sum(dim=-1)

    def entropy_per_head(self) -> torch.Tensor:
        return torch.stack([head.entropy() for head in self.heads], dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.entropy_per_head().sum(dim=-1)


class SharedActorCentralCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.actor_body = mlp(observation_dim, hidden, hidden)
        self.actor_heads = nn.ModuleList(nn.Linear(hidden, cardinality) for cardinality in HEAD_CARDINALITIES)
        # Safe initial scheduling prior: RELEASE, nominal speed, nominal time.
        for head in self.actor_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.actor_heads[0].bias.copy_(torch.tensor((0.0, 1.2)))
            self.actor_heads[1].bias[2] = 0.5
            self.actor_heads[2].bias[2] = 0.5
        self.critic_encoder = mlp(observation_dim, hidden, hidden)
        self.critic_head = mlp(2 * hidden, 1, hidden)

    def actor_distribution(self, observations: torch.Tensor) -> FactorizedLowLevelCategorical:
        body = self.actor_body(observations)
        return FactorizedLowLevelCategorical(tuple(head(body) for head in self.actor_heads))

    def critic(self, agent_sets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.critic_encoder(agent_sets)
        valid = mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (encoded * valid).sum(dim=1) / count
        maximum = encoded.masked_fill(~valid, float("-inf")).max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.critic_head(torch.cat((mean, maximum), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class PolicyActionBatch:
    categories: np.ndarray
    normalized: np.ndarray
    joint_log_prob: np.ndarray
    head_log_prob: np.ndarray
    value: float


@dataclass
class Transition:
    """PPO stores sampled V3 categories, never teacher/applied actions."""

    agent_id: str
    observation: np.ndarray
    world: np.ndarray
    action: np.ndarray  # requested categorical indices [release, speed, merge]
    log_prob: float  # original joint categorical log-probability
    value: float
    reward: float
    done: bool
    teacher_action: np.ndarray | None = None  # categorical teacher label
    applied_action: np.ndarray | None = None  # normalized, never PPO ratio
    teacher_used: bool = False
    shield_used: bool = False
    requested_normalized_action: np.ndarray | None = None
    old_head_log_prob: np.ndarray | None = None


class RolloutBuffer:
    def __init__(self) -> None:
        self.by_agent: dict[str, list[Transition]] = defaultdict(list)

    def add(self, transition: Transition) -> None:
        # Fail before a costly update if a caller silently supplies old
        # continuous actions instead of the V3 requested categorical sample.
        normalized_to_indices(indices_to_normalized(transition.action))
        self.by_agent[transition.agent_id].append(transition)

    def clear(self) -> None:
        self.by_agent.clear()

    def flattened(self, gamma: float, gae_lambda: float):
        rows = []
        for trace in self.by_agent.values():
            advantage = 0.0
            for index in range(len(trace) - 1, -1, -1):
                current = trace[index]
                next_value = 0.0 if current.done or index == len(trace) - 1 else trace[index + 1].value
                delta = current.reward + gamma * next_value - current.value
                advantage = delta + gamma * gae_lambda * (0.0 if current.done else advantage)
                rows.append((current, advantage, advantage + current.value))
        return list(reversed(rows))


class SharedPPO:
    def __init__(self, observation_dim: int, *, device: str = "cpu", learning_rate: float = 3e-4,
                 clip_ratio: float = .2, value_coef: float = .5, entropy_coef: float = .01,
                 teacher_bc_weight: float = 0.0) -> None:
        self.device = torch.device(device)
        self.model = SharedActorCentralCritic(observation_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clip_ratio, self.value_coef, self.entropy_coef, self.teacher_bc_weight = clip_ratio, value_coef, entropy_coef, teacher_bc_weight
        self.observation_dim = observation_dim

    def act_details(self, observations: np.ndarray, world: np.ndarray, *, deterministic: bool = False) -> PolicyActionBatch:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        world_tensor = torch.as_tensor(world[None], dtype=torch.float32, device=self.device)
        mask = torch.ones((1, world.shape[0]), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            distribution = self.model.actor_distribution(obs)
            categories = distribution.mode() if deterministic else distribution.sample()
            per_head = distribution.log_prob_per_head(categories)
            value = self.model.critic(world_tensor, mask)[0]
        category_array = categories.cpu().numpy().astype(np.int64)
        return PolicyActionBatch(category_array, indices_to_normalized(category_array),
                                 per_head.sum(dim=-1).cpu().numpy(), per_head.cpu().numpy(), float(value.cpu()))

    def act(self, observations: np.ndarray, world: np.ndarray, *, deterministic: bool = False):
        """Compatibility execution API: normalized values from the V3 catalogue."""
        batch = self.act_details(observations, world, deterministic=deterministic)
        return batch.normalized, batch.joint_log_prob, batch.value

    def update(self, buffer: RolloutBuffer, *, epochs: int = 4, minibatch_size: int = 128,
               gamma: float = .99, gae_lambda: float = .95) -> dict[str, float]:
        rows = buffer.flattened(gamma, gae_lambda)
        zero = {"loss": 0.0, "actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "teacher_bc_loss": 0.0,
                "entropy_release": 0.0, "entropy_speed": 0.0, "entropy_merge_time": 0.0, "approx_kl": 0.0,
                "clip_fraction": 0.0}
        if not rows:
            return zero
        advantages = np.asarray([row[1] for row in rows], np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        totals = defaultdict(float)
        updates = 0
        for _ in range(epochs):
            permutation = np.random.permutation(len(rows))
            for begin in range(0, len(rows), minibatch_size):
                choices = permutation[begin:begin + minibatch_size]
                selected = [rows[index] for index in choices]
                observations = torch.as_tensor(np.stack([row[0].observation for row in selected]), dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(np.stack([row[0].action for row in selected]), dtype=torch.long, device=self.device)
                old_log_prob = torch.as_tensor([row[0].log_prob for row in selected], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor([row[2] for row in selected], dtype=torch.float32, device=self.device)
                adv = torch.as_tensor(advantages[choices], dtype=torch.float32, device=self.device)
                worlds = [row[0].world for row in selected]
                max_n = max(len(world) for world in worlds)
                padded = np.zeros((len(worlds), max_n, self.observation_dim), np.float32)
                mask = np.zeros((len(worlds), max_n), bool)
                for index, world in enumerate(worlds):
                    padded[index, :len(world)] = world
                    mask[index, :len(world)] = True
                distribution = self.model.actor_distribution(observations)
                current_log_prob = distribution.log_prob(actions)
                ratio = (current_log_prob - old_log_prob).exp()
                clipped_ratio = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = -torch.minimum(ratio * adv, clipped_ratio * adv).mean()
                values = self.model.critic(torch.as_tensor(padded, device=self.device), torch.as_tensor(mask, device=self.device))
                value_loss = (values - returns).pow(2).mean()
                entropy_by_head = distribution.entropy_per_head()
                entropy = entropy_by_head.sum(-1).mean()
                teacher_rows = [row for row in selected if row[0].teacher_used and row[0].teacher_action is not None]
                if teacher_rows:
                    teacher_obs = torch.as_tensor(np.stack([row[0].observation for row in teacher_rows]), dtype=torch.float32, device=self.device)
                    teacher_targets = torch.as_tensor(np.stack([row[0].teacher_action for row in teacher_rows]), dtype=torch.long, device=self.device)
                    # Head-wise categorical CE on exactly the policy catalogue.
                    teacher_loss = -self.model.actor_distribution(teacher_obs).log_prob_per_head(teacher_targets).mean()
                else:
                    teacher_loss = torch.zeros((), device=self.device)
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy + self.teacher_bc_weight * teacher_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite categorical PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                log_ratio = current_log_prob - old_log_prob
                totals["loss"] += float(loss.detach()); totals["actor_loss"] += float(actor_loss.detach())
                totals["value_loss"] += float(value_loss.detach()); totals["entropy"] += float(entropy.detach())
                totals["teacher_bc_loss"] += float(teacher_loss.detach())
                totals["entropy_release"] += float(entropy_by_head[:, 0].mean().detach())
                totals["entropy_speed"] += float(entropy_by_head[:, 1].mean().detach())
                totals["entropy_merge_time"] += float(entropy_by_head[:, 2].mean().detach())
                totals["approx_kl"] += float((-log_ratio).mean().detach())
                totals["clip_fraction"] += float((torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().detach())
                updates += 1
        buffer.clear()
        return {key: value / max(1, updates) for key, value in (dict(zero) | totals).items()}
