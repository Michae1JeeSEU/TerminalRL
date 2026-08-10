"""Shared 3-D low-level PPO with a masked mean/max set critic.

The actor consumes one UAV observation at a time; the critic receives the
variable-size active-UAV set.  No attention, GNN, or fixed max-N action exists.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim: int, output_dim: int, hidden: int = 96) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))


class SharedActorCentralCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int = 3, hidden: int = 96) -> None:
        super().__init__()
        self.actor_body = mlp(observation_dim, hidden, hidden)
        self.actor_mean = nn.Linear(hidden, action_dim)
        # Start from a safe executable intent (request release, nominal speed,
        # modestly earlier merge) rather than a random no-release controller.
        # PPO remains free to move every component within [-1, 1].
        nn.init.zeros_(self.actor_mean.weight)
        with torch.no_grad(): self.actor_mean.bias.copy_(torch.tensor((1.2, 0.0, -0.50)))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))
        self.critic_encoder = mlp(observation_dim, hidden, hidden)
        self.critic_head = mlp(2 * hidden, 1, hidden)

    def actor_distribution(self, observations: torch.Tensor) -> Normal:
        mean = torch.tanh(self.actor_mean(self.actor_body(observations)))
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def critic(self, agent_sets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.critic_encoder(agent_sets)
        valid = mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (encoded * valid).sum(dim=1) / count
        maximum = encoded.masked_fill(~valid, float("-inf")).max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.critic_head(torch.cat((mean, maximum), dim=-1)).squeeze(-1)


@dataclass
class Transition:
    agent_id: str
    observation: np.ndarray
    world: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    def __init__(self) -> None: self.by_agent: dict[str, list[Transition]] = defaultdict(list)
    def add(self, transition: Transition) -> None: self.by_agent[transition.agent_id].append(transition)
    def clear(self) -> None: self.by_agent.clear()
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
    def __init__(self, observation_dim: int, *, device: str = "cpu", learning_rate: float = 3e-4, clip_ratio: float = .2, value_coef: float = .5, entropy_coef: float = .01) -> None:
        self.device = torch.device(device)
        self.model = SharedActorCentralCritic(observation_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clip_ratio, self.value_coef, self.entropy_coef = clip_ratio, value_coef, entropy_coef
        self.observation_dim = observation_dim

    def act(self, observations: np.ndarray, world: np.ndarray, *, deterministic: bool = False):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        world_tensor = torch.as_tensor(world[None], dtype=torch.float32, device=self.device)
        mask = torch.ones((1, world.shape[0]), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            distribution = self.model.actor_distribution(obs)
            actions = distribution.mean if deterministic else distribution.sample()
            actions = actions.clamp(-1, 1)
            log_prob = distribution.log_prob(actions).sum(dim=-1)
            value = self.model.critic(world_tensor, mask)[0]
        return actions.cpu().numpy().astype(np.float32), log_prob.cpu().numpy(), float(value.cpu())

    def update(self, buffer: RolloutBuffer, *, epochs: int = 4, minibatch_size: int = 128, gamma: float = .99, gae_lambda: float = .95) -> dict[str, float]:
        rows = buffer.flattened(gamma, gae_lambda)
        if not rows: return {"loss": 0.0, "actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        advantages = np.asarray([x[1] for x in rows], np.float32); advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        totals = defaultdict(float); updates = 0
        for _ in range(epochs):
            permutation = np.random.permutation(len(rows))
            for begin in range(0, len(rows), minibatch_size):
                choices = permutation[begin:begin + minibatch_size]
                selected = [rows[i] for i in choices]
                observations = torch.as_tensor(np.stack([x[0].observation for x in selected]), dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(np.stack([x[0].action for x in selected]), dtype=torch.float32, device=self.device)
                old_log_prob = torch.as_tensor([x[0].log_prob for x in selected], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor([x[2] for x in selected], dtype=torch.float32, device=self.device)
                adv = torch.as_tensor(advantages[choices], dtype=torch.float32, device=self.device)
                worlds = [x[0].world for x in selected]; max_n = max(len(x) for x in worlds)
                padded = np.zeros((len(worlds), max_n, self.observation_dim), np.float32); mask = np.zeros((len(worlds), max_n), bool)
                for i, world in enumerate(worlds): padded[i, :len(world)] = world; mask[i, :len(world)] = True
                distribution = self.model.actor_distribution(observations)
                ratio = (distribution.log_prob(actions).sum(-1) - old_log_prob).exp()
                actor_loss = -torch.minimum(ratio * adv, ratio.clamp(1-self.clip_ratio, 1+self.clip_ratio) * adv).mean()
                values = self.model.critic(torch.as_tensor(padded, device=self.device), torch.as_tensor(mask, device=self.device))
                value_loss = (values - returns).pow(2).mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                if not torch.isfinite(loss): raise FloatingPointError("non-finite PPO loss")
                self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()
                totals["loss"] += float(loss.detach()); totals["actor_loss"] += float(actor_loss.detach()); totals["value_loss"] += float(value_loss.detach()); totals["entropy"] += float(entropy.detach()); updates += 1
        buffer.clear()
        return {key: value / max(1, updates) for key, value in totals.items()}
