"""PPO for the canonical 10-D macro-level dynamic-airspace action."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def _mlp(input_dim: int, output_dim: int, hidden: int = 96) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))


class HighActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int = 10) -> None:
        super().__init__()
        self.actor = _mlp(observation_dim, action_dim)
        self.critic = _mlp(observation_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.8))
        # A modest executable initial intent ensures all four non-structural
        # dimensions are exercised from rollout one; PPO remains unconstrained.
        with torch.no_grad(): self.actor[-1].bias.copy_(torch.tensor((.20,.12,.06,.04,.02,.01,.10,-.10,-.10,-.06)))
    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(torch.tanh(self.actor(obs)), self.log_std.exp().expand(obs.shape[0], -1))
    def value(self, obs: torch.Tensor) -> torch.Tensor: return self.critic(obs).squeeze(-1)

@dataclass
class HighTransition:
    observation: np.ndarray; action: np.ndarray; log_prob: float; value: float; reward: float; done: bool

class HighRolloutBuffer:
    def __init__(self) -> None: self.rows: list[HighTransition] = []
    def add(self, row: HighTransition) -> None: self.rows.append(row)
    def clear(self) -> None: self.rows.clear()
    def returns(self, gamma: float) -> tuple[np.ndarray, np.ndarray]:
        values=np.asarray([row.value for row in self.rows],np.float32); returns=np.zeros(len(self.rows),np.float32); running=0.
        for index in range(len(self.rows)-1,-1,-1):
            running=self.rows[index].reward+gamma*running*(1.-float(self.rows[index].done)); returns[index]=running
        return returns, returns-values

class HighLevelPPO:
    def __init__(self, observation_dim: int, *, device: str = "cpu", learning_rate: float = 3e-4, clip_ratio: float = .2, value_coef: float = .5, entropy_coef: float = .01) -> None:
        self.device=torch.device(device); self.model=HighActorCritic(observation_dim).to(self.device); self.optimizer=torch.optim.Adam(self.model.parameters(),lr=learning_rate)
        self.observation_dim=observation_dim; self.clip_ratio=clip_ratio; self.value_coef=value_coef; self.entropy_coef=entropy_coef
    def act(self, observation: np.ndarray, *, deterministic: bool = False) -> tuple[np.ndarray,float,float]:
        obs=torch.as_tensor(observation[None],dtype=torch.float32,device=self.device)
        with torch.no_grad():
            distribution=self.model.distribution(obs); action=distribution.mean if deterministic else distribution.sample(); action=action.clamp(-1.,1.); log_prob=distribution.log_prob(action).sum(-1); value=self.model.value(obs)
        return action[0].cpu().numpy().astype(np.float32),float(log_prob[0].cpu()),float(value[0].cpu())
    def update(self, buffer: HighRolloutBuffer, *, epochs: int = 6, gamma: float = .99) -> dict[str,float]:
        if not buffer.rows: return {"loss":0.,"actor_loss":0.,"value_loss":0.,"entropy":0.}
        returns,advantages=buffer.returns(gamma); advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        observations=torch.as_tensor(np.stack([row.observation for row in buffer.rows]),dtype=torch.float32,device=self.device); actions=torch.as_tensor(np.stack([row.action for row in buffer.rows]),dtype=torch.float32,device=self.device); old_log=torch.as_tensor([row.log_prob for row in buffer.rows],dtype=torch.float32,device=self.device); returns_t=torch.as_tensor(returns,dtype=torch.float32,device=self.device); adv_t=torch.as_tensor(advantages,dtype=torch.float32,device=self.device)
        totals={"loss":0.,"actor_loss":0.,"value_loss":0.,"entropy":0.}
        for _ in range(epochs):
            distribution=self.model.distribution(observations); ratio=(distribution.log_prob(actions).sum(-1)-old_log).exp(); actor=-torch.minimum(ratio*adv_t,ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*adv_t).mean(); value=(self.model.value(observations)-returns_t).pow(2).mean(); entropy=distribution.entropy().sum(-1).mean(); loss=actor+self.value_coef*value-self.entropy_coef*entropy
            if not torch.isfinite(loss): raise FloatingPointError("non-finite high-level PPO loss")
            self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),1.);self.optimizer.step()
            for key,item in (("loss",loss),("actor_loss",actor),("value_loss",value),("entropy",entropy)): totals[key]+=float(item.detach())
        buffer.clear(); return {key:value/epochs for key,value in totals.items()}
