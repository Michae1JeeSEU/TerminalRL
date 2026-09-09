"""Joint-categorical PPO for Low-Level Action V4.

This module intentionally has no compatibility path that can substitute a
teacher/execution/shield action in PPO's behaviour ratio.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..core.low_level_joint_action import N_JOINT_ACTIONS, decode_joint_actions
from .shared_ppo import mlp


class JointActorCentralCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.actor_body = mlp(observation_dim, hidden, hidden)
        self.joint_action_logits = nn.Linear(hidden, N_JOINT_ACTIONS)
        nn.init.zeros_(self.joint_action_logits.weight); nn.init.zeros_(self.joint_action_logits.bias)
        self.joint_action_logits.bias.data[5 + 2 * 5 + 2] = 0.5  # RELEASE, nominal speed/time
        self.critic_encoder = mlp(observation_dim, hidden, hidden)
        self.critic_head = mlp(2 * hidden, 1, hidden)

    def actor_distribution(self, observations: torch.Tensor) -> Categorical:
        return Categorical(logits=self.joint_action_logits(self.actor_body(observations)))

    def critic(self, agent_sets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.critic_encoder(agent_sets); valid = mask.unsqueeze(-1)
        mean = (encoded * valid).sum(1) / valid.sum(1).clamp_min(1)
        maximum = encoded.masked_fill(~valid, float("-inf")).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.critic_head(torch.cat((mean, maximum), -1)).squeeze(-1)


@dataclass(frozen=True)
class JointPolicyBatch:
    joint_action_index: np.ndarray
    decoded: np.ndarray
    joint_log_prob: np.ndarray
    value: float


@dataclass
class JointTransition:
    agent_id: str; observation: np.ndarray; world: np.ndarray; joint_action_index: int; log_prob: float; value: float; reward: float; done: bool
    teacher_joint_action: int | None = None
    execution_applied_action: np.ndarray | None = None
    shield_applied_action: np.ndarray | None = None
    teacher_used: bool = False


class JointRolloutBuffer:
    def __init__(self) -> None: self.by_agent: dict[str, list[JointTransition]] = defaultdict(list)
    def add(self, row: JointTransition) -> None:
        decode_joint_actions(np.asarray(row.joint_action_index)); self.by_agent[row.agent_id].append(row)
    def clear(self) -> None: self.by_agent.clear()
    def flattened(self, gamma: float, gae_lambda: float):
        rows=[]
        for trace in self.by_agent.values():
            advantage=0.0
            for i in range(len(trace)-1,-1,-1):
                row=trace[i]; next_value=0.0 if row.done or i==len(trace)-1 else trace[i+1].value
                delta=row.reward+gamma*next_value-row.value; advantage=delta+gamma*gae_lambda*(0.0 if row.done else advantage)
                rows.append((row,advantage,advantage+row.value))
        return list(reversed(rows))


class JointPPO:
    def __init__(self, observation_dim: int, *, device="cpu", learning_rate=.0003, clip_ratio=.2, value_coef=.5, entropy_coef=.01, teacher_bc_weight=.0) -> None:
        self.device=torch.device(device); self.model=JointActorCentralCritic(observation_dim).to(self.device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=learning_rate); self.observation_dim=observation_dim
        self.clip_ratio,self.value_coef,self.entropy_coef,self.teacher_bc_weight=clip_ratio,value_coef,entropy_coef,teacher_bc_weight

    def act_details(self, observations: np.ndarray, world: np.ndarray, *, deterministic=False) -> JointPolicyBatch:
        obs=torch.as_tensor(observations,dtype=torch.float32,device=self.device); mask=torch.ones((1,len(world)),dtype=torch.bool,device=self.device)
        with torch.no_grad():
            distribution=self.model.actor_distribution(obs); action=distribution.probs.argmax(-1) if deterministic else distribution.sample()
            value=self.model.critic(torch.as_tensor(world[None],dtype=torch.float32,device=self.device),mask)[0]
        indices=action.cpu().numpy().astype(np.int64)
        return JointPolicyBatch(indices,decode_joint_actions(indices),distribution.log_prob(action).cpu().numpy(),float(value.cpu()))

    def update(self, buffer: JointRolloutBuffer, *, epochs=4, minibatch_size=128, gamma=.99, gae_lambda=.95) -> dict[str,float]:
        rows=buffer.flattened(gamma,gae_lambda); zero={"loss":0.,"actor_loss":0.,"value_loss":0.,"entropy":0.,"joint_teacher_bc_loss":0.,"approx_kl":0.,"clip_fraction":0.,"explained_variance":0.,"grad_norm":0.}
        if not rows:return zero
        advantages=np.asarray([x[1] for x in rows],np.float32); advantages=(advantages-advantages.mean())/(advantages.std()+1e-8); totals=defaultdict(float); updates=0
        for _ in range(epochs):
            for choices in (np.random.permutation(len(rows))[start:start+minibatch_size] for start in range(0,len(rows),minibatch_size)):
                selected=[rows[i] for i in choices]; obs=torch.as_tensor(np.stack([x[0].observation for x in selected]),dtype=torch.float32,device=self.device)
                action=torch.as_tensor([x[0].joint_action_index for x in selected],dtype=torch.long,device=self.device); old=torch.as_tensor([x[0].log_prob for x in selected],dtype=torch.float32,device=self.device)
                returns=torch.as_tensor([x[2] for x in selected],dtype=torch.float32,device=self.device); adv=torch.as_tensor(advantages[choices],dtype=torch.float32,device=self.device)
                worlds=[x[0].world for x in selected]; n=max(len(x) for x in worlds); padded=np.zeros((len(worlds),n,self.observation_dim),np.float32); mask=np.zeros((len(worlds),n),bool)
                for i,world in enumerate(worlds): padded[i,:len(world)]=world; mask[i,:len(world)]=True
                distribution=self.model.actor_distribution(obs); current=distribution.log_prob(action); ratio=(current-old).exp(); actor=-torch.minimum(ratio*adv,ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*adv).mean()
                values=self.model.critic(torch.as_tensor(padded,device=self.device),torch.as_tensor(mask,device=self.device)); value_loss=(values-returns).pow(2).mean(); entropy=distribution.entropy().mean()
                teacher=[x[0] for x in selected if x[0].teacher_used and x[0].teacher_joint_action is not None]
                bc=torch.zeros((),device=self.device) if not teacher else -self.model.actor_distribution(torch.as_tensor(np.stack([x.observation for x in teacher]),dtype=torch.float32,device=self.device)).log_prob(torch.as_tensor([x.teacher_joint_action for x in teacher],device=self.device)).mean()
                loss=actor+self.value_coef*value_loss-self.entropy_coef*entropy+self.teacher_bc_weight*bc
                self.optimizer.zero_grad();loss.backward();gn=nn.utils.clip_grad_norm_(self.model.parameters(),1.0);self.optimizer.step()
                variance=returns.var(unbiased=False).clamp_min(1e-8); ev=1-(returns-values).var(unbiased=False)/variance; logratio=current-old
                for key,val in {"loss":loss,"actor_loss":actor,"value_loss":value_loss,"entropy":entropy,"joint_teacher_bc_loss":bc,"approx_kl":(-logratio).mean(),"clip_fraction":(torch.abs(ratio-1)>self.clip_ratio).float().mean(),"explained_variance":ev,"grad_norm":gn}.items(): totals[key]+=float(val.detach())
                updates+=1
        buffer.clear(); return {k:v/max(1,updates) for k,v in (zero|totals).items()}
