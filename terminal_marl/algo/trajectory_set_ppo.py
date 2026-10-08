"""V2.5 PPO building blocks; construction never grants training admission.

Keep the qualified gate/proposal representation frozen. A zero-initialized
gate residual and an independent centralized critic are the only trainable
parameters. Every simulator tick, including empty arrival/service intervals,
has one team reward/value; only pre-action eligible actors enter the policy
objective. No environment or teacher is queried by the optimizer.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence

from .trajectory_set_actor import TrajectorySetActor, SetEncoder
from .joint_ppo_v4_2 import hierarchical_joint_logprob
from .student_ppo_v4_2 import team_gae, _state_finite

CENTRAL_CONTEXT_DIM = 26


class TrajectorySetActorCritic(nn.Module):
    def __init__(self, actor: TrajectorySetActor):
        super().__init__()
        self.actor = copy.deepcopy(actor).requires_grad_(False)
        self.gate_adapter = nn.Linear(96, 2)
        nn.init.zeros_(self.gate_adapter.weight)
        nn.init.zeros_(self.gate_adapter.bias)
        self.critic_encoder = SetEncoder()
        self.critic_head = nn.Sequential(nn.Linear(2*96+1+CENTRAL_CONTEXT_DIM,96), nn.Tanh(), nn.Linear(96,1))

    def distributions(self, ego, traffic, mask):
        with torch.no_grad():
            gate, proposal = self.actor(ego, traffic, mask)
            encoded = self.actor.gate_encoder(ego, traffic, mask)
        return Categorical(logits=gate+self.gate_adapter(encoded)), Categorical(logits=proposal)

    def value(self, ego, traffic, mask, context):
        if context.shape != (CENTRAL_CONTEXT_DIM,):
            raise ValueError('central context must have 26 features')
        if len(ego):
            encoded = self.critic_encoder(ego, traffic, mask)
            pooled = torch.cat((encoded.mean(0), encoded.max(0).values))
        else:
            pooled = context.new_zeros(192)
        count = context.new_tensor([np.log1p(len(ego))])
        return self.critic_head(torch.cat((pooled,count,context))).squeeze(-1)


@dataclass(frozen=True)
class SetStep:
    ego: np.ndarray
    traffic: np.ndarray
    traffic_mask: np.ndarray
    context: np.ndarray
    requested_gate: np.ndarray
    requested_proposal: np.ndarray
    gate_logprob: np.ndarray
    proposal_logprob: np.ndarray
    old_gate_logits: np.ndarray
    old_proposal_logits: np.ndarray
    actor_mask: np.ndarray
    value: float
    next_value: float
    reward: float
    terminated: bool
    episode_end: bool
    teacher_used: bool = False
    stochastic_behavior: bool = True

    @property
    def requested_logprob(self):
        return self.gate_logprob+np.where(self.requested_gate==1,self.proposal_logprob,0.)

    def validate(self):
        n=len(self.ego)
        if (self.ego.shape!=(n,105) or self.traffic.ndim!=3
            or self.traffic.shape[0]!=n or self.traffic.shape[2]!=50
            or self.traffic_mask.shape!=self.traffic.shape[:2]
            or self.traffic_mask.dtype!=np.bool_ or self.context.shape!=(CENTRAL_CONTEXT_DIM,)
            or self.old_gate_logits.shape!=(n,2) or self.old_proposal_logits.shape!=(n,25)):
            raise ValueError('invalid V2.5 rollout shapes')
        fields=(self.requested_gate,self.requested_proposal,self.gate_logprob,self.proposal_logprob,self.actor_mask)
        if any(np.shape(a)!=(n,) for a in fields) or self.actor_mask.dtype!=np.bool_:
            raise ValueError('unaligned actions or actor mask')
        release=self.requested_gate==1
        if (not np.isin(self.requested_gate,(0,1)).all()
            or not np.issubdtype(self.requested_proposal.dtype,np.integer)
            or np.any(self.requested_proposal[~release]!=-1)
            or np.any((self.requested_proposal[release]<0)|(self.requested_proposal[release]>=25))
            or not np.isnan(self.proposal_logprob[~release]).all()):
            raise ValueError('invalid conditional action')
        if np.any(release & ~self.actor_mask) or np.any(self.gate_logprob[~self.actor_mask]!=0):
            raise ValueError('forced rows require canonical WAIT likelihood')
        finite=(self.ego,self.traffic[self.traffic_mask],self.context,self.old_gate_logits,self.old_proposal_logits,
                self.gate_logprob,self.proposal_logprob[release],
                [self.value,self.next_value,self.reward])
        if not all(np.isfinite(a).all() for a in finite):
            raise ValueError('nonfinite rollout')
        if self.teacher_used or not self.stochastic_behavior:
            raise ValueError('PPO requires stochastic student behavior')
        if self.terminated and (not self.episode_end or self.next_value!=0):
            raise ValueError('terminal transition cannot bootstrap')
        if n:
            gate,proposal,joint=hierarchical_joint_logprob(
                Categorical(logits=torch.from_numpy(self.old_gate_logits)),
                Categorical(logits=torch.from_numpy(self.old_proposal_logits)),
                torch.as_tensor(self.requested_gate),torch.as_tensor(self.requested_proposal))
            if (not np.allclose(gate.numpy()[self.actor_mask],self.gate_logprob[self.actor_mask],atol=1e-6,rtol=0)
                or not np.allclose(proposal.numpy()[release],self.proposal_logprob[release],atol=1e-6,rtol=0)):
                raise ValueError('stored requested likelihood differs from behavior distribution')


def tensors(row):
    return tuple(torch.as_tensor(a) for a in (row.ego,row.traffic,row.traffic_mask))


def actor_batch(rows):
    active=[r for r in rows if r.actor_mask.any()]
    if not active:
        return None
    n=sum(int(r.actor_mask.sum()) for r in active)
    neighbors=max(r.traffic.shape[1] for r in active)
    traffic=np.zeros((n,neighbors,50),np.float32)
    mask=np.zeros((n,neighbors),bool)
    offset=0
    for r in active:
        count=int(r.actor_mask.sum()); width=r.traffic.shape[1]
        traffic[offset:offset+count,:width]=r.traffic[r.actor_mask]
        mask[offset:offset+count,:width]=r.traffic_mask[r.actor_mask]
        offset+=count
    def joined(name):
        return torch.as_tensor(np.concatenate([getattr(r,name)[r.actor_mask] for r in active]))
    return (joined('ego'),torch.from_numpy(traffic),torch.from_numpy(mask),
            joined('requested_gate'),joined('requested_proposal'),joined('requested_logprob'),
            joined('old_gate_logits'))


class TrajectorySetPPO:
    def __init__(self, actor, *, learning_rate=3e-5, reference=None,
                 reference_limit=.02, reference_coefficient=1.):
        self.model=TrajectorySetActorCritic(actor)
        self.optimizer=torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],lr=learning_rate)
        if not np.isfinite(reference_limit) or reference_limit<=0 or not np.isfinite(reference_coefficient) or reference_coefficient<0:
            raise ValueError('invalid fixed-reference KL settings')
        self.reference=None if reference is None else tuple(torch.as_tensor(a).detach().clone() for a in reference)
        if self.reference is not None and not len(self.reference[0]):
            raise ValueError('reference states cannot be empty')
        self.reference_limit=float(reference_limit)
        self.reference_coefficient=float(reference_coefficient)

    def reference_kl(self):
        if self.reference is None:
            return self.model.gate_adapter.weight.new_zeros(())
        with torch.no_grad():
            base,_=self.model.actor.distributions(*self.reference)
        current,_=self.model.distributions(*self.reference)
        return kl_divergence(base,current).mean()

    @torch.no_grad()
    def rollout_gate_kl(self, steps):
        values=[]
        # Bound memory by simulator states; exact categorical gate KL,
        # including unchosen gates, remains meaningful for saturated actors.
        for row in steps:
            batch=actor_batch([row])
            if batch is None: continue
            current,_=self.model.distributions(*batch[:3])
            values.append(kl_divergence(Categorical(logits=batch[6]),current).mean())
        return float(torch.stack(values).mean()) if values else 0.

    def update(self, steps, *, training_authorized=False, epochs=4, minibatch_size=32,
               gamma=.99, gae_lambda=.95, clip_ratio=.2, value_coef=.5,
               entropy_coef=.01, target_kl=.01, wall_seconds=600):
        if not training_authorized:
            raise PermissionError('V2.5 PPO requires evidenced student qualification and admission')
        if (not steps or epochs<1 or minibatch_size<1 or not 0<=gamma<=1
            or not 0<=gae_lambda<=1 or not 0<clip_ratio<1
            or not np.isfinite(target_kl) or target_kl<=0
            or not np.isfinite(wall_seconds) or wall_seconds<=0):
            raise ValueError('invalid PPO budget or hyperparameters')
        started=time.monotonic()
        for r in steps: r.validate()
        with torch.no_grad():
            for r in steps:
                batch=actor_batch([r])
                if batch is None: continue
                gd,pd=self.model.distributions(*batch[:3])
                _,_,lp=hierarchical_joint_logprob(gd,pd,batch[3],batch[4])
                if (not torch.allclose(lp,batch[5],atol=1e-5,rtol=0)
                    or not torch.allclose(pd.logits,torch.from_numpy(r.old_proposal_logits[r.actor_mask]),atol=1e-5,rtol=0)):
                    raise ValueError('requested behavior does not match the initial PPO policy')
        advantages,returns=team_gae(steps,gamma,gae_lambda)
        actor_steps=np.asarray([r.actor_mask.any() for r in steps])
        count=sum(int(r.actor_mask.sum()) for r in steps)
        normalizer=count/len(steps)
        if actor_steps.any():
            selected=advantages[actor_steps]
            advantages=(advantages-selected.mean())/(selected.std()+1e-8)
        totals=dict(accepted_updates=0,rejected_updates=0,team_steps=len(steps),actor_decisions=count,
                    actor_count_normalizer=normalizer,final_gate_kl=self.rollout_gate_kl(steps),
                    final_reference_kl=float(self.reference_kl().detach()),rejection_reason=None,budget_exhausted=False)
        metrics={k:0. for k in ('loss','actor_loss','value_loss','entropy','grad_norm','clip_fraction')}
        def finished():
            return {**totals,'elapsed_seconds':time.monotonic()-started,
                    **{k:v/max(1,totals['accepted_updates']) for k,v in metrics.items()}}
        if totals['final_gate_kl']>target_kl+1e-8:
            raise ValueError('rollout is stale before PPO')
        if not np.isfinite(totals['final_reference_kl']) or totals['final_reference_kl']>self.reference_limit:
            raise ValueError('policy already exceeds fixed-reference KL budget')
        for _ in range(epochs):
            order=np.random.permutation(len(steps))
            for start in range(0,len(steps),minibatch_size):
                if time.monotonic()-started>=wall_seconds:
                    totals['budget_exhausted']=True
                    return finished()
                indices=order[start:start+minibatch_size];rows=[steps[i] for i in indices]
                predicted=torch.stack([self.model.value(*tensors(r),torch.from_numpy(r.context)) for r in rows])
                value_loss=(predicted-torch.as_tensor(returns[indices])).square().mean()
                actor=entropy=clipped=value_loss.new_zeros(())
                batch=actor_batch(rows)
                if batch is not None:
                    gd,pd=self.model.distributions(*batch[:3])
                    _,_,current=hierarchical_joint_logprob(gd,pd,batch[3],batch[4])
                    ratio=(current-batch[5]).exp()
                    counts=[int(r.actor_mask.sum()) for r in rows]
                    advantage=torch.as_tensor(np.repeat(advantages[indices],counts))
                    objective=-torch.minimum(ratio*advantage,ratio.clamp(1-clip_ratio,1+clip_ratio)*advantage)
                    actor=objective.sum()/(len(rows)*normalizer)
                    entropy=torch.stack([v.mean() for v in torch.split(gd.entropy(),[c for c in counts if c])]).mean()
                    clipped=((ratio-1).abs()>clip_ratio).float().mean()
                loss=actor+value_coef*value_loss-entropy_coef*entropy+self.reference_coefficient*self.reference_kl()
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite PPO loss')
                state=copy.deepcopy(self.model.state_dict());optim=copy.deepcopy(self.optimizer.state_dict())
                self.optimizer.zero_grad()
                rejected=False
                try:
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                    self.optimizer.step()
                    kl=self.rollout_gate_kl(steps)
                    ref_kl=float(self.reference_kl().detach())
                    rejected=(not _state_finite(self.model.state_dict())
                              or not _state_finite(self.optimizer.state_dict())
                              or not np.isfinite(kl) or kl>target_kl
                              or not np.isfinite(ref_kl) or ref_kl>self.reference_limit)
                    if rejected:
                        totals['rejection_reason']='reference_kl' if not np.isfinite(ref_kl) or ref_kl>self.reference_limit else 'rollout_kl_or_nonfinite'
                except (RuntimeError,FloatingPointError):
                    rejected=True
                    totals['rejection_reason']='optimizer_error'
                if rejected:
                    self.model.load_state_dict(state);self.optimizer.load_state_dict(optim)
                    self.optimizer.zero_grad(set_to_none=True)
                    totals['rejected_updates']+=1
                    return finished()
                totals['accepted_updates']+=1;totals['final_gate_kl']=kl;totals['final_reference_kl']=ref_kl
                for name,value in (('loss',loss),('actor_loss',actor),('value_loss',value_loss),
                                   ('entropy',entropy),('grad_norm',norm),('clip_fraction',clipped)):
                    metrics[name]+=float(value.detach())
        return finished()
