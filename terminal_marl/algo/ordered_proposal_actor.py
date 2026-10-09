"""Train-only candidate-conditioned first-feasible factorization.

The output is a requested-action distribution, never a safety certificate.
No candidate planner, teacher, or runtime environment is called here.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from .trajectory_set_actor import SetEncoder
from ..core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from ..core.resource_context_observation import VERSION


RELEASE_ORDER = tuple(row.index-1 for row in sorted(
    JOINT_ACTION_CATALOG_V4_1[1:],
    key=lambda row: (abs(row.speed_residual), abs(row.merge_time_residual), row.index)))


def first_feasible_logits(candidate_logits):
    """25 catalogue-indexed Bernoulli scores -> gate and conditional proposal.

In preference order, the mass of candidate k is h[k] * product(1-h[:k]);
WAIT receives product(1-h). Log-space computation stays finite at extremes.
These are model scores, not independent physical-feasibility probabilities.
"""
    if candidate_logits.ndim != 2 or candidate_logits.shape[1] != 25:
        raise ValueError('expected candidate scores [B,25]')
    order=torch.as_tensor(RELEASE_ORDER,device=candidate_logits.device)
    ordered=candidate_logits[:,order]
    rejected=F.logsigmoid(-ordered)
    prefix=torch.cat((torch.zeros_like(rejected[:,:1]),rejected[:,:-1].cumsum(-1)),-1)
    ordered_joint=F.logsigmoid(ordered)+prefix
    joint=ordered_joint[:,torch.argsort(order)]
    gate=torch.stack((rejected.sum(-1),torch.logsumexp(joint,-1)),-1)
    # Categorical/CE normalizes these masses conditional on RELEASE.
    return gate,joint


class OrderedProposalActor(nn.Module):
    """One state encoder and a shared scorer for the 25 numeric proposals."""
    observation_schema=VERSION
    architecture='ordered_candidate_set_v1'
    ego_dim=159

    def __init__(self,hidden=96,neighbor_hidden=32):
        super().__init__()
        self.encoder=SetEncoder(hidden,neighbor_hidden,ego_dim=self.ego_dim)
        self.score=nn.Sequential(nn.Linear(hidden+2,64),nn.Tanh(),nn.Linear(64,1))
        proposals=[(row.speed_residual,row.merge_time_residual) for row in JOINT_ACTION_CATALOG_V4_1[1:]]
        self.register_buffer('proposals',torch.tensor(proposals,dtype=torch.float32))
        # Keep initial WAIT and RELEASE mass away from saturated extremes.
        nn.init.constant_(self.score[-1].bias,math.log(.03/.97))

    def candidate_logits(self,ego,traffic,mask):
        if (ego.ndim!=2 or ego.shape[1]!=self.ego_dim or traffic.ndim!=3
                or traffic.shape[0]!=len(ego) or traffic.shape[2]!=50
                or mask.shape!=traffic.shape[:2] or mask.dtype!=torch.bool):
            raise ValueError('expected ego[B,159], traffic[B,N,50], boolean mask[B,N]')
        state=self.encoder(ego,traffic,mask)
        state=state[:,None,:].expand(-1,25,-1)
        candidates=self.proposals[None,:,:].expand(len(ego),-1,-1)
        return self.score(torch.cat((state,candidates),-1)).squeeze(-1)

    def forward(self,ego,traffic,mask):
        return first_feasible_logits(self.candidate_logits(ego,traffic,mask))

    def distributions(self,ego,traffic,mask):
        gate,proposal=self(ego,traffic,mask)
        return Categorical(logits=gate),Categorical(logits=proposal)
