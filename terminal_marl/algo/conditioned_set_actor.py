"""Development actor: condition neighbor features before set aggregation."""
from __future__ import annotations

import torch
from torch import nn
from .trajectory_set_actor import TrajectorySetActor
from ..core.resource_context_observation import VERSION


class ConditionedSetEncoder(nn.Module):
    def __init__(self,hidden=96,neighbor_hidden=32,ego_dim=159):
        super().__init__()
        self.neighbor=nn.Sequential(nn.Linear(ego_dim+50,neighbor_hidden),nn.Tanh(),
                                    nn.Linear(neighbor_hidden,neighbor_hidden),nn.Tanh())
        self.combine=nn.Sequential(nn.Linear(ego_dim+2*neighbor_hidden+1,hidden),nn.Tanh(),
                                   nn.Linear(hidden,hidden),nn.Tanh())

    def forward(self,ego,traffic,mask):
        valid=mask.unsqueeze(-1)
        pairs=torch.cat((ego[:,None,:].expand(-1,traffic.shape[1],-1),traffic),-1)
        encoded=self.neighbor(pairs.masked_fill(~valid,0.))
        count=mask.sum(1,keepdim=True)
        mean=encoded.masked_fill(~valid,0.).sum(1)/count.clamp_min(1)
        if traffic.shape[1]:
            maximum=encoded.masked_fill(~valid,-torch.inf).max(1).values
            maximum=torch.where(count>0,maximum,torch.zeros_like(maximum))
        else:maximum=mean
        return self.combine(torch.cat((ego,mean,maximum,torch.log1p(count.to(ego.dtype))),-1))


class ConditionedSetActor(TrajectorySetActor):
    observation_schema=VERSION
    architecture='ego_conditioned_neighbor_set_v1'

    def __init__(self,hidden=96,neighbor_hidden=32):
        nn.Module.__init__(self)
        self.ego_dim=159
        self.gate_encoder=ConditionedSetEncoder(hidden,neighbor_hidden)
        self.proposal_encoder=ConditionedSetEncoder(hidden,neighbor_hidden)
        self.gate_head=nn.Linear(hidden,2)
        self.proposal_head=nn.Linear(hidden,25)
