"""Development actor retaining intent/destination/neighbor identity until scoring.

The pair scores are learned rejection predictions, never safety certificates.
Only existing V2.6 numeric reports enter forward; offline witness labels, UIDs,
native plan generation and the environment are not accessible to this module.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .trajectory_set_actor import SetEncoder
from .ordered_proposal_actor import first_feasible_logits
from ..core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from ..core.resource_context_observation import VERSION


class IntentNeighborActor(nn.Module):
    observation_schema=VERSION
    architecture='intent_destination_neighbor_risk_v1'
    gate_modes=('first_feasible_union','max_candidate')
    ego_dim=159
    # Bounded receiving interface. Overflow is explicit, never silent nearest-N
    # truncation which could discard a distant future blocker.
    max_neighbors=32

    def __init__(self,hidden=32,gate_mode='first_feasible_union'):
        if gate_mode not in self.gate_modes:
            raise ValueError("gate_mode must be 'first_feasible_union' or 'max_candidate'")
        super().__init__()
        self.gate_mode=gate_mode
        self.context=SetEncoder(hidden,16,ego_dim=159)
        self.ego_pair=nn.Linear(159+hidden,hidden)
        self.traffic_pair=nn.Linear(50,hidden,bias=False)
        self.intent=nn.Linear(2,hidden,bias=False)
        self.destination=nn.Linear(9,hidden,bias=False)
        self.pair_head=nn.Sequential(nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.scene=nn.Sequential(nn.Linear(159+hidden,hidden),nn.Tanh())
        self.scene_head=nn.Sequential(nn.Tanh(),nn.Linear(hidden,1))
        proposals=[(r.speed_residual,r.merge_time_residual) for r in JOINT_ACTION_CATALOG_V4_1[1:]]
        self.register_buffer('proposals',torch.tensor(proposals,dtype=torch.float32))
        nn.init.constant_(self.pair_head[-1].bias,2.)
        nn.init.constant_(self.scene_head[-1].bias,2.)

    @property
    def architecture_variant(self):
        return f'{self.architecture}__gate_{self.gate_mode}'

    def validate(self,ego,traffic,mask):
        if (ego.ndim!=2 or ego.shape[1]!=159 or traffic.ndim!=3 or traffic.shape[0]!=len(ego)
                or traffic.shape[2]!=50 or mask.shape!=traffic.shape[:2] or mask.dtype!=torch.bool
                or traffic.shape[1]>self.max_neighbors):
            raise ValueError('expected bounded V2.6 ego[B,159], traffic[B,N,50], boolean mask')

    def bases(self,ego,traffic,mask):
        self.validate(ego,traffic,mask)
        context=self.context(ego,traffic,mask)
        joined=torch.cat((ego,context),-1)
        pair=self.ego_pair(joined)[:,None,:]+self.traffic_pair(traffic.masked_fill(~mask[...,None],0.))
        scene=self.scene(joined)
        pads=ego[:,105:].reshape(-1,6,9)
        is_final=ego[:,10]>.5  # explicit inner-ring one-hot, not hidden state
        pads=torch.where(is_final[:,None,None],pads,torch.zeros_like(pads))
        eligible=(pads[:,:,3]>.5)&(pads[:,:,6]>.5)
        ring_destination=torch.zeros_like(eligible);ring_destination[:,0]=True
        eligible=torch.where(is_final[:,None],eligible,ring_destination)
        return pair,scene,pads,eligible

    def queried_scores(self,ego,traffic,mask,proposal,pad):
        """Auxiliary predictions for one known intent/destination per row.

        Inputs are still current reports; proposal/pad specify the query whose
        offline targets the caller compares against. No targets enter forward.
        """
        pair,scene,pads,_=self.bases(ego,traffic,mask)
        if (proposal.shape!=(len(ego),) or pad.shape!=(len(ego),)
                or bool(((proposal<0)|(proposal>=25)|(pad<0)|(pad>=6)).any())):
            raise ValueError('invalid intent/destination indices')
        index=torch.arange(len(ego),device=ego.device)
        query=self.intent(self.proposals[proposal])+self.destination(pads[index,pad])
        output=self.pair_head(pair+query[:,None,:]).squeeze(-1)
        return output,self.scene_head(scene+query).squeeze(-1)

    def relations(self,ego,traffic,mask):
        pair,scene,pads,eligible=self.bases(ego,traffic,mask)
        query=self.intent(self.proposals)[None,:,None,:]+self.destination(pads)[:,None,:,:]
        output=self.pair_head(pair[:,None,None,:,:]+query[:,:,:,None,:]).squeeze(-1)
        risk=output.masked_fill(~mask[:,None,None,:],-torch.inf)
        if traffic.shape[1]:
            maximum,blocker=risk.max(-1)
            blocker=torch.where(mask.any(-1)[:,None,None],blocker,torch.full_like(blocker,-1))
        else:
            maximum=torch.full(query.shape[:3],-torch.inf,device=ego.device,dtype=ego.dtype)
            blocker=torch.full(query.shape[:3],-1,device=ego.device,dtype=torch.long)
        scene_risk=self.scene_head(scene[:,None,None,:]+query).squeeze(-1)
        destination_risk=torch.maximum(maximum,scene_risk)
        destination_risk=destination_risk.masked_fill(~eligible[:,None,:],30.)
        best_risk,best_pad=destination_risk.min(-1)
        return dict(candidate_logits=-best_risk,pair_logits=risk,
            scene_logits=scene_risk,blocker_index=blocker,best_pad=best_pad,eligible_destinations=eligible)

    def forward(self,ego,traffic,mask):
        candidate_logits=self.relations(ego,traffic,mask)['candidate_logits']
        if self.gate_mode=='first_feasible_union':
            return first_feasible_logits(candidate_logits)
        _,proposal=first_feasible_logits(candidate_logits)
        best=candidate_logits.max(-1).values
        gate=torch.stack((torch.zeros_like(best),best),-1)
        return gate,proposal

    def distributions(self,ego,traffic,mask):
        gate,proposal=self(ego,traffic,mask)
        return Categorical(logits=gate),Categorical(logits=proposal)
