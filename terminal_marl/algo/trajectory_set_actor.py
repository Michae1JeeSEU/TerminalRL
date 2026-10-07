"""Experimental V2.5 set actor. No critic, executor, or PPO admission."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from ..core.trajectory_context_observation import VERSION


class SetEncoder(nn.Module):
    def __init__(self, hidden=96, neighbor_hidden=32):
        super().__init__()
        self.neighbor = nn.Sequential(nn.Linear(50, neighbor_hidden), nn.Tanh(),
                                      nn.Linear(neighbor_hidden, neighbor_hidden), nn.Tanh())
        self.combine = nn.Sequential(nn.Linear(105 + 2 * neighbor_hidden + 1, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, ego, traffic, mask):
        # Zero before the linear layer: even NaN padding cannot contaminate
        # either the forward pass or weight gradients.
        valid = mask.unsqueeze(-1)
        encoded = self.neighbor(traffic.masked_fill(~valid, 0.))
        count = mask.sum(1, keepdim=True)
        mean = encoded.masked_fill(~valid, 0.).sum(1) / count.clamp_min(1)
        if traffic.shape[1]:
            maximum = encoded.masked_fill(~valid, -torch.inf).max(1).values
            maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        else:
            maximum = mean
        return self.combine(torch.cat((ego, mean, maximum, torch.log1p(count.to(ego.dtype))), -1))


class TrajectorySetActor(nn.Module):
    """Independent gate/proposal encoders, invariant to neighbor order.

    Both arms of the information comparison instantiate this exact network.
    Numeric tensors only: no UID, teacher certificate or execution outcome.
    """
    observation_schema = VERSION

    def __init__(self, hidden=96, neighbor_hidden=32):
        super().__init__()
        self.gate_encoder = SetEncoder(hidden, neighbor_hidden)
        self.proposal_encoder = SetEncoder(hidden, neighbor_hidden)
        self.gate_head = nn.Linear(hidden, 2)
        self.proposal_head = nn.Linear(hidden, 25)

    def forward(self, ego, traffic, mask):
        if (ego.ndim != 2 or ego.shape[1] != 105 or traffic.ndim != 3
                or traffic.shape[0] != len(ego) or traffic.shape[2] != 50
                or mask.shape != traffic.shape[:2] or mask.dtype != torch.bool):
            raise ValueError('expected ego[B,105], traffic[B,N,50], boolean mask[B,N]')
        return (self.gate_head(self.gate_encoder(ego, traffic, mask)),
                self.proposal_head(self.proposal_encoder(ego, traffic, mask)))

    def distributions(self, ego, traffic, mask):
        gate, proposal = self(ego, traffic, mask)
        return Categorical(logits=gate), Categorical(logits=proposal)
