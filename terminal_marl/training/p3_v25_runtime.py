"""Explicit V2.5 requested-action bridge; all execution stays in the scheduler."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..algo.trajectory_set_actor import TrajectorySetActor
from ..algo.joint_ppo_v4_2 import hierarchical_joint_logprob
from ..core.low_level_joint_action_v4_2 import HierarchicalActionV42
from ..core.trajectory_context_observation import VERSION, pad_trajectory_context
from .student_rollout_v4_2 import validate_student_config


def actor_eligible(env, uav):
    interface = {'outer':'o2m', 'middle':'m2i', 'inner':'i2f'}.get(getattr(uav.ring_role, 'value', None))
    return bool(uav.committed_plan is None and uav.phase.value in {'ring_outer','ring_middle','ring_inner'}
                and uav.uid not in env.future_release_reservations and interface is not None
                and uav.soc > 0 and env.now >= uav.safety_retry_after
                and interface not in env.airspace.affected_interfaces())


def load_actor(path):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    from ..core.resource_context_observation import VERSION as RESOURCE_VERSION
    dimensions={VERSION:105,RESOURCE_VERSION:159}
    if payload['observation_schema'] not in dimensions or payload['arm'] != 'new_information':
        raise ValueError('set runtime requires an explicit supported new-information checkpoint')
    with torch.random.fork_rng(devices=[]):
        model = TrajectorySetActor(ego_dim=dimensions[payload['observation_schema']])
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, payload


def model_observations(model,env):
    """Select the observation contract by the actual actor, never by tensor padding."""
    actor=getattr(model,'actor',model)
    if actor.observation_schema==VERSION:
        return env.trajectory_context_observations()
    from ..core.resource_context_observation import VERSION as RESOURCE_VERSION,ResourceContextCache
    if actor.observation_schema!=RESOURCE_VERSION:
        raise ValueError('unsupported actor observation schema')
    cache=ResourceContextCache(env)
    return {str(uid):cache.build(env.active[uid]) for uid in sorted(env.active)}


@dataclass
class SetRequest:
    uids: tuple[str, ...]
    tensors: tuple
    actor_mask: np.ndarray
    gates: np.ndarray
    proposals: np.ndarray
    actions: dict
    shortlists: dict
    likelihoods: dict
    logits: tuple

    def tape(self):
        return dict(actions={k:v.tolist() for k,v in self.actions.items()},
                    shortlists={k:[v.tolist() for v in rows] for k,rows in self.shortlists.items()})


@torch.no_grad()
def request(model, env, *, deterministic=False):
    validate_student_config(env.config)
    if env.config.student_decision_mask_mode != 'pre_action_forced_wait' or env.config.runtime_proposal_shortlist_k != 3:
        raise ValueError('V2.5 bridge requires pre-action mask and frozen K3')
    observations = model_observations(model,env)
    uids = tuple(sorted(observations, key=int))
    if not uids:
        return SetRequest((), (), np.zeros(0,dtype=bool), np.zeros(0,dtype=int),
                          np.zeros(0,dtype=int), {}, {}, {}, ())
    tensors = tuple(torch.from_numpy(a) for a in pad_trajectory_context([observations[u] for u in uids]))
    gd, pd = model.distributions(*tensors)
    actor = torch.tensor([actor_eligible(env,env.active[int(uid)]) for uid in uids], dtype=torch.bool)
    gates = torch.zeros(len(uids),dtype=torch.long)
    proposals = torch.full((len(uids),),-1,dtype=torch.long)
    if actor.any():
        distribution = torch.distributions.Categorical(logits=gd.logits[actor])
        gates[actor] = distribution.logits.argmax(-1) if deterministic else distribution.sample()
    release = gates == 1
    if release.any():
        distribution = torch.distributions.Categorical(logits=pd.logits[release])
        proposals[release] = distribution.logits.argmax(-1) if deterministic else distribution.sample()
    gl, pl, jl = hierarchical_joint_logprob(gd,pd,gates,proposals)
    gl = torch.where(actor,gl,torch.zeros_like(gl))
    jl = torch.where(actor,jl,torch.zeros_like(jl))
    actions, shortlists, likelihoods = {}, {}, {}
    for i,uid in enumerate(uids):
        gate, proposal = int(gates[i]), int(proposals[i])
        actions[uid] = HierarchicalActionV42(gate,None if gate==0 else proposal).normalized.copy()
        likelihoods[uid] = dict(gate=float(gl[i]), proposal=None if gate==0 else float(pl[i]), joint=float(jl[i]))
        if gate:
            ranking = torch.argsort(pd.logits[i],descending=True,stable=True).tolist()
            order = [proposal]+[p for p in ranking if p!=proposal][:2]
            shortlists[uid] = tuple(HierarchicalActionV42(1,p).normalized.copy() for p in order)
    return SetRequest(uids,tensors,actor.numpy(),gates.numpy(),proposals.numpy(),actions,shortlists,
                      likelihoods,(gd.logits.numpy(),pd.logits.numpy()))


def check_request(row):
    """Independent numeric requested likelihood and K3 provenance checks."""
    for i,uid in enumerate(row.uids):
        gate,proposal=int(row.gates[i]),int(row.proposals[i])
        if not row.actor_mask[i] and gate:
            raise ValueError('forced-WAIT selected RELEASE')
        g = float(row.logits[0][i,gate]) if row.actor_mask[i] else 0.
        p = None if gate==0 else float(row.logits[1][i,proposal])
        actual = row.likelihoods[uid]
        if not np.isclose(actual['gate'],g,rtol=0,atol=1e-6) or not np.isclose(actual['joint'],g+(p or 0.),rtol=0,atol=1e-6):
            raise ValueError('requested likelihood mismatch')
        if (p is None and actual['proposal'] is not None) or (p is not None and not np.isclose(p,actual['proposal'],rtol=0,atol=1e-6)):
            raise ValueError('conditional likelihood mismatch')
        if gate:
            ranking=sorted(range(25),key=lambda p:(-float(row.logits[1][i,p]),p))
            expected=[proposal]+[p for p in ranking if p!=proposal][:2]
            if len(row.shortlists[uid])!=3 or any(not np.array_equal(a,HierarchicalActionV42(1,p).normalized) for a,p in zip(row.shortlists[uid],expected)):
                raise ValueError('K3 does not start with the requested proposal')
        elif proposal!=-1 or uid in row.shortlists or np.any(row.actions[uid]):
            raise ValueError('WAIT must have no proposal or shortlist')
