import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.algo.trajectory_set_actor import TrajectorySetActor
from terminal_marl.algo.trajectory_set_ppo import SetStep,TrajectorySetPPO,actor_batch,tensors
from terminal_marl.algo.joint_ppo_v4_2 import hierarchical_joint_logprob
from terminal_marl.algo.student_ppo_v4_2 import team_gae
from terminal_marl.training.p3_v25_training import central_context
from terminal_marl.core.config import RingRole


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.set_num_threads(1);torch.manual_seed(713);np.random.seed(713)


def step(policy,n=3,neighbors=2,*,reward=1.,terminated=False,actor=True):
    ego=torch.randn(n,105);traffic=torch.randn(n,neighbors,50);mask=torch.ones(n,neighbors,dtype=torch.bool)
    with torch.no_grad():
        gd,pd=policy.model.distributions(ego,traffic,mask) if n else (None,None)
        gates=gd.sample() if actor and n else torch.zeros(n,dtype=torch.long)
        proposals=torch.full((n,),-1,dtype=torch.long)
        release=gates==1
        if release.any(): proposals[release]=torch.distributions.Categorical(logits=pd.logits[release]).sample()
        gl,pl,joint=hierarchical_joint_logprob(gd,pd,gates,proposals) if n else (torch.empty(0),torch.empty(0),torch.empty(0))
    return SetStep(ego.numpy(),traffic.numpy(),mask.numpy(),np.zeros(26,np.float32),
        gates.numpy(),proposals.numpy(),gl.numpy() if actor else np.zeros(n,np.float32),pl.numpy(),
        gd.logits.numpy() if n else np.zeros((0,2),np.float32),
        pd.logits.numpy() if n else np.zeros((0,25),np.float32),
        np.full(n,actor,bool),0.,0.,reward,terminated,terminated)


def test_zero_adapter_preserves_base_and_proposal_exactly():
    base=TrajectorySetActor();policy=TrajectorySetPPO(base);row=step(policy)
    old=base.distributions(*tensors(row));new=policy.model.distributions(*tensors(row))
    assert all(torch.equal(a.logits,b.logits) for a,b in zip(old,new))
    assert all(not p.requires_grad for p in policy.model.actor.parameters())
    assert all(p.requires_grad for p in policy.model.gate_adapter.parameters())


def test_ragged_neighbors_and_forced_rows_do_not_enter_actor_batch():
    policy=TrajectorySetPPO(TrajectorySetActor())
    rows=[step(policy,3,1),step(policy,2,4,actor=False),step(policy,1,5)]
    for row in rows: row.validate()
    batch=actor_batch(rows)
    assert batch[0].shape==(4,105) and batch[1].shape==(4,5,50)
    assert batch[2].sum()==8
    assert torch.equal(batch[0],torch.from_numpy(np.concatenate([rows[0].ego,rows[2].ego])))


def test_stored_probability_and_mask_contract():
    policy=TrajectorySetPPO(TrajectorySetActor());row=step(policy)
    with pytest.raises(ValueError,match='likelihood'): replace(row,gate_logprob=row.gate_logprob+.2).validate()
    with pytest.raises(ValueError,match='stochastic'): replace(row,stochastic_behavior=False).validate()
    with pytest.raises(PermissionError): policy.update([row])
    with pytest.raises(ValueError,match='bootstrap'): replace(row,terminated=True,episode_end=True,next_value=2).validate()


def test_empty_steps_retain_delayed_team_reward_and_boundary_bootstrap():
    policy=TrajectorySetPPO(TrajectorySetActor())
    rows=[step(policy,reward=1),step(policy,0,0,reward=2),step(policy,0,0,reward=8,terminated=True)]
    for row in rows: row.validate()
    a,r=team_gae(rows,.99,1.)
    assert a==pytest.approx([1+.99*2+.99**2*8,2+.99*8,8])
    # A truncation bootstraps value but cannot mix the following episode.
    rows[1]=replace(rows[1],episode_end=True,next_value=7)
    assert team_gae(rows,.99,1.)[0][1]==pytest.approx(2+.99*7)


def test_update_freezes_base_and_updates_only_adapter_and_critic():
    policy=TrajectorySetPPO(TrajectorySetActor(),learning_rate=1e-3)
    rows=[step(policy,reward=float(i+1)) for i in range(4)]+[step(policy,0,0,reward=8,terminated=True)]
    before=copy.deepcopy(policy.model.state_dict())
    result=policy.update(rows,training_authorized=True,epochs=1,minibatch_size=5,target_kl=.1)
    after=policy.model.state_dict()
    assert result['accepted_updates']==1 and result['rejected_updates']==0
    assert result['team_steps']==5 and result['actor_decisions']==12
    assert result['actor_count_normalizer']==12/5
    assert all(torch.equal(v,after[k]) for k,v in before.items() if k.startswith('actor.'))
    assert any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith('gate_adapter.'))
    assert any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith('critic_'))
    with pytest.raises(ValueError,match='behavior'): policy.update(rows,training_authorized=True,epochs=1)


def test_kl_rejection_rolls_back_parameters_and_adam_state():
    policy=TrajectorySetPPO(TrajectorySetActor(),learning_rate=1.)
    rows=[step(policy,reward=1),step(policy,reward=10,terminated=True)]
    before=copy.deepcopy(policy.model.state_dict())
    result=policy.update(rows,training_authorized=True,epochs=1,minibatch_size=2,target_kl=1e-10)
    assert result['accepted_updates']==0 and result['rejected_updates']==1
    assert all(torch.equal(v,policy.model.state_dict()[k]) for k,v in before.items())
    assert policy.optimizer.state_dict()['state']=={}


def test_rollback_preserves_existing_adam_moments():
    policy=TrajectorySetPPO(TrajectorySetActor(),learning_rate=1e-3)
    warm=[step(policy,reward=1),step(policy,reward=10,terminated=True)]
    assert policy.update(warm,training_authorized=True,epochs=1,minibatch_size=2,target_kl=.1)['accepted_updates']==1
    rows=[step(policy,reward=1),step(policy,reward=10,terminated=True)]
    for group in policy.optimizer.param_groups: group['lr']=1.
    weights=copy.deepcopy(policy.model.state_dict());optim=copy.deepcopy(policy.optimizer.state_dict())
    result=policy.update(rows,training_authorized=True,epochs=1,minibatch_size=2,target_kl=1e-10)
    assert result['rejected_updates']==1 and result['accepted_updates']==0
    assert all(torch.equal(v,policy.model.state_dict()[k]) for k,v in weights.items())
    actual=policy.optimizer.state_dict()
    assert optim['param_groups']==actual['param_groups']
    for key,values in optim['state'].items():
        for name,value in values.items(): assert torch.equal(value,actual['state'][key][name])


def test_all_forced_and_empty_steps_only_update_critic():
    policy=TrajectorySetPPO(TrajectorySetActor(),learning_rate=1e-3)
    rows=[step(policy,actor=False,reward=1),step(policy,0,0,reward=10,terminated=True)]
    before=copy.deepcopy(policy.model.state_dict())
    result=policy.update(rows,training_authorized=True,epochs=1,minibatch_size=2)
    assert result['accepted_updates']==1 and result['actor_decisions']==0
    after=policy.model.state_dict()
    assert all(torch.equal(v,after[k]) for k,v in before.items() if not k.startswith('critic_'))
    assert any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith('critic_'))


def test_critic_keeps_service_context_when_no_active_agents():
    ring=SimpleNamespace(radius=200.,height=120.,nominal_speed=15.)
    zone=SimpleNamespace(status=SimpleNamespace(value='occupied'),service_remaining=50.,expected_release_at=None)
    env=SimpleNamespace(config=SimpleNamespace(total_requested_uavs=12),now=40.,
        arrivals=SimpleNamespace(generated=12,next_time=float('inf'),exhausted=lambda:True),
        admission=SimpleNamespace(queue=SimpleNamespace(items=[])),lz_manager=SimpleNamespace(zones=[zone]),
        active={},ledger=SimpleNamespace(unresolved={1}),airspace=SimpleNamespace(actual_ring=lambda r:ring))
    context=central_context(env)
    assert context.shape==(26,) and np.isfinite(context).all()
    assert context[3]==1 and context[10]==1 and context[11]==pytest.approx(.25)
    assert context[15]==0 and context[16]==pytest.approx(1/12)
    policy=TrajectorySetPPO(TrajectorySetActor());row=step(policy,0,0)
    value=policy.model.value(*tensors(row),torch.from_numpy(context))
    value.backward()
    assert any(p.grad is not None for p in policy.model.critic_head.parameters())
