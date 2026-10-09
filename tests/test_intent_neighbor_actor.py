import torch
import pytest

from terminal_marl.algo.intent_neighbor_actor import IntentNeighborActor


def inputs():
    ego=torch.zeros(2,159);ego[:,8]=1.
    traffic=torch.randn(2,3,50);mask=torch.tensor([[True,True,False],[False,False,False]])
    traffic[~mask]=torch.nan
    return ego,traffic,mask


def test_each_intent_keeps_neighbor_identity_and_permutation_equivariance():
    torch.manual_seed(7);model=IntentNeighborActor();e,t,m=inputs()
    a=model.relations(e,t,m);order=torch.tensor([2,0,1]);b=model.relations(e,t[:,order],m[:,order])
    assert a['pair_logits'].shape==(2,25,6,3)
    torch.testing.assert_close(a['pair_logits'][:,:,:,order],b['pair_logits'])
    torch.testing.assert_close(a['candidate_logits'],b['candidate_logits'])
    assert (a['blocker_index'][1]==-1).all()
    assert not torch.equal(a['pair_logits'][0,0,0,:2],a['pair_logits'][0,24,0,:2])
    gate,proposal=model(e,t,m)
    assert torch.isfinite(gate).all() and torch.isfinite(proposal).all()
    (gate.square().mean()+proposal.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_auxiliary_query_is_exact_same_score_used_by_policy():
    model=IntentNeighborActor();e,t,m=inputs();full=model.relations(e,t,m)
    p=torch.tensor([12,3]);d=torch.tensor([0,0]);pair,scene=model.queried_scores(e,t,m,p,d)
    for i in range(2):
        torch.testing.assert_close(pair[i,m[i]],full['pair_logits'][i,p[i],d[i],m[i]])
        torch.testing.assert_close(scene[i],full['scene_logits'][i,p[i],d[i]])


def test_missing_pad_and_overflow_cannot_silently_create_a_release():
    model=IntentNeighborActor();e,t,m=inputs();e[:,8]=0;e[:,10]=1
    result=model.relations(e,t,m)
    assert not result['eligible_destinations'].any()
    gate,_=model(e,t,m);assert (gate.argmax(-1)==0).all()
    with pytest.raises(ValueError):model(e,torch.zeros(2,33,50),torch.ones(2,33,dtype=torch.bool))
    # No-neighbor state remains a defined query, not an invented blocking UID.
    e[:,10]=0;e[:,8]=1
    result=model.relations(e,torch.zeros(2,0,50),torch.zeros(2,0,dtype=torch.bool))
    assert (result['blocker_index']==-1).all() and torch.isfinite(result['candidate_logits']).all()
