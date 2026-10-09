import torch
import pytest
from unittest.mock import patch

from terminal_marl.algo.intent_neighbor_actor import IntentNeighborActor
from terminal_marl.algo.ordered_proposal_actor import first_feasible_logits


def inputs():
    ego=torch.zeros(2,159);ego[:,8]=1.
    traffic=torch.randn(2,3,50);mask=torch.tensor([[True,True,False],[False,False,False]])
    traffic[~mask]=torch.nan
    return ego,traffic,mask


@pytest.mark.parametrize('gate_mode',['first_feasible_union','max_candidate'])
def test_each_intent_keeps_neighbor_identity_and_permutation_equivariance(gate_mode):
    torch.manual_seed(7);model=IntentNeighborActor(gate_mode=gate_mode);e,t,m=inputs()
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


@pytest.mark.parametrize('gate_mode',['first_feasible_union','max_candidate'])
def test_auxiliary_query_is_exact_same_score_used_by_policy(gate_mode):
    model=IntentNeighborActor(gate_mode=gate_mode);e,t,m=inputs();full=model.relations(e,t,m)
    p=torch.tensor([12,3]);d=torch.tensor([0,0]);pair,scene=model.queried_scores(e,t,m,p,d)
    for i in range(2):
        torch.testing.assert_close(pair[i,m[i]],full['pair_logits'][i,p[i],d[i],m[i]])
        torch.testing.assert_close(scene[i],full['scene_logits'][i,p[i],d[i]])


@pytest.mark.parametrize('gate_mode',['first_feasible_union','max_candidate'])
def test_missing_pad_and_overflow_cannot_silently_create_a_release(gate_mode):
    model=IntentNeighborActor(gate_mode=gate_mode);e,t,m=inputs();e[:,8]=0;e[:,10]=1
    result=model.relations(e,t,m)
    assert not result['eligible_destinations'].any()
    gate,_=model(e,t,m);assert (gate.argmax(-1)==0).all()
    with pytest.raises(ValueError):model(e,torch.zeros(2,33,50),torch.ones(2,33,dtype=torch.bool))
    # No-neighbor state remains a defined query, not an invented blocking UID.
    e[:,10]=0;e[:,8]=1
    result=model.relations(e,torch.zeros(2,0,50),torch.zeros(2,0,dtype=torch.bool))
    assert (result['blocker_index']==-1).all() and torch.isfinite(result['candidate_logits']).all()


def test_max_candidate_gate_uses_best_candidate_without_count_amplification():
    model=IntentNeighborActor(gate_mode='max_candidate')
    candidates=torch.full((2,25),-2.)
    candidates[1,-1]=2.
    with patch.object(model,'relations',return_value={'candidate_logits':candidates}):
        gate,_=model(torch.zeros(2,159),torch.zeros(2,0,50),torch.zeros(2,0,dtype=torch.bool))
    release=torch.softmax(gate,-1)[:,1]
    torch.testing.assert_close(release,torch.sigmoid(torch.tensor([-2.,2.])))
    assert release[0]<.5 and release[1]>.5


def test_gate_modes_keep_initialization_and_proposal_logits_identical():
    torch.manual_seed(11);legacy=IntentNeighborActor()
    torch.manual_seed(11);maximum=IntentNeighborActor(gate_mode='max_candidate')
    assert legacy.architecture==IntentNeighborActor.architecture
    assert legacy.architecture_variant!=maximum.architecture_variant
    assert legacy.architecture_variant.endswith('__gate_first_feasible_union')
    assert maximum.architecture_variant.endswith('__gate_max_candidate')
    assert legacy.state_dict().keys()==maximum.state_dict().keys()
    for name,value in legacy.state_dict().items():
        torch.testing.assert_close(value,maximum.state_dict()[name],rtol=0,atol=0)
    e,t,m=inputs()
    legacy_gate,legacy_proposal=legacy(e,t,m)
    maximum_gate,maximum_proposal=maximum(e,t,m)
    expected_gate,expected_proposal=first_feasible_logits(legacy.relations(e,t,m)['candidate_logits'])
    torch.testing.assert_close(legacy_gate,expected_gate,rtol=0,atol=0)
    torch.testing.assert_close(legacy_proposal,expected_proposal,rtol=0,atol=0)
    torch.testing.assert_close(legacy_proposal,maximum_proposal,rtol=0,atol=0)
    assert not torch.equal(legacy_gate,maximum_gate)


def test_gate_mode_rejects_unknown_value():
    with pytest.raises(ValueError,match='gate_mode'):
        IntentNeighborActor(gate_mode='unknown')
