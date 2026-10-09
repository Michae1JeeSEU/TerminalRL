import pytest
import torch
from terminal_marl.algo.ordered_proposal_actor import OrderedProposalActor,first_feasible_logits,RELEASE_ORDER


def test_first_feasible_joint_mass_matches_checked_prefix_likelihood():
    logits=torch.linspace(-3,2,25,dtype=torch.float64).repeat(2,1).requires_grad_()
    gate,proposal=first_feasible_logits(logits)
    p_gate=gate.softmax(-1);p_proposal=proposal.softmax(-1)
    torch.testing.assert_close(gate.exp().sum(-1),torch.ones(2,dtype=torch.float64))
    rank=8;chosen=RELEASE_ORDER[rank]
    joint_nll=-(gate[0,1]+proposal[0,chosen]-torch.logsumexp(proposal[0],-1))
    checked=torch.tensor(RELEASE_ORDER[:rank+1]);target=torch.zeros(rank+1,dtype=torch.float64);target[-1]=1
    expected=torch.nn.functional.binary_cross_entropy_with_logits(logits[0,checked],target,reduction='sum')
    torch.testing.assert_close(joint_nll,expected)
    joint_nll.backward()
    assert torch.equal(logits.grad[0,list(RELEASE_ORDER[rank+1:])],torch.zeros(24-rank,dtype=torch.float64))
    torch.testing.assert_close(p_gate[:,0],torch.sigmoid(-logits).prod(-1))
    torch.testing.assert_close((p_gate[:,1,None]*p_proposal).sum(-1)+p_gate[:,0],torch.ones(2,dtype=torch.float64))


@pytest.mark.parametrize('value',[-100.,100.])
def test_extreme_scores_are_finite_and_have_correct_gate(value):
    scores=torch.full((2,25),value,requires_grad=True)
    gate,proposal=first_feasible_logits(scores)
    assert torch.isfinite(gate).all() and torch.isfinite(proposal).all()
    assert gate.argmax(-1).tolist()==[int(value>0)]*2
    if value>0:assert proposal.argmax(-1).tolist()==[RELEASE_ORDER[0]]*2
    (gate.sum()+proposal.sum()).backward();assert torch.isfinite(scores.grad).all()


def test_teacher_catalogue_order_is_exact_without_importing_teacher_at_inference():
    from terminal_marl.training.v4_1_joint_teacher import STATE_ONLY_JOINT_CATALOGUE_V4_1
    assert RELEASE_ORDER==tuple(x.index-1 for x in STATE_ONLY_JOINT_CATALOGUE_V4_1 if x.index)


def test_numeric_actor_is_permutation_invariant_and_ignores_nan_padding():
    torch.manual_seed(10);model=OrderedProposalActor()
    ego=torch.randn(2,159);traffic=torch.randn(2,3,50);mask=torch.tensor([[True,True,False],[False,False,False]])
    traffic[~mask]=float('nan')
    expected=model(ego,traffic,mask);actual=model(ego,traffic.flip(1),mask.flip(1))
    for a,b in zip(expected,actual):torch.testing.assert_close(a,b)
    assert all(torch.isfinite(x).all() for x in actual)
    sum(x.sum() for x in actual).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):model(ego[:,:105],traffic,mask)
    assert all(torch.isfinite(x).all() for x in model(ego,torch.zeros(2,0,50),torch.zeros(2,0,dtype=torch.bool)))
