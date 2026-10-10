import json
import sys
from pathlib import Path

import pytest
import torch

from terminal_marl.scripts import diagnose_p3_intent_neighbor as diagnostic
from terminal_marl.training.p3_neighbor_witness_dataset import (
    WitnessTargets,
    auxiliary_loss,
    blocker_ranking_loss,
)


def _target(known,blocked):
    known=torch.tensor(known,dtype=torch.bool)
    blocked=torch.tensor(blocked,dtype=torch.float)
    rows=known.shape[0]
    return WitnessTargets(
        torch.arange(rows),torch.zeros(rows,dtype=torch.long),torch.zeros(rows,dtype=torch.long),
        blocked,known,torch.zeros(rows),torch.ones(rows,dtype=torch.bool),torch.ones(rows,dtype=torch.long))


def _legacy_auxiliary(pair_logits,scene_logits,target):
    losses=[]
    for positive in (False,True):
        selected=target.known & (target.blocked.bool()==positive)
        if selected.any():
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
                pair_logits[selected],target.blocked[selected]))
    pair_loss=torch.stack(losses).mean() if losses else pair_logits.sum()*0.
    scene_loss=(torch.nn.functional.binary_cross_entropy_with_logits(
        scene_logits[target.scene_known],target.scene_blocked[target.scene_known])
        if target.scene_known.any() else scene_logits.sum()*0.)
    return pair_loss+.25*scene_loss


def test_default_auxiliary_loss_is_legacy_value_and_gradient():
    target=_target([[True,True,False]],[[1.,0.,0.]])
    pair=torch.tensor([[.4,-.7,1.2]],requires_grad=True)
    scene=torch.tensor([.3],requires_grad=True)
    actual=auxiliary_loss(pair,scene,target)
    actual.backward()
    expected_pair=pair.detach().clone().requires_grad_()
    expected_scene=scene.detach().clone().requires_grad_()
    expected=_legacy_auxiliary(expected_pair,expected_scene,target)
    expected.backward()
    assert torch.equal(actual.detach(),expected.detach())
    assert torch.equal(pair.grad,expected_pair.grad)
    assert torch.equal(scene.grad,expected_scene.grad)
    weighted=auxiliary_loss(pair.detach(),scene.detach(),target,blocker_ranking_weight=0.)
    torch.testing.assert_close(actual.detach(),weighted)


def test_multi_positive_ranking_uses_total_known_mass():
    target=_target([[True,True,True,False]],[[1.,0.,1.,0.]])
    logits=torch.tensor([[0.,2.,1.,-100.]])
    expected=torch.logsumexp(torch.tensor([0.,2.,1.]),0)-torch.logsumexp(torch.tensor([0.,1.]),0)
    torch.testing.assert_close(blocker_ranking_loss(logits,target),expected)


def test_unknown_and_absent_nonfinite_logits_have_zero_gradient():
    target=_target([[True,False,True,False]],[[1.,0.,0.,0.]])
    logits=torch.tensor([[0.,float('nan'),1.,float('inf')]],requires_grad=True)
    loss=blocker_ranking_loss(logits,target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0,1]==0 and logits.grad[0,3]==0


def test_no_eligible_rows_returns_finite_differentiable_zero():
    target=_target([[False,False],[True,False]],[[0.,0.],[1.,0.]])
    logits=torch.tensor([[1e38,-1e38],[1e38,-1e38]],requires_grad=True)
    loss=blocker_ranking_loss(logits,target)
    assert loss.item()==0. and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.equal(logits.grad,torch.zeros_like(logits.grad))


def test_ranking_is_permutation_invariant_and_finite_for_extreme_logits():
    target=_target([[True,True,True]],[[1.,0.,1.]])
    logits=torch.tensor([[10000.,-10000.,9999.]])
    perm=torch.tensor([2,0,1])
    permuted=_target([target.known[0,perm].tolist()],[target.blocked[0,perm].tolist()])
    torch.testing.assert_close(blocker_ranking_loss(logits,target),
        blocker_ranking_loss(logits[:,perm],permuted))
    assert torch.isfinite(blocker_ranking_loss(logits,target))


def test_improving_blocker_logit_lowers_ranking_loss():
    target=_target([[True,True]],[[1.,0.]])
    bad=blocker_ranking_loss(torch.tensor([[-2.,2.]]),target)
    good=blocker_ranking_loss(torch.tensor([[2.,-2.]]),target)
    assert good<bad


def test_enabled_auxiliary_adds_fixed_ranking_term():
    target=_target([[True,True]],[[1.,0.]])
    pair=torch.tensor([[2.,-2.]],requires_grad=True)
    scene=torch.tensor([0.],requires_grad=True)
    base=auxiliary_loss(pair,scene,target)
    enabled=auxiliary_loss(pair,scene,target,blocker_ranking_weight=.25)
    torch.testing.assert_close(enabled,base+.25*blocker_ranking_loss(pair,target))


@pytest.mark.parametrize('argv',[
    ['--blocker-ranking','--smoke'],
    ['--blocker-ranking','--train-directories','a','b','c',
        '--additional-train-directories','d','e','f','--witness-directory','w'],
])
def test_cli_rejects_blocker_ranking_without_non_smoke_prefix_route(tmp_path,monkeypatch,argv):
    monkeypatch.setattr(sys,'argv',['diagnose_p3_intent_neighbor.py',*argv,
        '--output',str(tmp_path/'cli')])
    with pytest.raises(SystemExit) as error:
        diagnostic.main()
    assert error.value.code==2


def test_synthetic_parallel_reference_route_emits_ranking_fields(tmp_path):
    torch.set_num_threads(1)
    paired,witness=diagnostic.smoke_inputs(supplemental=True)
    specs=[dict(seed=seed,arm='native_pair_auxiliary',gate_mode='first_feasible_union',marks=(0,1,2),
        wall_seconds=10,progress_path=tmp_path/f'fold_{seed}.progress.json') for seed in range(1,10)]
    protocol=dict(loss_variant='native_pair_auxiliary_plus_blocker_ranking',blocker_ranking_weight=.25)
    result=diagnostic._execute_folds(paired,witness,specs,workers=4,root_path=tmp_path/'result.json',
        protocol=protocol,reference_witness=witness,blocker_ranking_weight=.25)
    assert result['status']=='COMPLETE' and len(result['folds'])==9
    saved=json.loads((tmp_path/'result.json').read_text())
    assert saved['status']=='COMPLETE' and len(saved['folds'])==9
    for fold in result['folds']:
        assert len(fold['curve'])==3
        for curve in fold['curve']:
            for key in ('blocker_ranking_loss','blocker_ranking_queries'):
                assert key in curve['validation_pairs']
                assert key in curve['validation_reference_pairs']


def test_run_forwards_fixed_ranking_and_records_reference_support(tmp_path,monkeypatch):
    paired,witness=diagnostic.smoke_inputs(supplemental=True)
    observed={}
    monkeypatch.setattr(diagnostic,'load_nine_cycle_train',lambda *args:paired)
    monkeypatch.setattr(diagnostic,'load_witness_targets',lambda *args:witness)
    monkeypatch.setattr(diagnostic,'combine_witness_targets',lambda parts:witness)
    def stub_execute(*args,**kwargs):
        observed['reference_witness']=kwargs['reference_witness']
        observed['blocker_ranking_weight']=kwargs['blocker_ranking_weight']
        return dict(status='COMPLETE',protocol=kwargs['protocol'],folds=[],error=None)
    monkeypatch.setattr(diagnostic,'_execute_folds',stub_execute)
    output=tmp_path/'prefix'
    result=diagnostic.run(
        (Path('old1'),Path('old2'),Path('old3')),
        (Path('ind1'),Path('ind2'),Path('ind3')),
        tmp_path/'old',output,
        supplemental=(Path('supp1'),Path('supp2'),Path('supp3')),
        supplemental_witness=tmp_path/'supp',prefix_witness=tmp_path/'prefix_witness',workers=4,
        blocker_ranking_weight=.25)
    protocol=json.loads((output/'protocol.json').read_text())
    assert result['status']=='COMPLETE'
    assert observed['reference_witness'] is witness
    assert observed['blocker_ranking_weight']==.25
    assert protocol['comparison_control_reference']=='p3_neighbor_prefix_ring_reuse_20261010/nine_cycle'
    assert 'same nine-fold 3149 actor rows' in protocol['comparison_note']
    assert 'same 1270 expanded and 473 original-held reference supports' in protocol['comparison_note']
    assert protocol['blocker_ranking_weight']==.25
    assert protocol['blocker_ranking_reference_support']['queries']==len(witness.state_index)
    assert protocol['blocker_ranking_training_support']['queries']==len(witness.state_index)


def test_run_rejects_unfixed_or_nonprefix_ranking_before_output(tmp_path):
    bad_weight=tmp_path/'bad_weight'
    with pytest.raises(ValueError,match='fixed at 0.25'):
        diagnostic.run(None,None,None,bad_weight,smoke=True,blocker_ranking_weight=.1)
    assert not bad_weight.exists()
    bad_route=tmp_path/'bad_route'
    with pytest.raises(ValueError,match='non-smoke prefix supplemental route'):
        diagnostic.run(None,None,None,bad_route,smoke=True,blocker_ranking_weight=.25)
    assert not bad_route.exists()
