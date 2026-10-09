import json

import pytest
import torch

from terminal_marl.training.p3_neighbor_witness_dataset import auxiliary_loss
from terminal_marl.scripts.diagnose_p3_intent_neighbor import smoke_inputs,split,run


def test_auxiliary_rows_are_held_out_with_their_parent_cycle():
    paired,witness=smoke_inputs()
    train,held,wtrain,wheld=split(paired,witness,1)
    assert set(witness.state_index[wtrain].tolist())<=set(train.tolist())
    assert set(witness.state_index[wheld].tolist())<=set(held.tolist())
    witness.seeds[0]=2
    with pytest.raises(ValueError):split(paired,witness,1)


def test_unknown_pair_has_no_supervision_gradient():
    _,witness=smoke_inputs();witness.known[0,0]=False
    logits=torch.zeros_like(witness.blocked,requires_grad=True)
    scene=torch.zeros(len(logits),requires_grad=True)
    auxiliary_loss(logits,scene,witness).backward()
    assert logits.grad[0,0]==0 and torch.isfinite(logits.grad).all()


def test_two_step_diagnostic_serializes_both_arms(tmp_path):
    torch.set_num_threads(1)
    result=run(None,None,None,tmp_path/'cv',smoke=True)
    assert result['status']=='COMPLETE' and len(result['folds'])==4
    saved=json.loads((tmp_path/'cv/result.json').read_text())
    assert saved['status']=='COMPLETE' and not saved['protocol']['checkpoint_written']
    assert all(len(f['curve'])==3 for f in saved['folds'])
    assert all('blocker_top1' in f['curve'][-1]['validation_pairs'] for f in saved['folds'])
    with pytest.raises(FileExistsError):run(None,None,None,tmp_path/'cv',smoke=True)


def test_max_candidate_smoke_serializes_one_auxiliary_arm_and_mode(tmp_path):
    torch.set_num_threads(1)
    root=tmp_path/'max_candidate'
    result=run(None,None,None,root,smoke=True,max_candidate_gate=True)
    saved=json.loads((root/'result.json').read_text())
    assert result['status']=='COMPLETE' and len(result['folds'])==2
    assert result['protocol']['gate_mode']=='max_candidate'
    assert result['protocol']['architecture']=='intent_destination_neighbor_risk_v1'
    assert result['protocol']['architecture_variant'].endswith('__gate_max_candidate')
    assert result['protocol']['arms']==['native_pair_auxiliary']
    assert result['protocol']['comparison_control_reference']=='p3_neighbor_relation_cv_20261009/six_cycle'
    assert result['protocol']['comparison_note']=='max gate tests count-amplification; no promotion'
    assert all(f['arm']=='native_pair_auxiliary' for f in result['folds'])
    assert all([v['update'] for v in f['curve']]==[0,1,2] for f in result['folds'])
    assert not result['protocol']['checkpoint_written'] and not result['protocol']['Q_accessed']
    assert json.loads(json.dumps(result))==saved
