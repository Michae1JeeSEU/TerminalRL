import json
import pytest
import torch
from terminal_marl.scripts.diagnose_p3_v25_train_generalization import run, subset
from terminal_marl.training.p3_v25_set_learnability import synthetic


def test_subset_keeps_complete_row_identity_and_inputs():
    paired=synthetic();indices=torch.tensor([1,8,13])
    data,tensors=subset(paired,indices)
    assert data.ids==tuple(paired.original.ids[i] for i in indices)
    torch.testing.assert_close(tensors[0],paired.new[0][indices])
    torch.testing.assert_close(data.gate,paired.original.gate[indices])


def test_additional_cycles_cannot_bypass_resource_or_real_data_contract(tmp_path):
    with pytest.raises(ValueError,match='real resource train mode'):
        run([],tmp_path/'mixed',smoke=True,resource_context=True,additional_directories=['Q'])
    with pytest.raises(ValueError,match='real resource train mode'):
        run([],tmp_path/'old',additional_directories=['Q'])
    with pytest.raises(ValueError,match='resource observations'):
        run([],tmp_path/'bad_ordered',smoke=True,ordered_proposals=True)


def test_ordered_proposal_smoke_serializes_fixed_architecture_and_complete_curves(tmp_path):
    torch.set_num_threads(1)
    root=tmp_path/'ordered';result=run([],root,smoke=True,resource_context=True,ordered_proposals=True)
    assert result['status']=='COMPLETE' and len(result['folds'])==2
    assert result['protocol']['architecture']=='ordered_candidate_set_v1'
    assert result['protocol']['ego_dim']==159 and not result['protocol']['Q_rows_accessed']
    assert all([v['update'] for v in f['curve']]==[1,2] for f in result['folds'])
    assert json.loads(json.dumps(result))==json.loads((root/'result.json').read_text())
    assert not list(root.glob('*.pt'))


@pytest.mark.parametrize('resource_context',[False,True])
def test_smoke_has_disjoint_splits_serialized_curves_and_no_checkpoint(tmp_path,resource_context):
    torch.set_num_threads(1)
    root=tmp_path/'smoke';result=run([],root,smoke=True,resource_context=resource_context)
    # The shared metric helper uses integer histogram keys; JSON preserves
    # their values while converting object keys to strings.
    assert json.loads(json.dumps(result))==json.loads((root/'result.json').read_text())
    assert result['status']=='COMPLETE' and len(result['folds'])==2
    assert not result['protocol']['Q_rows_accessed']
    assert result['protocol']['ego_dim']==(159 if resource_context else 105)
    for fold in result['folds']:
        assert fold['train_rows']==fold['validation_rows']==8
        assert [v['update'] for v in fold['curve']]==[1,2]
        assert fold['curve'][-1]['train']['loss']!=fold['curve'][0]['train']['loss']
    assert not list(root.glob('*.pt'))
    with pytest.raises(FileExistsError):run([],root,smoke=True)
