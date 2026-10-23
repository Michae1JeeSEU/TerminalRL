"""Targeted Stage3 input/architecture safeguards; no formal training campaign."""
import builtins
from dataclasses import replace
from types import SimpleNamespace
import math
import sys
import pytest

from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.world import AircraftState
from terminal_marl.core.rotating_slots.resources import Reservation
from terminal_marl.core.rotating_slots.geometry import slot_state
from terminal_marl.core.rotating_slots.stage3_learning.features import FeatureEncoder,FeatureInputError,QUERY_DIM,SET_DIMS
from terminal_marl.core.rotating_slots.stage3_learning.model import FeatureNormalizer,build_model,fingerprint_compatible,LearnedAdvisor
from terminal_marl.core.rotating_slots.stage3_learning.learning import train_model,auxiliary_targets


def fixture():
    config=load_config()
    own_state=slot_state('outer',0,10.,config)
    other_state=slot_state('middle',2,10.,config)
    own=AircraftState('a','HOLD_OUTER',.8,own_state.p,own_state.v,own_state.a,own_state.jerk,ring='outer',slot=0)
    other=AircraftState('b','HOLD_MIDDLE',.7,other_state.p,other_state.v,other_state.a,other_state.jerk,ring='middle',slot=2)
    snapshot=SimpleNamespace(time=10.,config=config,aircraft={'a':own,'b':other},active={},soft={},slots=(),pads=())
    descriptor=dict(descriptor_id='d',aircraft_id='a',source_node_id='n',source_ring='outer',source_slot=0,
        target_kind='next_ring_slot',target_id=2,start_offset_s=0.,duration_s=30.,parent_prefix_id=None,
        construction_stage='progress',proposal_origin='current')
    return snapshot,descriptor


def test_features_are_readonly_cheap_and_forbidden_fields_ignored(monkeypatch):
    import terminal_marl.core.rotating_slots.trajectory as trajectory
    import terminal_marl.core.rotating_slots.energy as energy
    import terminal_marl.core.rotating_slots.safety as safety
    def forbidden(*args,**kwargs):
        raise AssertionError('expensive call')
    for module,names in [(trajectory,('generate_transfer','generate_final')),(energy,('integrate_energy',)),(safety,('validate_separation','validate_plan'))]:
        for name in names:
            monkeypatch.setattr(module,name,forbidden)
    s,d=fixture(); before=dict(s.aircraft)
    first=FeatureEncoder().encode(s,[d])
    s.unrevealed_manifest={'future':'must not read'}; s.random_seed=99; s.final_solver_choice='d'
    second=FeatureEncoder().encode(s,[dict(d,random_seed=2,future_outcome='landed')],{'future_outcome':8,'seed':99})
    assert first == second
    assert s.aircraft == before
    f=first[0]
    assert len(f['query']) == QUERY_DIM
    assert all(len(r) == SET_DIMS[k] for k,rows in f['sets'].items() for r in rows)
    assert f['query'][-4:] == [0.,0.,0.,0.]  # absent shared downstream estimate


def test_permutation_and_identifier_invariance():
    s,d=fixture(); encoder=FeatureEncoder()
    one=encoder.encode(s,[d])[0]
    s.aircraft={'renamed-b':replace(s.aircraft['b'],id='renamed-b'),'renamed-a':replace(s.aircraft['a'],id='renamed-a')}
    two=encoder.encode(s,[dict(d,aircraft_id='renamed-a',descriptor_id='renamed-d')])[0]
    assert one['query'] == two['query']
    for name in SET_DIMS:
        assert sorted(one['sets'][name]) == sorted(two['sets'][name])
    torch=pytest.importorskip('torch')
    torch.manual_seed(1); model=build_model().eval()
    with torch.no_grad():
        assert torch.allclose(model([one]),model([two]),atol=1e-6)


def test_real_parent_epoch_and_no_hypothetical_soc():
    s,d=fixture(); d=dict(d,parent_prefix_id='parent',source_ring='middle',source_slot=2)
    with pytest.raises(FeatureInputError,match='unmaterialized'):
        FeatureEncoder().encode(s,[d])
    state={'parents':{'parent':dict(materialized=True,soc=.63,ring='middle',slot=2,end_time=65.,depth=1,peak=12.)}}
    f=FeatureEncoder().encode(s,[d],state)[0]
    p=slot_state('middle',2,65.,s.config).p
    assert f['query'][44:50:2] == pytest.approx([v/900. for v in p])
    assert f['query'][32] == .63
    assert f['query'][50:54:2] == [.1,.9]


def test_overflow_nonfinite_soc_and_infinite_reservation():
    s,d=fixture()
    with pytest.raises(FeatureInputError,match='ENTITY_LIMIT'):
        FeatureEncoder(entity_limits={'aircraft':1}).encode(s,[d])
    s.slots=(Reservation(('outer',0),'a',0.,math.inf),)
    assert FeatureEncoder().encode(s,[d])[0]['sets']['reservations']
    s.aircraft['a']=replace(s.aircraft['a'],soc=1.1)
    with pytest.raises(FeatureInputError,match='INPUT_OOD'):
        FeatureEncoder().encode(s,[d])
    s.aircraft['a']=replace(s.aircraft['a'],soc=.8,p=(math.nan,0.,0.))
    with pytest.raises(FeatureInputError,match='INPUT_OOD'):
        FeatureEncoder().encode(s,[d])


def test_normalizer_train_only_and_missing_values():
    s,d=fixture(); f=FeatureEncoder().encode(s,[d])[0]
    train={'split':'train','features':[f]}
    n=FeatureNormalizer().fit([train]); transformed=n.transform(f)
    assert transformed['query'][-4:] == [0.,0.,0.,0.]
    assert n.as_dict()['fit_split'] == 'train'
    with pytest.raises(ValueError,match='train contexts only'):
        FeatureNormalizer().fit([dict(train,split='validation')])
    assert n.statistics['query'][16]['count'] == 1


def test_auxiliary_masks_unknown_timeout_and_unselected():
    for outcome in ('UNKNOWN','TIMEOUT','NOT_ATTEMPTED','FIXED_BACKGROUND_CONFLICT','JOINT_CERTIFIED'):
        assert auxiliary_targets({'outcome':outcome,'wall_s':1.,'censored':True}) == (None,None)
    assert auxiliary_targets({'outcome':'SINGLE_INVALID_CERTIFIED','wall_s':.01,'censored':False}) == (0.,pytest.approx(math.log1p(10.)))
    assert auxiliary_targets({'outcome':'SINGLE_VALID','wall_s':None}) == (1.,None)


def test_zero_labels_no_formal_training_or_torch_import(tmp_path,monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('torch import without labels')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    result=train_model({'contexts':[],'preference_pairs':[],'fingerprint':{'objective_id':'o'}},tmp_path/'model',{'objective_id':'o'})
    assert result['status'] == 'DATA_INSUFFICIENT'
    assert not (tmp_path/'formal_training.guard.json').exists()
    assert not (tmp_path/'model'/'model.pt').exists()


def test_test_context_rejected_without_access(tmp_path):
    with pytest.raises(ValueError,match='test sealed'):
        train_model({'contexts':[{'split':'test'}]},tmp_path/'model',{})


def test_empty_set_pool_and_fixed_architecture():
    torch=pytest.importorskip('torch'); model=build_model()
    assert sum(p.numel() for p in model.parameters()) <= 250000
    x=torch.tensor([[[9.]*64,[5.]*64]])
    pooled=model.pool(x,torch.tensor([[False,False]]),256)
    assert torch.equal(pooled,torch.zeros((1,129)))
    empty=model.pool(torch.zeros((1,0,64)),torch.zeros((1,0),dtype=torch.bool),256)
    assert torch.equal(empty,torch.zeros((1,129)))
    s,d=fixture(); f=FeatureEncoder().encode(s,[d])[0]
    assert model([f]).shape == (1,3)
    assert all(type(m).__name__ not in ('BatchNorm1d','Dropout','LSTM','MultiheadAttention') for m in model.modules())


def test_fingerprint_and_checkpoint_rejection(tmp_path):
    assert fingerprint_compatible({'objective_id':'a','model_id':'NO_MODEL'},{'objective_id':'a','model_id':'hash'})
    assert not fingerprint_compatible({'objective_id':'a'},{'objective_id':'b'})
    torch=pytest.importorskip('torch')
    path=tmp_path/'model.pt'
    torch.save({'architecture_id':'wrong','schema_id':'candidate_context_sets_v1_0'},path)
    with pytest.raises(FeatureInputError,match='FINGERPRINT_MISMATCH'):
        LearnedAdvisor(path,{'objective_id':'a'},evaluation=True).rank([])


def test_debug_one_step_grouped_loss_and_frozen_checkpoint(tmp_path):
    """One toy optimizer step only; never invokes the formal training entry point."""
    import random
    from terminal_marl.core.rotating_slots.stage3_learning.learning import _loss
    from terminal_marl.core.rotating_slots.stage3_learning.model import ARCHITECTURE_ID
    from terminal_marl.core.rotating_slots.stage3_learning.features import SCHEMA_ID
    torch=pytest.importorskip('torch')
    F=torch.nn.functional
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        assert torch.get_num_interop_threads() == 1
    torch.manual_seed(20261023)
    snapshot,descriptor=fixture()
    features=FeatureEncoder().encode(snapshot,[dict(descriptor,descriptor_id=name,start_offset_s=offset)
        for name,offset in [('toy-a',0.),('toy-b',10.),('toy-c',20.)]])
    normalizer=FeatureNormalizer().fit([{'split':'train','features':features}])
    by_id={f['descriptor_id']:f for f in features}
    groups=[
        {'features':by_id,'pairs':[('toy-a','toy-b'),('toy-a','toy-c')],
         'facts':{'toy-a':{'outcome':'SINGLE_VALID','validity_label_mask':False,'wall_s':2.,'censored':True},
                  'toy-b':{'outcome':'UNKNOWN','wall_s':.01,'censored':False},
                  'toy-c':{'outcome':'TIMEOUT','wall_s':100000.,'censored':False}}},
        {'features':by_id,'pairs':[('toy-c','toy-b')],
         'facts':{'toy-c':{'outcome':'SINGLE_VALID','validity_label_mask':True,'wall_s':.005,'censored':False},
                  'toy-a':{'outcome':'SINGLE_INVALID_CERTIFIED','validity_label_mask':False,'wall_s':None}}},
    ]
    model=build_model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    prediction=dict(zip(by_id,model([normalizer.transform(f) for f in features]).unbind(0)))
    rank_one=torch.stack([F.softplus(-(prediction[a][0]-prediction[b][0])) for a,b in groups[0]['pairs']]).mean()
    rank_two=F.softplus(-(prediction['toy-c'][0]-prediction['toy-b'][0]))
    expected_rank=(rank_one+rank_two)/2
    expected_cost_one=F.huber_loss(prediction['toy-b'][2],torch.tensor(math.log1p(10.)),delta=1.)
    expected_valid_two=F.binary_cross_entropy_with_logits(prediction['toy-c'][1],torch.tensor(1.))
    expected_cost_two=F.huber_loss(prediction['toy-c'][2],torch.tensor(math.log1p(5.)),delta=1.)
    expected=(rank_one+.1*expected_cost_one+rank_two+.1*expected_valid_two+.1*expected_cost_two)/2
    loss,rank=_loss(model,groups,normalizer,random.Random(20261023))
    repeated,_=_loss(model,groups,normalizer,random.Random(20261023))
    assert torch.equal(loss,repeated)
    assert torch.allclose(rank,expected_rank,atol=1e-7)
    assert torch.allclose(loss,expected,atol=1e-7)
    before=[p.detach().clone() for p in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) for g in gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()  # This is the test's sole gradient update.
    assert any(not torch.equal(old,new) for old,new in zip(before,model.parameters()))
    fingerprint={'objective_id':'toy-development-only','model_id':'NO_MODEL'}
    checkpoint=tmp_path/'debug-development-only.pt'
    torch.save(dict(architecture_id=ARCHITECTURE_ID,schema_id=SCHEMA_ID,fingerprint=fingerprint,
        state_dict=model.state_dict(),normalizer=normalizer.as_dict()),checkpoint)
    advisor=LearnedAdvisor(checkpoint,dict(fingerprint,model_id='frozen-debug-hash'),evaluation=True)
    result=advisor.rank(features)
    model.eval()
    with torch.inference_mode():
        expected_scores=model([normalizer.transform(f) for f in features])[:,0].tolist()
    assert result.scores == pytest.approx(dict(zip(by_id,expected_scores)))
    assert set(result.ordered_ids) == set(by_id)
    assert advisor._normalizer.as_dict() == normalizer.as_dict()
    assert all(p.device.type == 'cpu' for p in advisor._model.parameters())
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    assert not (tmp_path/'formal_training.guard.json').exists()


def test_root_descriptor_uses_materialized_commitment_endpoint_without_parent():
    snapshot,descriptor=fixture()
    descriptor=dict(descriptor,source_ring='middle',source_slot=2,parent_prefix_id=None)
    snapshot.active={'a':SimpleNamespace(end_time=65.)}
    state={'nodes':{'n':dict(materialized=True,soc=.63,ring='middle',slot=2,end_time=65.,depth=0,peak=12.)}}
    feature=FeatureEncoder().encode(snapshot,[descriptor],state)[0]
    endpoint=slot_state('middle',2,65.,snapshot.config).p
    assert feature['query'][44:50:2] == pytest.approx([v/900. for v in endpoint])
    assert feature['query'][32] == .63
    assert feature['query'][36] == 0.  # Root descriptor has no hypothetical parent.
    assert feature['query'][54] == pytest.approx(55./240.)
    assert feature['query'][60] == pytest.approx(55./240.)
    state['nodes']['n']['materialized']=False
    with pytest.raises(FeatureInputError,match='unmaterialized'):
        FeatureEncoder().encode(snapshot,[descriptor],state)

@pytest.mark.parametrize('output_name',['model','replacement_model'])
def test_direct_training_refuses_prior_heldout_access_before_inputs(tmp_path,monkeypatch,output_name):
    import terminal_marl.core.rotating_slots.stage3_learning.learning as learning
    (tmp_path/'heldout_access.json').write_text('{"accessed":true}',encoding='utf8')
    assert not (tmp_path/'formal_training.guard.json').exists()
    def forbidden(*args,**kwargs):
        raise AssertionError('heldout guard must precede configuration, input access and model construction')
    monkeypatch.setattr(learning,'_fixed_config',forbidden)
    monkeypatch.setattr(learning,'_read',forbidden)
    monkeypatch.setattr(learning,'build_model',forbidden)
    with pytest.raises(ValueError,match='heldout access already recorded'):
        learning.train_model(tmp_path/'unread-dataset.json',tmp_path/output_name,tmp_path/'unread-binding.json')
    assert not (tmp_path/output_name).exists()
    assert not (tmp_path/'formal_training.guard.json').exists()
