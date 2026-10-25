import math

import pytest

from terminal_marl.core.rotating_slots.stage3_learning.features import SCHEMA_ID,SCHEMA_V2,FeatureEncoder,migrate_v1_feature_to_v2
from terminal_marl.core.rotating_slots.stage3_learning.model import build_model,FeatureNormalizer,cost_aware_scores,LearnedAdvisor,FeatureInputError,ARCHITECTURE_ID
from terminal_marl.core.rotating_slots.stage3_learning.learning import auxiliary_targets,_loss,fit_cost_statistics
from terminal_marl.core.rotating_slots.stage3_learning.labeling import complete_cost_facts,incremental_evidence,supplementary_pairs
from terminal_marl.core.rotating_slots.stage3_learning.data import merge_v2_datasets,query_observations
from tests.test_rotating_slots_stage3_model import fixture


def test_cost_changes_ranking_and_fixed_clip_ties():
    stats=dict(fit_split='train',mean=2.,std=1.,count=4)
    scores=cost_aware_scores(['expensive','cheap'],[[1.,0.,5.],[.9,0.,0.]],stats)
    assert scores['cheap']>scores['expensive']
    assert cost_aware_scores(['a','b'],[[1.,0.,2.],[1.,0.,2.]],stats)=={'a':1.,'b':1.}
    with pytest.raises(FeatureInputError,match='NONFINITE_SCORE'):
        cost_aware_scores(['a'],[[1.,0.,math.nan]],stats)


def test_v2_network_identical_parameter_count_strict_schema_and_migration():
    torch=pytest.importorskip('torch')
    v1,v2=build_model(),build_model(SCHEMA_V2)
    assert sum(p.numel() for p in v1.parameters())==sum(p.numel() for p in v2.parameters())==119555
    s,d=fixture();f=FeatureEncoder().encode(s,[d])[0];m=migrate_v1_feature_to_v2(f)
    assert m['query'][:38]==f['query'][:38]
    assert m['query'][48:]==[0.,0.]*9
    assert v2([m]).shape==(1,3)
    with pytest.raises(FeatureInputError):v1([m])


def test_v2_masks_partial_cost_and_trains_auxiliary_only_group():
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    s,d=fixture();f=FeatureEncoder(schema_id=SCHEMA_V2).encode(s,[d])[0]
    normalizer=FeatureNormalizer(schema_id=SCHEMA_V2).fit([dict(split='train',features=[f])])
    partial=dict(outcome='UNKNOWN',wall_s=.2,cost_complete=False,cost_scope='GENERATOR_ENERGY_TERMINAL_VALIDATION_V2')
    assert auxiliary_targets(partial,schema_id=SCHEMA_V2)==(None,None)
    fact=dict(partial,cost_complete=True)
    group=dict(features={'d':f},pairs=[],facts={'d':fact})
    model=build_model(SCHEMA_V2)
    loss,rank=_loss(model,[group],normalizer)
    assert loss is not None and rank is None
    loss.backward()
    assert model.score_head[-1].weight.grad[2].abs().sum()>0
    assert fit_cost_statistics([group])['mean']==pytest.approx(math.log1p(200.))


def test_cost_components_join_exact_query_no_unknown_negative():
    events=[dict(record_type='FactualResult',descriptor_id='d',features_before_query_ref='q1',wall_s=.1,trajectory_hash='t',outcome='UNKNOWN'),
        dict(record_type='FactualResult',descriptor_id='d',features_before_query_ref='q2',wall_s=.2,trajectory_hash='u',outcome='UNKNOWN'),
        dict(record_type='SingleVehicleEvidence',descriptor_id='d',features_before_query_ref='q1',wall_s=.3,outcome='SINGLE_VALID',validity_label_mask=True),
        dict(record_type='ColumnValidationCost',descriptor_id='d',features_before_query_ref='q1',wall_s=.4)]
    facts=complete_cost_facts(events)
    assert facts[0]['wall_s']==pytest.approx(.8) and facts[0]['cost_complete']
    assert facts[1]['wall_s'] is None and not facts[1]['cost_complete']
    assert auxiliary_targets(facts[1],schema_id=SCHEMA_V2)==(None,None)
    assert len(query_observations(events))==2


def test_strict_v2_checkpoint_and_cost_metadata(tmp_path):
    torch=pytest.importorskip('torch');s,d=fixture();f=FeatureEncoder(schema_id=SCHEMA_V2).encode(s,[d])[0]
    n=FeatureNormalizer(schema_id=SCHEMA_V2).fit([dict(split='train',features=[f])])
    model=build_model(SCHEMA_V2)
    path=tmp_path/'model.pt'
    torch.save(dict(architecture_id=ARCHITECTURE_ID,schema_id=SCHEMA_V2,fingerprint={'objective_id':'o'},state_dict=model.state_dict(),normalizer=n.as_dict(),cost_statistics=dict(fit_split='train',count=1,mean=1.,std=0.)),path)
    with pytest.raises(FeatureInputError,match='FINGERPRINT'):LearnedAdvisor(path,{'objective_id':'o'}).rank([f])
    result=LearnedAdvisor(path,{'objective_id':'o'},schema_id=SCHEMA_V2).rank([f])
    assert result.ordered_ids==('d',) and result.model_metadata['cost_coefficient']==.1


def test_paired_increment_certification_and_ancestry_merge_masks():
    inc=dict(certified=True,catalog_id='c',budget_settings={'seconds':3},required_descriptor='d',descriptor_ids=['d'],objective=[0,-2,1],solution_id='i')
    rem=dict(certified=True,catalog_id='c',budget_settings={'seconds':3},excluded_descriptor='d',descriptor_ids=[],objective=[0,-1,1],solution_id='r',removed_signatures=['candidate'],counterfactual_catalog_id='minus')
    evidence=incremental_evidence('d',inc,rem)
    assert evidence['contribution_mask'] and evidence['increment']==[0,1,0]
    assert not incremental_evidence('d',inc,dict(rem,certified=False))['contribution_mask']
    assert not incremental_evidence('d',inc,dict(rem,removed_signatures=['one','duplicate']))['contribution_mask']
    s,d=fixture();f=FeatureEncoder().encode(s,[d])[0]
    context=dict(context_id='c',ancestry_root_id='root',split='train',features=[f],facts=[dict(descriptor_id='d',wall_s=.1)])
    dataset=dict(fingerprint={},contexts=[context],preference_pairs=[])
    result=merge_v2_datasets([dataset],{})
    assert result['contexts'][0]['features'][0]['schema_id']==SCHEMA_V2
    assert not result['contexts'][0]['facts'][0]['cost_complete']
    with pytest.raises(ValueError,match='duplicate'):merge_v2_datasets([dataset,dataset],{})


def test_supplementary_labels_only_certified_quality_ties_and_comparable_cost():
    descriptors=[dict(descriptor_id=did,snapshot_id='s',source_node_id='n',aircraft_id='a',parent_prefix_id=None,construction_stage='alternative') for did in ('a','b')]
    solutions=[dict(required_descriptor=did,certified=True,catalog_id='c',budget_settings={'seconds':3},objective=[0,-1,1]) for did in ('a','b')]
    increments=[dict(descriptor_id=did,contribution_mask=True,increment=[0,1,0],evidence_refs=['i'+did,'r'+did]) for did in ('a','b')]
    facts=[dict(descriptor_id=did,cost_complete=True,cost_scope='GENERATOR_ENERGY_TERMINAL_VALIDATION_V2',wall_s=wall,features_before_query_ref='q'+did) for did,wall in (('a',.1),('b',.2))]
    kwargs=dict(root_id='root',split='train',fingerprint={'objective_id':'o'})
    pair=supplementary_pairs(descriptors,facts,solutions,increments,**kwargs)[0]
    assert pair['preferred_descriptor']=='a' and pair['label_source']=='equal_quality_measured_cost_preference'
    assert not supplementary_pairs(descriptors,facts,[solutions[0],dict(solutions[1],objective=[0,-2,1])],increments,**kwargs)
    assert not supplementary_pairs(descriptors,[dict(facts[0],cost_complete=False),facts[1]],solutions,increments,**kwargs)
    inc=[increments[0],dict(increments[1],increment=[0,2,0])]
    assert supplementary_pairs(descriptors,facts,solutions,inc,**kwargs)[0]['preferred_descriptor']=='b'


def test_cost_diagnostics_include_cost_only_validation_groups():
    from terminal_marl.core.rotating_slots.stage3_learning.learning import validation_cost_groups,cost_diagnostics
    torch=pytest.importorskip('torch')
    snapshot,descriptor=fixture()
    feature=FeatureEncoder(schema_id=SCHEMA_V2).encode(snapshot,[descriptor])[0]
    normalizer=FeatureNormalizer(schema_id=SCHEMA_V2).fit([dict(split='train',features=[feature])])
    fact=dict(outcome='UNKNOWN',wall_s=.2,cost_complete=True,cost_scope='GENERATOR_ENERGY_TERMINAL_VALIDATION_V2')
    cost_only=dict(context=dict(split='validation'),features={'d':feature},pairs=[],facts={'d':fact})
    rank_group=dict(cost_only,pairs=[('d','other')])
    train=dict(cost_only,context=dict(split='train'))
    partial=dict(cost_only,facts={'d':dict(fact,cost_complete=False)})
    selected=validation_cost_groups([cost_only,rank_group,train,partial])
    assert selected==[cost_only,rank_group]
    class Constant:
        def __call__(self,features):return torch.tensor([[0.,0.,2.] for _ in features])
    diagnostics=cost_diagnostics(Constant(),selected,normalizer,dict(mean=1.))
    assert diagnostics['count']==2
    assert diagnostics['mae_log1p_ms']==pytest.approx(abs(math.log1p(200.)-2.))
