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
    stats=dict(mean=1.,std=1.,count=2,fit_split='train')
    diagnostics=cost_diagnostics(Constant(),selected,normalizer,stats)
    assert diagnostics['count']==2
    assert diagnostics['mae_log1p_ms']==pytest.approx(abs(math.log1p(200.)-2.))
    assert all(r['cost_adjusted_score']==pytest.approx(cost_aware_scores([r['descriptor_id']],[[r['preference_score'],0.,r['predicted_log1p_ms']]],stats)[r['descriptor_id']]) for r in diagnostics['predictions'])
    assert diagnostics['penalty_min']==pytest.approx(.1) and diagnostics['ordering_changed_contexts']==0


def test_training_admission_rank_or_complete_cost_excludes_validity_only():
    from terminal_marl.core.rotating_slots.stage3_learning.learning import eligible_training_groups
    base=dict(context=dict(split='train'),features={},pairs=[],facts={})
    rank=dict(base,pairs=[('a','b')])
    cost=dict(base,facts={'a':dict(outcome='UNKNOWN',wall_s=.2,cost_complete=True,cost_scope='GENERATOR_ENERGY_TERMINAL_VALIDATION_V2')})
    validity=dict(base,facts={'a':dict(outcome='SINGLE_VALID',wall_s=.1,cost_complete=False)})
    validation=dict(cost,context=dict(split='validation'))
    assert eligible_training_groups([rank,cost,validity,validation],schema_id=SCHEMA_V2)==[rank,cost]
    assert eligible_training_groups([rank,cost,validity],schema_id=SCHEMA_ID)==[rank]


def test_cost_diagnostics_reports_order_change_from_existing_vectors():
    from terminal_marl.core.rotating_slots.stage3_learning.learning import cost_diagnostics
    torch=pytest.importorskip('torch')
    snapshot,descriptor=fixture()
    features=FeatureEncoder(schema_id=SCHEMA_V2).encode(snapshot,[dict(descriptor,descriptor_id=did) for did in ('expensive','cheap')])
    normalizer=FeatureNormalizer(schema_id=SCHEMA_V2).fit([dict(split='train',features=features)])
    facts={did:dict(outcome='UNKNOWN',wall_s=.2,cost_complete=True,cost_scope='GENERATOR_ENERGY_TERMINAL_VALIDATION_V2') for did in ('expensive','cheap')}
    group=dict(context=dict(context_id='covered'),features={f['descriptor_id']:f for f in features},facts=facts)
    class ExistingVectors:
        calls=0
        def __call__(self,rows):
            self.calls+=1
            return torch.tensor([[1.,0.,6.] if f['descriptor_id']=='expensive' else [.9,0.,0.] for f in rows])
    model=ExistingVectors()
    result=cost_diagnostics(model,[group],normalizer,dict(mean=2.,std=1.,count=2,fit_split='train'))
    assert model.calls==2 and result['count']==2
    assert result['ordering_changed_contexts']==1 and result['comparable_order_context_count']==1
    assert result['penalty_clipped_count']==1
    assert result['penalty_min']==pytest.approx(-.2) and result['penalty_max']==pytest.approx(.3)


def test_cost_only_train_cannot_start_formal_ranking_fit(tmp_path):
    from terminal_marl.core.rotating_slots.stage3_learning.learning import train_model
    snapshot,descriptor=fixture();feature=FeatureEncoder(schema_id=SCHEMA_V2).encode(snapshot,[descriptor])[0]
    fp={'objective_id':'o'}
    contexts=[]
    for split in ('train','validation'):
        fact=dict(descriptor_id='d',context_id=split,ancestry_root_id=split,fingerprint=fp,
            outcome='UNKNOWN',wall_s=.1,cost_complete=True,cost_scope='GENERATOR_ENERGY_TERMINAL_VALIDATION_V2')
        contexts.append(dict(context_id=split,ancestry_root_id=split,split=split,fingerprint=fp,features=[feature,dict(feature,descriptor_id='e')],facts=[fact]))
    pair=dict(context_id='validation',ancestry_root_id='validation',split='validation',preferred_descriptor='d',other_descriptor='e',objective_id='o',
        label_source='bounded_conditional_certified_objective_preference',label_policy_id='bounded_reference_supported_pair_v1_0',evidence_refs=['saved'])
    result=train_model(dict(fingerprint=fp,contexts=contexts,preference_pairs=[pair]),tmp_path/'model',fp,schema_id=SCHEMA_V2)
    assert result['status']=='DATA_INSUFFICIENT' and result['formal_training_started'] is False
    assert not (tmp_path/'formal_training.guard.json').exists()
    assert result['train_pairs']==0 and result['validation_pairs']==1


def test_saved_reindex_uses_interventions_not_colliding_old_ids_and_is_immutable():
    from copy import deepcopy
    from terminal_marl.core.rotating_slots.stage3_learning.labeling import reindex_saved_labels,derive_pairs,canonical_solution_id
    descriptors=[dict(descriptor_id=d,snapshot_id='s',source_node_id='n',aircraft_id='a',parent_prefix_id=None,construction_stage='alternative') for d in ('a','b')]
    conditional=[];increments=[]
    for did,objective,removed_objective in [('a',[0,-2],[0,-1]),('b',[0,-1],[0,0])]:
        common=dict(certified=True,catalog_id='catalog',budget_settings={'seconds':3},method='rh_cp_sat',status='FEASIBLE')
        inc=dict(common,required_descriptor=did,descriptor_ids=[did],objective=objective,selected_signatures=[did],solution_id='old:'+did)
        rem=dict(common,required_descriptor=None,excluded_descriptor=did,descriptor_ids=[],objective=removed_objective,selected_signatures=[],
                 removed_signatures=[did],counterfactual_catalog_id='without:'+did,solution_id='COLLIDING_OLD_REMOVAL_ID')
        conditional.extend([inc,rem]);increments.append(incremental_evidence(did,inc,rem))
    fp={'objective_id':'o'}
    pairs=derive_pairs(descriptors,[],[],conditional_attempts=conditional,context_id='s',root_id='root',split='train',fingerprint=fp)
    labels=dict(root_id='root',ancestry_root_id='root',context_id='s',split='train',fingerprint=fp,facts=[],reference_solutions=[deepcopy(conditional[1])],
                conditional_attempts=conditional,incremental_contribution_evidence=increments,preference_pairs=pairs)
    original=deepcopy(labels);repaired=reindex_saved_labels(labels,descriptors)
    assert labels==original
    assert repaired['conditional_attempts'][1]['solution_id']!=repaired['conditional_attempts'][3]['solution_id']
    assert repaired['reference_solutions'][0]['solution_id']==repaired['conditional_attempts'][1]['solution_id']
    assert [(p['preferred_descriptor'],p['other_descriptor'],p['label_source']) for p in repaired['preference_pairs']]==[(p['preferred_descriptor'],p['other_descriptor'],p['label_source']) for p in pairs]
    assert repaired['incremental_contribution_evidence'][0]['increment']==increments[0]['increment']
    assert repaired['reindex_provenance']['no_new_solves']
    assert canonical_solution_id(conditional[1])!=canonical_solution_id(dict(conditional[1],excluded_descriptor='other'))
