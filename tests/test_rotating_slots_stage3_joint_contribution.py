"""Pure bounded intervention fixtures; no simulator, solver, or training run."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import MappingProxyType, SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots.stage3_learning import labeling
from terminal_marl.core.rotating_slots.stage3_learning.features import SCHEMA_V2
from terminal_marl.core.rotating_slots.world import PlanningSnapshot


POLICY=labeling.JOINT_CONTRIBUTION_POLICY
KWARGS=dict(root_id='root',split='train',fingerprint={'objective_id':'objective','label_policy_id':POLICY})


def fixture(*,equal_delta=False):
    descriptors=[dict(descriptor_id=did,snapshot_id='snapshot',source_node_id='node',aircraft_id='aircraft',
        parent_prefix_id='prefix',construction_stage='alternative') for did in ('a','b')]
    solutions=[];increments=[]
    for did,included,removed in (('a',[0,-3,10],[0,-2,10]),('b',[0,-1,8],[0,0 if equal_delta else 2,8])):
        common=dict(certified=True,catalog_id='catalog',budget_settings={'seconds':3},method='rh_cp_sat',status='FEASIBLE')
        inc=dict(common,required_descriptor=did,descriptor_ids=[did],objective=included,selected_signatures=[did])
        rem=dict(common,required_descriptor=None,excluded_descriptor=did,descriptor_ids=[],objective=removed,
            selected_signatures=[],removed_signatures=[did],counterfactual_catalog_id='without:'+did)
        for row in (inc,rem):row['solution_id']=labeling.canonical_solution_id(row)
        solutions.extend((inc,rem));increments.append(labeling.incremental_evidence(did,inc,rem))
    return descriptors,solutions,increments


def derive(descriptors,solutions,increments,*,policy=POLICY,facts=()):
    return labeling.derive_pairs(descriptors,facts,[],conditional_attempts=solutions,context_id='snapshot',
        label_policy=policy,increments=increments,**KWARGS)


def joint(descriptors,solutions,increments,*,facts=()):
    return labeling.joint_contribution_pairs(descriptors,facts,solutions,increments,**KWARGS)


def test_observed_contribution_replaces_reverse_quality_with_four_exact_bindings():
    descriptors,solutions,increments=fixture()
    original=deepcopy((descriptors,solutions,increments))
    pairs=derive(descriptors,solutions,increments)
    assert len(pairs)==1
    pair=pairs[0]
    assert (pair['preferred_descriptor'],pair['other_descriptor'])==('b','a')
    assert pair['label_source']=='observed_bounded_joint_contribution_preference'
    assert pair['label_policy_id']==POLICY
    assert pair['catalog_id']=='catalog' and pair['budget_settings']=={'seconds':3}
    assert len(pair['evidence_refs'])==len(set(pair['evidence_refs']))==4
    by_ref={row['solution_id']:row for row in solutions}
    for evidence in pair['contribution_evidence']:
        inc=by_ref[evidence['included_solution_ref']];rem=by_ref[evidence['removed_solution_ref']]
        assert inc['required_descriptor']==rem['excluded_descriptor']==evidence['descriptor_id']
        assert inc['objective']==evidence['included_objective']
        assert rem['objective']==evidence['removed_objective']
        assert evidence['delta']==[r-i for i,r in zip(inc['objective'],rem['objective'])]
    assert pair['global_oracle'] is False and pair['causal_guarantee'] is False
    assert (descriptors,solutions,increments)==original


def test_old_policy_keeps_original_inclusion_quality_direction():
    descriptors,solutions,increments=fixture()
    old=derive(descriptors,solutions,increments,policy=None)
    assert len(old)==1 and old[0]['preferred_descriptor']=='a'
    assert old[0]['label_source']=='bounded_conditional_certified_objective_preference'
    assert old[0]['label_policy_id']==labeling.DEFAULT_LABEL_POLICY
    assert not labeling.supplementary_pairs(descriptors,[],solutions,increments,**KWARGS)


@pytest.mark.parametrize('key',['snapshot_id','source_node_id','aircraft_id','parent_prefix_id','construction_stage'])
def test_contribution_cannot_cross_comparison_group(key):
    descriptors,solutions,increments=fixture()
    descriptors[1][key]='other'
    assert not derive(descriptors,solutions,increments)


@pytest.mark.parametrize('key,value',[('catalog_id','other-catalog'),('budget_settings',{'seconds':4})])
def test_contribution_cannot_cross_catalog_or_budget(key,value):
    descriptors,solutions,increments=fixture()
    for row in solutions[2:]:row[key]=value
    increments[1]=labeling.incremental_evidence('b',solutions[2],solutions[3])
    assert not derive(descriptors,solutions,increments)


def test_unknown_untested_and_multi_signature_removal_never_make_contribution():
    descriptors,solutions,increments=fixture()
    solutions[2].update(certified=False,status='UNKNOWN')
    increments[1]=labeling.incremental_evidence('b',solutions[2],solutions[3])
    assert increments[1]['contribution_mask'] is False
    assert not derive(descriptors,solutions,increments,facts=[dict(descriptor_id='b',outcome='UNKNOWN')])
    descriptors,solutions,increments=fixture()
    solutions[3]['removed_signatures']=['b','descendant']
    increments[1]=labeling.incremental_evidence('b',solutions[2],solutions[3])
    assert increments[1]['contribution_mask'] is False
    assert not joint(descriptors,solutions,increments)
    assert not joint(descriptors,solutions,increments[:1])


def test_equal_delta_requires_complete_same_scope_unique_cost_observations():
    descriptors,solutions,increments=fixture(equal_delta=True)
    assert not joint(descriptors,solutions,increments)
    facts=[dict(descriptor_id=did,cost_complete=True,cost_scope='complete-scope',wall_s=wall,
        features_before_query_ref='query:'+did) for did,wall in (('a',.2),('b',.1))]
    pairs=derive(descriptors,solutions,increments,facts=facts)
    assert len(pairs)==1 and pairs[0]['preferred_descriptor']=='b'
    assert pairs[0]['label_source']=='observed_bounded_joint_contribution_measured_cost_preference'
    assert len(pairs[0]['evidence_refs'])==4
    assert pairs[0]['cost_evidence_refs']==['query:a','query:b']
    for key,value in (('cost_complete',False),('cost_scope','different'),('wall_s',None),('wall_s',float('nan')),
                      ('features_before_query_ref',None),('features_before_query_ref','query:a')):
        changed=deepcopy(facts);changed[1][key]=value
        assert not joint(descriptors,solutions,increments,facts=changed)
    assert not joint(descriptors,solutions,increments,facts=facts+[facts[0]])


@pytest.mark.parametrize('corruption',['duplicate_ref','wrong_intervention','forged_delta','wrong_ref_order'])
def test_four_references_are_uniquely_bound_to_the_actual_interventions(corruption):
    descriptors,solutions,increments=fixture()
    if corruption=='duplicate_ref':
        solutions[3]['solution_id']=solutions[1]['solution_id']
        increments[1]=labeling.incremental_evidence('b',solutions[2],solutions[3])
    elif corruption=='wrong_intervention':solutions[3]['excluded_descriptor']='a'
    elif corruption=='forged_delta':increments[1]['increment']=[0,100,0]
    else:increments[1]['evidence_refs'].reverse()
    assert not joint(descriptors,solutions,increments)


def test_unrelated_quality_and_explicit_invalid_pairs_are_preserved():
    descriptors,solutions,increments=fixture()
    descriptors.append(dict(descriptors[0],descriptor_id='invalid'))
    facts=[dict(descriptor_id='invalid',outcome='SINGLE_INVALID_CERTIFIED',validity_label_mask=True)]
    references=[dict(certified=True,descriptor_ids=['a'],solution_id='reference')]
    pairs=labeling.derive_pairs(descriptors,facts,references,conditional_attempts=solutions,context_id='snapshot',
        label_policy=POLICY,increments=increments,**KWARGS)
    assert len(pairs)==2
    assert any(p['preferred_descriptor']=='a' and p['other_descriptor']=='invalid' and
        p['label_source']=='supported_vs_certified_single_invalid' for p in pairs)
    assert len({frozenset((p['preferred_descriptor'],p['other_descriptor'])) for p in pairs})==len(pairs)


def test_new_policy_saved_reference_reindex_preserves_contribution_semantics():
    descriptors,solutions,increments=fixture()
    labels=dict(root_id='root',ancestry_root_id='root',context_id='snapshot',split='train',fingerprint=KWARGS['fingerprint'],
        label_policy_id=POLICY,facts=[],reference_solutions=[],conditional_attempts=solutions,
        incremental_contribution_evidence=increments,preference_pairs=derive(descriptors,solutions,increments))
    original=deepcopy(labels)
    repaired=labeling.reindex_saved_labels(labels,descriptors)
    assert labels==original
    assert repaired['preference_pairs']==labels['preference_pairs']
    assert repaired['reindex_provenance']['no_new_solves']


def snapshot_fixture():
    empty=MappingProxyType({})
    return PlanningSnapshot(time=30.,version=7,config=None,aircraft=empty,active=empty,soft=empty,requests=empty,
        slots=(),pads=(),dependencies=empty,source_clear=empty,trajectories=empty,energy_ledger=empty)


def provenance_fixture():
    return dict(source_time=0.,effective_time=30.,version=7,
        projection=dict(method='project_visible_snapshot',revealed_inputs_only=True))


def test_planning_snapshot_provenance_is_copied_validated_and_read_only():
    snapshot=snapshot_fixture();provenance=provenance_fixture()
    saved=labeling._planning_snapshot_provenance(snapshot,provenance)
    provenance['projection']['method']='caller-edited'
    assert saved['projection']['method']=='project_visible_snapshot'
    assert snapshot.time==30. and snapshot.version==7
    with pytest.raises(FrozenInstanceError):snapshot.time=31.
    for change in (dict(source_time=31.),dict(effective_time=29.),dict(effective_time=float('nan')),
                   dict(version=8),dict(projection={})):
        with pytest.raises(ValueError):labeling._planning_snapshot_provenance(snapshot,dict(provenance_fixture(),**change))
    with pytest.raises(ValueError):labeling._planning_snapshot_provenance(snapshot,None)
    with pytest.raises(ValueError):labeling._planning_snapshot_provenance(None,provenance_fixture())
    with pytest.raises(ValueError):labeling._planning_snapshot_provenance(SimpleNamespace(time=30.,version=7),provenance_fixture())


def test_label_snapshot_uses_supplied_projection_without_restoring_or_extra_solves(tmp_path,monkeypatch):
    import terminal_marl.core.rotating_slots.stage2_planning as planning
    import terminal_marl.core.rotating_slots.stage3_learning.runtime as runtime_module
    from terminal_marl.core.rotating_slots.stage3_learning.campaign import contract_config
    snapshot=snapshot_fixture();provenance=provenance_fixture();seen=[]
    root=dict(root_id='projected-root',ancestry_root_id='original-root',split='train',snapshot_id='source-snapshot')
    labeling.atomic_json(tmp_path/'roots'/'projected-root'/'root.json',root)
    def forbidden(*args,**kwargs):raise AssertionError('restoring a source world is forbidden for supplied projection')
    monkeypatch.setattr(labeling,'restore_root',forbidden)
    class Runtime:
        def __init__(self,**kwargs):self._plans={}
    class Provider:
        def __init__(self,**kwargs):pass
        def iter_progressive_columns(self,s,budget,context):
            assert s is snapshot and context.snapshot is snapshot
            yield SimpleNamespace(columns={})
    class Scheduler:
        def plan(self,s,catalog,context,budget):
            assert s is snapshot and context.snapshot is snapshot
            seen.append(budget.settings())
            return SimpleNamespace(certified=False,evaluation=None,selected=(),objective=(0,),status='UNKNOWN')
    monkeypatch.setattr(runtime_module,'AdvisorRuntime',Runtime)
    monkeypatch.setattr(planning,'CandidateProvider',Provider)
    monkeypatch.setattr(planning,'EnergySlackScheduler',Scheduler)
    monkeypatch.setattr(planning,'RollingHorizonCPSATScheduler',Scheduler)
    binding={'fingerprint':KWARGS['fingerprint']}
    result=labeling.label_snapshot(tmp_path,'projected-root',binding,schema_id=SCHEMA_V2,label_policy=POLICY,
        planning_snapshot=snapshot,planning_snapshot_provenance=provenance)
    assert len(seen)==len(contract_config()['labeling']['reference_solvers'])
    assert result['status']=='DATA_INSUFFICIENT' and result['preference_pairs']==[]
    assert result['planning_snapshot_provenance']==provenance
    assert result['ancestry_root_id']=='original-root' and result['split']=='train'
    assert result['label_policy_id']==POLICY and result['fingerprint']==binding['fingerprint']
    assert result['feature_schema_id']==SCHEMA_V2
    assert snapshot.time==30. and snapshot.aircraft=={}
    with pytest.raises(ValueError,match='different label policy'):
        labeling.label_snapshot(tmp_path,'projected-root',binding,schema_id=SCHEMA_V2)
    with pytest.raises(ValueError,match='fingerprint label policy'):
        labeling.label_snapshot(tmp_path,'another',{'fingerprint':{'objective_id':'objective'}},schema_id=SCHEMA_V2,label_policy=POLICY)
