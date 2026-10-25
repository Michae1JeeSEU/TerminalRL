from dataclasses import replace
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots.stage3_learning.runtime import AdvisorRuntime
from terminal_marl.core.rotating_slots.stage3_learning.online_observer import OnlineObserver
from terminal_marl.core.rotating_slots.stage3_learning.features import FeatureEncoder, SCHEMA_V2
from tests.test_rotating_slots_stage3_runtime import setup, ReverseAdvisor, CheapEncoder
from tests.test_rotating_slots_stage3_model import fixture


def test_online_filters_before_copy_and_does_not_encode_selected():
    class Uncopyable:
        def __deepcopy__(self,memo):raise AssertionError('full trace copied')
    class Counting(CheapEncoder):
        calls=0
        def encode(self,*args,**kwargs):
            self.calls+=1
            return super().encode(*args,**kwargs)
    _,snapshot,budget,runtime,_,_,ds=setup(trace_mode='online')
    runtime.feature_encoder=Counting()
    runtime.emit(dict(record_type='FeaturesBeforeQuery',features=Uncopyable()))
    token=runtime.record_before(snapshot,ds[0],budget,{})
    runtime.record_after(ds[0],token,True,None,budget,{})
    assert runtime.feature_encoder.calls==0
    assert not any(r['record_type'] in ('FeaturesBeforeQuery','WorkDescriptor','ColumnAncestry') for r in runtime.traces)
    assert runtime.coverage['attempted']==1


def test_online_aggregate_matches_full_event_stream():
    events=[dict(record_type='WorkDescriptor',descriptor_id='d'),
        dict(record_type='FeaturesBeforeQuery',features=[1,2,3]),
        dict(record_type='FactualResult',descriptor_id='d',wall_s=.2,physical_attempts=2,outcome='UNKNOWN'),
        dict(record_type='SingleVehicleEvidence',descriptor_id='d',wall_s=.1,outcome='SINGLE_VALID'),
        dict(record_type='ColumnValidationCost',descriptor_id='d',wall_s=.3),
        dict(record_type='Rank',policy='learned',scores={'d':4}),
        dict(record_type='Coverage',counts={'attempted':1}),dict(record_type='Fallback',reason='MODEL_MISSING')]
    full,online=OnlineObserver(),OnlineObserver()
    runtime=AdvisorRuntime('RECORD',observer=online,trace_mode='online')
    for event in events:
        full(event);runtime.emit(event)
    assert full.summary()==online.summary()
    assert online.cost_counts()['physical_attempts']==2
    assert online.cost_counts()['generation_wall_s']==.2


def test_three_consumption_cache_ignores_attempts_but_invalidates_additions_and_source():
    class Counting(ReverseAdvisor):
        calls=0
        def rank(self,*args):
            self.calls+=1
            return super().rank(*args)
    advisor=Counting()
    _,snapshot,budget,runtime,a,node,ds=setup(advisor=advisor,trace_mode='online',feature_schema=SCHEMA_V2)
    runtime.complete_initial_coverage()
    # Four descriptors allow three learned consumptions before singleton handling.
    ds.append(runtime.descriptor(snapshot,a,node,3,0.,35.,'progress'))
    ordered=runtime.order(snapshot,ds,budget)
    budget.attempts[a.id]+=1
    remaining=runtime.order(snapshot,ordered[1:],budget)
    assert advisor.calls==1 and runtime.coverage['rank_cache_hits']==1
    runtime.nodes[remaining[0].source_node_id]['soc']-=.01
    runtime.order(snapshot,remaining[1:]+[ordered[0]],budget)
    assert advisor.calls==2
    runtime._post_initial=3
    runtime.order(snapshot,ds,budget)
    assert runtime._rank_cache is None
    with pytest.raises(ValueError,match='registry'):
        runtime.order(snapshot,[replace(ds[0],snapshot_id='foreign'),ds[1]],budget)


def test_v2_context_masks_known_empty_and_query_geometry_in_entity_sets():
    snapshot,d=fixture()
    encoder=FeatureEncoder(schema_id=SCHEMA_V2)
    missing=encoder.encode(snapshot,[d])[0]
    known=encoder.encode(snapshot,[d],dict(retained_context={'a':dict(full_count=0,immediate_count=0,delayed_count=0,pad_count=0,window_count=0,directions={},missing_coverage_count=3)},known_conflicts={'a':0}))[0]
    assert len(known['query'])==66 and known['schema_id']==SCHEMA_V2
    assert missing['query'][50:60]==[0.,0.]*5
    assert known['query'][50:60]==[0.,1.]*5
    assert known['query'][60:66]==[0.,1.,0.,1.,3.,1.]
    assert known['sets']==missing['sets']


def test_context_coverage_epoch_not_changed_by_duplicate_multiplicity():
    _,snapshot,budget,runtime,a,node,ds=setup()
    p=SimpleNamespace(target_ring=None,target_slot=None,pad_id=0,start_time=0.,end_time=35.)
    c=SimpleNamespace(kind='MOVE',plans=(p,),downstream={})
    context=SimpleNamespace(observed_definite_conflicts=[])
    runtime.update_retained_context(snapshot,{a.id:[c]},{},context)
    epoch=runtime._coverage_epoch
    runtime.update_retained_context(snapshot,{a.id:[c,c]},{},context)
    assert runtime._coverage_epoch==epoch
    assert runtime.retained_context[a.id]['directions'][repr(('pad',0))]==2
    assert runtime.known_conflicts[a.id]==0


def test_cache_three_consumptions_and_unexpected_removal_reorder():
    class Counting(ReverseAdvisor):
        calls=0
        def rank(self,*args):
            self.calls+=1
            return super().rank(*args)
    advisor=Counting()
    _,s,b,r,a,n,ds=setup(advisor=advisor,feature_schema=SCHEMA_V2)
    r.complete_initial_coverage()
    ds += [r.descriptor(s,a,n,t,0.,35.,'progress') for t in (3,4)]
    one=r.order(s,ds,b)
    r.parents['unrelated']=dict(materialized=True,soc=.1)
    two=r.order(s,one[1:],b)
    three=r.order(s,two[1:],b)
    assert advisor.calls==1 and r.coverage['rank_cache_hits']==2
    r.order(s,three[1:],b)
    assert advisor.calls==2
    r._rank_cache=None
    one=r.order(s,ds,b)
    r.order(s,one[2:],b)  # An external removal beyond the selected item.
    assert advisor.calls==4


def test_v1_default_never_reuses_rank_and_online_keeps_submission_evidence():
    class Counting(ReverseAdvisor):
        calls=0
        def rank(self,*args):self.calls+=1;return super().rank(*args)
    advisor=Counting()
    _,s,b,r,_,_,ds=setup(advisor=advisor)
    r.complete_initial_coverage()
    ranked=r.order(s,ds,b);r.order(s,ranked[1:],b)
    assert advisor.calls==2
    online=AdvisorRuntime('RECORD',trace_mode='online')
    online.emit(dict(record_type='PlanningSelection',certified=True,objective=[0,-1],selected_column_ids=['c'],
        submitted_trajectory_hashes=['t'],commit_status='ACCEPTED'))
    assert online.traces[-1]['submitted_trajectory_hashes']==['t']
    assert online.traces[-1]['commit_status']=='ACCEPTED'


def test_progressive_v2_portable_observer_serializes_retained_context_and_costs(tmp_path):
    """Exercise the real collection sink after retention, through parse and cost join."""
    from terminal_marl.core.rotating_slots import stage2_planning as sp
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    from terminal_marl.core.rotating_slots.stage3_learning.runtime import scale_candidate_budget
    from terminal_marl.core.rotating_slots.stage3_learning.data import PortableObserver,read_jsonl
    from terminal_marl.core.rotating_slots.stage3_learning.labeling import complete_cost_facts
    world=RotatingSlotWorld();world.seed_hold('a','inner',0,.9)
    snapshot=world.build_planning_snapshot()
    observer=PortableObserver(tmp_path/'raw')
    runtime=AdvisorRuntime('RECORD',feature_schema=SCHEMA_V2,trace_mode='collection',observer=observer)
    provider=sp.CandidateProvider(planning_profile=sp.PLANNING_PROFILE_DOWNSTREAM,advisor_runtime=runtime)
    budget=sp.PlanningBudget.for_profile('quality_v5');scale_candidate_budget(budget,.5)
    budget.candidate_active_seconds=0.
    context=sp.EvaluationContext(snapshot,budget=budget,reuse_prefix_results=True)
    batches=provider.iter_progressive_columns(snapshot,budget,context)
    with budget.candidate_work():catalog=next(batches)
    batches.close()
    features=read_jsonl(tmp_path/'raw'/'features.jsonl')
    facts=read_jsonl(tmp_path/'raw'/'facts.jsonl')
    descriptors={d['descriptor_id']:d for d in read_jsonl(tmp_path/'raw'/'descriptors.jsonl')}
    assert features and facts and any(c.kind=='MOVE' for c in catalog.columns['a'])
    covered=[f for f in features if f['search_state']['retained_context']['a']['directions']]
    assert covered, 'a later query must serialize nonempty retained directions'
    for observation in covered:
        descriptor=descriptors[observation['descriptor_id']]
        directions=observation['search_state']['retained_context']['a']['directions']
        assert all(isinstance(key,str) for key in directions)
        direction=('pad',descriptor['target_id'])
        assert observation['features']['query'][62]==directions.get(repr(direction),0)
        assert observation['features']['query'][63]==1.
    by_ref={f['features_before_query_ref']:f for f in features}
    validations=[r for r in facts if r['record_type']=='ColumnValidationCost']
    assert validations
    for row in validations:
        assert by_ref[row['features_before_query_ref']]['descriptor_id']==row['descriptor_id']
    complete=[f for f in complete_cost_facts(facts) if f.get('trajectory_hash') is not None and f['cost_complete']]
    assert complete
    assert all(f['features_before_query_ref'] in by_ref and f['column_validation_wall_s']>=0. for f in complete)
