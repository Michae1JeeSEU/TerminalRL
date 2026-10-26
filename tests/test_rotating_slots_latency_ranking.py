"""Bounded comparable ranking and candidate/solver stop separation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots import stage2_planning as sp
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.stage3_learning.features import FeatureEncoder, SCHEMA_V2
from terminal_marl.core.rotating_slots.stage3_learning.advisor import RankResult
from terminal_marl.core.rotating_slots.stage3_learning.runtime import AdvisorRuntime
from tests.test_rotating_slots_stage3_runtime import setup, ReverseAdvisor
from tests.test_rotating_slots_stage3_model import fixture
from tests.test_rotating_slots_stage3_online_catalog import fake_move


class CountingAdvisor(ReverseAdvisor):
    def __init__(self): self.batches=[]

    def rank(self, features, budget):
        self.batches.append([f['descriptor_id'] for f in features])
        ranked=super().rank(features,budget)
        return replace(ranked,scores={did:score*1.e100 for did,score in ranked.scores.items()})


def grouped_setup(count=5):
    advisor=CountingAdvisor()
    world,s,b,r,a,n,ds=setup(advisor=advisor,feature_schema=SCHEMA_V2,
                            ranking_scope='comparable_group_v1')
    r.feature_encoder=FeatureEncoder(schema_id=SCHEMA_V2)
    r.complete_initial_coverage()
    ds.extend(r.descriptor(s,a,n,t,0.,35.,'progress') for t in range(3,count))
    return world,s,b,r,a,n,ds,advisor


def with_parent(runtime, descriptor, parent):
    runtime.parents[parent]=dict(runtime.nodes[descriptor.source_node_id])
    d=replace(descriptor,parent_prefix_id=parent,descriptor_id=descriptor.descriptor_id+':'+parent)
    runtime._descriptors[d.descriptor_id]=d
    return d


def test_cross_prefix_extreme_scores_only_reorder_original_group_positions():
    _,s,b,r,_,_,ds,advisor=grouped_setup(4)
    left=[with_parent(r,d,'left') for d in ds[:2]]
    right=[with_parent(r,d,'right') for d in ds[2:]]
    incoming=[left[0],right[0],left[1],right[1]]
    ordered=r.order(s,incoming,b)
    # ReverseAdvisor's largest score belongs to the right group. The first
    # deterministic position nevertheless stays in the left prefix group.
    assert ordered==[left[1],right[1],left[0],right[0]]
    assert [d.parent_prefix_id for d in ordered]==[d.parent_prefix_id for d in incoming]
    assert len(advisor.batches)==1


def test_same_group_three_pops_reuse_then_reinfer_without_extra_cache():
    _,s,b,r,_,_,ds,advisor=grouped_setup(6)
    one=r.order(s,ds,b)
    two=r.order(s,one[1:],b)
    three=r.order(s,two[1:],b)
    assert len(advisor.batches)==1 and r.coverage['rank_cache_hits']==2
    r.order(s,three[1:],b)
    assert len(advisor.batches)==2
    assert set(r._rank_cache)=={'groups'}


@pytest.mark.parametrize('change',('candidate','source','coverage','duplicate','conflict'))
def test_comparable_cache_invalidates_real_context_changes(change):
    _,s,b,r,a,n,ds,advisor=grouped_setup(5)
    r.retained_context[a.id]={'directions':{repr(('pad',0)):1},'full_count':1}
    ordered=r.order(s,ds,b)
    remaining=ordered[1:]
    if change=='candidate': remaining.append(r.descriptor(s,a,n,8,0.,35.,'progress'))
    elif change=='source': r.nodes[remaining[0].source_node_id]['soc']-=.01
    elif change=='coverage': r.retained_context[a.id]['full_count']+=1
    elif change=='duplicate': r.retained_context[a.id]['directions'][repr(('pad',0))]+=1
    elif change=='conflict': r.known_conflicts[a.id]=1
    r.order(s,remaining,b)
    assert len(advisor.batches)==2


def test_cached_parent_is_validated_at_each_pop():
    _,s,b,r,_,_,ds,advisor=grouped_setup(5)
    ds=[with_parent(r,d,'parent') for d in ds]
    ordered=r.order(s,ds,b)
    r.parents['parent']['materialized']=False
    with pytest.raises(ValueError,match='unmaterialized'):
        r.order(s,ordered[1:],b)
    assert len(advisor.batches)==1


def test_real_feature_dependencies_ignore_unrelated_coverage_but_keep_budget_drift_bounded():
    _,s,b,r,a,_,ds,advisor=grouped_setup(6)
    r.feature_encoder=FeatureEncoder(schema_id=SCHEMA_V2)
    b.candidate_active_seconds=0.
    one=r.order(s,ds,b)
    scored_fraction=r.traces[-1]['budget_fraction_at_score']
    r.retained_context['unrelated']={'full_count':1,'directions':{}}
    r.known_conflicts['unrelated']=2
    r._coverage_epoch+=1
    b.candidate_active_seconds=.2
    two=r.order(s,one[1:],b)
    assert len(advisor.batches)==1
    assert r.traces[-1]['budget_fraction_at_score']==scored_fraction
    assert r.traces[-1]['budget_fraction_at_pop']<scored_fraction
    assert r.traces[-1]['budget_fraction_drift']>0
    r.retained_context[a.id]={'full_count':1,'directions':{}}
    r.order(s,two[1:],b)
    assert len(advisor.batches)==2
    assert r.coverage['rank_cache_miss:FEATURE_CONTEXT_CHANGED']==1


def test_visible_content_binding_computed_once_per_order_not_per_group_or_pop():
    _,s,b,r,_,_,ds,advisor=grouped_setup(6)
    class CountingEncoder(FeatureEncoder):
        calls=0
        def _content_key(self,snapshot):
            self.calls+=1
            return super()._content_key(snapshot)
    encoder=CountingEncoder(schema_id=SCHEMA_V2);r.feature_encoder=encoder
    left=[with_parent(r,d,'left') for d in ds[:3]]
    right=[with_parent(r,d,'right') for d in ds[3:]]
    incoming=[d for pair in zip(left,right) for d in pair]
    ordered=r.order(s,incoming,b)
    # One dependency binding plus encode's existing independent content guard.
    assert encoder.calls==2 and len(advisor.batches)==1
    encoder.calls=0
    r.order(s,ordered[1:],b)
    assert encoder.calls==1 and len(advisor.batches)==1


def test_score_fraction_is_exact_encoder_input_and_pop_fraction_is_after_inference(monkeypatch):
    from tests.test_stage2_quality_budget import Clock
    _,s,b,r,_,_,ds,advisor=grouped_setup(4)
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    b.wall_start=0.;b.candidate_active_seconds=0.;b._candidate_phase_started=0.
    class ObservedEncoder(FeatureEncoder):
        def _content_key(self,snapshot):
            clock.now+=.1
            return super()._content_key(snapshot)
        def encode(self,snapshot,descriptors,observed_search_state=None):
            self.fraction=observed_search_state['remaining_budget_fraction']
            return super().encode(snapshot,descriptors,observed_search_state)
    encoder=ObservedEncoder(schema_id=SCHEMA_V2);r.feature_encoder=encoder
    original=advisor.rank
    def rank(features,budget):
        clock.now+=.2
        return original(features,budget)
    monkeypatch.setattr(advisor,'rank',rank)
    r.order(s,ds,b)
    row=r.traces[-1]
    assert row['budget_fraction_at_score']==encoder.fraction
    assert row['budget_fraction_at_score']==pytest.approx(1.-.1/b.candidate_seconds)
    assert row['budget_fraction_at_pop']==pytest.approx(1.-.4/b.candidate_seconds)
    assert row['budget_fraction_drift']==pytest.approx(.3/b.candidate_seconds)


def test_inference_overrun_does_not_begin_a_new_generation_query(monkeypatch):
    from tests.test_stage2_quality_budget import Clock
    _,s,b,r,a,n,ds,advisor=grouped_setup(4)
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    b.wall_start=0.;b.planning_wall_deadline=1.
    original=advisor.rank
    def rank(features,budget):
        ranked=original(features,budget)
        clock.now=2.
        return ranked
    monkeypatch.setattr(advisor,'rank',rank)
    chosen=r.order(s,ds,b)[0]
    provider=sp.ReferenceProvider();provider.advisor_runtime=r
    monkeypatch.setattr(sp,'_generate',lambda *args:pytest.fail('physical attempt after inference deadline overrun'))
    assert provider.generate(s,a,n,0,0.,35.,b,work_descriptor=chosen)==(False,None)
    assert not any(row['record_type'] in ('FeaturesBeforeQuery','FactualResult') for row in r.traces)
    assert 'READY_DEADLINE' in b.truncations


def test_groups_of_singletons_do_not_encode_or_infer():
    _,s,b,r,_,_,ds,advisor=grouped_setup(3)
    incoming=[with_parent(r,d,str(i)) for i,d in enumerate(ds)]
    assert r.order(s,incoming,b)==incoming
    assert not advisor.batches and not r._features


@pytest.mark.parametrize('rule',('initial','interleave','finish'))
def test_comparable_mode_preserves_deterministic_coverage_and_finish(rule):
    _,s,b,r,_,_,ds,advisor=grouped_setup(3)
    if rule=='initial': r._initial_finished=False
    elif rule=='interleave': r._post_initial=3
    else: b.candidate_active_seconds=.8*b.candidate_seconds
    assert r.order(s,ds,b)==ds
    assert not advisor.batches
    assert r._rank_cache is None


def test_legal_order_with_missing_scores_records_missing_measurements():
    _,s,b,r,_,_,ds,_=grouped_setup(3)
    class PartialScores:
        def rank(self,features,budget):
            ids=tuple(f['descriptor_id'] for f in features)
            return RankResult(ids[::-1],scores={ids[-1]:1.e100})
    r.advisor=PartialScores()
    assert r.order(s,ds,b)==ds[::-1]
    row=r.traces[-1]
    assert row['model_metadata']['missing_score_ids']==sorted(d.descriptor_id for d in ds[:-1])
    assert row['scores']=={ds[-1].descriptor_id:1.e100}


def test_checkpoint_scope_binding_preserves_historical_legacy_and_rejects_cross_scope():
    def callback(scope):
        runtime=AdvisorRuntime('RECORD',ranking_scope=scope)
        return sp.NormalSchedulerCallback(advisor_runtime=runtime)
    legacy=callback('legacy')
    historical=legacy.state_dict()
    historical['stage3'].pop('ranking_scope')
    legacy.load_state_dict(historical)
    comparable=callback('comparable_group_v1')
    comparable.load_state_dict(comparable.state_dict())
    with pytest.raises(ValueError,match='Stage3 callback binding'):
        comparable.load_state_dict(historical)
    with pytest.raises(ValueError,match='Stage3 callback binding'):
        legacy.load_state_dict(comparable.state_dict())


def test_future_view_metadata_preserves_numeric_features_and_training_fingerprint():
    snapshot,descriptor=fixture()
    snapshot.version='visible-state'
    projected=SimpleNamespace(**dict(vars(snapshot),version='effective-state',
        effective_state_version='planning_effective_state_v1'))
    encoder=FeatureEncoder(schema_id=SCHEMA_V2)
    assert encoder.encode(snapshot,[descriptor])==encoder.encode(projected,[descriptor])
    fingerprint={'feature_schema_id':SCHEMA_V2,'model_id':'original-model'}
    runtime=AdvisorRuntime('RECORD',fingerprint=fingerprint,ranking_scope='comparable_group_v1',trace_mode='online')
    runtime.begin_snapshot(projected,sp.PlanningBudget(),[])
    row=runtime.traces[-1]
    assert row['ranking_scope']=='comparable_group_v1'
    assert row['effective_state_version']=='planning_effective_state_v1'
    assert row['effective_snapshot_version']=='effective-state'
    assert runtime.fingerprint==fingerprint
    features=encoder.encode(projected,[descriptor])[0]
    assert len(features['query'])==66
    assert {k:len(v[0]) for k,v in features['sets'].items() if v}==dict(aircraft=84,slots=34,pads=26)


def test_future_plus_one_is_delayed_coverage_without_changing_five_second_buckets():
    from terminal_marl.core.rotating_slots.stage3_learning.catalog_control import coverage_key,coverage_report,retain_validated
    _,snapshot,_,runtime,aircraft,_,_,_=grouped_setup()
    effective=30.
    snapshot=replace(snapshot,time=effective,planning_timing=SimpleNamespace(effective_time=effective))
    immediate=fake_move(aircraft.id,start=effective)
    later=fake_move(aircraft.id,start=effective+1.)
    keep=sp.PredictionColumn(aircraft.id,'KEEP',(),(),(0,0,0,0,0),
        sp.PlanningNode('inner',0,effective,.9,0.,0))
    # Both proposals stay in the same historical five-second resource bins.
    assert coverage_key(immediate,effective)==coverage_key(later,effective)
    assert coverage_key(immediate,effective,effective_time=effective)[-1]==coverage_key(later,effective,effective_time=effective)[-1]
    assert coverage_key(immediate,effective,effective_time=effective)[1]
    assert not coverage_key(later,effective,effective_time=effective)[1]
    columns=[later]
    assert retain_validated(columns,immediate,effective,1,sp.Counter(),effective_time=effective)
    report=coverage_report(snapshot,{aircraft.id:[keep,later]},effective_time=effective)[aircraft.id]
    assert not report['immediate'] and report['delayed']
    assert 'NO_ELIGIBLE_IMMEDIATE_MOVE_OBSERVED' in report['missing_reasons']
    context=SimpleNamespace(observed_definite_conflicts=[])
    runtime.update_retained_context(snapshot,{aircraft.id:[later]},{},context)
    retained=runtime.retained_context[aircraft.id]
    assert retained['immediate_count']==0 and retained['delayed_count']==1 and retained['window_count']==1
    runtime.update_retained_context(snapshot,{aircraft.id:[later,immediate]},{},context)
    retained=runtime.retained_context[aircraft.id]
    assert retained['immediate_count']==1 and retained['delayed_count']==1 and retained['window_count']==1


@pytest.mark.parametrize('final_kind',('better','unknown','worse'))
def test_candidate_cutoff_allows_one_same_catalog_improvement_and_keeps_certified_best(monkeypatch,final_kind):
    world,s,_,runtime,_,_,_,_=grouped_setup()
    provider=sp.CandidateProvider(advisor_runtime=runtime)
    context=sp.EvaluationContext(s)
    aircraft=s.aircraft['i']
    keep=provider._column(s,aircraft,(),'KEEP',context)
    move=fake_move('i')
    catalog=sp.CandidateCatalog({'i':(keep,move)},{'i':(0,)},{},('i',))
    generated=[]; budgets=[]; calls=[]
    def batches(snapshot,budget,context):
        generated.append(0);budgets.append(budget)
        budget.candidate_active_seconds=budget.candidate_seconds
        yield catalog
        raise AssertionError('candidate generator resumed after exhaustion')
    monkeypatch.setattr(provider,'iter_progressive_columns',batches)
    import terminal_marl.core.rotating_slots.stage3_learning.catalog_control as control
    monkeypatch.setattr(control,'expansion_reasons',lambda *args:['NO_COMPLETE_CHAIN_OBSERVED'])
    callback=sp.NormalSchedulerCallback('rh_cp_sat',candidate_provider=provider,online_policy='progressive_v1')
    def plan(snapshot,actual,context,budget,**kwargs):
        calls.append((id(actual),id(budget),budget.solver_call_seconds,budget.solver_wall_used))
        budget.solver_wall_used+=2. if len(calls)==1 else 3.
        objective=move.objective if len(calls)==1 else (0,-1,0,0. if final_kind=='better' else 10.,0)
        certified=len(calls)==1 or final_kind!='unknown'
        evaluation=ValidationResult(certified,'SAFE' if certified else 'UNKNOWN')
        return sp.PlanningResult((move,),(move.plans[0],),evaluation,'FEASIBLE',objective,certified=certified)
    monkeypatch.setattr(callback.scheduler,'plan',plan)
    port=SimpleNamespace(build_planning_snapshot=lambda:s,
        commit_validated_prefix=lambda *args,**kwargs:ValidationResult(False,'REJECTED'))
    callback(port)
    assert len(generated)==1 and len(calls)==2
    assert calls[0][:2]==calls[1][:2] and calls[0][2]==2. and calls[1][2] is None and calls[1][3]==2.
    assert callback.last_result.certified
    expected=(0,-1,0,0.,0) if final_kind=='better' else move.objective
    assert callback.last_result.objective==expected
    record=callback.records[-1]
    assert record['generation_stop_reason']=='CANDIDATE_WORK_CUTOFF'
    assert record['same_catalog_improvement'] and record['catalog_rounds'][-1]['same_catalog_improvement']
    assert record['improvement_stop_reason']=='FINAL_EXISTING_CATALOG_CALL_COMPLETE'
    budget=budgets[0]
    assert budget.candidate_seconds==6. and budget.solver_seconds==8. and budget.solver_wall_used==5.
    assert budget.hint_seconds==3. and budget.soft_seconds==30. and budget.watchdog_seconds==60.
