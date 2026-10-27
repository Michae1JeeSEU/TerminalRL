"""Lazy comparable-group ordering over saved/static visible inputs only."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots.stage3_learning.features import FeatureEncoder,SCHEMA_V2
from terminal_marl.core.rotating_slots.stage3_learning.online_observer import OnlineObserver
from terminal_marl.core.rotating_slots.stage3_learning.advisor import RankResult
from tests.test_rotating_slots_stage3_runtime import setup
from tests.test_rotating_slots_latency_ranking import CountingAdvisor,with_parent
from tests.test_stage2_quality_budget import Clock
from terminal_marl.core.rotating_slots import stage2_planning as sp
from terminal_marl.core.rotating_slots.models import ValidationResult
from tests.test_rotating_slots_stage3_online_catalog import fake_move


LAZY='comparable_group_lazy_v2'


def ready(scope=LAZY,count=6):
    advisor=CountingAdvisor()
    world,s,b,r,a,n,ds=setup(advisor=advisor,feature_schema=SCHEMA_V2,ranking_scope=scope)
    r.feature_encoder=FeatureEncoder(schema_id=SCHEMA_V2);r.complete_initial_coverage()
    ds.extend(r.descriptor(s,a,n,t,0.,35.,'progress') for t in range(3,count))
    return s,b,r,a,n,ds,advisor


def groups(runtime,descriptors):
    left=[with_parent(runtime,d,'left') for d in descriptors[:3]]
    right=[with_parent(runtime,d,'right') for d in descriptors[3:]]
    return left,right,[d for pair in zip(left,right) for d in pair]


def test_only_head_group_is_scored_and_other_positions_are_unchanged():
    s,b,r,_,_,ds,advisor=ready()
    left,right,incoming=groups(r,ds)
    ordered=r.order(s,incoming,b)
    assert advisor.batches==[[d.descriptor_id for d in left]]
    assert ordered==[left[2],right[0],left[1],right[1],left[0],right[2]]
    row=r.traces[-1]
    assert (row['scored_groups'],row['scored_descriptors'],row['deferred_groups'],row['deferred_items'],row['batch_consumed'])==(1,3,1,3,1)
    assert set(r._rank_cache['groups'])=={r._comparable_key(left[0])}


def test_every_head_matches_fresh_eager_choice_with_fixed_clock(monkeypatch):
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    s,b,r,_,_,ds,advisor=ready();b.wall_start=0.
    es,eb,eager,_,_,eds,_=ready('comparable_group_v1');eb.wall_start=0.
    _,_,incoming=groups(r,ds);_,_,eager_incoming=groups(eager,eds)
    class FixedScores:
        def __init__(self,descriptors):
            self.batches=[]
            self.scores={d.descriptor_id:(d.target_id+1)*(1.e200 if d.parent_prefix_id=='right' else 1.e100) for d in descriptors}
        def rank(self,features,budget):
            ids=[f['descriptor_id'] for f in features];self.batches.append(ids)
            return RankResult(tuple(sorted(ids,key=lambda did:-self.scores[did])),{},
                {did:self.scores[did] for did in ids})
    advisor=FixedScores(incoming);r.advisor=advisor;eager.advisor=FixedScores(eager_incoming)
    identity=lambda d:(d.parent_prefix_id,d.target_id)
    mapping={identity(d):d for d in eager_incoming}
    for _ in range(len(incoming)):
        eager._rank_cache=None
        expected=eager.order(es,[mapping[identity(d)] for d in incoming],eb)[0]
        actual=r.order(s,incoming,b)
        assert identity(actual[0])==identity(expected)
        incoming=actual[1:]
    assert all(len(batch)<=3 for batch in advisor.batches)


def test_inactive_membership_parent_and_unrelated_coverage_changes_do_not_rescore_head():
    s,b,r,a,n,ds,advisor=ready()
    left,right,incoming=groups(r,ds)
    ordered=r.order(s,incoming,b)
    remaining=[d for d in ordered[1:] if d.parent_prefix_id=='left']
    new=with_parent(r,r.descriptor(s,a,n,8,0.,35.,'progress'),'right')
    r.parents['right']['soc']-=.01
    r.retained_context['unrelated']={'full_count':1,'directions':{}}
    r.known_conflicts['unrelated']=1;r._coverage_epoch+=1
    result=r.order(s,[remaining[0],new,*right,*remaining[1:]],b)
    assert len(advisor.batches)==1 and r.coverage['rank_cache_hits']==1
    assert result[0] in remaining
    assert all(d in remaining for d in result if d.descriptor_id in advisor.batches[0])
    # First activation of the changed group scores its actual current members.
    r.order(s,[new,*right],b)
    assert advisor.batches[1]==[d.descriptor_id for d in [new,*right]]


@pytest.mark.parametrize('change',('candidate','source','parent','coverage','duplicate','conflict'))
def test_active_dependencies_still_invalidate_cached_scores(change):
    s,b,r,a,n,ds,advisor=ready(count=5)
    ds=[with_parent(r,d,'active') for d in ds]
    r.retained_context[a.id]={'full_count':1,'directions':{repr(('pad',0)):1}}
    ordered=r.order(s,ds,b);remaining=ordered[1:]
    if change=='candidate':remaining.append(with_parent(r,r.descriptor(s,a,n,8,0.,35.,'progress'),'active'))
    elif change=='source':r.nodes[remaining[0].source_node_id]['soc']-=.01
    elif change=='parent':r.parents['active']['soc']-=.01
    elif change=='coverage':r.retained_context[a.id]['full_count']+=1
    elif change=='duplicate':r.retained_context[a.id]['directions'][repr(('pad',0))]+=1
    elif change=='conflict':r.known_conflicts[a.id]=1
    r.order(s,remaining,b)
    assert len(advisor.batches)==2
    assert r.traces[-1]['rank_cache_miss_reasons']=={('CANDIDATES_CHANGED' if change=='candidate' else 'FEATURE_CONTEXT_CHANGED'):1}


def test_three_consumptions_then_reinfer_and_global_parent_legality_remain():
    s,b,r,_,_,ds,advisor=ready(count=6)
    for _ in range(3):ds=r.order(s,ds,b)[1:]
    assert len(advisor.batches)==1 and r.coverage['rank_cache_hits']==2
    r.order(s,ds,b)
    assert len(advisor.batches)==2 and r.coverage['rank_cache_miss:CONSUMPTION_LIMIT']==1
    s,b,r,_,_,ds,advisor=ready();left,right,incoming=groups(r,ds)
    r.parents['right']['materialized']=False
    with pytest.raises(ValueError,match='unmaterialized'):r.order(s,incoming,b)
    assert not advisor.batches  # Even an inactive group must pass the global legal registry guard.


def test_singleton_head_and_original_budget_fallback_do_not_score_deferred_groups():
    s,b,r,_,_,ds,advisor=ready()
    left,right,_=groups(r,ds)
    incoming=[left[0],*right]
    assert r.order(s,incoming,b)==incoming and not advisor.batches
    assert r.traces[-1]['deferred_groups']==1 and r.traces[-1]['scored_descriptors']==0
    b.candidate_active_seconds=b.candidate_seconds
    assert r.order(s,[*left,*right],b)==[*left,*right] and not advisor.batches
    assert r.traces[-1]['policy']=='budget_exhausted'
    b.candidate_active_seconds=.8*b.candidate_seconds
    assert r.order(s,[*left,*right],b)==[*left,*right] and not advisor.batches
    assert r.traces[-1]['policy']=='deterministic_finish'


def test_lazy_scalar_counts_reach_online_coverage_without_feature_arrays():
    s,b,r,_,_,ds,advisor=ready();_,_,incoming=groups(r,ds)
    observer=OnlineObserver();r.observer=observer;r.trace_mode='online'
    r.order(s,incoming,b);r.finish_snapshot(b)
    counts=observer.summary()['coverage']['counts']
    assert counts['rank_scored_groups']==1 and counts['rank_scored_descriptors']==3
    assert counts['rank_deferred_groups']==1 and counts['rank_deferred_items']==3 and counts['rank_batch_consumed']==1
    rank=next(row for row in r.traces if row['record_type']=='Rank' and 'scored_descriptors' in row)
    assert rank['scored_descriptors']==3 and 'features' not in rank and 'scores' not in rank


def test_lazy_scope_keeps_search_stop_and_certified_prefix_without_final_improvement(monkeypatch):
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    s,_,r,a,_,_,_=ready()
    provider=sp.CandidateProvider(advisor_runtime=r);move=fake_move(a.id)
    cat=sp.CandidateCatalog({a.id:(move,)},{a.id:(0,)},{},())
    generated=[];calls=[]
    def catalogs(*args):
        generated.append(True);yield cat
        pytest.fail('generator resumed after search deadline')
    monkeypatch.setattr(provider,'iter_progressive_columns',catalogs)
    import terminal_marl.core.rotating_slots.stage3_learning.catalog_control as control
    monkeypatch.setattr(control,'expansion_reasons',lambda *args:['MORE'])
    callback=sp.NormalSchedulerCallback('rh_cp_sat',candidate_provider=provider,online_policy='progressive_v1')
    def plan(snapshot,catalog,context,budget,**kwargs):
        calls.append(budget);clock.now=1.
        return sp.PlanningResult((move,),move.plans,ValidationResult(True,'SAFE'),'FEASIBLE',move.objective,certified=True)
    monkeypatch.setattr(callback.scheduler,'plan',plan)
    port=SimpleNamespace(planning_wall_origin=0.,planning_wall_deadline=5.,search_wall_deadline=1.,
        search_handoff_reserve_seconds=4.,build_planning_snapshot=lambda:s,
        commit_validated_prefix=lambda *args,**kwargs:ValidationResult(False,'DEFERRED'))
    result=callback(port)
    assert result.certified and result.prefix==move.plans and len(calls)==len(generated)==1
    record=callback.records[-1]
    assert record['generation_stop_reason']=='SEARCH_HANDOFF_DEADLINE' and not record['same_catalog_improvement']
    assert 'rank_batch_cache' in record and (calls[0].candidate_seconds,calls[0].solver_seconds,calls[0].hint_seconds)==(6.,8.,3.)
