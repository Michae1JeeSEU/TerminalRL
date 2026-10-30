"""Bounded service-first search, exact proof reuse, and six-key CP ordering."""
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS
import time

import pytest

from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider, EvaluationContext, PlanningBudget, PlanningResult,
    RollingHorizonCPSATScheduler, EnergySlackScheduler, _selection_key,
    _service_empty_reason, _record_service_certificate,
)
from terminal_marl.core.rotating_slots.stage2_downstream import SERVICE_EVALUATION_PROFILE
from terminal_marl.core.rotating_slots.selective_replanning.local_problem import LocalPlanner


@dataclass(frozen=True)
class Column:
    aircraft_id: str
    kind: str
    objective: tuple = (0, 0, 0, 0, 0, 0)
    plans: tuple = ()
    @property
    def signature(self): return self.aircraft_id, self.kind


def continuation_problem():
    keep=Column('a','KEEP')
    downstream=NS(evaluation_profile=SERVICE_EVALUATION_PROFILE, demands={keep.signature:('demand',)},
        order=('a',),frame_id='frame',baseline_starts={'a':10})
    catalog=NS(columns={'a':(keep,)},slacks={'a':1},downstream=downstream)
    snapshot=NS(version='s',time=0.)
    context=NS(snapshot=snapshot,snapshot_key='s',scope=NS(identity='scope'),global_frame=NS(identity='frame'),
        prefix_evaluator=None,prefix_evaluator_identity='evaluator',_prefix_results={},prefix=lambda columns:())
    return snapshot,catalog,context


def test_append_only_growth_reuses_only_identical_full_column_context():
    snapshot,catalog,context=continuation_problem()
    state,_=RollingHorizonCPSATScheduler._continuation(snapshot,catalog,context)
    signature=catalog.columns['a'][0].signature
    state['single_exclusions'].add(signature)
    state['single_results'][signature]='known-result'
    state['resource_complete']=True
    move=Column('a','MOVE',(0,-1,0,0,1,20))
    grown=NS(columns={'a':(*catalog.columns['a'],move)},slacks={'a':2},
        downstream=NS(**dict(vars(catalog.downstream),demands=dict(catalog.downstream.demands,**{}))))
    grown.downstream.demands[move.signature]=('new-demand',)
    again,report=RollingHorizonCPSATScheduler._continuation(snapshot,grown,context)
    assert again is state and report['append_only_reused']
    assert report['single_cuts_reused']==1 and report['single_results_reused']==1
    assert not again['resource_complete']
    changed=replace(move,objective=(1,-1,0,0,1,20))
    grown.columns={'a':(*catalog.columns['a'],changed)}
    invalidated,report=RollingHorizonCPSATScheduler._continuation(snapshot,grown,context)
    assert report['state_invalidated'] and not invalidated['single_exclusions']


@pytest.mark.parametrize('mutation',['scope','frame','evaluator','snapshot','queue'])
def test_proof_reuse_invalidates_changed_state_or_scoring_context(mutation):
    snapshot,catalog,context=continuation_problem()
    state,_=RollingHorizonCPSATScheduler._continuation(snapshot,catalog,context)
    state['single_exclusions'].add(catalog.columns['a'][0].signature)
    if mutation=='scope': context.scope=NS(identity='other')
    elif mutation=='frame': context.global_frame=NS(identity='other')
    elif mutation=='evaluator': context.prefix_evaluator_identity='other'
    elif mutation=='snapshot': snapshot.version='other'
    else: catalog.downstream.baseline_starts={'a':20}
    state,report=RollingHorizonCPSATScheduler._continuation(snapshot,catalog,context)
    assert report['state_invalidated'] and not state['single_exclusions']


def test_expired_generation_preserves_exact_original_certificate_without_validation():
    snapshot,catalog,context=continuation_problem()
    certificate=NS(accepted=True,status='SAFE',refinement_required=False)
    seed=PlanningResult(catalog.columns['a'],(),certificate,'SAFE',(0,0,0,0,0,0),certified=True)
    state,_=RollingHorizonCPSATScheduler._continuation(snapshot,catalog,context)
    state['incumbent']=seed
    context.check_batch=lambda columns: pytest.fail('unchanged known certificate was revalidated')
    context.check_commit_prefix=lambda columns: pytest.fail('unchanged prefix was revalidated')
    catalog.downstream.key=lambda columns: pytest.fail('expired catalog triggered expensive rescore')
    # Adding the key callable changes the scoring model binding, so establish
    # that binding before saving the certificate used in this assertion.
    state,_=RollingHorizonCPSATScheduler._continuation(snapshot,catalog,context)
    state['incumbent']=seed
    budget=NS(evaluation_available=lambda:False)
    reused=LocalPlanner._mapped_incumbent(seed,catalog,context,budget)
    assert reused.evaluation is certificate and reused.objective==seed.objective


def real_final_catalog(soc=.9):
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,STAGE2_POLICY_ID
    from terminal_marl.core.rotating_slots.trajectory import generate_final
    from terminal_marl.core.rotating_slots.selective_replanning import build_global_frame,PlanningScope
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False;world.seed_hold('a','inner',0,soc)
    snapshot=world.build_planning_snapshot()
    provider=CandidateProvider(planning_profile='downstream_service_v8',evaluation_profile=SERVICE_EVALUATION_PROFILE)
    frame=build_global_frame(snapshot,provider,EvaluationContext(snapshot))
    scope=PlanningScope(snapshot.version,snapshot.time,('a',),('a',),(),actual_k=1,objective_frame_id=frame.identity)
    budget=PlanningBudget(soft_seconds=30.,candidate_seconds=6.,solver_seconds=3.,max_refinement_rounds=8)
    context=EvaluationContext(snapshot,budget=budget,scope=scope,global_frame=frame)
    plan=generate_final('a','inner',0,5,0.,source_version=0,config=snapshot.config).plan
    assert plan is not None
    columns={'a':[provider._column(snapshot,snapshot.aircraft['a'],(),kind,context) for kind in ('KEEP','WAIT')]}
    columns['a'].append(provider._column(snapshot,snapshot.aircraft['a'],(plan,),'MOVE',context))
    catalog=provider._progressive_catalog(snapshot,budget,context,columns,{'a':columns['a'][0].reference},[snapshot.aircraft['a']])
    return world,snapshot,provider,context,budget,catalog


def test_known_safe_final_has_joint_certificate_and_is_physically_committed():
    world,snapshot,provider,context,budget,catalog=real_final_catalog()
    result=EnergySlackScheduler().plan(snapshot,catalog,context,budget)
    assert result.certified and result.prefix and result.prefix[0].pad_id==5
    assert len(result.objective)==6
    commit=world._scheduler_port.commit_validated_prefix(result.evaluation,snapshot_version=snapshot.version)
    assert commit.accepted
    evidence=budget.service_opportunity_certificates
    assert evidence[0]['joint_prefix_certified'] and evidence[0]['final_hashes']==[result.prefix[0].trajectory_hash]
    assert evidence[0]['finals'][0]['plan']['trajectory_hash']==result.prefix[0].trajectory_hash


def test_unknown_validation_never_produces_service_final_witness():
    _,snapshot,_,context,budget,catalog=real_final_catalog()
    evaluation=NS(accepted=False,status='UNKNOWN',refinement_required=False)
    _record_service_certificate(snapshot,catalog.columns['a'],catalog.columns['a'][-1].plans,evaluation,context,budget)
    assert not getattr(budget,'service_opportunity_certificates',())


def test_minimal_middle_supply_is_completed_before_first_cp_and_committed():
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,STAGE2_POLICY_ID
    from terminal_marl.core.rotating_slots.selective_replanning import build_global_frame,PlanningScope
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False;world.seed_hold('a','middle',0,.9)
    snapshot=world.build_planning_snapshot()
    provider=CandidateProvider(planning_profile='downstream_service_v8',evaluation_profile=SERVICE_EVALUATION_PROFILE)
    frame=build_global_frame(snapshot,provider,EvaluationContext(snapshot))
    scope=PlanningScope(snapshot.version,snapshot.time,('a',),('a',),(),actual_k=1,objective_frame_id=frame.identity)
    budget=PlanningBudget(soft_seconds=30.,candidate_seconds=30.,candidate_active_seconds=0.,
        planning_wall_deadline=time.perf_counter()+19.,search_wall_deadline=time.perf_counter()+15.)
    context=EvaluationContext(snapshot,budget=budget,scope=scope,global_frame=frame)
    generator=provider.iter_progressive_columns(snapshot,budget,context,scope=scope,early_publish=True)
    catalog=next(generator);generator.close()
    report=provider.suffix_completion_report(budget)
    assert report['priority_before_first_cp'] and report['priority_reserved_seconds']>0
    assert report['priority_used_seconds']>0 and report['attempts'][0]['complete_candidate_in_catalog']
    result=EnergySlackScheduler().plan(snapshot,catalog,context,budget)
    assert result.certified and result.prefix and result.prefix[0].target_ring=='inner'
    assert result.selected[0].plans[-1].pad_id is not None
    assert world._scheduler_port.commit_validated_prefix(result.evaluation,snapshot_version=snapshot.version).accepted


def test_unknown_joint_validation_never_authorizes_execution(monkeypatch):
    from terminal_marl.core.rotating_slots.models import ValidationResult
    _,snapshot,_,context,budget,catalog=real_final_catalog()
    monkeypatch.setattr(context,'check_commit_prefix',lambda selected:ValidationResult(False,'UNKNOWN','TEST_TIMEOUT'))
    result=EnergySlackScheduler().plan(snapshot,catalog,context,budget)
    assert not result.certified and not result.prefix
    assert not getattr(budget,'service_opportunity_certificates',())


def test_service_priority_seed_precedes_reference_and_keeps_every_variable(monkeypatch):
    _,snapshot,provider,context,budget,_=real_final_catalog()
    events=[]
    def reference(*args):
        events.append('reference')
        yield None
    monkeypatch.setattr(provider.references,'build_steps',reference)
    started=time.perf_counter()
    generator=provider.iter_progressive_columns(snapshot,budget,context,early_publish=True)
    catalog=next(generator);generator.close()
    assert catalog.generation_coverage['phase']=='SERVICE_PRIORITY_SEED'
    assert set(catalog.columns)=={'a'} and any(c.plans and c.plans[-1].pad_id is not None for c in catalog.columns['a'])
    assert not events and time.perf_counter()-started<6.


def test_empty_reason_requires_bounded_certified_alternative():
    _,catalog,context=continuation_problem()
    budget=PlanningBudget()
    result=PlanningResult((),(),None,'LIMITED',(0,0,0,0,0,0),certified=True)
    assert _service_empty_reason(result,catalog,context,budget)=='CANDIDATE_INSUFFICIENT'
    budget._service_certified_move_alternative=True
    assert _service_empty_reason(result,catalog,context,budget)=='REASONABLE_WAIT'
    result=replace(result,details={'solver_stop_reason':'PREFIX_VALIDATION_UNKNOWN'})
    assert _service_empty_reason(result,catalog,context,budget)=='CERTIFICATION_TIMEOUT'


def test_observed_conflict_with_exhausted_certification_allocation_is_timeout():
    _,catalog,context=continuation_problem()
    context.observed_definite_conflicts=[('bounded','conflict')]
    budget=PlanningBudget()
    result=PlanningResult((),(),None,'LIMITED',(0,0,0,0,0,0),certified=True,
        details={'solver_stop_reason':'SOLVER_ALLOCATION_EXHAUSTED','batched_conflicts':1})
    assert _service_empty_reason(result,catalog,context,budget)=='CERTIFICATION_TIMEOUT'
    proven=replace(result,details={'solver_stop_reason':None,'batched_conflicts':1})
    assert _service_empty_reason(proven,catalog,context,budget)=='TRUE_CONFLICT'


def test_service_cp_combines_failure_and_landing_before_one_validation():
    pytest.importorskip('ortools')
    _,snapshot,_,context,budget,catalog=real_final_catalog()
    result=RollingHorizonCPSATScheduler().plan(snapshot,catalog,context,budget)
    assert result.certified and result.prefix
    assert [row['phase'] for row in result.solver_records]==[0,2,3,4,5]
    assert result.solver_records[0]['dominance_coefficient']==2
    assert result.solver_records[0]['objective_components']==('F','-L')
    assert all(len(row['model_integer_key'])==6 for row in result.solver_records)
    assert result.details['optimality_scope']=='bounded_candidate_catalog_model'


@pytest.mark.parametrize('method',['energy_slack','rh_cp_sat'])
def test_real_service_final_survives_20s_lead_delivery_and_starts_at_effective(method):
    if method=='rh_cp_sat': pytest.importorskip('ortools')
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,STAGE2_POLICY_ID
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    from terminal_marl.core.rotating_slots.planning_latency import LatencyAwareScheduler
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False;world.seed_hold('a','inner',0,.9)
    callback=NormalSchedulerCallback(method,planning_profile='downstream_service_v8',
        evaluation_profile=SERVICE_EVALUATION_PROFILE,prefix_reuse=True)
    wrapper=LatencyAwareScheduler(callback,lead_seconds=20.,preparation_margin=1.,search_handoff_reserve_seconds=4.)
    world.decision_callback=wrapper
    started=time.perf_counter();world.advance(20.01)
    assert time.perf_counter()-started<19.
    assert wrapper.records[0]['status']=='COMMITTED',wrapper.records[0]
    assert wrapper.records[0]['ready_deadline']==19. and wrapper.records[0]['effective_time']==20.
    assert wrapper.records[0]['normal_work_s']<15.
    assert world.active['a'].pad_id is not None
    assert any(e['type']=='maneuver_started' and e['time']==20. for e in world.events)
    witness=callback.records[0]['service_opportunity_certificates'][0]
    assert witness['source_snapshot_id'] and witness['effective_snapshot_id']
    assert witness['source_snapshot_time']==0. and witness['projection_protocol']=='planning_effective_state_v1'
    assert witness['remaining_ready_margin_s']>0
    import json
    state=json.loads(json.dumps(callback.state_dict(),allow_nan=False))
    assert set(state['records'][0]['objective_terms'])=={'F','-L','R','G_service','E','tau'}
    restored=NormalSchedulerCallback(method,planning_profile='downstream_service_v8',
        evaluation_profile=SERVICE_EVALUATION_PROFILE,prefix_reuse=True)
    restored.load_state_dict(state)
    assert restored.records[0]['objective']==state['records'][0]['objective']


def test_slow_later_growth_keeps_original_nonempty_certificate_for_handoff(monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_planning as sp
    from terminal_marl.core.rotating_slots.models import ValidationResult
    from tests.test_stage2_quality_budget import Clock
    _,snapshot,provider,original_context,budget,catalog=real_final_catalog()
    frame=original_context.global_frame;scope=original_context.scope
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    budget.wall_start=0.;budget.search_wall_deadline=15.;budget.planning_wall_deadline=19.
    budget.candidate_active_seconds=0.;budget.solver_seconds=8.
    generated=[]
    def catalogs(*args,**kwargs):
        generated.append('seed');yield catalog
        clock.now=14.5;generated.append('slow_growth');yield catalog
        pytest.fail('growth resumed after certification deadline')
    monkeypatch.setattr(provider,'iter_progressive_columns',catalogs)
    first=[]
    class SlowScheduler:
        def plan(self,snapshot,catalog,context,budget,*,incumbent=None):
            if not first:
                result=EnergySlackScheduler().plan(snapshot,catalog,context,budget)
                assert result.certified and result.prefix
                state,_=RollingHorizonCPSATScheduler._continuation(snapshot,catalog,context)
                state['incumbent']=result;first.append(result);clock.now=2.
                return result
            assert incumbent is not None and incumbent.evaluation is first[0].evaluation
            clock.now=15.2
            return PlanningResult((),(),ValidationResult(False,'UNKNOWN','SLOW_VALIDATION'),'UNKNOWN',(),certified=False)
    local=LocalPlanner(provider,SlowScheduler(),progressive_v2=True)
    result=local.plan(snapshot,scope,frame,budget)
    assert generated==['seed','slow_growth']
    assert result.certified and result.prefix==first[0].prefix
    assert result.evaluation is first[0].evaluation and result.objective==first[0].objective
    assert budget.ready_deadline_ok() and budget.remaining_seconds()==0.
    # A handoff certificate is retained while the independent delivery deadline
    # still has margin. Once that deadline expires it cannot imply quality wait.
    clock.now=19.1
    empty=replace(result,prefix=())
    budget._service_certified_move_alternative=True
    assert _service_empty_reason(empty,catalog,local.last_context,budget)=='DELIVERY_TIMEOUT'


def test_preconstructed_low_soc_final_never_bypasses_physical_safety():
    world,snapshot,_,context,budget,catalog=real_final_catalog(soc=.251)
    final=catalog.columns['a'][-1]
    checked=context.check_commit_prefix((final,))
    assert not checked.accepted and checked.status!='UNKNOWN'
    result=EnergySlackScheduler().plan(snapshot,catalog,context,budget)
    assert not result.prefix and not world.active and not world.soft
    assert not getattr(budget,'service_opportunity_certificates',())


def test_service_extra_catalogs_allow_only_one_bounded_scope_expansion():
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,STAGE2_POLICY_ID
    from terminal_marl.core.rotating_slots.selective_replanning import build_global_frame,PlanningScope
    from terminal_marl.core.rotating_slots.stage2_planning import NormalReference
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    for aid,ring in (('a','outer'),('b','middle'),('c','inner')): world.seed_hold(aid,ring,0,.9)
    snapshot=world.build_planning_snapshot()
    class GrowingProvider(CandidateProvider):
        def iter_progressive_columns(self,snapshot,budget,context,*,scope=None,early_publish=False):
            for _ in range(4):
                rows={aid:[self._column(snapshot,snapshot.aircraft[aid],(),kind,context) for kind in ('KEEP','WAIT')]
                    for aid in scope.replan_ids}
                yield self._progressive_catalog(snapshot,budget,context,rows,
                    {aid:NormalReference('UNKNOWN') for aid in rows},[snapshot.aircraft[aid] for aid in rows])
    provider=GrowingProvider(planning_profile='downstream_service_v8',evaluation_profile=SERVICE_EVALUATION_PROFILE)
    frame=build_global_frame(snapshot,provider,EvaluationContext(snapshot))
    scope=PlanningScope(snapshot.version,snapshot.time,('a','b','c'),('a',),('b','c'),
        actual_k=1,max_k=12,objective_frame_id=frame.identity)
    seen=[]
    class Blockers:
        def plan(self,snapshot,catalog,context,budget,*,incumbent=None):
            seen.append(tuple(context.scope.replan_ids))
            unopened=sorted(set(scope.eligible_ids)-set(context.scope.replan_ids))
            if unopened:
                context.background_blockers.append(dict(source='a',target=unopened[0],snapshot_id=snapshot.version,
                    reason='RESOURCE_CONFLICT',resource=('middle',0)))
            return EnergySlackScheduler().plan(snapshot,catalog,context,budget)
    budget=PlanningBudget(soft_seconds=30.,candidate_seconds=30.,candidate_active_seconds=0.)
    local=LocalPlanner(provider,Blockers(),progressive_v2=True)
    local.plan(snapshot,scope,frame,budget)
    assert seen==[('a',),('a','b'),('a','b'),('a','b')]
    assert local.telemetry['scope_expansions']==1 and local.last_scope.actual_k<=12


def test_service_hint_tries_inner_final_before_many_keep_only_variables(monkeypatch):
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,STAGE2_POLICY_ID
    from terminal_marl.core.rotating_slots.selective_replanning import build_global_frame,PlanningScope
    from terminal_marl.core.rotating_slots.stage2_planning import NormalReference
    from terminal_marl.core.rotating_slots.trajectory import generate_final
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False;world.seed_hold('final','inner',0,.9)
    for index in range(8): world.seed_hold('keep'+str(index),'outer',index*3,.9)
    snapshot=world.build_planning_snapshot()
    provider=CandidateProvider(planning_profile='downstream_service_v8',evaluation_profile=SERVICE_EVALUATION_PROFILE)
    frame=build_global_frame(snapshot,provider,EvaluationContext(snapshot));ids=tuple(sorted(snapshot.aircraft))
    scope=PlanningScope(snapshot.version,snapshot.time,ids,ids,(),actual_k=len(ids),status='FULL_H',objective_frame_id=frame.identity)
    budget=PlanningBudget(soft_seconds=30.,candidate_seconds=30.)
    context=EvaluationContext(snapshot,budget=budget,scope=scope,global_frame=frame)
    rows={aid:[provider._column(snapshot,snapshot.aircraft[aid],(),kind,context) for kind in ('KEEP','WAIT')] for aid in ids}
    plan=generate_final('final','inner',0,5,0.,source_version=0,config=snapshot.config).plan
    assert plan is not None
    rows['final'].append(provider._column(snapshot,snapshot.aircraft['final'],(plan,),'MOVE',context))
    catalog=provider._progressive_catalog(snapshot,budget,context,rows,
        {aid:NormalReference('UNKNOWN') for aid in ids},[snapshot.aircraft[aid] for aid in ids])
    # Force the old energy-slack scan to put all KEEP-only variables first.
    catalog=replace(catalog,slacks={aid:(99 if aid=='final' else 0) for aid in ids})
    calls=[];original=context.check_batch
    def batch(columns):
        calls.append(tuple(columns))
        return original(columns)
    monkeypatch.setattr(context,'check_batch',batch)
    result=EnergySlackScheduler().plan(snapshot,catalog,context,budget)
    assert result.certified and result.prefix==(plan,)
    assert len(calls)==2
    assert all({c.aircraft_id for c in columns}==set(ids) for columns in calls)
    assert all(any(c.plans==(plan,) for c in columns) for columns in calls)
    assert context.check_commit_prefix(result.selected).accepted


def test_seed_round_robin_rejected_inner_does_not_starve_real_middle_partial(monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_planning as sp
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,STAGE2_POLICY_ID
    from terminal_marl.core.rotating_slots.selective_replanning import build_global_frame,PlanningScope
    from terminal_marl.core.rotating_slots.trajectory import generate_transfer
    from tests.test_stage2_quality_budget import Clock
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False
    world.seed_hold('inner','inner',5,.9);world.seed_hold('middle','middle',0,.9)
    snapshot=world.build_planning_snapshot()
    provider=CandidateProvider(planning_profile='downstream_service_v8',evaluation_profile=SERVICE_EVALUATION_PROFILE)
    frame=build_global_frame(snapshot,provider,EvaluationContext(snapshot));ids=('inner','middle')
    scope=PlanningScope(snapshot.version,snapshot.time,ids,ids,(),actual_k=2,objective_frame_id=frame.identity)
    plan=generate_transfer('middle','middle',0,0,0.,durations=(40.,),config=snapshot.config).plan
    assert plan is not None
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    budget=PlanningBudget(wall_start=0.,soft_seconds=30.,candidate_seconds=30.,candidate_active_seconds=0.,
        search_wall_deadline=15.,planning_wall_deadline=19.)
    context=EvaluationContext(snapshot,budget=budget,scope=scope,global_frame=frame)
    events=[]
    def progress(snapshot,a,node,budget,context):
        if a.id=='inner':
            for _ in range(8):
                events.append('inner');clock.now+=.6
                yield ('ATTEMPT',None,None)
        else:
            events.append('middle');yield ('SCREEN',None,None)
            events.append('middle');yield ('COLUMN',(plan,),'NONE')
    monkeypatch.setattr(provider,'_progress_steps',progress)
    generator=provider.iter_progressive_columns(snapshot,budget,context,scope=scope,early_publish=True)
    catalog=next(generator);generator.close()
    assert events[:4]==['inner','middle','inner','middle']
    assert any(c.plans and c.plans[0]==plan for c in catalog.columns['middle'])
    assert set(catalog.columns)==set(ids)


@pytest.mark.parametrize('already_enrolled',[False,True],ids=['new_reservation','existing_pending_reservation'])
def test_empty_initial_reservation_accepts_late_real_partial_and_completes_before_more_growth(monkeypatch,already_enrolled):
    from terminal_marl.core.rotating_slots import stage2_planning as sp
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,STAGE2_POLICY_ID
    from terminal_marl.core.rotating_slots.selective_replanning import build_global_frame,PlanningScope
    from terminal_marl.core.rotating_slots.trajectory import generate_transfer
    from tests.test_stage2_quality_budget import Clock
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False;world.seed_hold('middle','middle',0,.9)
    snapshot=world.build_planning_snapshot()
    provider=CandidateProvider(planning_profile='downstream_service_v8',evaluation_profile=SERVICE_EVALUATION_PROFILE)
    frame=build_global_frame(snapshot,provider,EvaluationContext(snapshot))
    scope=PlanningScope(snapshot.version,snapshot.time,('middle',),('middle',),(),actual_k=1,objective_frame_id=frame.identity)
    context=EvaluationContext(snapshot,scope=scope,global_frame=frame)
    plan=generate_transfer('middle','middle',0,0,0.,durations=(40.,),config=snapshot.config).plan
    assert plan is not None
    partial=provider._column(snapshot,snapshot.aircraft['middle'],(plan,),'MOVE',context)
    ref=provider.references.get(snapshot,snapshot.aircraft['middle'],node=partial.node,
        budget=PlanningBudget.for_profile('quality_v5'))
    assert ref.status=='KNOWN' and ref.plans[-1].pad_id is not None
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    budget=PlanningBudget(wall_start=0.,soft_seconds=30.,candidate_seconds=30.,candidate_active_seconds=0.,
        search_wall_deadline=15.,planning_wall_deadline=19.)
    context.budget=budget
    monkeypatch.setattr(provider,'_progress_steps',lambda *args:iter(()))
    monkeypatch.setattr(provider,'_delayed_steps',lambda *args:iter(()))
    def reference(snapshot,a,node,budget):
        if node.ring=='inner':
            clock.now+=.2;yield None
            clock.now+=.2;return ref
        yield None
        return sp.NormalReference('UNKNOWN')
    monkeypatch.setattr(provider.references,'build_steps',reference)
    def paths(*args):
        yield ('SCREEN',None,None)
        yield ('COLUMN',(plan,),'NONE')
        pytest.fail('ordinary work resumed before publishing reserved completion')
    monkeypatch.setattr(provider,'_paths',paths)
    generator=provider.iter_progressive_columns(snapshot,budget,context,scope=scope,early_publish=True)
    first=next(generator)
    state=budget._suffix_completion_state
    assert state['records']==[] and state['priority_used_seconds']==0.
    reserved=state['priority_reserved_seconds']
    assert reserved==1.5 and budget.solver_validation_headroom()==4.5
    assert EnergySlackScheduler().plan(snapshot,first,context,budget).certified
    if already_enrolled:
        assert context.check_single(partial).accepted
        provider.reserve_suffix_completion(snapshot,replace(first,columns={'middle':(*first.columns['middle'],partial)}),budget)
        assert len(state['records'])==1 and state['records'][0]['suffix_status']=='RESERVED'
    later=next(generator);generator.close()
    assert later.generation_coverage['phase']=='SERVICE_LATE_PRIORITY_COMPLETION'
    assert state is budget._suffix_completion_state and state['priority_reserved_seconds']==reserved
    assert len(state['records'])==1 and state['records'][0]['complete_candidate_in_catalog']
    assert state['priority_used_seconds']==pytest.approx(.4)
    assert budget.service_completion_reserve()==pytest.approx(1.1)
    assert any(c.plans and c.plans[0]==plan and c.plans[-1].pad_id is not None for c in later.columns['middle'])
    for _ in range(3): provider.reserve_suffix_completion(snapshot,later,budget)
    assert state['priority_reserved_seconds']==reserved and state['priority_used_seconds']==pytest.approx(.4)
    result=EnergySlackScheduler().plan(snapshot,later,context,budget)
    assert result.certified and result.prefix==(plan,)
    assert world._scheduler_port.commit_validated_prefix(result.evaluation,snapshot_version=snapshot.version).accepted
    clock.now=14.
    assert not budget.available()  # Ordinary generation preserves unused 1.1s.
    with budget.stage_limit(.5,'SERVICE_COMPLETION_SLICE'):
        assert budget.available()


def test_service_reservation_refills_at_most_two_distinct_aircraft_without_reset():
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateCatalog,PredictionColumn,PlanningNode
    provider=CandidateProvider(planning_profile='downstream_service_v8',evaluation_profile=SERVICE_EVALUATION_PROFILE)
    snapshot=NS(version='immutable',time=0.,soft={})
    budget=PlanningBudget(soft_seconds=30.)
    empty=CandidateCatalog({}, {}, {}, ())
    provider.reserve_suffix_completion(snapshot,empty,budget)
    state=budget._suffix_completion_state;original=state['priority_reserved_seconds']
    for aid in ('a','b','c'):
        plan=NS(trajectory_hash='reservation-token-'+aid,start_time=0.,pad_id=None)
        column=PredictionColumn(aid,'MOVE',(plan,),(),(0,0,0,0,0,0),PlanningNode('inner',0,40.,.9,0.,1))
        provider.reserve_suffix_completion(snapshot,CandidateCatalog({aid:(column,)},{},{},()),budget)
        assert state['priority_reserved_seconds']==original
    assert [row['aircraft_id'] for row in state['records']]==['a','b']
    state['priority_used_seconds']=.5
    provider.reserve_suffix_completion(snapshot,empty,budget)
    assert state['priority_used_seconds']==.5 and state['priority_reserved_seconds']==original
    state['records'][0]['suffix_status']='KNOWN'
    assert budget.service_completion_reserve()==pytest.approx(original-.5)  # Other row is pending.
    state['records'][1]['suffix_status']='NOT_ADMISSIBLE'
    assert budget.service_completion_reserve()==0. and budget.solver_validation_headroom()==0.
    assert state['priority_reserve_released']
    assert state['priority_used_seconds']==.5 and state['priority_reserved_seconds']==original
