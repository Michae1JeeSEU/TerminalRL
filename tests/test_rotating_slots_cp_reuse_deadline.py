"""Context-scoped CP proofs and absolute wall deadlines; no world simulation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots import stage2_planning as sp
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.resources import Reservation
from tests.test_stage2_quality_budget import Clock,DiscreteContext,RefinementContext,discrete_catalog,refinement_catalog,keep_hint
from tests.test_rotating_slots_latency_ranking import grouped_setup
from tests.test_rotating_slots_stage3_online_catalog import fake_move


def certified_keep(catalog,context):
    selected=tuple(rows[0] for rows in catalog.columns.values())
    return sp.PlanningResult(selected,context.prefix(selected),context.check_commit_prefix(selected),
        'FEASIBLE',sp._sum_key(selected),certified=True)


def ordered_proof_catalog():
    """The first lex layer uniquely selects each bounded rejected alternative."""
    catalog=refinement_catalog()
    columns=dict(catalog.columns)
    columns['a']=tuple(replace(c,objective=(10 if not c.plans else index-1,*c.objective[1:]))
                       for index,c in enumerate(columns['a']))
    return replace(catalog,columns=columns)


def timed_solver(monkeypatch,clock,step=.25,startup=0.):
    cp=pytest.importorskip('ortools.sat.python.cp_model')
    original=cp.CpSolver
    observed=[]
    class Timed(original):
        def __init__(self):
            super().__init__()
            clock.now+=startup
        def Solve(self,model,*args,**kwargs):
            proto=model.Proto()
            observed.append(dict(limit=self.parameters.max_time_in_seconds,at=clock.now,
                hint_values=tuple(proto.solution_hint.values)))
            result=super().Solve(model,*args,**kwargs)
            clock.now+=step
            return result
    monkeypatch.setattr(cp,'CpSolver',Timed)
    return observed


def test_same_catalog_reuses_certified_incumbent_proved_pair_cut_without_new_hint_budget(monkeypatch):
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    observed=timed_solver(monkeypatch,clock)
    budget=sp.PlanningBudget.for_profile('quality_v5');budget.wall_start=0.
    budget.hint_wall_used=3.;budget.solver_call_seconds=.25
    catalog=ordered_proof_catalog();context=RefinementContext(budget)
    scheduler=sp.RollingHorizonCPSATScheduler()
    monkeypatch.setattr(sp.EnergySlackScheduler,'plan',lambda *args,**kwargs:pytest.fail('hint budget was recreated'))
    first=scheduler.plan(None,catalog,context,budget,incumbent=certified_keep(catalog,context))
    assert first.certified and first.details['pair_cuts']==1
    used=budget.solver_wall_used;rounds=budget.refinement_rounds_used
    second=scheduler.plan(None,catalog,context,budget,incumbent=first)
    reuse=second.details['continuation']
    assert reuse['state_reused'] and reuse['incumbent_source']=='continuation'
    assert reuse['caller_incumbent_accepted'] and reuse['pair_cuts_reused']==1
    assert reuse['single_results_reused']==sum(map(len,catalog.columns.values()))
    assert not reuse['solver_internal_state_reused']
    assert second.certified and second.objective<=first.objective
    assert budget.solver_wall_used==used+.25 and budget.refinement_rounds_used==rounds+1
    assert budget.solver_wall_used<=8. and budget.hint_wall_used==3.
    assert len(observed)==2 and all(sum(row['hint_values'])==2 for row in observed)


def test_proved_single_and_prefix_nogood_reused_but_unknown_never_becomes_cut(monkeypatch):
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    timed_solver(monkeypatch,clock)
    catalog=ordered_proof_catalog()
    budget=sp.PlanningBudget.for_profile('quality_v5');budget.wall_start=0.;budget.solver_call_seconds=.25
    class ProofContext(DiscreteContext):
        def check_single(self,column):
            bad=bool(column.plans and column.plans[0].trajectory_hash=='unsafe0')
            return ValidationResult(not bad,'REJECTED' if bad else 'SAFE')
        def check_batch(self,columns): return ValidationResult(True,'SAFE'),()
        def check_commit_prefix(self,columns):
            bad=any(c.plans and c.plans[0].trajectory_hash.startswith('unsafe') for c in columns)
            return ValidationResult(not bad,'REJECTED' if bad else 'ACCEPTED')
    context=ProofContext(budget);scheduler=sp.RollingHorizonCPSATScheduler()
    seed=certified_keep(catalog,context)
    first=scheduler.plan(None,catalog,context,budget,incumbent=seed)
    assert first.details['single_cuts']==1 and first.details['prefix_nogoods']==1
    second=scheduler.plan(None,catalog,context,budget)
    reuse=second.details['continuation']
    assert reuse['single_cuts_reused']==1 and reuse['prefix_nogoods_reused']==1
    unknown=RefinementContext(budget,unknown=True)
    third=scheduler.plan(None,catalog,unknown,budget,incumbent=certified_keep(catalog,unknown))
    assert third.details['solver_stop_reason']=='VALIDATION_UNKNOWN'
    assert third.details['pair_cuts']==third.details['single_cuts']==third.details['prefix_nogoods']==0


@pytest.mark.parametrize('change',('objective','resources','energy','reference','snapshot','evaluator'))
def test_full_catalog_or_context_change_invalidates_proofs(change):
    catalog=refinement_catalog();budget=sp.PlanningBudget.for_profile('quality_v5')
    context=DiscreteContext(budget);scheduler=sp.RollingHorizonCPSATScheduler()
    context.snapshot_key='visible-v1';context._prefix_results={'old':'certificate'}
    state,_=scheduler._continuation(None,catalog,context)
    state['exclusions'].add(('proof-a','proof-b'))
    columns={aid:tuple(rows) for aid,rows in catalog.columns.items()}
    old=columns['a'][1]
    if change=='objective': new=replace(old,objective=(0,-1,0,99.,0.))
    elif change=='resources': new=replace(old,resources=(Reservation(('new',0),'a',0.,10.),))
    elif change=='energy': new=replace(old,energy_j=1.)
    elif change=='reference': new=replace(old,reference=sp.NormalReference('KNOWN',energy_j=1.))
    else: new=old
    assert new.signature==old.signature
    columns['a']=(columns['a'][0],new,*columns['a'][2:])
    changed=replace(catalog,columns=columns)
    if change=='snapshot': context.snapshot_key='visible-v2'
    if change=='evaluator': context.prefix_evaluator=lambda prefix:ValidationResult(True,'ACCEPTED')
    state,report=scheduler._continuation(None,changed,context)
    assert report['state_invalidated'] and report['state_invalidations']==1
    assert not state['exclusions'] and report['pair_cuts_reused']==0
    if change=='evaluator': assert not context._prefix_results


def test_forged_caller_incumbent_cannot_authorize_conflicting_prefix(monkeypatch):
    pytest.importorskip('ortools.sat.python.cp_model')
    keep_hint(monkeypatch)
    budget=sp.PlanningBudget.for_profile('quality_v5');budget.solver_wall_used=8.
    catalog=refinement_catalog();context=RefinementContext(budget)
    selected=(catalog.columns['a'][1],catalog.columns['anchor'][0])
    forged=sp.PlanningResult(selected,context.prefix(selected),ValidationResult(True,'ACCEPTED'),
        'FEASIBLE',sp._sum_key(selected),certified=True)
    result=sp.RollingHorizonCPSATScheduler().plan(None,catalog,context,budget,incumbent=forged)
    assert not result.details['continuation']['caller_incumbent_accepted']
    assert result.certified and all(c.kind=='KEEP' for c in result.selected)


@pytest.mark.parametrize('stage_cap',(None,.9))
def test_actual_solver_allocation_respects_ready_and_stage_deadlines(monkeypatch,stage_cap):
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    observed=timed_solver(monkeypatch,clock,startup=.1)
    budget=sp.PlanningBudget.for_profile('quality_v5');budget.wall_start=0.;budget.candidate_seconds=6.
    budget.planning_wall_deadline=2.3
    catalog=discrete_catalog();context=DiscreteContext(budget);scheduler=sp.RollingHorizonCPSATScheduler()
    seed=certified_keep(catalog,context)
    if stage_cap is None:
        result=scheduler.plan(None,catalog,context,budget,incumbent=seed)
    else:
        with budget.stage_limit(stage_cap,'TEST_STAGE'):
            result=scheduler.plan(None,catalog,context,budget,incumbent=seed)
    absolute=min(2.3,stage_cap) if stage_cap is not None else 2.3
    assert observed
    for row in observed:
        assert row['limit']<=absolute-row['at']+1.e-9
    for row in result.solver_records:
        assert row['allocated_seconds']<=row['remaining_deadline_s']+1.e-9
    assert budget.wall_start==0. and budget.planning_wall_deadline==2.3
    assert budget.candidate_seconds==6. and budget.solver_seconds==8. and budget.soft_seconds==30.


def test_past_ready_deadline_starts_no_generation_inference_or_solve(monkeypatch):
    _,snapshot,budget,runtime,a,node,descriptors,advisor=grouped_setup()
    clock=Clock();clock.now=1.;monkeypatch.setattr(sp.time,'perf_counter',clock)
    budget.wall_start=0.;budget.planning_wall_deadline=0.
    assert runtime.order(snapshot,descriptors,budget)==descriptors and not advisor.batches
    monkeypatch.setattr(sp,'_generate',lambda *args:pytest.fail('new generation after deadline'))
    assert sp.ReferenceProvider()._generate_unobserved(snapshot,a,node,0,0.,35.,budget)==(False,None)
    cp=pytest.importorskip('ortools.sat.python.cp_model')
    monkeypatch.setattr(cp.CpSolver,'Solve',lambda *args,**kwargs:pytest.fail('solver after deadline'))
    context=DiscreteContext(budget)
    result=sp.RollingHorizonCPSATScheduler().plan(None,discrete_catalog(),context,budget)
    assert not result.solver_records and result.details['solver_stop_reason']=='READY_DEADLINE'
    assert not budget.available() and not budget.search_available() and not budget.evaluation_available()


def test_callback_copies_absolute_port_deadline_without_resetting_totals(monkeypatch):
    _,snapshot,_,runtime,_,_,_,_=grouped_setup()
    provider=sp.CandidateProvider(advisor_runtime=runtime)
    catalog=sp.CandidateCatalog({}, {}, {}, ())
    observed=[]
    def batches(snapshot,budget,context):
        observed.append(budget)
        yield catalog
    monkeypatch.setattr(provider,'iter_progressive_columns',batches)
    callback=sp.NormalSchedulerCallback(candidate_provider=provider,online_policy='progressive_v1')
    monkeypatch.setattr(callback.scheduler,'plan',lambda *args,**kwargs:sp.PlanningResult((),(),None,'UNKNOWN',(0,0,0,0,0)))
    deadline=sp.time.perf_counter()+10.
    port=SimpleNamespace(planning_wall_deadline=deadline,build_planning_snapshot=lambda:snapshot)
    callback(port)
    assert observed[0].planning_wall_deadline==deadline
    record=callback.records[-1]
    assert record['ready_budget']['planning_wall_deadline']==deadline
    assert 0.<record['ready_budget']['remaining_at_return_s']<=record['ready_budget']['remaining_at_start_s']<=10.
    assert record['planning_budget']['candidate_seconds']==6. and record['planning_budget']['solver_seconds']==8.


def test_active_stage_deadline_blocks_generation_and_search_without_reset(monkeypatch):
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    budget=sp.PlanningBudget.for_profile('quality_v5');budget.wall_start=0.
    with budget.stage_limit(1.,'BOUNDED_STAGE'):
        clock.now=1.
        assert not budget.available() and not budget.search_available() and not budget.evaluation_available()
        assert budget.generation_stop_reason()=='BOUNDED_STAGE' and budget.remaining_seconds()==0.
    assert budget.available() and budget.wall_start==0. and budget.solver_seconds==8.


def test_expired_deadline_rejects_caller_seed_without_query_or_new_certificate(monkeypatch):
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    budget=sp.PlanningBudget.for_profile('quality_v5');budget.wall_start=0.
    catalog=discrete_catalog();context=DiscreteContext(budget)
    seed=certified_keep(catalog,context)
    budget.planning_wall_deadline=0.;clock.now=1.
    monkeypatch.setattr(context,'check_batch',lambda *args:pytest.fail('seed query after deadline'))
    assert sp.RollingHorizonCPSATScheduler._certified_seed(seed,catalog,context,budget) is None


def test_direct_seed_validation_binds_current_budget_and_restores_original(monkeypatch):
    pytest.importorskip('ortools.sat.python.cp_model')
    _,snapshot,_,_,aircraft,_,_,_=grouped_setup()
    clock=Clock();monkeypatch.setattr(sp.time,'perf_counter',clock)
    budget=sp.PlanningBudget.for_profile('quality_v5');budget.wall_start=0.;budget.planning_wall_deadline=2.
    context=sp.EvaluationContext(snapshot,budget=None)
    keep=sp.PredictionColumn(aircraft.id,'KEEP',(),(),(0,0,0,0,0),sp._node(snapshot,aircraft))
    move=fake_move(aircraft.id)
    catalog=sp.CandidateCatalog({aircraft.id:(keep,move)},{aircraft.id:(0,)},{},())
    seed=sp.PlanningResult((move,),move.plans,ValidationResult(True,'ACCEPTED'),'FEASIBLE',move.objective,certified=True)
    seen=[]
    def batch(columns):
        seen.append(context.budget)
        clock.now=2.
        assert not context._time_ok()
        return context._time_rejection(),()
    monkeypatch.setattr(context,'check_batch',batch)
    monkeypatch.setattr(context,'check_commit_prefix',lambda *args:pytest.fail('certification after seed deadline'))
    result=sp.RollingHorizonCPSATScheduler().plan(snapshot,catalog,context,budget,incumbent=seed)
    assert seen==[budget] and context.budget is None
    assert not result.certified and not result.details['continuation']['caller_incumbent_accepted']


def test_callback_canonicalizes_best_before_passing_it_into_next_catalog(monkeypatch):
    _,snapshot,_,runtime,aircraft,_,_,_=grouped_setup()
    provider=sp.CandidateProvider(advisor_runtime=runtime)
    keep=sp.PredictionColumn(aircraft.id,'KEEP',(),(),(0,0,0,0,0),sp._node(snapshot,aircraft))
    first=fake_move(aircraft.id)
    updated=replace(first,objective=(0,-1,0,2.,0))
    catalogs=[sp.CandidateCatalog({aircraft.id:(keep,c)},{aircraft.id:(0,)},{},(aircraft.id,)) for c in (first,updated)]
    def fake_progressive(*args):
        yield from catalogs
    monkeypatch.setattr(provider,'iter_progressive_columns',fake_progressive)
    import terminal_marl.core.rotating_slots.stage3_learning.catalog_control as control
    monkeypatch.setattr(control,'expansion_reasons',lambda *args:['NO_COMPLETE_CHAIN_OBSERVED'])
    callback=sp.NormalSchedulerCallback('rh_cp_sat',candidate_provider=provider,online_policy='progressive_v1')
    seeds=[]
    def plan(snapshot,catalog,context,budget,*,incumbent=None):
        seeds.append(incumbent)
        column=catalog.columns[aircraft.id][1]
        certified=len(seeds)==1
        return sp.PlanningResult((column,),column.plans,ValidationResult(certified,'SAFE' if certified else 'UNKNOWN'),
            'FEASIBLE',column.objective,certified=certified)
    monkeypatch.setattr(callback.scheduler,'plan',plan)
    port=SimpleNamespace(build_planning_snapshot=lambda:snapshot,
        commit_validated_prefix=lambda *args,**kwargs:ValidationResult(False,'REJECTED'))
    callback(port)
    assert seeds[0] is None and seeds[1].selected==(updated,)
    assert seeds[1].objective==updated.objective
    assert callback.last_result.certified and callback.last_result.objective==updated.objective
