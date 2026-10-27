"""Search-only deadline and certificate handoff; no formal rollouts or fit."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots import planning_latency as latency
from terminal_marl.core.rotating_slots import stage2_planning as sp
from terminal_marl.core.rotating_slots.models import ValidationResult
from tests.test_stage2_quality_budget import Clock, discrete_catalog, DiscreteContext
from tests.test_rotating_slots_latency_ranking import grouped_setup
from tests.test_rotating_slots_stage3_online_catalog import fake_move


def budget_at(clock,monkeypatch):
    monkeypatch.setattr(sp.time,"perf_counter",clock)
    b=sp.PlanningBudget.for_profile("quality_v5"); b.wall_start=0.
    b.candidate_seconds=6.  # Existing online-policy .5 candidate scale.
    b.planning_wall_origin=0.; b.planning_wall_deadline=19.; b.search_wall_deadline=15.
    b.search_handoff_reserve_seconds=4.
    return b


def test_search_stop_preserves_ready_and_original_phase_limits(monkeypatch):
    clock=Clock();b=budget_at(clock,monkeypatch)
    with b.stage_limit(3.,"HINT_BUDGET"):
        assert b._stage_limits[-1][0]==3.
        clock.now=2.5
        assert b.search_available() and b.heavy_allowed("prefix")
    clock.now=15.
    assert not b.available() and not b.evaluation_available() and not b.search_available()
    assert b.evaluation_stop_reason()=="SEARCH_HANDOFF_DEADLINE" and b.ready_deadline_ok()
    assert b.planning_wall_deadline-clock.now==4.
    assert (b.candidate_seconds,b.hint_seconds,b.solver_seconds,b.soft_seconds)==(6.,3.,8.,30.)
    assert b.solver_validation_headroom()==3.
    b.solver_wall_used=2.
    assert b.solver_seconds-b.solver_wall_used==6.
    b.search_wall_deadline=None
    assert b.solver_validation_headroom()==0. and b.search_available()


def test_uncached_heavy_gates_and_cached_certificates_keep_unknown_out_of_cache(monkeypatch):
    _,snapshot,_,_,a,_,_,_=grouped_setup()
    clock=Clock();b=budget_at(clock,monkeypatch)
    context=sp.EvaluationContext(snapshot,budget=b,reuse_prefix_results=True,
        prefix_evaluator=lambda prefix:ValidationResult(True,"ACCEPTED"),prefix_evaluator_identity="fixture")
    move=fake_move(a.id)
    monkeypatch.setattr(context,"_check_single",lambda column:ValidationResult(True,"SAFE"))
    clock.now=1.
    assert context.check_single(move).accepted
    assert context.check_commit_prefix((move,)).accepted
    clock.now=14.5
    monkeypatch.setattr(context,"_check_single",lambda c:pytest.fail("uncached single started near search stop"))
    context.prefix_evaluator=lambda prefix:pytest.fail("uncached prefix started near search stop")
    assert context.check_single(move).accepted and context.check_commit_prefix((move,)).accepted
    other=fake_move(a.id,tag="new")
    assert context.check_single(other).status=="UNKNOWN"
    assert context.check_commit_prefix((other,)).status=="UNKNOWN"
    monkeypatch.setattr(sp,"validate_separation",lambda *args:pytest.fail("geometry started near search stop"))
    assert context._geometry_validation(None).status=="UNKNOWN"
    assert len(context.singles)==1 and len(context._prefix_results)==1
    b.heavy_highwater["geometry"]=2.
    assert b.heavy_estimate("geometry")==2.5
    clock.now=10.
    with b.heavy_work("geometry"): clock.now=16.
    assert b.heavy_checks["geometry"]["overrun_count"]==1
    assert b.heavy_checks["geometry"]["max_wall_s"]==6.


def test_unknown_resource_protection_cannot_turn_into_empty_background_or_cached_safe(monkeypatch):
    _,snapshot,_,_,a,_,_,_=grouped_setup()
    foreign=replace(a,id="foreign",status="INBOUND")
    snapshot=replace(snapshot,aircraft={a.id:a,foreign.id:foreign})
    clock=Clock();b=budget_at(clock,monkeypatch)
    context=sp.EvaluationContext(snapshot,budget=b);context.decisions={a.id}
    monkeypatch.setattr(context,"resources",lambda *args:None)
    monkeypatch.setattr(context,"check_pair",lambda *args:pytest.fail("unknown resources became a pair certificate"))
    result=context.check_single(replace(fake_move(a.id),kind="KEEP"))
    assert not result.accepted and result.status=="UNKNOWN"
    assert result.reason=="SEARCH_RESOURCE_PROTECTION_UNKNOWN" and not context.singles


def test_first_actual_prefix_and_strict_improvement_trace_are_scope_bound(monkeypatch):
    _,snapshot,_,_,a,_,_,_=grouped_setup()
    clock=Clock();b=budget_at(clock,monkeypatch)
    first=fake_move(a.id,objective=3.);better=fake_move(a.id,objective=1.,tag="better")
    cat=sp.CandidateCatalog({a.id:(first,better)},{a.id:(0,)},{},())
    b._certificate_context="snapshot1";b.bind_catalog(cat);b.certificate_source="cp_proposal"
    def certify(prefix):
        clock.now+=.2
        return ValidationResult(True,"ACCEPTED")
    context=sp.EvaluationContext(snapshot,budget=b,prefix_evaluator=certify,
        reuse_prefix_results=True,prefix_evaluator_identity="fixed_fixture")
    clock.now=1.
    assert context.check_commit_prefix((first,)).accepted
    b.certificate_event("cp_proposal",(first,),first.plans,event="CERTIFIED_OBJECTIVE",objective=first.objective)
    assert context.check_commit_prefix((better,)).accepted
    b.certificate_event("cp_proposal",(better,),better.plans,event="CERTIFIED_OBJECTIVE",objective=better.objective)
    assert [r["event"] for r in b.certificate_events]==["FIRST_NONEMPTY_CERTIFIED","STRICT_CERTIFIED_IMPROVEMENT"]
    assert [r["wall_elapsed_s"] for r in b.certificate_events]==pytest.approx([1.2,1.4])
    assert b.first_nonempty_certified["prefix_hashes"]==[first.plans[0].trajectory_hash]
    assert all(r["search_certificate_not_publication"] for r in b.certificate_events)
    clock.now=5.
    assert context.check_commit_prefix((better,)).accepted
    b.certificate_event("cp_proposal",(better,),better.plans,event="CERTIFIED_OBJECTIVE",objective=(0,-1,0,.75,0))
    assert b.certificate_events[-1]["wall_elapsed_s"]==5.
    assert b.certificate_events[-1]["certificate_wall_elapsed_s"]==pytest.approx(1.4)
    clock.now=6.
    assert context.check_commit_prefix((first,)).accepted
    b.certificate_event("cp_proposal",(first,),first.plans,event="CERTIFIED_OBJECTIVE",objective=(0,-1,0,.5,0))
    assert b.first_nonempty_certified["objective"]==list(first.objective)
    assert b.certificate_events[-1]["event"]=="STRICT_CERTIFIED_IMPROVEMENT"
    assert b.certificate_events[-1]["wall_elapsed_s"]==6.
    assert b.certificate_events[-1]["certificate_wall_elapsed_s"]==pytest.approx(1.2)
    changed=replace(cat,columns={a.id:(replace(first,objective=(0,-1,0,.5,0)),better)})
    b.bind_catalog(changed)
    b.certificate_event("canonical_rescore",(changed.columns[a.id][0],),first.plans,event="CATALOG_RESCORE",objective=changed.columns[a.id][0].objective)
    assert b.certificate_events[-1]["event"]=="CATALOG_RESCORE"
    assert b.certificate_events[-1]["catalog_scope"]!=b.certificate_events[0]["catalog_scope"]
    unknown=sp.EvaluationContext(snapshot,budget=b,prefix_evaluator=lambda prefix:ValidationResult(False,"UNKNOWN"))
    assert unknown.check_commit_prefix((first,)).status=="UNKNOWN"
    assert len(b.certificate_events)==5


@pytest.mark.parametrize("method",["energy_slack","rh_cp_sat"])
def test_callback_returns_certified_best_at_search_stop_without_new_improvement_or_explanation(monkeypatch,method,tmp_path):
    _,snapshot,_,runtime,a,_,_,_=grouped_setup()
    clock=Clock();monkeypatch.setattr(sp.time,"perf_counter",clock)
    move=fake_move(a.id)
    class Downstream:
        def key(self,columns): return sp._sum_key(columns)
        def evaluate(self,columns): pytest.fail("explanation evaluated after search stop")
    cat=sp.CandidateCatalog({a.id:(move,)},{a.id:(0,)},{},(),downstream=Downstream())
    provider=sp.CandidateProvider(advisor_runtime=runtime); generated=[];calls=[];committed=[]
    def catalogs(*args):
        generated.append(1);yield cat
        pytest.fail("generator resumed after search stop")
    monkeypatch.setattr(provider,"iter_progressive_columns",catalogs)
    callback=sp.NormalSchedulerCallback(method,candidate_provider=provider,online_policy="progressive_v1")
    def plan(s,c,context,budget,**kwargs):
        calls.append(budget)
        assert budget.search_wall_deadline==15. and budget.planning_wall_deadline==19.
        context.prefix_evaluator=lambda prefix:ValidationResult(True,"ACCEPTED")
        clock.now=3.
        evaluation=context.check_commit_prefix((move,))
        clock.now=15.
        return sp.PlanningResult((move,),move.plans,evaluation,"FEASIBLE",move.objective,certified=True)
    monkeypatch.setattr(callback.scheduler,"plan",plan)
    port=SimpleNamespace(planning_wall_deadline=19.,search_wall_deadline=15.,planning_wall_origin=-2.,
        search_handoff_reserve_seconds=4.,build_planning_snapshot=lambda:snapshot,
        commit_validated_prefix=lambda proof,**kwargs:committed.append(proof) or ValidationResult(False,"DEFERRED"))
    callback(port)
    assert len(calls)==len(generated)==len(committed)==1
    assert callback.last_result.certified and callback.last_result.prefix==move.plans
    assert callback.records[-1]["explanation_skipped_search_handoff"]
    trace=callback.records[-1]["search_handoff"]
    assert trace is port.search_handoff
    assert trace["first_nonempty_certified"]["wall_elapsed_s"]==5.
    assert trace["callback_return"]["wall_elapsed_s"]==17.
    assert trace["callback_return"]["prefix_hashes"]==[move.plans[0].trajectory_hash]
    assert (calls[0].candidate_seconds,calls[0].solver_seconds,calls[0].hint_seconds)==(6.,8.,3.)
    path=tmp_path/"search_handoff.json"
    path.write_text(json.dumps(trace,allow_nan=False),encoding="utf8")
    assert json.loads(path.read_text(encoding="utf8"))["first_nonempty_certified"]==trace["first_nonempty_certified"]


def test_wrapper_reserve_origin_and_checkpoint_binding_leave_ready_deadline_unchanged(monkeypatch):
    from tests.test_rotating_slots_planning_latency import CertifiedFinals,Durations
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    world=RotatingSlotWorld(); world.seed_hold("a","inner",0,.9)
    observed=[]
    class Callback(CertifiedFinals):
        def __call__(self,port):
            observed.append(port)
            super().__call__(port)
            port.search_handoff={"protocol":latency.SEARCH_HANDOFF_VERSION,"fixture":True}
        def state_dict(self): return {}
        def load_state_dict(self,state): pass
    clock=Clock();clock.now=100.;monkeypatch.setattr(latency.time,"perf_counter",clock)
    wrapper=latency.LatencyAwareScheduler(Callback(),lead_seconds=20.,search_handoff_reserve_seconds=4.,clock=Durations(2.))
    wrapper._service_until=2.
    wrapper(world._scheduler_port)
    port=observed[0]
    assert (port.planning_wall_origin,port.planning_wall_deadline,port.search_wall_deadline)==(100.,117.,113.)
    assert wrapper.records[0]["ready_deadline"]==19. and wrapper.records[0]["effective_time"]==20.
    assert wrapper.records[0]["search_handoff"] is port.search_handoff
    state=deepcopy(wrapper.state_dict())
    restored=latency.LatencyAwareScheduler(Callback(),lead_seconds=20.,search_handoff_reserve_seconds=4.)
    restored.load_state_dict(state)
    assert restored.pending==wrapper.pending
    with pytest.raises(ValueError,match="binding mismatch"):
        latency.LatencyAwareScheduler(Callback(),lead_seconds=20.).load_state_dict(state)
    legacy=latency.LatencyAwareScheduler(Callback(),lead_seconds=20.)
    legacy_state=legacy.state_dict();del legacy_state["search_handoff_reserve_seconds"]
    legacy.load_state_dict(legacy_state)


def test_real_cp_headroom_preserves_allocation_and_certified_incumbent(monkeypatch):
    pytest.importorskip("ortools.sat.python.cp_model")
    from tests.test_rotating_slots_cp_reuse_deadline import timed_solver
    clock=Clock();b=budget_at(clock,monkeypatch)
    observed=timed_solver(monkeypatch,clock)
    cat=discrete_catalog();context=DiscreteContext(b)
    scheduler=sp.RollingHorizonCPSATScheduler()
    selected=(cat.columns['a'][1],cat.columns['b'][0])
    seed=sp.PlanningResult(selected,context.prefix(selected),context.check_commit_prefix(selected),
        "FEASIBLE",sp._sum_key(selected),certified=True)
    b.solver_call_seconds=.25;clock.now=11.9
    first=scheduler.plan(None,cat,context,b,incumbent=seed)
    assert first.certified and observed and observed[0]["limit"]<=.10000001
    used=b.solver_wall_used
    clock.now=15.
    monkeypatch.setattr(context,"check_batch",lambda *args:pytest.fail("certified incumbent rechecked after stop"))
    result=scheduler.plan(None,cat,context,b)
    assert result.certified and result.prefix and result.objective==first.objective and b.solver_wall_used==used
    assert b.solver_seconds==8. and b.hint_wall_used==0.
