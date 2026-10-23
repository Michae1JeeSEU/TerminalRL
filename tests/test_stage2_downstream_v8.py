"""Bounded downstream approximation: actual worlds and shared CP/Python cost."""
from dataclasses import replace
from types import SimpleNamespace
import math
import pytest

from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider, EnergySlackScheduler, EvaluationContext, NormalSchedulerCallback,
    PlanningBudget, PLANNING_PROFILE_DOWNSTREAM, PLANNING_PROFILE_LEGACY,
    NormalReference, PlanningNode, ReferenceProvider, _node, objective_schema_id,
)
from terminal_marl.core.rotating_slots.stage2_downstream import (
    Demand, JointDownstream, ENERGY_UNIT_J, describe_column, ticks,
)
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID
from terminal_marl.core.rotating_slots.trajectory import generate_final, make_stable_plan
from terminal_marl.core.rotating_slots.resources import Reservation
from terminal_marl.core.rotating_slots.config import load_service_profile
from terminal_marl.core.rotating_slots.rescue_rules import RESCUE_PROFILE_V7


def world_catalog(*, release=0., ring="middle", soc=.9):
    w=RotatingSlotWorld(config=load_service_profile("service_half_v1"),experiment_policy_id=STAGE2_POLICY_ID)
    w.rescue_rule_profile=RESCUE_PROFILE_V7
    w.seed_hold("a",ring,0,soc)
    if release:
        for pad in range(6): w.seed_service("ground"+str(pad),pad,remaining=release)
    w.configure_practical_runtime(True)
    s=w.build_planning_snapshot(); b=PlanningBudget.for_profile("quality_v5")
    p=CandidateProvider(planning_profile=PLANNING_PROFILE_DOWNSTREAM)
    ctx=EvaluationContext(s,budget=b); catalog=p.get_columns(s,b,ctx)
    result=EnergySlackScheduler().plan(s,catalog,ctx,b)
    return w,s,p,ctx,catalog,result,b


def test_real_pad_soon_free_keeps_immediate_complete_opportunity():
    _,_,_,_,cat,result,b=world_catalog(release=10.)
    assert result.certified and result.selected[0].plans[-1].pad_id is not None
    assert result.selected[0].plans[0].start_time==0.
    assert len(cat.columns["a"])<=10 and b.attempts["a"]<=64


def test_real_busy_pad_delayed_complete_chain_enters_catalog_and_is_selected():
    w,s,p,ctx,cat,result,b=world_catalog(release=150.)
    delayed=[c for c in cat.columns["a"] if c.plans and c.plans[0].start_time>0 and c.plans[-1].pad_id is not None]
    assert delayed and result.selected[0] in delayed
    assert result.selected[0].plans[-1].end_time<=240.
    assert result.certified and not result.prefix  # prediction is not a booking
    assert b.screening["delayed_full_chain_retained"]>0
    # Reuse the certified epochs until the unchanged five-second commit window.
    p.previous={"a":result.selected[0].plans}
    start=result.selected[0].plans[0].start_time
    w.advance(start-5.)
    later=w.build_planning_snapshot(); laterctx=EvaluationContext(later)
    latercat=p.get_columns(later,PlanningBudget(max_attempts=0),laterctx)
    reused=next(c for c in latercat.columns["a"] if c.plans)
    assert reused.plans==result.selected[0].plans
    assert laterctx.prefix((reused,))==(reused.plans[0],)
    assert laterctx.check_commit_prefix((reused,)).accepted


def test_real_long_busy_pad_energy_sufficient_selects_upstream_wait():
    _,_,_,_,cat,result,b=world_catalog(release=600.,soc=.95)
    assert result.certified and result.selected[0].plans[0].start_time>0.
    assert result.selected[0].plans[0].source_ring=="middle"
    assert any(c.plans and c.plans[0].start_time==0 for c in cat.columns["a"])
    assert b.attempts["a"]<=64 and len(cat.columns["a"])<=10


def test_real_low_energy_inner_takes_feasible_immediate_landing():
    _,_,_,_,_,result,_=world_catalog(ring="inner",soc=.4)
    assert result.certified and result.prefix
    assert result.selected[0].plans[0].start_time==0.
    assert result.selected[0].plans[-1].pad_id is not None


def test_low_energy_middle_preserves_safe_immediate_chain_over_upstream_wait():
    # With every pad occupied until 150 there is no immediate safe complete
    # chain. Use the existing soon-free fixture, without changing physics.
    _,s,p,ctx,_,high_result,_=world_catalog(release=10.)
    immediate=high_result.selected[0]
    a=s.aircraft["a"]; node=_node(s,a)
    hold=make_stable_plan("a","middle",0,0.,110.,config=s.config)
    q=ctx.energy_portions(((hold,0.,110.),))
    later_node=replace(node,time=110.,soc=node.soc-q[1],peak=node.peak+q[2])
    later_ref=p.references.get(s,a,node=later_node,budget=PlanningBudget.for_profile("quality_v5"))
    assert later_ref.status=="KNOWN"
    delayed=p._column(s,a,later_ref.plans,"MOVE",ctx)
    assert ctx.check_single(delayed).accepted and delayed.energy_j>immediate.energy_j
    capacity=s.config.energy.equivalent_voltage_V*s.config.energy.reference_capacity_Ah*3600.
    soc=.25+(immediate.energy_j+delayed.energy_j)/(2.*capacity)
    low=replace(s,aircraft=dict(s.aircraft,a=replace(a,soc=soc)))
    lowctx=EvaluationContext(low)
    low_immediate=p._column(low,low.aircraft["a"],immediate.plans,"MOVE",lowctx)
    assert lowctx.check_single(low_immediate).accepted
    # check_single covers geometry/background; normal reserve is enforced by
    # the unchanged candidate energy replay, before catalog admission.
    assert soc-immediate.energy_j/capacity>.25
    assert soc-delayed.energy_j/capacity<.25
    immediate_ref=ReferenceProvider._reference(low,_node(low,low.aircraft["a"]),immediate.plans,"test")
    assert p._reference_seed(low,low.aircraft["a"],_node(low,low.aircraft["a"]),immediate_ref,PlanningBudget()) is not None
    assert p._reference_seed(low,low.aircraft["a"],_node(low,low.aircraft["a"]),later_ref,PlanningBudget()) is None
    p.previous={"a":immediate.plans}
    catalog=p.get_columns(low,PlanningBudget(max_attempts=0),lowctx)
    result=EnergySlackScheduler().plan(low,catalog,lowctx,PlanningBudget())
    assert result.certified and result.prefix and result.selected[0].plans==immediate.plans


def demand(a,*,ready=0,duration=100,full=False,real=(),base=1000,rate=10,unknown=0):
    return Demand(a,full,ready,duration,base,0 if full else rate,10000000,unknown,0,0,
        ready+135,"middle",{"reference_status":"UNKNOWN" if unknown else "ESTIMATED"},real)


def column(a,name="wait"):
    return SimpleNamespace(aircraft_id=a,signature=(a,name))


def test_shared_queue_uses_different_gaps_and_complete_chain_changes_other_tail():
    a,b=column("a"),column("b"); full=column("a","full")
    demands={a.signature:demand("a",duration=30),b.signature:demand("b",duration=30),
             full.signature:demand("a",full=True,real=((0,150,300,"a"),))}
    queue=JointDownstream(demands,((0,50,150,"ground"),),("a","b"),pad_count=1)
    _,rows=queue.evaluate((a,b))
    assert rows["a"]["tail_wait_seconds"]==0.
    assert rows["b"]["tail_wait_seconds"]==15.
    _,changed=queue.evaluate((full,b))
    assert changed["b"]["tail_wait_seconds"]==0.
    assert changed["a"]["virtual_pad"] is None
    assert changed["b"]["total_j"]<rows["b"]["total_j"]


@pytest.mark.parametrize("choose_full",[False,True])
def test_fixed_selected_cp_lex_cost_matches_python_joint_queue(choose_full):
    cp=pytest.importorskip("ortools.sat.python.cp_model")
    a,b=column("a"),column("b"); full=column("a","full")
    demands={a.signature:demand("a",duration=30,unknown=1),b.signature:demand("b",duration=30),
             full.signature:demand("a",full=True,real=((0,150,300,"a"),))}
    queue=JointDownstream(demands,((0,50,150,"ground"),),("a","b"),pad_count=1)
    selected=(full if choose_full else a,b)
    model=cp.CpModel(); lookup={c.signature:c for c in (a,full,b)}
    variables={key:model.NewBoolVar(str(key)) for key in lookup}
    model.AddExactlyOne([variables[a.signature],variables[full.signature]])
    for key,var in variables.items(): model.Add(var==int(key in {c.signature for c in selected}))
    expressions=queue.cp_expressions(model,variables,lookup)
    solver=cp.CpSolver(); solver.parameters.num_search_workers=1
    for expression in expressions:
        model.Minimize(expression)
        assert solver.Solve(model)==cp.OPTIMAL
        model.Add(expression==solver.Value(expression))
    assert tuple(solver.Value(e) for e in expressions)==queue.evaluate(selected)[0]


def test_complete_cross_horizon_energy_is_integrated_once():
    w=RotatingSlotWorld(); w.seed_hold("a","inner",0,.9); w.configure_practical_runtime(True)
    s=w.build_planning_snapshot(); a=s.aircraft["a"]; ctx=EvaluationContext(s)
    plan=next(r.plan for pad in range(6) if (r:=generate_final("a","inner",0,pad,230.,config=s.config)).accepted)
    p=CandidateProvider(planning_profile=PLANNING_PROFILE_DOWNSTREAM)
    c=p._column(s,a,(plan,),"MOVE",ctx)
    d=describe_column(s,c,p.references,ctx)
    hold=make_stable_plan("a","inner",0,0.,plan.start_time,config=s.config)
    expected=ctx.energy_portions(((hold,0.,plan.start_time),(plan,plan.start_time,plan.end_time)))[0]
    assert plan.end_time>240.
    assert math.isclose(d.base_energy*ENERGY_UNIT_J,expected,abs_tol=.11)
    assert math.isclose(d.diagnostics["window_j"]+d.diagnostics["tail_motion_j"],expected,abs_tol=.01)


def test_complete_final_after_horizon_reports_known_tail_hold_without_double_energy():
    w=RotatingSlotWorld(config=load_service_profile("service_half_v1"),experiment_policy_id=STAGE2_POLICY_ID)
    w.rescue_rule_profile=RESCUE_PROFILE_V7
    w.seed_hold("a","inner",0,.9);w.configure_practical_runtime(True)
    s=w.build_planning_snapshot();a=s.aircraft["a"];ctx=EvaluationContext(s)
    plan=next(r.plan for pad in range(6) if (r:=generate_final("a","inner",0,pad,300.,config=s.config)).accepted)
    p=CandidateProvider(planning_profile=PLANNING_PROFILE_DOWNSTREAM)
    c=p._column(s,a,(plan,),"MOVE",ctx)
    d=describe_column(s,c,p.references,ctx)
    queue=JointDownstream({c.signature:d},(),("a",),now=ticks(s.time),pad_count=6)
    integer_key,rows=queue.evaluate((c,));row=rows["a"]
    assert row["window_wait_seconds_by_ring"]["inner"]==240.
    assert row["tail_known_wait_seconds_by_ring"]["inner"]==60.
    assert row["estimated_wait_seconds_by_ring"]=={"outer":0.,"middle":0.,"inner":300.}
    assert row["tail_wait_seconds"]==0. and row["virtual_pad"] is None
    hold=make_stable_plan("a","inner",0,0.,300.,config=s.config)
    expected=ctx.energy_portions(((hold,0.,300.),(plan,plan.start_time,plan.end_time)))[0]
    assert integer_key[3]==d.base_energy
    assert math.isclose(row["total_j"],expected,abs_tol=.11)
    assert math.isclose(row["window_j"]+row["tail_j"],expected,abs_tol=.01)


def test_unknown_nonzero_proxy_is_uncertainty_not_observed_failure():
    w=RotatingSlotWorld();w.seed_hold("a","middle",0,.9)
    s=w.build_planning_snapshot();p=CandidateProvider(planning_profile=PLANNING_PROFILE_DOWNSTREAM)
    cat=p.get_columns(s,PlanningBudget(max_attempts=0)); c=cat.columns["a"][0]
    key,rows=cat.downstream.evaluate((c,))
    assert key[0]==0 and key[2]>0 and rows["a"]["unknown_risk_flag"]
    assert rows["a"]["tail_motion_j"]>0
    assert rows["a"]["reference_estimate_kind"]=="normal_energy_and_time_lower_bound_proxy"
    assert not w.failed


def test_profile_checkpoint_matching_and_legacy_schema():
    legacy=NormalSchedulerCallback(); modern=NormalSchedulerCallback(planning_profile=PLANNING_PROFILE_DOWNSTREAM)
    assert legacy.planning_profile==PLANNING_PROFILE_LEGACY
    assert "planning_profile" not in legacy.state_dict()
    legacy.load_state_dict(legacy.state_dict()); modern.load_state_dict(modern.state_dict())
    with pytest.raises(ValueError,match="planning profile mismatch"): modern.load_state_dict(legacy.state_dict())
    with pytest.raises(ValueError,match="planning profile mismatch"): legacy.load_state_dict(modern.state_dict())
    assert objective_schema_id(PLANNING_PROFILE_LEGACY)=="reliability_throughput_progress_tail_v2"


def test_declared_unreserved_rescue_aircraft_remains_unknown_queue_demand():
    w=RotatingSlotWorld();w.seed_hold("a","middle",0,.9);w.seed_hold("b","middle",8,.9)
    s=w.build_planning_snapshot();a=s.aircraft["a"]
    s=replace(s,aircraft=dict(s.aircraft, a=replace(a,emergency=replace(a.emergency,declared=True))))
    p=CandidateProvider(planning_profile=PLANNING_PROFILE_DOWNSTREAM)
    cat=p.get_columns(s,PlanningBudget(max_attempts=0)); selected=(cat.columns["b"][0],)
    key,rows=cat.downstream.evaluate(selected)
    assert set(cat.columns)=={"b"} and set(rows)=={"a","b"}
    assert set(cat.downstream.fixed_demands)=={"a"}
    assert rows["a"]["reference_status"]=="UNKNOWN" and rows["a"]["unknown_risk_flag"]
    assert key[0]==0


def test_fixed_final_calendar_is_counted_once_without_virtual_job():
    w=RotatingSlotWorld();w.seed_hold("a","inner",0,.9);s=w.build_planning_snapshot()
    final=generate_final("a","inner",0,0,0.,config=s.config).plan
    assert final is not None
    row=Reservation(0,"a",final.end_time-13.5,final.end_time+120.)
    s=replace(s,active={"a":final},pads=(row,))
    cat=CandidateProvider(planning_profile=PLANNING_PROFILE_DOWNSTREAM).get_columns(s,PlanningBudget(max_attempts=0))
    assert not cat.columns and not cat.downstream.assignment and not cat.downstream.fixed_demands
    assert len(cat.downstream.fixed_identity)==1


def test_delayed_reference_uses_supported_screen_keys_at_35_minus_ulp(monkeypatch):
    p=CandidateProvider(planning_profile=PLANNING_PROFILE_DOWNSTREAM)
    epoch=9.388737514630813
    final_start=epoch+40.; final_end=final_start+35.+13.5
    reference_duration=final_end-final_start-13.5
    assert reference_duration==math.nextafter(35.,0.) and int(reference_duration)==34
    transfer=SimpleNamespace(pad_id=None,target_slot=0,start_time=epoch,end_time=final_start)
    final=SimpleNamespace(pad_id=0,target_slot=None,start_time=final_start,end_time=final_end)
    ref=NormalReference("KNOWN",plans=(transfer,final))
    # Missing (0,35) must never be invented as an implicitly safe key;
    # reject witnesses and durations below the reference are also excluded.
    screen={(0,40):None,(3,35):("REJECTED",),(1,35):None,(0,30):None}
    calls=[]
    def generate(snapshot,a,node,target,start,duration,budget):
        calls.append((node.ring,target,duration))
        landing=node.ring=="inner"
        return True,SimpleNamespace(target_ring=None if landing else "inner",target_slot=None if landing else target,
            pad_id=target if landing else None,start_time=start,end_time=start+duration+(13.5 if landing else 0.))
    monkeypatch.setattr(p,"_search_offsets",lambda *args:(0.,10.))
    monkeypatch.setattr(p.references,"screen",lambda *args:screen)
    monkeypatch.setattr(p.references,"generate",generate)
    monkeypatch.setattr(p.references,"move_energy",lambda *args:(0.,0.,0.,0.,0.))
    node=PlanningNode("middle",0,epoch,.9,0.,0)
    result=list(p._delayed_reference_seed(SimpleNamespace(config=None),SimpleNamespace(id="a"),node,ref,PlanningBudget()))
    assert len(result)==1
    assert [row for row in calls if row[0]=="inner"]==[("inner",1,35)]


def test_real_noninteger_epoch_delayed_reference_catalog_and_prefix():
    w=RotatingSlotWorld(config=load_service_profile("service_half_v1"),experiment_policy_id=STAGE2_POLICY_ID)
    w.rescue_rule_profile=RESCUE_PROFILE_V7
    w.seed_hold("a","middle",0,.9)
    for pad in range(6): w.seed_service("ground"+str(pad),pad,remaining=150.)
    w.advance(9.388737514630813)
    s=w.build_planning_snapshot();p=CandidateProvider(planning_profile=PLANNING_PROFILE_DOWNSTREAM)
    budget=PlanningBudget.for_profile("quality_v5");ctx=EvaluationContext(s,budget=budget)
    catalog=p.get_columns(s,budget,ctx)
    source=p.references.cache[(s.config.config_hash,"middle",0,s.time,0,"a")]
    last=source.plans[-1]
    assert last.end_time-last.start_time-13.5==math.nextafter(35.,0.)
    assert budget.screening["delayed_full_chain_retained"]>0
    result=EnergySlackScheduler().plan(s,catalog,ctx,budget)
    assert result.certified and ctx.check_commit_prefix(result.selected).accepted
    assert len(catalog.columns["a"])<=10 and budget.attempts["a"]<=64
