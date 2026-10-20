"""Focused Stage2 planning checks; test IDs identify only exercised clauses."""
from dataclasses import replace
from types import MappingProxyType
import json
import math
import time
import pytest

from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateCatalog, CandidateProvider, EnergySlackScheduler, EvaluationContext,
    NormalReference, NormalSchedulerCallback, PlanningBudget, PlanningNode, PredictionColumn,
    ReferenceProvider, RollingHorizonCPSATScheduler, _node, energy_slack,
    solver_coefficients)
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.energy import integrate_energy
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.resources import Reservation


def transfer(a,ring,slot,target,start=0.,version=0,duration=40.):
    r=generate_transfer(a,ring,slot,target,start,source_version=version,durations=(duration,))
    assert r.accepted,r
    return r.plan


@pytest.fixture(scope="module")
def outer_catalog():
    w=RotatingSlotWorld(); w.seed_hold("o","outer",0,.9)
    w.configure_practical_runtime(True)
    snapshot=w.build_planning_snapshot(); budget=PlanningBudget()
    provider=CandidateProvider(); context=EvaluationContext(snapshot)
    catalog=provider.get_columns(snapshot,budget,context)
    return w,snapshot,provider,context,catalog,budget


def test_T02_actual_outer_middle_inner_pad_and_only_first_committed(outer_catalog):
    w,snapshot,provider,context,catalog,budget=outer_catalog
    moving=[c for c in catalog.columns["o"] if c.kind=="MOVE"]
    assert moving
    chain=moving[0].plans
    assert [p.source_ring for p in chain]==["outer","middle","inner"]
    assert chain[-1].pad_id is not None and len(chain)==3
    result=EnergySlackScheduler().plan(snapshot,catalog,context,PlanningBudget())
    assert result.certified and len(result.prefix)==1
    assert w.commit_validated_prefix(result.evaluation).accepted
    assert len(w.active)==1 and not w.soft
    assert w.active["o"].target_ring=="middle"
    assert len(result.selected[0].plans)==3


def test_T03_slack_depends_on_remaining_normal_energy_not_soc():
    w=RotatingSlotWorld(); w.seed_hold("a","outer",0,.7); w.seed_hold("b","inner",9,.7)
    s=w.build_planning_snapshot()
    outer=NormalReference("KNOWN",soc_required=.3,peak_required=10.,duration=120.,energy_j=1.)
    inner=NormalReference("KNOWN",soc_required=.1,peak_required=5.,duration=60.,energy_j=1.)
    assert energy_slack(s,s.aircraft["a"],outer)<energy_slack(s,s.aircraft["b"],inner)
    unknown=energy_slack(s,s.aircraft["a"],NormalReference("UNKNOWN"))
    assert unknown[0]==1
    negative=replace(outer,soc_required=.6)
    assert energy_slack(s,s.aircraft["a"],negative)[0]==0


def test_T04_reference_unknown_is_not_zero_or_failure():
    w=RotatingSlotWorld(); w.seed_hold("o","outer",0,.9)
    s=w.build_planning_snapshot(); b=PlanningBudget(max_attempts=0)
    r=ReferenceProvider().get(s,s.aircraft["o"],budget=b)
    assert r.status=="UNKNOWN" and r.soc_required is None and r.energy_j is None
    cat=CandidateProvider().get_columns(s,b)
    assert len(cat.columns["o"])==2
    assert cat.columns["o"][0].objective[:4]==(0,0,3,901.)
    assert not w.failed


def test_T05_wait_retains_slot_and_pair_resource_conflict():
    w=RotatingSlotWorld(); w.seed_hold("o","outer",1,.9); w.seed_hold("m","middle",1,.9)
    s=w.build_planning_snapshot(); context=EvaluationContext(s); provider=CandidateProvider()
    move=provider._column(s,s.aircraft["o"],(transfer("o","outer",1,1),),"MOVE",context)
    wait=provider._column(s,s.aircraft["m"],(),"WAIT",context)
    assert any(r.resource==("middle",1) and math.isinf(r.end) for r in wait.resources)
    assert not context.check_pair(move,wait).accepted


def test_T06_actual_geometric_crossing_with_distinct_resources():
    w=RotatingSlotWorld(); w.seed_hold("a","outer",0,.9);w.seed_hold("b","outer",3,.9)
    s=w.build_planning_snapshot(); context=EvaluationContext(s); provider=CandidateProvider()
    p=transfer("a","outer",0,1,duration=90.)
    q=transfer("b","outer",3,0,duration=90.)
    a=provider._column(s,s.aircraft["a"],(p,),"MOVE",context)
    b=provider._column(s,s.aircraft["b"],(q,),"MOVE",context)
    assert not context.resource_conflict(a,b)
    r=context.check_pair(a,b)
    assert not r.accepted and r.status!="UNKNOWN"


def test_T07_joint_departures_not_prefiltered_against_other_permanent_hold():
    w=RotatingSlotWorld();w.seed_hold("o","outer",1,.9);w.seed_hold("m","middle",1,.9)
    s=w.build_planning_snapshot(); context=EvaluationContext(s);p=CandidateProvider()
    outer=p._column(s,s.aircraft["o"],(transfer("o","outer",1,1,5.),),"MOVE",context)
    middle=p._column(s,s.aircraft["m"],(transfer("m","middle",1,0,0.),),"MOVE",context)
    assert context.check_single(outer).accepted
    assert context.check_pair(outer,middle).accepted
    assert context.check_commit_prefix((outer,middle)).accepted


def test_T08_pad_descent_interval_release_not_touchdown():
    w=RotatingSlotWorld(); w.seed_hold("i","inner",0,.9); w.seed_service("g",0,30.)
    s=w.build_planning_snapshot(); context=EvaluationContext(s); provider=CandidateProvider()
    final=generate_final("i","inner",0,0,0).plan
    c=provider._column(s,s.aircraft["i"],(final,),"MOVE",context)
    assert context.check_single(c).accepted
    reservation=next(r for r in c.resources if r.resource==("pad",0))
    assert reservation.start==final.end_time-13.5
    w2=RotatingSlotWorld();w2.seed_hold("i","inner",0,.9)
    w2.seed_service("g",0,remaining=final.end_time-5.)
    s2=w2.build_planning_snapshot();ctx2=EvaluationContext(s2)
    c2=provider._column(s2,s2.aircraft["i"],(final,),"MOVE",ctx2)
    assert not ctx2.check_single(c2).accepted


def test_T10_safe_full_prediction_cannot_authorize_uncommitted_release():
    w=RotatingSlotWorld();w.seed_hold("o","outer",1,.9);w.seed_hold("m","middle",1,.9)
    s=w.build_planning_snapshot(); context=EvaluationContext(s);provider=CandidateProvider()
    outer=provider._column(s,s.aircraft["o"],(transfer("o","outer",1,1,5.),),"MOVE",context)
    middle=provider._column(s,s.aircraft["m"],(transfer("m","middle",1,0,6.),),"MOVE",context)
    assert context.check_pair(outer,middle).accepted
    assert context.prefix((outer,middle))==(outer.plans[0],)
    before=w.build_planning_snapshot()
    result=context.check_commit_prefix((outer,middle))
    assert not result.accepted and result.reason=="RESOURCE_CONFLICT"
    assert w.build_planning_snapshot()==before and not w.active


def test_T11_fixed_active_endpoint_uses_successor_version_and_soft_gap_energy():
    w=RotatingSlotWorld();w.seed_hold("o","outer",0,.9)
    first=transfer("o","outer",0,0)
    assert w.submit((first,)).accepted
    second=transfer("o","middle",0,0,first.end_time+5.,version=1)
    assert w.submit((second,)).accepted
    s=w.build_planning_snapshot(); n=_node(s,s.aircraft["o"])
    assert (n.ring,n.version,n.time)==("inner",2,second.end_time)
    expected=integrate_energy(first,0,first.end_time).soc_used+integrate_energy(second,second.start_time,second.end_time).soc_used
    assert n.soc<.9-expected  # five seconds holding is charged


def test_T13_repeated_reversed_pure_catalog_checks_do_not_mutate_world():
    w=RotatingSlotWorld();w.seed_hold("o","outer",0,.9)
    s=w.build_planning_snapshot();context=EvaluationContext(s);provider=CandidateProvider()
    cat=provider.get_columns(s,PlanningBudget(),context)
    before=w.build_planning_snapshot(); events=tuple(w.events)
    for columns in (cat.columns["o"],tuple(reversed(cat.columns["o"]))):
        for c in columns: context.check_single(c); context.check_commit_prefix((c,))
    assert w.build_planning_snapshot()==before and tuple(w.events)==events


def test_cp_dependency_missing_is_explicit():
    import importlib.util
    if importlib.util.find_spec("ortools") is not None: pytest.skip("OR-Tools available")
    w=RotatingSlotWorld();s=w.build_planning_snapshot()
    cat=CandidateCatalog({}, {}, {}, ())
    with pytest.raises(RuntimeError,match="requires OR-Tools"):
        RollingHorizonCPSATScheduler().plan(s,cat,EvaluationContext(s),PlanningBudget())


def test_T16_catalog_budget_truncation_keeps_every_aircraft_and_round_robin():
    w=RotatingSlotWorld();w.seed_hold("a","outer",0,.9);w.seed_hold("b","outer",18,.9)
    s=w.build_planning_snapshot();b=PlanningBudget(max_attempts=8,max_per_aircraft=4)
    cat=CandidateProvider().get_columns(s,b)
    assert set(cat.columns)=={"a","b"} and all(len(v)>=2 for v in cat.columns.values())
    assert sum(b.attempts.values())<=8 and all(v<=4 for v in b.attempts.values())
    assert b.attempts["a"] and b.attempts["b"] and set(cat.incomplete)=={"a","b"}


def test_T16_real_catalog_counts_and_first_target_cap(outer_catalog):
    _,_,_,_,catalog,budget=outer_catalog
    moving=[c for c in catalog.columns["o"] if c.kind=="MOVE"]
    from collections import Counter
    assert moving and len(moving)<=8
    assert max(Counter(c.first_target for c in moving).values())<=2
    assert all(n<=64 for n in budget.attempts.values()) and sum(budget.attempts.values())<=2048


def test_T25_quantization_changes_only_solver_coefficients():
    c=PredictionColumn("a","MOVE",(),(),(1,-1,2,12.34567,3666.12345))
    before=c.objective
    assert solver_coefficients(c)==(1,-1,2,123,1)
    assert c.objective==before


def test_T26_exclusive_timing_and_callback_checkpoint():
    budget=PlanningBudget()
    with budget.measure("parent"):
        with budget.measure("child"):
            sum(range(10000))
    assert sum(t[0] for t in budget.timings.values())<=budget.elapsed
    w=RotatingSlotWorld();w.seed_hold("i","inner",0,.9)
    callback=NormalSchedulerCallback();callback(w._scheduler_port)
    record=callback.records[-1]
    assert all(k in record["timings"] for k in ("candidate_generation_and_single_vehicle","selected_validation","commit"))
    assert sum(v[0] for v in record["timings"].values())<=record["wall_s"]
    state=json.loads(json.dumps(callback.state_dict()))
    restored=NormalSchedulerCallback();restored.load_state_dict(state)
    assert restored.state_dict()==state


def test_T09_real_multistage_downstream_pad_changes_selected_first_step():
    pytest.importorskip("ortools")
    w=RotatingSlotWorld();w.seed_hold("o","outer",0,.9)
    s=w.build_planning_snapshot();ctx=EvaluationContext(s);provider=CandidateProvider()
    def chain(target,pad):
        first=transfer("o","outer",0,target)
        second=transfer("o","middle",target,target,first.end_time,1)
        last=generate_final("o","inner",target,pad,second.end_time,source_version=2).plan
        assert last is not None
        return provider._column(s,s.aircraft["o"],(first,second,last),"MOVE",ctx)
    a,b=chain(0,0),chain(1,4)
    # Both first transfers have identical duration; downstream flight determines
    # their different full arrival costs. A real pad reservation then removes A.
    assert a.objective<b.objective
    keep=provider._column(s,s.aircraft["o"],(),"KEEP",ctx)
    catalog=CandidateCatalog({"o":(keep,a,b)},{"o":(2,0.,0.,"o")},{},())
    r=RollingHorizonCPSATScheduler().plan(s,catalog,ctx,PlanningBudget())
    assert r.certified and r.selected[0].first_target==a.first_target
    w.seed_service("busy",a.plans[-1].pad_id,remaining=a.plans[-1].end_time-5.)
    s2=w.build_planning_snapshot();ctx2=EvaluationContext(s2)
    a2=provider._column(s2,s2.aircraft["o"],a.plans,"MOVE",ctx2)
    b2=provider._column(s2,s2.aircraft["o"],b.plans,"MOVE",ctx2)
    keep2=provider._column(s2,s2.aircraft["o"],(),"KEEP",ctx2)
    assert not ctx2.check_single(a2).accepted and ctx2.check_single(b2).accepted
    catalog=replace(catalog,columns={"o":(keep2,a2,b2)})
    r2=RollingHorizonCPSATScheduler().plan(s2,catalog,ctx2,PlanningBudget())
    assert r2.certified and r2.selected[0].first_target==b.first_target
    assert len(r2.selected[0].plans)==3


@pytest.mark.parametrize(("node_soc","prediction"),[(.8,"UNKNOWN"),(.35,"ELIGIBLE")])
def test_predicted_inner_alert_allows_only_immediate_normal_exception(node_soc,prediction):
    w=RotatingSlotWorld();w.seed_hold("m","middle",0,.9)
    a=w.aircraft["m"]
    w.aircraft["m"]=replace(a,emergency=replace(a.emergency,reference_missing=True))
    w.seed_service("busy",0,remaining=200.)
    first=transfer("m","middle",0,0)
    q=integrate_energy(first,first.start_time,first.end_time)
    w.aircraft["m"]=replace(w.aircraft["m"],soc=node_soc+q.soc_used)
    s=w.build_planning_snapshot();provider=CandidateProvider();ctx=EvaluationContext(s)
    node=PlanningNode("inner",0,first.end_time,node_soc,q.peak_seconds,1)
    columns=[item for item in provider._paths(s,s.aircraft["m"],node,PlanningBudget(),ctx,(first,)) if item[0]=="COLUMN"]
    complete=[item for item in columns if item[1][-1].pad_id is not None]
    assert complete
    assert all(item[1][-1].start_time==node.time and item[2]==prediction+"_IMMEDIATE_NORMAL_EXCEPTION" for item in complete)
    immediate=[p for p in provider._successors(s,s.aircraft["m"],node,PlanningBudget(),immediate_only=True) if p is not None]
    assert immediate and all(p.start_time==node.time for p in immediate)
    general=[p for p in provider._successors(s,s.aircraft["m"],node,PlanningBudget()) if p is not None]
    assert any(p.start_time>node.time for p in general)  # excluded delayed choices really exist


def test_predicted_inner_unknown_keeps_partial_when_immediate_unavailable(monkeypatch):
    import terminal_marl.core.rotating_slots.stage2_planning as planning
    w=RotatingSlotWorld();w.seed_hold("m","middle",0,.9)
    a=w.aircraft["m"];w.aircraft["m"]=replace(a,emergency=replace(a.emergency,reference_missing=True))
    s=w.build_planning_snapshot();p=transfer("m","middle",0,0)
    node=PlanningNode("inner",0,p.end_time,.8,10.,1)
    # This controlled iterator input offers delayed starts only. The real
    # generator must not turn any of them into the immediate-inner exception.
    monkeypatch.setattr(planning,"OFFSETS",(5.,10.))
    out=[x for x in CandidateProvider()._paths(s,s.aircraft["m"],node,PlanningBudget(),EvaluationContext(s),(p,)) if x[0]=="COLUMN"]
    assert out==[("COLUMN",(p,),"UNKNOWN")]


def test_T15_real_solver_unknown_retains_verified_plan(monkeypatch):
    cp_model=pytest.importorskip("ortools.sat.python.cp_model")
    real_solver=cp_model.CpSolver
    class TinyBudgetSolver(real_solver):
        def Solve(self,model,*args,**kwargs):
            self.parameters.max_time_in_seconds=1.e-9
            return super().Solve(model,*args,**kwargs)
    monkeypatch.setattr(cp_model,"CpSolver",TinyBudgetSolver)
    w=RotatingSlotWorld();w.seed_hold("i","inner",0,.9)
    s=w.build_planning_snapshot();ctx=EvaluationContext(s);provider=CandidateProvider()
    catalog=provider.get_columns(s,PlanningBudget(),ctx)
    r=RollingHorizonCPSATScheduler().plan(s,catalog,ctx,PlanningBudget())
    assert r.certified and r.status=="LEX_LIMITED"
    assert r.solver_records and r.solver_records[0]["status"]=="UNKNOWN"
    assert not w.failed and w.metrics().get("energy_failure_count",0)==0
