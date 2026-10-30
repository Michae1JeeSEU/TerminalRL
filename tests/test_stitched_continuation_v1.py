"""Stitched scoring timelines and bounded completion, with no episode runs."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace as NS

import pytest

from terminal_marl.core.rotating_slots.stage2_downstream import (
    LEGACY_EVALUATION_PROFILE as LEGACY, STITCHED_EVALUATION_PROFILE as NEW,
    Demand, JointDownstream, describe_column, joules, ticks)
from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider, CandidateCatalog, EvaluationContext, NormalReference, PlanningBudget, PlanningNode,
    PredictionColumn, NormalSchedulerCallback, _node)
from terminal_marl.core.rotating_slots.selective_replanning import PlanningScope, build_global_frame
from terminal_marl.core.rotating_slots.selective_replanning.frame import frame_policy
from terminal_marl.core.rotating_slots.planning_latency import PlanningTiming, project_visible_snapshot
from terminal_marl.core.rotating_slots.trajectory import generate_transfer, generate_final, make_stable_plan
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.resources import Reservation


def provider(profile=NEW):
    return CandidateProvider(planning_profile="downstream_service_v8", evaluation_profile=profile)


def world_snapshot(ring="middle", n=1, timing=True):
    world=RotatingSlotWorld()
    for i in range(n): world.seed_hold(chr(ord("a")+i),ring,i,.9)
    source=world.build_planning_snapshot()
    snapshot=project_visible_snapshot(source,20.,timing=PlanningTiming.from_lead(0.,20.,1.)) if timing else source
    return world,snapshot


def test_legacy_default_path_unchanged_and_profile_state_binding():
    world,s=world_snapshot()
    implicit=CandidateProvider(planning_profile="downstream_service_v8")
    explicit=provider(LEGACY)
    left=implicit._column(s,s.aircraft["a"],(),"WAIT",EvaluationContext(s))
    right=explicit._column(s,s.aircraft["a"],(),"WAIT",EvaluationContext(s))
    assert left==right and implicit.evaluation_profile==LEGACY
    make=lambda profile:NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",
        selective_method="RULE_LOCAL_K8",evaluation_profile=profile)
    old,new=make(LEGACY),make(NEW)
    old_state=old.state_dict(); new_state=new.state_dict()
    assert "evaluation_profile" not in old_state and new_state["evaluation_profile"]==NEW
    make(LEGACY).load_state_dict(old_state); make(NEW).load_state_dict(new_state)
    with pytest.raises(ValueError,match="evaluation profile mismatch"): new.load_state_dict(old_state)
    with pytest.raises(ValueError,match="evaluation profile mismatch"): old.load_state_dict(new_state)
    with pytest.raises(ValueError,match="candidate provider evaluation"):
        NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",
            candidate_provider=provider(LEGACY),evaluation_profile=NEW)


def test_partial_timeline_has_no_horizon_hold_and_no_fictional_landing():
    world,s=world_snapshot()
    p=provider(); context=EvaluationContext(s)
    plan=generate_transfer("a","middle",0,0,s.time,durations=(40,),config=s.config).plan
    c=p._column(s,s.aircraft["a"],(plan,),"MOVE",context)
    before=world.build_planning_snapshot()
    demand=p._describe_column(s,c,context)
    explicit=context.energy_portions(((plan,s.time,plan.end_time),))[0]
    assert demand.diagnostics["explicit_prefix_j"]==pytest.approx(explicit)
    assert demand.diagnostics["timeline_explicit_end"]==plan.end_time
    assert demand.diagnostics["estimated_adjustment_wait_s"]==0.
    assert demand.diagnostics["timeline_anchor"]==plan.end_time
    assert demand.landed==0 and demand.unknown==1
    assert demand.diagnostics["reference_status"]=="UNKNOWN"
    assert demand.diagnostics["suffix_peak_status"]=="UNKNOWN_NOT_A_SAFETY_CERTIFICATE"
    assert not demand.diagnostics["executed_reservations_modified"]
    joint=JointDownstream({c.signature:demand},(),("a",),now=ticks(s.time),pad_count=1,evaluation_profile=NEW)
    _,rows=joint.evaluate((c,)); diag=rows["a"]
    segments=diag["timeline_segments"]
    assert all(a["end"]==pytest.approx(b["start"],abs=.1) for a,b in zip(segments,segments[1:]))
    assert sum(segment["energy_j"] for segment in segments)==pytest.approx(diag["component_sum_j"])
    assert diag["component_sum_j"]+diag["base_energy_quantization_j"]==pytest.approx(diag["total_j"])
    assert world.build_planning_snapshot()==before and not world.active


def test_wait_explicit_length_zero_and_nominal_lead_is_labelled():
    _,s=world_snapshot(timing=False)
    p=provider(); context=EvaluationContext(s)
    c=p._column(s,s.aircraft["a"],(),"WAIT",context)
    d=p._describe_column(s,c,context)
    assert d.diagnostics["explicit_prefix_j"]==0.
    assert d.diagnostics["timeline_explicit_end"]==s.time
    assert d.diagnostics["estimated_adjustment_wait_s"]==20.
    assert d.diagnostics["lead_source"]=="nominal20_estimate"
    assert d.diagnostics["window_j"]==0.


def test_stitched_energy_and_failure_never_use_legacy_window_or_horizon_failure():
    _,s=world_snapshot(); p=provider(); context=EvaluationContext(s)
    c=p._column(s,s.aircraft["a"],(),"WAIT",context)
    changed=replace(c,objective=(1,0,999,999,1.e30),downstream=dict(window_j=1.e30))
    before=p._describe_column(s,c,context); after=p._describe_column(s,changed,context)
    assert before.base_energy==after.base_energy and before.failure==after.failure==0
    joint=JointDownstream({changed.signature:after},(),("a",),now=ticks(s.time),pad_count=1,evaluation_profile=NEW)
    assert joint.evaluate((changed,))[0][0]==0


def test_active_soft_and_real_gap_are_charged_once_to_last_committed_endpoint():
    world=RotatingSlotWorld(); world.seed_hold("a","outer",0,.9)
    first=generate_transfer("a","outer",0,0,0.,durations=(40,)).plan
    assert world.submit((first,)).accepted
    second=generate_transfer("a","middle",0,0,45.,source_version=1,durations=(40,)).plan
    assert world.submit((second,)).accepted
    s=world.build_planning_snapshot(); p=provider(); context=EvaluationContext(s)
    c=p._column(s,s.aircraft["a"],(),"WAIT",context)
    gap=make_stable_plan("a","middle",0,40.,45.,config=s.config)
    expected=context.energy_portions(((first,0.,40.),(gap,40.,45.),(second,45.,85.)))[0]
    diag=p._describe_column(s,c,context).diagnostics
    assert diag["explicit_prefix_j"]==pytest.approx(expected)
    assert diag["timeline_explicit_end"]==85. and diag["estimated_adjustment_wait_s"]==0.
    assert diag["window_wait_seconds_by_ring"]["middle"]==5.
    assert world.active["a"]==first and world.soft["a"]==second


def test_cached_suffix_requires_source_slot_version_and_never_generates(monkeypatch):
    _,s=world_snapshot(); p=provider(); context=EvaluationContext(s)
    first=generate_transfer("a","middle",0,0,s.time,durations=(40,)).plan
    c=p._column(s,s.aircraft["a"],(first,),"MOVE",context)
    terminal=c.node
    final=generate_final("a","inner",0,0,terminal.time,source_version=terminal.version,config=s.config).plan
    assert final is not None
    q=p.references.move_energy(terminal,final,s.config)
    key=(s.config.config_hash,terminal.ring,terminal.slot,terminal.time,terminal.version,"a")
    p.references.cache[key]=NormalReference("KNOWN",(final,),duration=final.end_time-final.start_time,energy_j=q[0])
    monkeypatch.setattr(p.references,"generate",lambda *a,**k:pytest.fail("scoring generated a trajectory"))
    d=p._describe_column(s,c,context)
    assert d.diagnostics["reference_status"]=="KNOWN"
    assert d.diagnostics["estimated_suffix_motion_j"]==pytest.approx(q[0])
    assert d.landed==0
    changed=replace(c,node=replace(terminal,version=terminal.version+1))
    assert p._describe_column(s,changed,context).diagnostics["reference_status"]=="UNKNOWN"
    occupied=replace(s,pads=(Reservation(final.pad_id,"unknown_owner",0.,float("inf")),))
    d=p._describe_column(occupied,c,EvaluationContext(occupied))
    assert d.unknown and d.diagnostics["resource_occupancy_unknown"]


def test_explicit_full_final_has_no_virtual_continuation():
    _,s=world_snapshot("inner"); p=provider(); context=EvaluationContext(s)
    plan=generate_final("a","inner",0,0,s.time,config=s.config).plan
    c=p._column(s,s.aircraft["a"],(plan,),"MOVE",context)
    d=p._describe_column(s,c,context)
    assert d.full and d.landed==-int(plan.end_time<=s.time+240.)
    assert d.diagnostics["explicit_prefix_j"]==pytest.approx(context.energy_portions(((plan,s.time,plan.end_time),))[0])
    assert d.diagnostics["estimated_suffix_motion_j"]==d.diagnostics["estimated_adjustment_wait_j"]==0.
    assert d.diagnostics["timeline_end"]==pytest.approx(plan.end_time,abs=.05)


def scoring_demand(aid, *, base=10, rate=1, available=1000, peak=0, unknown=0, ready=0):
    diagnostics=dict(evaluation_profile=NEW,explicit_prefix_peak_failure=peak,available_energy_threshold=available,
        explicit_prefix_j=base*.1,estimated_adjustment_wait_j=0.,estimated_adjustment_wait_s=0.,
        estimated_suffix_motion_j=0.,estimated_suffix_motion_s=13.5,timeline_start=0.,timeline_explicit_end=0.,
        timeline_anchor=0.,base_energy_quantization_j=0.)
    return Demand(aid,False,ready,20,base,rate,available//2,unknown,peak,0,ready+135,"inner",diagnostics)


def test_busy_calendar_can_favor_wait_and_failures_include_queue_energy():
    wait=PredictionColumn("a","WAIT",(),(),(0,)*5)
    move=PredictionColumn("a","MOVE",(NS(trajectory_hash="explicit"),),(),(0,)*5)
    demands={wait.signature:scoring_demand("a",base=10,available=35),move.signature:scoring_demand("a",base=20,available=35)}
    joint=JointDownstream(demands,((0,0,20,"ground"),),("a",),pad_count=1,evaluation_profile=NEW)
    assert joint.key((wait,)) < joint.key((move,))
    w,wr=joint.evaluate((wait,)); m,mr=joint.evaluate((move,))
    assert w[0]==0 and m[0]==1
    assert wr["a"]["estimated_queue_wait_s"]==2.
    assert mr["a"]["estimated_failure"]==1 and mr["a"]["total_j"]==4.


def test_same_profile_background_and_selected_aircraft_share_one_queue():
    _,s=world_snapshot(n=2); p=provider(); context=EvaluationContext(s)
    frame=build_global_frame(s,p,context)
    assert frame.policy==frame_policy(NEW)
    assert all(d.diagnostics["evaluation_profile"]==NEW for d in frame.baselines)
    frame=replace(frame,pad_count=1,assignment=(("a",0),("b",0)))
    c=p._column(s,s.aircraft["a"],(),"WAIT",context)
    d=frame.baseline_by_id["a"]
    joint=frame.downstream({c.signature:d},{"a"})
    _,rows=joint.evaluate((c,))
    assert "b" in joint.fixed_demands and len(rows)==2
    assert rows["b"]["timeline_queue_end"] >= rows["a"]["timeline_queue_end"]+d.duration/10.
    assert rows["b"]["estimated_queue_wait_j"] > 0.
    legacy=replace(frame,policy=frame_policy(LEGACY))
    with pytest.raises(ValueError,match="frame profile mismatch"):
        p._check_frame_profile(NS(global_frame=legacy))


def test_committed_final_and_ground_background_use_new_profile_without_continuation():
    world=RotatingSlotWorld(); world.seed_hold("a","inner",0,.9)
    final=generate_final("a","inner",0,0,0.,config=world.config).plan
    assert world.submit((final,)).accepted
    s=world.build_planning_snapshot(); p=provider()
    frame=build_global_frame(s,p,EvaluationContext(s))
    ground=RotatingSlotWorld(); ground.seed_service("g",1,100.)
    gs=ground.build_planning_snapshot(); gf=build_global_frame(gs,p,EvaluationContext(gs))
    demands=(*frame.baselines,*gf.baselines)
    assert {d.aircraft_id for d in demands}=={"a","g"}
    assert all(d.full and d.diagnostics["evaluation_profile"]==NEW for d in demands)
    assert all(d.diagnostics["estimated_suffix_motion_j"]==0. for d in demands)
    assert gf.baseline_by_id["g"].diagnostics["explicit_prefix_j"]==0.


@pytest.mark.parametrize("peak,unknown",((0,0),(1,0),(0,1)))
def test_python_and_real_cp_objective_use_identical_failure_thresholds(peak,unknown):
    cp=pytest.importorskip("ortools.sat.python.cp_model",reason="real CP backend required")
    c=PredictionColumn("a","WAIT",(),(),(0,)*5)
    d=scoring_demand("a",base=10,available=30,peak=peak,unknown=unknown)
    joint=JointDownstream({c.signature:d},((0,0,20,"ground"),),("a",),pad_count=1,evaluation_profile=NEW)
    model=cp.CpModel(); v=model.NewBoolVar("selected"); model.Add(v==1)
    terms=joint.cp_expressions(model,{c.signature:v},{c.signature:c})
    solver=cp.CpSolver(); solver.parameters.num_search_workers=1
    for term in terms:
        model.Minimize(term); assert solver.Solve(model) in (cp.OPTIMAL,cp.FEASIBLE)
        model.Add(term==solver.Value(term))
    assert tuple(solver.Value(term) for term in terms)==joint.evaluate((c,))[0]


def test_profile_archive_roundtrip_and_common_callback_share_interface(tmp_path):
    from scripts.stage3 import run_selective_common_initial_a4 as common
    from scripts.stage3 import run_selective_quality_gap_a4 as a4
    _,s=world_snapshot(); p=provider(); frame=build_global_frame(s,p,EvaluationContext(s))
    scope=PlanningScope(s.version,s.time,("a",),("a",),(),actual_k=1,objective_frame_id=frame.identity)
    context=EvaluationContext(s,scope=scope,global_frame=frame)
    c=p._column(s,s.aircraft["a"],(),"KEEP",context)
    catalog=p._progressive_catalog(s,PlanningBudget(),context,{"a":[c]},{"a":NormalReference("UNKNOWN")},[s.aircraft["a"]])
    path=tmp_path/"new.json"; a4.write_new(path,a4.pack_problem(dict(snapshot=s,catalog=catalog,global_frame=frame,selected=(c,))))
    restored,_=a4.unpack_problem(path)
    assert restored["catalog"].downstream.evaluation_profile==NEW
    assert restored["global_frame"].policy==frame_policy(NEW)
    for method in ("FULL_H","RULE_LOCAL_K8"):
        cb=common.common_callback(NS(NormalSchedulerCallback=NormalSchedulerCallback),method,evaluation_profile=NEW)
        assert cb.evaluation_profile==cb.provider.evaluation_profile==NEW
    assert common.common_callback(NS(NormalSchedulerCallback=NormalSchedulerCallback),"RULE_LOCAL_K8").evaluation_profile==LEGACY


@dataclass(frozen=True)
class Primitive:
    trajectory_hash: str
    aircraft_id: str
    source_ring: str="outer"
    source_slot: int=0
    source_version: int=0
    start_time: float=0.
    end_time: float=40.
    target_ring: str|None="inner"
    target_slot: int|None=0
    pad_id: int|None=None


@pytest.mark.parametrize("outcome",("KNOWN","UNKNOWN","NOT_ADMISSIBLE"))
def test_completion_keeps_absolute_prefix_and_two_object_cap_after_scope_rebuild(monkeypatch,outcome):
    _,s=world_snapshot("outer",n=3,timing=False); p=provider(); calls=[]
    frame=build_global_frame(s,p,EvaluationContext(s))
    scope1=PlanningScope(s.version,s.time,tuple(s.aircraft),("a","b"),("c",),actual_k=2,objective_frame_id=frame.identity)
    budget=PlanningBudget(max_per_aircraft=32,max_attempts=1024,candidate_seconds=6.,solver_seconds=8.,
        soft_seconds=30.,candidate_active_seconds=0.)
    originals={aid:Primitive("prefix_"+aid,aid,source_slot=a.slot,target_slot=a.slot) for aid,a in s.aircraft.items()}
    def column(snapshot,a,plans,kind,context,ref=None,*args,**kwargs):
        n=(_node(snapshot,a) if not plans else PlanningNode("inner",a.slot,plans[0].end_time,.8,0.,1))
        return PredictionColumn(a.id,kind,tuple(plans),(),(0,)*5,n,NormalReference("UNKNOWN"),downstream={"estimated_touchdown":100.+ord(a.id)})
    monkeypatch.setattr(p,"_column",column)
    monkeypatch.setattr(p,"_previous_chain",lambda *args:None)
    monkeypatch.setattr(p,"_progressive_catalog",lambda snapshot,budget,context,columns,refs,ordered:
        CandidateCatalog({aid:tuple(rows) for aid,rows in columns.items()},{},dict(budget.attempts),(),scope=context.scope))
    def progress(snapshot,a,node,budget,context):
        yield ("COLUMN",(originals[a.id],),"NONE")
    def empty(*args):
        if False: yield None
    def reference(snapshot,a,node,budget):
        if node.ring!="inner": return NormalReference("UNKNOWN")
        calls.append(a.id); budget.attempts[a.id]+=1
        yield None
        final=Primitive("suffix_"+a.id,a.id,"inner",a.slot,1,node.time,node.time+40.,None,None,0)
        return NormalReference("UNKNOWN" if outcome=="UNKNOWN" else "KNOWN",(final,))
    monkeypatch.setattr(p,"_progress_steps",progress); monkeypatch.setattr(p,"_paths",empty)
    monkeypatch.setattr(p,"_delayed_steps",empty); monkeypatch.setattr(p.references,"build_steps",reference)
    monkeypatch.setattr(p.references,"_reference",lambda *args:NormalReference("KNOWN",args[2],energy_j=0.))
    monkeypatch.setattr(p,"_reference_seed",lambda snapshot,a,node,ref,budget:
        None if outcome=="NOT_ADMISSIBLE" else (ref.plans,"NONE"))
    def context(scope):
        result=EvaluationContext(s,budget=budget,scope=scope,global_frame=frame)
        monkeypatch.setattr(result,"check_single",lambda c:ValidationResult(True,"SAFE"))
        return result
    initial_generator=p.iter_progressive_columns(s,budget,context(scope1),scope=scope1,early_publish=True)
    first=next(initial_generator)
    assert not calls and not hasattr(budget,"_suffix_completion_state")
    p.reserve_suffix_completion(s,first,budget)
    state=budget._suffix_completion_state
    assert [row["aircraft_id"] for row in state["records"]]==["a","b"]
    initial_generator.close()
    scope2=replace(scope1,replan_ids=tuple(s.aircraft),fixed_background_ids=(),actual_k=3)
    next_generator=p.iter_progressive_columns(s,budget,context(scope2),scope=scope2,early_publish=True)
    grown=next(next_generator); next_generator.close()
    assert calls==["a","b"] and budget._suffix_completion_state is state
    for aid in ("a","b"):
        assert {c.signature for c in first.columns[aid]} <= {c.signature for c in grown.columns[aid]}
        full=[c for c in grown.columns[aid] if c.plans and c.plans[-1].pad_id is not None]
        if outcome=="KNOWN":
            assert full and full[0].plans[0] == originals[aid]
            assert full[0].plans[1].start_time==originals[aid].end_time
        else:
            assert not full and {c.kind for c in grown.columns[aid]} >= {"KEEP","WAIT","MOVE"}
    p.reserve_suffix_completion(s,grown,budget)
    assert budget._suffix_completion_state is state and len(state["records"])==2
    assert budget.candidate_seconds==6. and budget.solver_seconds==8. and budget.max_per_aircraft==32
    report=p.suffix_completion_report(budget)
    assert all(row["complete_candidate_in_catalog"] == (outcome=="KNOWN") for row in report["attempts"])
    json.dumps(report)


def test_new_profile_full_plan_still_uses_unchanged_final_atomic_authorization(monkeypatch):
    from terminal_marl.core.rotating_slots.stage2_planning import RollingHorizonCPSATScheduler, PlanningResult
    class OneFinal(CandidateProvider):
        def iter_progressive_columns(self,snapshot,budget,context,*,scope=None,early_publish=False):
            a=snapshot.aircraft["a"]
            plan=generate_final("a",a.ring,a.slot,0,snapshot.time,source_version=a.version,config=snapshot.config).plan
            rows={"a":[self._column(snapshot,a,(),kind,context) for kind in ("KEEP","WAIT")]}
            rows["a"].append(self._column(snapshot,a,(plan,),"MOVE",context))
            catalog=self._progressive_catalog(snapshot,budget,context,rows,{"a":rows["a"][0].reference},[a])
            yield catalog
            yield catalog
    def choose_final(self,snapshot,catalog,context,budget,*,incumbent=None):
        selected=(catalog.columns["a"][-1],)
        assert context.check_batch(selected)[0].accepted
        checked=context.check_commit_prefix(selected)
        assert checked.accepted
        return PlanningResult(selected,context.prefix(selected),checked,"FIXTURE_EXPLICIT_FINAL",
            catalog.downstream.key(selected),certified=True)
    monkeypatch.setattr(RollingHorizonCPSATScheduler,"plan",choose_final)
    world=RotatingSlotWorld(); world.seed_hold("a","inner",0,.9)
    cb=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",evaluation_profile=NEW,
        candidate_provider=OneFinal(planning_profile="downstream_service_v8",evaluation_profile=NEW),selective_method="RULE_LOCAL_K8")
    result=cb(world._scheduler_port)
    assert result.certified and result.prefix==(world.active["a"],)
    assert result.evaluation.plans==result.prefix
    assert world.trajectories[result.prefix[0].trajectory_hash]==result.prefix[0].to_dict()
    assert cb.records[-1]["submitted"]==1 and cb.records[-1]["evaluation_profile"]==NEW
    assert cb.records[-1]["objective_id"].startswith(NEW)
    assert world.time==0.


def test_non_progressive_entry_scores_with_new_profile_without_enabling_completion():
    class OneShot(CandidateProvider):
        def get_columns(self,snapshot,budget=None,context=None,*,scope=None):
            a=snapshot.aircraft["a"]
            rows={"a":[self._column(snapshot,a,(),kind,context) for kind in ("KEEP","WAIT")]}
            return self._progressive_catalog(snapshot,budget,context,rows,{"a":NormalReference("UNKNOWN")},[a])
    records={}
    for profile in (LEGACY,NEW):
        world=RotatingSlotWorld(); world.seed_hold("a","inner",0,.9)
        cb=NormalSchedulerCallback("energy_slack",planning_profile="downstream_service_v8",evaluation_profile=profile,
            candidate_provider=OneShot(planning_profile="downstream_service_v8",evaluation_profile=profile))
        result=cb(world._scheduler_port)
        assert cb.online_policy is None and result.certified
        records[profile]=cb.records[-1]
        assert not world.active and world.time==0.
    row=records[NEW]
    assert row["evaluation_profile"]==NEW and row["objective_id"].startswith(NEW)
    assert row["downstream"]["a"]["evaluation_profile"]==NEW
    assert row["suffix_completion"]==dict(policy_id="bounded_suffix_completion_v1",
        status="NOT_ENABLED_NON_PROGRESSIVE",enabled=False,attempts=[])
    assert row["candidate_policy_id"]==records[LEGACY]["candidate_policy_id"]
    assert row["planning_budget"]==records[LEGACY]["planning_budget"]
