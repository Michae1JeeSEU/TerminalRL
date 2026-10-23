"""Stage3 ordering, factual records and unchanged authoritative transactions."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import json
import math
import subprocess
import sys

import pytest

from terminal_marl.core.rotating_slots import stage2_planning as sp
from terminal_marl.core.rotating_slots.stage3_learning import AdvisorRuntime, RankResult, scale_candidate_budget
from terminal_marl.core.rotating_slots.stage3_learning.adapters import stable_id
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld


class CheapEncoder:
    def encode(self, snapshot, descriptors, observed_search_state=None):
        return [dict(descriptor_id=d.descriptor_id, attempts=observed_search_state["attempts"]) for d in descriptors]


class ReverseAdvisor:
    def rank(self, features, budget):
        ids=tuple(f["descriptor_id"] for f in features)
        return RankResult(ids[::-1], {"model_id":"test"}, {i:float(n) for n,i in enumerate(ids)})


def setup(mode="GUIDED_ORDER", advisor=None, **kwargs):
    world=RotatingSlotWorld();world.configure_practical_runtime(True)
    world.seed_hold("i","inner",0,.9)
    snapshot=world.build_planning_snapshot();budget=sp.PlanningBudget()
    runtime=AdvisorRuntime(mode,advisor=advisor,feature_encoder=CheapEncoder(),**kwargs)
    runtime.begin_snapshot(snapshot,budget,["i"])
    a=snapshot.aircraft["i"];node=sp._node(snapshot,a)
    descriptors=[runtime.descriptor(snapshot,a,node,t,0.,35.,"progress") for t in range(3)]
    return world,snapshot,budget,runtime,a,node,descriptors


def test_off_import_does_not_load_torch():
    command="import sys; from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider; assert 'torch' not in sys.modules; assert CandidateProvider().advisor_runtime is None"
    result=subprocess.run([sys.executable,"-c",command],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_descriptor_is_immutable_and_schema_fields_are_portable():
    *_,descriptors=setup()
    descriptor=descriptors[0]
    with pytest.raises(FrozenInstanceError): descriptor.target_id=99
    assert descriptor.as_dict()["record_type"]=="WorkDescriptor"
    assert json.loads(json.dumps(descriptor.as_dict()))["target_kind"]=="normal_pad"


def test_reduced_budget_changes_only_three_fields_and_never_resets_clock():
    budget=sp.PlanningBudget.for_profile(sp.QUALITY_BUDGET_PROFILE)
    before=budget.settings();clock=budget.wall_start;attempts=budget.attempts
    scale_candidate_budget(budget,.5)
    for key,value in before.items():
        expected=value*.5 if key=="candidate_seconds" else max(1,math.floor(value*.5)) if key in ("max_per_aircraft","max_attempts") else value
        assert budget.settings()[key]==expected
    assert budget.wall_start==clock and budget.attempts is attempts
    scale_candidate_budget(budget,.5)
    assert budget.candidate_seconds==before["candidate_seconds"]*.5
    with pytest.raises(ValueError): scale_candidate_budget(budget,1.)


def test_real_parent_requires_generated_plan_and_computed_source_state():
    world=RotatingSlotWorld();world.configure_practical_runtime(True);world.seed_hold("o","outer",0,.9)
    snapshot=world.build_planning_snapshot();a=snapshot.aircraft["o"];root=sp._node(snapshot,a)
    runtime=AdvisorRuntime("RECORD",feature_encoder=CheapEncoder());budget=sp.PlanningBudget()
    runtime.begin_snapshot(snapshot,budget,[a.id])
    plan=generate_transfer(a.id,"outer",0,1,0.,config=snapshot.config,durations=(40.,)).plan
    assert plan is not None
    q=sp._move_energy(root,plan,snapshot.config)
    child=sp.PlanningNode(plan.target_ring,plan.target_slot,plan.end_time,root.soc-q[1],root.peak+q[2],
                          root.version+1,root.error_j+q[3],root.peak_error_s+q[4])
    with pytest.raises(ValueError,match="not been materialized"):
        runtime.descriptor(snapshot,a,child,0,child.time,40.,"alternative",(plan,))
    d=runtime.descriptor(snapshot,a,root,1,0.,40.,"alternative")
    token=runtime.record_before(snapshot,d,budget,{})
    runtime.record_after(d,token,True,plan,budget,{})
    with pytest.raises(ValueError,match="not been computed"):
        runtime.descriptor(snapshot,a,child,0,child.time,40.,"alternative",(plan,))
    runtime.record_energy(root,plan,q,False)
    descendant=runtime.descriptor(snapshot,a,child,0,child.time,40.,"alternative",(plan,))
    parent=runtime.parents[descendant.parent_prefix_id]
    assert parent["soc"]==child.soc and parent["materialized"] and parent["depth"]==1
    with pytest.raises(ValueError,match="not been computed"):
        runtime.descriptor(snapshot,a,replace(child,soc=1.),0,child.time,40.,"alternative",(plan,))


@pytest.mark.parametrize("failure",["NAN","ID","ERROR","ENTITY_LIMIT"])
def test_runtime_fallback_uses_remaining_shared_budget(failure):
    class BadAdvisor:
        def rank(self, features,budget):
            ids=tuple(f["descriptor_id"] for f in features)
            if failure=="ERROR": raise RuntimeError("intentional")
            if failure=="ENTITY_LIMIT":
                error=ValueError("entity limit");error.reason="ENTITY_LIMIT";raise error
            return RankResult(("invented",) if failure=="ID" else ids,{}, {ids[0]:math.nan})
    _,snapshot,budget,runtime,_,_,ds=setup(advisor=BadAdvisor())
    runtime._initial.clear();clock=budget.wall_start;attempts=budget.attempts
    assert runtime.order(snapshot,ds,budget)==ds
    row=next(row for row in runtime.traces if row["record_type"]=="Fallback")
    assert row["reason"] in ("NONFINITE_SCORE","ILLEGAL_DESCRIPTOR_ID","INFERENCE_ERROR","ENTITY_LIMIT")
    assert budget.wall_start==clock and budget.attempts is attempts


def test_three_learned_one_heuristic_and_deterministic_finish(monkeypatch):
    _,snapshot,budget,runtime,_,_,ds=setup(advisor=ReverseAdvisor())
    runtime._initial.clear()
    for index in range(4):
        ranked=runtime.order(snapshot,ds,budget)
        assert ranked[0]==(ds[0] if index==3 else ds[-1])
        token=runtime.record_before(snapshot,ranked[0],budget,{})
        budget.attempts[ranked[0].aircraft_id]+=1
        runtime.record_after(ranked[0],token,True,None,budget,{})
    assert runtime.coverage["learned_selected"]==3
    monkeypatch.setattr(sp.PlanningBudget,"elapsed",property(lambda self:.85*self.candidate_seconds))
    assert runtime.order(snapshot,ds,budget)==ds
    assert runtime.traces[-1]["policy"]=="deterministic_finish"


def test_generate_none_and_nonfinite_energy_are_unknown_not_certified_invalid():
    _,snapshot,budget,runtime,_,node,ds=setup("RECORD")
    token=runtime.record_before(snapshot,ds[0],budget,{})
    runtime.record_after(ds[0],token,True,None,budget,{})
    row=runtime.traces[-1]
    assert row["outcome"]=="UNKNOWN" and not row["censored"]
    plan=SimpleNamespace(trajectory_hash="generated",target_ring=None,pad_id=0)
    runtime._plans["generated"]=[ds[0].descriptor_id]
    runtime.record_energy(node,plan,(math.inf,)*5,False)
    assert runtime.traces[-1]["outcome"]=="UNKNOWN"
    assert not runtime.traces[-1]["validity_label_mask"]


def test_prequery_features_are_refreshed_and_events_do_not_alias_mutable_state():
    _,snapshot,budget,runtime,_,_,ds=setup(advisor=ReverseAdvisor())
    runtime._initial.clear();runtime.order(snapshot,ds,budget)
    budget.attempts["i"]=5
    runtime.record_before(snapshot,ds[0],budget,{})
    row=runtime.traces[-1]
    assert row["features"]["attempts"]==5
    runtime.parents["later"]={"materialized":True}
    assert "later" not in row["search_state"]["parents"]


def test_off_record_same_order_and_keep_wait_under_fixed_attempt_budget(monkeypatch):
    monkeypatch.setattr(sp.PlanningBudget,"elapsed",property(lambda self:0.))
    calls=[]
    monkeypatch.setattr(sp,"_generate",lambda snapshot,aid,node,target,start,duration:(calls.append((aid,target,start,duration)) or None))
    monkeypatch.setattr(sp.ReferenceProvider,"screen",lambda self,snapshot,node,start,budget:
                        {(target,duration):None for target in range(6) for duration in range(20,301,5)})
    monkeypatch.setattr(sp.CandidateProvider,"_column",lambda self,s,a,plans,kind,context,*args,**kwargs:
                        sp.PredictionColumn(a.id,kind,tuple(plans),(),(0,0,0,0,0)))
    world=RotatingSlotWorld();world.seed_hold("i","inner",0,.9);snapshot=world.build_planning_snapshot()
    before=snapshot
    results=[]
    for mode in ("OFF","RECORD"):
        calls.clear();runtime=AdvisorRuntime(mode,feature_encoder=CheapEncoder())
        budget=sp.PlanningBudget(max_attempts=8,max_per_aircraft=8)
        catalog=sp.CandidateProvider(advisor_runtime=runtime).get_columns(snapshot,budget)
        results.append((list(calls),dict(catalog.attempts),[c.kind for c in catalog.columns["i"]]))
    assert results[0]==results[1]
    assert results[1][2]==["KEEP","WAIT"]
    assert world.build_planning_snapshot()==before


def test_guided_initial_pass_counts_queries_after_screen_for_all_rings(monkeypatch):
    monkeypatch.setattr(sp.PlanningBudget,"elapsed",property(lambda self:0.))
    monkeypatch.setattr(sp.CandidateProvider,"_column",lambda self,s,a,plans,kind,context,*args,**kwargs:
                        sp.PredictionColumn(a.id,kind,tuple(plans),(),(0,0,0,0,0)))
    world=RotatingSlotWorld()
    for aid,ring,slot in (("o1","outer",0),("o2","outer",4),("m","middle",7),("i","inner",15)):
        world.seed_hold(aid,ring,slot,.9)
    snapshot=world.build_planning_snapshot();seen=[]
    runtime=AdvisorRuntime("GUIDED_LIMITED",advisor=ReverseAdvisor(),feature_encoder=CheapEncoder())
    def progress(self,s,a,node,b,context):
        yield "SCREEN",None,None
        d=runtime.descriptor(s,a,node,0,node.time,40.,"progress")
        token=runtime.record_before(s,d,b,{})
        seen.append(a.id);b.attempts[a.id]+=1
        runtime.record_after(d,token,True,None,b,{})
        yield "ATTEMPT",None,None
    monkeypatch.setattr(sp.CandidateProvider,"_progress_steps",progress)
    monkeypatch.setattr(sp.ReferenceProvider,"build_steps",lambda *args,**kwargs:iter(()))
    monkeypatch.setattr(sp.ReferenceProvider,"get",lambda *args,**kwargs:sp.NormalReference("UNKNOWN"))
    monkeypatch.setattr(sp.CandidateProvider,"_paths",lambda *args,**kwargs:iter(()))
    provider=sp.CandidateProvider(advisor_runtime=runtime)
    # Avoid entering the mocked empty reference generator after initial work.
    budget=sp.PlanningBudget(max_attempts=4,max_per_aircraft=1)
    provider.get_columns(snapshot,budget)
    assert seen==["o1","m","i","o2"]
    coverage=next(r for r in reversed(runtime.traces) if r["record_type"]=="Coverage")
    assert coverage["unserved_aircraft_count"]==0


def test_high_advisor_score_cannot_authorize_conflicting_commit():
    world=RotatingSlotWorld();world.configure_practical_runtime(True);world.seed_hold("i","inner",0,.9)
    plan=generate_final("i","inner",0,0,0.,config=world.config).plan
    assert plan is not None
    world.seed_service("busy",0,remaining=plan.end_time+50.)
    snapshot=world.build_planning_snapshot();context=sp.EvaluationContext(snapshot)
    runtime=AdvisorRuntime("GUIDED_ORDER",advisor=ReverseAdvisor(),feature_encoder=CheapEncoder())
    provider=sp.CandidateProvider(advisor_runtime=runtime)
    column=provider._column(snapshot,snapshot.aircraft["i"],(plan,),"MOVE",context)
    assert column is not None
    checked=context.check_commit_prefix((column,))
    assert not checked.accepted and checked.reason=="RESOURCE_CONFLICT"
    assert not world.active


def test_optin_callback_checkpoint_requires_identical_mode_scale_binding():
    callback=sp.NormalSchedulerCallback(budget_scale=.5,advisor_runtime=AdvisorRuntime("RECORD",fingerprint={"model_id":"NO_MODEL"}))
    state=callback.state_dict()
    assert state["stage3"]["budget_scale"]==.5
    with pytest.raises(ValueError,match="Stage3 callback binding mismatch"):
        sp.NormalSchedulerCallback().load_state_dict(state)
    callback.load_state_dict(state)
    assert "stage3" not in sp.NormalSchedulerCallback().state_dict()


def test_offline_probe_is_explicit_and_preserves_ready_pool():
    def probe(descriptors,selected,index):
        return descriptors[-1],dict(probe=True,selection_probability=1./len(descriptors),opportunity_index=index)
    _,snapshot,budget,runtime,_,_,ds=setup("SHADOW_OFFLINE",offline_probe=probe)
    ranked=runtime.order(snapshot,ds,budget)
    assert ranked[0]==ds[-1] and set(ranked)==set(ds)
    assert runtime.traces[-1]["record_type"]=="OfflineProbe"
    with pytest.raises(ValueError,match="requires SHADOW_OFFLINE"):
        AdvisorRuntime("GUIDED_ORDER",offline_probe=probe)


def test_trace_snapshot_time_follows_actual_snapshot_without_rewriting_prior_rows():
    unbound=AdvisorRuntime("RECORD",feature_encoder=CheapEncoder())
    unbound.emit(dict(record_type="BeforeSnapshot"))
    assert "as_of" not in unbound.traces[-1] and "snapshot_id" not in unbound.traces[-1]

    _,snapshot,budget,runtime,a,node,descriptors=setup("RECORD")
    token=runtime.record_before(snapshot,descriptors[0],budget,{})
    runtime.record_after(descriptors[0],token,True,None,budget,{})
    prior=list(runtime.traces)
    snapshot0_id=runtime.snapshot_id
    assert all(row["as_of"]==0. and row["snapshot_id"]==snapshot0_id for row in prior)
    assert prior[0]["record_type"]=="SnapshotStart" and prior[0]["time"]==0.

    later=replace(snapshot,time=5.,version=snapshot.version+"-time5")
    runtime.begin_snapshot(later,budget,[a.id])
    later_node=sp._node(later,a)
    descriptor=runtime.descriptor(later,a,later_node,0,5.,35.,"progress")
    runtime.order(later,[descriptor],budget)
    token=runtime.record_before(later,descriptor,budget,{})
    runtime.record_after(descriptor,token,True,None,budget,{})
    later_rows=runtime.traces[len(prior):]
    assert all(row["as_of"]==5. and row["snapshot_id"]==runtime.snapshot_id for row in later_rows)
    assert later_rows[0]["record_type"]=="SnapshotStart" and later_rows[0]["time"]==5.
    assert runtime.snapshot_id!=snapshot0_id
    assert all(row["as_of"]==0. and row["snapshot_id"]==snapshot0_id for row in prior)


def test_reduced_heuristic_control_matches_neutral_model_order_without_rank_cost():
    class NeutralAdvisor:
        def rank(self,features,budget):
            return RankResult(tuple(row["descriptor_id"] for row in features))
    class ForbiddenAdvisor:
        def rank(self,*args,**kwargs):
            pytest.fail("heuristic control called an advisor")
    class CountingEncoder(CheapEncoder):
        def __init__(self): self.calls=0
        def encode(self,*args,**kwargs):
            self.calls+=1
            return super().encode(*args,**kwargs)

    cases=[]
    for deterministic,advisor in ((True,ForbiddenAdvisor()),(False,NeutralAdvisor())):
        _,snapshot,budget,runtime,_,_,descriptors=setup("GUIDED_LIMITED",advisor=advisor,deterministic_only=deterministic)
        scale_candidate_budget(budget,.5)
        encoder=CountingEncoder();runtime.feature_encoder=encoder
        selected=[]
        for index in range(7):
            calls_before=encoder.calls
            chosen=runtime.order(snapshot,descriptors,budget)[0]
            if deterministic: assert encoder.calls==calls_before
            selected.append(chosen.target_id)
            token=runtime.record_before(snapshot,chosen,budget,{})
            budget.attempts[chosen.aircraft_id]+=1
            runtime.record_after(chosen,token,True,None,budget,{})
            if index==0: runtime.complete_initial_coverage()
        cases.append((selected,runtime,budget))
    control,neutral=cases
    assert control[0]==neutral[0]
    assert control[1].coverage["attempted"]==neutral[1].coverage["attempted"]==7
    assert control[1].coverage["heuristic_control_selected"]==7
    assert control[1]._initial==neutral[1]._initial==set()
    assert control[2].settings()==neutral[2].settings()
    assert "stage3_inference" not in control[2].timings
    assert not any(row["record_type"]=="Fallback" for row in control[1].traces)
    assert all(row["policy"]=="heuristic_control" for row in control[1].traces if row["record_type"]=="Rank")


def test_callback_checkpoint_rejects_crossarm_deterministic_control_resume():
    control=sp.NormalSchedulerCallback(budget_scale=.5,advisor_runtime=AdvisorRuntime("GUIDED_LIMITED",deterministic_only=True))
    state=control.state_dict()
    assert state["stage3"]["deterministic_only"]
    control.load_state_dict(state)
    learned=sp.NormalSchedulerCallback(budget_scale=.5,advisor_runtime=AdvisorRuntime("GUIDED_LIMITED"))
    with pytest.raises(ValueError,match="Stage3 callback binding mismatch"):
        learned.load_state_dict(state)
