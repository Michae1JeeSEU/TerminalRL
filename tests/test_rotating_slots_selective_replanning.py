"""Stage A invariants. IDs identify exercised clauses, not campaign acceptance."""
from dataclasses import replace
from types import MappingProxyType
import json
import time
import pytest

from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.stage2_planning import (CandidateProvider, EvaluationContext, PlanningBudget,
    NormalSchedulerCallback, PredictionColumn, PlanningResult, CandidateCatalog, RollingHorizonCPSATScheduler, _node)
from terminal_marl.core.rotating_slots.selective_replanning import (PlanningScope, PlanningHistory, RuleSelector, build_global_frame, LocalPlanner)
from terminal_marl.core.rotating_slots.selective_replanning.frame import FrameIncomplete
from terminal_marl.core.rotating_slots.selective_replanning.relations import build_relations
from terminal_marl.core.rotating_slots.trajectory import generate_transfer
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.planning_latency import LatencyAwareScheduler, PlanningTiming, project_visible_snapshot


def world(n=3):
    w=RotatingSlotWorld()
    for k in range(n): w.seed_hold(str(k),"outer",k,.9)
    return w


def setup_scope(w, ids):
    s=w.build_planning_snapshot(); p=CandidateProvider(planning_profile="downstream_service_v8")
    f=build_global_frame(s,p,EvaluationContext(s))
    u=tuple(sorted(a.id for a in __import__("terminal_marl.core.rotating_slots.stage2_planning",fromlist=["_decision_aircraft"])._decision_aircraft(s)))
    scope=PlanningScope(s.version,s.time,u,tuple(ids),tuple(sorted(set(s.aircraft)-set(ids))),actual_k=len(ids),objective_frame_id=f.identity)
    return s,p,f,scope,EvaluationContext(s,scope=scope,global_frame=f)


def test_A01_off_fixed_work_catalog_and_no_extra_latency_snapshot(monkeypatch):
    w=world(1); s=w.build_planning_snapshot(); p=CandidateProvider()
    a=p.get_columns(s,PlanningBudget(max_attempts=0),EvaluationContext(s))
    b=p.get_columns(s,PlanningBudget(max_attempts=0),EvaluationContext(s),scope=None)
    assert a==b
    cb=NormalSchedulerCallback(); wrapper=LatencyAwareScheduler(cb)
    wrapper.pending={}; wrapper.dirty=True
    class Port:
        time=0.
        def build_planning_snapshot(self): raise AssertionError("OFF history called")
    wrapper(None,Port())


def test_A02_scope_partition_and_decisions_complete_world():
    w=world(); s,p,f,scope,c=setup_scope(w,("0",))
    assert c.decisions=={"0"} and set(s.aircraft)=={"0","1","2"}
    assert set(scope.fixed_background_ids)=={"1","2"}
    with pytest.raises(ValueError): replace(scope,replan_ids=("missing",),actual_k=1).validate(s)


def test_A03_all_candidate_entrypoints_and_cold_reference_scoped(monkeypatch):
    s,p,f,scope,c=setup_scope(world(),("0",))
    b=PlanningBudget(max_attempts=0)
    calls=[]
    original=p.references.build_steps
    def counted(snapshot,a,node,budget):
        calls.append(a.id); return original(snapshot,a,node,budget)
    monkeypatch.setattr(p.references,"build_steps",counted)
    cat=p.get_columns(s,b,c,scope=scope)
    assert set(cat.columns)=={"0"} and set(calls)<={"0"}
    assert not b.attempts["1"] and not b.attempts["2"]
    with pytest.raises(ValueError):
        next(p.references.build_steps(s,s.aircraft["1"],_node(s,s.aircraft["1"]),b))


def test_A04_background_future_crossing_uses_full_pair_check():
    w=RotatingSlotWorld(); w.seed_hold("a","outer",0,.9); w.seed_hold("b","outer",3,.9)
    s,p,f,scope,c=setup_scope(w,("a",))
    x=generate_transfer("a","outer",0,1,0.,durations=(90.,)).plan
    y=generate_transfer("b","outer",3,0,0.,durations=(90.,)).plan
    assert x is not None and y is not None
    s=replace(s,active=MappingProxyType({"b":y}))
    f=build_global_frame(s,p,EvaluationContext(s)); scope=replace(scope,objective_frame_id=f.identity)
    c=EvaluationContext(s,scope=scope,global_frame=f)
    col=p._column(s,s.aircraft["a"],(x,),"MOVE",c)
    result=c.check_single(col)
    assert not result.accepted and result.status!="UNKNOWN"
    assert c.cache_stats["pair_selected_background"]>0


def test_A05_resource_blocker_and_expanded_joint_legal_inward_actions():
    w=RotatingSlotWorld(); w.seed_hold("a","outer",1,.9); w.seed_hold("b","middle",1,.9)
    s,p,f,scope,c=setup_scope(w,("a",))
    x=generate_transfer("a","outer",1,1,5.,durations=(40.,)).plan
    y=generate_transfer("b","middle",1,0,0.,durations=(40.,)).plan
    ca=p._column(s,s.aircraft["a"],(x,),"MOVE",c)
    assert not c.check_single(ca).accepted
    assert c.background_blockers[-1]["target"]=="b"
    expanded=replace(scope,replan_ids=("a","b"),fixed_background_ids=(),actual_k=2)
    joint=EvaluationContext(s,scope=expanded,global_frame=f)
    cb=p._column(s,s.aircraft["b"],(y,),"MOVE",joint)
    assert joint.check_batch((ca,cb))[0].accepted
    assert joint.check_commit_prefix((ca,cb)).accepted
    assert x.kind==y.kind=="TRANSFER"


def test_A06_A07_committed_background_resources_release_and_no_mutation():
    w=RotatingSlotWorld(); w.seed_hold("a","outer",1,.9); w.seed_hold("b","middle",1,.9)
    q=generate_transfer("b","middle",1,0,0.,durations=(40.,)).plan
    assert w.submit((q,)).accepted
    before=w.to_state() if hasattr(w,"to_state") else w.build_planning_snapshot()
    s,p,f,scope,c=setup_scope(w,("a",))
    rows=c.resources(s.aircraft["b"],())
    source=next(r for r in rows if r.resource==("middle",1))
    assert source.end<q.end_time and s.active["b"].trajectory_hash==q.trajectory_hash
    p.get_columns(s,PlanningBudget(max_attempts=0),c,scope=scope)
    assert w.build_planning_snapshot()==s


def test_A08_background_queue_wait_and_energy_not_constant():
    s,p,f,scope,c=setup_scope(world(2),("0",))
    cat=p.get_columns(s,PlanningBudget(max_attempts=0),c,scope=scope)
    from terminal_marl.core.rotating_slots.stage2_downstream import Demand
    base=f.baseline_by_id["1"]
    # Both demand identities are fixed; a real selected occupancy pushes the
    # unselected virtual job and its holding energy on its assigned pad.
    pad=dict(f.assignment)["1"]
    follow=cat.columns["0"][0]
    d=replace(cat.downstream.demands[follow.signature],full=True,real_intervals=((pad,base.ready,base.ready+base.duration+100,"0"),))
    changed=f.downstream({follow.signature:d},{"0"})
    original_rows=cat.downstream.evaluate((follow,))[1]
    changed_rows=changed.evaluate((follow,))[1]
    assert changed_rows["1"]["tail_wait_seconds"]>original_rows["1"]["tail_wait_seconds"]
    assert changed_rows["1"]["total_j"]>original_rows["1"]["total_j"]


@pytest.mark.parametrize("n",[2,4,6])
def test_A09_A10_fixed_follow_equivalence_and_frozen_frame(n):
    w=world(n); s,p,f,scope,c=setup_scope(w,("0",))
    local=p.get_columns(s,PlanningBudget(max_attempts=0),c,scope=scope)
    fullscope=replace(scope,replan_ids=scope.eligible_ids,fixed_background_ids=(),actual_k=n,status="FULL_H")
    fullctx=EvaluationContext(s,scope=fullscope,global_frame=f)
    full=p.get_columns(s,PlanningBudget(max_attempts=0),fullctx,scope=fullscope)
    ls=tuple(rows[0] for rows in local.columns.values()); fs=tuple(rows[0] for rows in full.columns.values())
    assert local.downstream.evaluate(ls)==full.downstream.evaluate(fs)
    assert local.downstream.order==full.downstream.order==f.order
    assert local.downstream.assignment==full.downstream.assignment==dict(f.assignment)
    assert c.check_batch(ls)[0].accepted==fullctx.check_batch(fs)[0].accepted
    with pytest.raises(TypeError): f.baselines[0].diagnostics["new"]=1


def test_A11_scope_cache_and_cut_identity_invalidation():
    s,p,f,scope,c=setup_scope(world(2),("0",))
    cat=p.get_columns(s,PlanningBudget(max_attempts=0),c,scope=scope)
    state,report=RollingHorizonCPSATScheduler._continuation(s,cat,c)
    state["single_exclusions"].add(next(iter(cat.columns["0"])).signature)
    alternate=replace(scope,replan_ids=("1",),fixed_background_ids=("0",))
    other=EvaluationContext(s,scope=alternate,global_frame=f,validation_cache=p.validation_cache)
    assert c.snapshot_key!=other.snapshot_key
    othercat=p.get_columns(s,PlanningBudget(max_attempts=0),other,scope=alternate)
    new,_=RollingHorizonCPSATScheduler._continuation(s,othercat,other)
    assert not new["single_exclusions"] and state["identity"]!=new["identity"]


def test_A12_unknown_not_cached_safe_or_cut(monkeypatch):
    s,p,f,scope,c=setup_scope(world(1),("0",))
    col=p._column(s,s.aircraft["0"],(),"WAIT",c)
    monkeypatch.setattr(c,"_check_single",lambda _:ValidationResult(False,"UNKNOWN","BUDGET"))
    assert c.check_single(col).status=="UNKNOWN" and not c.singles
    assert not c.completed_evaluations and not c.background_blockers


def test_A13_A19_delivery_ages_and_roundtrip_not_selection_or_execution():
    w=world(1); s=w.build_planning_snapshot(); h=PlanningHistory(); h.observe(s)
    h.queue_reviews("pending",s,("0",))
    assert h.ages("0",10)["review_age"]==10 and h.ages("0",10)["progress_age"]==10
    restored=PlanningHistory(); restored.load_state_dict(json.loads(json.dumps(h.state_dict())))
    assert restored.deliver("pending",s,11)==("0",)
    assert restored.ages("0",12)["review_age"]==1 and restored.ages("0",12)["progress_age"]==12
    stale=replace(s,aircraft={"0":replace(s.aircraft["0"],emergency=replace(s.aircraft["0"].emergency,eligible=True))})
    restored.queue_reviews("stale",s,("0",)); assert not restored.deliver("stale",stale,15)


def test_A13_imported_unknown_and_completed_actual_start_recovery():
    w=world(1); q=generate_transfer("0","outer",0,0,0.,durations=(40.,)).plan
    assert w.submit((q,)).accepted
    w.advance(45.)
    s=w.build_planning_snapshot(); h=PlanningHistory(); h.initialize_imported(s)
    age=h.ages("0",45.)
    assert age["review_unknown"] and not age["progress_unknown"] and h.entries["0"].progressed==q.start_time
    h.queue_reviews("done",replace(s,aircraft={"0":replace(s.aircraft["0"],emergency=replace(s.aircraft["0"].emergency,normal_margin=.2))}),("0",))
    assert h.deliver("done",s,46.)==("0",)  # advisory refresh is not qualification change


def test_A14_mandatory_overflow_explicit_full_shared_scope():
    w=world(13); s,p,f,scope,c=setup_scope(w,())
    h=PlanningHistory(); h.observe(s)
    selected,info=RuleSelector().select(s,f,h,now=61.)
    assert len(selected.mandatory_ids)==13 and selected.actual_k==13
    assert selected.status=="MANDATORY_OVERFLOW_FULL" and info["status"]=="MANDATORY_OVERFLOW"
    assert set(selected.replan_ids)==set(selected.eligible_ids)


@pytest.mark.parametrize("n,chosen",[(0,()),(1,()),(1,("0",)),(3,("0","1","2"))])
def test_A15_empty_full_single_and_keep(n,chosen):
    s,p,f,scope,c=setup_scope(world(n),chosen)
    cat=p.get_columns(s,PlanningBudget(max_attempts=0),c,scope=scope)
    assert set(cat.columns)==set(chosen)
    selected=tuple(rows[0] for rows in cat.columns.values())
    assert c.prefix(selected)==() and len(cat.downstream.evaluate(selected)[1])==n


def test_A16_only_revealed_snapshot_and_selector_soc_sensitive():
    w=world(10); s,p,f,scope,c=setup_scope(w,())
    h=PlanningHistory(); h.observe(s)
    chosen,info=RuleSelector().select(s,f,h)
    assert len(chosen.replan_ids)==8
    missed=next(aid for aid in scope.eligible_ids if aid not in chosen.replan_ids)
    aircraft=dict(s.aircraft); aircraft[missed]=replace(aircraft[missed],soc=.3)
    s2=replace(s,aircraft=aircraft); f2=build_global_frame(s2,p,EvaluationContext(s2))
    selected,info2=RuleSelector().select(s2,f2,h)
    assert missed in selected.replan_ids
    assert not info2["energy_masks"][missed]["known"]
    assert info2["energy_masks"][missed]["can_safely_postpone"] is False
    assert not hasattr(s,"manifest")


def test_A17_full_long_background_horizon(monkeypatch):
    s,p,f,scope,c=setup_scope(world(2),("0",))
    a=p._column(s,s.aircraft["0"],(),"WAIT",c)
    b=PredictionColumn("1","BACKGROUND",(),(),(0,0,0,0,0))
    original=c.traffic
    def traffic(column):
        t=original(column)
        if column.aircraft_id=="1": t.end_time=700.
        return t
    monkeypatch.setattr(c,"traffic",traffic)
    seen=[]
    monkeypatch.setattr(c,"_geometry_validation",lambda x,y,start,end,config:seen.append(end) or ValidationResult(True,"SAFE"))
    assert c.check_pair(a,b).accepted and seen[0]>240.


def test_A18_frame_budget_exhaustion_is_explicit_unknown():
    w=world(1); s=w.build_planning_snapshot(); p=CandidateProvider(planning_profile="downstream_service_v8")
    budget=PlanningBudget(search_wall_deadline=time.perf_counter()-1.)
    with pytest.raises(FrameIncomplete,match="BUDGET_UNKNOWN"): build_global_frame(s,p,EvaluationContext(s,budget=budget))


def test_A19_callback_binding_roundtrip_and_phase_limits():
    cb=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",selective_method="RULE_LOCAL_K8")
    cb.initialize_imported_history(world(1).build_planning_snapshot())
    state=cb.state_dict(); other=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",selective_method="RULE_LOCAL_K8")
    other.load_state_dict(state)
    assert state==other.state_dict() and cb.online_policy=="progressive_v1"
    with pytest.raises(ValueError): NormalSchedulerCallback("rh_cp_sat",selective_method="FULL_H")


def test_X01_visible_demand_conservation_and_unknown_masks():
    w=world(2); w.seed_service("ground",0,30.)
    s,p,f,scope,c=setup_scope(w,("0",))
    cat=p.get_columns(s,PlanningBudget(max_attempts=0),c,scope=scope)
    objective,rows=cat.downstream.evaluate((cat.columns["0"][0],))
    assert set(rows)==set(s.aircraft) and "ground" in rows
    assert rows["1"]["reference_unknown_flag"] and rows["ground"]["reference_status"]=="KNOWN"


def test_relations_include_pad_release_without_claiming_proof():
    w=world(1); w.seed_service("g",0,30.)
    s=w.build_planning_snapshot(); relations=build_relations(s,("0",))
    assert any(r.kind=="pad_release" and r.target=="g" and not r.definite and not r.required for r in relations)


def test_A05_A11_bounded_expansion_recertifies_with_same_budget():
    w=RotatingSlotWorld(); w.seed_hold("a","outer",1,.9); w.seed_hold("b","middle",1,.9)
    s,p,f,scope,c=setup_scope(w,("a",))
    plans={"a":generate_transfer("a","outer",1,1,5.,durations=(40.,)).plan,
           "b":generate_transfer("b","middle",1,0,0.,durations=(40.,)).plan}
    class Provider(CandidateProvider):
        def iter_progressive_columns(self,snapshot,budget,context,*,scope=None):
            cols={aid:[self._column(snapshot,snapshot.aircraft[aid],(),"KEEP",context),
                       self._column(snapshot,snapshot.aircraft[aid],(),"WAIT",context),
                       self._column(snapshot,snapshot.aircraft[aid],(plans[aid],),"MOVE",context)] for aid in scope.replan_ids}
            refs={aid:cols[aid][0].reference for aid in cols}
            yield self._progressive_catalog(snapshot,budget,context,cols,refs,[snapshot.aircraft[aid] for aid in cols])
    class Scheduler:
        budgets=[]; contexts=[]
        def plan(self,snapshot,catalog,context,budget,*,incumbent=None):
            self.budgets.append(id(budget)); self.contexts.append(context.snapshot_key)
            rows=tuple(next(c for c in v if c.kind=="MOVE") for v in catalog.columns.values())
            result,_=context.check_batch(rows)
            evaluation=context.check_commit_prefix(rows) if result.accepted else None
            return PlanningResult(rows,context.prefix(rows),evaluation,"FEASIBLE" if evaluation and evaluation.accepted else "INCOMPLETE",
                catalog.downstream.key(rows),certified=bool(evaluation and evaluation.accepted))
    scheduler=Scheduler(); local=LocalPlanner(Provider(planning_profile="downstream_service_v8"),scheduler)
    budget=PlanningBudget(candidate_active_seconds=0.)
    result=local.plan(s,scope,f,budget)
    assert result.certified and local.telemetry["solve_calls"]==2 and local.telemetry["scope_expansions"]==1
    assert len(set(scheduler.budgets))==1 and len(set(scheduler.contexts))==2
    assert set(local.last_scope.replan_ids)=={"a","b"}


def test_A13_expected_committed_completion_allows_delivered_review():
    w=world(1); p=generate_transfer("0","outer",0,0,0.,durations=(40.,)).plan
    assert w.submit((p,)).accepted
    source=w.build_planning_snapshot(); h=PlanningHistory(); h.observe(source); h.queue_reviews("pending",source,("0",))
    w.advance(45.)
    assert h.deliver("pending",w.build_planning_snapshot(),45.)==("0",)


def test_A18_callback_frame_unknown_returns_no_proposal(monkeypatch):
    from terminal_marl.core.rotating_slots.selective_replanning import frame
    def incomplete(*args): raise frame.FrameIncomplete("FIXED_CONTINUATION_PROTECTION_UNKNOWN")
    monkeypatch.setattr("terminal_marl.core.rotating_slots.selective_replanning.build_global_frame",incomplete)
    cb=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",selective_method="RULE_LOCAL_K8")
    result=cb(world(1)._scheduler_port)
    assert result.status=="GLOBAL_FRAME_INCOMPLETE" and not result.certified and not result.prefix
    assert cb.records[-1]["selective_replanning"]["solve_calls"]==0


@pytest.mark.parametrize("n",[2,4,6])
def test_A09_cp_frozen_background_matches_full_fixed_follow(n):
    pytest.importorskip("ortools.sat.python.cp_model",reason="real CP backend required; no substitute")
    from terminal_marl.core.rotating_slots.trajectory import generate_final
    from terminal_marl.core.rotating_slots.stage2_planning import NormalReference
    w=RotatingSlotWorld(); w.seed_hold("0","inner",0,.9)
    for k in range(1,n): w.seed_hold(str(k),"outer",k,.9)
    s,p,f,scope,c=setup_scope(w,("0",))
    plan=generate_final("0","inner",0,0,0.).plan
    assert plan is not None
    initial={"0":[p._column(s,s.aircraft["0"],(),kind,c) for kind in ("KEEP","WAIT")]+[p._column(s,s.aircraft["0"],(plan,),"MOVE",c)]}
    # Deliberate late cache improvement must not change any FOLLOW baseline.
    for aid,a in s.aircraft.items():
        node=_node(s,a); key=(s.config.config_hash,node.ring,node.slot,s.time+240.,node.version,aid)
        p.references.cache[key]=NormalReference("KNOWN",soc_required=.001,peak_required=0.,duration=1.,energy_j=1.)
    local=p._progressive_catalog(s,PlanningBudget(),c,initial,{"0":initial["0"][0].reference},[s.aircraft["0"]])
    fullscope=replace(scope,replan_ids=scope.eligible_ids,fixed_background_ids=(),actual_k=n,status="FULL_H")
    fullctx=EvaluationContext(s,scope=fullscope,global_frame=f)
    allcols={aid:list(local.columns[aid]) if aid=="0" else [p._column(s,a,(),kind,fullctx) for kind in ("KEEP","WAIT")]
             for aid,a in s.aircraft.items()}
    full=p._progressive_catalog(s,PlanningBudget(),fullctx,allcols,{aid:rows[0].reference for aid,rows in allcols.items()},list(s.aircraft.values()))
    full=replace(full,columns={aid:rows if aid=="0" else rows[:1] for aid,rows in full.columns.items()})
    scheduler=RollingHorizonCPSATScheduler()
    a=scheduler.plan(s,local,c,PlanningBudget(soft_seconds=10.,solver_seconds=2.))
    b=scheduler.plan(s,full,fullctx,PlanningBudget(soft_seconds=10.,solver_seconds=2.))
    assert a.certified and b.certified and a.objective==b.objective
    assert a.solver_records and b.solver_records
    assert any(col.kind=="MOVE" for col in local.columns["0"])


def test_A03_actual_selected_materialization_unselected_zero():
    s,p,f,scope,c=setup_scope(world(3),("0",))
    budget=PlanningBudget(candidate_seconds=2.,max_per_aircraft=2,max_attempts=2,candidate_active_seconds=0.)
    cat=p.get_columns(s,budget,c,scope=scope)
    assert set(cat.columns)=={"0"} and budget.screening["materialized:0"]>0
    assert all(budget.screening["materialized:"+aid]==0 and budget.attempts[aid]==0 for aid in ("1","2"))


def test_A10_follow_frozen_despite_reference_cache_changes():
    from terminal_marl.core.rotating_slots.stage2_planning import NormalReference
    s,p,f,scope,c=setup_scope(world(2),("0",))
    for aid,a in s.aircraft.items():
        node=_node(s,a); key=(s.config.config_hash,node.ring,node.slot,s.time+240.,node.version,aid)
        p.references.cache[key]=NormalReference("KNOWN",soc_required=.001,peak_required=0.,duration=1.,energy_j=1.)
    cat=p.get_columns(s,PlanningBudget(max_attempts=0),c,scope=scope)
    assert cat.downstream.demands[cat.columns["0"][0].signature] is f.baseline_by_id["0"]
    assert cat.downstream.fixed_demands["1"] is f.baseline_by_id["1"]


def test_A18_A19_real_latency_delivery_history_and_no_early_progress():
    pytest.importorskip("ortools.sat.python.cp_model",reason="real CP backend required; no substitute")
    w=RotatingSlotWorld(); w.seed_hold("a","inner",0,.9); w.configure_practical_runtime(True)
    cb=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",selective_method="RULE_LOCAL_K8")
    wrapper=LatencyAwareScheduler(cb,lead_seconds=20.,search_handoff_reserve_seconds=4.)
    wrapper(w._scheduler_port)
    assert cb.selective_history.entries["a"].reviewed is None
    state=wrapper.state_dict()
    assert state["pending"]["stage"]=="RESULT" and state["callback"]["selective_replanning"]["history"]["pending"]
    due=wrapper.pending["due"]; w.advance(due)
    wrapper.planning_event(w._scheduler_port)
    assert cb.selective_history.entries["a"].reviewed==due
    assert cb.selective_history.entries["a"].progressed is None
    assert wrapper.records[-1]["selective_history_work_s"]>0


def test_A16_X01_revealed_upstream_identity_mask_without_fictitious_energy():
    from terminal_marl.core.rotating_slots.demand import ArrivalRequest
    s,p,f,scope,c=setup_scope(world(1),("0",))
    request=ArrivalRequest("upstream",0.,0,.8,"short")
    revealed=replace(s,requests={"upstream":request})
    updated=build_global_frame(revealed,p,EvaluationContext(revealed))
    assert updated.identity!=f.identity and updated.upstream_request_ids==("upstream",)
    assert updated.objective_eligibility["upstream"]=="NOT_ADMITTED_NO_FLIGHT_STATE"
    assert updated.baselines==f.baselines and updated.order==f.order
    assert updated.upstream_requests[0][5:7]==(None,"UNKNOWN")


def test_A19_real_selective_checkpoint_restore_all_pending_stages(tmp_path):
    pytest.importorskip("ortools.sat.python.cp_model",reason="real CP backend required; no substitute")
    from terminal_marl.core.rotating_slots.world import STAGE2_POLICY_ID
    from terminal_marl.core.rotating_slots.stage2_campaign import checkpoint_identity,digest,save_checkpoint,restore_checkpoint
    def scheduler():
        cb=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",selective_method="RULE_LOCAL_K8")
        return LatencyAwareScheduler(cb,lead_seconds=20.,search_handoff_reserve_seconds=4.)
    w=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID); w.seed_hold("a","inner",0,.9)
    wrapper=scheduler(); wrapper(w._scheduler_port)
    assert wrapper.pending["stage"]=="RESULT" and wrapper.pending["proposal"]
    identity=checkpoint_identity(dict(source_hash="test",base_config_hash=w.config.config_hash,overlay_sha256="test",
        planning_profile="downstream_service_v8"),digest(w.manifest.to_dict()))
    def restore(label,current,active):
        path=tmp_path/(label+".json")
        before=active.state_dict()
        save_checkpoint(path,current,active,identity)
        loaded_scheduler=scheduler(); loaded=restore_checkpoint(path,loaded_scheduler,identity)
        assert loaded_scheduler.pending==active.pending
        assert loaded_scheduler.callback.state_dict()==before["callback"]
        return loaded,loaded_scheduler
    w,wrapper=restore("computing_result",w,wrapper)
    for _ in range(5):
        if wrapper.pending["stage"]=="PREPARED": break
        w.advance(wrapper.pending["due"],False); wrapper.planning_event(w._scheduler_port)
        assert wrapper.pending is not None,wrapper.records
    assert wrapper.pending["stage"]=="PREPARED"
    w,wrapper=restore("prepared",w,wrapper)
    due=wrapper.pending["due"]; hashes=tuple(p.trajectory_hash for p in wrapper.pending["proposal"])
    w.advance(due,False); wrapper.planning_event(w._scheduler_port)
    assert wrapper.pending is None and wrapper.records[-1]["status"]=="COMMITTED"
    assert tuple(p.trajectory_hash for p in w.active.values())==hashes
    starts=sum(e["type"]=="maneuver_started" for e in w.events)
    w,wrapper=restore("published",w,wrapper)
    wrapper.planning_event(w._scheduler_port)
    assert wrapper.pending is None and sum(e["type"]=="maneuver_started" for e in w.events)==starts
    assert wrapper.callback.selective_history.entries["a"].progressed==due


def test_A07_cp_real_reservation_identity_deduplicated():
    cp_model=pytest.importorskip("ortools.sat.python.cp_model",reason="real CP backend required; no substitute")
    from terminal_marl.core.rotating_slots.stage2_downstream import JointDownstream,Demand
    d=Demand("g",True,0,100,0,0,100,0,0,0,0,None,{},((0,0,100,"g"),))
    ds=JointDownstream({},((0,0,100,"g"),),("g",),fixed_demands=(d,),assignment={},frame_id="f")
    model=cp_model.CpModel(); ds.cp_expressions(model,{},{}); solver=cp_model.CpSolver()
    assert solver.Solve(model)==cp_model.OPTIMAL
    intervals=sum(r.has_interval() if hasattr(r,"has_interval") else r.HasField("interval") for r in model.Proto().constraints)
    assert intervals==1


def test_A19_multiaircraft_follow_frozen_diagnostics_checkpoint_roundtrip(tmp_path,monkeypatch):
    """Real FOLLOW/frame diagnostics and checkpoint path; fixed selection, no CP."""
    from terminal_marl.core.rotating_slots.world import STAGE2_POLICY_ID
    from terminal_marl.core.rotating_slots.config import Section
    from terminal_marl.core.rotating_slots.replay import _plain
    from terminal_marl.core.rotating_slots.stage2_campaign import checkpoint_identity,digest,save_checkpoint,restore_checkpoint
    class FollowProvider(CandidateProvider):
        def iter_progressive_columns(self,snapshot,budget,context,*,scope=None):
            columns={aid:[self._column(snapshot,snapshot.aircraft[aid],(),"KEEP",context)] for aid in scope.replan_ids}
            yield self._progressive_catalog(snapshot,budget,context,columns,
                {aid:rows[0].reference for aid,rows in columns.items()},[snapshot.aircraft[aid] for aid in columns])
    def choose_follow(self,snapshot,catalog,context,budget,*,incumbent=None):
        selected=tuple(rows[0] for rows in catalog.columns.values())
        checked,_=context.check_batch(selected)
        assert checked.accepted
        evaluation=context.check_commit_prefix(selected)
        assert evaluation.accepted
        return PlanningResult(selected,(),evaluation,"FIXED_FOLLOW_SERIALIZATION_FIXTURE",
            catalog.downstream.key(selected),certified=True)
    monkeypatch.setattr(RollingHorizonCPSATScheduler,"plan",choose_follow)
    def scheduler():
        cb=NormalSchedulerCallback("rh_cp_sat",candidate_provider=FollowProvider(planning_profile="downstream_service_v8"),
            planning_profile="downstream_service_v8",selective_method="FULL_H")
        return LatencyAwareScheduler(cb,lead_seconds=20.,search_handoff_reserve_seconds=4.)
    w=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    for aid,ring,slot in (("a","outer",0),("b","outer",5),("c","middle",2)):
        w.seed_hold(aid,ring,slot,.9)
    wrapper=scheduler();wrapper(w._scheduler_port)
    cb=wrapper.callback
    assert wrapper.pending["stage"]=="RESULT" and not wrapper.pending["proposal"]
    assert cb.records[-1]["keep"]==3 and len(cb.records[-1]["downstream"])==3
    assert all(isinstance(row["window_wait_seconds_by_ring"],Section) for row in cb.records[-1]["downstream"].values())
    with pytest.raises(TypeError,match="Section"):
        json.dumps(cb.records,allow_nan=False)
    assert cb.selective_history.pending and len(cb.selective_history.entries)==3
    # Match the observed failure boundary while executing only real holding.
    w.advance(5.,False)
    assert w.time==5. and not any(e["type"]=="maneuver_started" for e in w.events)
    before=wrapper.state_dict()
    assert before["callback"]["records"]==_plain(cb.records)
    identity=checkpoint_identity(dict(source_hash="follow_serialization_test",base_config_hash=w.config.config_hash,
        overlay_sha256="test",planning_profile="downstream_service_v8"),digest(w.manifest.to_dict()))
    path=tmp_path/"follow_t5.json"
    save_checkpoint(path,w,wrapper,identity)
    loaded_scheduler=scheduler();loaded=restore_checkpoint(path,loaded_scheduler,identity)
    assert loaded_scheduler.state_dict()==before
    assert loaded_scheduler.pending==wrapper.pending
    restored_history=loaded_scheduler.callback.selective_history
    assert _plain(restored_history.state_dict())==_plain(cb.selective_history.state_dict())
    review_key=next(iter(cb.selective_history.pending))
    original_delivered=cb.selective_history.deliver(review_key,w.build_planning_snapshot(),5.)
    restored_delivered=restored_history.deliver(review_key,loaded.build_planning_snapshot(),5.)
    assert set(original_delivered)==set(restored_delivered)=={"a","b","c"}
    assert restored_history.entries==cb.selective_history.entries
    assert all(entry.reviewed==5. and entry.progressed is None for entry in restored_history.entries.values())
    assert loaded.events==w.events and loaded.time==5.
    assert loaded.aircraft==w.aircraft and loaded.active==w.active and loaded.soft==w.soft
    assert not any(e["type"]=="maneuver_started" for e in loaded.events)


def test_A19_selective_state_dict_nonfinite_fails_closed_and_off_unchanged():
    from terminal_marl.core.rotating_slots.config import Section
    cb=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",selective_method="FULL_H")
    cb.records.append(dict(downstream=Section(dict(aircraft=Section(dict(invalid=float("nan")))))))
    with pytest.raises(ValueError,match="Out of range float"):
        cb.state_dict()
    off=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8")
    off.records.append(dict(diagnostic=Section(dict(value=1.))))
    with pytest.raises(TypeError,match="Section"):
        off.state_dict()


def _ground_frame_fixture(time_s=10.,calendar_change=None):
    from terminal_marl.core.rotating_slots.world import STAGE2_POLICY_ID
    w=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    for pad,end in ((0,15.),(2,30.),(4,45.)):
        w.seed_service("initial_ground_pad"+str(pad),pad,remaining=end)
    w.seed_hold("airborne","outer",0,.9)
    if time_s: w.advance(time_s,False)
    snapshot=w.build_planning_snapshot()
    if calendar_change is not None:
        snapshot=replace(snapshot,pads=tuple(calendar_change(row) if row.resource==0 else row for row in snapshot.pads))
    context=EvaluationContext(snapshot)
    provider=CandidateProvider(planning_profile="downstream_service_v8")
    frame=build_global_frame(snapshot,provider,context)
    return w,snapshot,context,frame


@pytest.mark.parametrize("time_s",[0.,10.])
def test_A07_ground_frame_covered_service_interval_only_once(time_s):
    from terminal_marl.core.rotating_slots.stage2_downstream import ticks
    w,snapshot,context,frame=_ground_frame_fixture(time_s)
    before=w.build_planning_snapshot().version
    actual=tuple((row.resource,ticks(row.start),ticks(row.end),row.aircraft_id) for row in snapshot.pads)
    assert frame.calendars==actual
    for pad,end in ((0,15.),(2,30.),(4,45.)):
        aid="initial_ground_pad"+str(pad)
        raw=context.resources(snapshot.aircraft[aid],())
        assert [(r.resource[1],ticks(r.start),ticks(r.end),aid) for r in raw]==[(pad,ticks(time_s),ticks(end),aid)]
        assert frame.baseline_by_id[aid].real_intervals==()
        assert (pad,0,ticks(end),aid) in frame.calendars
    assert w.build_planning_snapshot().version==before
    assert frame.downstream({},()).fixed_identity==set(actual)


@pytest.mark.parametrize("change",["different_owner","partial_extension"])
def test_A07_ground_frame_keeps_other_owner_and_partly_covered_intervals(change):
    from terminal_marl.core.rotating_slots.stage2_downstream import ticks
    def altered(row):
        return replace(row,aircraft_id="other_owner") if change=="different_owner" else replace(row,end=12.)
    w,snapshot,context,frame=_ground_frame_fixture(calendar_change=altered)
    original_version=w.build_planning_snapshot().version
    assert frame.baseline_by_id["initial_ground_pad0"].real_intervals==((0,100,150,"initial_ground_pad0"),)
    expected_owner="other_owner" if change=="different_owner" else "initial_ground_pad0"
    expected_end=150 if change=="different_owner" else 120
    assert (0,0,expected_end,expected_owner) in frame.calendars
    assert frame.calendars==tuple((r.resource,ticks(r.start),ticks(r.end),r.aircraft_id) for r in snapshot.pads)
    assert w.build_planning_snapshot().version==original_version


def test_A07_ground_frame_real_cp_old_duplicate_infeasible_normalized_feasible():
    cp_model=pytest.importorskip("ortools.sat.python.cp_model",reason="real CP backend required for ground-interval regression; no substitute")
    from terminal_marl.core.rotating_slots.stage2_downstream import ticks
    w,snapshot,context,frame=_ground_frame_fixture()
    before=w.build_planning_snapshot().version
    # Reconstruct precisely the previous frame representation from the real
    # current-state resource view, without changing a world/calendar.
    legacy=[]
    for demand in frame.baselines:
        aircraft=snapshot.aircraft[demand.aircraft_id]
        if aircraft.status=="LANDED_SERVICE":
            raw=tuple((r.resource[1],ticks(r.start),ticks(r.end),aircraft.id) for r in context.resources(aircraft,()))
            demand=replace(demand,real_intervals=raw)
        legacy.append(demand)
    old=replace(frame,baselines=tuple(legacy))
    def solve(candidate):
        model=cp_model.CpModel();candidate.downstream({},()).cp_expressions(model,{},{})
        solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=2.
        return solver.Solve(model)
    assert solve(old)==cp_model.INFEASIBLE
    assert solve(frame)==cp_model.OPTIMAL
    assert frame.calendars==old.calendars and frame.assignment==old.assignment
    assert frame.downstream({},()).evaluate(())==old.downstream({},()).evaluate(())
    # A different owner's overlapping actual reservation is still a conflict.
    _,_,_,conflicting=_ground_frame_fixture(calendar_change=lambda r:replace(r,aircraft_id="other_owner"))
    assert solve(conflicting)==cp_model.INFEASIBLE
    assert w.build_planning_snapshot().version==before
