"""Bounded v3 ordering contracts on fixed snapshots, without a rollout."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider,EvaluationContext,PlanningBudget,PlanningNode,PredictionColumn,CandidateCatalog,PlanningResult,NormalSchedulerCallback)
from terminal_marl.core.rotating_slots.selective_replanning import RuleSelector,PlanningScope,LocalPlanner
from terminal_marl.core.rotating_slots.selective_replanning.relations import Relation
from terminal_marl.core.rotating_slots.selective_replanning.near_term import (
    adjustment_access,summarize_opportunities,cooperative_pad_groups)
from terminal_marl.core.rotating_slots.resources import Reservation


V3="opportunity_executable_v3"


def snapshot(n=13):
    w=RotatingSlotWorld()
    for k in range(n): w.seed_hold(str(k),"outer",k,.9)
    return w.build_planning_snapshot()


def history():
    return SimpleNamespace(ages=lambda aid,now:dict(review_age=0.,review_unknown=False,
        progress_age=0.,progress_unknown=False))


def patch_nodes(monkeypatch,s,future=(),urgent=()):
    def node(snapshot,a):
        return PlanningNode(a.ring,a.slot,40. if a.id in future else snapshot.time,
                            .249 if a.id in urgent else .9,0.,a.version)
    monkeypatch.setattr("terminal_marl.core.rotating_slots.stage2_planning._node",node)
    return node


@pytest.mark.parametrize("end,allowed",[(19.,True),(20.,True),(21.,False)])
def test_committed_endpoint_before_equal_after_effective_not_capture(end,allowed):
    s=snapshot(1)
    s=replace(s,time=20.,planning_timing=SimpleNamespace(effective_time=20.,snapshot_time=0.),
              active={"0":SimpleNamespace(end_time=end)},
              slots=(Reservation(("outer",0),"0",0.,5.,committed=True),))
    node=PlanningNode("middle",0,end,.9,0.,1)
    row=adjustment_access(s,s.aircraft["0"],node)
    assert row["adjustable"]==allowed and row["committed_end_time"]==end
    assert row["effective_time"]==20. and row["node_time"]==end
    if not allowed: assert row["reason"]=="NOT_ADJUSTABLE_AT_EFFECTIVE"
    # A later soft commitment, rather than the first active endpoint, binds.
    s=replace(s,soft={"0":SimpleNamespace(end_time=30.)})
    assert not adjustment_access(s,s.aircraft["0"],node)["adjustable"]


def test_ordinary_future_nodes_do_not_consume_fairness_or_cooperative_scope(monkeypatch):
    s=replace(snapshot(),time=20.,planning_timing=SimpleNamespace(effective_time=20.,snapshot_time=0.))
    patch_nodes(monkeypatch,s,future={"0"})
    h=history(); h.ages=lambda aid,now:dict(review_age=100. if aid=="0" else 0.,review_unknown=False,
                                         progress_age=0.,progress_unknown=False)
    scope,info=RuleSelector(variant=V3).select(s,SimpleNamespace(identity="f"),h)
    assert "0" not in scope.replan_ids and "0" in scope.fixed_background_ids
    assert "0" not in info["fairness_selected_ids"]
    assert info["unselected_reasons"]["0"]=="NOT_ADJUSTABLE_AT_EFFECTIVE"
    assert info["adjustment_access"]["0"]["node_time"]==40.
    assert scope.actual_k==8 and set(scope.replan_ids)|set(scope.fixed_background_ids)==set(s.aircraft)
    full,_=RuleSelector(variant=V3).select(s,SimpleNamespace(identity="f"),h,method="FULL_H")
    assert set(full.replan_ids)==set(s.aircraft)


def test_future_hard_urgency_and_required_closure_remain_complete(monkeypatch):
    s=replace(snapshot(),time=20.,planning_timing=SimpleNamespace(effective_time=20.,snapshot_time=0.))
    patch_nodes(monkeypatch,s,future={"0","12"},urgent={"0"})
    relation=(Relation("0","12","validated_background_blocker",True,True,("outer",12)),)
    scope,info=RuleSelector(variant=V3).select(s,SimpleNamespace(identity="f"),history(),relation)
    assert {"0","12"}<=set(scope.mandatory_ids) and {"0","12"}<=set(scope.replan_ids)
    assert info["hard_urgent_count"]==1 and not info["adjustment_access"]["12"]["adjustable"]


def test_ordinary_closure_with_future_required_owner_is_skipped_without_promoting_hard(monkeypatch):
    s=replace(snapshot(),time=20.)
    patch_nodes(monkeypatch,s,future={"12"})
    relation=(Relation("0","12","validated_background_blocker",True,True,("outer",12)),)
    scope,info=RuleSelector(variant=V3).select(s,SimpleNamespace(identity="f"),history(),relation)
    assert not scope.mandatory_ids and "0" not in scope.replan_ids and "12" not in scope.replan_ids
    assert any(r["reason"]=="DEPENDENCY_NOT_ADJUSTABLE_AT_EFFECTIVE" for r in info["skipped_optional_groups"])


@pytest.mark.parametrize("background_ring",["inner","outer"])
def test_pad_anchor_has_specific_inner_middle_outer_links_and_keeps_background(background_ring):
    w=RotatingSlotWorld(); w.seed_hold("anchor","inner",0,.9)
    w.seed_hold("middle","middle",1,.9); w.seed_hold("outer","outer",1,.9)
    for k in range(10): w.seed_hold("other"+str(k),background_ring,k+1 if background_ring=="inner" else k+5,.9)
    s=w.build_planning_snapshot(); rows=summarize_opportunities(s,s.aircraft,executable=True)
    groups=cooperative_pad_groups(rows)
    chain=next(g for g in groups if {"anchor","middle"}<=set(g["members"]) and len(g["members"])==3)
    follower=next(aid for aid in chain["members"] if aid not in ("anchor","middle"))
    assert s.aircraft[follower].ring=="outer"
    assert len({tuple(link["resource"]) for link in chain["release_dependencies"]})==len(chain["release_dependencies"])
    assert any(link["follower"]==follower and link["owner"]=="middle" and link["resource"]==("middle",1)
               for link in chain["release_dependencies"])
    assert chain["anchor_id"]=="anchor" and chain["pad_id"] is not None
    assert chain["anchor_start_available"] and chain["anchor_start_time"]==s.time
    assert {tuple(r["resource"]) for r in chain["release_dependencies"]}=={("inner",0),("middle",1)}
    assert all(not r["certified"] for r in chain["release_dependencies"])
    scope,info=RuleSelector(variant=V3).select(s,SimpleNamespace(identity="f"),history())
    assert set(chain["members"])<=set(scope.replan_ids)
    assert len(scope.fixed_background_ids)==5 and scope.actual_k==8
    assert not scope.mandatory_ids


def test_pad_group_order_prioritizes_actual_arrival_over_group_size():
    def row(resource,arrival,members=()):
        return dict(adjustment_access={"adjustable":True},options=(dict(resource=resource,earliest_arrival=arrival,
            start_available=True,start_time=0.,pad_id=resource[1] if resource[0]=="pad" else None,release_time=0.,
            joint_members=members,release_dependencies=[dict(owner=m,resource=resource,estimated_release_time=10.,
                owner_start_available=True,certified=False) for m in members]),))
    opportunities={"early":row(("pad",0),30.),"late":row(("pad",1),100.),
                   "middle":row(("inner",0),110.,("late",)),"outer":row(("middle",0),120.,("middle",))}
    groups=cooperative_pad_groups(opportunities)
    assert groups[0]["anchor_id"]=="early" and len(groups[0]["members"])==1
    assert groups[1]["members"]==("late","middle","outer")
    assert max(map(lambda r:len(r["members"]),groups))==3


def test_anchor_fallback_fits_last_seat_but_retains_true_dependency_closure(monkeypatch):
    w=RotatingSlotWorld(); w.seed_hold("anchor","inner",0,.9)
    w.seed_hold("middle","middle",1,.9); w.seed_hold("outer","outer",1,.9)
    for k in range(7): w.seed_hold("hard"+str(k),"outer",k+5,.9)
    s=w.build_planning_snapshot(); urgent={"hard"+str(k) for k in range(7)}
    patch_nodes(monkeypatch,s,urgent=urgent)
    scope,info=RuleSelector(variant=V3).select(s,SimpleNamespace(identity="f"),history())
    assert set(scope.replan_ids)==urgent|{"anchor"} and scope.actual_k==8
    assert any(row["reason"]=="GROUP_EXCEEDS_REMAINING_NOMINAL_SEATS" for row in info["skipped_optional_groups"])
    relations=(Relation("anchor","middle","committed_release",True,True),)
    scope,_=RuleSelector(variant=V3).select(s,SimpleNamespace(identity="f"),history(),relations)
    assert urgent|{"anchor","middle"}<=set(scope.replan_ids) and scope.actual_k<=12


def test_local_ordinary_expansion_rejects_future_release_with_same_two_call_budget(monkeypatch):
    s=snapshot(2); patch_nodes(monkeypatch,s,future={"1"})
    frame=SimpleNamespace(snapshot_id=s.version,effective_time=s.time,identity="f",policy="fixed",
        order=("0","1"),upstream_request_ids=(),objective_eligibility={},upstream_requests=())
    scope=PlanningScope(s.version,s.time,("0","1"),("0",),("1",),actual_k=1,objective_frame_id="f")
    class Provider:
        validation_cache={}
        def iter_progressive_columns(self,snapshot,budget,context,*,scope,early_publish):
            for _ in range(3):
                cols={aid:(PredictionColumn(aid,"KEEP",(),(),(0,0,0,0,0),PlanningNode("outer",int(aid),0.,.9,0.,0)),)
                      for aid in scope.replan_ids}
                yield CandidateCatalog(cols,{aid:(0,aid) for aid in cols},{},(),scope=scope)
    class Scheduler:
        def __init__(self): self.calls=[]
        def plan(self,snapshot,catalog,context,budget,*,incumbent):
            self.calls.append((id(budget),tuple(context.scope.replan_ids)))
            context.background_blockers.append(dict(source="0",target="1",snapshot_id=s.version,
                reason="RESOURCE_CONFLICT",resource=("outer",1)))
            return PlanningResult(tuple(rows[0] for rows in catalog.columns.values()),(),None,"UNKNOWN",(0,)*5)
    scheduler=Scheduler(); budget=PlanningBudget(candidate_seconds=6.,solver_seconds=8.,candidate_active_seconds=0.)
    local=LocalPlanner(Provider(),scheduler,executable_v3=True); local.plan(s,scope,frame,budget)
    assert len(scheduler.calls)==2 and len({r[0] for r in scheduler.calls})==1
    assert all(ids==("0",) for _,ids in scheduler.calls) and local.telemetry["scope_expansions"]==0
    assert local.telemetry["skipped_unexecutable_expansions"][0]["access"][0]["node_time"]==40.
    assert budget.solver_seconds==8. and budget.candidate_seconds==6.


def test_target_cursor_queries_different_free_pads_before_joint_target(monkeypatch):
    w=RotatingSlotWorld(); w.seed_hold("a","inner",0,.9); s=w.build_planning_snapshot()
    p=CandidateProvider(); node=PlanningNode("inner",0,0.,.9,0.,0); calls=[]
    s=replace(s,pads=(Reservation(1,"ground",0.,100.,committed=True),))
    monkeypatch.setattr(p,"_offset_cursors",lambda *args,**kwargs:iter({"target":i} for i in range(3)))
    monkeypatch.setattr(p,"_next_descriptor",lambda snapshot,a,node,budget,cursor,seen,immediate:(cursor["target"],0.,40.,0.))
    monkeypatch.setattr(p.references,"generate",lambda snapshot,a,node,target,*args:(calls.append(target) is None,None))
    state=dict(attempted_targets=set(),viable_targets=set(),screened_without_descriptor_targets=set())
    list(p._target_coverage_steps(s,s.aircraft["a"],node,PlanningBudget(),None,state))
    assert calls==[0,2,1] and len(calls)==len(set(calls))
    assert state["attempted_targets"]=={0,1,2}
    calls.clear(); state=dict(attempted_targets={0},viable_targets=set(),screened_without_descriptor_targets=set())
    list(p._target_coverage_steps(s,s.aircraft["a"],node,PlanningBudget(),None,state))
    assert calls==[2,1]  # A valid, already retained target is not queried again.


def test_target_coverage_reuses_existing_strict_immutable_pruning_without_losing_joint_release(monkeypatch):
    w=RotatingSlotWorld(); w.seed_hold("a","inner",0,.9); s=w.build_planning_snapshot()
    s=replace(s,pads=(Reservation(0,"ground",0.,100.,committed=True),
                      Reservation(1,"selected_owner",0.,float("inf"),committed=False)))
    p=CandidateProvider(); node=PlanningNode("inner",0,0.,.9,0.,0); materialized=[]
    budget=PlanningBudget(); budget._strict_resource_context=object()
    monkeypatch.setattr(p,"_offset_cursors",lambda *args,**kwargs:iter({"target":i} for i in range(2)))
    monkeypatch.setattr(p,"_next_descriptor",lambda snapshot,a,node,budget,cursor,seen,immediate:(cursor["target"],0.,40.,0.))
    monkeypatch.setattr(p.references,"_generate_unobserved",lambda snapshot,a,node,target,*args:(materialized.append(target) is None,None))
    state=dict(attempted_targets=set(),viable_targets=set(),screened_without_descriptor_targets=set())
    list(p._target_coverage_steps(s,s.aircraft["a"],node,budget,None,state))
    assert materialized==[1] and budget.screening["strict_immutable_resource_pruned"]==1


def test_progressive_target_two_then_full_delayed_families_are_fairly_interleaved(monkeypatch):
    s=snapshot(2); p=CandidateProvider(); events=[]
    budget=PlanningBudget(max_per_aircraft=32,max_attempts=1024,candidate_seconds=6.,soft_seconds=30.,candidate_active_seconds=0.)
    budget._selective_executable_v3=True
    monkeypatch.setattr(p,"_previous_chain",lambda *args:None)
    monkeypatch.setattr(p,"_column",lambda snapshot,a,plans,kind,context,ref=None,*args:
        PredictionColumn(a.id,kind,plans,(),(0,)*5,PlanningNode(a.ring,a.slot,0.,.9,0.,0)))
    monkeypatch.setattr(p,"_progressive_catalog",lambda snapshot,budget,context,columns,refs,ordered:
        CandidateCatalog({aid:tuple(rows) for aid,rows in columns.items()},{aid:(0,aid) for aid in columns},{},()))
    def coverage(snapshot,a,node,budget,context,state):
        state["viable_targets"].update(range(4))
        for target in range(4):
            events.append(("target",a.id,target)); budget.attempts[a.id]+=1
            state["attempted_targets"].add(target); yield ("ATTEMPT",None,None)
    def family(name):
        def steps(snapshot,a,node,budget,*args):
            for _ in range(5):
                events.append((name,a.id)); budget.attempts[a.id]+=1
                yield None if name=="reference" else ("ATTEMPT",None,None)
        return steps
    monkeypatch.setattr(p,"_target_coverage_steps",coverage)
    monkeypatch.setattr(p,"_progress_steps",family("progress"))
    monkeypatch.setattr(p,"_paths",family("alternative"))
    monkeypatch.setattr(p.references,"build_steps",family("reference"))
    # Legacy-profile fixture adds the delayed family explicitly for this test.
    p.planning_profile="downstream_service_v8"
    monkeypatch.setattr(p,"_delayed_steps",family("delayed"))
    context=EvaluationContext(s,budget=budget)
    generator=p.iter_progressive_columns(s,budget,context,early_publish=True)
    first=next(generator)
    assert events==[("target","0",0),("target","1",0)]
    second=next(generator); generator.close()
    assert events[:4]==[("target","0",0),("target","1",0),("target","0",1),("target","1",1)]
    assert any(row[0]=="reference" for row in events) and any(row[0]=="delayed" for row in events)
    assert not any(row[0] in ("progress","alternative") for row in events[:8])
    assert first.generation_coverage["target_attempt_coverage"]["0"]["remaining_unseen_targets"]==[1,2,3]
    assert second.generation_coverage["target_attempt_coverage"]["0"]["claim"]=="BOUNDED_DESCRIPTOR_QUERIES_NOT_FEASIBILITY"


def test_variant_is_explicit_and_full_h_uses_same_progressive_v3_entrypoint():
    cb=NormalSchedulerCallback(method="rh_cp_sat",planning_profile="downstream_service_v8",
        selective_method="FULL_H",selective_rule_variant=V3)
    assert cb.selective_rule_variant==V3 and cb.selective_history.progress_policy=="maneuver_started_v2"
    assert NormalSchedulerCallback().selective_rule_variant=="baseline"
