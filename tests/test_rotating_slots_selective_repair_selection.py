"""Frozen v2 selection contracts; no campaign or new experimental protocol."""
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, EvaluationContext, NormalReference, _node
from terminal_marl.core.rotating_slots.planning_latency import PlanningTiming,project_visible_snapshot
from terminal_marl.core.rotating_slots.selective_replanning import PlanningHistory,RuleSelector,build_global_frame
from terminal_marl.core.rotating_slots.selective_replanning.near_term import summarize_opportunities,cooperative_seed_groups
from terminal_marl.core.rotating_slots.selective_replanning.relations import Relation,build_relations
from terminal_marl.core.rotating_slots.trajectory import generate_transfer


def fixture(n=19):
    w=RotatingSlotWorld()
    for i in range(n): w.seed_hold(str(i),"outer",i,.9)
    s=w.build_planning_snapshot(); p=CandidateProvider(planning_profile="downstream_service_v8")
    frame=build_global_frame(s,p,EvaluationContext(s)); h=PlanningHistory(progress_policy="maneuver_started_v2"); h.observe(s)
    return w,s,p,frame,h


def test_age_over_60_is_debt_not_overflow_and_background_preserved():
    w,s,p,f,h=fixture()
    scope,info=RuleSelector(variant="opportunity_cooperative_v2").select(s,f,h,now=68.5,references=p.references)
    assert scope.actual_k==8 and not scope.mandatory_ids and scope.status=="LOCAL"
    assert info["fairness_overdue_count"]==19 and len(info["fairness_selected_ids"])==2
    assert set(scope.replan_ids)|set(scope.fixed_background_ids)==set(s.aircraft)
    assert len(scope.fixed_background_ids)==11
    legacy,_=RuleSelector().select(s,f,h,now=68.5)
    assert legacy.status=="MANDATORY_OVERFLOW_FULL" and len(legacy.mandatory_ids)==19


@pytest.mark.parametrize("holding_cost_factor",[.5,1.5])
def test_twenty_second_wait_is_projected_once_and_protects_urgent(holding_cost_factor):
    w,s,p,f,h=fixture(1)
    from terminal_marl.core.rotating_slots.energy import stable_power
    from terminal_marl.core.rotating_slots.rescue_rules import normal_energy_lower_bound
    cost=stable_power("outer",s.config).current_a*20/(s.config.energy.reference_capacity_Ah*3600)
    soc=.25+normal_energy_lower_bound(s.config,"outer")+cost*holding_cost_factor
    s=replace(s,aircraft={"0":replace(s.aircraft["0"],soc=soc)})
    future=project_visible_snapshot(s,20.,timing=PlanningTiming.from_lead(0.,20.,1.))
    scope,info=RuleSelector(variant="opportunity_cooperative_v2").select(future,SimpleNamespace(identity="fixed"),h)
    mask=info["energy_masks"]["0"]
    assert scope.mandatory_ids==("0",) and mask["urgent_evidence"]
    assert mask["next_opportunity_wait_seconds"]==40.
    assert mask["current_adjustment_time"]==20. and mask["deferred_adjustment_time"]==40.
    assert mask["additional_uncharged_wait_soc"]==pytest.approx(cost)
    assert mask["margin_soc"]==pytest.approx(soc-cost-.25-normal_energy_lower_bound(s.config,"outer"),abs=1.e-7)
    assert mask["can_safely_postpone"] is None and not mask["known"]
    if holding_cost_factor>1: assert mask["margin_soc"]>0


def test_approximate_competition_never_mandatory_and_definite_closure_can_expand():
    w,s,p,f,h=fixture(13)
    approximate=tuple(Relation("0",str(i),"virtual_pad_competition") for i in range(1,13))
    scope,info=RuleSelector(variant="opportunity_cooperative_v2").select(s,f,h,approximate)
    assert scope.actual_k==8 and not scope.mandatory_ids
    exact=(Relation("0","9","committed_release",True),)
    scope,_=RuleSelector(variant="opportunity_cooperative_v2").select(s,f,h,exact)
    assert {"0","9"}<=set(scope.replan_ids) and scope.actual_k<=12


def test_cached_specific_slot_cooperative_release_chain_is_jointly_selected():
    w=RotatingSlotWorld(); w.seed_hold("a","outer",1,.9); w.seed_hold("b","middle",1,.9)
    s=w.build_planning_snapshot(); node=_node(s,s.aircraft["a"])
    plan=generate_transfer("a","outer",1,1,0.,durations=(40.,)).plan
    assert plan is not None
    key=(s.config.config_hash,node.ring,node.slot,node.time,node.version,"a")
    refs=SimpleNamespace(cache={key:NormalReference("KNOWN",(plan,))})
    opportunities=summarize_opportunities(s,s.aircraft,refs)
    assert opportunities["a"]["joint_members"]==("b",)
    assert opportunities["a"]["resource"]==("middle",1)
    assert ("a","b") in cooperative_seed_groups(opportunities)
    h=PlanningHistory(progress_policy="maneuver_started_v2"); h.observe(s)
    scope,info=RuleSelector(variant="opportunity_cooperative_v2").select(s,SimpleNamespace(identity="f"),h,references=refs)
    assert {"a","b"}<=set(scope.replan_ids)
    assert all(not row["certified"] for row in opportunities["a"]["options"])


def test_many_inner_singletons_do_not_displace_specific_inner_middle_outer_chain():
    w=RotatingSlotWorld()
    for i in range(8): w.seed_hold("inner"+str(i),"inner",i,.9)
    w.seed_hold("middle","middle",1,.9); w.seed_hold("outer","outer",1,.9)
    s=w.build_planning_snapshot(); cache={}
    for aid,ring,slot,target in (("middle","middle",1,0),("outer","outer",1,1)):
        node=_node(s,s.aircraft[aid]); plan=generate_transfer(aid,ring,slot,target,0.,durations=(40.,)).plan
        assert plan is not None
        key=(s.config.config_hash,node.ring,node.slot,node.time,node.version,aid)
        cache[key]=NormalReference("KNOWN",(plan,))
    refs=SimpleNamespace(cache=cache)
    h=PlanningHistory(progress_policy="maneuver_started_v2"); h.observe(s)
    scope,info=RuleSelector(variant="opportunity_cooperative_v2").select(s,SimpleNamespace(identity="f"),h,references=refs)
    chain={"inner0","middle","outer"}
    assert any(chain<=set(group["members"]) for group in info["cooperative_groups"])
    assert chain<=set(scope.replan_ids) and scope.actual_k==8


def test_required_targets_are_hard_only_when_reached_from_energy_hard():
    w,s,p,f,h=fixture(13)
    dependency=(Relation("12","11","validated_background_blocker",True,True,("middle",0)),)
    scope,info=RuleSelector(variant="opportunity_cooperative_v2").select(s,f,h,dependency)
    assert not scope.mandatory_ids and info["hard_mandatory_count"]==0
    changed=replace(s,aircraft={aid:replace(a,soc=.249) if aid=="12" else a for aid,a in s.aircraft.items()})
    scope,info=RuleSelector(variant="opportunity_cooperative_v2").select(changed,SimpleNamespace(identity="f"),h,dependency)
    assert set(scope.mandatory_ids)=={"12","11"}


def test_follow_tail_cannot_supply_near_term_opportunity():
    w,s,p,f,h=fixture(1)
    poison=SimpleNamespace(identity="unchanged",baseline_by_id=property(lambda _: (_ for _ in ()).throw(AssertionError("FOLLOW tail read"))))
    scope,info=RuleSelector(variant="opportunity_cooperative_v2").select(s,poison,h)
    assert info["near_term_opportunities"]["0"]["source"]!="FOLLOW"
    assert scope.actual_k==1


def test_history_unknown_review_and_progress_require_delivery_and_actual_event():
    w,s,p,f,h=fixture(1)
    h.queue_reviews("unknown",s,("0",),unknown_ids=("0",))
    assert not h.deliver("unknown",s,10.) and h.entries["0"].reviewed is None
    h.queue_reviews("valid",s,("0",)); assert h.deliver("valid",s,11.)==("0",)
    q=generate_transfer("0","outer",0,0,12.,durations=(40.,)).plan
    projected=replace(s,time=13.,active={"0":q})
    h.observe(projected)
    assert h.entries["0"].progressed is None
    event=dict(type="maneuver_started",time=12.,aircraft_id="0",details=dict(maneuver_kind="TRANSFER",trajectory_hash=q.trajectory_hash))
    h.observe(projected,events=dict(cursor=9,events=(event,)))
    assert h.entries["0"].progressed==12.
    state=json.loads(json.dumps(h.state_dict())); restored=PlanningHistory(progress_policy="maneuver_started_v2"); restored.load_state_dict(state)
    assert restored.event_cursor==9 and restored.entries==h.entries
    future=dict(event,time=20.); restored.observe(projected,events=(future,))
    assert restored.entries["0"].progressed==12.


def test_stale_or_unowned_required_resource_is_not_reused():
    w,s,p,f,h=fixture(2)
    rows=build_relations(s,("0","1"),[dict(snapshot_id=s.version,source="0",target="1",resource=("middle",1),reason="RESOURCE_CONFLICT")],variant="opportunity_cooperative_v2")
    assert not any(r.required for r in rows)


def test_hard_overflow_is_explicit_and_unknown_age_is_bounded():
    w,s,p,f,h=fixture(13)
    s=replace(s,aircraft={aid:replace(a,soc=.249) for aid,a in s.aircraft.items()})
    scope,info=RuleSelector(variant="opportunity_cooperative_v2").select(s,SimpleNamespace(identity="f"),h)
    assert len(scope.mandatory_ids)==13 and scope.actual_k==13
    assert info["overflow_cause"]=="HARD_MANDATORY_GT_MAX12"
    _,s,_,f,h=fixture(13); h.initialize_imported(s)
    scope,info=RuleSelector(variant="opportunity_cooperative_v2").select(s,f,h)
    assert len(info["unknown_review_age_ids"])==13 and len(info["fairness_selected_ids"])<=2
    assert not scope.mandatory_ids
