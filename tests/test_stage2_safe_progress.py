"""Independent safe progress, bounded coverage, and explicit tail heuristic v2."""
from collections import Counter

import pytest

from terminal_marl.core.rotating_slots.config import load_service_profile, service_duration
from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider, EnergySlackScheduler, EvaluationContext, NormalReference,
    PlanningBudget, ReferenceProvider, _node,
)
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, evaluate_planning_snapshot


def world_with_holds(*rows, config=None):
    world=RotatingSlotWorld(config=config)
    world.configure_practical_runtime(True)
    for aid,ring,slot in rows:
        world.seed_hold(aid,ring,slot,.9)
    return world


def test_safe_transfer_survives_its_full_seed_blocked_final(monkeypatch):
    monkeypatch.setattr(PlanningBudget,"elapsed",property(lambda self:0.))
    world=world_with_holds(("o","outer",0))
    snapshot=world.build_planning_snapshot()
    references=ReferenceProvider()
    reference=references.get(snapshot,snapshot.aircraft["o"],budget=PlanningBudget())
    assert reference.status=="KNOWN" and len(reference.plans)==3
    world.seed_service("busy",reference.plans[-1].pad_id,remaining=500.)
    snapshot=world.build_planning_snapshot();context=EvaluationContext(snapshot)
    provider=CandidateProvider(references)
    # Isolate the new progress pass and the original complete seed.
    monkeypatch.setattr(provider,"_paths",lambda *args,**kwargs:iter(()))
    budget=PlanningBudget();catalog=provider.get_columns(snapshot,budget,context)
    full=provider._column(snapshot,snapshot.aircraft["o"],reference.plans,"MOVE",context)
    assert not context.check_single(full).accepted
    assert context.check_single(full).reason=="RESOURCE_CONFLICT"
    partial=[c for c in catalog.columns["o"] if c.kind=="MOVE" and c.plans[-1].pad_id is None]
    assert partial and budget.screening["progress_retained"]==1
    assert all(evaluate_planning_snapshot(snapshot,(c.plans[0],)).accepted for c in partial)
    result=EnergySlackScheduler().plan(snapshot,catalog,context,PlanningBudget())
    assert result.certified and result.prefix and result.prefix[0].target_ring=="middle"
    assert world.time==0 and not world.active


def test_progress_round_robin_covers_each_ring_without_removing_full_chains(monkeypatch):
    monkeypatch.setattr(PlanningBudget,"elapsed",property(lambda self:0.))
    world=world_with_holds(("o","outer",0),("m","middle",9),("i","inner",17))
    snapshot=world.build_planning_snapshot();provider=CandidateProvider()
    budget=PlanningBudget();context=EvaluationContext(snapshot)
    catalog=provider.get_columns(snapshot,budget,context)
    assert budget.screening["progress_retained"]==3
    for aid,depth in (("o",3),("m",2),("i",1)):
        moves=[c for c in catalog.columns[aid] if c.kind=="MOVE"]
        assert any(len(c.plans)==1 and c.plans[0].start_time==0 for c in moves)
        assert any(len(c.plans)==depth and c.plans[-1].pad_id is not None for c in moves)
        assert len(moves)<=8 and max(Counter(c.first_target for c in moves).values())<=2
        assert budget.attempts[aid]<=64
    assert sum(budget.attempts.values())<=2048


def test_unknown_partial_uses_terminal_ring_lower_bound_and_never_queries_tail(monkeypatch):
    world=world_with_holds(("o","outer",0));snapshot=world.build_planning_snapshot()
    provider=CandidateProvider();context=EvaluationContext(snapshot);a=snapshot.aircraft["o"]
    plan=generate_transfer("o","outer",0,1,0.,config=snapshot.config,durations=(40,)).plan
    assert plan is not None and evaluate_planning_snapshot(snapshot,(plan,)).accepted
    monkeypatch.setattr(provider.references,"get",lambda *args,**kwargs:pytest.fail("implicit tail search"))
    keep=provider._column(snapshot,a,(),"KEEP",context,budget=PlanningBudget())
    move=provider._column(snapshot,a,(plan,),"MOVE",context,budget=PlanningBudget())
    assert keep.reference.status==move.reference.status=="UNKNOWN"
    assert keep.reference.soc_required is move.reference.soc_required is None
    assert keep.objective[:3]==move.objective[:3]==(0,0,3)
    assert 900<move.objective[3]<keep.objective[3]
    assert move.objective[3]==901+provider._remaining_lower(snapshot,"middle")
    assert move.objective<keep.objective and move.objective[1]==0


def test_complete_normal_landing_score_is_unchanged():
    world=world_with_holds(("i","inner",0));snapshot=world.build_planning_snapshot()
    a=snapshot.aircraft["i"];provider=CandidateProvider();context=EvaluationContext(snapshot)
    plan=generate_final("i","inner",0,5,0.,config=snapshot.config,durations=(35,)).plan
    assert plan is not None
    column=provider._column(snapshot,a,(plan,),"MOVE",context)
    assert column.objective[:4]==(0,-1,0,plan.end_time)


def test_exact_known_tail_cache_keeps_original_bounded_tau():
    world=world_with_holds(("o","outer",0));snapshot=world.build_planning_snapshot()
    a=snapshot.aircraft["o"];provider=CandidateProvider();context=EvaluationContext(snapshot)
    node=_node(snapshot,a)
    key=(snapshot.config.config_hash,node.ring,node.slot,240.,node.version,a.id)
    ref=NormalReference("KNOWN",soc_required=.1,peak_required=1.,duration=123.5,energy_j=1.,identity=key)
    provider.references.cache[key]=ref
    column=provider._column(snapshot,a,(),"KEEP",context)
    assert column.reference==ref and column.objective[3]==363.5<=900.


def test_progress_records_predicted_transfer_terminal_risk(monkeypatch):
    world=world_with_holds(("o","outer",0));snapshot=world.build_planning_snapshot()
    a=snapshot.aircraft["o"];provider=CandidateProvider();context=EvaluationContext(snapshot)
    recorded=[]
    def prediction(snapshot,aircraft,node):
        recorded.append(node)
        return "UNKNOWN"
    monkeypatch.setattr(provider,"_emergency_prediction",prediction)
    for tag,plans,emergency in provider._progress_steps(snapshot,a,_node(snapshot,a),PlanningBudget(),context):
        if tag=="COLUMN": break
    assert emergency=="UNKNOWN" and recorded
    assert (recorded[-1].ring,recorded[-1].slot,recorded[-1].time,recorded[-1].version)==(
        plans[0].target_ring,plans[0].target_slot,plans[0].end_time,a.version+1)
    assert recorded[-1].soc<a.soc


def test_path_prefix_caps_do_not_stop_expansion_to_complete_chain(monkeypatch):
    monkeypatch.setattr(PlanningBudget,"elapsed",property(lambda self:0.))
    world=world_with_holds(("o","outer",0));snapshot=world.build_planning_snapshot()
    a=snapshot.aircraft["o"];provider=CandidateProvider();context=EvaluationContext(snapshot)
    budget=PlanningBudget();counts=Counter();accepted=[]
    provider._aligned_by_node={}
    for tag,plans,_ in provider._paths(snapshot,a,_node(snapshot,a),budget,context,accepted_counts=counts):
        if tag!="COLUMN": continue
        c=provider._column(snapshot,a,plans,"MOVE",context)
        if c is None or not context.check_single(c).accepted: continue
        counts[c.first_target]=min(2,counts[c.first_target]+1)
        accepted.append(c)
        if plans[-1].pad_id is not None: break
    assert any(len(c.plans)==1 for c in accepted)
    assert any(len(c.plans)==2 for c in accepted)
    assert accepted[-1].plans[-1].pad_id is not None and len(accepted[-1].plans)==3


def test_progress_skips_occupied_slot_but_joint_enumeration_can_release_it():
    world=world_with_holds(("o","outer",1),("m","middle",1))
    snapshot=world.build_planning_snapshot();context=EvaluationContext(snapshot)
    provider=CandidateProvider();a=snapshot.aircraft["o"];node=_node(snapshot,a)
    assert provider._current_progress_reservation_conflict(snapshot,a,node,1,5.,40.)
    outer=generate_transfer("o","outer",1,1,5.,config=snapshot.config,durations=(40,)).plan
    middle=generate_transfer("m","middle",1,0,0.,config=snapshot.config,durations=(40,)).plan
    oc=provider._column(snapshot,a,(outer,),"MOVE",context)
    mc=provider._column(snapshot,snapshot.aircraft["m"],(middle,),"MOVE",context)
    assert context.check_pair(oc,mc).accepted and context.check_commit_prefix((oc,mc)).accepted


def test_progress_pass_does_not_spend_on_far_future_active_endpoint(monkeypatch):
    world=world_with_holds(("o","outer",0))
    plan=generate_transfer("o","outer",0,1,0.,config=world.config,durations=(40,)).plan
    assert world.submit((plan,)).accepted
    snapshot=world.build_planning_snapshot();provider=CandidateProvider()
    def forbidden(*args,**kwargs):
        pytest.fail("future active endpoint used immediate-progress budget")
    monkeypatch.setattr(provider,"_progress_steps",forbidden)
    catalog=provider.get_columns(snapshot,PlanningBudget(max_attempts=0))
    assert catalog.columns["o"] and snapshot.active["o"].end_time==40.


def test_two_prefix_slots_retain_useful_partial_and_valid_complete_chain(monkeypatch):
    monkeypatch.setattr(PlanningBudget,"elapsed",property(lambda self:0.))
    world=world_with_holds(("o","outer",0));snapshot=world.build_planning_snapshot()
    a=snapshot.aircraft["o"];provider=CandidateProvider()
    reference=provider.references.get(snapshot,a,budget=PlanningBudget())
    assert len(reference.plans)==3
    # Existing depth1 plus progress depth2 exhaust the first-target allowance;
    # checked replacement retains the useful depth2 partial with the full seed.
    provider.previous["o"]=reference.plans[:1]
    monkeypatch.setattr(provider,"_progress_steps",lambda *args,**kwargs:iter((("COLUMN",reference.plans[:2],"NONE"),)))
    monkeypatch.setattr(provider,"_paths",lambda *args,**kwargs:iter(()))
    catalog=provider.get_columns(snapshot,PlanningBudget())
    moving=[c for c in catalog.columns["o"] if c.kind=="MOVE"]
    assert len(moving)==2 and all(c.first_target==moving[0].first_target for c in moving)
    assert any(len(c.plans)==2 and c.plans[-1].pad_id is None for c in moving)
    assert any(len(c.plans)==3 and c.plans[-1].pad_id is not None for c in moving)


@pytest.mark.parametrize("profile",("baseline","service_half_v1"))
def test_prediction_and_executor_use_same_config_service_end(profile):
    config=load_service_profile(profile)
    world=world_with_holds(("i","inner",0),config=config)
    snapshot=world.build_planning_snapshot();a=snapshot.aircraft["i"]
    plan=generate_final("i","inner",0,5,0.,config=config,durations=(35,)).plan
    assert plan is not None
    context=EvaluationContext(snapshot)
    column=CandidateProvider()._column(snapshot,a,(plan,),"MOVE",context)
    predicted=next(r for r in column.resources if r.resource==("pad",5))
    expected=plan.end_time+service_duration(config,a.service_class)
    assert predicted.end==expected==context.traffic(column).end_time
    assert world.submit((plan,)).accepted
    actual=next(r for r in world.pads.reservations if r.aircraft_id=="i")
    assert (actual.start,actual.end)==(predicted.start,predicted.end)
