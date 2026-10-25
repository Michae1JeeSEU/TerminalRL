"""Prefix reuse is local to immutable visible inputs and never authorizes commit."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots import stage2_planning as sp
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.trajectory import generate_final
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld


def setup():
    world=RotatingSlotWorld(); world.seed_hold("a","inner",0,.9)
    return world,world.build_planning_snapshot()


def column(prefix="first",tail="tail"):
    plans=tuple(SimpleNamespace(trajectory_hash=h,start_time=0.,aircraft_id="a") for h in (prefix,tail))
    return sp.PredictionColumn("a","MOVE",plans,(),(0,0,0,0,0))


def test_same_prefix_different_prediction_tail_reuses_only_with_identity():
    _,snapshot=setup(); calls=[]
    evaluate=lambda plans: calls.append(plans) or ValidationResult(True,"SAFE")
    context=sp.EvaluationContext(snapshot,prefix_evaluator=evaluate,reuse_prefix_results=True,prefix_evaluator_identity="immutable-test-v1")
    assert context.check_commit_prefix((column(tail="x"),)).accepted
    assert context.check_commit_prefix((column(tail="y"),)).accepted
    assert len(calls)==1 and context.cache_stats["prefix_hits"]==1
    context.check_commit_prefix((column(prefix="other"),))
    assert len(calls)==2
    opaque=sp.EvaluationContext(snapshot,prefix_evaluator=evaluate,reuse_prefix_results=True)
    opaque.check_commit_prefix((column(),)); opaque.check_commit_prefix((column(),))
    assert len(calls)==4


@pytest.mark.parametrize("change",("soc","state","ledger","dependency","same"))
def test_new_context_cannot_reuse_prior_results(change):
    _,snapshot=setup(); cache={}; calls=[]
    def evaluate(plans): calls.append(plans); return ValidationResult(True,"SAFE")
    def context(s): return sp.EvaluationContext(s,evaluate,validation_cache=cache,reuse_prefix_results=True,prefix_evaluator_identity="v1")
    first=context(snapshot); first.check_commit_prefix((column(),))
    if change=="soc": snapshot=replace(snapshot,aircraft={"a":replace(snapshot.aircraft["a"],soc=.8)})
    elif change=="state": snapshot=replace(snapshot,aircraft={"a":replace(snapshot.aircraft["a"],version=1)})
    elif change=="ledger": snapshot=replace(snapshot,energy_ledger={"executed":{"a":({"error_j":1.,"peak_error_s":0.},)},"fine_plans":{}})
    elif change=="dependency": snapshot=replace(snapshot,dependencies={"p":("support",)})
    second=context(snapshot); second.check_commit_prefix((column(),))
    assert len(calls)==2 and second.cache_stats["prefix_hits"]==0
    assert second.cache_stats["prefix_invalidations"]==1
    if change!="same": assert second.snapshot_key!=first.snapshot_key


@pytest.mark.parametrize("accepted,status,refinement",((False,"UNKNOWN",False),(False,"REJECTED",False),
    (True,"UNKNOWN",False),(True,"INVALID",False),(True,"ACCEPTED",True)))
def test_uncertain_failed_or_refining_results_are_never_cached(accepted,status,refinement):
    _,snapshot=setup(); calls=[]
    def evaluate(plans):
        calls.append(plans)
        return SimpleNamespace(accepted=accepted,status=status,reason="TEST",refinement_required=refinement)
    context=sp.EvaluationContext(snapshot,evaluate,reuse_prefix_results=True,prefix_evaluator_identity="v1")
    context.check_commit_prefix((column(),)); context.check_commit_prefix((column(),))
    assert len(calls)==2 and not context._prefix_results


def test_budget_is_checked_before_cache_hit():
    _,snapshot=setup(); budget=sp.PlanningBudget(); available=[True]
    budget.evaluation_available=lambda:available[0]
    context=sp.EvaluationContext(snapshot,lambda plans:ValidationResult(True,"SAFE"),budget,
        reuse_prefix_results=True,prefix_evaluator_identity="v1")
    context.check_commit_prefix((column(),)); available[0]=False
    assert context.check_commit_prefix((column(),)).status=="UNKNOWN"
    assert context.cache_stats["prefix_hits"]==0


def test_cached_evaluation_still_requires_fresh_authoritative_commit():
    world,snapshot=setup()
    plan=generate_final("a","inner",0,0,0.,config=snapshot.config).plan
    assert plan is not None
    c=sp.PredictionColumn("a","MOVE",(plan,),(),(0,0,0,0,0))
    context=sp.EvaluationContext(snapshot,reuse_prefix_results=True)
    result=context.check_commit_prefix((c,))
    assert result.accepted and context.check_commit_prefix((c,)) is result
    world.aircraft["a"]=replace(world.aircraft["a"],soc=.8)
    assert world.commit_validated_prefix(result).reason=="STALE_PLANNING_SNAPSHOT"
