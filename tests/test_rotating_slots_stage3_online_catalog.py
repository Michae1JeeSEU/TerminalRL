"""Bounded progressive search and original-objective incumbent retention."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots import stage2_planning as sp
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.resources import Reservation
from terminal_marl.core.rotating_slots.stage3_learning.catalog_control import retain_validated,coverage_report
from terminal_marl.core.rotating_slots.trajectory import generate_final
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,evaluate_planning_snapshot


def fake_move(aid="a",pad=0,start=0.,duration=40.,tag="",objective=1.):
    p=SimpleNamespace(aircraft_id=aid,trajectory_hash=f"{aid}:{pad}:{start}:{duration}:{tag}",
        start_time=start,end_time=start+duration,target_ring=None,target_slot=None,pad_id=pad)
    return sp.PredictionColumn(aid,"MOVE",(p,),(Reservation(("pad",pad),aid,start,start+duration),),
        (0,-1,0,objective,0),sp.PlanningNode("inner",0,0.,.9,0.,0),sp.NormalReference("KNOWN"))


def test_validated_similarity_pruning_preserves_new_resource_opportunity():
    counts=sp.Counter(); first=fake_move(); columns=[first]
    assert not retain_validated(columns,fake_move(tag="duplicate"),0.,1,counts)
    assert counts["catalog_similar_pruned"]==1
    for c in (fake_move(pad=1),fake_move(start=10.),fake_move(objective=.5,tag="better")):
        assert retain_validated(columns,c,0.,1,counts)
    assert columns[0] is first and len(columns)==4
    # Unknown estimates do not establish the objective comparison.
    unknown=replace(fake_move(tag="unknown"),plans=(SimpleNamespace(
        **dict(vars(first.plans[0]),pad_id=None,trajectory_hash="partial",target_ring="inner",target_slot=0)),),reference=sp.NormalReference("UNKNOWN"))
    assert retain_validated(columns,unknown,0.,1,counts)


def test_generator_resumes_fair_per_aircraft_work_and_catalog_grows(monkeypatch):
    world=RotatingSlotWorld();world.seed_hold("a","inner",0,.9);world.seed_hold("b","inner",5,.9)
    snapshot=world.build_planning_snapshot(); provider=sp.CandidateProvider();budget=sp.PlanningBudget(candidate_seconds=100.,soft_seconds=100.,max_per_aircraft=32)
    context=sp.EvaluationContext(snapshot,budget=budget);calls=[];family_events=[]
    monkeypatch.setattr(context,"check_single",lambda c:ValidationResult(True,"SAFE"))
    def column(snapshot,a,plans,kind,context,reference=None,emergency="NONE",budget=None):
        if plans: return replace(fake_move(a.id,pad=plans[0].pad_id,start=plans[0].start_time,tag=plans[0].trajectory_hash),plans=plans)
        return sp.PredictionColumn(a.id,kind,(),(),(0,0,0,0,0),sp._node(snapshot,a),reference)
    monkeypatch.setattr(provider,"_column",column)
    monkeypatch.setattr(provider.references,"_reference",lambda *args:sp.NormalReference("KNOWN",energy_j=1.,soc_required=.1,peak_required=0.,duration=40.))
    def steps(snapshot,a,node,budget,context):
        for index in range(8):
            family_events.append((a.id,"progress"))
            calls.append((a.id,index));budget.attempts[a.id]+=1
            yield "COLUMN",fake_move(a.id,pad=index%6,start=10.*index,tag=str(index)).plans,"NONE"
    monkeypatch.setattr(provider,"_progress_steps",steps)
    monkeypatch.setattr(provider,"_paths",lambda *args,**kwargs:iter(()))
    def reference(snapshot,a,*args):
        family_events.append((a.id,"reference"))
        if False: yield None
        return sp.NormalReference("UNKNOWN")
    monkeypatch.setattr(provider.references,"build_steps",reference)
    gen=provider.iter_progressive_columns(snapshot,budget,context)
    first=next(gen); before=list(calls)
    assert before[:2]==[("a",0),("b",0)] and len(before)==4
    assert family_events[:2]==[("a","progress"),("b","progress")]
    assert all(len(rows)==4 for rows in first.columns.values())
    second=next(gen)
    assert calls[:len(before)]==before and len(calls)==8
    assert all({c.signature for c in first.columns[aid]}<={c.signature for c in second.columns[aid]} for aid in first.columns)
    third=next(gen); assert len(calls)==16
    assert all(len(rows)==10 for rows in third.columns.values())
    with pytest.raises(StopIteration): next(gen)


@pytest.mark.parametrize("complete,expected_rounds,rescore",((True,1,False),(False,3,False),(False,3,True)))
def test_callback_early_stop_or_expansion_keeps_best_and_one_budget(monkeypatch,complete,expected_rounds,rescore):
    world=RotatingSlotWorld();world.seed_hold("a","inner",0,.9)
    snapshot=world.build_planning_snapshot();provider=sp.CandidateProvider()
    context=sp.EvaluationContext(snapshot)
    plan=generate_final("a","inner",0,0,0.,config=snapshot.config).plan
    assert plan is not None
    keep=provider._column(snapshot,snapshot.aircraft["a"],(),"KEEP",context)
    move=provider._column(snapshot,snapshot.aircraft["a"],(plan,),"MOVE",context)
    rows=(keep,move) if complete else (keep,)
    catalog=sp.CandidateCatalog({"a":rows},{"a":(0,)},{},("a",))
    budgets=[]; produced=[]
    def batches(snapshot,budget,context):
        budgets.append(budget)
        for index in range(3): produced.append(index); yield catalog
    monkeypatch.setattr(provider,"iter_progressive_columns",batches)
    monkeypatch.setattr(sp,"_selection_key",lambda catalog,columns:(0,0,0,30. if rescore else 11.,0))
    callback=sp.NormalSchedulerCallback(candidate_provider=provider,online_policy="progressive_v1")
    calls=[]
    def plan_round(snapshot,catalog,context,budget):
        calls.append((id(budget),budget.solver_call_seconds,budget.solver_wall_used))
        budget.solver_wall_used+=min(budget.solver_call_seconds or 8.,8.-budget.solver_wall_used)
        selected=(move,) if complete else (keep,)
        prefix=(plan,) if complete else ()
        evaluation=evaluate_planning_snapshot(snapshot,prefix)
        return sp.PlanningResult(selected,prefix,evaluation,"FEASIBLE",(0,0,0,10.+len(calls),0),certified=True)
    monkeypatch.setattr(callback.scheduler,"plan",plan_round)
    callback(world)
    assert len(calls)==len(produced)==expected_rounds and len({row[0] for row in calls})==1
    assert budgets[0].settings()["candidate_seconds"]==6. and budgets[0].max_per_aircraft==32 and budgets[0].max_attempts==1024
    assert budgets[0].solver_wall_used<=8. and budgets[0].hint_seconds==3.
    assert callback.last_result.objective[3]==(13. if rescore else 11.)
    record=callback.records[-1]
    assert len(record["catalog_rounds"])==expected_rounds
    assert record["catalog_rounds"][-1]["cumulative_cp_wall_s"]<=8.
    assert record["catalog_rounds"][0]["coverage"]["a"]["coverage_claim"]=="observed_bounded_catalog_only"


def test_online_checkpoint_binding_and_default_legacy_contract():
    callback=sp.NormalSchedulerCallback(online_policy="progressive_v1")
    state=callback.state_dict(); assert state["online_policy"]=="progressive_v1"
    callback.load_state_dict(state)
    with pytest.raises(ValueError,match="online catalog"): sp.NormalSchedulerCallback().load_state_dict(state)
    assert "online_policy" not in sp.NormalSchedulerCallback().state_dict()


def test_active_candidate_clock_excludes_selection_but_obeys_original_global_soft(monkeypatch):
    now=[0.];monkeypatch.setattr(sp.time,"perf_counter",lambda:now[0])
    budget=sp.PlanningBudget.for_profile("quality_v5");budget.wall_start=0.
    budget.candidate_seconds=6.;budget.candidate_active_seconds=0.
    with budget.candidate_work(): now[0]=2.
    now[0]=12.  # Ten seconds outside candidate generation, on the same clock.
    assert budget.elapsed==12. and budget.candidate_elapsed==2. and budget.available()
    assert budget.candidate_seconds-budget.candidate_elapsed==4.
    assert budget.solver_seconds==8. and budget.hint_seconds==3.
    now[0]=30.
    assert not budget.available() and "SOFT_BUDGET" in budget.truncations


@pytest.mark.parametrize("status",("UNKNOWN","TIMEOUT","BUDGET_EXHAUSTED"))
def test_nondefinite_status_is_not_recorded_as_observed_conflict(status):
    world=RotatingSlotWorld();world.seed_hold("a","inner",0,.9)
    context=sp.EvaluationContext(world.build_planning_snapshot())
    context._observe(ValidationResult(False,status,"BUDGET"),(fake_move(),))
    assert not context.observed_definite_conflicts


def test_real_reused_full_chain_has_downstream_catalog_without_new_generation():
    world=RotatingSlotWorld();world.seed_hold("a","inner",0,.9)
    snapshot=world.build_planning_snapshot()
    final=generate_final("a","inner",0,0,0.,config=snapshot.config).plan
    assert final is not None
    provider=sp.CandidateProvider(planning_profile=sp.PLANNING_PROFILE_DOWNSTREAM)
    provider.previous={"a":(final,)}
    budget=sp.PlanningBudget(max_attempts=0)
    context=sp.EvaluationContext(snapshot,budget=budget)
    batches=provider.iter_progressive_columns(snapshot,budget,context)
    catalog=next(batches);batches.close()
    full=next(c for c in catalog.columns["a"] if c.plans)
    assert catalog.downstream is not None and not budget.attempts
    assert catalog.downstream.key((full,))[:2]==(0,-1)
    assert catalog.downstream.key((full,))[3]>0.
    assert coverage_report(snapshot,catalog.columns)["a"]["full_chain"]
    assert context.check_commit_prefix((full,)).accepted


def test_real_progressive_collection_joins_terminal_validation_and_v2_context():
    from terminal_marl.core.rotating_slots.stage3_learning.runtime import AdvisorRuntime,scale_candidate_budget
    from terminal_marl.core.rotating_slots.stage3_learning.features import FeatureEncoder,SCHEMA_V2
    from terminal_marl.core.rotating_slots.stage3_learning.labeling import complete_cost_facts
    world=RotatingSlotWorld();world.seed_hold("a","inner",0,.9)
    snapshot=world.build_planning_snapshot()
    runtime=AdvisorRuntime("RECORD",feature_encoder=FeatureEncoder(schema_id=SCHEMA_V2),
        feature_schema=SCHEMA_V2,trace_mode="collection")
    provider=sp.CandidateProvider(planning_profile=sp.PLANNING_PROFILE_DOWNSTREAM,advisor_runtime=runtime)
    budget=sp.PlanningBudget.for_profile("quality_v5");scale_candidate_budget(budget,.5)
    budget.candidate_active_seconds=0.
    context=sp.EvaluationContext(snapshot,budget=budget,reuse_prefix_results=True)
    batches=provider.iter_progressive_columns(snapshot,budget,context)
    with budget.candidate_work(): catalog=next(batches)
    batches.close()
    moves=[c for c in catalog.columns["a"] if c.kind=="MOVE"]
    assert moves
    features={r["features_before_query_ref"]:r for r in runtime.traces if r["record_type"]=="FeaturesBeforeQuery"}
    validations=[r for r in runtime.traces if r["record_type"]=="ColumnValidationCost"]
    assert validations and all(r["accepted"] for r in validations)
    for row in validations:
        query=features[row["features_before_query_ref"]]
        assert query["descriptor_id"]==row["descriptor_id"]
        feature=query["features"]
        assert not feature.get("missing") and feature["schema_id"]==SCHEMA_V2
        assert len(feature["query"])==66 and feature["query"][51:60:2]==[1.]*5
        assert "a" in query["search_state"]["retained_context"]
    ancestry=[r for r in runtime.traces if r["record_type"]=="ColumnAncestry" and r.get("retained")]
    terminal_ids={runtime._plans[c.plans[-1].trajectory_hash][-1] for c in moves}
    assert terminal_ids<={r["descriptor_id"] for r in validations}
    assert all(r["descriptor_id"] in terminal_ids for r in validations)
    assert ancestry
    complete=[fact for fact in complete_cost_facts(runtime.traces)
        if fact["trajectory_hash"] is not None and fact["cost_complete"]]
    assert complete, [(r["record_type"],r.get("descriptor_id"),r.get("features_before_query_ref"),r.get("wall_s"))
        for r in runtime.traces if r["record_type"] in ("FactualResult","SingleVehicleEvidence","ColumnValidationCost")]
    joined_refs={r["features_before_query_ref"] for r in validations}
    assert all(fact["features_before_query_ref"] in joined_refs and fact["column_validation_wall_s"]>=0. for fact in complete)
    cached=[r for r in runtime.traces if r["record_type"]=="SingleVehicleEvidence" and r.get("energy_cache_hit")]
    assert cached and all(isinstance(r["wall_s"],float) and r["wall_s"]>=0. and r["cpu_s"]>=0. for r in cached)
    ref=complete[0]["features_before_query_ref"]
    masked=[dict(r,wall_s=None) if r["record_type"]=="SingleVehicleEvidence" and r.get("features_before_query_ref")==ref
        and r.get("energy_cache_hit") else dict(r) for r in runtime.traces]
    assert not next(f for f in complete_cost_facts(masked) if f["features_before_query_ref"]==ref)["cost_complete"]


@pytest.mark.parametrize("v2",(True,False))
def test_reuse_observation_has_no_v2_query_attribution_and_keeps_successor(v2):
    from terminal_marl.core.rotating_slots.stage3_learning.runtime import AdvisorRuntime
    from terminal_marl.core.rotating_slots.stage3_learning.features import SCHEMA_ID,SCHEMA_V2
    from terminal_marl.core.rotating_slots.stage3_learning.labeling import complete_cost_facts
    world=RotatingSlotWorld();world.seed_hold("a","outer",0,.9)
    snapshot=world.build_planning_snapshot();a=snapshot.aircraft["a"];node=sp._node(snapshot,a)
    runtime=AdvisorRuntime("RECORD",feature_schema=SCHEMA_V2 if v2 else SCHEMA_ID)
    budget=sp.PlanningBudget();runtime.begin_snapshot(snapshot,budget,["a"])
    plan=SimpleNamespace(trajectory_hash="observed-cached-transfer",source_ring="outer",source_slot=0,
        source_version=0,target_ring="middle",target_slot=0,pad_id=None,start_time=0.,end_time=40.)
    descriptor=runtime.descriptor(snapshot,a,node,0,0.,40.,"alternative")
    token=runtime.record_before(snapshot,descriptor,budget,{})
    runtime.record_after(descriptor,token,True,plan,budget,{})
    q=(100.,.01,0.,.1,.01)
    runtime.record_energy(node,plan,q,True,.001,.001)
    c=sp.PredictionColumn("a","MOVE",(plan,),(),(0,0,0,0,0))
    runtime.record_column_validation(c,ValidationResult(True,"SAFE"),.001,.001,False)
    prior_ref=runtime._query_refs[descriptor.descriptor_id]
    assert complete_cost_facts(runtime.traces)[0]["cost_complete"]
    runtime.register_reuse(snapshot,a,node,plan,(),q)
    last=next(r for r in reversed(runtime.traces) if r["record_type"]=="SingleVehicleEvidence")
    assert last["descriptor_id"]==descriptor.descriptor_id and last["wall_s"] is None
    assert last["features_before_query_ref"]==(None if v2 else prior_ref)
    assert runtime._successor_states[plan.trajectory_hash]==("middle",0,40.,.89,0.,1,.1,.01)
    assert complete_cost_facts(runtime.traces)[0]["cost_complete"] is v2
