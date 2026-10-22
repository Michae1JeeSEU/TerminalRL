"""Bounded v6 candidate reuse and conflict collection regressions."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots import stage2_planning as p
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.resources import Reservation
from terminal_marl.core.rotating_slots.rescue_rules import RESCUE_PROFILE_V6
from terminal_marl.core.rotating_slots.trajectory import generate_transfer, generate_final
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld


def snapshot():
    world=RotatingSlotWorld();world.seed_hold("a","outer",0,.9)
    return world.build_planning_snapshot()


def discrete(aid,target=0,start=0.,full=False,tag=""):
    plan=SimpleNamespace(trajectory_hash=f"{aid}-{target}-{start}-{full}-{tag}",
        target_ring=None if full else "middle",target_slot=None if full else target,
        pad_id=target if full else None,start_time=start,end_time=start+40.,kind="FINAL_NORMAL" if full else "TRANSFER")
    return p.PredictionColumn(aid,"MOVE",(plan,),(),(0,-int(full),3,40.,0.))


def test_batch_collects_multiple_explicit_conflicts_in_one_pass(monkeypatch):
    context=p.EvaluationContext(snapshot())
    columns=tuple(discrete(a) for a in ("a","b","c"))
    monkeypatch.setattr(context,"check_single",lambda c:ValidationResult(True,"SAFE"))
    monkeypatch.setattr(context,"check_pair",lambda a,b:ValidationResult(False,"CONFLICT","PAIR"))
    result,conflicts=context.check_batch_conflicts(columns)
    assert result.status=="CONFLICT" and len(conflicts)==3
    assert {tuple(c.aircraft_id for c in row) for row in conflicts}=={("a","b"),("a","c"),("b","c")}
    assert context.cache_stats["batch_conflicts"]==3
    assert len(context.check_batch(columns)[1])==2


def test_batch_collects_single_and_pair_conflicts_and_is_bounded(monkeypatch):
    context=p.EvaluationContext(snapshot());columns=tuple(discrete(str(i)) for i in range(9))
    monkeypatch.setattr(context,"check_single",lambda c:ValidationResult(False,"REJECTED","SINGLE"))
    monkeypatch.setattr(context,"check_pair",lambda a,b:ValidationResult(False,"CONFLICT","PAIR"))
    result,conflicts=context.check_batch_conflicts(columns)
    assert not result.accepted and len(conflicts)==p.MAX_BATCH_CONFLICTS==16
    assert sum(len(row)==1 for row in conflicts)==9


def test_unknown_produces_no_explicit_conflict_and_is_not_cached(monkeypatch):
    context=p.EvaluationContext(snapshot());columns=(discrete("a"),discrete("b"))
    monkeypatch.setattr(context,"check_single",lambda c:ValidationResult(True,"SAFE"))
    calls=[]
    def unknown(*args):
        calls.append(args);return ValidationResult(False,"UNKNOWN","NUMERIC_BOUND")
    monkeypatch.setattr(p,"validate_separation",unknown)
    monkeypatch.setattr(context,"traffic",lambda c:SimpleNamespace(plans=(),end_time=float("inf")))
    result,conflicts=context.check_batch_conflicts(columns)
    assert result.status=="UNKNOWN" and not conflicts and not context.pairs
    assert context.check_pair(*columns).status=="UNKNOWN" and len(calls)==2


def test_single_unknown_is_retried_and_definite_results_reused(monkeypatch):
    context=p.EvaluationContext(snapshot());column=discrete("a");calls=[]
    def check(c):
        calls.append(c)
        return ValidationResult(False,"UNKNOWN","BOUND") if len(calls)==1 else ValidationResult(True,"SAFE")
    monkeypatch.setattr(context,"_check_single",check)
    assert context.check_single(column).status=="UNKNOWN"
    assert context.check_single(column).accepted and context.check_single(column).accepted
    assert len(calls)==2 and context.cache_stats["single_hits"]==1


@pytest.mark.parametrize("changed",("soc","resource","trajectory","time"))
def test_snapshot_or_column_change_invalidates_cached_checks(monkeypatch,changed):
    s=snapshot();cache={};first=p.EvaluationContext(s,validation_cache=cache)
    columns=(discrete("a"),discrete("b"));calls=[]
    monkeypatch.setattr(p,"validate_separation",lambda *args:(calls.append(args) or ValidationResult(True,"SAFE")))
    traffic=lambda c:SimpleNamespace(plans=(),end_time=float("inf"))
    monkeypatch.setattr(first,"traffic",traffic)
    assert first.check_pair(*columns).accepted and first.check_pair(*columns).accepted
    assert len(calls)==1
    unchanged=p.EvaluationContext(s,validation_cache=cache);monkeypatch.setattr(unchanged,"traffic",traffic)
    assert unchanged.check_pair(*columns).accepted and unchanged.cache_stats["pair_hits"]==1
    if changed=="soc":
        s=replace(s,aircraft={"a":replace(s.aircraft["a"],soc=.8)})
    elif changed=="time": s=replace(s,time=s.time+1.)
    elif changed=="resource":
        columns=(replace(columns[0],resources=(Reservation(("pad",2),"a",0.,50.),)),columns[1])
    else: columns=(discrete("a",tag="new_trajectory"),columns[1])
    second=p.EvaluationContext(s,validation_cache=cache);monkeypatch.setattr(second,"traffic",traffic)
    assert second.check_pair(*columns).accepted and len(calls)==2
    assert second.cache_stats["pair_misses"]==1
    if changed in ("soc","time"): assert second.cache_stats["snapshot_invalidations"]==1


def test_prior_chain_reuse_trims_fixed_prefix_and_preserves_exact_times():
    world=RotatingSlotWorld();world.seed_hold("a","outer",0,.9)
    first=generate_transfer("a","outer",0,0,0.).plan
    assert first is not None and world.submit((first,)).accepted
    s=world.build_planning_snapshot();a=s.aircraft["a"];node=p._node(s,a)
    second=generate_transfer("a","middle",0,0,node.time,source_version=node.version).plan
    assert second is not None
    provider=p.CandidateProvider();provider.previous["a"]=(first,second)
    budget=p.PlanningBudget();reused=provider._previous_chain(s,a,node,budget)
    assert reused==(second,) and reused[0].start_time==second.start_time
    assert budget.screening["previous_prefix_trimmed"]==1
    assert provider._previous_chain(s,a,replace(node,time=node.time+1.),budget)==()
    assert budget.screening["previous_state_invalidations"]==1
    assert provider._previous_chain(s,a,replace(node,soc=0.),budget)==()
    assert budget.screening["previous_energy_invalidations"]==1


def test_previous_chain_is_checked_against_current_calendar_with_no_new_generation():
    world=RotatingSlotWorld();world.seed_hold("a","inner",0,.9)
    original=world.build_planning_snapshot()
    final=generate_final("a","inner",0,0,0.,config=original.config).plan
    assert final is not None
    world.seed_service("occupied",0,remaining=500.)
    s=world.build_planning_snapshot();provider=p.CandidateProvider();provider.previous["a"]=(final,)
    budget=p.PlanningBudget(max_attempts=0);context=p.EvaluationContext(s)
    catalog=provider.get_columns(s,budget,context)
    assert not any(c.kind=="MOVE" for c in catalog.columns["a"])
    assert budget.screening["previous_chain_hits"]==1 and budget.screening["previous_safety_invalidations"]==1
    assert context.rejection_reasons["RESOURCE_CONFLICT"]>=1
    assert not budget.attempts


def test_retention_keeps_complete_and_partial_chains_with_target_diversity():
    keep=p.PredictionColumn("a","KEEP",(),(),(0,0,3,900.,0.));wait=replace(keep,kind="WAIT")
    columns=[keep,wait]
    for target in range(8): assert p.CandidateProvider._retain_column(columns,discrete("a",target))
    assert p.CandidateProvider._retain_column(columns,discrete("a",0,full=True))
    for target in range(8,16): p.CandidateProvider._retain_column(columns,discrete("a",target))
    assert len(columns)<=10 and columns[:2]==[keep,wait]
    assert any(c.plans and c.plans[-1].pad_id is not None for c in columns)
    assert any(c.plans and c.plans[-1].pad_id is None for c in columns)
    assert len({c.first_target for c in columns if c.plans})>=7
    only_full=[keep,wait,*[discrete("a",target,full=True) for target in range(8)]]
    assert p.CandidateProvider._retain_column(only_full,discrete("a",0))
    assert sum(c.plans[-1].pad_id is not None for c in only_full if c.plans)==7


def test_warning_only_aircraft_stays_normal_candidate_and_unknown_tail_has_risk():
    s=snapshot();a=s.aircraft["a"]
    emergency=replace(a.emergency,eligible=False,warning=True)
    s=replace(s,aircraft={"a":replace(a,soc=.34,emergency=emergency)},emergency_monitor={"rescue_rule_profile":RESCUE_PROFILE_V6})
    assert [a.id for a in p._decision_aircraft(s)]==["a"]
    provider=p.CandidateProvider();context=p.EvaluationContext(s)
    state=provider._emergency_prediction(s,s.aircraft["a"],p._node(s,s.aircraft["a"]))
    assert state=="WARNING_UNKNOWN" and not provider._stop_prediction(s,state)
    keep=provider._column(s,s.aircraft["a"],(),"KEEP",context)
    assert keep.objective[2]==3 and keep.objective[3]>901.
    forced=replace(s.aircraft["a"],emergency=replace(emergency,eligible=True))
    assert not p._decision_aircraft(replace(s,aircraft={"a":forced}))


def test_callback_pad_evidence_distinguishes_certified_candidates_from_unexamined():
    world=RotatingSlotWorld();world.seed_hold("a","inner",0,.9)
    s=world.build_planning_snapshot();context=p.EvaluationContext(s);provider=p.CandidateProvider()
    final=generate_final("a","inner",0,0,0.,config=s.config).plan;assert final is not None
    keep=provider._column(s,s.aircraft["a"],(),"KEEP",context)
    move=provider._column(s,s.aircraft["a"],(final,),"MOVE",context)
    catalog=p.CandidateCatalog({"a":(keep,move)},{"a":(1,0.,0.,"a")},{},())
    class FixedProvider(p.CandidateProvider):
        def get_columns(self,*args): return catalog
    callback=p.NormalSchedulerCallback(candidate_provider=FixedProvider())
    result=callback(world._scheduler_port);assert result.certified
    row=callback.records[-1];pad=row["pad_opportunities"]["0"]
    assert pad["candidate_count"]==pad["certified_single_count"]==pad["selected_count"]==pad["submitted_count"]==1
    assert pad["validation_status"]=="EXAMINED"
    assert row["pad_opportunities"]["5"]["validation_status"]=="UNKNOWN_NOT_EXAMINED"
    assert row["aircraft_choices"][0]["full_normal_chain_count"]==1
    assert not row["aircraft_choices"][0]["only_keep_wait"]


def test_callback_reports_exact_normal_reference_after_safety_transaction():
    world=RotatingSlotWorld();world.rescue_rule_profile=RESCUE_PROFILE_V6
    world.seed_hold("a","inner",0,.9)
    s=world.build_planning_snapshot();a=s.aircraft["a"];node=p._node(s,a)
    final=generate_final("a","inner",0,0,0.,config=s.config).plan;assert final is not None
    reference=p.ReferenceProvider._reference(s,node,(final,),"test_reference")
    class FixedProvider(p.CandidateProvider):
        def get_columns(self,snapshot,budget,context):
            keep=self._column(snapshot,snapshot.aircraft["a"],(),"KEEP",context)
            return p.CandidateCatalog({"a":(keep,)},{"a":(2,0.,0.,"a")},{},())
    provider=FixedProvider()
    provider.references.cache[(s.config.config_hash,node.ring,node.slot,node.time,node.version,"a")]=reference
    callback=p.NormalSchedulerCallback(candidate_provider=provider)
    result=callback(world._scheduler_port)
    assert result.certified and callback.records[-1]["commit_status"]=="ACCEPTED"
    assert callback.records[-1]["normal_evidence_report_status"]=="ACCEPTED"
    emergency=world.aircraft["a"].emergency
    assert emergency.normal_path_soc==reference.soc_required
    assert tuple(emergency.normal_estimate_source)==(node.ring,node.slot,node.version,node.time)
    assert not emergency.eligible and not emergency.declared


def test_energy_cache_requires_exact_node_and_trajectory_and_config(monkeypatch):
    s=snapshot();node=p._node(s,s.aircraft["a"]);provider=p.ReferenceProvider()
    plan=generate_transfer("a","outer",0,0,0.).plan;assert plan is not None
    calls=[];original=p._move_energy
    monkeypatch.setattr(p,"_move_energy",lambda *args:(calls.append(args) or original(*args)))
    q=provider.move_energy(node,plan,s.config)
    assert provider.move_energy(node,plan,s.config)==q and len(calls)==1
    provider.move_energy(replace(node,soc=.8),plan,s.config)
    assert len(calls)==2 and provider.energy_stats=={"hits":1,"misses":2}


def test_template_not_found_and_historical_none_entries_are_retried(monkeypatch):
    s=snapshot();a=s.aircraft["a"];node=p._node(s,a);provider=p.ReferenceProvider();calls=[]
    monkeypatch.setattr(p,"_generate",lambda *args:(calls.append(args) or None))
    key=(s.config.config_hash,a.id,node.ring,node.slot,node.version,0,0.,40.)
    provider.templates[key]=None  # A historical checkpoint may contain misses.
    for _ in range(2): assert provider.generate(s,a,node,0,0.,40.,p.PlanningBudget())==(True,None)
    assert len(calls)==2 and provider.hits==0


def test_quality_v5_budgets_are_unchanged():
    budget=p.PlanningBudget.for_profile("quality_v5")
    assert budget.settings()==dict(soft_seconds=30.,watchdog_seconds=60.,candidate_seconds=12.,
        hint_seconds=3.,solver_seconds=8.,solver_reserve_seconds=8.,max_per_aircraft=64,max_attempts=2048,
        max_refinement_rounds=64,hint_validation_reserve_seconds=.5)


def test_real_cp_adds_multiple_conflicts_together_and_unknown_creates_no_cuts(monkeypatch):
    pytest.importorskip("ortools.sat.python.cp_model")
    keep=lambda aid:p.PredictionColumn(aid,"KEEP",(),(),(0,0,3,900.,0.))
    catalog=p.CandidateCatalog({aid:(keep(aid),discrete(aid,full=True)) for aid in ("a","b","c")},
        {aid:(1,0.,0.,aid) for aid in ("a","b","c")},{},())
    def hint(self,snapshot,candidates,context,budget,**kwargs):
        selected=tuple(c[0] for c in candidates.columns.values())
        return p.PlanningResult(selected,(),ValidationResult(True,"SAFE"),"FEASIBLE",p._sum_key(selected),certified=True)
    monkeypatch.setattr(p.EnergySlackScheduler,"plan",hint)
    class Context:
        resource_conflict=staticmethod(lambda a,b:False)
        def __init__(self,unknown=False): self.unknown=unknown;self.single_calls=0
        def check_single(self,c): self.single_calls+=1;return ValidationResult(True,"SAFE")
        def check_batch_conflicts(self,columns):
            moves=tuple(c for c in columns if c.plans)
            if len(moves)>1:
                return (ValidationResult(False,"UNKNOWN" if self.unknown else "CONFLICT","PAIR"),
                    () if self.unknown else tuple((a,b) for a,b in __import__("itertools").combinations(moves,2)))
            return ValidationResult(True,"SAFE"),()
        def prefix(self,columns): return tuple(c.plans[0] for c in columns if c.plans)
        def check_commit_prefix(self,columns): return ValidationResult(True,"SAFE")
    context=Context();result=p.RollingHorizonCPSATScheduler().plan(None,catalog,context,p.PlanningBudget.for_profile("quality_v5"))
    assert result.certified
    # Equal phase-0 objectives can select KEEP; all three conflicts are still
    # collected together within the first refinement round.
    conflicts=[r for r in result.solver_records if r.get("batch_conflict_count",0)]
    assert len(conflicts)==1 and conflicts[0]["batch_conflict_count"]==3
    assert conflicts[0]["round"]==1 and conflicts[0]["cumulative_pair_cuts"]==3
    assert result.details["pair_cuts"]==3 and result.refinement_rounds==2
    assert context.single_calls==6  # Model screening is reused after all cuts.
    unknown=Context(True);result=p.RollingHorizonCPSATScheduler().plan(None,catalog,unknown,p.PlanningBudget.for_profile("quality_v5"))
    assert result.certified and result.details["stop_reason"]=="VALIDATION_UNKNOWN"
    assert result.details["pair_cuts"]==result.details["single_cuts"]==result.details["prefix_nogoods"]==0
