"""Budget/control tests; real CP tests explicitly require the optional solver."""
from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace
import json

import pytest

from terminal_marl.core.rotating_slots import stage2_planning as p
from terminal_marl.core.rotating_slots.energy import stable_power
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.resources import Reservation
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


class Clock:
    def __init__(self): self.now=0.
    def __call__(self): return self.now


def discrete_catalog():
    """Two competing resource options; greedy A blocks the better B option."""
    def column(aid,move):
        plan=SimpleNamespace(trajectory_hash=aid+"_move",target_ring="middle",target_slot=0,pad_id=None,
            start_time=0.,kind="TRANSFER_O_M")
        objective=(0,-1,0,1.,300.) if aid=="b" and move else (0,0,3,1.,200.) if move else (0,0,3,10.,100.)
        return p.PredictionColumn(aid,"MOVE" if move else "KEEP",(plan,) if move else (),
            (Reservation(("shared",0),aid,0.,100.),) if move else (),objective)
    return p.CandidateCatalog({aid:(column(aid,False),column(aid,True)) for aid in ("a","b")},
        {"a":(1,0.,0.,"a"),"b":(1,1.,0.,"b")},{},())


class DiscreteContext:
    """Resource-only test problem, not a physical-flight certification fixture."""
    def __init__(self,budget,clock=None): self.budget=budget;self.clock=clock;self.calls=0
    resource_conflict=staticmethod(p.EvaluationContext.resource_conflict)
    def check_batch(self,columns):
        self.calls+=1
        if self.clock is not None: self.clock.now+=1.
        for a,b in combinations(columns,2):
            if self.resource_conflict(a,b): return ValidationResult(False,"CONFLICT","RESOURCE_CONFLICT"),(a,b)
        return ValidationResult(True,"SAFE"),()
    def check_single(self,column): return ValidationResult(True,"SAFE")
    def prefix(self,columns): return tuple(c.plans[0] for c in columns if c.plans)
    def check_commit_prefix(self,columns): return ValidationResult(True,"ACCEPTED")


def test_named_profile_preserves_direct_legacy_budget_and_scales_solver_shares():
    legacy=p.PlanningBudget()
    assert (legacy.soft_seconds,legacy.watchdog_seconds,legacy.candidate_seconds,legacy.solver_seconds)==(5.,30.,3.,2.)
    quality=p.PlanningBudget.for_profile("quality_v5")
    assert quality.settings()==dict(p.QUALITY_BUDGET_SETTINGS,max_per_aircraft=64,max_attempts=2048,
        max_refinement_rounds=64,hint_validation_reserve_seconds=.5)
    assert legacy.max_refinement_rounds==3 and legacy.hint_validation_reserve_seconds==0.
    assert [share*quality.solver_seconds/sum(p.PHASE_SHARES) for share in p.PHASE_SHARES]==pytest.approx([1.2,3.2,1.2,1.6,.8])
    assert p.PlanningBudget(soft_seconds=.01,candidate_seconds=0.).soft_seconds==.01
    with pytest.raises(ValueError,match="unknown"): p.PlanningBudget.for_profile("typo")


def test_hint_stage_uses_original_clock_and_leaves_solver_reserve(monkeypatch):
    clock=Clock();monkeypatch.setattr(p.time,"perf_counter",clock)
    budget=p.PlanningBudget.for_profile("quality_v5");budget.wall_start=clock();origin=budget.wall_start
    clock.now=12.  # candidate phase has already consumed its own allowance
    context=DiscreteContext(budget,clock)
    with budget.stage_limit(budget.hint_seconds,"HINT_BUDGET",budget.solver_reserve_seconds):
        result=p.EnergySlackScheduler().plan(None,discrete_catalog(),context,budget)
    assert clock.now==15. and budget.wall_start==origin and not result.certified
    assert "HINT_BUDGET" in budget.truncations and budget.search_available()
    assert budget.soft_seconds-budget.elapsed>=budget.solver_seconds
    clock.now=22.
    assert not budget.preparation_available() and budget.search_available()
    assert budget.soft_seconds-budget.elapsed==8.


def test_callback_profile_roundtrip_and_historical_state_without_profile():
    callback=p.NormalSchedulerCallback()
    assert callback.budget_profile=="quality_v5"
    state=callback.state_dict()
    restored=p.NormalSchedulerCallback(budget_profile="legacy_v4");restored.load_state_dict(state)
    assert restored.state_dict()==state
    del state["budget_profile"]
    restored.load_state_dict(state)
    assert restored.budget_profile=="legacy_v4"


def test_hint_reserves_validation_inside_original_stage(monkeypatch):
    clock=Clock();monkeypatch.setattr(p.time,"perf_counter",clock)
    budget=p.PlanningBudget.for_profile("quality_v5");budget.wall_start=clock()
    base=discrete_catalog().columns["a"]
    columns={str(i):tuple(replace(c,aircraft_id=str(i),resources=()) for c in base) for i in range(9)}
    catalog=p.CandidateCatalog(columns,{i:(1,0.,0.,i) for i in columns},{},())
    class ValidationContext(DiscreteContext):
        def check_batch(self,selected):
            self.calls+=1
            clock.now+=.35 if self.calls<=8 else .1
            return ValidationResult(True,"SAFE"),()
        def check_commit_prefix(self,selected):
            clock.now+=.05
            assert budget.evaluation_available()
            return ValidationResult(True,"ACCEPTED")
    context=ValidationContext(budget)
    with budget.stage_limit(3.,"HINT_BUDGET",8.):
        result=p.EnergySlackScheduler().plan(None,catalog,context,budget,validation_reserve_seconds=.5)
    assert result.certified and sum(c.kind=="MOVE" for c in result.selected)==8
    assert clock.now==pytest.approx(2.95) and budget.wall_start==0.
    assert "HINT_VALIDATION_RESERVE" in budget.truncations and "HINT_BUDGET" not in budget.truncations


def test_unknown_slack_margin_depends_on_soc_and_remaining_ring_time():
    world=RotatingSlotWorld();world.seed_hold("a","outer",0,.8);world.seed_hold("z","outer",18,.6)
    snapshot=world.build_planning_snapshot();unknown=p.NormalReference("UNKNOWN")
    assert p.energy_slack(snapshot,snapshot.aircraft["z"],unknown)<p.energy_slack(snapshot,snapshot.aircraft["a"],unknown)
    for ring in ("outer","middle","inner"):
        a=replace(snapshot.aircraft["a"],ring=ring,slot=0)
        key=p.energy_slack(snapshot,a,unknown)
        rate=stable_power(ring,snapshot.config).current_a/(snapshot.config.energy.reference_capacity_Ah*3600.)
        assert key[0]==1 and key[1]==pytest.approx((a.soc-.25)/rate-p.remaining_normal_time_lower(snapshot,ring))
        assert unknown.status=="UNKNOWN" and unknown.soc_required is None
    assert len({p.energy_slack(snapshot,replace(snapshot.aircraft["a"],ring=r),unknown)[1] for r in ("outer","middle","inner")})==3


@pytest.mark.parametrize("source_status",("UNKNOWN","KNOWN"))
def test_callback_logs_budget_choices_and_emergency_priority_rejection(source_status):
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.seed_hold("urgent","outer",0,.3,emergency=True)
    world.seed_hold("normal","inner",9,.9)
    snapshot=world.build_planning_snapshot();context=p.EvaluationContext(snapshot);provider=p.CandidateProvider()
    final=generate_final("normal","inner",9,3,0.,config=snapshot.config).plan
    assert final is not None
    keep=provider._column(snapshot,snapshot.aircraft["normal"],(),"KEEP",context)
    # The source reference used by slack and the H-tail reference are different
    # epochs. Deliberately disagree so logging cannot infer one from the other.
    keep=replace(keep,reference=p.NormalReference("KNOWN" if source_status=="UNKNOWN" else "UNKNOWN"))
    move=provider._column(snapshot,snapshot.aircraft["normal"],(final,),"MOVE",context)
    catalog=p.CandidateCatalog({"normal":(keep,move)},{"normal":(1 if source_status=="UNKNOWN" else 2,0.,0.,"normal")},{},(),
        slack_reference_statuses={"normal":source_status})
    class FixedProvider(p.CandidateProvider):
        def get_columns(self,*args): self.aligned_starts=[];return catalog
    callback=p.NormalSchedulerCallback(candidate_provider=FixedProvider())
    callback(world._scheduler_port)
    record=callback.records[-1]
    assert record["budget_profile"]=="quality_v5" and record["planning_budget"]["soft_seconds"]==30.
    assert record["candidate_rejection_reasons"]["EMERGENCY_PRIORITY_WAIT"]>=1
    row=record["aircraft_choices"][0]
    assert row["selected_kind"]=="KEEP" and row["move_candidates"]==1
    assert row["source_reference_status"]==source_status
    assert row["keep_tail_reference_status"]!=source_status
    assert row["unknown_slack_source"]==(p.UNKNOWN_SLACK_SOURCE if source_status=="UNKNOWN" else None)
    assert record["move_aircraft_count"]==1 and record["skip_reason"]=="METHOD_ENERGY_SLACK"
    assert record["solver_entered"] is False and record["final_objective"]==record["objective"]


def test_priority_guard_does_not_block_transfer_for_future_final_suffix(monkeypatch):
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.seed_hold("urgent","outer",0,.3,emergency=True);world.seed_hold("normal","outer",18,.9)
    snapshot=world.build_planning_snapshot();context=p.EvaluationContext(snapshot)
    transfer=generate_transfer("normal","outer",18,14,0.,config=snapshot.config).plan
    assert transfer is not None
    suffix=SimpleNamespace(kind="FINAL_NORMAL",start_time=100.)
    column=p.PredictionColumn("normal","MOVE",(transfer,suffix),(),(0,0,3,900,0))
    monkeypatch.setattr(context,"check_pair",lambda *args:ValidationResult(True,"SAFE"))
    assert context.check_single(column).accepted
    assert not context.rejection_reasons


def test_real_cp_enters_nontrivial_catalog_and_reports_true_improvement():
    pytest.importorskip("ortools.sat.python.cp_model")
    budget=p.PlanningBudget.for_profile("quality_v5");context=DiscreteContext(budget)
    result=p.RollingHorizonCPSATScheduler().plan(None,discrete_catalog(),context,budget)
    assert result.certified and result.details["solver_entered"]
    assert result.solver_records and all(r["status"]=="OPTIMAL" for r in result.solver_records)
    assert result.details["skip_reason"] is None
    assert result.details["strict_improvement_over_hint"] is True
    assert result.details["final_objective"]==result.objective<result.details["hint_objective"]
    assert next(c for c in result.selected if c.aircraft_id=="b").kind=="MOVE"
    assert result.details["solver_wall_s"]==budget.solver_wall_used<=8.


@pytest.mark.parametrize("empty",(False,True))
def test_real_cp_zero_options_reports_reason_without_solver_calls(empty):
    pytest.importorskip("ortools.sat.python.cp_model")
    budget=p.PlanningBudget.for_profile("quality_v5");catalog=discrete_catalog()
    catalog=replace(catalog,columns={} if empty else {k:(v[0],) for k,v in catalog.columns.items()})
    result=p.RollingHorizonCPSATScheduler().plan(None,catalog,DiscreteContext(budget),budget)
    assert result.certified and not result.solver_records and not result.details["solver_entered"]
    assert result.details["skip_reason"]==("EMPTY_CATALOG" if empty else "NO_MOVE_OPTIONS")
    assert result.details["strict_improvement_over_hint"] is False


def test_real_cp_hint_cap_does_not_starve_solver(monkeypatch):
    pytest.importorskip("ortools.sat.python.cp_model")
    clock=Clock();monkeypatch.setattr(p.time,"perf_counter",clock)
    budget=p.PlanningBudget.for_profile("quality_v5");budget.wall_start=clock();clock.now=12.
    context=DiscreteContext(budget,clock)
    result=p.RollingHorizonCPSATScheduler().plan(None,discrete_catalog(),context,budget)
    assert budget.hint_wall_used==3. and result.details["solver_entered"]
    assert result.solver_records and result.details["hint_objective"] is None
    assert budget.wall_start==0. and budget.watchdog_ok()


def test_real_cp_truncated_resource_model_never_claims_catalog_optimal(monkeypatch):
    pytest.importorskip("ortools.sat.python.cp_model")
    budget=p.PlanningBudget.for_profile("quality_v5")
    monkeypatch.setattr(budget,"preparation_available",lambda:False)
    result=p.RollingHorizonCPSATScheduler().plan(None,discrete_catalog(),DiscreteContext(budget),budget)
    assert result.details["solver_entered"] and not result.details["resource_conflicts_complete"]
    assert not result.details["model_screen_complete"] and result.status=="LEX_LIMITED"


def test_real_cp_unknown_retains_verified_safe_hint(monkeypatch):
    cp_model=pytest.importorskip("ortools.sat.python.cp_model")
    original=cp_model.CpSolver
    class TinySolver(original):
        def Solve(self,model,*args,**kwargs):
            self.parameters.max_time_in_seconds=1.e-9
            return super().Solve(model,*args,**kwargs)
    monkeypatch.setattr(cp_model,"CpSolver",TinySolver)
    budget=p.PlanningBudget.for_profile("quality_v5")
    result=p.RollingHorizonCPSATScheduler().plan(None,discrete_catalog(),DiscreteContext(budget),budget)
    assert result.certified and result.status=="LEX_LIMITED"
    assert result.solver_records[0]["status"]=="UNKNOWN"
    assert result.objective==result.details["hint_objective"] and result.details["strict_improvement_over_hint"] is False


def test_real_cp_solver_allowance_is_not_recreated_on_reentry():
    pytest.importorskip("ortools.sat.python.cp_model")
    budget=p.PlanningBudget.for_profile("quality_v5");budget.solver_wall_used=8.
    result=p.RollingHorizonCPSATScheduler().plan(None,discrete_catalog(),DiscreteContext(budget),budget)
    assert result.certified and not result.solver_records
    assert result.details["skip_reason"]=="SOLVER_ALLOCATION_EXHAUSTED"
    assert budget.solver_wall_used==8.


def refinement_catalog():
    """Four proven geometric conflicts, hidden from resource-only screening."""
    keep=p.PredictionColumn("a","KEEP",(),(),(0,0,3,10.,0.))
    anchor=p.PredictionColumn("anchor","KEEP",(),(),(0,0,3,10.,0.))
    moves=[]
    for n in range(5):
        plan=SimpleNamespace(trajectory_hash="unsafe"+str(n) if n<4 else "safe",
            kind="TRANSFER_O_M",target_ring="middle",target_slot=n,pad_id=None,start_time=0.)
        moves.append(p.PredictionColumn("a","MOVE",(plan,),(),(0,-1,min(n,3),1. if n<4 else 2.,0.)))
    return p.CandidateCatalog({"a":(keep,*moves),"anchor":(anchor,)},
        {"a":(1,0.,0.,"a"),"anchor":(1,0.,0.,"anchor")},{},())


class RefinementContext(DiscreteContext):
    def __init__(self,budget,unknown=False): super().__init__(budget);self.unknown=unknown
    def check_batch(self,columns):
        bad=next((c for c in columns if c.plans and c.plans[0].trajectory_hash.startswith("unsafe")),None)
        if bad is not None:
            anchor=next(c for c in columns if c.aircraft_id=="anchor")
            return ValidationResult(False,"UNKNOWN" if self.unknown else "CONFLICT","HARD_SEPARATION"),(bad,anchor)
        return ValidationResult(True,"SAFE"),()


def keep_hint(monkeypatch):
    def hint(self,snapshot,candidates,context,budget,**kwargs):
        selected=tuple(cols[0] for cols in candidates.columns.values())
        evaluation=context.check_commit_prefix(selected)
        return p.PlanningResult(selected,(),evaluation,"FEASIBLE",p._sum_key(selected),certified=True)
    monkeypatch.setattr(p.EnergySlackScheduler,"plan",hint)


@pytest.mark.parametrize("profile,expected_limit",(("quality_v5",False),("legacy_v4",True)))
def test_real_cp_refines_more_than_three_proven_conflicts_or_logs_limit(monkeypatch,profile,expected_limit):
    pytest.importorskip("ortools.sat.python.cp_model")
    keep_hint(monkeypatch)
    budget=p.PlanningBudget.for_profile(profile);origin=budget.wall_start
    result=p.RollingHorizonCPSATScheduler().plan(None,refinement_catalog(),RefinementContext(budget),budget)
    assert result.certified and result.details["solver_entered"]
    assert budget.wall_start==origin and budget.solver_wall_used<=budget.solver_seconds
    assert all(r["status"]=="OPTIMAL" for r in result.solver_records)
    assert result.details["max_refinement_rounds"]==(3 if expected_limit else 64)
    assert (result.details["stop_reason"]=="REFINEMENT_LIMIT")==expected_limit
    assert ("REFINEMENT_LIMIT" in budget.truncations)==expected_limit
    if expected_limit:
        assert result.refinement_rounds==budget.refinement_rounds_used==3
    else:
        assert result.refinement_rounds==budget.refinement_rounds_used>=5
        assert result.solver_records[-1]["phase"]==4
        assert next(c for c in result.selected if c.aircraft_id=="a").plans[0].trajectory_hash=="safe"
    print(json.dumps(dict(test="real_cp_lazy_refinement",profile=profile,
        rounds=budget.refinement_rounds_used,stop_reason=result.details["stop_reason"],
        solver_wall_s=budget.solver_wall_used,phase_count=len(result.solver_records))))


def test_real_cp_unknown_validation_is_not_a_conflict_cut(monkeypatch):
    pytest.importorskip("ortools.sat.python.cp_model")
    keep_hint(monkeypatch)
    budget=p.PlanningBudget.for_profile("quality_v5")
    result=p.RollingHorizonCPSATScheduler().plan(None,refinement_catalog(),RefinementContext(budget,unknown=True),budget)
    assert result.certified and result.details["stop_reason"]=="VALIDATION_UNKNOWN"
    assert result.refinement_rounds==1 and budget.refinement_rounds_used==1
    assert "REFINEMENT_LIMIT" not in budget.truncations
    assert result.objective==result.details["hint_objective"]


def test_real_cp_refinement_allowance_is_not_recreated_on_reentry(monkeypatch):
    pytest.importorskip("ortools.sat.python.cp_model")
    keep_hint(monkeypatch)
    budget=p.PlanningBudget.for_profile("quality_v5");budget.refinement_rounds_used=64
    result=p.RollingHorizonCPSATScheduler().plan(None,refinement_catalog(),RefinementContext(budget),budget)
    assert result.certified and not result.solver_records and budget.refinement_rounds_used==64
    assert result.details["stop_reason"]==result.details["skip_reason"]=="REFINEMENT_LIMIT"


def test_real_cp_physical_prefix_accepted_with_quality_budget():
    pytest.importorskip("ortools.sat.python.cp_model")
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.seed_hold("normal","inner",0,.9)
    snapshot=world.build_planning_snapshot();budget=p.PlanningBudget.for_profile("quality_v5")
    context=p.EvaluationContext(snapshot,budget=budget);provider=p.CandidateProvider()
    final=generate_final("normal","inner",0,0,0.,config=snapshot.config).plan
    assert final is not None
    keep=provider._column(snapshot,snapshot.aircraft["normal"],(),"KEEP",context)
    move=provider._column(snapshot,snapshot.aircraft["normal"],(final,),"MOVE",context)
    catalog=p.CandidateCatalog({"normal":(keep,move)},{"normal":(1,0.,0.,"normal")},{},())
    result=p.RollingHorizonCPSATScheduler().plan(snapshot,catalog,context,budget)
    assert result.certified and result.prefix and result.details["solver_entered"]
    assert result.details["hint_certified"] is True
    committed=world.commit_validated_prefix(result.evaluation)
    assert committed.accepted and world.active["normal"].trajectory_hash==final.trajectory_hash
    print(json.dumps(dict(test="quality_v5_physical_prefix",hint_certified=result.details["hint_certified"],
        solver_entered=result.details["solver_entered"],skip_reason=result.details["skip_reason"],
        strict_improvement_over_hint=result.details["strict_improvement_over_hint"],
        solver_phases=len(result.solver_records),commit_status=committed.status,budget=budget.settings())))
