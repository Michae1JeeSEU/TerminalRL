"""Small deterministic v2 generation/selection checks, with real certificates."""
from dataclasses import replace
from types import SimpleNamespace
import time
import pytest
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.stage2_planning import (CandidateProvider,EvaluationContext,PlanningBudget,
    PlanningResult,NormalReference,NormalSchedulerCallback,_node,_selection_key)
from terminal_marl.core.rotating_slots.selective_replanning import PlanningScope,build_global_frame,LocalPlanner
from terminal_marl.core.rotating_slots.resources import Reservation
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.trajectory import generate_transfer


def setup(n=1):
    w=RotatingSlotWorld()
    for i in range(n): w.seed_hold(str(i),"outer",i*3,.9)
    s=w.build_planning_snapshot(); p=CandidateProvider(planning_profile="downstream_service_v8")
    frame=build_global_frame(s,p,EvaluationContext(s)); ids=tuple(sorted(s.aircraft))
    scope=PlanningScope(s.version,s.time,ids,ids,(),actual_k=n,objective_frame_id=frame.identity)
    budget=PlanningBudget(candidate_seconds=6.,solver_seconds=8.,soft_seconds=30.,max_per_aircraft=32,max_attempts=1024,
        candidate_active_seconds=0.,planning_wall_deadline=time.perf_counter()+19.,search_wall_deadline=time.perf_counter()+15.)
    return w,s,p,frame,scope,budget


def certify(snapshot,catalog,context,selected):
    selected=tuple(selected); checked,_=context.check_batch(selected)
    assert checked.accepted,checked
    evaluation=context.check_commit_prefix(selected)
    assert evaluation.accepted and not evaluation.refinement_required,evaluation
    return PlanningResult(selected,context.prefix(selected),evaluation,"TEST_REAL_CERTIFIED",_selection_key(catalog,selected),certified=True)


def test_real_provider_publishes_initial_small_catalog_before_reference_growth(monkeypatch):
    w,s,p,f,scope,budget=setup(2); order=[]
    def progress(snapshot,a,node,budget,context):
        order.append(("progress",a.id))
        yield ("ATTEMPT",None,None)
    def reference(*args):
        order.append(("reference",args[1].id))
        yield None
        return NormalReference("UNKNOWN")
    monkeypatch.setattr(p,"_progress_steps",progress)
    monkeypatch.setattr(p.references,"build_steps",reference)
    context=EvaluationContext(s,budget=budget,scope=scope,global_frame=f)
    generator=p.iter_progressive_columns(s,budget,context,scope=scope,early_publish=True)
    first=next(generator)
    assert order==[("progress","0"),("progress","1")]
    assert first.generation_coverage["phase"]=="INITIAL_PROGRESS"
    assert first.generation_coverage["all_initial_aircraft_visited"]
    assert all([c.kind for c in rows]==["KEEP","WAIT"] for rows in first.columns.values())
    generator.close()


def test_initial_unvisited_is_explicit_when_attempt_budget_exhausted():
    w,s,p,f,scope,budget=setup(2); budget.max_attempts=0
    context=EvaluationContext(s,budget=budget,scope=scope,global_frame=f)
    generator=p.iter_progressive_columns(s,budget,context,scope=scope,early_publish=True)
    first=next(generator); generator.close()
    assert first.generation_coverage["initial_unvisited_ids"]==["0","1"]
    assert not first.generation_coverage["all_initial_aircraft_visited"]
    assert all(row["only_keep_wait"] for row in LocalPlanner._coverage(s,first).values())


class TwoCatalogProvider(CandidateProvider):
    def __init__(self,plans):
        super().__init__(planning_profile="downstream_service_v8")
        self.plans=plans; self.events=[]; self.budgets=[]
    def iter_progressive_columns(self,snapshot,budget,context,*,scope=None,early_publish=False):
        assert early_publish
        for batch in range(3):
            self.events.append(("generate",batch,scope.identity)); self.budgets.append((id(budget),budget.wall_start))
            columns={}; refs={}
            for aid in scope.replan_ids:
                a=snapshot.aircraft[aid]; ref=NormalReference("UNKNOWN"); refs[aid]=ref
                columns[aid]=[self._column(snapshot,a,(),kind,context,ref) for kind in ("KEEP","WAIT")]
                for plan in self.plans.get(aid,()):
                    col=self._column(snapshot,a,(plan,),"MOVE",context,ref)
                    if col is not None and context.check_single(col).accepted: columns[aid].append(col)
            catalog=self._progressive_catalog(snapshot,budget,context,columns,refs,[snapshot.aircraft[aid] for aid in scope.replan_ids])
            yield replace(catalog,generation_coverage=dict(phase="INITIAL_PROGRESS" if batch==0 else "GROWTH_0"))


class ChoosingScheduler:
    def __init__(self,provider,modes): self.provider=provider; self.modes=modes; self.calls=[]; self.first=None
    def plan(self,snapshot,catalog,context,budget,*,incumbent=None):
        index=len(self.calls); self.provider.events.append(("solve",index,context.scope.identity))
        self.calls.append(dict(budget_id=id(budget),wall_start=budget.wall_start,scope=context.scope,
            allocation=budget.solver_call_seconds,incumbent=incumbent))
        budget.solver_wall_used+=.1
        mode=self.modes[index]
        if mode=="UNKNOWN":
            return PlanningResult((),(),ValidationResult(False,"UNKNOWN","TEST_UNKNOWN"),"TEST_UNKNOWN",(),certified=False)
        selected=[]
        for rows in catalog.columns.values():
            selected.append(next((c for c in rows if c.kind==mode),rows[0]))
        result=certify(snapshot,catalog,context,selected)
        if self.first is None: self.first=result
        return result


@pytest.mark.parametrize("mode",["MOVE","WAIT"])
def test_two_calls_share_budget_and_keep_certified_best_on_unknown(mode):
    w,s,p,f,scope,budget=setup()
    plan=generate_transfer("0","outer",0,0,0.,durations=(40.,)).plan
    assert plan is not None
    provider=TwoCatalogProvider({"0":(plan,)})
    scheduler=ChoosingScheduler(provider,(mode,"UNKNOWN"))
    local=LocalPlanner(provider,scheduler,progressive_v2=True)
    result=local.plan(s,scope,f,budget)
    assert result.certified and result.prefix==scheduler.first.prefix
    assert all(c.kind==mode for c in result.selected)
    assert [e[0] for e in provider.events]==["generate","solve","generate","solve"]
    assert len(scheduler.calls)==2 and local.telemetry["scope_expansions"]==0
    assert scheduler.calls[0]["allocation"]==2. and scheduler.calls[1]["allocation"]==pytest.approx(7.9)
    assert len(set(provider.budgets))==1 and len({(c["budget_id"],c["wall_start"]) for c in scheduler.calls})==1
    assert budget.solver_seconds==8. and budget.candidate_seconds==6.
    assert local.telemetry["calls"][1]["previous_best_retained"]
    assert local.telemetry["first_certified"] is not None
    assert not local.telemetry["actual_fallback"]


def test_real_blocker_expansion_happens_once_with_same_budget_and_complete_background():
    w=RotatingSlotWorld(); w.seed_hold("a","outer",1,.9); w.seed_hold("b","middle",1,.9)
    s=w.build_planning_snapshot(); provider=TwoCatalogProvider({})
    a=generate_transfer("a","outer",1,1,5.,durations=(40.,)).plan
    b=generate_transfer("b","middle",1,0,0.,durations=(40.,)).plan
    assert a and b
    provider.plans={"a":(a,),"b":(b,)}
    frame=build_global_frame(s,provider,EvaluationContext(s))
    scope=PlanningScope(s.version,s.time,("a","b"),("a",),("b",),actual_k=1,objective_frame_id=frame.identity)
    budget=PlanningBudget(candidate_seconds=6.,solver_seconds=8.,soft_seconds=30.,candidate_active_seconds=0.)
    scheduler=ChoosingScheduler(provider,("WAIT","MOVE")); local=LocalPlanner(provider,scheduler,progressive_v2=True)
    result=local.plan(s,scope,frame,budget)
    assert result.certified and len(scheduler.calls)==2
    assert local.telemetry["scope_expansions"]==1
    assert scheduler.calls[1]["scope"].replan_ids==("a","b")
    assert len(set(provider.budgets))==1
    assert local.telemetry["calls"][1]["incumbent_mapped"]
    assert set(local.last_scope.replan_ids)|set(local.last_scope.fixed_background_ids)==set(s.aircraft)


def test_current_frame_worse_certified_choice_does_not_replace_better_best():
    w,s,p,f,scope,budget=setup()
    plan=generate_transfer("0","outer",0,0,0.,durations=(40.,)).plan
    assert plan
    provider=TwoCatalogProvider({"0":(plan,)})
    class OrderedRealChoices(ChoosingScheduler):
        def plan(self,snapshot,catalog,context,budget,*,incumbent=None):
            self.calls.append(dict(allocation=budget.solver_call_seconds))
            choices=[certify(snapshot,catalog,context,(next(c for c in catalog.columns["0"] if c.kind==kind),))
                for kind in ("WAIT","MOVE")]
            choices.sort(key=lambda r:r.objective)
            assert choices[0].objective<choices[1].objective
            if len(self.calls)==1: self.first=choices[0]; return choices[0]
            return choices[1]
    scheduler=OrderedRealChoices(provider,())
    local=LocalPlanner(provider,scheduler,progressive_v2=True)
    result=local.plan(s,scope,f,budget)
    assert result.certified and result.objective==scheduler.first.objective
    assert local.telemetry["calls"][1]["previous_best_retained"]


def test_strict_resource_pruning_is_only_finite_committed_and_keeps_joint_release(monkeypatch):
    w,s,p,f,scope,budget=setup(); a=s.aircraft["0"]; node=_node(s,a)
    context=EvaluationContext(s,budget=budget,scope=scope,global_frame=f)
    budget._strict_resource_context=context
    resource=("middle",0)
    fixed=Reservation(resource,"fixed",0.,100.,committed=True)
    occupied=replace(s,slots=(fixed,))
    monkeypatch.setattr(p.references,"_generate_unobserved",lambda *args:(True,"MATERIALIZED"))
    assert p.references.generate(occupied,a,node,0,0.,40.,budget)==(True,None)
    assert budget.screening["strict_immutable_resource_pruned"]==1
    for row in (replace(fixed,committed=False),replace(fixed,end=float("inf"))):
        joint=replace(s,slots=(row,))
        assert p.references.generate(joint,a,node,0,0.,40.,budget)==(True,"MATERIALIZED")


def test_previous_chain_invalid_state_epoch_and_unknown_never_reused(monkeypatch):
    w,s,p,f,scope,budget=setup(); a=s.aircraft["0"]; node=_node(s,a)
    plan=generate_transfer("0","outer",0,0,0.,durations=(40.,)).plan
    assert plan
    p.previous["0"]=(plan,)
    assert p._previous_chain(s,a,node,budget)==(plan,)
    assert not p._previous_chain(s,a,replace(node,time=1.),budget)
    assert not p._previous_chain(s,a,replace(node,version=1),budget)
    monkeypatch.setattr(p.references,"move_energy",lambda *args:(float("inf"),)*5)
    assert not p._previous_chain(s,a,node,budget)


def test_joint_conflict_observation_has_pair_ids_and_excludes_unknown_and_background(monkeypatch):
    w,s,p,f,scope,budget=setup(2); context=EvaluationContext(s,scope=scope,global_frame=f)
    ca=p._column(s,s.aircraft["0"],(),"WAIT",context)
    cb=p._column(s,s.aircraft["1"],(),"WAIT",context)
    monkeypatch.setattr(context,"check_single",lambda c:ValidationResult(True,"SAFE"))
    monkeypatch.setattr(context,"check_pair",lambda a,b:ValidationResult(False,"CONFLICT","RESOURCE_CONFLICT"))
    context.check_batch_conflicts((ca,cb))
    row=context.joint_conflict_observations[0]
    assert row["aircraft_ids"]==["0","1"] and row["source"]=="SELECTED_SELECTED_BATCH"
    assert row["snapshot_id"]==s.version and row["context_scope"]==scope.identity
    context.joint_conflict_observations=[]
    monkeypatch.setattr(context,"check_pair",lambda a,b:ValidationResult(False,"UNKNOWN","BUDGET"))
    context.check_batch_conflicts((ca,cb)); assert not context.joint_conflict_observations
    monkeypatch.setattr(context,"check_pair",lambda a,b:ValidationResult(False,"CONFLICT","RESOURCE_CONFLICT"))
    context.check_batch_conflicts((ca,replace(cb,kind="BACKGROUND"))); assert not context.joint_conflict_observations


def test_pure_geometry_counts_are_reported_and_cold_worker_clear_resets():
    from terminal_marl.core.rotating_slots.safety import geometry_evidence_counts
    from terminal_marl.core.rotating_slots.performance.parallel import _clear_worker_accelerators
    reset=_clear_worker_accelerators()
    assert reset["after"]["safety.geometry_evidence"]["entries"]==0
    assert not any(geometry_evidence_counts().values())
    cb=NormalSchedulerCallback(planning_profile="downstream_service_v8")
    w=RotatingSlotWorld(); cb(w._scheduler_port)
    counts=cb.records[-1]["geometry_evidence_counts"]
    assert all(counts["delta"][key]==counts["after"][key]-counts["before"][key] for key in counts["after"])
