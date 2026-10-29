"""Fixed-catalog observations; no rollout or candidate generation required."""
from dataclasses import replace
import json
import math
import pytest

from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.trajectory import make_stable_plan
from terminal_marl.core.rotating_slots.stage2_planning import (
    EvaluationContext,PlanningBudget,PredictionColumn,PlanningNode,CandidateCatalog,RollingHorizonCPSATScheduler,NormalSchedulerCallback)
from terminal_marl.core.rotating_slots.selective_replanning.scope import PlanningScope
from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import (
    DiagnosticConfig,CandidateDiagnostics,candidate_hash,probe_catalog_choices,plain)


def fixture(n=2):
    world=RotatingSlotWorld()
    for k in range(n): world.seed_hold(chr(97+k),"outer",k*3,.9)
    snapshot=world.build_planning_snapshot(); columns={}
    for aid,a in snapshot.aircraft.items():
        keep=PredictionColumn(aid,"KEEP",(),(),(0,0,1,50,100))
        plan=make_stable_plan(aid,"outer",a.slot,0.,40.,config=snapshot.config)
        move=PredictionColumn(aid,"MOVE",(plan,),(),(0,0,0,20,200),
                              PlanningNode("outer",a.slot,40.,.8,0.,1))
        columns[aid]=(keep,move)
    return snapshot,CandidateCatalog(columns,{aid:(0.,aid) for aid in columns}, {},())


def test_pair_observation_preserves_actual_witness_and_absolute_interval(monkeypatch):
    s,cat=fixture(); calls=[]
    original=ValidationResult(False,"UNSAFE","HARD_SEPARATION",
                              {"witness_time":12.5,"distance_m":79.5,"interval":[12.,13.]})
    def check(*args): calls.append(args[2:4]); return original
    monkeypatch.setattr("terminal_marl.core.rotating_slots.stage2_planning.validate_separation",check)
    c=EvaluationContext(s); result=c.check_pair(cat.columns["a"][1],cat.columns["b"][0])
    assert result.status==original.status and result.accepted==original.accepted
    assert all(result.details[k]==v for k,v in original.details.items())
    evidence=result.details["pair_evidence"]
    assert evidence["check_interval"]==list(calls[0]) and evidence["check_interval"][0]==s.time
    assert evidence["hard_separation_m"]==80.
    assert evidence["participants"][0]["candidate_hash"]==candidate_hash(cat.columns["a"][1])
    again=c.check_pair(cat.columns["a"][1],cat.columns["b"][0])
    assert again==result and len(calls)==1 and len(c.pair_conflict_evidence)==1
    assert c.pair_conflict_evidence[0]["count"]==2


def test_unknown_is_not_sampled_or_cached_or_converted_to_safe(monkeypatch):
    s,cat=fixture(); calls=[]
    def check(*args):
        calls.append(1); return ValidationResult(False,"UNKNOWN","HARD_SEPARATION",{"interval":[3.,3.001]})
    monkeypatch.setattr("terminal_marl.core.rotating_slots.stage2_planning.validate_separation",check)
    c=EvaluationContext(s,budget=PlanningBudget(diagnostics=DiagnosticConfig()))
    for _ in range(2):
        result=c.check_pair(cat.columns["a"][1],cat.columns["b"][0])
        assert result.status=="UNKNOWN" and not result.accepted
        assert result.details["interval"]==[3.,3.001]
        assert "distance_m" not in result.details and "witness_time" not in result.details
    assert len(calls)==2 and not c.pairs
    assert c.pair_conflict_evidence[-1]["status"]=="UNKNOWN"


def test_formal_conflicts_deduplicate_and_bound_payload_construction(monkeypatch):
    from terminal_marl.core.rotating_slots.selective_replanning import diagnostics
    s,cat=fixture(); context=EvaluationContext(s); identities=[]; original=diagnostics.identity
    def identity(column): identities.append(column.aircraft_id); return original(column)
    monkeypatch.setattr(diagnostics,"identity",identity)
    a,b=cat.columns["a"][1],cat.columns["b"][0]
    for i in range(128):
        context._pair_observation(a,b,ValidationResult(False,"UNKNOWN","HARD_SEPARATION",{"interval":[i,i+.1]}),[0.,1000.])
    context._pair_observation(a,b,ValidationResult(False,"UNKNOWN","HARD_SEPARATION",{"interval":[0,.1]}),[0.,1000.])
    context._pair_observation(a,b,ValidationResult(False,"UNKNOWN","HARD_SEPARATION",{"interval":[500,500.1]}),[0.,1000.])
    assert len(context.pair_conflict_evidence)==128 and context.pair_conflict_evidence[0]["count"]==2
    assert context.pair_conflict_evidence_dropped==1 and len(identities)==256


def test_synthetic_catalog_has_complete_slacks_for_real_hint_scheduler(monkeypatch):
    from terminal_marl.core.rotating_slots.stage2_planning import EnergySlackScheduler
    s,cat=fixture(); budget=PlanningBudget()
    monkeypatch.setattr("terminal_marl.core.rotating_slots.stage2_planning.validate_separation",
                        lambda *args: ValidationResult(True,"SAFE"))
    context=EvaluationContext(s,budget=budget,prefix_evaluator=lambda p: ValidationResult(True,"SAFE"))
    result=EnergySlackScheduler().plan(s,cat,context,budget)
    assert result.certified and set(cat.slacks)==set(cat.columns)


@pytest.mark.parametrize("config",[None,DiagnosticConfig()])
def test_infinite_resource_overlap_telemetry_is_canonical_without_changing_raw_result(config):
    from terminal_marl.core.rotating_slots.resources import Reservation
    from terminal_marl.core.rotating_slots.stage2_campaign import canonical
    s,cat=fixture()
    a=replace(cat.columns["a"][1],resources=(Reservation(("inner",1),"a",0.,math.inf),))
    b=replace(cat.columns["b"][0],resources=(Reservation(("inner",1),"b",5.,math.inf),))
    context=EvaluationContext(s,budget=PlanningBudget(diagnostics=config))
    result=context.check_pair(a,b)
    assert not result.accepted and result.status=="CONFLICT" and result.reason=="RESOURCE_CONFLICT"
    raw=result.details["resource_overlaps"][0]
    assert raw["first_interval"]==[0.,math.inf] and raw["second_interval"]==[5.,math.inf]
    assert raw["interval"]==[5.,math.inf] and math.isinf(a.resources[0].end) and math.isinf(b.resources[0].end)
    telemetry=dict(selective_replanning=dict(calls=[dict(pair_conflict_evidence=context.pair_conflict_evidence)]))
    saved=json.loads(canonical(telemetry))
    overlap=saved["selective_replanning"]["calls"][0]["pair_conflict_evidence"][0]["details"]["resource_overlaps"][0]
    for name in ("first_interval","second_interval","interval"):
        assert overlap[name][1]=={"$nonfinite_float":"+inf"}
    assert context.check_pair(a,b) is result and context.pair_conflict_evidence[0]["count"]==2
    assert math.isinf(next(iter(context.pairs.values())).details["resource_overlaps"][0]["interval"][1])
    if context.diagnostics is not None:
        context.diagnostics.model({"a":(a,),"b":(b,)},{(a.signature,b.signature)},set(),set(),1)
        out=context.diagnostics.export(replace(cat,columns={"a":(a,),"b":(b,)}),(a,b))
        canonical(out)
        assert out["cp_models"][0]["constraints"][0]["resource_overlaps"][0]["interval"][1]=={"$nonfinite_float":"+inf"}


def test_diagnostic_infinity_encoding_is_explicit_and_nan_still_rejected():
    from terminal_marl.core.rotating_slots.stage2_campaign import canonical
    source={"interval":[-math.inf,math.inf],"finite":80.}
    saved=plain(source)
    assert saved=={"interval":[{"$nonfinite_float":"-inf"},{"$nonfinite_float":"+inf"}],"finite":80.}
    assert source["interval"]==[-math.inf,math.inf]
    canonical(saved)
    with pytest.raises(ValueError,match="NaN"): plain({"interval":[0.,math.nan]})
    with pytest.raises(ValueError): canonical({"interval":[0.,math.nan]})


def test_bounded_catalog_payload_queries_and_cp_variable_mapping():
    s,cat=fixture(); diag=CandidateDiagnostics(DiagnosticConfig(max_queries=1,max_trajectory_candidates=1,max_constraints=1))
    a,b=cat.columns["a"][1],cat.columns["b"][1]
    for row in (a,b): diag.query("PAIR",(row,),ValidationResult(False,"UNKNOWN","budget"))
    diag.model(cat.columns,{(a.signature,b.signature)},{a.signature},set(),1)
    out=diag.export(cat,(a,cat.columns["b"][0]))
    assert len(out["candidates"])==4 and len(out["trajectory_payloads"])==1
    assert len(out["validation_queries"])==1 and out["dropped"]["queries"]==1
    assert out["dropped"]["trajectory_candidates"]==1 and out["dropped"]["constraints"]==1
    assert out["cp_models"][0]["exactly_one"]["a"][1]==dict(variable="x_a_1",candidate_hash=candidate_hash(a))
    assert out["cp_models"][0]["constraints"][0]["kind"]=="PAIR_AT_MOST_ONE"
    move=next(r for r in out["candidates"] if r["candidate_hash"]==candidate_hash(a))
    assert move["node"]["time"]==40. and move["selected"]
    assert out["selected_evaluation"]["objective"]==[0.,0.,1.,70.,300.]
    json.dumps(out)


def test_probes_require_selected_baseline_and_do_not_change_catalog_or_budget(monkeypatch):
    s,cat=fixture(); baseline=(cat.columns["a"][0],cat.columns["b"][1])
    before=repr(cat); formal=PlanningBudget(solver_wall_used=1.75)
    with pytest.raises(ValueError,match="actual selected baseline"):
        probe_catalog_choices(s,cat,{"a":candidate_hash(cat.columns["a"][1])})
    monkeypatch.setattr("terminal_marl.core.rotating_slots.stage2_planning.validate_separation",
                        lambda *args: ValidationResult(True,"SAFE"))
    out=probe_catalog_choices(s,cat,{"a":candidate_hash(cat.columns["a"][1])},baseline=baseline,
                              prefix_evaluator=lambda prefix: ValidationResult(True,"SAFE"))
    assert out["batch"]["accepted"] and out["prefix"]["accepted"]
    assert out["objective_delta"]==[0.,0.,-1.,-30.,100.]
    assert out["selected"][1]["candidate_hash"]==candidate_hash(baseline[1])
    assert out["cp_calls"]==0 and out["formal_cp_budget_consumed_s"]==0.
    assert formal.solver_wall_used==1.75 and repr(cat)==before
    keep=probe_catalog_choices(s,cat,{},baseline_mode="keep",prefix_evaluator=lambda prefix: ValidationResult(True,"SAFE"))
    assert all(c["kind"]=="KEEP" for c in keep["selected"])
    assert keep["objective_delta"]==[0.]*5
    with pytest.raises(ValueError,match="outside catalog"):
        probe_catalog_choices(s,cat,{"UID14":"unknown"},baseline=baseline)


def test_probe_checks_complete_fixed_background_and_original_unknown(monkeypatch):
    s,cat=fixture(3)
    scope=PlanningScope(s.version,s.time,tuple(cat.columns),("a","b"),("c",),actual_k=2)
    cat=replace(cat,columns={k:v for k,v in cat.columns.items() if k!="c"},scope=scope)
    pairs=[]
    monkeypatch.setattr(EvaluationContext,"resources",lambda self,a,plans: ())
    def check(self,a,b):
        pairs.append((a.aircraft_id,b.aircraft_id))
        if b.aircraft_id=="c": return ValidationResult(False,"UNKNOWN","HARD_SEPARATION",{"interval":[6.,7.]})
        return ValidationResult(True,"SAFE")
    monkeypatch.setattr(EvaluationContext,"check_pair",check)
    out=probe_catalog_choices(s,cat,{"a":cat.columns["a"][1].signature},baseline_mode="keep")
    assert ("a","c") in pairs and ("b","c") in pairs
    assert out["batch"]["status"]=="UNKNOWN" and out["prefix"] is None
    assert out["batch"]["details"]["interval"]==[6.,7.]


def test_opt_in_does_not_add_validation_calls(monkeypatch):
    s,cat=fixture(); count=[]
    def check(*args): count.append(1); return ValidationResult(True,"SAFE")
    monkeypatch.setattr("terminal_marl.core.rotating_slots.stage2_planning.validate_separation",check)
    results=[]
    for config in (None,DiagnosticConfig()):
        context=EvaluationContext(s,budget=PlanningBudget(diagnostics=config))
        results.append(context.check_batch((cat.columns["a"][1],cat.columns["b"][0]))[0])
        if context.diagnostics:
            assert context.diagnostics.export(cat,(cat.columns["a"][1],cat.columns["b"][0]))["additional_validation_calls"]==0
    assert results[0]==results[1] and len(count)==2


def test_cp_observations_map_resource_cuts_and_retain_formal_selection(monkeypatch):
    pytest.importorskip("ortools")
    from terminal_marl.core.rotating_slots.resources import Reservation
    s,cat=fixture()
    cat=replace(cat,columns={aid:(rows[0],replace(rows[1],resources=(Reservation(("pad",2),aid,20.,30.),)))
                            for aid,rows in cat.columns.items()})
    monkeypatch.setattr("terminal_marl.core.rotating_slots.stage2_planning.validate_separation",
                        lambda *args: ValidationResult(True,"SAFE"))
    selected=[]
    for config in (None,DiagnosticConfig()):
        budget=PlanningBudget(soft_seconds=5.,watchdog_seconds=6.,solver_seconds=.5,diagnostics=config)
        context=EvaluationContext(s,budget=budget,prefix_evaluator=lambda p: ValidationResult(True,"SAFE"))
        result=RollingHorizonCPSATScheduler().plan(s,cat,context,budget)
        assert result.certified
        selected.append(tuple(c.signature for c in result.selected))
        if config is not None:
            out=context.diagnostics.export(cat,result.selected)
            constraint=out["cp_models"][0]["constraints"][0]
            assert constraint["kind"]=="PAIR_AT_MOST_ONE"
            assert constraint["resource_overlaps"]==[{"resource":["pad",2],"interval":[20.,30.]}]
            assert result.solver_records[0]["selected_candidate_hashes"]
            assert out["additional_cp_calls"]==0
    assert selected[0]==selected[1]


def test_callback_probe_uses_retained_actual_baseline(monkeypatch):
    s,cat=fixture(); cb=NormalSchedulerCallback(diagnostics=DiagnosticConfig())
    with pytest.raises(ValueError,match="no retained diagnostic catalog"): cb.probe_last_choices({})
    baseline=(cat.columns["a"][1],cat.columns["b"][0])
    cb.last_diagnostic_problem=dict(snapshot=s,catalog=cat,selected=baseline,global_frame=None)
    monkeypatch.setattr("terminal_marl.core.rotating_slots.stage2_planning.validate_separation",
                        lambda *args: ValidationResult(True,"SAFE"))
    out=cb.probe_last_choices({},prefix_evaluator=lambda p: ValidationResult(True,"SAFE"))
    assert [r["candidate_hash"] for r in out["selected"]]==[candidate_hash(c) for c in baseline]
    assert out["objective_delta"]==[0.]*5


def test_probe_reports_actual_shared_queue_before_after_including_fixed_demand(monkeypatch):
    from terminal_marl.core.rotating_slots.stage2_downstream import Demand,JointDownstream
    s,cat=fixture()
    def demand(aid,*,full=False):
        return Demand(aid,full,150 if full else 0,30,1000,0 if full else 10,
                      1000000,0,0,0,285 if full else 135,"outer",{},
                      ((0,150,300,aid),) if full else ())
    demands={c.signature:demand(aid,full=aid=="a" and c.kind=="MOVE")
             for aid,rows in cat.columns.items() for c in rows}
    queue=JointDownstream(demands,((0,50,150,"ground"),),("a","b","upstream_request"),pad_count=1,
                          fixed_demands=(demand("upstream_request"),),
                          assignment={"a":0,"b":0,"upstream_request":0})
    cat=replace(cat,downstream=queue); baseline=tuple(rows[0] for rows in cat.columns.values())
    monkeypatch.setattr("terminal_marl.core.rotating_slots.stage2_planning.validate_separation",
                        lambda *args: ValidationResult(True,"SAFE"))
    out=probe_catalog_choices(s,cat,{"a":candidate_hash(cat.columns["a"][1])},baseline=baseline,
                              prefix_evaluator=lambda p: ValidationResult(True,"SAFE"))
    assert out["baseline_integer_objective"]==list(queue.evaluate(baseline)[0])
    assert out["baseline_downstream"]["b"]["tail_wait_seconds"]==15.
    assert out["downstream"]["b"]["tail_wait_seconds"]==0.
    assert "upstream_request" in out["baseline_downstream"] and "upstream_request" in out["downstream"]
    assert out["objective_delta"]==[b-a for a,b in zip(out["baseline_objective"],out["objective"])]
    json.dumps(out)
