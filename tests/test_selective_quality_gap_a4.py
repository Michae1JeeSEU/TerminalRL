"""Bounded A4 artifact/protocol tests; no solver or physical world is run."""
from dataclasses import dataclass
import json
import math
from types import MappingProxyType, SimpleNamespace as NS
from contextlib import contextmanager
import os
from pathlib import Path

import pytest

from scripts.stage3 import run_selective_quality_gap_a4 as a4


@dataclass(frozen=True)
class Plan:
    trajectory_hash: str
    start_time: float = 20.
    pad_id: int | None = None


@dataclass(frozen=True)
class Column:
    aircraft_id: str
    kind: str
    plans: tuple = ()
    objective: tuple = (0, 0, 0, 0, 0)
    downstream: object = None
    node: object = None

    @property
    def signature(self): return self.aircraft_id, self.kind, tuple(p.trajectory_hash for p in self.plans)


def test_fixed_catalog_hash_stable_across_order_and_augmentation():
    a, b = Column("a", "KEEP"), Column("a", "MOVE", (Plan("p"),))
    original = NS(columns={"a": (a, b)})
    assert a4.catalog_hash(original) == a4.catalog_hash(NS(columns={"a": (b, a)}))
    assert a4.catalog_hash(original) != a4.catalog_hash(NS(columns={"a": (a,)}))


def test_inf_serialization_mappingproxy_and_exclusive_output(tmp_path):
    path = tmp_path / "R1" / "baseline" / "evidence.json"
    a4.write_new(path, {"reservation": MappingProxyType({"end": math.inf}), "negative": -math.inf})
    row = json.loads(path.read_text())
    assert row["reservation"]["end"] == {"$nonfinite_float": "+inf"}
    assert row["negative"] == {"$nonfinite_float": "-inf"}
    with pytest.raises(FileExistsError): a4.write_new(path, {})
    a4.write_new(tmp_path / "R4" / "baseline" / "evidence.json", {"separate": True})
    assert json.loads(path.read_text()) == row
    with pytest.raises(ValueError): a4.finite_view(float("nan"))


def test_certificate_json_excludes_executor_private_prepared_calendars():
    from terminal_marl.core.rotating_slots.world import SubmissionEvaluation
    certificate=SubmissionEvaluation(True, "ACCEPTED", "", {}, 20., "snap", (), (), False, object())
    row=a4.finite_view(certificate)
    assert row["accepted"] is True and row["snapshot_version"] == "snap"
    assert "_prepared" not in row
    json.dumps(row, allow_nan=False)


def test_r1_target_filter_and_one_per_pad_upper_bound(monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_planning as planning
    aid = "initial_inner_02"; calls=[]
    monkeypatch.setattr(planning, "_node", lambda *args: NS(ring="inner", time=20.))
    class References:
        def generate(self, snapshot, a, node, target, start, duration, budget):
            calls.append((target, duration)); return True, Plan("pad"+str(target), start, target)
    provider = NS(references=References(), _column=lambda snapshot, a, plans, *args: Column(a.id, "MOVE", plans))
    problem = dict(snapshot=NS(time=20., aircraft={aid: NS(id=aid)}), catalog=NS(columns={aid: ()}))
    monkeypatch.setattr(a4, "accept_addition", lambda problem, column, context, observations: True)
    budget=NS(available=lambda aid: True, screening={}, attempts={})
    additions, evidence=a4.generate_additions(problem, provider, NS(), budget, "R1")
    assert len(additions) == 2
    assert [c.plans[-1].pad_id for c in additions] == [0, 1]
    assert calls == [(0, 20), (1, 20)]
    assert [row["retained"] for row in evidence["attempts"]] == [1, 1]


def test_r1_pad_attempts_round_robin_before_next_duration(monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_planning as planning
    aid="initial_inner_02"; calls=[]
    monkeypatch.setattr(planning, "_node", lambda *args: NS(ring="inner", time=20.))
    def generate(snapshot, a, node, target, start, duration, budget):
        calls.append((target, duration))
        return True, None if duration == 20 else Plan("pad"+str(target), start, target)
    provider=NS(references=NS(generate=generate), _column=lambda snapshot, a, plans, *args: Column(a.id, "MOVE", plans))
    monkeypatch.setattr(a4, "accept_addition", lambda *args: True)
    problem=dict(snapshot=NS(time=20., aircraft={aid: NS(id=aid)}), catalog=NS(columns={aid: ()}))
    additions, _=a4.generate_additions(problem, provider, NS(), NS(available=lambda aid: True, screening={}, attempts={}), "R1")
    assert calls == [(0, 20), (1, 20), (0, 25), (1, 25)]
    assert len(additions) == 2


def test_single_valid_column_retained_when_actual_combination_conflicts():
    old=Column("a", "KEEP"); extra=Column("a", "MOVE", (Plan("new", pad_id=0),))
    problem=dict(catalog=NS(columns={"a": (old,)}), selected=(old,), snapshot=NS(time=20.))
    single=NS(accepted=True, status="SAFE", reason="", details={})
    batch=NS(accepted=False, status="REJECTED", reason="PAIR_CONFLICT", details={})
    context=NS(check_single=lambda column: single, check_batch_conflicts=lambda selected: (batch, ((old, extra),)))
    observations=[]
    assert a4.accept_addition(problem, extra, context, observations) is True
    row=observations[0]
    assert row["retained_as_candidate"] is True
    assert row["baseline_combination_certified"] is False
    assert row["candidate_status"] == "TENTATIVE_SINGLE_VALID_COLUMN"
    assert row["batch"]["reason"] == "PAIR_CONFLICT"


def test_uncertified_fixed_solver_result_cannot_count_as_improvement(monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_planning
    problem=fixed_problem()
    result=NS(status="UNKNOWN", objective=(0, -10, 0, 0, 0), certified=False, evaluation=None,
              prefix=(), selected=problem["selected"], solver_records=())
    class Scheduler:
        def plan(self, *args): return result
    monkeypatch.setattr(stage2_planning, "RollingHorizonCPSATScheduler", Scheduler)
    monkeypatch.setattr(a4, "offline_context", lambda *args, **kwargs: (NS(), NS(settings=lambda: {}, solver_wall_used=0.)))
    monkeypatch.setattr(a4, "export_catalog", lambda *args: {})
    row=a4.solve_fixed(problem, 8.)
    assert row["five_dimensional_key"] == result.objective
    assert row["final_combination_certified"] is False
    assert row["improvement_evidence_eligible"] is False
    assert row["objective_is_descriptive_only"] is True


def test_r1_no_generation_outside_fixed_scope(monkeypatch):
    calls=[]
    provider=NS(references=NS(generate=lambda *args: calls.append(args)))
    additions, evidence=a4.generate_additions(dict(catalog=NS(columns={}), snapshot=NS()), provider, NS(), NS(), "R1")
    assert not additions and not calls
    assert evidence["status"] == "TARGET_OUTSIDE_BASELINE_GROUP"


def test_r1_rejects_generator_returning_another_pad(monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_planning as planning
    aid="initial_inner_02"
    monkeypatch.setattr(planning, "_node", lambda *args: NS(ring="inner", time=20.))
    provider=NS(references=NS(generate=lambda *args: (True, Plan("wrong", pad_id=3))))
    problem=dict(catalog=NS(columns={aid: ()}), snapshot=NS(aircraft={aid: NS(id=aid)}))
    with pytest.raises(ValueError, match="another pad"):
        a4.generate_additions(problem, provider, NS(), NS(available=lambda aid: True), "R1")


def fixed_problem():
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateCatalog, PredictionColumn
    from terminal_marl.core.rotating_slots.stage2_downstream import Demand
    from terminal_marl.core.rotating_slots.selective_replanning.scope import PlanningScope
    from terminal_marl.core.rotating_slots.selective_replanning.frame import GlobalEvaluationFrame
    demand=Demand("background", False, 200, 150, 0, 1, 100, 1, 0, 0, 335, "inner", MappingProxyType({"frozen": True}))
    frame=GlobalEvaluationFrame("snapshot", 20., (demand,), (), ("background",), (("background", 2),), 6)
    scope=PlanningScope("snapshot", 20., ("a",), ("a",), ("background",), actual_k=1, objective_frame_id=frame.identity)
    keep=PredictionColumn("a", "KEEP", (), (), (0, 0, 0, 0, 0))
    from terminal_marl.core.rotating_slots.stage2_downstream import Demand as D
    a_demand=D("a", False, 200, 150, 0, 1, 100, 1, 0, 0, 335, "inner", {})
    # a needs a frozen virtual pad too, while background stays fixed.
    frame=GlobalEvaluationFrame("snapshot", 20., (demand, a_demand), (), ("a", "background"), (("a", 0), ("background", 2)), 6)
    scope=PlanningScope("snapshot", 20., ("a",), ("a",), ("background",), actual_k=1, objective_frame_id=frame.identity)
    downstream=frame.downstream({keep.signature: a_demand}, {"a"})
    catalog=CandidateCatalog(MappingProxyType({"a": (keep,)}), {}, {}, (), downstream=downstream, scope=scope,
                             fixed_background=MappingProxyType(dict(downstream.fixed_demands)))
    return dict(snapshot=None, catalog=catalog, global_frame=frame, selected=(keep,))


def test_safe_archive_roundtrip_fixed_hash_and_background(tmp_path):
    problem=fixed_problem(); path=tmp_path / "problem.json"
    a4.write_new(path, a4.pack_problem(problem, {"references": [], "inf": math.inf}))
    restored, state=a4.unpack_problem(path)
    assert a4.catalog_hash(restored["catalog"]) == a4.catalog_hash(problem["catalog"])
    assert restored["catalog"].scope == problem["catalog"].scope
    assert restored["global_frame"].identity == problem["global_frame"].identity
    assert restored["catalog"].downstream.fixed_demands == problem["catalog"].downstream.fixed_demands
    assert state["inf"] == math.inf


def test_safe_archive_preserves_real_snapshot_and_config_identity(tmp_path):
    from terminal_marl.core.rotating_slots.config import load_service_profile
    from terminal_marl.core.rotating_slots.world import PlanningSnapshot
    from terminal_marl.core.rotating_slots.planning_latency import PlanningTiming
    problem=fixed_problem(); empty=MappingProxyType({})
    problem["snapshot"]=PlanningSnapshot(20., "snapshot", load_service_profile("service_half_v1"), empty, empty, empty,
        empty, (), (), empty, empty, empty, empty, planning_timing=PlanningTiming.from_lead(0., 20., 1.))
    path=tmp_path/"snapshot.json"; a4.write_new(path, a4.pack_problem(problem))
    restored, _=a4.unpack_problem(path)
    assert restored["snapshot"] == problem["snapshot"]
    assert restored["snapshot"].config.config_hash == problem["snapshot"].config.config_hash
    assert restored["snapshot"].planning_timing.effective_time == 20.


def test_augmentation_retains_all_old_columns_and_complete_background(monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_downstream
    from terminal_marl.core.rotating_slots.stage2_planning import PredictionColumn
    problem=fixed_problem(); original=problem["catalog"]; before=a4.catalog_hash(original)
    extra=PredictionColumn("a", "WAIT", (), (), (0, 0, 0, 0, 0))
    demand=next(iter(original.downstream.demands.values()))
    monkeypatch.setattr(stage2_downstream, "describe_column", lambda *args: demand)
    augmented=a4.rebuild_catalog(problem, [extra], NS(references=None), NS())
    assert original.scope == augmented.scope
    assert augmented.fixed_background == original.fixed_background
    assert len(augmented.columns["a"]) == 2
    assert {c.signature for c in original.columns["a"]} <= {c.signature for c in augmented.columns["a"]}
    assert a4.catalog_hash(original) == before
    with pytest.raises(ValueError, match="outside fixed scope"):
        a4.rebuild_catalog(problem, [replace_aid(extra, "background")], NS(), NS())


def replace_aid(column, aid):
    from dataclasses import replace
    return replace(column, aircraft_id=aid)


def test_continuation_remains_explicit_reserved_inputs():
    row=a4.continuation_inputs()
    assert row["status"] == "EXTERNAL_ENTRYPOINT_NOT_AUTO_RUN"
    assert row["entrypoint"] == "scripts/stage3/run_selective_short_continuation_a4.py"
    assert row["duration_seconds"] == 120.
    assert row["publication_or_physics_authorized_by_this_entrypoint"] is False
    assert a4.POLICY["cap_child_wall_s"] == 7200.
    assert a4.POLICY["prior_audits_child_wall_s"] == 0.
    assert a4.POLICY["historical_campaign_used_s"] == 1374.067592
    assert a4.POLICY["run_role"] == "SHARED_A4_CAMPAIGN"


def test_default_reference_switches_are_explicitly_false():
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    callback=a4.new_callback(NS(NormalSchedulerCallback=NormalSchedulerCallback))
    assert callback.search_policy.to_dict() == dict(pad_anchor_priority=False, target_coverage_order=False,
                                                   ordinary_immediate_filter=False)
    assert callback.selective_rule_variant == "opportunity_reference_v4"


def test_fixed_solver_contexts_have_equal_limits_except_solver_allocation(monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_planning
    captured=[]
    class Context:
        def __init__(self, snapshot, **kwargs):
            self.snapshot_key="bound"; self.options=kwargs; captured.append(self)
    monkeypatch.setattr(stage2_planning, "EvaluationContext", Context)
    problem=fixed_problem()
    first, b8=a4.offline_context(problem, solver_seconds=8.)
    second, b24=a4.offline_context(problem, solver_seconds=24.)
    s8, s24=b8.settings(), b24.settings()
    assert s8.pop("solver_seconds") == 8.
    assert s24.pop("solver_seconds") == 24.
    assert s8 == s24
    assert b8.watchdog_seconds == b24.watchdog_seconds == 60.
    assert first.options["validation_cache"] == second.options["validation_cache"] == {}
    assert first.options["validation_cache"] is not second.options["validation_cache"]
    assert first.options["scope"] is second.options["scope"]


@pytest.mark.parametrize("entry", ("diagnostic", "short", "common"))
@pytest.mark.parametrize("launch_value", ("1", None, "2"))
def test_a4_cli_real_cpu_preflight_uses_environment_before_dependencies(monkeypatch, tmp_path, entry, launch_value):
    """Exercise real old.set_cpu; imports cannot repair missing launch settings."""
    from scripts.stage3 import run_selective_common_initial_a4 as common
    from scripts.stage3 import run_selective_short_continuation_a4 as short
    from scripts.stage3 import run_selective_reoptimization_repair as old
    class PassedPreflight(Exception): pass
    def stop(*args, **kwargs): raise PassedPreflight()
    for key in a4.base.THREADS:
        monkeypatch.setenv(key, "1")
    target = "OPENBLAS_NUM_THREADS"
    if launch_value is None: monkeypatch.delenv(target)
    else: monkeypatch.setenv(target, launch_value)
    probes, affinity, created = [], [], []
    real_dependencies = a4.base.dependencies
    def dependencies(*args):
        # Call the actual dependency importer, which sets thread variables.
        d = real_dependencies(*args)
        monkeypatch.setattr(d.campaign, "resource_probe",
            lambda: probes.append(True) or {"free_physical_cores": [12, 14]})
        created.append(d)
        return d
    monkeypatch.setattr(a4.base, "dependencies", dependencies)
    monkeypatch.setattr(old.os, "sched_setaffinity", lambda pid, cores: affinity.append((pid, cores)), raising=False)
    source = dict(source_hash="frozen", commit="test", files={
        "scripts/stage3/run_selective_quality_gap_a4.py": "test", "tests/test_selective_quality_gap_a4.py": "test",
        "scripts/stage3/run_selective_short_continuation_a4.py": "test", "tests/test_selective_short_continuation_a4.py": "test",
        "scripts/stage3/run_selective_common_initial_a4.py": "test"})
    freeze = tmp_path/"source.json"
    def load(path):
        if Path(path) == freeze: return source
        raise PassedPreflight()
    monkeypatch.setattr(a4.base, "load", load)
    monkeypatch.setattr(a4.base, "verify_sources", lambda source: None)
    monkeypatch.setattr(a4, "validated_output", lambda path: path)
    monkeypatch.setattr(a4, "baseline", stop)
    monkeypatch.setattr(short, "load_inputs", stop)
    jobs = {"R1_baseline": {}, "short_continuation:R4:prepare": {}, "common_initial:s3r23": {}}
    for job in jobs.values(): job.update(status="running", parent_pid=os.getppid())
    ledger = dict(jobs=jobs)
    @contextmanager
    def transaction(*args, **kwargs): yield ledger
    monkeypatch.setattr(a4, "ledger_transaction", transaction)
    output = tmp_path/"output"
    if entry == "diagnostic":
        main, args = a4.main, ["baseline", "--case", "R1", "--_child-job-id", "R1_baseline", "--cpu", "14"]
    elif entry == "short":
        main, args = short.main, ["--stage", "prepare-child", "--case", "R4", "--cpu", "12"]
    else:
        main, args = common.main, ["--stage", "initialize-child", "--root", "s3r23", "--cpu", "12",
            "--origin-output", str(tmp_path/"origin")]
    args += ["--output", str(output), "--source-freeze", str(freeze)]
    if launch_value == "1":
        with pytest.raises(PassedPreflight): main(args)
        assert probes == [True] and affinity == [(0, {14 if entry == "diagnostic" else 12})]
        assert len(created) == 1
        d = created[0]
        assert d.launch_threads == {key: "1" for key in a4.base.THREADS}
    else:
        with pytest.raises(ValueError, match="one numerical thread at launch required"): main(args)
        assert len(created) == 1
        d = created[0]
        assert d.launch_threads[target] == launch_value
        assert not probes and not affinity and not output.exists()
