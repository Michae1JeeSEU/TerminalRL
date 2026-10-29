"""Fixed recipes and isolated no-physics interfaces; never run actual P2."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from types import SimpleNamespace as NS

import pytest

from scripts.stage3 import run_selective_fixed_groups_a4 as runner
from scripts.stage3 import run_selective_quality_gap_a4 as a4
from terminal_marl.core.rotating_slots.selective_replanning import PlanningScope, RuleSelector
from terminal_marl.core.rotating_slots.selective_replanning.relations import Relation


def scope_fixture(monkeypatch, extra=(), mandatory=(), fairness=()):
    ids = tuple(sorted(set().union(*map(set, runner.RECIPES.values())) | set(extra)))
    snapshot = NS(version="fixed", time=255., aircraft={aid: object() for aid in ids})
    monkeypatch.setattr("terminal_marl.core.rotating_slots.stage2_planning._decision_aircraft",
        lambda s: tuple(NS(id=aid) for aid in ids))
    opened = tuple(sorted(set(runner.RECIPES["control"]) | set(mandatory)))
    scope = PlanningScope(snapshot.version, snapshot.time, ids, opened,
        tuple(sorted(set(ids)-set(opened))), mandatory_ids=tuple(mandatory), actual_k=len(opened),
        objective_frame_id="frame")
    return snapshot, scope, dict(fairness_selected_ids=list(fairness))


def test_three_recipes_are_distinct_eight_and_replace_only_two_ordinary_members():
    recipes = [set(ids) for ids in runner.RECIPES.values()]
    common = {"initial_middle_01", "initial_middle_02", "initial_outer_01", "initial_outer_03",
        "initial_outer_05", "initial_outer_10"}
    assert len({tuple(sorted(ids)) for ids in recipes}) == 3
    assert all(len(ids) == 8 and common <= ids for ids in recipes)
    assert recipes[0]-recipes[1] == recipes[0]-recipes[2] == {"initial_outer_04", "initial_outer_08"}
    assert recipes[1]-recipes[0] == {"initial_middle_03", "initial_middle_05"}
    assert recipes[2]-recipes[0] == {"initial_middle_07", "initial_middle_08"}


@pytest.mark.parametrize("group", tuple(runner.RECIPES))
def test_explicit_scope_preserves_hard_fair_and_complete_background(monkeypatch, group):
    snapshot, original, selection = scope_fixture(monkeypatch, mandatory=("initial_middle_01",),
        fairness=("initial_middle_01", "initial_middle_02"))
    applied, evidence = runner.fixed_scope(original, selection, snapshot, (), group)
    assert set(applied.replan_ids) == set(runner.RECIPES[group]) and applied.actual_k == 8
    assert set(applied.mandatory_ids) <= set(applied.replan_ids)
    assert set(applied.replan_ids).isdisjoint(applied.fixed_background_ids)
    assert set(applied.replan_ids) | set(applied.fixed_background_ids) == set(snapshot.aircraft)
    assert evidence["original_selection"] is selection and evidence["status"] == "APPLICABLE"


def test_definite_dependency_closure_matches_existing_semantics_and_keeps_background(monkeypatch):
    snapshot, original, selection = scope_fixture(monkeypatch, extra=("necessary", "approximate"))
    edges = (Relation("initial_middle_01", "necessary", "committed_release", True, False),
        Relation("initial_middle_01", "approximate", "virtual_pad_competition", False, False))
    scope, evidence = runner.fixed_scope(original, selection, snapshot, edges, "control")
    assert scope.actual_k == 9 and "necessary" in scope.replan_ids
    assert "approximate" in scope.fixed_background_ids
    assert evidence["dependency_added_ids"] == ["necessary"]


@pytest.mark.parametrize("violation", ("mandatory", "fairness", "ineligible", "closure"))
def test_unsafe_or_inapplicable_recipe_is_not_replaced_with_another_group(monkeypatch, violation):
    extra = tuple("extra"+str(i) for i in range(5))
    snapshot, original, selection = scope_fixture(monkeypatch, extra=extra,
        mandatory=(extra[0],) if violation == "mandatory" else ())
    relations = ()
    if violation == "fairness": selection["fairness_selected_ids"] = ["initial_middle_03"]
    if violation == "ineligible": original = replace(original,
        eligible_ids=tuple(a for a in original.eligible_ids if a != "initial_middle_01"),
        replan_ids=tuple(a for a in original.replan_ids if a != "initial_middle_01"), actual_k=7,
        fixed_background_ids=original.fixed_background_ids+("initial_middle_01",))
    if violation == "closure": relations = tuple(Relation("initial_middle_01", aid, "required", True, True) for aid in extra)
    with pytest.raises(runner.GroupNotApplicable) as caught:
        runner.fixed_scope(original, selection, snapshot, relations, "control")
    assert caught.value.evidence["status"] == "NOT_APPLICABLE"
    assert caught.value.evidence["requested_group"] == list(runner.RECIPES["control"])


@pytest.mark.parametrize("exit_kind", ("normal", "exception", "not_applicable"))
def test_temporary_selector_patch_always_restores_original_method(monkeypatch, exit_kind):
    snapshot, scope, selection = scope_fixture(monkeypatch, extra=("hard",),
        mandatory=("hard",) if exit_kind == "not_applicable" else ())
    calls = []
    def original(self, s, frame, history, relations=(), **kwargs):
        calls.append((s, frame, history, relations))
        return scope, selection
    monkeypatch.setattr(RuleSelector, "select", original)
    observations = []
    try:
        with runner.selector_intervention("alternative_a", observations):
            applied, info = RuleSelector().select(snapshot, "frame", "history")
            assert set(applied.replan_ids) == set(runner.RECIPES["alternative_a"])
            assert info["fixed_group_intervention"] is observations[0]
            if exit_kind == "exception": raise RuntimeError("fixture")
    except (RuntimeError, runner.GroupNotApplicable):
        assert exit_kind != "normal"
    assert RuleSelector.select is original and len(calls) == 1 and len(observations) == 1


def test_source_manifest_requires_identical_core_config_and_contracts():
    def source(files): return dict(files=files, source_hash=runner.base.digest(files))
    files = {"terminal_marl/core/x.py": "core", "configs/stage3/x.yaml": "config", "new10.16/contract.yaml": "contract",
        "scripts/stage3/run_selective_quality_gap_a4.py": "old"}
    changed = dict(files, **{"scripts/stage3/run_selective_fixed_groups_a4.py": "new", "tests/test_selective_fixed_groups_a4.py": "new"})
    differences = runner.source_differences(source(files), source(changed))
    assert {row["path"] for row in differences} == {"scripts/stage3/run_selective_fixed_groups_a4.py", "tests/test_selective_fixed_groups_a4.py"}
    for protected in tuple(files)[:3]:
        wrong = dict(files); wrong[protected] = "changed"
        with pytest.raises(ValueError, match="core/config/contract"):
            runner.source_differences(source(files), source(wrong))
        wrong.pop(protected)
        with pytest.raises(ValueError, match="core/config/contract"):
            runner.source_differences(source(files), source(wrong))


def test_group_comparison_counts_actual_first_and_final_convergence():
    current = dict(source_hash="same", canonical_frame_sha256="frame", group="alternative_b",
        first_group=["b"], final_group=["final"])
    saved = [dict(current, group="control", first_group=["c"]),
        dict(current, group="alternative_a", first_group=["a"]), dict(current, source_hash="other", group="not_comparable")]
    result = runner.compare_groups(current, saved)
    assert result["distinct_first_group_count"] == 3 and result["distinct_final_group_count"] == 1
    assert result["same_first_group_as"] == [] and result["same_final_group_as"] == ["control", "alternative_a"]


def tiny_inputs():
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    from terminal_marl.core.rotating_slots.planning_latency import PlanningTiming, project_visible_snapshot
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, EvaluationContext, NormalSchedulerCallback
    from terminal_marl.core.rotating_slots.selective_replanning import build_global_frame
    world = RotatingSlotWorld()
    ids = tuple(sorted(set().union(*map(set, runner.RECIPES.values()))))
    for slot, aid in enumerate(ids): world.seed_hold(aid, "outer", slot, .9)
    source = world.build_planning_snapshot()
    future = project_visible_snapshot(source, 20., timing=PlanningTiming.from_lead(0., 20., 1.))
    provider = CandidateProvider(planning_profile="downstream_service_v8")
    frame = build_global_frame(future, provider, EvaluationContext(future))
    callback = a4.new_callback(NS(NormalSchedulerCallback=NormalSchedulerCallback), frame=frame)
    callback.initialize_imported_history(source, events=world.events)
    return world, source, dict(snapshot=future, global_frame=frame), dict(
        history=callback.selective_history.state_dict(), blockers=[])


def test_tiny_real_callback_keeps_full_world_and_writes_readable_no_physics_output(monkeypatch, tmp_path):
    from terminal_marl.core.rotating_slots.stage2_planning import (
        CandidateProvider, NormalSchedulerCallback, RollingHorizonCPSATScheduler, PlanningResult)
    class KeepProvider(CandidateProvider):
        def iter_progressive_columns(self, snapshot, budget, context, *, scope=None, early_publish=False):
            rows = {aid: [self._column(snapshot, snapshot.aircraft[aid], (), kind, context) for kind in ("KEEP", "WAIT")]
                for aid in scope.replan_ids}
            catalog = self._progressive_catalog(snapshot, budget, context, rows,
                {aid: values[0].reference for aid,values in rows.items()}, [snapshot.aircraft[aid] for aid in rows])
            yield catalog
            yield catalog
    def factory(*args, **kwargs):
        kwargs["candidate_provider"] = KeepProvider(planning_profile="downstream_service_v8")
        return NormalSchedulerCallback(*args, **kwargs)
    world, source, problem, history = tiny_inputs()
    original = world.build_planning_snapshot()
    original_select = RuleSelector.select
    contexts = []
    def fixed_choice(self, snapshot, catalog, context, budget, *, incumbent=None):
        contexts.append(context)
        selected = tuple(values[0] for values in catalog.columns.values())
        checked, _ = context.check_batch(selected)
        assert checked.accepted
        evaluation = context.check_commit_prefix(selected)
        assert evaluation.accepted
        return PlanningResult(selected, (), evaluation, "FIXTURE_KEEP_ONLY", catalog.downstream.key(selected), certified=True)
    monkeypatch.setattr(RollingHorizonCPSATScheduler, "plan", fixed_choice)
    binding = dict(source_hash="tiny", source_commit="tiny")
    directory = tmp_path/"control"
    result = runner.run_group(problem, source, history, "control", NS(NormalSchedulerCallback=factory), directory, binding)
    assert result["status"] == "COMPLETE", result
    assert result["NO_PHYSICS"] and result["actual_submitted"] == 0 and result["immutable_inputs_unchanged"]
    assert result["first_group"] == result["final_group"] == sorted(runner.RECIPES["control"])
    assert result["solve_calls"] <= 2 and result["scope_expansions"] == 0
    assert result["planning_budget"]["candidate_seconds"] == 6.
    assert result["planning_budget"]["solver_seconds"] == 8.
    assert all(set(context.snapshot.aircraft) == set(source.aircraft) for context in contexts)
    assert all(set(context.decisions) == set(runner.RECIPES["control"]) for context in contexts)
    assert all(set(context.scope.fixed_background_ids) == set(source.aircraft)-set(runner.RECIPES["control"]) for context in contexts)
    assert RuleSelector.select is original_select and world.build_planning_snapshot() == original and world.time == 0.
    assert not world.active and not world.soft
    diagnostic = json.loads((directory/"diagnostic.json").read_text())
    assert diagnostic["catalog_export"] and diagnostic["callback_records"]
    archived, _ = a4.unpack_problem(directory/"problem.json")
    assert set(archived["catalog"].columns) == set(runner.RECIPES["control"])
    assert json.loads((directory/"result.json").read_text())["publication_permitted"] is False
    with pytest.raises(FileExistsError): runner.run_group(problem, source, history, "control", NS(), directory, binding)


def test_publication_sink_never_sets_deferred_proposal():
    from terminal_marl.core.rotating_slots.models import ValidationResult
    source, future = NS(), NS(version="future")
    port = runner.no_publication_port(future, source, 100.)
    certificate = NS(snapshot_version="future", accepted=True, refinement_required=False, cancel=(), plans=())
    result = port.commit_validated_prefix(certificate, snapshot_version="future")
    assert isinstance(result, ValidationResult) and result.status == "DIAGNOSTIC_ONLY"
    assert port.proposal is None and port.prefix_observations == [[]]
    assert port.planning_wall_deadline == 119. and port.search_wall_deadline == 115.


def test_parent_uses_separate_source_key_and_one_300_second_group_child(monkeypatch, tmp_path):
    source = dict(source_hash="groups", commit="test", files={
        runner.DRIVER.relative_to(runner.PROJECT).as_posix(): "x", "tests/test_selective_fixed_groups_a4.py": "y"})
    freeze = tmp_path/"source.json"; freeze.write_text(json.dumps(source))
    ledger = dict(a4_source_hash="keep_diagnostic_v8", jobs={})
    @contextmanager
    def transaction(*args, **kwargs): yield ledger
    def bounded(output, job_id, command, **kwargs):
        assert job_id == "fixed_group_P2_alternative_b" and kwargs == {"cap_s": 300.}
        assert command[command.index("--group")+1] == "alternative_b"
        ledger["jobs"][job_id] = {"status": "complete"}
        return {"status": "complete"}
    monkeypatch.setattr(a4, "validated_output", lambda path: path)
    monkeypatch.setattr(runner.base, "verify_sources", lambda source: None)
    monkeypatch.setattr(a4, "ledger_transaction", transaction)
    monkeypatch.setattr(a4, "run_bounded_job", bounded)
    runner.main(["--group", "alternative_b", "--output", str(tmp_path/"output"), "--source-freeze", str(freeze), "--cpu", "14"])
    assert ledger["a4_source_hash"] == "keep_diagnostic_v8"
    assert ledger["group_diagnostic_source_hash"] == "groups"
