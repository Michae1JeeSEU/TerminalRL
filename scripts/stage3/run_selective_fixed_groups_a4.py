"""One preregistered P2 group diagnostic, with no execution or publication.

The three new groups share the archived frame, cold provider and callback
timing convention. Historical baseline total wall time is a different scope.
This process-local selector intervention is never an online policy variant.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage3 import run_selective_quality_gap_a4 as a4
from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_reoptimization_repair as old

DRIVER = Path(__file__).resolve()
RECIPES = {
    "control": ("initial_middle_01", "initial_middle_02", "initial_outer_01", "initial_outer_03",
        "initial_outer_04", "initial_outer_05", "initial_outer_08", "initial_outer_10"),
    "alternative_a": ("initial_middle_01", "initial_middle_02", "initial_middle_03", "initial_middle_05",
        "initial_outer_01", "initial_outer_03", "initial_outer_05", "initial_outer_10"),
    "alternative_b": ("initial_middle_01", "initial_middle_02", "initial_middle_07", "initial_middle_08",
        "initial_outer_01", "initial_outer_03", "initial_outer_05", "initial_outer_10"),
}
ROLE = "OFFLINE_FIXED_GROUP_DIAGNOSTIC"
CHILD_CAP = 300.


class GroupNotApplicable(ValueError):
    def __init__(self, reason, evidence):
        super().__init__(reason)
        self.evidence = dict(evidence, status="NOT_APPLICABLE", reason=reason)


def source_differences(baseline, current):
    """Core, configurations and contracts must match in presence and bytes."""
    for source in (baseline, current):
        base.require(source["source_hash"] == base.digest(source["files"]), "source manifest hash differs")
    differences = []
    for path in sorted(set(baseline["files"]) | set(current["files"])):
        left, right = baseline["files"].get(path), current["files"].get(path)
        if path in baseline["files"] and path in current["files"] and left == right:
            continue
        protected = path.startswith(("terminal_marl/", "configs/", "new10.16/"))
        base.require(not protected, "fixed-group core/config/contract source differs: "+path)
        base.require(path.startswith(("scripts/", "tests/", "docs/rotating_slots_stage3/")),
            "unclassified fixed-group source difference: "+path)
        differences.append(dict(path=path, baseline_sha256=left, current_sha256=right))
    return differences


def fixed_scope(original, selection, snapshot, relations, group):
    """Apply only a declared recipe and the existing definite-edge closure."""
    requested = RECIPES[group]
    evidence = dict(group=group, requested_group=list(requested), original_scope=original.to_dict(),
        original_selection=selection,
        original_fairness_selected_ids=list(selection.get("fairness_selected_ids", ())),
        original_mandatory_ids=list(original.mandatory_ids), relations=[vars(r) for r in relations])
    def require(condition, reason):
        if not condition: raise GroupNotApplicable(reason, evidence)
    require(len(requested) == len(set(requested)) == 8, "RECIPE_NOT_EIGHT_DISTINCT_IDS")
    eligible = set(original.eligible_ids)
    require(set(requested) <= eligible, "RECIPE_IDS_NOT_ELIGIBLE")
    require(set(selection.get("fairness_selected_ids", ())) <= set(requested), "RECIPE_DROPS_FAIRNESS_MEMBER")
    opened, frontier = set(requested), list(requested)
    while frontier:
        current = frontier.pop()
        for relation in relations:
            if (relation.source == current and relation.definite and relation.target in eligible
                    and relation.target not in opened):
                opened.add(relation.target)
                frontier.append(relation.target)
    evidence.update(dependency_added_ids=sorted(opened-set(requested)), closed_group=sorted(opened))
    require(set(original.mandatory_ids) <= opened, "RECIPE_DROPS_HARD_MANDATORY")
    require(len(opened) <= original.max_k == 12, "NECESSARY_DEPENDENCY_CLOSURE_EXCEEDS_MAX12")
    require(original.status == "LOCAL", "ORIGINAL_SELECTOR_NOT_LOCAL")
    reasons = dict(original.selection_reasons)
    scope = replace(original, replan_ids=tuple(sorted(opened)), actual_k=len(opened),
        fixed_background_ids=tuple(sorted(set(snapshot.aircraft)-opened)),
        selection_reasons=tuple((aid, tuple(reasons.get(aid, ()))+
            (("FIXED_GROUP_RECIPE_"+group.upper(),) if aid in requested else ("FIXED_GROUP_DEFINITE_CLOSURE",)))
            for aid in sorted(opened)))
    scope.validate(snapshot)
    evidence.update(status="APPLICABLE", applied_scope=scope.to_dict())
    return scope, evidence


@contextmanager
def selector_intervention(group, observations):
    """Restore the class method on every normal, rejected and exception exit."""
    from terminal_marl.core.rotating_slots.selective_replanning.selectors import RuleSelector
    original = RuleSelector.select
    def select(self, snapshot, frame, history, relations=(), **kwargs):
        base.require(not observations, "one fixed-group selection per child required")
        scope, selection = original(self, snapshot, frame, history, relations, **kwargs)
        try:
            scope, evidence = fixed_scope(scope, selection, snapshot, relations, group)
        except GroupNotApplicable as exc:
            observations.append(exc.evidence)
            raise
        observations.append(evidence)
        return scope, dict(selection, actual_k=scope.actual_k, fixed_group_intervention=evidence)
    RuleSelector.select = select
    try:
        yield
    finally:
        RuleSelector.select = original


def no_publication_port(snapshot, source, origin):
    from terminal_marl.core.rotating_slots.planning_latency import _FuturePort
    from terminal_marl.core.rotating_slots.models import ValidationResult
    class DiagnosticPort(_FuturePort):
        def commit_validated_prefix(self, result, snapshot_version=None):
            base.require(result.snapshot_version == self._snapshot.version and result.accepted
                and not result.refinement_required and not result.cancel, "invalid diagnostic certificate")
            self.prefix_observations.append([p.trajectory_hash for p in result.plans])
            return ValidationResult(False, "DIAGNOSTIC_ONLY", "PUBLICATION_DISABLED")
    port = DiagnosticPort(snapshot, source, origin+19., search_wall_deadline=origin+15.,
        planning_wall_origin=origin, search_handoff_reserve_seconds=4.)
    port.prefix_observations = []
    return port


def compare_groups(current, saved):
    comparable = [row for row in saved if row.get("source_hash") == current["source_hash"]
        and row.get("canonical_frame_sha256") == current["canonical_frame_sha256"]]
    rows = comparable+[current]
    return dict(observed_groups=[r["group"] for r in rows],
        same_first_group_as=[r["group"] for r in comparable if r.get("first_group") is not None
            and r["first_group"] == current.get("first_group")],
        same_final_group_as=[r["group"] for r in comparable if r.get("final_group") is not None
            and r["final_group"] == current.get("final_group")],
        distinct_first_group_count=len({tuple(r["first_group"]) for r in rows if r.get("first_group") is not None}),
        distinct_final_group_count=len({tuple(r["final_group"]) for r in rows if r.get("final_group") is not None}),
        comparison_scope="ONLY_OBSERVED_NEW_GROUP_RUNS_SAME_SOURCE_AND_FRAME")


def run_group(problem, source, saved_history, group, d, directory, binding):
    """A single cold callback on immutable inputs; no world object is used."""
    from terminal_marl.core.rotating_slots.performance.parallel import _clear_worker_accelerators
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started, cpu = time.perf_counter(), time.process_time()
    frame = problem["global_frame"]
    before = base.digest(a4.codec().encode((source, problem["snapshot"], frame)))
    reset = _clear_worker_accelerators()
    callback = a4.new_callback(d, frame=frame)
    callback.selective_history.load_state_dict(deepcopy(saved_history["history"]))
    callback.selective_blockers = deepcopy(saved_history["blockers"])
    base.require(not callback.provider.previous and not callback.provider.references.cache
        and not callback.provider.validation_cache, "cold group provider required")
    setup_wall = time.perf_counter()-started
    observations, error = [], None
    origin = time.perf_counter()
    port = no_publication_port(problem["snapshot"], source, origin)
    try:
        with selector_intervention(group, observations):
            callback(port)
    except GroupNotApplicable as exc:
        error = dict(type=type(exc).__name__, reason=str(exc))
    except Exception as exc:
        error = dict(type=type(exc).__name__, reason=str(exc))
    callback_wall = time.perf_counter()-origin
    unchanged = before == base.digest(a4.codec().encode((source, problem["snapshot"], frame)))
    base.require(unchanged and port.proposal is None, "NO_PHYSICS diagnostic invariant changed")
    candidate = getattr(callback, "last_diagnostic_problem", None)
    result = callback.last_result
    telemetry = callback.records[-1].get("selective_replanning", {}) if callback.records else {}
    calls = telemetry.get("calls", ())
    first = (list(calls[0]["scope"]["replan_ids"]) if calls else
        observations[0].get("applied_scope", {}).get("replan_ids") if observations else None)
    final = list(candidate["catalog"].scope.replan_ids) if candidate is not None else None
    last_attempted = list(calls[-1]["scope"]["replan_ids"]) if calls else first
    final_certified = bool(result is not None and result.certified and result.evaluation is not None
        and result.evaluation.accepted and result.evaluation.status != "UNKNOWN"
        and not getattr(result.evaluation, "refinement_required", False))
    row = dict(status="NOT_APPLICABLE" if error and error["type"] == "GroupNotApplicable" else
        "DIAGNOSTIC_FAILED" if error else "COMPLETE", case="P2", group=group, run_role=ROLE,
        source_hash=binding["source_hash"], canonical_frame_sha256=base.digest(a4.codec().encode(frame)),
        group_recipe=list(RECIPES[group]), first_group=first, final_group=final, last_attempted_group=last_attempted,
        first_final_same=first == final if first is not None and final is not None else None,
        solve_calls=telemetry.get("solve_calls", 0), scope_expansions=telemetry.get("scope_expansions", 0),
        certified=final_certified, objective=None if result is None else result.objective,
        objective_descriptive_only=not final_certified, source_snapshot_time=source.time,
        effective_time=problem["snapshot"].time, NO_PHYSICS=True, publication_permitted=False,
        physical_advance=False, actual_submitted=0, immutable_inputs_unchanged=unchanged,
        cold_callback_setup_wall_s=setup_wall, callback_wall_s=callback_wall,
        frame_origin="ARCHIVED_PRECOMPUTED_FRAME_NOT_REBUILT",
        planning_budget=callback.records[-1].get("planning_budget") if callback.records else None,
        actual_timings=callback.records[-1].get("timings") if callback.records else None,
        full_diagnostic_wall_s=time.perf_counter()-started, process_cpu_s=time.process_time()-cpu,
        wall_comparison_scope="THREE_NEW_GROUPS_COLD_ARCHIVED_FRAME_ONLY_NOT_HISTORICAL_BASELINE_TOTAL",
        error=error)
    base.require(row["solve_calls"] <= 2 and row["scope_expansions"] <= 1, "existing local budget bounds changed")
    previous = []
    for other in RECIPES:
        path = directory.parent/other/"result.json"
        if other != group and path.is_file(): previous.append(base.load(path))
    row["group_comparison"] = compare_groups(row, previous)
    a4.write_new(directory/"binding.json", dict(binding, group=group, recipe=list(RECIPES[group]),
        cold_cache_reset=reset, saved_history_source="P2_BASELINE_PRE_CALLBACK_HISTORY",
        archived_frame_id=frame.identity, archive_identity_rebinding=problem.get("archive_identity_rebinding")))
    a4.write_new(directory/"selection_evidence.json", observations)
    a4.write_new(directory/"diagnostic.json", dict(result=result, callback_records=callback.records,
        publication_disabled_prefix_observations=port.prefix_observations,
        catalog_export=a4.export_catalog(candidate) if candidate is not None else None))
    if candidate is not None:
        a4.write_new(directory/"problem.json", a4.pack_problem(candidate, callback.state_dict()))
    a4.write_new(directory/"result.json", row)
    return row


def load_inputs(output, current_source):
    directory = Path(output)/"P2/baseline"
    paths = {name: directory/name for name in ("problem.json", "source_snapshot.json", "selector_history_before.json", "binding.json")}
    base.require(all(path.is_file() for path in paths.values()), "P2 baseline problem/source/history required")
    original_source = base.load(paths["binding.json"])["source"]
    differences = source_differences(original_source, current_source)
    problem, _ = a4.unpack_problem(paths["problem.json"])
    source = a4.codec().decode(base.load(paths["source_snapshot.json"]))
    timing = problem["snapshot"].planning_timing
    base.require(source.time == a4.CASES["P2"]["boundary"] and problem["snapshot"].time == source.time+20.
        and timing.snapshot_time == source.time and timing.effective_time == source.time+20.
        and timing.ready_deadline == source.time+19., "fixed P2 source/lead20 timing differs")
    history = base.load(paths["selector_history_before.json"])
    binding = dict(source_hash=current_source["source_hash"], source_commit=current_source["commit"],
        baseline_source_hash=original_source["source_hash"], source_differences=differences,
        inputs={name: a4.file_hash(path) for name,path in paths.items()}, case="P2",
        planning_profile="downstream_service_v8", backend="rh_cp_sat", lead_seconds=20.,
        ready_work_s=19., search_work_s=15., candidate_cache="COLD_PER_GROUP",
        group_assignment="PREDECLARED_RESOURCE_OPPORTUNITY_IDS_NO_OUTCOME_SELECTION", run_role=ROLE)
    return problem, source, history, binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=tuple(RECIPES), required=True)
    parser.add_argument("--output", type=Path, default=Path.home()/"training_outputs/terminal"/a4.OUTPUT_NAME)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--_child-job-id", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output = a4.validated_output(args.output)
    source = base.load(args.source_freeze)
    base.verify_sources(source)
    base.require(all(path in source["files"] for path in (DRIVER.relative_to(PROJECT).as_posix(),
        "tests/test_selective_fixed_groups_a4.py")), "fixed-group runner/test freeze required")
    job_id = "fixed_group_P2_"+args.group
    with a4.ledger_transaction(args.output) as ledger:
        bound = ledger.setdefault("group_diagnostic_source_hash", source["source_hash"])
        base.require(bound == source["source_hash"], "new fixed groups require one source freeze")
    if not args._child_job_id:
        command = [sys.executable, str(DRIVER), "--group", args.group, "--output", str(args.output),
            "--source-freeze", str(args.source_freeze), "--cpu", str(args.cpu), "--_child-job-id", job_id]
        result = a4.run_bounded_job(args.output, job_id, command, cap_s=CHILD_CAP)
        with a4.ledger_transaction(args.output) as ledger:
            ledger["jobs"][job_id]["run_role"] = ROLE
        base.print_result(result)
        return result
    base.require(args._child_job_id == job_id, "fixed-group child identity differs")
    with a4.ledger_transaction(args.output, write=False) as ledger:
        job = ledger["jobs"].get(job_id, {})
        base.require(job.get("status") == "running" and job.get("parent_pid") == os.getppid(), "active parent watchdog required")
    launch_threads = {key: os.environ.get(key) for key in base.THREADS}
    d = base.dependencies(PROJECT)
    d.launch_threads = launch_threads
    old.set_cpu(argparse.Namespace(stage="snapshot", cpu=args.cpu, root="s3r24"), d)
    problem, physical, history, binding = load_inputs(args.output, source)
    result = run_group(problem, physical, history, args.group, d,
        args.output/"fixed_groups/P2"/args.group, binding)
    base.print_result(result)
    return result


if __name__ == "__main__":
    main()
