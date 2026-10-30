"""One preregistered A4 quality diagnostic per watchdog child; never a matrix.

No action advances a world or publishes a proposal.  Offline factors are
separately charged child jobs and cannot change the frozen baseline archive.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import MappingProxyType

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_reoptimization_repair as old
from scripts.stage3 import run_selective_reoptimization_repair_v3 as v3

DRIVER = Path(__file__).resolve()
OUTPUT_NAME = "rs_selective_quality_a4_20261030"
VARIANT = "opportunity_reference_v4"
POLICY = dict(cap_child_wall_s=7200., prior_audits_child_wall_s=0.,
    historical_campaign_used_s=1374.067592, campaign=OUTPUT_NAME,
    previous_ledger_readonly="rs_selective_repair_v3_20261029/repair_budget_ledger.json",
    run_role="SHARED_A4_CAMPAIGN", old_ledgers_reset=False)
CASES = {
    "R1": dict(boundary=0., historical=0., checkpoint="0000",
               sha256="73f2d262f27d9c1567a9d3e0f67e9fe0cfeec769f6d9953a79c55e24083b34ed", aid="initial_inner_02"),
    "R4": dict(boundary=65., historical=60., checkpoint="0065",
               sha256="2f36969490f1dc01fa7c2bfb3a33efe96fa03e7ea02056dc7a11d69e797bdb8e", aid="initial_middle_02"),
    "R12": dict(boundary=210., historical=206.03508481464814, checkpoint="0210",
                sha256="5730466d2561f60bdb7cad69670247281184e6b1fc4aa4445952accaf030e367", aid="initial_middle_01"),
    "P2": dict(boundary=235., historical=233.65165837993845, checkpoint="0235",
               sha256="5cb5da997bef53d1c26b39fcf9152d23548b4c38e9e2e22e98b40b3f461348cf", aid=None),
}
FACTORS = ("baseline", "augment", "catalog_cp8", "augment_cp8", "catalog_cp24", "offset1", "offset2")


def ledger_transaction(output, *, write=True):
    """One A4 ledger for diagnostics, common-init, audit and formal children."""
    return old.ledger_transaction(output, write=write, policy=POLICY)


def run_bounded_job(output, job_id, command, *, cap_s, poll_s=.05):
    return old.run_bounded_job(output, job_id, command, cap_s=cap_s, poll_s=poll_s, policy=POLICY)


def finite_view(value):
    """asdict would deepcopy mappingproxy fields in frozen research objects."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value): raise ValueError("NaN is not diagnostic evidence")
        return {"$nonfinite_float": "+inf" if value > 0 else "-inf"}
    if is_dataclass(value):
        # SubmissionEvaluation._prepared contains executor-private mutable
        # calendars. Public result fields are the diagnostic certificate;
        # executable inputs are separately preserved by the whitelisted codec.
        return {f.name: finite_view(getattr(value, f.name)) for f in fields(value) if not f.name.startswith("_")}
    if hasattr(value, "items"): return {str(k): finite_view(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [finite_view(v) for v in value]
    if isinstance(value, (set, frozenset)): return sorted(finite_view(v) for v in value)
    if isinstance(value, Path): return str(value)
    return value


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(finite_view(value), stream, sort_keys=True, indent=2, allow_nan=False)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def catalog_hash(catalog):
    from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import candidate_hash
    return base.digest({aid: sorted(candidate_hash(c) for c in rows)
                        for aid, rows in sorted(catalog.columns.items())})


def codec():
    """Existing safe trajectory codec plus three explicitly named dataclasses."""
    from terminal_marl.core.rotating_slots.stage2_campaign import Stage2Codec
    from terminal_marl.core.rotating_slots.selective_replanning.scope import PlanningScope
    from terminal_marl.core.rotating_slots.selective_replanning.frame import GlobalEvaluationFrame
    from terminal_marl.core.rotating_slots.stage2_downstream import Demand
    class A4Codec(Stage2Codec):
        def encode(self, value):
            if isinstance(value, MappingProxyType):
                return {"$": "mappingproxy", "items": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
            return super().encode(value)
        def decode(self, value):
            if isinstance(value, dict) and value.get("$") == "mappingproxy":
                if set(value) != {"$", "items"}: raise ValueError("invalid diagnostic mappingproxy")
                rows=[(self.decode(k), self.decode(v)) for k, v in value["items"]]
                if len(dict(rows)) != len(rows): raise ValueError("duplicate diagnostic mappingproxy key")
                return MappingProxyType(dict(rows))
            return super().decode(value)
    result = A4Codec()
    for cls in (PlanningScope, GlobalEvaluationFrame, Demand):
        result.classes[cls.__module__.split(".")[-1] + "." + cls.__name__] = cls
    return result


def pack_problem(problem, callback_state=None):
    c = problem["catalog"]; d = c.downstream
    state = dict(snapshot=problem["snapshot"], frame=problem["global_frame"], selected=problem["selected"],
        catalog={f.name: getattr(c, f.name) for f in fields(c) if f.name != "downstream"},
        downstream=dict(demands=d.demands, fixed_calendars=tuple((p, lo, hi, aid) for p, lo, hi, aid in d.fixed_identity),
                        order=d.order, now=d.now, pad_count=len(d.calendars), fixed_demands=tuple(d.fixed_demands.values()),
                        assignment=d.assignment, frame_id=d.frame_id), callback_state=callback_state)
    from terminal_marl.core.rotating_slots.stage2_downstream import LEGACY_EVALUATION_PROFILE
    if d.evaluation_profile!=LEGACY_EVALUATION_PROFILE:
        state["downstream"]["evaluation_profile"]=d.evaluation_profile
    encoded = codec().encode(state)
    return dict(schema="a4_fixed_problem_v1", payload=encoded, sha256=base.digest(encoded), catalog_hash=catalog_hash(c))


def unpack_problem(path):
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateCatalog
    from terminal_marl.core.rotating_slots.stage2_downstream import JointDownstream, ticks
    row = base.load(path)
    base.require(row["schema"] == "a4_fixed_problem_v1" and row["sha256"] == base.digest(row["payload"]), "fixed problem archive changed")
    s = codec().decode(row["payload"])
    c = CandidateCatalog(**s["catalog"], downstream=JointDownstream(**s["downstream"]))
    base.require(catalog_hash(c) == row["catalog_hash"], "fixed candidate identities changed")
    frame, snapshot, scope = s["frame"], s["snapshot"], c.scope
    from terminal_marl.core.rotating_slots.selective_replanning.frame import frame_evaluation_profile
    base.require(frame_evaluation_profile(frame.policy)==c.downstream.evaluation_profile,
        "archive evaluation profile binding mismatch")
    saved_id = s["downstream"]["frame_id"]
    base.require(scope is not None and bool(saved_id) and scope.objective_frame_id == saved_id,
        "saved scope/downstream frame binding mismatch")
    base.require(snapshot is not None and scope.snapshot_id == frame.snapshot_id == snapshot.version
        and scope.effective_time == frame.effective_time == snapshot.time
        and c.downstream.now == ticks(snapshot.time), "fixed archive snapshot/time binding mismatch")
    canonical_frame_sha256 = base.digest(codec().encode(frame))
    decoded_id = frame.identity
    c.downstream.frame_id = decoded_id
    c = replace(c, scope=replace(scope, objective_frame_id=decoded_id))
    rebinding = dict(saved_frame_id=saved_id, decoded_frame_id=decoded_id,
        canonical_frame_sha256=canonical_frame_sha256, snapshot_id=snapshot.version,
        effective_time=snapshot.time, rebinding_applied=saved_id != decoded_id,
        only_changed=["catalog.scope.objective_frame_id", "catalog.downstream.frame_id"],
        reused_accepted_certificates=False)
    return dict(snapshot=snapshot, catalog=c, global_frame=frame, selected=s["selected"],
        archive_identity_rebinding=rebinding), s["callback_state"]


def memory_evidence():
    row = dict(pid=os.getpid(), observed_epoch=time.time())
    if sys.platform != "win32":
        import resource
        row["ru_maxrss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        row["ru_maxrss_unit"] = "bytes" if sys.platform == "darwin" else "KiB"
    path = Path("/proc/self/status")
    if path.is_file():
        row["proc_status_memory"] = [line for line in path.read_text().splitlines() if line.startswith(("VmRSS:", "VmHWM:"))]
    return row


def new_callback(d, *, frame=None, offset=0):
    # All reference_v4 experimental switches remain their false defaults.
    return d.NormalSchedulerCallback("rh_cp_sat", planning_profile="downstream_service_v8", online_policy="progressive_v1",
        selective_method="RULE_LOCAL_K8", selective_rule_variant=VARIANT, opportunity_seed_offset=offset,
        global_frame=frame, search_policy=dict(pad_anchor_priority=False, target_coverage_order=False,
            ordinary_immediate_filter=False), diagnostics=diagnostic_config())


def diagnostic_config():
    from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import DiagnosticConfig
    return DiagnosticConfig()


def baseline(args, d, directory):
    from terminal_marl.core.rotating_slots.performance.parallel import _clear_worker_accelerators
    case = CASES[args.case]
    checkpoint = args.server_root / "rs_selective_repair_20261029/arms/s3r24/evaluation/s3r24/RULE_LOCAL_K8" / ("checkpoint_" + case["checkpoint"] + ".json")
    base.require(file_hash(checkpoint) == case["sha256"], "preregistered old v2 checkpoint bytes differ")
    observation = v3.checkpoint_observation(checkpoint, case["historical"])
    base.require(observation["checkpoint_time"] == case["boundary"], "checkpoint boundary differs")
    reset = _clear_worker_accelerators(); before_memory = memory_evidence(); started = time.perf_counter(); cpu = time.process_time()
    world, saved = v3.restore_cold_with_history(checkpoint, d, directory)
    source = world.build_planning_snapshot(); before = source.version
    callback = new_callback(d); history = v3.restore_selector_history(callback, saved)
    timing = d.PlanningTiming.from_lead(source.time, 20., 1.)
    origin = time.perf_counter(); projected = d.project_visible_snapshot(source, timing.effective_time, timing=timing)
    port = d.FuturePort(projected, source, origin + 19., search_wall_deadline=origin + 15.,
        planning_wall_origin=origin, search_handoff_reserve_seconds=4.)
    callback_start = time.perf_counter(); callback(port); callback_wall = time.perf_counter() - callback_start
    problem = callback.last_diagnostic_problem
    base.require(problem is not None and problem["catalog"] is not None, "no final retained baseline catalog")
    base.require(world.build_planning_snapshot().version == before, "baseline mutated physical state")
    write_new(directory / "problem.json", pack_problem(problem, callback.state_dict()))
    write_new(directory / "source_snapshot.json", codec().encode(source))
    write_new(directory / "selector_history_before.json", history)
    write_new(directory / "baseline.json", dict(checkpoint_observation=observation, original_actual_times=dict(
        source=source.time, effective=projected.time, callback_wall_s=callback_wall, total_restore_and_callback_wall_s=time.perf_counter()-started,
        process_cpu_s=time.process_time()-cpu), memory_before=before_memory, memory_after=memory_evidence(), cache_reset=reset,
        records=callback.records, result=callback.last_result, catalog_hash=catalog_hash(problem["catalog"]),
        scope=problem["catalog"].scope, frame=problem["global_frame"], catalog_export=export_catalog(problem),
        cold_candidate_cache=True, physical_source_unchanged=True, no_physical_advance=True, actual_publication=False,
        only_changed=[], run_role="BASELINE_NORMAL_CALLBACK_COLD_SAVED_HISTORY"))
    return dict(status="COMPLETE", archive=str(directory / "problem.json"))


def export_catalog(problem):
    from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import CandidateDiagnostics, DiagnosticConfig
    catalog = problem["catalog"]; count = sum(map(len, catalog.columns.values()))
    collector = CandidateDiagnostics(DiagnosticConfig(max_candidates=count, max_trajectory_candidates=count), problem["snapshot"])
    for rows in catalog.columns.values():
        for column in rows: collector.candidate(column, trajectory=True)
    return collector.export(catalog, problem["selected"])


def offline_context(problem, *, solver_seconds=8., generation=False):
    from terminal_marl.core.rotating_slots.stage2_planning import EvaluationContext, PlanningBudget, QUALITY_BUDGET_PROFILE
    # The independent 8/24s solver factors have identical 60s validation/search
    # deadlines and hint/refinement settings. Only the solver allocation changes.
    limit = 6. if generation else 60.
    budget = PlanningBudget.for_profile(QUALITY_BUDGET_PROFILE)
    budget.soft_seconds=budget.watchdog_seconds=limit
    budget.candidate_seconds=6.; budget.solver_seconds=solver_seconds; budget.hint_seconds=3.
    budget.max_per_aircraft=32; budget.max_attempts=1024
    budget.planning_wall_deadline=time.perf_counter()+limit; budget.diagnostics=diagnostic_config()
    budget.search_wall_deadline = budget.planning_wall_deadline
    budget.allowed_replan_ids = set(problem["catalog"].scope.replan_ids)
    context = EvaluationContext(problem["snapshot"], budget=budget, validation_cache={}, reuse_prefix_results=False,
        scope=problem["catalog"].scope, global_frame=problem["global_frame"])
    budget._certificate_context = context.snapshot_key
    return context, budget


def rebuild_catalog(problem, additions, provider, context):
    """Preserve every old column/background; add single-valid scoped choices."""
    from terminal_marl.core.rotating_slots.stage2_downstream import describe_column
    c = problem["catalog"]; values = {aid: tuple(rows) for aid, rows in c.columns.items()}
    for column in additions:
        if column.aircraft_id not in values: raise ValueError("augmentation outside fixed scope")
        if not any(old.signature == column.signature for old in values[column.aircraft_id]):
            values[column.aircraft_id] += (column,)
    demands = dict(c.downstream.demands)
    for column in additions: demands[column.signature] = describe_column(problem["snapshot"], column, provider.references, context)
    downstream = problem["global_frame"].downstream(demands, set(values))
    # As in the core generator, scalar column keys are whole-world replacement
    # keys; the solver's joint objective is downstream.evaluate(selected).
    baseline = {aid: rows[0] for aid, rows in values.items()}
    values = {aid: tuple(replace(col, objective=downstream.key(tuple(dict(baseline, **{aid: col}).values())),
                                     downstream=demands[col.signature].diagnostics) for col in rows) for aid, rows in values.items()}
    return replace(c, columns=MappingProxyType(values), downstream=downstream,
                   fixed_background=MappingProxyType(dict(downstream.fixed_demands)))


def partial_choice(problem, case):
    from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import candidate_hash
    c = problem["catalog"]; effective = problem["snapshot"].time
    def eligible(col):
        return col.kind == "MOVE" and bool(col.plans) and col.plans[-1].pad_id is None and abs(col.plans[0].start_time-effective) <= 1.e-7
    if CASES[case]["aid"]:
        rows = [col for col in c.columns.get(CASES[case]["aid"], ()) if eligible(col)]
        return (min(rows, key=candidate_hash) if rows else None), "FIRST_HASH_SORTED_EXISTING_IMMEDIATE_PARTIAL"
    # P2 prioritizes the actual selected partial whose frozen virtual pad is 2.
    selected = {col.aircraft_id: col for col in problem["selected"]}
    priorities = [(aid, col) for aid, col in selected.items() if eligible(col) and c.downstream.assignment.get(aid) == 2]
    if priorities: return sorted(priorities, key=lambda pair: (pair[0], candidate_hash(pair[1])))[0][1], "SELECTED_IMMEDIATE_PARTIAL_VIRTUAL_PAD2"
    rows = [col for col in problem["selected"] if eligible(col)]
    if rows: return sorted(rows, key=lambda col: (col.aircraft_id, candidate_hash(col)))[0], "SELECTED_IMMEDIATE_PARTIAL_NO_PAD2_MATCH"
    rows = [col for group in c.columns.values() for col in group if eligible(col)]
    return (sorted(rows, key=lambda col: (col.aircraft_id, candidate_hash(col)))[0] if rows else None), "EXISTING_IMMEDIATE_PARTIAL_NO_SELECTED_MATCH"


def pad2_selected(problem):
    from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import identity
    return [dict(candidate=identity(c), frozen_virtual_pad=problem["catalog"].downstream.assignment.get(c.aircraft_id),
                 real_pad2=any(p.pad_id == 2 for p in c.plans), is_partial=bool(c.plans) and c.plans[-1].pad_id is None)
            for c in problem["selected"] if (problem["catalog"].downstream.assignment.get(c.aircraft_id) == 2
                                             or any(p.pad_id == 2 for p in c.plans))]


def accept_addition(problem, column, context, observations):
    from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import candidate_hash, result_row
    if column is None: observations.append(dict(status="COLUMN_RESOURCES_UNKNOWN")); return False
    if any(old.signature == column.signature for old in problem["catalog"].columns[column.aircraft_id]):
        observations.append(dict(candidate_hash=candidate_hash(column), status="ALREADY_IN_CATALOG")); return False
    single = context.check_single(column)
    single_valid = bool(single.accepted and single.status in ("SAFE", "ACCEPTED", "FEASIBLE", "VALID")
                        and not getattr(single, "refinement_required", False))
    complete_baseline = {c.aircraft_id for c in problem["selected"]} == set(problem["catalog"].columns)
    selected = tuple(column if old.aircraft_id == column.aircraft_id else old for old in problem["selected"])
    batch, conflicts = context.check_batch_conflicts(selected) if single_valid and complete_baseline else (None, ())
    prefix = context.check_commit_prefix(selected) if batch is not None and batch.accepted else None
    joint_valid = (single_valid and batch is not None and batch.accepted
             and batch.status != "UNKNOWN" and prefix is not None and prefix.accepted and prefix.status != "UNKNOWN"
             and not getattr(prefix, "refinement_required", False))
    observations.append(dict(candidate_hash=candidate_hash(column), single=result_row(single),
        batch=None if batch is None else result_row(batch), prefix=None if prefix is None else result_row(prefix),
        conflict_count=len(conflicts), retained_as_candidate=single_valid,
        candidate_status="TENTATIVE_SINGLE_VALID_COLUMN" if single_valid else "SINGLE_NOT_CERTIFIED",
        baseline_combination_certified=bool(joint_valid), complete_actual_selected_baseline=complete_baseline,
        baseline_combination_is_descriptive_probe_only=True, first_start=column.plans[0].start_time,
        effective=problem["snapshot"].time, final_pad=column.plans[-1].pad_id))
    return single_valid


def generate_additions(problem, provider, context, budget, case):
    from terminal_marl.core.rotating_slots.stage2_planning import _node
    snapshot = problem["snapshot"]; attempts=[]; additions=[]; chosen=None; policy=None
    if case == "R1":
        aid = CASES[case]["aid"]
        if aid not in problem["catalog"].columns:
            return additions, dict(status="TARGET_OUTSIDE_BASELINE_GROUP", target=aid, attempts=[])
        a = snapshot.aircraft[aid]; node = _node(snapshot, a)
        if node is None or node.ring != "inner":
            return additions, dict(status="TARGET_NOT_INNER_DECISION_NODE", target=aid, attempts=[])
        by_pad={pad: dict(aircraft_id=aid, target_pad=pad, tried=0, retained=0, validation=[], generation=[]) for pad in (0, 1)}
        attempts.extend(by_pad.values())
        stopped=False
        for duration in range(20, 301, 5):
            for pad in (0, 1):
                observed=by_pad[pad]
                if observed["retained"]: continue
                if not budget.available(aid): stopped=True; observed["stop"]="GENERATION_BUDGET"; break
                observed["tried"] += 1
                ok, plan = provider.references.generate(snapshot, a, node, pad, node.time, duration, budget)
                observed["generation"].append(dict(target_pad=pad, duration=duration, start=node.time,
                    budget_allows_return=ok, result="PLAN_RETURNED" if plan is not None else "NO_PLAN_RETURNED_NO_INFEASIBILITY_CERTIFICATE"))
                if not ok: observed["stop"] = "GENERATION_BUDGET"; stopped=True; break
                if plan is None: continue
                if plan.pad_id != pad: raise ValueError("specified-pad generator returned another pad")
                column = provider._column(snapshot, a, (plan,), "MOVE", context)
                if accept_addition(problem, column, context, observed["validation"]):
                    additions.append(column); observed["retained"] = 1
            if stopped or len(additions) == 2: break
        for observed in attempts:
            observed["status"] = "ONE_TENTATIVE_SINGLE_VALID_COLUMN" if observed["retained"] else "NO_ADDITION_OBSERVED_WITHIN_BOUND"
    else:
        chosen, policy = partial_choice(problem, case)
        if chosen is None:
            return additions, dict(status="NO_EXISTING_IMMEDIATE_PARTIAL", choice_policy=policy, attempts=[],
                pad2_selected_objects=pad2_selected(problem) if case == "P2" else [])
        a = snapshot.aircraft[chosen.aircraft_id]
        steps = provider.references.build_steps(snapshot, a, chosen.node, budget)
        suffix = None
        while budget.available(a.id):
            try: next(steps)
            except StopIteration as stopped: suffix = stopped.value; break
        if suffix is not None and suffix.status == "KNOWN" and suffix.plans and suffix.plans[-1].pad_id is not None:
            plans = (*chosen.plans, *suffix.plans)
            if tuple(p.trajectory_hash for p in plans[:len(chosen.plans)]) != tuple(p.trajectory_hash for p in chosen.plans):
                raise ValueError("completion changed required partial prefix")
            column = provider._column(snapshot, a, plans, "MOVE", context, reference=suffix)
            if accept_addition(problem, column, context, attempts): additions.append(column)
        else:
            attempts.append(dict(status="NO_COMPLETE_SUFFIX_OBSERVED_WITHIN_BOUND",
                reference_status=None if suffix is None else suffix.status))
    from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import identity
    return additions, dict(status="TENTATIVE_SINGLE_VALID_COLUMNS" if additions else "NO_ADDITION_OBSERVED_WITHIN_BOUND",
        choice_policy=policy, required_partial=None if chosen is None else identity(chosen), attempts=attempts,
        pad2_selected_objects=pad2_selected(problem) if case == "P2" else [],
        generation_screening=dict(budget.screening), generation_attempts=dict(budget.attempts),
        no_physical_infeasibility_claim=True, max_additions=2 if case == "R1" else 1, wall_allowance_s=6.)


def augmentation(args, d, directory, problem, callback_state):
    before = catalog_hash(problem["catalog"]); context, budget = offline_context(problem, generation=True)
    callback = new_callback(d); callback.load_state_dict(callback_state)
    started = time.perf_counter()
    additions, evidence = generate_additions(problem, callback.provider, context, budget, args.case)
    catalog = rebuild_catalog(problem, additions, callback.provider, context)
    augmented = dict(problem, catalog=catalog)
    base.require(catalog_hash(problem["catalog"]) == before, "augmentation mutated baseline")
    write_new(directory / "problem.json", pack_problem(augmented, callback.state_dict()))
    write_new(directory / "augmentation.json", dict(evidence=evidence, added_columns=additions,
        baseline_catalog_hash=before, augmented_catalog_hash=catalog_hash(catalog), scope=catalog.scope,
        background_ids=catalog.scope.fixed_background_ids, background_preserved=catalog.fixed_background == problem["catalog"].fixed_background,
        only_changed=["explicit_new_single_valid_columns", "demands_for_new_columns"],
        validation_context="INDEPENDENT_EMPTY_CACHE", generation_reference_cache="CLONED_POST_BASELINE_REFERENCES",
        wall_s=time.perf_counter()-started, run_role="OFFLINE_DIAGNOSTIC", no_physical_advance=True))
    return dict(status="COMPLETE", additions=len(additions))


def solve_fixed(problem, seconds):
    from terminal_marl.core.rotating_slots.stage2_planning import RollingHorizonCPSATScheduler
    before = catalog_hash(problem["catalog"]); context, budget = offline_context(problem, solver_seconds=seconds)
    started=time.perf_counter(); result = RollingHorizonCPSATScheduler().plan(problem["snapshot"], problem["catalog"], context, budget)
    base.require(catalog_hash(problem["catalog"]) == before, "fixed solver changed candidate identities")
    final_certified=bool(result.certified and result.evaluation is not None and result.evaluation.accepted
        and result.evaluation.status != "UNKNOWN" and not getattr(result.evaluation, "refinement_required", False))
    return dict(result=result, status=result.status, five_dimensional_key=result.objective, certified=result.certified,
        final_combination_certified=final_certified, improvement_evidence_eligible=final_certified,
        objective_is_descriptive_only=not final_certified,
        prefix=result.prefix, selected=result.selected, solver_records=result.solver_records,
        candidate_hash=before, scope=problem["catalog"].scope, solver_allocation_s=seconds,
        planning_budget=budget.settings(), solver_wall_s=budget.solver_wall_used, wall_s=time.perf_counter()-started,
        validation_allowance_s=60., validation_context="INDEPENDENT_EMPTY_CACHE", diagnostics=export_catalog(dict(problem, selected=result.selected)),
        run_role="OFFLINE_DIAGNOSTIC", optimality_scope="THIS_FIXED_CATALOG_ONLY_NO_GLOBAL_OPTIMALITY_CLAIM",
        only_changed=["solver_allocation_seconds"] if seconds == 24. else [], no_physical_advance=True, publication_permitted=False)


def offset_callback(args, d, directory, problem):
    from terminal_marl.core.rotating_slots.planning_latency import _FuturePort
    source = codec().decode(base.load(args.output / args.case / "baseline/source_snapshot.json"))
    saved = base.load(args.output / args.case / "baseline/selector_history_before.json")
    callback = new_callback(d, frame=problem["global_frame"], offset=int(args.factor[-1]))
    callback.selective_history.load_state_dict(saved["history"]); callback.selective_blockers=saved["blockers"]
    origin=time.perf_counter(); port=_FuturePort(problem["snapshot"], source, origin+19., search_wall_deadline=origin+15.,
        planning_wall_origin=origin, search_handoff_reserve_seconds=4.)
    callback(port); candidate=callback.last_diagnostic_problem
    base.require(candidate is not None, "offset callback retained no catalog")
    write_new(directory / "problem.json", pack_problem(candidate, callback.state_dict()))
    group=tuple(candidate["catalog"].scope.replan_ids)
    write_new(directory / "offset.json", dict(result=callback.last_result, records=callback.records,
        scope=candidate["catalog"].scope, baseline_group=problem["catalog"].scope.replan_ids, actual_group=group,
        same_group_as_baseline=set(group)==set(problem["catalog"].scope.replan_ids), frame_id=candidate["global_frame"].identity,
        only_changed=["opportunity_seed_offset"], offset=int(args.factor[-1]), candidate_context="COLD_PROVIDER_WITH_SAVED_SELECTOR_HISTORY",
        no_physical_advance=True, run_role="OFFLINE_DIAGNOSTIC"))
    return dict(status="COMPLETE", actual_group=group)


def continuation_inputs():
    return dict(status="EXTERNAL_ENTRYPOINT_NOT_AUTO_RUN", run_role="OFFLINE_DIAGNOSTIC",
        entrypoint="scripts/stage3/run_selective_short_continuation_a4.py",
        selection_requires_primary_review=True, allowed_cases=["R4", "R12"], duration_seconds=120.,
        required_inputs=["reviewed_baseline_problem_sha256", "reviewed_SAFE_immediate_MOVE_candidate_hash",
                         "baseline_vs_selected_action_pair", "same_normal_policy_and_saved_history",
                         "source_physical_checkpoint_sha256", "lead20_real_compute_timing_receipt",
                         "primary_selected_public_boundary_capture_hook", "separate_continuation_job_budget_receipt"],
        publication_or_physics_authorized_by_this_entrypoint=False)


def validated_output(output):
    output=Path(output).resolve()
    base.require(output.name == OUTPUT_NAME and output.parent == (Path.home()/"training_outputs/terminal").resolve(),
                 "A4 output must be the fresh authorized server directory")
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factor", choices=FACTORS); parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--output", type=Path, default=Path.home()/"training_outputs/terminal"/OUTPUT_NAME)
    parser.add_argument("--server-root", type=Path, default=Path.home()/"training_outputs/terminal")
    parser.add_argument("--source-freeze", type=Path, required=True); parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--_child-job-id", help=argparse.SUPPRESS)
    args=parser.parse_args(argv); args.output=validated_output(args.output)
    base.require(args.server_root.resolve() == (Path.home()/"training_outputs/terminal").resolve(), "fixed server input root required")
    base.require(args.factor not in ("catalog_cp24", "offset1", "offset2") or args.case in (("R1", "P2") if args.factor == "catalog_cp24" else ("P2",)),
                 "factor outside preregistered minimal table")
    job_id=args.case+"_"+args.factor
    if not args._child_job_id:
        command=[sys.executable, str(DRIVER), *(sys.argv[1:] if argv is None else argv), "--_child-job-id", job_id]
        result=run_bounded_job(args.output, job_id, command, cap_s=300.)
        with ledger_transaction(args.output) as ledger:
            ledger["jobs"][job_id]["run_role"]="OFFLINE_DIAGNOSTIC"
        result["run_role"]="OFFLINE_DIAGNOSTIC"
        print(json.dumps(finite_view(dict(job_id=job_id, **result)), allow_nan=False)); return
    base.require(args._child_job_id == job_id, "child identity differs")
    with ledger_transaction(args.output, write=False) as ledger:
        row=ledger["jobs"].get(job_id, {})
        base.require(row.get("status") == "running" and row.get("parent_pid") == os.getppid(), "active parent watchdog required")
    source=base.load(args.source_freeze); base.verify_sources(source)
    base.require(all(name in source["files"] for name in (DRIVER.relative_to(PROJECT).as_posix(), "tests/test_selective_quality_gap_a4.py")),
                 "reviewed A4 runner and tests freeze required")
    with ledger_transaction(args.output) as ledger:
        bound=ledger.setdefault("a4_source_hash", source["source_hash"])
        base.require(bound == source["source_hash"], "A4 diagnostic source differs between factors")
    launch_threads={key:os.environ.get(key) for key in base.THREADS}
    d=base.dependencies(PROJECT); d.launch_threads=launch_threads
    args.stage="snapshot"; old.set_cpu(args,d)
    directory=args.output / args.case / args.factor; directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / "binding.json", dict(source=source, case=CASES[args.case], factor=args.factor, cpu=args.cpu,
        run_role="OFFLINE_DIAGNOSTIC", accounting=POLICY, job_id=job_id, lead_seconds=20., continuation=continuation_inputs()))
    if args.factor == "baseline": result=baseline(args,d,directory)
    else:
        archive=args.output / args.case / "baseline/problem.json"
        problem, state=unpack_problem(archive)
        if args.factor == "augment": result=augmentation(args,d,directory,problem,state)
        elif args.factor.startswith("offset"): result=offset_callback(args,d,directory,problem)
        else:
            if args.factor == "augment_cp8":
                problem, _=unpack_problem(args.output / args.case / "augment/problem.json")
            result=solve_fixed(problem, 24. if args.factor == "catalog_cp24" else 8.)
            if args.factor == "augment_cp8": result["only_changed"]=["explicit_single_valid_catalog_additions"]
        result["archive_identity_rebinding"]=problem["archive_identity_rebinding"]
    result["run_role"]="OFFLINE_DIAGNOSTIC"
    write_new(directory / "result.json", result)
    print(json.dumps(finite_view(dict(status="COMPLETE", result=str(directory / "result.json"))), allow_nan=False))


if __name__ == "__main__": main()
