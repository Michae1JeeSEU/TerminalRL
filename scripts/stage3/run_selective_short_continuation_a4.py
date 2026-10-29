"""Prepare, or explicitly run, one fixed A4 first-action continuation branch.

Offline choice discovery is separate from online revalidation and physical
execution. The first choice uses the existing deferred executor transaction;
all later choices use the same cold reference_v4 normal policy in both arms.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage3 import run_selective_quality_gap_a4 as a4
from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_reoptimization_repair as old
from scripts.stage3 import run_selective_reoptimization_repair_v3 as v3

DRIVER = Path(__file__).resolve()
CASES = ("R4", "R12")
BRANCHES = ("baseline_choice", "forced_partial")
SWITCHES = dict(pad_anchor_priority=False, target_coverage_order=False, ordinary_immediate_filter=False)
HORIZON = 120.
CHILD_CAP = 300.
ROLE = "DIAGNOSTIC_SHORT_CONTINUATION"
SOURCE_ADAPTER_FILES = frozenset(("scripts/stage3/run_selective_quality_gap_a4.py",
    "scripts/stage3/run_selective_short_continuation_a4.py", "tests/test_selective_quality_gap_a4.py",
    "tests/test_selective_short_continuation_a4.py"))


def hashes(columns):
    from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import candidate_hash
    return {c.aircraft_id: candidate_hash(c) for c in columns}


def branch_choices(problem, case):
    """The UID and first hash are fixed; never search another UID/case."""
    base.require(case in CASES, "only R4 and R12 short continuations are authorized")
    baseline = tuple(problem["selected"])
    base.require(len({c.aircraft_id for c in baseline}) == len(baseline)
        and set(hashes(baseline)) == set(problem["catalog"].columns), "complete baseline selected tuple required")
    partial, policy = a4.partial_choice(problem, case)
    if partial is None:
        return None, dict(status="NOT_APPLICABLE", reason="NO_FIXED_UID_IMMEDIATE_PARTIAL", case=case,
            aircraft_id=a4.CASES[case]["aid"], choice_policy=policy)
    forced = tuple(partial if c.aircraft_id == partial.aircraft_id else c for c in baseline)
    return {"baseline_choice": baseline, "forced_partial": forced}, dict(case=case,
        aircraft_id=partial.aircraft_id, candidate_hash=hashes((partial,))[partial.aircraft_id],
        choice_policy=policy, baseline_hashes=hashes(baseline), forced_hashes=hashes(forced))


def certified(result):
    return bool(result is not None and result.accepted and result.status != "UNKNOWN"
        and not getattr(result, "refinement_required", False))


def certify_selection(context, selected):
    from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import result_row
    batch, conflicts = context.check_batch_conflicts(selected)
    prefix = context.check_commit_prefix(selected) if certified(batch) else None
    return prefix, dict(batch=result_row(batch), prefix=result_row(prefix) if prefix is not None else None,
        conflict_count=len(conflicts), certified=certified(batch) and certified(prefix),
        prefix_hashes=[p.trajectory_hash for p in context.prefix(selected)])


def prepare_pair(problem, case):
    """Offline certificate and objective evidence only; never call a world/CP."""
    started, cpu = time.perf_counter(), time.process_time()
    choices, row = branch_choices(problem, case)
    row.update(physical_execution=False, solver_called=False, online_performance_evidence=False,
        accounting_scope="OFFLINE_DISCOVERY_AND_CERTIFICATION_ONLY",
        archive_identity_rebinding=problem.get("archive_identity_rebinding"))
    if choices is not None:
        baseline_key = problem["catalog"].downstream.key(choices["baseline_choice"])
        forced_key = problem["catalog"].downstream.key(choices["forced_partial"])
        row.update(baseline_objective=list(baseline_key), forced_objective=list(forced_key),
            objective_profile="downstream_service_v8", forced_strictly_worse=forced_key > baseline_key)
        certificates = {}
        for branch in BRANCHES:
            context, _ = a4.offline_context(problem)
            _, certificates[branch] = certify_selection(context, choices[branch])
        row["certificates"] = certificates
        row["status"] = "PREPARED" if row["forced_strictly_worse"] and all(c["certified"] for c in certificates.values()) else "NOT_APPLICABLE"
        row["reason"] = ("SAFE_STRICTLY_WORSE_FIXED_PARTIAL" if row["status"] == "PREPARED" else
            "FORCED_CHOICE_NOT_STRICTLY_WORSE" if not row["forced_strictly_worse"] else "FIXED_COMBINATION_NOT_CERTIFIED")
    row.update(offline_wall_s=time.perf_counter()-started, offline_cpu_s=time.process_time()-cpu)
    return row


def normal_callback(d):
    return d.NormalSchedulerCallback("rh_cp_sat", planning_profile="downstream_service_v8",
        online_policy="progressive_v1", selective_method="RULE_LOCAL_K8",
        selective_rule_variant=a4.VARIANT, search_policy=dict(SWITCHES))


def cold_search(callback):
    """Discard search products, preserving actual execution/review history."""
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider
    from terminal_marl.core.rotating_slots.performance.parallel import _clear_worker_accelerators
    callback.provider = CandidateProvider(planning_profile="downstream_service_v8")
    callback.selective_blockers = []
    return _clear_worker_accelerators()


def snapshot_equivalence(expected, current):
    """Compare every codec field except the top-level generated version."""
    encoded_old = a4.codec().encode(replace(expected, version="GENERATED_VERSION_EXCLUDED"))
    encoded_new = a4.codec().encode(replace(current, version="GENERATED_VERSION_EXCLUDED"))
    return dict(old_version=expected.version, new_version=current.version,
        old_canonical_sha256=base.digest(encoded_old), new_canonical_sha256=base.digest(encoded_new),
        equivalent=encoded_old == encoded_new, excluded_fields=["version"])


def rebind_validation(problem, snapshot):
    """Refresh only snapshot/frame identity links after full input equivalence."""
    frame = replace(problem["global_frame"], snapshot_id=snapshot.version)
    scope = replace(problem["catalog"].scope, snapshot_id=snapshot.version, objective_frame_id=frame.identity)
    scope.validate(snapshot)
    return scope, frame


class FirstDecisionAdapter:
    """One offline intervention through FuturePort, then ordinary delegation."""
    def __init__(self, delegate, problem, source, selected, branch):
        base.require(branch in BRANCHES, "unknown short continuation branch")
        self.delegate, self.problem, self.source = delegate, problem, source
        self.selected, self.branch = tuple(selected), branch
        self.first_pending = True
        self.first_evidence = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def state_dict(self):
        return dict(schema="a4_first_decision_adapter_v1", branch=self.branch, first_pending=self.first_pending,
            selected_hashes=hashes(self.selected), source_version=self.source.version,
            effective_version=self.problem["snapshot"].version, first_evidence=self.first_evidence,
            delegate=self.delegate.state_dict())

    def load_state_dict(self, state):
        base.require(state["schema"] == "a4_first_decision_adapter_v1" and state["branch"] == self.branch
            and state["selected_hashes"] == hashes(self.selected) and state["source_version"] == self.source.version
            and state["effective_version"] == self.problem["snapshot"].version, "first intervention binding differs")
        self.delegate.load_state_dict(state["delegate"])
        self.first_pending, self.first_evidence = state["first_pending"], state["first_evidence"]

    def __call__(self, port):
        if not self.first_pending:
            return self.delegate(port)
        from terminal_marl.core.rotating_slots.planning_latency import _FuturePort
        from terminal_marl.core.rotating_slots.stage2_planning import EvaluationContext, PlanningBudget, PlanningResult
        # A SchedulerPort would publish immediately, so this adapter only accepts
        # the wrapper's actual projected/deferred port.
        base.require(isinstance(port, _FuturePort), "first intervention requires wrapper FuturePort")
        self.first_pending = False
        started, cpu = time.perf_counter(), time.process_time()
        snapshot = port.build_planning_snapshot()
        evidence = dict(record_type="DIAGNOSTIC_FORCED_FIRST_CALLBACK", branch=self.branch,
            offline_intervention=True, solver_called=False, evidence_phase="CALLBACK_RETURN",
            selected_hashes=hashes(self.selected), source_version=port._source.version,
            effective_version=snapshot.version, deadline=port.planning_wall_deadline,
            review_policy="NO_NORMAL_PATH_REVIEW_FROM_OFFLINE_CATALOG", certified=False, proposed_prefix_hashes=[])
        evaluation, prefix, reason = None, (), None
        try:
            evidence["input_equivalence"] = dict(source=snapshot_equivalence(self.source, port._source),
                effective=snapshot_equivalence(self.problem["snapshot"], snapshot))
            base.require(evidence["input_equivalence"]["source"]["equivalent"], "source canonical snapshot differs")
            base.require(evidence["input_equivalence"]["effective"]["equivalent"], "effective canonical snapshot differs")
            base.require(snapshot.time == self.source.time+20., "fixed effective time differs")
            base.require(port.planning_wall_deadline is not None and time.perf_counter() <= port.planning_wall_deadline,
                "first validation ready deadline missed")
            budget = PlanningBudget.for_profile("quality_v5")
            budget.planning_wall_deadline = port.planning_wall_deadline
            budget.search_wall_deadline = port.planning_wall_deadline
            budget.planning_wall_origin = port.planning_wall_origin
            budget.soft_seconds = budget.watchdog_seconds = 19.
            scope, frame = rebind_validation(self.problem, snapshot)
            evidence["fresh_binding"] = dict(scope=scope.to_dict(), frame_id=frame.identity,
                old_frame_id=self.problem["global_frame"].identity)
            context = EvaluationContext(snapshot, budget=budget, validation_cache={}, reuse_prefix_results=False,
                scope=scope, global_frame=frame)
            evaluation, check = certify_selection(context, self.selected)
            evidence["fresh_validation"] = check
            base.require(check["certified"], "fresh fixed combination not certified")
            base.require(time.perf_counter() <= port.planning_wall_deadline, "first validation ready deadline missed")
            prefix = context.prefix(self.selected)
            deferred = port.commit_validated_prefix(evaluation, snapshot_version=snapshot.version)
            base.require(deferred.status == "DEFERRED", "first proposal was not deferred")
            evidence.update(certified=True, deferred_status=deferred.status,
                proposed_prefix_hashes=[p.trajectory_hash for p in prefix])
        except ValueError as exc:
            reason = str(exc)
            evaluation, prefix = None, ()
            evidence["rejection_reason"] = reason
        finally:
            evidence["cold_reset_after_intervention"] = cold_search(self.delegate)
        evidence.update(online_validation_wall_s=time.perf_counter()-started, online_validation_cpu_s=time.process_time()-cpu)
        result = PlanningResult(self.selected, prefix, evaluation,
            "DIAGNOSTIC_FIRST_CERTIFIED" if evidence["certified"] else "DIAGNOSTIC_FIRST_REJECTED",
            self.problem["catalog"].downstream.key(self.selected), certified=evidence["certified"])
        record = dict(time=snapshot.time, method="rh_cp_sat", status=result.status, record_type=evidence["record_type"],
            planning_timing=asdict(snapshot.planning_timing), certified=result.certified, submitted=0,
            commit_status="DEFERRED" if result.certified else "NOT_COMMITTED", actual_submitted_trajectory_hashes=[],
            selective_replanning=dict(method="RULE_LOCAL_K8", rule_variant=a4.VARIANT, search_policy=dict(SWITCHES),
                first_intervention=evidence, scope=self.problem["catalog"].scope.to_dict()))
        self.delegate.records.append(record)
        self.delegate.last_result = result
        self.first_evidence = evidence
        return result


def holding_observer(world):
    """Observe actual integration deltas without changing physical ledgers."""
    cls, original = type(world), type(world)._advance_energy
    rows = {ring: dict(seconds=0., energy_j=0.) for ring in ("outer", "middle", "inner")}
    def observed(instance, end):
        before = {aid: (a.ring, a.holding_time, a.energy_j) for aid, a in instance.aircraft.items()}
        result = original(instance, end)
        for aid, (ring, held, energy) in before.items():
            after = instance.aircraft.get(aid)
            if after is not None and ring in rows and after.holding_time > held:
                rows[ring]["seconds"] += after.holding_time-held
                rows[ring]["energy_j"] += after.energy_j-energy
        return result
    cls._advance_energy = observed
    return rows, lambda: setattr(cls, "_advance_energy", original)


def first_delivery(adapter, wrapper):
    record = adapter.records[0] if adapter.records else {}
    task = wrapper.records[0] if wrapper.records else {}
    expected = (adapter.first_evidence or {}).get("proposed_prefix_hashes", [])
    actual = record.get("actual_submitted_trajectory_hashes", [])
    delivered = bool(expected and task.get("status") == "COMMITTED" and set(actual) == set(expected))
    valid_wait = bool(getattr(adapter, "branch", None) == "baseline_choice"
        and (adapter.first_evidence or {}).get("certified") and not expected
        and all(c.kind in ("KEEP", "WAIT") for c in adapter.selected)
        and task.get("status") == "NO_EXECUTABLE_PREFIX" and task.get("reason") == "NO_EXECUTABLE_PREFIX"
        and record.get("actual_delivery_status") == "NO_EXECUTABLE_PREFIX")
    return dict(evidence_phase="ACTUAL_EXECUTOR_ACK", first_action_delivered=delivered,
        valid_baseline_wait=valid_wait,
        action_comparison_status="FIRST_ACTION_DELIVERED" if delivered else "VALID_BASELINE_WAIT" if valid_wait else "FIRST_ACTION_NOT_DELIVERED",
        expected_prefix_hashes=expected, actual_prefix_hashes=actual,
        actual_delivery_status=record.get("actual_delivery_status"), actual_delivery_time=record.get("actual_delivery_time"),
        normal_task=task, valid_action_contrast=delivered or valid_wait)


def terminal_increment(world, initial, initial_touchdowns, d):
    current = d.episode_metrics(world)
    cumulative = ("L_success", "L_normal", "L_emergency", "F_energy_total", "normal_landed", "emergency_landed",
        "energy_failures", "failed_before_landing", "energy_j", "energy_kwh", "airborne_time", "holding_time",
        "upstream_wait_time", "cost_emergency", "cost_reroute")
    return dict(metrics=current, increments={key: current[key]-initial[key] for key in cumulative
        if type(current.get(key)) in (int,float) and type(initial.get(key)) in (int,float)},
        pending={key:current.get(key) for key in ("N_airborne_pending", "N_upstream_pending", "emergency_unresolved")},
        new_touchdowns=[dict(aircraft_id=aid, time=a.touchdown_time, normal=not a.emergency.charged)
            for aid,a in sorted(world.aircraft.items()) if a.touchdown_time is not None and aid not in initial_touchdowns])


def emit_outputs(directory, world, adapter, wrapper, prepared, initial, initial_touchdowns, holding, d, binding, error=None):
    directory = Path(directory)
    first = first_delivery(adapter, wrapper)
    terminal = dict(source_time=adapter.source.time, absolute_end_time=adapter.source.time+HORIZON,
        observed_time=world.time, physical_ledger_reset=False, holding_by_ring=holding,
        **terminal_increment(world, initial, initial_touchdowns, d))
    for name,value in (("firstcert.json", adapter.first_evidence), ("actualsubmit.json", first),
        ("counterfactualobjective.json", prepared), ("terminal.json", terminal),
        ("normal_tasks.json", wrapper.records), ("scheduler_records.json", adapter.records),
        ("online_services.json", wrapper.online_services), ("trajectories.json", world.trajectories)):
        a4.write_new(directory/name, value)
    identity = dict(binding["checkpoint_identity"], source_hash=binding["a4_source_hash"])
    d.save_checkpoint(directory/"checkpoint_terminal.json", world, wrapper, identity)
    record = world.to_dict()
    record.update(decisions=adapter.records, stage3_trace=[dict(record_type="NormalTask", **r) for r in wrapper.records],
        stage2_provenance=dict(execution_scope=binding),
        tentative_predictions=[dict(aircraft_id=aid, as_of=world.time, plans=[p.to_dict() for p in plans])
            for aid,plans in adapter.provider.previous.items()])
    d.ReplayBundle.write(directory/"replay", record=record, code_commit=binding["source_commit"],
        config_path=world.config.path, command=str(DRIVER), seed=binding["seed"], eager=False)
    result = dict(status="COMPLETE" if world.time == adapter.source.time+HORIZON and error is None else "CONTINUATION_PARTIAL",
        case=binding["case"], branch=adapter.branch, run_role=ROLE, first_delivery=first,
        source_time=adapter.source.time, observed_time=world.time, absolute_end_time=adapter.source.time+HORIZON,
        error=error, replay_status="WRITTEN_HUMAN_PENDING")
    a4.write_new(directory/"result.json", result)
    return result


def fixed_checkpoint_seed(payload):
    manifest = a4.codec().decode(payload["world"])["manifest"]
    base.require(base.digest(manifest.to_dict()) == payload["identity"]["manifest_sha256"]
        and manifest.seed == base.ROOTS["s3r24"][0] == 7325, "fixed high-flow checkpoint manifest/seed differs")
    return manifest.seed


def source_adapter_differences(baseline, current):
    """Only the four named diagnostic adapters may differ from a baseline."""
    for source in (baseline, current):
        base.require(source["source_hash"] == base.digest(source["files"]), "source manifest hash differs")
    differences = [dict(path=path, baseline_sha256=baseline["files"].get(path), current_sha256=current["files"].get(path))
        for path in sorted(set(baseline["files"]) | set(current["files"]))
        if path not in baseline["files"] or path not in current["files"]
        or baseline["files"].get(path) != current["files"].get(path)]
    base.require(all(row["path"] in SOURCE_ADAPTER_FILES for row in differences),
        "baseline source differs outside authorized instrumentation files")
    return differences


def load_inputs(output, server_root, case, source):
    base.require(case in CASES, "only R4/R12 inputs permitted")
    baseline = Path(output)/case/"baseline"
    paths = {name:baseline/name for name in ("problem.json", "source_snapshot.json", "binding.json", "selector_history_before.json")}
    base.require(all(path.is_file() for path in paths.values()), "required baseline problem/source/history artifacts missing")
    baseline_binding = base.load(paths["binding.json"])
    source_differences = source_adapter_differences(baseline_binding["source"], source)
    item = a4.CASES[case]
    checkpoint = Path(server_root)/"rs_selective_repair_20261029/arms/s3r24/evaluation/s3r24/RULE_LOCAL_K8"/(
        "checkpoint_"+item["checkpoint"]+".json")
    base.require(a4.file_hash(checkpoint) == item["sha256"], "fixed old checkpoint SHA256 differs")
    problem, _ = a4.unpack_problem(paths["problem.json"])
    physical = a4.codec().decode(base.load(paths["source_snapshot.json"]))
    base.require(physical.time == item["boundary"] and problem["snapshot"].time == physical.time+20., "fixed source/effective boundary differs")
    payload = base.load(checkpoint)["payload"]
    seed = fixed_checkpoint_seed(payload)
    binding = dict(case=case, a4_source_hash=source["source_hash"], source_commit=source["commit"],
        baseline_source_hash=baseline_binding["source"]["source_hash"], source_differences=source_differences,
        canonical_frame_sha256=problem["archive_identity_rebinding"]["canonical_frame_sha256"],
        checkpoint=str(checkpoint), checkpoint_sha256=item["sha256"], checkpoint_identity=payload["identity"],
        seed=seed, seed_basis="IDENTITY_BOUND_CHECKPOINT_MANIFEST",
        baseline_files={name:a4.file_hash(path) for name,path in paths.items()}, source_version=physical.version,
        effective_version=problem["snapshot"].version, lead_seconds=20., heartbeat_seconds=5.,
        horizon_seconds=HORIZON, search_policy=dict(SWITCHES), normal_policy=a4.VARIANT,
        objective_profile="downstream_service_v8", run_role=ROLE)
    return problem, physical, checkpoint, binding


def run_branch(directory, problem, physical, checkpoint, prepared, binding, branch, d):
    base.require(prepared["status"] == "PREPARED", "NOT_APPLICABLE pair cannot be executed")
    choices, identity = branch_choices(problem, binding["case"])
    base.require(identity["candidate_hash"] == prepared["candidate_hash"] and
        identity["baseline_hashes"] == prepared["baseline_hashes"] and identity["forced_hashes"] == prepared["forced_hashes"],
        "prepared first choices changed")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    world, saved = v3.restore_cold_with_history(checkpoint, d, directory)
    base.require(world.build_planning_snapshot().version == physical.version and world.scheduler_period == 5.,
        "restored physical state or heartbeat differs")
    delegate = normal_callback(d)
    v3.restore_selector_history(delegate, saved)
    # Only executed age/history is retained. Historical search blockers, pending
    # computation, candidate/reference caches and previous chains are not reused.
    cache_reset = cold_search(delegate)
    adapter = FirstDecisionAdapter(delegate, problem, physical, choices[branch], branch)
    wrapper = d.LatencyAwareScheduler(adapter, lead_seconds=20., preparation_margin=1.,
        search_handoff_reserve_seconds=4., restart_policy="coalesced_restart_v2")
    world.decision_callback, world.admission_callback = wrapper, d.shared_fifo_admission
    initial = d.episode_metrics(world)
    touch = {aid for aid,a in world.aircraft.items() if a.touchdown_time is not None}
    a4.write_new(directory/"binding.json", dict(binding, branch=branch, cache_reset=cache_reset,
        history=delegate.selective_history.state_dict(), historical_pending_not_resumed=True,
        candidates_references_previous="COLD_BOTH_BRANCHES", offline_discovery_online_charge=False,
        archive_identity_rebinding=problem.get("archive_identity_rebinding")))
    holding, restore_observer = holding_observer(world)
    error = None
    try:
        wrapper(world._scheduler_port)
        for step in range(5, int(HORIZON)+1, 5):
            target = physical.time+step
            world.advance(target, wrapper, settle_only_at_end=True)
            if world.failed or getattr(world, "terminated_invalid", False) or abs(world.time-target)>1.e-7:
                break
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        restore_observer()
    return emit_outputs(directory, world, adapter, wrapper, prepared, initial, touch, holding, d, binding, error)


def verify_freeze(source):
    base.verify_sources(source)
    required = (DRIVER.relative_to(PROJECT).as_posix(), "tests/test_selective_short_continuation_a4.py")
    base.require(all(path in source["files"] for path in required), "reviewed short-continuation script/test freeze required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "prepare-child", "run-child"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--branch", choices=BRANCHES)
    parser.add_argument("--output", type=Path, default=Path.home()/"training_outputs/terminal"/a4.OUTPUT_NAME)
    parser.add_argument("--server-root", type=Path, default=Path.home()/"training_outputs/terminal")
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args(argv)
    args.output = a4.validated_output(args.output)
    base.require(args.server_root.resolve() == (Path.home()/"training_outputs/terminal").resolve(), "fixed server input root required")
    is_run = args.stage.startswith("run")
    base.require((args.branch is not None) == is_run, "run needs one branch; preparation prepares the fixed pair")
    source = base.load(args.source_freeze)
    verify_freeze(source)
    job_id = "short_continuation:"+args.case+":"+(args.branch if is_run else "prepare")
    with a4.ledger_transaction(args.output) as ledger:
        bound = ledger.setdefault("a4_source_hash", source["source_hash"])
        base.require(bound == source["source_hash"], "short continuation differs from A4 diagnostic freeze")
    if not args.stage.endswith("-child"):
        command = [sys.executable, str(DRIVER), "--stage", "run-child" if is_run else "prepare-child",
            "--case", args.case, "--output", str(args.output), "--server-root", str(args.server_root),
            "--source-freeze", str(args.source_freeze), "--cpu", str(args.cpu)]
        if is_run: command += ["--branch", args.branch]
        result = a4.run_bounded_job(args.output, job_id, command, cap_s=CHILD_CAP)
        with a4.ledger_transaction(args.output) as ledger:
            ledger["jobs"][job_id]["run_role"] = ROLE if is_run else "OFFLINE_CONTINUATION_PREPARATION"
        base.print_result(result)
        return result
    with a4.ledger_transaction(args.output, write=False) as ledger:
        job = ledger["jobs"].get(job_id, {})
        base.require(job.get("status") == "running" and job.get("parent_pid") == os.getppid(), "active parent watchdog required")
    launch_threads = {key: os.environ.get(key) for key in base.THREADS}
    d = base.dependencies(PROJECT)
    d.launch_threads = launch_threads
    old.set_cpu(argparse.Namespace(stage="snapshot", cpu=args.cpu, root="s3r24"), d)
    problem, physical, checkpoint, binding = load_inputs(args.output, args.server_root, args.case, source)
    root = args.output/"diagnostic_continuations"/args.case
    if not is_run:
        directory = root/"preparation"
        directory.mkdir(parents=True, exist_ok=False)
        prepared = prepare_pair(problem, args.case)
        prepared["binding"] = binding
        a4.write_new(directory/"prepared.json", prepared)
        base.print_result(prepared)
        return prepared
    prepared = base.load(root/"preparation/prepared.json")
    base.require(prepared["binding"] == binding, "prepared continuation inputs changed")
    result = run_branch(root/args.branch, problem, physical, checkpoint, prepared, binding, args.branch, d)
    base.print_result(result)
    return result


if __name__ == "__main__":
    main()
