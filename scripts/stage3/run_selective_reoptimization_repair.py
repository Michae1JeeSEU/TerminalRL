"""One explicit v2 repair action, under a shared active-child wall budget.

No action launches a matrix. ``run_bounded_job`` is also the mechanical wrapper
for separately authorized checks/audits, so their cost uses the same ledger.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage3 import run_selective_reoptimization_a as base

DRIVER = Path(__file__).resolve()
OUTPUT_NAME = "rs_selective_repair_20261029"
CAP = 5400.
PRIOR_AUDIT_S = 82.8723364593
HISTORICAL_USED_S = 7323.47022066447
RULE = "opportunity_cooperative_v2"
RESTART = "coalesced_restart_v2"
ROOT_CPUS = {"s3r23": 12, "s3r24": 14}


def treatment(method="RULE_LOCAL_K8", offset=0):
    base.require(method in base.METHODS and offset in (0, 1, 2), "fixed repair method/offset required")
    return dict(protocol="rs_selective_opportunity_cooperative_v2_20261029", selective_method=method,
        selective_rule_variant=RULE, opportunity_seed_offset=offset, restart_policy=RESTART,
        lead_seconds=20., heartbeat_seconds=5., preparation_margin_seconds=1.,
        search_handoff_reserve_seconds=4., ready_budget_seconds=19., search_budget_seconds=15.,
        nominal_k=8, max_k=12, learning=False, dynamic_k=False, lead_scan=False,
        objective_and_physics="UNCHANGED", formal_offset=0,
        comparison_scope="WHOLE_VERSION_BEFORE_AFTER_NOT_FULLY_FAIR_CONTROL")


def active_elapsed(data, now=None):
    """Includes every currently running child, not only settled receipts."""
    epoch_now=time.time() if now is None else now
    mono_now=time.monotonic() if now is None else now
    return data["settled_child_wall_s"] + math.fsum(max(0.,
        mono_now-row["started_monotonic"] if "started_monotonic" in row else epoch_now-row["started_epoch"])
        for row in data["jobs"].values() if row["status"] == "running")


@contextmanager
def ledger_transaction(output, *, write=True):
    """Short interprocess lock; never held while a subprocess runs."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output/"repair_budget.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0, 2)
            if lock.tell() == 0: lock.write(b"0"); lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            path = output/"repair_budget_ledger.json"
            data = base.load(path) if path.exists() else dict(schema="repair_child_wall_ledger_v1",
                cap_child_wall_s=CAP, settled_child_wall_s=PRIOR_AUDIT_S,
                prior_audits_child_wall_s=PRIOR_AUDIT_S, prior_audits="prior_audits",
                historical_campaign_used_s=HISTORICAL_USED_S,
                historical_campaign_budget_reset=False, jobs={}, no_automatic_retry=True)
            base.require(data["cap_child_wall_s"] == CAP and data["historical_campaign_used_s"] == HISTORICAL_USED_S,
                         "repair ledger limits/history changed")
            yield data
            if write:
                data["observed_active_child_wall_s"] = active_elapsed(data)
                data["updated_epoch"] = time.time()
                temporary=path.with_name(path.name+".partial")
                with temporary.open("w", encoding="utf8") as stream:
                    json.dump(data,stream,sort_keys=True,indent=2,allow_nan=False)
                    stream.flush(); os.fsync(stream.fileno())
                temporary.replace(path)
        finally:
            if os.name == "nt":
                lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _stop(process):
    if process.poll() is not None: return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    process.wait()


def run_bounded_job(output, job_id, command, *, cap_s, poll_s=.05):
    """Parent-enforced per-job and concurrent aggregate child wall limits.

    Used by Luna's explicit test/audit wrapper as well as the four entrypoints.
    A reused ID fails, including failed/timed-out attempts. Unknown active jobs
    remain charged until primary recovery; no stale-job credit is invented.
    """
    output = Path(output)
    base.require(cap_s > 0 and poll_s > 0 and poll_s <= .25, "bounded positive watchdog required")
    with ledger_transaction(output) as ledger:
        base.require(job_id not in ledger["jobs"], "job already attempted; no automatic retry")
        remaining = CAP-active_elapsed(ledger)
        base.require(remaining > 0, "repair child-wall budget exhausted")
        started = time.time(); started_mono=time.monotonic()
        ledger["jobs"][job_id] = dict(status="running", started_epoch=started, started_monotonic=started_mono,parent_pid=os.getpid(),
            command=list(map(str, command)), cap_s=cap_s, child_pid=None)
    log = output/"logs"/(job_id+".log"); log.parent.mkdir(parents=True, exist_ok=True)
    process = None; stop_reason = None; exit_code = None
    try:
        with log.open("x", encoding="utf8") as stream:
            process = subprocess.Popen(list(map(str, command)), cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt")
            with ledger_transaction(output) as ledger: ledger["jobs"][job_id]["child_pid"] = process.pid
            while process.poll() is None:
                with ledger_transaction(output, write=False) as ledger:
                    elapsed = time.monotonic()-started_mono
                    total = active_elapsed(ledger)
                    running = sum(row["status"] == "running" for row in ledger["jobs"].values())
                    # Reserve one polling quantum per active child and stop
                    # before the budget boundary, rather than detect it later.
                    if elapsed+poll_s >= cap_s: stop_reason = "JOB_WALL_LIMIT"
                    elif total+running*poll_s >= CAP: stop_reason = "CAMPAIGN_CHILD_WALL_LIMIT"
                if stop_reason: _stop(process); break
                time.sleep(poll_s)
            exit_code = process.wait()
    finally:
        if process is not None and process.poll() is None: _stop(process)
        ended = time.time(); elapsed = max(0., time.monotonic()-started_mono)
        with ledger_transaction(output) as ledger:
            row = ledger["jobs"][job_id]
            row.update(status="BUDGET_LIMITED" if stop_reason else "completed" if exit_code == 0 else "failed",
                finished_epoch=ended, child_wall_s=elapsed, exit_code=exit_code, stop_reason=stop_reason, log=str(log))
            ledger["settled_child_wall_s"] += elapsed
            result = deepcopy(row)
    return result


def credit_observed_cost(output, job_id, wall_s, *, evidence):
    """Import an already measured local preflight once; never run work here."""
    base.require(isinstance(wall_s,(float,int)) and not isinstance(wall_s,bool) and math.isfinite(wall_s) and wall_s>=0. and bool(evidence),
                 "finite measured cost and evidence required")
    with ledger_transaction(output) as ledger:
        base.require(job_id not in ledger["jobs"], "cost/job already recorded")
        ledger["jobs"][job_id]=dict(status="credited_preexisting_measurement",child_wall_s=wall_s,evidence=evidence)
        ledger["settled_child_wall_s"]+=wall_s
        exhausted=active_elapsed(ledger)>=CAP
    return dict(status="CREDITED",child_wall_s=wall_s,budget_exhausted=exhausted)


def validated_output(path):
    output = Path(path).resolve()
    base.require(output.name == OUTPUT_NAME and output.parent == (Path.home()/"training_outputs/terminal").resolve(),
                 "repair outputs must use the authorized fresh server directory")
    return output


def claim_snapshot_selection(output, *, root, frame_id, selected_ids, result_path, method, offset):
    """Atomically claim one solve for an identical preregistered selected set."""
    selected=sorted(selected_ids)
    key=base.digest(dict(root=root,frame_id=frame_id,selector_version=RULE,selected_ids=selected))
    with ledger_transaction(output) as ledger:
        registered=ledger.setdefault("snapshot_selected_sets", {})
        duplicate=deepcopy(registered.get(key))
        if duplicate is None:
            registered[key]=dict(result_path=str(result_path),method=method,offset=offset,
                status="REGISTERED_SOLVE_MAY_BE_RUNNING",root=root,frame_id=frame_id,
                selector_version=RULE,selected_ids=selected)
    return key,duplicate


def snapshot_scope_observations(records, preselected_ids):
    """Compare preselection with the first local call, before legal expansion."""
    telemetry=(records[0].get("selective_replanning") or {}) if records else {}
    calls=telemetry.get("calls") or []
    initial_scope=(calls[0].get("scope") or {}) if calls else {}
    returned_scope=telemetry.get("scope") or {}
    observed_initial=initial_scope.get("replan_ids")
    returned=returned_scope.get("replan_ids",returned_scope.get("selected_ids"))
    attempted=telemetry.get("attempted_replan_ids")
    if attempted is None and calls and all("replan_ids" in (call.get("scope") or {}) for call in calls):
        attempted=sorted({aid for call in calls for aid in call["scope"]["replan_ids"]})
    matches=None if observed_initial is None else sorted(observed_initial)==sorted(preselected_ids)
    return dict(initial_selected_ids=sorted(preselected_ids),
        observed_initial_call_selected_ids=None if observed_initial is None else sorted(observed_initial),
        returned_selected_ids=None if returned is None else sorted(returned),
        attempted_union=None if attempted is None else sorted(attempted),
        initial_scope_matches_preselection=matches,
        comparison_status="UNAVAILABLE_NO_INITIAL_CALL_SCOPE" if matches is None else "MATCH" if matches else "MISMATCH",
        comparison_basis="FIRST_LOCAL_CALL_SCOPE_BEFORE_ANY_BOUNDED_EXPANSION",
        observed_solve_calls=len(calls))


def binding(args, d):
    source = base.load(args.source_freeze); base.verify_sources(source)
    required=(DRIVER.relative_to(PROJECT).as_posix(), "scripts/stage3/report_selective_reoptimization_repair.py",
              "tests/test_rotating_slots_selective_repair_campaign.py")
    base.require(all(name in source["files"] for name in required), "reviewed repair runner/report/tests freeze required")
    with ledger_transaction(args.output) as ledger:
        bound_hash=ledger.setdefault("repair_source_hash",source["source_hash"])
        base.require(bound_hash == source["source_hash"],"repair source differs from the first attempted action")
    origin_path = args.origin_output/"resolved_run_binding.json"; origin = base.load(origin_path)
    base.require(origin["binding_id"] == base.digest({k:v for k,v in origin.items() if k != "binding_id"}), "original binding changed")
    item = deepcopy(origin["inputs"][args.root])
    base.require(base.file_hash(item["run_checkpoint"]) == item["initial_checkpoint_sha256"], "original t0 changed")
    base.require(item["row"]["seed"] == base.ROOTS[args.root][0], "original root seed changed")
    return dict(schema="rs_selective_repair_binding_v2", source=source, source_commit=source["commit"],
        source_hash=source["source_hash"], inputs={args.root:item}, root=args.root, seed=base.ROOTS[args.root][0],
        origin_input_binding=origin["binding_id"], origin_input_binding_sha256=base.file_hash(origin_path),
        treatment=treatment(), old_followup_readonly=str(args.followup_output), episode_seconds=300,
        historical_used_s=HISTORICAL_USED_S, new_child_wall_cap_s=CAP, no_automatic_retry=True,
        human_replay_status="HUMAN_PENDING", fullH_version="SAME_V2_COMMON_PROCESS_DIAGNOSTIC_ONLY")


def set_cpu(args, d):
    base.require(all(d.launch_threads.get(key) == "1" for key in base.THREADS), "one numerical thread at launch required")
    base.require(hasattr(os, "sched_setaffinity"), "server physical CPU affinity required")
    if args.stage in ("smoke", "arm"):
        base.require(args.cpu == ROOT_CPUS[args.root], "fixed root CPU12/14 required")
    resource = d.campaign.resource_probe()
    base.require(args.cpu in resource.get("free_physical_cores", []), "requested physical core unavailable")
    os.sched_setaffinity(0, {args.cpu})


def arm(args, d):
    set_cpu(args, d)
    output = args.output
    destination = output/("smokes" if args.stage == "smoke" else "arms")/args.root
    destination.mkdir(parents=True, exist_ok=False)
    row = binding(args, d); row["episode_seconds"] = 10 if args.stage == "smoke" else 300
    row["run_role"] = "SMOKE_NOT_FORMAL" if args.stage == "smoke" else "FORMAL_SINGLE_ARM"
    row["cpu"] = args.cpu; row["binding_id"] = base.digest(row)
    d.atomic_json(destination/"resolved_run_binding.json", row)
    result = base.closed_job(destination, args.root, "RULE_LOCAL_K8", row, d, lead_seconds=20.,
        selector_variant=RULE, episode_seconds=row["episode_seconds"], opportunity_seed_offset=0, restart_policy=RESTART)
    d.atomic_json(destination/"arm.finished.json", result)
    return result


def snapshot(args, d):
    """One cold restored 40s job, never an advancing physical rollout."""
    from terminal_marl.core.rotating_slots.selective_replanning import build_global_frame
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, EvaluationContext, PlanningBudget, QUALITY_BUDGET_PROFILE
    from terminal_marl.core.rotating_slots.performance.parallel import _clear_worker_accelerators
    from terminal_marl.core.rotating_slots.selective_replanning import RuleSelector
    from terminal_marl.core.rotating_slots.selective_replanning.relations import build_relations
    from terminal_marl.core.rotating_slots.stage2_planning import _decision_aircraft
    set_cpu(args, d)
    baseline = args.followup_output/"arms"/args.root/"baseline20"/"evaluation"/args.root/"RULE_LOCAL_K8"
    checkpoint = baseline/"checkpoint_0040.json"
    base.require(checkpoint.is_file(), "fixed baseline20 40s checkpoint missing; no replacement authorized")
    raw = base.load(checkpoint)
    base.require(raw["payload"]["time"] == 40. and raw["payload"]["identity"]["manifest_sha256"] == base.RAW_HASHES[args.root][0], "wrong checkpoint root/time")
    old_binding=base.load(args.followup_output/"arms"/args.root/"baseline20"/"resolved_run_binding.json")
    base.require(old_binding["treatment"]["selector_variant"] == "baseline" and old_binding["treatment"]["lead_seconds"] == 20.,
                 "fixed saved baseline20 treatment differs")
    directory = args.output/"snapshots"/args.root/("FULL_H" if args.method == "FULL_H" else f"offset{args.offset}")
    base.require(not directory.exists(), "snapshot already attempted")
    bound = binding(args, d)
    reset_started=time.perf_counter(); reset=_clear_worker_accelerators(); reset_s=time.perf_counter()-reset_started
    world = base.restore(checkpoint, d, directory)
    physical_before = base.digest(world.to_dict())
    source = world.build_planning_snapshot(); timing = d.PlanningTiming.from_lead(source.time, 20., 1.)
    method_started = time.perf_counter()
    projected = d.project_visible_snapshot(source, timing.effective_time, timing=timing)
    projection_s = time.perf_counter()-method_started
    frame_started = time.perf_counter()
    frame = build_global_frame(projected, CandidateProvider(planning_profile="downstream_service_v8"),
        EvaluationContext(projected, budget=PlanningBudget.for_profile(QUALITY_BUDGET_PROFILE)))
    frame_s = time.perf_counter()-frame_started
    callback = base.scheduler(d, args.method, frame, selector_variant=RULE, opportunity_seed_offset=args.offset)
    callback.initialize_imported_history(source, events=world.events)
    setup_s = time.perf_counter()-method_started-projection_s-frame_s
    preselection_started=time.perf_counter()
    if getattr(source, "planning_timing", None) is None: callback.selective_history.observe(source)
    relations=build_relations(projected, tuple(a.id for a in _decision_aircraft(projected)), callback.selective_blockers,
        frame=frame, variant=RULE)
    pre_scope, pre_telemetry=RuleSelector(variant=RULE).select(projected, frame, callback.selective_history, relations,
        method=args.method, now=source.time, references=callback.provider.references, opportunity_seed_offset=args.offset)
    preselection_s=time.perf_counter()-preselection_started
    selected=sorted(pre_scope.replan_ids); selected_digest=base.digest(selected)
    dedup_key,duplicate=claim_snapshot_selection(args.output,root=args.root,frame_id=frame.identity,selected_ids=selected,
        result_path=directory/"result.json",method=args.method,offset=args.offset)
    if duplicate is not None:
        result=dict(status="DUPLICATE_SELECTED_SET_NO_SOLVE",root=args.root,snapshot_time=40.,method=args.method,
            offset=args.offset,fixed_checkpoint=str(checkpoint),checkpoint_sha256=base.file_hash(checkpoint),
            physical_source_version=source.version,physical_source_unchanged=base.digest(world.to_dict()) == physical_before,
            physical_execution=False,actual_publication=False,cold_normal_cache=True,frame_id=frame.identity,
            selected_ids=selected,selected_set_digest=selected_digest,duplicate_of=duplicate,
            selected_ids_basis="INITIAL_SCOPE_DEDUPLICATION_SET",initial_selected_ids=selected,
            returned_selected_ids=None,attempted_union=None,
            deduplication_key=dedup_key,preselection_scope=vars(pre_scope),preselection_measurements=pre_telemetry,
            scheduler_records=[],selection_records=[],preparation=None,solve_executed=False,
            costs=dict(common_projection_s=projection_s,shared_public_frame_s=frame_s,cache_reset_s=reset_s,
                callback_setup_s=setup_s,preselection_s=preselection_s,
                observed_preselection_chain_s=time.perf_counter()-method_started+reset_s,
                full_online_pipeline_cost_s=None),
            treatment=treatment(args.method,args.offset),human_replay_status="NOT_APPLICABLE_NO_EXECUTION",
            shared_result_or_timing_counted_as_new_sample=False)
        d.atomic_json(directory/"result.json",result)
        return dict(status=result["status"],result_path=str(directory/"result.json"),duplicate_of=duplicate)
    port = d.FuturePort(projected, source, method_started+19., search_wall_deadline=method_started+15.,
        planning_wall_origin=method_started, search_handoff_reserve_seconds=4.)
    callback_started=time.perf_counter(); callback(port); callback_s=time.perf_counter()-callback_started
    preparation = base.counterfactual_preparation(world, port.proposal, timing, ready_wall_deadline=method_started+19.)
    unchanged = base.digest(world.to_dict()) == physical_before
    base.require(unchanged, "counterfactual preparation changed saved physical world/commitments")
    bound["treatment"] = treatment(args.method, args.offset); bound["binding_id"] = base.digest(bound)
    selections = base.selection_envelopes(callback.records, f"{args.root}:40:{args.method}:{args.offset}", bound["binding_id"], args.root)
    scope_observations=snapshot_scope_observations(callback.records,selected)
    frame_observed=bool(selections) and all(r["scope"]["objective_frame_id"] == frame.identity for r in selections)
    status=("INCOMPLETE_SCOPE_EVIDENCE" if scope_observations["initial_scope_matches_preselection"] is None else
        "PRESELECTION_SCOPE_MISMATCH" if scope_observations["initial_scope_matches_preselection"] is False else
        "SHARED_FRAME_UNAVAILABLE_OR_MISMATCH" if not frame_observed else "COMPLETE")
    result = dict(status=status, root=args.root, snapshot_time=40., method=args.method, offset=args.offset,
        fixed_checkpoint=str(checkpoint), checkpoint_sha256=base.file_hash(checkpoint), physical_source_version=source.version,
        physical_source_unchanged=unchanged, physical_execution=False, actual_publication=False, cold_normal_cache=True,
        frame_id=frame.identity, frame_identity_basis="SAME_CHECKPOINT_PROJECTED_EFFECTIVE60_GLOBAL_FRAME",
        selected_ids=selected, selected_set_digest=selected_digest, selection_records=selections,
        selected_ids_basis="INITIAL_SCOPE_DEDUPLICATION_SET",scope_observations=scope_observations,
        initial_selected_ids=scope_observations["initial_selected_ids"],
        returned_selected_ids=scope_observations["returned_selected_ids"],attempted_union=scope_observations["attempted_union"],
        callback_frame_observed_matches=frame_observed,
        preselection_scope=vars(pre_scope),preselection_measurements=pre_telemetry,
        deduplication_key=dedup_key,solve_executed=bool(scope_observations["observed_solve_calls"]),
        scheduler_records=callback.records, preparation=preparation, cache_reset=reset,
        costs=dict(common_projection_s=projection_s, shared_public_frame_s=frame_s, cache_reset_s=reset_s,
            callback_setup_s=setup_s, preselection_s=preselection_s, callback_s=callback_s, preparation_s=preparation["wall_s"],
            observed_preparation_chain_s=time.perf_counter()-method_started+reset_s,
            full_online_pipeline_cost_s=None),
        unobserved_online_stages=["AFFECTED_TRAFFIC_REVALIDATION", "ONLINE_SERVICE_DELAY", "ACTUAL_DELIVERY"],
        treatment=bound["treatment"], human_replay_status="NOT_APPLICABLE_NO_EXECUTION")
    d.atomic_json(directory/"result.json", result)
    return dict(status=status, result_path=str(directory/"result.json"), frame_id=frame.identity,
        initial_selected_ids=selected,returned_selected_ids=scope_observations["returned_selected_ids"],
        attempted_union=scope_observations["attempted_union"])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "snapshot", "arm", "report"))
    parser.add_argument("--output", type=Path, default=Path.home()/"training_outputs/terminal"/OUTPUT_NAME)
    parser.add_argument("--origin-output", type=Path, default=Path.home()/"training_outputs/terminal/rs_selective_reoptimization_a_20261027")
    parser.add_argument("--followup-output", type=Path, default=Path.home()/"training_outputs/terminal/rs_selective_followup_20261028")
    parser.add_argument("--source-freeze", type=Path)
    parser.add_argument("--root", choices=tuple(ROOT_CPUS)); parser.add_argument("--cpu", type=int)
    parser.add_argument("--method", choices=base.METHODS, default="RULE_LOCAL_K8")
    parser.add_argument("--offset", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--_child-job-id", help=argparse.SUPPRESS)
    args=parser.parse_args(argv); args.output=validated_output(args.output)
    if args.stage != "report":
        base.require(args.root and args.cpu is not None and args.source_freeze, "root/cpu/source-freeze required")
        base.require(args.stage == "snapshot" or (args.method == "RULE_LOCAL_K8" and args.offset == 0), "formal and smoke use fixed K8 offset0")
        base.require(args.method != "FULL_H" or args.offset == 0, "FULL_H diagnostic has one fixed offset")
    suffix = "" if args.stage == "report" else "_"+args.root+("_"+args.method+"_"+str(args.offset) if args.stage == "snapshot" else "")
    job_id = args.stage+suffix
    if not args._child_job_id:
        command = [sys.executable, str(DRIVER), *(sys.argv[1:] if argv is None else argv), "--_child-job-id", job_id]
        receipt=run_bounded_job(args.output, job_id, command, cap_s=1200. if args.stage == "arm" else 180.)
        base.print_result(dict(job_id=job_id, **receipt)); return
    base.require(args._child_job_id == job_id, "child job identity differs")
    with ledger_transaction(args.output, write=False) as ledger:
        row=ledger["jobs"].get(job_id, {})
        base.require(row.get("status") == "running" and row.get("parent_pid") == os.getppid(), "active parent watchdog required")
    if args.stage == "report":
        from scripts.stage3.report_selective_reoptimization_repair import report
        result=report(args.output, args.followup_output)
    else:
        d=base.dependencies(PROJECT); d.launch_threads={k:os.environ.get(k) for k in base.THREADS}
        result=snapshot(args,d) if args.stage == "snapshot" else arm(args,d)
    base.print_result(result)


if __name__ == "__main__": main()
