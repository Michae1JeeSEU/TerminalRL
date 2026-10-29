"""Stage A4's one executed common initial state and cold method clones.

This module does not launch a campaign. Callers must charge initialization,
restoration and all physical advances to the enclosing bounded A4 child ledger.
The initializer always uses the frozen old downstream objective.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.stage3 import run_selective_reoptimization_a as base

RULE = "opportunity_reference_v4"
RESTART = "coalesced_restart_v2"
OUTPUT_NAME = "rs_selective_quality_a4_20261030"
INITIALIZATION_SOURCE_LIMIT_S = 60.
INITIALIZATION_CHILD_CAP_S = 300.
NEW_CHILD_WALL_CAP_S = 7200.
METHOD_HORIZON_S = 300.
SWITCHES = dict(pad_anchor_priority=False, target_coverage_order=False,
                ordinary_immediate_filter=False)


def _dependencies(d):
    return base.dependencies(base.PROJECT) if d is None else d


def common_callback(d, method, *, evaluation_profile=None):
    # A nonempty evaluation profile needs the explicit A4 objective API. Never
    # silently substitute a score or change the initializer's objective.
    base.require(evaluation_profile is None, "A4 evaluation profile API not bound")
    return d.NormalSchedulerCallback("rh_cp_sat", planning_profile="downstream_service_v8",
        online_policy="progressive_v1", selective_method=method,
        selective_rule_variant=RULE, opportunity_seed_offset=0, search_policy=dict(SWITCHES))


def clean_boundary(world, wrapper):
    """Predeclared first nonempty normal delivery, before its successor starts."""
    records = wrapper.records
    if wrapper.pending is not None or not records:
        return False
    record = records[-1]
    if record.get("status") != "COMMITTED" or record.get("submitted", 0) <= 0:
        return False
    commits = [r for r in records if r.get("status") == "COMMITTED" and r.get("submitted", 0) > 0]
    base.require(len(commits) == 1, "missed the first nonempty common initialization delivery")
    base.require(records[0]["snapshot_time"] == 0. and
        sum(r["snapshot_time"] == 0. for r in records) == 1, "common initialization source0 task differs")
    base.require(record["finished_time"] == world.time, "common initialization delivery boundary changed")
    base.require(not any(p.start_time <= world.time for p in world.soft.values()),
                 "due soft action was not activated before the common boundary")
    return True


def inherited_finals(world):
    return [dict(aircraft_id=plan.aircraft_id, trajectory_hash=plan.trajectory_hash,
                 kind=plan.kind, start_time=plan.start_time, end_time=plan.end_time, pad_id=plan.pad_id,
                 commitment_state=state)
            for state, plans in (("active", world.active), ("soft", world.soft))
            for plan in sorted(plans.values(), key=lambda p: p.aircraft_id)
            if plan.kind == "FINAL_NORMAL"]


def baseline_observation(world):
    from terminal_marl.core.rotating_slots.stage2_campaign import episode_metrics
    return dict(metrics=episode_metrics(world), event_count=len(world.events),
        frame_count=world._record_count, dispatch_count=len(world.dispatch_timings),
        energy_by_aircraft={aid:a.energy_j for aid,a in world.aircraft.items()},
        holding_by_aircraft={aid:a.holding_time for aid,a in world.aircraft.items()},
        touchdown_ids=sorted(aid for aid,a in world.aircraft.items() if a.touchdown_time is not None))


def increment_observation(world, common_summary):
    """Absolute physics stays untouched; report only the post-common interval."""
    from terminal_marl.core.rotating_slots.stage2_campaign import episode_metrics
    baseline = common_summary["baseline"]
    initial = baseline["metrics"]
    current = episode_metrics(world)
    cumulative = ("L_success", "L_normal", "L_emergency", "failed_before_landing",
                  "F_energy_total", "energy_j", "energy_kwh", "airborne_time",
                  "upstream_wait_time", "holding_time", "cost_emergency", "cost_reroute")
    delta = {name:current[name]-initial[name] for name in cumulative
             if type(current.get(name)) in (int,float) and type(initial.get(name)) in (int,float)}
    previous_touchdowns = set(baseline["touchdown_ids"])
    new_touchdowns = sorted(aid for aid,a in world.aircraft.items()
                            if a.touchdown_time is not None and aid not in previous_touchdowns)
    inherited = {r["aircraft_id"]:r for r in common_summary["inherited_finals"]}
    inherited_touchdowns = [aid for aid in new_touchdowns if aid in inherited and
        world.aircraft[aid].touchdown_time == inherited[aid]["end_time"]]
    return dict(t0=common_summary["t0"], time=world.time, elapsed_s=world.time-common_summary["t0"],
        cumulative_delta=delta,
        ending_pending={name:current[name] for name in
            ("N_airborne_pending", "N_upstream_pending", "airborne_or_censored") if name in current},
        new_touchdown_ids=new_touchdowns,
        inherited_final_touchdown_ids=inherited_touchdowns,
        treatment_final_touchdown_ids=[aid for aid in new_touchdowns if aid not in inherited_touchdowns],
        post_t0_event_count=len(world.events)-baseline["event_count"],
        post_t0_frame_count=world._record_count-baseline["frame_count"],
        post_t0_dispatch_count=len(world.dispatch_timings)-baseline["dispatch_count"],
        energy_by_aircraft={aid:a.energy_j-baseline["energy_by_aircraft"].get(aid,0.)
                            for aid,a in world.aircraft.items()})


def initialize_common(checkpoint, directory, *, d=None, identity=None, source_binding=None):
    """Physically advance once from original t=0; never reset the reached state."""
    started = time.perf_counter()
    d = _dependencies(d)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    envelope = base.load(checkpoint)
    original_identity = envelope["payload"]["identity"]
    cold = base.ColdRestore()
    world = d.restore_checkpoint(checkpoint, cold, original_identity,
                                 stream_copy=directory/"restored_states.jsonl")
    base.require(world.time == 0. and world.scheduler_period == 5., "original t0 and heartbeat required")
    world.configure_practical_runtime(True)
    callback = common_callback(d, "RULE_LOCAL_K8")
    callback.initialize_imported_history(world.build_planning_snapshot(), events=world.events)
    wrapper = d.LatencyAwareScheduler(callback, lead_seconds=20., preparation_margin=1.,
        search_handoff_reserve_seconds=4., restart_policy=RESTART)
    world.decision_callback, world.admission_callback = wrapper, d.shared_fifo_admission
    reached = []
    def stop_at_first_delivery(current, latency):
        if clean_boundary(current, latency):
            reached.append(current.time)
            return True
        return False
    error = None
    try:
        world.advance(INITIALIZATION_SOURCE_LIMIT_S, wrapper, settle_only_at_end=False,
                      decision_boundary_hook=stop_at_first_delivery)
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    status = "COMMON_INITIAL_READY" if reached and error is None and not world.failed else "INIT_NOT_REACHED"
    summary = dict(schema="rs_selective_common_initial_a4_v1", status=status,
        t0=world.time, source_limit_s=INITIALIZATION_SOURCE_LIMIT_S,
        original_checkpoint=str(Path(checkpoint).resolve()), original_checkpoint_sha256=base.file_hash(checkpoint),
        source_binding=source_binding, method="RULE_LOCAL_K8", selective_rule_variant=RULE,
        search_policy=dict(SWITCHES), planning_profile="downstream_service_v8", restart_policy=RESTART,
        lead_seconds=20., heartbeat_seconds=5., normal_task_count=len(wrapper.records),
        pending=wrapper.pending is not None, inherited_finals=inherited_finals(world),
        baseline=baseline_observation(world), wall_s=time.perf_counter()-started, error=error,
        checkpoint_state="FULL_WORLD_WRAPPER_HISTORY_NO_RESET", retry_policy="NO_AUTOMATIC_RETRY")
    if status == "COMMON_INITIAL_READY":
        base.require(clean_boundary(world,wrapper), "common initialization was not stopped at clean boundary")
        d.save_checkpoint(directory/"common.checkpoint.json", world, wrapper,
                          original_identity if identity is None else identity, boundary_summary=summary)
    else:
        d.save_checkpoint(directory/"not_reached.checkpoint.json", world, wrapper,
                          original_identity if identity is None else identity, boundary_summary=summary)
    for name,value in (("boundary_summary.json",summary), ("initializer_tasks.json",wrapper.records),
                       ("initializer_callback_records.json",callback.records),
                       ("initializer_services.json",wrapper.online_services)):
        d.atomic_json(directory/name,value)
    return summary


def restore_common(checkpoint, method, evaluation_profile=None, *, d=None, directory=None):
    """Clone exact physics/history, cold algorithms and preserved CPU service clocks."""
    d = _dependencies(d)
    base.require(method in base.METHODS, "fixed A4 method required")
    envelope = base.load(checkpoint)
    summary = deepcopy(envelope["payload"]["boundary_summary"])
    base.require(summary and summary["status"] == "COMMON_INITIAL_READY", "ready common checkpoint required")
    cold = base.ColdRestore()
    copy_path = None
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        copy_path = directory/"restored_states.jsonl"
    # A source sink must always be copied; restoration truncates an attached
    # stream to its checkpoint offset and must not touch the common artifact.
    base.require(envelope["payload"]["sink"] is None or copy_path is not None,
                 "streamed common checkpoint requires an independent clone directory")
    world = d.restore_checkpoint(checkpoint, cold, envelope["payload"]["identity"], stream_copy=copy_path)
    saved = cold.saved
    base.require(saved["pending"] is None, "common checkpoint contains pending normal work")
    from terminal_marl.core.rotating_slots.performance.parallel import _clear_worker_accelerators
    cache_reset = _clear_worker_accelerators()
    callback = common_callback(d,method,evaluation_profile=evaluation_profile)
    selective = saved["callback"]["selective_replanning"]
    base.require(selective["method"] == "RULE_LOCAL_K8" and selective["rule_variant"] == RULE,
                 "unexpected initializer history binding")
    callback.selective_history.load_state_dict(deepcopy(selective["history"]))
    # Background blockers are prior search products, rather than executed age
    # history. Both treatments rebuild them from their own cold first search.
    callback.selective_blockers = []
    wrapper = d.LatencyAwareScheduler(callback,lead_seconds=20.,preparation_margin=1.,
        search_handoff_reserve_seconds=4.,restart_policy=RESTART)
    # Load the exact scheduling clocks through its own binding validator while
    # replacing algorithm products and historical normal/service record arrays.
    resumed = deepcopy(saved)
    resumed.update(callback=callback.state_dict(), records=[], online_services=[])
    if resumed.get("restart_requested") is not None:
        resumed["restart_requested"]["record_index"] = None
        resumed["restart_requested"]["external_predecessor"] = dict(
            common_checkpoint_sha256=base.file_hash(checkpoint),
            initializer_record_index=saved["restart_requested"]["record_index"],
            initializer_finished_time=saved["restart_requested"]["finished_time"])
    wrapper.load_state_dict(resumed)
    world.decision_callback, world.admission_callback = wrapper, d.shared_fifo_admission
    metadata = dict(common_summary=summary, t0=world.time, horizon=[world.time,world.time+METHOD_HORIZON_S],
        checkpoint_sha256=base.file_hash(checkpoint), evaluation_profile=evaluation_profile,
        cache_reset=cache_reset, normal_records="NEW_EMPTY", initializer_records="PROVENANCE_ONLY",
        initializer_tasks=deepcopy(saved["records"]), initializer_services=deepcopy(saved["online_services"]),
        initializer_callback_records=deepcopy(saved["callback"]["records"]),
        initializer_blockers=deepcopy(selective.get("blockers",[])),
        inherited_restart=deepcopy(saved.get("restart_requested")))
    return world,callback,wrapper,metadata


def resume_common(world, wrapper):
    """Start the first treatment normal task at t0, charging inherited CPU queues."""
    base.require(wrapper.pending is None and not wrapper.records,
                 "common method may be resumed exactly once before normal work")
    previous = wrapper._restart_requested
    base.require(previous is not None and previous.get("external_predecessor") and
                 previous["finished_time"] == world.time, "common same-time predecessor required")
    base.require(wrapper.restart_coalesced(world._scheduler_port), "common same-time restart did not start")
    base.require(len(wrapper.records) == 1 and wrapper.records[0]["snapshot_time"] == world.time,
                 "first treatment normal task was delayed from common t0")
    return wrapper.records[0]


def main(argv=None):
    from scripts.stage3 import run_selective_quality_gap_a4 as a4
    from scripts.stage3 import run_selective_reoptimization_repair as old
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("initialize","initialize-child"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--origin-output",type=Path,required=True)
    parser.add_argument("--source-freeze",type=Path,required=True)
    parser.add_argument("--root",choices=tuple(base.ROOTS),required=True)
    parser.add_argument("--cpu",type=int,required=True)
    args=parser.parse_args(argv)
    args.output=a4.validated_output(args.output)
    source=base.load(args.source_freeze)
    base.verify_sources(source)
    base.require(Path(__file__).relative_to(PROJECT).as_posix() in source["files"],
                 "common-initial runner missing from reviewed freeze")
    if args.stage=="initialize":
        command=[sys.executable,str(Path(__file__).resolve()),"--stage","initialize-child",
            "--output",str(args.output),"--origin-output",str(args.origin_output),
            "--source-freeze",str(args.source_freeze),"--root",args.root,"--cpu",str(args.cpu)]
        result=a4.run_bounded_job(args.output,"common_initial:"+args.root,command,
                                  cap_s=INITIALIZATION_CHILD_CAP_S)
        with a4.ledger_transaction(args.output) as ledger:
            ledger["jobs"]["common_initial:"+args.root]["run_role"]="EXECUTED_COMMON_INITIALIZATION"
        return result
    with a4.ledger_transaction(args.output,write=False) as ledger:
        job=ledger["jobs"].get("common_initial:"+args.root,{})
        base.require(job.get("status")=="running" and job.get("parent_pid")==os.getppid(),
                     "active common-initial parent watchdog required")
    d=_dependencies(None)
    # Resource probing and affinity are inside the charged child. No world is
    # constructed until the original root and reviewed source are bound.
    resource_args=argparse.Namespace(stage="snapshot",cpu=args.cpu,root=args.root)
    old.set_cpu(resource_args,d)
    origin_path=args.origin_output/"resolved_run_binding.json"
    origin=base.load(origin_path)
    base.require(origin["binding_id"]==base.digest({k:v for k,v in origin.items() if k!="binding_id"}),
                 "original Stage A binding changed")
    item=origin["inputs"][args.root]
    base.require((item["row"]["seed"],item["row"]["scenario_profile"])==base.ROOTS[args.root][:2],
                 "common initialization root differs")
    checkpoint=Path(item["run_checkpoint"])
    base.require(base.file_hash(checkpoint)==item["initial_checkpoint_sha256"], "original t0 checkpoint changed")
    with a4.ledger_transaction(args.output) as ledger:
        bound=ledger.setdefault("common_initial_source_hash",source["source_hash"])
        base.require(bound==source["source_hash"], "A4 common source differs from first common initialization attempt")
    identity=dict(base.load(checkpoint)["payload"]["identity"],source_hash=source["source_hash"])
    summary=initialize_common(checkpoint,args.output/"common_initial"/args.root,d=d,identity=identity,
        source_binding=dict(root=args.root,source_hash=source["source_hash"],source_commit=source["commit"],
            origin_binding=origin["binding_id"],origin_binding_sha256=base.file_hash(origin_path),
            run_role="EXECUTED_COMMON_INITIALIZATION",pid=os.getpid(),cpu=args.cpu))
    base.print_result(summary)
    return summary


if __name__=="__main__":
    main()
