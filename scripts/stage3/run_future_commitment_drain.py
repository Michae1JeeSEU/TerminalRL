"""Resume the saved R4 world to t600, then drain its admitted aircraft.

Each invocation owns one arm and one wall-bounded segment. Only explicitly
PAUSED_CHECKPOINT segments may resume; no new seeds or scheduler resets occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_common_initial_a4 as common
from scripts.stage3 import run_selective_common_comparison_a4 as comparison
from scripts.stage3 import run_service_opportunity_comparison as campaign
from terminal_marl.core.rotating_slots import stage2_campaign as archive
from terminal_marl.core.rotating_slots.demand import ArrivalManifest
from terminal_marl.core.rotating_slots.replay import ReplayBundle, _plain
from terminal_marl.core.rotating_slots.world import AIRBORNE

DRIVER = Path(__file__).resolve()
OUTPUT_NAME = "rs_future_commitment_drain_20261101"
START = 320.
CUTOFF = 600.
STEP = 5.
CHECKPOINT_INTERVAL = 60.


def write(path, value):
    archive.atomic_json(path, _plain(value))


def verify_algorithm_source(source):
    """Check frozen algorithm/config bytes; historical prose is not executable."""
    base.require(source["source_hash"] == base.digest(source["files"]), "original freeze digest differs")
    selected = {name: expected for name, expected in source["files"].items()
                if name.startswith("terminal_marl/core/rotating_slots/")
                or name.startswith("new10.16/") and name.endswith((".yaml", ".yml", ".json"))}
    base.require("terminal_marl/core/rotating_slots/world.py" in selected, "algorithm freeze missing world")
    for name, expected in selected.items():
        relative = Path(name)
        base.require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
        base.require(base.file_hash(PROJECT / relative) == expected, "original algorithm/config bytes differ: " + name)
    return selected


def code_binding():
    names = ("scripts/stage3/run_future_commitment_drain.py", "tests/test_future_commitment_drain.py",
             "scripts/stage3/run_selective_reoptimization_a.py", "scripts/stage3/run_selective_common_initial_a4.py",
             "scripts/stage3/run_selective_common_comparison_a4.py", "scripts/stage3/run_service_opportunity_comparison.py",
             "scripts/stage3/run_latency_interface.py", "scripts/stage3/report_selective_reoptimization_a.py",
             "scripts/stage3/report_latency_interface.py")
    files = {name: base.file_hash(PROJECT / name) for name in names}
    return dict(files=files, source_hash=base.digest(files))


def source_inputs(source_output, root, method, source):
    directory = comparison.evaluation_directory(source_output, root, method)
    binding = base.load(Path(source_output) / "arms" / root / method / "resolved_run_binding.json")
    result = base.load(directory / "result.json")
    checkpoint = directory / "checkpoint_terminal.json"
    envelope = base.load(checkpoint)
    payload = envelope["payload"]
    base.require(envelope["sha256"] == archive.digest(payload), "original checkpoint digest differs")
    base.require(binding["source_hash"] == source["source_hash"] and binding["source_commit"] == source["commit"],
                 "original source binding differs")
    base.require(binding["binding_id"] == base.digest({key:value for key,value in binding.items() if key != "binding_id"}),
                 "original arm binding digest differs")
    base.require(binding["root"] == root and binding["method"] == method and
                 binding["evaluation_profile"] == campaign.PROFILE and binding["horizon"] == [20., 320.],
                 "original arm/profile/horizon differs")
    base.require(result["status"] == "COMPLETE" and result["time"] == START and payload["time"] == START,
                 "only completed original t320 arms are accepted")
    base.require(payload["identity"]["source_hash"] == source["source_hash"] and
                 payload["boundary_summary"]["source_binding"] == binding["binding_id"] and
                 result["source_binding"] == binding["binding_id"], "original checkpoint/result arm binding differs")
    common_dir = Path(source_output) / "common_initial" / root
    metadata = dict(common_summary=base.load(common_dir / "boundary_summary.json"),
                    initializer_tasks=base.load(common_dir / "initializer_tasks.json"),
                    initializer_callback_records=base.load(common_dir / "initializer_callback_records.json"),
                    initializer_services=base.load(common_dir / "initializer_services.json"),
                    t0=20., checkpoint_sha256=binding["common_checkpoint_sha256"])
    base.require(base.file_hash(common_dir / "common.checkpoint.json") == binding["common_checkpoint_sha256"],
                 "original common provenance differs")
    return checkpoint, payload, binding, metadata


def restored_state_checks(world, callback, wrapper, payload):
    codec = archive.Stage2Codec()
    saved_world = codec.decode(payload["world"])
    saved_scheduler = codec.decode(payload["scheduler"])
    actual = wrapper.state_dict()
    checks = dict(world_fields=all(codec.encode(getattr(world, key)) == codec.encode(value)
                                 for key, value in saved_world.items()),
                  scheduler=codec.encode(actual) == codec.encode(saved_scheduler),
                  time=world.time == payload["time"], scheduler_period=world.scheduler_period == STEP)
    return dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                pending_stage=None if wrapper.pending is None else wrapper.pending["stage"],
                future_reservations=len(getattr(world, "_future_reservations", {})),
                persisted_cache_scope="WORLD_CACHES_CALLBACK_PREVIOUS_REFERENCES_TEMPLATES_HISTORY",
                nonpersisted_provider_validation_cache_size=len(callback.provider.validation_cache),
                physical_advance_calls=0, solver_calls=0)


def restore(checkpoint, payload, method, stream_copy, *, cutoff=None, d=None):
    d = base.dependencies(PROJECT) if d is None else d
    callback = common.common_callback(d, method, evaluation_profile=campaign.PROFILE)
    wrapper = d.LatencyAwareScheduler(callback, lead_seconds=20., preparation_margin=1.,
        search_handoff_reserve_seconds=4., restart_policy=common.RESTART)
    world = d.restore_checkpoint(checkpoint, wrapper, payload["identity"], stream_copy=stream_copy)
    checks = restored_state_checks(world, callback, wrapper, payload)
    base.require(checks["status"] == "PASS", "full persisted restore differs")
    world.decision_callback = wrapper
    if cutoff is not None:
        base.require(world.time >= CUTOFF and archive.digest(world.manifest.to_dict()) == cutoff["manifest_sha256"],
                     "drain checkpoint cutoff identity differs")
        world.admission_callback = None
    return world, callback, wrapper, checks


def statistics(world):
    admitted = [a for a in world.aircraft.values() if a.admitted and not
                (a.fixture and a.touchdown_time == 0. and a.airborne_time == 0.)]
    return dict(time=world.time, energy_j=sum(a.energy_j for a in world.aircraft.values()),
                normal=sum(a.touchdown_time is not None and not a.emergency.charged for a in admitted),
                emergency=sum(a.touchdown_time is not None and a.emergency.charged for a in admitted),
                energy_failed=sum(a.status == "ENERGY_MISSION_FAILED" for a in admitted),
                admitted=len(admitted), released=world._manifest_index,
                airborne=sum(a.status in AIRBORNE for a in world.aircraft.values()),
                upstream_pending=sum(key not in world.aircraft for key in world.requests))


def delta(after, before):
    return {key: after[key] - before[key] for key in after if key != "time"} | dict(
        start_time=before["time"], end_time=after["time"])


def initial_finals(world):
    return [dict(aircraft_id=plan.aircraft_id,trajectory_hash=plan.trajectory_hash,kind=plan.kind,
                 start_time=plan.start_time,end_time=plan.end_time,source=source)
            for source,plans in (("active",world.active),("soft",world.soft))
            for plan in plans.values() if plan.kind in ("FINAL_NORMAL","FINAL_EMERGENCY")]


def apply_cutoff(world, identity):
    base.require(world.time == CUTOFF, "input cutoff requires exact settled t600")
    original = world.manifest
    released = original.requests[:world._manifest_index]
    suppressed = original.requests[world._manifest_index:]
    base.require(all(r.release_time <= CUTOFF + 1.e-8 for r in released), "released prefix extends beyond cutoff")
    base.require(all(r.release_time > CUTOFF + 1.e-8 for r in suppressed), "cutoff contains unreleased due requests")
    receipt = dict(time=CUTOFF, original_manifest_sha256=archive.digest(original.to_dict()),
        original_manifest_count=len(original.requests), original_arrival_window=original.arrival_window,
        suppressed_request_ids=[r.id for r in suppressed], suppressed_count=len(suppressed),
        excluded_upstream_ids=sorted(key for key in world.requests if key not in world.aircraft),
        excluded_upstream_status="EXTERNAL_PENDING_EXCLUDED_NOT_FAILED",
        cohort_airborne_ids=sorted(a.id for a in world.aircraft.values() if a.status in AIRBORNE),
        admitted_ids=sorted(a.id for a in world.aircraft.values() if a.admitted and a.status != "REROUTED"
                            and not (a.fixture and a.touchdown_time == 0. and a.airborne_time == 0.)),
        event_index=len(world.events), statistics=statistics(world))
    world.manifest = ArrivalManifest(tuple(released), original.seed, original.rate_per_hour, CUTOFF)
    world.admission_callback = None
    identity = dict(identity, manifest_sha256=archive.digest(world.manifest.to_dict()))
    receipt["manifest_sha256"] = identity["manifest_sha256"]
    return identity, receipt


def aircraft_outcome(world, aircraft):
    failure = getattr(world, "failure_archive", {}).get(aircraft.id)
    if aircraft.touchdown_time is not None and aircraft.status in ("LANDED_SERVICE", "CLEARED"):
        outcome = "EMERGENCY_LANDING" if aircraft.emergency.charged else "NORMAL_LANDING"
        terminal_time = aircraft.touchdown_time
        terminal_soc = aircraft.soc
    elif aircraft.status == "ENERGY_MISSION_FAILED" and failure is not None:
        reason = failure["reason"]
        outcome = reason if reason in ("BATTERY_DEPLETED", "PEAK_BUDGET_EXHAUSTED") else "UNKNOWN_TERMINAL"
        terminal_time, terminal_soc = failure["time"], aircraft.soc
    elif aircraft.status in AIRBORNE:
        outcome, terminal_time, terminal_soc = "AIRBORNE_PENDING", None, None
    else:
        outcome, terminal_time, terminal_soc = "UNKNOWN_TERMINAL", None, None
    return dict(aircraft_id=aircraft.id, status=aircraft.status, outcome=outcome,
                terminal_time=terminal_time, terminal_soc=terminal_soc,
                current_soc=aircraft.soc, energy_j=aircraft.energy_j,
                failure_reason=None if failure is None else failure["reason"])


def completion(world, cutoff):
    if world.failed or getattr(world, "terminated_invalid", False):
        return "INVALID"
    if cutoff is None:
        return "TRAFFIC"
    rows = [aircraft_outcome(world, world.aircraft[key]) for key in cutoff["cohort_airborne_ids"]]
    admitted = [aircraft_outcome(world, world.aircraft[key]) for key in cutoff["admitted_ids"]]
    if any(row["outcome"] == "UNKNOWN_TERMINAL" for row in admitted):
        return "UNKNOWN_TERMINAL"
    if any(a.status in AIRBORNE for a in world.aircraft.values()) or any(
            row["outcome"] == "AIRBORNE_PENDING" for row in rows):
        return "DRAINING"
    return "COMPLETE"


def checkpoint(directory, world, wrapper, identity, state, status):
    """Publish stable checkpoint before atomically switching the resume pointer."""
    name = f"checkpoint_s{state['segment']:04d}_t{world.time:012.5f}_{status}.json"
    path = directory / name
    summary = dict(schema="future_commitment_drain_checkpoint_v1", status=status,
                   continuation=state, statistics=statistics(world))
    archive.save_checkpoint(path, world, wrapper, identity, boundary_summary=summary)
    write(directory / "latest.json", dict(path=name, sha256=base.file_hash(path), status=status,
        time=world.time, segment=state["segment"], source_binding=state["binding_id"]))
    return path


def save_final(directory, world, callback, wrapper, metadata, binding, state, status, error):
    callbacks, tasks, services, mapping = comparison.merged_records(metadata, callback, wrapper)
    mapping["calculation_event_rows"] = comparison.calculation_event_mapping(world, metadata, mapping["offsets"]["tasks"])
    for name, rows in (("scheduler_records.json", callbacks), ("normal_tasks.json", tasks),
                       ("online_services.json", services), ("record_index_mapping.json", mapping),
                       ("method_scheduler_records.json", callback.records), ("method_normal_tasks.json", wrapper.records),
                       ("method_online_services.json", wrapper.online_services), ("dispatch_timings.json", world.dispatch_timings)):
        write(directory / name, rows)
    cutoff = state.get("cutoff")
    outcomes = [] if cutoff is None else [aircraft_outcome(world, world.aircraft[key]) for key in cutoff["admitted_ids"]]
    cohort = [] if cutoff is None else [row for row in outcomes if row["aircraft_id"] in cutoff["cohort_airborne_ids"]]
    after_cutoff = [] if cutoff is None else world.events[cutoff["event_index"]:]
    terminal = statistics(world)
    inherited_finals=[]
    for row in state["initial_finals"]:
        aircraft=world.aircraft[row["aircraft_id"]]
        outcome=aircraft_outcome(world,aircraft)
        inherited_finals.append(dict(row,outcome=outcome,
            original_final_completed_to_touchdown=aircraft.touchdown_time is not None and
                aircraft.trajectory_hash==row["trajectory_hash"]))
    result = dict(schema="future_commitment_drain_result_v1", status=status, root_id=binding["root"], method=binding["method"],
        time=world.time, source_binding=binding["binding_id"], error=error, segment=state["segment"],
        initial=state["initial"], at600=None if cutoff is None else cutoff["statistics"], terminal=terminal,
        traffic_increment=None if cutoff is None else delta(cutoff["statistics"], state["initial"]),
        drain_increment=None if cutoff is None else delta(terminal, cutoff["statistics"]),
        normal_landed_before320=state["initial"]["normal"],
        emergency_landed_before320=state["initial"]["emergency"], initial_finals=state["initial_finals"],
        inherited_final_outcomes=inherited_finals,
        last_cohort_terminal_time=max((row["terminal_time"] for row in cohort if row["terminal_time"] is not None),default=None),
        cutoff=cutoff,
        admitted_outcomes=outcomes, cutoff_airborne_outcomes=cohort,
        outcome_counts=dict(Counter(row["outcome"] for row in outcomes)),
        arrivals_after600=sum(e["type"] == "request_released" for e in after_cutoff),
        admissions_after600=sum(e["type"] == "admitted" for e in after_cutoff),
        reservation_registry_unexpected_live=[dict(reservation_id=key,aircraft_id=row.get("aircraft_id"),state=row.get("state"))
            for key,row in getattr(world,"_future_reservations",{}).items()
            if row.get("state") in ("RESERVED","EXECUTING") and
            (row.get("aircraft_id") not in world.aircraft or world.aircraft[row["aircraft_id"]].status not in AIRBORNE)],
        human_replay_status="HUMAN_PENDING", terminal_observation=dict(failed=world.failed,
            terminated_invalid=getattr(world, "terminated_invalid", False)))
    if status == "COMPLETE":
        base.require(terminal["airborne"] == 0 and result["arrivals_after600"] == result["admissions_after600"] == 0,
                     "complete drain violates input/airborne invariants")
    record = world.to_dict()
    record.update(decisions=callbacks, stage3_trace=[dict(record_type="NormalTask", **row) for row in tasks],
        stage2_provenance=dict(execution_scope=binding, common_initial=metadata, record_index_mapping=mapping,
            continuation=state, method_computation_scope="METHOD_ONLY_FILES_EXCLUDE_INITIALIZER"),
        tentative_predictions=[dict(aircraft_id=key, as_of=world.time, plans=[p.to_dict() for p in plans])
            for key, plans in callback.provider.previous.items()])
    # Unique segment replay; paused archives and original archives are immutable.
    replay_path = directory / f"replay_segment_{state['segment']:04d}"
    ReplayBundle.write(replay_path, record=record, code_commit=binding["source_commit"],
        config_path=world.config.path, command=str(DRIVER), seed=world.manifest.seed, eager=False)
    result["replay_directory"] = replay_path.name
    write(directory / "result.json", result)
    return result


def run_segment(args, *, d=None, clock=time.perf_counter):
    started = clock()
    smoke = args.stage == "smoke"
    directory = (Path(args.output)/"restore_checks"/args.root/args.method if smoke else
                 comparison.evaluation_directory(args.output, args.root, args.method))
    source = base.load(args.source_freeze)
    algorithm_files = verify_algorithm_source(source)
    original, original_payload, old_binding, metadata = source_inputs(args.source_output, args.root, args.method, source)
    home = directory if smoke else Path(args.output) / "arms" / args.root / args.method
    if args.resume:
        binding = base.load(home / "resolved_run_binding.json")
        base.require(binding["code"] == code_binding() and binding["algorithm_files"] == algorithm_files and
                     binding["source_checkpoint_sha256"] == base.file_hash(original), "resume source binding differs")
        latest = base.load(directory / "latest.json")
        base.require(latest["status"] == "PAUSED_CHECKPOINT", "only a normally paused checkpoint may resume")
        original = directory / latest["path"]
        base.require(original.parent == directory and base.file_hash(original) == latest["sha256"], "resume checkpoint differs")
        original_payload = base.load(original)["payload"]
        state = deepcopy(original_payload["boundary_summary"]["continuation"])
        base.require(state["binding_id"] == binding["binding_id"], "resume binding differs")
        state["segment"] += 1
    else:
        base.require(not home.exists(), "arm already exists; only explicit paused --resume is permitted")
        directory.mkdir(parents=True, exist_ok=False)
        binding = dict(schema="future_commitment_drain_binding_v1", root=args.root, method=args.method,
            cpu=args.cpu, source_commit=args.code_commit, code=code_binding(), algorithm_files=algorithm_files,
            original_source_binding=old_binding, source_checkpoint=str(original),
            source_checkpoint_sha256=base.file_hash(original), original_identity=original_payload["identity"],
            start_time=START, input_cutoff=CUTOFF, no_physical_horizon=True,
            evaluation_profile=campaign.PROFILE, lead_seconds=20., search_policy=dict(common.SWITCHES))
        binding["binding_id"] = base.digest(binding)
        write(home / "resolved_run_binding.json", binding)
        state = dict(segment=1, binding_id=binding["binding_id"], cutoff=None)
    segment_dir = directory / f"segment_{state['segment']:04d}"
    segment_dir.mkdir(exist_ok=False)
    world, callback, wrapper, checks = restore(original, original_payload, args.method,
        segment_dir / "states.jsonl", cutoff=state.get("cutoff"), d=d)
    write(segment_dir / "restore_invariants.json", checks)
    if smoke:
        result=dict(status="RESTORE_ONLY_PASS",time=world.time,checks=checks,
                    source_checkpoint_sha256=binding["source_checkpoint_sha256"],
                    output_directory=str(directory),physical_advance_calls=0,solver_calls=0)
        try:
            write(directory/"receipt.json",result)
            return result
        finally:
            if world.state_sink is not None: world.state_sink.close()
    identity = dict(original_payload["identity"])
    state.setdefault("initial", statistics(world))
    state.setdefault("initial_finals", initial_finals(world))
    error = None
    status = "PAUSED_CHECKPOINT"
    saved_time = world.time
    try:
        checkpoint(directory, world, wrapper, identity, state, "RESTORED")
        if not smoke:
            while True:
                phase = completion(world, state.get("cutoff"))
                if phase in ("INVALID", "UNKNOWN_TERMINAL"):
                    status = "ERROR"; error = dict(type=phase, message="physical invalidity or unknown admitted outcome")
                    break
                if phase == "COMPLETE":
                    status = "COMPLETE"; break
                if clock() - started >= args.segment_wall_s:
                    status = "PAUSED_CHECKPOINT"; break
                target = world.time + STEP
                if state.get("cutoff") is None:
                    target = min(target, CUTOFF)
                world.advance(target, wrapper, settle_only_at_end=True)
                base.require(abs(world.time - target) < 1.e-7 and not world.failed and
                             not getattr(world, "terminated_invalid", False), "advance did not reach valid stable boundary")
                if world.time == CUTOFF and state.get("cutoff") is None:
                    identity, state["cutoff"] = apply_cutoff(world, identity)
                    write(directory / "cutoff_receipt.json", state["cutoff"])
                    checkpoint(directory, world, wrapper, identity, state, "CUTOFF")
                    saved_time = world.time
                elif world.time - saved_time >= CHECKPOINT_INTERVAL:
                    checkpoint(directory, world, wrapper, identity, state, "RUNNING_CHECKPOINT")
                    saved_time = world.time
                write(directory / "progress.json", dict(phase=completion(world, state.get("cutoff")),
                    time=world.time, pid=os.getpid(), segment=state["segment"], statistics=statistics(world),
                    elapsed_wall_s=clock()-started))
    except Exception as exc:
        status = "ERROR"
        error = dict(type=type(exc).__name__, message=str(exc))
    try:
        checkpoint(directory, world, wrapper, identity, state, status)
        try:
            result = save_final(directory, world, callback, wrapper, metadata, binding, state, status, error)
        except Exception as exc:
            status = "ERROR"
            error = dict(type=type(exc).__name__, message=str(exc), phase="EXPORT")
            result = dict(status=status,time=world.time,error=error,source_binding=binding["binding_id"])
            write(directory / "result.json",result)
            latest = base.load(directory / "latest.json")
            latest["status"] = status
            write(directory / "latest.json",latest)
        write(segment_dir / "receipt.json", dict(status=status, time=world.time, pid=os.getpid(),
            elapsed_wall_s=clock()-started, error=error))
        return result
    finally:
        if world.state_sink is not None:
            world.state_sink.close()


def audit(output, root, method):
    from scripts.stage3.report_selective_reoptimization_a import audit_job
    directory = comparison.evaluation_directory(output, root, method)
    result = base.load(directory / "result.json")
    base.require(result["status"] == "COMPLETE", "only completed drain replay may be audited")
    replay = directory / result["replay_directory"]
    # audit_job has a fixed sibling name. A directory symlink avoids copying history.
    target = directory / "replay"
    base.require(not target.exists(), "audit replay alias already exists")
    target.symlink_to(replay.name, target_is_directory=True)
    return audit_job(directory, PROJECT)


def report(output):
    rows = []
    for root in base.ROOTS:
        for method in base.METHODS:
            directory = comparison.evaluation_directory(output, root, method)
            path = directory / "result.json"
            rows.append(dict(root=root, method=method, result=base.load(path) if path.is_file() else {"status":"NOT_RUN"},
                audit=base.load(directory / "audit_result.json") if (directory / "audit_result.json").is_file() else None))
    result = dict(schema="future_commitment_drain_report_v1", arms=rows,
                  all_complete=all(row["result"]["status"] == "COMPLETE" for row in rows), human_review="HUMAN_PENDING")
    write(Path(output) / "analysis_summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "smoke", "audit", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-output", type=Path)
    parser.add_argument("--source-freeze", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--root", choices=tuple(base.ROOTS))
    parser.add_argument("--method", choices=base.METHODS)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--segment-wall-s", type=float, default=1800.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    base.require(args.output.name == OUTPUT_NAME and args.output.parent == (Path.home()/"training_outputs/terminal").resolve(),
                 "output must be the dedicated server training_outputs drain directory")
    if args.stage == "report":
        value = report(args.output)
    else:
        base.require(args.root is not None and args.method is not None, "explicit root and method required")
        if args.stage == "audit":
            value = audit(args.output, args.root, args.method)
        else:
            base.require(args.source_output and args.source_freeze and args.code_commit, "source output/freeze and code commit required")
            args.source_output = args.source_output.resolve()
            base.require(args.source_output.name == campaign.FUTURE_COMMITMENT_OUTPUT_NAME and
                         args.source_output.parent == args.output.parent, "only original sibling R4 source accepted")
            base.require(args.cpu == campaign.CPUS[(args.root,args.method)], "arm CPU differs")
            base.require(math.isfinite(args.segment_wall_s) and args.segment_wall_s > 0, "positive segment wall cap required")
            base.require(hasattr(os,"sched_setaffinity"), "server CPU affinity API required")
            os.sched_setaffinity(0,{args.cpu})
            for key in base.THREADS: os.environ[key] = "1"
            if args.stage == "smoke":
                base.require(not args.resume,"restore-only smoke cannot resume")
            value = run_segment(args)
    base.print_result(value)
    return value


if __name__ == "__main__":
    result = main()
    if result.get("status") == "ERROR": sys.exit(1)
