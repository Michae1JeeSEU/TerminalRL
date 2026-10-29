"""One explicitly requested A4 method arm or its separately charged audit.

The common checkpoint owns absolute physical time.  This runner adds a relative
300-second method window, exports initializer provenance for existing audits,
and keeps method-only computation records separate for incremental accounting.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import os
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_common_initial_a4 as common

DRIVER=Path(__file__).resolve()
ARM_CAP_S=1200.
AUDIT_CAP_S=300.
EPISODE_SECONDS=300
BOUNDARY_SECONDS=5


def require_profile(value):
    base.require(isinstance(value,str) and bool(value.strip()), "nonempty explicit evaluation_profile required")
    return value


def evaluation_directory(output,root,method):
    base.require(root in base.ROOTS and method in base.METHODS,"fixed root and method required")
    return Path(output)/"arms"/root/method/"evaluation"/root/method


def write_new(path,value):
    from terminal_marl.core.rotating_slots.stage2_campaign import canonical
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream: stream.write(canonical(value))


def merged_records(metadata,callback,wrapper):
    """Offset copied task references; neither the live scheduler nor physics change."""
    callbacks=deepcopy(metadata["initializer_callback_records"])
    tasks=deepcopy(metadata["initializer_tasks"])
    services=deepcopy(metadata["initializer_services"])
    offsets=dict(callbacks=len(callbacks),tasks=len(tasks),services=len(services))
    mapping=dict(offsets=offsets,task_rows=[],callback_rows=[],service_rows=[],
        physical_events="UNCHANGED",method_records="UNCHANGED",external_predecessors=[])
    for kind,initial,method_rows in (("callback",callbacks,callback.records),("task",tasks,wrapper.records),
                                     ("service",services,wrapper.online_services)):
        offset=len(initial)
        for index,row in enumerate(initial):
            row.update(a4_record_role="INITIALIZER_PROVENANCE",audit_record_index=index)
        for index,original in enumerate(method_rows):
            row=deepcopy(original)
            row.update(a4_record_role="METHOD",audit_record_index=offset+index,method_record_index=index)
            if kind=="task":
                previous=row.get("previous_record_index")
                if type(previous) is int:
                    row["previous_record_index"]=previous+offset
                elif previous is None and row.get("external_predecessor"):
                    predecessor=row["external_predecessor"]["initializer_record_index"]
                    base.require(type(predecessor) is int and 0<=predecessor<offset,
                                 "external predecessor is outside initializer task provenance")
                    row["previous_record_index"]=predecessor
                    mapping["external_predecessors"].append(dict(method_record_index=index,
                        audit_record_index=offset+index,initializer_record_index=predecessor))
            if kind=="service" and "sequence" in row:
                row["sequence"]=offset+index
            initial.append(row)
            mapping[kind+"_rows"].append(dict(method_record_index=index,audit_record_index=offset+index))
    pending=deepcopy(wrapper.pending)
    if pending is not None:
        pending=dict(stage=pending["stage"],due=pending["due"],
            method_record_index=pending["record_index"],audit_record_index=offsets["tasks"]+pending["record_index"])
    mapping["pending_task"]=pending
    return callbacks,tasks,services,mapping


def calculation_event_mapping(world,metadata,task_offset):
    """Describe computation-event indices without rewriting authoritative events."""
    rows=[]
    start=metadata["common_summary"]["baseline"]["event_count"]
    for index,event in enumerate(world.events[start:],start):
        details=event.get("details",{})
        if event.get("type")=="normal_coalesced_restart" and "record_index" in details:
            original=details["record_index"]
            external=details.get("external_predecessor")
            audit_index=(original+task_offset if type(original) is int else
                         external["initializer_record_index"] if external else None)
            rows.append(dict(event_index=index,event_type=event["type"],time=event["time"],
                method_record_index=original,audit_record_index=audit_index,
                external_predecessor=deepcopy(external)))
    return rows


def observe_method_rings(world,d):
    """Use the existing holding-energy observer; separately measure time/all energy."""
    holding_energy,restore_holding=d.evaluation.observe_executed_holding(world)
    cls=type(world); inner=cls._advance_energy
    holding_time={ring:0. for ring in ("outer","middle","inner")}
    total_energy=dict(holding_time)
    def observed(instance,end):
        before={aid:(a.ring,a.holding_time,a.energy_j) for aid,a in instance.aircraft.items()}
        result=inner(instance,end)
        for aid,(ring,held,energy) in before.items():
            after=instance.aircraft.get(aid)
            if after is not None and ring in total_energy:
                total_energy[ring]+=after.energy_j-energy
                holding_time[ring]+=after.holding_time-held
        return result
    cls._advance_energy=observed
    def restore():
        cls._advance_energy=inner
        restore_holding()
    return holding_energy,holding_time,total_energy,restore


def check_treatment(world,callback,wrapper,method):
    base.require(world.scheduler_period==5. and wrapper.lead_seconds==20. and
        wrapper.preparation_margin==1. and wrapper.search_handoff_reserve_seconds==4. and
        wrapper.restart_policy==common.RESTART,"common method timing differs")
    base.require(callback.selective_method==method and callback.selective_rule_variant==common.RULE and
        callback.search_policy.to_dict()==common.SWITCHES,"common method search policy differs")
    base.require(not callback.records and not wrapper.records and wrapper.pending is None,
                 "method restore must be cold with no pending computation")


def run_arm(output,root,method,checkpoint,binding,*,evaluation_profile,d=None,episode_seconds=EPISODE_SECONDS):
    """Internal single-arm implementation. Formal CLI fixes the horizon to 300s."""
    require_profile(evaluation_profile)
    base.require(root in base.ROOTS and method in base.METHODS,"fixed root and method required")
    base.require(type(episode_seconds) is int and 0<episode_seconds<=EPISODE_SECONDS and
        episode_seconds%BOUNDARY_SECONDS==0,"positive bounded five-second method window required")
    d=base.dependencies(PROJECT) if d is None else d
    directory=evaluation_directory(output,root,method)
    arm_home=Path(output)/"arms"/root/method
    started,cpu_started=time.perf_counter(),time.process_time()
    arm_home.mkdir(parents=True,exist_ok=False)
    io_wall=io_cpu=online_wall=online_cpu=resume_wall=0.
    def io(function,*args,**kwargs):
        nonlocal io_wall,io_cpu
        before,before_cpu=time.perf_counter(),time.process_time()
        try: return function(*args,**kwargs)
        finally:
            io_wall+=time.perf_counter()-before; io_cpu+=time.process_time()-before_cpu
    io(write_new,arm_home/"arm.started.json",dict(root=root,method=method,pid=os.getpid(),evaluation_profile=evaluation_profile))
    world,callback,wrapper,metadata=common.restore_common(checkpoint,method,evaluation_profile,d=d,directory=directory)
    check_treatment(world,callback,wrapper,method)
    base.require(world.manifest.seed==base.ROOTS[root][0],"restored common manifest seed differs")
    base.require(world.time==metadata["t0"] and metadata["common_summary"]["t0"]==world.time,
                 "absolute common t0 differs")
    base.require(metadata["evaluation_profile"]==evaluation_profile,"restored evaluation profile differs")
    t0=world.time
    resolved=dict(binding,root=root,method=method,seed=base.ROOTS[root][0],scenario_profile=base.ROOTS[root][1],
        common_checkpoint=str(Path(checkpoint).resolve()),common_checkpoint_sha256=base.file_hash(checkpoint),
        evaluation_profile=evaluation_profile,t0=t0,horizon=[t0,t0+episode_seconds],
        episode_seconds=episode_seconds,lead_seconds=20.,heartbeat_seconds=5.,max_k=12,
        nominal_k=8,search_policy=dict(common.SWITCHES),selective_rule_variant=common.RULE,
        run_role="FORMAL_SINGLE_ARM" if episode_seconds==EPISODE_SECONDS else "TINY_TEST_ONLY",
        no_automatic_retry=True,human_replay_status="HUMAN_PENDING")
    resolved["binding_id"]=base.digest(resolved)
    io(write_new,arm_home/"resolved_run_binding.json",resolved)
    identity=dict(base.load(checkpoint)["payload"]["identity"],source_hash=binding["source_hash"])
    initial_dispatch=len(world.dispatch_timings)
    holding_energy,holding_time,ring_energy,restore_observers=observe_method_rings(world,d)
    setup_wall=time.perf_counter()-started-io_wall
    setup_cpu=time.process_time()-cpu_started-io_cpu
    completed=0; error=None
    def costs():
        elapsed=time.perf_counter()-started; used_cpu=time.process_time()-cpu_started
        return dict(full_job_wall_s=elapsed,process_cpu_s=used_cpu,setup_wall_s=setup_wall,setup_cpu_s=setup_cpu,
            online_advance_wall_s=online_wall,online_advance_cpu_s=online_cpu,initial_resume_wall_s=resume_wall,
            output_io_wall_s=io_wall,output_io_cpu_s=io_cpu,
            other_bookkeeping_wall_s=max(0.,elapsed-setup_wall-online_wall-io_wall),
            dispatch_wall_s=sum(row.get("dispatch_wall_s",0.) for row in world.dispatch_timings[initial_dispatch:]),
            timing_scope="RESTORE_ENTRY_THROUGH_CURRENT_OBSERVATION_INITIALIZER_COMPUTE_EXCLUDED",
            restore_io_in_setup=True,online_includes_t0_resume=True,
            ledger_receipt_includes_final_result_write=True)
    def summaries():
        quality=d.episode_metrics(world)
        from terminal_marl.core.rotating_slots.world import AIRBORNE
        quality.update(unresolved_low_soc_count=sum(a.status in AIRBORNE and a.soc<.25 for a in world.aircraft.values()),
                       unresolved_low_soc_threshold=.25)
        return dict(root_id=root,method=method,t0=t0,time=world.time,relative_time_s=world.time-t0,
            quality=quality,increment=common.increment_observation(world,metadata["common_summary"]),
            inherited_finals=deepcopy(metadata["common_summary"]["inherited_finals"]),
            normal_lifecycle=base.lifecycle_summary(wrapper.records,callback.records,wrapper.online_services,d.latency),
            selection_summary=base.module(PROJECT/"scripts/stage3/report_selective_reoptimization_a.py",
                                          "_common_comparison_selection").selection_summary(callback.records),
            holding_energy_by_ring=dict(holding_energy),holding_time_by_ring=dict(holding_time),
            total_executed_energy_by_source_ring=dict(ring_energy),costs=costs(),pending=wrapper.pending is not None,
            source_binding=resolved["binding_id"],evaluation_profile=evaluation_profile)
    def save_logs():
        callbacks,tasks,services,mapping=merged_records(metadata,callback,wrapper)
        mapping["calculation_event_rows"]=calculation_event_mapping(world,metadata,mapping["offsets"]["tasks"])
        for name,value in (("scheduler_records.json",callbacks),("normal_tasks.json",tasks),
            ("online_services.json",services),("method_scheduler_records.json",callback.records),
            ("method_normal_tasks.json",wrapper.records),("method_online_services.json",wrapper.online_services),
            ("record_index_mapping.json",mapping),("dispatch_timings.json",world.dispatch_timings),
            ("method_dispatch_timings.json",world.dispatch_timings[initial_dispatch:]),
            ("selection_records.json",base.selection_envelopes(callbacks,directory.name,resolved["binding_id"],root)),
            ("method_selection_records.json",base.selection_envelopes(callback.records,directory.name,resolved["binding_id"],root))):
            io(d.atomic_json,directory/name,value)
        return callbacks,tasks,mapping
    try:
        # Baseline is serialized before the first method computation. Both arms
        # then resume at this exact absolute t0, before their first 5s advance.
        row=summaries()
        io(d.save_checkpoint,directory/"checkpoint_0000.json",world,wrapper,identity,boundary_summary=row)
        io(d.atomic_json,directory/"boundary_0000.json",row)
        before,before_cpu=time.perf_counter(),time.process_time()
        try: common.resume_common(world,wrapper)
        finally:
            resume_wall=time.perf_counter()-before; online_wall+=resume_wall; online_cpu+=time.process_time()-before_cpu
        for relative in range(BOUNDARY_SECONDS,episode_seconds+1,BOUNDARY_SECONDS):
            target=t0+relative
            io(d.atomic_json,directory/"progress.json",dict(root=root,method=method,pid=os.getpid(),
                phase="ADVANCING",t0=t0,completed_relative_s=completed,target_relative_s=relative,target_time=target,
                elapsed_wall_s=time.perf_counter()-started))
            before,before_cpu=time.perf_counter(),time.process_time()
            try: world.advance(target,wrapper,settle_only_at_end=True)
            finally:
                online_wall+=time.perf_counter()-before; online_cpu+=time.process_time()-before_cpu
            if abs(world.time-target)>1.e-7 or world.failed or getattr(world,"terminated_invalid",False): break
            completed=relative
            for record in callback.records:
                scope=record.get("selective_replanning",{}).get("scope") or {}
                base.require(not scope or scope["max_k"]==12,"method scope max_k differs")
            row=summaries()
            io(d.save_checkpoint,directory/f"checkpoint_{relative:04d}.json",world,wrapper,identity,boundary_summary=row)
            io(d.atomic_json,directory/f"boundary_{relative:04d}.json",row)
            save_logs()
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        restore_observers()
        callbacks,tasks,mapping=save_logs()
        terminal=d.evaluation.terminal_observation(world)
        io(d.atomic_json,directory/"terminal_observation.json",terminal)
        terminal_summary=summaries()
        io(d.save_checkpoint,directory/"checkpoint_terminal.json",world,wrapper,identity,boundary_summary=terminal_summary)
        before,before_cpu=time.perf_counter(),time.process_time()
        record=world.to_dict()
        record.update(decisions=callbacks,stage3_trace=[dict(record_type="NormalTask",**row) for row in tasks],
            stage2_provenance=dict(execution_scope=resolved,common_initial=metadata,
                record_index_mapping=mapping,method_computation_scope="METHOD_ONLY_FILES_EXCLUDE_INITIALIZER"),
            tentative_predictions=[dict(aircraft_id=aid,as_of=world.time,plans=[p.to_dict() for p in plans])
                for aid,plans in callback.provider.previous.items()])
        io_wall+=time.perf_counter()-before; io_cpu+=time.process_time()-before_cpu
        io(d.ReplayBundle.write,directory/"replay",record=record,code_commit=binding["source_commit"],
            config_path=world.config.path,command=str(DRIVER),seed=base.ROOTS[root][0],eager=False)
        result=dict(status="COMPLETE" if completed==episode_seconds and error is None else "EVALUATION_PARTIAL",
            root_id=root,method=method,t0=t0,time=world.time,completed_relative_s=completed,
            target_relative_s=episode_seconds,evaluation_profile=evaluation_profile,error=error,
            terminal_observation=terminal,increment=common.increment_observation(world,metadata["common_summary"]),
            costs=costs(),normal_lifecycle=terminal_summary["normal_lifecycle"],
            replay_status="WRITTEN_HUMAN_PENDING",source_binding=resolved["binding_id"])
        io(d.atomic_json,directory/"progress.json",dict(result,phase="TERMINAL"))
        io(d.atomic_json,directory/"result.json",result)
        io(write_new,arm_home/"arm.finished.json",result)
        if world.state_sink is not None: world.state_sink.close()
    if error: raise RuntimeError(str(error))
    return result


def audit_arm(output,root,method,*,checkout=PROJECT):
    from scripts.stage3 import report_selective_reoptimization_a as report
    directory=evaluation_directory(output,root,method)
    base.require((directory/"result.json").is_file(),"completed arm artifacts required for audit")
    write_new(directory/"comparison_audit.started.json",dict(pid=os.getpid(),root=root,method=method))
    return report.audit_job(directory,Path(checkout))


def main(argv=None):
    from scripts.stage3 import run_selective_quality_gap_a4 as a4
    from scripts.stage3 import run_selective_reoptimization_repair as old
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("arm","audit"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--root",choices=tuple(base.ROOTS),required=True)
    parser.add_argument("--method",choices=base.METHODS,required=True)
    parser.add_argument("--evaluation-profile",required=True)
    parser.add_argument("--source-freeze",type=Path,required=True)
    parser.add_argument("--cpu",type=int,required=True)
    parser.add_argument("--_child-job-id",help=argparse.SUPPRESS)
    values=sys.argv[1:] if argv is None else argv
    args=parser.parse_args(values)
    require_profile(args.evaluation_profile)
    args.output=a4.validated_output(args.output)
    job_id="comparison_"+args.stage+"_"+args.root+"_"+args.method
    if not args._child_job_id:
        command=[sys.executable,str(DRIVER),*values,"--_child-job-id",job_id]
        receipt=a4.run_bounded_job(args.output,job_id,command,
            cap_s=ARM_CAP_S if args.stage=="arm" else AUDIT_CAP_S)
        with a4.ledger_transaction(args.output) as ledger:
            ledger["jobs"][job_id]["run_role"]="FORMAL_SINGLE_ARM" if args.stage=="arm" else "SAVED_REPLAY_AUDIT"
        base.print_result(dict(job_id=job_id,**receipt))
        return receipt
    base.require(args._child_job_id==job_id,"comparison child identity differs")
    with a4.ledger_transaction(args.output,write=False) as ledger:
        row=ledger["jobs"].get(job_id,{})
        base.require(row.get("status")=="running" and row.get("parent_pid")==os.getppid(),
                     "active comparison parent watchdog required")
    source=base.load(args.source_freeze); base.verify_sources(source)
    required=(DRIVER.relative_to(PROJECT).as_posix(),"tests/test_selective_common_comparison_a4.py")
    base.require(all(name in source["files"] for name in required),"reviewed comparison runner/tests freeze required")
    with a4.ledger_transaction(args.output) as ledger:
        bound=ledger.setdefault("comparison_source_hash",source["source_hash"])
        profile=ledger.setdefault("comparison_evaluation_profile",args.evaluation_profile)
        base.require(bound==source["source_hash"] and profile==args.evaluation_profile,
                     "comparison source/profile differs from first attempted arm or audit")
    launch_threads={key:os.environ.get(key) for key in base.THREADS}
    d=base.dependencies(PROJECT); d.launch_threads=launch_threads
    old.set_cpu(argparse.Namespace(stage="comparison",cpu=args.cpu,root=args.root),d)
    if args.stage=="audit":
        saved=base.load(args.output/"arms"/args.root/args.method/"resolved_run_binding.json")
        base.require(saved["source_hash"]==source["source_hash"] and
            saved["evaluation_profile"]==args.evaluation_profile,"audit binding differs from saved arm")
        result=audit_arm(args.output,args.root,args.method,checkout=PROJECT)
    else:
        checkpoint=args.output/"common_initial"/args.root/"common.checkpoint.json"
        summary=base.load(checkpoint)["payload"]["boundary_summary"]
        base.require(summary["status"]=="COMMON_INITIAL_READY" and
            summary["source_binding"]["root"]==args.root,"common checkpoint root/status differs")
        with a4.ledger_transaction(args.output) as ledger:
            hashes=ledger.setdefault("comparison_common_checkpoints",{})
            expected=hashes.setdefault(args.root,base.file_hash(checkpoint))
            base.require(expected==base.file_hash(checkpoint),"common comparison checkpoint changed")
        result=run_arm(args.output,args.root,args.method,checkpoint,
            dict(source_commit=source["commit"],source_hash=source["source_hash"],cpu=args.cpu),
            evaluation_profile=args.evaluation_profile,d=d)
    base.print_result(result)
    return result


if __name__=="__main__": main()
