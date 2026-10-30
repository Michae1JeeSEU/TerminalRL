"""Export an already completed checkpoint; never advance or resume a simulation.

The original failed watchdog receipt and original arms remain authoritative and
unchanged. Export/audit children are charged to that same campaign ledger.
Run exports serially within one campaign: shared common copies and large
checkpoint/stream reads deliberately have no concurrent-export coordination.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.stage3 import run_service_opportunity_comparison as runner
from scripts.stage3 import run_selective_common_comparison_a4 as comparison
from scripts.stage3 import run_selective_common_initial_a4 as common
from scripts.stage3 import report_service_opportunity_comparison as report
from terminal_marl.core.rotating_slots.replay import _plain
from terminal_marl.core.rotating_slots.stage2_campaign import digest

base=runner.base
old=runner.old
DRIVER=Path(__file__).resolve()
CAPS={"export":180.,"audit":300.}


def exclusive_copy(source,target,*,shared=False):
    source,target=Path(source),Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if shared and target.exists():
        base.require(base.file_hash(source)==base.file_hash(target),"existing common export copy differs")
        return
    with source.open("rb") as src,target.open("xb") as dst:
        shutil.copyfileobj(src,dst)
    base.require(base.file_hash(source)==base.file_hash(target),"export copy bytes differ")


def original_inputs(output,root,method,source,ledger):
    """Validate terminal state and its original immutable execution bindings."""
    base.require(root=="s3r24" and method in runner.base.METHODS,"only the two authorized high-load arms may be exported")
    home=Path(output)/"arms"/root/method
    checkpoint=comparison.evaluation_directory(output,root,method)/"checkpoint_0300.json"
    binding_path=home/"resolved_run_binding.json"
    binding=base.load(binding_path); envelope=base.load(checkpoint); payload=envelope["payload"]
    base.require(envelope["sha256"]==digest(payload),"terminal checkpoint envelope hash mismatch")
    base.require(binding.get("binding_id")==base.digest({k:v for k,v in binding.items() if k!="binding_id"}),"original arm binding hash mismatch")
    base.require(binding.get("source_hash")==source["source_hash"] and binding.get("source_commit")==source["commit"] and
        payload["identity"].get("source_hash")==source["source_hash"],"original source freeze binding mismatch")
    base.require(ledger.get("service_source_hash")==source["source_hash"] and
        ledger.get("service_evaluation_profile")==runner.PROFILE,"campaign source/profile binding mismatch")
    row=payload.get("boundary_summary") or {}
    base.require(payload.get("time")==320. and row.get("time")==320. and row.get("t0")==20. and
        row.get("relative_time_s")==300. and row.get("root_id")==root and row.get("method")==method,
        "only an exact completed t320/relative300 checkpoint is accepted")
    base.require(binding.get("root")==root and binding.get("method")==method and binding.get("horizon")==[20.,320.] and
        binding.get("evaluation_profile")==runner.PROFILE and row.get("evaluation_profile")==runner.PROFILE and
        row.get("source_binding")==binding["binding_id"],"terminal arm/time/profile binding mismatch")
    base.require(all(k in row for k in ("costs","holding_energy_by_ring","holding_time_by_ring",
        "total_executed_energy_by_source_ring","increment","normal_lifecycle")),"complete boundary accounting required")
    job_id="service_arm_"+root+"_"+method
    job=ledger.get("jobs",{}).get(job_id,{})
    base.require(job.get("status")=="BUDGET_LIMITED","original arm must retain its BUDGET_LIMITED receipt")
    common_checkpoint=Path(output)/"common_initial"/root/"common.checkpoint.json"
    base.require(base.file_hash(common_checkpoint)==runner.COMMON_HASHES[root]==binding.get("common_checkpoint_sha256"),
        "original common checkpoint binding mismatch")
    return checkpoint,binding_path,binding,payload,row,job_id,deepcopy(job),common_checkpoint


def export_terminal(output,root,method,source,ledger,*,exporter_commit,d=None):
    started=time.perf_counter()
    checkpoint,binding_path,binding,payload,boundary,job_id,job,common_checkpoint=original_inputs(output,root,method,source,ledger)
    d=base.dependencies(PROJECT) if d is None else d
    export_root=Path(output)/"terminal_export"
    home=export_root/"arms"/root/method
    directory=comparison.evaluation_directory(export_root,root,method)
    base.require(not home.exists(),"terminal export already exists; no overwrite/retry")
    # Read all algorithm state through its unchanged checkpoint binding validator.
    callback=common.common_callback(d,method,evaluation_profile=runner.PROFILE)
    wrapper=d.LatencyAwareScheduler(callback,lead_seconds=20.,preparation_margin=1.,
        search_handoff_reserve_seconds=4.,restart_policy=common.RESTART)
    directory.mkdir(parents=True,exist_ok=False)
    world=d.restore_checkpoint(checkpoint,wrapper,payload["identity"],stream_copy=directory/"restored_states.jsonl")
    try:
        base.require(world.time==320. and not world.failed and not getattr(world,"terminated_invalid",False),
                     "failed/invalid or nonterminal checkpoint cannot be exported")
        # Common restoration supplies provenance only. Its cloned world is
        # discarded and never attached to the actual terminal scheduler.
        initial,_,_,metadata=common.restore_common(common_checkpoint,method,runner.PROFILE,d=d,directory=directory/"common_clone")
        if initial.state_sink is not None: initial.state_sink.close()
        base.require(metadata["t0"]==20. and metadata["checkpoint_sha256"]==binding["common_checkpoint_sha256"],"common metadata differs")
        initial_dispatch=metadata["common_summary"]["baseline"]["dispatch_count"]
        callbacks,tasks,services,mapping=comparison.merged_records(metadata,callback,wrapper)
        mapping["calculation_event_rows"]=comparison.calculation_event_mapping(world,metadata,mapping["offsets"]["tasks"])
        exclusive_copy(binding_path,home/"resolved_run_binding.json")
        exclusive_copy(common_checkpoint,export_root/"common_initial"/root/"common.checkpoint.json",shared=True)
        exclusive_copy(checkpoint,directory/"checkpoint_terminal.json")
        exclusive_copy(checkpoint,directory/"checkpoint_0300.json")
        comparison.write_new(directory/"boundary_0300.json",boundary)
        for name,value in (("scheduler_records.json",callbacks),("normal_tasks.json",tasks),
            ("online_services.json",services),("method_scheduler_records.json",callback.records),
            ("method_normal_tasks.json",wrapper.records),("method_online_services.json",wrapper.online_services),
            ("record_index_mapping.json",mapping),("dispatch_timings.json",world.dispatch_timings),
            ("method_dispatch_timings.json",world.dispatch_timings[initial_dispatch:]),
            ("selection_records.json",base.selection_envelopes(callbacks,directory.name,binding["binding_id"],root)),
            ("method_selection_records.json",base.selection_envelopes(callback.records,directory.name,binding["binding_id"],root))):
            comparison.write_new(directory/name,value)
        terminal=d.evaluation.terminal_observation(world)
        comparison.write_new(directory/"terminal_observation.json",terminal)
        record=world.to_dict()
        record.update(decisions=callbacks,stage3_trace=[dict(record_type="NormalTask",**row) for row in tasks],
            stage2_provenance=dict(execution_scope=binding,common_initial=metadata,record_index_mapping=mapping,
                method_computation_scope="METHOD_ONLY_FILES_EXCLUDE_INITIALIZER"),
            tentative_predictions=[dict(aircraft_id=aid,as_of=world.time,plans=[p.to_dict() for p in plans])
                for aid,plans in callback.provider.previous.items()])
        d.ReplayBundle.write(directory/"replay",record=record,code_commit=binding["source_commit"],
            config_path=world.config.path,command=str(DRIVER),seed=base.ROOTS[root][0],eager=False)
        receipt=dict(schema="local_recovery_terminal_export_v1",completed_state_export_only=True,
            source_checkpoint=str(checkpoint),source_checkpoint_sha256=base.file_hash(checkpoint),
            source_checkpoint_envelope_sha256=base.load(checkpoint)["sha256"],
            source_original_job_id=job_id,source_original_job_status=job["status"],source_original_job_wall_s=job.get("child_wall_s"),
            source_original_job_exit_code=job.get("exit_code"),source_original_job_stop_reason=job.get("stop_reason"),
            original_source_commit=source["commit"],original_source_hash=source["source_hash"],
            exporter_commit=exporter_commit,exporter_script_sha256=base.file_hash(DRIVER),export_wall_s=time.perf_counter()-started,
            simulation_advance_calls=0,callback_calls=0,algorithm_resume_calls=0,
            source_binding=binding["binding_id"],pending_task_preserved=wrapper.pending is not None,
            accounting="original boundary costs unchanged; export wall separately charged to shared ledger")
        comparison.write_new(home/"export_receipt.json",receipt)
        result=dict(status="COMPLETE",root_id=root,method=method,t0=20.,time=320.,completed_relative_s=300,
            target_relative_s=300,evaluation_profile=runner.PROFILE,error=None,terminal_observation=terminal,
            increment=boundary["increment"],costs=boundary["costs"],normal_lifecycle=boundary["normal_lifecycle"],
            replay_status="WRITTEN_HUMAN_PENDING",**receipt)
        comparison.write_new(directory/"result.json",result)
        comparison.write_new(directory/"progress.json",dict(result,phase="TERMINAL_STATE_EXPORT_ONLY"))
        comparison.write_new(home/"arm.finished.json",result)
        return result
    finally:
        if world.state_sink is not None: world.state_sink.close()


def audit_export(output,root,method,source,ledger):
    checkpoint,binding_path,binding,_,_,_,_,_=original_inputs(output,root,method,source,ledger)
    export_root=Path(output)/"terminal_export"; home=export_root/"arms"/root/method
    directory=comparison.evaluation_directory(export_root,root,method)
    receipt=base.load(home/"export_receipt.json")
    base.require(receipt.get("completed_state_export_only") is True and
        receipt.get("source_checkpoint_sha256")==base.file_hash(checkpoint) and
        receipt.get("source_binding")==binding["binding_id"] and
        receipt.get("original_source_hash")==source["source_hash"] and
        receipt.get("original_source_commit")==source["commit"],"export receipt lineage differs")
    base.require(base.file_hash(home/"resolved_run_binding.json")==base.file_hash(binding_path) and
        base.file_hash(directory/"checkpoint_terminal.json")==base.file_hash(checkpoint),"export binding/checkpoint copy differs")
    return comparison.audit_arm(export_root,root,method,checkout=PROJECT)


def combined_report(old_root,output):
    original=report.collect(old_root,output); exported=report.collect(old_root,Path(output)/"terminal_export")
    combined=deepcopy(original)
    for key in ("metrics",):
        combined["campaigns"]["new"][key]=[deepcopy(a) for source in (original,exported)
            for a in source["campaigns"]["new"][key] if a["root"]==("s3r23" if source is original else "s3r24")]
    ordinary=combined["campaigns"]["new"]["ordinary"]
    for key in ("arms","task_timeline","callback_timings"):
        ordinary[key]=[deepcopy(a) for source in (original,exported)
            for a in source["campaigns"]["new"]["ordinary"][key] if a["root"]==("s3r23" if source is original else "s3r24")]
    expected={(r,m) for r in report.ROOTS for m in report.METHODS}
    for rows in (ordinary["arms"],combined["campaigns"]["new"]["metrics"],
                 combined["campaigns"]["old"]["ordinary"]["arms"],combined["campaigns"]["old"]["metrics"]):
        base.require(len(rows)==4 and {(a["root"],a["method"]) for a in rows}==expected,"combined report requires four unique authorized arms")
    for metric in combined["campaigns"]["new"]["metrics"]:
        root=Path(output)/"terminal_export" if metric["root"]=="s3r24" else Path(output)
        receipt=root/"arms"/metric["root"]/metric["method"]/"export_receipt.json"
        lineage=base.load(receipt) if receipt.is_file() else None
        if metric["root"]=="s3r24":
            base.require(lineage is not None and lineage.get("completed_state_export_only") is True,
                         "high-load completed-state export lineage required")
        metric.update(arm_input_root=str(root.resolve()),export_lineage=lineage)
        arm=next(a for a in ordinary["arms"] if (a["root"],a["method"])==(metric["root"],metric["method"]))
        arm.update(arm_input_root=str(root.resolve()),export_lineage=lineage)
        audit_path=comparison.evaluation_directory(root,metric["root"],metric["method"])/"audit_result.json"
        audit=base.load(audit_path) if audit_path.is_file() else None
        arm["saved_audit_result"]=audit; metric["saved_audit_result"]=audit
    for arm in combined["campaigns"]["old"]["ordinary"]["arms"]:
        arm.update(arm_input_root=str(Path(old_root).resolve()),export_lineage=None)
    for metric in combined["campaigns"]["old"]["metrics"]:
        metric.update(arm_input_root=str(Path(old_root).resolve()),export_lineage=None)
    checks={root:len({a.get("common_checkpoint_sha256") for a in ordinary["arms"] if a["root"]==root})==1 and
        all(a.get("common_checkpoint_sha256") is not None for a in ordinary["arms"] if a["root"]==root) for root in report.ROOTS}
    ordinary["same_common_sha_by_root"]=checks
    profiles={a.get("evaluation_profile") for a in ordinary["arms"]}
    ordinary["same_evaluation_profile"]=len(profiles)==1 and None not in profiles
    ordinary["status"]="COMPLETE" if all(a["status"]=="COMPLETE" for a in ordinary["arms"]) and all(checks.values()) and ordinary["same_evaluation_profile"] else "INCOMPLETE"
    combined["same_common_checkpoint_across_campaigns"]={root:len({a.get("common_checkpoint_sha256")
        for c in combined["campaigns"].values() for a in c["ordinary"]["arms"] if a["root"]==root})==1 and
        all(a.get("common_checkpoint_sha256") is not None for c in combined["campaigns"].values()
            for a in c["ordinary"]["arms"] if a["root"]==root) for root in report.ROOTS}
    combined["terminal_export_semantics"]="medium arms original; high arms completed-state exports; original BUDGET_LIMITED jobs unchanged"
    return combined


def write_report(output_dir,value,campaign_output):
    target=Path(output_dir).resolve(); campaign_output=Path(campaign_output).resolve()
    local=(PROJECT/"artifacts").resolve(); server=(Path.home()/"training_outputs/terminal"/runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME).resolve()
    base.require(target.is_relative_to(local) or (campaign_output==server and target.is_relative_to(server)),"report output requires local artifacts or this server campaign directory")
    base.require(target!=campaign_output and not target.is_relative_to(campaign_output/"arms") and
        not target.is_relative_to(campaign_output/"terminal_export") and not target.is_relative_to(campaign_output/"common_initial"),"report must be separate from arm/common inputs")
    for label,campaign in value["campaigns"].items():
        source=Path(campaign["input_root"]).resolve()
        base.require(not source.is_relative_to(target),"report cannot contain an input campaign")
        if label=="old": base.require(not target.is_relative_to(source),"old report inputs are read-only")
    target.mkdir(parents=True,exist_ok=False)
    comparison.write_new(target/"summary.json",value)
    rows=[dict(campaign=label,**m) for label,c in value["campaigns"].items() for m in c["metrics"]]
    fields=sorted({k for row in rows for k in row})
    with (target/"comparison.csv").open("x",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(row.get(k),sort_keys=True) if isinstance(row.get(k),(dict,list)) or row.get(k) is None else row[k] for k in fields})
    return target


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=("export","audit","report"))
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--root",choices=("s3r24",))
    parser.add_argument("--method",choices=base.METHODS); parser.add_argument("--cpu",type=int)
    parser.add_argument("--source-freeze",type=Path); parser.add_argument("--exporter-commit")
    parser.add_argument("--old-root",type=Path); parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--_child-job-id",help=argparse.SUPPRESS)
    values=sys.argv[1:] if argv is None else argv; args=parser.parse_args(values)
    base.require(args.output.name==runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME,"fixed serialization-fix campaign required")
    if args.stage=="report":
        base.require(args.old_root is not None and args.output_dir is not None,"old-root and output-dir required")
        base.require(args.old_root.name==runner.SUPPLY_R2_OUTPUT_NAME,"fixed read-only R2 comparison required")
        value=combined_report(args.old_root,args.output); write_report(args.output_dir,value,args.output); return value
    args.output=runner.validated_output(args.output)
    base.require(args.root is not None and args.method is not None and args.source_freeze is not None and
        args.cpu==runner.CPUS[args.root,args.method],"fixed high-load arm, CPU and original source freeze required")
    base.require(args.stage!="export" or bool(args.exporter_commit),"exporter-commit required for export provenance")
    threads=runner.launch_threads(); policy=runner.policy_for_output(args.output)
    job_id="terminal_"+args.stage+"_"+args.root+"_"+args.method
    if not args._child_job_id:
        result=old.run_bounded_job(args.output,job_id,[sys.executable,"-B",str(DRIVER),*values,"--_child-job-id",job_id],cap_s=CAPS[args.stage],policy=policy)
        base.print_result(dict(job_id=job_id,**result)); return result
    base.require(args._child_job_id==job_id,"export/audit child identity differs")
    source=base.load(args.source_freeze); base.verify_sources(source)
    with old.ledger_transaction(args.output,write=False,policy=policy) as ledger:
        active=ledger["jobs"].get(job_id,{})
        base.require(active.get("status")=="running" and active.get("parent_pid")==os.getppid(),"active parent watchdog required")
        saved=deepcopy(ledger)
    d=base.dependencies(PROJECT); d.launch_threads=threads
    old.set_cpu(argparse.Namespace(stage="comparison",cpu=args.cpu,root=args.root),d)
    if args.stage=="export": result=export_terminal(args.output,args.root,args.method,source,saved,exporter_commit=args.exporter_commit,d=d)
    else:
        result=audit_export(args.output,args.root,args.method,source,saved)
    base.print_result(result); return result


if __name__=="__main__": main()
