"""One explicitly requested exploratory arm; no Stage B, training or campaign launch."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.stage3 import run_selective_reoptimization_a as base

VARIANTS={"baseline20":("baseline",20.),"outer20":("outer_priority_v1",20.),
          "baseline15":("baseline",15.),"outer15":("outer_priority_v1",15.)}
ROOT_CPUS={"s3r23":12,"s3r24":14}


def treatment(variant,short_lead):
    base.require(variant in VARIANTS,"unknown follow-up treatment")
    base.require(short_lead==15.,"authorized short lead is exactly 15 seconds")
    rule,lead=VARIANTS[variant]
    return dict(treatment_id=variant,selective_method="RULE_LOCAL_K8",selector_variant=rule,
        lead_seconds=lead,short_lead_factor_seconds=short_lead,heartbeat_seconds=5.,preparation_margin_seconds=1.,
        search_handoff_reserve_seconds=4.,nominal_k=8,max_k=12,
        ready_budget_seconds=lead-1.,search_budget_seconds=lead-1.-4.,
        learning=False,training=False,stage_B=False,stage_C=False,
        factor_design="EXPLICITLY_AUTHORIZED_2_BY_2_RULE_AND_LEAD")


def build_binding(origin,source,args,environment):
    """Only original input lineage is imported; no old treatment results/model."""
    item=deepcopy(origin["inputs"][args.root])
    return dict(schema="rs_selective_followup_binding_v1",experiment_family="EXPLORATORY_CAUSAL_FOLLOWUP",
        source=source,source_commit=source["commit"],source_hash=source["source_hash"],
        inputs={args.root:item},root=args.root,seed=base.ROOTS[args.root][0],cpu=args.cpu,
        treatment=treatment(args.variant,args.short_lead),episode_seconds=args.seconds,
        run_role="FORMAL_SINGLE_ARM" if args.seconds==300 else "SMOKE_NOT_FORMAL",
        origin_input_binding=origin["binding_id"],baseline_output_readonly=str(args.baseline_output.resolve()),
        environment=environment,original_stage_A_outputs_modified=False,automatic_phase_advance=False,
        no_automatic_retry=True,human_replay_status="HUMAN_PENDING")


def json_lines(path):
    with Path(path).open(encoding="utf8") as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def measured(values,latency):
    values=list(values);result=latency.measured(values)
    result["mean_s"]=result["total_s"]/result["count"] if result["count"] else None
    return result


def saved_observables(directory,d,seconds):
    """Derive from saved authority and raw clocks; never step or replan a world."""
    callbacks=base.load(directory/"scheduler_records.json")
    tasks=base.load(directory/"normal_tasks.json")
    services=base.load(directory/"online_services.json")
    life=base.lifecycle_summary(tasks,callbacks,services,d.latency)
    rings=("outer","middle","inner")
    selections=[];scope_kind={}
    for record in callbacks:
        telemetry=record.get("selective_replanning",{});scope=telemetry.get("scope")
        if not scope:continue
        ring_by_id=telemetry.get("eligible_ring_by_id",{})
        selected=scope["replan_ids"];attempted=telemetry.get("attempted_replan_ids",selected)
        count=lambda ids:{ring:sum(ring_by_id.get(aid)==ring for aid in ids) for ring in rings}
        snapshot_time=record.get("planning_timing",{}).get("snapshot_time")
        kind="OVERFLOW_FULL" if "OVERFLOW" in scope.get("status","") else "LOCAL"
        if snapshot_time is not None:scope_kind[snapshot_time]=kind
        selections.append(dict(snapshot_time=snapshot_time,effective_time=scope["effective_time"],
            rule_variant=telemetry.get("rule_variant"),scope_kind=kind,selected_by_ring=count(selected),
            eligible_by_ring=count(scope["eligible_ids"]),attempted_union_by_ring=count(attempted),
            unknown_ring_ids=sorted(aid for aid in scope["eligible_ids"] if aid not in ring_by_id),
            outer_queue_seed_ids=[aid for aid,reasons in scope["selection_reasons"] if "OUTER_RING_QUEUE" in reasons]))
    starts=sorted(r["snapshot_time"] for r in tasks)
    ready=[]
    for row in tasks:
        if row.get("status")=="COMMITTED" and isinstance(row.get("ready_time"),(int,float)):
            gap=row["effective_time"]-row["ready_time"]
            base.require(math.isfinite(gap) and gap>=-1.e-7,"committed final-ready wait is negative/nonfinite")
            ready.append(dict(snapshot_time=row["snapshot_time"],ready_time=row["ready_time"],
                effective_time=row["effective_time"],final_ready_wait_s=gap))
    kinds={kind:measured((row["full_normal_planning_work_s"] for row in life["per_task_work"]
        if scope_kind.get(row["snapshot_time"],"UNKNOWN")==kind),d.latency) for kind in ("LOCAL","OVERFLOW_FULL","UNKNOWN")}
    from terminal_marl.core.rotating_slots.world import AIRBORNE
    plans=base.load(directory/"replay/trajectories.json")
    events=list(json_lines(directory/"replay/events.jsonl"))
    transition_events=[]
    for event in events:
        if event.get("type")!="maneuver_started":continue
        plan=plans.get(event.get("details",{}).get("trajectory_hash"),{})
        key=(plan.get("source_ring"),plan.get("target_ring"))
        if plan.get("kind")=="TRANSFER" and key in (("outer","middle"),("middle","inner")):
            transition_events.append(dict(time=event["time"],aircraft_id=event.get("aircraft_id"),
                transition="_to_".join(key),trajectory_hash=event["details"]["trajectory_hash"]))
    wanted=set(range(0,seconds+1,5));frames={}
    for frame in json_lines(directory/"replay/states.jsonl"):
        nearest=round(frame["time"])
        if nearest in wanted and abs(frame["time"]-nearest)<=1.e-8:frames[nearest]=frame
    boundaries=[]
    for boundary in sorted(wanted):
        frame=frames.get(boundary)
        boundaries.append(dict(time=boundary,airborne_by_ring=None if frame is None else
            {ring:sum(a.get("status") in AIRBORNE and a.get("ring")==ring for a in frame["aircraft"]) for ring in rings},
            airborne_ring_basis="RECORDED_AIRCRAFT_RING_NOT_PROJECTED_DECISION_NODE",exact_saved_frame=frame is not None,
            transition_starts={key:sum(e["time"]<=boundary and e["transition"]==key for e in transition_events)
                for key in ("outer_to_middle","middle_to_inner")}))
    result=dict(schema="rs_selective_followup_observables_v1",selections=selections,boundaries=boundaries,
        transition_start_events=transition_events,normal_lifecycle=life,complete_normal_work_by_scope_kind=kinds,
        final_ready_wait_committed=measured((r["final_ready_wait_s"] for r in ready),d.latency),
        final_ready_wait_committed_rows=ready,
        final_ready_wait_scope="COMMITTED_TASKS_WITH_RECORDED_READY_TIME_ONLY_EFFECTIVE_MINUS_FINAL_READY",
        snapshot_start_intervals=measured((later-earlier for earlier,later in zip(starts,starts[1:])),d.latency),
        snapshot_start_interval_scope="ALL_RECORDED_NORMAL_TASK_STARTS",
        note="Selection is not execution; all task work includes pending partial. Overflow full fallback is separate; no stall-cause inference.")
    d.atomic_json(directory/"followup_observables.json",result)
    return result


def arm(args,d):
    started=time.perf_counter()
    treatment_row=treatment(args.variant,args.short_lead)
    base.require(args.root in ROOT_CPUS and args.cpu==ROOT_CPUS[args.root],"root-specific physical CPU12/14 required")
    base.require(args.seconds in (10,300),"only ten-second smoke or preregistered 300-second arm")
    output=args.output.resolve();baseline=args.baseline_output.resolve();home_outputs=(Path.home()/"training_outputs/terminal").resolve()
    base.require(output.parent==home_outputs and baseline.parent==home_outputs,"outputs must be separate direct children of ~/training_outputs/terminal")
    base.require(not output.is_relative_to(baseline) and not baseline.is_relative_to(output),"follow-up must not overlap original Stage A outputs")
    base.require(not output.is_relative_to(PROJECT),"outputs cannot be written into source")
    source=base.load(args.source_freeze);base.verify_sources(source)
    base.require("scripts/stage3/run_selective_reoptimization_followup.py" in source["files"],"new follow-up runner must be in reviewed source freeze")
    origin_path=baseline/"resolved_run_binding.json";origin=base.load(origin_path)
    item=origin["inputs"][args.root]
    base.require(base.file_hash(item["run_checkpoint"])==item["initial_checkpoint_sha256"],"original t0 checkpoint changed")
    base.require(item["row"]["seed"]==base.ROOTS[args.root][0],"original seed changed")
    resource=d.campaign.resource_probe()
    base.require(args.cpu in resource.get("free_physical_cores",[]),"requested physical CPU unavailable")
    base.require(hasattr(os,"sched_setaffinity"),"formal paired runs require CPU affinity")
    os.sched_setaffinity(0,{args.cpu})
    arm_root=output/"arms"/args.root/args.variant
    arm_root.mkdir(parents=True,exist_ok=False)
    binding=build_binding(origin,source,args,resource)
    binding["origin_input_binding_sha256"]=base.file_hash(origin_path)
    binding["source_freeze_sha256"]=base.file_hash(args.source_freeze)
    binding["binding_id"]=base.digest(binding)
    d.atomic_json(arm_root/"resolved_run_binding.json",binding)
    d.atomic_json(arm_root/"treatment.json",dict(treatment_row,source_binding=binding["binding_id"],
        episode_seconds=args.seconds,cpu=args.cpu,seed=base.ROOTS[args.root][0]))
    result=base.closed_job(arm_root,args.root,"RULE_LOCAL_K8",binding,d,
        lead_seconds=treatment_row["lead_seconds"],selector_variant=treatment_row["selector_variant"],episode_seconds=args.seconds)
    directory=arm_root/"evaluation"/args.root/"RULE_LOCAL_K8"
    saved_observables(directory,d,args.seconds)
    final=dict(status=result["status"],root=args.root,variant=args.variant,source_binding=binding["binding_id"],
        scientific_source=source["commit"],episode_seconds=args.seconds,completed_boundary_s=result["completed_boundary_s"],
        treatment=treatment_row,result_path=str(directory/"result.json"),observables_path=str(directory/"followup_observables.json"),
        full_arm_wall_including_observables_s=time.perf_counter()-started,human_replay_status="HUMAN_PENDING")
    d.atomic_json(arm_root/"arm.finished.json",final)
    return final


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("arm",))
    parser.add_argument("--baseline-output",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-freeze",type=Path,required=True)
    parser.add_argument("--root",choices=tuple(ROOT_CPUS),required=True);parser.add_argument("--variant",choices=tuple(VARIANTS),required=True)
    parser.add_argument("--short-lead",type=float,choices=(15.,),required=True)
    parser.add_argument("--cpu",type=int,choices=(12,14),required=True);parser.add_argument("--seconds",type=int,choices=(10,300),required=True)
    args=parser.parse_args(argv)
    launch_threads={key:os.environ.get(key) for key in base.THREADS}
    base.require(all(value=="1" for value in launch_threads.values()),"set numerical threads before Python starts")
    d=base.dependencies(PROJECT);d.launch_threads=launch_threads
    base.print_result(arm(args,d))


if __name__=="__main__":main()
