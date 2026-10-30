"""Read-only A4 four-arm extraction. Missing inputs remain null/INCOMPLETE.

No run, observation update, history mutation, experiment write, or scientific
interpretation is performed. Output must be a fresh local analysis directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
ROOTS=("s3r23","s3r24")
METHODS=("FULL_H","RULE_LOCAL_K8")
PHASES=("candidate_generation_and_single_vehicle","trajectory_validation","prefix_validation")
DEADLINE_REASONS=("READY_DEADLINE_MISSED","LATENCY_PACKET_EXPIRED","TRAFFIC_CHANGED_AFTER_DEADLINE","REVALIDATION_WORK_DEADLINE")


def load(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def digest_file(path): return sha256(Path(path).read_bytes()).hexdigest()
def finite(value): return type(value) in (int,float) and math.isfinite(value)
def pick(value,*names): return {name:value.get(name) for name in names}


def stats(values):
    values=[x for x in values if finite(x)]
    return dict(n=len(values),sum=math.fsum(values) if values else None,
                mean=math.fsum(values)/len(values) if values else None,max=max(values) if values else None)


def raw_counts(rows,fields):
    result=[]
    for field in fields:
        counter=Counter(json.dumps(row[field],sort_keys=True) for row in rows if field in row)
        result.extend(dict(field=field,value=json.loads(value),count=n) for value,n in sorted(counter.items()))
    return result


def index_by_time(rows,field):
    result=defaultdict(list)
    for index,row in enumerate(rows):
        value=row.get(field)
        if finite(value): result[value].append(index)
    return result


def ending_history(checkpoint):
    """Decode saved state; ages() is queried without observe()/world restoration."""
    from terminal_marl.core.rotating_slots.stage2_campaign import Stage2Codec,digest
    from terminal_marl.core.rotating_slots.selective_replanning.history import PlanningHistory
    from terminal_marl.core.rotating_slots.world import AIRBORNE
    envelope=load(checkpoint); payload=envelope["payload"]
    if envelope["sha256"]!=digest(payload): raise ValueError("terminal checkpoint digest mismatch")
    codec=Stage2Codec(); world=codec.decode(payload["world"]); scheduler=codec.decode(payload["scheduler"])
    saved=scheduler["callback"]["selective_replanning"]["history"]
    history=PlanningHistory(progress_policy=saved.get("progress_policy","legacy"))
    history.load_state_dict(deepcopy(saved)); before=codec.encode(history.state_dict())
    now=payload["time"]
    rows=[dict(aircraft_id=aid,status=a.status,**history.ages(aid,now))
          for aid,a in sorted(world["aircraft"].items()) if a.status in AIRBORNE]
    if before!=codec.encode(history.state_dict()): raise ValueError("ages query changed saved history")
    reviews=[r["review_age"] for r in rows if finite(r["review_age"])]
    progress=[r["progress_age"] for r in rows if finite(r["progress_age"])]
    return dict(absolute_time=now,history_last_observed_time=history.last_observed_time,
        history_unmodified=True,airborne_count=len(rows),rows=rows,
        known_review_count=len(reviews),known_review_ge60_count=sum(x>=60. for x in reviews),
        unknown_review_count=sum(r["review_unknown"] for r in rows),max_known_review_age_s=max(reviews,default=None),
        known_progress_count=len(progress),known_progress_ge60_count=sum(x>=60. for x in progress),
        unknown_progress_count=sum(r["progress_unknown"] for r in rows),max_known_progress_age_s=max(progress,default=None))


def callback_timings(callbacks,root,method):
    rows=[]; names=set(PHASES)
    for cb in callbacks: names.update((cb.get("timings") or {}).keys())
    for index,cb in enumerate(callbacks):
        for name in sorted(names):
            value=(cb.get("timings") or {}).get(name)
            valid=isinstance(value,(list,tuple)) and len(value)==2
            rows.append(dict(root=root,method=method,callback_index=index,
                snapshot_time=(cb.get("planning_timing") or {}).get("snapshot_time"),phase=name,
                present=name in (cb.get("timings") or {}),
                wall_s=value[0] if valid else None,cpu_s=value[1] if valid else None))
    summary={name:dict(wall=stats(r["wall_s"] for r in rows if r["phase"]==name),
        cpu=stats(r["cpu_s"] for r in rows if r["phase"]==name),
        missing_callbacks=sum(not r["present"] for r in rows if r["phase"]==name)) for name in sorted(names)}
    return rows,summary


def starts_for(hashes,event_index,t0,end,effective):
    if hashes is None: return None
    result=[]
    for h in hashes:
        matches=[dict(event,within_method_window=t0<=event["time"]<=end,
            matches_task_effective=(abs(event["time"]-effective)<=1.e-8 if finite(effective) else None))
            for event in event_index.get(h,())]
        result.append(dict(trajectory_hash=h,starts=matches))
    return result


def task_rows(tasks,callbacks,work_rows,events,root,method,t0,end):
    by_cb=defaultdict(list)
    for index,cb in enumerate(callbacks):
        key=(cb.get("planning_timing") or {}).get("snapshot_time")
        if finite(key): by_cb[key].append(index)
    by_work=index_by_time(work_rows,"snapshot_time"); by_task=index_by_time(tasks,"snapshot_time")
    event_index=defaultdict(list)
    for index,event in enumerate(events):
        if event.get("type")!="maneuver_started": continue
        details=event.get("details") or {}; h=details.get("trajectory_hash")
        event_index[h].append(dict(event_index=index,time=event.get("time"),aircraft_id=event.get("aircraft_id"),
            trajectory_hash=h,maneuver_kind=details.get("maneuver_kind")))
    rows=[]; issues=[]
    for index,task in enumerate(tasks):
        now=task.get("snapshot_time"); matches=by_cb.get(now,[]); work_matches=by_work.get(now,[])
        cb=callbacks[matches[0]] if len(matches)==1 else None
        work=work_rows[work_matches[0]] if len(work_matches)==1 else None
        match_status="MATCHED" if len(matches)==1 else "MISSING" if not matches else "DUPLICATE"
        work_status="MATCHED" if len(work_matches)==1 else "MISSING" if not work_matches else "DUPLICATE"
        if match_status!="MATCHED": issues.append(dict(task_index=index,kind="callback_match",status=match_status,indices=matches))
        if work_status!="MATCHED": issues.append(dict(task_index=index,kind="work_match",status=work_status,indices=work_matches))
        if len(by_task.get(now,()))!=1: issues.append(dict(task_index=index,kind="duplicate_task_time",snapshot_time=now))
        selective=(cb or {}).get("selective_replanning") or {}; scope=selective.get("scope") or {}
        handoff=(cb or {}).get("search_handoff") or {}
        first=handoff.get("first_nonempty_certified")
        first_source="search_handoff.first_nonempty_certified" if first is not None else None
        if first is None and selective.get("first_nonempty_certified") is not None:
            first=selective["first_nonempty_certified"]; first_source="selective_replanning.first_nonempty_certified"
        first_hashes=first.get("prefix_hashes") if first is not None else None
        final_hashes=(cb or {}).get("actual_submitted_trajectory_hashes")
        finished=task.get("finished_time"); next_started=task.get("next_started_time")
        # The wrapper writes the raw gap on both the incoming new record and
        # its finished predecessor; only next_started_time is outgoing here.
        outgoing_gap=next_started-finished if finite(finished) and finite(next_started) else None
        row=dict(root=root,method=method,task_index=index,callback_indices=matches,callback_match=match_status,
            work_indices=work_matches,work_match=work_status,
            **pick(task,"snapshot_time","effective_time","result_time","preparation_started_time","ready_time",
                "finished_time","next_started_time","previous_finished_time","previous_record_index",
                "status","reason","submitted","restart_reason"),
            raw_restart_gap_s=task.get("finish_to_next_start_s"),finish_to_next_start_s=outgoing_gap,
            scope_status=scope.get("status"),actual_k=scope.get("actual_k"),selection_status=selective.get("selection_status"),
            full_normal_planning_work_s=(work or {}).get("full_normal_planning_work_s"),
            observed_work_components=(work or {}).get("observed_components"),
            first_nonempty_certified=first,first_certificate_source=first_source,
            first_nonempty_certified_wall_elapsed_s=(first or {}).get("wall_elapsed_s"),
            first_prefix_hashes=first_hashes,final_submitted_hashes=final_hashes,
            proposed_hashes=task.get("proposed_trajectory_hashes"),
            first_prefix_start_matches=starts_for(first_hashes,event_index,t0,end,task.get("effective_time")),
            actual_starts=starts_for(final_hashes,event_index,t0,end,task.get("effective_time")))
        rows.append(row)
    missing_timing=[i for i,cb in enumerate(callbacks) if not finite((cb.get("planning_timing") or {}).get("snapshot_time"))]
    unmatched=[i for key,indices in by_cb.items() if key not in by_task for i in indices]
    return rows,dict(issues=issues,callbacks_missing_snapshot_time=missing_timing,callbacks_without_task=unmatched)


def completion_rows(callbacks):
    result=[]
    for index,cb in enumerate(callbacks):
        selective=cb.get("selective_replanning") or {}
        sources=[("suffix_completion",cb.get("suffix_completion")),
                 ("selective_replanning.suffix_completion",selective.get("suffix_completion"))]
        for source,report in sources:
            if report is None: continue
            # Preserve both paths rather than summing duplicated telemetry.
            result.append(dict(callback_index=index,snapshot_time=(cb.get("planning_timing") or {}).get("snapshot_time"),
                source=source,report=deepcopy(report)))
    return result


def extract_arm(input_root,root,method):
    home=input_root/"arms"/root/method; directory=home/"evaluation"/root/method
    paths={"result":directory/"result.json","boundary":directory/"boundary_0300.json",
        "callbacks":directory/"method_scheduler_records.json","tasks":directory/"method_normal_tasks.json",
        "services":directory/"method_online_services.json","checkpoint":directory/"checkpoint_terminal.json",
        "events":directory/"replay/events.jsonl","binding":home/"resolved_run_binding.json",
        "common":input_root/"common_initial"/root/"common.checkpoint.json"}
    missing=[str(p) for p in paths.values() if not p.is_file()]
    arm=dict(root=root,method=method,status="INCOMPLETE",missing_inputs=missing,
        inputs={name:dict(path=str(p),sha256=digest_file(p) if p.is_file() else None) for name,p in paths.items()})
    if missing: return arm,[],[]
    try:
        result,boundary,binding=(load(paths[key]) for key in ("result","boundary","binding"))
        callbacks,tasks,services=(load(paths[key]) for key in ("callbacks","tasks","services"))
        events=[json.loads(line) for line in paths["events"].read_text(encoding="utf8").splitlines() if line.strip()]
        checks=dict(result_complete=result.get("status")=="COMPLETE",
            result_duration=finite(result.get("time")) and finite(result.get("t0")) and abs(result["time"]-result["t0"]-300.)<=1.e-7,
            result_boundary=result.get("completed_relative_s")==300 and result.get("target_relative_s")==300,
            boundary_duration=finite(boundary.get("time")) and finite(boundary.get("t0")) and abs(boundary["time"]-boundary["t0"]-300.)<=1.e-7,
            time_consistent=result.get("time")==boundary.get("time") and result.get("t0")==boundary.get("t0"),
            profile_consistent=isinstance(binding.get("evaluation_profile"),str) and bool(binding["evaluation_profile"]) and
                result.get("evaluation_profile")==boundary.get("evaluation_profile")==binding["evaluation_profile"],
            common_sha_matches=binding.get("common_checkpoint_sha256")==arm["inputs"]["common"]["sha256"])
        arm.update(checks=checks,evaluation_profile=binding.get("evaluation_profile"),
            common_checkpoint_sha256=binding.get("common_checkpoint_sha256"),t0=result.get("t0"),time=result.get("time"))
        increment=boundary.get("increment") or {}; delta=increment.get("cumulative_delta") or {}
        required=("L_normal","L_emergency","F_energy_total","failed_before_landing","energy_j","holding_time")
        arm["missing_metric_fields"]=["increment.cumulative_delta."+name for name in required if not finite(delta.get(name))]
        checks["post_t0_metrics_present"]=not arm["missing_metric_fields"]
        if not finite(result.get("t0")) or not finite(result.get("time")):
            arm["error"]="numeric t0/time missing"; return arm,[],[]
        life=boundary.get("normal_lifecycle") or {}; work=life.get("per_task_work") or []
        timeline,matches=task_rows(tasks,callbacks,work,events,root,method,result["t0"],result["time"])
        timing_rows,timing_summary=callback_timings(callbacks,root,method)
        committed=[t for t in tasks if t.get("status")=="COMMITTED" and finite(t.get("submitted")) and t["submitted"]>0]
        finished=[t for t in tasks if "finished_time" in t]
        submitted=[t.get("submitted") for t in tasks]
        all_submitted=math.fsum(submitted) if all(finite(v) for v in submitted) else None
        by_scope=defaultdict(list)
        for row in timeline: by_scope[row["scope_status"]].append(row)
        scope_work=[dict(scope_status=label,task_count=len(rows),
            full_work=stats(row["full_normal_planning_work_s"] for row in rows),
            actual_k=[row["actual_k"] for row in rows]) for label,rows in by_scope.items()]
        latest_times=[(i,(cb.get("planning_timing") or {}).get("snapshot_time")) for i,cb in enumerate(callbacks)]
        maximum=max((t for _,t in latest_times if finite(t)),default=None)
        latest=[dict(callback_index=i,age_basis_snapshot_time=t,callback_projected_time=callbacks[i].get("time"),
            ages=(callbacks[i].get("selective_replanning") or {}).get("ages"))
            for i,t in latest_times if maximum is not None and t==maximum]
        raw=[]
        for i,cb in enumerate(callbacks):
            s=cb.get("selective_replanning") or {}; row=dict(callback_index=i)
            for key in ("fallback","status","skip_reason","solver_stop_reason","generation_stop_reason","stop_reason","truncations"):
                if key in cb: row[key]=cb[key]
            for key in ("actual_fallback","overflow_cause","selection_status"):
                if key in s: row["selective_"+key]=s[key]
            raw.append(row)
        ending=ending_history(paths["checkpoint"])
        checks["terminal_checkpoint_time"]=ending["absolute_time"]==result["time"]
        history=(life.get("selective_history_work") or {}).get("unattributed_service_s")
        arm.update(increment=deepcopy(boundary.get("increment")),quality=deepcopy(boundary.get("quality")),
            holding_time_by_ring=boundary.get("holding_time_by_ring"),holding_energy_by_ring=boundary.get("holding_energy_by_ring"),
            total_executed_energy_by_source_ring=boundary.get("total_executed_energy_by_source_ring"),
            inherited_finals=boundary.get("inherited_finals"),
            normal_planning=dict(task_count=len(tasks),finished_task_count=len(finished),committed_nonempty_count=len(committed),
                committed_nonempty_over_all_finished=len(committed)/len(finished) if finished else None,
                submitted_plan_total=all_submitted,pending_task_count=sum("finished_time" not in t for t in tasks),
                existing_lifecycle=pick(life,"join_rate","join_rate_denominator","finished_nonempty_groups","committed_nonempty_groups",
                    "actual_submitted_plans","full_normal_planning_work","finished_task_full_planning_work"),
                task_reason_counts=raw_counts(tasks,("status","reason")),
                deadline_reason_counts=[dict(reason=reason,count=sum(str(t.get("reason","")).split(":",1)[0]==reason for t in tasks))
                    for reason in DEADLINE_REASONS],scope_work=scope_work,
                unattributed_history_work_s=history,per_task_work_observed_sum=stats(row["full_normal_planning_work_s"] for row in timeline),
                history_accounting="per_task_work already includes task history; unattributed history stays separate"),
            callback_matching=matches,timing_summary=timing_summary,timing_semantics="exclusive seconds per callback; mean is not per primitive",
            ending_history=ending,latest_callback_ages=latest,completion=completion_rows(callbacks),
            fallback_records=raw,fallback_counts=raw_counts(raw,tuple(sorted({k for row in raw for k in row if k!="callback_index"}))),
            method_service_count=len(services))
        arm["status"]="COMPLETE" if all(checks.values()) and not any(matches.values()) else "INCOMPLETE"
        return arm,timeline,timing_rows
    except Exception as exc:
        arm["error"]=dict(type=type(exc).__name__,message=str(exc)); return arm,[],[]


def collect(input_root):
    input_root=Path(input_root).resolve(); arms=[]; timeline=[]; timings=[]
    for root in ROOTS:
        for method in METHODS:
            arm,rows,phases=extract_arm(input_root,root,method)
            arms.append(arm); timeline.extend(rows); timings.extend(phases)
    pair_checks={root:len({a.get("common_checkpoint_sha256") for a in arms if a["root"]==root})==1 and
        all(a.get("common_checkpoint_sha256") is not None for a in arms if a["root"]==root) for root in ROOTS}
    profiles={a.get("evaluation_profile") for a in arms}
    same_profile=len(profiles)==1 and None not in profiles
    report=dict(schema="a4_common_comparison_numeric_summary_v1",input_root=str(input_root),
        status="COMPLETE" if all(a["status"]=="COMPLETE" for a in arms) and all(pair_checks.values()) and same_profile else "INCOMPLETE",
        same_common_sha_by_root=pair_checks,same_evaluation_profile=same_profile,arms=arms,
        task_timeline=timeline,callback_timings=timings,
        semantics=dict(no_historical_cold_start_comparison=True,first_certificate_wall_is_not_simulation_time=True,
            missing_values="null; no zero imputation",history_queried_without_observe=True,
            raw_restart_gap_s="saved finish_to_next_start_s; may describe the incoming predecessor gap",
            finish_to_next_start_s="outgoing next_started_time minus finished_time; null unless both finite"))
    return report,timeline,timings


def csv_rows(path,rows,columns):
    with path.open("x",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
        for row in rows:
            converted={k:("null" if row.get(k) is None else json.dumps(row[k],sort_keys=True) if isinstance(row[k],(list,dict)) else row[k]) for k in columns}
            writer.writerow(converted)


def write_outputs(output,report,timeline,timings):
    output=Path(output).resolve()
    if not output.is_relative_to(HERE) or not output.name.startswith("analysis"):
        raise ValueError("output must be a fresh local analysis directory under this helper's artifact directory")
    input_root=Path(report["input_root"])
    if output.is_relative_to(input_root) or input_root.is_relative_to(output):
        raise ValueError("analysis output must be separate from all experiment inputs")
    output.mkdir(parents=True,exist_ok=False)
    with (output/"summary.json").open("x",encoding="utf8") as stream:
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False)
    rows=[]
    for arm in report["arms"]:
        increment=arm.get("increment") or {}; delta=increment.get("cumulative_delta") or {}
        normal=arm.get("normal_planning") or {}; existing=normal.get("existing_lifecycle") or {}
        row=pick(arm,"root","method","status","t0","time","evaluation_profile","common_checkpoint_sha256")
        row.update(pick(delta,"L_normal","L_emergency","F_energy_total","failed_before_landing","energy_j","holding_time"))
        row.update(pick(normal,"task_count","finished_task_count","committed_nonempty_count","committed_nonempty_over_all_finished",
            "submitted_plan_total","pending_task_count","unattributed_history_work_s"))
        row.update(ending_pending=increment.get("ending_pending"),existing_join_rate=existing.get("join_rate"),
            existing_join_rate_denominator=existing.get("join_rate_denominator"),
            full_normal_planning_work_s=(existing.get("full_normal_planning_work") or {}).get("total_s"),
            per_task_work_observed_sum_s=(normal.get("per_task_work_observed_sum") or {}).get("sum"),
            scope_work=normal.get("scope_work"),
            holding_time_by_ring=arm.get("holding_time_by_ring"),holding_energy_by_ring=arm.get("holding_energy_by_ring"),
            total_executed_energy_by_source_ring=arm.get("total_executed_energy_by_source_ring"),
            inherited_final_touchdown_ids=increment.get("inherited_final_touchdown_ids"),
            treatment_final_touchdown_ids=increment.get("treatment_final_touchdown_ids"))
        row.update(pick(arm.get("ending_history") or {},"known_review_ge60_count","unknown_review_count",
            "max_known_review_age_s","known_progress_ge60_count","unknown_progress_count","max_known_progress_age_s"))
        rows.append(row)
    csv_rows(output/"four_arm_summary.csv",rows,list(rows[0]))
    columns=list(timeline[0]) if timeline else ["root","method","task_index","callback_match","first_nonempty_certified_wall_elapsed_s"]
    csv_rows(output/"task_timeline.csv",timeline,columns)
    columns=list(timings[0]) if timings else ["root","method","callback_index","snapshot_time","phase","present","wall_s","cpu_s"]
    csv_rows(output/"callback_timings.csv",timings,columns)
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    report,timeline,timings=collect(args.input_root)
    output=write_outputs(args.output_dir,report,timeline,timings)
    print(json.dumps(dict(status=report["status"],output=str(output)),sort_keys=True))
    return 0 if report["status"]=="COMPLETE" else 2


if __name__=="__main__": raise SystemExit(main())
