"""Local, read-only extraction of a COMPLETE R3/R4 final comparison.

No world, solver, source modules, or remote commands are imported or invoked.
Full future telemetry is retained in a detailed JSON sidecar; compact rows keep
per-ID published-future lifecycles and callback-scoped unpublished selections.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ARTIFACT_ROOT=Path(__file__).resolve().parent
EXPECTED={(r,m) for r in ("s3r23","s3r24") for m in ("FULL_H","RULE_LOCAL_K8")}
TOL=1.e-7
HEAVY={"first_action_rows","reservation_rows","predicted_suffix_rows","final_selected_occurrence_rows"}


def require(ok,message):
    if not ok: raise ValueError(message)


def finite(value): return type(value) in (int,float) and math.isfinite(value)
def sha(data): return hashlib.sha256(data).hexdigest()
def encoded(value): return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf8")


def read(path):
    path=Path(path).resolve(); data=path.read_bytes()
    def invalid(value): raise ValueError("nonfinite JSON constant: "+value)
    return json.loads(data.decode("utf-8-sig"),parse_constant=invalid),dict(path=str(path),sha256=sha(data),bytes=len(data))


def index(rows,label):
    keys=[(r.get("root"),r.get("method")) for r in rows]
    require(len(rows)==4 and len(set(keys))==4 and set(keys)==EXPECTED,label+" requires four unique authorized arms")
    return dict(zip(keys,rows))


def validate(value):
    require(value.get("schema")=="service_opportunity_comparison_v1" and value.get("window")==[20.,320.],"complete t20-320 comparison schema required")
    require(set(value.get("campaigns",{}))=={"old","new"},"old/new campaigns required")
    for label,campaign in value["campaigns"].items():
        ordinary=campaign.get("ordinary") or {}
        require(ordinary.get("status")=="COMPLETE",label+" ordinary campaign is incomplete")
        arms=index(ordinary.get("arms",[]),label+" ordinary")
        index(campaign.get("metrics",[]),label+" metrics")
        for arm in arms.values():
            require(arm.get("status")=="COMPLETE" and arm.get("t0")==20. and arm.get("time")==320.,label+" arm is not a complete terminal result")
            require(not arm.get("checks") or all(arm["checks"].values()),label+" arm has failed saved checks")
    for root in ("s3r23","s3r24"):
        hashes=[a.get("common_checkpoint_sha256") for c in value["campaigns"].values() for a in c["ordinary"]["arms"] if a["root"]==root]
        require(len(hashes)==4 and None not in hashes and len(set(hashes))==1,"common checkpoint mismatch or missing: "+root)


def future_review(future,lo=20.,hi=320.):
    if future is None: return None
    raw=future.get("reservation_rows")
    if not isinstance(raw,list): return dict(status="unknown",reason="reservation_rows_missing")
    units={}; missing_ids=0
    for row in raw:
        rid=row.get("reservation_id")
        if not rid: missing_ids+=1; continue
        require(rid not in units or units[rid]==row,"conflicting duplicate reservation ID: "+str(rid))
        units[rid]=row
    published=[r for r in units.values() if r.get("published") is True]
    unknown=[r for r in published if not finite(r.get("start_time")) or not finite(r.get("effective_time"))]
    selected=[r for r in published if finite(r.get("start_time")) and finite(r.get("effective_time")) and r["start_time"]>r["effective_time"]+TOL]
    compact=[]
    for row in selected:
        start=row["start_time"]; raw_starts=row.get("started_times")
        starts=[t for t in raw_starts or [] if finite(t)]
        observed=[t for t in starts if lo-TOL<=t<=hi+TOL]
        cancelled=row.get("cancelled") is True
        due=lo-TOL<=start<=hi+TOL
        completion_times=[t for t in row.get("completed_times",[]) if finite(t) and lo-TOL<=t<=hi+TOL]
        completed=bool(completion_times)
        complete_start_times=isinstance(raw_starts,list) and all(finite(t) for t in raw_starts)
        timely=(False if any(abs(t-start)>TOL for t in starts) else True if starts and complete_start_times else None)
        compact.append(dict(reservation_id=row["reservation_id"],trajectory_hash=row.get("trajectory_hash"),
            aircraft_id=row.get("aircraft_id"),effective_time=row["effective_time"],absolute_start=start,absolute_end=row.get("end_time"),
            published_times=deepcopy(row.get("published_times")),started_times=deepcopy(raw_starts),completed_times=deepcopy(row.get("completed_times")),
            cancelled_times=deepcopy(row.get("cancelled_times")),cancellation_reasons=deepcopy(row.get("cancellation_reasons")),
            cancelled=cancelled,pending_cancelled=row.get("pending_cancelled"),superseded_pending=row.get("superseded_pending"),
            start_due_within_window=due,started_within_window=bool(observed),all_recorded_starts_match_intended=timely,
            completed_within_window=completed,pending_outside_window=start>hi+TOL and not cancelled and not observed and not completed,
            start_time_metadata_complete=complete_start_times,
            unexplained_due_absent=due and not cancelled and not observed and not completed,
            due_at_window_end=abs(start-hi)<=TOL))
    occurrences=future.get("final_selected_occurrence_rows")
    occurrence_fields=("callback_index","aircraft_id","candidate_hash","trajectory_hash","snapshot_time","effective_time",
        "absolute_start","absolute_end","state","first_action_middle_inner","already_reserved_ids","published_for_occurrence_ids",
        "bound_reservation_ids","callback_delivery_observation","reservation_reason_observations")
    missing=[dict({k:deepcopy(o.get(k)) for k in occurrence_fields},detailed_occurrence_index=i) for i,o in enumerate(occurrences or [])
        if o.get("state")=="not_published_by_window_end"]
    return dict(status="measured_with_coverage_counts",tolerance_s=TOL,
        published_first_action_count=len(published),published_future_count=len(compact),
        published_future_start_due_within_window_count=sum(r["start_due_within_window"] for r in compact),
        published_future_uncancelled_due_count=sum(r["start_due_within_window"] and not r["cancelled"] for r in compact),
        published_future_started_count=sum(r["started_within_window"] for r in compact),
        published_future_timely_started_count=sum(r["started_within_window"] and r["all_recorded_starts_match_intended"] is True for r in compact),
        published_future_mistimed_started_count=sum(r["started_within_window"] and r["all_recorded_starts_match_intended"] is False for r in compact),
        published_future_started_missing_time_metadata_count=sum(r["started_within_window"] and not r["start_time_metadata_complete"] for r in compact),
        published_future_completed_count=sum(r["completed_within_window"] for r in compact),
        published_future_cancelled_count=sum(r["cancelled"] for r in compact),
        published_future_pending_outside_count=sum(r["pending_outside_window"] for r in compact),
        published_future_unexplained_due_absent_count=sum(r["unexplained_due_absent"] for r in compact),
        boundary_due_unobserved_count=sum(r["unexplained_due_absent"] and r["due_at_window_end"] for r in compact),
        published_future_lifecycle=compact,
        replacement_count=future.get("replacement_count"),absolute_start_postponement_count=future.get("absolute_start_postponement_count"),
        explicit_missed_start_count=future.get("missed_start_count"),pending_cancelled_reservation_count=future.get("pending_cancelled_reservation_count"),
        superseded_pending_reservation_count=future.get("superseded_pending_reservation_count"),
        missing_reservation_id_rows=missing_ids,published_missing_future_time_metadata_count=len(unknown),
        published_missing_identity_metadata_count=sum(not r.get("trajectory_hash") or not r.get("aircraft_id") for r in published),
        first_action_unclassified_count=future.get("unclassified_first_action_count"),
        malformed_event_count=len(future["malformed_event_rows"]) if isinstance(future.get("malformed_event_rows"),list) else None,
        selected_unpublished_occurrence_count=len(missing) if isinstance(occurrences,list) else None,
        selected_unpublished_middle_inner_occurrence_count=sum(o.get("first_action_middle_inner") is True for o in missing) if isinstance(occurrences,list) else None,
        selected_unpublished_callback_group_reason_counts=dict(Counter((o.get("callback_delivery_observation") or {}).get("reason")
            for o in missing if (o.get("callback_delivery_observation") or {}).get("reason"))),
        selected_unpublished_callback_group_reason_unknown_count=sum(not (o.get("callback_delivery_observation") or {}).get("reason") for o in missing),
        selected_unpublished_reservation_reason_counts=dict(Counter(r["reason"] for o in missing for r in (o.get("reservation_reason_observations") or []) if r.get("reason"))),
        selected_unpublished_occurrences=missing,
        occurrence_state_counts=deepcopy(future.get("final_selected_occurrence_state_counts")),
        semantics="counts deduplicate reservation IDs; future means absolute_start>effective+tol; all recorded starts must match intended start; cancelled IDs are excluded from unexplained absence; absence is observational, not inferred missed-start causality; callback/group reasons are associated only")


def extract(value,provenance,ledger=None):
    validate(value); rows=[]; details={}
    for label,campaign in value["campaigns"].items():
        arms=index(campaign["ordinary"]["arms"],label)
        for metric in campaign["metrics"]:
            root,method=metric["root"],metric["method"]; arm=arms[root,method]
            future=metric.get("future_reservation_metrics"); key="/".join((label,root,method)); details[key]=deepcopy(future)
            physical=metric.get("physical_idle") or {}; supply=metric.get("middle_inner_supply") or {}
            normal=arm.get("normal_planning") or {}; life=normal.get("existing_lifecycle") or {}
            candidate=metric.get("candidate_lifecycle")
            future_compact={k:deepcopy(v) for k,v in future.items() if k not in HEAVY} if isinstance(future,dict) else None
            if future_compact is not None:
                for section in ("all_first_actions","middle_inner"):
                    for transition in (future_compact.get(section) or {}).get("transitions",[]): transition.pop("trajectory_hashes",None)
            review=future_review(future)
            if review and review.get("status")!="unknown":
                for occurrence in review["selected_unpublished_occurrences"]:
                    matches=[t for t in campaign["ordinary"].get("task_timeline",[]) if t.get("root")==root and t.get("method")==method
                        and occurrence["callback_index"] in (t.get("callback_indices") or [])]
                    fields=("task_index","status","reason","submitted","snapshot_time","effective_time","result_time",
                        "preparation_started_time","ready_time","finished_time")
                    occurrence["task_timeline_matches"]=[{k:deepcopy(t.get(k)) for k in fields} for t in matches]
                    occurrence["task_observation"]=("task unfinished at observation end" if len(matches)==1 and
                        matches[0].get("status")=="COMPUTING" and matches[0].get("finished_time") is None else
                        "matched task observation" if len(matches)==1 else "missing or nonunique task match")
            certificate_records=metric.get("first_certificate_times")
            times=[r["first_nonempty_certified"]["wall_elapsed_s"] for r in certificate_records or []
                if isinstance(r.get("first_nonempty_certified"),dict) and finite(r["first_nonempty_certified"].get("wall_elapsed_s"))]
            audit=metric.get("saved_audit_result") or arm.get("saved_audit_result")
            n=len(times); committed=normal.get("committed_nonempty_count"); finished=normal.get("finished_task_count")
            idle=physical.get("idle_by_pad")
            idle_by_pad=({str(p):math.fsum(b-a for a,b in intervals) for p,intervals in idle.items()} if isinstance(idle,dict) else None)
            row=dict(campaign=label,root=root,method=method,evidence_type="preexisting_artifact" if label=="old" else "newly_run",
                status=arm["status"],input_root=metric.get("arm_input_root",arm.get("arm_input_root",campaign.get("input_root"))),
                export_lineage=deepcopy(metric.get("export_lineage",arm.get("export_lineage"))),
                treatment_normal=metric.get("treatment_normal_touchdown_count"),inherited_normal=metric.get("inherited_final_touchdown_count"),
                normal_touchdown_delta=metric.get("normal_touchdown_delta"),increment_energy_kwh=metric.get("increment_energy_kwh"),
                emergencies=metric.get("emergencies"),failures=metric.get("failures"),
                pad_seconds_idle=physical.get("pad_seconds_idle"),pad_idle_seconds_by_pad=idle_by_pad,
                longest_all_pad_idle_s=physical.get("longest_all_pad_idle_s"),
                inner_stock_empty_longest_s=(metric.get("inner_stock_empty") or {}).get("longest_inner_stock_empty_with_upstream_demand_s"),
                middle_inner_starts_count=supply.get("actual_maneuver_start_count"),middle_inner_completed_count=supply.get("completed_effective_supply_count"),
                middle_inner_actual_timing=deepcopy(metric.get("middle_inner_supply")),
                holding_time_by_ring=deepcopy(metric.get("holding_time_by_ring")),holding_energy_by_ring=deepcopy(metric.get("holding_energy_by_ring")),
                normal_planning_committed_nonempty=committed,normal_planning_finished_nonempty=life.get("finished_nonempty_groups"),
                normal_planning_finished_task_count=finished,normal_planning_join_rate=life.get("join_rate"),
                normal_planning_join_rate_denominator=life.get("join_rate_denominator"),
                normal_planning_committed_nonempty_over_all_finished=(committed/finished if finite(committed) and finite(finished) and finished else None),
                first_nonempty_cert_count=n if isinstance(certificate_records,list) else None,
                first_nonempty_cert_elapsed_values_s=times if isinstance(certificate_records,list) else None,
                first_nonempty_cert_elapsed_median_s=statistics.median(times) if n else None,
                first_nonempty_cert_elapsed_p95_s=sorted(times)[math.ceil(.95*n)-1] if n else None,
                audit_status=deepcopy(audit),audit_internal_wall_s=(audit or {}).get("wall_s"),
                candidate_lifecycle_summary=({k:deepcopy(v) for k,v in candidate.items() if k not in ("joint_check_rows","middle_inner_actual_timing")} if isinstance(candidate,dict) else None),
                local_recovery=({k:deepcopy(v) for k,v in metric["local_recovery"].items() if k not in
                    ("local_delta_rows","repair_rows","recovery_event_rows","real_joint_conflicts","failed_repair_attempts")} if isinstance(metric.get("local_recovery"),dict) else None),
                future_reservation_metrics=future_compact,future_reservation_review=review,future_details_key=key,
                common_checkpoint_sha256=arm.get("common_checkpoint_sha256"),evaluation_profile=arm.get("evaluation_profile"),
                baseline_provenance=deepcopy(campaign.get("baseline_provenance")),
                audit_charged_child_wall_s=None,formal_charged_child_wall_s=None)
            if label=="new" and ledger is not None:
                jobs=ledger.get("jobs",{}); audit_id=("terminal_audit_" if row["export_lineage"] else "service_audit_")+root+"_"+method
                row.update(audit_job_id=audit_id,audit_charged_child_wall_s=(jobs.get(audit_id) or {}).get("child_wall_s"),
                    formal_charged_child_wall_s=(jobs.get("service_arm_"+root+"_"+method) or {}).get("child_wall_s"))
            rows.append(row)
    return dict(schema="rs_future_commitment_compact_results_v1",provenance=provenance,window=value["window"],rows=rows,
        ledger_summary=({k:deepcopy(v) for k,v in ledger.items() if k!="jobs"} if ledger is not None else None),
        ledger_remaining_child_wall_s=(ledger["cap_child_wall_s"]-ledger["settled_child_wall_s"] if ledger is not None and
            finite(ledger.get("cap_child_wall_s")) and finite(ledger.get("settled_child_wall_s")) else None)),details


def run(summary,output,ledger_path=None):
    summary=Path(summary).resolve(); output=Path(output).resolve()
    require(output.is_relative_to(ARTIFACT_ROOT),"local output must stay under the R4 artifact directory")
    require(not output.is_relative_to(summary.parent) and not summary.is_relative_to(output),"output must be separate from read-only summary inputs")
    require(not output.exists(),"fresh output directory required; no overwrite")
    value,source=read(summary); ledger=None; ledger_source=None
    if ledger_path is not None:
        ledger,ledger_source=read(ledger_path)
        require(not Path(ledger_source["path"]).is_relative_to(output),"output cannot contain the read-only ledger")
        require(not any(j.get("status")=="running" for j in ledger.get("jobs",{}).values()),"ledger still contains active unsettled jobs")
    provenance=dict(source_summary=source,source_schema=value.get("schema"),source_ledger=ledger_source,
        extractor_path=str(Path(__file__).resolve()),extractor_sha256=sha(Path(__file__).read_bytes()),
        extraction_utc=datetime.now(timezone.utc).isoformat(),percentile="nearest-rank ceil(.95*n)")
    compact,details=extract(value,provenance,ledger)
    detailed=dict(schema="rs_future_commitment_full_future_telemetry_v1",provenance=provenance,arms=details)
    detail_bytes=encoded(detailed)
    for row in compact["rows"]:
        row["future_details_reference"]=dict(file="future_reservation_details.json",key=row["future_details_key"],sha256=sha(detail_bytes))
    compact_bytes=encoded(compact); stream=io.StringIO(newline=""); fields=sorted({k for r in compact["rows"] for k in r})
    writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
    for row in compact["rows"]:
        writer.writerow({k:(row[k] if isinstance(row.get(k),str) else json.dumps(row.get(k),ensure_ascii=False,sort_keys=True,allow_nan=False)) for k in fields})
    files={"compact_metrics.json":compact_bytes,"compact_metrics.csv":stream.getvalue().encode("utf8"),"future_reservation_details.json":detail_bytes}
    require(sha(summary.read_bytes())==source["sha256"],"summary changed while being read")
    if ledger_source: require(sha(Path(ledger_source["path"]).read_bytes())==ledger_source["sha256"],"ledger changed while being read")
    output.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        with (output/name).open("xb") as sink: sink.write(data)
    manifest=dict(schema="rs_future_commitment_extraction_manifest_v1",provenance=provenance,
        files=[dict(name=name,bytes=len(data),sha256=sha(data)) for name,data in files.items()])
    with (output/"extraction_manifest.json").open("xb") as sink: sink.write(encoded(manifest))
    return compact


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True); parser.add_argument("--ledger",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args(argv)
    value=run(args.summary,args.output_dir,args.ledger)
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()),schema=value["schema"],rows=len(value["rows"]))))


if __name__=="__main__": main()
