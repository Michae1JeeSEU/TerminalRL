"""Read-only service-opportunity before/after extraction with physical idle.

Reuse the saved A4 summarizer for ordinary lifecycle and timing measures. Missing
certificate or physical evidence remains null, never evidence of zero waste.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
ROOTS = ("s3r23","s3r24")
METHODS = ("FULL_H","RULE_LOCAL_K8")
AIRBORNE = {"HOLD_OUTER","HOLD_MIDDLE","HOLD_INNER","INBOUND","TRANSFER_O_M","TRANSFER_M_I",
            "FINAL_NORMAL","FINAL_EMERGENCY","VERTICAL_DESCENT"}


def load(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def finite(value): return type(value) in (float,int) and math.isfinite(value)
def jsonl(path):
    with Path(path).open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def existing_summary():
    path=PROJECT/"artifacts/rs_select_quality_a4_20261030/summarize_common_comparison.py"
    spec=importlib.util.spec_from_file_location("_service_existing_summary",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def merged(intervals, lo, hi):
    result=[]
    for start,end in sorted((max(lo,a),min(hi,b)) for a,b in intervals if a<hi and b>lo):
        if end<=start: continue
        if result and start<=result[-1][1]: result[-1]=(result[-1][0],max(end,result[-1][1]))
        else: result.append((start,end))
    return result


def complement(intervals,lo,hi):
    rows=[]; cursor=lo
    for start,end in merged(intervals,lo,hi):
        if start>cursor: rows.append((cursor,start))
        cursor=max(cursor,end)
    if cursor<hi: rows.append((cursor,hi))
    return rows


def intersection(left,right):
    return [(max(a,c),min(b,d)) for a,b in left for c,d in right if max(a,c)<min(b,d)]


def measure(intervals): return math.fsum(b-a for a,b in intervals)


def demand_intervals(frames,lo,hi):
    """Observed admitted airborne demand only; pending upstream is excluded."""
    if not frames or frames[0].get("time",hi+1)>lo or frames[-1].get("time",lo-1)<hi:
        return None
    result=[]
    for current,nxt in zip(frames,frames[1:]):
        if any(a.get("status") in AIRBORNE for a in current.get("aircraft",())):
            result.append((current["time"],nxt["time"]))
    return merged(result,lo,hi)


def middle_inner(events,trajectories,lo,hi):
    """Count actual starts and observed completions, never aircraft ID names."""
    starts=[]; completed=[]; missing=[]
    for index,event in enumerate(events):
        if event.get("type")!="maneuver_started": continue
        plan=trajectories.get((event.get("details") or {}).get("trajectory_hash"))
        if plan is None:
            if lo<=event.get("time",lo-1)<=hi: missing.append(index)
            continue
        if plan.get("source_ring")!="middle" or plan.get("target_ring")!="inner": continue
        row=dict(aircraft_id=event["aircraft_id"],trajectory_hash=plan.get("trajectory_hash"),
                 start=event["time"],end=plan["end_time"])
        if lo<=event["time"]<=hi: starts.append(row)
        matches=[e for e in events if e.get("type")=="transfer_completed" and
            e.get("aircraft_id")==event["aircraft_id"] and
            abs(e.get("time",-1)-plan["end_time"])<1.e-7]
        if matches and lo<=plan["end_time"]<=hi: completed.append(row)
    return dict(actual_maneuver_start_count=len(starts),completed_effective_supply_count=len(completed),
                starts=starts,completed=completed,missing_started_trajectory_indices=missing)


def physical_idle(events,trajectories,frames,pad_count,service_seconds,*,lo=20.,hi=320.):
    """Executed FINAL descent (13.5s) plus service, and observed initial service.

    Future soft reservations are excluded from physical occupancy. Committed
    pad reservations confirm initial service/approach and retain fixture blocks.
    """
    coverage=[]
    if not frames or frames[0].get("time",hi+1)>lo or frames[-1].get("time",lo-1)<hi:
        return dict(pad_seconds_idle=None,longest_all_pad_idle_s=None,
                    coverage_issues=["states_do_not_cover_window"],idle_by_pad=None)
    occupied={pad:[] for pad in range(pad_count)}
    by_aircraft={}
    for frame in frames:
        for a in frame.get("aircraft",()): by_aircraft[a["id"]]=a
    starts={(e.get("details") or {}).get("trajectory_hash"):e for e in events if e.get("type")=="maneuver_started"}
    for hash_,event in starts.items():
        plan=trajectories.get(hash_)
        if plan is None:
            if lo<=event.get("time",-1)<=hi: coverage.append("started_trajectory_missing:"+str(hash_))
            continue
        pad=plan.get("pad_id")
        if pad is None: continue
        aircraft=by_aircraft.get(plan["aircraft_id"],{})
        duration=service_seconds.get(aircraft.get("service_class"))
        if duration is None: coverage.append("service_duration_missing:"+plan["aircraft_id"]); continue
        end=plan["end_time"]+duration
        # Failure isolation physically ends managed occupancy at its event.
        failure=[e["time"] for e in events if e.get("aircraft_id")==plan["aircraft_id"] and
                 e.get("type") in ("aircraft_energy_failure","failure_isolated","invalid_run_stopped") and e["time"]>=event["time"]]
        if failure: end=min(end,min(failure))
        occupied[pad].append((plan["end_time"]-13.5,end))
    initial=max((f for f in frames if f["time"]<=lo),key=lambda f:f["time"],default=None)
    for a in (initial or {}).get("aircraft",()):
        if a.get("status")=="LANDED_SERVICE" and a.get("pad_id") is not None:
            if finite(a.get("service_end")): occupied[a["pad_id"]].append((lo,a["service_end"]))
            else: coverage.append("initial_service_end_missing:"+a["id"])
    # Existing fixture or pre-t0 committed approach can be absent from start
    # events. Accept only commitments whose plan is actually active/started.
    active_hashes=set(starts)
    active_hashes.update(a.get("trajectory_hash") for a in (initial or {}).get("aircraft",())
                         if str(a.get("status","")).startswith("FINAL"))
    for reservation in (initial or {}).get("pads",()):
        if reservation.get("committed") and (reservation.get("kind")=="service" or
                reservation.get("plan_hash") in active_hashes):
            if finite(reservation.get("end")):
                occupied[reservation["resource"]].append((reservation["start"],reservation["end"]))
            else: coverage.append("initial_committed_pad_end_unknown")
    if coverage:
        return dict(pad_seconds_idle=None,longest_all_pad_idle_s=None,coverage_issues=coverage,idle_by_pad=None)
    occupied={pad:merged(rows,lo,hi) for pad,rows in occupied.items()}
    idle={pad:complement(rows,lo,hi) for pad,rows in occupied.items()}
    all_idle=complement([row for rows in occupied.values() for row in rows],lo,hi)
    demand=demand_intervals(frames,lo,hi)
    return dict(pad_seconds_idle=math.fsum(measure(rows) for rows in idle.values()),
        longest_all_pad_idle_s=max((b-a for a,b in all_idle),default=0.),all_pad_idle_intervals=all_idle,
        idle_by_pad=idle,occupied_by_pad=occupied,coverage_issues=[],
        no_demand_pad_seconds_idle=(math.fsum(measure(intersection(rows,complement(demand,lo,hi)))
            for rows in idle.values()) if demand is not None else None),
        classification="PHYSICAL_IDLE_ONLY; avoidability_requires_joint_witness")


def supply_gap(supplies,demand,lo=20.,hi=320.):
    """Largest demand-present gap between completed M->I supplies; edges censored."""
    if demand is None: return dict(longest_no_inner_supply_s=None,intervals=None)
    times=sorted({r["end"] for r in supplies if lo<=r["end"]<=hi})
    cuts=[lo,*times,hi]; rows=[]
    for a,b in zip(cuts,cuts[1:]):
        for start,end in intersection([(a,b)],demand):
            rows.append(dict(start=start,end=end,seconds=end-start,
                             left_censored=a==lo,right_censored=b==hi))
    return dict(longest_no_inner_supply_s=max((r["seconds"] for r in rows),default=0.),intervals=rows,
                definition="completed_middle_to_inner_supply; admitted_airborne_demand")


def inner_stock_empty(frames,lo=20.,hi=320.):
    """Observed no holding INNER stock while admitted inbound/outer/middle demand exists."""
    if not frames or frames[0].get("time",hi+1)>lo or frames[-1].get("time",lo-1)<hi:
        return dict(longest_inner_stock_empty_with_upstream_demand_s=None,intervals=None)
    empty=[]
    for current,nxt in zip(frames,frames[1:]):
        aircraft=current.get("aircraft",())
        stock=any(a.get("status")=="HOLD_INNER" for a in aircraft)
        demand=any(a.get("status") in AIRBORNE and (a.get("ring") in ("outer","middle") or
                   a.get("status") in ("INBOUND","TRANSFER_O_M","TRANSFER_M_I")) for a in aircraft)
        if demand and not stock: empty.append((current["time"],nxt["time"]))
    intervals=merged(empty,lo,hi)
    return dict(longest_inner_stock_empty_with_upstream_demand_s=max((b-a for a,b in intervals),default=0.),
        intervals=[dict(start=a,end=b,seconds=b-a,left_censored=a==lo,right_censored=b==hi) for a,b in intervals],
        definition="no_HOLD_INNER_aircraft; admitted_inbound_outer_middle_airborne_demand; pending_unadmitted_requests_excluded")


def witnessed_idle(callbacks,events,idle,demand,lo=20.,hi=320.,*,event_log_complete=False):
    """Union only exported, physically validated joint FINAL witness intervals."""
    reasons=Counter(); intervals={}; accepted=0; total=0
    for cb in callbacks:
        source=(cb.get("planning_timing") or {}).get("snapshot_time")
        certificates=cb.get("service_opportunity_certificates")
        if certificates is None: reasons["certificate_export_missing"]+=1; continue
        for cert in certificates:
            total+=1
            if cert.get("joint_prefix_certified") is not True:
                reasons["joint_certificate_missing"]+=1; continue
            bindings=("source_snapshot_id","projection_protocol","effective_snapshot_id")
            if any(not isinstance(cert.get(key),str) or not cert[key] for key in bindings):
                reasons["source_projection_effective_binding_missing"]+=1; continue
            if not event_log_complete:
                reasons["complete_event_log_not_established"]+=1; continue
            if not finite(cert.get("remaining_ready_margin_s")) or cert["remaining_ready_margin_s"]<=0:
                reasons["delivery_margin_missing_or_nonpositive"]+=1; continue
            effective=cert.get("effective_time")
            if not finite(source) or not finite(effective): reasons["source_or_effective_time_missing"]+=1; continue
            if cert.get("source_snapshot_time",source)!=source:
                reasons["source_time_binding_mismatch"]+=1; continue
            # Event names are deliberately conservative: only explicitly
            # observational bookkeeping can leave a witness valid. Every other
            # physical/admission/commitment/emergency event ends the interval.
            bookkeeping={"normal_task_started","normal_task_finished","normal_coalesced_restart",
                         "normal_planning_started","normal_planning_finished"}
            next_change=min((e["time"] for e in events if e.get("type") not in bookkeeping and e["time"]>=source),default=hi)
            for final in cert.get("finals",()):
                required=("pad_id","flight_start","descent_start","service_end")
                if any(not finite(final.get(k)) for k in required): reasons["final_interval_missing"]+=1; continue
                if final["flight_start"]<effective: reasons["final_precedes_delivery"]+=1; continue
                a=max(lo,effective,final["descent_start"]); b=min(hi,next_change,final["service_end"])
                if a<b:
                    intervals.setdefault(final["pad_id"],[]).append((a,b)); accepted+=1
                else: reasons["validity_ends_before_final_occupancy"]+=1
    if idle.get("idle_by_pad") is None or demand is None: reasons["physical_idle_or_demand_evidence_missing"]+=1
    available=accepted>0 and idle.get("idle_by_pad") is not None and demand is not None
    witnessed={}
    if available:
        for pad,windows in intervals.items():
            free=idle["idle_by_pad"].get(pad,[])
            witnessed[pad]=merged(intersection(intersection(merged(windows,lo,hi),free),demand),lo,hi)
    return dict(witnessed_avoidable_idle_pad_seconds=(math.fsum(measure(r) for r in witnessed.values()) if available else None),
        witnessed_intervals_by_pad=witnessed if available else None,certificate_rows=total,
        accepted_window_count=accepted,coverage_reasons=dict(reasons),
        unclassified_idle_pad_seconds=(idle["pad_seconds_idle"]-(idle.get("no_demand_pad_seconds_idle") or 0.)-math.fsum(measure(r) for r in witnessed.values())
            if idle.get("pad_seconds_idle") is not None else None),
        semantics="conservative_lower_bound; validity_ends_at_next_state_change; no_witness_is_unknown",
        assumption="same_projected_committed_and_hold_evolution; not_global_proof_of_all_feasible_paths")


def extract_metrics(input_root,root,method,ordinary):
    directory=Path(input_root)/"arms"/root/method/"evaluation"/root/method
    paths={name:directory/relative for name,relative in dict(events="replay/events.jsonl",frames="replay/states.jsonl",
        trajectories="replay/trajectories.json",callbacks="method_scheduler_records.json",config="replay/config.yaml").items()}
    missing=[str(path) for path in paths.values() if not path.is_file()]
    increment=ordinary.get("increment") or {}; delta=increment.get("cumulative_delta") or {}
    inherited=increment.get("inherited_final_touchdown_ids")
    treatments=increment.get("treatment_final_touchdown_ids")
    result=dict(root=root,method=method,missing_inputs=missing,
        normal_touchdown_delta=delta.get("L_normal"),inherited_final_touchdown_count=len(inherited) if inherited is not None else None,
        treatment_final_touchdown_count=len(treatments) if treatments is not None else None,
        treatment_normal_touchdown_count=None,
        energy_j=delta.get("energy_j"),emergencies=delta.get("L_emergency"),failures=delta.get("failed_before_landing"),
        holding_time_by_ring=ordinary.get("holding_time_by_ring"),holding_energy_by_ring=ordinary.get("holding_energy_by_ring"))
    if missing: return dict(result,physical_idle=None,middle_inner_supply=None,witnessed_idle=None,
        supply_gap=None,inner_stock_empty=None,replay_input_hash_checks=None,event_log_complete=None,
        first_certificate_times=None,forecast_service_terms=None,empty_reason_counts=None)
    import yaml
    config=yaml.safe_load(paths["config"].read_text(encoding="utf8"))
    services={k:v["duration_s"] for k,v in config["pads"]["service_classes"].items()}
    events=jsonl(paths["events"]); frames=jsonl(paths["frames"])
    trajectories=load(paths["trajectories"]); callbacks=load(paths["callbacks"])
    index_path=directory/"replay/replay.json"
    index=load(index_path) if index_path.is_file() else {}
    replay_checks={name:((index.get("files") or {}).get(path.name)==sha256(path.read_bytes()).hexdigest())
        for name,path in paths.items() if name in ("events","frames","trajectories","config")}
    event_log_complete=all(replay_checks.values()) and all((ordinary.get("checks") or {}).values()) and bool(ordinary.get("checks"))
    physical=physical_idle(events,trajectories,frames,config["pads"]["count"],services)
    supplies=middle_inner(events,trajectories,20.,320.)
    demand=demand_intervals(frames,20.,320.)
    normal=[e["aircraft_id"] for e in events if e.get("type")=="touchdown" and 20.<e.get("time",-1)<=320.
            and (e.get("details") or {}).get("emergency") is False]
    result["treatment_normal_touchdown_count"]=len([aid for aid in normal if aid not in set(inherited or ())]) if inherited is not None else None
    result.update(physical_idle=physical,middle_inner_supply=supplies,
        replay_input_hash_checks=replay_checks,event_log_complete=event_log_complete,
        supply_gap=supply_gap(supplies["completed"],demand),
        inner_stock_empty=inner_stock_empty(frames),
        witnessed_idle=witnessed_idle(callbacks,events,physical,demand,event_log_complete=event_log_complete),
        forecast_service_terms=[dict(snapshot_time=(cb.get("planning_timing") or {}).get("snapshot_time"),
            evaluation_profile=cb.get("evaluation_profile"),objective_id=cb.get("objective_id"),objective=cb.get("objective"),
            G_service_forecast_s=(cb["objective"][3] if cb.get("evaluation_profile")=="service_opportunity_v1" and
                isinstance(cb.get("objective"),(list,tuple)) and len(cb["objective"])==6 else None),
            downstream_service={aid:{key:row.get(key) for key in ("service_baseline_start","service_start",
                "service_opportunity_delay_s","service_opportunity_eligible","service_opportunity_mask","estimated_touchdown")}
                for aid,row in (cb.get("downstream") or {}).items()}) for cb in callbacks],
        empty_reason_counts=dict(Counter(cb.get("empty_reason") for cb in callbacks if cb.get("empty_reason") is not None)),
        missing_six_objective_callbacks=sum(not isinstance(cb.get("objective"),(list,tuple)) or len(cb["objective"])!=6
            for cb in callbacks if cb.get("evaluation_profile")=="service_opportunity_v1"),
        first_certificate_times=[dict(snapshot_time=(cb.get("planning_timing") or {}).get("snapshot_time"),
            first_nonempty_certified=(cb.get("search_handoff") or {}).get("first_nonempty_certified") or
                (cb.get("selective_replanning") or {}).get("first_nonempty_certified")) for cb in callbacks])
    return result


def collect(old_root,new_root):
    summarizer=existing_summary(); campaigns={}
    for label,path in (("old",old_root),("new",new_root)):
        ordinary,timeline,timings=summarizer.collect(path)
        metrics=[extract_metrics(path,a["root"],a["method"],a) for a in ordinary["arms"]]
        campaigns[label]=dict(input_root=str(Path(path).resolve()),ordinary=ordinary,metrics=metrics)
    common_checks={root:len({arm.get("common_checkpoint_sha256") for campaign in campaigns.values()
        for arm in campaign["ordinary"]["arms"] if arm["root"]==root})==1 and all(
        arm.get("common_checkpoint_sha256") is not None for campaign in campaigns.values()
        for arm in campaign["ordinary"]["arms"] if arm["root"]==root) for root in ROOTS}
    return dict(schema="service_opportunity_comparison_v1",window=[20.,320.],campaigns=campaigns,
        same_common_checkpoint_across_campaigns=common_checks,
        semantics=dict(forecast_service_is_not_realized_idle=True,missing_values="null; never zero-imputed",
                       physical_idle_is_not_automatically_avoidable=True))


def write_outputs(output,report):
    output=Path(output).resolve()
    if not output.is_relative_to(PROJECT/"artifacts"):
        raise ValueError("fresh local report directory under project artifacts required")
    for campaign in report["campaigns"].values():
        source=Path(campaign["input_root"]).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output): raise ValueError("report must be separate from inputs")
    output.mkdir(parents=True,exist_ok=False)
    (output/"summary.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False),encoding="utf8")
    rows=[dict(campaign=label,**metric) for label,c in report["campaigns"].items() for metric in c["metrics"]]
    columns=sorted({key for row in rows for key in row})
    with (output/"comparison.csv").open("x",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(row.get(k),sort_keys=True) if isinstance(row.get(k),(dict,list)) or row.get(k) is None else row[k] for k in columns})
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root",type=Path,required=True); parser.add_argument("--new-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv); report=collect(args.old_root,args.new_root)
    output=write_outputs(args.output_dir,report)
    print(json.dumps(dict(output=str(output),schema=report["schema"])))
    return report


if __name__=="__main__": main()
