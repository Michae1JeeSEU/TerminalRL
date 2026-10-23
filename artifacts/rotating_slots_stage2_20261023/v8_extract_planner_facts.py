"""Compact, read-only extraction from completed v8 logs and local analysis.

No simulation imports or execution. Repeated planner predictions are observation
statistics, never cumulative physical energy or observed landings.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

METHODS=("energy_slack","rh_cp_sat")
RINGS=("outer","middle","inner")
CANDIDATE_STAGES=("candidate_generation_and_single_vehicle","reference_seed_generation",
    "safe_progress_generation","bounded_delayed_generation","delayed_reference_regeneration","geometry_template_screen")
VALIDATION_STAGES=("trajectory_validation","selected_validation","prefix_validation")


def read(path): return json.loads(Path(path).read_bytes())
def lines(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line]
def path_text(path): return str(path).replace(chr(92),"/").replace("//","/")
def hash_file(path): return sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values=sorted(float(value) for value in values if value is not None and math.isfinite(value))
    if not values: return dict(count=0,mean=None,p95=None,min=None,max=None,sum=0.)
    position=.95*(len(values)-1); lo=int(position); hi=min(lo+1,len(values)-1)
    return dict(count=len(values),mean=math.fsum(values)/len(values),p95=values[lo]+(values[hi]-values[lo])*(position-lo),
        min=values[0],max=values[-1],sum=math.fsum(values))


def strip_raw(value):
    if isinstance(value,dict): return {key:strip_raw(item) for key,item in value.items() if key not in ("raw_s","raw","planner_downstream_decisions")}
    if isinstance(value,list): return [strip_raw(item) for item in value]
    return value


def numeric_delta(current,baseline):
    return {key:dict(new=value,baseline=baseline[key],delta=value-baseline[key],
        percent=None if baseline[key]==0 else 100.*(value-baseline[key])/baseline[key])
        for key,value in current.items() if isinstance(value,(int,float)) and not isinstance(value,bool) and key in baseline and isinstance(baseline[key],(int,float))}


def latency_summary(rows):
    return dict(inner_entry_count=len(rows),inner_to_final_s=stats(row["inner_to_final_s"] for row in rows),
        inner_to_final_censored=sum(row["inner_to_final_censored"] for row in rows),
        inner_to_touchdown_s=stats(row["inner_to_touchdown_s"] for row in rows),
        inner_to_touchdown_censored=sum(row["inner_to_touchdown_censored"] for row in rows),
        note="Statistics use uncensored durations only; censored counts remain separate, no survival extrapolation.")


def compact_physical(analysis,baseline,full_path):
    result=dict(schema="stage2_v8_medium_final_summary_v1",root=path_text(analysis["root"]),time_s=analysis["time_s"],
        planning_profile=analysis["planning_profile"],full_local_analysis_path=path_text(full_path),
        full_analysis_note="Large local analysis retained separately; original per-dispatch records remain in the completed root.",
        human_review="HUMAN_PENDING",methods={})
    for method,current in analysis["methods"].items():
        old=baseline["methods"][method]
        delta=dict(metrics=numeric_delta(current["metrics"],old["metrics"]),
            holding_seconds_by_ring=numeric_delta(current["holding"]["seconds_by_ring"],old["holding"]["seconds_by_ring"]),
            holding_kwh_by_ring=numeric_delta(current["holding"]["kwh_by_ring"],old["holding"]["kwh_by_ring"]),
            transfers=numeric_delta(current["transfer_completed_by_route"],old["transfer_completed_by_route"]),
            terminal_queues={ring:numeric_delta(current["queues"]["terminal"][ring],old["queues"]["terminal"][ring]) for ring in RINGS},
            every_60s_queues=[dict(time_s=new_row["time_s"],rings={ring:numeric_delta(new_row["rings"][ring],old_row["rings"][ring]) for ring in RINGS})
                for new_row,old_row in zip(current["queues"]["every_60s"],old["queues"]["every_60s"])])
        result["methods"][method]=dict(metrics=current["metrics"],holding=current["holding"],checks=current["checks"],
            queues=current["queues"],transfer_completed_by_route=current["transfer_completed_by_route"],
            inner_latency_summary=latency_summary(current["inner_latencies"]),timing=strip_raw(current["timing"]),
            planner_timing_by_stage=current["planner_timing_by_stage"],new_minus_v7_baseline=delta)
    return result


def planner_method(root,method,physical):
    directory=root/method
    decisions=lines(directory/"decisions.jsonl"); dispatch=lines(directory/"dispatch_timings.jsonl")
    events=lines(directory/"replay_0600/events.jsonl"); trajectories=read(directory/"replay_0600/trajectories.json")
    dispatch_by_time={row["time"]:row for row in dispatch}
    stage_keys=sorted({key for row in decisions for key in row.get("timings",{})})
    timings={key:stats(row.get("timings",{}).get(key,(0.,0.))[0] for row in decisions if key in row.get("timings",{})) for key in stage_keys}
    def stage_sum(row,keys): return math.fsum(row.get("timings",{}).get(key,(0.,0.))[0] for key in keys)
    timing=dict(total_normal_wall_s=stats(row["normal_wall_s"] for row in dispatch),
        full_dispatch_wall_s=stats(row["dispatch_wall_s"] for row in dispatch),
        algorithm_record_wall_s=stats(row["wall_s"] for row in decisions),
        candidate_reported_exclusive_stage_sum_s=stats(stage_sum(row,CANDIDATE_STAGES) for row in decisions),
        validation_reported_exclusive_stage_sum_s=stats(stage_sum(row,VALIDATION_STAGES) for row in decisions),
        solver_wall_s=stats(row.get("solver_wall_s",0.) for row in decisions),stages=timings,
        definition="mean/p95/count across real dispatches; p95 linear interpolation; grouped candidate/validation sum only recorded exclusive stage counters, inactive/unrecorded keys contribute zero.")
    choices=[choice for row in decisions for choice in row.get("aircraft_choices",())]
    coverage={}
    for ring in RINGS:
        rows=[choice for choice in choices if choice.get("ring")==ring]
        coverage[ring]=dict(aircraft_dispatch_observations=len(rows),with_full_chain=sum(choice.get("full_normal_chain_count",0)>0 for choice in rows),
            without_full_chain=sum(choice.get("full_normal_chain_count",0)==0 for choice in rows),
            full_chain_columns_reported=sum(choice.get("full_normal_chain_count",0) for choice in rows),
            candidate_insufficiency_reasons=dict(Counter(reason for choice in rows for reason in choice.get("candidate_insufficiency_reasons",()))),
            only_keep_wait=sum(bool(choice.get("only_keep_wait")) for choice in rows))
    solver_records=[record for row in decisions for record in row.get("solver",())]
    cuts=Counter()
    for row in decisions:
        grouped=defaultdict(list)
        for record in row.get("solver",()): grouped[record.get("uniform_refinement_restart",0)].append(record)
        for records in grouped.values():
            for key in ("cumulative_pair_cuts","cumulative_single_cuts","cumulative_prefix_nogoods"):
                cuts[key]+=max((record.get(key,0) for record in records),default=0)
    starts=defaultdict(list); starts_by_hash={}; commits=[]
    for event in events:
        if event["type"]=="maneuver_started":
            starts[event["aircraft_id"]].append(event); starts_by_hash.setdefault(event["details"]["trajectory_hash"],event)
        elif event["type"] in ("plan_committed","soft_plan_reserved"): commits.append(event)
    delayed=[]; selected_counts=Counter(); energy_summaries=[]; delay_counts=[]
    for row in decisions:
        by_aid={choice["aircraft_id"]:choice for choice in row.get("aircraft_choices",())}
        moving_delayed=0
        energy_rows=[]
        for aid,data in row.get("downstream",{}).items():
            choice=by_aid.get(aid)
            if not choice: continue
            moving=choice.get("selected_kind")=="MOVE"
            selected_counts["selected_kind_"+str(choice.get("selected_kind"))]+=1
            if moving: selected_counts["MOVE_"+data.get("candidate_timing","UNKNOWN")]+=1
            energy_rows.append(data)
            if not moving or data.get("candidate_timing")!="DELAYED": continue
            moving_delayed+=1
            action=data.get("decision_explanation",{}).get("actual_action","UNKNOWN")
            selected_counts["DELAYED_action_"+action]+=1
            target=choice.get("first_target")
            current_commits=[event for event in commits if event["aircraft_id"]==aid and abs(event["time"]-row["time"])<=1e-8
                and trajectories[event["details"]["trajectory_hash"]]["kind"]!="FINAL_EMERGENCY"
                and target==[trajectories[event["details"]["trajectory_hash"]].get(key) for key in ("target_ring","target_slot","pad_id")]]
            exact_started=[event for event in current_commits if event["details"]["trajectory_hash"] in starts_by_hash]
            same_target=[]
            for event in starts[aid]:
                if event["time"]<row["time"]-1e-8: continue
                plan=trajectories[event["details"]["trajectory_hash"]]
                if target==[plan.get("target_ring"),plan.get("target_slot"),plan.get("pad_id")]: same_target.append(event)
            delayed.append(dict(dispatch_time_s=row["time"],aircraft_id=aid,source_ring=choice.get("ring"),first_target=target,
                actual_action=action,predicted_touchdown_s=data.get("estimated_touchdown"),reference_status=data.get("reference_status"),
                committed_hash_at_same_dispatch=bool(current_commits),same_dispatch_committed_hash_started=bool(exact_started),
                later_same_target_start_proxy=bool(same_target),first_same_target_start_s=None if not same_target else same_target[0]["time"]))
        delay_counts.append((moving_delayed,row["time"]))
        if energy_rows:
            energy_summaries.append(dict(time_s=row["time"],
                energy={key:math.fsum(data.get(key,0.) for data in energy_rows) for key in ("window_j","tail_j","tail_motion_j","tail_wait_j","total_j")},
                wait={ring:math.fsum(data.get("estimated_wait_seconds_by_ring",{}).get(ring,0.) for data in energy_rows) for ring in RINGS},
                touchdown=stats(data.get("estimated_touchdown") for data in energy_rows)))
    actual_delayed=[]
    seen=set()
    for event in commits:
        key=event["details"]["trajectory_hash"]
        if key in seen: continue
        seen.add(key);plan=trajectories[key]
        if plan["kind"]=="FINAL_EMERGENCY" or plan["start_time"]<=event["time"]+1e-8: continue
        started=starts_by_hash.get(key)
        actual_delayed.append(dict(aircraft_id=event["aircraft_id"],committed_at_s=event["time"],planned_start_s=plan["start_time"],
            started=started is not None,actual_start_s=None if started is None else started["time"],
            start_matches=started is not None and abs(started["time"]-plan["start_time"])<=1e-6,
            censored_future_start=plan["start_time"]>600.))
    queue_rows=physical["queues"]["every_60s"]
    peak_queue=max(queue_rows,key=lambda row:(sum(value["lower_bound_shortage"] for value in row["rings"].values()),-row["time_s"]))
    most_delay=max(delay_counts,key=lambda pair:(pair[0],-pair[1]))
    reps=[dict(reason="maximum_selected_delayed_moves",time_s=most_delay[1],count=most_delay[0]),
        dict(reason="maximum_60s_lower_bound_shortage",time_s=peak_queue["time_s"],count=sum(value["lower_bound_shortage"] for value in peak_queue["rings"].values())),
        dict(reason="terminal",time_s=600.)]
    vector_rows=[row for row in decisions if row.get("hint_objective") is not None and row.get("final_objective") is not None]
    return dict(dispatch_count=len(dispatch),normal_record_count=len(decisions),timing=timing,candidate_coverage=coverage,
        catalog_statistics=dict(columns_per_dispatch=stats(row.get("columns",0) for row in decisions),
            incomplete_aircraft_per_dispatch=stats(len(row.get("catalog_incomplete",())) for row in decisions)),
        cp=dict(entered_count=sum(bool(row.get("solver_entered")) for row in decisions),hint_comparable_calls=len(vector_rows),
            strict_improvement_reported_true=sum(row.get("strict_improvement_over_hint") is True for row in decisions),
            final_vector_less_than_hint=sum(tuple(row["final_objective"])<tuple(row["hint_objective"]) for row in vector_rows),
            rounds=stats(row.get("refinement_rounds",0) for row in decisions),phase_record_count=len(solver_records),
            phase_status_counts=dict(Counter(record.get("status") for record in solver_records)),
            cuts=dict(cuts),cuts_definition="Sum per-dispatch/per-uniform-restart maxima of cumulative cut counters; do not sum cumulative phase values."),
        fallback=dict(count=sum(bool(row.get("fallback")) for row in decisions),status_counts=dict(Counter(row["status"] for row in decisions)),
            skip_reasons=dict(Counter(str(row.get("skip_reason")) for row in decisions)),
            truncations=dict(Counter(item for row in decisions for item in row.get("truncations",())))),
        cache=dict(reference_hits_per_call_sum=sum(row.get("cache_hits",0) for row in decisions),
            move_energy_cumulative_last=decisions[-1].get("move_energy_cache"),
            validation_per_call_sums={key:sum(row.get("validation_cache",{}).get(key,0) for row in decisions) for key in {key for row in decisions for key in row.get("validation_cache",{})}},
            note="Move energy cache is cumulative: last recorded counters retained, not summed across callbacks."),
        selected_observations=dict(counts=dict(selected_counts),note="Repeated aircraft-dispatch predictions; not distinct aircraft, reservations, or actual landings."),
        predicted_snapshot_statistics=dict(energy_j={key:stats(row["energy"][key] for row in energy_summaries) for key in ("window_j","tail_j","tail_motion_j","tail_wait_j","total_j")},
            estimated_wait_seconds_by_ring={ring:stats(row["wait"][ring] for row in energy_summaries) for ring in RINGS},
            predicted_touchdown_absolute_s=stats(data.get("estimated_touchdown") for row in decisions for data in row.get("downstream",{}).values()),
            predicted_touchdown_offset_from_dispatch_s=stats(data["estimated_touchdown"]-row["time"] for row in decisions
                for data in row.get("downstream",{}).values() if data.get("estimated_touchdown") is not None),
            reference_status_counts=dict(Counter(data.get("reference_status","UNKNOWN") for row in decisions for data in row.get("downstream",{}).values())),
            note="Snapshot sums then distribution over dispatches; these are predictions with KNOWN/ESTIMATED/UNKNOWN, never integrate/sum them into physical terminal energy."),
        delayed_selection_fulfillment=dict(selected_delayed_move_observations=len(delayed),
            same_dispatch_committed_hash_started=sum(row["same_dispatch_committed_hash_started"] for row in delayed),
            selected_delayed_unreserved_matching_commit_absent=sum(not row["committed_hash_at_same_dispatch"] for row in delayed),
            selected_delayed_definition="Repeated selected candidate_timing=DELAYED observations. Only matching same-dispatch aircraft/first-target commit hashes subsequently seen in actual maneuver_started support same_dispatch_committed_hash_started; this does not establish that every selected DELAYED prediction executed.",
            later_same_target_proxy_count=sum(row["later_same_target_start_proxy"] for row in delayed),
            exact_identity_unavailable_for_unreserved_predictions=True,
            note="Unreserved future selected signatures are absent from logs. Later same-target starts are a proxy only, not proof that the predicted plan was executed.",
            actual_delayed_committed_unique_plans=len(actual_delayed),actual_delayed_committed_started=sum(row["started"] for row in actual_delayed),
            actual_delayed_committed_definition="All unique normal plan hashes with planned start_time later than their commit event time. May include different normal candidate types; selected candidate_timing=DELAYED membership is not required. This is not a fulfillment count for selected-DELAYED observations.",
            actual_delayed_starts_match_planned=sum(row["start_matches"] for row in actual_delayed),
            actual_delayed_censored_future_start=sum(row["censored_future_start"] for row in actual_delayed),
            examples=delayed[:3],actual_delayed_committed_examples=actual_delayed[:3]),
        representative_replay_times=reps,
        source_files={name:dict(path=path_text(directory/name),sha256=hash_file(directory/name)) for name in ("decisions.jsonl","dispatch_timings.jsonl","summary_0600.json")})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--analysis",required=True,type=Path)
    parser.add_argument("--baseline",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--summary-output",required=True,type=Path)
    args=parser.parse_args()
    for output in (args.output,args.summary_output):
        if output.resolve().is_relative_to(args.root.resolve()): raise ValueError("output must be outside read-only run root")
    analysis=read(args.analysis);baseline=read(args.baseline)
    compact=compact_physical(analysis,baseline,args.analysis.resolve())
    result=dict(schema="stage2_v8_planner_final_compact_v1",root=path_text(args.root.resolve()),time_s=600.,human_review="HUMAN_PENDING",
        methods={method:planner_method(args.root,method,analysis["methods"][method]) for method in METHODS},
        analysis_path=path_text(args.analysis.resolve()),physical_summary_path=path_text(args.summary_output.resolve()),
        interpretation="Mechanical observed statistics and prediction diagnostics only; no causal attribution or global optimality claim.")
    for output,value in ((args.output,result),(args.summary_output,compact)):
        output.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(dict(planner_output=path_text(args.output),summary_output=path_text(args.summary_output),
        counts={method:dict(dispatch=result["methods"][method]["dispatch_count"],delayed_selected=result["methods"][method]["delayed_selection_fulfillment"]["selected_delayed_move_observations"]) for method in METHODS}),indent=2))


if __name__=="__main__": main()
