"""Compare saved v2/v3 development evidence; never launch a physical run or solve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.stage3 import report_selective_reoptimization_repair as saved
from scripts.stage3 import report_selective_replay_decisions as decisions


def named_values(value, names):
    """Retain exact saved fields, including their location and missing values."""
    rows=[]
    def walk(item,path):
        if isinstance(item,dict):
            for key,child in item.items():
                location=f"{path}.{key}" if path else key
                if key in names: rows.append(dict(path=location,value=child))
                walk(child,location)
        elif isinstance(item,list):
            for index,child in enumerate(item): walk(child,f"{path}[{index}]")
    walk(value,"")
    return rows


def conflict_facts(payload, uid_by_id):
    evidence=named_values(payload,{"pair_conflict_evidence"})
    rows=[]
    for collection in evidence:
        for item in collection["value"] or []:
            details=item.get("details",{})
            pair=details.get("pair_evidence",{})
            participants=pair.get("participants")
            rows.append(dict(source_path=collection["path"],raw=item,
                participants=None if participants is None else [dict(participant,
                    uid=uid_by_id.get(str(participant.get("aircraft_id")))) for participant in participants],
                check_interval=pair.get("check_interval"),
                witness={key:details.get(key) for key in
                    ("witness_time","distance_m","upper_bound_m","lower_bound_m")},
                missing_value_semantics="NULL_IS_UNOBSERVED_NO_MINIMUM_DISTANCE_INFERRED"))
    return dict(rows=rows,recorded_collections=len(evidence),
        dropped=named_values(payload,{"pair_conflict_evidence_dropped"}),
        status="OBSERVED" if evidence else "UNAVAILABLE_NOT_RECORDED")


def p2_facts(events, states, actions, endpoint):
    """Separate approach starts, saved physical pad occupancy, and touch downs."""
    by_id={str(a["id"]):a for frame in states for a in frame.get("aircraft",[])}
    action_p2=[a for a in actions if a.get("pad_id")==2 and a.get("start_event_index") is not None
               and saved.finite(endpoint) and a["start_event_time"]<=endpoint]
    started_hashes={a.get("trajectory_hash") for a in action_p2}
    occupancy={}
    for frame in states:
        if not saved.finite(endpoint) or frame["time"]>endpoint: continue
        aircraft={str(a["id"]):a for a in frame.get("aircraft",[])}
        for row in frame.get("pads",[]):
            if row.get("resource")!=2 or not row.get("committed") or row.get("start",float("inf"))>frame["time"]:
                continue
            aid=str(row.get("aircraft_id")); actual=aircraft.get(aid,{})
            if actual.get("status")=="ENERGY_MISSION_FAILED": continue
            if row.get("plan_hash") not in started_hashes and row.get("kind")!="service": continue
            key=(aid,row.get("plan_hash"),row.get("start"),row.get("kind"))
            occupancy.setdefault(key,dict(aircraft_id=aid,trajectory_hash=row.get("plan_hash"),
                time=row.get("start"),first_saved_frame_time=frame["time"],raw_reservation=row))
    rows=[]
    for index,event in enumerate(events):
        if event.get("type")!="pad_service_completed": continue
        aid=str(event.get("aircraft_id"))
        if by_id.get(aid,{}).get("pad_id")!=2: continue
        end=event["time"]
        next_start=min((dict(time=a["start_event_time"],aircraft_id=a["aircraft_id"],
            trajectory_hash=a.get("trajectory_hash")) for a in action_p2 if a["start_event_time"]>=end),
            key=lambda x:x["time"],default=None)
        next_occupancy=min((row for row in occupancy.values() if row["time"]>=end),key=lambda x:x["time"],default=None)
        next_touchdown=min((dict(time=a["completion_time"],aircraft_id=a["aircraft_id"],
            trajectory_hash=a.get("trajectory_hash")) for a in action_p2
            if a.get("completion_type")=="touchdown" and a.get("completion_time") is not None
            and end<=a["completion_time"]<=endpoint),key=lambda x:x["time"],default=None)
        row=dict(service_completed_time=end,service_aircraft_id=aid,event_index=index)
        row["approaches_already_underway_at_service_end"]=[dict(aircraft_id=a["aircraft_id"],
            trajectory_hash=a.get("trajectory_hash"),start_event_time=a["start_event_time"],
            observed_touchdown_time=a.get("completion_time")) for a in action_p2
            if a["start_event_time"]<end and (a.get("completion_time") is None or a["completion_time"]>end)]
        for label,value in (("actual_start",next_start),("occupancy",next_occupancy),("touchdown",next_touchdown)):
            row[label]=value
            row[label+"_gap_s"]=None if value is None else value["time"]-end
            row[label+"_censored"]=value is None
            row[label+"_censor_time"]=endpoint if value is None else None
        rows.append(row)
    return dict(pad_id=2,rows=rows,endpoint_s=endpoint,
        occupancy_basis="SAVED_COMMITTED_PAD_INTERVAL_PRESENT_AT_OR_AFTER_START_WITH_ACTUAL_ACTION_OR_SERVICE",
        semantics="APPROACH_START_IS_NOT_PAD_OCCUPANCY; NO_TOUCHDOWN_EVENT_IS_CENSORED")


def cell(directory, endpoint=None):
    directory=Path(directory)
    if not (directory/"normal_tasks.json").is_file():
        return dict(status="NOT_RUN",directory=str(directory),statistics=None,rounds=[])
    statistics=saved.saved_cell(directory,endpoint)
    records=saved.load(directory/"scheduler_records.json",[])
    tasks=saved.load(directory/"normal_tasks.json",[])
    events=saved.lines(directory/"replay/events.jsonl")
    states=saved.lines(directory/"replay/states.jsonl")
    plans=saved.load(directory/"replay/trajectories.json",{})
    replay=decisions.extract(str(directory),records,tasks,events,states,plans)
    work={row["snapshot_time"]:row for row in statistics["normal_lifecycle"].get("per_task_work",[])}
    timelines={row["snapshot_time"]:row for row in statistics["task_timeline"]}
    callbacks={r.get("planning_timing",{}).get("snapshot_time"):r for r in records}
    task_by_time={task["snapshot_time"]:task for task in tasks}
    rounds=[]
    for row in replay["rounds"]:
        stamp=row["snapshot_time"]; record=callbacks.get(stamp,{})
        rounds.append(dict(row,full_planning_work=work.get(stamp),
            exclusive_phase_timings=record.get("timings"),timeline=timelines.get(stamp),
            pair_conflicts=conflict_facts(record,replay["uid_by_id"]),
            revalidations=task_by_time[stamp].get("revalidations",[]),
            last_revalidation_budget=task_by_time[stamp].get("last_revalidation_budget")))
    evidence=arm_evidence(directory,endpoint)
    return dict(status="SAVED",directory=str(directory),statistics=statistics,rounds=rounds,
        run_evidence=evidence,run_binding=evidence["run_binding"],
        replay_checks=replay["checks"],replay_issues=replay["issues"],uid_by_id=replay["uid_by_id"],
        fairness_debt_observations=named_values(records,{"fairness_debt","fair_debt","fairdebt","fairness_bonus",
            "fairness_overdue_ids","fairness_unselected_overdue_ids","progress_age","review_age"}),
        fallback_observations=named_values(records,{"full_fallback","fallback_reason","overflow_cause","actual_k"}),
        actual_k_basis="STATISTICS_USE_SCOPE_ACTUAL_K; TOP_LEVEL_ACTUAL_K_IS_INITIAL_NOMINAL_AND_RETAINED_ONLY_AS_LOCATED_RAW_EVIDENCE",
        pair_conflicts=conflict_facts(records,replay["uid_by_id"]),
        p2=p2_facts(events,states,replay["all_started_actions"],statistics["endpoint_s"]))


def arm_evidence(directory, endpoint=None):
    directory=Path(directory); arm=directory.parent.parent.parent
    binding=saved.load(arm/"resolved_run_binding.json")
    finished=saved.load(arm/"arm.finished.json")
    result=saved.load(directory/"result.json")
    provenance=saved.load(arm/"continuation_provenance.json")
    finished_provenance=saved.load(arm/"continuation_finished_provenance.json")
    if provenance is None and isinstance(binding,dict): provenance=binding.get("continuation")
    available=sorted(directory.glob("boundary_*.json"))
    boundary=saved.load(directory/f"boundary_{endpoint:04d}.json") if endpoint is not None else saved.load(available[-1]) if available else None
    costs=(boundary or {}).get("costs",{}) if isinstance(boundary,dict) else {}
    observation=finished if isinstance(finished,dict) else result if isinstance(result,dict) else {}
    segment_costs=observation.get("continuation_costs") or {}
    terminal=observation.get("terminal_observation") or {}
    complete=(observation.get("status")=="COMPLETE" and observation.get("completed_boundary_s")==300
        and observation.get("error") is None and terminal.get("observed_time_s")==300.
        and terminal.get("failed") is False and terminal.get("terminated_invalid") is False
        and (directory/"checkpoint_0300.json").is_file() and (directory/"boundary_0300.json").is_file())
    provenance_map=provenance if isinstance(provenance,dict) else {}
    failed_job=provenance_map.get("original_failed_job") or {}
    return dict(arm_directory=str(arm),binding_path=str(arm/"resolved_run_binding.json"),run_binding=binding,
        arm_finished=finished,evaluation_result=result,completion_status="COMPLETE_300" if complete else "INCOMPLETE_OR_UNKNOWN",
        continuation_provenance=provenance,continuation_finished_provenance=finished_provenance,
        wall_accounting=dict(cumulative_boundary_costs=costs,
            whole_child_full_job_wall_s=observation.get("full_job_wall_s"),
            whole_child_full_job_wall_scope=observation.get("full_job_wall_scope"),
            continuation_costs=segment_costs,
            continuation_job_wall_s=segment_costs.get("continuation_job_wall_s",costs.get("continuation_job_wall_s")),
            original_failed_child_wall_s=observation.get("original_failed_child_wall_s",failed_job.get("child_wall_s")),
            semantics="CUMULATIVE_BOUNDARY_COST_CONTINUATION_SEGMENT_AND_FAILED_CHILD_OVERHEAD_ARE_SEPARATE_NOT_SUMMED"))


def parse_arm_overrides(values):
    result={}
    for value in values or []:
        root,separator,path=value.partition("=")
        if not separator or not root or not path or root in result:
            raise ValueError("each --new-arm must be one unique ROOT=ARM_DIRECTORY")
        result[root]=Path(path)
    return result


def validate_source_bindings(binding,*,original_source,continuation_source=None):
    """Require separate original/continuation identities; never relabel the old arm."""
    current=(binding.get("source_hash"),binding.get("source_commit"))
    continuation=binding.get("continuation")
    if continuation is None:
        if current!=tuple(original_source): raise ValueError("original arm source binding differs")
        return dict(kind="ORIGINAL_ARM",source_hash=current[0],source_commit=current[1])
    if continuation_source is None or current!=tuple(continuation_source):
        raise ValueError("continuation source binding differs or is missing")
    old=(continuation.get("old_source_hash"),continuation.get("old_source_commit"))
    new=(continuation.get("new_source_hash"),continuation.get("new_source_commit"))
    if old!=tuple(original_source) or new!=tuple(continuation_source):
        raise ValueError("continuation dual-source provenance differs")
    return dict(kind="CONTINUATION",original_source_hash=old[0],original_source_commit=old[1],
        continuation_source_hash=new[0],continuation_source_commit=new[1],continuation=continuation)


def diagnostic_inputs(root):
    rows=[]
    for directory in sorted((Path(root)/"snapshots").glob("*/*")):
        if not directory.is_dir(): continue
        actual=saved.load(directory/"actual_diagnostic.json")
        legacy=saved.load(directory/"result.json") if actual is None else None
        payload=actual if actual is not None else legacy
        if payload is None: continue
        observation=payload.get("checkpoint_observation",{})
        records=payload.get("scheduler_records",[])
        timings=[record.get("planning_timing") for record in records]
        # Exact saved timing dictionaries are required; a nominal snapshot_time is insufficient.
        timing=timings[0] if len(timings)==1 and timings[0] is not None else None
        binding=saved.load(directory/"resolved_run_binding.json",{})
        rows.append(dict(root=directory.parent.name,case=directory.name,path=str(directory),
            variant=payload.get("treatment",{}).get("selective_rule_variant"),
            checkpoint_sha256=observation.get("checkpoint_sha256",payload.get("checkpoint_sha256")),
            timing=timing,physical_source_version=payload.get("actual_source_version",payload.get("physical_source_version")),
            projected_version=payload.get("projected_version"),source_hash=binding.get("source_hash"),
            boundary_delta_s=observation.get("boundary_minus_historical_source_s"),
            cold_normal_cache=payload.get("cold_normal_cache"),
            historical_input_recovered=observation.get("historical_input_recovered",False),
            catalog_basis="COLD_DIAGNOSTIC_CATALOG_NOT_HISTORICAL_CATALOG",
            actual_diagnostic=actual,legacy_snapshot_result=legacy,
            probes=saved.load(directory/"probe_results.json"),
            pair_conflicts=conflict_facts(payload,{})))
    return rows


def pair_diagnostics(old_rows,new_rows):
    pairs=[]
    for new in new_rows:
        matches=[old for old in old_rows if old["root"]==new["root"]
            and new.get("checkpoint_sha256") is not None
            and old.get("checkpoint_sha256")==new["checkpoint_sha256"]
            and new.get("timing") is not None and old.get("timing")==new["timing"]]
        pairs.append(dict(new_path=new["path"],old_paths=[old["path"] for old in matches],
            status="MATCHED" if len(matches)==1 else "UNMATCHED" if not matches else "AMBIGUOUS",
            version_observations=[dict(old_physical_source_version=old.get("physical_source_version"),
                new_physical_source_version=new.get("physical_source_version"),
                old_projected_version=old.get("projected_version"),new_projected_version=new.get("projected_version"),
                content_changed=None,meaning="VERSION_HASH_DIFFERENCE_DOES_NOT_PROVE_CONTENT_CHANGE") for old in matches]))
    return pairs


def boundaries(directory):
    return {int(path.stem.split("_")[-1]) for path in Path(directory).glob("boundary_*.json")}


def build_summary(new_root,old_root,roots=("s3r23","s3r24"),*,new_arms=None):
    new_arms=new_arms or {}
    if set(new_arms)-set(roots): raise ValueError("new-arm override contains unknown root")
    cells={}
    for root in roots:
        relative=Path("arms")/root/"evaluation"/root/"RULE_LOCAL_K8"
        arm=Path(new_arms[root]) if root in new_arms else Path(new_root)/"arms"/root
        new=arm/"evaluation"/root/"RULE_LOCAL_K8"; old=Path(old_root)/relative
        common=max(boundaries(new)&boundaries(old),default=None)
        # Use endpoint aggregates only when both endpoints are actually saved.
        cells[root]=dict(common_endpoint_s=common,new_arm_directory=str(arm),new_arm_overridden=root in new_arms,
            old_v2=cell(old,common),new_v3=cell(new,common),
            endpoint_aggregate_comparable=common is not None,
            longitudinal_trace_scope="COMPLETE_SAVED_TRACE_MAY_EXTEND_BEYOND_COMMON_ENDPOINT",
            comparison_scope="WHOLE_VERSION_BEFORE_AFTER_ONE_DEVELOPMENT_EXAMPLE_PER_ROOT_NOT_CAUSAL_ISOLATION")
    preexisting_diagnostics=diagnostic_inputs(old_root); current_diagnostics=diagnostic_inputs(new_root)
    diagnostic_v2=[row for row in current_diagnostics if row["variant"]=="opportunity_cooperative_v2"]
    diagnostic_v3=[row for row in current_diagnostics if row["variant"]=="opportunity_executable_v3"]
    return dict(schema="rs_selective_repair_saved_report_v3",old_v2_root=str(old_root),new_v3_root=str(new_root),
        cells=cells,diagnostics=dict(current_v2=diagnostic_v2,current_v3=diagnostic_v3,
            unclassified_current=[row for row in current_diagnostics if row not in diagnostic_v2 and row not in diagnostic_v3],
            preexisting_reference=preexisting_diagnostics,
            pairs=pair_diagnostics(diagnostic_v2,diagnostic_v3),
            pairing_scope="CURRENT_NEW_ROOT_V2_VERSUS_V3_ONLY_PREEXISTING_REFERENCE_EXCLUDED"),
        budget_ledger=saved.load(Path(new_root)/"repair_budget_ledger.json"),
        original_formal_finished=saved.load(Path(new_root)/"formal_finished.json"),
        report_scope="SAVED_EVIDENCE_ONLY_NO_SIMULATION_OR_SOLVER",scientific_acceptance="PENDING_PRIMARY_REVIEW",
        comparison_scope="WHOLE_VERSION_BEFORE_AFTER_NOT_CAUSALLY_ISOLATED",
        missing_semantics="NULL_OR_NOT_RUN_IS_MISSING_NOT_ZERO; PREDICTIONS_ARE_NOT_EXECUTION")


def write_report(new_root,old_root,roots=("s3r23","s3r24"),*,new_arms=None):
    new_root=Path(new_root); old_root=Path(old_root)
    if new_root.resolve()==old_root.resolve() or old_root.resolve() in new_root.resolve().parents:
        raise ValueError("output root must differ from and stay outside read-only old v2 root")
    summary=build_summary(new_root,old_root,roots,new_arms=new_arms)
    new_root.mkdir(parents=True,exist_ok=True)
    (new_root/"v3_analysis_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    csv_rows=[]
    text=["# Selective repair v3 saved evidence","",
        "Whole-version old_v2 versus new_v3; one development example per root. These observations do not isolate causes.",
        "Missing results remain NOT_RUN/null. Complete saved round traces can extend beyond the common aggregate endpoint.","",
        "| Root | Version | Saved status | Completion evidence | Endpoint | Normal | Emergency | Energy failures | Air pending | Upstream pending |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for root,pair in summary["cells"].items():
        for version in ("old_v2","new_v3"):
            entry=pair[version]; statistics=entry.get("statistics") or {}; counts=statistics.get("system_counts",{})
            text.append("| "+" | ".join(str(value) for value in (root,version,entry["status"],
                entry.get("run_evidence",{}).get("completion_status"),statistics.get("endpoint_s"),
                counts.get("L_normal"),counts.get("L_emergency"),counts.get("F_energy_total"),
                counts.get("N_airborne_pending"),counts.get("N_upstream_pending")))+" |")
            for row in entry["rounds"]:
                csv_rows.append(dict(root=root,version=version,round=row["round"],snapshot_sim_time=row["snapshot_time"],
                    effective_sim_time=row["effective_time"],status=row["status"],reason=row["reason"],
                    selection=row["selection"],full_planning_work=row["full_planning_work"],
                    exclusive_phase_timings=row["exclusive_phase_timings"],timeline=row["timeline"],
                    actual_actions=row["actual_actions"],pair_conflicts=row["pair_conflicts"],revalidations=row["revalidations"],
                    last_revalidation_budget=row["last_revalidation_budget"]))
    text.extend(["","JSON retains existing saved-cell statistics: ring holding energy/time, normal/emergency/failure/pending counts, fairness ages/debt, actual K and fallback causes, admission denominators, full planning work, exclusive phases and certificate wall → final ready simulation → actual start simulation → next start simulation.",
        "P2 service completion → next actual approach start, recorded pad occupancy and touchdown are separate; absent touchdown is censored. Conflict evidence preserves raw hashes, participants, intervals and recorded witnesses; missing distance is never fabricated.",
        "Snapshots are cold boundary diagnostics, not original historical catalogs. Pair only current new-root v2/v3 variants with identical checkpoint SHA and complete planning timing; old-root snapshots are preexisting references. Different version hashes alone do not establish different content.",
        "Completion requires saved COMPLETE/300 terminal, checkpoint and boundary evidence; a launcher exit alone is insufficient. Continuation provenance and original/new source bindings are retained verbatim. Cumulative boundary wall, continuation segment wall and failed-child wall remain separate and are never summed by this report.",
        "Actual K statistics use final scope.actual_k; top-level nominal actual_k stays separate. Algorithm attribution and acceptance remain with the primary review.",""])
    (new_root/"REPORT.md").write_text("\n".join(text),encoding="utf8")
    decisions.write_csv(new_root/"rounds.csv",csv_rows)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root",type=Path,default=Path("/home/user/training_outputs/terminal/rs_selective_repair_v3_20261029"))
    parser.add_argument("--old-root",type=Path,default=Path("/home/user/training_outputs/terminal/rs_selective_repair_20261029"))
    parser.add_argument("--new-arm",action="append",default=[],metavar="ROOT=ARM_DIRECTORY",
        help="explicit arm root containing resolved_run_binding.json and evaluation/ROOT/RULE_LOCAL_K8")
    args=parser.parse_args()
    write_report(args.new_root,args.old_root,new_arms=parse_arm_overrides(args.new_arm))


if __name__=="__main__": main()
