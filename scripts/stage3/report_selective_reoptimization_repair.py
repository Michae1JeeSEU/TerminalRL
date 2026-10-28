"""Read saved repair evidence; missing observations stay unknown/pending."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median

from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import report_selective_reoptimization_a as audit
from scripts.stage3 import run_latency_interface as latency


def load(path, default=None):
    return base.load(path) if Path(path).is_file() else default


def lines(path):
    if not Path(path).is_file(): return []
    with Path(path).open(encoding="utf8") as stream:
        return [json.loads(row) for row in stream if row.strip()]


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def work_stats(values):
    observed=[value for value in values if finite(value)]
    return dict(latency.measured(observed),median_s=median(observed) if observed else None)


def canonical_hold_powers(config_path):
    from terminal_marl.core.rotating_slots.config import load_config
    from terminal_marl.core.rotating_slots.energy import stable_power
    config=load_config(config_path)
    return {ring:stable_power(ring,config).battery_w for ring in ("outer","middle","inner")}


def derived_holding_times(energy_by_ring,total_holding_s,config_path):
    """Analytic conversion of recorded executed holding energy, no trajectories."""
    rings=("outer","middle","inner");config_path=Path(config_path)
    result=dict(status="UNAVAILABLE",basis="DERIVED_EXECUTED_HOLD_ENERGY_DIV_CANONICAL_POWER",
        holding_time_by_ring_s=None,power_battery_w=None,derived_sum_s=None,
        recorded_total_holding_s=total_holding_s,sum_difference_s=None,verification_passed=False,
        absolute_tolerance_s=1.e-6,config_path=str(config_path))
    if not isinstance(energy_by_ring,dict) or any(ring not in energy_by_ring for ring in rings):
        return dict(result,reason="MISSING_EXECUTED_HOLD_ENERGY_RING")
    if any(not finite(energy_by_ring[ring]) or energy_by_ring[ring]<0. for ring in rings):
        return dict(result,reason="INVALID_EXECUTED_HOLD_ENERGY")
    if not finite(total_holding_s) or total_holding_s<0.:
        return dict(result,reason="MISSING_OR_INVALID_RECORDED_TOTAL_HOLDING_TIME")
    if not config_path.is_file(): return dict(result,reason="MISSING_REPLAY_CONFIG")
    try: powers=canonical_hold_powers(config_path)
    except Exception as exc:
        return dict(result,reason="CANONICAL_POWER_UNAVAILABLE",error=dict(type=type(exc).__name__,message=str(exc)))
    result["power_battery_w"]=powers
    if any(not finite(powers.get(ring)) or powers[ring]<=0. for ring in rings):
        return dict(result,reason="INVALID_CANONICAL_POWER")
    seconds={ring:energy_by_ring[ring]/powers[ring] for ring in rings}
    total=math.fsum(seconds.values());difference=total-total_holding_s
    result.update(derived_sum_s=total,sum_difference_s=difference)
    if abs(difference)>result["absolute_tolerance_s"]:
        return dict(result,reason="DERIVED_SUM_MISMATCH_WITH_RECORDED_TOTAL")
    return dict(result,status="VERIFIED_DERIVED",reason=None,verification_passed=True,holding_time_by_ring_s=seconds)


def recorded_ring_pending(frame,endpoint=None):
    from terminal_marl.core.rotating_slots.world import AIRBORNE
    if not frame or "aircraft" not in frame:
        return dict(status="UNAVAILABLE",reason="MISSING_SAVED_STATE",airborne_by_recorded_ring=None,unmapped=None)
    if endpoint is not None and (not finite(frame.get("time")) or abs(frame["time"]-endpoint)>1.e-7):
        return dict(status="UNAVAILABLE",reason="LAST_SAVED_FRAME_NOT_REQUESTED_ENDPOINT",
            requested_endpoint_s=endpoint,last_saved_frame_s=frame.get("time"),airborne_by_recorded_ring=None,unmapped=None)
    rings=("outer","middle","inner");counts={ring:0 for ring in rings};statuses={ring:Counter() for ring in rings}
    unmapped=[];nonholding=[]
    for row in frame["aircraft"]:
        if row.get("status") not in AIRBORNE: continue
        ring=row.get("ring")
        if ring in counts: counts[ring]+=1;statuses[ring][row["status"]]+=1
        else: unmapped.append(dict(aircraft_id=row.get("id"),status=row.get("status"),recorded_ring=ring))
        if not row["status"].startswith("HOLD_"):
            nonholding.append(dict(aircraft_id=row.get("id"),status=row["status"],recorded_ring=ring))
    return dict(status="OBSERVED",basis="LAST_SAVED_PHYSICAL_AIRCRAFT_RING_AND_AIRBORNE_STATUS_NOT_PROJECTED_NODE",
        saved_frame_time=frame.get("time"),airborne_by_recorded_ring=counts,
        status_by_recorded_ring={ring:dict(value) for ring,value in statuses.items()},
        unmapped=unmapped,unmapped_count=len(unmapped),airborne_total=sum(counts.values())+len(unmapped),
        nonholding_airborne_rows=nonholding)


def scope_work_facts(lifecycle,records):
    by_time=defaultdict(list)
    for record in records:
        stamp=record.get("planning_timing",{}).get("snapshot_time")
        if stamp is not None: by_time[stamp].append(record)
    groups={kind:[] for kind in ("LOCAL","OVERFLOW_FULL","UNKNOWN")}
    for work in lifecycle.get("per_task_work",[]):
        matching=by_time.get(work["snapshot_time"],[])
        telemetry=matching[0].get("selective_replanning",{}) if len(matching)==1 else {}
        scope=telemetry.get("scope") or {};status=scope.get("status","")
        kind="OVERFLOW_FULL" if "OVERFLOW" in status else "LOCAL" if status=="LOCAL" else "UNKNOWN"
        groups[kind].append(dict(work,snapshot_time=work["snapshot_time"],scope_status=scope.get("status"),
            actual_k=scope.get("actual_k"),overflow_cause=telemetry.get("overflow_cause"),
            mandatory_ids=scope.get("mandatory_ids"),selection_reasons=scope.get("selection_reasons"),
            join_status="MATCHED_UNIQUE_SNAPSHOT_TIME" if len(matching)==1 else "MISSING_CALLBACK" if not matching else "AMBIGUOUS_CALLBACK_TIME"))
    per_task_total=math.fsum(row["full_normal_planning_work_s"] for rows in groups.values() for row in rows if finite(row.get("full_normal_planning_work_s")))
    full_total=lifecycle.get("full_normal_planning_work",{}).get("total_s")
    return dict(basis="FULL_PER_TASK_NORMAL_WORK_JOINED_BY_RECORDED_SNAPSHOT_TIME_PENDING_PARTIAL_INCLUDED",
        per_task_total_s=per_task_total,lifecycle_full_normal_work_total_s=full_total,
        unattributed_history_service_s=lifecycle.get("selective_history_work",{}).get("unattributed_service_s"),
        lifecycle_minus_per_task_s=full_total-per_task_total if finite(full_total) else None,
        groups={kind:dict(task_count=len(rows),pending_task_count=sum(not row["finished"] for row in rows),
            work=work_stats(row.get("full_normal_planning_work_s") for row in rows),
            actual_k_counts={"UNKNOWN" if key is None else str(key):value for key,value in Counter(row["actual_k"] for row in rows).items()},
            recorded_overflow_cause_counts=dict(Counter(row["overflow_cause"] for row in rows if row["overflow_cause"] is not None)),
            rows=rows) for kind,rows in groups.items()})


def task_continuation_facts(tasks):
    ordered=sorted(tasks,key=lambda row:row["snapshot_time"]);rows=[]
    for index,task in enumerate(ordered):
        finished=task.get("finished_time");next_task=ordered[index+1] if index+1<len(ordered) else None
        recorded_next_start=task.get("next_started_time")
        next_start=next_task["snapshot_time"] if next_task is not None else recorded_next_start
        gap=next_start-finished if finite(finished) and finite(next_start) else None
        matched_handoff=finite(recorded_next_start) and finite(next_start) and abs(recorded_next_start-next_start)<=1.e-7
        restart=task.get("restart_reason") if matched_handoff else None
        rows.append(dict(snapshot_time=task["snapshot_time"],finished_sim_time=finished,next_started_sim_time=next_start,
            finish_to_next_start_sim_s=gap,recorded_finish_to_next_start_s=task.get("finish_to_next_start_s"),
            recorded_next_started_time=recorded_next_start,
            restart_reason_source="CURRENT_TASK_RESTART_REASON_WITH_MATCHED_NEXT_STARTED_TIME" if matched_handoff and restart is not None else "UNKNOWN_NO_EXPLICIT_MATCHING_HANDOFF_REASON",
            next_restart_reason=restart,same_time=bool(finite(gap) and abs(gap)<=1.e-7),
            coalesced_same_time=bool(finite(gap) and abs(gap)<=1.e-7 and (restart or "").startswith("COALESCED_")),
            status=task.get("status"),reason=task.get("reason")))
    reasons=Counter(task["reason"] for task in tasks if task.get("reason"))
    return dict(clock="EXECUTOR_SIMULATION_TIME",rows=rows,
        finish_to_next_start=work_stats(row["finish_to_next_start_sim_s"] for row in rows),
        same_time_finish_next_start_count=sum(row["same_time"] for row in rows),
        explicitly_coalesced_same_time_count=sum(row["coalesced_same_time"] for row in rows),
        negative_gap_count=sum(finite(row["finish_to_next_start_sim_s"]) and row["finish_to_next_start_sim_s"] < -1.e-7 for row in rows),
        finished_without_observed_next_count=sum(finite(row["finished_sim_time"]) and row["next_started_sim_time"] is None for row in rows),
        reason_counts=dict(reasons),deadline_reason_counts={reason:count for reason,count in reasons.items()
            if "DEADLINE" in reason or reason.split(":",1)[0]=="LATENCY_PACKET_EXPIRED"},
        no_executable_prefix_count=reasons.get("NO_EXECUTABLE_PREFIX",0),
        revalidation_recorded_reason_counts=dict(Counter(event["reason"] for task in tasks for event in task.get("revalidations",[]) if event.get("reason"))),
        interpretation="Reason counts are saved observations; pending revalidation reasons are not final task rejection claims.")


def selection_facts(records):
    rows=[]
    for record in records:
        t=record.get("selective_replanning", {}); scope=t.get("scope") or {}
        if not scope: continue
        ages=t.get("ages", {}); reasons=dict(scope.get("selection_reasons", []))
        hard={aid:list(reasons.get(aid, ()))
              for aid in scope.get("mandatory_ids", [])}
        overdue_ids=t.get("fairness_overdue_ids", [aid for aid,age in ages.items() if (age.get("review_age") or 0.)>=60.])
        overdue={aid:{"review_age":ages.get(aid,{}).get("review_age"), "progress_age":ages.get(aid,{}).get("progress_age")}
                 for aid in overdue_ids}
        rows.append(dict(snapshot_time=record.get("planning_timing", {}).get("snapshot_time"),
            effective_time=scope.get("effective_time"), status=scope.get("status"), actual_k=scope.get("actual_k"),
            selected_ids=scope.get("replan_ids", scope.get("selected_ids", [])), eligible_ids=scope.get("eligible_ids", []),
            selected_ids_basis="RETURNED_CALLBACK_SCOPE_MAY_INCLUDE_BOUNDED_EXPANSION",
            mandatory_ids=scope.get("mandatory_ids", []), selection_reasons=reasons,
            hard_causes_recorded=hard, overdue=overdue,
            fairness_overdue_basis="RECORDED_FAIRNESS_OVERDUE_IDS" if "fairness_overdue_ids" in t else "RECORDED_REVIEW_AGE_GE60_SECONDS",
            review_unknown=[aid for aid,a in ages.items() if a.get("review_unknown")],
            progress_unknown=[aid for aid,a in ages.items() if a.get("progress_unknown")],
            raw_ages=ages, overflow="OVERFLOW" in scope.get("status", ""),
            candidates_by_aircraft=t.get("candidate_generation_by_aircraft"), solve_calls=t.get("solve_calls"),
            coverage=record.get("catalog_rounds", [{}])[-1].get("coverage") if record.get("catalog_rounds") else record.get("aircraft_choices"),
            final_objective=record.get("final_objective", record.get("objective")),
            first_nonempty_certified=record.get("search_handoff", {}).get("first_nonempty_certified"),
            budget=record.get("ready_budget"), planning_budget=record.get("planning_budget"),
            opportunity_features=t.get("opportunity_features"), raw_selector_measurements=t))
    return dict(rows=rows, overflow_rows=[r for r in rows if r["overflow"]],
        actual_k_counts=dict(Counter(r["actual_k"] for r in rows)),
        unknown_semantics="Missing ages/features/causes are unobserved, not zero or inferred.")


def timing_facts(records):
    phases=defaultdict(lambda:dict(wall_s=0., cpu_s=0., observations=0))
    for record in records:
        for key,value in record.get("timings", {}).items():
            phases[key]["wall_s"]+=value[0]; phases[key]["cpu_s"]+=value[1]; phases[key]["observations"]+=1
    names=("candidate_generation_and_single_vehicle", "trajectory_validation", "prefix_validation", "cp_solve")
    return dict(exclusive={name:phases.get(name) for name in names},
        feature={name:row for name,row in phases.items() if "feature" in name},
        shared_public={name:phases.get(name) for name in ("global_evaluation_frame", "selective_selection", "selective_opportunity_features")},
        all_exclusive_phases=dict(phases),
        semantics="PlanningBudget.measure subtracts nested child timing; absent phase is null. Shared/public and features are separate; do not add dispatch again.")


def admission_facts(tasks, records, plans):
    """Separate task denominator, aircraft-opportunities and unique aircraft."""
    callbacks={r["planning_timing"]["snapshot_time"]:r for r in records if r.get("planning_timing")}
    offered=[task for task in tasks if callbacks.get(task["snapshot_time"], {}).get("deferred") is True
             and callbacks[task["snapshot_time"]].get("noop") is False]
    finished=[task for task in offered if "finished_time" in task]
    joined=[task for task in finished if task.get("status") == "COMMITTED" and (task.get("submitted") or 0)>0]
    selected_opportunities=0; selected_ids=set(); submitted_ids=set(); submitted_opportunities=0; missing_hash_aircraft=[]
    for record in records:
        scope=record.get("selective_replanning", {}).get("scope") or {}
        selected=set(scope.get("replan_ids", scope.get("selected_ids", [])))
        selected_opportunities+=len(selected); selected_ids.update(selected)
        if record.get("actual_delivery_status") != "COMMITTED": continue
        submitted=set()
        for h in record.get("actual_submitted_trajectory_hashes", []):
            aid=plans.get(h, {}).get("aircraft_id")
            if aid is None: missing_hash_aircraft.append(h)
            else: submitted.add(aid)
        submitted_opportunities+=len(submitted); submitted_ids.update(submitted)
    return dict(submitted_task_count=len(joined), finished_nonempty_task_count=len(finished),
        submitted_task_rate=ratio(len(joined), len(finished)), task_rate_denominator="FINISHED_NONEMPTY_OFFERED_TASKS_PENDING_EXCLUDED",
        selected_aircraft_task_opportunities=selected_opportunities, submitted_aircraft_task_opportunities=submitted_opportunities,
        submitted_aircraft_opportunity_rate=None if missing_hash_aircraft else ratio(submitted_opportunities, selected_opportunities),
        aircraft_opportunity_denominator="SELECTED_AIRCRAFT_PER_CALLBACK_INCLUDING_PENDING_CALLBACKS",
        selected_unique_aircraft=len(selected_ids), submitted_unique_aircraft=len(submitted_ids),
        submitted_unique_aircraft_rate=None if missing_hash_aircraft else ratio(len(submitted_ids),len(selected_ids)),
        actual_submitted_plans=sum(task.get("submitted") or 0 for task in joined),
        missing_submitted_hash_aircraft=missing_hash_aircraft)


def task_timelines(tasks, records, events, plans):
    callbacks={r["planning_timing"]["snapshot_time"]:r for r in records if r.get("planning_timing")}
    starts=sorted(task["snapshot_time"] for task in tasks); rows=[]
    for task in tasks:
        record=callbacks.get(task["snapshot_time"], {}); handoff=task.get("search_handoff", record.get("search_handoff", {})) or {}
        first=(record.get("selective_replanning") or {}).get("first_certified")
        nonempty=handoff.get("first_nonempty_certified")
        hashes=set(record.get("actual_submitted_trajectory_hashes", []))
        actual=[dict(time=e["time"], aircraft_id=e.get("aircraft_id"), trajectory_hash=e.get("details", {}).get("trajectory_hash"))
                for e in events if e.get("type") == "maneuver_started" and e.get("details", {}).get("trajectory_hash") in hashes
                and e["time"] >= task["snapshot_time"]]
        # Hashes may recur: the acknowledged absolute effective-time start is
        # authoritative, while later suffix starts are listed separately.
        actual=[e for e in actual if abs(plans.get(e["trajectory_hash"], {}).get("start_time", -1.)-e["time"]) < 1.e-7]
        rows.append(dict(snapshot_time=task["snapshot_time"],
            first_certified_call_return_wall_elapsed_s=None if first is None else first.get("wall_elapsed_s"),
            first_certified_call_return=first,
            first_certified_call_return_raw_source="scheduler_records[].selective_replanning.first_certified",
            first_any_certificate_instant=None,first_any_certificate_instant_wall_elapsed_s=None,
            first_any_certificate_instant_status="UNKNOWN_NO_DIRECT_FIRST_ANY_CERTIFICATE_EVENT",
            first_nonempty_certified_wall_elapsed_s=None if nonempty is None else nonempty.get("wall_elapsed_s"),
            first_nonempty_certified=nonempty, final_ready_sim_time=task.get("ready_time"),
            actual_publish_sim_time=record.get("actual_delivery_time") if record.get("actual_delivery_status") == "COMMITTED" else None,
            actual_maneuver_starts=actual, next_snapshot_sim_time=next((t for t in starts if t>task["snapshot_time"]),None),
            final_objective=record.get("final_objective", record.get("objective")), status=task.get("status"), reason=task.get("reason"),
            certificate_wall_clock_basis=dict(first_certified_call_return="SELECTIVE_LOCAL_PLAN_CALL_RETURN_ELAPSED_WALL_TIME_NOT_CERTIFICATE_INSTANT_INCLUDES_WAIT",
                first_nonempty_certified="SEARCH_HANDOFF_PLANNING_WALL_ORIGIN_ELAPSED_TIME_NONEMPTY_PREFIX_ONLY"),
            clocks="Certificate-event and call-return fields use elapsed wall time; snapshot/ready/publication/maneuver-start/next-snapshot fields use executor simulation time. No inferred publication or conversion between clocks."))
    return rows


def nonlanding_facts(records, events, plans, terminal_aircraft, tasks=()):
    """Labels are observations that may coexist; avoid causal reconstruction."""
    labels=defaultdict(lambda:defaultdict(list)); known=set(); joint_observations=[]; admission_unattributed=[]
    task_by_time={task["snapshot_time"]:task for task in tasks}
    for i,record in enumerate(records):
        t=record.get("selective_replanning", {}); scope=t.get("scope") or {}
        eligible=set(scope.get("eligible_ids", [])); selected=set(scope.get("replan_ids", scope.get("selected_ids", [])))
        known.update(eligible)
        for aid in eligible-selected: labels[aid]["UNSELECTED"].append(i)
        # This describes only the returned finite catalog. Neither a missing
        # choice row nor a generator count of zero establishes physical failure.
        for choice in record.get("aircraft_choices", []):
            aid=choice.get("aircraft_id")
            if aid and choice.get("move_candidates") == 0:
                labels[aid]["NO_VALID_CANDIDATE"].append(dict(callback=i,
                    meaning="NO_VALID_MOVE_OBSERVED_IN_RETURNED_BOUNDED_CATALOG_NOT_PHYSICAL_INFEASIBILITY",
                    move_candidates=0,only_keep_wait=choice.get("only_keep_wait"),
                    candidate_insufficiency_reasons=choice.get("candidate_insufficiency_reasons"),
                    rejection_reasons=choice.get("rejection_reasons")))
        for observation in t.get("joint_conflict_observations", record.get("joint_conflict_observations", [])):
            joint_observations.append(dict(callback=i,observation=observation))
            for aid in observation.get("aircraft_ids", []):
                labels[aid]["JOINT_CONFLICT"].append(dict(callback=i,observation=observation))
        if record.get("actual_delivery_status") == "REJECTED":
            task=task_by_time.get(record.get("planning_timing", {}).get("snapshot_time"), {})
            hashes=record.get("proposed_trajectory_hashes", record.get("proposal_trajectory_hashes", task.get("proposed_trajectory_hashes", [])))
            proposed={plans.get(h, {}).get("aircraft_id") for h in hashes}
            for aid in proposed-{None}: labels[aid]["ADMISSION_FAILED"].append(dict(callback=i, reason=record.get("actual_delivery_reason")))
            if not hashes or None in proposed:
                admission_unattributed.append(dict(callback=i,reason=record.get("actual_delivery_reason"),
                    proposal_hashes=hashes,missing_aircraft_hashes=[h for h in hashes if plans.get(h,{}).get("aircraft_id") is None]))
    landed={aid for aid,row in terminal_aircraft.items() if row.get("status") in ("LANDED", "LANDED_SERVICE", "CLEARED", "GROUND")}
    landed.update(event.get("aircraft_id") for event in events if event.get("type") == "touchdown")
    for event in events:
        aid=event.get("aircraft_id")
        if event.get("type") == "maneuver_started" and aid and aid not in landed:
            labels[aid]["EXECUTED_NOT_LANDED"].append(dict(time=event["time"], trajectory_hash=event.get("details", {}).get("trajectory_hash")))
    result=[]
    for aid in sorted((known | set(terminal_aircraft))-landed):
        observed=dict(labels.get(aid, {}))
        if not observed: observed={"UNKNOWN":[]}
        result.append(dict(aircraft_id=aid, terminal_status=terminal_aircraft.get(aid, {}).get("status"), observation_labels=observed))
    return dict(rows=result, reason_counts=dict(Counter(key for row in result for key in row["observation_labels"])),
        joint_conflict_observations=joint_observations,
        admission_failed_unattributed=admission_unattributed,
        joint_conflict_evidence="OBSERVED" if joint_observations else "UNKNOWN_NO_SAVED_OBSERVATION",
        attribution="NONEXCLUSIVE_SAVED_OBSERVATIONS_NOT_CAUSAL_PROOF", predictions_are_execution=False)


def saved_cell(directory, endpoint=None, *, audit_override=None):
    directory=Path(directory)
    paths=sorted(directory.glob("boundary_*.json"))
    boundary=load(directory/f"boundary_{endpoint:04d}.json") if endpoint is not None else load(paths[-1]) if paths else None
    records=load(directory/"scheduler_records.json", []); tasks=load(directory/"normal_tasks.json", [])
    services=load(directory/"online_services.json", []); events=lines(directory/"replay/events.jsonl")
    plans=load(directory/"replay/trajectories.json", {})
    frames=lines(directory/"replay/states.jsonl"); terminal=frames[-1] if frames else {}
    terminal_aircraft={row["id"]:row for row in terminal.get("aircraft", [])}
    life=base.lifecycle_summary(tasks,records,services,latency)
    holding_energy=None if boundary is None else boundary.get("holding_energy_by_ring")
    quality={} if boundary is None else boundary.get("quality",{})
    return dict(directory=str(directory), endpoint_s=None if boundary is None else boundary["time"],
        system_counts=quality,holding_energy_by_ring=holding_energy,
        holding_time_by_ring=derived_holding_times(holding_energy,quality.get("holding_time"),directory/"replay/config.yaml"),
        endpoint_ring_pending=recorded_ring_pending(terminal,None if boundary is None else boundary["time"]),
        selection=selection_facts(records), normal_lifecycle=life,normal_work_by_scope=scope_work_facts(life,records),
        task_continuations=task_continuation_facts(tasks),
        admission=admission_facts(tasks,records,plans), timing=timing_facts(records),
        task_timeline=task_timelines(tasks,records,events,plans),
        nonlanding=nonlanding_facts(records,events,plans,terminal_aircraft,tasks),
        automated_audit=load(audit_override or directory/"audit_result.json", {"status":"PENDING_NOT_RUN", "full_a20_covered":False}),
        automated_audit_evidence_path=str(audit_override or directory/"audit_result.json"),
        human_replay_status="HUMAN_PENDING", replay_path=str(directory/"replay"),
        result=load(directory/"result.json", {"status":"NOT_RUN_OR_CENSORED"}))


def snapshot_comparisons(output):
    roots={}
    for root in base.ROOTS:
        rows=[load(p) for p in sorted((output/"snapshots"/root).glob("*/result.json"))]
        for row in rows:
            row["saved_scheduler_observables"]=selection_facts(row.get("scheduler_records", []))
            row["saved_exclusive_timings"]=timing_facts(row.get("scheduler_records", []))
        frames={row["frame_id"] for row in rows}; hashes={row["checkpoint_sha256"] for row in rows}
        groups=defaultdict(list)
        for row in rows:
            if row["method"] != "FULL_H": groups[row["selected_set_digest"]].append(row["offset"])
        expected={("RULE_LOCAL_K8",offset) for offset in range(3)} | {("FULL_H",0)}
        observed={(row["method"],row["offset"]) for row in rows}
        roots[root]=dict(rows=rows, expected_jobs=4, observed_jobs=len(rows),
            missing_jobs=[dict(method=method,offset=offset) for method,offset in sorted(expected-observed)],
            same_frame=len(frames)==1 if rows else None,
            same_checkpoint=len(hashes)==1 if rows else None,
            fullH_same_version=all(row["treatment"]["selective_rule_variant"] == "opportunity_cooperative_v2" for row in rows),
            offset_set_deduplication=[dict(selected_set_digest=h, offsets=sorted(offsets), independent_sets=1) for h,offsets in sorted(groups.items())],
            unique_local_sets=len(groups), fixed_formal_offset=0, formal_offset_selected_from_snapshot=False)
        roots[root]["deduplication_set_basis"]="SAVED_INITIAL_SCOPE_SELECTED_SET_DIGEST_NEVER_RETURNED_OR_EXPANDED_SCOPE"
    return roots


def supplemental_evidence_index(output):
    """Compact index only: leave canonical fullpayloads in their saved files."""
    output=Path(output);manual={};canonical={}
    for variant in ("A","B"):
        path=output/"diagnostic_scope_alternatives"/variant/"result.json";row=load(path)
        if row is None:
            manual[variant]=dict(status="UNAVAILABLE",evidence_path=str(path));continue
        record=(row.get("scheduler_records") or [{}])[0];t=record.get("selective_replanning",{})
        manual[variant]=dict(status=row.get("status"),evidence_path=str(path),root=row.get("root"),
            explicit_manual_scope_override=row.get("explicit_manual_scope_override"),
            initial_selected_ids=row.get("initial_selected_ids"),returned_selected_ids=row.get("returned_selected_ids"),
            attempted_union=row.get("attempted_union"),canonical_content_sha256=row.get("content_comparison",{}).get("canonical_sha256"),
            first_nonempty_certified=record.get("search_handoff",{}).get("first_nonempty_certified"),
            first_certified_call_return=t.get("first_certified"),preparation=row.get("preparation"),
            physical_source_unchanged=row.get("physical_source_unchanged"),physical_execution=row.get("physical_execution"),
            actual_publication=row.get("actual_publication"),source_commit=row.get("source_commit"),
            artifact_script_sha256=row.get("artifact_script_sha256"),
            constraint_checks=[observation.get("checks") for observation in row.get("manual_selection_observations",[])])
    for root in base.ROOTS:
        path=output/"identity_content_checks"/root/"result.json";row=load(path)
        if row is None:
            canonical[root]=dict(status="UNAVAILABLE",evidence_path=str(path));continue
        canonical[root]=dict(status=row.get("status"),evidence_path=str(path),
            canonical_content_equal=row.get("canonical_content_equal"),
            production_projected_version_equal=row.get("production_projected_version_equal"),
            production_frame_id_equal=row.get("production_frame_id_equal"),
            repeated_content_sha256=[r.get("content_comparison",{}).get("canonical_sha256") for r in row.get("rows",[])],
            repr_address_observations=[r.get("content_comparison",{}).get("raw_repr_has_default_memory_address") for r in row.get("rows",[])],
            fullpayload_refs=[str(path.parent/f"rebuild{index}"/"content.json") for index,_ in enumerate(row.get("rows",[]))],
            authorization_scope="CONTENT_COMPARISON_ONLY_NOT_PRODUCTION_IDENTITY_OR_CACHE_AUTHORIZATION")
    return dict(manual_scope_alternatives=manual,canonical_content_checks=canonical,
        token_identity_comparison="NOT_ASSERTED_PREPARATION_DISCARD_OBSERVED_PER_JOB",fullpayloads_embedded=False)


def report(output, followup_output):
    output=Path(output); followup_output=Path(followup_output); cells={}
    for root in base.ROOTS:
        new=output/"arms"/root/"evaluation"/root/"RULE_LOCAL_K8"
        old=followup_output/"arms"/root/"baseline20"/"evaluation"/root/"RULE_LOCAL_K8"
        available=lambda path:{int(p.stem.split("_")[-1]) for p in path.glob("boundary_*.json")}
        common=max(available(new) & available(old), default=None)
        cells[root]=dict(common_endpoint_s=common, current=saved_cell(new,common),
            baseline20=saved_cell(old,common,audit_override=output/"prior_audits"/root/"baseline20"/"audit_result.json"),
            comparison_scope="WHOLE_VERSION_BEFORE_AFTER_NOT_FULLY_FAIR_CONTROL")
    report_path=Path(__file__).resolve();test_path=report_path.parents[2]/"tests/test_rotating_slots_selective_repair_campaign.py"
    summary=dict(schema="rs_selective_repair_saved_report_v2", cells=cells, snapshots=snapshot_comparisons(output),
        supplemental_evidence=supplemental_evidence_index(output),
        postprocessing_source=dict(report_path=str(report_path),report_sha256=base.file_hash(report_path),
            test_path=str(test_path),test_sha256=base.file_hash(test_path) if test_path.is_file() else None,
            scope="SAVED_EVIDENCE_POSTPROCESSING_ONLY_PHYSICAL_RUN_SOURCE_REMAINS_ORIGINAL_BINDING"),
        budget_ledger=load(output/"repair_budget_ledger.json"), human_replay_status="HUMAN_PENDING",
        automated_audit_policy="PENDING_UNLESS_SAVED_AUDIT_RESULT_EXISTS", scientific_acceptance="PENDING_PRIMARY_REVIEW",
        no_learning=True, no_dynamic_k=True, no_lead_scan=True,
        prediction_scope="CANDIDATE_CP_SUFFIX_IS_PREDICTION_NEVER_EXECUTION")
    audit.write(output/"repair_analysis_summary.json",summary)
    text="# Selective repair v2 saved results\n\n"
    text+="Fixed K8/max12, lead20, heartbeat5, preparation1 and handoff4; offset0 is fixed for both formal runs. Baseline20 is a whole-version before/after reference. This is not a fully fair control comparison.\n\n"
    text+="| Root | Version | Endpoint s | Normal | Emergency | Energy failures | Air pending | Upstream pending |\n|---|---|---:|---:|---:|---:|---:|---:|\n"
    for root,pair in cells.items():
        for version in ("baseline20", "current"):
            row=pair[version]; q=row["system_counts"]
            text+="| "+" | ".join(map(str,(root,version,row["endpoint_s"],q.get("L_normal"),q.get("L_emergency"),q.get("F_energy_total"),q.get("N_airborne_pending"),q.get("N_upstream_pending"))))+" |\n"
    text+="\nMachine-readable evidence includes per-ring holding/energy, fairness unknown/overdue, overflow times/K/recorded hard causes, full lifecycle work (including pending partial work), separate admission denominators, exclusive phase timing and the certificate/ready/publication/start/next-snapshot chain. Null fields are unavailable evidence. Rejection labels can coexist and do not establish causation.\n\n"
    text+="Per-ring holding time is derived from executed holding energy divided by the saved configuration's canonical battery power, accepted only after its sum matches recorded total holding time. Endpoint ring counts use saved physical ring/status and retain unmapped aircraft. first_certified_call_return is the first certified local call return, not the first certificate instant; the direct first-nonempty event remains separate and first-any instant stays unknown. Manual A/B and canonical checks are compact evidence indexes, not execution or cache-authorization claims.\n\n"
    text+="Automated replay audits remain pending until an actual audit_result.json exists. Human viewing remains HUMAN_PENDING. Candidate CP suffixes are predictions; only saved executor starts and physical states are execution evidence. Snapshot preparation discards its token and never publishes.\n\n"
    text+="Open each saved replay with `ReplayController('/absolute/path/to/replay')` and `RotatingSlotsReplayViewer(controller).run()` from `terminal_marl.visualization.rotating_slots_replay`. Record actual human observations separately.\n"
    (output/"REPAIR_REPORT.md").write_text(text,encoding="utf8")
    return dict(status="REPORT_WRITTEN", summary_path=str(output/"repair_analysis_summary.json"), human_replay_status="HUMAN_PENDING")
