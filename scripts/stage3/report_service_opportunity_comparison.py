"""Read-only service-opportunity before/after extraction with physical idle.

Reuse the saved A4 summarizer for ordinary lifecycle and timing measures. Missing
certificate or physical evidence remains null, never evidence of zero waste.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
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


def candidate_lifecycle_metrics(callbacks, supplies):
    """Distinct physical candidates; catalog refreshes never enlarge denominators.

    A solve event is evidence of actual CP submission, rather than eligibility
    or inclusion in a generated catalog. Missing legacy exports stay unknown.
    """
    observed=[cb for cb in callbacks if "service_candidate_lifecycle" in cb]
    if not observed: return None
    candidates={}; malformed=0
    for cb in observed:
        for row in cb["service_candidate_lifecycle"]:
            key=row.get("candidate_hash")
            if not key: malformed+=1; continue
            item=candidates.setdefault(key,dict(row,callback_stop_reasons=[],_rows=[]))
            item["_rows"].append(row)
            item["callback_stop_reasons"].extend(cb[k] for k in ("empty_reason","stop_reason","solver_stop_reason") if cb.get(k))
    def has(item,field): return any(row.get(field) for row in item["_rows"])
    critical={key:item for key,item in candidates.items() if any(r.get("critical") is True for r in item["_rows"])}
    eligible={key:item for key,item in critical.items() if has(item,"catalog_identities") or has(item,"retained_in_catalog")}
    solved={key:item for key,item in critical.items() if has(item,"actual_solve_events")}
    single={key:item for key,item in critical.items() if any(r.get("single_accepted") is True for r in item["_rows"])}
    unsolved=Counter(); uncertified=Counter(); timing=[]
    for key,item in critical.items():
        if key not in solved:
            reasons={r["single_rejection_reason"] for r in item["_rows"] if r.get("single_rejection_reason")}
            reasons.update(item["callback_stop_reasons"])
            unsolved.update(reasons)
        if not has(item,"certified_events"):
            uncertified.update(set(item["callback_stop_reasons"]))
    for key,item in candidates.items():
        if not any(r.get("source_ring")=="middle" for r in item["_rows"]): continue
        hashes={h for r in item["_rows"] for h in r.get("action_hashes",())}
        forecasts=sorted({r["predicted_service_window"] for r in item["_rows"] if finite(r.get("predicted_service_window"))})
        for start in supplies["starts"]:
            if start["trajectory_hash"] not in hashes: continue
            timing.append(dict(candidate_hash=key,aircraft_id=start["aircraft_id"],trajectory_hash=start["trajectory_hash"],
                actual_start=start["start"],actual_completion=(start["end"] if start in supplies["completed"] else None),
                predicted_service_windows=forecasts or None,
                start_minus_predicted_window_s=[start["start"]-t for t in forecasts] or None,
                completion_minus_predicted_window_s=([start["end"]-t for t in forecasts] if start in supplies["completed"] else None)))
    joint=[dict(callback_index=i,**event) for i,cb in enumerate(callbacks)
        for event in cb.get("service_search_events",()) if event.get("event")=="JOINT_CHECKED"]
    return dict(distinct_generated_candidate_count=len(candidates),critical_generated_count=len(critical),
        critical_single_accepted_count=len(single),critical_catalog_eligible_count=len(eligible),
        critical_actual_cp_participated_count=len(solved),
        critical_catalog_eligible_actual_cp_participated_count=len(set(solved)&set(eligible)),
        critical_actual_cp_participation_rate=(len(set(solved)&set(eligible))/len(eligible) if eligible else None),
        critical_generated_actual_cp_participation_rate=(len(solved)/len(critical) if critical else None),
        participation_denominator="distinct critical candidates admitted to at least one catalog",
        unsolved_reason_counts=dict(unsolved),uncertified_reason_counts=dict(uncertified),
        reason_semantics="exported single rejection and callback stop reasons associated with candidate; no per-candidate causal attribution",
        critical_unsolved_count=len(set(critical)-set(solved)),
        critical_uncertified_count=sum(not has(item,"certified_events") for item in critical.values()),
        critical_joint_checked_count=sum(has(item,"joint_checked_events") for item in critical.values()),
        critical_certified_count=sum(has(item,"certified_events") for item in critical.values()),
        real_joint_check_count=sum(e.get("certificate_reused") is False for e in joint),
        reused_joint_certificate_count=sum(e.get("certificate_reused") is True for e in joint),
        joint_conflict_count=sum(len(e.get("conflicts",())) for e in joint),joint_check_rows=joint,
        malformed_candidate_rows=malformed,callbacks_with_lifecycle=len(observed),callbacks_without_lifecycle=len(callbacks)-len(observed),
        middle_inner_actual_timing=timing,
        timing_semantics="actual replay maneuver start/completion relative to candidate forecast service time; forecast is not an actual service or certification")


def local_recovery_metrics(tasks):
    """Read wrapper task evidence, keeping absent R2 instrumentation null."""
    if tasks is None: return None
    local=[t for t in tasks if t.get("traffic_policy")=="action_local_recovery_v1"]
    if not local: return None
    deltas=[]
    for index,task in enumerate(local):
        rows=[r["local_recovery"] for r in task.get("revalidations",()) if "local_recovery" in r]
        rows.extend(task.get("guard_local_recovery",()))
        deltas.extend(dict(task_index=index,**row) for row in rows)
    delivered=Counter(); delivered_plans=Counter(); reasons=Counter(); withdrawals=Counter()
    whole_batch=0; no_standby=0; joins=Counter(); conflicts=[]; repair_failures=[]; recovery_events=[]
    for task in local:
        reason=task.get("reason")
        if reason: reasons[reason]+=1
        committed=task.get("status")=="COMMITTED" and task.get("submitted",0)>0
        if committed:
            delivery=task.get("recovery_delivery","original")
            delivered[delivery]+=1; delivered_plans[delivery]+=task["submitted"]
        task_deltas=[r["local_recovery"] for r in task.get("revalidations",()) if "local_recovery" in r]
        task_deltas.extend(task.get("guard_local_recovery",()))
        changed=any(r.get("affected_action_hashes") for r in task_deltas)
        if changed and "finished_time" in task and not committed and task.get("submitted")==0:
            whole_batch+=1
            no_standby+=int(reason=="NO_CERTIFIED_STANDBY")
        joins.update(r["status"] for r in task.get("revalidations",()) if r.get("status"))
        # Recovery logs can repeat a revalidation receipt. Preserve the normal
        # per-check observations, then add only unseen explicit conflict rows
        # from the supplemental copies and failed repair attempts.
        seen=set()
        for row in task_deltas:
            for conflict in row.get("conflicts",()):
                conflicts.append(conflict); seen.add(json.dumps(conflict,sort_keys=True))
        extra=list(task.get("local_recovery_events",()))
        recovery_events.extend(extra)
        if isinstance(task.get("repair"),dict): extra.append(task["repair"])
        seen_repairs=set()
        for row in extra:
            for attempt in row.get("repair_attempts",()):
                identity=json.dumps(attempt,sort_keys=True)
                if attempt.get("accepted") is False and identity not in seen_repairs:
                    repair_failures.append(attempt); seen_repairs.add(identity)
            for source in (row,*row.get("repair_attempts",())):
                for conflict in source.get("conflicts",()):
                    identity=json.dumps(conflict,sort_keys=True)
                    if identity not in seen: conflicts.append(conflict); seen.add(identity)
    for row in deltas: withdrawals.update(row.get("withdrawn_action_reasons",{}).values())
    repair=[t["repair"] for t in local if isinstance(t.get("repair"),dict)]
    return dict(local_policy_task_count=len(local),local_delta_check_count=len(deltas),
        affected_action_occurrences=sum(len(r.get("affected_action_hashes",())) for r in deltas),
        retained_action_occurrences=sum(len(r.get("retained_action_hashes",())) for r in deltas),
        withdrawn_action_reason_counts=dict(withdrawals),
        recursive_release_edge_occurrences=sum(len(r.get("recursive_release_edges",())) for r in deltas),
        real_joint_pair_checks=sum(r.get("pairs_checked",0) for r in deltas),
        reused_pair_certificates=sum(r.get("reused_pairs",0) for r in deltas),
        envelope_proofs=sum(r.get("envelope_proofs",0) for r in deltas),
        real_joint_conflict_count=len(conflicts),real_joint_conflicts=conflicts,
        failed_repair_attempt_count=len(repair_failures),failed_repair_attempts=repair_failures,
        failed_repair_attempt_status_counts=dict(Counter(r.get("status") for r in repair_failures if r.get("status"))),
        failed_repair_attempt_reason_counts=dict(Counter(r.get("reason") for r in repair_failures if r.get("reason"))),
        local_change_whole_batch_rejection_count=whole_batch,
        local_change_no_certified_standby_cancellation_count=no_standby,
        delivered_task_counts={k:delivered[k] for k in ("original","standby","repaired")},
        delivered_plan_counts={k:delivered_plans[k] for k in ("original","standby","repaired")},
        task_finish_reason_counts=dict(reasons),repair_timed_out_count=sum(r.get("timed_out") is True for r in repair),
        certification_join_status_counts=dict(joins),certified_revalidation_join_count=joins["CERTIFIED"],
        timeout_reason_counts={k:v for k,v in reasons.items() if "TIMEOUT" in k or "DEADLINE" in k or "EXPIRED" in k},
        certification_join_reason_counts={k:v for k,v in reasons.items() if "CERT" in k or "REVALIDATION" in k or "JOIN" in k},
        local_delta_rows=deltas,repair_rows=repair,recovery_event_rows=recovery_events,
        semantics="action/pair/edge totals are occurrences per local delta check; extra conflict/repair log copies deduplicated within task; failed repair status alone does not establish a pair-conflict trace; delivery counts require COMMITTED and submitted>0; whole-batch rejection observes an earlier affected-action delta, with NO_CERTIFIED_STANDBY separately counted")


def future_reservation_metrics(callbacks,events,trajectories,*,lo=20.,hi=320.):
    """Join first-action trajectory identities to world-issued reservation IDs.

    A prepared event is a delivery certificate, whereas candidate certified
    events are search facts. Event times own publication/start/completion time.
    """
    names={"reservation_prepared","reservation_published","reservation_started","reservation_completed",
           "reservation_cancelled","reservation_pending_superseded","reservation_start_guard","reservation_traffic_guard"}
    reservation_events=[e for e in events if e.get("type") in names]
    final_export=any("final_selected_first_actions" in cb for cb in callbacks)
    complete_protocol=any(cb.get("delivery_certification_scope")=="effective_hold_then_absolute_first_action" for cb in callbacks)
    if not reservation_events and not final_export: return None
    metadata=dict(trajectories); first={}; suffix={}; candidates={}; final_hashes=set(); malformed=[]
    generation_export=any("service_candidate_lifecycle" in cb for cb in callbacks)
    for index,cb in enumerate(callbacks):
        for row in cb.get("service_candidate_lifecycle",()):
            actions=row.get("actions") or []
            for action in actions:
                if action.get("trajectory_hash"): metadata[action["trajectory_hash"]]=dict(action,aircraft_id=row.get("aircraft_id",action.get("aircraft_id")))
            hashes=row.get("action_hashes") or [a.get("trajectory_hash") for a in actions]
            if not hashes: continue
            candidates.setdefault(hashes[0],[]).append((index,row))
            for hash_ in hashes[1:]: suffix.setdefault(hash_,set()).add(row.get("candidate_hash"))
        for row in cb.get("final_selected_first_actions",()):
            actions=row.get("actions") or []
            for action in actions:
                if action.get("trajectory_hash"): metadata[action["trajectory_hash"]]=dict(action,aircraft_id=row.get("aircraft_id",action.get("aircraft_id")))
            action=row.get("first_action")
            if action is not None and action.get("trajectory_hash"):
                metadata[action["trajectory_hash"]]=dict(action,aircraft_id=row.get("aircraft_id",action.get("aircraft_id"))); final_hashes.add(action["trajectory_hash"])
            for action in actions[1:]:
                suffix.setdefault(action["trajectory_hash"],set()).add(row.get("candidate_hash"))
    reservations={}; replacements={}
    for e in reservation_events:
        details=e.get("details") or {}; rid=details.get("reservation_id")
        if rid and details.get("replacement_reservation_id"):
            new_id=details["replacement_reservation_id"]
            replacements[(rid,new_id)]=dict(old_reservation_id=rid,new_reservation_id=new_id,event_time=e.get("time"))
        if rid:
            for old_id in details.get("replaced_reservation_ids",()):
                replacements[(old_id,rid)]=dict(old_reservation_id=old_id,new_reservation_id=rid,event_time=e.get("time"))
        if not rid:
            malformed.append(dict(type=e["type"],time=e.get("time"),reason="reservation_id_missing"))
            continue
        item=reservations.setdefault(rid,dict(reservation_id=rid,trajectory_hash=None,aircraft_id=None,
            effective_time=None,start_time=None,end_time=None,events=[]))
        for key in ("trajectory_hash","effective_time","start_time","end_time"):
            if details.get(key) is not None: item[key]=details[key]
        if e.get("aircraft_id") is not None: item["aircraft_id"]=e["aircraft_id"]
        item["events"].append({**details,"type":e["type"],"time":e.get("time")})
    for rid,item in reservations.items():
        hash_=item["trajectory_hash"]; action=metadata.get(hash_,{})
        item["aircraft_id"]=item["aircraft_id"] or action.get("aircraft_id")
        for field,event in (("delivery_certified","reservation_prepared"),("published","reservation_published"),
            ("started","reservation_started"),("completed","reservation_completed")):
            matching=[e for e in item["events"] if e["type"]==event and finite(e.get("time"))]
            item[field]=bool(matching); item[field+"_times"]=sorted({e["time"] for e in matching})
        cancellations=[e for e in item["events"] if e["type"]=="reservation_cancelled" and finite(e.get("time"))]
        pending_cancel=[e for e in cancellations if e.get("published") is False or e.get("previous_state")=="PENDING_DELIVERY"]
        published_cancel=[e for e in cancellations if e not in pending_cancel and
            (e.get("published") is True or any(t<=e["time"] for t in item["published_times"]))]
        item["cancelled"]=bool(published_cancel)
        item["cancelled_times"]=sorted({e["time"] for e in published_cancel})
        item["pending_cancelled"]=bool(pending_cancel)
        item["pending_cancelled_times"]=sorted({e["time"] for e in pending_cancel})
        item["superseded_pending"]=any(e["type"]=="reservation_pending_superseded" for e in item["events"])
        start=item["start_time"]
        pending_discarded=not item["published"] and (item["pending_cancelled"] or item["superseded_pending"])
        item["pending_beyond_window"]=finite(start) and start>hi and not item["cancelled"] and not item["started"] and not pending_discarded
        item["due_without_observed_start"]=(item["published"] and finite(start) and start<hi and
            not item["started"] and not item["completed"] and not item["cancelled"])
        item["missed_start_observed"]=item["published"] and any(e.get("reason")=="RESERVATION_START_TIME_MISMATCH" and
            finite(e.get("time")) and finite(start) and e["time"]>start for e in item["events"])
        item["cancellation_reasons"]=sorted({e["reason"] for e in item["events"] if e["type"]=="reservation_cancelled" and e.get("reason")})
    all_hashes=set(candidates)|final_hashes|{r["trajectory_hash"] for r in reservations.values() if r["trajectory_hash"]}
    for hash_ in sorted(all_hashes):
        source=candidates.get(hash_,[]); action=metadata.get(hash_,{})
        related=[r for r in reservations.values() if r["trajectory_hash"]==hash_]
        aid=action.get("aircraft_id") or next((r.get("aircraft_id") for _,r in source if r.get("aircraft_id")),None)
        aid=aid or next((r["aircraft_id"] for r in related if r["aircraft_id"]),None)
        item=dict(trajectory_hash=hash_,aircraft_id=aid,action=action or None,
            candidate_hashes=sorted({r["candidate_hash"] for _,r in source if r.get("candidate_hash")}),
            generated=bool(source) if generation_export else None,
            actual_cp=any(r.get("actual_solve_events") for _,r in source) if generation_export else None,
            selected_any_intermediate=any(r.get("selected_events") for _,r in source) if generation_export else None,
            selected_final=(hash_ in final_hashes) if final_export else None,
            search_certified=any(r.get("certified_events") for _,r in source) if generation_export else None,
            effective_times=sorted({(callbacks[i].get("planning_timing") or {})["effective_time"] for i,_ in source
                if finite((callbacks[i].get("planning_timing") or {}).get("effective_time"))}),
            absolute_start=action.get("start_time"),absolute_end=action.get("end_time"),
            reservation_ids=sorted(r["reservation_id"] for r in related),
            candidate_reason_observations=[dict(scope="candidate",callback_index=i,candidate_hash=r.get("candidate_hash"),
                reason=r["single_rejection_reason"]) for i,r in source if r.get("single_rejection_reason")],
            callback_reason_observations=[dict(scope="callback_or_joint_group",callback_index=i,field=k,reason=callbacks[i][k])
                for i,_ in source for k in ("empty_reason","stop_reason","solver_stop_reason","actual_delivery_reason") if callbacks[i].get(k)],
            reservation_reason_observations=[dict(scope="reservation",reservation_id=r["reservation_id"],
                event_type=e["type"],time=e.get("time"),reason=e["reason"]) for r in related for e in r["events"] if e.get("reason")],
            reasons=sorted({e["reason"] for r in related for e in r["events"] if e.get("reason")}))
        for field in ("delivery_certified","published","started","completed","cancelled","pending_cancelled","superseded_pending"):
            item[field]=any(r[field] for r in related)
        item["first_action_middle_inner"]=(action.get("source_ring")=="middle" and action.get("target_ring")=="inner") if action else None
        item["also_observed_as_predicted_suffix"]=hash_ in suffix
        item["completed_or_cancelled"]=item["completed"] or item["cancelled"]
        first[hash_]=item
    stages=("generated","actual_cp","selected_final","delivery_certified","published","started","completed","cancelled")
    def funnel(rows):
        counts={field:(sum(r[field] is True for r in rows) if all(r[field] is not None for r in rows) else None) for field in stages}
        if not generation_export:
            counts["generated"]=counts["actual_cp"]=None
        if not final_export: counts["selected_final"]=None
        counts["completed_or_cancelled"]=sum(r["completed_or_cancelled"] for r in rows)
        counts["pending_cancelled"]=sum(r["pending_cancelled"] for r in rows)
        counts["superseded_pending"]=sum(r["superseded_pending"] for r in rows)
        losses=[]
        pairs=list(zip(stages[:5],stages[1:6]))+[("started","completed_or_cancelled")]
        for left,right in pairs:
            missing=[r for r in rows if r[left] is True and r[right] is False]
            losses.append(dict(from_stage=left,to_stage=right,not_observed_count=(len(missing) if counts[left] is not None and counts[right] is not None else None),
                trajectory_hashes=[r["trajectory_hash"] for r in missing],
                exported_reason_counts=dict(Counter(reason for r in missing for reason in r["reasons"])),
                associated_candidate_reason_counts=dict(Counter(o["reason"] for r in missing for o in r["candidate_reason_observations"])),
                associated_callback_reason_counts=dict(Counter(o["reason"] for r in missing for o in r["callback_reason_observations"])),
                pending_not_due_reservation_count=sum(unit["pending_beyond_window"] for r in missing for rid in r["reservation_ids"] for unit in (reservations[rid],)),
                reason_unknown_count=sum(not r["reasons"] for r in missing)))
        return dict(counts=counts,selected_any_intermediate_count=sum(r["selected_any_intermediate"] is True for r in rows),
            selected_final_not_published_count=(sum(r["selected_final"] is True and not r["published"] for r in rows) if final_export else None),transitions=losses)
    occurrences=[]
    for index,cb in enumerate(callbacks):
        timing=cb.get("planning_timing") or {}; snapshot=timing.get("snapshot_time"); effective=timing.get("effective_time")
        for selected in cb.get("final_selected_first_actions",()):
            action=selected.get("first_action")
            if not action or not action.get("trajectory_hash"): continue
            hash_=action["trajectory_hash"]; units=[r for r in reservations.values() if r["trajectory_hash"]==hash_]
            already=[]; newly_published=[]; bound=[]
            for unit in units:
                prior=[t for t in unit["published_times"] if finite(snapshot) and t<=snapshot+1.e-8]
                if prior and not any(max(prior)<=t<=snapshot+1.e-8 for t in unit["completed_times"]+unit["cancelled_times"]):
                    already.append(unit["reservation_id"])
                matches=finite(effective) and finite(unit["effective_time"]) and abs(effective-unit["effective_time"])<1.e-7
                if matches:
                    bound.append(unit["reservation_id"])
                    if finite(snapshot) and any(snapshot-1.e-8<=t<=hi for t in unit["published_times"]): newly_published.append(unit["reservation_id"])
            known=finite(snapshot) and finite(effective)
            state=("already_reserved" if already else "published_for_occurrence" if newly_published else
                "unknown_binding" if not known else "pending_effective_outside_window" if effective>hi else "not_published_by_window_end")
            occurrences.append(dict(callback_index=index,aircraft_id=selected.get("aircraft_id"),candidate_hash=selected.get("candidate_hash"),
                trajectory_hash=hash_,snapshot_time=snapshot,effective_time=effective,absolute_start=action.get("start_time"),
                absolute_end=action.get("end_time"),first_action=action,state=state,
                first_action_middle_inner=action.get("source_ring")=="middle" and action.get("target_ring")=="inner",
                already_reserved_ids=sorted(already),published_for_occurrence_ids=sorted(newly_published),bound_reservation_ids=sorted(bound),
                callback_delivery_observation=dict(scope="callback_or_joint_group",status=cb.get("actual_delivery_status"),
                    reason=cb.get("actual_delivery_reason"),time=cb.get("actual_delivery_time")),
                delivery_selection_binding=deepcopy(cb.get("delivery_selection_binding")),
                actual_delivery_selection_binding=deepcopy(cb.get("actual_delivery_selection_binding")),
                reservation_reason_observations=[dict(reservation_id=rid,event_type=e["type"],reason=e["reason"],time=e.get("time"))
                    for rid in bound for e in reservations[rid]["events"] if e.get("reason")]))
    middle=[r for r in first.values() if r["first_action_middle_inner"] is True]
    middle_funnel=funnel(middle)
    middle_funnel["denominator_excludes_unclassified_count"]=sum(r["first_action_middle_inner"] is None for r in first.values())
    postponed=[]
    for (old_id,new_id),row in replacements.items():
        one,two=reservations.get(old_id),reservations.get(new_id)
        if one and two and finite(one["start_time"]) and finite(two["start_time"]) and two["start_time"]>one["start_time"]:
            postponed.append(dict(old_reservation_id=old_id,new_reservation_id=new_id,
                old_start=one["start_time"],new_start=two["start_time"]))
    return dict(schema="future_reservation_metrics_v1",window=[lo,hi],all_first_actions=funnel(list(first.values())),
        middle_inner=middle_funnel,first_action_rows=list(first.values()),reservation_rows=list(reservations.values()),
        final_selected_occurrence_rows=occurrences,
        final_selected_occurrence_state_counts=dict(Counter(o["state"] for o in occurrences)),
        final_selected_middle_inner_occurrence_state_counts=dict(Counter(o["state"] for o in occurrences if o["first_action_middle_inner"])),
        predicted_suffix_rows=[dict(trajectory_hash=h,candidate_hashes=sorted(x for x in keys if x),
            also_first_action=h in first,action=metadata.get(h)) for h,keys in sorted(suffix.items())],
        unclassified_first_action_count=sum(r["first_action_middle_inner"] is None for r in first.values()),
        reservation_count=len(reservations),cancelled_reservation_count=sum(r["cancelled"] for r in reservations.values()),
        pending_cancelled_reservation_count=sum(r["pending_cancelled"] for r in reservations.values()),
        superseded_pending_reservation_count=sum(r["superseded_pending"] for r in reservations.values()),
        missed_start_count=sum(r["missed_start_observed"] for r in reservations.values()),
        due_without_observed_start_count=sum(r["due_without_observed_start"] for r in reservations.values()),
        pending_beyond_window_count=sum(r["pending_beyond_window"] for r in reservations.values()),
        replacement_count=len(replacements) if replacements or complete_protocol else None,
        absolute_start_postponement_count=len(postponed) if replacements or complete_protocol else None,
        replacement_rows=list(replacements.values()) if replacements or complete_protocol else None,
        postponement_rows=postponed if replacements or complete_protocol else None,
        malformed_event_rows=malformed,
        semantics="trajectory hashes count first actions; reservation IDs count issued units; primary cancellations require published evidence; pending cancellations and superseded unselected candidates are separate; callback/action occurrences retain repeated selections, already-reserved state and exact delivery selection bindings; callback/group reasons are associated observations, not per-trajectory attribution; intermediate selection and predicted suffix are separate; not_observed is not causal loss; future/cancelled reservations are not missed starts; absent replacement protocol is unknown")


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
        energy_j=delta.get("energy_j"),increment_energy_kwh=delta.get("energy_kwh"),
        emergencies=delta.get("L_emergency"),failures=delta.get("failed_before_landing"),
        holding_time_by_ring=ordinary.get("holding_time_by_ring"),holding_energy_by_ring=ordinary.get("holding_energy_by_ring"))
    if missing: return dict(result,physical_idle=None,middle_inner_supply=None,witnessed_idle=None,
        supply_gap=None,inner_stock_empty=None,replay_input_hash_checks=None,event_log_complete=None,
        first_certificate_times=None,forecast_service_terms=None,empty_reason_counts=None,
        candidate_lifecycle=None,local_recovery=None,future_reservation_metrics=None)
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
    task_path=directory/"method_normal_tasks.json"
    demand=demand_intervals(frames,20.,320.)
    normal=[e["aircraft_id"] for e in events if e.get("type")=="touchdown" and 20.<e.get("time",-1)<=320.
            and (e.get("details") or {}).get("emergency") is False]
    result["treatment_normal_touchdown_count"]=len([aid for aid in normal if aid not in set(inherited or ())]) if inherited is not None else None
    result.update(physical_idle=physical,middle_inner_supply=supplies,
        candidate_lifecycle=candidate_lifecycle_metrics(callbacks,supplies),
        local_recovery=local_recovery_metrics(load(task_path) if task_path.is_file() else None),
        future_reservation_metrics=future_reservation_metrics(callbacks,events,trajectories),
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


def saved_baseline(path):
    """Read R3's already combined new campaign, including its export lineage."""
    path=Path(path); saved=load(path)
    if saved.get("window")!=[20.,320.] or "new" not in saved.get("campaigns",{}):
        raise ValueError("saved baseline must be a t20-320 comparison with a new campaign")
    baseline=deepcopy(saved["campaigns"]["new"])
    if Path(baseline.get("input_root","")).name!="rs_local_recovery_serializationfix_20261031":
        raise ValueError("saved baseline must select the R3 serialization-fix new campaign, not R2")
    expected={(r,m) for r in ROOTS for m in METHODS}
    for rows in ((baseline.get("ordinary") or {}).get("arms",()),baseline.get("metrics",())):
        if len(rows)!=4 or {(r.get("root"),r.get("method")) for r in rows}!=expected:
            raise ValueError("saved baseline requires exactly four unique R3 new arms")
    if any(a.get("status")!="COMPLETE" for a in baseline["ordinary"]["arms"]):
        raise ValueError("saved baseline arms must be complete")
    provenance=dict(evidence_type="preexisting_artifact",source_summary=str(path.resolve()),
        source_summary_sha256=sha256(path.read_bytes()).hexdigest(),original_campaign_label="new")
    baseline["baseline_provenance"]=provenance
    for row in baseline["metrics"]:
        row.setdefault("future_reservation_metrics",None)
        row["evidence_type"]="preexisting_artifact"
    for row in baseline["ordinary"]["arms"]: row["evidence_type"]="preexisting_artifact"
    return baseline


def collect(old_root,new_root,*,old_summary=None):
    summarizer=existing_summary(); campaigns={}
    for label,path in (("old",old_root),("new",new_root)):
        if label=="old" and old_summary is not None:
            campaigns[label]=saved_baseline(old_summary); continue
        if path is None: raise ValueError("old-root or explicit old-summary required")
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
    parser.add_argument("--old-root",type=Path); parser.add_argument("--old-summary",type=Path)
    parser.add_argument("--new-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv); report=collect(args.old_root,args.new_root,old_summary=args.old_summary)
    output=write_outputs(args.output_dir,report)
    print(json.dumps(dict(output=str(output),schema=report["schema"])))
    return report


if __name__=="__main__": main()
