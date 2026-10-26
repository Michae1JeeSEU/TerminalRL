"""Read saved latency-interface artifacts; export facts and figures, never run a world."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ARMS=("heuristic_reduced","learned_reduced")
LABELS={ARMS[0]:"H",ARMS[1]:"L"}
COLORS={ARMS[0]:"#606060",ARMS[1]:"#527C9C"}
OUTPUTS=("FINAL_REPORT.md","analysis_summary.json","timeline.csv","boundary_quality.csv",
         "replay_sample_errors.csv","planning_timeline.svg","planning_timeline.png",
         "figure_contract.json","figure_QA.json")
CONTACT_OUTPUTS=("contact_positions.csv","saved_xy_contact.svg","saved_xy_contact.png")
AIRBORNE={"INBOUND","HOLD_OUTER","HOLD_MIDDLE","HOLD_INNER","TRANSFER_O_M","TRANSFER_M_I",
          "FINAL_NORMAL","FINAL_EMERGENCY","VERTICAL_DESCENT"}


def load(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def optional(path,default=None): return load(path) if Path(path).is_file() else default


def file_hash(path):
    h=sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    with Path(path).open("x",encoding="utf8") as stream: json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)


def finite(value): return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def stats(values):
    values=sorted(float(x) for x in values if finite(x))
    return dict(count=len(values),total_s=math.fsum(values),p95_s=values[max(0,math.ceil(.95*len(values))-1)] if values else None,
                max_s=max(values) if values else None,rule="NEAREST_RANK")


def same_known(values): return None if any(v is None for v in values) else all(v==values[0] for v in values)


def json_lines(path):
    with Path(path).open(encoding="utf8") as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def saved_core(checkout):
    """Only replay validation/serialized curve evaluation is exposed here."""
    if checkout is None or not Path(checkout).is_dir(): return None
    checkout=Path(checkout).resolve(); sys.path.insert(0,str(checkout))
    from terminal_marl.core.rotating_slots import replay,trajectory,config
    if not all(Path(m.__file__).resolve().is_relative_to(checkout) for m in (replay,trajectory,config)):
        raise ValueError("replay modules loaded outside the selected frozen checkout")
    return dict(validate=replay.ReplayBundle.validate,config=config.load_config,
                decode=lambda raw,c:trajectory.TrajectoryPlan.from_dict(raw,config=c))


def compact_callbacks(value):
    drop={"catalogs","coverage_by_callback","final_selection_only"}
    result={k:v for k,v in value.items() if k not in drop}
    catalogs=value.get("catalogs",[])
    observed=[r["pruned_columns"] for r in catalogs if finite(r.get("pruned_columns"))]
    result["pruned_columns_observed_total"]=sum(observed) if observed else None
    return result


def task_facts(tasks,callbacks,observed_time):
    by_time={r["planning_timing"]["snapshot_time"]:r for r in callbacks if r.get("planning_timing")}
    components=("normal_work_s","prepare_work_s","revalidation_work_s","dependency_guard_work_s","ready_registration_bookkeeping_s",
                "publication_bookkeeping_wall_s","normal_evidence_wall_s")
    timeline=[]
    for index,task in enumerate(tasks):
        callback=by_time.get(task["snapshot_time"],{})
        ready=task.get("ready_time"); deadline=task.get("ready_deadline")
        pending="finished_time" not in task
        timely=(task.get("status") in ("PREPARED","COMMITTED") and finite(ready) and finite(deadline) and ready<=deadline)
        past=bool(pending and finite(observed_time) and finite(deadline) and observed_time>deadline and not timely)
        known_late=bool(pending and finite(ready) and finite(deadline) and ready>deadline)
        row=dict(task_index=index+1,snapshot_time=task["snapshot_time"],result_time=task.get("result_time"),
            ready_time=ready,ready_deadline=deadline,effective_time=task.get("effective_time"),finished_time=task.get("finished_time"),
            status=task["status"],reason=task.get("reason"),submitted=task.get("submitted"),pending=pending,
            nonempty_proposed=callback.get("deferred") is True and callback.get("noop") is False,
            revalidation_count=task.get("revalidation_count"),revalidations=task.get("revalidations"),
            revalidation_work_cpu_s=task.get("revalidation_work_cpu_s"),
            dependency_guard_work_cpu_s=task.get("dependency_guard_work_cpu_s"),
            ready_registration_bookkeeping_cpu_s=task.get("ready_registration_bookkeeping_cpu_s"),
            timely_registered_ready=timely,pending_past_ready_deadline=past,pending_known_late_ready_time=known_late,
            full_observed_planning_work_s=math.fsum(task.get(k) or 0. for k in components))
        row.update({k:task.get(k) for k in components}); timeline.append(row)
    proposed=[r for r in timeline if r["nonempty_proposed"]]
    finished=[r for r in proposed if not r["pending"]]
    committed=[r for r in finished if r["status"]=="COMMITTED" and (r.get("submitted") or 0)>0]
    return timeline,dict(tasks=len(tasks),status_counts=dict(Counter(r["status"] for r in timeline)),
        nonempty_offered=len(proposed),finished_nonempty=len(finished),committed_nonempty=len(committed),
        pending_nonempty=sum(r["pending"] for r in proposed),pending_tasks=sum(r["pending"] for r in timeline),
        actual_joined_groups=len(committed),actual_submitted_plans=sum(r.get("submitted") or 0 for r in committed),
        submitted_plans_by_joined_group=[r["submitted"] for r in committed],revalidation=revalidation_summary(tasks),
        join_rate=len(committed)/len(finished) if finished else None,join_denominator="finished nonempty offered groups; pending/no-prefix excluded",
        all_task_observed_work=stats(r["full_observed_planning_work_s"] for r in timeline),
        finished_task_complete_work=stats(r["full_observed_planning_work_s"] for r in timeline if not r["pending"]),
        components={k:stats(r.get(k) for r in tasks) for k in components},
        ready_guard_already_in_preparation=stats(r.get("ready_guard_work_s") for r in tasks),
        ready_registration_bookkeeping_cpu=stats(r.get("ready_registration_bookkeeping_cpu_s") for r in tasks),
        ready_deadline_rejections=sum(r.get("reason")=="READY_DEADLINE_MISSED" for r in tasks),
        traffic_changed_after_deadline_rejections=sum(r.get("reason")=="TRAFFIC_CHANGED_AFTER_DEADLINE" for r in tasks),
        external_traffic_missing_rejections=sum(r.get("reason")=="EXTERNAL_TRAFFIC_MISSING" for r in tasks),
        packet_expiry_rejections=sum(r.get("reason")=="LATENCY_PACKET_EXPIRED" for r in tasks),
        revalidation_work_deadline_rejections=sum(r.get("reason")=="REVALIDATION_WORK_DEADLINE" for r in tasks),
        pending_past_ready_deadline=sum(r["pending_past_ready_deadline"] for r in timeline),
        pending_known_late_ready_time=sum(r["pending_known_late_ready_time"] for r in timeline),
        rejection_reasons=dict(Counter(r.get("reason") for r in tasks if r["status"]=="REJECTED")),
        no_prefix=sum(r.get("reason")=="NO_EXECUTABLE_PREFIX" for r in tasks),
        old_5s_complete_work_overruns=sum(r["full_observed_planning_work_s"]>5. for r in timeline if not r["pending"]),
        old_5s_observed_work_overruns=sum(r["full_observed_planning_work_s"]>5. for r in timeline),
        old_callback_budget_misses=sum(bool(r.get("deadline_miss")) for r in callbacks))


def revalidation_summary(tasks):
    observed=[r for r in tasks if "revalidations" in r or "revalidation_count" in r]
    rows=[event for task in tasks for event in task.get("revalidations",[])]
    def total(key):
        values=[r[key] for r in rows if finite(r.get(key))]
        return math.fsum(values) if values else None
    counts=Counter(r.get("status","MISSING") for r in rows)
    return dict(availability="OBSERVED" if observed else "UNAVAILABLE_LEGACY_FIELDS",
        tasks_with_fields=len(observed),tasks_missing_fields=len(tasks)-len(observed),
        declared_count=sum(r["revalidation_count"] for r in tasks if "revalidation_count" in r) if observed else None,
        recorded_rounds=len(rows) if observed else None,status_counts=dict(counts),
        certified_rounds=counts["CERTIFIED"] if observed else None,
        rejected_rounds=counts["REJECTED"] if observed else None,late_rounds=counts["LATE"] if observed else None,
        pending_rounds=sum(r.get("status") in ("COMPUTING","GUARDING") and r.get("completed_time") is None for r in rows) if observed else None,
        affected_aircraft_ids=sorted({aid for r in rows for aid in r.get("affected_aircraft_ids",[])}) if observed else None,
        pairs_checked=total("pairs_checked"),reused_pairs=total("reused_pairs"),
        full_fallback_rounds=sum(bool(r["full_fallback"]) for r in rows if "full_fallback" in r) if any("full_fallback" in r for r in rows) else None,
        wall=stats(r.get("wall_s") for r in rows),cpu=stats(r.get("cpu_s") for r in rows))


def search_reuse_summary(callbacks):
    cp=[row for callback in callbacks for row in callback.get("cp_continuation",[])]
    rank=[r["rank_batch_cache"] for r in callbacks if "rank_batch_cache" in r]
    budgets=[r["ready_budget"] for r in callbacks if "ready_budget" in r]
    solvers=[r for callback in callbacks for r in callback.get("solver",[])]
    def total(rows,key):
        values=[r[key] for r in rows if finite(r.get(key)) or isinstance(r.get(key),bool)]
        return sum(values) if values else None
    misses=Counter()
    for row in rank: misses.update(row.get("miss_reasons",{}))
    drifts=[r["max_budget_fraction_drift"] for r in rank if finite(r.get("max_budget_fraction_drift"))]
    return dict(cp_observed_calls=len(cp),cp_callbacks_with_fields=sum("cp_continuation" in r for r in callbacks),
        cp={key:total(cp,key) for key in ("state_reused","state_invalidated","caller_incumbent_accepted",
            "pair_cuts_reused","single_cuts_reused","prefix_nogoods_reused","single_results_reused","solver_internal_state_reused")},
        incumbent_sources=dict(Counter(r["incumbent_source"] for r in cp if "incumbent_source" in r)),
        rank_callbacks_with_fields=len(rank),rank_cache_hits=total(rank,"hits"),rank_cache_miss_reasons=dict(misses) if rank else None,
        max_budget_fraction_drift=max(drifts) if drifts else None,
        ready_budget_callbacks=len(budgets),ready_budget_remaining_at_start=stats(r.get("remaining_at_start_s") for r in budgets),
        ready_budget_remaining_at_return=stats(r.get("remaining_at_return_s") for r in budgets),
        solver_allocated_seconds=stats(r.get("allocated_seconds") for r in solvers),
        solver_remaining_deadline_s=stats(r.get("remaining_deadline_s") for r in solvers))


def audit_replay(base,callbacks,core=None):
    """Hash/check existing files and evaluate only their saved serialized curves."""
    root=base/"replay"; index_path=root/"replay.json"
    unavailable=dict(status="UNAVAILABLE_REPLAY",c2_status="UNAVAILABLE",human_acceptance="HUMAN_PENDING")
    if not index_path.is_file(): return unavailable,[],[]
    try:
        index=load(index_path)
        hashes={name:file_hash(root/name)==expected for name,expected in index.get("files",{}).items()
                if Path(name).name==name and (root/name).is_file()}
        valid_names=len(hashes)==len(index.get("files",{})) and bool(hashes)
        plans=load(root/"trajectories.json"); events=list(json_lines(root/"events.jsonl"))
        starts=[r for r in events if r.get("type")=="maneuver_started"]
        start_rows=[]
        for event in starts:
            h=event.get("details",{}).get("trajectory_hash"); p=plans.get(h)
            start_rows.append(dict(trajectory_hash=h,time=event["time"],plan_present=p is not None,
                start_time_matches=(abs(event["time"]-p["start_time"])<=1e-8 if p else False)))
        by_hash={r["trajectory_hash"]:r for r in start_rows}
        groups=[]; acknowledged=set()
        for r in callbacks:
            timing=r.get("planning_timing")
            if not timing: continue
            hs=r.get("actual_submitted_trajectory_hashes",[])
            if r.get("actual_delivery_status")=="COMMITTED" and hs:
                acknowledged.update(hs)
                groups.append(dict(snapshot_time=timing["snapshot_time"],status="CHECKED",hash_count=len(hs),
                    complete_same_time=all(h in by_hash and by_hash[h]["start_time_matches"] and
                        abs(by_hash[h]["time"]-timing["effective_time"])<=1e-8 for h in hs)))
            else:
                groups.append(dict(snapshot_time=timing["snapshot_time"],status="UNAVAILABLE_PROPOSED_HASHES",
                    ack_hash_count=len(hs),ack_submitted=r.get("submitted")))
        archive=None; config=None; decoded={}; validation_error=None
        if core is not None:
            try:
                archive=core["validate"](root); config=core["config"](root/"config.yaml")
                decoded={h:core["decode"](p,config) for h,p in plans.items()}
            except Exception as exc: validation_error=dict(type=type(exc).__name__,message=str(exc))
        errors=[]; contacts=[]; frames=0; initial_normal=set(); previous={}; c2=[]; checked_c2=set()
        committed_starts=[e for e in starts if e.get("details",{}).get("trajectory_hash") in acknowledged]
        for frame in json_lines(root/"states.jsonl"):
            t=frame["time"]; frames+=1
            if frames==1:
                initial_normal={a.get("trajectory_hash") for a in frame["aircraft"] if plans.get(a.get("trajectory_hash"),{}).get("kind") in ("TRANSFER","FINAL_NORMAL")}
            for event in committed_starts:
                h=event["details"]["trajectory_hash"]
                if h in checked_c2 or t<event["time"]: continue
                checked_c2.add(h); aid=event.get("aircraft_id"); prev=previous.get(aid)
                row=dict(trajectory_hash=h,time=event["time"],status="UNAVAILABLE_PREDECESSOR")
                if (decoded and abs(t-event["time"])<=1e-8 and prev and prev[1].get("status") in AIRBORNE
                        and prev[1].get("trajectory_hash") in decoded and prev[1]["trajectory_hash"]!=h
                        and not any(e.get("aircraft_id")==aid and prev[0]<e["time"]<event["time"] for e in starts)):
                    before=decoded[prev[1]["trajectory_hash"]].state(event["time"]); after=decoded[h].state(event["time"])
                    row.update(status="CHECKED",predecessor_hash=prev[1]["trajectory_hash"],
                        **{k+"_error":math.dist(getattr(before,k),getattr(after,k)) for k in ("p","v","a")})
                c2.append(row)
            for aircraft in frame["aircraft"]:
                if (any(abs(t-target)<=1e-8 for target in (0.,60.,120.))
                        and len(aircraft.get("p",()))>=2 and all(finite(x) for x in aircraft["p"][:2])):
                    contacts.append(dict(time=t,aircraft_id=aircraft["id"],status=aircraft["status"],
                                         x=aircraft["p"][0],y=aircraft["p"][1]))
                if aircraft.get("status") in AIRBORNE:
                    h=aircraft.get("trajectory_hash")
                    row=dict(time=t,aircraft_id=aircraft["id"],trajectory_hash=h,p_error=None,v_error=None,a_error=None)
                    if h not in plans: row["status"]="MISSING_APPROVED_PLAN"
                    elif h not in decoded: row["status"]="UNAVAILABLE_CURVE_EVALUATOR"
                    elif not all(len(aircraft.get(k,()))==3 for k in ("p","v","a")): row["status"]="MISSING_PVA"
                    else:
                        expected=decoded[h].state(t)
                        row.update(status="CHECKED",**{k+"_error":math.dist(aircraft[k],getattr(expected,k)) for k in ("p","v","a")})
                    errors.append(row)
                previous[aircraft["id"]]=(t,aircraft)
        actual_normal={r["details"]["trajectory_hash"] for r in starts if plans.get(r["details"].get("trajectory_hash"),{}).get("kind") in ("TRANSFER","FINAL_NORMAL")}
        max_errors={k:max((r[k+"_error"] for r in errors if r[k+"_error"] is not None),default=None) for k in ("p","v","a")}
        checked=[r for r in errors if r["status"]=="CHECKED"]
        failed=any((r["p_error"]>1e-6 or r["v_error"]>1e-7 or r["a_error"]>1e-7) for r in checked)
        result=dict(status="CHECKED" if core is not None and archive is not None else "PARTIAL_CORE_VALIDATION_UNAVAILABLE",
            code_commit=index.get("code_commit"),file_hashes_match=valid_names and all(hashes.values()),files_checked=len(hashes),
            trajectory_index_matches=sorted(index.get("trajectory_hashes",[]))==sorted(plans),
            replay_bundle_validate="PASS" if archive is not None else "FAILED" if validation_error else "UNAVAILABLE_CHECKOUT",
            validation_error=validation_error,state_frames=frames,sample_rows=len(errors),checked_pva_samples=len(checked),
            sample_status_counts=dict(Counter(r["status"] for r in errors)),max_pva_errors=max_errors,pva_within_tolerance=not failed if checked else None,
            actual_starts=len(starts),start_time_mismatch_count=sum(not r["start_time_matches"] for r in start_rows),
            authoritative_trajectory_hashes=sorted(plans),
            committed_group_checks=groups,normal_start_coverage=dict(actual_normal=len(actual_normal),acknowledged=len(acknowledged),
                t0_approved_normal=len(initial_normal),unattributed_hashes=sorted(actual_normal-acknowledged-initial_normal),
                acknowledged_not_started=sorted(acknowledged-actual_normal),t0_normal_not_started=sorted(initial_normal-actual_normal),
                actual_equals_ack_plus_t0=actual_normal==(acknowledged|initial_normal),actual_normal_hashes=sorted(actual_normal)),c2_checks=c2,
            c2_status="CHECKED_WHERE_UNIQUE_PREDECESSOR" if any(r["status"]=="CHECKED" for r in c2) else "UNAVAILABLE_PREDECESSOR",
            human_acceptance="HUMAN_PENDING")
        return result,errors,contacts
    except Exception as exc:
        return dict(status="FAILED_ARTIFACT_READ",error_type=type(exc).__name__,error=str(exc),human_acceptance="HUMAN_PENDING"),[],[]


def collect(output,core=None):
    summary=load(output/"summary.json"); scope=load(output/"execution_scope.json")
    roots=optional(output/"roots/s3r23/root.json",{})
    cells={}; timeline=[]; quality=[]; errors=[]; contacts=[]; identities=[]
    times={arm:{int(p.stem.split("_")[-1]):p for p in (output/"evaluation"/arm).glob("boundary_*.json")} for arm in ARMS}
    common=max(set(times[ARMS[0]]) & set(times[ARMS[1]]),default=0)
    for arm in ARMS:
        base=output/"evaluation"/arm; result=load(base/"result.json")
        terminal=optional(base/"terminal_observation.json",result.get("terminal_observation",{}))
        tasks=optional(base/"normal_tasks.json",[]); callbacks=optional(base/"scheduler_records.json",[])
        dispatch=optional(base/"dispatch_timings.json",[]); services=optional(base/"online_services.json",[])
        observed=terminal.get("observed_time_s")
        rows,life=task_facts(tasks,callbacks,observed)
        timeline.extend(dict(arm=arm,**r) for r in rows)
        for t,path in sorted(times[arm].items()):
            q=load(path)["quality"]
            quality.append(dict(arm=arm,time_s=t,**{k:q.get(k) for k in
                ("L_normal","L_emergency","F_energy_total","N_airborne_pending","N_admitted_total","E_terminal_J","holding_time")}))
        replay,sample_rows,points=audit_replay(base,callbacks,core)
        errors.extend(dict(arm=arm,**r) for r in sample_rows); contacts.extend(dict(arm=arm,**r) for r in points)
        boundary=load(times[arm][common]) if common else None
        costs=result.get("costs",{}); terminal_costs=result.get("terminal_costs",{})
        actual_scope=result.get("execution_scope",{})
        identities.append(dict(arm=arm,source_commit=actual_scope.get("source_commit"),source_hash=actual_scope.get("source_hash"),
            manifest=result.get("shared_manifest_hash"),initial=result.get("initial_state_hash"),fingerprint=result.get("model_fingerprint")))
        cells[arm]=dict(status=result["status"],completed_boundary_s=result.get("completed_boundary_s"),
            terminal_observed_time_s=observed,common_quality=boundary["quality"] if boundary else None,
            terminal=terminal,receipt=optional(output/"receipts"/(arm+".json"),{}),job=optional(base/"job.finished.json",{}),
            actual_cpu=result.get("actual_cpu"),source_commit=actual_scope.get("source_commit"),source_hash=actual_scope.get("source_hash"),
            normal_lifecycle=life,search_reuse=search_reuse_summary(callbacks),all_callbacks=compact_callbacks(result.get("all_callbacks",{})),
            dispatch_cost=stats(r.get("dispatch_wall_s") for r in dispatch),dispatch_cpu=stats(r.get("dispatch_cpu_s") for r in dispatch),
            world_normal_dispatch_cost=stats(r.get("normal_wall_s") for r in dispatch),
            online_service={kind:stats(r.get("wall_s") for r in services if r.get("kind")==kind) for kind in sorted({r["kind"] for r in services})},
            clocks=dict(boundary_job_elapsed_s=costs.get("full_online_wall_s"),terminal_job_elapsed_s=terminal_costs.get("full_online_wall_s"),
                full_job_including_replay_s=result.get("full_job_wall_s"),receipt_wall_s=optional(output/"receipts"/(arm+".json"),{}).get("wall_s"),
                world_advance_wall_s=terminal_costs.get("world_advance_wall_s"),
                simulation_integration_and_other_executor_wall_s=terminal_costs.get("simulation_integration_and_other_executor_wall_s")),
            replay=replay,result_sha256=file_hash(base/"result.json"))
    checks=dict(common_boundary_matches_saved_summary=summary.get("common_completed_boundary_s")==common,
        same_manifest=same_known([r["manifest"] for r in identities]+[scope.get("shared_manifest_hash"),roots.get("manifest_hash")]),
        same_initial_state=same_known([r["initial"] for r in identities]+[scope.get("initial_state_hash"),roots.get("initial_state_hash")]),
        same_source=same_known([r["source_hash"] for r in identities]+[scope.get("source_hash")]),
        same_commit=same_known([r["source_commit"] for r in identities]+[scope.get("source_commit")]),
        original_model_fingerprint_preserved=same_known([r["fingerprint"] for r in identities]+[scope.get("model_fingerprint")]))
    checks["same_actual_normal_start_hashes"]=same_known([cells[arm]["replay"].get("normal_start_coverage",{}).get("actual_normal_hashes") for arm in ARMS])
    checks["same_authoritative_trajectory_hash_sets"]=same_known([cells[arm]["replay"].get("authoritative_trajectory_hashes") for arm in ARMS])
    checks["replay_execution_commit_matches"]=same_known([cells[arm]["replay"].get("code_commit") for arm in ARMS]+[scope.get("source_commit")])
    model=output/"model/model.pt"; normalizer=output/"model/feature_normalizer.json"
    value=dict(schema="normal_latency_analysis_v1",scope=scope,common_completed_boundary_s=common,cells=cells,identity_checks=checks,
        model_sha256=file_hash(model) if model.exists() else None,normalizer_sha256=file_hash(normalizer) if normalizer.exists() else None,
        scientific_interpretation="PENDING_PRIMARY_REVIEW",human_replay_acceptance="HUMAN_PENDING",
        c2_limit="Only uniquely identified saved predecessors are checked; unavailable cases are not claimed continuous. Small-world C2 evidence is separate.")
    return value,timeline,quality,errors,contacts


def write_csv(path,rows,fields):
    with Path(path).open("x",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def figures(output,value,timeline,quality,contacts,contact_sheet=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":10,"axes.labelsize":9,
        "legend.fontsize":8,"svg.fonttype":"none","pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    xmax=max([value["scope"].get("protocol",{}).get("target_sim_seconds",120)]+[r["effective_time"] for r in timeline if finite(r.get("effective_time"))])
    fig,axes=plt.subplots(3,1,figsize=(10,8),gridspec_kw=dict(height_ratios=[1,1,1.2]))
    for panel,arm in enumerate(ARMS):
        ax=axes[panel]; rows=[r for r in timeline if r["arm"]==arm]
        end=value["cells"][arm]["terminal_observed_time_s"]
        for i,r in enumerate(rows):
            finish=r["finished_time"] if finite(r["finished_time"]) else end
            if finite(finish): ax.plot([r["snapshot_time"],finish],[i,i],color=COLORS[arm],ls="--" if r["pending"] else "-",lw=1.5)
            for key,marker,dy in (("snapshot_time","o",0),("result_time","s",0),("ready_time","^",0),
                                  ("ready_deadline","|",.16),("effective_time","D",-.16)):
                if finite(r.get(key)):
                    hollow=key in ("effective_time",) or key=="ready_time" and finite(end) and r[key]>end
                    ax.plot(r[key],i+dy,marker=marker,ms=5,color=COLORS[arm],mfc="white" if hollow else COLORS[arm],linestyle="none")
            if finite(finish):
                ax.plot(finish,i,marker=">" if r["pending"] else "*" if r["status"]=="COMMITTED" else "x",ms=8,color=COLORS[arm])
            for event in r.get("revalidations") or []:
                start=event.get("started_time"); done=event.get("completed_time")
                if finite(start):
                    stop=done if finite(done) else end
                    if finite(stop): ax.plot([start,stop],[i+.10,i+.10],color="#A35C33",lw=3,alpha=.7)
                    ax.plot(start,i+.10,marker="+",ms=5,color="#A35C33")
            label=r["status"] if not r.get("reason") else r["reason"]
            ax.text(.99,i,label,transform=ax.get_yaxis_transform(),ha="right",va="bottom",fontsize=7)
        ax.set(title=f"({chr(97+panel)}) {LABELS[arm]} normal-task delivery",xlim=(0,xmax+2),ylim=(-.6,max(len(rows)-.4,.6)),
               yticks=range(len(rows)),yticklabels=[f"Task {r['task_index']}" for r in rows],xlabel="Simulation time (s)")
        if finite(end): ax.axvline(end,color="#A0A0A0",ls=":",lw=.8)
        if finite(end) and end<xmax: ax.axvspan(end,xmax+2,color="#EEEEEE",zorder=-1)
    ax=axes[2]
    for arm in ARMS:
        rows=[r for r in quality if r["arm"]==arm]
        for field,marker,style in (("L_normal","o" if arm==ARMS[0] else "s","--" if arm==ARMS[0] else "-"),
                                   ("L_emergency","^" if arm==ARMS[0] else "v",":" if arm==ARMS[0] else "-.")):
            observed=[r for r in rows if finite(r.get(field))]
            ax.plot([r["time_s"] for r in observed],[r[field] for r in observed],color=COLORS[arm],marker=marker,
                    ms=5 if arm==ARMS[0] else 3,mfc="white" if arm==ARMS[0] else COLORS[arm],
                    lw=2. if arm==ARMS[0] else 1.1,ls=style,
                    label=LABELS[arm]+(" normal" if field=="L_normal" else " emergency"))
    ax.set(title="(c) Saved 5 s boundary cumulative landings",xlabel="Simulation time (s)",ylabel="Aircraft",xlim=(0,xmax+2),ylim=(0,None))
    ax.legend(loc="upper left",ncol=2,frameon=False)
    handles=[Line2D([],[],color="#606060",marker=m,ls="none",mfc="white" if m=="D" else "#606060",label=label)
             for m,label in (("o","Snapshot"),("s","Actual result"),("^","Recorded ready"),("|","Ready deadline"),
                             ("D","Effective"),("*","Committed"),("x","Rejected / no-prefix"),(">","Censored pending"),
                             ("+","Revalidation start"))]
    fig.legend(handles=handles,loc="upper center",ncol=4,bbox_to_anchor=(.5,.998),frameon=False)
    fig.subplots_adjust(top=.87,bottom=.07,left=.09,right=.98,hspace=.57)
    fig.savefig(output/"planning_timeline.svg"); fig.savefig(output/"planning_timeline.png",dpi=300); plt.close(fig)
    made_contact=False
    if contact_sheet and contacts:
        write_csv(output/"contact_positions.csv",contacts,("arm","time","aircraft_id","status","x","y"))
        fig,axes=plt.subplots(2,3,figsize=(10,6),sharex=True,sharey=True)
        bound=max(100.,max(max(abs(r["x"]),abs(r["y"])) for r in contacts)*1.05)
        for i,arm in enumerate(ARMS):
            for j,t in enumerate((0.,60.,120.)):
                ax=axes[i,j]; rows=[r for r in contacts if r["arm"]==arm and r["time"]==t]
                for status in sorted({r["status"] for r in rows}):
                    points=[r for r in rows if r["status"]==status]
                    marker="o" if status.startswith("HOLD") else "v" if "FINAL" in status or status=="VERTICAL_DESCENT" else "s" if status not in AIRBORNE else ">"
                    ax.scatter([r["x"] for r in points],[r["y"] for r in points],s=18,color=COLORS[arm],marker=marker)
                ax.set(title=f"{LABELS[arm]} saved t={t:g}s; n={len(rows)}",xlim=(-bound,bound),ylim=(-bound,bound),xlabel="x (m)",ylabel="y (m)")
                ax.set_aspect("equal"); ax.axhline(0,color="#DDDDDD",lw=.5); ax.axvline(0,color="#DDDDDD",lw=.5)
                if not rows: ax.text(0,0,"SAVED FRAME UNAVAILABLE",ha="center",fontsize=7)
        fig.tight_layout(); fig.savefig(output/"saved_xy_contact.svg"); fig.savefig(output/"saved_xy_contact.png",dpi=300); plt.close(fig); made_contact=True
    return made_contact


def fmt(value): return "unavailable" if value is None else f"{value:.6g}" if isinstance(value,float) else str(value)


def report_text(value,timeline=()):
    cells=value["cells"]; h,l=(cells[a] for a in ARMS)
    life={a:cells[a]["normal_lifecycle"] for a in ARMS}
    def phase(arm,key): return cells[arm]["all_callbacks"].get("exclusive_phase_timings",{}).get(key,{}).get("wall_s")
    def counts(values): return "; ".join(f"{k}: {v}" for k,v in values.items()) or "none recorded"
    lines=["# Normal planning latency interface — saved-artifact report","",
        "Final interpretation: PENDING_PRIMARY_REVIEW. Figure review and replay human acceptance remain pending.","",
        f"Common completed boundary: {value['common_completed_boundary_s']} s. Execution source `{value['scope'].get('source_commit')}` / `{value['scope'].get('source_hash')}`.","",
        "One preregistered H/L pair. H is heuristic candidate order with the same CP-SAT backend as L. Detailed observations and checks are in [analysis_summary.json](analysis_summary.json); raw inputs are unchanged.","",
        "## Quality and timing","","| Observed metric | H | L |","| --- | ---: | ---: |"]
    metrics=[("Terminal time (s) / process RC",lambda a:f"{fmt(cells[a]['terminal_observed_time_s'])} / {fmt(cells[a]['receipt'].get('exit_code'))}"),
        ("Common normal / emergency / energy-failed",lambda a:" / ".join(fmt((cells[a]['common_quality'] or {}).get(k)) for k in ("L_normal","L_emergency","F_energy_total"))),
        ("Common airborne pending / admitted",lambda a:" / ".join(fmt((cells[a]['common_quality'] or {}).get(k)) for k in ("N_airborne_pending","N_admitted_total"))),
        ("Common cumulative energy (J)",lambda a:fmt((cells[a]['common_quality'] or {}).get("E_terminal_J"))),
        ("Common holding (aircraft-s)",lambda a:fmt((cells[a]['common_quality'] or {}).get("holding_time"))),
        ("Nonempty offered / finished / committed / pending",lambda a:" / ".join(str(life[a][k]) for k in ("nonempty_offered","finished_nonempty","committed_nonempty","pending_nonempty"))),
        ("Join rate: committed / finished nonempty",lambda a:fmt(life[a]["join_rate"])),
        ("Actual joined groups / submitted plans",lambda a:f"{life[a]['actual_joined_groups']} / {life[a]['actual_submitted_plans']}"),
        ("Submitted plans in each joined group",lambda a:str(life[a]["submitted_plans_by_joined_group"])),
        ("All-task observed planning work (s; pending included)",lambda a:fmt(life[a]["all_task_observed_work"]["total_s"])),
        ("Finished complete planning work: count / P95 (s)",lambda a:f"{life[a]['finished_task_complete_work']['count']} / {fmt(life[a]['finished_task_complete_work']['p95_s'])}"),
        ("Dispatch total / P95 (s)",lambda a:f"{fmt(cells[a]['dispatch_cost']['total_s'])} / {fmt(cells[a]['dispatch_cost']['p95_s'])}"),
        ("Job including replay / subprocess receipt (s)",lambda a:f"{fmt(cells[a]['clocks']['full_job_including_replay_s'])} / {fmt(cells[a]['clocks']['receipt_wall_s'])}"),
        ("Ready-deadline rejected / unready pending past deadline",lambda a:f"{life[a]['ready_deadline_rejections']} / {life[a]['pending_past_ready_deadline']}"),
        ("Traffic changed after deadline / external traffic missing",lambda a:f"{life[a]['traffic_changed_after_deadline_rejections']} / {life[a]['external_traffic_missing_rejections']}"),
        ("Packet expiry rejections",lambda a:fmt(life[a]['packet_expiry_rejections'])),
        ("Revalidation compute-budget exhaustion rejections",lambda a:fmt(life[a]['revalidation_work_deadline_rejections'])),
        ("Old 5 s observed-work overruns / callback budget misses",lambda a:f"{life[a]['old_5s_observed_work_overruns']} / {life[a]['old_callback_budget_misses']}")]
    lines.extend(f"| {label} | {fn(ARMS[0])} | {fn(ARMS[1])} |" for label,fn in metrics)
    lines += ["","## Computation and search","","| Measured phase or work | H | L |","| --- | ---: | ---: |"]
    for key,label in (("cp_solve","CP solve (s)"),("prefix_validation","Prefix validation (s)"),
                      ("trajectory_validation","Trajectory validation (s)"),("stage3_features","Learning features (s)"),
                      ("stage3_inference","Model inference (s)")):
        lines.append(f"| {label} | {fmt(phase(ARMS[0],key))} | {fmt(phase(ARMS[1],key))} |")
    for key,label in (("physical_attempts","Physical attempts"),("descriptor_queries","Descriptor queries"),
                      ("catalog_round_count","Catalog rounds"),("pruned_columns_observed_total","Observed pruned columns"),
                      ("rank_learned","Learned ranks"),("rank_cache_hits","Cached learned ranks"),
                      ("coverage_missing_full_callback_aircraft","Missing full-chain callback-aircraft observations")):
        lines.append(f"| {label} | {fmt(h['all_callbacks'].get(key))} | {fmt(l['all_callbacks'].get(key))} |")
    for key,label in (("state_reused","CP calls reusing state"),("state_invalidated","CP calls invalidating state"),
                      ("caller_incumbent_accepted","Accepted caller incumbents"),("pair_cuts_reused","Reused pair cuts"),
                      ("single_cuts_reused","Reused single cuts"),("prefix_nogoods_reused","Reused prefix nogoods"),
                      ("single_results_reused","Reused single results"),("solver_internal_state_reused","Solver internal state reused")):
        lines.append(f"| {label} | "+" | ".join(fmt(cells[a]['search_reuse']['cp'][key]) for a in ARMS)+" |")
    for key,label in (("cp_observed_calls","Observed CP continuation records"),("incumbent_sources","Observed incumbent sources"),
                      ("rank_cache_hits","Comparable group cache hits"),("rank_cache_miss_reasons","Group cache miss reasons"),
                      ("max_budget_fraction_drift","Maximum cached budget-fraction drift")):
        lines.append(f"| {label} | "+" | ".join(fmt(cells[a]['search_reuse'][key]) for a in ARMS)+" |")
    for key,label in (("ready_budget_remaining_at_start","Ready budget remaining at start"),
                      ("ready_budget_remaining_at_return","Ready budget remaining at return"),
                      ("solver_allocated_seconds","Solver allocated seconds"),("solver_remaining_deadline_s","Solver remaining deadline")):
        lines.append(f"| {label}: observed count / P95 (s) | "+" | ".join(
            f"{cells[a]['search_reuse'][key]['count']} / {fmt(cells[a]['search_reuse'][key]['p95_s'])}" for a in ARMS)+" |")
    for key,label in (("normal_work_s","Snapshot/projection/callback work"),("prepare_work_s","Preparation including ready guard"),
                      ("revalidation_work_s","Affected revalidation, separate from preparation"),
                      ("dependency_guard_work_s","Prepared-packet dependency inspection"),
                      ("ready_registration_bookkeeping_s","Ready registration"),("publication_bookkeeping_wall_s","Publication bookkeeping"),
                      ("normal_evidence_wall_s","Normal evidence")):
        lines.append(f"| {label} (s) | "+" | ".join(fmt(life[a]['components'][key]['total_s'] if life[a]['components'][key]['count'] else None) for a in ARMS)+" |")
    lines += ["","## Affected revalidation observations","","| Saved metric | H | L |","| --- | ---: | ---: |"]
    for key,label in (("availability","Field availability"),("declared_count","Declared revalidations"),
                      ("recorded_rounds","Recorded rounds"),("certified_rounds","Certified rounds"),
                      ("rejected_rounds","Rejected rounds"),("late_rounds","Late rounds"),
                      ("pending_rounds","Pending rounds"),("affected_aircraft_ids","Affected aircraft IDs"),
                      ("pairs_checked","Checked pairs"),("reused_pairs","Reused pairs"),
                      ("full_fallback_rounds","Full fallback rounds")):
        lines.append(f"| {label} | "+" | ".join(fmt(life[a]['revalidation'][key]) for a in ARMS)+" |")
    for field,label in (("wall","Per-round measured wall (s)"),("cpu","Per-round measured CPU (s)")):
        lines.append(f"| {label} | "+" | ".join(fmt(life[a]['revalidation'][field]['total_s'] if life[a]['revalidation'][field]['count'] else None) for a in ARMS)+" |")
    lines.append("Revalidation status counts H/L: "+" / ".join(counts(life[a]['revalidation']['status_counts']) for a in ARMS)+". Simulation due times and observed completion times are retained per task in timeline.csv; COMPUTING or GUARDING without observed completion is censored. geometry_completed_time records geometry computation only; it does not establish completed delivery. SUPERSEDED is reported separately in status counts.")
    lines.append("READY_DEADLINE_MISSED records computation returning after its ready deadline. REVALIDATION_WORK_DEADLINE records exhaustion of the measured revalidation compute budget and does not alone establish a late physical return. TRAFFIC_CHANGED_AFTER_DEADLINE records a newly observed traffic change after the last revalidation opportunity; it is a late dependency change, not a computation overrun. EXTERNAL_TRAFFIC_MISSING and packet expiry retain their own rejection counts.")
    lines += ["","## Task delivery facts",""]
    for arm in ARMS:
        c=cells[arm]; task=life[arm]; q=c["terminal"].get("quality",{})
        lines.append(f"{LABELS[arm]}: {counts(task['status_counts'])}. Rejection reasons: {counts(task['rejection_reasons'])}. No-prefix groups: {task['no_prefix']}; actual CPU: {c['actual_cpu']}. Independent terminal normal/emergency/failed/pending: "+"/".join(fmt(q.get(k)) for k in ("L_normal","L_emergency","F_energy_total","N_airborne_pending"))+".")
        lines.append("")
        for row in (r for r in timeline if r["arm"]==arm):
            lines.append(f"- {LABELS[arm]} task {row['task_index']}: snapshot {fmt(row['snapshot_time'])}; result {fmt(row['result_time'])}; ready {fmt(row['ready_time'])}, deadline {fmt(row['ready_deadline'])}; effective {fmt(row['effective_time'])}; {row['status']} at {fmt(row['finished_time'])}, submitted {fmt(row['submitted'])}"+(f"; {row['reason']}." if row.get("reason") else "."))
            for event in row.get("revalidations") or []:
                lines.append(f"  Revalidation {event.get('status','MISSING')}: started {fmt(event.get('started_time'))}; due {fmt(event.get('due'))}; completed {fmt(event.get('completed_time'))}; reason {fmt(event.get('reason'))}; checked/reused pairs {fmt(event.get('pairs_checked'))}/{fmt(event.get('reused_pairs'))}; measured wall/CPU {fmt(event.get('wall_s'))}/{fmt(event.get('cpu_s'))} s.")
        lines.append("")
    lines += ["","## Saved replay evidence",""]
    for arm in ARMS:
        audit=cells[arm]["replay"]; coverage=audit.get("normal_start_coverage",{})
        groups=[r for r in audit.get("committed_group_checks",[]) if r["status"]=="CHECKED"]
        c2=audit.get("c2_checks",[]); checked=[r for r in c2 if r["status"]=="CHECKED"]
        maximum=audit.get("max_pva_errors",{})
        lines.append(f"{LABELS[arm]}: ReplayBundle.validate {audit.get('replay_bundle_validate',audit['status'])}; file hashes {fmt(audit.get('file_hashes_match'))}; trajectory index {fmt(audit.get('trajectory_index_matches'))}. PVA samples checked {fmt(audit.get('checked_pva_samples'))}; maximum p/v/a errors "+" / ".join(fmt(maximum.get(k)) for k in ("p","v","a"))+f"; within tolerance {fmt(audit.get('pva_within_tolerance'))}. Start-time mismatches {fmt(audit.get('start_time_mismatch_count'))}.")
        lines.append(f"Committed groups checked: {len(groups)}; all hashes start together: {fmt(all(r['complete_same_time'] for r in groups) if groups else None)}. Actual normal-start set equals committed acknowledgements plus t0 approved normal plans: {fmt(coverage.get('actual_equals_ack_plus_t0'))}; unattributed starts: {fmt(len(coverage['unattributed_hashes']) if 'unattributed_hashes' in coverage else None)}. Unique-predecessor C2 comparisons: {len(checked)}; unavailable: {len(c2)-len(checked)}.")
        if audit.get("validation_error") or audit.get("error"): lines.append(f"Replay exception: {audit.get('validation_error') or audit.get('error')}.")
    lines += ["", "Rejected/pending proposed hashes were not saved, so independent per-proposal nonexecution is UNAVAILABLE_PROPOSED_HASHES. This is separate from checking coverage of all actual normal starts. Detailed group/predecessor rows and per-sample errors are in analysis_summary.json and replay_sample_errors.csv.","",
        "## Limited primary interpretation",""]
    hw,lw=(life[a]["all_task_observed_work"]["total_s"] for a in ARMS)
    hq,lq=h["common_quality"] or {},l["common_quality"] or {}
    if lw<hw:
        lines.append(f"Observed planning computation is lower for L in this pair: {fmt(hw)} to {fmt(lw)} s ({(hw-lw)/hw*100:.2f}% less). Actual committed groups are H/L {life[ARMS[0]]['committed_nonempty']}/{life[ARMS[1]]['committed_nonempty']}; common normal landings are {fmt(hq.get('L_normal'))}/{fmt(lq.get('L_normal'))}.")
        if life[ARMS[0]]["committed_nonempty"]==life[ARMS[1]]["committed_nonempty"] and hq.get("L_normal")==lq.get("L_normal"):
            lines.append("Faster computation did not yield more actual joint joins or normal landings within this horizon.")
    lines.append("This is one pair, without a significance/generalization claim or an isolated causal learning ablation. H and L share the progressive catalog, CP-SAT backend and physical policies.")
    if all(finite(phase(a,k)) for a in ARMS for k in ("cp_solve","prefix_validation","trajectory_validation")):
        lines.append(f"CP totals are H/L {fmt(phase(ARMS[0],'cp_solve'))}/{fmt(phase(ARMS[1],'cp_solve'))} s. Phase totals and search volume are descriptive observations; causal attribution remains PENDING_PRIMARY_REVIEW.")
    if all(finite(phase(ARMS[1],k)) for k in ("stage3_features","stage3_inference")):
        learning=phase(ARMS[1],"stage3_features")+phase(ARMS[1],"stage3_inference")
        lines.append(f"L features plus inference consume {fmt(learning)} s across callbacks and also occupy the same fixed candidate-active clock (6 s per callback). Fewer attempts or a smaller catalog alone cannot establish better candidate identification; inference can displace generation work. Shared prefix-cache and geometry improvements belong to both H and L. No extra ablation is run.")
    lines.append(f"L no-prefix groups: {life[ARMS[1]]['no_prefix']}; finished nonempty denominators H/L: {life[ARMS[0]]['finished_nonempty']}/{life[ARMS[1]]['finished_nonempty']}. Different denominators, no-prefix and remaining coverage gaps prevent interpreting the raw join fractions as sustained join improvement.")
    if any(life[a]["rejection_reasons"].get("NEW_TRAFFIC",0) for a in ARMS):
        lines.append("NEW_TRAFFIC rejection counts are retained above. Their presence alone does not establish the bottleneck; compare recorded revalidation certification, lateness, affected traffic, fallback and actual joined groups. Legacy missing revalidation fields cannot establish that revalidation occurred.")
    lines += ["","## Identity, clocks and review limits",""]
    lines.extend(f"- {name}: {fmt(result)}." for name,result in value["identity_checks"].items())
    lines.append("")
    lines += [f"Model SHA `{value['model_sha256']}`; normalizer SHA `{value['normalizer_sha256']}`.",
        "Complete-planning P95 uses finished tasks only. All-task work includes computation already executed for censored pending tasks. Ready guard is included in preparation exactly once; separate cumulative revalidation and dependency-inspection work are each added once, without summing per-round or online_services diagnostics again. Service-side publication is not summed again. Join rate excludes pending/no-prefix; actual joined group and submitted-plan counts come from recorded acknowledgements.",
        "Boundary/terminal full_online_wall_s include setup and previous checkpoint/log I/O; they are job-wall diagnostics, not pure planning time. Job wall includes replay export, and subprocess receipts include startup. Integration/execution/recording overhead is separate from planning delay.",
        "A timely-ready PREPARED token censored at 120 s is pending, not a timeout or successful join. No endpoint dispatch is executed by this report. Recorded result_time includes handoff completion, while finished_time is the outcome event; negative acknowledgements may therefore precede the recorded handoff by a small interval. These metadata are not asserted strictly chronological.",
        value["c2_limit"],
        "Shared emergency response keeps its prior timestamp semantics and charges CPU service against normal work. Publication uses the documented atomic-event bookkeeping abstraction. Saved hash/state checks provide evidence only where the checks above are available and pass; they do not create human safety acceptance.",
        "Figures: planning_timeline.svg/PNG (300 dpi), editable text and source CSV. One pair, no CI/error bars; missing result times stay missing. Grey regions are beyond observed terminal time. See timeline.csv and boundary_quality.csv.",
        "Final image visual QA: PENDING_PRIMARY_VIEW_IMAGE. Replay HUMAN_PENDING is unchanged."]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkout",type=Path); parser.add_argument("--contact-sheet",action="store_true")
    args=parser.parse_args(argv); output=args.output.resolve()
    existing=[name for name in (*OUTPUTS,*CONTACT_OUTPUTS) if (output/name).exists()]
    if existing: raise ValueError("derived outputs already exist; no overwrite: "+str(existing))
    scope=load(output/"execution_scope.json")
    value,timeline,quality,errors,contacts=collect(output,saved_core(args.checkout or scope.get("checkout")))
    write_csv(output/"timeline.csv",timeline,tuple(timeline[0]) if timeline else ("arm","snapshot_time","status"))
    write_csv(output/"boundary_quality.csv",quality,("arm","time_s","L_normal","L_emergency","F_energy_total","N_airborne_pending","N_admitted_total","E_terminal_J","holding_time"))
    write_csv(output/"replay_sample_errors.csv",errors,("arm","time","aircraft_id","trajectory_hash","status","p_error","v_error","a_error"))
    made_contact=figures(output,value,timeline,quality,contacts,args.contact_sheet)
    contract=dict(core_question="Do saved task delivery/actual joins follow fixed timing, and how do H/L timeouts and dependency rejections evolve?",
        core_conclusion="Descriptive evidence; scientific interpretation is reserved for primary review",evidence_hierarchy=["normal_tasks and acknowledged scheduler records","actual replay events/states","saved 5 s boundary quality"],
        archetype="aligned task timelines plus cumulative boundary trends",panels=["H tasks","L tasks","normal/emergency cumulative landings"],
        reviewer_risk="One pair; censored pending; missing per-rejected-proposal hashes; no independent causal learning attribution",
        target="research review, full width",exports=list(OUTPUTS),statistics="n=1 pair, no CI or significance tests",qa=dict(visual_review="PENDING_PRIMARY_VIEW_IMAGE"))
    write_json(output/"figure_contract.json",contract)
    text_count=len(ET.parse(output/"planning_timeline.svg").getroot().findall(".//{http://www.w3.org/2000/svg}text"))
    qa=dict(svg_editable_text_elements=text_count,svg_text_editable=text_count>0,export_dpi=300,
        source_csvs=["timeline.csv","boundary_quality.csv"],visual_review="PENDING_PRIMARY_VIEW_IMAGE",human_replay_acceptance="HUMAN_PENDING",
        contact_sheet_saved_frames_only=made_contact)
    write_json(output/"figure_QA.json",qa)
    value.update(figure_contract=contract,figure_qa=qa,derived_script_sha256=file_hash(Path(__file__)),
                 derived_files_sha256={name:file_hash(output/name) for name in OUTPUTS if (output/name).exists()})
    write_json(output/"analysis_summary.json",value)
    with (output/"FINAL_REPORT.md").open("x",encoding="utf8") as stream: stream.write(report_text(value,timeline))
    print(json.dumps(dict(status="DERIVED_OUTPUTS_WRITTEN",output=str(output),visual_review="PENDING_PRIMARY_VIEW_IMAGE")))


if __name__=="__main__": main()
