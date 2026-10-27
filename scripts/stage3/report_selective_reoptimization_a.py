"""Saved-artifact Stage A accounting; no simulation, training, or acceptance claim."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import time

METHODS = ("FULL_H", "RULE_LOCAL_K8")
ROOTS = ("s3r23", "s3r24")


def load(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).is_file() else default


def lines(path):
    with Path(path).open(encoding="utf8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def conservation(quality):
    needed = ("N_offered", "N_admitted_total", "N_rerouted", "N_upstream_pending", "L_success", "F_energy_total", "N_airborne_pending")
    if not all(key in quality for key in needed):
        return dict(status="UNAVAILABLE", missing=[key for key in needed if key not in quality])
    offered = quality["N_offered"] == quality["N_admitted_total"] + quality["N_rerouted"] + quality["N_upstream_pending"]
    admitted = quality["N_admitted_total"] == quality["L_success"] + quality["F_energy_total"] + quality["N_airborne_pending"]
    return dict(status="PASS" if offered and admitted else "FAIL", offered_equation=offered, admitted_equation=admitted)


def proposal_nonexecution(callbacks, events, tasks=()):
    """Attribute starts by absolute time; same immutable hash may be reproposed."""
    starts = {(e.get("details", {}).get("trajectory_hash"), e["time"]) for e in events if e.get("type") == "maneuver_started"}
    legitimate = set()
    for row in callbacks:
        timing = row.get("planning_timing", {})
        if row.get("actual_delivery_status") == "COMMITTED":
            legitimate.update((h, timing.get("effective_time")) for h in row.get("actual_submitted_trajectory_hashes", []))
    checks = []
    task_by_time={r["snapshot_time"]:r for r in tasks}
    for index, row in enumerate(callbacks):
        if row.get("actual_delivery_status") == "COMMITTED":
            continue
        # These hashes must be captured before submission; missing telemetry is
        # unavailable evidence, never proof that no proposal was executed.
        hashes = row.get("proposed_trajectory_hashes")
        if hashes is None:
            hashes = row.get("proposal_trajectory_hashes")
        if hashes is None:
            hashes = task_by_time.get(row.get("planning_timing",{}).get("snapshot_time"),{}).get("proposed_trajectory_hashes")
        effective = row.get("planning_timing", {}).get("effective_time")
        if hashes is None or effective is None:
            checks.append(dict(index=index, status="UNAVAILABLE_PROPOSED_HASHES")); continue
        offending = sorted(h for h in hashes if (h, effective) in starts - legitimate)
        checks.append(dict(index=index, status="FAIL" if offending else "PASS", effective_time=effective,
                           proposed_hashes=hashes, unauthorized_started_hashes=offending,
                           legitimately_committed_elsewhere=sorted(h for h in hashes if (h, effective) in legitimate)))
    return dict(status="FAIL" if any(x["status"]=="FAIL" for x in checks) else "INCOMPLETE" if any(x["status"].startswith("UNAVAILABLE") for x in checks) else "PASS", checks=checks)


def resource_and_c2_audit(replay):
    """Check full saved calendars with the existing half-open resource kernel.

    PVA transitions use authoritative serialized plans; missing predecessors are
    retained as unknown rather than silently accepted.
    """
    from terminal_marl.core.rotating_slots.resources import ResourceCalendar
    from terminal_marl.core.rotating_slots.config import load_config
    from terminal_marl.core.rotating_slots.trajectory import TrajectoryPlan
    config = load_config(replay / "config.yaml")
    plans = {h: TrajectoryPlan.from_dict(p, config=config) for h,p in load(replay / "trajectories.json", {}).items()}
    events = list(lines(replay / "events.jsonl"))
    starts = {e["details"]["trajectory_hash"]: e["time"] for e in events if e.get("type")=="maneuver_started" and "trajectory_hash" in e.get("details", {})}
    previous = {}; issues = []; transitions = []; frames = reservations = 0
    for frame in lines(replay / "states.jsonl"):
        frames += 1; now = frame["time"]
        for name in ("slots", "pads"):
            calendar = ResourceCalendar()
            for r in frame.get(name, ()):
                reservations += 1
                try:
                    resource = tuple(r["resource"]) if isinstance(r["resource"], list) else r["resource"]
                    calendar.reserve(resource, r["aircraft_id"], r["start"], math.inf if r["end"] is None else r["end"])
                except (ValueError, KeyError, TypeError) as exc:
                    if len(issues)<100:
                        issues.append(dict(time=now, calendar=name, error=str(exc)))
        for a in frame.get("aircraft", ()):
            aid = a["id"]; h = a.get("trajectory_hash"); old = previous.get(aid)
            if old and h != old[1].get("trajectory_hash") and h in starts:
                old_hash = old[1].get("trajectory_hash"); transition = starts[h]
                row = dict(aircraft_id=aid, trajectory_hash=h, predecessor_hash=old_hash, time=transition, status="UNAVAILABLE_PREDECESSOR")
                if old_hash in plans and h in plans and old[0] <= transition <= now:
                    before, after = plans[old_hash].state(transition), plans[h].state(transition)
                    errors = {k+"_error": math.dist(getattr(before,k), getattr(after,k)) for k in ("p","v","a")}
                    row.update(errors, status="PASS" if errors["p_error"]<=1e-6 and errors["v_error"]<=1e-7 and errors["a_error"]<=1e-7 else "FAIL")
                transitions.append(row)
            previous[aid]=(now,a)
    missing = sorted(set(starts) - {r["trajectory_hash"] for r in transitions})
    # INBOUND may first appear at admission without a preceding physical state.
    required_missing = [h for h in missing if plans[h].kind in ("TRANSFER","FINAL_NORMAL","FINAL_EMERGENCY")]
    return dict(resource_status="FAIL" if issues else "PASS", saved_calendar_frames=frames,
        reservations_checked=reservations, resource_issues=issues, c2_checks=transitions,
        c2_status="FAIL" if any(r["status"]=="FAIL" for r in transitions) else "INCOMPLETE" if required_missing or any(r["status"]!="PASS" for r in transitions) else "PASS",
        missing_maneuver_predecessors=required_missing)


def audit_job(base, checkout):
    """Called only in a parent-owned bounded ledger subprocess."""
    from terminal_marl.core.rotating_slots.audit import audit_replay
    started = time.perf_counter()
    write(base / "audit_progress.json", dict(status="RUNNING", stage="INDEPENDENT_CONTINUOUS_ENERGY_LEDGER"))
    independent = audit_replay(base / "replay")
    write(base / "independent_replay_audit.json", independent)
    write(base / "audit_progress.json", dict(status="RUNNING", stage="FULL_SAVED_RESOURCES_AND_C2"))
    resources = resource_and_c2_audit(base / "replay")
    write(base / "saved_resource_c2_audit.json", resources)
    latency = source_module(checkout / "scripts/stage3/report_latency_interface.py", "_selective_saved_latency_report")
    callbacks = load(base / "scheduler_records.json", [])
    pva, errors, _ = latency.audit_replay(base, callbacks, latency.saved_core(checkout))
    write(base / "saved_pva_audit.json", pva)
    nonexecution = proposal_nonexecution(callbacks, list(lines(base / "replay/events.jsonl")), load(base/"normal_tasks.json",[]))
    write(base / "proposal_nonexecution_audit.json", nonexecution)
    covered = (independent.get("status")=="PASS" and independent.get("continuous_separation")=="CERTIFIED"
               and resources["resource_status"]=="PASS" and resources["c2_status"]=="PASS"
               and pva.get("pva_within_tolerance") is True and nonexecution["status"]=="PASS")
    result = dict(status="PASS" if covered else "INCOMPLETE_OR_FAILED", full_a20_covered=covered,
        independent_status=independent.get("status"), continuous_separation=independent.get("continuous_separation"),
        resource_status=resources["resource_status"], c2_status=resources["c2_status"],
        pva_status=pva.get("status"), proposal_nonexecution=nonexecution["status"],
        wall_s=time.perf_counter()-started, human_review="HUMAN_PENDING")
    write(base / "audit_result.json", result); write(base / "audit_progress.json", result)
    return result


def selection_summary(records):
    rows = [r["selective_replanning"] for r in records if r.get("selective_replanning")]
    counts = Counter(r.get("status","UNKNOWN") for r in rows)
    overflow = [r for r in rows if "OVERFLOW" in r.get("status","") or "OVERFLOW" in (r.get("scope") or {}).get("status","")]
    ages = [v for r in rows for v in r.get("ages",{}).values()]
    return dict(tasks=len(rows), status_counts=dict(counts), overflow_full_fallback_tasks=len(overflow),
        local_tasks=sum((r.get("scope") or {}).get("status")=="LOCAL" for r in rows),
        full_scope_tasks=sum((r.get("scope") or {}).get("status")=="FULL_H" for r in rows),
        max_review_age_s=max((v["review_age"] for v in ages if v.get("review_age") is not None), default=None),
        max_progress_age_s=max((v["progress_age"] for v in ages if v.get("progress_age") is not None), default=None),
        review_age_unknown=sum(v.get("review_unknown",False) for v in ages),
        progress_age_unknown=sum(v.get("progress_unknown",False) for v in ages),
        scope_rows=[dict(scope=r.get("scope"),candidate_counts=r.get("candidate_generation_by_aircraft"),pair_checks=r.get("pair_checks"),model_scale=r.get("model_scale"),solve_calls=r.get("solve_calls"),status=r.get("status")) for r in rows])


def report(output, checkout):
    started = time.perf_counter()
    binding = load(output / "resolved_run_binding.json")
    runner = source_module(checkout / "scripts/stage3/run_selective_reoptimization_a.py", "_selective_runner_accounting")
    latency = source_module(checkout / "scripts/stage3/run_latency_interface.py", "_selective_lifecycle")
    pairs = {}; table = []
    for root in ROOTS:
        bases = {m:output/"evaluation"/root/m for m in METHODS}
        boundaries = {m:{int(p.stem.split("_")[-1]):p for p in base.glob("boundary_*.json")} for m,base in bases.items()}
        common = max(set(boundaries[METHODS[0]]) & set(boundaries[METHODS[1]]),default=None)
        rows = {}
        for method, base in bases.items():
            row = load(boundaries[method][common]) if common is not None else None
            callbacks = load(base/"scheduler_records.json",[])
            tasks, services = load(base/"normal_tasks.json",[]),load(base/"online_services.json",[])
            terminal_lifecycle = runner.lifecycle_summary(tasks,callbacks,services,latency)
            lifecycle = row["normal_lifecycle"] if row else {}
            quality = row.get("quality",{}) if row else {}
            selections = row.get("selection_summary",{}) if row else {}
            pending = lifecycle.get("pending_tasks")
            rows[method] = dict(result=load(base/"result.json",dict(status="NOT_RUN_OR_WATCHDOG")),
                common_boundary=row, conservation=conservation(quality), lifecycle=lifecycle,terminal_lifecycle=terminal_lifecycle,
                selections=selections, pending_computation_callbacks=pending,
                offline_audit=load(base/"audit_result.json",dict(status="UNAVAILABLE",full_a20_covered=False)))
            table.append(dict(root=root,method=method,common_endpoint_s=common,
                initial_airborne=quality.get("initial_airborne_count"),initial_ground=quality.get("initial_ground_count"),
                new_offered=quality.get("manifest_offered"),normal=quality.get("L_normal"),emergency=quality.get("L_emergency"),
                failures=quality.get("F_energy_total"),rerouted=quality.get("N_rerouted"),
                airborne_pending=quality.get("N_airborne_pending"),upstream_pending=quality.get("N_upstream_pending"),
                unresolved_low_soc=quality.get("unresolved_low_soc_count"),
                terminal_energy_J=quality.get("E_terminal_J"),holding_time_s=quality.get("holding_time"),
                pending_computations=pending,max_review_age_s=selections.get("max_review_age_s"),
                max_progress_age_s=selections.get("max_progress_age_s"),overflow_full_fallback_tasks=selections.get("overflow_full_fallback_tasks"),
                full_normal_planning_work_s=lifecycle.get("full_normal_planning_work",{}).get("total_s"),
                dispatch_wall_s=row.get("costs",{}).get("dispatch_wall_s") if row else None,
                available_delay_total_s=lifecycle.get("availability_delay_s",{}).get("total_s"),
                late_rate=lifecycle.get("late_rate"),conservation=rows[method]["conservation"]["status"]))
        pairs[root]=dict(common_completed_boundary_s=common,methods=rows)
    snapshots=[load(p) for p in sorted((output/"snapshots").glob("*/*/result.json"))]
    acceptance={f"A{i:02d}":dict(status="PENDING_PRIMARY_EVIDENCE_REVIEW",evidence_refs=["resolved_run_binding.json:checks"]) for i in range(1,21)}
    acceptance["A20"]=dict(status="AUTOMATED_SAVED_REPLAY_COVERED" if all(v["methods"][m]["offline_audit"].get("full_a20_covered") for v in pairs.values() for m in METHODS) else "INCOMPLETE_OR_FAILED",evidence_refs=[f"evaluation/{r}/{m}/audit_result.json" for r in ROOTS for m in METHODS])
    acceptance["X01"]=dict(status="PASS_AT_COMMON_ENDPOINTS" if all(v["methods"][m]["conservation"]["status"]=="PASS" for v in pairs.values() for m in METHODS) else "INCOMPLETE_OR_FAILED",evidence_refs=["system_counts.csv","analysis_summary.json:pairs"])
    acceptance["X02"]=dict(status="HUMAN_PENDING",implementation="READ_ONLY_SELECTION_OVERLAY",evidence_refs=["ReplayController.selective_overlay","replay_instructions.md"])
    summary=dict(schema="rs_selective_stage_a_report_v1",source_binding=binding["binding_id"],pairs=pairs,
        snapshots=snapshots,expected_snapshot_comparisons=8,observed_snapshot_comparisons=len(snapshots),
        acceptance=acceptance,research_status="DATA_INSUFFICIENT_PENDING_PRIMARY_REVIEW",
        human_replay_status="HUMAN_PENDING",automatic_phase_B=False,
        budget_ledger=load(output/"budget_ledger.json"),offline_reporting_wall_s=time.perf_counter()-started,
        timing_semantics=dict(normal_work="lifecycle.full_normal_planning_work includes actual selector/frame/candidate/CP/preparation/revalidation/publish services",
            dispatch="separately measured dispatch_timings; overlaps normal-work entries, do not add twice",
            available_delay="simulation snapshot to actual delivery, includes service and wait; censored pending not successful",
            physics="world advance minus dispatch includes execution checks, integration and recording; pure physics unavailable",
            offline_audit="separate measured bounded jobs; never normal online planning cost"))
    write(output/"analysis_summary.json",summary)
    with (output/"system_counts.csv").open("w",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    text="# Selective reoptimization: Stage A saved evidence\n\n"
    text+=f"Source `{binding['source_commit']}`; binding `{binding['binding_id']}`. Eight preregistered saved development snapshots; observed {len(snapshots)}. No model fit.\n\n"
    text+="| Root | Method | Common s | Normal | Emergency | Energy failure | Air pending | Upstream pending | Full fallback |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in table:
        text+="| "+" | ".join(str(r[k]) for k in ("root","method","common_endpoint_s","normal","emergency","failures","airborne_pending","upstream_pending","overflow_full_fallback_tasks"))+" |\n"
    text+="\nFull normal work, dispatch, simulated availability delay, integration/executor work and offline audit costs are separate in analysis_summary.json. Whole-job cost includes setup/checkpoint/output I/O. Pending computation and unfinished service at the fixed endpoint are censored; a watchdog is BUDGET_LIMITED, never a physical failure. Lower energy must be read together with service and pending counts.\n\n"
    text+="Snapshot rows retain common frame/projection cost, candidate counts by aircraft, pair checks, decision/auxiliary model scale, solve calls, objective, actual K, and overflow status in snapshots/*/*/result.json. Overflow full fallbacks are nonlocal; no speedup claim is made.\n\n"
    text+="A01–A20/X01/X02 have explicit evidence/status entries. Passing XML alone does not establish all contract conditions. A20 requires all saved replay audits, including continuous separation, resources, full PVA/C2 and rejected-proposal nonexecution. Missing coverage stays incomplete. Research interpretation requires primary review. Human replay remains HUMAN_PENDING; no automatic B/C.\n"
    (output/"FINAL_REPORT.md").write_text(text,encoding="utf8")
    instructions="Replay each evaluation/<root>/<method>/replay in the existing viewer.\n\n```python\nfrom terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer\ncontroller = ReplayController('/absolute/output/evaluation/s3r23/RULE_LOCAL_K8/replay')\nRotatingSlotsReplayViewer(controller).run()\n```\n\nS is selected, B fixed background, M mandatory; inspect blocker edges and the effective-time countdown/status. These are computation annotations. Authoritative trajectories and commits remain the execution record. Inspect both root/method pairs and the audit artifacts before recording actual human observations. Automatic checks do not change HUMAN_PENDING.\n"
    (output/"replay_instructions.md").write_text(instructions,encoding="utf8")
    return dict(status="REPORT_WRITTEN",research_status=summary["research_status"],human_replay_status="HUMAN_PENDING")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkout",type=Path,default=Path(__file__).resolve().parents[2]);args=parser.parse_args(argv)
    print(json.dumps(report(args.output.resolve(),args.checkout.resolve()),sort_keys=True))


if __name__=="__main__": main()
