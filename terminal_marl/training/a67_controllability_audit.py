"""A6.7 causal pipeline and controllability audit; never trains PPO."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

import numpy as np

from ..core.config import RingRole
from ..core.merge_scheduler import LowLevelIntent
from ..core.uav import FlightPhase
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
from ..policies import DynamicRulePolicy
from .a6_effectiveness import _collect, _sample, scenario_config
from .a63_dynamic_screen import SHORT_SCENARIOS
from .a65_counterfactual_audit import Candidate, NO_OP, available_cpus


TARGET_TIMES = (720., 1080., 1260., 1800.)
TRACE_ACTIONS = (
    Candidate("no_op", "NO_OP", tuple(NO_OP)),
    Candidate("headway_m2i_minus", "HEADWAY", (0, 0, 0, 0, 0, 0, 0, 0, -.35, 0)),
    Candidate("speed_minus", "SPEED", (0, 0, 0, 0, 0, 0, -.25, 0, 0, 0)),
    Candidate("outer_capacity_plus1", "GEOMETRY", (.60, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    Candidate("outer_plus_o2m_minus", "COMBINED", (.60, 0, 0, 0, 0, 0, 0, -.35, 0, 0)),
)
HORIZONS = (180., 540., 1080.)


def _worker_init() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    try:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
    except ImportError:
        pass


def _capture_worker(target_time: float) -> tuple[dict, bytes]:
    """One deterministic prefix rollout; independent target times share no state."""
    captures = capture_high_seed0((target_time,))
    if len(captures) != 1:
        raise RuntimeError(f"failed to capture high seed0 at t={target_time}")
    return captures[0]


def _interface_snapshot(env, interface: str) -> dict:
    phases = {
        "o2m": FlightPhase.RING_OUTER,
        "m2i": FlightPhase.RING_MIDDLE,
        "i2f": FlightPhase.RING_INNER,
    }
    target = {"o2m": RingRole.MIDDLE, "m2i": RingRole.INNER, "i2f": None}[interface]
    eligible = [uav for uav in env.active.values() if uav.phase is phases[interface] and uav.committed_plan is None]
    downstream_capacity = env.config.final_queue_capacity if interface == "i2f" else env.airspace.capacity(target)
    downstream_occupancy = sum(uav.phase is FlightPhase.FINAL for uav in env.active.values()) if interface == "i2f" else env.airspace.occupancy(target)
    return {"eligible": len(eligible), "downstream_capacity": downstream_capacity, "downstream_occupancy": downstream_occupancy,
            "downstream_slack": downstream_capacity - downstream_occupancy, "attempts": 0, "approvals": 0, "rejections": 0,
            "reason_histogram": Counter(), "headway_blocked": 0, "merge_gap_blocked": 0, "safety_rejected": 0,
            "capacity_blocked": 0, "interface_frozen": 0, "soc_blocked": 0, "resource_blocked": 0}


def _classify_reason(row: dict, reason: str) -> None:
    row["reason_histogram"][reason] += 1
    if reason in {"geometry_frozen"}: row["interface_frozen"] += 1
    elif reason in {"safety_reject", "safety_wait"}: row["safety_rejected"] += 1
    elif reason in {"final_capacity", "lz_unavailable", "lz_reservation_reject"}: row["resource_blocked"] += 1
    elif reason in {"plan_active"}: row["merge_gap_blocked"] += 1
    elif reason in {"headway_wait"}: row["headway_blocked"] += 1
    elif reason in {"not_requested", "phase_masked"}: pass
    else: row["capacity_blocked"] += 1


def _reservation_audit(env) -> dict:
    active_ids = set(env.active); rows = list(env.scheduler.reservations.values())
    interfaces = Counter({"o2m": 0, "m2i": 0, "i2f": 0}); statuses = Counter(); ages = []; orphaned = []
    for item in rows:
        key = {"middle": "o2m", "inner": "m2i", "final": "i2f"}[item.target_stage]
        interfaces[key] += 1; statuses[item.status] += 1; ages.append(max(0., env.now - item.merge_time))
        owner = env.active.get(item.uav_id)
        if owner is None or owner.merge_reservation_id != item.reservation_id: orphaned.append(item.reservation_id)
    return {"active_reservation_count": len(rows), "pending_reservation_count": len(rows), "reservation_by_interface": dict(interfaces),
            "reservation_by_status": dict(statuses), "reservation_age_mean": float(np.mean(ages)) if ages else 0.,
            "reservation_age_p95": float(np.percentile(ages, 95)) if ages else 0., "reservation_age_max": max(ages, default=0.),
            "expired_or_stale_count": sum(item.expires_at <= env.now for item in rows), "orphaned_reservation_ids": orphaned,
            "committed_transfer_load": sum(uav.committed_plan is not None for uav in env.active.values()),
            "active_canonical_plans": sum(uav.committed_plan is not None for uav in env.active.values()),
            "last_release_tentative": dict(env.scheduler.last_release), "last_release_executed": dict(env.scheduler.last_executed_release)}


def _state(env, interface_rows: dict) -> dict:
    phase_counts = Counter(uav.phase.value for uav in env.active.values())
    occupancy = {role.value: env.airspace.occupancy(role) for role in RingRole}; capacity = {role.value: env.airspace.capacity(role) for role in RingRole}
    actual = env.airspace.state_vector()
    return {"time": env.now, "external_queue": len(env.admission.queue.items), "external_oldest_wait": max((item.external_wait_time for item in env.admission.queue.items), default=0.),
            "occupancy": occupancy, "capacity": capacity, "slack": {key: capacity[key] - occupancy[key] for key in capacity},
            "actual_state_10d": actual.tolist(),
            "target_rh": [env.airspace.target_ring(role).radius for role in RingRole] + [env.airspace.target_ring(role).height for role in RingRole],
            "headways": dict(env.airspace.headways), "speed_profile_bias": env.airspace.speed_profile_bias,
            "phase_counts": dict(phase_counts), "final_count": sum(uav.phase is FlightPhase.FINAL for uav in env.active.values()),
            "lz": [{"id": zone.lz_id, "status": zone.status.value, "remaining": zone.service_remaining} for zone in env.lz_manager.zones],
            "interfaces": {key: dict(value) | {"reason_histogram": dict(value["reason_histogram"]), "safety_reason_histogram": dict(value.get("safety_reason_histogram", {}))} for key, value in interface_rows.items()},
            "reservations": _reservation_audit(env)}


def capture_high_seed0(target_times=TARGET_TIMES, *, parallel_prefixes: bool = False) -> list[tuple[dict, bytes]]:
    if parallel_prefixes and len(target_times) > 1:
        workers=min(len(target_times), max(1, available_cpus()-2))
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
            captures=list(executor.map(_capture_worker, target_times))
        return sorted(captures, key=lambda item: item[0]["time"])
    config = scenario_config(SHORT_SCENARIOS["high"], "dynamic_rule")
    wrapper = DynamicHighLevelWrapper(config, RuleLowLevelAdapter()); observation, _ = wrapper.reset(0)
    targets = iter(sorted(target_times)); next_target = next(targets, None); captures = []
    interface_rows = {key: _interface_snapshot(wrapper.base, key) for key in ("o2m", "m2i", "i2f")}
    while next_target is not None and wrapper.base.now < config.max_simulation_seconds:
        action = DynamicRulePolicy().act(observation, wrapper._context())
        base = wrapper.base
        decision = wrapper.coordinator.coordinate(action, geometry_gate=wrapper.geometry_event_gate())
        for _ in range(round(config.high_level_decision_interval_seconds / config.dt)):
            for key in interface_rows:
                current = _interface_snapshot(base, key); interface_rows[key]["eligible"] = current["eligible"]; interface_rows[key]["downstream_capacity"] = current["downstream_capacity"]; interface_rows[key]["downstream_occupancy"] = current["downstream_occupancy"]; interface_rows[key]["downstream_slack"] = current["downstream_slack"]
            phase_before = {str(uid): uav.phase for uid, uav in base.active.items()}
            actions = wrapper.coordinator.apply_speed_repairs(wrapper.low_level.act(base._observations()))
            _, _, _, _, info = base.step(actions)
            for uid, reason in info["release_decisions"].items():
                phase = phase_before.get(uid)
                interface = {FlightPhase.RING_OUTER: "o2m", FlightPhase.RING_MIDDLE: "m2i", FlightPhase.RING_INNER: "i2f"}.get(phase)
                if interface is None:
                    continue
                interface_rows[interface]["attempts"] += int(reason != "not_requested")
                if reason == "approved":
                    interface_rows[interface]["approvals"] += 1
                elif reason != "not_requested":
                    interface_rows[interface]["rejections"] += 1; _classify_reason(interface_rows[interface], reason)
                    if reason == "safety_reject":
                        interface_rows[interface].setdefault("safety_reason_histogram", Counter())[base.last_safety_decisions.get(int(uid), "unknown")] += 1
            if base.now >= next_target - 1e-9:
                snapshot = _state(base, interface_rows); snapshot["high_action"] = {"requested": list(decision.requested_action), "applied": list(decision.applied_action), "outcome": decision.outcome}
                captures.append((snapshot, pickle.dumps(wrapper, protocol=pickle.HIGHEST_PROTOCOL)))
                # Each requested snapshot reports its own recent interface
                # behavior rather than cumulative counts since time zero.
                interface_rows = {key: _interface_snapshot(base, key) for key in ("o2m", "m2i", "i2f")}
                next_target = next(targets, None)
                if next_target is None: break
            if wrapper._done(): break
        observation = wrapper.observation()
        if wrapper._done(): break
    return captures


def explain_o2m_capture(payload: bytes) -> dict:
    """Read-only reproduction of one O2M SafetyFilter verdict and nearest holder."""
    wrapper = pickle.loads(payload); env = wrapper.base
    candidates = [uav for uav in env.active.values() if uav.phase is FlightPhase.RING_OUTER and uav.committed_plan is None]
    if not candidates:
        return {"time": env.now, "error": "no_eligible_outer_holder"}
    uav = candidates[0]; intent = LowLevelIntent(1., 0., 0.)
    snapshot = type("Snapshot", (), {"now": env.now, "airspace": env.airspace, "active": env.active})()
    proposal = env.scheduler.propose(uav, intent, snapshot, RingRole.MIDDLE)
    reservation = SimpleNamespace(reservation_id="diagnostic", merge_time=proposal.merge_time, merge_angle=proposal.merge_angle)
    plan = env.generator.ring_transfer(uav, RingRole.OUTER, RingRole.MIDDLE, reservation, env.airspace, env.now, 0.)
    verdict = env.safety.filter_plan(plan, {uid: item.committed_plan for uid, item in env.active.items() if item.committed_plan is not None}, snapshot)
    nearest = []
    times = np.arange(plan.start_time, plan.end_time + 1e-9, 1.)
    if times[-1] < plan.end_time: times = np.append(times, plan.end_time)
    for other in env.active.values():
        if other.uid == uav.uid:
            continue
        distance = min(float(np.linalg.norm(plan.sample(float(moment)).position - env.safety._holding_prediction(other, snapshot, float(moment)))) for moment in times)
        current_distance=float(np.linalg.norm(uav.position-other.position))
        nearest.append({"uid": other.uid, "phase": other.phase.value, "ring": None if other.ring_role is None else other.ring_role.value, "ring_progress": other.ring_progress, "current_distance": current_distance, "minimum_distance": distance})
    actual_pairs=[]
    outer=[item for item in env.active.values() if item.ring_role is RingRole.OUTER]
    for index,left in enumerate(outer):
        for right in outer[index+1:]: actual_pairs.append({"uids": (left.uid,right.uid), "distance": float(np.linalg.norm(left.position-right.position))})
    return {"time": env.now, "uav_id": uav.uid, "source_phase": uav.ring_progress, "proposal": {"merge_time": proposal.merge_time, "merge_angle": proposal.merge_angle}, "plan": {"start": plan.start_time, "end": plan.end_time, "duration": plan.end_time-plan.start_time}, "safety_verdict": {"approved": verdict.approved, "reason": verdict.reason}, "nearest_holders": sorted(nearest, key=lambda row: row["minimum_distance"])[:8], "minimum_actual_outer_pair": min(actual_pairs,key=lambda row:row["distance"],default=None)}


def _run_branch(snapshot: bytes, action: Candidate, horizon: float) -> dict:
    wrapper = pickle.loads(snapshot); base = wrapper.base; start = base.now
    before = _state(base, {key: _interface_snapshot(base, key) for key in ("o2m", "m2i", "i2f")})
    safety_before = {key: base.telemetry.get(key, 0.) for key in ("collisions", "nmac", "hard_separation_violations", "final_overcapacity_steps")}
    # Snapshots retain prefix diagnostics; branch evidence must be scoped to
    # events created after this candidate action is applied.
    conflict_start = len(base.conflict_events)
    decision = wrapper.coordinator.coordinate(np.asarray(action.action, np.float32), geometry_gate=wrapper.geometry_event_gate())
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}
    effects = Counter(); first_divergence = None
    safety_reason_hist = {"o2m": Counter(), "m2i": Counter(), "i2f": Counter()}
    while base.now < start + horizon and not wrapper._done():
        advanced = wrapper.coordinator.advance()
        phase_before = {str(uid): uav.phase for uid, uav in base.active.items()}
        acts = wrapper.coordinator.apply_speed_repairs(wrapper.low_level.act(base._observations()))
        event_start = len(base.events)
        _, _, _, _, info = base.step(acts); _sample(base, samples)
        effects["release_attempts"] += sum(reason != "not_requested" for reason in info["release_decisions"].values())
        effects["approved_releases"] += sum(reason == "approved" for reason in info["release_decisions"].values())
        for uid, reason in info["release_decisions"].items():
            if reason != "safety_reject":
                continue
            interface = {FlightPhase.RING_OUTER: "o2m", FlightPhase.RING_MIDDLE: "m2i", FlightPhase.RING_INNER: "i2f"}.get(phase_before.get(uid))
            if interface is not None:
                safety_reason_hist[interface][base.last_safety_decisions.get(int(uid), "unknown")] += 1
        new_events = base.events[event_start:]
        effects["touchdowns"] += sum(event[0] == "touchdown" for event in new_events)
        effects["completions"] += sum(event[0] == "completed" for event in new_events)
    after = _state(base, {key: _interface_snapshot(base, key) for key in ("o2m", "m2i", "i2f")})
    metrics = _collect(base, samples, wrapper.coordinator, [])
    safety_delta = {key: metrics.get(key, 0.) - safety_before[key] for key in safety_before}
    branch_conflicts = base.conflict_events[conflict_start:]
    return {"action": action.name, "family": action.family, "horizon": horizon, "decision": {"requested": list(decision.requested_action), "applied": list(decision.applied_action), "outcome": decision.outcome, "reason": decision.reason_code}, "before": before, "after": after, "metrics": metrics, "safety_delta": safety_delta, "effects": dict(effects), "first_conflict": next((row for row in branch_conflicts if row.get("kind") == "first_hard_separation_trace"), None), "preexisting_first_conflict": next((row for row in base.conflict_events[:conflict_start] if row.get("kind") == "first_hard_separation_trace"), None), "safety_reason_histogram": {key: dict(value) for key, value in safety_reason_hist.items()}, "candidate_audit": list(wrapper.coordinator.history), "geometry_before": before["actual_state_10d"][:6], "geometry_after": after["actual_state_10d"][:6], "target_before": before["target_rh"], "target_after": after["target_rh"], "capacity_before": before["capacity"], "capacity_after": after["capacity"], "headways_before": before["headways"], "headways_after": dict(base.airspace.headways), "speed_profile_before": before["speed_profile_bias"], "speed_profile_after": base.airspace.speed_profile_bias}


def _task(snapshot_id: str, payload: bytes, action: Candidate, horizon: float):
    started = time.perf_counter()
    result = _run_branch(payload, action, horizon)
    return snapshot_id, action.name, horizon, time.perf_counter() - started, result


def run_audit(captures: list[tuple[dict, bytes]], *, full_horizon: bool) -> tuple[dict, dict]:
    tasks = []
    for state, payload in captures:
        selected = TRACE_ACTIONS
        horizons = HORIZONS if full_horizon and state["time"] in (1080., 1800.) else (540.,)
        for horizon in horizons:
            for action in selected: tasks.append((f"high-seed0-t{int(state['time'])}", payload, action, horizon))
    workers = min(len(tasks), max(1, available_cpus() - 2)); results = {}; durations=[]; started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        futures = [executor.submit(_task, *task) for task in tasks]
        for future in as_completed(futures):
            sid, action, horizon, duration, row = future.result(); results.setdefault(sid, {}).setdefault(str(int(horizon)), {})[action] = row; durations.append(duration)
    telemetry={"available_cpu_count":available_cpus(),"worker_count":workers,"branch_count":len(tasks),"wall_clock_seconds":time.perf_counter()-started,"sequential_equivalent_seconds":float(sum(durations)),"parallel_speedup_estimate":float(sum(durations)/max(1e-9,time.perf_counter()-started))}
    return results, telemetry


def _classification(row: dict, baseline: dict) -> str:
    decision=row["decision"]
    if not any(abs(value)>1e-8 for value in decision["applied"]): return "ACTION_NOT_APPLIED"
    effects=row["effects"]; before=row["before"]
    if sum(item["eligible"] for item in before["interfaces"].values()) == 0: return "ACTION_APPLIED_NO_TRAFFIC_TO_INFLUENCE"
    if effects.get("approved_releases", 0) == 0 and any(item["downstream_slack"] > 0 for item in before["interfaces"].values()): return "ACTION_APPLIED_BUT_BLOCKED_DOWNSTREAM"
    if row["metrics"]["completed"] != baseline["metrics"]["completed"] or row["metrics"]["all_request_waiting_area"] != baseline["metrics"]["all_request_waiting_area"] or effects.get("approved_releases",0) != baseline["effects"].get("approved_releases",0): return "ACTION_APPLIED_WITH_OPERATIONAL_EFFECT"
    return "ACTION_APPLIED_NO_TRAFFIC_TO_INFLUENCE"


def analyze(captures, results, telemetry, *, root_cause_evidence: dict | None = None) -> dict:
    pipeline_pass=True
    snapshot_rows={f"high-seed0-t{int(state['time'])}":state for state,_ in captures}
    operational=[]; horizon=[]
    for sid, groups in results.items():
        for horizon_key, actions in groups.items():
            baseline=actions["no_op"]
            for name,row in actions.items():
                safe=all(row["safety_delta"].get(key, -1)==0 for key in ("collisions","nmac","hard_separation_violations","final_overcapacity_steps"))
                reservations=row["after"]["reservations"]
                consistency=reservations["expired_or_stale_count"] == 0 and not reservations["orphaned_reservation_ids"]
                pipeline_pass &= consistency
                category=_classification(row,baseline)
                operational.append({"snapshot_id":sid,"horizon":float(horizon_key),"action":name,"category":category,"safe":safe,"decision":row["decision"],"safety_delta":row["safety_delta"],"effects":row["effects"],"first_conflict":row["first_conflict"],"safety_reason_histogram":row["safety_reason_histogram"],"reservation_after":reservations,"requested_state_delta":(np.asarray(row["decision"]["requested"][:6])-np.asarray(baseline["decision"]["requested"][:6])).tolist(),"applied_state_delta":(np.asarray(row["after"]["actual_state_10d"])-np.asarray(baseline["after"]["actual_state_10d"])).tolist(),"target_delta":(np.asarray(row["after"]["target_rh"])-np.asarray(baseline["after"]["target_rh"])).tolist(),"metric_delta":{"completed":row["metrics"]["completed"]-baseline["metrics"]["completed"],"waiting_area":row["metrics"]["all_request_waiting_area"]-baseline["metrics"]["all_request_waiting_area"],"unresolved":row["metrics"]["unresolved_count"]-baseline["metrics"]["unresolved_count"],"energy":row["metrics"]["energy_mean"]-baseline["metrics"]["energy_mean"],"fairness":row["metrics"]["priority_fairness_std"]-baseline["metrics"]["priority_fairness_std"]}})
            if horizon_key in {"180","540","1080"}:
                for name,row in actions.items():
                    if name == "no_op":continue
                    horizon.append({"snapshot_id":sid,"horizon":float(horizon_key),"action":name,"completed_delta":row["metrics"]["completed"]-baseline["metrics"]["completed"],"waiting_delta":row["metrics"]["all_request_waiting_area"]-baseline["metrics"]["all_request_waiting_area"],"unresolved_delta":row["metrics"]["unresolved_count"]-baseline["metrics"]["unresolved_count"],"energy_delta":row["metrics"]["energy_mean"]-baseline["metrics"]["energy_mean"],"fairness_delta":row["metrics"]["priority_fairness_std"]-baseline["metrics"]["priority_fairness_std"]})
    long = [item for item in horizon if item["horizon"] == 1080 and (item["completed_delta"] != 0 or item["waiting_delta"] != 0)]
    medium = [item for item in horizon if item["horizon"] == 540 and (item["completed_delta"] != 0 or item["waiting_delta"] != 0)]
    observed_pipeline_bug=bool(root_cause_evidence and not root_cause_evidence.get("safe", True))
    if observed_pipeline_bug or not pipeline_pass: result="PIPELINE_BUG"
    elif long and not medium: result="EFFECT_LATENCY_MISMATCH"
    elif not long: result="VALID_BUT_UNCONTROLLABLE_REGIME"
    else: result="INCONCLUSIVE"
    causal="arrival overload -> external backlog -> outer accumulation -> "
    late=[state for state in snapshot_rows.values() if state["external_queue"] > 0 and state["slack"]["middle"]>0]
    if late: causal += "outer O2M safety rejection / holder-spacing state leaves middle capacity unused; the operational trace separates action application from release approval"
    else: causal += "no representative downstream-starvation state captured"
    return {"result":result,"safety":"PASS" if all(item["safe"] for item in operational) else "FAIL","pipeline_consistency":"PASS" if pipeline_pass else "FAIL","pre_fix_pipeline_bug_evidence":root_cause_evidence,"operational_action_effect":"PROVEN" if any(item["category"]=="ACTION_APPLIED_WITH_OPERATIONAL_EFFECT" for item in operational) else "NOT_PROVEN","horizon_540_adequacy":"FAIL" if result=="EFFECT_LATENCY_MISMATCH" else "PASS","late_high_controllability":"FAIL" if result=="VALID_BUT_UNCONTROLLABLE_REGIME" else "UNCERTAIN","causal_bottleneck_chain":causal,"snapshot_states":snapshot_rows,"operational":operational,"horizon":horizon,"telemetry":telemetry}


def write_report(report: dict, output: str | Path) -> None:
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/"a67_controllability_audit.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    with (output/"a67_pipeline_trace.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["snapshot_id","horizon","action","category","safe","safety_delta","requested_state_delta","applied_state_delta","target_delta","metric_delta","effects","first_conflict","safety_reason_histogram","decision","reservation_after"]); writer.writeheader();writer.writerows(report["operational"])
    with (output/"a67_horizon_sensitivity.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["snapshot_id","horizon","action","completed_delta","waiting_delta","unresolved_delta","energy_delta","fairness_delta"]);writer.writeheader();writer.writerows(report["horizon"])
    evidence=report.get("pre_fix_pipeline_bug_evidence") or {}
    first=evidence.get("first_conflict") or {}
    compact_evidence={"safe": evidence.get("safe"), "time": evidence.get("time"), "telemetry": evidence.get("telemetry"),
                      "first_conflict": {key:first.get(key) for key in ("time","pair","phase_pair","actual_distance","swept_min_distance")}}
    unsafe=[row for row in report["operational"] if not row["safe"]]
    lines=["# A6.7 controllability & bottleneck causal audit","",f"A6.7 result: **{report['result']}**",f"Post-fix branch safety: **{report['safety']}**",f"Post-fix reservation consistency: **{report['pipeline_consistency']}**",f"Operational action effect: **{report['operational_action_effect']}**",f"540 s horizon adequacy: **{report['horizon_540_adequacy']}**",f"Late-high controllability: **{report['late_high_controllability']}**","",f"Causal chain: {report['causal_bottleneck_chain']}","",f"Pre-fix root-cause evidence: {json.dumps(compact_evidence, default=str)}",f"Unsafe short branches: {len(unsafe)}",f"Parallel telemetry: {report['telemetry']}","","PPO authorization: **STOP**. Evidence type: newly_run internal causal diagnostic; no PPO checkpoint was loaded, modified, or trained."]
    (output/"a67_bottleneck_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--full-horizon",action="store_true");parser.add_argument("--output",default=None);parser.add_argument("--time",action="append",type=float,default=[]);parser.add_argument("--capture-only",action="store_true");parser.add_argument("--explain-o2m",action="store_true");parser.add_argument("--parallel-prefixes",action="store_true");parser.add_argument("--root-cause-evidence");args=parser.parse_args()
    captures=capture_high_seed0(tuple(args.time) if args.time else TARGET_TIMES, parallel_prefixes=args.parallel_prefixes)
    output=Path(args.output) if args.output else Path.home()/"training_outputs"/"terminal"/"event_gated_stage15_diagnostics"/"a67"
    if args.capture_only:
        output.mkdir(parents=True,exist_ok=True)
        (output/"a67_capture_trace.json").write_text(json.dumps({f"high-seed0-t{int(state['time'])}": state for state,_ in captures},indent=2,default=str),encoding="utf-8")
        print(json.dumps({"output":str(output),"capture_count":len(captures)})); return
    if args.explain_o2m:
        output.mkdir(parents=True,exist_ok=True)
        explanation={f"high-seed0-t{int(state['time'])}": explain_o2m_capture(payload) for state,payload in captures}
        (output/"a67_o2m_explanation.json").write_text(json.dumps(explanation,indent=2,default=str),encoding="utf-8")
        print(json.dumps({"output":str(output),"capture_count":len(captures)})); return
    evidence=json.loads(Path(args.root_cause_evidence).read_text(encoding="utf-8")) if args.root_cause_evidence else None
    results,telemetry=run_audit(captures,full_horizon=args.full_horizon); report=analyze(captures,results,telemetry,root_cause_evidence=evidence)
    write_report(report,output); print(json.dumps({"output":str(output),"result":report["result"],"safety":report["safety"],"pipeline":report["pipeline_consistency"]}))

if __name__=="__main__":main()
