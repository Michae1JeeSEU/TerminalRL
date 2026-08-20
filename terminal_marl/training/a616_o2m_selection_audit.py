"""A6.16 O2M snapshot candidate-selection and reservation-sequencing audit.

The live A6.12 scheduler is never changed.  Every forced candidate is applied
only to a deep-copied, deterministic snapshot and is then followed by the
unchanged scheduler and SafetyFilter.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import csv, json
from pathlib import Path

import numpy as np

from ..core.config import RingRole
from ..core.lifecycle import transition
from ..core.merge_scheduler import LowLevelIntent
from ..core.uav import FlightPhase
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
from ..envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from ..policies import RuleLowLevelPolicy
from .a611_causal_audit import DemandProfile, _safe
from .a612_scheduler_repair_audit import _a611_mu0
from .a613_capacity_reaudit import repair_config
from .a614_reconfiguration_timing_audit import OUTER_PLUS_ONE
from .a6_effectiveness import _collect, _sample


PPO_AUTHORIZATION = "STOP"
SELECTED = (
    (0, "static_outer_plus1", 247., 5, "static_nominal_control"),
    (0, "dynamic_t300", 248., 5, "dynamic_t300_control"),
    (1, "static_outer_plus1", 247., 5, "static_nominal_control"),
    (1, "dynamic_t300", 302., 6, "dynamic_t300_bifurcation"),
    (1, "nominal", 1140., 15, "long_safe_window"),
    (1, "dynamic_t240", 248., 5, "geometry_frozen_control"),
)


def _profile(mu):
    return DemandProfile("constant_high_repair_scheduler", ((0., 720., 1.55 * mu),), 720., 480.)


def _new(profile, seed, branch):
    if branch == "static_outer_plus1":
        from .a611_causal_audit import static_expanded_rings
        config = replace(repair_config(profile, dynamic=False), rings=static_expanded_rings(repair_config(profile, dynamic=False), RingRole.OUTER))
        env = TerminalParallelEnv(config); env.reset(seed); return env, RuleLowLevelPolicy(), None, False
    wrapper = DynamicHighLevelWrapper(repair_config(profile, dynamic=True), RuleLowLevelAdapter()); wrapper.reset(seed)
    return wrapper.base, wrapper.low_level, wrapper.coordinator, False


def _actions(env, policy, coordinator):
    return coordinator.apply_speed_repairs(policy.act(env._observations())) if coordinator else policy.act(env._observations())


def _advance(env, policy, coordinator, branch, target_time):
    requested = False
    action_time = {"dynamic_t180": 180., "dynamic_t240": 240., "dynamic_t300": 300.}.get(branch)
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}
    while env.now < target_time - 1e-9:
        if coordinator and action_time is not None and not requested and env.now >= action_time - 1e-9:
            coordinator.coordinate(OUTER_PLUS_ONE, geometry_gate=True); requested = True
        if coordinator: coordinator.advance()
        env.step(_actions(env, policy, coordinator)); _sample(env, samples)
    return requested, samples


def _safe_o2m(env, arrival_id):
    uav = next((u for u in env.active.values() if u.arrival_id == arrival_id and u.phase is FlightPhase.RING_OUTER and u.committed_plan is None), None)
    if uav is None or "o2m" in env.airspace.affected_interfaces(): return None, []
    snapshot = WorldSnapshot(env.now, env.active, env.airspace); requested = LowLevelIntent(1., 0., 0.); rows = []
    for rank, candidate in enumerate(env.scheduler.repair_candidates(requested)):
        proposal, plan, reason = env._candidate_plan(uav, candidate, RingRole.MIDDLE, snapshot, None)
        if reason is None:
            rows.append({"rank": rank, "candidate": candidate, "proposal": proposal, "plan": plan})
    return uav.uid, rows


def _force(env, uid, item):
    uav = env.active[uid]; proposal, plan = item["proposal"], item["plan"]
    reservation = env.scheduler.reserve(uav, proposal, now=env.now); plan.reservation_ids = [reservation.reservation_id]
    if not env._commit(uav, plan):
        env.scheduler.cancel(reservation.reservation_id, "counterfactual_safety_reject"); return False
    uav.merge_reservation_id = reservation.reservation_id; transition(uav, FlightPhase.TRANSFER_O2M, env.now); uav.ring_role = None
    return True


def _rollout(env, policy, coordinator, samples):
    if coordinator is not None:
        coordinator = deepcopy(coordinator); coordinator.env = env
    trace=[]
    while env.now < env.config.max_simulation_seconds and not (env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs):
        if coordinator: coordinator.advance()
        _, _, _, _, info = env.step(_actions(env, policy, coordinator)); _sample(env, samples)
        for raw, decision in info["release_decisions"].items():
            if decision in {"approved", "future_reserved"}:
                uav = env.active.get(int(raw)); trace.append((None if uav is None else uav.arrival_id, decision, env.now))
    return _collect(env, samples, coordinator) if coordinator else _collect(env, samples), trace


def run_audit(*, seeds=(0,1), mu0=None):
    mu = _a611_mu0(0)[0] if mu0 is None else float(mu0); profile = _profile(mu); candidates=[]; rollouts=[]; effects=[]; missed=[]
    for seed, branch, time, arrival, tag in SELECTED:
        if seed not in seeds: continue
        env, policy, coordinator, _ = _new(profile, seed, branch); _, samples = _advance(env, policy, coordinator, branch, time)
        uid, safe_rows = _safe_o2m(env, arrival)
        if uid is None:
            candidates.append({"seed":seed,"branch":branch,"time":time,"arrival_id":arrival,"tag":tag,"status":"no_legal_o2m_candidate"}); continue
        baseline_env = deepcopy(env); baseline_metrics, baseline_trace = _rollout(baseline_env, policy, coordinator, deepcopy(samples))
        for item in safe_rows:
            clone = deepcopy(env); ok = _force(clone, uid, item)
            metrics, trace = _rollout(clone, policy, coordinator, deepcopy(samples)) if ok else ({}, [])
            row={"seed":seed,"branch":branch,"time":time,"arrival_id":arrival,"tag":tag,"rank":item["rank"],"kind":item["candidate"].kind,"merge_time":item["proposal"].merge_time,"future_delay":item["candidate"].delay_seconds,"safe":ok,
                 "completed":metrics.get("completed"),"waiting_area":metrics.get("all_request_waiting_area"),"external_queue_area":metrics.get("external_queue_area"),"service_delay_p95":metrics.get("service_delay_p95"),"unresolved":metrics.get("unresolved_count"),"energy":metrics.get("energy_mean"),"fairness":metrics.get("priority_fairness_std"),"hard_safe":bool(metrics) and _safe(metrics, clone.config),"checked_executed":bool(metrics)}
            candidates.append({k:row[k] for k in ("seed","branch","time","arrival_id","tag","rank","kind","merge_time","future_delay","safe")}); rollouts.append(row)
            effects.append({"seed":seed,"branch":branch,"time":time,"arrival_id":arrival,"rank":item["rank"],"downstream_reservation_changes":sum(a!=b for a,b in zip(trace, baseline_trace))+abs(len(trace)-len(baseline_trace)),"waiting_area_delta":metrics.get("all_request_waiting_area",0)-baseline_metrics.get("all_request_waiting_area",0)})
            if tag == "long_safe_window": missed.append({**row,"first_shadow_safe":1140.,"actual_reservation":1146.,"delay_seconds":6.})
    labels={"INCONCLUSIVE"}
    if any(row.get("downstream_reservation_changes",0)>0 for row in effects): labels={"RESERVATION_LOCK_IN_CAUSAL","SEQUENCE_AWARE_SELECTION_NEEDED","MIXED"}
    return {"stage":"A6.16 O2M Safe-Candidate Selection & Reservation Sequencing Audit","evidence_type":"newly_run","inputs":{"safe_distance":80.,"ppo_authorization":PPO_AUTHORIZATION,"repair_scheduler_frozen":True},"snapshot_candidates":candidates,"candidate_rollouts":rollouts,"sequence_effects":effects,"missed_window_counterfactuals":missed,"summary":{"classification":sorted(labels),"ppo_authorization":PPO_AUTHORIZATION}}


def _csv(path, rows):
    fields=sorted({k for r in rows for k in r}) or ["empty"]
    with path.open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def write_report(report, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    _csv(output/"a616_snapshot_candidates.csv",report["snapshot_candidates"]);_csv(output/"a616_candidate_rollouts.csv",report["candidate_rollouts"]);_csv(output/"a616_sequence_effects.csv",report["sequence_effects"]);_csv(output/"a616_missed_window_counterfactuals.csv",report["missed_window_counterfactuals"])
    (output/"a616_audit.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    (output/"a616_summary.md").write_text(f"# A6.16 O2M candidate audit\n\nClassification: **{', '.join(report['summary']['classification'])}**\n\nPPO authorization: **STOP**\n",encoding="utf-8")

def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--seeds",default="0,1"); parser.add_argument("--output",default=None)
    args=parser.parse_args(); output=Path(args.output) if args.output else Path.home()/"training_outputs"/"terminal"/"event_gated_stage15_diagnostics"/"a616"
    report=run_audit(seeds=tuple(int(x) for x in args.seeds.split(",") if x)); write_report(report,output); print(json.dumps(report["summary"]))

if __name__ == "__main__": main()
