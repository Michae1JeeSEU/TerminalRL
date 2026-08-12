"""Frozen-checkpoint A6 effectiveness qualification with paired CRN reports."""
from __future__ import annotations

from dataclasses import dataclass, replace
import csv
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.config import SimConfig
from ..core.uav import TerminalOutcome
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, FrozenLowLevelPolicy, RuleLowLevelAdapter
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import DynamicRulePolicy
from .high_level_a6 import load_high_level
from .run_a6 import A5_HASH


@dataclass(frozen=True)
class DemandScenario:
    name: str; total_uavs: int; rate: float; max_seconds: float; arrival_model: str = "finite_cohort"


SCENARIOS = (
    DemandScenario("low", 80, .020, 5200.), DemandScenario("medium", 120, .040, 5200.),
    DemandScenario("high", 180, .080, 5000.), DemandScenario("bursty", 180, .040, 5500., "bursty_wave"),
)


def scenario_config(scenario: DemandScenario, mode: str) -> SimConfig:
    return SimConfig(experiment_mode=mode, high_level_enabled=mode != "fixed_rule", reset_mode="flow_from_empty",
                     arrival_model=scenario.arrival_model, total_requested_uavs=scenario.total_uavs,
                     arrival_rate_per_second=scenario.rate, max_simulation_seconds=scenario.max_seconds)


def _percentiles(values: list[float], prefix: str) -> dict[str, float]:
    if not values: return {f"{prefix}_{name}": float("inf") for name in ("mean", "median", "p95", "p99")}
    return {f"{prefix}_mean": float(np.mean(values)), f"{prefix}_median": float(np.median(values)),
            f"{prefix}_p95": float(np.percentile(values, 95)), f"{prefix}_p99": float(np.percentile(values, 99))}


def _action_ranges(actions: list[list[float]]) -> dict[str, list[float]]:
    """Retain the complete 10-D decision trace and compact, auditable ranges."""
    matrix = np.asarray(actions or [np.zeros(10)], dtype=float)
    return {"mean": matrix.mean(axis=0).tolist(), "min": matrix.min(axis=0).tolist(), "max": matrix.max(axis=0).tolist()}


def _collect(base: TerminalParallelEnv, samples: dict[str, list[float]], coordinator=None, action_events: list[dict] | None = None) -> dict:
    completed = [u for u in base.completed_uavs if u.completion_time is not None]
    service = [u.completion_time-u.request_time for u in completed]
    touchdown = [u.touchdown_time-u.request_time for u in completed if u.touchdown_time is not None]
    holding = {role.value: float(np.mean([u.holding_time_by_ring.get(role, 0.) for u in completed])) if completed else 0. for role in base.airspace._actual}
    result = {"requested":len(base.ledger.requested), "completed":len(completed), "throughput":len(completed)/max(1.,base.now),
              "external_queue_area":float(np.sum(samples["queue"])*base.config.dt), "external_queue_max":max(samples["queue"],default=0.),
              "oldest_external_wait":max(samples["oldest_wait"],default=0.), "final_utilization":float(np.mean(samples["final_util"],dtype=float)) if samples["final_util"] else 0.,
              "lz_utilization":float(np.mean(samples["lz_util"],dtype=float)) if samples["lz_util"] else 0.,
              "energy_mean":float(np.mean([u.energy_consumed for u in completed])) if completed else 0.,
              "priority_fairness_std":float(np.std(service)) if service else float("inf"), **{f"holding_{key}":value for key,value in holding.items()},
              **_percentiles(service,"service_delay"), **_percentiles(touchdown,"touchdown_delay"), **{key:float(value) for key,value in base.telemetry.items()}}
    if coordinator is not None:
        history=coordinator.history; applied=[row["applied_action"] for row in history]; requested=[row["requested_action"] for row in history]
        actual=np.asarray(samples["actual"] or [base.airspace.state_vector()],float); cap=np.asarray(samples["capacity"] or [[base.airspace.capacity(role) for role in base.airspace._actual]],float)
        structural_requests=sum(any(abs(x)>coordinator.deadband for x in row[:6]) for row in requested)
        delays=[event["deferred_until"]-event["time"] for event in (action_events or []) if event["outcome"]=="delayed" and event["deferred_until"] is not None]
        result.update({"high_decisions":len(requested),"structural_requests":structural_requests,
                       "deadband_noops":sum(row["outcome"]=="no_op" for row in history),"effective_geometry_updates":base.telemetry["geometry_updates"],
                       "headway_speed_updates":sum(any(abs(x)>1e-8 for x in row[6:]) for row in applied),"traffic_repair_count":coordinator.telemetry["traffic_repairs"],
                       "traffic_repair_magnitude":coordinator.telemetry["traffic_repair_magnitude"],"transition_delay_count":coordinator.telemetry["geometry_delays"],
                       "transition_delay_seconds":float(sum(delays)),"projection_magnitude":coordinator.telemetry["geometry_projection_magnitude"],
                       "projection_count":sum(row["outcome"]=="projected" for row in history),"defer_count":sum(row["outcome"]=="delayed" for row in history),"reject_count":coordinator.telemetry["geometry_rejects"],
                       "coordinator_attempts":coordinator.telemetry["coordinator_attempts"],"pending_retries":coordinator.telemetry["pending_retries"],"effective_updates":coordinator.telemetry["effective_updates"],
                       "reconfiguration_execution_rate":base.telemetry["geometry_updates"]/max(1,structural_requests),"shield_cost":coordinator.telemetry["traffic_repair_magnitude"]+.1*coordinator.telemetry["geometry_projection_magnitude"]+.5*(coordinator.telemetry["geometry_delays"]+coordinator.telemetry["geometry_rejects"]),
                       "actual_radius_min":actual[:, :3].min(axis=0).tolist(),"actual_radius_max":actual[:, :3].max(axis=0).tolist(),
                       "actual_height_min":actual[:, 3:6].min(axis=0).tolist(),"actual_height_max":actual[:, 3:6].max(axis=0).tolist(),
                       "speed_profile_min":float(actual[:, 6].min()),"speed_profile_max":float(actual[:, 6].max()),
                       "headway_min":actual[:, 7:10].min(axis=0).tolist(),"headway_max":actual[:, 7:10].max(axis=0).tolist(),
                       "capacity_min":cap.min(axis=0).tolist(),"capacity_max":cap.max(axis=0).tolist(),
                       "requested_action_summary":_action_ranges(requested),"applied_action_summary":_action_ranges(applied),"action_events":action_events or []})
    return result


def _sample(base: TerminalParallelEnv, samples: dict) -> None:
    queue=list(base.admission.queue.items); samples["queue"].append(len(queue)); samples["oldest_wait"].append(max((item.external_wait_time for item in queue),default=0.))
    samples["final_util"].append(sum(u.phase.value=="final" for u in base.active.values())/max(1,base.config.final_queue_capacity))
    samples["lz_util"].append(sum(zone.status.value!="free" for zone in base.lz_manager.zones)/len(base.lz_manager.zones))
    samples["actual"].append(base.airspace.state_vector().copy()); samples["capacity"].append([base.airspace.capacity(role) for role in base.airspace._actual])


def _run_dynamic(config: SimConfig, seed: int, high_action: Callable, low_level) -> dict:
    wrapper=DynamicHighLevelWrapper(config,low_level); observation,_=wrapper.reset(seed); samples={key:[] for key in ("queue","oldest_wait","final_util","lz_util","actual","capacity")}; action_events=[]
    while True:
        action=high_action(observation,wrapper); decision=wrapper.coordinator.coordinate(action,geometry_gate=wrapper.geometry_event_gate())
        action_events.append({"time":wrapper.base.now,"requested_action":list(decision.requested_action),"applied_action":list(decision.applied_action),"outcome":decision.outcome,"reason_code":decision.reason_code,"repair_count":decision.repair_count,"repair_magnitude":decision.repair_magnitude,"projection_magnitude":decision.projection_magnitude,"deferred_until":decision.deferred_until,"safety_reason":decision.safety_reason})
        for _ in range(round(config.high_level_decision_interval_seconds/config.dt)):
            advanced=wrapper.coordinator.advance()
            if advanced is not None:
                action_events.append({"time":wrapper.base.now,"requested_action":list(advanced.requested_action),"applied_action":list(advanced.applied_action),"outcome":advanced.outcome,"reason_code":advanced.reason_code,"repair_count":advanced.repair_count,"repair_magnitude":advanced.repair_magnitude,"projection_magnitude":advanced.projection_magnitude,"deferred_until":advanced.deferred_until,"safety_reason":advanced.safety_reason})
            low_actions=wrapper.coordinator.apply_speed_repairs(wrapper.low_level.act(wrapper.base._observations())); wrapper.base.step(low_actions); _sample(wrapper.base,samples)
            if wrapper._done() or wrapper.base.now>=config.max_simulation_seconds: break
        observation=wrapper.observation()
        if wrapper._done() or wrapper.base.now>=config.max_simulation_seconds: break
    return _collect(wrapper.base,samples,wrapper.coordinator,action_events)


def _run_fixed(config: SimConfig, seed: int) -> dict:
    env=TerminalParallelEnv(config);env.reset(seed);samples={key:[] for key in ("queue","oldest_wait","final_util","lz_util","actual","capacity")}
    while True:
        env.step();_sample(env,samples)
        if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs: break
        if env.now>=config.max_simulation_seconds: break
    return _collect(env,samples)


_WORKER_CACHE: dict[tuple[str, str], object] = {}

def _task(scenario: DemandScenario, policy_name: str, seed: int, rule_checkpoint: str, frozen_high_checkpoint: str, frozen_low_checkpoint: str) -> tuple[str, str, int, dict]:
    """One independent CRN rollout; safe to execute in a separate process."""
    if policy_name == "fixed_rule":
        return scenario.name, policy_name, seed, _run_fixed(scenario_config(scenario,"fixed_rule"),seed)
    if policy_name == "dynamic_rule":
        config=scenario_config(scenario,"dynamic_rule")
        return scenario.name, policy_name, seed, _run_dynamic(config,seed,lambda obs,w:DynamicRulePolicy().act(obs,w._context()),RuleLowLevelAdapter())
    if policy_name == "learned_high_rule_low":
        config=scenario_config(scenario,"learned_high_rule_low"); key=("high_rule",rule_checkpoint)
        if key not in _WORKER_CACHE: _WORKER_CACHE[key]=load_high_level(rule_checkpoint,config)
        policy=_WORKER_CACHE[key]
        return scenario.name, policy_name, seed, _run_dynamic(config,seed,lambda obs,w:policy.act(obs,deterministic=True)[0],RuleLowLevelAdapter())
    config=scenario_config(scenario,"learned_high_frozen_low"); high_key=("high_frozen",frozen_high_checkpoint); low_key=("low_frozen",frozen_low_checkpoint)
    if high_key not in _WORKER_CACHE: _WORKER_CACHE[high_key]=load_high_level(frozen_high_checkpoint,config,expected_low_hash=A5_HASH)
    if low_key not in _WORKER_CACHE: _WORKER_CACHE[low_key]=FrozenLowLevelPolicy(frozen_low_checkpoint,config,A5_HASH)
    high=_WORKER_CACHE[high_key]; low=_WORKER_CACHE[low_key]
    return scenario.name, policy_name, seed, _run_dynamic(config,seed,lambda obs,w:high.act(obs,deterministic=True)[0],low)


def run_matrix(*, rule_checkpoint: str, frozen_high_checkpoint: str, frozen_low_checkpoint: str, seeds: list[int], workers: int = 6) -> dict:
    policies=("fixed_rule","dynamic_rule","learned_high_rule_low","learned_high_frozen_a5_low")
    raw={scenario.name:{policy:[] for policy in policies} for scenario in SCENARIOS}
    tasks=[(scenario,policy,seed,rule_checkpoint,frozen_high_checkpoint,frozen_low_checkpoint) for scenario in SCENARIOS for policy in policies for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(_task,*task) for task in tasks]
        for future in futures:
            scenario_name, policy, seed, row = future.result()
            raw[scenario_name][policy].append((seed,row))
    for policies_rows in raw.values():
        for policy, rows in policies_rows.items(): policies_rows[policy]=[row for _,row in sorted(rows)]
    return raw


def _aggregate(rows: list[dict]) -> dict:
    result={}
    for key in rows[0]:
        values=[row[key] for row in rows if isinstance(row[key],(int,float)) and np.isfinite(row[key])]
        if values:
            mean=float(np.mean(values)); std=float(np.std(values,ddof=1)) if len(values)>1 else 0.; ci=1.96*std/max(1,len(values))**.5
            result[key]={"mean":mean,"std":std,"ci95_low":mean-ci,"ci95_high":mean+ci}
    return result


def write_report(raw: dict, output: str | Path) -> dict:
    output=Path(output);output.mkdir(parents=True,exist_ok=True); aggregate={scenario:{policy:_aggregate(rows) for policy,rows in policies.items()} for scenario,policies in raw.items()}
    paired={}
    for scenario, policies in raw.items():
        paired[scenario]={}
        fixed=policies["fixed_rule"]
        for policy,rows in policies.items():
            if policy=="fixed_rule":continue
            paired[scenario][policy]={}
            scalar_keys=set.intersection(*(set(row) for row in [*rows,*fixed]))
            for key in sorted(scalar_keys):
                deltas=[row[key]-base[key] for row,base in zip(rows,fixed) if isinstance(row[key],(int,float)) and isinstance(base[key],(int,float)) and np.isfinite(row[key]) and np.isfinite(base[key])]
                if deltas:
                    mean=float(np.mean(deltas)); std=float(np.std(deltas,ddof=1)) if len(deltas)>1 else 0.; ci=1.96*std/max(1,len(deltas))**.5
                    paired[scenario][policy][key]={"per_seed":deltas,"mean":mean,"std":std,"ci95_low":mean-ci,"ci95_high":mean+ci}
    required_safety = ("collisions", "nmac", "hard_separation_violations", "hard_separation_exposure_steps", "nmac_exposure_steps", "collision_exposure_steps")
    safety=all(all(key in row for key in required_safety) and row["collisions"]==0 and row["nmac"]==0 and row["hard_separation_violations"]==0 for policies in raw.values() for rows in policies.values() for row in rows)
    report={"seeds":len(next(iter(next(iter(raw.values())).values()))),"safety_passed":safety,"aggregate":aggregate,"paired_vs_fixed_rule":paired,"raw":raw}
    (output/"a6_effectiveness_report.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    with (output/"a6_effectiveness_report.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle);writer.writerow(["scenario","policy","metric","mean","std","ci95_low","ci95_high"])
        for scenario,policies in aggregate.items():
            for policy,metrics in policies.items():
                for metric,values in metrics.items():writer.writerow([scenario,policy,metric,*[values[k] for k in ("mean","std","ci95_low","ci95_high")]])
    lines=["# A6 effectiveness qualification",f"\nHard-safety precondition: **{'PASS' if safety else 'FAIL'}**.\n", "| Scenario | Policy | Completed | Throughput | Service P95 delay | Queue area | Geometry updates |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scenario,policies in aggregate.items():
        for policy,metrics in policies.items():
            get=lambda key:metrics.get(key,{}).get("mean",float("nan"));lines.append(f"| {scenario} | {policy} | {get('completed'):.2f} | {get('throughput'):.5f} | {get('service_delay_p95'):.2f} | {get('external_queue_area'):.1f} | {get('effective_geometry_updates'):.2f} |")
    (output/"a6_effectiveness_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report
