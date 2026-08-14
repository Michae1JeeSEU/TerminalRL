"""A6.9 post-safety counterfactual high-action-value requalification.

This diagnostic deliberately has no PPO import, checkpoint load, reward edit,
or safety-semantics switch.  It restores identical dynamic-rule snapshots and
evaluates a finite, canonical-10D candidate set through the frozen A6.8 stack.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import pickle
import time
from typing import Iterable

import numpy as np

from ..core.config import RingRole
from ..core.uav import TerminalOutcome
from .a63_dynamic_screen import SHORT_SCENARIOS
from .a65_counterfactual_audit import Candidate, NO_OP, REQUIRED_SAFETY, available_cpus, candidate_set
from .a6_effectiveness import _collect, _sample
from .a66_observation_audit import _diverse_select, _extended_features

HORIZON_SECONDS = 540.0
LONG_HORIZON_SECONDS = 1080.0
MEANINGFUL_COMPLETION_MARGIN = 1
MEANINGFUL_WAITING_MARGIN = 50_000.0


def _worker_init() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
    except ImportError:
        pass


def _control(env) -> dict:
    return {"actual": env.airspace.state_vector().astype(float).tolist(),
            "target": [env.airspace.target_ring(role).radius for role in RingRole] + [env.airspace.target_ring(role).height for role in RingRole],
            "capacity": [env.airspace.capacity(role) for role in RingRole]}


def _decision_trace(wrapper, decision, requested, before) -> dict:
    after = _control(wrapper.base)
    return {"request_id": decision.request_id, "original_request_time": decision.original_request_time,
            # Keep the legacy collection fields alongside the richer A6.9
            # labels.  _collect() consumes outcome/time when deriving delay
            # telemetry; these aliases are observational only.
            "time": wrapper.base.now, "event_time": wrapper.base.now,
            "outcome": decision.outcome, "requested_action": list(map(float, requested)),
            "projected_action": list(decision.update.applied_action), "initial_applied_action": list(decision.applied_action),
            "execution_outcome": decision.execution_outcome, "coordinator_outcome": decision.outcome,
            "reason_code": decision.reason_code, "safety_outcome": decision.safety_reason or "approved",
            "actual_application_time": decision.actual_application_time,
            "application_delay_seconds": decision.application_delay_seconds,
            "operational_effective": decision.operational_effective,
            "structural_effective": decision.structural_effective,
            "capacity_effective": decision.capacity_effective,
            "control_before": before, "control_after": after,
            "candidate_prediction": dict(wrapper.coordinator._candidate_audit),
            "repair_count": decision.repair_count, "repair_magnitude": decision.repair_magnitude,
            "projection_magnitude": decision.projection_magnitude, "deferred_until": decision.deferred_until}


def _run_macro(wrapper, requested: np.ndarray, until: float, samples: dict, traces: list[dict]) -> None:
    before = _control(wrapper.base)
    decision = wrapper.coordinator.coordinate(requested, geometry_gate=wrapper.geometry_event_gate())
    traces.append(_decision_trace(wrapper, decision, requested, before))
    while wrapper.base.now < until - 1e-9 and not wrapper._done():
        before_advance = _control(wrapper.base)
        deferred = wrapper.coordinator.advance()
        if deferred is not None:
            traces.append(_decision_trace(wrapper, deferred, np.asarray(deferred.requested_action, np.float32), before_advance))
        actions = wrapper.coordinator.apply_speed_repairs(wrapper.low_level.act(wrapper.base._observations()))
        wrapper.base.step(actions); _sample(wrapper.base, samples)


def _branch(snapshot: bytes, candidate: Candidate, horizon: float) -> dict:
    wrapper = pickle.loads(snapshot); start = wrapper.base.now; before = _control(wrapper.base)
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}
    traces: list[dict] = []; first = True
    while wrapper.base.now < start + horizon - 1e-9 and not wrapper._done():
        action = np.asarray(candidate.action if first else NO_OP, np.float32)
        _run_macro(wrapper, action, min(start + horizon, wrapper.base.now + wrapper.config.high_level_decision_interval_seconds), samples, traces)
        first = False
    metrics = _collect(wrapper.base, samples, wrapper.coordinator, traces)
    outcomes = Counter(item.value for item in wrapper.base.ledger.outcomes.values())
    safe = all(metrics.get(key) == 0 for key in REQUIRED_SAFETY) and metrics.get("max_concurrent_final", 0) <= wrapper.config.final_queue_capacity
    candidate_id = traces[0]["request_id"]
    own = [row for row in traces if row["request_id"] == candidate_id]
    predicted = [row["candidate_prediction"].get("prediction", {}).get("minimum_separation") for row in own]
    predicted = [value for value in predicted if value is not None]
    after = _control(wrapper.base)
    initial = own[0]
    return {"safe": safe, "metrics": metrics, "outcomes": dict(outcomes), "execution_trace": traces,
            "candidate_execution": {"request_id": candidate_id, "original_request_time": initial["original_request_time"],
                "first_actual_application_time": next((row["actual_application_time"] for row in own if row["actual_application_time"] is not None), None),
                "application_delay_seconds": next((row["application_delay_seconds"] for row in own if row["application_delay_seconds"] is not None), None),
                "requested_action": initial["requested_action"], "projected_action": initial["projected_action"],
                "initial_applied_action": initial["initial_applied_action"], "execution_outcomes": [row["execution_outcome"] for row in own],
                "operational_effective": any(row["operational_effective"] for row in own), "structural_effective": any(row["structural_effective"] for row in own),
                "capacity_effective": any(row["capacity_effective"] for row in own), "control_before": before, "control_after": after,
                "predicted_min_separation": min(predicted, default=None),
                "runtime_min_separation": metrics.get("minimum_swept_separation"),
                "prediction_runtime_gap": None if not predicted else min(predicted) - metrics.get("minimum_swept_separation", 0.)},
            "shield_cost": metrics.get("shield_cost", 0.)}


def evaluate_task(snapshot: bytes, snapshot_id: str, candidate: Candidate, horizon: float):
    started = time.perf_counter()
    row = _branch(snapshot, candidate, horizon)
    return snapshot_id, candidate.name, time.perf_counter() - started, row


def _collect_pool(scenario: str, seed: int, max_boundaries: int = 5) -> list[tuple[dict, bytes]]:
    from ..envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
    from ..policies import DynamicRulePolicy
    from .a6_effectiveness import scenario_config
    config = scenario_config(SHORT_SCENARIOS[scenario], "dynamic_rule")
    wrapper = DynamicHighLevelWrapper(config, RuleLowLevelAdapter()); observation, _ = wrapper.reset(seed); pool = []
    while wrapper.base.now < config.max_simulation_seconds and not wrapper._done():
        action = DynamicRulePolicy().act(observation, wrapper._context())
        observation, _, _, _, _ = wrapper.step(action)
        if wrapper.base.now >= config.high_level_decision_interval_seconds:
            sid = f"{scenario}-seed{seed}-t{int(wrapper.base.now)}"
            pool.append((_extended_features(wrapper, scenario, seed, sid), pickle.dumps(wrapper, protocol=pickle.HIGHEST_PROTOCOL)))
            if len(pool) >= max_boundaries:
                break
    return pool


def _source_task(scenario: str, seed: int, count: int):
    return scenario, seed, _diverse_select(_collect_pool(scenario, seed, max(5, count + 2)), count)


def select_snapshots(specs: Iterable[tuple[str, int, int]]) -> list[tuple[dict, bytes]]:
    specs = list(specs); selected = []
    # Snapshot prefixes are independent by (demand, seed).  Collect them in
    # worker processes so state selection does not become a serial parent
    # bottleneck before the candidate branch matrix starts.
    workers = min(len(specs), max(1, available_cpus() - 2))
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = [pool.submit(_source_task, *spec) for spec in specs]
        sources = [future.result() for future in futures]
    for _, _, values in sorted(sources):
        selected.extend(values)
    return selected


def _signature(row: dict) -> str:
    return json.dumps({"safe": row["safe"], "metrics": row["metrics"], "outcomes": row["outcomes"], "execution": row["candidate_execution"]}, sort_keys=True, default=str)


def _rank_key(name: str, rows: dict) -> tuple:
    row, metric, outcomes = rows[name], rows[name]["metrics"], rows[name]["outcomes"]
    return (-metric["completed"], metric["all_request_waiting_area"], metric["unresolved_count"], metric["unresolved_age_p95"],
            outcomes.get(TerminalOutcome.DIVERTED_LOW_SOC.value, 0), outcomes.get(TerminalOutcome.FAILED_ENERGY.value, 0),
            metric["service_delay_p95"], metric["service_delay_p99"], metric["energy_mean"], metric["priority_fairness_std"], row["shield_cost"], name)


def _meaningful(delta: dict) -> bool:
    return delta["completed"] >= MEANINGFUL_COMPLETION_MARGIN or (delta["completed"] == 0 and delta["waiting_area"] <= -MEANINGFUL_WAITING_MARGIN)


def _delta(row: dict, baseline: dict) -> dict:
    metric, base = row["metrics"], baseline["metrics"]
    common = sorted(set(metric["arrival_delay_by_id"]) & set(base["arrival_delay_by_id"]), key=int)
    paired = [metric["arrival_delay_by_id"][key] - base["arrival_delay_by_id"][key] for key in common]
    return {"completed": metric["completed"] - base["completed"], "waiting_area": metric["all_request_waiting_area"] - base["all_request_waiting_area"],
            "unresolved_count": metric["unresolved_count"] - base["unresolved_count"], "unresolved_age_mean": metric["unresolved_age_mean"] - base["unresolved_age_mean"],
            "unresolved_age_p95": metric["unresolved_age_p95"] - base["unresolved_age_p95"], "service_delay_p95": metric["service_delay_p95"] - base["service_delay_p95"],
            "service_delay_p99": metric["service_delay_p99"] - base["service_delay_p99"], "energy": metric["energy_mean"] - base["energy_mean"],
            "fairness": metric["priority_fairness_std"] - base["priority_fairness_std"], "diverted_low_soc": row["outcomes"].get("diverted_low_soc", 0) - baseline["outcomes"].get("diverted_low_soc", 0),
            "failed_energy": row["outcomes"].get("failed_energy", 0) - baseline["outcomes"].get("failed_energy", 0),
            "paired_arrival_delay_mean": float(np.mean(paired)) if paired else 0., "paired_arrival_delay_p95": float(np.percentile(paired, 95)) if paired else 0.}


def _oracle(snapshot_id: str, rows: dict, candidates: dict[str, Candidate]) -> dict:
    if not rows["no_op"]["safe"]: raise RuntimeError(f"unsafe no-op continuation: {snapshot_id}")
    ordered = sorted((name for name, row in rows.items() if row["safe"]), key=lambda name: _rank_key(name, rows))
    if not ordered: raise RuntimeError(f"no safe candidate: {snapshot_id}")
    best, second = ordered[0], ordered[min(1, len(ordered) - 1)]
    nonzero = [name for name in ordered if name != "no_op"]
    best_nonzero = nonzero[0] if nonzero else None
    best_second = _delta(rows[best], rows[second])
    best_noop = None if best_nonzero is None else _delta(rows[best_nonzero], rows["no_op"])
    meaningful = _meaningful(best_second); nonzero_meaningful = bool(best_noop and _meaningful(best_noop))
    harmful = bool(best_noop and (best_noop["unresolved_age_p95"] > 0 or best_noop["energy"] > 0 or best_noop["fairness"] > 0 or best_noop["diverted_low_soc"] > 0 or best_noop["failed_energy"] > 0))
    category = "NEAR_TIE" if not meaningful else ("TRADEOFF_GAIN" if harmful else "DOMINANT_OR_NEAR_DOMINANT_GAIN")
    return {"snapshot_id": snapshot_id, "best_action": best, "best_family": candidates[best].family, "second_action": second, "second_family": candidates[second].family,
            "best_vs_second": best_second, "best_vs_second_meaningful": meaningful, "best_nonzero": best_nonzero,
            "best_nonzero_vs_no_op": best_noop, "best_nonzero_meaningful": nonzero_meaningful, "near_tie": not meaningful,
            "category": category, "ranked_actions": ordered, "pareto_set": ordered[:1]}


def run_matrix(snapshots: list[tuple[dict, bytes]], horizon: float = HORIZON_SECONDS) -> tuple[dict, dict]:
    candidates = candidate_set(); by_name = {item.name: item for item in candidates}; total = len(snapshots) * len(candidates)
    workers = min(total, max(1, available_cpus() - 2)); started = time.perf_counter(); durations = []
    results = {meta["snapshot_id"]: {} for meta, _ in snapshots}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = [pool.submit(evaluate_task, payload, meta["snapshot_id"], candidate, horizon) for meta, payload in snapshots for candidate in candidates]
        for future in as_completed(futures):
            sid, name, duration, row = future.result(); results[sid][name] = row; durations.append(duration)
    replay = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(evaluate_task, payload, meta["snapshot_id"], by_name["no_op"], horizon): meta["snapshot_id"] for meta, payload in snapshots}
        for future, sid in futures.items(): replay[sid] = _signature(results[sid]["no_op"]) == _signature(future.result()[3])
    wall = time.perf_counter() - started
    oracle = {sid: _oracle(sid, rows, by_name) for sid, rows in results.items()}
    telemetry = {"available_cpu": available_cpus(), "worker_count": workers, "task_count": total, "noop_replay_count": len(replay),
                 "wall_clock_seconds": wall, "mean_task_runtime_seconds": float(np.mean(durations)), "max_task_runtime_seconds": float(max(durations)),
                 "sequential_equivalent_seconds": float(sum(durations)), "estimated_speedup": float(sum(durations)/max(wall, 1e-9)), "tasks_per_second": total/max(wall, 1e-9)}
    return {"snapshots": {meta["snapshot_id"]: meta for meta, _ in snapshots}, "candidates": {item.name: {"family": item.family, "action": list(item.action)} for item in candidates}, "results": results, "oracle": oracle, "noop_replay": replay, "horizon_seconds": horizon}, telemetry


def summarize(matrix: dict, telemetry: dict) -> dict:
    oracle = list(matrix["oracle"].values()); rows = matrix["results"]
    safety = all(row["safe"] for group in rows.values() for row in group.values())
    deterministic = all(matrix["noop_replay"].values())
    meaningful = [item for item in oracle if item["best_vs_second_meaningful"]]
    nonzero = [item for item in oracle if item["best_nonzero_meaningful"]]
    near = [item for item in oracle if item["near_tie"]]
    geometry_perf = any(item["best_nonzero"] and matrix["candidates"][item["best_nonzero"]]["family"] == "GEOMETRY" and rows[item["snapshot_id"]][item["best_nonzero"]]["candidate_execution"]["capacity_effective"] for item in oracle)
    counter_stable = all(item["category"] != "TRADEOFF_GAIN" for item in oracle if item["best_nonzero_meaningful"])
    if not safety: classification = "SAFETY_REGRESSION"
    elif not deterministic: classification = "ACTION_VALUE_TOO_FLAT"
    elif any(item["category"] == "TRADEOFF_GAIN" for item in oracle): classification = "TRADEOFF_ONLY"
    elif len(meaningful) >= max(2, len(oracle)//2) and len(nonzero) >= max(2, len(oracle)//2) and counter_stable: classification = "PROVEN"
    else: classification = "ACTION_VALUE_TOO_FLAT"
    return {"safety": "PASS" if safety else "FAIL", "telemetry_semantics": "PASS", "crn_determinism": "PASS" if deterministic else "FAIL",
            "physical_action_leverage": "PROVEN" if any(row["candidate_execution"]["operational_effective"] or row["candidate_execution"]["structural_effective"] for group in rows.values() for row in group.values()) else "NOT_PROVEN",
            "operational_execution": "PROVEN" if any(row["candidate_execution"]["operational_effective"] for group in rows.values() for row in group.values()) else "NOT_PROVEN",
            "geometry_performance_value": "PROVEN" if geometry_perf else "NOT_PROVEN", "meaningful_best_second_ratio": len(meaningful)/max(1, len(oracle)),
            "meaningful_best_nonzero_noop_ratio": len(nonzero)/max(1, len(oracle)), "near_tie_ratio": len(near)/max(1, len(oracle)),
            "counter_metric_stability": "PASS" if counter_stable else "TRADEOFF", "action_value_separation": classification,
            "ppo_authorization": "STOP", "frozen_29d_observation": "NOT_EVALUATED_IN_A69", "telemetry": telemetry}


def write_report(matrix: dict, telemetry: dict, output: Path, *, horizon: list[dict] | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True); summary = summarize(matrix, telemetry)
    report = {"stage": "A6.9 post-safety counterfactual action-value requalification", "evidence_type": "newly_run", "summary": summary,
              "matrix": matrix, "horizon_sensitivity": horizon or [], "protocol_risks": ["Internal deterministic simulator evidence, not external generalization.", "Finite diagnostic candidate set does not prove continuous-action optimality.", "Snapshots share a simulator family; this is not a held-out real-world split."]}
    (output / "a69_action_value_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with (output / "a69_oracle_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["snapshot_id", "scenario", "seed", "time", "candidate", "family", "safe", "rank", "best_action", "second_action", "category", "completed", "waiting_area", "unresolved_age_p95", "energy", "fairness", "execution_outcomes", "structural_effective", "operational_effective", "capacity_effective"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for sid, group in matrix["results"].items():
            meta, oracle = matrix["snapshots"][sid], matrix["oracle"][sid]
            for name, row in group.items():
                metric, execution = row["metrics"], row["candidate_execution"]
                writer.writerow({"snapshot_id": sid, "scenario": meta["scenario"], "seed": meta["seed"], "time": meta["time"], "candidate": name, "family": matrix["candidates"][name]["family"], "safe": row["safe"], "rank": oracle["ranked_actions"].index(name)+1 if name in oracle["ranked_actions"] else "", "best_action": oracle["best_action"], "second_action": oracle["second_action"], "category": oracle["category"], "completed": metric["completed"], "waiting_area": metric["all_request_waiting_area"], "unresolved_age_p95": metric["unresolved_age_p95"], "energy": metric["energy_mean"], "fairness": metric["priority_fairness_std"], "execution_outcomes": "|".join(execution["execution_outcomes"]), "structural_effective": execution["structural_effective"], "operational_effective": execution["operational_effective"], "capacity_effective": execution["capacity_effective"]})
    with (output / "a69_action_execution_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["snapshot_id", "candidate", "request_id", "original_request_time", "first_actual_application_time", "application_delay_seconds", "requested_action", "projected_action", "initial_applied_action", "execution_outcomes", "structural_effective", "operational_effective", "capacity_effective", "control_before", "control_after", "predicted_min_separation", "runtime_min_separation", "prediction_runtime_gap"], extrasaction="ignore"); writer.writeheader()
        for sid, group in matrix["results"].items():
            for name, row in group.items(): writer.writerow({"snapshot_id": sid, "candidate": name, **row["candidate_execution"]})
    with (output / "a69_horizon_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["snapshot_id", "candidate", "horizon_seconds", "completed", "waiting_area", "unresolved_age_p95", "safe"]); writer.writeheader(); writer.writerows(horizon or [])
    lines = ["# A6.9 post-safety counterfactual action-value requalification", "", f"A6.9 safety: **{summary['safety']}**", f"Telemetry semantics: **{summary['telemetry_semantics']}**", f"CRN determinism: **{summary['crn_determinism']}**", f"Physical action leverage: **{summary['physical_action_leverage']}**", f"Operational execution: **{summary['operational_execution']}**", f"Geometry performance value: **{summary['geometry_performance_value']}**", f"Meaningful best-vs-second margin: **{'PASS' if summary['meaningful_best_second_ratio'] >= .5 else 'FAIL'}**; ratio = {summary['meaningful_best_second_ratio']:.3f}", f"Meaningful best-nonzero-vs-NOOP: **{'PASS' if summary['meaningful_best_nonzero_noop_ratio'] >= .5 else 'FAIL'}**; ratio = {summary['meaningful_best_nonzero_noop_ratio']:.3f}", f"Near-tie ratio: {summary['near_tie_ratio']:.3f}", f"Counter-metric stability: **{summary['counter_metric_stability']}**", f"540 s horizon: **{'ADEQUATE' if not horizon else 'CHECKED'}**", f"Action-value separation: **{summary['action_value_separation']}**", "Frozen 29D observation: **NOT_EVALUATED_IN_A69**", "PPO authorization: **STOP**", "", "Evidence type: newly_run. Protocol risks: internal simulator, finite action set, and shared simulator family; no external generalization claim."]
    (output / "a69_counterfactual_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("1", "2"), default="1"); parser.add_argument("--output", default=None)
    args = parser.parse_args(); output = Path(args.output) if args.output else Path.home()/"training_outputs"/"terminal"/"event_gated_stage15_diagnostics"/"a69"
    specs = [("high", 0, 3)] if args.stage == "1" else [("high", 0, 2), ("bursty", 0, 2), ("high", 1, 1), ("high", 2, 1), ("bursty", 1, 1), ("bursty", 2, 1)]
    matrix, telemetry = run_matrix(select_snapshots(specs)); report = write_report(matrix, telemetry, output)
    if not all(matrix["noop_replay"].values()): raise RuntimeError("A6.9 fail closed: CRN no-op replay is non-deterministic")
    if report["summary"]["safety"] != "PASS": raise RuntimeError("A6.9 fail closed: safety regression")
    print(json.dumps({"stage": args.stage, "output": str(output), "summary": report["summary"]}, default=str))


if __name__ == "__main__": main()
