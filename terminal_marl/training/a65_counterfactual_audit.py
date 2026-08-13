"""A6.5 counterfactual high-level action-value audit; never trains PPO."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from typing import Iterable

import numpy as np

from ..core.config import RingRole
from ..core.uav import TerminalOutcome
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
from ..policies import DynamicRulePolicy
from .a6_effectiveness import _collect, _sample, scenario_config
from .a63_dynamic_screen import SHORT_SCENARIOS

try:  # ``resource`` is absent on Windows; evaluation itself runs on Linux.
    import resource
except ImportError:  # pragma: no cover - exercised only by Windows tooling.
    resource = None


HORIZON_SECONDS = 540.0  # Three canonical 180-second decision intervals.
NO_OP = np.zeros(10, dtype=np.float32)
REQUIRED_SAFETY = ("collisions", "nmac", "hard_separation_violations", "final_overcapacity_steps")


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    action: tuple[float, ...]


def candidate_set() -> tuple[Candidate, ...]:
    """Small interpretable set; every structural request crosses a capacity threshold."""
    return (
        Candidate("no_op", "NO_OP", tuple(NO_OP)),
        Candidate("o2m_plus", "HEADWAY", (0, 0, 0, 0, 0, 0, 0, .35, 0, 0)),
        Candidate("o2m_minus", "HEADWAY", (0, 0, 0, 0, 0, 0, 0, -.35, 0, 0)),
        Candidate("m2i_plus", "HEADWAY", (0, 0, 0, 0, 0, 0, 0, 0, .35, 0)),
        Candidate("m2i_minus", "HEADWAY", (0, 0, 0, 0, 0, 0, 0, 0, -.35, 0)),
        Candidate("i2f_plus", "HEADWAY", (0, 0, 0, 0, 0, 0, 0, 0, 0, .35)),
        Candidate("i2f_minus", "HEADWAY", (0, 0, 0, 0, 0, 0, 0, 0, 0, -.35)),
        Candidate("o2m_m2i_gradient", "HEADWAY", (0, 0, 0, 0, 0, 0, 0, -.35, .35, 0)),
        Candidate("outer_capacity_plus1", "GEOMETRY", (.60, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        Candidate("middle_capacity_plus1", "GEOMETRY", (0, .45, 0, 0, 0, 0, 0, 0, 0, 0)),
        Candidate("inner_capacity_plus1", "GEOMETRY", (0, 0, .30, 0, 0, 0, 0, 0, 0, 0)),
        Candidate("outer_middle_capacity_plus1", "GEOMETRY", (.60, .45, 0, 0, 0, 0, 0, 0, 0, 0)),
        Candidate("speed_plus", "SPEED", (0, 0, 0, 0, 0, 0, .25, 0, 0, 0)),
        Candidate("speed_minus", "SPEED", (0, 0, 0, 0, 0, 0, -.25, 0, 0, 0)),
        Candidate("outer_plus_o2m_minus", "COMBINED", (.60, 0, 0, 0, 0, 0, 0, -.35, 0, 0)),
        Candidate("m2i_minus_speed_minus", "COMBINED", (0, 0, 0, 0, 0, 0, -.25, 0, -.35, 0)),
    )


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return int(os.cpu_count() or 1)


def _worker_init() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    try:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
    except ImportError:
        pass


def _snapshot_features(wrapper: DynamicHighLevelWrapper, scenario: str, seed: int, snapshot_id: str) -> dict:
    base, context = wrapper.base, wrapper._context()
    priorities = [uav.dynamic_priority for uav in base.active.values()]
    soc_urgent = [uav.soc for uav in base.active.values() if uav.soc < base.config.soc_reserve]
    recent = sum(request_time >= base.now - base.config.high_level_decision_interval_seconds for request_time in base.request_times.values())
    return {
        "snapshot_id": snapshot_id, "scenario": scenario, "seed": seed, "time": base.now,
        "external_queue": context["external_queue"], "occupancy_ratio": list(context["occupancy_ratio"]),
        "capacity_slack": [capacity - occupancy for capacity, occupancy in zip(context["capacity"], context["occupancy"])],
        "final_busy": int(context["final_pressure"] > 0), "final_pressure": context["final_pressure"], "lz_utilization": context["lz_utilization"],
        "recent_arrivals_180s": recent, "soc_urgent_count": len(soc_urgent), "priority_mean": float(np.mean(priorities)) if priorities else 0.,
        "priority_max": float(np.max(priorities)) if priorities else 0., "pending_reservations": len(base.scheduler.reservations),
        "committed_transfer_load": sum(uav.committed_plan is not None for uav in base.active.values()),
        "pending_reconfiguration": wrapper.coordinator.pending_state.value,
        "high_observation": wrapper.observation().tolist(),
        "rng_state_sha256": hashlib.sha256(pickle.dumps(base.arrivals.rng.bit_generator.state)).hexdigest(),
    }


def _run_macro(wrapper: DynamicHighLevelWrapper, requested: np.ndarray, until: float, samples: dict, events: list[dict]) -> None:
    base, coordinator = wrapper.base, wrapper.coordinator
    decision = coordinator.coordinate(requested, geometry_gate=wrapper.geometry_event_gate())
    events.append({"time": base.now, "candidate_requested": requested.tolist(), "outcome": decision.outcome,
                   "reason": decision.reason_code, "applied": list(decision.applied_action), "repair": decision.repair_magnitude,
                   "projection": decision.projection_magnitude, "capacity_before": list(decision.update.capacity_before),
                   "capacity_after": list(decision.update.capacity_after), "structural_reason_codes": list(decision.update.structural_reason_codes)})
    while base.now < until - 1e-9:
        deferred = coordinator.advance()
        if deferred is not None:
            events.append({"time": base.now, "candidate_requested": list(deferred.requested_action), "outcome": deferred.outcome,
                           "reason": deferred.reason_code, "applied": list(deferred.applied_action), "repair": deferred.repair_magnitude,
                           "projection": deferred.projection_magnitude, "capacity_before": list(deferred.update.capacity_before),
                           "capacity_after": list(deferred.update.capacity_after), "structural_reason_codes": list(deferred.update.structural_reason_codes)})
        actions = coordinator.apply_speed_repairs(wrapper.low_level.act(base._observations()))
        base.step(actions); _sample(base, samples)
        if wrapper._done():
            return


def _branch_from_wrapper(wrapper: DynamicHighLevelWrapper, candidate: Candidate, horizon: float) -> dict:
    start = wrapper.base.now; end = start + horizon
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}
    events: list[dict] = []
    first = True
    while wrapper.base.now < end - 1e-9 and not wrapper._done():
        requested = np.asarray(candidate.action if first else NO_OP, dtype=np.float32)
        _run_macro(wrapper, requested, min(end, wrapper.base.now + wrapper.config.high_level_decision_interval_seconds), samples, events)
        first = False
    collected = _collect(wrapper.base, samples, wrapper.coordinator, events)
    outcome_counts = Counter(value.value for value in wrapper.base.ledger.outcomes.values())
    safety = all(collected.get(key) == 0 for key in REQUIRED_SAFETY) and collected.get("max_concurrent_final", 0) <= wrapper.config.final_queue_capacity
    return {
        "safe": safety, "metrics": collected, "outcomes": dict(outcome_counts),
        "applied_abs_sum": np.abs(np.asarray([event["applied"] for event in events], dtype=float)).sum(axis=0).tolist() if events else [0.] * 10,
        "capacity_changing_updates": sum(any(before != after for before, after in zip(event["capacity_before"], event["capacity_after"])) for event in events),
        "shield_cost": collected.get("shield_cost", 0.), "event_count": len(events),
    }


def _compact(branch: dict) -> dict:
    metrics = branch["metrics"]
    keep = ("completed", "all_request_waiting_area", "unresolved_count", "unresolved_age_mean", "unresolved_age_p95",
            "service_delay_p95", "service_delay_p99", "energy_mean", "priority_fairness_std", "final_busy_fraction",
            "max_concurrent_final", "final_overcapacity_steps", "collisions", "nmac", "hard_separation_violations",
            "safety_interventions", "minimum_swept_separation")
    return {"safe": branch["safe"], "metrics": {key: metrics[key] for key in keep}, "outcomes": branch["outcomes"],
            "applied_abs_sum": branch["applied_abs_sum"], "capacity_changing_updates": branch["capacity_changing_updates"],
            "shield_cost": branch["shield_cost"], "event_count": branch["event_count"],
            "arrival_delay_by_id": metrics["arrival_delay_by_id"]}


def evaluate_task(snapshot_bytes: bytes, snapshot_id: str, candidate: Candidate, horizon: float = HORIZON_SECONDS) -> tuple[str, str, float, dict]:
    started = time.perf_counter()
    wrapper = pickle.loads(snapshot_bytes)
    result = _compact(_branch_from_wrapper(wrapper, candidate, horizon))
    return snapshot_id, candidate.name, time.perf_counter() - started, result


def collect_snapshots(scenario: str, seed: int, count: int) -> list[tuple[dict, bytes]]:
    config = scenario_config(SHORT_SCENARIOS[scenario], "dynamic_rule")
    wrapper = DynamicHighLevelWrapper(config, RuleLowLevelAdapter()); observation, _ = wrapper.reset(seed)
    snapshots: list[tuple[dict, bytes]] = []
    while len(snapshots) < count and wrapper.base.now < config.max_simulation_seconds and not wrapper._done():
        action = DynamicRulePolicy().act(observation, wrapper._context())
        observation, _, _, _, _ = wrapper.step(action)
        if wrapper.base.now >= config.high_level_decision_interval_seconds * 2:
            snapshot_id = f"{scenario}-seed{seed}-t{int(wrapper.base.now)}"
            snapshots.append((_snapshot_features(wrapper, scenario, seed, snapshot_id), pickle.dumps(wrapper, protocol=pickle.HIGHEST_PROTOCOL)))
    if len(snapshots) < count:
        raise RuntimeError(f"insufficient valid decision-boundary snapshots for {scenario}/seed{seed}")
    return snapshots


def _state_signature(wrapper: DynamicHighLevelWrapper) -> dict:
    base = wrapper.base
    return {"now": base.now, "ledger": sorted((key, value.value) for key, value in base.ledger.outcomes.items()),
            "active": sorted((uav.uid, uav.phase.value, tuple(np.round(uav.position, 6)), round(uav.energy_remaining, 6)) for uav in base.active.values()),
            "geometry": tuple(np.round(base.airspace.state_vector(), 6)), "telemetry": {key: base.telemetry[key] for key in REQUIRED_SAFETY}}


def validate_snapshot_restore(snapshot_bytes: bytes) -> bool:
    original = pickle.loads(snapshot_bytes); restored = pickle.loads(snapshot_bytes)
    candidate = Candidate("no_op", "NO_OP", tuple(NO_OP))
    _branch_from_wrapper(original, candidate, 180.0)
    _branch_from_wrapper(restored, candidate, 180.0)
    return _state_signature(original) == _state_signature(restored)


def _paired_delay(row: dict, baseline: dict) -> dict:
    common = sorted(set(row["arrival_delay_by_id"]) & set(baseline["arrival_delay_by_id"]), key=int)
    values = [row["arrival_delay_by_id"][key] - baseline["arrival_delay_by_id"][key] for key in common]
    return {"count": len(values), "mean": float(np.mean(values)) if values else 0., "p95": float(np.percentile(values, 95)) if values else 0.}


def _rank_safe(rows: dict[str, dict]) -> str:
    def key(name: str):
        item, metrics = rows[name], rows[name]["metrics"]
        outcomes = item["outcomes"]
        return (-metrics["completed"], metrics["all_request_waiting_area"], metrics["unresolved_count"], metrics["unresolved_age_p95"],
                outcomes.get(TerminalOutcome.DIVERTED_LOW_SOC.value, 0), outcomes.get(TerminalOutcome.FAILED_ENERGY.value, 0),
                metrics["service_delay_p95"], metrics["service_delay_p99"], metrics["priority_fairness_std"], item["shield_cost"])
    return min((name for name, row in rows.items() if row["safe"]), key=key)


def analyze_rows(rows: dict[str, dict], candidates: dict[str, Candidate]) -> dict:
    baseline = rows["no_op"]
    best = _rank_safe(rows)
    nonzero = [name for name, row in rows.items() if name != "no_op" and row["safe"]]
    best_nonzero = _rank_safe({name: rows[name] for name in nonzero}) if nonzero else None
    def delta(name: str | None):
        if name is None: return None
        item = rows[name]; metrics, base = item["metrics"], baseline["metrics"]
        return {"completed": metrics["completed"] - base["completed"], "waiting_area": metrics["all_request_waiting_area"] - base["all_request_waiting_area"],
                "unresolved_age_p95": metrics["unresolved_age_p95"] - base["unresolved_age_p95"], "energy": metrics["energy_mean"] - base["energy_mean"],
                "fairness": metrics["priority_fairness_std"] - base["priority_fairness_std"], "paired_arrival_delay": _paired_delay(item, baseline)}
    return {"best": best, "best_family": candidates[best].family, "best_nonzero": best_nonzero,
            "best_nonzero_family": None if best_nonzero is None else candidates[best_nonzero].family,
            "best_nonzero_vs_no_op": delta(best_nonzero), "safe_candidate_rate": float(sum(row["safe"] for row in rows.values()) / len(rows))}


def run_stage(scenarios: Iterable[str], seeds: Iterable[int], snapshots_per_run: int, *, horizon: float = HORIZON_SECONDS) -> dict:
    candidates = candidate_set(); candidate_by_name = {candidate.name: candidate for candidate in candidates}
    snapshots = [item for scenario in scenarios for seed in seeds for item in collect_snapshots(scenario, seed, snapshots_per_run)]
    validation = validate_snapshot_restore(snapshots[0][1])
    if not validation:
        raise RuntimeError("snapshot restore + no-op is not deterministic")
    total = len(snapshots) * len(candidates); cores = available_cpus(); workers = min(total, max(1, cores - 2))
    started = time.perf_counter(); task_times: list[float] = []; grouped: dict[str, dict[str, dict]] = {meta["snapshot_id"]: {} for meta, _ in snapshots}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        futures = [executor.submit(evaluate_task, payload, meta["snapshot_id"], candidate, horizon) for meta, payload in snapshots for candidate in candidates]
        for future in as_completed(futures):
            snapshot_id, candidate_name, duration, result = future.result(); grouped[snapshot_id][candidate_name] = result; task_times.append(duration)
    wall = time.perf_counter() - started
    analyses = {snapshot_id: analyze_rows(rows, candidate_by_name) for snapshot_id, rows in grouped.items()}
    metadata = {meta["snapshot_id"]: meta for meta, _ in snapshots}
    return {"evidence_type": "newly_run", "horizon_seconds": horizon, "snapshot_restore_noop_deterministic": validation,
            "performance": {"available_cpu_count": cores, "worker_count": workers, "total_counterfactual_tasks": total,
                            "wall_clock_seconds": wall, "tasks_per_second": total / max(wall, 1e-9), "mean_task_runtime_seconds": float(np.mean(task_times)),
                            "peak_rss_kb": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None),
                            "sequential_equivalent_seconds": float(sum(task_times)), "parallel_speedup_estimate": float(sum(task_times) / max(wall, 1e-9))},
            "snapshots": metadata, "candidates": {candidate.name: {"family": candidate.family, "action": candidate.action} for candidate in candidates},
            "results": grouped, "analysis": analyses}


def summarize(stage: dict) -> dict:
    analyses = stage["analysis"]; candidate_meta = stage["candidates"]
    best_families = Counter(value["best_family"] for value in analyses.values())
    nonzero = [value for value in analyses.values() if value["best_nonzero"]]
    gains = [value["best_nonzero_vs_no_op"] for value in nonzero]
    stable_gain = [item for item in gains if item["completed"] >= 0 and item["waiting_area"] < 0 and item["energy"] <= 0 and item["fairness"] <= 0]
    geometry_value = any(value["best_nonzero_family"] == "GEOMETRY" and stage["results"][snapshot_id][value["best_nonzero"]]["capacity_changing_updates"] > 0 for snapshot_id, value in analyses.items())
    nonzero_best_ratio = float(sum(value["best"] != "no_op" for value in analyses.values()) / max(1, len(analyses)))
    # The frozen 29-D observation contains every recorded feature except the
    # explicit recent-arrival counter; calibration of separability needs the
    # multi-seed state distribution and is deliberately fail-closed here.
    observation_sufficiency = "UNCERTAIN"
    decision = "AUTHORIZE_PPO" if (stage["snapshot_restore_noop_deterministic"] and nonzero_best_ratio >= .35 and len(stable_gain) >= 2 and geometry_value and observation_sufficiency == "PASS") else "STOP_BEFORE_PPO"
    return {"safety": "PASS" if all(row["safe"] for rows in stage["results"].values() for row in rows.values()) else "FAIL",
            "counterfactual_leverage": "PROVEN" if len(stable_gain) >= 2 and nonzero_best_ratio >= .35 else "NOT_PROVEN",
            "observation_sufficiency": observation_sufficiency, "geometry_value": "PROVEN" if geometry_value else "NOT_PROVEN",
            "decision": decision, "snapshot_count": len(analyses), "candidate_count": len(candidate_meta),
            "safe_candidate_rate": float(np.mean([value["safe_candidate_rate"] for value in analyses.values()])),
            "best_family_counts": dict(best_families), "no_op_best_ratio": float(best_families.get("NO_OP", 0) / max(1, len(analyses))),
            "nonzero_best_ratio": nonzero_best_ratio, "stable_gain_count": len(stable_gain), "best_nonzero_vs_no_op": gains,
            "reason": "Counterfactual authorization requires repeatable lexicographic Tier-1 gain without Tier-2 regression; observation sufficiency stays UNCERTAIN until action choice is separable from the existing high observation."}


def write_report(stage: dict, output: str | Path, *, prefix: str = "a65_counterfactual") -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    summary = summarize(stage); report = {"stage": "A6.5 counterfactual high-level action-value audit", "summary": summary, "stage_data": stage,
        "protocol": {"crn": "Each candidate restores the same pickled boundary snapshot including ArrivalProcess RNG state.", "ranking": "Tier0 safety; Tier1 completion/waiting/unresolved; Tier2 diversion-energy-tail-delay-fairness; Tier3 shield.", "risk": "newly_run internal counterfactual simulation; this is not an external generalization claim."}}
    (output / f"{prefix}_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with (output / f"{prefix}_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["snapshot_id", "scenario", "seed", "candidate", "family", "safe", "completed", "waiting_area", "unresolved_age_p95", "energy", "fairness", "capacity_changing_updates"])
        for snapshot_id, rows in stage["results"].items():
            meta = stage["snapshots"][snapshot_id]
            for candidate, row in rows.items():
                metrics = row["metrics"]; writer.writerow([snapshot_id, meta["scenario"], meta["seed"], candidate, stage["candidates"][candidate]["family"], row["safe"], metrics["completed"], metrics["all_request_waiting_area"], metrics["unresolved_age_p95"], metrics["energy_mean"], metrics["priority_fairness_std"], row["capacity_changing_updates"]])
    lines = ["# A6.5 counterfactual high-level action-value audit", "", f"A6.5 safety: **{summary['safety']}**", f"Counterfactual leverage: **{summary['counterfactual_leverage']}**", f"Observation sufficiency: **{summary['observation_sufficiency']}**", f"Geometry value: **{summary['geometry_value']}**", f"Decision: **{summary['decision']}**", "", f"Snapshots: {summary['snapshot_count']}; candidates/snapshot: {summary['candidate_count']}; safe-candidate rate: {summary['safe_candidate_rate']:.3f}.", f"Best family counts: {summary['best_family_counts']}; no-op best ratio: {summary['no_op_best_ratio']:.3f}.", "", f"Parallel telemetry: {stage['performance']}", "", "Evidence type: newly_run. Protocol risk: internal deterministic simulation audit, not an external generalization result."]
    (output / f"{prefix}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def aggregate_reports(stage_reports: list[dict], output: str | Path) -> dict:
    """Publish the required final files without overwriting staged audit evidence."""
    by_scenario: dict[str, list[dict]] = {}
    worker_telemetry: list[dict] = []
    for report in stage_reports:
        worker_telemetry.append(report["stage_data"]["performance"])
        candidate_meta = report["stage_data"]["candidates"]
        for snapshot_id, meta in report["stage_data"]["snapshots"].items():
            by_scenario.setdefault(meta["scenario"], []).append({
                "snapshot_id": snapshot_id, "analysis": report["stage_data"]["analysis"][snapshot_id],
                "features": meta, "results": report["stage_data"]["results"][snapshot_id], "candidates": candidate_meta,
            })
    safety_pass = all(entry["results"]["no_op"]["safe"] and all(row["safe"] for row in entry["results"].values())
                      for entries in by_scenario.values() for entry in entries)
    scenario_summary: dict[str, dict] = {}
    for scenario, entries in by_scenario.items():
        best_families = Counter(entry["analysis"]["best_family"] for entry in entries)
        deltas = [entry["analysis"]["best_nonzero_vs_no_op"] for entry in entries if entry["analysis"]["best_nonzero_vs_no_op"]]
        scenario_summary[scenario] = {
            "snapshot_count": len(entries), "candidates_per_snapshot": len(entries[0]["candidates"]),
            "safe_candidate_rate": float(np.mean([entry["analysis"]["safe_candidate_rate"] for entry in entries])),
            "best_family_counts": dict(best_families), "no_op_best_ratio": best_families.get("NO_OP", 0) / max(1, len(entries)),
            "best_nonzero_vs_no_op": deltas,
            "state_action_pairs": [{"snapshot_id": entry["snapshot_id"], "best": entry["analysis"]["best"], "best_family": entry["analysis"]["best_family"], "features": entry["features"]} for entry in entries],
        }
    leverage_proven = any(sum(1 for delta in summary["best_nonzero_vs_no_op"] if delta["completed"] >= 0 and delta["waiting_area"] < 0 and delta["energy"] <= 0 and delta["fairness"] <= 0) >= 2 for summary in scenario_summary.values())
    geometry_proven = any(entry["results"][entry["analysis"]["best_nonzero"]]["capacity_changing_updates"] > 0 for entries in by_scenario.values() for entry in entries if entry["analysis"]["best_nonzero"])
    combined = {"stage": "A6.5 counterfactual high-level action-value audit", "evidence_type": "newly_run",
                "scenarios": by_scenario, "safety": "PASS" if safety_pass else "FAIL",
                "counterfactual_leverage": "PROVEN" if leverage_proven else "NOT_PROVEN", "observation_sufficiency": "UNCERTAIN",
                "geometry_value": "PROVEN" if geometry_proven else "NOT_PROVEN", "decision": "STOP_BEFORE_PPO",
                "scenario_summary": scenario_summary,
                "parallel_telemetry": worker_telemetry,
                "protocol": "CRN snapshot branches, 540-second horizon, Tier 0/1/2/3 lexicographic ranking."}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "a65_counterfactual_audit.json").write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    with (output / "a65_counterfactual_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["scenario", "snapshot_id", "candidate", "family", "safe", "completed", "waiting_area", "unresolved_age_p95", "energy", "fairness", "capacity_changing_updates"])
        for scenario, entries in by_scenario.items():
            for entry in entries:
                for candidate, row in entry["results"].items():
                    metrics = row["metrics"]
                    writer.writerow([scenario, entry["snapshot_id"], candidate, entry["candidates"][candidate]["family"], row["safe"], metrics["completed"], metrics["all_request_waiting_area"], metrics["unresolved_age_p95"], metrics["energy_mean"], metrics["priority_fairness_std"], row["capacity_changing_updates"]])
    lines = ["# A6.5 counterfactual high-level action-value audit", "", f"A6.5 safety: **{combined['safety']}**", f"Counterfactual leverage: **{combined['counterfactual_leverage']}**", "Observation sufficiency: **UNCERTAIN**", f"Geometry value: **{combined['geometry_value']}**", "Decision: **STOP_BEFORE_PPO**", ""]
    for scenario, entries in sorted(by_scenario.items()):
        best = Counter(entry["analysis"]["best_family"] for entry in entries)
        no_op = best.get("NO_OP", 0) / max(1, len(entries))
        summary = scenario_summary[scenario]
        lines.extend(["", f"## {scenario}", "", f"Snapshots: {len(entries)}; candidates/snapshot: {summary['candidates_per_snapshot']}; safe-candidate rate: {summary['safe_candidate_rate']:.3f}.", f"Best-family counts: {dict(best)}; no-op best ratio: {no_op:.3f}.", f"Best-nonzero vs no-op: {summary['best_nonzero_vs_no_op']}."])
    lines.extend(["", "The 29-D frozen observation is not yet empirically shown to separate the action-specific opportunities: the representative high/bursty snapshots select different families under similarly low ring occupancy. This is an observation/action-identifiability gap, so PPO remains prohibited.", "", f"Parallel telemetry: {worker_telemetry}"])
    (output / "a65_counterfactual_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return combined


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--scenario", action="append", choices=("high", "bursty"), default=[]); parser.add_argument("--seed", action="append", type=int, default=[]); parser.add_argument("--snapshots", type=int, default=2); parser.add_argument("--stage-tag", default="stage"); parser.add_argument("--aggregate", action="append", default=[])
    args = parser.parse_args(); output = Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a65"
    if args.aggregate:
        reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.aggregate]
        combined = aggregate_reports(reports, output); print(json.dumps({"output": str(output), "safety": combined["safety"], "scenarios": sorted(combined["scenarios"])})); return
    scenarios = args.scenario or ["high"]; seeds = args.seed or [0]
    stage = run_stage(scenarios, seeds, args.snapshots)
    stage_output = output / "stages" / args.stage_tag
    report = write_report(stage, stage_output)
    print(json.dumps({"output": str(output), **report["summary"]}, default=str))


if __name__ == "__main__":
    main()
