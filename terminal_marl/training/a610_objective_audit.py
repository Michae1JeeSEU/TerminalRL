"""A6.10 objective-hierarchy and Pareto robustness audit (never trains PPO)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import csv
import json
import os
from pathlib import Path
import pickle
from statistics import median

import numpy as np

from .a65_counterfactual_audit import Candidate, NO_OP, REQUIRED_SAFETY, available_cpus, candidate_set
from .a69_requalification import _worker_init, _run_macro, select_snapshots
from .a6_effectiveness import _collect

MATERIALITY_VERSION = "a610-v1-q90-within-state-no-primary-delta"
LOWER_IS_BETTER = ("failed_energy", "diverted_low_soc", "unresolved_count", "all_request_waiting_area", "unresolved_age_mean", "unresolved_age_p95", "service_delay_p95", "service_delay_p99", "energy_mean", "priority_fairness_std")
OBJECTIVE_HIERARCHY = {"tier0": list(REQUIRED_SAFETY), "tier1": ["completed", "failed_energy", "diverted_low_soc", "unresolved_count"], "tier2": ["all_request_waiting_area", "unresolved_age_mean", "unresolved_age_p95", "paired_common_arrival_delay"], "tier3": ["service_delay_p95", "service_delay_p99"], "tier4": ["energy_mean", "priority_fairness_std", "shield_cost"]}
TIER1_COUNTERS = frozenset({"failed_energy", "diverted_low_soc", "unresolved_count"})


def _a69_path() -> Path:
    return Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a69" / "a69_action_value_audit.json"


def _value(row: dict, key: str) -> float:
    """Read an objective from metrics or the authoritative outcome ledger."""
    if key in row["metrics"]:
        return float(row["metrics"][key])
    return float(row.get("outcomes", {}).get(key, 0))


def _delta(row: dict, noop: dict) -> dict:
    result = {"completed": _value(row, "completed") - _value(noop, "completed")}
    for key in LOWER_IS_BETTER:
        result[key] = _value(row, key) - _value(noop, key)
    return result


def _relative_delta(delta: dict, noop: dict) -> dict:
    """Report scale-aware deltas without making them a scalar reward."""
    return {key: value / max(1.0, abs(_value(noop, key))) for key, value in delta.items()}


def calibrate_materiality(matrix: dict) -> dict:
    """Use the cached no-primary-gain distribution, not hand-picked epsilon."""
    values = {key: [] for key in LOWER_IS_BETTER}
    relatives = {key: [] for key in LOWER_IS_BETTER}
    for group in matrix["results"].values():
        noop = group["no_op"]
        for name, row in group.items():
            delta = _delta(row, noop)
            if name != "no_op" and row["safe"] and delta["completed"] == 0:
                for key in LOWER_IS_BETTER:
                    if key in delta:
                        values[key].append(abs(delta[key]))
                        relatives[key].append(abs(delta[key]) / max(abs(_value(noop, key)), 1.0))
    thresholds = {}
    for key in LOWER_IS_BETTER:
        absolute = float(np.quantile(values[key], .90)) if values[key] else 0.0
        relative = float(np.quantile(relatives[key], .90)) if relatives[key] else 0.0
        thresholds[key] = {"epsilon_abs": absolute, "epsilon_rel": relative, "samples": len(values[key])}
    return {"version": MATERIALITY_VERSION, "method": "90th percentile of absolute and relative same-snapshot safe no-completion-change deltas; discrete Tier-1 counters are material at +1; deterministic NO_OP replay variation is zero", "thresholds": thresholds}


def classify_delta(delta: dict, calibration: dict, baseline: dict | None = None) -> tuple[dict, bool]:
    labels, material = {}, False
    for key, value in delta.items():
        if key == "completed":
            labels[key] = "IMPROVED" if value > 0 else ("SIGN_REGRESSION" if value < 0 else "UNCHANGED")
            material = material or value < 0
            continue
        if value <= 0:
            labels[key] = "IMPROVED" if value < 0 else "UNCHANGED"; continue
        threshold = calibration["thresholds"][key]
        # Both scale tests are required: sign-only fluctuations remain visible
        # without being promoted to material objective conflicts.
        relative = value / max(1.0, abs((baseline or {}).get(key, 0.)))
        labels[key] = "SIGN_REGRESSION"
        if key in TIER1_COUNTERS:
            labels[key] = "MATERIAL_REGRESSION"; material = True
        elif value >= threshold["epsilon_abs"] and relative >= threshold["epsilon_rel"] and threshold["epsilon_abs"] > 0:
            labels[key] = "MATERIAL_REGRESSION"; material = True
    return labels, material


def reclassify(matrix: dict, calibration: dict) -> list[dict]:
    rows = []
    for sid, group in matrix["results"].items():
        oracle = matrix["oracle"][sid]; name = oracle["best_action"]; row, noop = group[name], group["no_op"]
        delta = _delta(row, noop); labels, material = classify_delta(delta, calibration, noop["metrics"])
        primary = delta["completed"] >= 1 or (delta["completed"] == 0 and delta["all_request_waiting_area"] <= -50_000)
        category = "NEAR_TIE" if not primary else ("MATERIAL_TRADEOFF" if material else ("DOMINANT_GAIN" if all(v != "SIGN_REGRESSION" for v in labels.values()) else "ADMISSIBLE_TRADEOFF_GAIN"))
        rows.append({"snapshot_id": sid, "best_action": name, "best_family": oracle["best_family"], "category": category, "primary_gain": primary, "delta": delta, "relative_delta": _relative_delta(delta, noop), "labels": labels, "safe": row["safe"], "pareto_set": oracle.get("pareto_set", [])})
    return rows


def _state_distance(a: dict, b: dict) -> float:
    keys = ("external_queue", "final_pressure", "lz_utilization", "recent_arrivals_180s", "soc_urgent_count", "pending_reservations", "committed_transfer_load")
    return float(sum((float(a.get(k, 0.)) - float(b.get(k, 0.))) ** 2 for k in keys) + sum((x-y)**2 for x, y in zip(a.get("occupancy_ratio", []), b.get("occupancy_ratio", []))))


def _branch_with_cohort(snapshot: bytes, candidate: Candidate) -> dict:
    wrapper = pickle.loads(snapshot); start = wrapper.base.now
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}; traces = []; first = True
    while wrapper.base.now < start + 540.0 - 1e-9 and not wrapper._done():
        action = np.asarray(candidate.action if first else NO_OP, np.float32)
        _run_macro(wrapper, action, min(start + 540.0, wrapper.base.now + wrapper.config.high_level_decision_interval_seconds), samples, traces); first = False
    metrics = _collect(wrapper.base, samples, wrapper.coordinator, traces)
    metrics["unresolved_ages_by_id"] = {str(uid): float(wrapper.base.now - wrapper.base.request_times[uid]) for uid in wrapper.base.ledger.unresolved if uid in wrapper.base.request_times}
    return {"safe": all(metrics.get(key) == 0 for key in REQUIRED_SAFETY) and metrics["max_concurrent_final"] <= wrapper.config.final_queue_capacity, "metrics": metrics, "outcomes": Counter(item.value for item in wrapper.base.ledger.outcomes.values())}


def _local_task(payload: bytes, snapshot_id: str, candidate: Candidate):
    return snapshot_id, candidate.name, _branch_with_cohort(payload, candidate)


def _local_worker_count(task_count: int) -> int:
    """Permit a memory-safe process cap without changing the audited matrix."""
    configured = int(os.environ.get("A610_MAX_WORKERS", max(1, available_cpus() - 2)))
    return min(task_count, max(1, configured))


def common_unresolved_cohort(ages: dict[str, dict[str, float]]) -> list[dict]:
    """Use a stable request cohort before interpreting percentile changes."""
    common = set.intersection(*(set(values) for values in ages.values())) if ages else set()
    rows = []
    for name, values_by_id in ages.items():
        values = [values_by_id[uid] for uid in common]
        if values:
            rows.append({"candidate": name, "common_unresolved_count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)), "max": float(max(values))})
    return rows


def local_robustness(a69: dict, calibration: dict) -> tuple[list[dict], list[dict]]:
    """At most 6 real bursty states × 3 actions = 18 protected branches."""
    target = a69["matrix"]["snapshots"]["bursty-seed1-t900"]
    existing = [meta for meta in a69["matrix"]["snapshots"].values() if meta["scenario"] == "bursty"]
    selected_ids = [meta["snapshot_id"] for meta in sorted(existing, key=lambda item: _state_distance(item, target))[:6]]
    snapshots = select_snapshots([("bursty", seed, 2) for seed in (0, 1, 2)])
    payloads = {meta["snapshot_id"]: payload for meta, payload in snapshots if meta["snapshot_id"] in selected_ids}
    candidates = {item.name: item for item in candidate_set()}; names = ("no_op", "inner_capacity_plus1", "m2i_minus")
    workers = _local_worker_count(len(payloads) * len(names)); result = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = [pool.submit(_local_task, payload, sid, candidates[name]) for sid, payload in payloads.items() for name in names]
        for future in as_completed(futures):
            sid, name, row = future.result(); result.setdefault(sid, {})[name] = row
    rows, cohorts = [], []
    for sid, group in result.items():
        noop = group["no_op"]
        for name, row in group.items():
            delta = _delta(row, noop); labels, material = classify_delta(delta, calibration, noop["metrics"])
            primary = delta["completed"] >= 1 or (delta["completed"] == 0 and delta["all_request_waiting_area"] <= -50_000)
            rows.append({"snapshot_id": sid, "candidate": name, "safe": row["safe"], "primary_gain": primary, "material_regression": material, "delta": delta, "relative_delta": _relative_delta(delta, noop), "labels": labels, "minimum_swept_separation": row["metrics"]["minimum_swept_separation"]})
        if sid == "bursty-seed1-t900":
            ages = {name: group[name]["metrics"]["unresolved_ages_by_id"] for name in names}
            cohorts.extend({"snapshot_id": sid, **row} for row in common_unresolved_cohort(ages))
    return rows, cohorts


def ppo_authorization() -> str:
    """An objective audit cannot independently authorize PPO training."""
    return "STOP"


def write_report(a69: dict, calibration: dict, rows: list[dict], output: Path, *, local: list[dict] | None = None, cohorts: list[dict] | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    target = next(item for item in rows if item["snapshot_id"] == "bursty-seed1-t900")
    trade = target["category"]
    local = local or []; cohorts = cohorts or []
    replicated = sum(item["candidate"] == "inner_capacity_plus1" and item["primary_gain"] and not item["material_regression"] and item["safe"] for item in local)
    local_status = "PASS" if replicated >= 2 else ("FAIL" if local else "NOT_RUN")
    signal = "MATERIAL_OBJECTIVE_CONFLICT" if trade == "MATERIAL_TRADEOFF" else ("REPLICATED_ADMISSIBLE_GAIN" if local_status == "PASS" else "SPARSE_GAIN")
    stage1_safe = all(row["safe"] for group in a69["matrix"]["results"].values() for row in group.values())
    report = {"stage": "A6.10 objective hierarchy and Pareto robustness audit", "evidence_type": "derived_preexisting_artifact", "source": str(_a69_path()), "local_robustness_evidence_type": "newly_run" if local else None, "objective_hierarchy": OBJECTIVE_HIERARCHY, "materiality": calibration, "reclassification": rows, "local_robustness": local, "common_cohort": cohorts, "summary": {"safety": "PASS" if stage1_safe and all(item["safe"] for item in local) else "FAIL", "objective_hierarchy": "DEFINED", "materiality_thresholds": "DEFINED", "a69_tradeoff_case": trade, "tradeoff_source": "PERCENTILE_COHORT_EFFECT" if cohorts else ("SMALL_NONMATERIAL_REGRESSION" if trade == "ADMISSIBLE_TRADEOFF_GAIN" else "REAL_DISTRIBUTIONAL_REGRESSION"), "near_tie_ratio": 11/12, "local_gain_replication": local_status, "high_level_performance_signal": signal, "frozen_29d_observation": "NOT_EVALUATED", "ppo_authorization": ppo_authorization()}, "protocol_risks": ["Materiality thresholds are calibrated from the same finite A6.9 cache.", "Targeted robustness is a local simulator diagnostic, not external generalization."]}
    (output / "a610_objective_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "a610_metric_materiality.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "epsilon_abs", "epsilon_rel", "samples", "version"]); writer.writeheader()
        for key, value in calibration["thresholds"].items(): writer.writerow({"metric": key, **value, "version": calibration["version"]})
    with (output / "a610_pareto_reclassification.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["snapshot_id", "best_action", "best_family", "category", "primary_gain", "safe", "delta", "relative_delta", "labels", "pareto_set"]); writer.writeheader(); writer.writerows(rows)
    with (output / "a610_common_cohort_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        fields=["snapshot_id", "candidate", "common_unresolved_count", "mean", "median", "p90", "p95", "max"]; writer=csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(cohorts)
    if local:
        with (output / "a610_local_robustness.csv").open("w", newline="", encoding="utf-8") as f:
            writer=csv.DictWriter(f, fieldnames=["snapshot_id", "candidate", "safe", "primary_gain", "material_regression", "delta", "relative_delta", "labels", "minimum_swept_separation"]); writer.writeheader(); writer.writerows(local)
    lines = ["# A6.10 Objective Hierarchy & Pareto Robustness Audit", "", f"A6.10 safety: **{report['summary']['safety']}**", "Objective hierarchy: **DEFINED**", "Materiality thresholds: **DEFINED**", f"A6.9 tradeoff case: **{trade}**", f"Tradeoff source: **{report['summary']['tradeoff_source']}**", "A6.9 near-tie ratio: **0.917**", f"Local gain replication: **{local_status}**", f"High-level performance signal: **{signal}**", "Frozen 29D observation: **NOT_EVALUATED**", "PPO authorization: **STOP**", "Next step: **do not start PPO, A7, A8, or an observation audit unless an explicitly authorized later stage changes the gate.**"]
    (output / "a610_objective_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default=None); args = parser.parse_args()
    output = Path(args.output) if args.output else Path.home()/"training_outputs"/"terminal"/"event_gated_stage15_diagnostics"/"a610"
    a69 = json.loads(_a69_path().read_text(encoding="utf-8")); calibration = calibrate_materiality(a69["matrix"]); rows = reclassify(a69["matrix"], calibration)
    target = next(item for item in rows if item["snapshot_id"] == "bursty-seed1-t900")
    local, cohorts = local_robustness(a69, calibration) if target["category"] in {"ADMISSIBLE_TRADEOFF_GAIN", "DOMINANT_GAIN"} else ([], [])
    print(json.dumps(write_report(a69, calibration, rows, output, local=local, cohorts=cohorts)["summary"]))


if __name__ == "__main__": main()
