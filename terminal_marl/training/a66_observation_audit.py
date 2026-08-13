"""A6.6 oracle-action identifiability audit; explicitly never trains PPO."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import time
from typing import Callable, Iterable

import numpy as np

from ..envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
from ..policies import DynamicRulePolicy
from .a63_dynamic_screen import SHORT_SCENARIOS
from .a65_counterfactual_audit import (
    Candidate, HORIZON_SECONDS, NO_OP, REQUIRED_SAFETY, _compact,
    _branch_from_wrapper, available_cpus, candidate_set, evaluate_task,
)


MEANINGFUL_COMPLETION_MARGIN = 1
MEANINGFUL_WAITING_MARGIN = 50_000.0
NEAR_TIE_WAITING_MARGIN = 25_000.0
RANDOM_SEED = 6606


def _worker_init() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    try:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
    except ImportError:
        pass


def _extended_features(wrapper: DynamicHighLevelWrapper, scenario: str, seed: int, snapshot_id: str) -> dict:
    base, context = wrapper.base, wrapper._context()
    queue = list(base.admission.queue.items)
    active = list(base.active.values())
    priorities = [uav.dynamic_priority for uav in active]
    reserves = [uav.soc for uav in active if uav.soc < base.config.soc_reserve]
    remain = [zone.service_remaining for zone in base.lz_manager.zones if zone.service_remaining > 0.]
    now = base.now
    recent = sum(t >= now - base.config.high_level_decision_interval_seconds for t in base.request_times.values())
    return {
        "snapshot_id": snapshot_id, "scenario": scenario, "seed": seed, "time": now,
        "observation_29d": wrapper.observation().astype(float).tolist(),
        "external_queue": context["external_queue"], "external_oldest_wait": max((item.external_wait_time for item in queue), default=0.),
        "occupancy_ratio": list(context["occupancy_ratio"]), "capacity_slack": [cap - occ for cap, occ in zip(context["capacity"], context["occupancy"])],
        "final_pressure": context["final_pressure"], "lz_utilization": context["lz_utilization"],
        "recent_arrivals_180s": recent, "soc_urgent_count": len(reserves),
        "soc_urgent_min": min(reserves, default=1.), "priority_mean": float(np.mean(priorities)) if priorities else 0.,
        "priority_max": float(np.max(priorities)) if priorities else 0., "pending_reservations": len(base.scheduler.reservations),
        "committed_transfer_load": sum(uav.committed_plan is not None for uav in active),
        "earliest_lz_release": min(remain, default=0.), "pending_reconfiguration": wrapper.coordinator.pending_state.value,
        "transition_active": bool(base.airspace.transition is not None),
    }


def _collect_pool(scenario: str, seed: int) -> list[tuple[dict, bytes]]:
    """Collect every legal 180-second boundary from one real dynamic-rule run."""
    from .a6_effectiveness import scenario_config
    config = scenario_config(SHORT_SCENARIOS[scenario], "dynamic_rule")
    wrapper = DynamicHighLevelWrapper(config, RuleLowLevelAdapter())
    observation, _ = wrapper.reset(seed)
    pool: list[tuple[dict, bytes]] = []
    while wrapper.base.now < config.max_simulation_seconds and not wrapper._done():
        action = DynamicRulePolicy().act(observation, wrapper._context())
        observation, _, _, _, _ = wrapper.step(action)
        if wrapper.base.now >= config.high_level_decision_interval_seconds:
            snapshot_id = f"{scenario}-seed{seed}-t{int(wrapper.base.now)}"
            pool.append((_extended_features(wrapper, scenario, seed, snapshot_id), pickle.dumps(wrapper, protocol=pickle.HIGHEST_PROTOCOL)))
    return pool


def _diverse_select(pool: list[tuple[dict, bytes]], count: int) -> list[tuple[dict, bytes]]:
    """Deterministic max-min selection avoids a cluster of nearly identical boundaries."""
    if len(pool) <= count:
        return pool
    values = np.asarray([meta["observation_29d"] for meta, _ in pool], dtype=float)
    scale = np.maximum(values.std(axis=0), 1e-6); values = (values - values.mean(axis=0)) / scale
    selected = [0, len(pool) - 1]
    while len(selected) < count:
        remaining = [index for index in range(len(pool)) if index not in selected]
        index = max(remaining, key=lambda item: float(np.min(np.sum((values[item] - values[selected]) ** 2, axis=1))))
        selected.append(index)
    return [pool[index] for index in sorted(selected)]


def collect_representative_snapshots(scenarios: Iterable[str], seeds: Iterable[int], per_source: int) -> list[tuple[dict, bytes]]:
    snapshots: list[tuple[dict, bytes]] = []
    for scenario in scenarios:
        for seed in seeds:
            snapshots.extend(_diverse_select(_collect_pool(scenario, seed), per_source))
    return snapshots


def _result_signature(result: dict) -> str:
    return json.dumps(result, sort_keys=True, default=str)


def _relative_value(row: dict, baseline: dict) -> float:
    """Diagnostic value only; oracle ranking remains lexicographic below, never PPO reward."""
    metric, base = row["metrics"], baseline["metrics"]
    return float(100_000. * (metric["completed"] - base["completed"]) - (metric["all_request_waiting_area"] - base["all_request_waiting_area"]) - 2_000. * (metric["unresolved_age_p95"] - base["unresolved_age_p95"]))


def _rank_key(name: str, rows: dict[str, dict]) -> tuple:
    row, metrics = rows[name], rows[name]["metrics"]
    outcomes = row["outcomes"]
    return (
        -metrics["completed"], metrics["all_request_waiting_area"], metrics["unresolved_count"], metrics["unresolved_age_p95"],
        outcomes.get("diverted_low_soc", 0), outcomes.get("failed_energy", 0), metrics["service_delay_p95"],
        metrics["service_delay_p99"], metrics["energy_mean"], metrics["priority_fairness_std"], row["shield_cost"], name,
    )


def _pareto(rows: dict[str, dict]) -> list[str]:
    safe = [name for name, row in rows.items() if row["safe"]]
    def dominates(left: str, right: str) -> bool:
        a, b = rows[left], rows[right]
        am, bm = a["metrics"], b["metrics"]
        values_a = [-am["completed"], am["all_request_waiting_area"], am["unresolved_count"], am["unresolved_age_p95"], am["energy_mean"], am["priority_fairness_std"], a["shield_cost"]]
        values_b = [-bm["completed"], bm["all_request_waiting_area"], bm["unresolved_count"], bm["unresolved_age_p95"], bm["energy_mean"], bm["priority_fairness_std"], b["shield_cost"]]
        return all(x <= y for x, y in zip(values_a, values_b)) and any(x < y for x, y in zip(values_a, values_b))
    return [name for name in safe if not any(dominates(other, name) for other in safe if other != name)]


def _oracle(snapshot_id: str, rows: dict[str, dict], candidates: dict[str, Candidate]) -> dict:
    if not all(name in rows for name in candidates):
        raise RuntimeError(f"incomplete candidate matrix for {snapshot_id}")
    if not rows["no_op"]["safe"]:
        raise RuntimeError(f"unsafe no-op continuation for {snapshot_id}")
    ordered = sorted((name for name, row in rows.items() if row["safe"]), key=lambda name: _rank_key(name, rows))
    if not ordered:
        raise RuntimeError(f"no safe candidate for {snapshot_id}")
    best, second = ordered[0], ordered[1] if len(ordered) > 1 else ordered[0]
    b, s = rows[best]["metrics"], rows[second]["metrics"]
    completion_margin = b["completed"] - s["completed"]
    waiting_margin = s["all_request_waiting_area"] - b["all_request_waiting_area"]
    meaningful = bool(completion_margin >= MEANINGFUL_COMPLETION_MARGIN or (completion_margin == 0 and waiting_margin >= MEANINGFUL_WAITING_MARGIN))
    near_tie = not meaningful
    compatible = [best]
    if near_tie:
        compatible.extend(name for name in ordered[1:] if rows[name]["metrics"]["completed"] == b["completed"] and abs(rows[name]["metrics"]["all_request_waiting_area"] - b["all_request_waiting_area"]) <= NEAR_TIE_WAITING_MARGIN)
    baseline = rows["no_op"]
    return {
        "snapshot_id": snapshot_id, "best_action": best, "best_family": candidates[best].family,
        "second_action": second, "second_family": candidates[second].family,
        "best_vs_second_completion_margin": completion_margin, "best_vs_second_waiting_margin": waiting_margin,
        "meaningful_margin": meaningful, "near_tie": near_tie, "compatible_actions": sorted(set(compatible)),
        "compatible_families": sorted({candidates[name].family for name in compatible}), "pareto_set": _pareto(rows),
        "ranked_actions": ordered, "best_nonzero_vs_no_op": {
            "completed": b["completed"] - baseline["metrics"]["completed"],
            "waiting_area": b["all_request_waiting_area"] - baseline["metrics"]["all_request_waiting_area"],
            "unresolved_age_p95": b["unresolved_age_p95"] - baseline["metrics"]["unresolved_age_p95"],
            "energy": b["energy_mean"] - baseline["metrics"]["energy_mean"],
            "fairness": b["priority_fairness_std"] - baseline["metrics"]["priority_fairness_std"],
        },
    }


def run_counterfactuals(snapshots: list[tuple[dict, bytes]]) -> tuple[dict, dict]:
    candidates = candidate_set(); candidate_by_name = {candidate.name: candidate for candidate in candidates}
    total = len(snapshots) * len(candidates)
    workers = min(total, max(1, available_cpus() - 2))
    results = {meta["snapshot_id"]: {} for meta, _ in snapshots}
    started = time.perf_counter(); durations: list[float] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        futures = [executor.submit(evaluate_task, payload, meta["snapshot_id"], candidate, HORIZON_SECONDS)
                   for meta, payload in snapshots for candidate in candidates]
        for future in as_completed(futures):
            snapshot_id, candidate_name, duration, result = future.result()
            results[snapshot_id][candidate_name] = result; durations.append(duration)
    # An independent no-op replay provides restore/determinism evidence for each stored state.
    replay_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        replays = {executor.submit(evaluate_task, payload, meta["snapshot_id"], Candidate("no_op", "NO_OP", tuple(NO_OP)), HORIZON_SECONDS): meta["snapshot_id"] for meta, payload in snapshots}
        replay_ok = {snapshot_id: _result_signature(results[snapshot_id]["no_op"]) == _result_signature(future.result()[3]) for future, snapshot_id in replays.items()}
    wall = time.perf_counter() - started
    meta = {item[0]["snapshot_id"]: item[0] for item in snapshots}
    telemetry = {
        "available_cpu_count": available_cpus(), "worker_count": workers, "snapshot_count": len(snapshots),
        "counterfactual_branch_count": total, "noop_replay_branch_count": len(snapshots), "wall_clock_seconds": wall,
        "counterfactual_wall_seconds": replay_started - started, "mean_task_runtime_seconds": float(np.mean(durations)),
        "sequential_equivalent_seconds": float(sum(durations)), "parallel_speedup_estimate": float(sum(durations) / max(1e-9, replay_started - started)),
        "tasks_per_second": total / max(1e-9, replay_started - started),
    }
    return {"metadata": meta, "candidates": {c.name: {"family": c.family, "action": list(c.action)} for c in candidates}, "results": results, "noop_replay_ok": replay_ok}, telemetry


def _feature_arrays(matrix: dict, oracle: dict) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    ids = sorted(oracle); meta = matrix["metadata"]
    diagnostic_names = ["external_queue", "external_oldest_wait", "slack_outer", "slack_middle", "slack_inner", "recent_arrivals_180s", "soc_urgent_count", "soc_urgent_min", "priority_mean", "priority_max", "pending_reservations", "committed_transfer_load", "final_pressure", "lz_utilization", "earliest_lz_release", "transition_active"]
    def diagnostic(item: dict) -> list[float]:
        return [item["external_queue"], item["external_oldest_wait"], *item["capacity_slack"], item["recent_arrivals_180s"], item["soc_urgent_count"], item["soc_urgent_min"], item["priority_mean"], item["priority_max"], item["pending_reservations"], item["committed_transfer_load"], item["final_pressure"], item["lz_utilization"], item["earliest_lz_release"], float(item["transition_active"])]
    frozen = np.asarray([meta[sid]["observation_29d"] for sid in ids], float)
    augmented = np.column_stack((frozen, np.asarray([diagnostic(meta[sid]) for sid in ids], float)))
    labels = np.asarray([oracle[sid]["best_family"] for sid in ids], object)
    return frozen, augmented, ids, diagnostic_names, labels


class SoftmaxClassifier:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "SoftmaxClassifier":
        self.classes = np.unique(y); self.mean = x.mean(0); self.std = np.maximum(x.std(0), 1e-6); z = (x - self.mean) / self.std
        self.weights = np.zeros((z.shape[1] + 1, len(self.classes))); encoded = np.searchsorted(self.classes, y)
        if len(self.classes) == 1: return self
        onehot = np.eye(len(self.classes))[encoded]
        for _ in range(700):
            logits = np.clip(np.column_stack((z, np.ones(len(z)))) @ self.weights, -30., 30.); probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
            gradient = np.column_stack((z, np.ones(len(z)))).T @ (probs - onehot) / len(z)
            gradient[:-1] += .001 * self.weights[:-1]; self.weights -= .12 * gradient
        return self
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if len(self.classes) == 1: return np.ones((len(x), 1))
        z = (x - self.mean) / self.std; logits = np.clip(np.column_stack((z, np.ones(len(z)))) @ self.weights, -30., 30.); out = np.exp(logits - logits.max(1, keepdims=True)); return out / out.sum(1, keepdims=True)


@dataclass
class _Leaf:
    prediction: str; probs: dict[str, float]; feature: int | None = None; threshold: float = 0.; left: "_Leaf | None" = None; right: "_Leaf | None" = None


class TinyTreeClassifier:
    """Small deterministic CART-style diagnostic model; no external ML dependency."""
    def fit(self, x: np.ndarray, y: np.ndarray) -> "TinyTreeClassifier":
        self.classes = np.unique(y); self.root = self._build(x, y, 0); return self
    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> _Leaf:
        counts = Counter(y); probs = {key: value / len(y) for key, value in counts.items()}; node = _Leaf(max(counts, key=counts.get), probs)
        if depth >= 3 or len(counts) == 1 or len(y) < 5: return node
        def gini(values: np.ndarray) -> float:
            c = Counter(values); return 1. - sum((n / len(values)) ** 2 for n in c.values())
        best: tuple[float, int, float] | None = None
        parent = gini(y)
        for feature in range(x.shape[1]):
            for threshold in np.unique(np.quantile(x[:, feature], (.25, .5, .75))):
                mask = x[:, feature] <= threshold
                if not mask.any() or mask.all(): continue
                gain = parent - (mask.mean() * gini(y[mask]) + (~mask).mean() * gini(y[~mask]))
                if best is None or gain > best[0]: best = (gain, feature, float(threshold))
        if best is None or best[0] < 1e-7: return node
        _, feature, threshold = best; mask = x[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold; node.left = self._build(x[mask], y[mask], depth + 1); node.right = self._build(x[~mask], y[~mask], depth + 1)
        return node
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), len(self.classes)))
        for index, row in enumerate(x):
            node = self.root
            while node.feature is not None: node = node.left if row[node.feature] <= node.threshold else node.right
            for label, value in node.probs.items(): out[index, np.searchsorted(self.classes, label)] = value
        return out


class TinyMLPClassifier:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "TinyMLPClassifier":
        self.classes = np.unique(y); self.mean = x.mean(0); self.std = np.maximum(x.std(0), 1e-6); z = (x - self.mean) / self.std; rng = np.random.default_rng(RANDOM_SEED)
        hidden = min(12, max(4, x.shape[0] // 2)); self.w1 = rng.normal(0., .12, (z.shape[1], hidden)); self.b1 = np.zeros(hidden); self.w2 = rng.normal(0., .12, (hidden, len(self.classes))); self.b2 = np.zeros(len(self.classes)); encoded = np.searchsorted(self.classes, y)
        if len(self.classes) == 1: return self
        target = np.eye(len(self.classes))[encoded]
        for _ in range(900):
            h_raw = z @ self.w1 + self.b1; h = np.maximum(0., h_raw); logits = np.clip(h @ self.w2 + self.b2, -30., 30.); prob = np.exp(logits - logits.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
            delta = (prob - target) / len(z); grad_w2 = h.T @ delta + .001 * self.w2; grad_b2 = delta.sum(0); grad_h = (delta @ self.w2.T) * (h_raw > 0); grad_w1 = z.T @ grad_h + .001 * self.w1; grad_b1 = grad_h.sum(0)
            self.w2 -= .08 * grad_w2; self.b2 -= .08 * grad_b2; self.w1 -= .08 * grad_w1; self.b1 -= .08 * grad_b1
        return self
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if len(self.classes) == 1: return np.ones((len(x), 1))
        z = (x - self.mean) / self.std; logits = np.clip(np.maximum(0., z @ self.w1 + self.b1) @ self.w2 + self.b2, -30., 30.); out = np.exp(logits - logits.max(1, keepdims=True)); return out / out.sum(1, keepdims=True)


def _classification_metrics(y: np.ndarray, classes: np.ndarray, probabilities: np.ndarray, compatible: list[list[str]]) -> dict:
    prediction = classes[probabilities.argmax(1)]; top2 = [set(classes[np.argsort(row)[-min(2, len(row)):]]) for row in probabilities]
    labels = sorted(set(y) | set(classes)); f1 = []
    recalls = []
    confusion = {actual: {pred: 0 for pred in labels} for actual in labels}
    for actual, predicted in zip(y, prediction): confusion[actual][predicted] += 1
    for label in labels:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(y, prediction)); fp = sum(actual != label and predicted == label for actual, predicted in zip(y, prediction)); fn = sum(actual == label and predicted != label for actual, predicted in zip(y, prediction));
        f1.append(2 * tp / max(1, 2 * tp + fp + fn)); recalls.append(tp / max(1, tp + fn))
    return {"accuracy": float(np.mean(prediction == y)), "macro_f1": float(np.mean(f1)), "balanced_accuracy": float(np.mean(recalls)),
            "top2_accuracy": float(np.mean([actual in choices for actual, choices in zip(y, top2)])),
            "pareto_compatible_top2": float(np.mean([bool(choices & set(options)) for choices, options in zip(top2, compatible)])), "confusion_matrix": confusion}


def _split_specs(meta: dict[str, dict], ids: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    seeds = np.asarray([meta[sid]["seed"] for sid in ids]); times = np.asarray([meta[sid]["time"] for sid in ids]); scenarios = np.asarray([meta[sid]["scenario"] for sid in ids])
    specs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for seed in (0, 1, 2): specs[f"cross_seed_train_not_{seed}_test_{seed}"] = (np.where(seeds != seed)[0], np.where(seeds == seed)[0])
    median = float(np.median(times)); specs["time_train_early_test_later"] = (np.where(times <= median)[0], np.where(times > median)[0])
    specs["scenario_train_high_test_bursty"] = (np.where(scenarios == "high")[0], np.where(scenarios == "bursty")[0])
    specs["scenario_train_bursty_test_high"] = (np.where(scenarios == "bursty")[0], np.where(scenarios == "high")[0])
    return specs


def _evaluate_models(x: np.ndarray, labels: np.ndarray, ids: list[str], meta: dict[str, dict], oracle: dict) -> list[dict]:
    output: list[dict] = []; specs = _split_specs(meta, ids)
    compatible = [oracle[sid]["compatible_families"] for sid in ids]
    models: dict[str, Callable[[], object]] = {"logistic": SoftmaxClassifier, "tiny_tree": TinyTreeClassifier, "tiny_mlp": TinyMLPClassifier}
    for split_name, (train, test) in specs.items():
        if not len(train) or not len(test): continue
        majority = Counter(labels[train]).most_common(1)[0][0]
        baseline = np.full(len(test), majority, dtype=object); baseline_prob = np.ones((len(test), 1)); baseline_classes = np.asarray([majority], object)
        output.append({"feature_set": "majority", "model": "majority", "split": split_name, **_classification_metrics(labels[test], baseline_classes, baseline_prob, [compatible[i] for i in test])})
        for name, factory in models.items():
            model = factory().fit(x[train], labels[train]); probabilities = model.predict_proba(x[test])
            output.append({"feature_set": "", "model": name, "split": split_name, **_classification_metrics(labels[test], model.classes, probabilities, [compatible[i] for i in test])})
    return output


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0: return 0.
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def _ranking_diagnostics(x: np.ndarray, ids: list[str], matrix: dict, meta: dict) -> list[dict]:
    candidates = matrix["candidates"]; candidate_names = sorted(candidates); actions = np.asarray([candidates[name]["action"] for name in candidate_names], float)
    values = np.asarray([[_relative_value(matrix["results"][sid][name], matrix["results"][sid]["no_op"]) for name in candidate_names] for sid in ids])
    output: list[dict] = []
    for split_name, (train, test) in _split_specs(meta, ids).items():
        if not len(train) or not len(test):
            continue
        tx = np.vstack([np.column_stack((np.repeat(x[index:index + 1], len(actions), axis=0), actions)) for index in train]); ty = values[train].ravel()
        mean = tx.mean(0); std = np.maximum(tx.std(0), 1e-6); z = np.column_stack(((tx - mean) / std, np.ones(len(tx)))); weights = np.linalg.pinv(z.T @ z + 1e-3 * np.eye(z.shape[1])) @ z.T @ ty
        errors = []; correlations = []
        for index in test:
            row = np.column_stack((np.repeat(x[index:index + 1], len(actions), axis=0), actions)); predicted = np.column_stack(((row - mean) / std, np.ones(len(row)))) @ weights
            errors.extend(predicted - values[index]); correlations.append(_spearman(predicted, values[index]))
        output.append({"split": split_name, "rank_spearman_mean": float(np.mean(correlations)), "relative_value_mae": float(np.mean(np.abs(errors)))})
    return output


def analyze(matrix: dict, telemetry: dict) -> dict:
    candidates = {name: Candidate(name, item["family"], tuple(item["action"])) for name, item in matrix["candidates"].items()}
    oracle = {sid: _oracle(sid, rows, candidates) for sid, rows in matrix["results"].items()}
    frozen, augmented, ids, diagnostic_names, labels = _feature_arrays(matrix, oracle)
    model_a = _evaluate_models(frozen, labels, ids, matrix["metadata"], oracle)
    model_b = _evaluate_models(augmented, labels, ids, matrix["metadata"], oracle)
    for row in model_a: row["feature_set"] = "frozen_29d"
    for row in model_b: row["feature_set"] = "augmented_29d_plus_diagnostics"
    ablations: list[dict] = []
    blocks = {"reservation_transfer": [10, 11], "demand_trend": [5], "soc_urgency": [6, 7], "priority": [8, 9], "final_lz": [12, 13, 14, 15]}
    for name, positions in blocks.items():
        result = _evaluate_models(np.column_stack((frozen, augmented[:, 29 + np.asarray(positions)])), labels, ids, matrix["metadata"], oracle)
        for row in result:
            if row["model"] != "majority": row["feature_set"] = f"29d_plus_{name}"; ablations.append(row)
    ranking_a = _ranking_diagnostics(frozen, ids, matrix, matrix["metadata"])
    ranking_b = _ranking_diagnostics(augmented, ids, matrix, matrix["metadata"])
    safety = all(all(row["safe"] and all(row["metrics"].get(metric) == 0 for metric in REQUIRED_SAFETY) for row in rows.values()) for rows in matrix["results"].values())
    meaningful_ratio = float(np.mean([entry["meaningful_margin"] for entry in oracle.values()])); near_tie_ratio = float(np.mean([entry["near_tie"] for entry in oracle.values()]))
    labels_by_family = Counter(labels)
    def best_metric(rows: list[dict], feature: str) -> float:
        selected = [row["macro_f1"] for row in rows if row["feature_set"] == feature and row["model"] != "majority"]
        return float(np.mean(selected)) if selected else 0.
    frozen_f1, augmented_f1 = best_metric(model_a + model_b, "frozen_29d"), best_metric(model_a + model_b, "augmented_29d_plus_diagnostics")
    flat_values = meaningful_ratio < .5
    benefit = (not flat_values) and augmented_f1 > frozen_f1 + .10
    frozen_sufficient = meaningful_ratio >= .5 and frozen_f1 >= .50
    augmented_sufficient = meaningful_ratio >= .5 and augmented_f1 >= .50
    final_schema = "phaseA_high_obs_v2" if benefit and augmented_sufficient else "phaseA_high_obs_v1_29d"
    frozen_status = "UNCERTAIN" if flat_values else ("PASS" if frozen_sufficient else "FAIL")
    benefit_status = "NOT_NEEDED" if flat_values else ("YES" if benefit else "NO")
    reason = "ACTION_VALUE_TOO_FLAT" if flat_values else ("OBSERVATION_INSUFFICIENT" if benefit and not frozen_sufficient else ("INSUFFICIENT_SAMPLE_COVERAGE" if len(ids) < 30 else "OBJECTIVE_AMBIGUOUS"))
    decision = "AUTHORIZE_PPO" if safety and frozen_sufficient and meaningful_ratio >= .5 else "STOP_BEFORE_PPO"
    return {"oracle": oracle, "oracle_label_distribution": dict(labels_by_family), "meaningful_margin": {"definition": {"completion": MEANINGFUL_COMPLETION_MARGIN, "waiting_area": MEANINGFUL_WAITING_MARGIN}, "ratio": meaningful_ratio, "near_tie_ratio": near_tie_ratio, "pass": meaningful_ratio >= .5},
            "models": {"frozen_29d": model_a, "augmented_29d_plus_diagnostics": model_b, "feature_ablations": ablations, "ranking_29d": ranking_a, "ranking_augmented": ranking_b, "diagnostic_feature_names": diagnostic_names},
            "safety": "PASS" if safety else "FAIL", "counterfactual_leverage": "PROVEN" if any(entry["best_action"] != "no_op" for entry in oracle.values()) else "NOT_PROVEN", "frozen_29d_sufficiency": frozen_status, "augmented_observation_benefit": benefit_status, "final_observation_schema": final_schema, "decision": decision, "stop_reason": None if decision == "AUTHORIZE_PPO" else reason}


def _write_csv(path: Path, matrix: dict, audit: dict) -> None:
    fields = ["snapshot_id", "scenario", "seed", "time", "candidate", "family", "safe", "rank", "best_action", "best_family", "meaningful_margin", "near_tie", "pareto_set", "relative_value", "completed", "waiting_area", "unresolved_age_p95", "energy", "fairness"] + [f"obs_{index}" for index in range(29)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for sid, rows in matrix["results"].items():
            meta, item = matrix["metadata"][sid], audit["oracle"][sid]
            baseline = rows["no_op"]
            for name, row in rows.items():
                metrics = row["metrics"]; values = {f"obs_{index}": value for index, value in enumerate(meta["observation_29d"])}
                writer.writerow({"snapshot_id": sid, "scenario": meta["scenario"], "seed": meta["seed"], "time": meta["time"], "candidate": name, "family": matrix["candidates"][name]["family"], "safe": row["safe"], "rank": item["ranked_actions"].index(name) + 1 if name in item["ranked_actions"] else "", "best_action": item["best_action"], "best_family": item["best_family"], "meaningful_margin": item["meaningful_margin"], "near_tie": item["near_tie"], "pareto_set": "|".join(item["pareto_set"]), "relative_value": _relative_value(row, baseline), "completed": metrics["completed"], "waiting_area": metrics["all_request_waiting_area"], "unresolved_age_p95": metrics["unresolved_age_p95"], "energy": metrics["energy_mean"], "fairness": metrics["priority_fairness_std"], **values})


def write_report(matrix: dict, telemetry: dict, audit: dict, output: str | Path) -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report = {"stage": "A6.6 observation sufficiency / oracle-action identifiability audit", "evidence_type": "newly_run", "telemetry": telemetry, "snapshot_matrix": matrix, "audit": audit, "protocol_risks": ["Internal simulation counterfactual evidence only; no external generalization claim.", "Held-out splits are grouped by seed/time/scenario but share one simulator family.", "Candidate set is deliberately finite and does not establish continuous-action optimality."]}
    (output / "a66_observation_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_csv(output / "a66_oracle_dataset.csv", matrix, audit)
    with (output / "a66_feature_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = audit["models"]["frozen_29d"] + audit["models"]["augmented_29d_plus_diagnostics"] + audit["models"]["feature_ablations"]
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    margin = audit["meaningful_margin"]
    lines = ["# A6.6 observation sufficiency / oracle-action identifiability audit", "", f"A6.6 safety: **{audit['safety']}**", f"Counterfactual leverage: **{audit['counterfactual_leverage']}**", f"Meaningful action margin: **{'PASS' if margin['pass'] else 'FAIL'}** ({margin['ratio']:.3f}; completion ≥ {MEANINGFUL_COMPLETION_MARGIN} or equal-completion waiting improvement ≥ {MEANINGFUL_WAITING_MARGIN:.0f})", f"Frozen 29D sufficiency: **{audit['frozen_29d_sufficiency']}**", f"Augmented observation benefit: **{audit['augmented_observation_benefit']}**", f"Final observation schema: **{audit['final_observation_schema']}**", f"Decision: **{audit['decision']}**", "", f"Snapshots: {telemetry['snapshot_count']}; counterfactual branches: {telemetry['counterfactual_branch_count']}; no-op replay branches: {telemetry['noop_replay_branch_count']}.", f"Oracle family distribution: {audit['oracle_label_distribution']}; near-tie ratio: {margin['near_tie_ratio']:.3f}.", f"Parallel telemetry: {telemetry}.", "", f"Stop reason: {audit['stop_reason'] or 'none'}.", "", "Evidence type: newly_run. All branches use a restored CRN snapshot through the frozen coordinator/safety/runtime stack; results are an internal identifiability diagnostic, not an external generalization claim."]
    (output / "a66_observation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(scenarios: Iterable[str], seeds: Iterable[int], per_source: int, output: str | Path) -> dict:
    snapshots = collect_representative_snapshots(scenarios, seeds, per_source)
    matrix, telemetry = run_counterfactuals(snapshots)
    if not all(matrix["noop_replay_ok"].values()): raise RuntimeError("snapshot restore + no-op is not deterministic")
    audit = analyze(matrix, telemetry)
    return write_report(matrix, telemetry, audit, output)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--scenario", action="append", choices=("high", "bursty"), default=[]); parser.add_argument("--seed", action="append", type=int, default=[]); parser.add_argument("--snapshots-per-source", type=int, default=5); parser.add_argument("--stage-tag", default="formal"); parser.add_argument("--finalize-from", default=None)
    args = parser.parse_args(); scenarios = args.scenario or ["high", "bursty"]; seeds = args.seed or [0, 1, 2]
    output = Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a66" / args.stage_tag
    if args.finalize_from:
        previous = json.loads(Path(args.finalize_from).read_text(encoding="utf-8")); matrix, telemetry = previous["snapshot_matrix"], previous["telemetry"]
        audit = analyze(matrix, telemetry); report = write_report(matrix, telemetry, audit, output)
        print(json.dumps({"output": str(output), **{key: report["audit"][key] for key in ("safety", "counterfactual_leverage", "frozen_29d_sufficiency", "augmented_observation_benefit", "decision", "stop_reason")}}, default=str)); return
    report = run(scenarios, seeds, args.snapshots_per_source, output)
    print(json.dumps({"output": str(output), **{key: report["audit"][key] for key in ("safety", "counterfactual_leverage", "frozen_29d_sufficiency", "augmented_observation_benefit", "decision", "stop_reason")}}, default=str))


if __name__ == "__main__":
    main()
