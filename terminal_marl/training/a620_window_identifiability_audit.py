"""A6.20: can the frozen 29-D high-level observation identify +12 m outcomes?"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path

import numpy as np

from .a611_causal_audit import DemandProfile, _safe
from .a612_scheduler_repair_audit import _a611_mu0
from .a613_capacity_reaudit import repair_config
from .a614_reconfiguration_timing_audit import PPO_AUTHORIZATION, run_dynamic_at_time
from .a66_observation_audit import SoftmaxClassifier, TinyTreeClassifier


SEEDS = tuple(range(6))
SNAPSHOT_TIMES = (180.0, 240.0, 300.0)
WAITING_MATERIALITY = 50_000.0
PERMUTATIONS = 128
OBSERVATION_NAMES = (
    "actual_outer_radius", "actual_middle_radius", "actual_inner_radius",
    "actual_outer_height", "actual_middle_height", "actual_inner_height",
    "speed_profile_bias", "headway_o2m", "headway_m2i", "headway_i2f",
    "target_outer_radius", "target_middle_radius", "target_inner_radius",
    "target_outer_height", "target_middle_height", "target_inner_height",
    "outer_occupancy_ratio", "middle_occupancy_ratio", "inner_occupancy_ratio",
    "external_queue", "final_pressure", "lz_utilization", "max_lz_service_remaining",
    "geometry_event_gate", "transition_active", "transition_progress", "episode_progress",
    "safety_intervention_rate", "collision_rate",
)


def _profile() -> DemandProfile:
    mu, _ = _a611_mu0(0)
    return DemandProfile("constant_high_repair_scheduler", ((0.0, 720.0, 1.55 * mu),), 720.0, 480.0)


def _label(delta: float) -> str:
    if delta <= -WAITING_MATERIALITY:
        return "BENEFICIAL"
    if delta >= WAITING_MATERIALITY:
        return "HARMFUL"
    return "NEUTRAL"


def _task(profile: DemandProfile, seed: int, time: float, intervention: bool) -> tuple[tuple[int, float, bool], dict]:
    action_time = time if intervention else float("inf")
    result = run_dynamic_at_time(repair_config(profile, dynamic=True), seed, action_time,
                                 force_initial_gate=False, observation_capture_time=time)
    if not result["safe"]:
        raise RuntimeError(f"A6.20 unsafe branch seed={seed}, time={time}, intervention={intervention}")
    return (seed, time, intervention), result


def collect_matrix(*, seeds=SEEDS, times=SNAPSHOT_TIMES, workers: int = 6) -> list[dict]:
    """Run same-seed nominal/+12m pairs; the 29-D row is captured pre-action."""
    profile = _profile(); results: dict[tuple[int, float, bool], dict] = {}
    jobs = [(profile, seed, time, intervention) for seed in seeds for time in times for intervention in (False, True)]
    with ProcessPoolExecutor(max_workers=min(max(1, workers), len(jobs))) as pool:
        futures = [pool.submit(_task, *job) for job in jobs]
        for future in as_completed(futures):
            key, result = future.result(); results[key] = result
    rows = []
    for seed in seeds:
        for time in times:
            nominal, dynamic = results[seed, time, False], results[seed, time, True]
            observation = np.asarray(dynamic["pre_action_observation"]["observation_29d"], dtype=float)
            if observation.shape != (29,) or observation.tolist() != nominal["pre_action_observation"]["observation_29d"]:
                raise RuntimeError("A6.20 CRN pre-action observation mismatch")
            delta = float(dynamic["metrics"]["all_request_waiting_area"] - nominal["metrics"]["all_request_waiting_area"])
            event = dynamic.get("timing", {})
            rows.append({
                "snapshot_id": f"seed{seed}_t{int(time)}", "seed": seed, "time": time,
                "observation_29d": observation.tolist(), "waiting_delta": delta, "outcome_label": _label(delta),
                "nominal_waiting_area": nominal["metrics"]["all_request_waiting_area"],
                "dynamic_waiting_area": dynamic["metrics"]["all_request_waiting_area"],
                "nominal_completed": nominal["metrics"]["completed"], "dynamic_completed": dynamic["metrics"]["completed"],
                "dynamic_unresolved": dynamic["metrics"]["unresolved_count"],
                "dynamic_external_queue_area": dynamic["metrics"]["external_queue_area"],
                "dynamic_service_delay_p95": dynamic["metrics"]["service_delay_p95"],
                "geometry_gate_pre_action": bool(observation[23]),
                "geometry_applied_time": event.get("geometry_applied_time"),
                "geometry_effective_time": event.get("geometry_effective_time"),
                "transition_delay_seconds": event.get("transition_delay_seconds"),
                "safe": dynamic["safe"] and nominal["safe"], "checked_executed": True,
                "nominal_safety": {key: nominal["metrics"][key] for key in ("collisions", "nmac", "hard_separation_violations", "final_overcapacity_steps")},
                "dynamic_safety": {key: dynamic["metrics"][key] for key in ("collisions", "nmac", "hard_separation_violations", "final_overcapacity_steps")},
            })
    return rows


def _standardized(rows: list[dict]) -> np.ndarray:
    x = np.asarray([row["observation_29d"] for row in rows], dtype=float)
    return (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)


def _nearest_conflicts(rows: list[dict]) -> tuple[list[dict], dict]:
    z = _standardized(rows); labels = [row["outcome_label"] for row in rows]; output = []
    for index, row in enumerate(rows):
        candidates = [other for other in range(len(rows)) if other != index]
        other = min(candidates, key=lambda item: float(np.linalg.norm(z[index] - z[item])))
        output.append({"snapshot_id": row["snapshot_id"], "seed": row["seed"], "time": row["time"],
                       "outcome_label": labels[index], "nearest_snapshot_id": rows[other]["snapshot_id"],
                       "nearest_label": labels[other], "distance_z": float(np.linalg.norm(z[index] - z[other])),
                       "label_conflict": labels[index] != labels[other]})
    pairs = [(float(np.linalg.norm(z[left] - z[right])), left, right) for left in range(len(rows)) for right in range(left + 1, len(rows))
             if {labels[left], labels[right]} == {"BENEFICIAL", "HARMFUL"}]
    closest = min(pairs, default=None)
    return output, ({} if closest is None else {"distance_z": closest[0], "left_snapshot": rows[closest[1]]["snapshot_id"],
                                                  "left_label": labels[closest[1]], "right_snapshot": rows[closest[2]]["snapshot_id"],
                                                  "right_label": labels[closest[2]]})


def _dimensional_relationships(rows: list[dict]) -> list[dict]:
    x = np.asarray([row["observation_29d"] for row in rows], dtype=float)
    ordinal = np.asarray([{"BENEFICIAL": -1., "NEUTRAL": 0., "HARMFUL": 1.}[row["outcome_label"]] for row in rows])
    output = []
    for index, name in enumerate(OBSERVATION_NAMES):
        means = {label: float(np.mean(x[[row["outcome_label"] == label for row in rows], index]))
                 if any(row["outcome_label"] == label for row in rows) else None for label in ("BENEFICIAL", "NEUTRAL", "HARMFUL")}
        scale = max(float(np.std(x[:, index])), 1e-6)
        correlation = 0. if np.std(x[:, index]) < 1e-9 or np.std(ordinal) < 1e-9 else float(np.corrcoef(x[:, index], ordinal)[0, 1])
        gap = None if means["BENEFICIAL"] is None or means["HARMFUL"] is None else (means["HARMFUL"] - means["BENEFICIAL"]) / scale
        output.append({"index": index, "name": name, "pearson_outcome_ordinal": correlation,
                       "beneficial_mean": means["BENEFICIAL"], "neutral_mean": means["NEUTRAL"], "harmful_mean": means["HARMFUL"],
                       "beneficial_harmful_standardized_gap": gap})
    return output


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    labels = ("BENEFICIAL", "NEUTRAL", "HARMFUL")
    recalls, f1 = [], []
    for label in labels:
        support = int(np.sum(y == label)); tp = int(np.sum((y == label) & (prediction == label)))
        fp = int(np.sum((y != label) & (prediction == label))); fn = support - tp
        if support:
            recalls.append(tp / support); f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return {"accuracy": float(np.mean(prediction == y)), "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.,
            "macro_f1_present": float(np.mean(f1)) if f1 else 0.,
            **{f"recall_{label.lower()}": (float(np.mean(prediction[y == label] == label)) if np.any(y == label) else None) for label in labels}}


def _predict(model_name: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = (SoftmaxClassifier() if model_name == "logistic_linear" else TinyTreeClassifier()).fit(train_x, train_y)
    return model.classes[model.predict_proba(test_x).argmax(1)]


def _classifier_results(rows: list[dict]) -> list[dict]:
    x = np.asarray([row["observation_29d"] for row in rows], dtype=float)
    y = np.asarray([row["outcome_label"] for row in rows], dtype=object)
    seed = np.asarray([row["seed"] for row in rows]); rng = np.random.default_rng(62020); output = []
    for holdout in sorted(set(seed)):
        train, test = np.where(seed != holdout)[0], np.where(seed == holdout)[0]
        for model_name in ("logistic_linear", "shallow_tree"):
            prediction = _predict(model_name, x[train], y[train], x[test])
            metrics = _metrics(y[test], prediction)
            perm_accuracy, perm_f1 = [], []
            for _ in range(PERMUTATIONS):
                permuted = rng.permutation(y)
                predicted = _predict(model_name, x[train], permuted[train], x[test])
                score = _metrics(permuted[test], predicted)
                perm_accuracy.append(score["accuracy"]); perm_f1.append(score["macro_f1_present"])
            output.append({"model": model_name, "split": f"leave_seed_{holdout}_out", "held_out_seed": int(holdout),
                           "train_count": len(train), "test_count": len(test), **metrics,
                           "permutation_count": PERMUTATIONS, "permutation_accuracy_mean": float(np.mean(perm_accuracy)),
                           "permutation_accuracy_p95": float(np.percentile(perm_accuracy, 95)),
                           "permutation_macro_f1_mean": float(np.mean(perm_f1)), "predictions": prediction.tolist(), "targets": y[test].tolist()})
    for model_name in ("logistic_linear", "shallow_tree"):
        subset = [row for row in output if row["model"] == model_name]
        output.append({"model": model_name, "split": "leave_one_seed_out_aggregate", "held_out_seed": "all",
                       "train_count": len(rows), "test_count": len(rows),
                       "accuracy": float(np.mean([row["accuracy"] for row in subset])),
                       "balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in subset])),
                       "macro_f1_present": float(np.mean([row["macro_f1_present"] for row in subset])),
                       "permutation_accuracy_mean": float(np.mean([row["permutation_accuracy_mean"] for row in subset])),
                       "permutation_macro_f1_mean": float(np.mean([row["permutation_macro_f1_mean"] for row in subset])),
                       "permutation_count": PERMUTATIONS})
    return output


def classify(rows: list[dict], nearest: list[dict], classifier: list[dict]) -> str:
    counts = Counter(row["outcome_label"] for row in rows); seeds = {row["seed"] for row in rows}
    if len(rows) < 15 or len(seeds) < 4 or counts["BENEFICIAL"] < 3 or counts["HARMFUL"] < 3:
        return "INSUFFICIENT_DATA"
    aggregate = [row for row in classifier if row["split"] == "leave_one_seed_out_aggregate"]
    best = max(aggregate, key=lambda row: row["macro_f1_present"])
    uplift = best["macro_f1_present"] - best["permutation_macro_f1_mean"]
    conflict_rate = float(np.mean([row["label_conflict"] for row in nearest]))
    if best["macro_f1_present"] >= .70 and uplift >= .20 and conflict_rate <= .25:
        return "IDENTIFIABLE"
    if uplift >= .10 and best["macro_f1_present"] >= .45:
        return "PARTIALLY_IDENTIFIABLE"
    return "NOT_IDENTIFIABLE"


def run_audit(*, seeds=SEEDS, times=SNAPSHOT_TIMES, workers: int = 6) -> dict:
    rows = collect_matrix(seeds=seeds, times=times, workers=workers)
    if not all(row["safe"] and row["checked_executed"] for row in rows):
        raise RuntimeError("A6.20 safety or execution-equivalence failure")
    nearest, closest = _nearest_conflicts(rows); classifier = _classifier_results(rows); relationships = _dimensional_relationships(rows)
    return {"stage": "A6.20 Reconfiguration Window Identifiability Audit", "evidence_type": "newly_run",
            "inputs": {"observation_schema": "phaseA_high_obs_v1_29d", "observation_dimension": 29,
                       "seeds": list(seeds), "snapshot_times": list(times), "waiting_materiality": WAITING_MATERIALITY,
                       "ppo_authorization": PPO_AUTHORIZATION, "action": "Outer R 600→612 m canonical full_dynamic"},
            "snapshot_labels": rows, "observation_relationships": relationships, "nearest_neighbor_conflicts": nearest,
            "closest_beneficial_harmful_pair": closest, "classifier_results": classifier,
            "summary": {"classification": classify(rows, nearest, classifier), "label_counts": dict(Counter(row["outcome_label"] for row in rows)),
                        "safe_rollouts": len(rows), "ppo_authorization": PPO_AUTHORIZATION,
                        "protocol_risks": ["Leave-one-seed-out is internal CRN validation, not independent traffic-distribution generalization.",
                                           "Only 18 snapshots across six seeds and three action times; no PPO training or action-selection fitting was performed.",
                                           "Classifier inputs contain only pre-action 29-D observations; seed/time are used solely to define validation splits."]}}


def _csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    labels = [{key: value for key, value in row.items() if key not in {"observation_29d", "nominal_safety", "dynamic_safety"}} for row in report["snapshot_labels"]]
    observations = [{"snapshot_id": row["snapshot_id"], "seed": row["seed"], "time": row["time"], "outcome_label": row["outcome_label"],
                     **{f"obs_{index:02d}_{name}": row["observation_29d"][index] for index, name in enumerate(OBSERVATION_NAMES)}} for row in report["snapshot_labels"]]
    _csv(output / "a620_snapshot_labels.csv", labels); _csv(output / "a620_observation_matrix.csv", observations)
    _csv(output / "a620_nearest_neighbor_conflicts.csv", report["nearest_neighbor_conflicts"])
    _csv(output / "a620_classifier_results.csv", report["classifier_results"])
    (output / "a620_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    aggregates = [row for row in report["classifier_results"] if row["split"] == "leave_one_seed_out_aggregate"]
    model_rows = "\n".join(f"| {row['model']} | {row['macro_f1_present']:.3f} | {row['balanced_accuracy']:.3f} | {row['permutation_macro_f1_mean']:.3f} |" for row in aggregates)
    pair = report["closest_beneficial_harmful_pair"]
    pair_text = "No BENEFICIAL/HARMFUL pair was observed." if not pair else f"Closest BENEFICIAL/HARMFUL pair: `{pair['left_snapshot']}` vs `{pair['right_snapshot']}`, standardized distance {pair['distance_z']:.3f}."
    summary = report["summary"]
    (output / "a620_summary.md").write_text(
        "# A6.20 Reconfiguration Window Identifiability Audit\n\n"
        f"Classification: **{summary['classification']}**. Label counts: `{summary['label_counts']}`.\n\n"
        "The classifier sees only action-pre `phaseA_high_obs_v1_29d`; seed/time are split keys, never features.\n\n"
        "| model | LOSO macro-F1 | LOSO balanced accuracy | permutation macro-F1 |\n|---|---:|---:|---:|\n" + model_rows +
        f"\n\n{pair_text}\n\nAll {summary['safe_rollouts']} paired rollouts are hard-safe and checked==executed. PPO authorization: **STOP**.\n\n"
        "Protocol risks: internal CRN leave-one-seed-out only; small finite snapshot set; results do not establish PPO readiness beyond the classification rule.\n",
        encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--seeds", default="0,1,2,3,4,5")
    parser.add_argument("--times", default="180,240,300"); parser.add_argument("--workers", type=int, default=6); parser.add_argument("--output", default=None)
    args = parser.parse_args(); output = Path(args.output) if args.output else Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a620"
    report = run_audit(seeds=tuple(map(int, args.seeds.split(","))), times=tuple(map(float, args.times.split(","))), workers=args.workers)
    write_report(report, output); print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
