"""Read-only stage4 label/ranking diagnostics; no simulation or model fitting."""
import os
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from terminal_marl.core.fixed_terminal.learning_data import load_contexts
from terminal_marl.core.fixed_terminal.learning_models import Scorer, root_objects, true_pairs
from terminal_marl.core.fixed_terminal.learning_scheduler import prediction_order
from terminal_marl.core.fixed_terminal.reference import action_key, value_comparison_key
from terminal_marl.scripts.evaluate_fixed_terminal_stage4 import timing_summary

METHODS = ("beam_topk", "value_topk", "rank_topk")
COMPONENTS = ("failed_energy", "normal_landed", "emergency_landed", "arrived_unresolved",
              "observed_request_seconds", "normal_delay_sum_seconds", "energy_consumed_soc_all_requests")
QUANTILES = (0., .25, .5, .75, .9, .95, 1.)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_lines(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def context_id(context):
    m = context["metadata"]
    return f"{m['load']}-{m['soc']}-{m['seed']}-t{m['root_time']:g}"


def selected_value(context, root_ids):
    """Look up labels and reproduce exact-key/action tie-breaking, without rollout."""
    roots = root_objects(context)
    keys = [value_comparison_key(v) for v in context["values"]]
    indices = [i for i, r in enumerate(roots) if r.root_id in set(root_ids)]
    if not indices:
        raise ValueError("empty shortlist")
    best = max(keys[i] for i in indices)
    chosen = min((i for i in indices if keys[i] == best), key=lambda i: action_key(roots[i].plans))
    full_best = max(keys)
    full = min((i for i in range(len(roots)) if keys[i] == full_best), key=lambda i: action_key(roots[i].plans))
    value, reference = context["values"][chosen], context["values"][full]
    return {"root_ids": [r.root_id for r in roots if r.root_id in set(root_ids)],
            "selected_count": len(indices), "best_key_hit": best == full_best,
            "chosen_root_id": roots[chosen].root_id, "batch_size": len(roots[chosen].plans),
            "chosen_action_key": action_key(roots[chosen].plans), "chosen_key": best,
            "chosen_value": value, "full_best_root_id": roots[full].root_id,
            "raw_J_regret_full_minus_chosen": [a-b for a, b in zip(reference["J"], value["J"])],
            "raw_component_regret_full_minus_chosen": {k: reference["components"][k]-value["components"][k] for k in COMPONENTS}}


def analyze_context(context, predictions):
    roots = root_objects(context)
    ids = [r.root_id for r in roots]
    keys = [value_comparison_key(v) for v in context["values"]]
    beam = list(context["pool_metadata"]["beam_order"])
    anchors = list(dict.fromkeys(context["pool_metadata"]["anchors"]))
    if not set(beam+anchors) <= set(ids):
        raise ValueError("unknown pool root")
    result = {"context_id": context_id(context), "source": context["path"],
              "pool_size": len(roots), "degenerate": len(roots) == 1,
              "informative": len(set(keys)) > 1, "anchors": anchors,
              "root_batch_sizes": {r.root_id: len(r.plans) for r in roots},
              "true_J": {r.root_id: v["J"] for r, v in zip(roots, context["values"])},
              "true_delta_J": dict(zip(ids, context["targets"])), "methods": {}}
    pairs = true_pairs(context)
    for method in METHODS:
        original = beam if method == "beam_topk" else prediction_order(roots, predictions[method], method.split("_")[0])
        ranked = original+[r.root_id for r in sorted(roots, key=lambda r: action_key(r.plans)) if r.root_id not in original]
        positions = {root_id: i for i, root_id in enumerate(ranked)}
        correct = sum(positions[ids[i]] < positions[ids[j]] for i, j in pairs)
        entry = {"original_order": original, "complete_ranking": ranked,
                 "predictions": predictions.get(method), "non_tie_pairs": len(pairs), "pair_correct": correct,
                 "all_non_tie_pairs_correct": correct == len(pairs) if pairs else None, "topk": {}}
        for k in (1, 3, 5, "all"):
            before = ids if k == "all" else original[:k]
            # Beam generation always includes at least one root; reject malformed fixtures.
            entry["topk"][str(k)] = {"before_anchors": selected_value(context, before),
                                         "after_anchors": selected_value(context, set(before) | set(anchors))}
        result["methods"][method] = entry
    return result


def summarize_contexts(rows):
    grouped = {}
    axes = {"all": lambda r: "all", "degenerate": lambda r: str(r["degenerate"]),
            "informative": lambda r: str(r["informative"]), "pool_size": lambda r: str(r["pool_size"])}
    for method in METHODS:
        grouped[method] = {}
        for k in ("1", "3", "5", "all"):
            grouped[method][k] = {}
            for phase in ("before_anchors", "after_anchors"):
                output = {}
                for axis, key in {**axes, "batch_size": lambda r: str(r["methods"][method]["topk"][k][phase]["batch_size"])}.items():
                    buckets = defaultdict(list)
                    for row in rows:
                        buckets[key(row)].append(row["methods"][method]["topk"][k][phase])
                    output[axis] = {name: {"contexts": len(items), "best_key_hits": sum(x["best_key_hit"] for x in items),
                                          "selected_candidates_sum": sum(x["selected_count"] for x in items),
                                          "raw_J_regret_sum": np.sum([x["raw_J_regret_full_minus_chosen"] for x in items], axis=0).tolist()}
                                    for name, items in sorted(buckets.items())}
                grouped[method][k][phase] = output
    cross, lists = Counter(), defaultdict(list)
    for row in rows:
        rank, beam = (row["methods"][m] for m in ("rank_topk", "beam_topk"))
        r, b = rank["topk"]["3"]["after_anchors"], beam["topk"]["3"]["after_anchors"]
        cross[json.dumps({"informative": row["informative"], "rank_all_pairs_correct": rank["all_non_tie_pairs_correct"],
                          "rank_top1_hit": rank["topk"]["1"]["before_anchors"]["best_key_hit"],
                          "same_chosen_key_after_anchors": r["chosen_key"] == b["chosen_key"],
                          "same_chosen_action_after_anchors": r["chosen_action_key"] == b["chosen_action_key"]}, sort_keys=True)] += 1
        for phase in ("before_anchors", "after_anchors"):
            rh, bh = rank["topk"]["3"][phase]["best_key_hit"], beam["topk"]["3"][phase]["best_key_hit"]
            lists[phase+"_rank_hit_beam_miss" if rh and not bh else phase+"_beam_hit_rank_miss" if bh and not rh else phase+"_both_hit" if rh else phase+"_both_miss"].append(row["context_id"])
        for method in METHODS:
            before, after = (row["methods"][method]["topk"]["3"][p]["best_key_hit"] for p in ("before_anchors", "after_anchors"))
            if after and not before:
                lists[method+"_anchor_rescued"].append(row["context_id"])
    return {"groups": grouped, "rank_beam_top3_crosstab": [{**json.loads(key), "contexts": count} for key, count in sorted(cross.items())],
            "top3_context_lists": dict(lists)}


def error_summary(errors):
    errors = np.asarray(errors, dtype=float)
    if not len(errors):
        return {"rows": 0}
    return {"rows": len(errors), "MAE": np.abs(errors).mean(axis=0).tolist(),
            "signed_error_quantiles": {str(q): np.quantile(errors, q, axis=0).tolist() for q in QUANTILES},
            "absolute_error_quantiles": {str(q): np.quantile(np.abs(errors), q, axis=0).tolist() for q in QUANTILES},
            "quantile_definition": "numpy_linear_interpolation"}


def value_diagnostics(contexts, predictions):
    errors, rounded, holds, targets, absolute_J = [], [], [], [], []
    per_context = []
    for context, prediction in zip(contexts, predictions):
        target = np.asarray(context["targets"])
        prediction = np.asarray(prediction)
        error = prediction-target
        rounded_error = np.array([[round(float(x)) for x in row[:4]] for row in prediction])-target[:, :4]
        errors.extend(error); rounded.extend(rounded_error); targets.extend(target)
        absolute_J.extend(v["J"] for v in context["values"])
        hold_indices = [i for i, r in enumerate(context["roots"]) if "HOLD" in r["labels"]]
        if len(hold_indices) != 1 or np.any(target[hold_indices[0]] != 0):
            raise ValueError("HOLD delta label is not exactly zero")
        holds.append(error[hold_indices[0]])
        per_context.append({"context_id": context_id(context), "root_ids": [r["root_id"] for r in context["roots"]],
                            "raw_signed_errors": error.tolist(), "rounded_first_four_errors": rounded_error.tolist(),
                            "HOLD_root_id": context["roots"][hold_indices[0]]["root_id"], "HOLD_prediction": prediction[hold_indices[0]].tolist()})
    distributions = {}
    for name, values in (("delta_J", targets), ("absolute_J", absolute_J)):
        distributions[name] = {str(i): dict(sorted(Counter(str(int(row[i])) for row in values).items())) for i in range(4)}
    return {"target": "raw_delta_J_vs_HOLD", "J_component_order": list(COMPONENTS),
            "J_signs": [-1, 1, -1, -1, -1, -1, -1], "all_candidates": error_summary(errors),
            "rounded_first_four": error_summary(rounded), "HOLD": error_summary(holds),
            "true_integer_distributions": distributions, "contexts": per_context}


def identical_feature_labels(contexts):
    buckets = defaultdict(list)
    for context in contexts:
        for root, features, value, delta in zip(context["roots"], context["features"], context["values"], context["targets"]):
            buckets[tuple(features)].append({"context_id": context_id(context), "root_id": root["root_id"],
                                             "J": value["J"], "delta_J": delta, "key": value_comparison_key(value)})
    duplicates = [rows for rows in buckets.values() if len(rows) > 1]
    different = {}
    for field in ("J", "delta_J", "key"):
        groups = [rows for rows in duplicates if len({tuple(r[field]) for r in rows}) > 1]
        pair_count = 0
        for rows in groups:
            counts = Counter(tuple(r[field]) for r in rows)
            pair_count += (len(rows)**2-sum(n*n for n in counts.values()))//2
        different[field] = {"groups": len(groups), "rows": sum(map(len, groups)), "unequal_label_pairs": pair_count,
                            "examples_first_5_groups": [rows[:5] for rows in groups[:5]]}
    return {"equality": "exact_numeric_equality_of_all_93_features; no_approximate_clustering",
            "candidate_rows": sum(len(c["roots"]) for c in contexts), "unique_feature_vectors": len(buckets),
            "duplicate_groups": len(duplicates), "different_labels": different}


def plan_key(plans):
    return action_key(tuple(SimpleNamespace(aircraft_id=p["aircraft_id"], wait_cycles=tuple(p["wait_cycles"])) for p in plans))


def episode_diagnostics(stage4_root):
    records, selected_decisions = {}, {}
    for path in sorted((stage4_root/"episodes").glob("*/result.json")):
        result = read_json(path)
        if result.get("schema") != "fixed-terminal-stage4-episode-v1":
            continue
        scenario = result["scenario"]
        key = (scenario["load"], scenario["soc_profile"], scenario["seed"])
        method = result["scheduler"]
        decisions = list(read_lines(path.parent/"decisions.jsonl"))
        timings = timing_summary(decisions)
        if timings != result["decision_timings"]:
            raise ValueError("saved decision timing mismatch: "+str(path))
        if (method, key) in records:
            raise ValueError("duplicate episode")
        counts = result["summary"]["outcomes"]
        records[method, key] = {"source": str(path), "N": counts["normal_landed"], "E": counts["emergency_landed"],
                               "F": counts["failed_energy"], "C": counts["censored"],
                               "request_count": result["summary"]["request_count"], "decision_count": len(decisions),
                               "timings": timings, "cold_model_load_seconds": result["model_load_seconds"],
                               "environment_step_seconds": result["environment_step_seconds"]}
        if key == ("medium", "low", 4303) and method in ("rank_topk", "beam_topk", "full_reference"):
            selected_decisions[method] = decisions
    paired = {}
    scenarios = sorted(key for method, key in records if method == "rank_topk")
    for reference in ("beam_topk", "full_reference"):
        paired[reference] = [{"scenario": dict(zip(("load", "soc", "seed"), key)),
                              "rank_topk": records["rank_topk", key], "reference": records[reference, key],
                              "delta_rank_minus_reference": {name: records["rank_topk", key][name]-records[reference, key][name] for name in ("N", "E", "F", "C")}}
                             for key in scenarios]
    first = {}
    for reference in ("beam_topk", "full_reference"):
        left, right = selected_decisions["rank_topk"], selected_decisions[reference]
        if [r["time"] for r in left] != [r["time"] for r in right]:
            raise ValueError("decision times differ")
        divergence = next(((i, a, b) for i, (a, b) in enumerate(zip(left, right)) if plan_key(a["normal_plans"]) != plan_key(b["normal_plans"])), None)
        if divergence:
            i, a, b = divergence
            first[reference] = {"time": a["time"], "zero_based_decision_index": i,
                                "rank_action_key": plan_key(a["normal_plans"]), "reference_action_key": plan_key(b["normal_plans"]),
                                "rank_normal_plans": a["normal_plans"], "reference_normal_plans": b["normal_plans"],
                                "decision_sources": {m: str(stage4_root/"episodes"/f"{m}-medium-low-4303"/"decisions.jsonl") for m in ("rank_topk", reference)},
                                "state_snapshot": "not_recorded_in_decisions; no_state_replay_or_inference_performed"}
        else:
            first[reference] = None
    return {"episodes": len(records), "rows": [{"method": method, "scenario": dict(zip(("load", "soc", "seed"), key)), **row} for (method, key), row in records.items()],
            "paired_rank_minus_reference": paired, "medium_low_4303_first_different_normal_action": first}


def stage3_counts(stage4_root):
    path = stage4_root.parent/"tmp_stage3_results"/"summary"/"conditional_margins.json"
    if not path.is_file():
        return {"source": str(path), "status": "missing", "new_counterfactuals_run": 0}
    rows = read_json(path)
    return {"source": str(path), "status": "read_existing", "rows": len(rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "context_count": len({r["context"] for r in rows}), "new_counterfactuals_run": 0,
            "interpretation": "counts_of_preexisting_stage3_conditional_remove_one_records_only"}


def analyze(stage4_root):
    stage4_root = Path(stage4_root).resolve()
    saved = read_json(stage4_root/"summary"/"offline_test.json")
    contexts = load_contexts(stage4_root/"data"/"test", "test")
    if len(contexts) != saved["contexts"]:
        raise ValueError("saved offline context count mismatch")
    predictions, hashes = {}, {}
    for kind in ("value", "rank"):
        path = stage4_root/"models"/f"{kind}.pt"
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        if hashes[path.name] != saved["models"]["files"][path.name]:
            raise ValueError("frozen checkpoint hash mismatch")
        scorer = Scorer.load(path)
        predictions[kind+"_topk"] = [scorer.predict(c["features"]) for c in contexts]
    rows = [analyze_context(c, {m: predictions[m][i] for m in predictions}) for i, c in enumerate(contexts)]
    for method in METHODS:
        old = {str(Path(r["context"]).parent.name)+"-"+Path(r["context"]).name: r for r in saved["methods"][method]["contexts"]}
        for row in rows:
            previous = old[row["context_id"]]
            actual = row["methods"][method]
            chosen = actual["topk"]["3"]["after_anchors"]
            if (chosen["chosen_root_id"] != previous["filtered_chosen"] or actual["pair_correct"] != previous["pair_correct"]
                    or chosen["raw_J_regret_full_minus_chosen"] != previous["raw_J_regret_full_minus_filtered"]):
                raise ValueError("stage4 saved offline reproduction mismatch")
    return {"schema": "fixed-terminal-stage5-stage4-diagnostics-v1", "stage4_root": str(stage4_root),
            "source_offline": str(stage4_root/"summary"/"offline_test.json"), "models_sha256": hashes,
            "scope": "saved_test_labels_and_frozen_predictions_only; no_rollouts_or_training",
            "selection_key": "four_exact_integer_counts; exposure_delay_round8; energy_round10; action_key_tie",
            "beam_order_scope": "original_beam_order_for_finite_k; full_pool_for_all; complete_ranking_appends_nonbeam_roots_by_action_key",
            "saved_offline_top3_reproduction": True, "contexts": rows, "summary": summarize_contexts(rows),
            "value_errors": value_diagnostics(contexts, predictions["value_topk"]),
            "identical_features": identical_feature_labels(contexts), "episodes": episode_diagnostics(stage4_root),
            "stage3_remove_one": stage3_counts(stage4_root)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.stage4_root)
    serialized = json.dumps(result, indent=2, allow_nan=False)+"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    print(json.dumps({"output": str(args.output), "contexts": len(result["contexts"]), "episodes": result["episodes"]["episodes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
