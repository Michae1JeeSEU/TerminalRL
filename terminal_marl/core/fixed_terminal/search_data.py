"""New partial-search traces on reused stage4 frozen inputs and prefixes.

Labels describe contribution to a retained finite-reference best leaf. A zero
does not establish a branch's infeasibility, counterfactual value or global merit.
Internal nodes inherit leaf linkage; they do not receive fabricated rollouts.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from math import isfinite
import json
import os
from pathlib import Path
from time import perf_counter

from .baselines import observed_clone, apply_scheduler
from .domain import Intent
from .learning_data import write_json
from .reference import value_comparison_key
from .search_features import FEATURE_NAMES, SCHEMA as FEATURE_SCHEMA, extract_search_features
from .search_scheduler import _Search
from .assisted_scheduler import visible_counts, completion_certificate
from .scenarios import generate_scenario, Scenario, Request
from .value_analysis import visible_snapshot, canonical_json, config_from_dict
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs.fixed_terminal_stage7 import validate_split, SNAPSHOT_TIMES, REFERENCE_WIDTH, TRACE_WIDTHS

CONTEXT_SCHEMA = "fixed-terminal-stage7-context-v2"
LABEL_SCOPE = "real_processed_lineage_prefix_contribution_to_any_best_retained_width128_leaf; zero_is_not_counterfactual_infeasibility_or_no_value"


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def reference_trace(env):
    tick = perf_counter()
    state = _Search(observed_clone(env))
    for width in TRACE_WIDTHS:
        roots, stage = state.run(width)
    reference_stage = stage["stage"]
    chosen, _ = state.evaluate(roots)
    best_key = value_comparison_key(state.evaluations[chosen.root_id]["value"])
    best_ids = [r.root_id for r in roots if value_comparison_key(state.evaluations[r.root_id]["value"]) == best_key]
    node_by_id = {n["node_id"]: n for n in state.nodes}
    reference_links = {r.root_id: [key for key in state.leaf_links.get(r.root_id, []) if node_by_id[key]["stage"] == reference_stage] for r in roots}
    descendants = {n["node_id"]: [] for n in state.nodes}
    def lineage_key(lineage):
        return tuple((row["holder"], tuple(row["wait_cycles"]) if row["wait_cycles"] is not None else None) for row in lineage)
    prefix_roots = {}
    for root in roots:
        for leaf_id in reference_links[root.root_id]:
            lineage = lineage_key(node_by_id[leaf_id]["lineage"])
            for depth in range(len(lineage)+1):
                prefix_roots.setdefault(lineage[:depth], set()).add(root.root_id)
    for node in state.nodes:
        descendants[node["node_id"]] = sorted(prefix_roots.get(lineage_key(node["lineage"]), ()))
    feature_tick = perf_counter()
    feature_rows, label_rows = [], []
    for node in state.nodes:
        eligible = node["depth"] < len(state.catalogue) and (node["validation"] is None or node["validation"]["status"] == "feasible")
        linked = descendants[node["node_id"]]
        contributes = any(rid in best_ids for rid in linked)
        label_rows.append({"node_id": node["node_id"], "expand_label": int(contributes),
                           "eligible_expand_training": eligible, "descendant_evaluated_root_ids": linked,
                           "best_descendant_root_ids": [rid for rid in linked if rid in best_ids],
                           "own_rollout": False, "label_scope": LABEL_SCOPE})
        if eligible:
            plans = tuple(env.propose(Intent(key, tuple(cycles))) for key, cycles in node["action_signature"])
            features = extract_search_features(env, plans, state.catalogue, node["depth"], node["width"], node["frontier_count"])
            feature_rows.append({"node_id": node["node_id"], "values": features})
    feature_seconds = perf_counter()-feature_tick
    counts = visible_counts(env)
    leaves = [{**r.serializable(), "tree_leaf_node_ids": reference_links[r.root_id],
               "evaluation": state.evaluations[r.root_id],
               "certificate": completion_certificate(state.evaluations[r.root_id]["value"], counts),
               "best_retained_leaf": r.root_id in best_ids} for r in roots]
    trace = state.trace()
    trace.update(trace_widths=list(TRACE_WIDTHS), evaluated_reference_stage=reference_stage,
                 label_linkage="exact_holder_and_None_or_wait_cycles_processed_prefix_across_real_search_stages")
    return {"trace": trace, "features": {"schema": FEATURE_SCHEMA, "feature_names": FEATURE_NAMES, "rows": feature_rows},
            "labels": {"scope": LABEL_SCOPE, "rows": label_rows, "best_root_ids": best_ids,
                       "unlinked_best_anchors": [r for r in best_ids if not reference_links[r]],
                       "trace_widths": list(TRACE_WIDTHS), "reference_stage": reference_stage,
                       "cross_stage_linkage": "exact_real_lineage_processed_prefix; only_width128_roots_evaluated"},
            "leaves": {"rows": leaves, "scope": "at_most14_retained_roots_width128; no_internal_node_counterfactual_rollouts"},
            "timing": {**state.timing, "dataset_features": feature_seconds, "trace_and_label_seconds": perf_counter()-tick},
            "statistics": asdict(state.stats), "work": {**trace["counters"],
                **{k: sum(v["work"][k] for v in state.evaluations.values()) for k in
                   ("rollouts", "simulation_steps", "root_validation_calls", "continuation_submission_calls")}}}


def compare_source_snapshot(env, source_snapshot):
    """Reuse the historical artifact-only roundoff rule; physics stays exact."""
    from terminal_marl.scripts.evaluate_fixed_terminal_stage5 import compare_continuation_artifact
    stored = json.loads(Path(source_snapshot).read_text(encoding="utf-8"))
    compare_continuation_artifact(stored["config"], canonical_json(asdict(env.config)), "$.snapshot.config", exact=True)
    return compare_continuation_artifact(stored, canonical_json(visible_snapshot(env)), "$.snapshot")


def compare_source_scenario(stored, rebuilt):
    """Allow only finite float SOC roundoff in old input artifacts.

    Normalize a comparison copy, never the scenario used by the simulator.
    All non-SOC values, types, structure and order remain exactly checked.
    """
    from terminal_marl.scripts.evaluate_fixed_terminal_stage5 import compare_continuation_artifact
    if (type(stored) is not dict or type(rebuilt) is not dict
            or type(stored.get("requests")) is not list or type(rebuilt.get("requests")) is not list
            or len(stored["requests"]) != len(rebuilt["requests"])):
        raise ValueError("scenario artifact requests structure mismatch")
    normalized = deepcopy(rebuilt)
    differences = []
    for index, (old, new) in enumerate(zip(stored["requests"], rebuilt["requests"])):
        path = f"scenario.requests[{index}].soc"
        if type(old) is not dict or type(new) is not dict or "soc" not in old or "soc" not in new:
            raise ValueError("scenario artifact request structure mismatch at "+path)
        left, right = old["soc"], new["soc"]
        if type(left) is not float or type(right) is not float or not isfinite(left) or not isfinite(right):
            raise ValueError("scenario SOC requires same-type finite floats at "+path)
        audit = compare_continuation_artifact(left, right, path)
        if not audit["exact_match"]:
            differences.append({"path": path, "stored": left, "rebuilt": right,
                                "abs_difference": audit["max_abs_difference"]})
        normalized["requests"][index]["soc"] = left
    compare_continuation_artifact(stored, normalized, "scenario", exact=True)
    return {"exact_match": not differences, "float_difference_count": len(differences),
            "max_abs_difference": max((r["abs_difference"] for r in differences), default=0.),
            "absolute_tolerance": 1e-12, "relative_tolerance": 0.,
            "scope": "artifact_only_scenario.requests[i].soc; all_other_fields_types_structure_order_exact; stored_values_used_for_simulation",
            "differing_paths": [r["path"] for r in differences], "differences": differences,
            "exact_match_definition": "parsed_types_and_values_equal; not_bitwise_file_equality"}


def generate_context(env, output, metadata, source_snapshot=None):
    output = Path(output)
    snapshot = visible_snapshot(env)
    snapshot_audit = compare_source_snapshot(env, source_snapshot) if source_snapshot else None
    output.mkdir(parents=True, exist_ok=False)
    begin = perf_counter()
    data = reference_trace(env)
    serialize = 0.
    for name, value in [("snapshot", snapshot)]+[(key, data[key]) for key in ("trace", "features", "labels", "leaves")]:
        tick = perf_counter()
        write_json(output/(name+".json"), value)
        serialize += perf_counter()-tick
    record = {"schema": CONTEXT_SCHEMA, **metadata, "root_time": env.time, "status": "completed",
              "source_commit": os.environ.get("TERMINAL_SOURCE_COMMIT"),
              "time": env.time, "context": str(output),
              "scenario_group": [metadata["seed"], metadata["load"], metadata["soc"]], "prefix": "joint_window",
              "trace_widths": list(TRACE_WIDTHS), "evaluated_reference_width": REFERENCE_WIDTH,
              "feature_schema": FEATURE_SCHEMA, "feature_dim": len(FEATURE_NAMES), "config": asdict(env.config),
              "node_count": len(data["trace"]["nodes"]), "training_node_count": len(data["features"]["rows"]),
              "root_count": len(data["leaves"]["rows"]), "label_scope": LABEL_SCOPE,
              "timing": {**data["timing"], "serialization_seconds": serialize, "context_seconds_before_metadata_write": perf_counter()-begin},
              "work": data["work"], "statistics": data["statistics"],
              "source_snapshot": str(source_snapshot) if source_snapshot else None,
              "source_snapshot_equal": True if source_snapshot else None,
              "source_snapshot_comparison": snapshot_audit,
              "source_scenario_comparison": metadata.get("source_input", {}).get("source_scenario_comparison"),
              "trace_source": "new_instrumented_partial_search; historical_logs_have_no_partial_nodes"}
    record["files"] = {name+".json": file_hash(output/(name+".json")) for name in ("snapshot", "trace", "features", "labels", "leaves")}
    write_json(output/"context.json", record)
    print(json.dumps({"status": "completed", "context": str(output), "nodes": record["node_count"], "roots": record["root_count"]}), flush=True)
    return record


def _source_input(directory, split, seed, load, soc, config, times):
    if directory is None:
        return None
    matches = []
    for path in sorted(Path(directory).rglob("frozen_input.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("split"), row.get("seed"), row.get("load"), row.get("soc")) == (split, seed, load, soc):
            matches.append((path, row))
    if len(matches) != 1:
        raise ValueError("stage4 frozen scenario input missing or duplicated")
    path, row = matches[0]
    from terminal_marl.scripts.evaluate_fixed_terminal_stage5 import compare_continuation_artifact
    compare_continuation_artifact(row["config"], canonical_json(asdict(config)), "$.config", exact=True)
    compare_continuation_artifact(row["times"], list(times), "$.times", exact=True)
    return path, row


def generate_scenario_data(output, split, seed, load, soc, *, source_stage4=None, smoke=False,
                           config=DEFAULT_CONFIG, scenario=None, times=SNAPSHOT_TIMES):
    validate_split(split, seed)
    if split not in ("train", "val", "pilot") or smoke != (split == "pilot"):
        raise ValueError("new training traces only accept train/val or independent pilot smoke")
    times = tuple(float(t) for t in times)
    if not smoke and (config != DEFAULT_CONFIG or times != SNAPSHOT_TIMES or scenario is not None):
        raise ValueError("formal trace geometry, times and generator are frozen")
    if not times or tuple(sorted(set(times))) != times or any(t < 0 or t >= config.horizon or t/config.dt != round(t/config.dt) for t in times):
        raise ValueError("snapshot times must be ordered pre-cutoff ticks")
    source = _source_input(source_stage4, split, seed, load, soc, config, times)
    if source:
        path, row = source
        s = dict(row["scenario"])
        s["requests"] = tuple(Request(**r) for r in s["requests"])
        selected_scenario = Scenario(**s)
        scenario_audit = compare_source_scenario(row["scenario"], canonical_json(asdict(generate_scenario(config, seed, load, soc))))
        source_meta = {"kind": "reused_stage4_frozen_input", "path": str(path), "sha256": file_hash(path),
                       "source_scenario_comparison": scenario_audit}
    else:
        selected_scenario = scenario or generate_scenario(config, seed, load, soc)
        source_meta = {"kind": "reconstructed_frozen_stage4_generator" if not smoke else "independent_pilot_smoke", "path": None}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = FixedTerminalEnv(config, selected_scenario)
    write_json(output/"frozen_input.json", {"schema": "fixed-terminal-stage7-trace-input-v1", "split": split,
        "source_commit": os.environ.get("TERMINAL_SOURCE_COMMIT"), "trace_widths": list(TRACE_WIDTHS),
        "seed": seed, "load": load, "soc": soc, "scenario": asdict(selected_scenario), "config": asdict(config),
        "times": times, "prefix": "joint_window", "source": source_meta, "smoke_only": smoke,
        "source_scenario_comparison": source_meta.get("source_scenario_comparison")})
    next_decision, records = 0., []
    for target in times:
        while env.time < target-1e-9:
            if env.time >= next_decision-1e-9:
                apply_scheduler(env, "joint_window")
                next_decision += 5.
            env.step()
        snapshot = source[0].parent/f"t{target:g}"/"snapshot.json" if source else None
        records.append(generate_context(env, output/f"t{target:g}", {"split": split, "seed": seed,
            "load": load, "soc": soc, "smoke_only": smoke, "source_input": source_meta}, source_snapshot=snapshot))
    write_json(output/"generation.json", {"schema": "fixed-terminal-stage7-generation-v1", "status": "completed",
        "source_commit": os.environ.get("TERMINAL_SOURCE_COMMIT"), "trace_widths": list(TRACE_WIDTHS),
        "contexts": records, "split": split, "seed": seed, "load": load, "soc": soc,
        "scenario": asdict(selected_scenario), "config": asdict(config), "times": times,
        "prefix_decisions_before_snapshot": True, "source": source_meta, "smoke_only": smoke,
        "source_scenario_comparison": source_meta.get("source_scenario_comparison")})
    return records


def load_contexts(directory, expected_split, *, training=False):
    if training and expected_split not in ("train", "val", "pilot"):
        raise ValueError("training cannot read test traces")
    contexts = []
    for path in sorted(Path(directory).rglob("context.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["schema"] != CONTEXT_SCHEMA or metadata["split"] != expected_split or metadata["status"] != "completed":
            raise ValueError("context schema/status/split mismatch")
        if metadata["trace_widths"] != list(TRACE_WIDTHS):
            raise ValueError("partial trace budget coverage mismatch")
        validate_split(metadata["split"], metadata["seed"])
        for name, digest in metadata["files"].items():
            if file_hash(path.parent/name) != digest:
                raise ValueError("trace dependency hash mismatch")
        features = json.loads((path.parent/"features.json").read_text(encoding="utf-8"))
        labels = json.loads((path.parent/"labels.json").read_text(encoding="utf-8"))
        if features["schema"] != FEATURE_SCHEMA or features["feature_names"] != list(FEATURE_NAMES) or labels["scope"] != LABEL_SCOPE:
            raise ValueError("feature/label schema mismatch")
        by_id = {r["node_id"]: r for r in labels["rows"]}
        ids = [r["node_id"] for r in features["rows"]]
        if len(by_id) != len(labels["rows"]) or len(ids) != len(set(ids)) or set(ids) != {k for k, v in by_id.items() if v["eligible_expand_training"]}:
            raise ValueError("node feature/label alignment mismatch")
        contexts.append({"metadata": metadata, "path": str(path.parent), "context_hash": file_hash(path),
                         "features": [r["values"] for r in features["rows"]],
                         "targets": [by_id[key]["expand_label"] for key in ids], "node_ids": ids})
    if not contexts:
        raise ValueError("no completed contexts")
    keys = [tuple(c["metadata"]["scenario_group"])+ (c["metadata"]["root_time"],) for c in contexts]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate context")
    return contexts


def analyze_contexts(directory):
    depths, totals = {}, Counter()
    timing, contexts = Counter(), []
    for path in sorted(Path(directory).rglob("context.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        trace = json.loads((path.parent/"trace.json").read_text(encoding="utf-8"))
        labels = json.loads((path.parent/"labels.json").read_text(encoding="utf-8"))
        leaves = json.loads((path.parent/"leaves.json").read_text(encoding="utf-8"))["rows"]
        by_id = {r["node_id"]: r for r in labels["rows"]}
        totals["contexts"] += 1
        totals["retained_evaluated_roots"] += meta["root_count"]
        totals["unlinked_best_anchors"] += len(labels["unlinked_best_anchors"])
        local_depths = {}
        for node in trace["nodes"]:
            d = depths.setdefault(str(node["depth"]), Counter())
            label = by_id[node["node_id"]]
            d["nodes"] += 1
            d["positive"] += label["expand_label"]
            d["negative_noncontribution"] += 1-label["expand_label"]
            d["eligible_training"] += label["eligible_expand_training"]
            if label["eligible_expand_training"]:
                d["training_positive"] += label["expand_label"]
                d["training_negative_noncontribution"] += 1-label["expand_label"]
                totals["training_positive"] += label["expand_label"]
                totals["training_negative_noncontribution"] += 1-label["expand_label"]
            local = local_depths.setdefault(str(node["depth"]), Counter())
            local["positive_contribution_branches"] += label["expand_label"]
            local["nodes"] += 1
            d["linked_to_evaluated_leaf"] += bool(label["descendant_evaluated_root_ids"])
            d["check_cache_hits"] += node["check_cache_hit"]
            result = node["validation"]
            if result:
                d["validated_nodes"] += 1
                d["actual_batch_checks"] += not node["check_cache_hit"]
                d["validated_noncontribution"] += not label["expand_label"]
                d["actual_check_noncontribution"] += not node["check_cache_hit"] and not label["expand_label"]
                d["cached_revisit_noncontribution"] += node["check_cache_hit"] and not label["expand_label"]
                d["unknown_validation"] += result["status"] == "unknown"
        for name, amount in meta["work"].items():
            if isinstance(amount, int):
                totals[name] += amount
        for name, seconds in meta["timing"].items():
            if isinstance(seconds, (int, float)):
                timing[name] += seconds
        contexts.append({"path": str(path.parent), "scenario_group": meta["scenario_group"], "time": meta["root_time"],
                         "retained_leaves": len(leaves), "distinct_leaf_J_keys": len({value_comparison_key(r["evaluation"]["value"]) for r in leaves}),
                         "best_retained_leaves": len(labels["best_root_ids"]), "best_retained_fraction": len(labels["best_root_ids"])/len(leaves),
                         "depths": {key: dict(row) for key, row in local_depths.items()}})
    if not totals["contexts"]:
        raise ValueError("no contexts for analysis")
    return {"schema": "fixed-terminal-stage7-bottleneck-v1", "totals": dict(totals),
            "timing_seconds": dict(timing), "contexts": contexts,
            "training_positive_fraction": totals["training_positive"]/(totals["training_positive"]+totals["training_negative_noncontribution"]) if totals["training_positive"]+totals["training_negative_noncontribution"] else None,
            "depths": {d: {**dict(row), "validated_noncontribution_ratio": row["validated_noncontribution"]/row["validated_nodes"] if row["validated_nodes"] else None}
                       for d, row in depths.items()}, "label_scope": LABEL_SCOPE,
            "waste_scope": "observed_validated_noncontribution_to_retained_best_leaf; includes_unknown_and_cached_revisits_separately; not_proven_counterfactual_waste"}
