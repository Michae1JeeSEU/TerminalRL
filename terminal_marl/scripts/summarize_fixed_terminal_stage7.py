"""Strict stage7 quality/compute summary; reads artifacts, never reruns a model."""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import json
from math import isclose
from pathlib import Path

from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs.fixed_terminal_stage7 import ARMS, scenario_jobs, panel_scenario_jobs, PANEL_TIMES, SNAPSHOT_TIMES, TRACE_WIDTHS, COMPUTE_REDUCTION_TARGET
from hashlib import sha256
from terminal_marl.core.fixed_terminal.reference import value_comparison_key
from terminal_marl.scripts.evaluate_fixed_terminal_stage4 import timing_summary
from terminal_marl.scripts.summarize_fixed_terminal_stage5 import (
    read_json, require_finite, rows_jsonl, scenario_key, scenario_fields,
    COUNT_COMPONENTS, WORK_FIELDS, SEARCH_FIELDS, search_totals, group_rows,
    aggregate_episodes as _aggregate_episodes, flat_row, numeric_summary,
)

LEARNING_ARMS = ("learned_pruning", "learned_beam_width", "hybrid")
REFERENCES = ("beam_b3", "beam_b5", "beam_b8", "full_reference", "beam_adaptive")
SUMMARY_COUNTS = dict(episodes=264, generations=15, timing_panels=30, timing_rows=240)


class IncompleteEvidence(ValueError):
    def __init__(self, missing):
        self.missing = sorted(set(missing))
        super().__init__("incomplete stage7 evidence: " + json.dumps(self.missing))


def stem(job):
    return f"{job['load']}-{job['soc']}-{job['seed']}-{job['variant']}"


def expected_paths():
    paths = ["execution_data.json", "execution_episodes.json", "execution_panels.json"]
    paths.extend("models/" + name for name in ("manifest.json", "protocol.json", "training.json", "gradient_boost_expand.joblib"))
    for split in ("train", "val"):
        for job in scenario_jobs(split):
            base = f"data/{split}/{job['load']}-{job['soc']}-{job['seed']}"
            paths.extend(base+"/"+name for name in ("generation.json", "frozen_input.json"))
            paths.extend(base+f"/t{time:g}/{name}.json" for time in SNAPSHOT_TIMES
                         for name in ("snapshot", "context", "trace", "features", "labels", "leaves"))
    expected = {"episodes": set(), "panels": set(), "generations": set()}
    for job in scenario_jobs("test"):
        for arm in ARMS:
            expected["episodes"].add((arm,)+scenario_key(job))
            paths.extend(f"episodes/{arm}-{stem(job)}/{name}" for name in
                         ("result.json", "decisions.jsonl", "events.jsonl", "requests.jsonl", "frozen_input.json"))
    for job in panel_scenario_jobs("test"):
        expected["generations"].add(scenario_key(job))
        paths.extend(f"panels/{stem(job)}/{name}.json" for name in ("generation", "frozen_input"))
        for time in PANEL_TIMES:
            expected["panels"].add(scenario_key(job)+(time,))
            paths.extend(f"panels/{stem(job)}/t{time:g}/{name}.json" for name in
                         (*ARMS, "snapshot", "summary", "full_values"))
    return paths, expected


def validate_stages(row):
    """Validate cumulative exact cache and recorded partial expansion evidence."""
    search = row["search"]
    if not isinstance(search.get("nodes"), list) or not isinstance(search.get("stages"), list):
        raise ValueError("partial search traces absent")
    nodes = search["nodes"]
    ids = [n["node_id"] for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate partial node identity")
    node_map = {n["node_id"]: n for n in nodes}
    for node in nodes:
        parent = node.get("parent_id")
        if parent is not None and (parent not in node_map or node_map[parent]["depth"] + 1 != node["depth"]):
            raise ValueError("partial node parent/depth mismatch")
        if node["depth"] < 0 or node["candidate_count"] < 0:
            raise ValueError("negative partial depth/candidate count")
    seen = set()
    adaptive = row["adaptive"]
    for stage in adaptive["stages"]:
        new = stage["new_root_ids"]
        if len(set(new)) != len(new) or seen.intersection(new):
            raise ValueError("duplicate cached exact root evaluation")
        seen.update(new)
        if stage["cumulative_count"] != len(seen):
            raise ValueError("cumulative exact work mismatch")
        if stage["chosen_root_id"] not in seen:
            raise ValueError("stage choice lacks exact evaluation")
        certificate = stage.get("certificate")
        if certificate is not None:
            a, f, e = (certificate[k] for k in ("A", "F0", "E0"))
            passed = certificate["F"] == f and certificate["E"] == e and certificate["N"] == a-f-e and certificate["C"] == 0
            if certificate["passed"] != passed or sum(certificate[k] for k in ("N", "E", "F", "C")) != a:
                raise ValueError("invalid observed completion certificate")
    if len(seen) != row["work"]["root_rollout_count"] or row["budget"] != len(seen):
        raise ValueError("cumulative unique exact work/budget mismatch")
    if set(row["evaluations"]) != seen:
        raise ValueError("exact rollout records differ from stages")
    terminal_certificate = adaptive["stages"][-1]["certificate"] if adaptive["stages"] else None
    if adaptive["quality_certified"] != (terminal_certificate["passed"] if terminal_certificate else None):
        raise ValueError("terminal certificate mismatch")
    if row["pool_count"] != len(row["pool"]) or len({r["root_id"] for r in row["pool"]}) != row["pool_count"] or not seen <= {r["root_id"] for r in row["pool"]}:
        raise ValueError("method cumulative pool inventory mismatch")
    if row["degenerate"] != (row["pool_count"] == 1):
        raise ValueError("pool/degeneracy mismatch")
    if row["chosen_root_id"] not in {r["root_id"] for r in row["pool"]}:
        raise ValueError("chosen root outside method pool")
    if row["final_check"]["status"] != "feasible":
        raise ValueError("final independent execution validation failed")
    require_finite(row)


def adaptive_totals(rows):
    nondeg = [r for r in rows if not r["degenerate"]]
    enabled = [r for r in nondeg if r["adaptive"]["enabled"]]
    ratio = lambda n, d: n/d if d else None
    return {"decisions": len(rows), "nondegenerate_decisions": len(nondeg),
            "adaptive_nondegenerate_decisions": len(enabled),
            "stop_reasons": dict(Counter(r["adaptive"]["stop_reason"] for r in rows)),
            "fallback_count": sum(r["adaptive"]["fallback_count"] for r in rows),
            "fallback_rate": ratio(sum(r["adaptive"]["fallback_count"] > 0 for r in enabled), len(enabled)),
            "full_restoration_rate": ratio(sum(any(s.get("width") == 128 for s in r["adaptive"]["stages"][1:]) for r in enabled), len(enabled)),
            "partial_width_counts": dict(Counter(str(width) for r in rows for stage in r["adaptive"]["stages"] for width in stage.get("actual_widths", []))),
            "quality_certificate_rate": ratio(sum(r["adaptive"]["quality_certified"] is True for r in nondeg), len(nondeg)),
            "unique_exact_root_evaluations": sum(r["work"]["root_rollout_count"] for r in rows),
            "pool_count": numeric_summary([r["pool_count"] for r in rows]),
            "search_counter_totals": sum_numeric([r["search"]["counters"] for r in rows]),
            "work_totals": sum_numeric([r["work"] for r in rows]),
            "timing_summary": timing_summary(rows),
            "method_pool_single_root_decisions": sum(r["pool_count"] == 1 for r in rows)}


def sum_numeric(rows):
    fields = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return {k: sum(r.get(k, 0) for r in rows) for k in fields}


def search_summary(trace):
    """Keep measured search metadata; detailed nodes and links stay in raw files."""
    return {**{key: trace[key] for key in ("schema", "counters", "catalogue_holder_count",
             "catalogue_candidate_counts", "counter_scope", "reference_scope", "stages")},
            "node_count": len(trace["nodes"])}


def decision_summary(row):
    return {**{key: row[key] for key in ("time", "timing", "work", "statistics", "degenerate",
             "pool_count", "budget", "adaptive", "final_check")}, "search": search_summary(row["search"])}


def load_episodes(root, allow_incomplete):
    records, seen, configurations = [], set(), {}
    for path in sorted((root/"episodes").glob("*/result.json")):
        result = read_json(path)
        if result.get("schema") != "fixed-terminal-stage7-episode-v1":
            raise ValueError("unexpected episode schema: "+str(path))
        if allow_incomplete and result.get("smoke_only") is not True:
            raise ValueError("--allow-incomplete accepts only explicit fixture/pilot artifacts")
        fields = scenario_fields(result)
        arm = result["scheduler"]
        identity = (arm,)+scenario_key(fields)
        if identity in seen:
            raise ValueError("duplicate episode identity: "+repr(identity))
        seen.add(identity)
        frozen = read_json(path.parent/"frozen_input.json")
        if (frozen["scenario"] != result["scenario"] or frozen["config"] != result["config"] or
                frozen["models"] != result["models"] or frozen["arm"] != result["arm"] or frozen["variant"] != result["variant"]):
            raise ValueError("episode frozen input/result mismatch")
        config_scenario = {"config": frozen["config"], "scenario": frozen["scenario"]}
        key = scenario_key(fields)
        if key in configurations and config_scenario != configurations[key]:
            raise ValueError("unpaired scenario/config across methods")
        configurations[key] = config_scenario
        if not allow_incomplete and (result.get("smoke_only") is not False or frozen["config"] != json.loads(json.dumps(asdict(DEFAULT_CONFIG)))):
            raise ValueError("formal episode differs from frozen physical configuration")
        decisions = []
        for raw_decision in rows_jsonl(path.parent/"decisions.jsonl"):
            # Validate the full original record before dropping detailed trace objects.
            validate_stages(raw_decision)
            decisions.append(decision_summary(raw_decision))
        if result.get("model_hashes_unchanged") is not True or result.get("models_after") != result["models"]:
            raise ValueError("episode frozen model hashes changed")
        if result.get("request_conservation") is not True or result["summary"].get("conservation_ok") is not True:
            raise ValueError("request conservation failed")
        requests = list(rows_jsonl(path.parent/"requests.jsonl"))
        if requests != result["requests"] or len(requests) != result["summary"]["request_count"] or len({r["aircraft_id"] for r in requests}) != len(requests):
            raise ValueError("request records/count/identity mismatch")
        observed_outcomes = Counter(r["outcome"] for r in requests)
        if any(observed_outcomes[k] != v for k, v in result["summary"]["outcomes"].items()):
            raise ValueError("request outcome mismatch")
        times = [r["time"] for r in decisions]
        if len(times) != len(set(times)):
            raise ValueError("duplicate decision time")
        if not allow_incomplete and times != list(range(0, 240, 5)):
            raise ValueError("formal episode decision cadence incomplete")
        timings = timing_summary(decisions)
        if timings != result["decision_timings"]:
            raise ValueError("decision timing aggregate mismatch")
        work = sum_numeric([row["work"] for row in decisions])
        if work != result["work"]:
            raise ValueError("decision work aggregate mismatch")
        value, summary = result["final_value"], result["summary"]
        components = value["components"]
        counts = {short: components[name] for short, name in COUNT_COMPONENTS.items()}
        summary_counts = {"N": summary["outcomes"]["normal_landed"], "E": summary["outcomes"]["emergency_landed"],
                          "F": summary["outcomes"]["failed_energy"], "C": summary["outcomes"]["censored"]}
        not_arrived = summary["outcomes"]["not_arrived"]
        if counts != summary_counts or sum(counts.values())+not_arrived != summary["request_count"] or value["request_denominator"] != summary["request_count"]:
            raise ValueError("request/value conservation mismatch")
        if not allow_incomplete and (not_arrived or value["time"] != 240. or value["episode_cumulative"] is not True):
            raise ValueError("formal episode final value is incomplete")
        observed_counts = Counter(r["outcome"] for r in requests)
        if any(observed_counts.get(name, 0) != count for name, count in summary["outcomes"].items()):
            raise ValueError("request rows/outcome counts mismatch")
        if len(result["scenario"]["requests"]) != summary["request_count"]:
            raise ValueError("scenario/request denominator mismatch")
        raw_components = {
            "normal_delay_sum_seconds": sum(r["delay_seconds"] for r in requests if r["outcome"] == "normal_landed"),
            "observed_request_seconds": sum(r["observed_request_seconds"] for r in requests),
            "energy_consumed_soc_all_requests": sum(r["energy_consumed_soc"] for r in requests)}
        if any(not isclose(components[k], amount, abs_tol=1e-8, rel_tol=1e-10) for k, amount in raw_components.items()):
            raise ValueError("raw request/J component mismatch")
        actual_j = [-counts["F"], counts["N"], -counts["E"], -counts["C"],
                    -components["observed_request_seconds"], -components["normal_delay_sum_seconds"], -components["energy_consumed_soc_all_requests"]]
        if len(value["J"]) != 7 or any(not isclose(a, b, abs_tol=1e-8, rel_tol=1e-10) for a, b in zip(value["J"], actual_j)):
            raise ValueError("J/components mismatch")
        delay = components["normal_delay_sum_seconds"]
        if counts["N"] and not isclose(summary["normal_delay_mean_seconds"], delay/counts["N"], abs_tol=1e-8):
            raise ValueError("conditional normal delay mismatch")
        records.append({"arm": arm, **fields, **counts, "not_arrived": not_arrived,
                        "request_count": summary["request_count"], "normal_delay_mean_seconds": delay/counts["N"] if counts["N"] else None,
                        "normal_delay_denominator": counts["N"], "normal_delay_sum_seconds": delay,
                        "normal_exposure_seconds": sum(r["observed_request_seconds"] for r in requests if r["outcome"] == "normal_landed"),
                        "normal_energy_soc": sum(r["energy_consumed_soc"] for r in requests if r["outcome"] == "normal_landed"),
                        "exposure_seconds": components["observed_request_seconds"], "energy_soc": components["energy_consumed_soc_all_requests"],
                        "spacing_violations": summary["separation_violation_pair_ticks"], "spacing_uncertified": summary["spacing_uncertified_pair_ticks"],
                        "minimum_landed_soc": summary["minimum_landed_soc"], "timings": timings, "work": work,
                        "search": search_totals(decisions), "pool_count_sum": sum(r["pool_count"] for r in decisions),
                        "pool_count_min": min((r["pool_count"] for r in decisions), default=None),
                        "pool_count_max": max((r["pool_count"] for r in decisions), default=None),
                        "decisions": len(decisions), "degenerate_decisions": sum(r["degenerate"] is True for r in decisions),
                        "decision_event_io_seconds": result["decision_event_io_seconds"], "final_execution_checks_passed": True, "adaptive": adaptive_totals(decisions), "model_load_seconds": result["model_load_seconds"], "environment_step_seconds": result["environment_step_seconds"],
                        "final_value": value, "source": str(path), "source_commit": result.get("source_commit"), "models": result["models"],
                        "_decisions": decisions})
    return records, seen


def executor_evidence(root, allow_incomplete=False):
    evidence = {}
    for phase, count in (("data", 36), ("episodes", 264), ("panels", 15)):
        path = root/f"execution_{phase}.json"
        if not path.exists() and allow_incomplete:
            continue
        state = read_json(path)
        jobs = state["jobs"]
        names = [job["name"] for job in jobs]
        expected = ({f"{arm}-{stem(job)}" for job in scenario_jobs("test") for arm in ARMS} if phase == "episodes"
                    else {"panels-"+stem(job) for job in panel_scenario_jobs("test")} if phase == "panels"
                    else {f"{split}-{job['load']}-{job['soc']}-{job['seed']}" for split in ("train", "val") for job in scenario_jobs(split)})
        if (state.get("schema") != "fixed-terminal-stage7-dispatch-v1" or state.get("status") != "completed"
                or state.get("exit_code") != 0 or state.get("model_hashes_unchanged") is not True
                or state.get("frozen_models_before") != state.get("frozen_models_after")
                or len(names) != len(set(names)) or any(j["status"] != "completed" or j["returncode"] != 0 for j in jobs)):
            raise IncompleteEvidence([f"execution_{phase}: status/exit/hash/unique jobs mismatch"])
        if not allow_incomplete and (set(names) != expected or state["counts"] != dict(total=count, pending=0, running=0, completed=count, failed=0)):
            raise IncompleteEvidence([f"execution_{phase}: frozen task matrix mismatch"])
        if state.get("thread_env") != {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}:
            raise ValueError("dispatcher numerical thread limits mismatch")
        import re
        if re.fullmatch(r"[0-9a-f]{40}", state.get("source_commit") or "") is None:
            raise ValueError("formal dispatcher source commit missing or malformed")
        if phase == "panels" and state["workers"] != 1:
            raise ValueError("same-state panels must use one worker")
        evidence[phase] = {k: state.get(k) for k in ("schema", "status", "exit_code", "counts", "source_commit",
            "started_utc", "finished_utc", "duration_seconds", "workers", "thread_env", "frozen_models_before", "frozen_models_after", "model_hashes_unchanged")}
    return evidence


def load_panels(root, allow_incomplete=False):
    rows, keys, generation_keys, generations = [], set(), set(), []
    for path in sorted((root/"panels").glob("*/generation.json")):
        generation = read_json(path)
        if generation.get("schema") != "fixed-terminal-stage7-generation-v1" or generation["status"] != "completed":
            raise ValueError("incomplete stage7 panel generation")
        if generation.get("smoke_only") is not allow_incomplete:
            raise ValueError("panel fixture/pilot versus formal marker mismatch")
        frozen = read_json(path.parent/"frozen_input.json")
        if any(frozen[k] != generation[k] for k in ("config", "scenario", "models", "variant", "times")):
            raise ValueError("panel generation/frozen input mismatch")
        if generation["models"] != generation["models_after"] or generation["model_hashes_unchanged"] is not True:
            raise ValueError("panel model hash mismatch")
        if not allow_incomplete and (generation["config"] != json.loads(json.dumps(asdict(DEFAULT_CONFIG))) or generation["times"] != list(PANEL_TIMES)):
            raise ValueError("formal panel configuration/time mismatch")
        fields = scenario_fields(generation)
        generation_key = scenario_key(fields)
        if generation_key in generation_keys:
            raise ValueError("duplicate generation scenario")
        generation_keys.add(generation_key)
        generations.append(generation)
        for context in generation["contexts"]:
            key = generation_key+(context["time"],)
            if key in keys or context["status"] != "completed":
                raise ValueError("duplicate/incomplete panel")
            keys.add(key)
            directory = path.parent/f"t{context['time']:g}"
            panel = read_json(directory/"summary.json")
            if (panel["schema"] != "fixed-terminal-stage7-panel-v1" or panel["identical_snapshot_and_catalogue_domain"] is not True
                    or panel["cross_method_cache"] is not False or panel["original_environment_unchanged"] is not True):
                raise ValueError("panel execution conditions mismatch")
            names = [r["arm"] for r in panel["rows"]]
            if len(names) != len(set(names)) or set(names) != set(ARMS):
                raise ValueError("panel arm inventory mismatch")
            values = read_json(directory/"full_values.json")["values"]
            labels = read_json(directory/"full_values.json")
            full_ids = labels["full_pool_root_ids"]
            full_best = max(value_comparison_key(values[rid]) for rid in full_ids)
            full_raw = read_json(directory / "full_reference.json")
            if set(full_ids) != {r["root_id"] for r in full_raw["pool"]}:
                raise ValueError("finite reference root inventory mismatch")
            for row in panel["rows"]:
                raw = read_json(directory/(row["arm"]+".json"))
                validate_stages(raw)
                if (raw["search"]["catalogue_holder_count"], raw["search"]["catalogue_candidate_counts"]) != (full_raw["search"]["catalogue_holder_count"], full_raw["search"]["catalogue_candidate_counts"]):
                    raise ValueError("panel catalogue domain mismatch")
                if raw["final_check"]["status"] != "feasible":
                    raise ValueError("panel final execution check failed")
                if any(raw[k] != row[k] for k in ("arm", "timing", "work", "degenerate", "pool_count", "budget", "adaptive", "chosen_root_id", "search")):
                    raise ValueError("panel summary/raw mismatch")
                from terminal_marl.scripts.evaluate_fixed_terminal_stage7 import scientific_value
                if row["chosen_root_id"] in raw["evaluations"] and scientific_value(raw["evaluations"][row["chosen_root_id"]]["value"]) != scientific_value(row["chosen_value"]):
                    raise ValueError("panel raw chosen scientific value mismatch")
                if row["chosen_value"] != values[row["chosen_root_id"]] or value_comparison_key(row["full_best_value"]) != full_best:
                    raise ValueError("panel full value annotation mismatch")
                if row["best_key_retained"] != (value_comparison_key(row["chosen_value"]) >= full_best):
                    raise ValueError("panel retention mismatch")
                if row["chosen_in_full_pool"] != (row["chosen_root_id"] in full_ids) or row["new_value_above_finite_reference"] != (value_comparison_key(row["chosen_value"]) > full_best):
                    raise ValueError("new-root finite reference annotation mismatch")
                rows.append({**fields, "time": context["time"], **row, "search": search_summary(raw["search"]),
                             "analysis_label_cost": panel["analysis_label_cost"], "source": str(directory)})
    return rows, keys, generation_keys, generations


def training_data_evidence(root, allow_incomplete=False):
    """Recheck frozen train/val trace dependencies and split identities."""
    result = {"train": [], "val": []}
    if allow_incomplete and not (root / "data").exists():
        return {"contexts": {"train": 0, "val": 0}, "fixture_only": True}
    from terminal_marl.core.fixed_terminal.search_data import load_contexts
    generations = set()
    for split in ("train", "val"):
        loaded = load_contexts(root / "data" / split, split, training=True)
        expected = {(j["seed"], j["load"], j["soc"], time) for j in scenario_jobs(split) for time in SNAPSHOT_TIMES}
        observed = set()
        for record in loaded:
            meta = record["metadata"]
            key = (meta["seed"], meta["load"], meta["soc"], meta["root_time"])
            if key in observed:
                raise ValueError("duplicate training context identity")
            observed.add(key)
            if not allow_incomplete and (meta.get("smoke_only") is not False or meta.get("source_snapshot_equal") is not True or meta.get("trace_widths") != list(TRACE_WIDTHS) or meta.get("source_commit") != read_json(root / "execution_data.json")["source_commit"]):
                raise ValueError("formal training context is smoke-only")
            result[split].append({"seed": key[0], "load": key[1], "soc": key[2], "time": key[3],
                                  "context_hash": record["context_hash"], "nodes": len(record["node_ids"])})
        if not allow_incomplete and observed != expected:
            raise IncompleteEvidence([split + ": training context matrix differs"])
        for path in sorted((root / "data" / split).glob("*/generation.json")):
            generation = read_json(path)
            frozen = read_json(path.parent / "frozen_input.json")
            identity = (split, generation["seed"], generation["load"], generation["soc"])
            if identity in generations:
                raise ValueError("duplicate training generation")
            generations.add(identity)
            if (generation["status"] != "completed" or generation["split"] != split or
                    any(generation[k] != frozen[k] for k in ("scenario", "config", "times", "split", "seed", "load", "soc"))):
                raise ValueError("training generation/frozen input mismatch")
            contexts = generation["contexts"]
            if len(contexts) != len({c["time"] for c in contexts}) or {c["time"] for c in contexts} != set(generation["times"]):
                raise ValueError("training generation context coverage mismatch")
            for context in contexts:
                saved_context = read_json(path.parent / f"t{context['time']:g}" / "context.json")
                if saved_context != context:
                    raise ValueError("training generation/context record mismatch")
            if not allow_incomplete and (generation["times"] != list(SNAPSHOT_TIMES) or generation["config"] != json.loads(json.dumps(asdict(DEFAULT_CONFIG))) or generation["source"]["kind"] != "reused_stage4_frozen_input" or generation.get("trace_widths") != list(TRACE_WIDTHS) or generation.get("source_commit") != read_json(root / "execution_data.json")["source_commit"]):
                raise ValueError("formal training source/config/times mismatch")
    expected_generations = {(split, j["seed"], j["load"], j["soc"]) for split in ("train", "val") for j in scenario_jobs(split)}
    if not allow_incomplete and generations != expected_generations:
        raise IncompleteEvidence(["training generations matrix differs"])
    return {"contexts": {split: len(result[split]) for split in result}, "generation_count": len(generations),
            "splits": result, "test_read": False, "scope": "new_partial_traces_from_frozen_stage4_train_val"}


def model_training_evidence(root, data):
    """Bind the sole fit to the exact raw context identities and bytes."""
    from terminal_marl.scripts.evaluate_fixed_terminal_stage7 import frozen_models
    current = frozen_models(root / "models", smoke=False)
    protocol = read_json(root / "models" / "protocol.json")
    training = read_json(root / "models" / "training.json")
    manifest = current["manifest"]
    metrics = manifest["metrics"]["train"]
    one_class = metrics["positive"] == 0 or metrics["negative"] == 0
    if (protocol.get("fit_count") != 1 or protocol.get("test_read") is not False or protocol.get("smoke_only") is not False or
            protocol.get("training") != manifest["training"] or protocol.get("threshold") != manifest["threshold"] or
            training.get("status") != "completed" or training.get("protocol") != protocol or
            training.get("model_class") != manifest["model_class"] or training.get("one_class_dummy") is not one_class or
            protocol.get("trace_widths") != list(TRACE_WIDTHS) or
            training.get("roundtrip_predictions_equal") is not True or training.get("metrics") != manifest["metrics"]):
        raise ValueError("frozen single-fit training protocol mismatch")
    for split in ("train", "val"):
        actual = {(r["seed"], r["load"], r["soc"], r["time"]): r["context_hash"] for r in data["splits"][split]}
        recorded = {(r["scenario_group"][0], r["scenario_group"][1], r["scenario_group"][2], r["time"]): r["context_sha256"] for r in protocol[split]}
        if len(recorded) != len(protocol[split]) or recorded != actual:
            raise ValueError("frozen model training context identity/hash mismatch: " + split)
    for phase in ("episodes", "panels"):
        state = read_json(root / f"execution_{phase}.json")
        for key in ("frozen_models_before", "frozen_models_after"):
            recorded = state[key]
            if recorded["manifest_sha256"] != current["manifest_sha256"] or recorded["manifest"] != current["manifest"]:
                raise ValueError("current model dependencies differ from formal evaluation freeze")
    return {"fit_count": 1, "training": manifest["training"], "threshold": manifest["threshold"], "splits": manifest["splits"],
            "context_hashes_match": True, "roundtrip_predictions_equal": True, "test_read": False,
            "model_class": manifest["model_class"], "one_class_dummy": one_class,
            "trace_widths": list(TRACE_WIDTHS), "metrics": manifest["metrics"], "fit_seconds": training["fit_seconds"],
            "manifest_sha256": current["manifest_sha256"]}


def aggregate_episodes(rows):
    result = _aggregate_episodes(rows)
    for group in result:
        selected = [r for r in rows if r["arm"] == group["arm"]
                    and (group["scope"] == "all" or r["variant"] == group["scope"])
                    and (group["axis"] == "all" or r[group["axis"]] == group["group"])]
        group.update({"all_work_totals": sum_numeric([r["work"] for r in selected]),
                      "decision_event_io_seconds": sum(r["decision_event_io_seconds"] for r in selected),
                      "normal_exposure_seconds": sum(r["normal_exposure_seconds"] for r in selected),
                      "normal_energy_soc": sum(r["normal_energy_soc"] for r in selected),
                      "J_component_sums": [sum(r["final_value"]["J"][i] for r in selected) for i in range(7)],
                      "all_final_execution_checks_passed": all(r["final_execution_checks_passed"] for r in selected)})
        for name in ("normal_exposure_seconds", "normal_energy_soc"):
            group[name+"_mean_conditional_normal"] = group[name]/group["N"] if group["N"] else None
        for label in ("F", "C"):
            group[label+"_proportion_all_requests"] = group[label]/group["request_count"] if group["request_count"] else None
    return result


def paired_episodes(rows):
    by_key = {(r["arm"],)+scenario_key(r): r for r in rows}
    pairs, conditions = [], []
    for arm in LEARNING_ARMS:
        for reference in REFERENCES:
            matched = []
            for row in (r for r in rows if r["arm"] == arm):
                other = by_key.get((reference,)+scenario_key(row))
                if other is None:
                    continue
                if row["request_count"] != other["request_count"]:
                    raise ValueError("paired episode request denominator differs")
                a, b = value_comparison_key(row["final_value"]), value_comparison_key(other["final_value"])
                pair = {"arm": arm, "reference": reference, **{k: row[k] for k in ("seed", "load", "soc", "variant")},
                        "comparison": "win" if a > b else "loss" if a < b else "tie",
                        "request_count": row["request_count"], "arm_N": row["N"], "reference_N": other["N"],
                        "arm_E": row["E"], "reference_E": other["E"], "F_increased": row["F"] > other["F"],
                        "all_safety_checks_passed": all(r["spacing_violations"] == r["spacing_uncertified"] == 0 and r["final_execution_checks_passed"] for r in (row, other)),
                        "raw_J_delta_arm_minus_reference": [x-y for x, y in zip(row["final_value"]["J"], other["final_value"]["J"])],
                        "arm_final_value": row["final_value"], "reference_final_value": other["final_value"]}
                pairs.append(pair)
                matched.append(pair)
            for (scope, axis, name), selected in sorted(group_rows(matched).items()):
                denominator = sum(r["request_count"] for r in selected)
                n, ref_n = sum(r["arm_N"] for r in selected), sum(r["reference_N"] for r in selected)
                e, ref_e = sum(r["arm_E"] for r in selected), sum(r["reference_E"] for r in selected)
                conditions.append({"arm": arm, "reference": reference, "scope": scope, "axis": axis, "group": name,
                    "paired_scenarios": len(selected), "request_count": denominator,
                    "arm_normal_proportion": n/denominator if denominator else None,
                    "reference_normal_proportion": ref_n/denominator if denominator else None,
                    "epsilon_normal": 0., "normal_condition": n >= ref_n if denominator else None,
                    "failed_scenario_increases": sum(r["F_increased"] for r in selected),
                    "emergency_condition": e <= ref_e, "all_safety_checks_passed": all(r["all_safety_checks_passed"] for r in selected),
                    "quality_condition_passed": bool(denominator and n >= ref_n and e <= ref_e and not any(r["F_increased"] for r in selected) and all(r["all_safety_checks_passed"] for r in selected)),
                    **{k: sum(r["comparison"] == k for r in selected) for k in ("win", "tie", "loss")}})
    return {"rows": pairs, "conditions": conditions}


def same_state_comparisons(rows):
    by_key = {(r["arm"],)+scenario_key(r)+(r["time"],): r for r in rows}
    pairs, groups = [], []
    threshold = 1. - COMPUTE_REDUCTION_TARGET
    for row in (r for r in rows if r["arm"] in LEARNING_ARMS):
        for reference in REFERENCES:
            other = by_key.get((reference,)+scenario_key(row)+(row["time"],))
            if other is None:
                continue
            # Partial controls can produce different root pools on the same snapshot.
            # Strata use the finite traditional reference's degeneracy consistently.
            reference_degenerate = by_key[("full_reference",)+scenario_key(row)+(row["time"],)]["degenerate"]
            a, b = row["timing"]["total"], other["timing"]["total"]
            x, y = row["work"]["root_rollout_count"], other["work"]["root_rollout_count"]
            checks, ref_checks = (r["search"]["counters"]["candidate_checks"] for r in (row, other))
            expanded, ref_expanded = (r["search"]["counters"]["expanded_nodes"] for r in (row, other))
            primary_passed = bool(not reference_degenerate and b > 0 and ref_checks > 0 and a <= threshold*b and checks <= threshold*ref_checks)
            pairs.append({"arm": row["arm"], "reference": reference, **{k: row[k] for k in ("seed", "load", "soc", "variant", "time")},
                          "degenerate": reference_degenerate, "method_pool_single_root": row["pool_count"] == 1, "arm_pool_count": row["pool_count"], "reference_pool_count": other["pool_count"],
                          "arm_seconds": a, "reference_seconds": b, "arm_rollouts": x, "reference_rollouts": y,
                          "wall_ratio": a/b if b else None, "rollout_ratio": x/y if y else None,
                          "arm_unique_rollouts": x, "reference_unique_rollouts": y, "unique_rollout_ratio": x/y if y else None,
                          "arm_candidate_checks": checks, "reference_candidate_checks": ref_checks,
                          "candidate_check_ratio": checks/ref_checks if ref_checks else None,
                          "arm_expanded_nodes": expanded, "reference_expanded_nodes": ref_expanded,
                          "expanded_node_ratio": expanded/ref_expanded if ref_expanded else None,
                          "primary_compute_condition_passed": primary_passed, "compute_condition_passed": primary_passed,
                          "rollout_wall_condition_passed": bool(b > 0 and y > 0 and a <= threshold*b and x <= threshold*y),
                          "threshold": threshold, "descriptive_only": True,
                          "arm_search_counters": row["search"]["counters"], "reference_search_counters": other["search"]["counters"],
                          "arm_work": row["work"], "reference_work": other["work"],
                          "raw_J_delta_arm_minus_reference": [u-v for u, v in zip(row["chosen_value"]["J"], other["chosen_value"]["J"])]})
    for arm in LEARNING_ARMS:
        for reference in REFERENCES:
            for stratum in ("all", "nondegenerate", "degenerate"):
                chosen = [r for r in pairs if r["arm"] == arm and r["reference"] == reference and (stratum == "all" or r["degenerate"] == (stratum == "degenerate"))]
                for (scope, axis, name), selected in sorted(group_rows(chosen).items()):
                    totals = {k: sum(r[k] for r in selected) for k in ("arm_seconds", "reference_seconds", "arm_rollouts", "reference_rollouts", "arm_candidate_checks", "reference_candidate_checks", "arm_expanded_nodes", "reference_expanded_nodes")}
                    a, b, x, y = (totals[k] for k in ("arm_seconds", "reference_seconds", "arm_rollouts", "reference_rollouts"))
                    checks, ref_checks = (totals[k] for k in ("arm_candidate_checks", "reference_candidate_checks"))
                    expanded, ref_expanded = (totals[k] for k in ("arm_expanded_nodes", "reference_expanded_nodes"))
                    primary_passed = bool(stratum == "nondegenerate" and b > 0 and ref_checks > 0 and a <= threshold*b and checks <= threshold*ref_checks)
                    groups.append({"arm": arm, "reference": reference, "stratum": stratum, "scope": scope, "axis": axis, "group": name,
                                   "states": len(selected), **totals, "wall_ratio": a/b if b else None, "rollout_ratio": x/y if y else None,
                                   "primary_compute_condition_passed": primary_passed, "compute_condition_passed": primary_passed,
                                   "primary_compute_scope": "common_full_reference_nondegenerate_states; actual_candidate_checks_AND_complete_wall",
                                   "rollout_wall_condition_passed": bool(b > 0 and y > 0 and a <= threshold*b and x <= threshold*y),
                                   "arm_unique_rollouts": x, "reference_unique_rollouts": y, "unique_rollout_ratio": x/y if y else None,
                                   "candidate_check_ratio": checks/ref_checks if ref_checks else None,
                                   "expanded_node_ratio": expanded/ref_expanded if ref_expanded else None,
                                   "arm_unique_rollouts_mean": x/len(selected) if selected else None,
                                   "reference_unique_rollouts_mean": y/len(selected) if selected else None,
                                   "arm_candidate_checks_mean": checks/len(selected) if selected else None,
                                   "reference_candidate_checks_mean": ref_checks/len(selected) if selected else None,
                                   "arm_expanded_nodes_mean": expanded/len(selected) if selected else None,
                                   "reference_expanded_nodes_mean": ref_expanded/len(selected) if selected else None,
                                   "arm_seconds_mean": a/len(selected) if selected else None,
                                   "reference_seconds_mean": b/len(selected) if selected else None,
                                   "arm_over_5s_proportion": sum(r["arm_seconds"] > 5 for r in selected)/len(selected) if selected else None,
                                   "reference_over_5s_proportion": sum(r["reference_seconds"] > 5 for r in selected)/len(selected) if selected else None,
                                   "arm_search_counter_totals": sum_numeric([r["arm_search_counters"] for r in selected]),
                                   "reference_search_counter_totals": sum_numeric([r["reference_search_counters"] for r in selected]),
                                   "arm_work_totals": sum_numeric([r["arm_work"] for r in selected]),
                                   "reference_work_totals": sum_numeric([r["reference_work"] for r in selected]),
                                   "threshold": threshold, "descriptive_only": True})
    return {"pairs": pairs, "groups": groups}


def summarize(root, allow_incomplete=False):
    root = Path(root)
    paths, expected = expected_paths()
    missing = [p for p in paths if not (root/p).is_file()]
    if missing and not allow_incomplete:
        raise IncompleteEvidence(missing)
    execution = executor_evidence(root, allow_incomplete)
    episodes, episode_keys = load_episodes(root, allow_incomplete)
    panels, panel_keys, generation_keys, generations = load_panels(root, allow_incomplete)
    actual = dict(episodes=episode_keys, panels=panel_keys, generations=generation_keys)
    if not allow_incomplete:
        mismatch = [f"{k}: matrix identities differ" for k in expected if expected[k] != actual[k]]
        if mismatch:
            raise IncompleteEvidence(mismatch)
        model_record = execution["episodes"]["frozen_models_before"]
        if not model_record or model_record["manifest"]["smoke_only"] is not False:
            raise ValueError("formal models missing or smoke-only")
        if any(r.get("source_commit") != execution["episodes"]["source_commit"] for r in episodes) or any(r.get("source_commit") != execution["panels"]["source_commit"] for r in generations):
            raise ValueError("formal generation/episode source differs from dispatcher")
        if execution["panels"]["frozen_models_before"] != model_record or any(r["models"] != model_record for r in episodes+generations):
            raise ValueError("model hashes differ across formal artifacts")
        if data_state := execution.get("data"):
            if data_state["finished_utc"] > execution["episodes"]["started_utc"] or data_state["finished_utc"] > execution["panels"]["started_utc"]:
                raise ValueError("data and evaluation phases overlap or are out of order")
        # The phases must be serial to keep same-state CPU timing isolated.
        a, b = execution["episodes"], execution["panels"]
        if not (a["finished_utc"] <= b["started_utc"] or b["finished_utc"] <= a["started_utc"]):
            raise ValueError("episode and panel phases overlap")
    data = training_data_evidence(root, allow_incomplete)
    training = model_training_evidence(root, data) if not allow_incomplete else {"fixture_only": True}
    groups = aggregate_episodes(episodes)
    fallback = []
    for source, rows in (("episode", episodes), ("same_state", panels)):
        for arm in ARMS:
            chosen = [r for r in rows if r["arm"] == arm]
            for (scope, axis, name), selected in sorted(group_rows(chosen).items()):
                decisions = [d for r in selected for d in r["_decisions"]] if source == "episode" else selected
                fallback.append({"source": source, "arm": arm, "scope": scope, "axis": axis, "group": name, **adaptive_totals(decisions)})
    result = {"schema": "fixed-terminal-stage7-summary-v1", "status": "fixture_or_pilot_incomplete" if allow_incomplete else "completed",
        "root": str(root), "allow_incomplete": allow_incomplete, "missing_formal_paths": missing,
        "counts": dict(episodes=len(episodes), generations=len(generations), timing_panels=len(panel_keys), timing_rows=len(panels)),
        "denominator_note": "N/E/F/C all requests; delay conditional on N; all/standard/sparse/dense and load/SOC groups overlap",
        "runtime_note": "same-state independent decisions versus distinct online trajectories; cold load/IO/env steps separate; no significance or real-time claim",
        "execution": execution, "training_data": data, "training": training, "episode_rows": [{k:v for k,v in r.items() if k not in ("_decisions", "models")} for r in episodes],
        "episode_groups": groups, "paired_online": paired_episodes(episodes),
        "same_state": {"rows": panels, **same_state_comparisons(panels)}, "fallback": fallback,
        "source_commits_observed": sorted({r["source_commit"] for r in episodes+generations if r.get("source_commit")}),
        "finite_reference_contribution": {"rows": [{k: r[k] for k in ("arm", "seed", "load", "soc", "variant", "time", "chosen_in_full_pool", "new_value_above_finite_reference", "best_key_retained")} for r in panels],
            "new_value_above_finite_reference_counts": dict(Counter(r["arm"] for r in panels if r["new_value_above_finite_reference"])),
            "scope": "finite_reference_pool; not_global_value_or_long_term_Q"},
        "interpretation": "A/B/C deferred to primary review; learning contribution requires beam_adaptive comparison"}
    require_finite(result)
    return result


def write_outputs(output, result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tables = {"episodes": result["episode_rows"], "groups": result["episode_groups"],
              "paired": result["paired_online"]["rows"], "same_state": result["same_state"]["pairs"], "fallback": result["fallback"]}
    for name, rows in tables.items():
        flat = [flat_row(r) for r in rows]
        with (output/(name+".csv")).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for row in flat for k in row}))
            writer.writeheader()
            writer.writerows(flat)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        result = summarize(args.root, args.allow_incomplete)
    except IncompleteEvidence as exc:
        print(json.dumps({"status": "incomplete", "missing": exc.missing}))
        return 2
    write_outputs(args.output, result)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
