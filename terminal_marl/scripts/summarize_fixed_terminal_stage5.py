"""Summarize completed stage5 artifacts without training or simulation."""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import median

from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs.fixed_terminal_stage5 import ONLINE_ARMS, scenario_jobs
from terminal_marl.core.fixed_terminal.reference import value_comparison_key
from terminal_marl.scripts.evaluate_fixed_terminal_stage4 import timing_summary

ARMS = tuple(a["name"] for a in ONLINE_ARMS)
COUNT_COMPONENTS = {"N": "normal_landed", "E": "emergency_landed", "F": "failed_energy", "C": "arrived_unresolved"}
WORK_FIELDS = ("root_rollout_count", "simulation_steps", "root_validation_calls", "continuation_submission_calls", "final_submit_count")
SEARCH_FIELDS = ("generated", "examined", "joint_assignments", "lookahead_states")
DENOMINATOR_NOTE = ("all combines27 standard and3 initial_rotated scenarios per arm; medium/medium occurs as3 standard and3 rotated scenarios. "
                    "Separate standard and initial_rotated groups are provided. Group rows overlap and are not additive across axes.")


class IncompleteEvidence(ValueError):
    def __init__(self, missing):
        self.missing = sorted(set(missing))
        super().__init__("incomplete stage5 evidence: "+json.dumps(self.missing))


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON constant: "+value)
    with Path(path).open(encoding="utf-8-sig") as stream:
        result = json.load(stream, parse_constant=invalid)
    require_finite(result)
    return result


def require_finite(value):
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("nonfinite evidence value")
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            require_finite(item)


def rows_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                require_finite(row)
                yield row


def scenario_key(row):
    return row["seed"], row["load"], row["soc"], row["variant"]


def scenario_fields(result):
    scenario = result["scenario"]
    return {"seed": scenario["seed"], "load": scenario["load"], "soc": scenario["soc_profile"], "variant": result["variant"]}


def expected_paths():
    required, expected = [], {"episodes": set(), "contexts": set(), "panels": set(), "continuations": set()}
    for job in scenario_jobs("test"):
        stem = f"{job['load']}-{job['soc']}-{job['seed']}-{job['variant']}"
        key = scenario_key(job)
        required.append(f"snapshots/{stem}/generation.json")
        for arm in ARMS:
            expected["episodes"].add((arm,)+key)
            required.extend(f"episodes/{arm}-{stem}/{name}" for name in ("result.json", "decisions.jsonl", "frozen_input.json"))
        profile = (job["load"], job["soc"]) in (("low", "high"), ("medium", "medium"), ("high", "low"))
        for time in (0., 15., 90.):
            expected["contexts"].add(key+(time, 14))
            required.extend(f"snapshots/{stem}/t{time:g}/{name}.json" for name in ("context", "budget_curves", "labels"))
            if profile and time in (0., 15.):
                expected["panels"].add(key+(time,))
                required.append(f"snapshots/{stem}/t{time:g}/timing_panel/summary.json")
        if profile and job["variant"] == "standard":
            expected["contexts"].add(key+(15., 32))
            required.extend(f"snapshots/{stem}/large_pool/t15/{name}.json" for name in ("context", "budget_curves", "labels"))
    for load in ("low", "medium", "high"):
        for soc in ("high", "medium", "low"):
            expected["continuations"].add((load, soc, 4303, 15.))
            required.append(f"continuation/{load}-{soc}-4303/result.json")
    required += ["execution_test.json", "execution_continuation.json"]
    return required, expected


def executor_evidence(root, allow_incomplete):
    result = {}
    for phase, count in (("test", 360), ("continuation", 9)):
        path = root/f"execution_{phase}.json"
        if not path.exists():
            if allow_incomplete:
                result[phase] = {"source": str(path), "status": "missing_in_fixture_or_pilot"}
                continue
            raise IncompleteEvidence([str(path)])
        state = read_json(path)
        names = [r["name"] for r in state["jobs"]]
        if len(names) != len(set(names)):
            raise ValueError("duplicate dispatcher job names")
        if (state.get("schema") != "fixed-terminal-stage5-dispatch-v1" or state["status"] != "completed" or
                state.get("exit_code") != 0 or any(j["status"] != "completed" or j["returncode"] != 0 for j in state["jobs"]) or
                state.get("model_hashes_unchanged") is not True):
            raise IncompleteEvidence([f"execution_{phase}: not completed with all zero exits and unchanged model hashes"])
        if not allow_incomplete and (len(names) != count or state["counts"] != {"total": count, "pending": 0, "running": 0, "completed": count, "failed": 0}):
            raise IncompleteEvidence([f"execution_{phase}: expected {count} completed jobs"])
        if not allow_incomplete:
            if phase == "test":
                expected_names = set()
                for job in scenario_jobs("test"):
                    stem = f"{job['load']}-{job['soc']}-{job['seed']}-{job['variant']}"
                    expected_names.add("snapshot-"+stem)
                    expected_names.update(arm+"-"+stem for arm in ARMS)
            else:
                expected_names = {f"continuation-{load}-{soc}-4303" for load in ("low", "medium", "high") for soc in ("high", "medium", "low")}
            if set(names) != expected_names:
                raise IncompleteEvidence([f"execution_{phase}: job identities differ from frozen matrix"])
        if state["frozen_models_before"] != state["frozen_models_after"]:
            raise ValueError("dispatcher before/after model records differ")
        result[phase] = {"source": str(path), **{k: state.get(k) for k in
            ("schema", "status", "exit_code", "counts", "source_commit", "started_utc", "finished_utc", "duration_seconds",
             "workers", "peak_running", "thread_env", "frozen_models_before", "frozen_models_after", "model_hashes_unchanged", "model_check_scope")},
            "job_returncode_counts": dict(Counter(str(j["returncode"]) for j in state["jobs"]))}
    return result


def search_totals(decisions):
    result = {name: sum(row["statistics"][name] for row in decisions) for name in SEARCH_FIELDS}
    reasons = Counter()
    for row in decisions:
        reasons.update(row["statistics"]["rejection_reasons"])
    result.update({"truncated_decisions": sum(row["statistics"]["truncated"] for row in decisions),
                   "rejection_reasons": dict(sorted(reasons.items())),
                   "counter_scope": "sum_of_recorded_operations_including_rollout_continuation; not_unique_candidates"})
    return result


def load_episodes(root, allow_incomplete):
    records, seen, configurations = [], set(), {}
    for path in sorted((root/"episodes").glob("*/result.json")):
        result = read_json(path)
        if result.get("schema") != "fixed-terminal-stage5-episode-v1":
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
        decisions = [{k: row[k] for k in ("time", "timing", "degenerate", "statistics", "work", "pool_count")}
                     for row in rows_jsonl(path.parent/"decisions.jsonl")]
        times = [r["time"] for r in decisions]
        if len(times) != len(set(times)):
            raise ValueError("duplicate decision time")
        if not allow_incomplete and times != list(range(0, 240, 5)):
            raise ValueError("formal episode decision cadence incomplete")
        timings = timing_summary(decisions)
        if timings != result["decision_timings"]:
            raise ValueError("decision timing aggregate mismatch")
        work = {name: sum(row["work"][name] for row in decisions) for name in WORK_FIELDS}
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
        delay = components["normal_delay_sum_seconds"]
        if counts["N"] and not isclose(summary["normal_delay_mean_seconds"], delay/counts["N"], abs_tol=1e-8):
            raise ValueError("conditional normal delay mismatch")
        records.append({"arm": arm, **fields, **counts, "not_arrived": not_arrived,
                        "request_count": summary["request_count"], "normal_delay_mean_seconds": delay/counts["N"] if counts["N"] else None,
                        "normal_delay_denominator": counts["N"], "normal_delay_sum_seconds": delay,
                        "exposure_seconds": components["observed_request_seconds"], "energy_soc": components["energy_consumed_soc_all_requests"],
                        "spacing_violations": summary["separation_violation_pair_ticks"], "spacing_uncertified": summary["spacing_uncertified_pair_ticks"],
                        "minimum_landed_soc": summary["minimum_landed_soc"], "timings": timings, "work": work,
                        "search": search_totals(decisions), "pool_count_sum": sum(r["pool_count"] for r in decisions),
                        "pool_count_min": min((r["pool_count"] for r in decisions), default=None),
                        "pool_count_max": max((r["pool_count"] for r in decisions), default=None),
                        "decisions": len(decisions), "degenerate_decisions": sum(r["degenerate"] is True for r in decisions),
                        "model_load_seconds": result["model_load_seconds"], "environment_step_seconds": result["environment_step_seconds"],
                        "final_value": value, "source": str(path), "source_commit": result.get("source_commit"), "models": result["models"],
                        "_decisions": decisions})
    return records, seen


def group_rows(rows):
    """Nonexclusive all/variant scopes, each with all/load/SOC group axes."""
    buckets = defaultdict(list)
    for row in rows:
        for scope in ("all", row["variant"]):
            for axis, value in (("all", "all"), ("load", row["load"]), ("soc", row["soc"])):
                buckets[scope, axis, value].append(row)
    return buckets


def aggregate_episodes(rows):
    grouped = []
    for arm in sorted({r["arm"] for r in rows}):
        for (scope, axis, name), selected in sorted(group_rows([r for r in rows if r["arm"] == arm]).items()):
            counts = {key: sum(r[key] for r in selected) for key in ("N", "E", "F", "C", "not_arrived", "request_count")}
            delay = sum(r["normal_delay_sum_seconds"] for r in selected)
            decisions = [d for r in selected for d in r["_decisions"]]
            minimums = [r["minimum_landed_soc"] for r in selected if r["minimum_landed_soc"] is not None]
            grouped.append({"arm": arm, "scope": scope, "axis": axis, "group": name, "episodes": len(selected), **counts,
                            "normal_delay_sum_seconds": delay, "normal_delay_denominator": counts["N"],
                            "normal_delay_weighted_mean_seconds": delay/counts["N"] if counts["N"] else None,
                            "normal_proportion_all_requests": counts["N"]/counts["request_count"] if counts["request_count"] else None,
                            "emergency_proportion_all_requests": counts["E"]/counts["request_count"] if counts["request_count"] else None,
                            "exposure_seconds": sum(r["exposure_seconds"] for r in selected), "energy_soc": sum(r["energy_soc"] for r in selected),
                            "spacing_violations": sum(r["spacing_violations"] for r in selected), "spacing_uncertified": sum(r["spacing_uncertified"] for r in selected),
                            "minimum_landed_soc": min(minimums, default=None), "timings": timing_summary(decisions),
                            "work": {k: sum(r["work"][k] for r in selected) for k in WORK_FIELDS}, "search": search_totals(decisions),
                            "pool_count_sum": sum(r["pool_count_sum"] for r in selected), "decisions": len(decisions),
                            "degenerate_decisions": sum(r["degenerate_decisions"] for r in selected),
                            "model_load_seconds_sum": sum(sum(r["model_load_seconds"].values()) for r in selected),
                            "environment_step_seconds": sum(r["environment_step_seconds"] for r in selected)})
    return grouped


def paired_episodes(rows):
    by_key = {(r["arm"],)+scenario_key(r): r for r in rows}
    output = []
    for row in rows:
        for reference in ("beam_b5", "full_reference"):
            other = by_key.get((reference,)+scenario_key(row))
            if other is None:
                continue
            a, b = value_comparison_key(row["final_value"]), value_comparison_key(other["final_value"])
            output.append({"arm": row["arm"], "reference": reference, **{k: row[k] for k in ("seed", "load", "soc", "variant")},
                           "comparison": "win" if a > b else "loss" if a < b else "tie",
                           "raw_J_delta_arm_minus_reference": [x-y for x, y in zip(row["final_value"]["J"], other["final_value"]["J"])],
                           "raw_component_delta_arm_minus_reference": {k: row["final_value"]["components"][k]-other["final_value"]["components"][k] for k in row["final_value"]["components"]},
                           "arm_final_value": row["final_value"], "reference_final_value": other["final_value"]})
    counts = []
    for method, reference in sorted({(r["arm"], r["reference"]) for r in output}):
        for (scope, axis, name), selected in sorted(group_rows([r for r in output if r["arm"] == method and r["reference"] == reference]).items()):
            counts.append({"arm": method, "reference": reference, "scope": scope, "axis": axis, "group": name, "paired_scenarios": len(selected),
                           **{k: sum(r["comparison"] == k for r in selected) for k in ("win", "tie", "loss")}})
    return {"rows": output, "counts": counts, "comparison": "saved_episode_final_J; four_ints_then_T_D_round8_energy_round10"}


def load_snapshots(root, allow_incomplete):
    contexts, curve_rows, panels, seen, panel_seen = [], [], [], set(), set()
    generations = []
    for path in sorted((root/"snapshots").glob("*/generation.json")):
        generation = read_json(path)
        if generation["status"] != "completed":
            raise IncompleteEvidence([str(path)+": generation not completed"])
        generations.append({"source": str(path), "contexts": len(generation["contexts"]), "models": generation.get("models"),
                            "model_load_seconds": generation.get("model_load_seconds", {})})
    for path in sorted((root/"snapshots").rglob("context.json")):
        context = read_json(path)
        if context.get("schema") != "fixed-terminal-stage5-context-v1" or context["status"] != "completed":
            raise ValueError("unexpected/incomplete context: "+str(path))
        if allow_incomplete and context.get("smoke_only") is not True:
            raise ValueError("--allow-incomplete context is not explicit fixture/pilot")
        fields = {k: context[k] for k in ("seed", "load", "soc", "variant", "root_time", "pool_limit")}
        key = scenario_key(fields)+(context["root_time"], context["pool_limit"])
        if key in seen:
            raise ValueError("duplicate snapshot identity")
        seen.add(key)
        label_rows = read_json(path.parent/"labels.json")["rows"]
        labels = {r["root_id"]: r["value"] for r in label_rows}
        curves = read_json(path.parent/"budget_curves.json")
        if len(labels) != context["root_count"] or len(labels) != len(label_rows):
            raise ValueError("duplicate/missing root labels")
        best = max(value_comparison_key(v) for v in labels.values())
        if value_comparison_key(curves["full_best_value"]) != best:
            raise ValueError("snapshot full reference label not maximal")
        contexts.append({**fields, "root_count": len(labels), "informative": curves["informative"], "degenerate": curves["degenerate"], "source": str(path),
                         "full_best_root_id": curves["full_best_root_id"], "source_commit": context.get("source_commit")})
        combinations = set()
        for curve in curves["rows"]:
            identity = (curve["method"], curve["budget"])
            if identity in combinations:
                raise ValueError("duplicate offline curve method/budget")
            combinations.add(identity)
            selected_ids = curve["selected_root_ids"]
            chosen = labels[curve["chosen_root_id"]]
            full = curves["full_best_value"]
            regret = [x-y for x, y in zip(full["J"], chosen["J"])]
            if (len(selected_ids) != len(set(selected_ids)) or len(selected_ids) > curve["budget"] or
                    chosen != curve["chosen_value"] or regret != curve["raw_J_regret_full_minus_chosen"] or
                    curve["best_key_retained"] != (value_comparison_key(chosen) == best)):
                raise ValueError("offline curve selection/value mismatch")
            curve_rows.append({**fields, "method": curve["method"], "budget": curve["budget"], "root_count": len(labels),
                               "informative": curves["informative"], "degenerate": curves["degenerate"], "best_key_retained": curve["best_key_retained"],
                               "would_evaluate_root_count": curve["would_evaluate_root_count"], "chosen_root_id": curve["chosen_root_id"],
                               "raw_J_regret_full_minus_chosen": regret,
                               "chosen_minus_full_counts": {short: chosen["components"][name]-full["components"][name] for short, name in COUNT_COMPONENTS.items()},
                               "chosen_value": chosen, "full_value": full, "source": str(path.parent/"budget_curves.json")})
        if not allow_incomplete and combinations != {(m, b) for m in ("beam", "linear_rank", "mlp_rank", "gradient_boost", "value", "hybrid") for b in (3, 5, 8)}:
            raise ValueError("formal offline curves missing methods/budgets")
        panel_path = path.parent/"timing_panel"/"summary.json"
        if panel_path.exists():
            panel = read_json(panel_path)
            panel_key = scenario_key(fields)+(fields["root_time"],)
            if panel_key in panel_seen:
                raise ValueError("duplicate timing panel")
            panel_seen.add(panel_key)
            arm_names = [r["arm"] for r in panel["rows"]]
            if len(arm_names) != len(set(arm_names)) or (not allow_incomplete and set(arm_names) != set(ARMS)):
                raise ValueError("duplicate/missing timing panel arms")
            if panel["identical_snapshot_and_pool"] is not True or panel["cross_method_cache"] is not False:
                raise ValueError("timing panel inputs/cache contract mismatch")
            reference = next((r["timing"]["total"] for r in panel["rows"] if r["arm"] == "beam_b5"), None)
            for row in panel["rows"]:
                value = labels[row["chosen_root_id"]]
                panels.append({**fields, **row, "chosen_closed_value": value,
                               "ratio_total_to_beam_b5": row["timing"]["total"]/reference if reference is not None and reference > 0 else None,
                               "selection_objective": "rolling_commitment_surrogate" if row["arm"].startswith("rolling") else "closed_actual_J",
                               "chosen_value_scope": "same_context_saved_closed_label_lookup; does_not_change_rolling_selection_objective",
                               "source": str(panel_path)})
    return contexts, curve_rows, panels, seen, panel_seen, generations


def aggregate_curves(rows):
    output = []
    for pool, method, budget in sorted({(r["pool_limit"], r["method"], r["budget"]) for r in rows}):
        subset = [r for r in rows if (r["pool_limit"], r["method"], r["budget"]) == (pool, method, budget)]
        for stratum in ("all", "informative", "nondegenerate"):
            selected = [r for r in subset if stratum == "all" or (r["informative"] if stratum == "informative" else not r["degenerate"])]
            for (scope, axis, name), group in sorted(group_rows(selected).items()):
                hits = sum(r["best_key_retained"] for r in group)
                output.append({"pool_limit": pool, "method": method, "budget": budget, "stratum": stratum,
                               "scope": scope, "axis": axis, "group": name, "contexts": len(group),
                               "best_key_hits": hits, "retention": hits/len(group),
                               "would_evaluate_root_count_sum": sum(r["would_evaluate_root_count"] for r in group),
                               "raw_J_regret_sum_full_minus_chosen": [sum(r["raw_J_regret_full_minus_chosen"][i] for r in group) for i in range(7)],
                               "chosen_minus_full_counts_sum": {k: sum(r["chosen_minus_full_counts"][k] for r in group) for k in COUNT_COMPONENTS},
                               "timing_scope": "cached_label_quality_only_not_measured_runtime"})
    return output


def numeric_summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "sum": 0., "median": None, "p95": None, "max": None}
    index = .95*(len(values)-1)
    low = int(index)
    high = min(low+1, len(values)-1)
    return {"count": len(values), "sum": sum(values), "median": median(values),
            "p95": values[low]+(values[high]-values[low])*(index-low), "max": max(values)}


def aggregate_panels(rows):
    output = []
    for arm in sorted({r["arm"] for r in rows}):
        for stratum in ("all", "nondegenerate", "degenerate"):
            selected = [r for r in rows if r["arm"] == arm and (stratum == "all" or r["degenerate"] == (stratum == "degenerate"))]
            fields = sorted({k for r in selected for k in r["timing"]})
            output.append({"arm": arm, "stratum": stratum, "states": len(selected),
                           "timing_seconds": {k: numeric_summary([r["timing"][k] for r in selected]) for k in fields},
                           "total_over5_seconds_count": sum(r["timing"]["total"] > 5 for r in selected),
                           "work": {k: sum(r["work"][k] for r in selected) for k in WORK_FIELDS},
                           "paired_ratio_total_to_beam_b5": numeric_summary([r["ratio_total_to_beam_b5"] for r in selected if r["ratio_total_to_beam_b5"] is not None]),
                           "scope": "real_same_state_independent_execution; model_loading_excluded; separate_from_online_trajectory"})
    return output


def load_continuation(root, allow_incomplete):
    rows, seen = [], set()
    for path in sorted((root/"continuation").glob("*/result.json")):
        result = read_json(path)
        if result.get("schema") != "fixed-terminal-stage5-continuation-diagnostic-v1":
            raise ValueError("unexpected continuation schema")
        if allow_incomplete and result.get("fixture_only") is not True:
            raise ValueError("--allow-incomplete continuation must be explicitly fixture_only")
        key = tuple(result["scenario"][k] for k in ("load", "soc", "seed", "time"))
        if key in seen:
            raise ValueError("duplicate continuation identity")
        seen.add(key)
        rows.append({"source": str(path), "source_commit": result.get("source_commit"), "scenario": result["scenario"],
                     "root_count": len(result["rows"]), "pair_count": len(result["pairs"]),
                     "order_changed_count": sum(r["order_changed"] for r in result["pairs"]),
                     "strict_reversal_count": sum(r["strict_reversal"] for r in result["pairs"]),
                     "selected_by": result["selected_by"], "values": result["rows"], "pairs": result["pairs"],
                     "interpretation": "two_worlds_have_different_request_denominators; completion_difference_is_not_a_policy_gain"})
    return rows, seen


def summarize(root, allow_incomplete=False):
    root = Path(root)
    required, expected = expected_paths()
    missing = [name for name in required if not (root/name).is_file()]
    if missing and not allow_incomplete:
        raise IncompleteEvidence(missing)
    execution = executor_evidence(root, allow_incomplete)
    episodes, episode_keys = load_episodes(root, allow_incomplete)
    contexts, curves, panels, context_keys, panel_keys, generations = load_snapshots(root, allow_incomplete)
    continuation, continuation_keys = load_continuation(root, allow_incomplete)
    actual = {"episodes": episode_keys, "contexts": context_keys, "panels": panel_keys, "continuations": continuation_keys}
    if not allow_incomplete:
        mismatches = [f"{name}: missing={sorted(expected[name]-actual[name])}; unexpected={sorted(actual[name]-expected[name])}" for name in expected if expected[name] != actual[name]]
        if len(generations) != 30:
            mismatches.append(f"generations: {len(generations)} instead of30")
        if mismatches:
            raise IncompleteEvidence(mismatches)
        model_record = execution["test"]["frozen_models_before"]
        if not model_record or model_record["manifest"]["smoke_only"] is not False:
            raise ValueError("formal model manifest missing or smoke-only")
        if any(r["models"] != model_record for r in episodes) or any(r["models"] != model_record for r in generations):
            raise ValueError("models differ across formal input artifacts")
    episode_groups = aggregate_episodes(episodes)
    paired = paired_episodes(episodes)
    result = {"schema": "fixed-terminal-stage5-summary-v1", "status": "fixture_or_pilot_incomplete" if allow_incomplete else "completed",
              "root": str(root), "allow_incomplete": allow_incomplete, "missing_formal_paths": missing,
              "counts": {"episodes": len(episodes), "generations": len(generations), "base_pool_contexts": sum(r["pool_limit"] == 14 for r in contexts),
                         "large_pool_contexts": sum(r["pool_limit"] == 32 for r in contexts), "timing_panels": len(panel_keys),
                         "timing_rows": len(panels), "continuations": len(continuation)},
              "denominator_note": DENOMINATOR_NOTE,
              "runtime_note": "online_trajectory_and_same_state_costs_are_separate; concurrent_wall_times_are_not_real_time_guarantees; no_significance_claim",
              "execution": execution, "source_commits_observed": sorted({r["source_commit"] for r in episodes+contexts+continuation if r["source_commit"] is not None}),
              "episode_rows": [{k: v for k, v in r.items() if k not in ("_decisions", "models")} for r in episodes],
              "episode_groups": episode_groups, "paired_online": paired,
              "offline": {"contexts": contexts, "rows": curves, "groups": aggregate_curves(curves), "pool14_and_pool32_never_pooled": True,
                          "scenario_generation_metadata": [{k: v for k, v in row.items() if k != "models"} for row in generations]},
              "same_state": {"rows": panels, "groups": aggregate_panels(panels)}, "continuation": continuation}
    require_finite(result)
    return result


def flat_row(row, prefix=""):
    out = {}
    for key, value in row.items():
        name = prefix+key
        if isinstance(value, dict):
            out.update(flat_row(value, name+"."))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(value, allow_nan=False, separators=(",", ":"))
        else:
            out[name] = value
    return out


def write_outputs(output, result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/"summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    tables = {"episode_rows": result["episode_rows"], "episode_groups": result["episode_groups"],
              "paired_online": result["paired_online"]["rows"], "paired_counts": result["paired_online"]["counts"],
              "offline_rows": result["offline"]["rows"], "offline_groups": result["offline"]["groups"],
              "same_state_rows": result["same_state"]["rows"], "same_state_groups": result["same_state"]["groups"],
              "continuation": result["continuation"]}
    for name, rows in tables.items():
        flat = [flat_row(row) for row in rows]
        fields = sorted({key for row in flat for key in row})
        with (output/(name+".csv")).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="explicit fixture/pilot artifacts only")
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
