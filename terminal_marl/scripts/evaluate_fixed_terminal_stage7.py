"""Frozen stage7 episodes and isolated same-state timing panels."""
import os
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter

from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG, SOC_PROFILES
from terminal_marl.core.fixed_terminal.baselines import apply_scheduler
from terminal_marl.core.fixed_terminal.domain import Intent
from terminal_marl.core.fixed_terminal.learning_data import write_json
from terminal_marl.core.fixed_terminal.reference import Root, value_comparison_key
from terminal_marl.core.fixed_terminal.scenarios import generate_scenario, stream_seed, truncated_normal
from terminal_marl.core.fixed_terminal.validation_scheduler import closed_evaluation
from terminal_marl.core.fixed_terminal.value_analysis import visible_snapshot, canonical_json, system_value
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.scripts.evaluate_fixed_terminal_stage4 import timing_summary
from hashlib import sha256
import re


def _config():
    from terminal_marl.configs import fixed_terminal_stage7
    return fixed_terminal_stage7


def _arm(name):
    for arm in _config().ONLINE_ARMS:
        if arm["name"] == name:
            return arm
    raise ValueError("unknown stage7 arm: " + str(name))


LEARNING_ARMS = ("learned_pruning", "learned_beam_width", "hybrid")


def scorer_name(arm_name):
    _arm(arm_name)
    return "gradient_boost_expand" if arm_name in LEARNING_ARMS else None


def frozen_models(directory, smoke=False):
    """Hash every manifest dependency; formal evaluation requires a frozen model."""
    if directory is None:
        if smoke:
            return None
        raise ValueError("formal evaluation requires the stage7 frozen model")
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    from terminal_marl.scripts.summarize_fixed_terminal_stage5 import read_json
    from terminal_marl.core.fixed_terminal.search_features import FEATURE_NAMES, FEATURE_DIM, SCHEMA
    manifest = read_json(manifest_path)
    if (manifest.get("schema") != "fixed-terminal-stage7-model-manifest-v1" or
            manifest.get("kind") != "expand" or manifest.get("name") != "gradient_boost_expand" or
            manifest.get("feature_schema") != SCHEMA or manifest.get("feature_names") != list(FEATURE_NAMES) or
            manifest.get("feature_dim") != FEATURE_DIM or manifest.get("training") != _config().TRAINING or
            manifest.get("trace_widths") != list(_config().TRACE_WIDTHS) or
            manifest.get("threshold") != _config().EXPAND_THRESHOLD or manifest.get("fit_count") != 1 or
            manifest.get("roundtrip_predictions_equal") is not True or
            manifest.get("status") != "completed" or manifest.get("frozen") is not True or
            manifest.get("test_read") is not False or
            (not smoke and manifest.get("smoke_only") is not False)):
        raise ValueError("model manifest is not a frozen compatible artifact")
    if not smoke:
        if manifest.get("splits") != {"train": list(_config().TRAIN_SEEDS), "val": list(_config().VAL_SEEDS)}:
            raise ValueError("formal model training split mismatch")
        metrics = manifest.get("metrics", {}).get("train", {})
        positive, negative, nodes = (metrics.get(k) for k in ("positive", "negative", "nodes"))
        if any(type(v) is not int or v < 0 for v in (positive, negative, nodes)) or positive + negative != nodes or nodes == 0:
            raise ValueError("formal training class accounting mismatch")
        one_class = positive == 0 or negative == 0
        expected_class = "DummyClassifier" if one_class else "GradientBoostingClassifier"
        if manifest.get("model_class") != expected_class:
            raise ValueError("formal classifier differs from observed training classes")
    if not manifest.get("files") or not {"gradient_boost_expand.joblib", "protocol.json", "training.json"} <= set(manifest["files"]):
        raise ValueError("model manifest dependencies absent")
    for filename, specification in manifest["files"].items():
        if Path(filename).name != filename or any(c in filename for c in ("/", "\\", ":")):
            raise ValueError("unsafe model manifest filename")
        expected = specification["sha256"] if isinstance(specification, dict) else specification
        if sha256((directory / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("model artifact hash mismatch: " + filename)
    return {"directory": str(directory), "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
            "manifest": manifest}


def load_scorers(directory, names, smoke=False):
    provenance = frozen_models(directory, smoke)
    names = tuple(dict.fromkeys(names))
    if any(name != "gradient_boost_expand" for name in names):
        raise ValueError("stage7 uses only its frozen expansion policy")
    if names and directory is None:
        raise ValueError("learned arms require frozen stage7 models")
    scorers, seconds = {}, {}
    if names:
        from terminal_marl.core.fixed_terminal.search_models import FrozenSearchPolicy
        start = perf_counter()
        policy = FrozenSearchPolicy.from_directory(directory)
        seconds[policy.name] = perf_counter() - start
        if policy.name != "gradient_boost_expand" or policy.kind != "expand":
            raise ValueError("frozen policy identity/kind mismatch")
        scorers[policy.name] = policy
    return scorers, seconds, provenance


def _apply(env, arm_name, scorers):
    from terminal_marl.core.fixed_terminal.search_scheduler import apply_search_scheduler
    name = scorer_name(arm_name)
    policy = scorers.get(name) if name else None
    if name and policy is None:
        raise ValueError("learned arm requires frozen expansion policy")
    return apply_search_scheduler(env, arm_name, policy=policy)


from terminal_marl.scripts.evaluate_fixed_terminal_stage6 import scenario_for, scenario_metadata


def validate_run(seed, smoke, config, scenario=None):
    allowed = _config().PILOT_SEEDS if smoke else _config().TEST_SEEDS
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in allowed:
        raise ValueError("seed outside declared pilot/formal group")
    if not smoke and (config != DEFAULT_CONFIG or scenario is not None):
        raise ValueError("formal evaluation requires unchanged defaults and generated scenario")
    if scenario is not None and scenario.seed != seed:
        raise ValueError("fixture scenario seed must match the pilot seed")


def _environment(seed, load, soc, variant, smoke, config, scenario):
    validate_run(seed, smoke, config, scenario)
    jobs = _config().scenario_jobs("pilot" if smoke else "test")
    if not any(all(j[k] == v for k, v in (("seed", seed), ("load", load), ("soc", soc), ("variant", variant))) for j in jobs):
        raise ValueError("scenario outside frozen stage7 matrix")
    if scenario is not None and (scenario.load != load or scenario.soc_profile != soc):
        raise ValueError("fixture profile does not match the requested job")
    return FixedTerminalEnv(config, scenario if scenario is not None else scenario_for(seed, load, soc, variant, config))


def _postcheck(models, smoke, before):
    after = frozen_models(models, smoke=smoke)
    if before != after:
        raise ValueError("frozen model provenance changed during evaluation")
    return after


def source_commit(smoke=False):
    commit = os.environ.get("TERMINAL_SOURCE_COMMIT")
    if not smoke and (commit is None or re.fullmatch(r"[0-9a-f]{40}", commit) is None):
        raise ValueError("formal evaluation requires TERMINAL_SOURCE_COMMIT full SHA")
    return commit


def aggregate_work(rows):
    fields = sorted({key for row in rows for key, value in row["work"].items()
                     if isinstance(value, (int, float)) and not isinstance(value, bool)})
    return {key: sum(row["work"].get(key, 0) for row in rows) for key in fields}


def adaptive_summary(rows):
    adaptive = [r["adaptive"] for r in rows]
    return {"decisions": len(rows), "enabled_decisions": sum(a["enabled"] for a in adaptive),
            "fallback_count": sum(a["fallback_count"] for a in adaptive),
            "decisions_with_fallback": sum(a["fallback_count"] > 0 for a in adaptive),
            "quality_certified_count": sum(a["quality_certified"] is True for a in adaptive),
            "quality_uncertified_count": sum(a["quality_certified"] is False for a in adaptive),
            "quality_not_applicable_count": sum(a["quality_certified"] is None for a in adaptive),
            "stop_reason_counts": dict(Counter(str(a["stop_reason"]) for a in adaptive)),
            "actual_budget_counts": dict(Counter(str(r["budget"]) for r in rows)),
            "stage_count": sum(len(a["stages"]) for a in adaptive)}


def run_episode(output, arm_name, seed=7001, load="low", soc="medium", variant="standard", *,
                models=None, smoke=False, config=DEFAULT_CONFIG, scenario=None):
    arm = _arm(arm_name)
    env = _environment(seed, load, soc, variant, smoke, config, scenario)
    name = scorer_name(arm_name)
    scorers, load_seconds, provenance = load_scorers(models, (name,) if name else (), smoke)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = source_commit(smoke)
    metadata = scenario_metadata(env.scenario, variant, config)
    write_json(output / "frozen_input.json", {"schema": "fixed-terminal-stage7-input-v1", "arm": arm,
               "scenario": asdict(env.scenario), "variant": variant, "scenario_metadata": metadata,
               "config": asdict(config), "models": provenance, "smoke_only": smoke, "source_commit": source,
               "model_load_seconds": load_seconds, "cadence_seconds": 5., "pool_limit": 14,
               "objective": "closed_observed_FCFS_actual_J; deterministic_safety_energy_resources"})
    rows, event_index, next_decision, step_seconds, io_seconds = [], 0, 0., 0., 0.
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as decisions, (output / "events.jsonl").open("w", encoding="utf-8") as events:
        while not env.done:
            if env.time >= next_decision - 1e-9:
                row = _apply(env, arm_name, scorers)
                row["arm"] = arm_name
                rows.append(row)
                tick = perf_counter()
                decisions.write(json.dumps(row, allow_nan=False) + "\n")
                decisions.flush()
                io_seconds += perf_counter() - tick
                next_decision += 5.
            tick = perf_counter()
            env.step()
            step_seconds += perf_counter() - tick
            tick = perf_counter()
            for event in env.events[event_index:]:
                events.write(json.dumps(event, allow_nan=False) + "\n")
            events.flush()
            io_seconds += perf_counter() - tick
            event_index = len(env.events)
    after = _postcheck(models, smoke, provenance)
    result = env.report()
    result.update({"schema": "fixed-terminal-stage7-episode-v1", "scheduler": arm_name, "arm": arm,
                   "variant": variant, "scenario_metadata": metadata, "smoke_only": smoke,
                   "models": provenance, "models_after": after, "model_hashes_unchanged": True,
                   "source_commit": source, "model_load_seconds": load_seconds,
                   "environment_step_seconds": step_seconds, "decision_event_io_seconds": io_seconds,
                   "decision_timings": timing_summary(rows), "final_value": system_value(env),
                   "work": aggregate_work(rows), "adaptive_summary": adaptive_summary(rows)})
    result["assumptions"]["normal_domain"] = "selected_candidates_jointly_committed; unselected_holders_hold; commitments_fixed"
    result["assumptions"]["emergency_domain"] = "normal_domain_infeasibility_certificate_required; UNKNOWN_never_authorizes_emergency"
    requests = result["requests"]
    landed = [r for r in requests if r["outcome"] in ("normal_landed", "emergency_landed")]
    result["summary"]["minimum_landed_soc"] = min((r["minimum_soc"] for r in landed), default=None)
    result["energy_failure_request_ids"] = [r["aircraft_id"] for r in requests if r["outcome"] == "failed_energy"]
    result["request_outcome_counts"] = dict(Counter(r["outcome"] for r in requests))
    result["request_conservation"] = len(requests) == len(env.scenario.requests) and sum(result["request_outcome_counts"].values()) == len(env.scenario.requests)
    if not result["request_conservation"]:
        raise AssertionError("complete request outcome accounting failed")
    write_json(output / "result.json", result)
    with (output / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for request in requests:
            stream.write(json.dumps(request, allow_nan=False) + "\n")
    return result


def _pool_keys(row):
    return sorted(tuple(sorted((p["aircraft_id"], tuple(p["wait_cycles"])) for p in root["plans"]))
                  for root in row["pool"])


def _label_root(env, saved):
    plans = tuple(env.propose(Intent(p["aircraft_id"], tuple(p["wait_cycles"]))) for p in saved["plans"])
    if canonical_json([asdict(p) for p in plans]) != canonical_json(saved["plans"]):
        raise AssertionError("analysis reconstruction mismatch")
    return Root(saved["root_id"], plans, saved["labels"], saved["provenance"], saved["feasibility"])


def scientific_value(value):
    """Exclude separately measured runtime from deterministic scientific values."""
    return {k: v for k, v in value.items() if k != "elapsed_through_stop_seconds"}


def timing_panel(env, output, scorers, rotation_index):
    """Independent partial searches, then exact labels for all chosen/new roots."""
    policy = scorers.get("gradient_boost_expand")
    if policy is None or policy.kind != "expand":
        raise ValueError("panel requires frozen stage7 expansion policy")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arms = [a["name"] for a in _config().ONLINE_ARMS]
    offset = rotation_index % len(arms)
    arms = arms[offset:] + arms[:offset]
    original = canonical_json(visible_snapshot(env))
    write_json(output / "snapshot.json", visible_snapshot(env))
    methods, catalogue_domain = {}, None
    for arm in arms:
        clone = deepcopy(env)
        if canonical_json(visible_snapshot(clone)) != original:
            raise AssertionError("different panel snapshot")
        row = _apply(clone, arm, scorers)
        row["arm"] = arm
        domain = (row["search"]["catalogue_holder_count"], row["search"]["catalogue_candidate_counts"])
        if catalogue_domain is not None and domain != catalogue_domain:
            raise AssertionError("different panel catalogue domain")
        catalogue_domain = domain
        methods[arm] = row
        write_json(output / (arm + ".json"), row)
        print(json.dumps({"phase": "same_state_timing", "time": env.time, "arm": arm}), flush=True)
    if canonical_json(visible_snapshot(env)) != original:
        raise AssertionError("panel mutated original environment")
    full = methods["full_reference"]
    full_ids = {r["root_id"] for r in full["pool"]}
    # Root IDs encode the original snapshot and action, independent of arm.
    roots, values = {}, {}
    for method in methods.values():
        for root in method["pool"]:
            rid = root["root_id"]
            if rid in roots and _pool_keys({"pool": [roots[rid]]}) != _pool_keys({"pool": [root]}):
                raise AssertionError("root identity collision")
            roots[rid] = root
        for rid, evaluation in method["evaluations"].items():
            if rid in values and scientific_value(values[rid]) != scientific_value(evaluation["value"]):
                raise AssertionError("same action exact values differ")
            values[rid] = evaluation["value"]
    cost = {"seconds": 0., "work": {}, "scope": "post_panel_labels_only_excluded_from_method_cost"}
    for rid in sorted(set(roots) - set(values)):
        tick = perf_counter()
        evaluation = closed_evaluation(deepcopy(env), _label_root(env, roots[rid]))
        cost["seconds"] += perf_counter() - tick
        for name, value in evaluation["work"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cost["work"][name] = cost["work"].get(name, 0) + value
        if evaluation["status"] != "feasible":
            raise AssertionError("post panel exact label failed")
        values[rid] = evaluation["value"]
    best_id = max(sorted(full_ids), key=lambda rid: value_comparison_key(values[rid]))
    best_key = value_comparison_key(values[best_id])
    rows = []
    for arm in arms:
        row = methods[arm]
        chosen = values[row["chosen_root_id"]]
        rows.append({**{k: row[k] for k in ("timing", "work", "degenerate", "pool_count", "budget",
                     "configured_budget", "selected_root_ids", "chosen_root_id", "adaptive", "search")},
                     "arm": arm, "chosen_value": chosen, "full_best_root_id": best_id,
                     "full_best_value": values[best_id], "best_key_retained": value_comparison_key(chosen) >= best_key,
                     "chosen_in_full_pool": row["chosen_root_id"] in full_ids,
                     "new_value_above_finite_reference": value_comparison_key(chosen) > best_key})
    if canonical_json(visible_snapshot(env)) != original:
        raise AssertionError("analysis labels mutated original environment")
    write_json(output / "full_values.json", {"schema": "fixed-terminal-stage7-panel-values-v1", "values": values,
               "full_pool_root_ids": sorted(full_ids), "full_best_root_id": best_id, "analysis_label_cost": cost,
               "source": "independent_method_exact_values_then_post_panel_labels; finite_reference_not_global_oracle",
               "duplicate_value_comparison": "all_scientific_fields_except_elapsed_through_stop_seconds; raw_method_values_preserved"})
    result = {"schema": "fixed-terminal-stage7-panel-v1", "time": env.time, "arm_order": arms,
              "rotation_index": rotation_index, "rows": rows, "pool_count": len(full_ids),
              "informative": len({value_comparison_key(values[rid]) for rid in full_ids}) > 1,
              "degenerate": full["degenerate"], "full_best_root_id": best_id, "full_best_value": values[best_id],
              "identical_snapshot_and_catalogue_domain": True, "original_environment_unchanged": True,
              "model_loading_excluded": True, "cross_method_cache": False, "analysis_label_cost": cost}
    write_json(output / "summary.json", result)
    return result


def run_panels(output, seed=7001, load="low", soc="high", variant="standard", *,
               models=None, smoke=False, config=DEFAULT_CONFIG, scenario=None, times=None):
    env = _environment(seed, load, soc, variant, smoke, config, scenario)
    times = tuple(_config().PANEL_TIMES if times is None else times)
    if not times or tuple(sorted(set(times))) != times or any(not isfinite(t) or t < 0 or t >= config.horizon or t / config.dt != round(t / config.dt) for t in times):
        raise ValueError("panel times must be ordered unique pre-cutoff ticks")
    if not smoke and times != tuple(_config().PANEL_TIMES):
        raise ValueError("formal panel times are frozen")
    jobs = _config().panel_scenario_jobs("pilot" if smoke else "test")
    key = {"seed": seed, "load": load, "soc": soc, "variant": variant}
    if not any(all(j[k] == v for k, v in key.items()) for j in jobs):
        raise ValueError("scenario outside frozen panel subset")
    all_jobs = _config().scenario_jobs("pilot" if smoke else "test")
    scenario_index = next(i for i, j in enumerate(all_jobs) if all(j[k] == v for k, v in key.items()))
    scorers, load_seconds, provenance = load_scorers(models, ("gradient_boost_expand",), smoke)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = scenario_metadata(env.scenario, variant, config)
    source = source_commit(smoke)
    write_json(output / "frozen_input.json", {"schema": "fixed-terminal-stage7-panel-input-v1", "scenario": asdict(env.scenario),
               "variant": variant, "scenario_metadata": metadata, "config": asdict(config), "times": times,
               "models": provenance, "model_load_seconds": load_seconds, "smoke_only": smoke,
               "prefix": "joint_window", "cadence_seconds": 5., "pool_limit": 14, "source_commit": source})
    next_decision, records = 0., []
    for time_index, target in enumerate(times):
        while env.time < target - 1e-9:
            if env.time >= next_decision - 1e-9:
                apply_scheduler(env, "joint_window")
                next_decision += 5.
            env.step()
        panel = timing_panel(env, output / f"t{target:g}", scorers, scenario_index * 2 + time_index)
        records.append({**key, "time": target, "status": "completed", "context": str(output / f"t{target:g}"),
                        "arm_rows": len(panel["rows"]), "pool_count": panel["pool_count"],
                        "degenerate": panel["degenerate"], "smoke_only": smoke, "split": "pilot" if smoke else "test"})
    after = _postcheck(models, smoke, provenance)
    write_json(output / "generation.json", {"schema": "fixed-terminal-stage7-generation-v1", "status": "completed",
               "contexts": records, "models": provenance, "models_after": after, "model_hashes_unchanged": True,
               "model_load_seconds": load_seconds, "source_commit": source, "smoke_only": smoke,
               "prefix_decisions_before_snapshot": True, "scenario_metadata": metadata,
               "scenario": asdict(env.scenario), "config": asdict(config), "variant": variant, "times": times})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("episode", "panels"), required=True)
    parser.add_argument("--arm", choices=tuple(a["name"] for a in _config().ONLINE_ARMS))
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--load", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--soc", choices=("high", "medium", "low"), default="high")
    parser.add_argument("--variant", choices=("standard", "initial_sparse", "initial_dense"), default="standard")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "episode" and args.arm is None:
        parser.error("episode requires --arm")
    common = dict(seed=args.seed, load=args.load, soc=args.soc, variant=args.variant, models=args.models, smoke=args.smoke)
    if args.mode == "episode":
        run_episode(args.output, args.arm, **common)
    else:
        run_panels(args.output, **common)
    print(json.dumps({"mode": args.mode, "output": str(args.output), "status": "completed"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
