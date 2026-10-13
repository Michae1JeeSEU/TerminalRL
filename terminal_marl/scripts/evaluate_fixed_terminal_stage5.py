"""Stage5 episodes, saved-label budget curves, same-state timing, and diagnostics."""
import os
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import pi
from pathlib import Path, PurePosixPath
from time import perf_counter

from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.core.fixed_terminal.baselines import apply_scheduler
from terminal_marl.core.fixed_terminal.domain import Intent
from terminal_marl.core.fixed_terminal.learning_data import write_json
from terminal_marl.core.fixed_terminal.learning_features import FEATURE_NAMES, extract_features
from terminal_marl.core.fixed_terminal.reference import Root, action_key, value_comparison_key
from terminal_marl.core.fixed_terminal.scenarios import generate_scenario
from terminal_marl.core.fixed_terminal.validation_scheduler import (make_pool, shortlist, closed_evaluation,
                                                                   apply_validation_scheduler)
from terminal_marl.core.fixed_terminal.value_analysis import (visible_snapshot, delta_value, canonical_json,
                                                              config_from_dict, evaluate_root, system_value)
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.scripts.evaluate_fixed_terminal_stage4 import timing_summary


def _config():
    from terminal_marl.configs import fixed_terminal_stage5
    return fixed_terminal_stage5


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def frozen_models(directory, smoke=False):
    if directory is None:
        if smoke:
            return None
        raise ValueError("formal evaluation requires frozen model manifest")
    directory = Path(directory)
    manifest = read_json(directory/"manifest.json")
    if (manifest.get("schema") != "fixed-terminal-stage5-model-manifest-v1" or
            manifest.get("status") != "completed" or manifest.get("frozen") is not True or
            manifest.get("roundtrip_predictions_equal") is not True or
            manifest.get("test_read") is not False or
            manifest.get("producer") != "terminal_marl.core.fixed_terminal.validation_models" or
            (not smoke and manifest.get("smoke_only") is not False) or
            manifest.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("model manifest is not a frozen compatible formal artifact")
    required = set(_config().MODEL_FILES.values())
    if not required <= set(manifest["files"]):
        raise ValueError("manifest missing required models")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256((directory/name).read_bytes()).hexdigest() != expected:
            raise ValueError("model artifact hash mismatch: "+name)
    return {"directory": str(directory), "manifest_sha256": sha256((directory/"manifest.json").read_bytes()).hexdigest(),
            "manifest": manifest}


def load_scorers(directory, smoke=False):
    provenance = frozen_models(directory, smoke)
    if directory is None:
        return {}, {}, provenance
    from terminal_marl.core.fixed_terminal.validation_models import ValidationScorer
    scorers, seconds = {}, {}
    for name, filename in _config().MODEL_FILES.items():
        start = perf_counter()
        scorers[name] = ValidationScorer.load(Path(directory)/filename)
        seconds[name] = perf_counter()-start
    return scorers, seconds, provenance


def scenario_for(seed, load, soc, variant="standard", config=DEFAULT_CONFIG):
    scenario = generate_scenario(config, seed, load, soc)
    if variant == "standard":
        return scenario
    if variant != "initial_rotated" or (load, soc) != ("medium", "medium"):
        raise ValueError("initial_rotated is restricted to medium/medium")
    return replace(scenario, requests=tuple(replace(r, theta=(r.theta+(r.ring+1)*pi/7) % (2*pi))
                                           if r.arrival_time == 0 and r.ring is not None else r for r in scenario.requests))


def validate_run(seed, smoke, config, scenario=None):
    allowed = _config().PILOT_SEEDS if smoke else _config().TEST_SEEDS
    if seed not in allowed:
        raise ValueError("seed outside the declared smoke/formal group")
    if not smoke and (config != DEFAULT_CONFIG or scenario is not None):
        raise ValueError("formal evaluation requires unchanged research defaults/generated scenarios")


def _arm(name):
    return next(a for a in _config().ONLINE_ARMS if a["name"] == name)


def _scorer(arm, scorers):
    name = "mlp_rank" if arm["method"] == "hybrid" else arm["method"]
    return scorers.get(name)


def _apply(env, arm, scorers, pool_limit=14):
    return apply_validation_scheduler(env, arm["method"], _scorer(arm, scorers),
                                      budget=arm["budget"], pool_limit=pool_limit,
                                      rolling_seconds=arm.get("rolling_seconds"))


def run_episode(output, arm_name, seed=5001, load="low", soc="medium", variant="standard", *,
                models=None, smoke=False, config=DEFAULT_CONFIG, scenario=None):
    validate_run(seed, smoke, config, scenario)
    arm = _arm(arm_name)
    scorers, load_seconds, provenance = load_scorers(models, smoke)
    if arm["method"] in ("linear_rank", "mlp_rank", "gradient_boost", "value", "hybrid") and _scorer(arm, scorers) is None:
        raise ValueError("learned arm requires its frozen scorer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = FixedTerminalEnv(config, scenario or scenario_for(seed, load, soc, variant, config))
    write_json(output/"frozen_input.json", {"schema": "fixed-terminal-stage5-input-v1", "arm": arm,
               "scenario": asdict(env.scenario), "variant": variant, "config": asdict(config), "models": provenance,
               "smoke_only": smoke, "source_commit": os.environ.get("TERMINAL_SOURCE_COMMIT"), "model_load_seconds": load_seconds, "cadence_seconds": 5.,
               "pool_limit": 14, "objective": "closed_observed_FCFS_actual_J_except_rolling_commitment_surrogate"})
    rows, event_index, next_decision, step_seconds, io_seconds = [], 0, 0., 0., 0.
    with (output/"decisions.jsonl").open("w", encoding="utf-8") as stream, (output/"events.jsonl").open("w", encoding="utf-8") as events:
        while not env.done:
            if env.time >= next_decision-1e-9:
                row = _apply(env, arm, scorers)
                row["arm"] = arm_name
                rows.append(row)
                tick = perf_counter()
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
                io_seconds += perf_counter()-tick
                next_decision += 5.
            tick = perf_counter()
            env.step()
            step_seconds += perf_counter()-tick
            tick = perf_counter()
            for event in env.events[event_index:]:
                events.write(json.dumps(event, allow_nan=False)+"\n")
            events.flush()
            io_seconds += perf_counter()-tick
            event_index = len(env.events)
    result = env.report()
    result.update({"schema": "fixed-terminal-stage5-episode-v1", "scheduler": arm_name, "arm": arm,
                   "variant": variant, "smoke_only": smoke, "models": provenance, "source_commit": os.environ.get("TERMINAL_SOURCE_COMMIT"),
                   "model_load_seconds": load_seconds, "environment_step_seconds": step_seconds,
                   "decision_event_io_seconds": io_seconds, "decision_timings": timing_summary(rows),
                   "final_value": system_value(env),
                   "work": {k: sum(r["work"][k] for r in rows) for k in
                            ("root_rollout_count", "simulation_steps", "root_validation_calls", "continuation_submission_calls", "final_submit_count")}})
    result["assumptions"]["normal_domain"] = "selected_candidates_jointly_committed; unselected_holders_hold; commitments_fixed"
    result["assumptions"]["emergency_domain"] = "normal_domain_infeasibility_certificate_required; UNKNOWN_never_authorizes_emergency"
    landed = [r for r in result["requests"] if r["outcome"] in ("normal_landed", "emergency_landed")]
    result["summary"]["minimum_landed_soc"] = min((r["minimum_soc"] for r in landed), default=None)
    write_json(output/"result.json", result)
    with (output/"requests.jsonl").open("w", encoding="utf-8") as stream:
        for request in result["requests"]:
            stream.write(json.dumps(request, allow_nan=False)+"\n")
    return result


def cached_budget_curves(roots, metadata, values, features, scorers):
    """Saved-label selection quality only: these lookups are not runtime benchmarks."""
    keys = {r.root_id: value_comparison_key(values[r.root_id]) for r in roots}
    best = max(keys.values())
    full = min((r for r in roots if keys[r.root_id] == best), key=lambda r: action_key(r.plans))
    predictions = {name: scorer.predict(features) for name, scorer in scorers.items()}
    methods = ["beam"]+list(scorers)+(["hybrid"] if "mlp_rank" in scorers else [])
    rows = []
    for method in methods:
        prediction = predictions.get("mlp_rank" if method == "hybrid" else method)
        for budget in (3, 5, 8):
            selected = shortlist(roots, metadata, method, budget, prediction)
            chosen_key = max(keys[r.root_id] for r in selected)
            chosen = min((r for r in selected if keys[r.root_id] == chosen_key), key=lambda r: action_key(r.plans))
            rows.append({"method": method, "budget": budget, "selected_root_ids": [r.root_id for r in selected],
                         "chosen_root_id": chosen.root_id, "chosen_value": values[chosen.root_id],
                         "full_best_root_id": full.root_id, "best_key_retained": chosen_key == best,
                         "raw_J_regret_full_minus_chosen": [a-b for a, b in zip(values[full.root_id]["J"], values[chosen.root_id]["J"])],
                         "raw_component_regret_full_minus_chosen": {k: values[full.root_id]["components"][k]-values[chosen.root_id]["components"][k] for k in values[full.root_id]["components"]},
                         "would_evaluate_root_count": 0 if len(roots) == 1 else len(selected),
                         "scope": "offline_cached_labels_quality; not_measured_execution_time"})
    return {"rows": rows, "full_best_root_id": full.root_id, "full_best_value": values[full.root_id],
            "informative": len(set(keys.values())) > 1, "degenerate": len(roots) == 1,
            "predictions": predictions}


def timing_panel(env, output, scorers, rotation_index):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arms = list(_config().ONLINE_ARMS)
    offset = rotation_index % len(arms)
    arms = arms[offset:]+arms[:offset]
    initial = canonical_json(visible_snapshot(env))
    pool_keys, rows = None, []
    for arm in arms:
        work = deepcopy(env)
        if canonical_json(visible_snapshot(work)) != initial:
            raise AssertionError("different timing-panel input snapshot")
        row = _apply(work, arm, scorers)
        keys = sorted(tuple((p["aircraft_id"], tuple(p["wait_cycles"])) for p in sorted(r["plans"], key=lambda p: p["aircraft_id"])) for r in row["pool"])
        if pool_keys is None:
            pool_keys = keys
        elif keys != pool_keys:
            raise AssertionError("different timing-panel candidate pool")
        row["arm"] = arm["name"]
        write_json(output/(arm["name"]+".json"), row)
        rows.append({"arm": arm["name"], "timing": row["timing"], "work": row["work"], "degenerate": row["degenerate"],
                     "chosen_root_id": row["chosen_root_id"]})
        print(json.dumps({"phase": "same_state_timing", "time": env.time, "arm": arm["name"]}), flush=True)
    if canonical_json(visible_snapshot(env)) != initial:
        raise AssertionError("timing panel mutated input")
    result = {"arm_order": [a["name"] for a in arms], "rotation_index": rotation_index, "rows": rows,
              "identical_snapshot_and_pool": True, "model_loading_excluded": True, "cross_method_cache": False}
    write_json(output/"summary.json", result)
    return result


def snapshot_context(env, output, metadata, scorers, *, pool_limit=14, panel=False, rotation_index=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"snapshot.json", visible_snapshot(env))
    roots, pool_meta = make_pool(env, pool_limit)
    write_json(output/"roots.json", {"roots": [r.serializable() for r in roots], "pool_metadata": pool_meta})
    started = perf_counter()
    features = [extract_features(env, r.plans) for r in roots]
    feature_seconds = perf_counter()-started
    write_json(output/"features.json", {"feature_names": list(FEATURE_NAMES),
               "rows": [{"root_id": r.root_id, "values": f} for r, f in zip(roots, features)]})
    values, evaluations = {}, {}
    for root in roots:
        evaluation = closed_evaluation(env, root)
        write_json(output/"rollouts"/(root.root_id+".json"), evaluation)
        values[root.root_id] = evaluation["value"]
        evaluations[root.root_id] = {"seconds": evaluation["seconds"], "work": evaluation["work"]}
        print(json.dumps({"phase": "snapshot_label", "output": str(output), "root": root.root_id, "seconds": evaluation["seconds"]}), flush=True)
    hold = next(r for r in roots if "HOLD" in r.labels)
    write_json(output/"labels.json", {"information_scope": "observed_arrived_cohort_FCFS_until_original_horizon",
               "rows": [{"root_id": r.root_id, "value": values[r.root_id], "delta_vs_HOLD": delta_value(values[r.root_id], values[hold.root_id])} for r in roots]})
    curves = cached_budget_curves(roots, pool_meta, values, features, scorers)
    write_json(output/"budget_curves.json", curves)
    panel_result = timing_panel(env, output/"timing_panel", scorers, rotation_index) if panel else None
    record = {"schema": "fixed-terminal-stage5-context-v1", **metadata, "status": "completed",
              "root_time": env.time, "root_count": len(roots), "pool_limit": pool_limit, "source_commit": os.environ.get("TERMINAL_SOURCE_COMMIT"),
              "feature_dim": len(FEATURE_NAMES), "pool_metadata": pool_meta, "feature_seconds": feature_seconds,
              "config": asdict(env.config), "prefix": "joint_window", "evaluations": evaluations,
              "informative": curves["informative"], "degenerate": len(roots) == 1,
              "timing_panel": panel_result, "offline_quality_is_not_runtime_measurement": True}
    write_json(output/"context.json", record)
    return record


def run_snapshots(output, seed=5001, load="low", soc="medium", variant="standard", *, models=None,
                  smoke=False, config=DEFAULT_CONFIG, scenario=None, times=(0., 15., 90.), enable_panels=True):
    validate_run(seed, smoke, config, scenario)
    scorers, load_seconds, provenance = load_scorers(models, smoke)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = FixedTerminalEnv(config, scenario or scenario_for(seed, load, soc, variant, config))
    if tuple(sorted(set(times))) != tuple(times) or any(t < 0 or t >= config.horizon or t/config.dt != round(t/config.dt) for t in times):
        raise ValueError("snapshot times must be ordered unique pre-cutoff ticks")
    if not smoke and tuple(times) != (0., 15., 90.):
        raise ValueError("formal snapshot times are frozen")
    write_json(output/"frozen_input.json", {"schema": "fixed-terminal-stage5-snapshot-input-v1", "scenario": asdict(env.scenario),
               "variant": variant, "config": asdict(config), "times": times, "models": provenance,
               "model_load_seconds": load_seconds, "smoke_only": smoke, "prefix": "joint_window", "source_commit": os.environ.get("TERMINAL_SOURCE_COMMIT")})
    selected_profile = (load, soc) in (("low", "high"), ("medium", "medium"), ("high", "low"))
    panel_scenario = selected_profile or variant == "initial_rotated"
    all_jobs = scenario_jobs("pilot" if smoke else "test")
    scenario_index = next(i for i, j in enumerate(all_jobs) if all(j[k] == v for k, v in (("seed", seed), ("load", load), ("soc", soc), ("variant", variant))))
    next_decision, records = 0., []
    for index, target in enumerate(times):
        while env.time < target-1e-9:
            if env.time >= next_decision-1e-9:
                apply_scheduler(env, "joint_window")
                next_decision += 5.
            env.step()
        meta = {"seed": seed, "load": load, "soc": soc, "variant": variant, "smoke_only": smoke, "split": "pilot" if smoke else "test"}
        panel = enable_panels and panel_scenario and target in (0., 15.)
        if panel and len(scorers) != 4:
            raise ValueError("timing panel requires all four frozen scorers")
        records.append(snapshot_context(env, output/f"t{target:g}", meta, scorers, panel=panel, rotation_index=scenario_index*3+index))
        if selected_profile and variant == "standard" and target == 15.:
            records.append(snapshot_context(env, output/"large_pool"/"t15", meta, scorers, pool_limit=32))
    write_json(output/"generation.json", {"status": "completed", "contexts": records, "models": provenance,
               "model_load_seconds": load_seconds, "prefix_decisions_before_snapshot": True})
    return records


def scenario_jobs(split="test"):
    seeds = _config().TEST_SEEDS if split == "test" else _config().PILOT_SEEDS
    return ([{"seed": seed, "load": load, "soc": soc, "variant": "standard"} for seed in seeds
             for load in ("low", "medium", "high") for soc in ("high", "medium", "low")]+
            [{"seed": seed, "load": "medium", "soc": "medium", "variant": "initial_rotated"} for seed in seeds])


def saved_context_path(value):
    """Stage4 artifacts may contain Windows paths when read on a Linux host."""
    return PurePosixPath(str(value).replace("\\", "/"))


def continuation_diagnostic(stage4_root, output, load, soc):
    """One fixed old test context; saved old selections, no checkpoint re-inference."""
    stage4_root, output = Path(stage4_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    base = stage4_root/"data"/"test"/f"{load}-{soc}-4303"
    frozen = read_json(base/"frozen_input.json")
    config = config_from_dict(frozen["config"])
    env = FixedTerminalEnv(config, generate_scenario(config, 4303, load, soc))
    if canonical_json(asdict(env.scenario)) != frozen["scenario"]:
        raise ValueError("old frozen scenario mismatch")
    next_decision = 0.
    while env.time < 15.-1e-9:
        if env.time >= next_decision-1e-9:
            apply_scheduler(env, "joint_window")
            next_decision += 5.
        env.step()
    if canonical_json(visible_snapshot(env)) != read_json(base/"t15"/"snapshot.json"):
        raise ValueError("old visible snapshot mismatch")
    roots_doc = read_json(base/"t15"/"roots.json")
    roots = {}
    for row in roots_doc["roots"]:
        plans = tuple(env.propose(Intent(p["aircraft_id"], tuple(p["wait_cycles"]))) for p in row["plans"])
        if canonical_json([asdict(p) for p in plans]) != row["plans"]:
            raise ValueError("old canonical plan mismatch")
        roots[row["root_id"]] = Root(row["root_id"], plans, row["labels"], row["provenance"], row["feasibility"])
    old = read_json(stage4_root/"summary"/"offline_test.json")
    chosen = {}
    for method in ("beam_topk", "rank_topk", "value_topk"):
        row = next(r for r in old["methods"][method]["contexts"] if saved_context_path(r["context"]).parent.name == base.name and saved_context_path(r["context"]).name == "t15")
        chosen[method] = row["filtered_chosen"]
        chosen["full_best"] = row["full_best_root"]
    chosen["HOLD"] = next(r.root_id for r in roots.values() if "HOLD" in r.labels)
    ids = sorted(set(chosen.values()), key=lambda key: action_key(roots[key].plans))
    if len(ids) > 5:
        raise AssertionError("continuation union exceeded5")
    labels = {r["root_id"]: r["value"] for r in read_json(base/"t15"/"labels.json")["rows"]}
    output.mkdir(parents=True)
    write_json(output/"input.json", {"source": str(base), "snapshot_match": True, "selected_by": chosen,
               "root_ids": ids, "selection_source": "preexisting_stage4_offline_test; no_checkpoint_reinference"})
    rows = []
    for root_id in ids:
        evaluation = evaluate_root(env, roots[root_id].plans, (config.horizon,), realized_future=True)
        if evaluation["status"] != "feasible":
            raise RuntimeError("realized continuation root rejected")
        write_json(output/"roots"/(root_id+".json"), evaluation)
        actual = evaluation["values"][f"{config.horizon:g}"]
        rows.append({"root_id": root_id, "closed_value": labels[root_id], "realized_value": actual,
                     "raw_component_difference_realized_minus_closed": {k: actual["components"][k]-labels[root_id]["components"][k] for k in actual["components"]}})
        print(json.dumps({"phase": "continuation_diagnostic", "root": root_id, "seconds": evaluation["seconds"]}), flush=True)
    pairs = []
    for i, left in enumerate(rows):
        for right in rows[i+1:]:
            def compare(field):
                a, b = value_comparison_key(left[field]), value_comparison_key(right[field])
                return int(a > b)-int(a < b)
            closed, actual = compare("closed_value"), compare("realized_value")
            pairs.append({"left": left["root_id"], "right": right["root_id"], "closed_order": closed,
                          "realized_order": actual, "strict_reversal": closed*actual == -1, "order_changed": closed != actual})
    result = {"schema": "fixed-terminal-stage5-continuation-diagnostic-v1", "source_commit": os.environ.get("TERMINAL_SOURCE_COMMIT"), "scenario": {"load": load, "soc": soc, "seed": 4303, "time": 15.},
              "selected_by": chosen, "rows": rows, "pairs": pairs,
              "scope": "ex_post_realized_future_vs_closed_arrived_world; different_denominators; not_expected_Q"}
    write_json(output/"result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("episode", "snapshot", "continuation"), default="episode")
    parser.add_argument("--arm", default="beam_b5")
    parser.add_argument("--seed", type=int, default=5301)
    parser.add_argument("--load", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--soc", choices=("high", "medium", "low"), default="medium")
    parser.add_argument("--variant", choices=("standard", "initial_rotated"), default="standard")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--stage4-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args(argv)
    if args.list_jobs:
        jobs = scenario_jobs("pilot" if args.smoke else "test")
        if args.mode == "episode":
            jobs = [{**job, "arm": arm["name"]} for job in jobs for arm in _config().ONLINE_ARMS]
        elif args.mode == "continuation":
            jobs = [{"seed": 4303, "load": load, "soc": soc, "time": 15.} for load in ("low", "medium", "high") for soc in ("high", "medium", "low")]
        print(json.dumps(jobs))
        return 0
    if args.output is None:
        parser.error("--output required")
    if args.mode == "continuation":
        if args.stage4_root is None:
            parser.error("--stage4-root required for continuation")
        result = continuation_diagnostic(args.stage4_root, args.output, args.load, args.soc)
        print(json.dumps({"roots": len(result["rows"])}))
    else:
        kwargs = dict(seed=args.seed, load=args.load, soc=args.soc, variant=args.variant, models=args.models, smoke=args.smoke)
        result = run_episode(args.output, args.arm, **kwargs) if args.mode == "episode" else run_snapshots(args.output, **kwargs)
        print(json.dumps({"mode": args.mode, "output": str(args.output), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
