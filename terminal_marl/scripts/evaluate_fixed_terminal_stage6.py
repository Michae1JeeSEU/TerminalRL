"""Frozen stage6 episodes and isolated same-state timing panels."""
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
from terminal_marl.scripts.evaluate_fixed_terminal_stage5 import frozen_models


def _config():
    from terminal_marl.configs import fixed_terminal_stage6
    return fixed_terminal_stage6


def _arm(name):
    for arm in _config().ONLINE_ARMS:
        if arm["name"] == name:
            return arm
    raise ValueError("unknown stage6 arm: " + str(name))


def scorer_name(arm_name):
    _arm(arm_name)
    return {"rank_adaptive": "mlp_rank", "hybrid_adaptive": "mlp_rank",
            "gb_adaptive": "gradient_boost"}.get(arm_name)


def load_scorers(directory, names, smoke=False):
    """Validate all four frozen artifacts, but load only the requested scorers."""
    provenance = frozen_models(directory, smoke=smoke)
    names = tuple(dict.fromkeys(names))
    if any(name not in ("mlp_rank", "gradient_boost") for name in names):
        raise ValueError("stage6 uses only frozen stage5 MLP rank and gradient boost")
    if names and directory is None:
        raise ValueError("learned arms require frozen stage5 models")
    scorers, seconds = {}, {}
    if names:
        from terminal_marl.core.fixed_terminal.validation_models import ValidationScorer
        for name in names:
            filename = name + (".joblib" if name == "gradient_boost" else ".pt")
            start = perf_counter()
            scorer = ValidationScorer.load(Path(directory) / filename)
            seconds[name] = perf_counter() - start
            if scorer.name != name or scorer.kind != "rank":
                raise ValueError("frozen scorer identity/kind mismatch")
            scorers[name] = scorer
    return scorers, seconds, provenance


def _apply(env, arm_name, scorers):
    from terminal_marl.core.fixed_terminal.assisted_scheduler import apply_assisted_scheduler
    name = scorer_name(arm_name)
    scorer = scorers.get(name) if name else None
    if name and (scorer is None or scorer.name != name or scorer.kind != "rank"):
        raise ValueError("arm requires its matching frozen scorer")
    return apply_assisted_scheduler(env, arm_name, scorer=scorer, pool_limit=14)


def scenario_for(seed, load, soc, variant="standard", config=DEFAULT_CONFIG):
    base = generate_scenario(config, seed, load, soc)
    if variant == "standard":
        result = base
    else:
        if variant not in ("initial_sparse", "initial_dense") or (load, soc) != ("medium", "medium"):
            raise ValueError("occupancy variants are restricted to medium/medium")
        source_load = "low" if variant == "initial_sparse" else "high"
        initial_source = generate_scenario(config, seed, source_load, soc)
        initial = [r for r in initial_source.requests if r.ring is not None]
        base_initial = [r for r in base.requests if r.ring is not None]
        extra_rng = Random(stream_seed(seed, "stage6-extra-initial-soc"))
        spec = SOC_PROFILES[soc]
        initial = [replace(r, soc=base_initial[i].soc if i < len(base_initial) else
                           truncated_normal(extra_rng, spec.mean, spec.std, spec.lower, spec.upper))
                   for i, r in enumerate(initial)]
        future = [r for r in base.requests if r.ring is None]
        requests = tuple(replace(r, aircraft_id=f"uav-{index:04d}") for index, r in enumerate(initial + future))
        result = replace(base, requests=requests)
    # Includes operational capacity and continuous initial protected spacing checks.
    FixedTerminalEnv(config, result)
    return result


def scenario_metadata(scenario, variant, config):
    initial = [r for r in scenario.requests if r.ring is not None]
    return {"variant": variant,
            "initial_source_load": {"initial_sparse": "low", "initial_dense": "high"}.get(variant, scenario.load),
            "initial_counts_by_ring": [sum(r.ring == i for r in initial) for i in range(len(config.rings))],
            "initial_request_count": len(initial),
            "future_request_count": sum(r.ring is None for r in scenario.requests),
            "future_pairing": "base_profile_future_arrival_time_SOC_unchanged; IDs_reindexed_after_initial",
            "initial_soc_source": "base_initial_prefix; dense_extras_independent_stage6_stream" if variant != "standard" else "base_profile_soc_stream",
            "extra_initial_soc_stream_seed": stream_seed(scenario.seed, "stage6-extra-initial-soc") if variant == "initial_dense" else None,
            "initial_capacity_and_spacing_validated": True}


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
        raise ValueError("scenario outside frozen stage6 matrix")
    if scenario is not None and (scenario.load != load or scenario.soc_profile != soc):
        raise ValueError("fixture profile does not match the requested job")
    return FixedTerminalEnv(config, scenario if scenario is not None else scenario_for(seed, load, soc, variant, config))


def _postcheck(models, smoke, before):
    after = frozen_models(models, smoke=smoke)
    if before != after:
        raise ValueError("frozen model provenance changed during evaluation")
    return after


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


def run_episode(output, arm_name, seed=6001, load="low", soc="medium", variant="standard", *,
                models=None, smoke=False, config=DEFAULT_CONFIG, scenario=None):
    arm = _arm(arm_name)
    env = _environment(seed, load, soc, variant, smoke, config, scenario)
    name = scorer_name(arm_name)
    scorers, load_seconds, provenance = load_scorers(models, (name,) if name else (), smoke)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = os.environ.get("TERMINAL_SOURCE_COMMIT")
    metadata = scenario_metadata(env.scenario, variant, config)
    write_json(output / "frozen_input.json", {"schema": "fixed-terminal-stage6-input-v1", "arm": arm,
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
    result.update({"schema": "fixed-terminal-stage6-episode-v1", "scheduler": arm_name, "arm": arm,
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


def timing_panel(env, output, scorers, rotation_index):
    """No shared method cache; label annotation happens only after all arms run."""
    for name in ("mlp_rank", "gradient_boost"):
        if name not in scorers or scorers[name].name != name or scorers[name].kind != "rank":
            raise ValueError("panel requires both frozen stage5 rank scorers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arms = [a["name"] for a in _config().ONLINE_ARMS]
    offset = rotation_index % len(arms)
    arms = arms[offset:] + arms[:offset]
    original = canonical_json(visible_snapshot(env))
    write_json(output / "snapshot.json", visible_snapshot(env))
    pool_keys, method_rows = None, {}
    for arm in arms:
        clone = deepcopy(env)
        if canonical_json(visible_snapshot(clone)) != original:
            raise AssertionError("different panel snapshot")
        row = _apply(clone, arm, scorers)
        keys = _pool_keys(row)
        if pool_keys is not None and keys != pool_keys:
            raise AssertionError("different panel candidate action keys")
        pool_keys = keys
        row["arm"] = arm
        method_rows[arm] = row
        write_json(output / (arm + ".json"), row)
        print(json.dumps({"phase": "same_state_timing", "time": env.time, "arm": arm}), flush=True)
    if canonical_json(visible_snapshot(env)) != original:
        raise AssertionError("panel mutated original environment")
    full = method_rows["full_reference"]
    values = {rid: evaluation["value"] for rid, evaluation in full["evaluations"].items()}
    analysis_cost = {"seconds": 0., "work": {}, "scope": "post_panel_label_only; excluded_from_all_method_timings"}
    if not values:
        if not full["degenerate"] or len(full["pool"]) != 1:
            raise AssertionError("full reference has no exact values outside single-root fast path")
        saved = full["pool"][0]
        plans = tuple(env.propose(Intent(p["aircraft_id"], tuple(p["wait_cycles"]))) for p in saved["plans"])
        if canonical_json([asdict(p) for p in plans]) != canonical_json(saved["plans"]):
            raise AssertionError("single-root analysis reconstruction mismatch")
        root = Root(saved["root_id"], plans, saved["labels"], saved["provenance"], saved["feasibility"])
        tick = perf_counter()
        evaluation = closed_evaluation(deepcopy(env), root)
        analysis_cost.update(seconds=perf_counter() - tick, work=evaluation["work"])
        values[root.root_id] = evaluation["value"]
    if set(values) != {root["root_id"] for root in full["pool"]}:
        raise AssertionError("full reference did not evaluate the complete pool")
    best_key = max(value_comparison_key(value) for value in values.values())
    best_id = min((root for root in full["pool"] if value_comparison_key(values[root["root_id"]]) == best_key),
                  key=lambda root: tuple(sorted((p["aircraft_id"], tuple(p["wait_cycles"])) for p in root["plans"])))["root_id"]
    rows = []
    for arm in arms:
        row = method_rows[arm]
        chosen = values[row["chosen_root_id"]]
        rows.append({"arm": arm, "timing": row["timing"], "work": row["work"], "degenerate": row["degenerate"],
                     "pool_count": row["pool_count"], "budget": row["budget"], "configured_budget": row["configured_budget"],
                     "selected_root_ids": row["selected_root_ids"], "chosen_root_id": row["chosen_root_id"],
                     "chosen_value": chosen, "full_best_root_id": best_id, "full_best_value": values[best_id],
                     "best_key_retained": value_comparison_key(chosen) == best_key, "adaptive": row["adaptive"]})
    if canonical_json(visible_snapshot(env)) != original:
        raise AssertionError("analysis labels mutated original environment")
    write_json(output / "full_values.json", {"schema": "fixed-terminal-stage6-panel-values-v1", "values": values,
               "full_best_root_id": best_id, "analysis_label_cost": analysis_cost,
               "source": "full_reference_evaluations; single_root_post_panel_closed_label_if_needed"})
    result = {"schema": "fixed-terminal-stage6-panel-v1", "time": env.time, "arm_order": arms,
              "rotation_index": rotation_index, "rows": rows, "pool_count": len(full["pool"]),
              "informative": len({value_comparison_key(v) for v in values.values()}) > 1,
              "degenerate": full["degenerate"], "full_best_root_id": best_id, "full_best_value": values[best_id],
              "identical_snapshot_and_pool": True, "original_environment_unchanged": True,
              "model_loading_excluded": True, "cross_method_cache": False, "analysis_label_cost": analysis_cost}
    write_json(output / "summary.json", result)
    return result


def run_panels(output, seed=6001, load="low", soc="high", variant="standard", *,
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
    scorers, load_seconds, provenance = load_scorers(models, ("mlp_rank", "gradient_boost"), smoke)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = scenario_metadata(env.scenario, variant, config)
    source = os.environ.get("TERMINAL_SOURCE_COMMIT")
    write_json(output / "frozen_input.json", {"schema": "fixed-terminal-stage6-panel-input-v1", "scenario": asdict(env.scenario),
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
    write_json(output / "generation.json", {"schema": "fixed-terminal-stage6-generation-v1", "status": "completed",
               "contexts": records, "models": provenance, "models_after": after, "model_hashes_unchanged": True,
               "model_load_seconds": load_seconds, "source_commit": source, "smoke_only": smoke,
               "prefix_decisions_before_snapshot": True, "scenario_metadata": metadata,
               "scenario": asdict(env.scenario), "config": asdict(config), "variant": variant, "times": times})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("episode", "panels"), required=True)
    parser.add_argument("--arm", choices=tuple(a["name"] for a in _config().ONLINE_ARMS))
    parser.add_argument("--seed", type=int, default=6001)
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
