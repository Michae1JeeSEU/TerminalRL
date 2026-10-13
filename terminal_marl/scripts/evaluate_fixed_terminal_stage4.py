"""Offline candidate quality or fresh full-episode scheduler evaluation."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from dataclasses import asdict
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs.fixed_terminal_stage4 import METHODS, SPLIT_SEEDS, jobs, REFERENCE_SETTINGS
from terminal_marl.core.fixed_terminal.scenarios import generate_scenario
from terminal_marl.core.fixed_terminal.learning_data import load_contexts, write_json
from terminal_marl.core.fixed_terminal.learning_scheduler import apply_learning_scheduler
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.scripts.generate_fixed_terminal_stage4 import frozen_models


def timing_summary(rows):
    def p95(values):
        ordered = sorted(values)
        index = .95*(len(ordered)-1)
        low = int(index)
        high = min(len(ordered)-1, low+1)
        return ordered[low]+(ordered[high]-ordered[low])*(index-low)
    groups = {}
    for name, selected in (("all", rows), ("degenerate", [r for r in rows if r["degenerate"] is True]),
                           ("nondegenerate", [r for r in rows if r["degenerate"] is False])):
        fields = sorted({k for row in selected for k in row["timing"]})
        groups[name] = {"decisions": len(selected), "total_over_5_seconds_count": sum(r["timing"]["total"] > 5. for r in selected),
                       "p95_definition": "linear_interpolation_at_0.95_times_n_minus_1", "seconds": {field: {
            "total": sum(r["timing"][field] for r in selected),
            "p50": median(r["timing"][field] for r in selected),
            "p95": p95([r["timing"][field] for r in selected]),
            "max": max(r["timing"][field] for r in selected)} for field in fields}}
    return groups


def run_episode(output, method, load="low", soc="medium", seed=4001, *, models=None, config=DEFAULT_CONFIG, scenario=None):
    if seed not in SPLIT_SEEDS["test"]+SPLIT_SEEDS["pilot"]:
        raise ValueError("online evaluation accepts only preregistered test or pilot seeds")
    if method not in METHODS:
        raise ValueError("unknown method")
    provenance = frozen_models(models) if seed in SPLIT_SEEDS["test"] else (frozen_models(models, allow_smoke=True) if models else None)
    scorer, load_seconds = None, 0.
    if method in ("value_topk", "rank_topk"):
        if models is None:
            raise ValueError("learned filtering requires checkpoint directory")
        tick = perf_counter()
        from terminal_marl.core.fixed_terminal.learning_models import Scorer
        scorer = Scorer.load(Path(models)/(method.split("_")[0]+".pt"))
        load_seconds = perf_counter()-tick
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = FixedTerminalEnv(config, scenario or generate_scenario(config, seed, load, soc))
    report = env.report()
    assumptions = report["assumptions"]
    assumptions["normal_domain"] = "selected_candidates_jointly_committed; unselected_holders_hold; commitments_fixed"
    assumptions["emergency_domain"] = "normal_domain_infeasibility_certificate_required; UNKNOWN_never_authorizes_emergency"
    frozen = {"schema": "fixed-terminal-stage4-input-v1", "method": method, "config": asdict(config),
              "scenario": asdict(env.scenario), "pool_settings": asdict(REFERENCE_SETTINGS), "models": provenance,
              "assumptions": assumptions, "cadence_seconds": 5., "model_load_seconds": load_seconds,
              "root_objective": "observed_cohort_actual_J_under_FCFS_until_horizon",
              "rolling_horizon_objective": "original_stage2_20s_commitment_surrogate_differs_from_stage4_J"}
    write_json(output/"frozen_input.json", frozen)
    decisions, next_decision, step_seconds, event_index = [], 0., 0., 0
    with (output/"decisions.jsonl").open("w", encoding="utf-8") as stream, (output/"events.jsonl").open("w", encoding="utf-8") as events:
        while not env.done:
            if env.time >= next_decision-1e-9:
                row = apply_learning_scheduler(env, method, scorer)
                decisions.append(row)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
                next_decision += 5.
            tick = perf_counter()
            env.step()
            step_seconds += perf_counter()-tick
            for event in env.events[event_index:]:
                events.write(json.dumps(event, allow_nan=False)+"\n")
            events.flush()
            event_index = len(env.events)
    result = env.report()
    result.update({"schema": "fixed-terminal-stage4-episode-v1", "scheduler": method, "assumptions": assumptions,
                   "model_load_seconds": load_seconds, "environment_step_seconds": step_seconds,
                   "decision_timings": timing_summary(decisions), "models": provenance})
    landed = [r for r in result["requests"] if r["outcome"] in ("normal_landed", "emergency_landed")]
    result["summary"]["minimum_landed_soc"] = min((r["minimum_soc"] for r in landed), default=None)
    reasons = {}
    for row in decisions:
        groups = ([row["statistics"]] if "statistics" in row else [row["pool_metadata"]["statistics"], row["emergency_statistics"]])
        for stats in groups:
            for reason, count in stats["rejection_reasons"].items():
                reasons[reason] = reasons.get(reason, 0)+count
    result["summary"]["constraint_rejection_reasons"] = reasons
    write_json(output/"result.json", result)
    with (output/"requests.jsonl").open("w", encoding="utf-8") as stream:
        for row in result["requests"]:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    return result


def evaluate_offline(data, models, output, split="test"):
    frozen = frozen_models(models, allow_smoke=split == "pilot")
    contexts = load_contexts(data, split)
    from terminal_marl.core.fixed_terminal.learning_models import Scorer, offline_metrics
    result = {"schema": "fixed-terminal-stage4-offline-v1", "split": split, "models": frozen, "contexts": len(contexts),
              "methods": {"beam_topk": offline_metrics(contexts, "beam_topk")}}
    for kind in ("value", "rank"):
        scorer = Scorer.load(Path(models)/f"{kind}.pt")
        result["methods"][kind+"_topk"] = offline_metrics(contexts, kind+"_topk", scorer)
    write_json(output, result)
    return result


def aggregate(directory, output):
    rows = []
    for path in sorted(Path(directory).rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema") != "fixed-terminal-stage4-episode-v1":
            continue
        s = result["summary"]
        rows.append({"method": result["scheduler"], "load": result["scenario"]["load"], "soc": result["scenario"]["soc_profile"],
                     "seed": result["scenario"]["seed"], "requests": s["request_count"], **s["outcomes"],
                     "normal_delay": s["normal_delay_mean_seconds"], "exposure": s["observed_request_exposure_seconds"],
                     "decision_seconds": result["decision_timings"]["all"]["seconds"]["total"]["total"], "path": str(path)})
    output = Path(output)
    write_json(output.with_suffix(".json"), rows)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="fcfs")
    parser.add_argument("--seed", type=int, default=4301)
    parser.add_argument("--load", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--soc", choices=("high", "medium", "low"), default="medium")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offline", type=Path, help="root dataset directory")
    parser.add_argument("--split", choices=("test", "pilot"), default="test")
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args(argv)
    if args.list_jobs:
        print(json.dumps([{**job, "method": method} for job in jobs(args.split) for method in METHODS]))
        return 0
    if args.output is None:
        parser.error("--output required")
    if args.aggregate:
        print(json.dumps({"rows": len(aggregate(args.aggregate, args.output))}))
    elif args.offline:
        result = evaluate_offline(args.offline, args.models, args.output, args.split)
        print(json.dumps({"contexts": result["contexts"]}))
    else:
        result = run_episode(args.output, args.method, args.load, args.soc, args.seed, models=args.models)
        print(json.dumps({"scheduler": result["scheduler"], "summary": result["summary"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
