"""Paired stage2 comparison and offline finite-catalogue snapshot references."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG, LOAD_PROFILES, SOC_PROFILES, STANDARD_SEEDS
from terminal_marl.configs.fixed_terminal_stage2 import DEFAULT_SCHEDULER_CONFIG, METHODS
from terminal_marl.core.fixed_terminal.baselines import apply_scheduler, observed_clone, make_catalogue, choose_normal_batch, Statistics
from terminal_marl.core.fixed_terminal.scenarios import generate_scenario, Scenario
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def decision_row(decision, time):
    return {"time": time, "method": decision.method, "seconds": decision.seconds,
            "objective": decision.objective,
            "normal_plans": [{"aircraft_id": p.aircraft_id, "wait_cycles": p.wait_cycles, "completion": p.end} for p in decision.plans],
            **asdict(decision.statistics)}


def run(output, method="fcfs", load="low", soc="high", seed=101,
        config=DEFAULT_CONFIG, settings=DEFAULT_SCHEDULER_CONFIG):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = FixedTerminalEnv(config, generate_scenario(config, seed, load, soc))
    frozen = env.report()
    frozen["assumptions"]["normal_domain"] = "selected_candidates_jointly_committed; unselected_uncommitted_holders_hold; existing_commitments_fixed"
    frozen["assumptions"]["emergency_domain"] = "stage1_individual_SOC_bound_domain; normal_domain_infeasibility_certificate_required; UNKNOWN_never_authorizes_emergency"
    write_json(output/"frozen_input.json", {"stage": 2, "scheduler": method, "scheduler_config": asdict(settings),
               **{key: frozen[key] for key in ("config", "scenario", "capacity", "assumptions")}})
    decisions, next_decision = [], 0.
    while not env.done:
        if env.time >= next_decision-1e-9:
            decisions.append(decision_row(apply_scheduler(env, method, settings), env.time))
            next_decision += settings.decision_interval
        env.step()
    result = env.report()
    result["schema"] = "fixed-terminal-stage2-v1"
    result["scheduler"] = method
    result["scheduler_config"] = asdict(settings)
    result["assumptions"] = frozen["assumptions"]
    times = [row["seconds"] for row in decisions]
    summary = result["summary"]
    summary.update({"online_call_count": len(times), "online_call_seconds_total": sum(times),
                    "online_call_seconds_p50": median(times) if times else None,
                    "online_call_seconds_max": max(times) if times else None,
                    "candidates_generated": sum(row["generated"] for row in decisions),
                    "candidates_examined": sum(row["examined"] for row in decisions),
                    "joint_assignments_examined": sum(row["joint_assignments"] for row in decisions),
                    "truncated_decisions": sum(row["truncated"] for row in decisions),
                    "lookahead_states": sum(row["lookahead_states"] for row in decisions)})
    reasons = {}
    for row in decisions:
        for key, value in row["rejection_reasons"].items():
            reasons[key] = reasons.get(key, 0)+value
    summary["constraint_rejection_reasons"] = reasons
    write_json(output/"result.json", result)
    for filename, rows in (("requests.jsonl", result["requests"]), ("events.jsonl", env.events), ("decisions.jsonl", decisions)):
        with (output/filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
    return result


def reference_cohort(size, seed=101, config=DEFAULT_CONFIG):
    original = generate_scenario(config, seed, "high", "medium")
    rings = [[r for r in original.requests if r.ring == i] for i in range(len(config.rings))]
    selected = []
    while len(selected) < size:
        for ring in rings:
            if ring and len(selected) < size:
                selected.append(ring.pop(0))
        if not any(rings) and len(selected) < size:
            raise ValueError("reference cohort exceeds available initial requests")
    return Scenario(seed, "closed_high_initial_subset", "medium", tuple(selected), original.demand_stream_seed, original.soc_stream_seed)


def run_reference(output, size=3, seed=101, settings=DEFAULT_SCHEDULER_CONFIG):
    from terminal_marl.core.fixed_terminal.oracle import finite_catalogue_milp, objective_gap
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = FixedTerminalEnv(DEFAULT_CONFIG, reference_cohort(size, seed))
    stats = Statistics()
    catalogue = make_catalogue(env, settings, stats, reference=True)
    frozen = env.report()
    write_json(output/"frozen_input.json", {"stage": 2, "reference": "finite_catalogue_snapshot_not_dynamic_optimum",
               "scheduler_config": asdict(settings), "config": frozen["config"], "scenario": frozen["scenario"],
               "catalogue": {key: [asdict(p) for p in group] for key, group in catalogue.items()},
               "catalogue_statistics": asdict(stats)})
    oracle = finite_catalogue_milp(env, catalogue, settings.oracle_seconds, settings.oracle_iterations)
    comparisons = []
    for method in METHODS:
        decision = choose_normal_batch(method, observed_clone(env), settings, catalogue)
        comparisons.append({**decision_row(decision, env.time), "gap": objective_gap(decision.objective, oracle)})
    result = {"reference": "same_restricted_finite_catalogue_snapshot", "size": size, "seed": seed,
              "oracle": oracle.serializable(), "comparisons": comparisons,
              "future_arrivals": False, "full_episode_optimality_claim": False}
    write_json(output/"result.json", result)
    return result


def aggregate(root, output=None):
    """Compact paired table, without replacing per-request evidence."""
    rows = []
    for path in sorted(Path(root).glob("**/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "summary" not in data:
            continue
        s, scenario = data["summary"], data["scenario"]
        rows.append({"method": data["scheduler"], "load": scenario["load"], "soc": scenario["soc_profile"], "seed": scenario["seed"],
                     "requests": s["request_count"], "normal": s["outcomes"]["normal_landed"],
                     "emergency": s["outcomes"]["emergency_landed"], "failed": s["outcomes"]["failed_energy"], "censored": s["outcomes"]["censored"],
                     "normal_delay_mean": s["normal_delay_mean_seconds"], "normal_delay_max": s["normal_delay_max_seconds"],
                     "normal_throughput": s["normal_throughput_per_second"], "minimum_soc": s["minimum_soc_all_requests"],
                     "energy_mean": s["energy_mean_soc_all_requests"], "external_backlog": s["state_at_cutoff_counts"]["external"],
                     "separation_violations": s["separation_violation_pair_ticks"], "spacing_uncertified": s["spacing_uncertified_pair_ticks"],
                     "online_seconds": s["online_call_seconds_total"], "online_p50": s["online_call_seconds_p50"], "online_max": s["online_call_seconds_max"],
                     "examined": s["candidates_examined"], "joint_assignments": s["joint_assignments_examined"], "truncated_decisions": s["truncated_decisions"],
                     "path": str(path)})
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination.with_suffix(".json"), rows)
        with destination.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["method"])
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="fcfs")
    parser.add_argument("--load", choices=LOAD_PROFILES, default="low")
    parser.add_argument("--soc", choices=SOC_PROFILES, default="high")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--matrix", action="store_true", help="all162 paired default episode cells")
    parser.add_argument("--oracle", action="store_true", help="closed finite catalogue reference")
    parser.add_argument("--size", type=int, choices=(3, 4, 5, 6), default=3)
    parser.add_argument("--oracle-matrix", action="store_true", help="12 preregistered small reference cohorts")
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.aggregate:
        print(json.dumps({"rows": len(aggregate(args.aggregate, args.output))}))
        return 0
    if args.oracle or args.oracle_matrix:
        cells = [(n, seed) for n in (3, 4, 5, 6) for seed in STANDARD_SEEDS] if args.oracle_matrix else [(args.size, args.seed)]
        for n, seed in cells:
            target = args.output/f"n{n}-seed{seed}" if args.oracle_matrix else args.output
            result = run_reference(target, n, seed)
            print(json.dumps({"output": str(target), "oracle": result["oracle"]}, allow_nan=False))
    else:
        cells = [(m, load, soc, seed) for m in METHODS for load in LOAD_PROFILES for soc in SOC_PROFILES for seed in STANDARD_SEEDS] if args.matrix else [(args.method, args.load, args.soc, args.seed)]
        for method, load, soc, seed in cells:
            target = args.output/f"{method}-{load}-{soc}-{seed}" if args.matrix else args.output
            result = run(target, method, load, soc, seed)
            print(json.dumps({"output": str(target), "summary": result["summary"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
