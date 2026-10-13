"""Stage3 per-context decision explanation and paired counterfactual artifacts.

Run one context with --load/--soc/--seed/--time. Formal15 contexts are a
stage2-informed mechanism sample, not a holdout or full performance matrix.
"""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import argparse
import json
from terminal_marl.configs.fixed_terminal_stage3 import Context, FORMAL_CONTEXTS, REMOVE_ONE_LIMIT
from terminal_marl.core.fixed_terminal.value_analysis import replay_episode, visible_snapshot, system_value, evaluate_root, conditional_margin, delta_value
from terminal_marl.core.fixed_terminal.reference import make_root_pool, observed_evaluation, select_observed_roots


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def run_context(output, context, stage2_root="tmp_stage2_results/episodes"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    source = Path(stage2_root)/f"fcfs-{context.load}-{context.soc}-{context.seed}"
    env = replay_episode(source, context.time)
    snapshot = visible_snapshot(env)
    replay_and_setup_seconds = perf_counter()-started
    write_json(output/"snapshot.json", snapshot)
    write_json(output/"input.json", {"context": asdict(context), "source_fcfs_prefix": str(source),
               "sample_scope": "stage2-informed mechanism sample; not new holdout or full algorithm matrix",
               "formal_context": context in FORMAL_CONTEXTS, "config": asdict(env.config),
               "episode_cumulative": True, "root_time": env.time,
               "value_order": ["-failed_energy", "normal_landed", "-emergency_landed", "-arrived_unresolved",
                               "-observed_request_seconds", "-normal_delay_sum_seconds", "-energy_consumed_soc_all_requests"],
               "continuation": "fixed FCFS; globally aligned stage2 cadence and common emergency",
               "worlds": {"observed": "only requests already arrived at root, including historical terminal requests",
                          "realized": "full original demand path retained only by ex-post simulator; policy inputs remove pending"}})
    roots, metadata = make_root_pool(env, context, stage2_root)
    write_json(output/"root_pool.json", {"metadata": metadata, "roots": [root.serializable() for root in roots]})
    print(json.dumps({"context": context.key, "phase": "root_pool", "unique_roots": len(roots)}), flush=True)
    observed, realized = {}, {}
    times = {"replay_and_setup_seconds": replay_and_setup_seconds, "root_generation_seconds": metadata["root_generation_seconds"],
             "control_acquisition_seconds": metadata["control_acquisition_seconds"],
             "observed_rollouts_seconds": 0., "expost_rollouts_seconds": 0., "conditional_margins_seconds": 0.}
    errors = []
    for index, root in enumerate(roots):
        root_dir = output/root.root_id
        root_dir.mkdir()
        write_json(root_dir/"root.json", root.serializable())
        for kind in ("observed", "realized"):
            tick = perf_counter()
            try:
                evaluation = observed_evaluation(env, root) if kind == "observed" else evaluate_root(env, root.plans, (env.config.horizon,), realized_future=True)
            except Exception as exc:
                error = {"status": "error", "root_id": root.root_id, "world": kind,
                         "error_type": type(exc).__name__, "message": str(exc)}
                write_json(root_dir/f"{kind}.json", error)
                errors.append(error)
                write_json(output/"errors.json", errors)
                raise
            evaluation["root_time"] = env.time
            evaluation["episode_cumulative"] = True
            evaluation["seconds"] = perf_counter()-tick
            (observed if kind == "observed" else realized)[root.root_id] = evaluation
            times["observed_rollouts_seconds" if kind == "observed" else "expost_rollouts_seconds"] += evaluation["seconds"]
            write_json(root_dir/f"{kind}.json", evaluation)
        print(json.dumps({"context": context.key, "phase": "root_evaluated", "index": index+1,
                          "total": len(roots), "root_id": root.root_id,
                          "observed_status": observed[root.root_id]["status"], "realized_status": realized[root.root_id]["status"]}), flush=True)
    selections = select_observed_roots(env, roots, observed)
    write_json(output/"observed_selections.json", selections)
    hold = next(root for root in roots if "HOLD" in root.labels)
    fcfs = next(root for root in roots if "fcfs" in root.labels)
    end_key = f"{env.config.horizon:g}"
    rows = []
    for root in roots:
        actual = realized[root.root_id]
        row = {"context": asdict(context), "root_id": root.root_id, "labels": root.labels,
               "provenance": root.provenance, "feasibility": root.feasibility,
               "normal_actions": [{"aircraft_id": p.aircraft_id, "wait_cycles": p.wait_cycles, "planned_end": p.end,
                                   "resources": [asdict(r) for r in p.reservations]} for p in root.plans],
               "observed_values": observed[root.root_id].get("values", {}), "realized_values": actual.get("values", {}),
               "realized_interpretation": "ex-post realized outcome on one fixed demand path, not expected Q or online visible future",
               "delta_realized_vs_HOLD": None, "delta_realized_vs_FCFS": None,
               "delta_observed_vs_HOLD": {}, "delta_observed_vs_FCFS": {}}
        if observed[root.root_id]["status"] == "feasible":
            for label, base in (("HOLD", hold), ("FCFS", fcfs)):
                if observed[base.root_id]["status"] == "feasible":
                    row["delta_observed_vs_"+label] = {
                        stop: delta_value(value, observed[base.root_id]["values"][stop])
                        for stop, value in observed[root.root_id]["values"].items()}
        if actual["status"] == "feasible":
            for label, base in (("HOLD", hold), ("FCFS", fcfs)):
                if realized[base.root_id]["status"] == "feasible":
                    row["delta_realized_vs_"+label] = delta_value(actual["values"][end_key], realized[base.root_id]["values"][end_key])
        rows.append(row)
    with (output/"decision_database.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    winner = next(root for root in roots if root.root_id == selections["until240"]["root_id"])
    tick = perf_counter()
    margins = [conditional_margin(env, winner.plans, p.aircraft_id, realized[winner.root_id])
               for p in sorted(winner.plans, key=lambda p: p.aircraft_id)[:REMOVE_ONE_LIMIT]]
    times["conditional_margins_seconds"] = perf_counter()-tick
    write_json(output/"conditional_margins.json", {"selected_observed_until240_root": winner.root_id,
               "removed_plan_limit": REMOVE_ONE_LIMIT, "margins": margins, "additivity_claim": False})
    total = perf_counter()-started
    times["serialization_and_other_seconds"] = max(0., total-sum(times.values()))
    times["total_seconds"] = total
    result = {"schema": "fixed-terminal-stage3-v1", "context": asdict(context), "root_count": len(roots),
              "root_metadata": {k: v for k, v in metadata.items() if k != "catalogue"},
              "observed_selections": selections,
              "realized_comparison_of_observed_until240_selection": next(row for row in rows if row["root_id"] == winner.root_id),
              "conditional_margin_status_counts": {s: sum(m["status"] == s for m in margins) for s in ("feasible", "infeasible", "unknown")},
              "timings": times, "errors": errors,
              "learning_or_teacher_training": False, "global_optimality_claim": False,
              "cross_world_same_denominator_claim": False}
    write_json(output/"result.json", result)
    print(json.dumps({"context": context.key, "phase": "completed", "seconds": total}), flush=True)
    return result


def aggregate(root, output):
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(root).glob("**/result.json"))]
    results = [row for row in results if row.get("schema") == "fixed-terminal-stage3-v1"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"context_count": len(results), "contexts": results,
                        "scope": "selected mechanism contexts only; no holdout performance claim"})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--soc", choices=("high", "medium", "low"), default="medium")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--time", type=float, default=0.)
    parser.add_argument("--stage2-root", type=Path, default=Path("tmp_stage2_results/episodes"))
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--list-contexts", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.list_contexts:
        print(json.dumps([asdict(c) for c in FORMAL_CONTEXTS]))
        return 0
    if args.output is None:
        parser.error("--output required for run or aggregation")
    if args.aggregate:
        aggregate(args.aggregate, args.output)
    else:
        run_context(args.output, Context(args.load, args.soc, args.seed, args.time), args.stage2_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
