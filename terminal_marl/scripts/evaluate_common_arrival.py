"""No-training common arrival evaluation; outputs must be outside the repo."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np

from terminal_marl.core.common_arrival import CommonArrivalConfig, generate_requests, request_digest
from terminal_marl.envs.common_arrival_env import CommonArrivalEnv
from terminal_marl.policies.common_arrival_baselines import make_policy, MissingCheckpointError


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(clean(rows))


def distribution(values):
    return {"count": len(values), "mean": float(np.mean(values)) if values else None,
            "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "max": max(values) if values else None}


def metrics(env, policy):
    completed = [a for a in env.aircraft.values() if a.completion_time is not None]
    observed = [a for a in env.aircraft.values()]
    unresolved = [a for a in observed if a.outcome is None]
    arrivals_due = sum(r.request_time <= env.now + 1e-8 for r in env.requests)
    latency = [c["seconds"] for c in policy.calls] if policy else []
    outcomes = {}
    for outcome in env.ledger.outcomes.values():
        outcomes[outcome.value] = outcomes.get(outcome.value, 0) + 1
    records = [{**asdict(a.request), "admission_time": a.admission_time,
                "capture_time": a.capture_time, "touchdown_time": a.touchdown_time,
                "completion_time": a.completion_time, "outcome_time": a.outcome_time, "outcome": a.outcome,
                "reason": a.reason, "final_phase": a.phase, "path_length_m": a.path_length,
                "energy_proxy": a.energy_consumed, "final_soc": env.soc(a)} for a in observed]
    result = {"schema": env.config.schema, "status": env.status, "stop_reason": env.stop_reason,
              "performance_valid": env.status in {"complete", "truncated"},
              "request_digest": request_digest(env.requests), "simulated_seconds": env.now,
              "planned_horizon_seconds": env.config.duration, "requested": len(env.requests),
              "requests_due": arrivals_due, "unspawned": len(env.requests) - env.next_request,
              "admitted": sum(a.admission_time is not None for a in observed),
              "completed": len(completed), "outcomes": outcomes,
              "unresolved_total": len(env.ledger.unresolved), "unresolved_due": len(unresolved),
              "conservation_ok": len(env.requests) == len(env.ledger.outcomes) + len(env.ledger.unresolved),
              "completed_fraction_due": len(completed) / max(1, arrivals_due),
              "finite_cohort_completions_per_elapsed_hour": len(completed) * 3600 / env.now if env.now else None,
              "completions_per_planned_hour": len(completed) * 3600 / env.config.duration,
              "request_to_completion_seconds": distribution([a.completion_time - a.request.request_time for a in completed]),
              "external_wait_seconds_admitted": distribution([a.admission_time - a.request.request_time for a in observed if a.admission_time is not None]),
              "unresolved_age_seconds": distribution([env.now - a.request.request_time for a in unresolved]),
              "total_time_in_system_observed_seconds": sum((a.outcome_time if a.outcome_time is not None else env.now) - a.request.request_time for a in observed),
              "integrated_airborne_seconds": env.integrated_airborne, "integrated_queue_seconds": env.integrated_queue,
              "max_airborne": env.max_airborne, "max_external_queue": env.max_queue,
              "pad_reserved_seconds": env.pad_reserved_seconds, "pad_service_seconds": env.pad_service_seconds,
              "path_length_m": sum(a.path_length for a in observed),
              "energy_proxy": sum(a.energy_consumed for a in observed),
              "shield_steps": env.intervention_steps, "requested_control_steps": env.requested_steps,
              "shield_step_fraction": env.intervention_steps / max(1, env.requested_steps),
              "resource_violations": env.resource_violations, "safety": dict(env.monitor.telemetry),
              "planner_latency_seconds": distribution(latency),
              "deadline_fallbacks": sum(c["deadline_fallback"] for c in policy.calls) if policy else 0,
              "solver_calls": len(policy.solver_records) if policy else 0,
              "solver_accepted": sum(r["accepted"] for r in policy.solver_records) if policy else 0,
              "airspace": {"area_m2": math.pi * env.config.radius ** 2,
                           "arrival_altitude_m": env.config.altitude,
                           "mandatory_rings": False, "pad_count": 1}}
    return clean(result), records


def run_one(config, tape, name, seed, output, checkpoint=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "requests.json", [asdict(r) for r in tape])
    env, policy = CommonArrivalEnv(config, tape), None
    start = time.perf_counter()
    error = None
    try:
        policy = make_policy(name, config, seed, checkpoint)
        while env.status == "running":
            env.prepare()
            env.step(policy.act(env))
    except MissingCheckpointError as exc:
        env.status, env.stop_reason = "skipped_missing_checkpoint", str(exc)
    except Exception:
        error = traceback.format_exc()
        env.status, env.stop_reason = "error", error.splitlines()[-1]
        (output / "error.txt").write_text(error, encoding="utf-8")
    summary, records = metrics(env, policy)
    summary.update(method=name, seed=seed, wall_seconds=time.perf_counter() - start)
    write_json(output / "metrics.json", summary)
    write_json(output / "metadata.json", {"config": asdict(config), "policy": policy.metadata if policy else None,
                                          "python": platform.python_version(),
                                          "versions": {pkg: version(pkg) for pkg in ("numpy", "scipy", "torch")}})
    write_json(output / "events.json", env.events)
    write_json(output / "planner_calls.json", policy.calls if policy else [])
    write_json(output / "solver_records.json", policy.solver_records if policy else [])
    for filename, rows in (("aircraft.csv", records), ("actions.csv", env.action_rows),
                           ("ticks.csv", env.tick_rows), ("trajectories.csv", env.trajectory_rows)):
        write_csv(output / filename, rows)
    return summary


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", nargs="+", default=["fcfs_rule", "yang_mcts_adapted", "saxena_milp_adapted", "waltz_actor_adapted"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261003])
    parser.add_argument("--output", required=True)
    parser.add_argument("--waltz-checkpoint")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if output == repo or repo in output.parents:
        parser.error("evaluation outputs must be outside the source repository")
    config = CommonArrivalConfig.read(args.config)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for seed in args.seeds:
        tape = generate_requests(config, seed)
        for name in args.methods:
            result = run_one(config, tape, name, seed, output / f"{name}_{seed}", args.waltz_checkpoint)
            summaries.append(result)
            write_json(output / "summary.json", summaries)
            print(json.dumps({k: result[k] for k in ("method", "seed", "status", "completed", "simulated_seconds", "stop_reason")}), flush=True)
    write_csv(output / "summary.csv", [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in summaries])
    return int(any(r["status"] in {"error", "fail_closed", "invalid_safety"} for r in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
