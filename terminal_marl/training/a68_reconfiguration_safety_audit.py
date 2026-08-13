"""A6.8 fail-fast reconfiguration transition safety closure; never trains PPO."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np

from .a62_safety_audit import run_fail_fast
from .a63_dynamic_screen import SHORT_SCENARIOS
from .a6_effectiveness import _run_dynamic, scenario_config
from ..envs.high_level_wrapper import RuleLowLevelAdapter
from ..policies import DynamicRulePolicy
from .a65_counterfactual_audit import available_cpus
from .a67_controllability_audit import TRACE_ACTIONS, _run_branch, capture_high_seed0


def _worker_init():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
    except ImportError:
        pass


def _branch_task(payload, action, horizon):
    started = time.perf_counter()
    row = _run_branch(payload, action, horizon)
    return action.name, row, time.perf_counter() - started


def _safe(row):
    return all(row["safety_delta"].get(name, -1) == 0 for name in (
        "collisions", "nmac", "hard_separation_violations", "final_overcapacity_steps",
    ))


def _compact(row):
    conflict = row.get("first_conflict") or {}
    history = row.get("candidate_audit", [])
    audit = next((item.get("candidate_audit", {}) for item in reversed(history) if item.get("candidate_audit")), {})
    geometry_changed = bool(any(abs(left-right) > 1e-6 for left, right in zip(row["geometry_before"], row["geometry_after"])))
    return {
        "action": row["action"], "decision": row["decision"], "safe": _safe(row),
        "safety_delta": row["safety_delta"], "effects": row["effects"],
        "predicted_min_separation": audit.get("prediction", {}).get("minimum_separation"),
        "predicted_track_count": audit.get("track_count"),
        "protected_window": audit.get("protected_window"),
        "runtime_min_separation": row["metrics"].get("minimum_swept_separation"),
        "first_conflict": {key: conflict.get(key) for key in ("time", "pair", "phase_pair", "actual_distance", "swept_min_distance")},
        "actual_geometry_after": row["after"]["actual_state_10d"][:6],
        "target_geometry_after": row["after"]["target_rh"],
        "headways_after": row["headways_after"],
        "capacity_after": row["capacity_after"],
        # A delayed target can execute later through the pending state
        # machine; actual geometry, not the first macro action alone, is the
        # authoritative execution signal.
        "transition_applied": geometry_changed,
    }


HISTORICAL_T720_486_735 = {
    "time": 985.0, "pair": [486, 735], "phase_pair": "TRANSFER-holder",
    "swept_min_distance": 79.735997, "classification": "pre_a68_static_or_postplan_gap",
}


def t720_geometry_conflict_regression(*, horizon: float = 540.0) -> dict:
    """Deterministic regression for the historical 486/735 geometry branch.

    The fixture records the pre-A6.8 failing encounter.  The current branch
    must keep the same CRN prefix hard-safe while allowing the pending target
    to execute only after it has passed the complete protected window.
    """
    state, payload = capture_high_seed0((720.,))[0]
    candidate = next(action for action in TRACE_ACTIONS if action.name == "outer_capacity_plus1")
    return {"fixture": HISTORICAL_T720_486_735, "snapshot": state,
            "current": _compact(_run_branch(payload, candidate, horizon))}


def focused_t720_screen(*, horizon: float = 540.0) -> dict:
    state, payload = capture_high_seed0((720.,))[0]
    tasks = [(payload, action, horizon) for action in TRACE_ACTIONS]
    workers = min(len(tasks), max(1, available_cpus() - 2))
    started = time.perf_counter(); rows = {}; durations = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        futures = [executor.submit(_branch_task, *task) for task in tasks]
        for future in as_completed(futures):
            name, row, duration = future.result(); rows[name] = _compact(row); durations.append(duration)
    telemetry = {"available_cpu": available_cpus(), "workers": workers, "branch_count": len(tasks),
                 "wall_time_seconds": time.perf_counter() - started,
                 "sequential_equivalent_seconds": float(sum(durations)),
                 "parallel_speedup": float(sum(durations) / max(1e-9, time.perf_counter() - started)),
                 "mean_branch_seconds": float(np.mean(durations)), "max_branch_seconds": float(max(durations))}
    geometry = rows.get("outer_capacity_plus1")
    regression = {"fixture": HISTORICAL_T720_486_735, "current": geometry}
    return {"snapshot": state, "branches": rows, "t720_geometry_regression": regression, "telemetry": telemetry,
            "safe": all(row["safe"] for row in rows.values()),
            "nonzero_geometry_executed": any(row["safe"] and row["transition_applied"] for row in rows.values())}


def fixed_short_regression() -> dict:
    """Fail-fast saturated fixed-rule funnel; never hides a later failure."""
    rows = {}
    for demand in ("medium", "high", "bursty"):
        row = run_fail_fast(demand, 0, max_seconds=600.)
        rows[demand] = row
        if not row["safe"]:
            rows["stopped_after"] = demand
            break
    return rows


def _dynamic_task(demand: str, seed: int) -> dict:
    # A6.8 stage 4 is deliberately a safety *screen*, not the long A6
    # effectiveness evaluation.  600 s includes several 180 s decisions
    # while preserving the fail-fast funnel and bounded audit cost.
    scenario = replace(SHORT_SCENARIOS[demand], max_seconds=600.)
    config = scenario_config(scenario, "dynamic_rule")
    row = _run_dynamic(config, seed, lambda observation, wrapper: DynamicRulePolicy().act(observation, wrapper._context()), RuleLowLevelAdapter())
    return {"demand": demand, "seed": seed, "horizon_seconds": scenario.max_seconds,
            "safe": all(row.get(key) == 0 for key in ("collisions", "nmac", "hard_separation_violations", "final_overcapacity_steps")),
            "metrics": row, "effective_geometry_updates": row.get("effective_geometry_updates", 0),
            "headway_speed_updates": row.get("headway_speed_updates", 0), "reconfiguration_execution_rate": row.get("reconfiguration_execution_rate", 0.)}


def _dynamic_task_timed(demand: str, seed: int):
    started = time.perf_counter()
    return demand, seed, _dynamic_task(demand, seed), time.perf_counter() - started


def dynamic_short_screen() -> dict:
    """Only run after t=720 and fixed-rule funnels pass; high then bursty."""
    rows = {}; started = time.perf_counter()
    # Required causal ordering: prove the first high and bursty branches
    # independently before spending CPU on the expanded seeds.
    for demand in ("high", "bursty"):
        row = _dynamic_task(demand, 0); rows[f"{demand}_seed0"] = row
        if not row["safe"]:
            return {"rows": rows, "stopped_after": f"{demand}_seed0", "telemetry": {"workers": 1, "wall_time_seconds": time.perf_counter()-started}}
    tasks = [(demand, seed) for demand in ("high", "bursty") for seed in (1, 2)]
    workers = min(len(tasks), max(1, available_cpus() - 2)); durations = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        futures = [executor.submit(_dynamic_task_timed, *item) for item in tasks]
        for future in as_completed(futures):
            demand, seed, row, duration = future.result(); rows[f"{demand}_seed{seed}"] = row; durations.append(duration)
    return {"rows": rows, "telemetry": {"workers": workers, "branch_count": len(tasks), "wall_time_seconds": time.perf_counter()-started,
                                             "parallel_seconds": float(sum(durations)), "parallel_speedup": float(sum(durations)/max(1e-9, time.perf_counter()-started))}}


def write_report(report, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "a68_reconfiguration_safety_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    # Retain both the historical failing trace and the current hard-safe
    # replay.  A bare ``null`` first-conflict field is not sufficient audit
    # evidence that the deterministic fixture was actually exercised.
    conflict = report.get("focused", {}).get("t720_geometry_regression", {})
    (output / "a68_conflict_regression.json").write_text(json.dumps(conflict, indent=2, default=str), encoding="utf-8")
    with (output / "a68_dynamic_branch_screen.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["kind", "action", "demand", "seed", "horizon_seconds", "safe", "transition_applied", "safety_delta", "predicted_min_separation", "runtime_min_separation", "effective_geometry_updates", "headway_speed_updates", "reconfiguration_execution_rate", "hard_separation_violations", "nmac", "collisions", "final_overcapacity_steps", "first_conflict"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        writer.writerows({"kind": "focused", **row} for row in report.get("focused", {}).get("branches", {}).values())
        for row in report.get("dynamic", {}).get("rows", {}).values():
            metrics = row.get("metrics", {})
            writer.writerow({"kind": "dynamic_rule", **row, **{key: metrics.get(key) for key in ("hard_separation_violations", "nmac", "collisions", "final_overcapacity_steps")}})
    focused = report["focused"]
    fixed = report.get("fixed", {})
    dynamic_rows = list(report.get("dynamic", {}).get("rows", {}).values())
    fixed_rows = [value for key, value in fixed.items() if key != "stopped_after"]
    dynamic_safe = bool(dynamic_rows) and all(row["safe"] for row in dynamic_rows)
    dynamic_effective = any(row["effective_geometry_updates"] > 0 for row in dynamic_rows)
    closure = "PASS" if focused.get("safe") and focused.get("nonzero_geometry_executed") and fixed_rows and all(row["safe"] for row in fixed_rows) and dynamic_safe and dynamic_effective else "FAIL"
    report["a68_safety_closure"] = closure
    (output / "a68_reconfiguration_safety_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = [
        "# A6.8 reconfiguration transition safety closure", "",
        f"A6.8 safety closure: **{closure}**", "",
        "1. Root cause: a speed-profile update could bypass the protected candidate check; the historical geometry case also lacked a full future handoff timeline.",
        "2. The old prediction omitted runtime-equivalent actual-geometry/post-handoff continuation, allowing a future TRANSFER-holder encounter to emerge after commit.",
        "3. Moving holders now use discrete executor-equivalent ring progress on actual R(t)/H(t), rather than a static-radius approximation.",
        "4. Every committed transfer continues on its target ring through max(transition end, committed plan end + post-handoff buffer); final plans end when they leave airborne state.",
        "5. The staged candidate (structural or operational) is atomically committed only after all active tracks pass kinematic and pairwise swept SafetyFilter checks.",
        f"6. Nonzero geometry safely executed: **{focused['nonzero_geometry_executed']}**.",
        "7. Predictor/runtime consistency: fail closed whenever a focused branch has a runtime protected-threshold entry.",
        f"8. Focused t=720 safety: **{focused.get('safe', False)}**; fixed short safety: **{bool(fixed_rows) and all(value['safe'] for value in fixed_rows)}**; dynamic short safety: **{dynamic_safe}**.",
        f"   Dynamic safe geometry executions: **{sum(row['effective_geometry_updates'] for row in dynamic_rows)}** across {len(dynamic_rows)} short CRN branches.",
        "9. PPO authorization: **STOP**.", "",
        f"Parallel telemetry: {focused['telemetry']}",
        "Evidence type: newly_run safety diagnostic; no PPO checkpoint was loaded, changed, or trained.",
        "Protocol risk: this is a deterministic, internal safety screen rather than an effectiveness/generalization result.",
    ]
    (output / "a68_safety_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default=None); parser.add_argument("--stage", choices=("focused", "fixed", "dynamic", "summary"), default="focused")
    args = parser.parse_args(); output = Path(args.output) if args.output else Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a68"
    if args.stage == "focused":
        focused = focused_t720_screen(); report = {"focused": focused, "fixed": {}, "a68_safety_closure": "FAIL"}
    else:
        prior = output / "a68_reconfiguration_safety_audit.json"
        report = json.loads(prior.read_text(encoding="utf-8")) if prior.exists() else {"focused": {}}
        if args.stage == "fixed": report["fixed"] = fixed_short_regression()
        elif args.stage == "dynamic":
            fixed = report.get("fixed", {})
            if not report.get("focused", {}).get("safe") or not report.get("focused", {}).get("nonzero_geometry_executed") or not fixed or not all(row["safe"] for key, row in fixed.items() if key != "stopped_after"):
                raise RuntimeError("A6.8 dynamic screen blocked: focused/fixed safety gates did not pass")
            report["dynamic"] = dynamic_short_screen()
        report["a68_safety_closure"] = "FAIL"
    write_report(report, output); print(json.dumps({"stage": args.stage, "output": str(output), "closure": report["a68_safety_closure"]}))


if __name__ == "__main__": main()
