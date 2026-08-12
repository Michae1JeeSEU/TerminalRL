"""A6.3 deterministic mechanism-sensitivity screen (not PPO training)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path

import numpy as np

from ..core.config import SimConfig
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
from ..envs.parallel_env import TerminalParallelEnv
from .a6_effectiveness import _collect, _run_fixed, _sample


HIGH_SEED = 0
HORIZON_SECONDS = 1800.0
CONDITIONS: dict[str, np.ndarray] = {
    "geometry_only": np.asarray((1, 1, 1, 0, 0, 0, 0, 0, 0, 0), dtype=np.float32),
    "headway_only": np.asarray((0, 0, 0, 0, 0, 0, 0, -1, -1, -1), dtype=np.float32),
    "geometry_headway": np.asarray((1, 1, 1, 0, 0, 0, 0, -1, -1, -1), dtype=np.float32),
}


def high_config(*, dynamic: bool) -> SimConfig:
    """High-load finite cohort used for all A6.3 deterministic conditions."""
    return SimConfig(
        experiment_mode="dynamic_rule" if dynamic else "fixed_rule",
        high_level_enabled=dynamic,
        reset_mode="flow_from_empty",
        arrival_model="finite_cohort",
        total_requested_uavs=180,
        arrival_rate_per_second=.080,
        max_simulation_seconds=HORIZON_SECONDS,
    )


def _short_metrics(result: dict) -> dict:
    required_safety = (
        "collisions", "nmac", "hard_separation_violations",
        "collision_exposure_steps", "nmac_exposure_steps", "hard_separation_exposure_steps",
    )
    if any(key not in result for key in required_safety):
        raise RuntimeError("A6.3 screen is fail-closed: missing authoritative safety metric")
    backlog = result["requested"] - result["completed"]
    return {
        "completed": result["completed"], "backlog": backlog,
        "external_queue_area": result["external_queue_area"],
        "external_queue_max": result["external_queue_max"],
        "service_delay_p95": result["service_delay_p95"],
        "holding_outer": result["holding_outer"], "holding_middle": result["holding_middle"],
        "holding_inner": result["holding_inner"], "energy_mean": result["energy_mean"],
        "final_utilization": result["final_utilization"], "lz_utilization": result["lz_utilization"],
        "effective_geometry_updates": result.get("effective_geometry_updates", 0),
        "headway_speed_updates": result.get("headway_speed_updates", 0),
        "runtime_traffic_interventions": result["runtime_traffic_interventions"],
        "runtime_traffic_repair_magnitude": result["runtime_traffic_repair_magnitude"],
        "minimum_swept_separation": result["minimum_swept_separation"],
        **{key: result[key] for key in required_safety},
    }


def _run_dynamic_constant(action: np.ndarray, seed: int) -> dict:
    config = high_config(dynamic=True)
    wrapper = DynamicHighLevelWrapper(config, RuleLowLevelAdapter())
    observation, _ = wrapper.reset(seed)
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity")}
    action_events: list[dict] = []
    while True:
        decision = wrapper.coordinator.coordinate(action, geometry_gate=wrapper.geometry_event_gate())
        action_events.append({
            "time": wrapper.base.now, "outcome": decision.outcome,
            "requested_action": list(decision.requested_action), "applied_action": list(decision.applied_action),
            "repair_count": decision.repair_count, "repair_magnitude": decision.repair_magnitude,
            "projection_magnitude": decision.projection_magnitude, "deferred_until": decision.deferred_until,
        })
        for _ in range(round(config.high_level_decision_interval_seconds / config.dt)):
            deferred = wrapper.coordinator.advance()
            if deferred is not None:
                action_events.append({
                    "time": wrapper.base.now, "outcome": deferred.outcome,
                    "requested_action": list(deferred.requested_action), "applied_action": list(deferred.applied_action),
                    "repair_count": deferred.repair_count, "repair_magnitude": deferred.repair_magnitude,
                    "projection_magnitude": deferred.projection_magnitude, "deferred_until": deferred.deferred_until,
                })
            low_actions = wrapper.coordinator.apply_speed_repairs(wrapper.low_level.act(wrapper.base._observations()))
            wrapper.base.step(low_actions)
            _sample(wrapper.base, samples)
            if wrapper._done() or wrapper.base.now >= config.max_simulation_seconds:
                break
        observation = wrapper.observation()
        if wrapper._done() or wrapper.base.now >= config.max_simulation_seconds:
            break
    del observation
    return _collect(wrapper.base, samples, wrapper.coordinator, action_events)


def run_condition(condition: str, seed: int = HIGH_SEED) -> dict:
    if condition == "nominal_fixed":
        return _short_metrics(_run_fixed(high_config(dynamic=False), seed))
    if condition not in CONDITIONS:
        raise ValueError(f"unknown A6.3 condition: {condition}")
    return _short_metrics(_run_dynamic_constant(CONDITIONS[condition], seed))


def _reproducible(left: dict, right: dict) -> bool:
    return all(np.isclose(left[key], right[key], equal_nan=True) for key in left)


def _positive_leverage(nominal: dict, candidate: dict) -> dict:
    """Require a material terminal benefit without a completion regression."""
    completion_gain = candidate["completed"] - nominal["completed"]
    queue_change = candidate["external_queue_area"] - nominal["external_queue_area"]
    delay_change = candidate["service_delay_p95"] - nominal["service_delay_p95"]
    positive = bool(
        candidate["completed"] >= nominal["completed"]
        and (completion_gain >= 1 or queue_change <= -.02 * max(1., nominal["external_queue_area"])
             or delay_change <= -.02 * max(1., nominal["service_delay_p95"]))
    )
    return {"positive": positive, "completed_delta": completion_gain,
            "queue_area_delta": queue_change, "p95_delay_delta": delay_change}


def run_screen(*, workers: int = 8) -> dict:
    conditions = ("nominal_fixed", *CONDITIONS)
    tasks = [(condition, repeat) for condition in conditions for repeat in range(2)]
    rows: dict[str, list[dict]] = {condition: [] for condition in conditions}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_condition, condition, HIGH_SEED): (condition, repeat) for condition, repeat in tasks}
        for future in as_completed(futures):
            condition, repeat = futures[future]
            rows[condition].append((repeat, future.result()))
    ordered = {condition: [row for _, row in sorted(values)] for condition, values in rows.items()}
    reproducible = {condition: _reproducible(*values) for condition, values in ordered.items()}
    primary = {condition: values[0] for condition, values in ordered.items()}
    protected = all(
        row[metric] == 0
        for row in primary.values()
        for metric in ("collisions", "nmac", "hard_separation_violations")
    )
    leverage = {condition: _positive_leverage(primary["nominal_fixed"], primary[condition]) for condition in CONDITIONS}
    return {
        "stage": "A6.3 deterministic mechanism-sensitivity screening",
        "scenario": {"demand": "high", "seed": HIGH_SEED, "horizon_seconds": HORIZON_SECONDS, "crn": True},
        "protected_safety_passed": protected,
        "repeat_identical": reproducible,
        "rows": ordered,
        "leverage_vs_nominal": leverage,
        "mechanism_leverage_proven": bool(protected and all(reproducible.values()) and any(item["positive"] for item in leverage.values())),
    }


def write_screen(report: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "a63_mechanism_screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = list(next(iter(report["rows"].values()))[0])
    with (output / "a63_mechanism_screen.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "repeat", *fields])
        writer.writeheader()
        for condition, values in report["rows"].items():
            for repeat, row in enumerate(values): writer.writerow({"condition": condition, "repeat": repeat, **row})
    lines = ["# A6.3 deterministic mechanism screen", "", f"Protected safety: **{'PASS' if report['protected_safety_passed'] else 'FAIL'}**.",
             f"Repeat-identical CRN reruns: **{'PASS' if all(report['repeat_identical'].values()) else 'FAIL'}**.",
             f"Mechanism leverage: **{'PROVEN' if report['mechanism_leverage_proven'] else 'NOT PROVEN'}**.", "",
             "| Condition | Completed delta | Queue-area delta | P95 service-delay delta | Positive |", "|---|---:|---:|---:|---:|"]
    for condition, values in report["leverage_vs_nominal"].items():
        lines.append(f"| {condition} | {values['completed_delta']:.0f} | {values['queue_area_delta']:.1f} | {values['p95_delay_delta']:.1f} | {values['positive']} |")
    (output / "a63_mechanism_screen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output = Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a63"
    report = run_screen()
    write_screen(report, output)
    print(json.dumps({"output": str(output), "protected_safety_passed": report["protected_safety_passed"],
                      "mechanism_leverage_proven": report["mechanism_leverage_proven"]}))


if __name__ == "__main__":
    main()
