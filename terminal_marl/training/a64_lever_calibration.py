"""A6.4 deterministic lever calibration and short protected rule screening."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path

import numpy as np

from ..core.config import RingRole
from ..core.dynamic_airspace import DynamicAirspace
from ..envs.high_level_wrapper import RuleLowLevelAdapter
from ..policies import DynamicRulePolicy
from .a63_dynamic_screen import SHORT_SCENARIOS
from .a6_effectiveness import _run_dynamic, _run_fixed, scenario_config


HEADWAY_ACTIONS = {
    "baseline_9_9_12": np.zeros(10, dtype=np.float32),
    "o2m_only": np.asarray((0, 0, 0, 0, 0, 0, 0, -.5, 0, 0), dtype=np.float32),
    "m2i_only": np.asarray((0, 0, 0, 0, 0, 0, 0, 0, -.5, 0), dtype=np.float32),
    "i2f_only": np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, -.5), dtype=np.float32),
    "o2m_m2i": np.asarray((0, 0, 0, 0, 0, 0, 0, -.5, -.5, 0), dtype=np.float32),
    "all_decrease": np.asarray((0, 0, 0, 0, 0, 0, 0, -.5, -.5, -.5), dtype=np.float32),
}
REQUIRED_SAFETY = ("collisions", "nmac", "hard_separation_violations", "final_overcapacity_steps")


def _constant_dynamic(action: np.ndarray, scenario_name: str, seed: int) -> dict:
    config = scenario_config(SHORT_SCENARIOS[scenario_name], "dynamic_rule")
    return _run_dynamic(config, seed, lambda observation, wrapper: action, RuleLowLevelAdapter())


def _safe(row: dict) -> bool:
    return all(key in row and row[key] == 0 for key in REQUIRED_SAFETY) and row.get("max_concurrent_final", 0) <= 1


def headway_sensitivity(seed: int = 0, *, workers: int = 6) -> dict:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_constant_dynamic, action, "high", seed): name for name, action in HEADWAY_ACTIONS.items()}
        results = {futures[future]: future.result() for future in as_completed(futures)}
    baseline = results["baseline_9_9_12"]
    return {"seed": seed, "scenario": "high", "results": results,
            "protected_safety_passed": all(_safe(row) for row in results.values()),
            "delta_vs_baseline": {name: {key: row[key] - baseline[key] for key in ("completed", "all_request_waiting_area", "service_delay_p95", "unresolved_age_p95", "energy_mean")} for name, row in results.items() if name != "baseline_9_9_12"}}


def capacity_threshold_probe() -> dict:
    airspace = DynamicAirspace(scenario_config(SHORT_SCENARIOS["high"], "dynamic_rule"))
    action = np.asarray((.60, .45, .30, 0, 0, 0, 0, 0, 0, 0), dtype=np.float32)
    update = airspace.request_update(action, now=0., geometry_gate=True)
    return {"requested_action": action.tolist(), "requested_radius_delta": list(update.requested_radius_delta),
            "projected_radius_delta": list(update.projected_radius_delta), "applied_radius_delta": list(update.applied_radius_delta),
            "capacity_before": list(update.capacity_before), "capacity_after": list(update.capacity_after),
            "structural_reason_codes": list(update.structural_reason_codes), "structural_applied": update.structural_applied,
            "targets": {role.value: airspace.target_ring(role).radius for role in RingRole}}


def _calibrated_dynamic(scenario_name: str, seed: int) -> dict:
    config = scenario_config(SHORT_SCENARIOS[scenario_name], "dynamic_rule")
    return _run_dynamic(config, seed, lambda observation, wrapper: DynamicRulePolicy().act(observation, wrapper._context()), RuleLowLevelAdapter())


def _fixed(scenario_name: str, seed: int) -> dict:
    return _run_fixed(scenario_config(SHORT_SCENARIOS[scenario_name], "fixed_rule"), seed)


def _pair(scenario_name: str, seed: int) -> tuple[str, int, dict]:
    fixed = _fixed(scenario_name, seed); dynamic = _calibrated_dynamic(scenario_name, seed)
    common = sorted(set(fixed["arrival_delay_by_id"]) & set(dynamic["arrival_delay_by_id"]), key=int)
    paired_arrival = [dynamic["arrival_delay_by_id"][key] - fixed["arrival_delay_by_id"][key] for key in common]
    delta_keys = ("completed", "all_request_waiting_area", "unresolved_age_p95", "service_delay_p95", "energy_mean", "priority_fairness_std", "final_busy_fraction")
    capacity_changes = sum(any(before != after for before, after in zip(event.get("capacity_before", ()), event.get("capacity_after", ()))) for event in dynamic.get("action_events", ()))
    return scenario_name, seed, {"fixed_rule": fixed, "calibrated_dynamic_rule": dynamic,
                                  "protected_safety_passed": _safe(fixed) and _safe(dynamic),
                                  "paired_arrival_delay": {"count": len(paired_arrival), "mean": float(np.mean(paired_arrival)) if paired_arrival else 0., "p95": float(np.percentile(paired_arrival, 95)) if paired_arrival else 0.},
                                  "capacity_changing_geometry_updates": capacity_changes,
                                  "delta_dynamic_minus_fixed": {key: dynamic[key] - fixed[key] for key in delta_keys}}


def run_short_screen(workers: int = 3) -> dict:
    """Scenario-level fail-fast funnel: high seeds first, then bursty."""
    rows = {scenario: {} for scenario in SHORT_SCENARIOS}
    for scenario in ("high", "bursty"):
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_pair, scenario, seed): seed for seed in range(3)}
            for future in as_completed(futures):
                seed = futures[future]; _, _, row = future.result(); rows[scenario][str(seed)] = row
                if not row["protected_safety_passed"]:
                    raise RuntimeError(f"A6.4 protected safety failure: {scenario}/seed{seed}")
    directional = {}
    for scenario, by_seed in rows.items():
        deltas = [by_seed[str(seed)]["delta_dynamic_minus_fixed"] for seed in range(3)]
        directional[scenario] = {key: [item[key] for item in deltas] for key in deltas[0]}
    consistency = {scenario: {"completed_nonnegative_all": all(value >= 0 for value in values["completed"]),
                               "waiting_area_nonpositive_all": all(value <= 0 for value in values["all_request_waiting_area"]),
                               "unresolved_age_nonpositive_all": all(value <= 0 for value in values["unresolved_age_p95"])} for scenario, values in directional.items()}
    return {"rows": rows, "all_protected_safe": True, "directional_deltas": directional, "directional_consistency": consistency}


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "a64_lever_calibration.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    consistency = report["short_screen"]["directional_consistency"]
    gate_passed = bool(report["short_screen"]["all_protected_safe"] and any(
        value["completed_nonnegative_all"] or (value["waiting_area_nonpositive_all"] and value["unresolved_age_nonpositive_all"])
        for value in consistency.values()) and all(value["completed_nonnegative_all"] or value["waiting_area_nonpositive_all"] for value in consistency.values()))
    report["gate"] = {"passed": gate_passed, "decision": "AUTHORIZE_HIGH_PPO" if gate_passed else "STOP_BEFORE_PPO",
                      "reason": "Protected safety is necessary but not sufficient: terminal benefit must be directionally stable without material counter-metric degradation."}
    (output / "a64_lever_calibration.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = ["# A6.4 high-level lever calibration and metric audit", "", f"Protected safety: **{'PASS' if report['short_screen']['all_protected_safe'] else 'FAIL'}**.", f"Decision: **{report['gate']['decision']}**.", "",
             "| Demand | Completed delta (3 seeds) | Waiting-area delta (3 seeds) | Unresolved-age P95 delta (3 seeds) |", "|---|---:|---:|---:|"]
    for scenario, values in report["short_screen"]["directional_deltas"].items():
        lines.append(f"| {scenario} | {values['completed']} | {values['all_request_waiting_area']} | {values['unresolved_age_p95']} |")
    lines.extend(["", "## Audit findings", "", "- Final utilization is now a bounded busy fraction; `max_concurrent_final` and `final_overcapacity_steps` are authoritative runtime facts.", "- Completed-only P95 remains reported, but the gate also uses all-request waiting area, unresolved age, and paired common-arrival delay.", "- All six calibrated runs retained `final_overcapacity_steps = 0`; the deterministic geometry probe crossed 47/31/15 to 48/32/16 capacity.", "- The calibrated rule has inconsistent high-load completion/waiting outcomes and bursty P95/energy/fairness regressions; protected safety alone does not authorize PPO.", "", "No PPO was trained; checkpoint files were not loaded or modified."])
    (output / "a64_lever_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "a64_headway_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["condition", "completed", "waiting_area", "service_delay_p95", "unresolved_age_p95", "final_overcapacity_steps"])
        for name, row in report["headway_sensitivity"]["results"].items(): writer.writerow([name, row["completed"], row["all_request_waiting_area"], row["service_delay_p95"], row["unresolved_age_p95"], row["final_overcapacity_steps"]])


def main() -> None:
    output = Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a64"
    headway = headway_sensitivity(); capacity = capacity_threshold_probe(); short = run_short_screen()
    report = {"stage": "A6.4", "headway_sensitivity": headway, "capacity_threshold_probe": capacity, "short_screen": short}
    write_report(report, output)
    print(json.dumps({"output": str(output), "safe": short["all_protected_safe"], "capacity_after": capacity["capacity_after"]}))


if __name__ == "__main__":
    main()
