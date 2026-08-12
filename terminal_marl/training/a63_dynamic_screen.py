"""Sequential A6.3 dynamic-rule safety/effectiveness smoke, never a formal gate."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from ..policies import DynamicRulePolicy
from ..envs.high_level_wrapper import RuleLowLevelAdapter
from .a6_effectiveness import DemandScenario, _run_dynamic, _run_fixed, scenario_config


SHORT_SCENARIOS = {
    "high": DemandScenario("high", 180, .080, 1800.),
    "bursty": DemandScenario("bursty", 180, .040, 1800., "bursty_wave"),
}
REQUIRED_SAFETY = ("collisions", "nmac", "hard_separation_violations")


def _fixed(scenario: DemandScenario, seed: int) -> dict:
    return _run_fixed(scenario_config(scenario, "fixed_rule"), seed)


def _dynamic(scenario: DemandScenario, seed: int) -> dict:
    config = scenario_config(scenario, "dynamic_rule")
    return _run_dynamic(config, seed, lambda observation, wrapper: DynamicRulePolicy().act(observation, wrapper._context()), RuleLowLevelAdapter())


def _safe(row: dict) -> bool:
    return all(key in row and row[key] == 0 for key in REQUIRED_SAFETY)


def run_pair(scenario_name: str, seed: int = 0, *, workers: int = 2) -> dict:
    scenario = SHORT_SCENARIOS[scenario_name]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        fixed_future = executor.submit(_fixed, scenario, seed)
        dynamic_future = executor.submit(_dynamic, scenario, seed)
        fixed, dynamic = fixed_future.result(), dynamic_future.result()
    return {
        "scenario": scenario_name, "seed": seed, "horizon_seconds": scenario.max_seconds, "crn": True,
        "fixed_rule": fixed, "dynamic_rule": dynamic,
        "protected_safety_passed": _safe(fixed) and _safe(dynamic),
        "delta_dynamic_minus_fixed": {
            "completed": dynamic["completed"] - fixed["completed"],
            "throughput": dynamic["throughput"] - fixed["throughput"],
            "external_queue_area": dynamic["external_queue_area"] - fixed["external_queue_area"],
            "service_delay_p95": dynamic["service_delay_p95"] - fixed["service_delay_p95"],
            "effective_geometry_updates": dynamic["effective_geometry_updates"],
            "headway_speed_updates": dynamic["headway_speed_updates"],
        },
    }


def write_pair(result: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    name = f"a63_dynamic_{result['scenario']}_seed{result['seed']}.json"
    (output / name).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("scenario", choices=tuple(SHORT_SCENARIOS)); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_pair(args.scenario, args.seed)
    output = Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a63"
    write_pair(result, output)
    print(json.dumps({"scenario": args.scenario, "seed": args.seed, "protected_safety_passed": result["protected_safety_passed"], **result["delta_dynamic_minus_fixed"]}))
    if not result["protected_safety_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
