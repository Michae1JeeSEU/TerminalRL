"""Fail-fast saturated fixed-rule safety audit (A6.2)."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ..core.config import SimConfig
from ..envs.parallel_env import TerminalParallelEnv


DEMANDS = {
    "medium": (120, .040, 5200.),
    "high": (180, .080, 5000.),
    "bursty": (180, .040, 5500.),
}


def fixed_rule_config(demand: str) -> SimConfig:
    total, rate, maximum = DEMANDS[demand]
    return SimConfig(experiment_mode="fixed_rule", high_level_enabled=False, reset_mode="flow_from_empty",
                     arrival_model="bursty_wave" if demand == "bursty" else "finite_cohort",
                     total_requested_uavs=total, arrival_rate_per_second=rate, max_simulation_seconds=maximum)


def run_fail_fast(demand: str, seed: int, *, stop_on_first_hard_separation: bool = True, max_seconds: float | None = None) -> dict:
    config = fixed_rule_config(demand)
    if max_seconds is not None: config = replace(config, max_simulation_seconds=max_seconds)
    env = TerminalParallelEnv(config); env.reset(seed)
    for _ in range(int(env.config.max_simulation_seconds / env.config.dt)):
        _, _, _, _, info = env.step()
        if stop_on_first_hard_separation and env.telemetry["hard_separation_violations"]:
            trace = next(item for item in env.conflict_events if item.get("kind") == "first_hard_separation_trace")
            return {"safe": False, "demand": demand, "seed": seed, "time": env.now, "telemetry": dict(env.telemetry),
                    "first_conflict": trace, "ledger_closes": env.ledger.closes()}
        if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
            break
    safety = all(env.telemetry[key] == 0 for key in ("collisions", "nmac", "hard_separation_violations"))
    return {"safe": safety, "demand": demand, "seed": seed, "time": env.now, "telemetry": dict(env.telemetry),
            "first_conflict": None, "ledger_closes": env.ledger.closes(),
            "outcomes": {key: value.value for key, value in env.ledger.outcomes.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--demand", choices=DEMANDS, required=True); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--output"); parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(); result = run_fail_fast(args.demand, args.seed, max_seconds=args.max_seconds)
    if args.output: Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "first_conflict"}, default=str))


if __name__ == "__main__": main()
