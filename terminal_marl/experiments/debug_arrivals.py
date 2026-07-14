"""Low-cost debug tool for high-level traffic arrivals and count logging."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..config import load_experiment_config
from ..envs.high_level_wrapper import HighLevelHRLWrapper
from ..envs.parallel_env import TerminalParallelEnv
from ..core.uav import FlightPhase
from ..policies import DynamicRuleHighLevelPolicy
from .ablation_runner import apply_demand_profile


DEBUG_POLICIES = ("rule-only", "dynamic-rule", "high-level-random")


def _occupancy(env: TerminalParallelEnv) -> dict[str, float]:
    rings = sorted(env.config.rings, key=lambda ring: ring.radius, reverse=True)
    outer = rings[0]
    inner = rings[-1]
    return {
        "outer_occupancy": float(len(env.vtac.queues[outer.index])),
        "inner_occupancy": float(len(env.vtac.queues[inner.index])),
        "final_occupancy": float(len(env.vtac.queues[0])),
    }


def _phase_counts(env: TerminalParallelEnv) -> dict[str, float]:
    return {
        "approach_phase_count": float(
            sum(
                not aircraft.finished and aircraft.phase == FlightPhase.APPROACH
                for aircraft in env.aircraft
            )
        ),
        "ring_phase_count": float(
            sum(
                not aircraft.finished and aircraft.phase == FlightPhase.RING
                for aircraft in env.aircraft
            )
        ),
        "final_phase_count": float(
            sum(
                not aircraft.finished and aircraft.phase == FlightPhase.FINAL
                for aircraft in env.aircraft
            )
        ),
        "finished_phase_count": float(sum(aircraft.finished for aircraft in env.aircraft)),
    }


def _approval_summary(env: TerminalParallelEnv) -> dict[str, float | str]:
    approvals = [
        *env.last_release_approval.values(),
        *env.last_landing_approval.values(),
    ]
    rejected = [item for item in approvals if not item.get("approved", False)]
    reasons: dict[str, int] = {}
    for item in rejected:
        reason = str(item.get("reason") or "none")
        reasons[reason] = reasons.get(reason, 0) + 1
    top_reason = max(reasons.items(), key=lambda item: item[1])[0] if reasons else ""
    return {
        "release_attempt_count": float(len(approvals)),
        "release_reject_count": float(len(rejected)),
        "top_rejection_reason": top_reason,
    }


def run_debug_arrivals(
    *,
    config_name: str = "dynamic_hrl",
    demand_level: str = "medium",
    policy: str = "rule-only",
    high_level_steps: int = 8,
    output_dir: str | Path = "runs/debug_arrivals_local",
    seed: int = 0,
) -> list[dict[str, float | int | str]]:
    """Run a short high-level rollout and write per-interval count diagnostics."""

    if policy not in DEBUG_POLICIES:
        raise ValueError(f"policy must be one of: {', '.join(DEBUG_POLICIES)}")
    config = apply_demand_profile(load_experiment_config(config_name), demand_level)
    wrapper = HighLevelHRLWrapper(
        TerminalParallelEnv(config),
        low_level_policy="rule-only",
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    dynamic_rule = DynamicRuleHighLevelPolicy()
    observation, _ = wrapper.reset(seed=seed)
    rows: list[dict[str, float | int | str]] = []
    for step in range(1, high_level_steps + 1):
        if policy == "high-level-random":
            action = rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
        elif policy == "dynamic-rule":
            action = dynamic_rule.act(wrapper.env)
        else:
            action = np.zeros(8, dtype=np.float32)
        observation, reward, terminated, truncated, info = wrapper.step(action)
        diagnostics = info["reward_diagnostics"]
        row = {
            "config_name": config_name,
            "traffic_demand_level": demand_level,
            "policy": policy,
            "high_level_step": step,
            "reward": float(reward),
            "spawned_count_delta": float(diagnostics["spawned_count_delta"]),
            "active_count_start": float(diagnostics["active_count_start"]),
            "active_count_mean": float(diagnostics["active_count_mean"]),
            "active_count_max": float(diagnostics["active_count_max"]),
            "active_count_end": float(diagnostics["active_count_end"]),
            "landed_count_delta": float(diagnostics["landed_count_delta"]),
            "safety_override_delta": float(diagnostics["safety_override_delta"]),
            "raw_safety_override_delta": float(
                diagnostics.get("raw_safety_override_delta", 0.0)
            ),
            "total_spawned_count": float(diagnostics["total_spawned_count"]),
            "total_landed_count": float(diagnostics["total_landed_count"]),
            "total_safety_override_count": float(
                diagnostics["total_safety_override_count"]
            ),
            "total_raw_safety_override_count": float(
                diagnostics.get("total_raw_safety_override_count", 0.0)
            ),
            **_occupancy(wrapper.env),
            **_phase_counts(wrapper.env),
            **_approval_summary(wrapper.env),
        }
        rows.append(row)
        if terminated or truncated:
            break

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "config_name",
        "traffic_demand_level",
        "policy",
        "high_level_step",
        "reward",
        "spawned_count_delta",
        "active_count_start",
        "active_count_mean",
        "active_count_max",
        "active_count_end",
        "landed_count_delta",
        "safety_override_delta",
        "raw_safety_override_delta",
        "total_spawned_count",
        "total_landed_count",
        "total_safety_override_count",
        "total_raw_safety_override_count",
        "outer_occupancy",
        "inner_occupancy",
        "final_occupancy",
        "approach_phase_count",
        "ring_phase_count",
        "final_phase_count",
        "finished_phase_count",
        "release_attempt_count",
        "release_reject_count",
        "top_rejection_reason",
    ]
    with (output / "debug_arrivals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "debug_arrivals.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dynamic_hrl")
    parser.add_argument("--demand-level", default="medium")
    parser.add_argument("--policy", choices=DEBUG_POLICIES, default="rule-only")
    parser.add_argument("--high-level-steps", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/debug_arrivals_local"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = run_debug_arrivals(
        config_name=args.config,
        demand_level=args.demand_level,
        policy=args.policy,
        high_level_steps=args.high_level_steps,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
