"""Debug phase progression, release approvals, and trajectory safety locally.

This tool is intentionally simulation-only.  It does not train PPO and does
not connect to ROS2, Crazyswarm2, or Crazyflie hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_experiment_config
from ..core.uav import FlightPhase, UAVState
from ..envs.high_level_wrapper import HighLevelHRLWrapper
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import DynamicRuleHighLevelPolicy
from .ablation_runner import apply_demand_profile


DEBUG_POLICIES = ("rule-only", "dynamic-rule", "high-level-random")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "mode") and hasattr(value, "points"):
        return {
            "mode": getattr(value, "mode"),
            "point_count": len(getattr(value, "points")),
            "metadata": _jsonable(getattr(value, "metadata", {})),
        }
    return value


def _distance_to_next_stage(env: TerminalParallelEnv, aircraft: UAVState) -> float:
    if aircraft.finished:
        return 0.0
    if aircraft.phase == FlightPhase.APPROACH:
        return max(0.0, float(np.linalg.norm(aircraft.position[:2])) - env.config.approach_threshold_radius)
    if aircraft.phase == FlightPhase.RING and aircraft.assigned_ring not in (None, 0):
        target_ring = (
            0
            if aircraft.assigned_ring == env.vtac.inner_ring_index
            else aircraft.assigned_ring - 1
        )
        return float(env.vtac.distance_to_release_reference(aircraft, target_ring))
    if aircraft.phase == FlightPhase.FINAL:
        if not aircraft.final_entry_fix_reached:
            return float(
                np.linalg.norm(aircraft.position - env.vtac.final_entry_hold_point(aircraft))
            )
        return float(np.linalg.norm(aircraft.position - np.asarray(env.config.center, dtype=float)))
    return math.nan


def _latest_approval(env: TerminalParallelEnv, uid: int) -> dict:
    return (
        env.last_landing_approval.get(uid)
        or env.last_release_approval.get(uid)
        or {}
    )


def _phase_rows(env: TerminalParallelEnv, high_level_step: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aircraft in env.aircraft:
        approval = _latest_approval(env, aircraft.uid)
        diagnostics = approval.get("diagnostics") or {}
        rows.append(
            {
                "high_level_step": high_level_step,
                "low_level_step": env.step_count,
                "time": env.time,
                "uid": aircraft.uid,
                "phase": FlightPhase(aircraft.phase).name,
                "phase_id": int(aircraft.phase),
                "assigned_ring": -1 if aircraft.assigned_ring is None else aircraft.assigned_ring,
                "transition_active": bool(aircraft.transition_active),
                "transition_kind": aircraft.transition_kind or "",
                "final_entry_fix_reached": bool(aircraft.final_entry_fix_reached),
                "finished": bool(aircraft.finished),
                "x": float(aircraft.position[0]),
                "y": float(aircraft.position[1]),
                "z": float(aircraft.position[2]),
                "speed": float(aircraft.speed),
                "distance_to_next_stage": _distance_to_next_stage(env, aircraft),
                "last_target_stage": approval.get("target_stage", ""),
                "last_approved": approval.get("approved", ""),
                "last_reason": approval.get("reason") or "",
                "current_min_distance": diagnostics.get("current_min_distance"),
            }
        )
    return rows


def _approval_rows(env: TerminalParallelEnv, high_level_step: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    approvals = {
        **{(uid, "release"): approval for uid, approval in env.last_release_approval.items()},
        **{(uid, "landing"): approval for uid, approval in env.last_landing_approval.items()},
    }
    for (uid, approval_type), approval in approvals.items():
        diagnostics = approval.get("diagnostics") or {}
        trajectory = diagnostics.get("trajectory") or {}
        trajectory_diagnostics = trajectory.get("diagnostics") or {}
        kinematics = trajectory_diagnostics.get("kinematic_detail") or {}
        rows.append(
            {
                "high_level_step": high_level_step,
                "low_level_step": env.step_count,
                "time": env.time,
                "uid": uid,
                "approval_type": approval_type,
                "target_stage": approval.get("target_stage", ""),
                "approved": bool(approval.get("approved", False)),
                "decision": approval.get("decision", ""),
                "reason": approval.get("reason") or "",
                "violations": "|".join(approval.get("violations") or []),
                "current_min_distance": diagnostics.get("current_min_distance"),
                "trajectory_min_distance": diagnostics.get("trajectory_min_distance"),
                "safe_distance": diagnostics.get("safe_distance"),
                "trajectory_reason": trajectory.get("reason"),
                "max_speed_observed": kinematics.get("max_speed_observed"),
                "speed_max_allowed": kinematics.get("speed_max_allowed"),
                "max_acceleration_observed": kinematics.get("max_acceleration_observed"),
                "acceleration_max_allowed": kinematics.get("acceleration_max_allowed"),
                "speed_violation_segments": json.dumps(
                    kinematics.get("speed_violation_segments", [])
                ),
                "acceleration_violation_segments": json.dumps(
                    kinematics.get("acceleration_violation_segments", [])
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_debug_phase_transitions(
    *,
    config_name: str = "dynamic_hrl",
    demand_level: str = "medium",
    policy: str = "rule-only",
    initial_uavs: int | None = None,
    disable_runtime_arrivals: bool = False,
    high_level_steps: int = 20,
    output_dir: str | Path = "runs/debug_phase_transitions_local",
    seed: int = 0,
) -> dict[str, Any]:
    """Run a short dry-run rollout and persist phase/approval diagnostics."""

    if policy not in DEBUG_POLICIES:
        raise ValueError(f"policy must be one of: {', '.join(DEBUG_POLICIES)}")
    config = apply_demand_profile(load_experiment_config(config_name), demand_level)
    if initial_uavs is not None:
        config.initial_aircraft_count = int(initial_uavs)
        config.formation_rows = max(1, int(initial_uavs))
        config.formation_cols = 1
    if disable_runtime_arrivals:
        config.arrival_rate = 0.0
        config.max_active_uavs = max(config.initial_aircraft_count, 1)

    env = TerminalParallelEnv(config)
    wrapper = HighLevelHRLWrapper(env, low_level_policy="rule-only", seed=seed)
    rng = np.random.default_rng(seed)
    dynamic_rule = DynamicRuleHighLevelPolicy()
    wrapper.reset(seed=seed)

    phase_trace: list[dict[str, Any]] = []
    release_attempts: list[dict[str, Any]] = []
    for high_level_step in range(1, high_level_steps + 1):
        if policy == "high-level-random":
            action = rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
        elif policy == "dynamic-rule":
            action = dynamic_rule.act(env)
        else:
            action = np.zeros(8, dtype=np.float32)
        _observation, _reward, terminated, truncated, _info = wrapper.step(action)
        phase_trace.extend(_phase_rows(env, high_level_step))
        release_attempts.extend(_approval_rows(env, high_level_step))
        if terminated or truncated:
            break

    trajectory_diagnostics = [
        row
        for row in release_attempts
        if row.get("target_stage") and row.get("trajectory_reason") is not None
    ]
    metrics = env.metrics()
    summary = {
        "config_name": config_name,
        "traffic_demand_level": demand_level,
        "policy": policy,
        "seed": seed,
        "initial_uavs": config.initial_aircraft_count,
        "disable_runtime_arrivals": disable_runtime_arrivals,
        "high_level_steps_requested": high_level_steps,
        "high_level_steps_completed": (
            max((row["high_level_step"] for row in phase_trace), default=0)
        ),
        "metrics": _jsonable(metrics),
        "phase_counts": {
            name: int(sum(FlightPhase(aircraft.phase).name == name for aircraft in env.aircraft))
            for name in ("APPROACH", "RING", "FINAL", "FINISHED")
        },
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "phase_trace.csv", phase_trace)
    _write_csv(output / "release_attempts.csv", release_attempts)
    _write_csv(output / "trajectory_diagnostics.csv", trajectory_diagnostics)
    (output / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dynamic_hrl")
    parser.add_argument("--demand-level", default="medium")
    parser.add_argument("--policy", choices=DEBUG_POLICIES, default="rule-only")
    parser.add_argument("--initial-uavs", type=int, default=None)
    parser.add_argument("--disable-runtime-arrivals", action="store_true")
    parser.add_argument("--high-level-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/debug_phase_transitions_local"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = run_debug_phase_transitions(
        config_name=args.config,
        demand_level=args.demand_level,
        policy=args.policy,
        initial_uavs=args.initial_uavs,
        disable_runtime_arrivals=args.disable_runtime_arrivals,
        high_level_steps=args.high_level_steps,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
