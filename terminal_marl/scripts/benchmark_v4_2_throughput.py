"""Short deterministic throughput benchmark for the V4.2 low-level runtime.

The benchmark is intentionally a measurement utility rather than a training
entry point.  It prints JSON to stdout and does not create output files.  The
same script can be rerun after a runtime change to obtain a paired benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action_v4_2 import (
    RELEASE_GATE,
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv


ACTION = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)


def make_config(
    warm_start_uavs: int,
    *,
    observation_schema_version: str = "phaseA_low_obs_v2_1_retry_state",
    neighbor_comm_range_m: float | None = None,
    initial_soc_min: float = 0.35,
    initial_soc_max: float = 0.55,
    warm_start_energy_check: bool = False,
    emergency_priority_guard_enabled: bool = False,
    energy_reachability_guard_enabled: bool = False,
) -> SimConfig:
    """Build the frozen short V4.2 phase-motion measurement scene."""
    return SimConfig(
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version=observation_schema_version,
        neighbor_comm_range_m=neighbor_comm_range_m,
        experiment_mode="fixed_low_student",
        high_level_enabled=False,
        reset_mode="warm_start_rings",
        warm_start_uavs=warm_start_uavs,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,) * warm_start_uavs,
        total_requested_uavs=warm_start_uavs,
        max_active_uavs=60,
        max_simulation_seconds=240.0,
        initial_soc_mode="uniform",
        initial_soc_min=initial_soc_min,
        initial_soc_max=initial_soc_max,
        lz_assignment_mode="nearest_available",
        release_arbitration_mode="downstream_priority",
        final_queue_capacity=3,
        energy_model_mode="phase_motion_v2",
        phase_motion_nominal_solo_energy=240.9,
        emergency_landing_enabled=True,
        emergency_priority_guard_enabled=emergency_priority_guard_enabled,
        energy_reachability_guard_enabled=energy_reachability_guard_enabled,
        warm_start_energy_reachability_check_enabled=warm_start_energy_check,
        emergency_trigger_soc=0.40,
        emergency_energy_margin_seconds=15.0,
        runtime_proposal_shortlist_k=3,
        safe_distance=80.0,
        nmac_distance=40.0,
        collision_distance=10.0,
        scheduler_mode="execution",
        teacher_enabled=False,
        teacher_release_label_enabled=False,
        shield_enabled=True,
    )


def actions_for(env: TerminalParallelEnv) -> dict[str, np.ndarray]:
    """Send the same explicit V4.2 RELEASE proposal to every active UAV."""
    return {str(uid): ACTION.copy() for uid in env.active}


def _stats(step_times: list[float]) -> dict[str, float]:
    values = np.asarray(step_times, dtype=float)
    total = float(values.sum())
    return {
        "steps": int(values.size),
        "wall_seconds": total,
        "mean_step_seconds": float(values.mean()),
        "median_step_seconds": float(np.median(values)),
        "p95_step_seconds": float(np.percentile(values, 95.0)),
        "steps_per_second": float(values.size / max(total, 1e-12)),
    }


def run_simulation(
    config: SimConfig,
    seed: int,
    *,
    repeats: int,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_step_times: list[float] = []
    for repeat in range(repeats):
        env = TerminalParallelEnv(config)
        env.reset(seed=seed)
        for _ in range(warmup_steps):
            env.step(actions_for(env))
        step_times: list[float] = []
        for _ in range(timed_steps):
            started = time.perf_counter()
            env.step(actions_for(env))
            step_times.append(time.perf_counter() - started)
        all_step_times.extend(step_times)
        row = _stats(step_times)
        row.update({
            "repeat": repeat,
            "seed": seed,
            "release_approved": int(env.telemetry.get("release_approved", 0)),
            "emergency_committed": int(env.telemetry.get("emergency_committed", 0)),
            "active_end": len(env.active),
            "active_after": len(env.active),
            "service_after": len(env.service_uavs),
            "emergency_touchdowns": int(env.telemetry.get("emergency_touchdowns", 0)),
            "hard_separation_violations": int(env.telemetry.get("hard_separation_violations", 0)),
            "energy_reachability_checks": int(env.telemetry.get("energy_reachability_checks", 0)),
            "energy_reachability_negative_budgets": int(
                env.telemetry.get("energy_reachability_negative_budgets", 0)
            ),
            "terminal_outcome_counts": dict(sorted(
                {
                    outcome.value: sum(
                        1 for recorded in env.ledger.outcomes.values() if recorded is outcome
                    )
                    for outcome in set(env.ledger.outcomes.values())
                }.items()
            )),
        })
        rows.append(row)
    return {"seed": seed, "repetitions": rows, "aggregate": _stats(all_step_times)}


def run_viewer(
    config: SimConfig,
    seed: int,
    *,
    warmup_steps: int,
    frames: int,
    steps_per_frame: int,
) -> dict[str, Any]:
    """Measure real off-screen PyVista updates, when the optional dependency works."""
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    try:
        from terminal_marl.visualization.config import VisualizationConfig
        from terminal_marl.visualization.pyvista_viewer import Terminal3DViewer
        from terminal_marl.visualization.snapshot import build_world_render_snapshot
    except Exception as error:  # pragma: no cover - optional dependency path
        return {"seed": seed, "available": False, "error": repr(error)}

    viewer = None
    try:
        env = TerminalParallelEnv(config)
        env.reset(seed=seed)
        render = VisualizationConfig(
            frame_stride=steps_per_frame,
            show_trajectories=False,
            trajectory_max_points=20,
        )
        initial = build_world_render_snapshot(
            env, trajectory_max_points=render.trajectory_max_points,
        )
        started = time.perf_counter()
        viewer = Terminal3DViewer(initial, render)
        startup_seconds = time.perf_counter() - started
        if not viewer.update(initial):
            raise RuntimeError("initial off-screen viewer update returned false")
        for _ in range(warmup_steps):
            env.step(actions_for(env))

        rows: list[dict[str, Any]] = []
        update_times: list[float] = []
        for frame in range(frames):
            frame_started = time.perf_counter()
            for _ in range(steps_per_frame):
                env.step(actions_for(env))
            snapshot_started = time.perf_counter()
            snapshot = build_world_render_snapshot(
                env, trajectory_max_points=render.trajectory_max_points,
            )
            snapshot_seconds = time.perf_counter() - snapshot_started
            update_started = time.perf_counter()
            ok = viewer.update(snapshot)
            update_seconds = time.perf_counter() - update_started
            update_times.append(update_seconds)
            rows.append({
                "frame": frame,
                "step_count": env.step_count,
                "snapshot_build_seconds": snapshot_seconds,
                "viewer_update_seconds": update_seconds,
                "frame_total_seconds": time.perf_counter() - frame_started,
                "viewer_update_ok": bool(ok),
                "uavs": len(snapshot.uavs),
                "landing_zones": len(snapshot.landing_zones),
            })
        return {
            "seed": seed,
            "available": True,
            "startup_seconds": startup_seconds,
            "frames": rows,
            "update_summary": _stats(update_times),
            "total_summary": _stats([row["frame_total_seconds"] for row in rows]),
        }
    except Exception as error:  # pragma: no cover - backend-dependent path
        return {"seed": seed, "available": False, "error": repr(error)}
    finally:
        if viewer is not None:
            viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="before")
    parser.add_argument("--seeds", nargs="+", type=int, default=(840600, 840601))
    parser.add_argument("--warm-start-uavs", type=int, default=12)
    parser.add_argument(
        "--observation-schema",
        default="phaseA_low_obs_v2_1_retry_state",
        choices=(
            "phaseA_low_obs_v2_1_retry_state",
            "phaseA_low_obs_v2_2_scheduler_context_94d",
            "phaseA_low_obs_v2_3_source_bound_88d",
        ),
    )
    parser.add_argument("--neighbor-comm-range-m", type=float, default=None)
    parser.add_argument("--initial-soc-min", type=float, default=0.35)
    parser.add_argument("--initial-soc-max", type=float, default=0.55)
    parser.add_argument("--warm-start-energy-check", action="store_true")
    parser.add_argument("--emergency-priority-guard", action="store_true")
    parser.add_argument("--energy-reachability-guard", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--viewer-frames", type=int, default=4)
    parser.add_argument("--steps-per-frame", type=int, default=5)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    if args.warm_start_uavs <= 0 or args.repeats <= 0 or args.warmup_steps < 0 or args.timed_steps <= 0:
        parser.error("warm-start UAVs, repeats, and timed steps must be positive; warmup cannot be negative")
    if not 0.0 < args.initial_soc_min <= args.initial_soc_max <= 1.0:
        parser.error("initial SOC bounds must satisfy 0 < min <= max <= 1")

    config = make_config(
        args.warm_start_uavs,
        observation_schema_version=args.observation_schema,
        neighbor_comm_range_m=args.neighbor_comm_range_m,
        initial_soc_min=args.initial_soc_min,
        initial_soc_max=args.initial_soc_max,
        warm_start_energy_check=args.warm_start_energy_check,
        emergency_priority_guard_enabled=args.emergency_priority_guard,
        energy_reachability_guard_enabled=args.energy_reachability_guard,
    )
    decoded = hierarchical_from_normalized_v4_2(ACTION)
    if decoded.gate != RELEASE_GATE:
        raise RuntimeError(f"benchmark action did not decode as RELEASE: {decoded.as_dict()}")

    simulation = [
        run_simulation(
            config,
            seed,
            repeats=args.repeats,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
        )
        for seed in args.seeds
    ]
    viewers = [] if args.no_viewer else [
        run_viewer(
            config,
            seed,
            warmup_steps=args.warmup_steps,
            frames=args.viewer_frames,
            steps_per_frame=args.steps_per_frame,
        )
        for seed in args.seeds
    ]
    print(json.dumps({
        "label": args.label,
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "action": ACTION.tolist(),
        "decoded_action": decoded.as_dict(),
        "config": {
            "action_schema_version": config.action_schema_version,
            "observation_schema_version": config.observation_schema_version,
            "neighbor_comm_range_m": config.neighbor_comm_range_m,
            "initial_soc_min": config.initial_soc_min,
            "initial_soc_max": config.initial_soc_max,
            "energy_model_mode": config.energy_model_mode,
            "reset_mode": config.reset_mode,
            "warm_start_uavs": config.warm_start_uavs,
            "warm_start_steady_motion": config.warm_start_steady_motion,
            "dt": config.dt,
            "physics_dt": config.physics_dt,
            "safety_sampling_dt": config.safety_sampling_dt,
            "safe_distance": config.safe_distance,
            "nmac_distance": config.nmac_distance,
            "collision_distance": config.collision_distance,
            "shield_enabled": config.shield_enabled,
            "shield_horizon_seconds": config.shield_horizon_seconds,
            "emergency_priority_guard_enabled": config.emergency_priority_guard_enabled,
            "energy_reachability_guard_enabled": config.energy_reachability_guard_enabled,
            "warm_start_energy_reachability_check_enabled": config.warm_start_energy_reachability_check_enabled,
        },
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "repetitions": args.repeats,
        "simulation": simulation,
        "viewer": viewers,
    }, indent=2))


if __name__ == "__main__":
    main()
