"""Exercise the real PyVista viewer without allowing it to pace simulation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from terminal_marl.core.config import SimConfig
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.visualization.config import VisualizationConfig
from terminal_marl.visualization.rollout_viewer import _attach_viewer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--off-screen", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.off_screen:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    config = SimConfig(
        experiment_mode="fixed_rule", reset_mode="warm_start_rings", warm_start_uavs=30,
        fixed_arrival_schedule=(0.0,) * 30, total_requested_uavs=30,
        max_active_uavs=60, max_simulation_seconds=240.0,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=17)
    render = VisualizationConfig(frame_stride=5, show_trajectories=False, trajectory_max_points=20)
    started = time.perf_counter()
    _, observer = _attach_viewer(env, render)
    ready = observer.wait_ready(timeout=20.0)
    startup = time.perf_counter() - started
    step_started = time.perf_counter()
    for _ in range(args.steps):
        env.step({str(uid): np.zeros(3, dtype=np.float32) for uid in env.active})
    step_wall = time.perf_counter() - step_started
    error = None if observer.error is None else repr(observer.error)
    observer.close()
    report = {
        "backend": "pyvista-off-screen" if args.off_screen else "pyvista-live",
        "ready": ready,
        "error": error,
        "steps": args.steps,
        "startup_wall_seconds": startup,
        "step_wall_seconds": step_wall,
        "steps_per_second": args.steps / max(step_wall, 1e-9),
        "frame_stride": render.frame_stride,
        "show_all_trajectories": render.show_trajectories,
        "trajectory_max_points": render.trajectory_max_points,
        "hard_separation": env.telemetry.get("hard_separation", 0),
        "nmac": env.telemetry.get("nmac", 0),
        "collisions": env.telemetry.get("collisions", 0),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not ready or error is not None:
        raise RuntimeError(f"viewer qualification failed: {report}")


if __name__ == "__main__":
    main()
