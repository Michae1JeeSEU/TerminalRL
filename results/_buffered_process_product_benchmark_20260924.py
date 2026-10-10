"""Temporary product observer benchmark; output is an untracked results artifact."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
import time
from queue import Empty

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEPS = 20
SEED = 840626
OUTPUT = Path(__file__).with_suffix(".json")


def _gap_stats(times: list[float]) -> dict[str, float | int | None]:
    gaps = [right - left for left, right in zip(times, times[1:])]
    return {
        "frames": len(times),
        "effective_fps": (len(times) - 1) / sum(gaps) if gaps else None,
        "median_gap_seconds": statistics.median(gaps) if gaps else None,
        "p95_gap_seconds": float(np.percentile(gaps, 95)) if gaps else None,
    }


def _metrics_target(target, target_args, metrics_queue):
    """Wrap the production child target to collect render timestamps only."""
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    from terminal_marl.visualization import pyvista_viewer

    events = {"full": [], "dynamic": [], "interpolation": []}
    original_full = pyvista_viewer.Terminal3DViewer.update
    original_dynamic = pyvista_viewer.Terminal3DViewer.update_dynamic
    original_interpolation = pyvista_viewer.Terminal3DViewer.update_interpolated_positions

    def full(viewer, *args, **kwargs):
        result = original_full(viewer, *args, **kwargs)
        events["full"].append((time.perf_counter(), bool(kwargs.get("render", True))))
        return result

    def dynamic(viewer, *args, **kwargs):
        result = original_dynamic(viewer, *args, **kwargs)
        events["dynamic"].append(time.perf_counter())
        return result

    def interpolation(viewer, *args, **kwargs):
        result = original_interpolation(viewer, *args, **kwargs)
        events["interpolation"].append(time.perf_counter())
        return result

    pyvista_viewer.Terminal3DViewer.update = full
    pyvista_viewer.Terminal3DViewer.update_dynamic = dynamic
    pyvista_viewer.Terminal3DViewer.update_interpolated_positions = interpolation
    target(*target_args)
    metrics_queue.put(events)


class _ContextProxy:
    def __init__(self, real_context, metrics_queue):
        self._real_context = real_context
        self._metrics_queue = metrics_queue

    def Queue(self, maxsize=0):
        return self._real_context.Queue(maxsize=maxsize)

    def Event(self):
        return self._real_context.Event()

    def Value(self, *args):
        return self._real_context.Value(*args)

    def Process(self, *, target, args, name=None):
        return self._real_context.Process(
            target=_metrics_target,
            args=(target, args, self._metrics_queue),
            name=name,
        )


def _config():
    from terminal_marl.training.diagnose_energy_reachability_guard_smoke import build_config

    return build_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=True,
        max_simulation_seconds=float(STEPS),
        dense_warm_start_uavs=30,
        final_approach_geometry_mode="continuous_inward_v1",
    )


def _actions(env):
    return {str(uid): np.asarray([1.0, 0.0, 0.0], dtype=np.float32) for uid in env.active}


def _headless(config):
    from terminal_marl.envs.parallel_env import TerminalParallelEnv

    env = TerminalParallelEnv(config)
    env.reset(SEED)
    started = time.perf_counter()
    for _ in range(STEPS):
        env.step(_actions(env))
    elapsed = time.perf_counter() - started
    return {"steps": STEPS, "wall_seconds": elapsed, "steps_per_second": STEPS / elapsed}


def _product_observer(config, headless):
    from terminal_marl.envs.parallel_env import TerminalParallelEnv
    from terminal_marl.visualization import pyvista_viewer
    from terminal_marl.visualization.config import VisualizationConfig
    from terminal_marl.visualization.snapshot import build_world_render_snapshot

    env = TerminalParallelEnv(config)
    env.reset(SEED)
    initial = build_world_render_snapshot(env, trajectory_max_points=20)
    real_context = mp.get_context("spawn")
    metrics_queue = real_context.Queue(maxsize=1)
    pyvista_viewer.mp.get_context = lambda _name: _ContextProxy(real_context, metrics_queue)
    observer = pyvista_viewer.BufferedProcessTerminalViewerObserver(
        initial,
        VisualizationConfig(frame_stride=1, show_trajectories=False, trajectory_max_points=20),
    )
    startup_started = time.perf_counter()
    ready = observer.wait_ready(timeout=30.0)
    startup_elapsed = time.perf_counter() - startup_started
    source_times: list[float] = []
    simulation_started = time.perf_counter()
    if ready:
        for _ in range(STEPS):
            env.step(_actions(env))
            source_times.append(time.perf_counter())
            observer(env)
    simulation_elapsed = time.perf_counter() - simulation_started
    time.sleep(1.5)
    observer.close()
    try:
        events = metrics_queue.get(timeout=5.0)
    except Empty:
        events = {"error": "render metrics unavailable"}
    source_gaps = [b - a for a, b in zip(source_times, source_times[1:])]
    interpolation = events.get("interpolation", []) if isinstance(events, dict) else []
    return {
        "steps": STEPS,
        "ready": ready,
        "startup_wall_seconds": startup_elapsed,
        "simulation_wall_seconds": simulation_elapsed,
        "sim_steps_per_second": STEPS / simulation_elapsed if simulation_elapsed else None,
        "headless_steps_per_second": headless["steps_per_second"],
        "sim_slowdown_fraction": 1.0 - (STEPS / simulation_elapsed) / headless["steps_per_second"] if simulation_elapsed else None,
        "source_frames": len(source_times),
        "source_rate_hz": (len(source_times) - 1) / sum(source_gaps) if source_gaps else None,
        "interpolation": _gap_stats(interpolation),
        "full_updates": len(events.get("full", [])) if isinstance(events, dict) else None,
        "dynamic_updates": len(events.get("dynamic", [])) if isinstance(events, dict) else None,
        "process_exitcode": observer._process.exitcode if observer._process_started else None,
        "stop_requested": observer.stop_requested,
        "observer_error": repr(observer.error),
    }


def main():
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    mp.freeze_support()
    config = _config()
    headless = _headless(config)
    product = _product_observer(config, headless)
    report = {"headless": headless, "product_observer": product}
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
