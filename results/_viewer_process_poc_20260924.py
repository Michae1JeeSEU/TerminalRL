"""One-off Windows spawn-process viewer benchmark; never used by training."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import replace
from pathlib import Path
import statistics
import sys
import time
import traceback
from queue import Empty, Full

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


STEPS = 20
OUTPUT = Path(__file__).with_name("_viewer_process_poc_20260924.json")


def _gap_stats(times: list[float]) -> dict[str, float | int | None]:
    gaps = [right - left for left, right in zip(times, times[1:])]
    return {
        "frames": len(times),
        "effective_fps": (len(times) - 1) / sum(gaps) if gaps else None,
        "median_gap_seconds": statistics.median(gaps) if gaps else None,
        "p95_gap_seconds": float(np.percentile(gaps, 95)) if gaps else None,
        "min_gap_seconds": min(gaps) if gaps else None,
        "max_gap_seconds": max(gaps) if gaps else None,
    }


def _plain_snapshot(snapshot):
    """Make the mapping field explicitly pickle-friendly for multiprocessing."""
    return replace(snapshot, telemetry=dict(snapshot.telemetry))


def _put_latest(queue, item, counters: dict[str, int]) -> None:
    try:
        queue.put_nowait(item)
        counters["put"] += 1
        return
    except Full:
        counters["dropped"] += 1
    try:
        queue.get_nowait()
    except Empty:
        pass
    try:
        queue.put_nowait(item)
        counters["put"] += 1
    except Full:
        counters["put_failed"] += 1


def _renderer_process(frame_queue, stop_event, ready_queue, metrics_queue, initial, config):
    """Run the existing viewer in a Windows-spawned child process."""
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    events = {"full": [], "dynamic": [], "interpolation": []}
    try:
        from terminal_marl.visualization import pyvista_viewer
        from terminal_marl.visualization.pyvista_viewer import AsyncTerminalViewerObserver

        original_full = pyvista_viewer.Terminal3DViewer.update
        original_dynamic = pyvista_viewer.Terminal3DViewer.update_dynamic
        original_interpolation = pyvista_viewer.Terminal3DViewer.update_interpolated_positions

        def full(viewer, snapshot, *args, **kwargs):
            started = time.perf_counter()
            result = original_full(viewer, snapshot, *args, **kwargs)
            events["full"].append((started, kwargs.get("render", True)))
            return result

        def dynamic(viewer, snapshot, *args, **kwargs):
            started = time.perf_counter()
            result = original_dynamic(viewer, snapshot, *args, **kwargs)
            events["dynamic"].append(started)
            return result

        def interpolation(viewer, snapshot, *args, **kwargs):
            started = time.perf_counter()
            result = original_interpolation(viewer, snapshot, *args, **kwargs)
            events["interpolation"].append(started)
            return result

        pyvista_viewer.Terminal3DViewer.update = full
        pyvista_viewer.Terminal3DViewer.update_dynamic = dynamic
        pyvista_viewer.Terminal3DViewer.update_interpolated_positions = interpolation
        observer = AsyncTerminalViewerObserver(initial, config)
        ready = observer.wait_ready(timeout=20.0)
        ready_queue.put({"ready": ready, "error": None if observer.error is None else repr(observer.error)})
        if not ready:
            return
        forwarded = 0
        forwarded_dropped = 0
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.02)
            except Empty:
                continue
            try:
                observer._frames.put_nowait(item)
                forwarded += 1
            except Full:
                forwarded_dropped += 1
                try:
                    observer._frames.get_nowait()
                except Empty:
                    pass
                try:
                    observer._frames.put_nowait(item)
                    forwarded += 1
                except Full:
                    pass
        error = None if observer.error is None else repr(observer.error)
        stop_requested = observer.stop_requested
        observer.close()
        metrics_queue.put({
            "error": error,
            "stop_requested": stop_requested,
            "forwarded": forwarded,
            "forwarded_dropped": forwarded_dropped,
            "queue_remaining": observer._frames.qsize(),
            "full_calls": len(events["full"]),
            "full_render_calls": sum(1 for _, render in events["full"] if render),
            "dynamic_calls": len(events["dynamic"]),
            "interpolation_times": events["interpolation"],
        })
    except Exception as error:  # pragma: no cover - exercised only on process failure
        metrics_queue.put({"error": repr(error), "traceback": traceback.format_exc()})


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
    return {
        str(uid): np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        for uid in env.active
    }


def _headless(config):
    from terminal_marl.envs.parallel_env import TerminalParallelEnv

    env = TerminalParallelEnv(config)
    env.reset(seed=840626)
    started = time.perf_counter()
    for _ in range(STEPS):
        env.step(_actions(env))
    elapsed = time.perf_counter() - started
    return {
        "steps": STEPS,
        "wall_seconds": elapsed,
        "steps_per_second": STEPS / elapsed,
    }


def _process_run(config):
    from terminal_marl.envs.parallel_env import TerminalParallelEnv
    from terminal_marl.visualization.config import VisualizationConfig
    from terminal_marl.visualization.snapshot import build_world_render_snapshot

    env = TerminalParallelEnv(config)
    env.reset(seed=840626)
    initial = _plain_snapshot(build_world_render_snapshot(env, trajectory_max_points=20))
    context = mp.get_context("spawn")
    frame_queue = context.Queue(maxsize=1)
    stop_event = context.Event()
    ready_queue = context.Queue(maxsize=1)
    metrics_queue = context.Queue(maxsize=1)
    render_config = VisualizationConfig(frame_stride=1, show_trajectories=False, trajectory_max_points=20)
    process = context.Process(
        target=_renderer_process,
        args=(frame_queue, stop_event, ready_queue, metrics_queue, initial, render_config),
        name="terminal-viewer-process-poc",
    )
    startup_started = time.perf_counter()
    process.start()
    ready = ready_queue.get(timeout=30.0)
    startup = time.perf_counter() - startup_started
    queue_counters = {"put": 0, "dropped": 0, "put_failed": 0}
    source_times: list[float] = []
    started = time.perf_counter()
    for _ in range(STEPS):
        env.step(_actions(env))
        source_times.append(time.perf_counter())
        snapshot = _plain_snapshot(build_world_render_snapshot(env, trajectory_max_points=20))
        _put_latest(frame_queue, (snapshot, time.perf_counter()), queue_counters)
    simulation_wall = time.perf_counter() - started
    time.sleep(1.0)
    stop_event.set()
    process.join(timeout=10.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2.0)
    try:
        metrics = metrics_queue.get(timeout=2.0)
    except Empty:
        metrics = {"error": "renderer metrics unavailable"}
    source_gaps = [right - left for left, right in zip(source_times, source_times[1:])]
    interpolation_times = metrics.pop("interpolation_times", [])
    metrics["interpolation"] = _gap_stats(interpolation_times)
    metrics.update({
        "ready": ready.get("ready"),
        "startup_wall_seconds": startup,
        "simulation_wall_seconds": simulation_wall,
        "simulation_steps_per_second": STEPS / simulation_wall,
        "process_exitcode": process.exitcode,
        "source_frames": len(source_times),
        "source_rate_hz": (len(source_times) - 1) / sum(source_gaps),
        "source_median_gap_seconds": statistics.median(source_gaps),
        "source_p95_gap_seconds": float(np.percentile(source_gaps, 95)),
        "ipc_queue_maxsize": frame_queue._maxsize,
        "ipc_put_counters": queue_counters,
        "telemetry": {
            key: env.telemetry.get(key)
            for key in ("hard_separation_violations", "nmac", "collisions")
        },
    })
    return metrics


def main() -> None:
    mp.freeze_support()
    config = _config()
    report = {"headless": _headless(config), "spawn_process": _process_run(config)}
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
