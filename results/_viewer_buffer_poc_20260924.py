"""One-off bounded presentation-buffer benchmark for the optional viewer."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from queue import Empty, Full

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEPS = 20
BUFFER_CAPACITY = int(os.environ.get("VIEWER_BUFFER_CAPACITY", "5"))
START_BUFFER = int(os.environ.get("VIEWER_START_BUFFER", "3"))
TRANSITION_SECONDS = float(os.environ.get("VIEWER_TRANSITION_SECONDS", "0.30"))
RENDER_HZ = 12.0
OUTPUT = Path(os.environ.get(
    "VIEWER_BUFFER_OUTPUT",
    str(Path(__file__).with_name("_viewer_buffer_poc_20260924.json")),
))


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
    return snapshot.__class__(
        time=snapshot.time,
        step_count=snapshot.step_count,
        rings=snapshot.rings,
        gates=snapshot.gates,
        landing_zones=snapshot.landing_zones,
        uavs=snapshot.uavs,
        external_queue_size=snapshot.external_queue_size,
        service_count=snapshot.service_count,
        completed_count=snapshot.completed_count,
        transition_active=snapshot.transition_active,
        transition_progress=snapshot.transition_progress,
        telemetry=dict(snapshot.telemetry),
        recent_conflict_uids=snapshot.recent_conflict_uids,
    )


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


def _renderer_process(frame_queue, control_queue, stop_event, ready_queue, metrics_queue, initial, render_config):
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    try:
        from terminal_marl.visualization import pyvista_viewer
        from terminal_marl.visualization.pyvista_viewer import Terminal3DViewer

        viewer = Terminal3DViewer(initial, render_config)
        ready_queue.put({"ready": True, "error": None})
        buffer: list[tuple[object, float]] = []
        displayed_snapshot = initial
        displayed_positions = tuple(uav.position for uav in initial.uavs)
        active = None
        playback_started = False
        simulation_start = None
        simulation_end = None
        latest_source_sim_time = initial.time
        latest_source_enqueued_at = time.perf_counter()
        max_displayed_source_lag = 0.0
        max_displayed_source_lag_wall = 0.0
        max_buffer_size = 0
        source_drops = 0
        interpolation_times: list[float] = []
        displayed_sim_times: list[float] = []
        next_tick = time.perf_counter()
        last_source_identity = pyvista_viewer._source_identity(initial)

        while not stop_event.is_set():
            now = time.perf_counter()
            if now < next_tick:
                if stop_event.wait(next_tick - now):
                    break
                continue
            next_tick += 1.0 / RENDER_HZ
            if next_tick < now:
                next_tick = now

            while True:
                try:
                    control, value = control_queue.get_nowait()
                except Empty:
                    break
                if control == "simulation_start":
                    simulation_start = float(value)
                elif control == "simulation_end":
                    simulation_end = float(value)

            while True:
                try:
                    item = frame_queue.get_nowait()
                except Empty:
                    break
                buffer.append(item)
                latest_source_sim_time = max(latest_source_sim_time, float(item[0].time))
                latest_source_enqueued_at = max(latest_source_enqueued_at, float(item[1]))
                while len(buffer) > BUFFER_CAPACITY:
                    buffer.pop(0)
                    source_drops += 1
                max_buffer_size = max(max_buffer_size, len(buffer))

            if not playback_started and len(buffer) >= START_BUFFER:
                playback_started = True

            if active is None and playback_started and buffer:
                target, _enqueued_at = buffer.pop(0)
                previous = displayed_snapshot
                previous_positions = displayed_positions
                start_positions = pyvista_viewer._interpolated_uav_positions(
                    previous, target, 0.0, previous_positions=previous_positions
                )
                target_identity = pyvista_viewer._source_identity(target)
                if target_identity != last_source_identity:
                    viewer.update(target, uav_positions=start_positions, render=False)
                    last_source_identity = target_identity
                else:
                    viewer.update_dynamic(target, start_positions, render=False)
                active = (previous, target, previous_positions, now, float(_enqueued_at))

            if active is not None:
                previous, target, previous_positions, started_at, target_enqueued_at = active
                alpha = min(1.0, max(0.0, (now - started_at) / TRANSITION_SECONDS))
                displayed_positions = pyvista_viewer._interpolated_uav_positions(
                    previous, target, alpha, previous_positions=previous_positions
                )
                if not viewer.update_interpolated_positions(target, displayed_positions):
                    break
                displayed_snapshot = target
                interpolation_times.append(now)
                displayed_sim_times.append(float(target.time))
                max_displayed_source_lag = max(
                    max_displayed_source_lag,
                    latest_source_sim_time - float(target.time),
                )
                max_displayed_source_lag_wall = max(
                    max_displayed_source_lag_wall,
                    latest_source_enqueued_at - target_enqueued_at,
                )
                if alpha >= 1.0:
                    active = None

        viewer.close()
        metrics_queue.put({
            "error": None,
            "playback_started": playback_started,
            "simulation_start": simulation_start,
            "simulation_end": simulation_end,
            "source_drops": source_drops,
            "max_buffer_size": max_buffer_size,
            "max_displayed_source_lag_sim_seconds": max_displayed_source_lag,
            "max_displayed_source_lag_wall_seconds": max_displayed_source_lag_wall,
            "displayed_sim_times": displayed_sim_times,
            "interpolation_times": interpolation_times,
        })
    except Exception as error:  # pragma: no cover - process failure path
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
    return {"steps": STEPS, "wall_seconds": elapsed, "steps_per_second": STEPS / elapsed}


def _process_run(config):
    from terminal_marl.envs.parallel_env import TerminalParallelEnv
    from terminal_marl.visualization.config import VisualizationConfig
    from terminal_marl.visualization.snapshot import build_world_render_snapshot

    env = TerminalParallelEnv(config)
    env.reset(seed=840626)
    initial = _plain_snapshot(build_world_render_snapshot(env, trajectory_max_points=20))
    context = mp.get_context("spawn")
    frame_queue = context.Queue(maxsize=BUFFER_CAPACITY)
    control_queue = context.Queue(maxsize=4)
    stop_event = context.Event()
    ready_queue = context.Queue(maxsize=1)
    metrics_queue = context.Queue(maxsize=1)
    render_config = VisualizationConfig(frame_stride=1, show_trajectories=False, trajectory_max_points=20)
    process = context.Process(
        target=_renderer_process,
        args=(frame_queue, control_queue, stop_event, ready_queue, metrics_queue, initial, render_config),
        name="terminal-viewer-buffer-poc",
    )
    startup_started = time.perf_counter()
    process.start()
    ready = ready_queue.get(timeout=30.0)
    startup = time.perf_counter() - startup_started
    counters = {"put": 0, "dropped": 0, "put_failed": 0}
    source_times: list[float] = []
    simulation_started = time.perf_counter()
    control_queue.put(("simulation_start", simulation_started))
    for _ in range(STEPS):
        env.step(_actions(env))
        source_times.append(time.perf_counter())
        snapshot = _plain_snapshot(build_world_render_snapshot(env, trajectory_max_points=20))
        _put_latest(frame_queue, (snapshot, time.perf_counter()), counters)
    simulation_ended = time.perf_counter()
    control_queue.put(("simulation_end", simulation_ended))
    simulation_wall = simulation_ended - simulation_started
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
        "ipc_queue_capacity": BUFFER_CAPACITY,
        "ipc_put_counters": counters,
        "telemetry": {
            key: env.telemetry.get(key)
            for key in ("hard_separation_violations", "nmac", "collisions")
        },
    })
    return metrics


def main() -> None:
    mp.freeze_support()
    config = _config()
    report = {"headless": _headless(config), "buffered_spawn_process": _process_run(config)}
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
