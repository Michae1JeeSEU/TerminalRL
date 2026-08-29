"""Repeatable P0 simulation-performance benchmark.

This is deliberately a qualification utility, not a training entry point.  It
creates deterministic traffic witnesses and writes all profiles/reports to an
explicit output directory supplied by the caller.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
import json
import os
from pathlib import Path
import pstats
import time

import numpy as np

from terminal_marl.core.config import SimConfig
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.core.runtime_separation import swept_minimum_distance
from terminal_marl.core.runtime_traffic_safety import PredictiveSafetyShield
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.trajectory_time_parameterization import GeometricPath, TrajectoryConstraints, TrajectoryTimeParameterizer
from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL
from terminal_marl.core.uav import FlightPhase
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.visualization.config import VisualizationConfig
from terminal_marl.visualization.pyvista_viewer import ViewerStepObserver


def _config(name: str, count: int) -> SimConfig:
    schedule = tuple(0.0 for _ in range(count))
    base = SimConfig(
        experiment_mode="fixed_teacher_heavy" if name == "full_teacher" else "fixed_rule",
        reset_mode="warm_start_rings", warm_start_uavs=count,
        fixed_arrival_schedule=schedule, total_requested_uavs=count,
        max_active_uavs=max(60, count), max_simulation_seconds=240.0,
        teacher_enabled=name == "full_teacher",
        scheduler_mode="teacher_full" if name == "full_teacher" else "execution",
    )
    if name == "inbound_heavy":
        return replace(base, reset_mode="flow_from_empty", warm_start_uavs=0,
                       arrival_model="finite_cohort", fixed_arrival_schedule=tuple(float(i % 4) * 20.0 for i in range(count)))
    return base


def _actions(env: TerminalParallelEnv, name: str, step: int) -> dict[str, np.ndarray]:
    # Holder/observation witnesses remain in their rings.  Teacher and mixed
    # witnesses issue normal requests, exercising candidate generation.
    if name == "mixed":
        return {str(uid): np.asarray((1.0 if (uid + step) % 10 == 0 else 0.0, 0.0, 0.0), dtype=np.float32)
                for uid in env.active}
    if name == "full_teacher":
        target = max(env.active, default=-1)
        return {str(uid): np.asarray((1.0 if uid == target else 0.0, 0.0, 0.0), dtype=np.float32)
                for uid in env.active}
    release = 0.0 if name in {"holder_only", "observation_v2", "committed_heavy", "shield_risk", "gui_off", "gui_on"} else 1.0
    return {str(uid): np.asarray((release, 0.0, 0.0), dtype=np.float32) for uid in env.active}


class _LaggingQualificationViewer:
    """A deterministic slow renderer used to prove main-loop decoupling."""
    def __init__(self) -> None:
        self.frames = 0
        self.paused = False
        self.playback_speed = 0.0

    def update(self, _snapshot) -> bool:
        time.sleep(0.02)
        self.frames += 1
        return True


def _prepare(env: TerminalParallelEnv, name: str) -> ViewerStepObserver | None:
    if name == "committed_heavy":
        env.step({str(uid): np.asarray((1.0, 0.0, 0.0), dtype=np.float32) for uid in env.active})
    if name == "gui_on":
        try:
            viewer_config = VisualizationConfig(frame_stride=5, show_trajectories=False, trajectory_max_points=20)
        except TypeError:  # compatibility when profiling the pre-refactor revision
            viewer_config = VisualizationConfig(frame_stride=5, show_trajectories=False)
        observer = ViewerStepObserver(_LaggingQualificationViewer(), viewer_config)
        env.set_step_observer(observer)
        return observer
    if name == "shield_risk":
        holder, traffic = list(env.active.values())[:2]
        ring = env.airspace.actual_ring(holder.ring_role)
        endpoint = PredictiveSafetyShield(env.config)._holder_position(holder, env.airspace, 30.0, 0.0)
        start = np.asarray(traffic.position, dtype=float)
        velocity = (endpoint - start) / 30.0
        traffic.committed_plan = TrajectoryPlan(
            "qualification-shield-risk", traffic.uid, "transfer",
            (TrajectoryPoint(env.now, start, velocity, np.zeros(3)),
             TrajectoryPoint(env.now + 30.0, endpoint, velocity, np.zeros(3))),
            env.now, env.now + 30.0, metadata={"committed": True, "target_phase": "ring_middle"},
        )
        traffic.trajectory_plan_id = traffic.committed_plan.plan_id
        env.safety.checked_trajectory_hashes[traffic.committed_plan.plan_id] = traffic.committed_plan.trajectory_hash
        traffic.phase = FlightPhase.TRANSFER_O2M
        traffic.ring_role = None
    return None


def _process_metrics():
    cpu_seconds = time.process_time()
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        return True, cpu_seconds, int(getattr(memory, "peak_wset", memory.rss))
    except ImportError:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters(); counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            query = ctypes.windll.psapi.GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
            query.restype = wintypes.BOOL
            if not query(handle, ctypes.byref(counters), counters.cb):
                return True, cpu_seconds, 0
            return True, cpu_seconds, int(counters.PeakWorkingSetSize)
        return True, cpu_seconds, 0


def _components(rows: list[dict]) -> dict[str, float]:
    groups = {
        "Dynamics": ("vehicle_dynamics.py",),
        "TrajectoryGenerator": ("trajectory_generator.py", "trajectory_time_parameterization.py"),
        "SafetyFilter": ("safety_filter.py",),
        "Shield": ("runtime_traffic_safety.py",),
        "Teacher": ("forecast_world.py", "teacher_scheduler.py"),
        "Observation": ("observations.py", "topology_neighbors.py"),
        "Viewer": ("visualization", "snapshot.py", "pyvista_viewer.py"),
    }
    return {name: max((row["cumulative_s"] for row in rows if any(token in row["function"] for token in tokens)), default=0.0)
            for name, tokens in groups.items()}


def run_case(name: str, count: int, steps: int, output: Path) -> dict:
    config = _config(name, count)
    env = TerminalParallelEnv(config)
    env.reset(seed=17)
    observer = _prepare(env, name)
    profiler = cProfile.Profile()
    process, cpu_start, peak_start = _process_metrics()
    wall_start = time.perf_counter()
    profiler.enable()
    for step in range(steps):
        env.step(_actions(env, name, step))
    profiler.disable()
    wall = time.perf_counter() - wall_start
    if observer is not None:
        if hasattr(observer, "close"):
            observer.close()
    _, cpu_end, peak_end = _process_metrics()
    profile_path = output / f"{name}_n{count}.prof"
    profiler.dump_stats(profile_path)
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    rows = []
    for (file, line, function), values in list(stats.stats.items()):
        primitive, total, tottime, cumulative, callers = values
        rows.append({"function": f"{Path(file).name}:{line}({function})", "calls": total,
                     "self_s": round(tottime, 6), "cumulative_s": round(cumulative, 6)})
    rows.sort(key=lambda row: row["cumulative_s"], reverse=True)
    return {"scenario": name, "n": count, "steps": steps, "wall_seconds": wall,
            "simulated_seconds": steps * config.dt, "steps_per_second": steps / max(wall, 1e-9),
            "wall_per_simulated_second": wall / max(steps * config.dt, 1e-9),
            "cpu_seconds": cpu_end - cpu_start if process is not None else None,
            "peak_rss_bytes": max(peak_start, peak_end) if process is not None else None,
            "profile": str(profile_path), "top_cumulative": rows[:20],
            "components_cumulative_seconds": _components(rows), "telemetry": dict(env.telemetry)}


def safety_sampling_convergence() -> list[dict]:
    rows = []
    for sampling_dt in (0.2, 0.5, 1.0):
        config = SimConfig(reset_mode="warm_start_rings", warm_start_uavs=1,
                           fixed_arrival_schedule=(0.0,), total_requested_uavs=1,
                           safety_sampling_dt=sampling_dt)
        env = TerminalParallelEnv(config)
        env.reset(7)
        holder = next(iter(env.active.values()))
        ring = env.airspace.actual_ring(holder.ring_role)
        holder.position = np.asarray((ring.radius, 0.0, ring.height)); holder.ring_progress = 0.0
        shield = PredictiveSafetyShield(config)
        endpoint = shield._holder_position(holder, env.airspace, 10.0, 0.0)
        crossing = TrajectoryPlan("sampling-witness", 999, "transfer", (
            TrajectoryPoint(0.0, np.asarray((ring.radius * 2.0, 0.0, ring.height))),
            TrajectoryPoint(10.0, endpoint),
        ), 0.0, 10.0, metadata={"committed": True})
        env.active[999] = type("CommittedTraffic", (), {"uid": 999, "ring_role": None, "committed_plan": crossing})()
        assessment = shield.assess(env)
        shield.protect(env, {holder.uid: LowLevelIntent(0.0, 0.0, 0.0)})

        candidate = TrajectoryPlan("filter-witness", 1001, "hold", (
            TrajectoryPoint(0.0, np.asarray((0.0, 0.0, 0.0))),
            TrajectoryPoint(10.0, np.asarray((0.0, 0.0, 0.0))),
        ), 0.0, 10.0)
        other = TrajectoryPlan("filter-other", 1002, "hold", (
            TrajectoryPoint(0.0, np.asarray((60.0, 0.0, 0.0))),
            TrajectoryPoint(10.0, np.asarray((60.0, 0.0, 0.0))),
        ), 0.0, 10.0, metadata={"committed": True})
        snapshot = type("Snapshot", (), {"now": 0.0, "active": {}, "airspace": env.airspace})()
        verdict = SafetyFilter(config).filter_plan(candidate, {1002: other}, snapshot)
        rows.append({
            "safety_sampling_dt": sampling_dt, "safety_filter_approved": verdict.approved,
            "safety_filter_reason": verdict.reason, "shield_trigger": assessment.trigger_reason,
            "minimum_predicted_separation": assessment.predicted_min_swept_separation,
            "minimum_actual_separation": 60.0, "hard_separation_violation": True,
            "nmac": False, "collision": False,
            "shield_interventions": shield.telemetry["shield_interventions"],
            "shield_fail_closed": shield.telemetry["shield_fail_closed"],
        })
    return rows


def trajectory_sampling_convergence() -> dict[str, list[dict]]:
    parameterizer = TrajectoryTimeParameterizer(VehicleDynamics(RESEARCH_EVTOL))
    definitions = {
        "bezier": ((81, 41, 33), (np.zeros(3), np.asarray((80., 0., 20.)), np.asarray((160., 80., 50.)), np.asarray((220., 80., 60.))),
                   18.0, TrajectoryConstraints()),
        "final": ((101, 51, 41), (np.asarray((200., 0., 40.)), np.asarray((170., 90., 35.)), np.asarray((60., 70., 8.)), np.zeros(3)),
                  10.0, TrajectoryConstraints(end_horizontal_speed=0.0)),
    }
    report = {}
    for name, (densities, control, speed, constraints) in definitions.items():
        dense = GeometricPath.cubic_bezier(*control, samples=801).points
        rows = []
        for density in densities:
            path = GeometricPath.cubic_bezier(*control, samples=density)
            points = parameterizer.parameterize(path, start_time=0.0, initial_velocity=np.zeros(3),
                                                 target_horizontal_speed=speed, constraints=constraints)
            normalized_dense = np.linspace(0.0, 1.0, len(dense))
            normalized_path = np.linspace(0.0, 1.0, len(path.points))
            interpolated = np.column_stack([np.interp(normalized_dense, normalized_path, path.points[:, axis]) for axis in range(3)])
            offset = np.asarray((0.0, 0.0, 100.0))
            min_separation = float(np.min(np.linalg.norm(interpolated - (dense + offset), axis=1)))
            rows.append({
                "samples": density, "duration": points[-1].t,
                "max_curvature_xy": float(np.max(parameterizer._curvature_xy(path.points))),
                "max_acceleration": float(max(np.linalg.norm(point.acceleration) for point in points)),
                "dynamics_feasible": not parameterizer.dynamics.validate_plan(points),
                "minimum_separation": min_separation, "safety_filter_approved": min_separation >= 80.0,
                "dense_reference_tracking_error": float(np.max(np.linalg.norm(interpolated - dense, axis=1))),
            })
        report[name] = rows
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("holder_only", n) for n in (10, 30, 60)]
    cases += [("inbound_heavy", 30), ("mixed", 30), ("committed_heavy", 30),
              ("shield_risk", 30), ("gui_off", 30), ("gui_on", 30)]
    cases += [("observation_v2", n) for n in (10, 30, 60)]
    # Keep the legacy deepcopy-heavy reference last: a before-run can apply
    # an explicit timeout without losing all other scenario profiles.
    cases += [("full_teacher", 30)]
    report = {"label": args.label, "cases": [run_case(name, count, args.steps, args.output) for name, count in cases],
              "safety_sampling_convergence": safety_sampling_convergence(),
              "trajectory_sampling_convergence": trajectory_sampling_convergence()}
    (args.output / f"performance_{args.label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
