"""Collect derivative consistency values for the four final-approach fixtures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "tests" / "test_final_approach_proposal_execution.py"
OUTPUT_PATH = ROOT / ".tmp" / "p3_trajectory_derivative_consistency_20261002.json"
sys.path.insert(0, str(ROOT))


def _load_fixture_module():
    spec = importlib.util.spec_from_file_location("final_approach_fixture", TEST_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture module: {TEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vector(value) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=float).tolist()]


def _point(point) -> dict:
    return {
        "t": float(point.t),
        "position": _vector(point.position),
        "velocity": _vector(point.velocity),
        "acceleration": _vector(point.acceleration),
    }


def _segment(left, right, index: int) -> dict:
    dt = float(right.t - left.t)
    dx = np.asarray(right.position, dtype=float) - np.asarray(left.position, dtype=float)
    chord_velocity = dx / dt
    average_velocity = (
        np.asarray(left.velocity, dtype=float) + np.asarray(right.velocity, dtype=float)
    ) / 2.0
    velocity_difference = chord_velocity - average_velocity
    dx_minus_average_v_dt = dx - average_velocity * dt
    return {
        "index": int(index),
        "left": _point(left),
        "right": _point(right),
        "dt": dt,
        "dx": _vector(dx),
        "chord_velocity_dx_dt": _vector(chord_velocity),
        "average_reported_velocity": _vector(average_velocity),
        "chord_minus_average_velocity": _vector(velocity_difference),
        "chord_minus_average_velocity_norm": float(np.linalg.norm(velocity_difference)),
        "dx_minus_average_v_dt": _vector(dx_minus_average_v_dt),
        "dx_minus_average_v_dt_norm": float(np.linalg.norm(dx_minus_average_v_dt)),
    }


def _execution_grid(plan, physics_dt: float) -> tuple[np.ndarray, bool]:
    times = [float(plan.start_time)]
    step_index = 1
    while True:
        candidate = float(plan.start_time + step_index * physics_dt)
        if candidate >= float(plan.end_time) - 1.0e-12:
            break
        times.append(candidate)
        step_index += 1
    end_appended = not np.isclose(times[-1], float(plan.end_time), rtol=0.0, atol=1.0e-12)
    if end_appended:
        times.append(float(plan.end_time))
    else:
        times[-1] = float(plan.end_time)
    return np.asarray(times, dtype=float), end_appended


def _max_speed_record(values: np.ndarray, times: np.ndarray, positions: np.ndarray, *, vertical: bool) -> dict:
    magnitudes = np.abs(values[:, 2]) if vertical else np.linalg.norm(values[:, :2], axis=1)
    index = int(np.argmax(magnitudes))
    return {
        "interval_index": index,
        "value": float(magnitudes[index]),
        "signed_vertical_value": float(values[index, 2]) if vertical else None,
        "vector": _vector(values[index]),
        "time_start": float(times[index]),
        "time_end": float(times[index + 1]),
        "position_start": _vector(positions[index]),
        "position_end": _vector(positions[index + 1]),
    }


def _max_acceleration_record(values: np.ndarray, interval_midpoints: np.ndarray, *, vertical: bool) -> dict:
    magnitudes = np.abs(values[:, 2]) if vertical else np.linalg.norm(values[:, :2], axis=1)
    index = int(np.argmax(magnitudes))
    return {
        "between_interval_index": index,
        "value": float(magnitudes[index]),
        "signed_vertical_value": float(values[index, 2]) if vertical else None,
        "vector": _vector(values[index]),
        "midpoint_time": float(interval_midpoints[index]),
    }


def _execution_sample_metrics(plan, env) -> dict:
    physics_dt = float(env.config.physics_dt)
    times, end_appended = _execution_grid(plan, physics_dt)
    positions = np.asarray([plan.sample_position(float(time)) for time in times], dtype=float)
    interval_dt = np.diff(times)
    interval_velocity = np.diff(positions, axis=0) / interval_dt[:, None]
    interval_midpoints = (times[:-1] + times[1:]) / 2.0
    midpoint_dt = np.diff(interval_midpoints)
    interval_acceleration = np.diff(interval_velocity, axis=0) / midpoint_dt[:, None]
    acceleration_midpoints = (interval_midpoints[:-1] + interval_midpoints[1:]) / 2.0
    profile = env.config.vehicle_profile
    return {
        "physics_dt": physics_dt,
        "sample_times": [float(value) for value in times],
        "sample_positions": [_vector(value) for value in positions],
        "sample_count": int(len(times)),
        "interval_count": int(len(interval_dt)),
        "interval_dt": [float(value) for value in interval_dt],
        "interval_midpoints": [float(value) for value in interval_midpoints],
        "interval_dx": [_vector(value) for value in np.diff(positions, axis=0)],
        "interval_velocity_dx_dt": [_vector(value) for value in interval_velocity],
        "interval_acceleration_between_midpoints": [
            _vector(value) for value in interval_acceleration
        ],
        "end_time_appended_without_duplicate": bool(end_appended),
        "last_interval_dt": float(interval_dt[-1]),
        "last_interval_is_short": bool(interval_dt[-1] < physics_dt - 1.0e-12),
        "max_horizontal_speed": _max_speed_record(interval_velocity, times, positions, vertical=False),
        "max_vertical_speed_abs": _max_speed_record(interval_velocity, times, positions, vertical=True),
        "max_horizontal_acceleration": _max_acceleration_record(
            interval_acceleration, acceleration_midpoints, vertical=False,
        ) if len(interval_acceleration) else None,
        "max_vertical_acceleration_abs": _max_acceleration_record(
            interval_acceleration, acceleration_midpoints, vertical=True,
        ) if len(interval_acceleration) else None,
        "profile_limits": {
            "max_horizontal_speed_mps": float(profile.max_horizontal_speed),
            "max_horizontal_accel_mps2": float(profile.max_horizontal_accel),
            "max_horizontal_decel_mps2": float(profile.max_horizontal_decel),
            "max_vertical_speed_up_mps": float(profile.max_climb_rate),
            "max_vertical_speed_down_mps": float(profile.max_descent_rate),
            "max_vertical_accel_up_mps2": float(profile.max_vertical_accel_up),
            "max_vertical_accel_down_mps2": float(profile.max_vertical_accel_down),
        },
    }


def _plan_record(mode: str, merge_time: float, residual: float, plan, env) -> dict:
    segments = [
        _segment(left, right, index)
        for index, (left, right) in enumerate(zip(plan.points, plan.points[1:]))
    ]
    chord_max = max(segments, key=lambda row: row["chord_minus_average_velocity_norm"])
    dx_max = max(segments, key=lambda row: row["dx_minus_average_v_dt_norm"])

    first = plan.points[0]
    second = plan.points[1]
    first_dt = float(second.t - first.t)
    midpoint = float((first.t + second.t) / 2.0)
    epsilon = float(min(1.0e-4, 0.01 * first_dt))
    sample_minus = plan.sample(midpoint - epsilon)
    sample_mid = plan.sample(midpoint)
    sample_plus = plan.sample(midpoint + epsilon)
    central_position_derivative = (
        np.asarray(sample_plus.position, dtype=float) - np.asarray(sample_minus.position, dtype=float)
    ) / (2.0 * epsilon)
    sample_velocity_difference = central_position_derivative - np.asarray(sample_mid.velocity, dtype=float)
    midpoint_sample = {
        "segment_index": 0,
        "midpoint_time": midpoint,
        "epsilon": epsilon,
        "sample_minus_time": float(sample_minus.t),
        "sample_mid_time": float(sample_mid.t),
        "sample_plus_time": float(sample_plus.t),
        "sample_minus_position": _vector(sample_minus.position),
        "sample_mid_position": _vector(sample_mid.position),
        "sample_plus_position": _vector(sample_plus.position),
        "central_position_derivative": _vector(central_position_derivative),
        "sample_mid_velocity": _vector(sample_mid.velocity),
        "central_derivative_minus_sample_velocity": _vector(sample_velocity_difference),
        "central_derivative_minus_sample_velocity_norm": float(np.linalg.norm(sample_velocity_difference)),
    }
    return {
        "mode": mode,
        "reservation_merge_time": float(merge_time),
        "speed_residual_mps": float(residual),
        "plan_id": plan.plan_id,
        "trajectory_hash": plan.trajectory_hash,
        "start_time": float(plan.start_time),
        "end_time": float(plan.end_time),
        "duration": float(plan.end_time - plan.start_time),
        "point_count": len(plan.points),
        "validate_kinematics": list(env.generator.validate_kinematics(plan)),
        "metadata": {
            key: value
            for key, value in plan.metadata.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "segment_count": len(segments),
        "max_chord_minus_average_velocity": {
            "index": int(chord_max["index"]),
            "norm": float(chord_max["chord_minus_average_velocity_norm"]),
        },
        "max_dx_minus_average_v_dt": {
            "index": int(dx_max["index"]),
            "norm": float(dx_max["dx_minus_average_v_dt_norm"]),
        },
        "execution_sample_metrics": _execution_sample_metrics(plan, env),
        "first_three_segments": segments[:3],
        "first_segment_midpoint_difference": midpoint_sample,
    }


def main() -> None:
    fixture = _load_fixture_module()
    rows: list[dict] = []
    for mode in ("legacy", "speed_timing_v1"):
        for merge_time in (0.0, 30.0):
            env, uav, lz_id = fixture._env(execution_mode=mode)
            plan = env.generator.final_approach(
                uav,
                env.lz_manager.zones[lz_id],
                fixture._reservation(uav, merge_time),
                0.0,
                speed_residual_mps=0.0,
                airspace=env.airspace,
            )
            rows.append(_plan_record(mode, merge_time, 0.0, plan, env))
    payload = {
        "fixture_source": str(TEST_PATH.relative_to(ROOT)),
        "scope": {
            "modes": ["legacy", "speed_timing_v1"],
            "reservation_merge_times": [0.0, 30.0],
            "speed_residual_mps": 0.0,
            "environment_steps": False,
            "training": False,
        },
        "plans": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    print(json.dumps({"output": str(OUTPUT_PATH), "plans": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
