"""Compact, diagnostic-only trajectory kinematic consistency report.

This module deliberately observes generated plans without changing the
production execution, safety, or validation paths.  A consistency finding is
reported with exit code zero so that it is distinguishable from a technical
error while still making the finding explicit in the JSON status.

The node chord-versus-average-velocity and displacement-integral residuals
describe the current V2 position/velocity fields, which are each linearly
interpolated; future curve versions must check their analytic derivative and
integral separately rather than treating a trapezoidal-approximation residual
as a universal physical-inconsistency criterion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.merge_scheduler import MergeReservation
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import ArrivalRequest, FlightPhase, MissionType, QueuedArrivalState, UAVState
from terminal_marl.core.vehicle_profiles import VehiclePerformanceProfile
from terminal_marl.envs.parallel_env import TerminalParallelEnv


PHYSICS_DT = 0.2
NODE_RESIDUAL_TOLERANCE = 1.0e-8
PROFILE_TOLERANCE = 1.0e-3
ANALYTIC_TOLERANCE = 1.0e-6
TRAJECTORY_SCHEMA_V2 = "executable_trajectory_v2"
TRAJECTORY_SCHEMA_V3 = "executable_trajectory_v3"


def _vector(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float).tolist()]


def _point_summary(point: Any) -> dict[str, Any]:
    return {
        "t": float(point.t),
        "position": _vector(point.position),
        "velocity": _vector(point.velocity),
        "acceleration": _vector(point.acceleration),
    }


def _segment_summary(left: Any, right: Any, index: int) -> dict[str, Any]:
    dt = float(right.t - left.t)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"segment {index} has invalid dt={dt!r}")
    dx = np.asarray(right.position, dtype=float) - np.asarray(left.position, dtype=float)
    chord_velocity = dx / dt
    average_velocity = (
        np.asarray(left.velocity, dtype=float) + np.asarray(right.velocity, dtype=float)
    ) / 2.0
    chord_minus_average = chord_velocity - average_velocity
    displacement_residual = dx - average_velocity * dt
    return {
        "index": int(index),
        "left": _point_summary(left),
        "right": _point_summary(right),
        "dt": dt,
        "dx": _vector(dx),
        "chord_velocity_dx_dt": _vector(chord_velocity),
        "average_reported_velocity": _vector(average_velocity),
        "chord_minus_average_velocity": _vector(chord_minus_average),
        "chord_minus_average_velocity_norm": float(np.linalg.norm(chord_minus_average)),
        "displacement_integral_residual": _vector(displacement_residual),
        "displacement_integral_residual_norm": float(np.linalg.norm(displacement_residual)),
    }


def _execution_grid(start_time: float, end_time: float, physics_dt: float) -> tuple[np.ndarray, bool]:
    if not np.isfinite(start_time) or not np.isfinite(end_time) or end_time <= start_time:
        raise ValueError("trajectory time bounds must be finite and increasing")
    if physics_dt <= 0.0:
        raise ValueError("physics_dt must be positive")
    times = [float(start_time)]
    index = 1
    while True:
        candidate = float(start_time + index * physics_dt)
        if candidate >= end_time - 1.0e-12:
            break
        times.append(candidate)
        index += 1
    end_appended = not np.isclose(times[-1], end_time, rtol=0.0, atol=1.0e-12)
    if end_appended:
        times.append(float(end_time))
    else:
        times[-1] = float(end_time)
    return np.asarray(times, dtype=float), bool(end_appended)


def _speed_record(values: np.ndarray, times: np.ndarray, positions: np.ndarray, *, vertical: bool) -> dict[str, Any]:
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


def _acceleration_record(values: np.ndarray, times: np.ndarray, *, vertical: bool) -> dict[str, Any] | None:
    if len(values) == 0:
        return None
    magnitudes = np.abs(values[:, 2]) if vertical else np.linalg.norm(values[:, :2], axis=1)
    index = int(np.argmax(magnitudes))
    return {
        "between_interval_index": index,
        "value": float(magnitudes[index]),
        "signed_vertical_value": float(values[index, 2]) if vertical else None,
        "vector": _vector(values[index]),
        "midpoint_time": float(times[index]),
    }


def _profile_limits(profile: VehiclePerformanceProfile) -> dict[str, float]:
    return {
        "max_horizontal_speed_mps": float(profile.max_horizontal_speed),
        "max_horizontal_accel_mps2": float(profile.max_horizontal_accel),
        "max_horizontal_decel_mps2": float(profile.max_horizontal_decel),
        "max_vertical_speed_up_mps": float(profile.max_climb_rate),
        "max_vertical_speed_down_mps": float(profile.max_descent_rate),
        "max_vertical_accel_up_mps2": float(profile.max_vertical_accel_up),
        "max_vertical_accel_down_mps2": float(profile.max_vertical_accel_down),
    }


def _execution_metrics(plan: TrajectoryPlan, profile: VehiclePerformanceProfile, physics_dt: float) -> dict[str, Any]:
    times, end_appended = _execution_grid(float(plan.start_time), float(plan.end_time), physics_dt)
    positions = np.asarray([plan.sample_position(float(time)) for time in times], dtype=float)
    interval_dt = np.diff(times)
    if np.any(interval_dt <= 0.0):
        raise ValueError("execution grid contains a non-positive interval")
    interval_velocity = np.diff(positions, axis=0) / interval_dt[:, None]
    interval_midpoints = (times[:-1] + times[1:]) / 2.0
    midpoint_dt = np.diff(interval_midpoints)
    interval_acceleration = np.diff(interval_velocity, axis=0) / midpoint_dt[:, None]
    acceleration_midpoints = (interval_midpoints[:-1] + interval_midpoints[1:]) / 2.0
    return {
        "physics_dt": float(physics_dt),
        "sample_count": int(len(times)),
        "interval_count": int(len(interval_dt)),
        "end_time_appended_without_duplicate": bool(end_appended),
        "last_interval_dt": float(interval_dt[-1]),
        "last_interval_is_short": bool(interval_dt[-1] < physics_dt - 1.0e-12),
        "max_horizontal_speed": _speed_record(interval_velocity, times, positions, vertical=False),
        "max_vertical_speed_abs": _speed_record(interval_velocity, times, positions, vertical=True),
        "max_horizontal_acceleration": _acceleration_record(
            interval_acceleration, acceleration_midpoints, vertical=False,
        ),
        "max_vertical_acceleration_abs": _acceleration_record(
            interval_acceleration, acceleration_midpoints, vertical=True,
        ),
        "profile_limits": _profile_limits(profile),
    }


def _midpoint_derivative(plan: TrajectoryPlan, first_segment: dict[str, Any]) -> dict[str, Any]:
    midpoint = (float(first_segment["left"]["t"]) + float(first_segment["right"]["t"])) / 2.0
    dt = float(first_segment["dt"])
    epsilon = min(1.0e-4, 0.01 * dt)
    sample_minus = plan.sample(midpoint - epsilon)
    sample_mid = plan.sample(midpoint)
    sample_plus = plan.sample(midpoint + epsilon)
    central_derivative = (
        np.asarray(sample_plus.position, dtype=float) - np.asarray(sample_minus.position, dtype=float)
    ) / (2.0 * epsilon)
    difference = central_derivative - np.asarray(sample_mid.velocity, dtype=float)
    return {
        "segment_index": 0,
        "midpoint_time": midpoint,
        "epsilon": float(epsilon),
        "sample_minus_time": float(sample_minus.t),
        "sample_mid_time": float(sample_mid.t),
        "sample_plus_time": float(sample_plus.t),
        "sample_minus_position": _vector(sample_minus.position),
        "sample_mid_position": _vector(sample_mid.position),
        "sample_plus_position": _vector(sample_plus.position),
        "central_position_derivative": _vector(central_derivative),
        "sample_mid_velocity": _vector(sample_mid.velocity),
        "central_derivative_minus_sample_velocity": _vector(difference),
        "central_derivative_minus_sample_velocity_norm": float(np.linalg.norm(difference)),
    }


def _gauss_legendre_3(segment: Any, *, component: str) -> np.ndarray:
    """Integrate a quintic segment's reported velocity or acceleration."""
    duration = float(segment.end_time - segment.start_time)
    nodes = np.asarray((-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)), dtype=float)
    weights = np.asarray((5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0), dtype=float)
    midpoint = (float(segment.start_time) + float(segment.end_time)) / 2.0
    half = duration / 2.0
    result = np.zeros(3, dtype=float)
    for node, weight in zip(nodes, weights):
        _position, velocity, acceleration = segment.sample(midpoint + half * float(node))
        result += float(weight) * (velocity if component == "velocity" else acceleration)
    return result * half


def _v3_analytic_consistency(plan: TrajectoryPlan) -> dict[str, Any]:
    """Check V3 derivatives and polynomial integrals independently of node chords."""
    segments = getattr(plan, "polynomial_segments", ())
    if not segments or len(segments) != len(plan.points) - 1:
        raise ValueError("V3 plan does not expose one polynomial segment per point interval")
    rows: list[dict[str, Any]] = []
    max_position_derivative_error = 0.0
    max_velocity_derivative_error = 0.0
    max_velocity_integral_error = 0.0
    max_acceleration_integral_error = 0.0
    for index, segment in enumerate(segments):
        duration = float(segment.end_time - segment.start_time)
        epsilon = max(1.0e-8, min(1.0e-4, 1.0e-3 * duration))
        derivative_rows: list[dict[str, Any]] = []
        for fraction in (0.25, 0.5, 0.75):
            moment = float(segment.start_time + fraction * duration)
            sample_minus = segment.sample(moment - epsilon)
            sample = segment.sample(moment)
            sample_plus = segment.sample(moment + epsilon)
            position_derivative = (
                np.asarray(sample_plus[0], dtype=float) - np.asarray(sample_minus[0], dtype=float)
            ) / (2.0 * epsilon)
            velocity_derivative = (
                np.asarray(sample_plus[1], dtype=float) - np.asarray(sample_minus[1], dtype=float)
            ) / (2.0 * epsilon)
            position_error = position_derivative - np.asarray(sample[1], dtype=float)
            velocity_error = velocity_derivative - np.asarray(sample[2], dtype=float)
            position_error_norm = float(np.linalg.norm(position_error))
            velocity_error_norm = float(np.linalg.norm(velocity_error))
            max_position_derivative_error = max(max_position_derivative_error, position_error_norm)
            max_velocity_derivative_error = max(max_velocity_derivative_error, velocity_error_norm)
            derivative_rows.append({
                "fraction": float(fraction),
                "time": moment,
                "epsilon": epsilon,
                "position_derivative_error": _vector(position_error),
                "position_derivative_error_norm": position_error_norm,
                "velocity_derivative_error": _vector(velocity_error),
                "velocity_derivative_error_norm": velocity_error_norm,
            })
        left = segment.sample(float(segment.start_time))
        right = segment.sample(float(segment.end_time))
        position_integral = _gauss_legendre_3(segment, component="velocity")
        acceleration_integral = _gauss_legendre_3(segment, component="acceleration")
        displacement = np.asarray(right[0], dtype=float) - np.asarray(left[0], dtype=float)
        delta_velocity = np.asarray(right[1], dtype=float) - np.asarray(left[1], dtype=float)
        position_integral_error = position_integral - displacement
        acceleration_integral_error = acceleration_integral - delta_velocity
        position_integral_error_norm = float(np.linalg.norm(position_integral_error))
        acceleration_integral_error_norm = float(np.linalg.norm(acceleration_integral_error))
        max_velocity_integral_error = max(max_velocity_integral_error, position_integral_error_norm)
        max_acceleration_integral_error = max(max_acceleration_integral_error, acceleration_integral_error_norm)
        rows.append({
            "index": int(index),
            "duration": duration,
            "central_derivative_checks": derivative_rows,
            "velocity_integral": _vector(position_integral),
            "displacement": _vector(displacement),
            "velocity_integral_error": _vector(position_integral_error),
            "velocity_integral_error_norm": position_integral_error_norm,
            "acceleration_integral": _vector(acceleration_integral),
            "delta_velocity": _vector(delta_velocity),
            "acceleration_integral_error": _vector(acceleration_integral_error),
            "acceleration_integral_error_norm": acceleration_integral_error_norm,
        })
    bound_rows = [segment.bounds() for segment in segments]
    analytic_bounds = {
        key: float(
            min(row[key] for row in bound_rows)
            if key == "min_altitude"
            else max(row[key] for row in bound_rows)
        )
        for key in bound_rows[0]
    }
    consistent = (
        max_position_derivative_error <= ANALYTIC_TOLERANCE
        and max_velocity_derivative_error <= ANALYTIC_TOLERANCE
        and max_velocity_integral_error <= ANALYTIC_TOLERANCE
        and max_acceleration_integral_error <= ANALYTIC_TOLERANCE
    )
    return {
        "tolerance": ANALYTIC_TOLERANCE,
        "segments": rows,
        "max_position_derivative_error": max_position_derivative_error,
        "max_velocity_derivative_error": max_velocity_derivative_error,
        "max_velocity_integral_error": max_velocity_integral_error,
        "max_acceleration_integral_error": max_acceleration_integral_error,
        "analytic_bounds": analytic_bounds,
        "consistent": bool(consistent),
    }


def _finding_flags(
    node_max_chord: dict[str, Any],
    node_max_displacement: dict[str, Any],
    midpoint: dict[str, Any],
    execution: dict[str, Any],
    validate_kinematics: Sequence[str],
    *,
    trajectory_schema: str = TRAJECTORY_SCHEMA_V2,
    analytic: dict[str, Any] | None = None,
) -> list[str]:
    findings: list[str] = []
    if trajectory_schema == TRAJECTORY_SCHEMA_V3:
        if analytic is None:
            findings.append("v3_analytic_consistency_unavailable")
        else:
            if not bool(analytic["consistent"]):
                findings.append("v3_analytic_derivative_or_integral_mismatch")
        limits = execution["profile_limits"]
        if execution["max_horizontal_speed"]["value"] > limits["max_horizontal_speed_mps"] + PROFILE_TOLERANCE:
            findings.append("sampled_horizontal_speed_profile_exceeded")
        if execution["max_vertical_speed_abs"]["value"] > max(
            limits["max_vertical_speed_up_mps"], limits["max_vertical_speed_down_mps"]
        ) + PROFILE_TOLERANCE:
            findings.append("sampled_vertical_speed_profile_exceeded")
        if execution["max_horizontal_acceleration"] is not None and execution["max_horizontal_acceleration"]["value"] > limits["max_horizontal_accel_mps2"] + PROFILE_TOLERANCE:
            findings.append("sampled_horizontal_acceleration_profile_exceeded")
        if execution["max_vertical_acceleration_abs"] is not None and execution["max_vertical_acceleration_abs"]["value"] > max(
            limits["max_vertical_accel_up_mps2"], limits["max_vertical_accel_down_mps2"]
        ) + PROFILE_TOLERANCE:
            findings.append("sampled_vertical_acceleration_profile_exceeded")
        return findings
    if node_max_chord["norm"] > NODE_RESIDUAL_TOLERANCE:
        findings.append("node_chord_velocity_mismatch")
    if node_max_displacement["norm"] > NODE_RESIDUAL_TOLERANCE:
        findings.append("node_displacement_integral_residual")
    if midpoint["central_derivative_minus_sample_velocity_norm"] > NODE_RESIDUAL_TOLERANCE:
        findings.append("first_segment_midpoint_velocity_mismatch")
    limits = execution["profile_limits"]
    if execution["max_horizontal_speed"]["value"] > limits["max_horizontal_speed_mps"] + PROFILE_TOLERANCE:
        findings.append("sampled_horizontal_speed_profile_exceeded")
    if execution["max_vertical_speed_abs"]["value"] > max(
        limits["max_vertical_speed_up_mps"], limits["max_vertical_speed_down_mps"]
    ) + PROFILE_TOLERANCE:
        findings.append("sampled_vertical_speed_profile_exceeded")
    if execution["max_horizontal_acceleration"] is not None and execution["max_horizontal_acceleration"]["value"] > limits["max_horizontal_accel_mps2"] + PROFILE_TOLERANCE:
        findings.append("sampled_horizontal_acceleration_profile_exceeded")
    if execution["max_vertical_acceleration_abs"] is not None and execution["max_vertical_acceleration_abs"]["value"] > max(
        limits["max_vertical_accel_up_mps2"], limits["max_vertical_accel_down_mps2"]
    ) + PROFILE_TOLERANCE:
        findings.append("sampled_vertical_acceleration_profile_exceeded")
    if validate_kinematics:
        findings.append("validate_kinematics_errors")
    return findings


def diagnose_plan(
    plan: TrajectoryPlan,
    *,
    profile: VehiclePerformanceProfile,
    validate_kinematics: Sequence[str] = (),
    physics_dt: float = PHYSICS_DT,
    mode: str | None = None,
    reservation_slot: float | None = None,
    trajectory_schema: str | None = None,
) -> dict[str, Any]:
    effective_schema = trajectory_schema or str(plan.metadata.get("trajectory_schema_version", TRAJECTORY_SCHEMA_V2))
    if effective_schema not in {TRAJECTORY_SCHEMA_V2, TRAJECTORY_SCHEMA_V3}:
        raise ValueError(f"unsupported trajectory schema {effective_schema!r}")
    segments = [
        _segment_summary(left, right, index)
        for index, (left, right) in enumerate(zip(plan.points, plan.points[1:]))
    ]
    if not segments:
        raise ValueError("trajectory requires at least one segment")
    chord_max = max(segments, key=lambda row: row["chord_minus_average_velocity_norm"])
    displacement_max = max(segments, key=lambda row: row["displacement_integral_residual_norm"])
    midpoint = _midpoint_derivative(plan, segments[0])
    execution = _execution_metrics(plan, profile, physics_dt)
    analytic = _v3_analytic_consistency(plan) if effective_schema == TRAJECTORY_SCHEMA_V3 else None
    findings = _finding_flags(
        {
            "index": chord_max["index"],
            "norm": chord_max["chord_minus_average_velocity_norm"],
        },
        {
            "index": displacement_max["index"],
            "norm": displacement_max["displacement_integral_residual_norm"],
        },
        midpoint,
        execution,
        validate_kinematics,
        trajectory_schema=effective_schema,
        analytic=analytic,
    )
    requested_slot = None if reservation_slot is None else float(reservation_slot)
    canonical_departure = float(plan.metadata.get("scheduled_departure", plan.start_time))
    return {
        "mode": mode if mode is not None else str(plan.mode),
        "reservation_slot": requested_slot,
        "trajectory_hash": plan.trajectory_hash,
        "duration": float(plan.end_time - plan.start_time),
        "start_time": float(plan.start_time),
        "end_time": float(plan.end_time),
        "canonical_departure": canonical_departure,
        "initial_state": _point_summary(plan.points[0]),
        "trajectory_schema_version": effective_schema,
        "validate_kinematics": list(validate_kinematics),
        "first_three_segments": segments[:3],
        "first_segment_midpoint_difference": midpoint,
        "max_chord_minus_average_velocity": {
            "index": int(chord_max["index"]),
            "norm": float(chord_max["chord_minus_average_velocity_norm"]),
            "vector": chord_max["chord_minus_average_velocity"],
        },
        "max_displacement_integral_residual": {
            "index": int(displacement_max["index"]),
            "norm": float(displacement_max["displacement_integral_residual_norm"]),
            "vector": displacement_max["displacement_integral_residual"],
        },
        "legacy_node_diagnostics": {
            "applicability": "v2_binding" if effective_schema == TRAJECTORY_SCHEMA_V2 else "v2_only_nonbinding",
            "max_chord_minus_average_velocity": {
                "index": int(chord_max["index"]),
                "norm": float(chord_max["chord_minus_average_velocity_norm"]),
            },
            "max_displacement_integral_residual": {
                "index": int(displacement_max["index"]),
                "norm": float(displacement_max["displacement_integral_residual_norm"]),
            },
        },
        "v3_analytic_consistency": analytic,
        "execution_sample_metrics": execution,
        "findings": findings,
        "consistent": not findings,
    }


def make_linear_plan(
    *,
    reported_velocity: Sequence[float] = (1.0, 0.0, 0.0),
    trajectory_schema: str = TRAJECTORY_SCHEMA_V2,
) -> TrajectoryPlan:
    """Construct a small in-memory plan for diagnostic regression tests."""
    velocity = np.asarray(reported_velocity, dtype=float)
    points = tuple(
        TrajectoryPoint(
            float(index),
            np.asarray((float(index), 0.0, 0.0), dtype=float),
            velocity,
            np.zeros(3, dtype=float),
        )
        for index in range(3)
    )
    return TrajectoryPlan(
        "diagnostic-linear-plan",
        0,
        "diagnostic",
        points,
        0.0,
        2.0,
        metadata={
            "scheduled_departure": 0.0,
            "trajectory_schema_version": trajectory_schema,
        },
    )


def _fixture_env(
    execution_mode: str,
    *,
    trajectory_schema: str = TRAJECTORY_SCHEMA_V2,
) -> tuple[TerminalParallelEnv, UAVState, int]:
    action_schema = (
        "low_action_schema_v4_2_hierarchical"
        if execution_mode == "speed_timing_v1"
        else "low_action_schema_v4_joint_discrete"
    )
    config = SimConfig(
        arrival_model="finite_cohort",
        total_requested_uavs=0,
        arrival_rate_per_second=0.0,
        action_schema_version=action_schema,
        final_approach_geometry_mode="continuous_inward_v1",
        final_approach_execution_mode=execution_mode,
        trajectory_schema_version=trajectory_schema,
        vehicle_execution_mode="performance_constrained",
        max_simulation_seconds=120.0,
    )
    env = TerminalParallelEnv(config)
    request = ArrivalRequest(1, 0.0, 0, MissionType.STANDARD, 1, config.energy_capacity, 1.0)
    queued = QueuedArrivalState(request, request.energy_capacity)
    uav = UAVState.from_request(0, request, queued, np.zeros(3), env.now)
    uav.phase = FlightPhase.RING_INNER
    uav.ring_role = RingRole.INNER
    uav.ring_progress = 0.0
    ring = env.airspace.actual_ring(RingRole.INNER)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    uav.acceleration = np.asarray((-(ring.nominal_speed**2) / ring.radius, 0.0, 0.0), dtype=float)
    env.active[uav.uid] = uav
    lz_id = env.lz_manager.choose_rule_zone(uav, env.now)
    if lz_id is None:
        raise RuntimeError("fixture did not produce a landing zone")
    return env, uav, lz_id


def build_fixture_plans(
    *,
    trajectory_schema: str = TRAJECTORY_SCHEMA_V2,
) -> list[tuple[str, float, TerminalParallelEnv, TrajectoryPlan]]:
    if trajectory_schema not in {TRAJECTORY_SCHEMA_V2, TRAJECTORY_SCHEMA_V3}:
        raise ValueError(f"unsupported trajectory schema {trajectory_schema!r}")
    plans: list[tuple[str, float, TerminalParallelEnv, TrajectoryPlan]] = []
    for mode in ("legacy", "speed_timing_v1"):
        for slot in (0.0, 30.0):
            env, uav, lz_id = _fixture_env(mode, trajectory_schema=trajectory_schema)
            reservation = MergeReservation(
                f"final-{slot}", uav.uid, "ring_inner", "final", slot, None, None, None,
            )
            plan = env.generator.final_approach(
                uav,
                env.lz_manager.zones[lz_id],
                reservation,
                0.0,
                speed_residual_mps=0.0,
                airspace=env.airspace,
            )
            plans.append((mode, slot, env, plan))
    return plans


def run_diagnostic(
    output_path: str | Path,
    *,
    trajectory_schema: str = TRAJECTORY_SCHEMA_V2,
) -> dict[str, Any]:
    output = Path(output_path)
    technical_errors: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    profile: VehiclePerformanceProfile | None = None
    try:
        fixture_plans = build_fixture_plans(trajectory_schema=trajectory_schema)
        if len(fixture_plans) != 4:
            raise RuntimeError(f"expected four fixture plans, got {len(fixture_plans)}")
        profile = fixture_plans[0][2].config.vehicle_profile
        for mode, slot, env, plan in fixture_plans:
            try:
                rows.append(
                    diagnose_plan(
                        plan,
                        profile=env.config.vehicle_profile,
                        validate_kinematics=env.generator.validate_kinematics(plan),
                        physics_dt=env.config.physics_dt,
                        mode=mode,
                        reservation_slot=slot,
                        trajectory_schema=trajectory_schema,
                    )
                )
            except Exception as exc:  # diagnostic surface retains the technical failure as data
                technical_errors.append({
                    "mode": mode,
                    "reservation_slot": float(slot),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
    except Exception as exc:
        technical_errors.append({"stage": "fixture_build", "error_type": type(exc).__name__, "error": str(exc)})

    if technical_errors:
        status = "TECHNICAL_ERROR"
    else:
        consistent = all(bool(row["consistent"]) for row in rows)
        status = "KINEMATIC_CONSISTENCY_PASS" if consistent else "KINEMATIC_CONSISTENCY_FAIL"
    payload: dict[str, Any] = {
        "stage": "trajectory_kinematic_consistency",
        "trajectory_schema_version": trajectory_schema,
        "status": status,
        "consistent": bool(status == "KINEMATIC_CONSISTENCY_PASS"),
        "finding_count": int(sum(len(row["findings"]) for row in rows)),
        "technical_error_count": len(technical_errors),
        "technical_errors": technical_errors,
        "profile": None if profile is None else _profile_limits(profile),
        "checks": {
            "fixture_count": len(rows),
            "validate_kinematics_results_included": all("validate_kinematics" in row for row in rows),
            "execution_sample_results_included": all("execution_sample_metrics" in row for row in rows),
            "physics_dt": PHYSICS_DT,
            "environment_steps": False,
            "training": False,
        },
        "plans": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    json.loads(output.read_text(encoding="utf-8"))
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="explicit JSON output path")
    parser.add_argument(
        "--trajectory-schema",
        choices=(TRAJECTORY_SCHEMA_V2, TRAJECTORY_SCHEMA_V3),
        default=TRAJECTORY_SCHEMA_V2,
        help="trajectory execution schema used for fixture generation and checks",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = run_diagnostic(args.output, trajectory_schema=args.trajectory_schema)
    except Exception as exc:
        print(f"TECHNICAL_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "output": str(Path(args.output)),
        "status": payload["status"],
        "finding_count": payload["finding_count"],
        "technical_error_count": payload["technical_error_count"],
        "exit_code": 0 if payload["technical_error_count"] == 0 else 1,
    }, sort_keys=True))
    return 0 if payload["technical_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
