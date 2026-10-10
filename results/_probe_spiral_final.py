"""Offline comparison of rotating-frame and Cartesian final approaches."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terminal_marl.core.config import RingRole
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.flow_trajectories import FlowTrajectoryGenerator
from terminal_marl.core.landing_zone import LandingZoneManager
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryGenerator, TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.trajectory_time_parameterization import GeometricPath, TrajectoryConstraints
from terminal_marl.core.uav import ArrivalRequest, FlightPhase, MissionType, QueuedArrivalState, UAVState
from terminal_marl.training.diagnose_landing_flow import _build_diagnostic_config


OUTPUT = Path("results/flow_spiral_final_geometry_probe.json")
ANGLES = (0.0, math.pi / 2.0, math.pi, 1.5 * math.pi)
DURATIONS = (28.0, 36.0, 48.0, 60.0, 90.0)
HOLDER_CHORD_M = 96.0


def make_uav(config, ring, angle: float) -> UAVState:
    request = ArrivalRequest(
        1, 0.0, 0, MissionType.STANDARD, 1, config.energy_capacity, 1.0,
    )
    queued = QueuedArrivalState(request, request.energy_capacity)
    uav = UAVState.from_request(
        1, request, queued, np.zeros(3, dtype=float), 0.0,
    )
    radius = float(ring.radius)
    speed = float(ring.nominal_speed)
    uav.ring_role = RingRole.INNER
    uav.phase = FlightPhase.RING_INNER
    uav.position = np.asarray(
        (radius * math.cos(angle), radius * math.sin(angle), float(ring.height)),
        dtype=float,
    )
    uav.speed = speed
    uav.velocity = np.asarray(
        (-speed * math.sin(angle), speed * math.cos(angle), 0.0),
        dtype=float,
    )
    uav.acceleration = np.asarray(
        (-(speed * speed / radius) * math.cos(angle),
         -(speed * speed / radius) * math.sin(angle),
         0.0),
        dtype=float,
    )
    uav.ring_progress = float(angle)
    return uav


def evaluate_polynomial(coefficients: np.ndarray, tau: np.ndarray, duration: float):
    powers = np.column_stack((
        np.ones_like(tau), tau, tau**2, tau**3, tau**4, tau**5,
    ))
    first = np.column_stack((
        np.zeros_like(tau), np.ones_like(tau), 2.0 * tau,
        3.0 * tau**2, 4.0 * tau**3, 5.0 * tau**4,
    ))
    second = np.column_stack((
        np.zeros_like(tau), np.zeros_like(tau), 2.0 * np.ones_like(tau),
        6.0 * tau, 12.0 * tau**2, 20.0 * tau**3,
    ))
    return powers @ coefficients, first @ coefficients / duration, second @ coefficients / duration**2


def rotate_xy(values: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cosines = np.cos(angles)
    sines = np.sin(angles)
    result = np.empty_like(values)
    result[:, 0] = cosines * values[:, 0] - sines * values[:, 1]
    result[:, 1] = sines * values[:, 0] + cosines * values[:, 1]
    result[:, 2] = values[:, 2]
    return result


def rotating_departure(flow, uav, staging: np.ndarray, duration: float):
    times = flow._time_grid(duration)
    tau = times / duration
    omega = float(flow.config.vehicle_profile.max_horizontal_speed * 0.0)
    ring = DynamicAirspace(flow.config).actual_ring(RingRole.INNER)
    omega = float(ring.nominal_speed) / float(ring.radius)
    j = lambda values: np.column_stack((-values[:, 1], values[:, 0], np.zeros(len(values))))
    b0 = np.asarray(uav.position, dtype=float) - staging
    b1 = np.asarray(uav.velocity, dtype=float) - omega * j(b0.reshape(1, 3))[0]
    b2 = (
        np.asarray(uav.acceleration, dtype=float)
        - 2.0 * omega * j(b1.reshape(1, 3))[0]
        + omega**2 * np.asarray((b0[0], b0[1], 0.0), dtype=float)
    )
    coefficients = flow._quintic_coefficients(
        b0, b1, b2, np.zeros(3, dtype=float), duration,
    )
    b, b_prime, b_second = evaluate_polynomial(coefficients, tau, duration)
    angles = omega * times
    j_b = j(b)
    j_b_prime = j(b_prime)
    b_xy = b.copy()
    b_xy[:, 2] = 0.0
    positions = staging + rotate_xy(b, angles)
    velocities = rotate_xy(b_prime + omega * j_b, angles)
    accelerations = rotate_xy(
        b_second + 2.0 * omega * j_b_prime - omega**2 * b_xy,
        angles,
    )
    positions[0] = np.asarray(uav.position, dtype=float)
    velocities[0] = np.asarray(uav.velocity, dtype=float)
    accelerations[0] = np.asarray(uav.acceleration, dtype=float)
    positions[-1] = staging
    velocities[-1] = 0.0
    accelerations[-1] = 0.0
    return tuple(
        TrajectoryPoint(float(t), p, v, a)
        for t, p, v, a in zip(times, positions, velocities, accelerations)
    )


def make_spiral_plan(flow, uav, pad, airspace, duration: float) -> TrajectoryPlan:
    cruise_height = float(uav.position[2])
    staging = np.asarray((pad.position[0], pad.position[1], cruise_height), dtype=float)
    departure = rotating_departure(flow, uav, staging, duration)
    leg_path = GeometricPath.line(staging, np.asarray(pad.position, dtype=float), samples=17)
    leg = flow.parameterizer.parameterize(
        leg_path,
        start_time=duration,
        initial_velocity=departure[-1].velocity,
        target_horizontal_speed=5.0,
        constraints=TrajectoryConstraints(end_horizontal_speed=0.0, tangent_yaw=False),
    )
    details = {
        "target_phase": "landing",
        "source_ring": RingRole.INNER.value,
        "source_ring_role": RingRole.INNER.value,
        "flow_geometry": "rotating_spiral_quintic_probe_v1",
        "scheduled_departure": 0.0,
        "staging_point": staging.tolist(),
        "corridor_height": cruise_height,
        "lz_id": int(pad.lz_id),
        "vehicle_profile_id": flow.config.vehicle_profile_id,
        "vehicle_profile_version": flow.config.vehicle_profile.profile_version,
    }
    points = tuple(departure[:-1]) + tuple(leg)
    return TrajectoryPlan(
        str(uuid.uuid4()), uav.uid, "spiral_final_approach", points,
        0.0, float(leg[-1].t), (), details,
    )


def error_code(exc: Exception) -> str:
    message = str(exc)
    if "dynamics validation" in message or "samples_are_feasible" in message:
        return "dynamics"
    return type(exc).__name__


def clearance(plan, source_angle: float, radius: float, speed: float, safety: SafetyFilter):
    gap = 2.0 * math.asin(HOLDER_CHORD_M / (2.0 * radius))
    samples = np.arange(0.0, float(plan.end_time) + 1e-9, 1.0, dtype=float)
    if samples[-1] < float(plan.end_time) - 1e-9:
        samples = np.append(samples, float(plan.end_time))
    candidate = np.asarray(plan.sample_positions(samples), dtype=float)
    omega = speed / radius
    by_side = {}
    for name, sign in (("ahead", 1.0), ("behind", -1.0)):
        holder_angle = source_angle + sign * gap + omega * samples
        holder = np.column_stack((
            radius * np.cos(holder_angle),
            radius * np.sin(holder_angle),
            np.full_like(samples, radius * 0.0 + 40.0),
        ))
        point, swept = safety._track_distances(candidate, holder)
        by_side[name] = [
            round(float(np.min(point)), 6),
            round(float(np.min(swept)), 6) if len(swept) else round(float(np.min(point)), 6),
        ]
    return [
        round(by_side["ahead"][0], 6),
        round(by_side["ahead"][1], 6),
        round(by_side["behind"][0], 6),
        round(by_side["behind"][1], 6),
    ]


def rounded(value):
    return None if value is None else round(float(value), 6)


def main():
    config = _build_diagnostic_config(True)
    airspace = DynamicAirspace(config)
    base = TrajectoryGenerator(config)
    flow = FlowTrajectoryGenerator(config, base)
    safety = SafetyFilter(config)
    ring = airspace.actual_ring(RingRole.INNER)
    pads = LandingZoneManager().zones
    rows = []
    rotating_valid = 0
    cartesian_valid = 0
    rotating_endpoint_max_abs_error = 0.0
    rotating_endpoint_checks = 0
    for angle_index, angle in enumerate(ANGLES):
        for pad in pads:
            for duration in DURATIONS:
                uav_rotating = make_uav(config, ring, angle)
                uav_cartesian = make_uav(config, ring, angle)
                staging = np.asarray((pad.position[0], pad.position[1], ring.height), dtype=float)
                rotating = {"valid": False, "error": None, "clearance": None}
                try:
                    plan = make_spiral_plan(flow, uav_rotating, pad, airspace, duration)
                    # Check the rotating-frame departure segment itself.  The
                    # first point of the appended vertical leg is parameterized
                    # separately and is therefore not the quintic endpoint.
                    departure = rotating_departure(flow, uav_rotating, staging, duration)
                    source = departure[0]
                    endpoint = departure[-1]
                    expected_start = (
                        uav_rotating.position,
                        uav_rotating.velocity,
                        uav_rotating.acceleration,
                    )
                    expected_end = (
                        staging,
                        np.zeros(3, dtype=float),
                        np.zeros(3, dtype=float),
                    )
                    endpoint_error = max(
                        max(
                            float(np.max(np.abs(actual - expected)))
                            for actual, expected in zip(
                                (source.position, source.velocity, source.acceleration),
                                expected_start,
                            )
                        ),
                        max(
                            float(np.max(np.abs(actual - expected)))
                            for actual, expected in zip(
                                (endpoint.position, endpoint.velocity, endpoint.acceleration),
                                expected_end,
                            )
                        ),
                    )
                    rotating_endpoint_max_abs_error = max(
                        rotating_endpoint_max_abs_error, endpoint_error,
                    )
                    rotating_endpoint_checks += 1
                    errors = flow.dynamics.validate_plan(plan.points)
                    if errors:
                        rotating["error"] = "dynamics"
                    else:
                        rotating["valid"] = True
                        rotating["clearance"] = clearance(
                            plan, angle, float(ring.radius), float(ring.nominal_speed), safety,
                        )
                        rotating_valid += 1
                except Exception as exc:
                    rotating["error"] = error_code(exc)
                cartesian = {"valid": False, "error": None, "clearance": None}
                try:
                    plan = flow.final_approach(
                        uav_cartesian, pad, airspace, 0.0, duration, emergency=False,
                    )
                    errors = flow.dynamics.validate_plan(plan.points)
                    if errors:
                        cartesian["error"] = "dynamics"
                    else:
                        cartesian["valid"] = True
                        cartesian["clearance"] = clearance(
                            plan, angle, float(ring.radius), float(ring.nominal_speed), safety,
                        )
                        cartesian_valid += 1
                except Exception as exc:
                    cartesian["error"] = error_code(exc)
                rows.append([
                    angle_index,
                    int(pad.lz_id),
                    duration,
                    bool(rotating["valid"]),
                    rotating["error"],
                    rotating["clearance"],
                    bool(cartesian["valid"]),
                    cartesian["error"],
                    cartesian["clearance"],
                ])
    result = {
        "probe": "rotating-frame spiral final approach versus Cartesian quintic",
        "schema": {
            "row": "[angle_index,pad_id,duration_s,rotating_valid,rotating_error,rotating_clearance,cartesian_valid,cartesian_error,cartesian_clearance]",
            "clearance": "[ahead_point_m,ahead_swept_m,behind_point_m,behind_swept_m], 1 s grid, holder chord 96 m",
            "error": "null or dynamics",
        },
        "config": {
            "coordinated": True,
            "inner_radius_m": float(ring.radius),
            "inner_height_m": float(ring.height),
            "inner_nominal_speed_mps": float(ring.nominal_speed),
            "angles_rad": list(ANGLES),
            "durations_s": list(DURATIONS),
            "pads": [
                {"id": int(pad.lz_id), "position_m": [float(x) for x in pad.position]}
                for pad in pads
            ],
            "holder_chord_m": HOLDER_CHORD_M,
        },
        "counts": {
            "total": len(rows),
            "rotating_valid": rotating_valid,
            "cartesian_valid": cartesian_valid,
            "rotating_clearance_evaluated": rotating_valid,
            "cartesian_clearance_evaluated": cartesian_valid,
        },
        "rotating_endpoint_checks": {
            "plans_checked": rotating_endpoint_checks,
            "max_abs_error_mps2_or_m": rotating_endpoint_max_abs_error,
        },
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "bytes": OUTPUT.stat().st_size, "counts": result["counts"]}))


if __name__ == "__main__":
    main()
