"""Bounded final-approach clearance probe for the coordinated I ring."""

from __future__ import annotations

from copy import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from terminal_marl.core.config import RingRole
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    UAVState,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from terminal_marl.training.diagnose_energy_reachability_guard_smoke import build_config


OUTPUT = Path("results/inner_final_clearance_probe.json")
START_ANGLES = tuple(float(value) for value in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False))
DURATIONS = (20.0, 28.0, 36.0, 48.0, 60.0)
GAP_CHORDS = tuple(float(value) for value in range(80, 401, 10))
SAFE_DISTANCE = 80.0


def _config():
    base = build_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=True,
        max_simulation_seconds=1800.0,
        dense_warm_start_uavs=1,
        final_approach_geometry_mode="continuous_inward_v1",
        release_arbitration_mode="downstream_priority",
        reconcile_late_merge_slots_enabled=False,
    )
    return replace(base, rule_flow_coordination_enabled=True)


def _uav(env: TerminalParallelEnv, uid: int, angle: float) -> UAVState:
    config = env.config
    ring = env.airspace.actual_ring(RingRole.INNER)
    request = ArrivalRequest(
        uid,
        0.0,
        0,
        MissionType.STANDARD,
        1,
        config.energy_capacity,
        0.80,
    )
    queued = QueuedArrivalState(request, request.energy_capacity * request.initial_soc)
    position = np.asarray(
        (ring.radius * math.cos(angle), ring.radius * math.sin(angle), ring.height),
        dtype=float,
    )
    uav = UAVState.from_request(uid, request, queued, position, env.now)
    uav.phase = FlightPhase.RING_INNER
    uav.ring_role = RingRole.INNER
    uav.ring_progress = float(angle)
    uav.position = position
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray(
        (-ring.nominal_speed * math.sin(angle), ring.nominal_speed * math.cos(angle), 0.0),
        dtype=float,
    )
    uav.acceleration = np.asarray(
        (-ring.nominal_speed**2 * math.cos(angle) / ring.radius,
         -ring.nominal_speed**2 * math.sin(angle) / ring.radius,
         0.0),
        dtype=float,
    )
    env.active[uid] = uav
    env.ledger.request(request.arrival_id)
    return uav


def _holder_for(uav: UAVState, ring, angle: float) -> UAVState:
    holder = copy(uav)
    holder.uid = -abs(int(uav.uid)) - 1
    holder.ring_role = RingRole.INNER
    holder.ring_progress = float(angle)
    holder.position = np.asarray(
        (ring.radius * math.cos(angle), ring.radius * math.sin(angle), ring.height),
        dtype=float,
    )
    holder.speed = ring.nominal_speed
    holder.velocity = np.asarray(
        (-ring.nominal_speed * math.sin(angle), ring.nominal_speed * math.cos(angle), 0.0),
        dtype=float,
    )
    holder.acceleration = np.asarray(
        (-ring.nominal_speed**2 * math.cos(angle) / ring.radius,
         -ring.nominal_speed**2 * math.sin(angle) / ring.radius,
         0.0),
        dtype=float,
    )
    return holder


def _samples(plan, step: float) -> np.ndarray:
    values = np.arange(float(plan.start_time), float(plan.end_time) + 1e-9, step)
    if values[-1] < float(plan.end_time):
        values = np.append(values, float(plan.end_time))
    return values


def _certificate(safety, plan, holder, snapshot, start_angle: float, chord: float) -> dict[str, object]:
    ring = snapshot.airspace.actual_ring(RingRole.INNER)
    angular_gap = 2.0 * math.asin(float(chord) / (2.0 * float(ring.radius)))
    times = _samples(plan, 1.0)
    candidate_positions = np.asarray(plan.sample_positions(times), dtype=float)
    holder_positions = np.asarray(
        safety._holder_positions(
            holder,
            snapshot,
            times,
            0.0,
        ),
        dtype=float,
    )
    point, swept = safety._track_distances(candidate_positions, holder_positions)
    point_min = float(np.min(point)) if len(point) else float("inf")
    swept_min = float(np.min(swept)) if len(swept) else float("inf")
    return {
        "chord_m": float(chord),
        "angular_gap_rad": float(angular_gap),
        "holder_start_angle_rad": float(holder.ring_progress),
        "candidate_start_angle_rad": float(start_angle),
        "sample_count": int(len(times)),
        "point_min_m": point_min,
        "swept_min_m": swept_min,
        "passes": bool(point_min >= SAFE_DISTANCE and swept_min >= SAFE_DISTANCE),
    }


def _first_clearance(safety, plan, uav, snapshot, start_angle: float, *, following: bool):
    ring = snapshot.airspace.actual_ring(RingRole.INNER)
    rows = []
    sign = -1.0 if following else 1.0
    for chord in GAP_CHORDS:
        angle = float(start_angle + sign * 2.0 * math.asin(chord / (2.0 * ring.radius)))
        holder = _holder_for(uav, ring, angle)
        row = _certificate(safety, plan, holder, snapshot, start_angle, chord)
        rows.append(row)
        if row["passes"]:
            return row, rows
    return None, rows


def main() -> None:
    config = _config()
    ring = config.rings[RingRole.INNER]
    all_rows = []
    angle_summaries = []
    dynamic_valid_count = 0
    dynamic_invalid_count = 0

    for angle_index, angle in enumerate(START_ANGLES):
        env = TerminalParallelEnv(config)
        env.active.clear()
        env.service_uavs.clear()
        uav = _uav(env, 1000 + angle_index, angle)
        snapshot = WorldSnapshot(env.now, env.active, env.airspace)
        angle_rows = []
        for pad in env.lz_manager.zones:
            for duration in DURATIONS:
                row = {
                    "angle_index": int(angle_index),
                    "start_angle_rad": float(angle),
                    "pad_id": int(pad.lz_id),
                    "requested_duration_s": float(duration),
                    "dynamic_valid": False,
                    "plan_end_time_s": None,
                    "following": None,
                    "ahead": None,
                }
                try:
                    plan = env.flow_coordinator.trajectory.final_approach(
                        uav,
                        pad,
                        env.airspace,
                        env.now,
                        duration,
                        emergency=False,
                    )
                except (RuntimeError, ValueError) as exc:
                    row["dynamic_error"] = str(exc)
                    dynamic_invalid_count += 1
                    angle_rows.append(row)
                    all_rows.append(row)
                    continue
                dynamic_valid_count += 1
                row["dynamic_valid"] = True
                row["plan_end_time_s"] = float(plan.end_time)
                following, following_grid = _first_clearance(
                    env.safety,
                    plan,
                    uav,
                    snapshot,
                    angle,
                    following=True,
                )
                ahead, ahead_grid = _first_clearance(
                    env.safety,
                    plan,
                    uav,
                    snapshot,
                    angle,
                    following=False,
                )
                row["following"] = following
                row["ahead"] = ahead
                row["following_grid"] = following_grid
                row["ahead_grid"] = ahead_grid
                angle_rows.append(row)
                all_rows.append(row)

        valid_rows = [item for item in angle_rows if item["dynamic_valid"]]
        ranked = [
            item for item in valid_rows
            if item["following"] is not None and item["ahead"] is not None
        ]
        ranked.sort(
            key=lambda item: (
                float(item["following"]["chord_m"]),
                float(item["ahead"]["chord_m"]),
                float(item["requested_duration_s"]),
                int(item["pad_id"]),
            )
        )
        best = ranked[0] if ranked else None
        angle_summaries.append(
            {
                "angle_index": int(angle_index),
                "start_angle_rad": float(angle),
                "dynamic_valid_trajectories": int(len(valid_rows)),
                "best_pad_id": None if best is None else int(best["pad_id"]),
                "best_requested_duration_s": None if best is None else float(best["requested_duration_s"]),
                "best_following": None if best is None else best["following"],
                "best_ahead": None if best is None else best["ahead"],
            }
        )

    payload = {
        "diagnostic_only": True,
        "method": {
            "config": "diagnose_landing_flow.build_config(... continuous_inward_v1, downstream_priority) + rule_flow_coordination_enabled=True",
            "inner_ring_radius_m": float(ring.radius),
            "inner_ring_nominal_speed_mps": float(ring.nominal_speed),
            "safe_distance_m": SAFE_DISTANCE,
            "sample_dt_s": 1.0,
            "gap_grid_chord_m": list(GAP_CHORDS),
            "point_and_swept": True,
            "following_definition": "holder starts behind candidate by the tested positive angular gap",
            "ahead_definition": "holder starts ahead of candidate by the tested positive angular gap",
        },
        "counts": {
            "start_angles": len(START_ANGLES),
            "pads": 6,
            "durations": len(DURATIONS),
            "candidate_geometries": len(all_rows),
            "dynamic_valid": dynamic_valid_count,
            "dynamic_invalid": dynamic_invalid_count,
        },
        "angle_summaries": angle_summaries,
        "candidate_rows": all_rows,
        "symmetric_population_checks": {
            "performed": False,
            "note": "No full simulation or population sweep was run.",
        },
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT),
        "candidate_geometries": len(all_rows),
        "dynamic_valid": dynamic_valid_count,
        "dynamic_invalid": dynamic_invalid_count,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
