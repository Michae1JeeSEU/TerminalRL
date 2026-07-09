from __future__ import annotations

from typing import List, Tuple, Optional
import math
import numpy as np

from .config import SimConfig
from .geometry import angle_to_point, wrap_angle, norm_xy, distance_3d, clamp, rotate90
from .uav import UAVState, FlightPhase
from .vtac import VTAC


def gate_positions(config: SimConfig) -> List[np.ndarray]:
    return [
        np.array(
            [
                config.approach_threshold_radius * math.cos(angle),
                config.approach_threshold_radius * math.sin(angle),
                config.gate_height,
            ],
            dtype=float,
        )
        for angle in config.gate_angles_rad()
    ]


def choose_nearest_gate(ac: UAVState, config: SimConfig) -> int:
    gates = gate_positions(config)
    distances = [distance_3d(ac.position, g) for g in gates]
    return int(np.argmin(distances))


def _ring_spec(ac: UAVState, config: SimConfig):
    return next(r for r in config.rings if r.index == ac.assigned_ring)


def _ring_angle(ac: UAVState) -> float:
    return math.atan2(ac.position[1], ac.position[0])


def _forward_angle_delta(own_angle: float, other_angle: float, ring_direction: int) -> float:
    if ring_direction >= 0:
        return (other_angle - own_angle) % (2.0 * math.pi)
    return (own_angle - other_angle) % (2.0 * math.pi)


def transfer_reference_angle(config: SimConfig, ac: UAVState, target_ring_idx: int) -> float:
    gate_idx = ac.assigned_gate if ac.assigned_gate is not None else 0
    base = config.gate_angles_rad()[gate_idx % config.gate_count]
    offset_deg = config.inner_to_final_release_offset_deg if target_ring_idx == 0 else config.outer_to_inner_release_offset_deg
    return base + math.radians(offset_deg) * config.ring_direction


def final_entry_hold_point(config: SimConfig, ac: UAVState) -> np.ndarray:
    ref = transfer_reference_angle(config, ac, 0)
    return np.array(
        [
            config.final_entry_hold_radius * math.cos(ref),
            config.final_entry_hold_radius * math.sin(ref),
            config.final_entry_hold_altitude,
        ],
        dtype=float,
    )


def bezier_point(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    u = 1.0 - t
    return (u ** 3) * p0 + 3.0 * (u ** 2) * t * p1 + 3.0 * u * (t ** 2) * p2 + (t ** 3) * p3


def bezier_tangent(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    u = 1.0 - t
    return 3.0 * (u ** 2) * (p1 - p0) + 6.0 * u * t * (p2 - p1) + 3.0 * (t ** 2) * (p3 - p2)


def closest_bezier_progress(ac: UAVState) -> float:
    if not ac.transition_active or ac.transition_p0 is None:
        return 0.0
    ts = np.linspace(max(0.0, ac.transition_progress - 0.1), 1.0, 21)
    best_t = ac.transition_progress
    best_d = float('inf')
    for t in ts:
        pt = bezier_point(ac.transition_p0, ac.transition_p1, ac.transition_p2, ac.transition_p3, float(t))
        d = distance_3d(ac.position, pt)
        if d < best_d:
            best_d = d
            best_t = float(t)
    return best_t


def same_ring_spacing_speed_limit(ac: UAVState, aircraft: List[UAVState], config: SimConfig) -> Tuple[float | None, float | None]:
    if ac.phase != FlightPhase.RING or ac.assigned_ring in (None, 0):
        return None, None

    own_angle = _ring_angle(ac)
    own_z = ac.position[2]
    candidates: List[Tuple[float, float]] = []

    for other in aircraft:
        if other.uid == ac.uid or other.finished:
            continue
        if other.phase != FlightPhase.RING or other.assigned_ring != ac.assigned_ring or other.transition_active:
            continue
        if abs(other.position[2] - own_z) > config.ring_capture_height_tolerance * 2.0:
            continue
        delta = _forward_angle_delta(own_angle, _ring_angle(other), config.ring_direction)
        if delta <= 1e-6 or delta > math.radians(config.same_ring_lookahead_deg):
            continue
        candidates.append((delta, distance_3d(ac.position, other.position)))

    if not candidates:
        return None, None

    min_delta, gap_dist = min(candidates, key=lambda x: x[0])
    gap_deg = math.degrees(min_delta)
    if gap_deg >= config.same_ring_desired_spacing_deg:
        return config.speed_max, gap_deg
    if gap_deg <= config.same_ring_min_spacing_deg:
        return config.speed_min, gap_deg

    ratio = (config.same_ring_desired_spacing_deg - gap_deg) / max(1e-6, config.same_ring_desired_spacing_deg - config.same_ring_min_spacing_deg)
    reduction = ratio * config.same_ring_speed_reduction_max
    speed_cap = clamp(config.nominal_speed - reduction, config.speed_min, config.speed_max)
    if gap_dist < config.safe_distance:
        extra = (config.safe_distance - gap_dist) / max(1e-6, config.safe_distance)
        speed_cap = clamp(speed_cap - 0.5 * config.same_ring_speed_reduction_max * extra, config.speed_min, config.speed_max)
    return speed_cap, gap_deg


def arrival_time_guided_speed(
    ac: UAVState,
    base_speed: float,
    target_ring_idx: int,
    aircraft: List[UAVState],
    config: SimConfig,
    vtac: Optional[VTAC],
) -> float | None:
    if vtac is None or ac.phase != FlightPhase.RING or ac.assigned_ring in (None, 0):
        return None
    if not vtac.is_head_of_queue(ac.assigned_ring, ac.uid):
        return None
    eta = vtac.time_to_release_reference(ac, target_ring_idx)
    horizon = config.inner_to_final_release_horizon_s if target_ring_idx == 0 else config.outer_to_inner_release_horizon_s
    if eta > horizon:
        return None
    aircraft_map = {a.uid: a for a in aircraft}
    target_abs = vtac.desired_reference_arrival_time(ac, target_ring_idx, aircraft_map)
    remaining = max(config.time_guidance_min_remaining_s, target_abs - vtac.current_time_s)
    distance = max(1.0, vtac.distance_to_release_reference(ac, target_ring_idx))
    tracked_speed = clamp(distance / remaining, config.speed_min, config.speed_max)
    desired = (1.0 - config.time_guidance_blend) * base_speed + config.time_guidance_blend * tracked_speed
    desired = min(desired, base_speed + config.time_guidance_max_speedup)
    return clamp(desired, config.speed_min, config.speed_max)


def final_entry_time_guided_speed(ac: UAVState, base_speed: float, config: SimConfig, vtac: Optional[VTAC]) -> float:
    if vtac is None or ac.final_entry_fix_reached or ac.final_entry_target_time is None:
        return base_speed
    stage_point = final_entry_hold_point(config, ac)
    distance = max(1.0, distance_3d(ac.position, stage_point))
    remaining = max(config.time_guidance_min_remaining_s, ac.final_entry_target_time - vtac.current_time_s)
    tracked_speed = clamp(distance / remaining, config.speed_min, config.speed_max)
    desired = (1.0 - config.time_guidance_blend) * base_speed + config.time_guidance_blend * tracked_speed
    desired = min(desired, base_speed + config.time_guidance_max_speedup)
    return clamp(desired, config.speed_min, config.speed_max)


def _transition_spacing_speed_cap(
    ac: UAVState,
    aircraft: List[UAVState],
    config: SimConfig,
    own_progress: float,
) -> Tuple[float | None, float | None]:
    if not ac.transition_active or ac.transition_p3 is None:
        return None, None

    endpoint = ac.transition_p3
    endpoint_radius = norm_xy(endpoint)
    endpoint_angle = math.atan2(endpoint[1], endpoint[0]) if endpoint_radius > 1e-6 else 0.0
    nearest_sep: float | None = None

    for other in aircraft:
        if other.uid == ac.uid or other.finished:
            continue

        candidate_sep: float | None = None

        if other.transition_active and other.transition_p3 is not None:
            if distance_3d(other.transition_p3, endpoint) <= config.transition_same_target_radius:
                other_progress = closest_bezier_progress(other)
                if other_progress >= own_progress - 0.02:
                    candidate_sep = distance_3d(ac.position, other.position)

        if ac.transition_kind == 'ring_merge' and other.phase == FlightPhase.RING and other.assigned_ring == ac.assigned_ring:
            if not other.transition_active and abs(other.position[2] - endpoint[2]) <= config.ring_capture_height_tolerance * 2.0:
                other_angle = math.atan2(other.position[1], other.position[0])
                delta = _forward_angle_delta(endpoint_angle, other_angle, config.ring_direction)
                if 0.0 <= delta <= math.radians(config.transition_target_ring_lookahead_deg):
                    arc_gap = max(endpoint_radius, 1.0) * delta
                    straight_gap = distance_3d(ac.position, other.position)
                    candidate_sep = min(straight_gap, arc_gap + distance_3d(ac.position, endpoint))

        if ac.transition_kind == 'final_entry' and other.phase == FlightPhase.FINAL:
            if distance_3d(other.position, endpoint) <= config.transition_endpoint_hold_radius * 1.35:
                straight_gap = distance_3d(ac.position, other.position)
                candidate_sep = straight_gap if candidate_sep is None else min(candidate_sep, straight_gap)

        if candidate_sep is None:
            continue
        nearest_sep = candidate_sep if nearest_sep is None else min(nearest_sep, candidate_sep)

    if nearest_sep is None:
        return None, None

    if own_progress >= config.transition_endpoint_progress_guard:
        for other in aircraft:
            if other.uid == ac.uid or other.finished:
                continue
            if distance_3d(other.position, endpoint) <= config.transition_endpoint_hold_radius:
                nearest_sep = min(nearest_sep, distance_3d(ac.position, other.position))

    if nearest_sep >= config.transition_front_desired_spacing:
        return config.speed_max, nearest_sep
    if nearest_sep <= config.transition_front_min_spacing:
        return config.speed_min, nearest_sep

    ratio = (config.transition_front_desired_spacing - nearest_sep) / max(
        1e-6, config.transition_front_desired_spacing - config.transition_front_min_spacing
    )
    reduction = ratio * config.transition_speed_reduction_max
    speed_cap = clamp(config.nominal_speed - reduction, config.speed_min, config.speed_max)
    return speed_cap, nearest_sep


def transition_curve_command(
    ac: UAVState,
    aircraft: List[UAVState],
    config: SimConfig,
    vtac: Optional[VTAC],
) -> Tuple[float, float, float]:
    t_now = closest_bezier_progress(ac)
    ac.transition_progress = t_now
    t_cmd = min(1.0, t_now + config.merge_curve_lookahead)
    target_pt = bezier_point(ac.transition_p0, ac.transition_p1, ac.transition_p2, ac.transition_p3, t_cmd)
    tangent = bezier_tangent(ac.transition_p0, ac.transition_p1, ac.transition_p2, ac.transition_p3, t_cmd)

    if np.linalg.norm(tangent[:2]) > 1e-6:
        raw_heading = math.atan2(tangent[1], tangent[0])
    else:
        raw_heading = angle_to_point(ac.position, target_pt)
    desired_heading = wrap_angle(ac.heading + config.transition_heading_smoothing * wrap_angle(raw_heading - ac.heading))

    if ac.transition_kind == 'final_entry':
        desired_speed = final_entry_time_guided_speed(ac, config.final_entry_speed, config, vtac)
    else:
        desired_speed = clamp(config.nominal_speed - config.merge_curve_speed_reduction, config.speed_min, config.speed_max)

    speed_cap, _ = _transition_spacing_speed_cap(ac, aircraft, config, t_now)
    if speed_cap is not None:
        desired_speed = min(desired_speed, speed_cap)

    desired_altitude = float(target_pt[2])

    if t_now >= 0.97 or distance_3d(ac.position, ac.transition_p3) <= config.merge_curve_capture_radius:
        ac.clear_transition()
        if ac.phase == FlightPhase.FINAL:
            ac.final_entry_fix_reached = True
        else:
            ac.ring_slot_active = False
            ac.ring_established_step = None

    return desired_heading, desired_speed, desired_altitude


def ring_guidance_command(ac: UAVState, aircraft: List[UAVState], config: SimConfig, vtac: Optional[VTAC] = None) -> Tuple[float, float, float]:
    ring_spec = _ring_spec(ac, config)
    xy = ac.position[:2]
    radius_now = max(1.0, norm_xy(ac.position))
    radial_unit = xy / radius_now
    tangent_unit = rotate90(radial_unit)
    if config.ring_direction < 0:
        tangent_unit = -tangent_unit

    radial_error = radius_now - ring_spec.radius
    radial_term = clamp(radial_error / max(1.0, config.ring_radial_error_scale), -1.5, 1.5)
    desired_vec = config.ring_tangent_gain * tangent_unit - config.ring_radial_gain * radial_term * radial_unit
    if np.linalg.norm(desired_vec) < 1e-6:
        desired_vec = tangent_unit
    raw_heading = math.atan2(desired_vec[1], desired_vec[0])
    desired_heading = wrap_angle(ac.heading + config.ring_heading_smoothing * wrap_angle(raw_heading - ac.heading))

    speed_reduction = min(config.ring_speed_reduction_max, abs(radial_error) / max(1.0, config.ring_radial_error_scale) * config.ring_speed_reduction_max)
    ring_nominal_speed = (
        config.nominal_speed if ring_spec.speed is None else float(ring_spec.speed)
    )
    desired_speed = clamp(
        ring_nominal_speed - 1.0 - speed_reduction,
        config.speed_min,
        config.speed_max,
    )

    speed_cap, _ = same_ring_spacing_speed_limit(ac, aircraft, config)
    if speed_cap is not None:
        desired_speed = min(desired_speed, speed_cap)

    if vtac is not None and vtac.is_head_of_queue(ac.assigned_ring, ac.uid):
        target_ring_idx = 0 if ac.assigned_ring == vtac.inner_ring_index else ac.assigned_ring - 1
        guided_speed = arrival_time_guided_speed(ac, desired_speed, target_ring_idx, aircraft, config, vtac)
        if guided_speed is not None:
            desired_speed = guided_speed

    return desired_heading, desired_speed, ring_spec.height


def simple_separation_adjustment(ac: UAVState, aircraft: List[UAVState], desired_heading: float, config: SimConfig) -> float:
    repulse = 0.0
    for other in aircraft:
        if other.uid == ac.uid or other.finished:
            continue
        d = distance_3d(ac.position, other.position)
        if d < config.safe_distance and d > 1e-6:
            rel = other.position[:2] - ac.position[:2]
            bearing = math.atan2(rel[1], rel[0])
            away = wrap_angle(bearing + math.pi - ac.heading)
            repulse += -0.35 if away < 0 else 0.35
    return wrap_angle(desired_heading + repulse)


def command_for_aircraft(ac: UAVState, aircraft: List[UAVState], config: SimConfig, vtac: Optional[VTAC] = None) -> Tuple[float, float, float]:
    gates = gate_positions(config)

    if ac.phase == FlightPhase.FINISHED:
        return ac.heading, 0.0, ac.position[2]

    if ac.assigned_gate is None:
        ac.assigned_gate = choose_nearest_gate(ac, config)

    if ac.sequence_number is None or ac.phase == FlightPhase.APPROACH:
        gate = gates[ac.assigned_gate]
        desired_heading = simple_separation_adjustment(ac, aircraft, angle_to_point(ac.position, gate), config)
        return desired_heading, config.nominal_speed, config.gate_height

    if ac.transition_active:
        desired_heading, desired_speed, desired_altitude = transition_curve_command(ac, aircraft, config, vtac)
        desired_heading = simple_separation_adjustment(ac, aircraft, desired_heading, config)
        return desired_heading, desired_speed, desired_altitude

    if ac.phase == FlightPhase.RING and ac.assigned_ring not in (None, 0):
        desired_heading, desired_speed, desired_altitude = ring_guidance_command(ac, aircraft, config, vtac)
        desired_heading = simple_separation_adjustment(ac, aircraft, desired_heading, config)
        return desired_heading, desired_speed, desired_altitude

    if ac.phase == FlightPhase.FINAL:
        stage_point = final_entry_hold_point(config, ac)
        if not ac.final_entry_fix_reached:
            if distance_3d(ac.position, stage_point) <= config.final_entry_capture_radius:
                ac.final_entry_fix_reached = True
            desired_heading = angle_to_point(ac.position, stage_point)
            desired_speed = final_entry_time_guided_speed(ac, config.final_entry_speed, config, vtac)
            desired_altitude = config.final_entry_hold_altitude
        else:
            target = np.array([0.0, 0.0, 0.0], dtype=float)
            desired_heading = angle_to_point(ac.position, target)
            desired_speed = clamp(config.final_approach_speed, config.speed_min, config.speed_max)
            desired_altitude = 0.0
        desired_heading = simple_separation_adjustment(ac, aircraft, desired_heading, config)
        return desired_heading, desired_speed, desired_altitude

    return ac.heading, config.nominal_speed, ac.position[2]
