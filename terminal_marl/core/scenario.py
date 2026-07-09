
from __future__ import annotations

from typing import List
import math
import numpy as np

from .config import SimConfig
from .geometry import unit_xy_from_angle, rotate90, angle_to_point
from .uav import UAVState


def build_initial_2x4_queue(config: SimConfig) -> List[UAVState]:
    count = config.initial_aircraft_count
    if count != config.formation_rows * config.formation_cols:
        raise ValueError("initial_aircraft_count must equal formation_rows * formation_cols")

    inbound_angle = math.radians(config.inbound_bearing_deg)
    anchor_radius = config.approach_threshold_radius + config.spawn_margin
    anchor_xy = unit_xy_from_angle(inbound_angle) * anchor_radius
    forward = -unit_xy_from_angle(inbound_angle)
    lateral = rotate90(forward)

    aircraft: List[UAVState] = []
    uid = 1
    for row in range(config.formation_rows):
        for col in range(config.formation_cols):
            lateral_offset = (col - (config.formation_cols - 1) / 2.0) * config.col_spacing
            longitudinal_offset = row * config.row_spacing
            pos_xy = anchor_xy + lateral * lateral_offset - forward * longitudinal_offset
            pos = np.array([pos_xy[0], pos_xy[1], config.spawn_altitude], dtype=float)
            heading = angle_to_point(pos, np.array(config.center))
            horizontal_distance = float(np.linalg.norm(pos[:2] - np.asarray(config.center[:2])))
            horizontal_time = horizontal_distance / max(config.nominal_speed, 1e-6)
            descent_time = max(0.0, pos[2] - config.center[2]) / max(
                config.max_descent_rate,
                1e-6,
            )
            nominal_landing_time = (
                horizontal_time
                + descent_time
                + (uid - 1) * config.final_min_time_gap_s
            )
            aircraft.append(
                UAVState(
                    uid=uid,
                    position=pos,
                    speed=config.nominal_speed,
                    heading=heading,
                    spawn_time=0.0,
                    planned_nominal_landing_time=nominal_landing_time,
                )
            )
            uid += 1
    return aircraft
