
from __future__ import annotations

from typing import List
import math
import numpy as np

from .config import SimConfig
from .geometry import unit_xy_from_angle, rotate90, angle_to_point
from .uav import UAVState
from ..metrics.nominal_schedule import NominalScheduleEstimator


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
    estimator = NominalScheduleEstimator(config)
    uid = 1
    for row in range(config.formation_rows):
        for col in range(config.formation_cols):
            lateral_offset = (col - (config.formation_cols - 1) / 2.0) * config.col_spacing
            longitudinal_offset = row * config.row_spacing
            pos_xy = anchor_xy + lateral * lateral_offset - forward * longitudinal_offset
            pos = np.array([pos_xy[0], pos_xy[1], config.spawn_altitude], dtype=float)
            heading = angle_to_point(pos, np.array(config.center))
            nominal_landing_time = estimator.estimate(
                pos, spawn_time=0.0, sequence_index=uid
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
