
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import numpy as np

from .config import SimConfig
from .geometry import angle_to_point, wrap_angle, norm_xy, distance_3d
from .uav import UAVState, FlightPhase
from .vtac import VTAC
from .baseline import gate_positions, final_entry_hold_point
from .dynamic_airspace import DynamicAirspace


def _forward_target(ac: UAVState, config: SimConfig) -> np.ndarray:
    if ac.phase == FlightPhase.APPROACH or ac.sequence_number is None:
        gate_idx = ac.assigned_gate if ac.assigned_gate is not None else 0
        return gate_positions(config)[gate_idx]
    if ac.transition_active and ac.transition_p3 is not None:
        return ac.transition_p3
    if ac.phase == FlightPhase.RING and ac.assigned_ring not in (None, 0):
        ring = next(r for r in config.rings if r.index == ac.assigned_ring)
        theta = math.atan2(ac.position[1], ac.position[0]) + config.ring_direction * math.radians(20.0)
        return np.array([ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height], dtype=float)
    if ac.phase == FlightPhase.FINAL and not ac.final_entry_fix_reached:
        return final_entry_hold_point(config, ac)
    return np.array([0.0, 0.0, 0.0], dtype=float)


def progress_error(ac: UAVState, config: SimConfig) -> float:
    if ac.phase == FlightPhase.FINISHED or ac.finished:
        return 0.0
    if ac.phase == FlightPhase.APPROACH or ac.sequence_number is None:
        gate_idx = ac.assigned_gate if ac.assigned_gate is not None else 0
        target = gate_positions(config)[gate_idx]
        return distance_3d(ac.position, target)
    if ac.transition_active and ac.transition_p3 is not None:
        return distance_3d(ac.position, ac.transition_p3)
    if ac.phase == FlightPhase.RING and ac.assigned_ring not in (None, 0):
        ring = next(r for r in config.rings if r.index == ac.assigned_ring)
        radial_error = abs(norm_xy(ac.position) - ring.radius)
        height_error = abs(ac.position[2] - ring.height)
        return radial_error + 0.5 * height_error
    if ac.phase == FlightPhase.FINAL and not ac.final_entry_fix_reached:
        return distance_3d(ac.position, final_entry_hold_point(config, ac))
    return distance_3d(ac.position, np.array([0.0, 0.0, 0.0], dtype=float))


@dataclass
class AgentObservationBuilder:
    config: SimConfig
    airspace: Optional[DynamicAirspace] = None

    def observation_length(self) -> int:
        # self(12) + phase_one_hot(4) + task(8) + queues(3) + neighbors(k*8)
        base_length = (
            12
            + len(FlightPhase)
            + 8
            + (len(self.config.rings) + 1)
            + self.config.neighbor_k * 8
        )
        return base_length + (DynamicAirspace.ACTION_SIZE if self.airspace is not None else 0)

    def build(self, ac: UAVState, aircraft: List[UAVState], vtac: VTAC) -> np.ndarray:
        cfg = self.config
        target = _forward_target(ac, cfg)
        rel_target = target - ac.position
        target_bearing = angle_to_point(ac.position, target)
        bearing_err = wrap_angle(target_bearing - ac.heading)
        ring_norm = 0.0 if ac.assigned_ring is None else ac.assigned_ring / max(1, len(cfg.rings))
        seq_norm = 0.0 if ac.sequence_number is None else ac.sequence_number / max(1, cfg.initial_aircraft_count)
        self_vec = [
            ac.position[0] / cfg.world_xy_limit,
            ac.position[1] / cfg.world_xy_limit,
            ac.position[2] / cfg.world_z_limit,
            ac.speed / cfg.speed_max,
            math.sin(ac.heading),
            math.cos(ac.heading),
            math.sin(ac.pitch),
            math.cos(ac.pitch),
            1.0 if ac.transition_active else 0.0,
            1.0 if ac.ring_slot_active else 0.0,
            1.0 if ac.final_entry_fix_reached else 0.0,
            1.0 if ac.finished else 0.0,
        ]
        phase_one_hot = [1.0 if ac.phase == phase else 0.0 for phase in FlightPhase]
        task_vec = [
            ring_norm,
            seq_norm,
            rel_target[0] / cfg.world_xy_limit,
            rel_target[1] / cfg.world_xy_limit,
            rel_target[2] / cfg.world_z_limit,
            math.sin(bearing_err),
            math.cos(bearing_err),
            progress_error(ac, cfg) / (cfg.world_xy_limit + cfg.world_z_limit),
        ]
        queue_vec = [len(vtac.queues[0]) / max(1, cfg.final_queue_capacity)]
        for ring in cfg.rings:
            queue_vec.append(len(vtac.queues[ring.index]) / max(1, ring.capacity))

        others: List[Tuple[float, UAVState]] = []
        for other in aircraft:
            if other.uid == ac.uid or other.finished:
                continue
            others.append((distance_3d(ac.position, other.position), other))
        others.sort(key=lambda x: x[0])

        nbr_vec: List[float] = []
        for _, other in others[: cfg.neighbor_k]:
            rel = other.position - ac.position
            rel_bearing = wrap_angle(angle_to_point(ac.position, other.position) - ac.heading)
            nbr_vec.extend([
                rel[0] / cfg.world_xy_limit,
                rel[1] / cfg.world_xy_limit,
                rel[2] / cfg.world_z_limit,
                (other.speed - ac.speed) / cfg.speed_max,
                math.sin(rel_bearing),
                math.cos(rel_bearing),
                distance_3d(ac.position, other.position) / (cfg.world_xy_limit + cfg.world_z_limit),
                1.0 if other.transition_active else 0.0,
            ])
        while len(nbr_vec) < cfg.neighbor_k * 8:
            nbr_vec.extend([0.0] * 8)

        airspace_vec = (
            self.airspace.normalized_state_vector().tolist()
            if self.airspace is not None
            else []
        )
        obs = np.asarray(
            self_vec + phase_one_hot + task_vec + queue_vec + nbr_vec + airspace_vec,
            dtype=np.float32,
        )
        return obs
