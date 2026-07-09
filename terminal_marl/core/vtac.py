from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
import math
import numpy as np

from .config import SimConfig
from .geometry import norm_xy, wrap_angle, distance_3d, clamp
from .uav import UAVState, FlightPhase


@dataclass
class VTAC:
    config: SimConfig
    queues: Dict[int, Deque[int]] = field(init=False)
    sequence_counter: int = 0
    merge_reserved_until: Dict[int, int] = field(init=False)
    last_release_step: Dict[int, int] = field(init=False)
    current_step: int = 0
    current_time_s: float = 0.0

    def __post_init__(self) -> None:
        self.queues = {0: deque()}
        for ring in self.config.rings:
            self.queues[ring.index] = deque()
        self.merge_reserved_until = {idx: -1 for idx in self.queues}
        self.last_release_step = {idx: -10**9 for idx in self.queues}

    def reset(self) -> None:
        self.sequence_counter = 0
        self.current_step = 0
        self.current_time_s = 0.0
        for q in self.queues.values():
            q.clear()
        for idx in self.merge_reserved_until:
            self.merge_reserved_until[idx] = -1
            self.last_release_step[idx] = -10**9

    def ring_spec(self, ring_idx: int):
        return next(r for r in self.config.rings if r.index == ring_idx)

    @property
    def outermost_ring_index(self) -> int:
        return max(r.index for r in self.config.rings)

    @property
    def inner_ring_index(self) -> int:
        return min(r.index for r in self.config.rings)

    def queue_capacity(self, queue_idx: int) -> int:
        if queue_idx == 0:
            return self.config.final_queue_capacity
        return self.ring_spec(queue_idx).capacity

    def queue_has_space(self, queue_idx: int) -> bool:
        return len(self.queues[queue_idx]) < self.queue_capacity(queue_idx)

    def gate_angle(self, gate_idx: Optional[int]) -> float:
        if gate_idx is None:
            return 0.0
        return self.config.gate_angles_rad()[gate_idx % self.config.gate_count]

    def transfer_reference_angle(self, gate_idx: Optional[int], target_ring_idx: int) -> float:
        base = self.gate_angle(gate_idx)
        offset_deg = (
            self.config.inner_to_final_release_offset_deg
            if target_ring_idx == 0
            else self.config.outer_to_inner_release_offset_deg
        )
        return base + math.radians(offset_deg) * self.config.ring_direction

    def final_entry_hold_point(self, ac: UAVState) -> np.ndarray:
        ref = self.transfer_reference_angle(ac.assigned_gate, 0)
        return np.array(
            [
                self.config.final_entry_hold_radius * math.cos(ref),
                self.config.final_entry_hold_radius * math.sin(ref),
                self.config.final_entry_hold_altitude,
            ],
            dtype=float,
        )

    def head_of_queue(self, queue_idx: int) -> Optional[int]:
        return self.queues[queue_idx][0] if self.queues[queue_idx] else None

    def is_head_of_queue(self, queue_idx: int, uid: int) -> bool:
        return self.head_of_queue(queue_idx) == uid

    def _forward_angle_delta(self, own_angle: float, other_angle: float) -> float:
        if self.config.ring_direction >= 0:
            return (other_angle - own_angle) % (2.0 * math.pi)
        return (own_angle - other_angle) % (2.0 * math.pi)

    def _ring_tangent_unit(self, angle_rad: float) -> np.ndarray:
        radial = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)
        tangent = np.array([-radial[1], radial[0]], dtype=float)
        if self.config.ring_direction < 0:
            tangent = -tangent
        return tangent

    def _refresh_ring_capture(self, aircraft: List[UAVState], current_step: int) -> None:
        for ac in aircraft:
            if ac.finished or ac.assigned_ring in (None, 0) or ac.transition_active:
                continue
            ring = self.ring_spec(ac.assigned_ring)
            radial_error = abs(norm_xy(ac.position) - ring.radius)
            height_error = abs(ac.position[2] - ring.height)
            captured = (
                radial_error <= self.config.ring_capture_radius_tolerance
                and height_error <= self.config.ring_capture_height_tolerance
            )
            ac.ring_slot_active = captured
            if captured and ac.ring_established_step is None:
                ac.ring_established_step = current_step
            if not captured and ac.phase == FlightPhase.RING and ac.assigned_ring_step == current_step:
                ac.ring_established_step = None

    def _angular_spacing_ok(
        self,
        candidate: UAVState,
        target_ring_idx: int,
        aircraft_map: Dict[int, UAVState],
    ) -> bool:
        if target_ring_idx == 0:
            return True
        candidate_theta = math.atan2(candidate.position[1], candidate.position[0])
        min_sep = math.radians(self.config.ring_merge_spacing_deg)
        for uid in self.queues[target_ring_idx]:
            other = aircraft_map[uid]
            if other.uid == candidate.uid or other.finished:
                continue
            other_theta = math.atan2(other.position[1], other.position[0])
            if abs(wrap_angle(candidate_theta - other_theta)) < min_sep:
                return False
        return True

    def _merge_point_clear(
        self,
        candidate: UAVState,
        target_ring_idx: int,
        aircraft_map: Dict[int, UAVState],
        current_step: int,
    ) -> bool:
        if current_step <= self.merge_reserved_until.get(target_ring_idx, -1):
            return False

        if target_ring_idx == 0:
            return True

        merge_ref = self.transfer_reference_angle(candidate.assigned_gate, target_ring_idx)
        protection = math.radians(self.config.merge_point_protection_deg)
        for uid in self.queues[target_ring_idx]:
            other = aircraft_map[uid]
            if other.uid == candidate.uid or other.finished:
                continue
            other_theta = math.atan2(other.position[1], other.position[0])
            if abs(wrap_angle(other_theta - merge_ref)) <= protection:
                return False
        return True

    def distance_to_release_reference(self, ac: UAVState, target_ring_idx: int) -> float:
        if target_ring_idx == 0:
            return distance_3d(ac.position, self.final_entry_hold_point(ac))
        current_angle = math.atan2(ac.position[1], ac.position[0])
        reference = self.transfer_reference_angle(ac.assigned_gate, target_ring_idx)
        delta = self._forward_angle_delta(current_angle, reference)
        source_ring_idx = ac.assigned_ring if ac.assigned_ring not in (None, 0) else target_ring_idx
        source_radius = self.ring_spec(source_ring_idx).radius
        return delta * source_radius

    def time_to_release_reference(self, ac: UAVState, target_ring_idx: int) -> float:
        return self.distance_to_release_reference(ac, target_ring_idx) / max(ac.speed, self.config.speed_min)

    def desired_reference_arrival_time(
        self,
        candidate: UAVState,
        target_ring_idx: int,
        aircraft_map: Dict[int, UAVState],
    ) -> float:
        min_gap = self.config.final_min_time_gap_s if target_ring_idx == 0 else self.config.ring_to_ring_min_time_gap_s
        desired_abs = self.current_time_s + self.time_to_release_reference(candidate, target_ring_idx)

        elapsed = (self.current_step - self.last_release_step[target_ring_idx]) * self.config.dt
        if elapsed < min_gap:
            desired_abs = max(desired_abs, self.current_time_s + (min_gap - elapsed))

        if target_ring_idx == 0:
            if self.queues[0]:
                predecessor = aircraft_map[self.queues[0][-1]]
                predecessor_eta = distance_3d(predecessor.position, self.final_entry_hold_point(predecessor)) / max(predecessor.speed, self.config.speed_min)
                desired_abs = max(desired_abs, self.current_time_s + predecessor_eta + min_gap)
            return desired_abs

        merge_ref = self.transfer_reference_angle(candidate.assigned_gate, target_ring_idx)
        target_radius = self.ring_spec(target_ring_idx).radius
        for uid in self.queues[target_ring_idx]:
            other = aircraft_map[uid]
            if other.uid == candidate.uid or other.finished:
                continue
            other_speed = max(other.speed, self.config.speed_min)
            other_theta = math.atan2(other.position[1], other.position[0])
            other_delta = self._forward_angle_delta(other_theta, merge_ref)
            other_eta = target_radius * other_delta / other_speed
            period = 2.0 * math.pi * target_radius / other_speed
            candidate_eta = desired_abs - self.current_time_s
            while other_eta + min_gap < candidate_eta - period:
                other_eta += period
            for k in (-1, 0, 1, 2):
                other_abs = self.current_time_s + other_eta + k * period
                if abs(desired_abs - other_abs) < min_gap:
                    desired_abs = other_abs + min_gap
        return desired_abs

    def _within_release_horizon(self, ac: UAVState, target_ring_idx: int) -> bool:
        eta = self.time_to_release_reference(ac, target_ring_idx)
        horizon = self.config.inner_to_final_release_horizon_s if target_ring_idx == 0 else self.config.outer_to_inner_release_horizon_s
        return eta <= horizon

    def _target_ring_time_gap_ok(
        self,
        candidate: UAVState,
        target_ring_idx: int,
        aircraft_map: Dict[int, UAVState],
        current_step: int,
    ) -> bool:
        desired_abs = self.desired_reference_arrival_time(candidate, target_ring_idx, aircraft_map)
        current_abs = self.current_time_s + self.time_to_release_reference(candidate, target_ring_idx)
        return current_abs >= desired_abs - self.config.release_time_alignment_tolerance_s

    def _eligible_for_inward_transfer(
        self,
        ac: UAVState,
        source_ring_idx: int,
        target_ring_idx: int,
        aircraft_map: Dict[int, UAVState],
        current_step: int,
    ) -> bool:
        if ac.finished or ac.phase != FlightPhase.RING:
            return False
        if ac.assigned_ring != source_ring_idx:
            return False
        if not ac.ring_slot_active or ac.ring_established_step is None:
            return False
        if not self._within_release_horizon(ac, target_ring_idx):
            return False

        dwell = current_step - ac.ring_established_step
        required = self.config.inner_to_final_min_dwell_steps if target_ring_idx == 0 else self.config.outer_to_inner_min_dwell_steps
        if dwell < required:
            return False
        if not self._angular_spacing_ok(ac, target_ring_idx, aircraft_map):
            return False
        if not self._merge_point_clear(ac, target_ring_idx, aircraft_map, current_step):
            return False
        if not self._target_ring_time_gap_ok(ac, target_ring_idx, aircraft_map, current_step):
            return False
        return True

    def _activate_transition_curve(self, ac: UAVState, target_ring_idx: int) -> None:
        p0 = ac.position.copy()
        if target_ring_idx == 0:
            ref = self.transfer_reference_angle(ac.assigned_gate, 0)
            p3 = self.final_entry_hold_point(ac)
            current_angle = math.atan2(p0[1], p0[0])
            t0 = self._ring_tangent_unit(current_angle)
            radial_in = -p3[:2] / max(1.0, np.linalg.norm(p3[:2]))
            t3 = radial_in
        else:
            ref = self.transfer_reference_angle(ac.assigned_gate, target_ring_idx)
            ring = self.ring_spec(target_ring_idx)
            p3 = np.array([ring.radius * math.cos(ref), ring.radius * math.sin(ref), ring.height], dtype=float)
            current_angle = math.atan2(p0[1], p0[0])
            t0 = self._ring_tangent_unit(current_angle)
            t3 = self._ring_tangent_unit(ref)

        chord = max(60.0, np.linalg.norm(p3[:2] - p0[:2]))
        tangent_len = clamp(chord * self.config.merge_curve_tangent_fraction,
                            self.config.merge_curve_tangent_min,
                            self.config.merge_curve_tangent_max)
        p1 = p0.copy()
        p1[:2] += tangent_len * t0
        p2 = p3.copy()
        p2[:2] -= tangent_len * t3
        p1[2] = p0[2] + 0.35 * (p3[2] - p0[2])
        p2[2] = p0[2] + 0.75 * (p3[2] - p0[2])

        ac.transition_active = True
        ac.transition_kind = 'final_entry' if target_ring_idx == 0 else 'ring_merge'
        ac.transition_p0 = p0
        ac.transition_p1 = p1
        ac.transition_p2 = p2
        ac.transition_p3 = p3
        ac.transition_progress = 0.0
        ac.ring_slot_active = False
        ac.ring_established_step = None

        if target_ring_idx == 0:
            baseline = distance_3d(p0, p3) / max(self.config.final_entry_speed, self.config.speed_min)
            ac.final_entry_target_time = self.current_time_s + baseline
            ac.final_entry_fix_reached = False

    def update(
        self,
        aircraft: List[UAVState],
        current_step: int,
        release_requests: Optional[Dict[int, bool]] = None,
    ) -> None:
        self.current_step = current_step
        self.current_time_s = current_step * self.config.dt
        aircraft_map = {a.uid: a for a in aircraft}

        for a in aircraft:
            if a.finished or a.sequence_number is not None:
                continue
            dist_xy = norm_xy(a.position)
            if dist_xy <= self.config.approach_threshold_radius and self.queue_has_space(self.outermost_ring_index):
                self.sequence_counter += 1
                a.sequence_number = self.sequence_counter
                a.assigned_ring = self.outermost_ring_index
                a.assigned_ring_step = current_step
                a.last_transfer_step = current_step
                a.phase = FlightPhase.RING
                a.ring_slot_active = False
                a.ring_established_step = None
                a.final_entry_target_time = None
                self.queues[self.outermost_ring_index].append(a.uid)

        self._refresh_ring_capture(aircraft, current_step)

        inner_ring_idx = self.inner_ring_index
        if self.queue_has_space(0) and self.queues[inner_ring_idx]:
            uid = self.queues[inner_ring_idx][0]
            ac = aircraft_map[uid]
            release_requested = release_requests is None or release_requests.get(uid, True)
            if release_requested and self._eligible_for_inward_transfer(ac, inner_ring_idx, 0, aircraft_map, current_step):
                self.queues[inner_ring_idx].popleft()
                self.queues[0].append(uid)
                self.merge_reserved_until[0] = current_step + self.config.merge_reservation_steps
                self.last_release_step[0] = current_step
                ac.final_cleared = True
                ac.assigned_ring = 0
                ac.assigned_ring_step = current_step
                ac.last_transfer_step = current_step
                ac.phase = FlightPhase.FINAL
                self._activate_transition_curve(ac, 0)

        for source_ring_idx in sorted([r.index for r in self.config.rings if r.index != inner_ring_idx], reverse=True):
            target_ring_idx = source_ring_idx - 1
            if not self.queue_has_space(target_ring_idx) or not self.queues[source_ring_idx]:
                continue
            uid = self.queues[source_ring_idx][0]
            ac = aircraft_map[uid]
            if release_requests is not None and not release_requests.get(uid, True):
                continue
            if not self._eligible_for_inward_transfer(ac, source_ring_idx, target_ring_idx, aircraft_map, current_step):
                continue
            self.queues[source_ring_idx].popleft()
            self.queues[target_ring_idx].append(uid)
            self.merge_reserved_until[target_ring_idx] = current_step + self.config.merge_reservation_steps
            self.last_release_step[target_ring_idx] = current_step
            ac.assigned_ring = target_ring_idx
            ac.assigned_ring_step = current_step
            ac.last_transfer_step = current_step
            ac.phase = FlightPhase.RING
            self._activate_transition_curve(ac, target_ring_idx)

        for uid in list(self.queues[0]):
            aircraft_map[uid].phase = FlightPhase.FINAL

    def release_finished(self, aircraft: UAVState) -> None:
        for q in self.queues.values():
            if aircraft.uid in q:
                try:
                    q.remove(aircraft.uid)
                except ValueError:
                    pass
