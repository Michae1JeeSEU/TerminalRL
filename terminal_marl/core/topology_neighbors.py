"""Deterministic, online traffic-topology features for the V2 low actor.

The selector deliberately exposes *roles*, not an arbitrary nearest-k list:
the learned actor therefore has a constant input width even when the active
traffic set changes.  All tie breaks use ``uid`` so agent-container ordering
cannot change an observation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .config import RingRole


TAU = 2.0 * math.pi


@dataclass(frozen=True)
class NeighborSlot:
    present: bool = False
    uid: int | None = None
    path_gap: float = 0.0
    euclidean_distance: float = 0.0
    relative_speed: float = 0.0
    closing_rate: float = 0.0
    d_cpa: float = 0.0
    t_cpa: float = 0.0
    priority_delta: float = 0.0
    reserved_or_committed: bool = False


def _angle(uav) -> float:
    if uav.ring_progress is not None:
        return float(uav.ring_progress) % TAU
    return float(math.atan2(uav.position[1], uav.position[0])) % TAU


def _tangent_velocity(uav, airspace) -> np.ndarray:
    if np.linalg.norm(uav.velocity) > 1e-6:
        return np.asarray(uav.velocity, dtype=float)
    if uav.ring_role is None:
        return np.zeros(3, dtype=float)
    ring = airspace.actual_ring(uav.ring_role)
    theta = _angle(uav)
    speed = ring.nominal_speed + airspace.speed_profile_bias
    return np.asarray((-speed * math.sin(theta), speed * math.cos(theta), 0.0))


def _slot(ego, other, airspace, *, path_gap: float, horizon: float) -> NeighborSlot:
    relative_position = np.asarray(other.position, dtype=float) - np.asarray(ego.position, dtype=float)
    ego_velocity = _tangent_velocity(ego, airspace)
    other_velocity = _tangent_velocity(other, airspace)
    relative_velocity = other_velocity - ego_velocity
    distance = float(np.linalg.norm(relative_position))
    speed_sq = float(np.dot(relative_velocity, relative_velocity))
    t_cpa = 0.0 if speed_sq <= 1e-10 else float(np.clip(-np.dot(relative_position, relative_velocity) / speed_sq, 0.0, horizon))
    d_cpa = float(np.linalg.norm(relative_position + relative_velocity * t_cpa))
    direction = relative_position / max(distance, 1e-9)
    closing = max(0.0, -float(np.dot(relative_velocity, direction)))
    return NeighborSlot(
        present=True, uid=int(other.uid), path_gap=float(path_gap), euclidean_distance=distance,
        relative_speed=float(np.linalg.norm(other_velocity) - np.linalg.norm(ego_velocity)),
        closing_rate=closing, d_cpa=d_cpa, t_cpa=t_cpa,
        priority_delta=float(other.dynamic_priority - ego.dynamic_priority),
        reserved_or_committed=bool(other.committed_plan is not None or other.merge_reservation_id is not None),
    )


class TopologyNeighborSelector:
    def __init__(self, airspace, *, now: float, cpa_horizon: float = 30.0) -> None:
        self.airspace = airspace
        self.now = float(now)
        self.cpa_horizon = float(cpa_horizon)

    @staticmethod
    def _ordered(items: Iterable) -> list:
        return sorted(items, key=lambda item: int(item.uid))

    def same_ring_leader(self, ego, active: Iterable) -> NeighborSlot:
        if ego.ring_role is None:
            return NeighborSlot()
        ring = self.airspace.actual_ring(ego.ring_role)
        candidates = [item for item in self._ordered(active) if item.uid != ego.uid and item.ring_role is ego.ring_role]
        if not candidates:
            return NeighborSlot()
        ego_angle = _angle(ego)
        item, gap_angle = min(((item, (_angle(item) - ego_angle) % TAU) for item in candidates), key=lambda row: (row[1], row[0].uid))
        return _slot(ego, item, self.airspace, path_gap=gap_angle * ring.radius, horizon=self.cpa_horizon)

    def same_ring_follower(self, ego, active: Iterable) -> NeighborSlot:
        if ego.ring_role is None:
            return NeighborSlot()
        ring = self.airspace.actual_ring(ego.ring_role)
        candidates = [item for item in self._ordered(active) if item.uid != ego.uid and item.ring_role is ego.ring_role]
        if not candidates:
            return NeighborSlot()
        ego_angle = _angle(ego)
        item, gap_angle = min(((item, (ego_angle - _angle(item)) % TAU) for item in candidates), key=lambda row: (row[1], row[0].uid))
        return _slot(ego, item, self.airspace, path_gap=-gap_angle * ring.radius, horizon=self.cpa_horizon)

    def _target_role(self, ego) -> RingRole | None:
        return {RingRole.OUTER: RingRole.MIDDLE, RingRole.MIDDLE: RingRole.INNER}.get(ego.ring_role)

    def _target_neighbors(self, ego, active: Iterable) -> tuple[NeighborSlot, NeighborSlot]:
        target = self._target_role(ego)
        if target is None:
            return NeighborSlot(), NeighborSlot()
        ring = self.airspace.actual_ring(target)
        # This is a baseline online prediction only; it never sees a teacher
        # action or a post-action reservation.
        source = self.airspace.actual_ring(ego.ring_role)
        merge_time = max(0.0, abs(source.radius - ring.radius) / max(1.0, ring.nominal_speed) + 6.0)
        expected = (_angle(ego) + ring.nominal_speed * merge_time / ring.radius) % TAU
        candidates = [item for item in self._ordered(active) if item.uid != ego.uid and item.ring_role is target]
        if not candidates:
            return NeighborSlot(), NeighborSlot()
        predicted = [(item, (_angle(item) + ring.nominal_speed * merge_time / ring.radius) % TAU) for item in candidates]
        pred, pred_gap = min(predicted, key=lambda row: ((row[1] - expected) % TAU, row[0].uid))
        succ, succ_gap = min(predicted, key=lambda row: ((expected - row[1]) % TAU, row[0].uid))
        return (
            _slot(ego, pred, self.airspace, path_gap=pred_gap * ring.radius, horizon=self.cpa_horizon),
            _slot(ego, succ, self.airspace, path_gap=-succ_gap * ring.radius, horizon=self.cpa_horizon),
        )

    def target_ring_predecessor(self, ego, active: Iterable) -> NeighborSlot:
        return self._target_neighbors(ego, active)[0]

    def target_ring_successor(self, ego, active: Iterable) -> NeighborSlot:
        return self._target_neighbors(ego, active)[1]

    def nearest_predicted_conflict(self, ego, active: Iterable) -> NeighborSlot:
        candidates = [item for item in self._ordered(active) if item.uid != ego.uid]
        if not candidates:
            return NeighborSlot()
        rows = [_slot(ego, item, self.airspace, path_gap=0.0, horizon=self.cpa_horizon) for item in candidates]
        return min(rows, key=lambda row: (row.d_cpa, row.t_cpa, row.uid if row.uid is not None else -1))


class TopologyCache:
    """One-step cache of V2 topology roles with unchanged selector semantics."""

    def __init__(self, airspace, active: Iterable, *, now: float, cpa_horizon: float) -> None:
        self._slots: dict[int, tuple[NeighborSlot, NeighborSlot, NeighborSlot, NeighborSlot, NeighborSlot]] = {}
        ordered = tuple(sorted(active, key=lambda item: int(item.uid)))
        count = len(ordered)
        if count == 0:
            return
        positions = np.asarray([item.position for item in ordered], dtype=float)
        velocities = np.asarray([_tangent_velocity(item, airspace) for item in ordered], dtype=float)
        relative_position = positions[None, :, :] - positions[:, None, :]
        relative_velocity = velocities[None, :, :] - velocities[:, None, :]
        distances = np.linalg.norm(relative_position, axis=2)
        speed_sq = np.einsum("ijk,ijk->ij", relative_velocity, relative_velocity)
        dot = np.einsum("ijk,ijk->ij", relative_position, relative_velocity)
        t_cpa = np.zeros((count, count), dtype=float)
        moving = speed_sq > 1e-10
        t_cpa[moving] = np.clip(-dot[moving] / speed_sq[moving], 0.0, cpa_horizon)
        d_cpa = np.linalg.norm(relative_position + relative_velocity * t_cpa[:, :, None], axis=2)
        direction = relative_position / np.maximum(distances[:, :, None], 1e-9)
        closing = np.maximum(0.0, -np.einsum("ijk,ijk->ij", relative_velocity, direction))
        speeds = np.linalg.norm(velocities, axis=1)
        angles = np.asarray([_angle(item) for item in ordered], dtype=float)
        uids = np.asarray([int(item.uid) for item in ordered], dtype=int)

        def slot(ego_index: int, other_index: int, path_gap: float) -> NeighborSlot:
            ego, other = ordered[ego_index], ordered[other_index]
            return NeighborSlot(
                present=True, uid=int(other.uid), path_gap=float(path_gap),
                euclidean_distance=float(distances[ego_index, other_index]),
                relative_speed=float(speeds[other_index] - speeds[ego_index]),
                closing_rate=float(closing[ego_index, other_index]),
                d_cpa=float(d_cpa[ego_index, other_index]), t_cpa=float(t_cpa[ego_index, other_index]),
                priority_delta=float(other.dynamic_priority - ego.dynamic_priority),
                reserved_or_committed=bool(other.committed_plan is not None or other.merge_reservation_id is not None),
            )

        for ego_index, ego in enumerate(ordered):
            same = [index for index, item in enumerate(ordered)
                    if index != ego_index and item.ring_role is ego.ring_role] if ego.ring_role is not None else []
            if same:
                ring = airspace.actual_ring(ego.ring_role)
                forward = [((angles[index] - angles[ego_index]) % TAU, int(uids[index]), index) for index in same]
                backward = [((angles[ego_index] - angles[index]) % TAU, int(uids[index]), index) for index in same]
                leader_gap, _, leader_index = min(forward)
                follower_gap, _, follower_index = min(backward)
                leader = slot(ego_index, leader_index, leader_gap * ring.radius)
                follower = slot(ego_index, follower_index, -follower_gap * ring.radius)
            else:
                leader = follower = NeighborSlot()

            target = {RingRole.OUTER: RingRole.MIDDLE, RingRole.MIDDLE: RingRole.INNER}.get(ego.ring_role)
            target_indices = [index for index, item in enumerate(ordered)
                              if index != ego_index and item.ring_role is target] if target is not None else []
            if target_indices:
                source_ring = airspace.actual_ring(ego.ring_role)
                target_ring = airspace.actual_ring(target)
                merge_time = max(0.0, abs(source_ring.radius - target_ring.radius) / max(1.0, target_ring.nominal_speed) + 6.0)
                expected = (angles[ego_index] + target_ring.nominal_speed * merge_time / target_ring.radius) % TAU
                predicted = [
                    (index, (angles[index] + target_ring.nominal_speed * merge_time / target_ring.radius) % TAU)
                    for index in target_indices
                ]
                _, _, pred_index, pred_angle = min(
                    ((angle - expected) % TAU, int(uids[index]), index, angle) for index, angle in predicted
                )
                _, _, succ_index, succ_angle = min(
                    ((expected - angle) % TAU, int(uids[index]), index, angle) for index, angle in predicted
                )
                # Preserve the established V2 feature semantics: the selector
                # stores the selected predicted absolute angle as path_gap.
                predecessor = slot(ego_index, pred_index, pred_angle * target_ring.radius)
                successor = slot(ego_index, succ_index, -succ_angle * target_ring.radius)
            else:
                predecessor = successor = NeighborSlot()

            conflict_indices = [index for index in range(count) if index != ego_index]
            if conflict_indices:
                conflict_index = min(conflict_indices, key=lambda index: (
                    d_cpa[ego_index, index], t_cpa[ego_index, index], int(uids[index])
                ))
                conflict = slot(ego_index, conflict_index, 0.0)
            else:
                conflict = NeighborSlot()
            self._slots[int(ego.uid)] = (leader, follower, predecessor, successor, conflict)

    def slots_for(self, uid: int) -> tuple[NeighborSlot, NeighborSlot, NeighborSlot, NeighborSlot, NeighborSlot]:
        return self._slots.get(int(uid), (NeighborSlot(),) * 5)
