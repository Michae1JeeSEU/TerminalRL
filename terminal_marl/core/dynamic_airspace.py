"""Canonical three-ring, event-gated dynamic terminal geometry."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .config import RingRole, SimConfig
from .ring_capacity import geometric_ring_capacity, minimum_radius_for_occupancy


@dataclass
class RingState:
    role: RingRole
    radius: float
    height: float
    nominal_speed: float
    derived_capacity: int = 0


@dataclass
class GeometryTransition:
    start_time: float
    end_time: float
    start: dict[RingRole, RingState]
    target: dict[RingRole, RingState]
    affected_interfaces: set[str]


@dataclass(frozen=True)
class AirspaceUpdateResult:
    accepted: bool
    structural_applied: bool
    violations: tuple[str, ...]
    affected_interfaces: frozenset[str]
    requested_action: tuple[float, ...]
    applied_action: tuple[float, ...]
    structural_mask: tuple[bool, ...]
    requested_radius_delta: tuple[float, ...] = (0.0, 0.0, 0.0)
    projected_radius_delta: tuple[float, ...] = (0.0, 0.0, 0.0)
    applied_radius_delta: tuple[float, ...] = (0.0, 0.0, 0.0)
    capacity_before: tuple[int, ...] = (0, 0, 0)
    capacity_after: tuple[int, ...] = (0, 0, 0)
    structural_reason_codes: tuple[str, ...] = ()


class DynamicAirspace:
    """Owns actual/target geometry; capacity is always derived from *actual*."""

    ACTION_SIZE = 10

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self._actual = {
            role: RingState(role, spec.radius, spec.height, spec.nominal_speed)
            for role, spec in config.rings.items()
        }
        self._target = {role: replace(state) for role, state in self._actual.items()}
        self.speed_profile_bias = 0.0
        self.headways = {
            "o2m": config.headway_o2m,
            "m2i": config.headway_m2i,
            "i2f": config.headway_i2f,
        }
        self.transition: GeometryTransition | None = None
        self.last_geometry_update = -float("inf")
        self._occupancy = {role: 0 for role in RingRole}
        self._committed = {role: 0 for role in RingRole}
        self._update_capacity()

    def _update_capacity(self) -> None:
        for role in RingRole:
            self._actual[role].derived_capacity = geometric_ring_capacity(
                self._actual[role].radius, self.config.safe_distance
            )

    def actual_ring(self, role: RingRole) -> RingState:
        return self._actual[role]

    def target_ring(self, role: RingRole) -> RingState:
        return self._target[role]

    def capacity(self, role: RingRole) -> int:
        return self._actual[role].derived_capacity

    def occupancy(self, role: RingRole) -> int:
        return self._occupancy[role]

    def set_occupancy(
        self,
        values: Mapping[RingRole, int],
        committed_incoming: Mapping[RingRole, int] | None = None,
    ) -> None:
        self._occupancy = {role: int(values.get(role, 0)) for role in RingRole}
        incoming = committed_incoming or {}
        self._committed = {role: int(incoming.get(role, 0)) for role in RingRole}

    def state_vector(self) -> np.ndarray:
        return np.asarray(
            [self._actual[role].radius for role in RingRole]
            + [self._actual[role].height for role in RingRole]
            + [self.speed_profile_bias, self.headways["o2m"], self.headways["m2i"], self.headways["i2f"]],
            dtype=np.float32,
        )

    def normalized_state_vector(self) -> np.ndarray:
        c = self.config
        low = np.asarray([c.radius_min] * 3 + [c.height_min] * 3 + [-2.0, c.headway_min] * 1 + [c.headway_min] * 2)
        high = np.asarray([c.radius_max] * 3 + [c.height_max] * 3 + [2.0] + [c.headway_max] * 3)
        return (2.0 * (self.state_vector() - low) / (high - low) - 1.0).astype(np.float32)

    def affected_interfaces(self) -> set[str]:
        return set() if self.transition is None else set(self.transition.affected_interfaces)

    def transition_progress(self, now: float) -> float:
        if self.transition is None:
            return 0.0
        span = max(1e-9, self.transition.end_time - self.transition.start_time)
        return float(np.clip((now - self.transition.start_time) / span, 0.0, 1.0))

    def step_transition(self, now: float) -> None:
        if self.transition is None:
            return
        transition = self.transition
        progress = self.transition_progress(now)
        for role in RingRole:
            start, target = transition.start[role], transition.target[role]
            self._actual[role].radius = start.radius + (target.radius - start.radius) * progress
            self._actual[role].height = start.height + (target.height - start.height) * progress
        self._update_capacity()
        if progress >= 1.0:
            self.transition = None

    def _project_candidate(self, candidate: dict[RingRole, RingState], violations: list[str]) -> None:
        c = self.config
        # Start at inner and work outward so ordering remains valid after the
        # occupancy projection changes a radius.
        for role in (RingRole.INNER, RingRole.MIDDLE, RingRole.OUTER):
            minimum = minimum_radius_for_occupancy(
                self._occupancy[role] + self._committed[role], c.safe_distance
            )
            if candidate[role].radius < minimum:
                candidate[role].radius = minimum
                violations.append("occupancy_radius_projection")
        candidate[RingRole.MIDDLE].radius = max(
            candidate[RingRole.MIDDLE].radius,
            candidate[RingRole.INNER].radius + c.min_ring_gap,
        )
        candidate[RingRole.OUTER].radius = max(
            candidate[RingRole.OUTER].radius,
            candidate[RingRole.MIDDLE].radius + c.min_ring_gap,
        )
        candidate[RingRole.MIDDLE].height = max(
            candidate[RingRole.MIDDLE].height,
            candidate[RingRole.INNER].height + c.min_height_gap,
        )
        candidate[RingRole.OUTER].height = max(
            candidate[RingRole.OUTER].height,
            candidate[RingRole.MIDDLE].height + c.min_height_gap,
        )
        for role in RingRole:
            candidate[role].radius = float(np.clip(candidate[role].radius, c.radius_min, c.radius_max))
            candidate[role].height = float(np.clip(candidate[role].height, c.height_min, c.height_max))

    @staticmethod
    def _affected_interfaces(changed_roles: set[RingRole]) -> set[str]:
        affected: set[str] = set()
        if RingRole.OUTER in changed_roles or RingRole.MIDDLE in changed_roles:
            affected.add("o2m")
        if RingRole.MIDDLE in changed_roles or RingRole.INNER in changed_roles:
            affected.add("m2i")
        if RingRole.INNER in changed_roles:
            affected.add("i2f")
        return affected

    def request_update(
        self,
        action: np.ndarray,
        *,
        now: float,
        occupancies: Mapping[RingRole, int] | None = None,
        committed_incoming: Mapping[RingRole, int] | None = None,
        geometry_gate: bool = True,
    ) -> AirspaceUpdateResult:
        raw = np.asarray(action, dtype=np.float32)
        if raw.shape != (self.ACTION_SIZE,) or not np.all(np.isfinite(raw)):
            raise ValueError("high-level action must be finite shape (10,)")
        self.set_occupancy(occupancies or self._occupancy, committed_incoming)
        clipped = np.clip(raw, -1.0, 1.0)
        c = self.config
        violations: list[str] = []
        structural_reasons: list[str] = []
        structural_mask = [False] * 6
        candidate = {role: replace(state) for role, state in self._target.items()}
        previous_target = {role: replace(state) for role, state in self._target.items()}
        capacity_before = tuple(geometric_ring_capacity(previous_target[role].radius, c.safe_distance) for role in RingRole)
        eligible = geometry_gate and self.transition is None and now - self.last_geometry_update >= c.geometry_min_dwell_seconds
        if eligible:
            for index, role in enumerate(RingRole):
                candidate[role].radius += float(clipped[index] * c.max_radius_change_per_update)
                candidate[role].height += float(clipped[index + 3] * c.max_height_change_per_update)
            self._project_candidate(candidate, violations)
            for index, role in enumerate(RingRole):
                requested_delta = float(clipped[index] * c.max_radius_change_per_update)
                prior_capacity = geometric_ring_capacity(previous_target[role].radius, c.safe_distance)
                candidate_capacity = geometric_ring_capacity(candidate[role].radius, c.safe_distance)
                if abs(requested_delta) <= 1e-9 or candidate_capacity != prior_capacity:
                    continue
                if requested_delta > 0.0:
                    threshold = minimum_radius_for_occupancy(prior_capacity + 1, c.safe_distance)
                    if threshold <= previous_target[role].radius + c.max_radius_change_per_update + 1e-9:
                        candidate[role].radius = max(candidate[role].radius, threshold)
                        structural_reasons.append(f"capacity_threshold_projection:{role.value}")
                    else:
                        candidate[role].radius = previous_target[role].radius
                        structural_reasons.append(f"capacity_threshold_noop:{role.value}")
                else:
                    candidate[role].radius = previous_target[role].radius
                    structural_reasons.append(f"capacity_threshold_noop:{role.value}")
            self._project_candidate(candidate, violations)
            changed = {
                role for role in RingRole
                if abs(candidate[role].radius - self._target[role].radius) > 1e-7
                or abs(candidate[role].height - self._target[role].height) > 1e-7
            }
            if changed:
                structural_mask = [True] * 6
                affected = self._affected_interfaces(changed)
                self.transition = GeometryTransition(
                    now, now + c.geometry_transition_seconds,
                    {role: replace(state) for role, state in self._actual.items()},
                    candidate, affected,
                )
                self._target = candidate
                self.last_geometry_update = now
            else:
                affected = set()
        else:
            violations.append("geometry_gate" if not geometry_gate else "geometry_dwell_or_transition")
            affected = self.affected_interfaces()
        projected_radius_delta = tuple(candidate[role].radius - previous_target[role].radius for role in RingRole)
        capacity_after = tuple(geometric_ring_capacity(candidate[role].radius, c.safe_distance) for role in RingRole)

        # The last four dimensions are deliberately never event-masked.
        self.speed_profile_bias = float(np.clip(
            self.speed_profile_bias + clipped[6] * c.max_speed_profile_change_per_update, -2.0, 2.0
        ))
        for key, index in (("o2m", 7), ("m2i", 8), ("i2f", 9)):
            self.headways[key] = float(np.clip(
                self.headways[key] + clipped[index] * c.max_headway_change_per_update,
                c.headway_min, c.headway_max,
            ))
        applied = np.concatenate((clipped[:6] * np.asarray(structural_mask, dtype=np.float32), clipped[6:]))
        return AirspaceUpdateResult(
            accepted=True,
            structural_applied=bool(any(structural_mask)),
            violations=tuple(violations),
            affected_interfaces=frozenset(affected),
            requested_action=tuple(float(x) for x in raw),
            applied_action=tuple(float(x) for x in applied),
            structural_mask=tuple(structural_mask),
            requested_radius_delta=tuple(float(clipped[index] * c.max_radius_change_per_update) for index in range(3)),
            projected_radius_delta=tuple(float(value) for value in projected_radius_delta),
            applied_radius_delta=tuple(float(value) if structural_mask[index] else 0.0 for index, value in enumerate(projected_radius_delta)),
            capacity_before=tuple(int(value) for value in capacity_before),
            capacity_after=tuple(int(value) for value in capacity_after),
            structural_reason_codes=tuple(structural_reasons),
        )
