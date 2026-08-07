"""Runtime configuration for a dynamically reconfigurable terminal airspace.

The module is intentionally independent of the environment and hardware
backends.  A high-level policy supplies bounded *increments* while this class
owns the projection that keeps the two-ring geometry executable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from .ring_capacity import geometric_ring_capacity, minimum_radius_for_occupancy


@dataclass
class DynamicRingSpec:
    """Mutable ring parameters exposed to the terminal-area components."""

    index: int
    radius: float
    height: float
    speed: float
    capacity: int


@dataclass(frozen=True)
class AirspaceBounds:
    """Bounds and per-update rate limits for dynamic airspace parameters."""

    radius_min: float = 50.0
    radius_max: float = 1000.0
    height_min: float = 20.0
    height_max: float = 700.0
    speed_min: float = 5.0
    speed_max: float = 36.0
    min_ring_gap: float = 50.0
    min_height_gap: float = 50.0
    max_radius_change_per_update: float = 20.0
    max_height_change_per_update: float = 20.0
    max_speed_change_per_update: float = 2.0
    headway_min: float = 1.0
    headway_max: float = 60.0
    max_headway_change_per_update: float = 1.0

    def __post_init__(self) -> None:
        pairs = (
            ("radius", self.radius_min, self.radius_max),
            ("height", self.height_min, self.height_max),
            ("speed", self.speed_min, self.speed_max),
            ("headway", self.headway_min, self.headway_max),
        )
        for name, lower, upper in pairs:
            if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
                raise ValueError(f"{name} bounds must be finite and strictly ordered")

        nonnegative = (
            ("min_ring_gap", self.min_ring_gap),
            ("min_height_gap", self.min_height_gap),
            ("max_radius_change_per_update", self.max_radius_change_per_update),
            ("max_height_change_per_update", self.max_height_change_per_update),
            ("max_speed_change_per_update", self.max_speed_change_per_update),
            ("max_headway_change_per_update", self.max_headway_change_per_update),
        )
        for name, value in nonnegative:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        if self.radius_max - self.radius_min < self.min_ring_gap:
            raise ValueError("radius bounds cannot accommodate min_ring_gap")
        if self.height_max - self.height_min < self.min_height_gap:
            raise ValueError("height bounds cannot accommodate min_height_gap")


class DynamicAirspace:
    """Maintain and safely update a two-ring terminal airspace.

    Ring indices and capacities are preserved.  The default geometry mirrors
    :class:`terminal_marl.core.config.SimConfig`, so the class can be created
    without changing the current environment configuration.
    """

    ACTION_SIZE = 8

    def __init__(
        self,
        outer_ring: DynamicRingSpec | None = None,
        inner_ring: DynamicRingSpec | None = None,
        bounds: AirspaceBounds | None = None,
        ring_headway: float = 7.0,
        final_headway: float = 12.0,
        dynamic_enabled: bool = True,
        required_separation: float = 80.0,
        ring_separation_margin: float = 1e-6,
    ) -> None:
        self.bounds = bounds or AirspaceBounds()
        self.outer_ring = replace(
            outer_ring
            or DynamicRingSpec(index=2, radius=400.0, height=400.0, speed=28.0, capacity=6)
        )
        self.inner_ring = replace(
            inner_ring
            or DynamicRingSpec(index=1, radius=200.0, height=200.0, speed=20.0, capacity=2)
        )
        self.ring_headway = float(ring_headway)
        self.final_headway = float(final_headway)
        self.dynamic_enabled = bool(dynamic_enabled)
        self.required_separation = float(required_separation) + float(ring_separation_margin)
        if not math.isfinite(self.required_separation) or self.required_separation <= 0.0:
            raise ValueError("required_separation must be positive and finite")
        self.update_capacities()

        violations = self._validation_violations()
        if violations:
            raise ValueError(f"invalid initial airspace: {', '.join(violations)}")

    @classmethod
    def from_ring_specs(
        cls,
        rings: Sequence[object],
        *,
        bounds: AirspaceBounds | None = None,
        nominal_speeds: Sequence[float] | None = None,
        ring_headway: float = 7.0,
        final_headway: float = 12.0,
        dynamic_enabled: bool = True,
        required_separation: float = 80.0,
        ring_separation_margin: float = 1e-6,
    ) -> "DynamicAirspace":
        """Build from existing ``RingSpec``-like objects.

        The existing project stores rings inner-to-outer and does not include
        speed on ``RingSpec``.  ``nominal_speeds`` fills that compatibility
        gap; otherwise the midpoint of the configured speed bounds is used.
        """

        if len(rings) != 2:
            raise ValueError("DynamicAirspace currently requires exactly two rings")
        active_bounds = bounds or AirspaceBounds()
        if nominal_speeds is not None and len(nominal_speeds) != 2:
            raise ValueError("nominal_speeds must contain one value per ring")

        converted: list[DynamicRingSpec] = []
        default_speed = 0.5 * (active_bounds.speed_min + active_bounds.speed_max)
        for i, ring in enumerate(rings):
            speed = (
                float(nominal_speeds[i])
                if nominal_speeds is not None
                else float(getattr(ring, "speed", default_speed))
            )
            converted.append(
                DynamicRingSpec(
                    index=int(getattr(ring, "index")),
                    radius=float(getattr(ring, "radius")),
                    height=float(getattr(ring, "height")),
                    speed=speed,
                    capacity=int(getattr(ring, "capacity")),
                )
            )

        inner, outer = sorted(converted, key=lambda item: item.radius)
        return cls(
            outer_ring=outer,
            inner_ring=inner,
            bounds=active_bounds,
            ring_headway=ring_headway,
            final_headway=final_headway,
            dynamic_enabled=dynamic_enabled,
            required_separation=required_separation,
            ring_separation_margin=ring_separation_margin,
        )

    def state_vector(self) -> np.ndarray:
        """Return the stable eight-dimensional high-level state ordering."""

        return np.asarray(
            [
                self.outer_ring.radius,
                self.inner_ring.radius,
                self.outer_ring.height,
                self.inner_ring.height,
                self.outer_ring.speed,
                self.inner_ring.speed,
                self.ring_headway,
                self.final_headway,
            ],
            dtype=np.float64,
        )

    def normalized_state_vector(self) -> np.ndarray:
        """Return the airspace state normalized to ``[-1, 1]``."""

        b = self.bounds
        lower = np.asarray(
            [
                b.radius_min,
                b.radius_min,
                b.height_min,
                b.height_min,
                b.speed_min,
                b.speed_min,
                b.headway_min,
                b.headway_min,
            ],
            dtype=np.float64,
        )
        upper = np.asarray(
            [
                b.radius_max,
                b.radius_max,
                b.height_max,
                b.height_max,
                b.speed_max,
                b.speed_max,
                b.headway_max,
                b.headway_max,
            ],
            dtype=np.float64,
        )
        return (2.0 * (self.state_vector() - lower) / (upper - lower) - 1.0).astype(
            np.float32
        )

    def set_state_vector(self, state: np.ndarray) -> None:
        """Replace the actual airspace state and recompute geometric capacity."""

        values = np.asarray(state, dtype=np.float64)
        if values.shape != (self.ACTION_SIZE,):
            raise ValueError(f"state must have shape ({self.ACTION_SIZE},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("state must contain only finite values")
        (
            self.outer_ring.radius,
            self.inner_ring.radius,
            self.outer_ring.height,
            self.inner_ring.height,
            self.outer_ring.speed,
            self.inner_ring.speed,
            self.ring_headway,
            self.final_headway,
        ) = (float(value) for value in values)
        self.update_capacities()
        violations = self._validation_violations()
        if violations:
            raise ValueError(f"invalid airspace state: {', '.join(violations)}")

    def update_capacities(self) -> None:
        """Synchronize ring capacities from the current actual radii."""

        self.outer_ring.capacity = geometric_ring_capacity(
            self.outer_ring.radius,
            self.required_separation,
        )
        self.inner_ring.capacity = geometric_ring_capacity(
            self.inner_ring.radius,
            self.required_separation,
        )

    def capacity_state(self) -> dict[str, int]:
        """Return current geometric capacities keyed by ring role."""

        self.update_capacities()
        return {
            "outer_capacity": int(self.outer_ring.capacity),
            "inner_capacity": int(self.inner_ring.capacity),
        }

    def apply_high_level_action(
        self,
        action: np.ndarray,
        *,
        occupancy_by_ring: dict[int, int] | None = None,
    ) -> dict:
        """Apply a normalized eight-dimensional incremental action.

        Values outside ``[-1, 1]`` are clipped and reported.  Geometry is then
        projected onto the feasible ring-gap and height-gap region while also
        respecting global and per-update bounds.
        """

        raw_action = np.asarray(action, dtype=np.float64)
        if raw_action.shape != (self.ACTION_SIZE,):
            raise ValueError(f"action must have shape ({self.ACTION_SIZE},)")
        if not np.all(np.isfinite(raw_action)):
            raise ValueError("action must contain only finite values")

        old_state = self.state_vector()
        if not self.dynamic_enabled:
            return {
                "clipped": False,
                "violations": [],
                "old_state": old_state,
                "new_state": old_state.copy(),
                "dynamic_enabled": False,
            }

        violations: list[str] = []
        normalized = np.clip(raw_action, -1.0, 1.0)
        if not np.array_equal(normalized, raw_action):
            violations.append("action_bounds")

        b = self.bounds
        scales = np.asarray(
            [
                b.max_radius_change_per_update,
                b.max_radius_change_per_update,
                b.max_height_change_per_update,
                b.max_height_change_per_update,
                b.max_speed_change_per_update,
                b.max_speed_change_per_update,
                b.max_headway_change_per_update,
                b.max_headway_change_per_update,
            ],
            dtype=np.float64,
        )
        candidate = old_state + normalized * scales

        effective_bounds = (
            self._step_bounds(old_state[0], b.radius_min, b.radius_max, b.max_radius_change_per_update),
            self._step_bounds(old_state[1], b.radius_min, b.radius_max, b.max_radius_change_per_update),
            self._step_bounds(old_state[2], b.height_min, b.height_max, b.max_height_change_per_update),
            self._step_bounds(old_state[3], b.height_min, b.height_max, b.max_height_change_per_update),
            self._step_bounds(old_state[4], b.speed_min, b.speed_max, b.max_speed_change_per_update),
            self._step_bounds(old_state[5], b.speed_min, b.speed_max, b.max_speed_change_per_update),
            self._step_bounds(
                old_state[6], b.headway_min, b.headway_max, b.max_headway_change_per_update
            ),
            self._step_bounds(
                old_state[7], b.headway_min, b.headway_max, b.max_headway_change_per_update
            ),
        )
        names = (
            "outer_radius_bounds",
            "inner_radius_bounds",
            "outer_height_bounds",
            "inner_height_bounds",
            "outer_speed_bounds",
            "inner_speed_bounds",
            "ring_headway_bounds",
            "final_headway_bounds",
        )
        for i, ((lower, upper), name) in enumerate(zip(effective_bounds, names)):
            clipped_value = float(np.clip(candidate[i], lower, upper))
            if not math.isclose(clipped_value, float(candidate[i]), rel_tol=0.0, abs_tol=1e-12):
                violations.append(name)
            candidate[i] = clipped_value

        candidate[1], candidate[0], gap_clipped = self._project_ordered_pair(
            lower_value=float(candidate[1]),
            upper_value=float(candidate[0]),
            lower_bounds=effective_bounds[1],
            upper_bounds=effective_bounds[0],
            minimum_gap=b.min_ring_gap,
        )
        if gap_clipped:
            violations.append("min_ring_gap")

        candidate[3], candidate[2], height_clipped = self._project_ordered_pair(
            lower_value=float(candidate[3]),
            upper_value=float(candidate[2]),
            lower_bounds=effective_bounds[3],
            upper_bounds=effective_bounds[2],
            minimum_gap=b.min_height_gap,
        )
        if height_clipped:
            violations.append("min_height_gap")

        occupancy_constraints: dict[str, dict] = {}
        if occupancy_by_ring:
            ring_roles = (
                ("outer", int(self.outer_ring.index), 0),
                ("inner", int(self.inner_ring.index), 1),
            )
            for role, index, state_idx in ring_roles:
                occupancy = int(occupancy_by_ring.get(index, 0))
                if occupancy <= 1:
                    continue
                min_radius = minimum_radius_for_occupancy(
                    occupancy,
                    self.required_separation,
                )
                requested_radius = float(candidate[state_idx])
                if requested_radius + 1e-9 < min_radius:
                    candidate[state_idx] = min(
                        effective_bounds[state_idx][1],
                        max(effective_bounds[state_idx][0], min_radius),
                    )
                    violations.append(f"{role}_radius_occupancy_constraint")
                    occupancy_constraints[role] = {
                        "occupancy": occupancy,
                        "requested_radius": requested_radius,
                        "applied_radius": float(candidate[state_idx]),
                        "minimum_feasible_radius": float(min_radius),
                        "radius_contraction_limited": True,
                    }

        candidate[1], candidate[0], second_gap_clipped = self._project_ordered_pair(
            lower_value=float(candidate[1]),
            upper_value=float(candidate[0]),
            lower_bounds=effective_bounds[1],
            upper_bounds=effective_bounds[0],
            minimum_gap=b.min_ring_gap,
        )
        if second_gap_clipped:
            violations.append("min_ring_gap_after_occupancy_projection")

        (
            self.outer_ring.radius,
            self.inner_ring.radius,
            self.outer_ring.height,
            self.inner_ring.height,
            self.outer_ring.speed,
            self.inner_ring.speed,
            self.ring_headway,
            self.final_headway,
        ) = (float(value) for value in candidate)
        self.update_capacities()

        validation_violations = self._validation_violations()
        if validation_violations:
            raise RuntimeError(
                "airspace projection produced invalid state: "
                + ", ".join(validation_violations)
            )

        return {
            "clipped": bool(violations),
            "violations": list(dict.fromkeys(violations)),
            "old_state": old_state,
            "new_state": self.state_vector(),
            "capacity_state": self.capacity_state(),
            "occupancy_constraints": occupancy_constraints,
            "dynamic_enabled": True,
        }

    def validate(self) -> bool:
        """Return whether the complete current state satisfies all constraints."""

        return not self._validation_violations()

    def to_ring_specs(self) -> list[DynamicRingSpec]:
        """Return defensive ring copies ordered by their existing indices."""

        self.update_capacities()
        return [
            replace(ring)
            for ring in sorted(
                (self.inner_ring, self.outer_ring),
                key=lambda item: item.index,
            )
        ]

    def get_headways(self) -> dict[str, float]:
        """Return ring-transfer and final-approach release headways."""

        return {
            "ring": float(self.ring_headway),
            "final": float(self.final_headway),
        }

    @staticmethod
    def _step_bounds(
        old_value: float,
        global_min: float,
        global_max: float,
        max_change: float,
    ) -> tuple[float, float]:
        return max(global_min, old_value - max_change), min(global_max, old_value + max_change)

    @staticmethod
    def _project_ordered_pair(
        *,
        lower_value: float,
        upper_value: float,
        lower_bounds: tuple[float, float],
        upper_bounds: tuple[float, float],
        minimum_gap: float,
    ) -> tuple[float, float, bool]:
        """Project ``upper_value - lower_value >= minimum_gap``."""

        deficit = minimum_gap - (upper_value - lower_value)
        if deficit <= 1e-12:
            return lower_value, upper_value, False

        lower_min, _lower_max = lower_bounds
        _upper_min, upper_max = upper_bounds
        lower_room = lower_value - lower_min
        upper_room = upper_max - upper_value
        if lower_room + upper_room + 1e-12 < deficit:
            raise RuntimeError("per-update bounds cannot preserve the required ring ordering")

        lower_shift = min(0.5 * deficit, lower_room)
        upper_shift = min(0.5 * deficit, upper_room)
        remaining = deficit - lower_shift - upper_shift
        if remaining > 0.0:
            extra_upper = min(remaining, upper_room - upper_shift)
            upper_shift += extra_upper
            remaining -= extra_upper
        if remaining > 0.0:
            lower_shift += min(remaining, lower_room - lower_shift)

        return lower_value - lower_shift, upper_value + upper_shift, True

    def _validation_violations(self) -> list[str]:
        b = self.bounds
        violations: list[str] = []
        values = self.state_vector()
        if not np.all(np.isfinite(values)):
            violations.append("non_finite_state")

        for label, value, lower, upper in (
            ("outer_radius_bounds", self.outer_ring.radius, b.radius_min, b.radius_max),
            ("inner_radius_bounds", self.inner_ring.radius, b.radius_min, b.radius_max),
            ("outer_height_bounds", self.outer_ring.height, b.height_min, b.height_max),
            ("inner_height_bounds", self.inner_ring.height, b.height_min, b.height_max),
            ("outer_speed_bounds", self.outer_ring.speed, b.speed_min, b.speed_max),
            ("inner_speed_bounds", self.inner_ring.speed, b.speed_min, b.speed_max),
            ("ring_headway_bounds", self.ring_headway, b.headway_min, b.headway_max),
            ("final_headway_bounds", self.final_headway, b.headway_min, b.headway_max),
        ):
            if not lower <= value <= upper:
                violations.append(label)

        if self.outer_ring.radius - self.inner_ring.radius < b.min_ring_gap - 1e-9:
            violations.append("min_ring_gap")
        if self.outer_ring.height - self.inner_ring.height < b.min_height_gap - 1e-9:
            violations.append("min_height_gap")
        if self.outer_ring.capacity <= 0 or self.inner_ring.capacity <= 0:
            violations.append("ring_capacity")
        if self.outer_ring.index == self.inner_ring.index:
            violations.append("ring_index")
        return violations
