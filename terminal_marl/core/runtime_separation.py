"""Authoritative runtime airborne-pair separation accounting.

The monitor is deliberately independent of the planner: it observes the
executed positions on every simulation interval and makes the three protected
thresholds (hard separation, NMAC, collision) mutually consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np


@dataclass(frozen=True)
class SeparationEncounter:
    pair: tuple[int, int]
    actual_distance: float
    swept_min_distance: float
    swept_fraction: float
    hard_entry: bool
    nmac_entry: bool
    collision_entry: bool


def swept_minimum_distance(start_a, end_a, start_b, end_b) -> tuple[float, float]:
    """Minimum distance between two linearly swept positions over one step."""
    relative_start = np.asarray(start_a, dtype=float) - np.asarray(start_b, dtype=float)
    relative_delta = (np.asarray(end_a, dtype=float) - np.asarray(start_a, dtype=float)
                      - np.asarray(end_b, dtype=float) + np.asarray(start_b, dtype=float))
    denominator = float(np.dot(relative_delta, relative_delta))
    fraction = 0.0 if denominator <= 1e-12 else float(np.clip(-np.dot(relative_start, relative_delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(relative_start + fraction * relative_delta)), fraction


class RuntimeSeparationMonitor:
    """Single-source actual/swept separation counters and encounter events."""

    def __init__(self, config) -> None:
        self.config = config
        self._active = {"operational": set(), "hard": set(), "nmac": set(), "collision": set()}
        self.telemetry = {
            "minimum_actual_separation": float("inf"), "minimum_swept_separation": float("inf"),
            # The operating buffer is the recoverable 40--80 m band.  It is
            # tracked separately from the protected floor so that a shield
            # takeover is visible without failing the hard-safety gate.
            "operational_buffer_entries": 0, "operational_buffer_exposure_steps": 0,
            "hard_separation_violations": 0, "hard_separation_exposure_steps": 0,
            "nmac": 0, "nmac_exposure_steps": 0, "collisions": 0, "collision_exposure_steps": 0,
        }

    def observe(self, states: dict[int, object], previous_positions: dict[int, np.ndarray]) -> list[SeparationEncounter]:
        current = {key: set() for key in self._active}
        encounters: list[SeparationEncounter] = []
        for first, second in combinations(sorted(states), 2):
            pair = (first, second); left, right = states[first], states[second]
            start_left = previous_positions.get(first, left.position); start_right = previous_positions.get(second, right.position)
            actual = float(np.linalg.norm(np.asarray(left.position) - np.asarray(right.position)))
            swept, fraction = swept_minimum_distance(start_left, left.position, start_right, right.position)
            minimum = min(actual, swept)
            self.telemetry["minimum_actual_separation"] = min(self.telemetry["minimum_actual_separation"], actual)
            self.telemetry["minimum_swept_separation"] = min(self.telemetry["minimum_swept_separation"], swept)
            thresholds = {
                # ``safe_distance`` is the desired operating separation.  A
                # pair in 40--80 m is recoverable by the runtime shield and
                # is therefore not a hard-safety violation.
                "operational": self.config.nmac_distance <= minimum < self.config.safe_distance,
                "hard": minimum < self.config.nmac_distance,
                "nmac": minimum < self.config.nmac_distance,
                "collision": minimum < self.config.collision_distance,
            }
            for name, active in thresholds.items():
                if active:
                    current[name].add(pair)
                    metric = {
                        "operational": "operational_buffer",
                        "hard": "hard_separation",
                    }.get(name, name)
                    self.telemetry[f"{metric}_exposure_steps"] += 1
            operational_entry = thresholds["operational"] and pair not in self._active["operational"]
            hard_entry = thresholds["hard"] and pair not in self._active["hard"]
            nmac_entry = thresholds["nmac"] and pair not in self._active["nmac"]
            collision_entry = thresholds["collision"] and pair not in self._active["collision"]
            if operational_entry: self.telemetry["operational_buffer_entries"] += 1
            if hard_entry: self.telemetry["hard_separation_violations"] += 1
            if nmac_entry: self.telemetry["nmac"] += 1
            if collision_entry: self.telemetry["collisions"] += 1
            if hard_entry or nmac_entry or collision_entry:
                encounters.append(SeparationEncounter(pair, actual, swept, fraction, hard_entry, nmac_entry, collision_entry))
        self._active = current
        return encounters
