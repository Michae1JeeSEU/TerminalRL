"""Event gate for structural airspace reconfiguration.

The gate only decides whether structural geometry dimensions may be applied.
It never chooses the direction or magnitude of a geometry update; PPO or an
explainable high-level rule still proposes the requested 8D action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping

import numpy as np


ACTION_GROUPS = {
    "structural_geometry": (0, 1, 2, 3),
    "operational": (4, 5, 6, 7),
}


class GateState(str, Enum):
    NORMAL = "NORMAL"
    PENDING_TRIGGER = "PENDING_TRIGGER"
    TRIGGERED = "TRIGGERED"
    DWELL_LOCKED = "DWELL_LOCKED"
    RECOVERY_ELIGIBLE = "RECOVERY_ELIGIBLE"


class GateMode(str, Enum):
    NONE = "none"
    EVENT = "event"
    PERIODIC = "periodic"

    @classmethod
    def parse(cls, value: "GateMode | str | None") -> "GateMode":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.NONE
        return cls(str(value).lower())


@dataclass(frozen=True)
class GateConfig:
    metrics_sampling_interval_seconds: float = 20.0
    geometry_min_dwell_seconds: float = 600.0
    theta_on: float = 0.7
    theta_off: float = 0.4
    k_on: int = 3
    k_off: int = 6
    recovery_k: int = 6
    nominal_recovery_tolerance: float = 1e-3
    max_active_uavs: int = 40

    def __post_init__(self) -> None:
        if self.metrics_sampling_interval_seconds <= 0.0:
            raise ValueError("metrics_sampling_interval_seconds must be positive")
        if self.geometry_min_dwell_seconds < 0.0:
            raise ValueError("geometry_min_dwell_seconds must be non-negative")
        if not (0.0 <= self.theta_off < self.theta_on):
            raise ValueError("theta_off must be lower than theta_on and non-negative")
        if self.k_on <= 0 or self.k_off <= 0 or self.recovery_k <= 0:
            raise ValueError("persistence counts must be positive")


@dataclass(frozen=True)
class GateMetrics:
    normalized_congestion: float
    approach_backlog_ratio: float
    occupancy_pressure: float
    backlog_growth: float
    normalized_conflict_rate: float
    recent_arrivals: float
    recent_landings: float
    arrival_service_imbalance: float
    active_capacity_ratio: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "GateMetrics":
        return cls(
            normalized_congestion=float(values.get("normalized_congestion", 0.0)),
            approach_backlog_ratio=float(values.get("approach_backlog_ratio", 0.0)),
            occupancy_pressure=float(values.get("occupancy_pressure", 0.0)),
            backlog_growth=float(values.get("backlog_growth", 0.0)),
            normalized_conflict_rate=float(values.get("normalized_conflict_rate", 0.0)),
            recent_arrivals=float(values.get("recent_arrivals", 0.0)),
            recent_landings=float(values.get("recent_landings", 0.0)),
            arrival_service_imbalance=float(values.get("arrival_service_imbalance", 0.0)),
            active_capacity_ratio=float(values.get("active_capacity_ratio", 0.0)),
        )


@dataclass
class GeometryEventGate:
    """Persistence/hysteresis gate for structural geometry dimensions."""

    config: GateConfig = field(default_factory=GateConfig)
    mode: GateMode = GateMode.EVENT
    state: GateState = GateState.NORMAL
    trigger_active: bool = False
    trigger_persistence_count: int = 0
    recovery_persistence_count: int = 0
    last_geometry_change_time: float = -math.inf
    last_sample_time: float = -math.inf
    last_trigger_score: float = 0.0
    backlog_history: list[float] = field(default_factory=list)
    trigger_count: int = 0
    nominal_recovery_count: int = 0
    geometry_opportunity_count: int = 0
    applied_geometry_change_count: int = 0

    def reset(self) -> None:
        self.state = GateState.NORMAL
        self.trigger_active = False
        self.trigger_persistence_count = 0
        self.recovery_persistence_count = 0
        self.last_geometry_change_time = -math.inf
        self.last_sample_time = -math.inf
        self.last_trigger_score = 0.0
        self.backlog_history.clear()
        self.trigger_count = 0
        self.nominal_recovery_count = 0
        self.geometry_opportunity_count = 0
        self.applied_geometry_change_count = 0

    def trigger_score(self, metrics: GateMetrics) -> float:
        """Transparent pressure score in [0, 1]."""

        recent_flow = max(0.0, metrics.recent_arrivals - metrics.recent_landings)
        score = (
            0.22 * metrics.normalized_congestion
            + 0.18 * metrics.approach_backlog_ratio
            + 0.18 * metrics.occupancy_pressure
            + 0.12 * max(0.0, metrics.backlog_growth)
            + 0.12 * metrics.active_capacity_ratio
            + 0.10 * metrics.arrival_service_imbalance
            + 0.05 * recent_flow
            + 0.03 * metrics.normalized_conflict_rate
        )
        return float(np.clip(score, 0.0, 1.0))

    def should_sample(self, time_seconds: float) -> bool:
        return (
            not math.isfinite(self.last_sample_time)
            or time_seconds - self.last_sample_time
            >= self.config.metrics_sampling_interval_seconds - 1e-9
        )

    def update(
        self,
        metrics: GateMetrics,
        *,
        time_seconds: float,
        geometry_deviation_l1: float = 0.0,
        force_sample: bool = False,
    ) -> dict:
        if not (force_sample or self.should_sample(time_seconds)):
            return self.info(time_seconds, geometry_deviation_l1=geometry_deviation_l1)

        self.last_sample_time = float(time_seconds)
        score = self.trigger_score(metrics)
        self.last_trigger_score = score
        self.backlog_history.append(metrics.approach_backlog_ratio)
        if len(self.backlog_history) > 32:
            self.backlog_history = self.backlog_history[-32:]

        was_active = self.trigger_active
        if score >= self.config.theta_on:
            self.trigger_persistence_count += 1
            self.recovery_persistence_count = 0
        elif score <= self.config.theta_off:
            self.recovery_persistence_count += 1
            self.trigger_persistence_count = 0
        else:
            # Hysteresis band: keep current active state but do not accumulate.
            self.trigger_persistence_count = 0
            self.recovery_persistence_count = 0

        if not self.trigger_active and self.trigger_persistence_count >= self.config.k_on:
            self.trigger_active = True
            self.trigger_count += 1
        if self.trigger_active and self.recovery_persistence_count >= self.config.k_off:
            self.trigger_active = False

        if (
            not self.trigger_active
            and self.recovery_persistence_count >= self.config.recovery_k
            and geometry_deviation_l1 > self.config.nominal_recovery_tolerance
            and self.dwell_satisfied(time_seconds)
        ):
            self.nominal_recovery_count += int(was_active or self.state != GateState.RECOVERY_ELIGIBLE)
            self.state = GateState.RECOVERY_ELIGIBLE
        elif self.trigger_active and not self.dwell_satisfied(time_seconds):
            self.state = GateState.DWELL_LOCKED
        elif self.trigger_active:
            self.state = GateState.TRIGGERED
        elif self.trigger_persistence_count > 0:
            self.state = GateState.PENDING_TRIGGER
        else:
            self.state = GateState.NORMAL
        return self.info(time_seconds, geometry_deviation_l1=geometry_deviation_l1)

    def dwell_satisfied(self, time_seconds: float) -> bool:
        if not math.isfinite(self.last_geometry_change_time):
            return True
        return (
            float(time_seconds) - self.last_geometry_change_time
            >= self.config.geometry_min_dwell_seconds - 1e-9
        )

    def seconds_since_geometry_change(self, time_seconds: float) -> float:
        if not math.isfinite(self.last_geometry_change_time):
            return math.inf
        return max(0.0, float(time_seconds) - self.last_geometry_change_time)

    def seconds_until_geometry_allowed(self, time_seconds: float) -> float:
        if self.dwell_satisfied(time_seconds):
            return 0.0
        return max(
            0.0,
            self.config.geometry_min_dwell_seconds
            - self.seconds_since_geometry_change(time_seconds),
        )

    def geometry_allowed(
        self,
        *,
        time_seconds: float,
        geometry_deviation_l1: float = 0.0,
    ) -> tuple[bool, str]:
        mode = GateMode.parse(self.mode)
        dwell = self.dwell_satisfied(time_seconds)
        if mode is GateMode.NONE:
            return True, "gate_disabled"
        if mode is GateMode.PERIODIC:
            if dwell:
                return True, "periodic_dwell_satisfied"
            return False, "dwell_locked"
        if self.trigger_active and dwell:
            return True, "trigger_active_and_dwell_satisfied"
        if (
            self.state is GateState.RECOVERY_ELIGIBLE
            and dwell
            and geometry_deviation_l1 > self.config.nominal_recovery_tolerance
        ):
            return True, "nominal_recovery_eligible"
        if not self.trigger_active:
            return False, "trigger_inactive"
        return False, "dwell_locked"

    def transform_action(
        self,
        requested_action: np.ndarray,
        *,
        time_seconds: float,
        geometry_deviation_l1: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        requested = np.asarray(requested_action, dtype=np.float64).reshape(8)
        applied = requested.copy()
        allowed, reason = self.geometry_allowed(
            time_seconds=time_seconds,
            geometry_deviation_l1=geometry_deviation_l1,
        )
        requested_geometry = requested[list(ACTION_GROUPS["structural_geometry"])].copy()
        if not allowed:
            applied[list(ACTION_GROUPS["structural_geometry"])] = 0.0
        else:
            self.geometry_opportunity_count += 1
        return applied, {
            "requested_action": requested.copy(),
            "applied_action": applied.copy(),
            "requested_geometry_action": requested_geometry,
            "applied_geometry_action": applied[list(ACTION_GROUPS["structural_geometry"])].copy(),
            "geometry_action_allowed": bool(allowed),
            "geometry_gate_reason": reason,
            "geometry_masked": bool(not allowed and np.any(np.abs(requested_geometry) > 1e-9)),
        }

    def mark_geometry_applied(self, time_seconds: float, old_state: np.ndarray, new_state: np.ndarray) -> bool:
        old = np.asarray(old_state, dtype=float).reshape(8)
        new = np.asarray(new_state, dtype=float).reshape(8)
        changed = bool(np.linalg.norm(new[:4] - old[:4], ord=1) > 1e-9)
        if changed:
            self.last_geometry_change_time = float(time_seconds)
            self.applied_geometry_change_count += 1
        return changed

    def info(self, time_seconds: float, *, geometry_deviation_l1: float = 0.0) -> dict:
        dwell = self.dwell_satisfied(time_seconds)
        allowed, reason = self.geometry_allowed(
            time_seconds=time_seconds,
            geometry_deviation_l1=geometry_deviation_l1,
        )
        seconds_since = self.seconds_since_geometry_change(time_seconds)
        seconds_since_norm = (
            1.0
            if not math.isfinite(seconds_since)
            else min(1.0, seconds_since / max(1e-9, self.config.geometry_min_dwell_seconds))
        )
        return {
            "gate_state": self.state.value,
            "geometry_gate_active": bool(self.trigger_active),
            "geometry_dwell_satisfied": bool(dwell),
            "geometry_action_allowed": bool(allowed),
            "geometry_gate_reason": reason,
            "trigger_score": float(self.last_trigger_score),
            "theta_on": float(self.config.theta_on),
            "theta_off": float(self.config.theta_off),
            "trigger_persistence_count": int(self.trigger_persistence_count),
            "recovery_persistence_count": int(self.recovery_persistence_count),
            "seconds_since_geometry_change": float(seconds_since if math.isfinite(seconds_since) else 1e9),
            "seconds_since_geometry_change_normalized": float(seconds_since_norm),
            "seconds_until_geometry_allowed": float(self.seconds_until_geometry_allowed(time_seconds)),
            "geometry_deviation_l1": float(geometry_deviation_l1),
            "trigger_count": int(self.trigger_count),
            "geometry_opportunity_count": int(self.geometry_opportunity_count),
            "applied_geometry_change_count": int(self.applied_geometry_change_count),
            "nominal_recovery_count": int(self.nominal_recovery_count),
        }

    def observation_vector(
        self,
        *,
        time_seconds: float,
        geometry_deviation_l1: float = 0.0,
        backlog_trend: float = 0.0,
        arrival_service_imbalance: float = 0.0,
    ) -> np.ndarray:
        info = self.info(time_seconds, geometry_deviation_l1=geometry_deviation_l1)
        state_index = list(GateState).index(self.state) / max(1, len(GateState) - 1)
        return np.asarray(
            [
                info["trigger_score"],
                float(info["geometry_gate_active"]),
                float(info["geometry_action_allowed"]),
                info["seconds_since_geometry_change_normalized"],
                min(1.0, info["seconds_until_geometry_allowed"] / max(1e-9, self.config.geometry_min_dwell_seconds)),
                min(1.0, geometry_deviation_l1),
                state_index,
                float(np.clip(backlog_trend, -1.0, 1.0)),
                float(np.clip(arrival_service_imbalance, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
