"""A5 low-level reward components and optional P3 urgency shaping."""
from __future__ import annotations

from dataclasses import dataclass
import math


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def low_level_urgency_penalty(
    base_time_penalty: float,
    dt: float,
    base_priority: float,
    soc: float,
    priority_weight: float,
    soc_weight: float,
    soc_urgency_threshold: float,
    soc_reserve: float,
) -> float:
    """Return the optional per-step urgency time cost for one active UAV.

    The two urgency terms are deliberately independent and additive.  Priority
    uses the frozen 1/3/5 scale and SOC urgency ramps linearly from zero at the
    configured threshold to one at the reserve boundary.
    """
    base_penalty = _finite_float(base_time_penalty, "base_time_penalty")
    step_seconds = _finite_float(dt, "dt")
    priority = _finite_float(base_priority, "base_priority")
    state_soc = _finite_float(soc, "soc")
    priority_scale = _finite_float(priority_weight, "priority_weight")
    soc_scale = _finite_float(soc_weight, "soc_weight")
    threshold = _finite_float(soc_urgency_threshold, "soc_urgency_threshold")
    reserve = _finite_float(soc_reserve, "soc_reserve")
    if base_penalty < 0.0 or step_seconds < 0.0:
        raise ValueError("base_time_penalty and dt must be non-negative")
    if priority_scale < 0.0 or soc_scale < 0.0:
        raise ValueError("priority_weight and soc_weight must be non-negative")
    if not reserve < threshold <= 1.0:
        raise ValueError(
            "soc_urgency_threshold must satisfy soc_reserve < threshold <= 1"
        )
    priority_term = min(1.0, max(0.0, (priority - 1.0) / 4.0))
    soc_term = min(1.0, max(0.0, (threshold - state_soc) / (threshold - reserve)))
    return float(-base_penalty * step_seconds * (
        priority_scale * priority_term + soc_scale * soc_term
    ))


# Explicit descriptive alias for callers that prefer a verb-first name.
calculate_low_level_urgency_penalty = low_level_urgency_penalty
compute_low_level_urgency_penalty = low_level_urgency_penalty


@dataclass
class LowLevelReward:
    time: float = 0.0
    progress: float = 0.0
    release: float = 0.0
    touchdown: float = 0.0
    failure: float = 0.0
    teacher: float = 0.0
    shield: float = 0.0
    urgency: float = 0.0
    emergency: float = 0.0

    @property
    def total(self):
        return (
            self.time
            + self.progress
            + self.release
            + self.touchdown
            + self.failure
            + self.teacher
            + self.shield
            + self.urgency
            + self.emergency
        )

    def as_dict(self):
        return {
            "time": self.time,
            "progress": self.progress,
            "release": self.release,
            "touchdown": self.touchdown,
            "failure": self.failure,
            "teacher": self.teacher,
            "shield": self.shield,
            "urgency": self.urgency,
            "emergency": self.emergency,
            "total": self.total,
        }
