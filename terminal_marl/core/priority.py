"""Canonical dynamic-priority formula from requirements.md."""
from __future__ import annotations
from dataclasses import dataclass
from .config import SimConfig

@dataclass(frozen=True)
class PriorityEngine:
    config: SimConfig
    def compute(self, item, now: float | None = None) -> float:
        capacity = item.request.energy_capacity if hasattr(item, "request") else item.energy_capacity
        remaining = item.energy_remaining
        base = (item.request.base_priority if hasattr(item, "request") else item.base_priority) \
            if self.config.task_priority_enabled else 1
        wait = item.external_wait_time
        if self.config.priority_wait_mode == "total_wait_v1":
            wait += getattr(item, "internal_wait_time", 0.0)
        soc = max(0.0, remaining / capacity)
        if self.config.emergency_landing_enabled:
            threshold = self.config.emergency_trigger_soc
            span = max(threshold - self.config.min_landing_soc, 1e-9)
            urgency = min(1.0, max(0.0, (threshold - soc) / span))
        else:
            # Preserve historical qualification semantics unless the new
            # emergency curriculum is explicitly enabled.
            urgency = max(0.0, (self.config.soc_reserve - soc) / self.config.soc_reserve)
        c = self.config
        return float(base * (1 + c.priority_alpha * (wait / c.priority_wait_reference_seconds) ** c.priority_beta) * (1 + c.priority_gamma * urgency ** c.priority_delta))
    def update(self, item, now: float | None = None) -> float:
        item.dynamic_priority = self.compute(item, now)
        return item.dynamic_priority
    def rank_key(self, item, now: float = 0.0) -> tuple[float, float, int]:
        self.update(item, now)
        soc = item.soc
        request_time = item.request.request_time if hasattr(item, "request") else item.request_time
        urgent_threshold = (self.config.emergency_trigger_soc if self.config.emergency_landing_enabled
                            else self.config.soc_emergency)
        return (0.0 if soc <= urgent_threshold else 1.0, -item.dynamic_priority, request_time)
