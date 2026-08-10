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
        base = item.request.base_priority if hasattr(item, "request") else item.base_priority
        wait = item.external_wait_time if hasattr(item, "external_wait_time") else item.external_wait_time + item.internal_wait_time
        soc = max(0.0, remaining / capacity)
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
        return (0.0 if soc < self.config.soc_emergency else 1.0, -item.dynamic_priority, request_time)
