"""Runtime UAV arrival processes for traffic-demand experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DEMAND_RATES = {"low": 0.01, "medium": 0.03, "high": 0.06, "bursty": 0.10}


@dataclass
class ArrivalProcess:
    demand_level: str
    model: str
    arrival_rate: float
    max_active_uavs: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.demand_level not in DEMAND_RATES:
            raise ValueError("unsupported demand level")
        if self.model not in {"deterministic_interval", "poisson", "bursty_wave"}:
            raise ValueError("unsupported arrival model")
        if self.max_active_uavs <= 0:
            raise ValueError("max_active_uavs must be positive")
        if self.arrival_rate < 0.0:
            raise ValueError("arrival_rate must be non-negative")
        self.rng = np.random.default_rng(self.seed)
        self.next_arrival_time = math.inf
        self.wave_index = 0
        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.wave_index = 0
        self.next_arrival_time = self._next_time(0.0)

    def _rate(self) -> float:
        return self.arrival_rate or DEMAND_RATES[self.demand_level]

    def _next_time(self, current_time: float) -> float:
        rate = self._rate()
        if rate <= 0.0:
            return math.inf
        if self.model == "poisson":
            return current_time + float(self.rng.exponential(1.0 / rate))
        if self.model == "bursty_wave":
            gap = (0.35 / rate) if self.wave_index % 4 < 3 else (2.5 / rate)
            self.wave_index += 1
            return current_time + gap
        return current_time + 1.0 / rate

    def arrivals_due(self, current_time: float, active_count: int) -> int:
        if active_count >= self.max_active_uavs or current_time < self.next_arrival_time:
            return 0
        count = 1
        if self.model == "bursty_wave":
            count = min(3, self.max_active_uavs - active_count)
        self.next_arrival_time = self._next_time(current_time)
        return count
