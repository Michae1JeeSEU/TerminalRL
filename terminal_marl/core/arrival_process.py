"""Runtime UAV arrival request processes for traffic-demand experiments.

The arrival process represents *external demand requests*.  It must not censor
requests based on the terminal's internal active-aircraft cap; admission is an
environment responsibility so all policies can be evaluated against identical
request demand.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


DEMAND_RATES = {"low": 0.01, "medium": 0.03, "high": 0.06, "bursty": 0.10}


@dataclass
class ArrivalProcess:
    demand_level: str
    model: str
    arrival_rate: float
    max_active_uavs: int
    seed: int | None = None
    fixed_schedule: tuple[float, ...] | None = None
    total_requested_uavs: int | None = None

    def __post_init__(self) -> None:
        if self.demand_level not in DEMAND_RATES:
            raise ValueError("unsupported demand level")
        if self.model not in {"deterministic_interval", "poisson", "bursty_wave"}:
            raise ValueError("unsupported arrival model")
        if self.max_active_uavs <= 0:
            raise ValueError("max_active_uavs must be positive")
        if self.arrival_rate < 0.0:
            raise ValueError("arrival_rate must be non-negative")
        if self.total_requested_uavs is not None and self.total_requested_uavs < 0:
            raise ValueError("total_requested_uavs must be non-negative")
        self.rng = np.random.default_rng(self.seed)
        self.next_arrival_time = math.inf
        self.wave_index = 0
        self.schedule_index = 0
        self.generated_count = 0
        if self.fixed_schedule is not None:
            self.fixed_schedule = tuple(sorted(float(t) for t in self.fixed_schedule))
        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.wave_index = 0
        self.schedule_index = 0
        self.generated_count = 0
        if self.fixed_schedule is not None:
            self.next_arrival_time = (
                self.fixed_schedule[0] if self.fixed_schedule else math.inf
            )
        else:
            self.next_arrival_time = self._next_time(0.0)

    def _rate(self) -> float:
        return self.arrival_rate

    def _next_time(self, current_time: float) -> float:
        rate = self._rate()
        if rate <= 0.0:
            return math.inf
        if self.model == "poisson":
            return current_time + float(self.rng.exponential(1.0 / rate))
        if self.model == "bursty_wave":
            # Piecewise burst/off intervals with unit requests.  The four-gap
            # cycle is scaled so its long-run mean rate equals ``arrival_rate``:
            # three short burst gaps plus one longer recovery gap average 1/rate.
            short_gap = 0.25 / rate
            long_gap = 3.25 / rate
            gap = short_gap if self.wave_index % 4 < 3 else long_gap
            self.wave_index += 1
            return current_time + gap
        return current_time + 1.0 / rate

    def arrivals_due(self, current_time: float, active_count: int | None = None) -> int:
        """Return the number of external arrival requests due by ``current_time``.

        ``active_count`` is accepted for backward compatibility but intentionally
        ignored.  Active-cap admission is handled by ``TerminalParallelEnv`` via
        an external/pre-entry queue, preventing policy-dependent demand
        censoring.
        """

        if self.total_requested_uavs is not None and self.generated_count >= self.total_requested_uavs:
            self.next_arrival_time = math.inf
            return 0
        if current_time < self.next_arrival_time:
            return 0
        if self.fixed_schedule is not None:
            count = 0
            while (
                self.schedule_index < len(self.fixed_schedule)
                and self.fixed_schedule[self.schedule_index] <= current_time + 1e-9
            ):
                count += 1
                self.schedule_index += 1
                if (
                    self.total_requested_uavs is not None
                    and self.generated_count + count >= self.total_requested_uavs
                ):
                    break
            if self.total_requested_uavs is not None:
                count = min(count, self.total_requested_uavs - self.generated_count)
            self.generated_count += count
            self.next_arrival_time = (
                self.fixed_schedule[self.schedule_index]
                if self.schedule_index < len(self.fixed_schedule)
                and (
                    self.total_requested_uavs is None
                    or self.generated_count < self.total_requested_uavs
                )
                else math.inf
            )
            return count
        count = 1
        if self.total_requested_uavs is not None:
            count = min(count, self.total_requested_uavs - self.generated_count)
        self.generated_count += count
        self.next_arrival_time = self._next_time(current_time)
        if self.total_requested_uavs is not None and self.generated_count >= self.total_requested_uavs:
            self.next_arrival_time = math.inf
        return count


def generate_arrival_schedule(
    *,
    demand_level: str,
    model: str,
    arrival_rate: float,
    seed: int,
    duration_seconds: float,
    total_requested_uavs: int | None = None,
) -> list[dict[str, float | int | str]]:
    """Generate a deterministic external-demand request schedule.

    The generated schedule is policy-independent and can be replayed by passing
    the ``request_time`` values to ``ArrivalProcess(fixed_schedule=...)``.
    """

    process = ArrivalProcess(
        demand_level=demand_level,
        model=model,
        arrival_rate=arrival_rate,
        max_active_uavs=1,
        seed=seed,
        total_requested_uavs=total_requested_uavs,
    )
    rows: list[dict[str, float | int | str]] = []
    arrival_id = 0
    current_time = 0.0
    while process.next_arrival_time <= duration_seconds + 1e-9:
        current_time = float(process.next_arrival_time)
        due = process.arrivals_due(current_time)
        for burst_index in range(due):
            rows.append(
                {
                    "arrival_id": int(arrival_id),
                    "request_time": float(current_time),
                    "scenario": demand_level,
                    "seed": int(seed),
                    "model": model,
                    "burst_index": int(burst_index),
                    "burst_size": int(due),
                }
            )
            arrival_id += 1
        if not math.isfinite(process.next_arrival_time):
            break
        if total_requested_uavs is not None and arrival_id >= total_requested_uavs:
            break
    return rows


def schedule_times(rows: Iterable[dict[str, float | int | str]]) -> tuple[float, ...]:
    """Extract sorted request times from a generated schedule."""

    return tuple(sorted(float(row["request_time"]) for row in rows))
