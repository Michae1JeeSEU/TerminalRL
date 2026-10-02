"""Frozen inputs for the opt-in, single-pad common arrival benchmark.

This is a bridge between free-flight guidance and terminal service, not the
three-ring training environment. Units are metres, seconds and radians.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CommonArrivalConfig:
    schema: str = "common_arrival_bridge_v1"
    radius: float = 800.0
    capture_radius: float = 200.0
    holding_radius: float = 440.0
    altitude: float = 100.0
    cruise_speed: float = 13.0
    max_turn_deg_s: float = 5.0
    dt: float = 1.0
    physics_dt: float = 0.2
    decision_period: float = 1.0
    prediction_horizon: float = 10.0
    safety_distance: float = 80.0
    nmac_distance: float = 40.0
    collision_distance: float = 10.0
    service_seconds: float = 60.0
    energy_capacity: float = 1000000.0
    initial_soc: float = 1.0
    duration: float = 1500.0
    requests: int = 6
    arrival_interval: float = 75.0
    arrival_pattern: str = "balanced"
    mcts_simulations: int = 100
    mcts_depth: int = 3
    milp_time_limit: float = 2.0
    planner_deadline: float = 5.0
    trace_period: float = 5.0

    def __post_init__(self):
        if self.schema != "common_arrival_bridge_v1":
            raise ValueError("unsupported common-arrival schema")
        positive = ("radius", "capture_radius", "holding_radius", "altitude",
                    "cruise_speed", "max_turn_deg_s", "dt", "physics_dt",
                    "decision_period", "prediction_horizon", "service_seconds",
                    "energy_capacity", "duration", "arrival_interval",
                    "milp_time_limit", "planner_deadline", "trace_period")
        if any(not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0 for key in positive):
            raise ValueError("physical values and budgets must be finite and positive")
        if not 0 < self.collision_distance < self.nmac_distance < self.safety_distance:
            raise ValueError("collision < NMAC < operating separation required")
        if not self.capture_radius + self.safety_distance < self.holding_radius < self.radius:
            raise ValueError("holding region must clear the capture region")
        if not 0 < self.initial_soc <= 1 or self.altitude > 120 or self.cruise_speed > 25:
            raise ValueError("invalid SOC or research-vehicle envelope")
        if self.requests < 1 or self.mcts_simulations < 3 or self.mcts_depth < 1:
            raise ValueError("invalid request count or MCTS budget")
        if abs(self.dt - 1.0) > 1e-9:
            raise ValueError("v1 heading adapters require dt=1 second")
        for value in (self.dt, self.decision_period, self.prediction_horizon, self.trace_period):
            if not math.isclose(value / self.physics_dt, round(value / self.physics_dt)):
                raise ValueError("time periods must be multiples of physics_dt")
        if self.prediction_horizon < self.decision_period or self.prediction_horizon <= self.dt:
            raise ValueError("prediction horizon must cover a decision period")
        if self.arrival_pattern not in {"balanced", "bursty", "direction_shift"}:
            raise ValueError("unknown arrival pattern")

    @classmethod
    def read(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))


@dataclass(frozen=True)
class CommonRequest:
    uid: int
    request_time: float
    gate: int
    speed: float
    initial_soc: float
    service_seconds: float


def generate_requests(config: CommonArrivalConfig, seed: int) -> tuple[CommonRequest, ...]:
    """Generate once, using UID-specific RNG independent of any planner."""
    result = []
    for uid in range(config.requests):
        rng = np.random.default_rng(np.random.SeedSequence([seed, uid, 20261003]))
        when = uid * config.arrival_interval
        gate = uid % 4
        if config.arrival_pattern == "bursty":
            when = (uid // 4) * config.arrival_interval * 4
        elif config.arrival_pattern == "direction_shift":
            probabilities = [0.7, 0.1, 0.1, 0.1] if uid < config.requests / 2 else [0.1, 0.7, 0.1, 0.1]
            gate = int(rng.choice(4, p=probabilities))
        result.append(CommonRequest(uid, when, gate, config.cruise_speed,
                                    config.initial_soc, config.service_seconds))
    return tuple(sorted(result, key=lambda item: (item.request_time, item.uid)))


def validate_requests(requests, config):
    if not requests or len({r.uid for r in requests}) != len(requests):
        raise ValueError("request IDs must be unique and tape nonempty")
    for r in requests:
        if r.uid < 0 or r.gate not in range(4):
            raise ValueError("invalid request ID/gate")
        if not all(math.isfinite(v) for v in (r.request_time, r.speed, r.initial_soc, r.service_seconds)):
            raise ValueError("nonfinite request")
        if not 0 <= r.request_time < config.duration or not 0 < r.initial_soc <= 1:
            raise ValueError("request time/SOC outside declared scenario")
        if r.speed != config.cruise_speed or r.service_seconds != config.service_seconds:
            raise ValueError("v1 bridge uses homogeneous speed and service")
    if list(requests) != sorted(requests, key=lambda r: (r.request_time, r.uid)):
        raise ValueError("request tape must be ordered by time and UID")


def request_digest(requests):
    payload = json.dumps([asdict(r) for r in requests], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi
