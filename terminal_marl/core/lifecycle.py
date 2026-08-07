"""Finite-cohort UAV lifecycle and terminal outcome records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class UAVLifecycleState(str, Enum):
    PENDING_REQUEST = "PENDING_REQUEST"
    EXTERNAL_QUEUE = "EXTERNAL_QUEUE"
    ACTIVE_APPROACH = "ACTIVE_APPROACH"
    ACTIVE_RING_OUTER = "ACTIVE_RING_OUTER"
    ACTIVE_RING_INNER = "ACTIVE_RING_INNER"
    ACTIVE_FINAL = "ACTIVE_FINAL"
    LANDED = "LANDED"
    CRASHED_COLLISION = "CRASHED_COLLISION"
    CRASHED_BATTERY = "CRASHED_BATTERY"
    INVALID = "INVALID"
    UNRESOLVED = "UNRESOLVED"


TERMINAL_OUTCOMES = {
    UAVLifecycleState.LANDED.value,
    UAVLifecycleState.CRASHED_COLLISION.value,
    UAVLifecycleState.CRASHED_BATTERY.value,
    UAVLifecycleState.INVALID.value,
    UAVLifecycleState.UNRESOLVED.value,
}


@dataclass
class UAVOutcomeRecord:
    arrival_id: int
    request_time: float
    scenario: str
    seed: int | None = None
    lifecycle_state: str = UAVLifecycleState.PENDING_REQUEST.value
    admission_time: float | None = None
    terminal_time: float | None = None
    terminal_outcome: str | None = None
    reason: str | None = None
    uid: int | None = None
    last_phase: str | None = None
    last_position: tuple[float, float, float] | None = None
    last_speed: float | None = None
    battery_remaining_seconds: float | None = None
    collision_partners: list[int] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.terminal_outcome in TERMINAL_OUTCOMES

    def set_terminal(
        self,
        *,
        outcome: str,
        time: float,
        reason: str,
        last_phase: str | None = None,
        last_position: tuple[float, float, float] | None = None,
        last_speed: float | None = None,
        battery_remaining_seconds: float | None = None,
        collision_partners: list[int] | None = None,
    ) -> None:
        if self.terminal_outcome is not None:
            return
        self.terminal_outcome = str(outcome)
        self.lifecycle_state = str(outcome)
        self.terminal_time = float(time)
        self.reason = str(reason)
        self.last_phase = last_phase
        self.last_position = last_position
        self.last_speed = last_speed
        self.battery_remaining_seconds = battery_remaining_seconds
        if collision_partners is not None:
            self.collision_partners = [int(uid) for uid in collision_partners]

    def to_row(self) -> dict[str, Any]:
        return {
            "arrival_id": int(self.arrival_id),
            "uid": "" if self.uid is None else int(self.uid),
            "scenario": self.scenario,
            "seed": "" if self.seed is None else int(self.seed),
            "request_time": float(self.request_time),
            "admission_time": "" if self.admission_time is None else float(self.admission_time),
            "terminal_time": "" if self.terminal_time is None else float(self.terminal_time),
            "lifecycle_state": self.lifecycle_state,
            "terminal_outcome": "" if self.terminal_outcome is None else self.terminal_outcome,
            "reason": "" if self.reason is None else self.reason,
            "last_phase": "" if self.last_phase is None else self.last_phase,
            "last_x": "" if self.last_position is None else float(self.last_position[0]),
            "last_y": "" if self.last_position is None else float(self.last_position[1]),
            "last_z": "" if self.last_position is None else float(self.last_position[2]),
            "last_speed": "" if self.last_speed is None else float(self.last_speed),
            "battery_remaining_seconds": (
                ""
                if self.battery_remaining_seconds is None
                else float(self.battery_remaining_seconds)
            ),
            "collision_partners": " ".join(str(uid) for uid in self.collision_partners),
        }
