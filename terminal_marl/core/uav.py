"""Phase-A domain entities.  Request attributes remain immutable."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np

from .config import RingRole


class MissionType(str, Enum):
    STANDARD = "standard"
    CARGO = "cargo"
    EMERGENCY = "emergency"
    SURVEILLANCE = "surveillance"


MISSION_PROBABILITIES = (0.45, 0.25, 0.10, 0.20)
MISSION_SERVICE_SECONDS = {
    MissionType.STANDARD: 30.0,
    MissionType.CARGO: 120.0,
    MissionType.EMERGENCY: 10.0,
    MissionType.SURVEILLANCE: 60.0,
}


class FlightPhase(str, Enum):
    EXTERNAL_QUEUE = "external_queue"
    INBOUND = "inbound"
    RING_OUTER = "ring_outer"
    TRANSFER_O2M = "transfer_o2m"
    RING_MIDDLE = "ring_middle"
    TRANSFER_M2I = "transfer_m2i"
    RING_INNER = "ring_inner"
    FINAL = "final"
    LANDING = "landing"
    SERVICE = "service"
    COMPLETED = "completed"
    DIVERTED = "diverted"
    FAILED = "failed"


class TerminalOutcome(str, Enum):
    COMPLETED = "completed"
    DIVERTED_LOW_SOC = "diverted_low_soc"
    DIVERTED_CAPACITY = "diverted_capacity"
    FAILED_COLLISION = "failed_collision"
    FAILED_ENERGY = "failed_energy"
    INVALID = "invalid"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ArrivalRequest:
    arrival_id: int
    request_time: float
    gate_id: int
    mission_type: MissionType
    base_priority: int
    energy_capacity: float
    initial_soc: float


@dataclass
class QueuedArrivalState:
    """Mutable airborne-upstream state for an immutable arrival request."""
    request: ArrivalRequest
    energy_remaining: float
    energy_consumed: float = 0.0
    external_wait_time: float = 0.0
    dynamic_priority: float = 0.0
    diverted_outcome: TerminalOutcome | None = None
    diversion_reason: str | None = None

    @property
    def soc(self) -> float:
        return max(0.0, self.energy_remaining / self.request.energy_capacity)


@dataclass
class UAVState:
    uid: int
    arrival_id: int
    request_time: float
    mission_type: MissionType
    base_priority: int
    assigned_gate: int
    energy_capacity: float
    phase: FlightPhase
    position: np.ndarray
    velocity: np.ndarray
    speed: float
    heading: float
    phase_entry_time: float
    energy_remaining: float
    energy_consumed: float = 0.0
    dynamic_priority: float = 0.0
    ring_role: RingRole | None = None
    ring_progress: float | None = None
    assigned_lz: int | None = None
    merge_reservation_id: str | None = None
    trajectory_plan_id: str | None = None
    admission_time: float | None = None
    outer_entry_time: float | None = None
    final_entry_time: float | None = None
    touchdown_time: float | None = None
    completion_time: float | None = None
    external_wait_time: float = 0.0
    internal_wait_time: float = 0.0
    holding_time_by_ring: dict[RingRole, float] = field(default_factory=dict)
    path_length: float = 0.0
    terminal_outcome: TerminalOutcome | None = None
    terminal_reason: str | None = None
    committed_plan: Any | None = None

    @property
    def soc(self) -> float:
        return max(0.0, self.energy_remaining / self.energy_capacity)

    @classmethod
    def from_request(cls, uid: int, request: ArrivalRequest, queued: QueuedArrivalState, position: np.ndarray, now: float) -> "UAVState":
        return cls(uid=uid, arrival_id=request.arrival_id, request_time=request.request_time,
                   mission_type=request.mission_type, base_priority=request.base_priority,
                   assigned_gate=request.gate_id, energy_capacity=request.energy_capacity,
                   phase=FlightPhase.INBOUND, position=np.asarray(position, dtype=float),
                   velocity=np.zeros(3), speed=0.0, heading=0.0, phase_entry_time=now,
                   energy_remaining=queued.energy_remaining, energy_consumed=queued.energy_consumed,
                   dynamic_priority=queued.dynamic_priority, admission_time=now,
                   external_wait_time=queued.external_wait_time)
