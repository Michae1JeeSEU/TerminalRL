"""Renderer-neutral immutable data contract for terminal-world snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class RingRenderState:
    role: str
    radius: float
    height: float
    target_radius: float
    target_height: float
    capacity: int
    occupancy: int


@dataclass(frozen=True)
class GateRenderState:
    gate_id: int
    spawn_position: Point3D
    gate_position: Point3D


@dataclass(frozen=True)
class LandingZoneRenderState:
    lz_id: int
    position: Point3D
    status: str
    reserved_by: int | None
    occupied_by: int | None
    service_remaining: float


@dataclass(frozen=True)
class UAVRenderState:
    uid: int
    arrival_id: int
    phase: str
    ring_role: str | None
    position: Point3D
    velocity: Point3D
    speed: float
    soc: float
    energy_remaining: float
    energy_capacity: float
    mission_type: str
    base_priority: int
    assigned_gate: int
    assigned_lz: int | None
    trajectory_plan_id: str | None
    merge_reservation_id: str | None
    committed: bool
    trajectory_points: tuple[Point3D, ...]
    dynamic_priority: float
    last_release_decision: str | None
    last_safety_decision: str | None
    emergency_landing: bool = False
    energy_alert: str = "normal"
    rescue_status: str | None = None


@dataclass(frozen=True)
class TerminalEventRenderState:
    """Immutable record of an arrival which left the airborne lifecycle."""

    uid: int | None
    arrival_id: int
    outcome: str
    reason: str
    time: float
    position: Point3D | None
    soc: float
    energy_remaining: float
    phase: str


@dataclass(frozen=True)
class WorldRenderSnapshot:
    time: float
    step_count: int
    rings: tuple[RingRenderState, ...]
    gates: tuple[GateRenderState, ...]
    landing_zones: tuple[LandingZoneRenderState, ...]
    uavs: tuple[UAVRenderState, ...]
    external_queue_size: int
    service_count: int
    completed_count: int
    transition_active: bool
    transition_progress: float
    telemetry: Mapping[str, float | int]
    recent_conflict_uids: tuple[int, ...]
    terminal_events: tuple[TerminalEventRenderState, ...] = ()
