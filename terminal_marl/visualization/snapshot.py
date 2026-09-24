"""Read-only extraction of authoritative terminal-world state."""
from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from ..core.config import RingRole
from .schema import (
    GateRenderState,
    LandingZoneRenderState,
    RingRenderState,
    TerminalEventRenderState,
    UAVRenderState,
    WorldRenderSnapshot,
)

if TYPE_CHECKING:
    from ..envs.parallel_env import TerminalParallelEnv


def _point(value) -> tuple[float, float, float]:
    """Copy a numerical position into the immutable rendering boundary."""
    return (float(value[0]), float(value[1]), float(value[2]))


def _recent_conflict_uids(env: "TerminalParallelEnv") -> tuple[int, ...]:
    if not env.conflict_events:
        return ()
    event = env.conflict_events[-1]
    ids = event.get("uids", event.get("pair", ()))
    return tuple(int(uid) for uid in ids)


def _trajectory_points(
    plan,
    maximum: int | None,
    *,
    now: float,
    current_position,
) -> tuple[tuple[float, float, float], ...]:
    """Return only the still-executable portion of the committed plan."""
    if plan is None:
        return ()
    remaining = [_point(current_position)]
    remaining.extend(
        _point(point.position)
        for point in plan.points
        if point.t > now + 1e-9
    )
    if len(remaining) < 2:
        return ()
    if maximum is None or len(remaining) <= maximum:
        return tuple(remaining)
    indices = np.linspace(0, len(remaining) - 1, num=maximum, dtype=int)
    return tuple(remaining[int(index)] for index in np.unique(indices))


def _terminal_event_positions(env: "TerminalParallelEnv") -> dict[int, tuple[float, float, float]]:
    """Index retained ground states for legacy outcome rows without a position."""
    positions: dict[int, tuple[float, float, float]] = {}
    for uav in env.completed_uavs:
        positions[int(uav.uid)] = _point(uav.position)
    for uav in env.service_uavs.values():
        positions[int(uav.uid)] = _point(uav.position)
    return positions


def _terminal_events(env: "TerminalParallelEnv") -> tuple[TerminalEventRenderState, ...]:
    """Copy the cumulative outcome log without fabricating queue positions."""
    retained_positions = _terminal_event_positions(env)
    events: list[TerminalEventRenderState] = []
    for row in env.energy_outcomes:
        raw_uid = row.get("uid")
        uid = None if raw_uid is None else int(raw_uid)
        position_value = row.get("position")
        if position_value is None and uid is not None:
            position_value = retained_positions.get(uid)
        position = None if position_value is None else _point(position_value)
        events.append(
            TerminalEventRenderState(
                uid=uid,
                arrival_id=int(row["arrival_id"]),
                outcome=str(row["outcome"]),
                reason=str(row.get("reason", "")),
                time=float(row["time"]),
                position=position,
                soc=float(row.get("soc", 0.0)),
                energy_remaining=float(row.get("energy_remaining", 0.0)),
                phase=str(row.get("phase", "")),
            )
        )
    return tuple(events)


def build_world_render_snapshot(env: "TerminalParallelEnv", *, trajectory_max_points: int | None = None) -> WorldRenderSnapshot:
    """Copy the current authoritative state without mutating the environment.

    No trajectory, conflict, or physics calculation is performed here.  In
    particular, plan geometry comes directly from ``committed_plan.points``.
    """
    rings = tuple(
        RingRenderState(
            role=role.value,
            radius=float(env.airspace.actual_ring(role).radius),
            height=float(env.airspace.actual_ring(role).height),
            target_radius=float(env.airspace.target_ring(role).radius),
            target_height=float(env.airspace.target_ring(role).height),
            capacity=int(env.airspace.capacity(role)),
            occupancy=int(env.airspace.occupancy(role)),
        )
        for role in RingRole
    )
    gates = tuple(
        GateRenderState(gate.gate_id, _point(gate.spawn_point), _point(gate.gate_point))
        for gate in env.admission.gates
    )
    landing_zones = tuple(
        LandingZoneRenderState(
            lz_id=zone.lz_id,
            position=_point(zone.position),
            status=zone.status.value,
            reserved_by=zone.reserved_by,
            occupied_by=zone.occupied_by,
            service_remaining=float(zone.service_remaining),
        )
        for zone in env.lz_manager.zones
    )
    uavs = tuple(
        UAVRenderState(
            uid=uav.uid,
            arrival_id=uav.arrival_id,
            phase=uav.phase.value,
            ring_role=None if uav.ring_role is None else uav.ring_role.value,
            position=_point(uav.position),
            velocity=_point(uav.velocity),
            speed=float(uav.speed),
            soc=float(uav.soc),
            energy_remaining=float(uav.energy_remaining),
            energy_capacity=float(uav.energy_capacity),
            mission_type=uav.mission_type.value,
            base_priority=int(uav.base_priority),
            assigned_gate=uav.assigned_gate,
            assigned_lz=uav.assigned_lz,
            trajectory_plan_id=uav.trajectory_plan_id,
            merge_reservation_id=uav.merge_reservation_id,
            committed=uav.committed_plan is not None,
            trajectory_points=_trajectory_points(
                uav.committed_plan,
                trajectory_max_points,
                now=float(env.now),
                current_position=uav.position,
            ),
            dynamic_priority=float(uav.dynamic_priority),
            last_release_decision=env.last_release_decisions.get(uav.uid),
            last_safety_decision=env.last_safety_decisions.get(uav.uid),
            emergency_landing=bool(uav.emergency_landing),
            energy_alert=(
                "emergency"
                if uav.emergency_landing
                else (
                    "low_soc"
                    if env.config.emergency_landing_enabled
                    and uav.soc <= env.config.emergency_trigger_soc
                    else "normal"
                )
            ),
        )
        for uav in sorted(env.active.values(), key=lambda item: item.uid)
    )
    numeric_telemetry = {
        str(key): value
        for key, value in env.telemetry.items()
        if isinstance(value, (int, float))
    }
    return WorldRenderSnapshot(
        time=float(env.now),
        step_count=int(env.step_count),
        rings=rings,
        gates=gates,
        landing_zones=landing_zones,
        uavs=uavs,
        external_queue_size=len(env.admission.queue.items),
        service_count=len(env.service_uavs),
        completed_count=len(env.completed_uavs),
        transition_active=env.airspace.transition is not None,
        transition_progress=float(env.airspace.transition_progress(env.now)),
        telemetry=MappingProxyType(numeric_telemetry),
        recent_conflict_uids=_recent_conflict_uids(env),
        terminal_events=_terminal_events(env),
    )
