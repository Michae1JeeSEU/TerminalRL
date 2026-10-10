"""Candidate energy and landing-zone slot reachability calculations.

The calculations in this module are deliberately standalone.  They inspect a
candidate trajectory and the currently reported landing-zone lifecycle state;
they do not reserve a zone, mutate a UAV, or run the safety filter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Literal

import numpy as np

from .config import RingRole, SimConfig
from .energy import EnergyModel
from .landing_zone import LZStatus, LandingZone, LandingZoneManager
from .trajectory_generator import TrajectoryGenerator, TrajectoryPlan
from .uav import FlightPhase, UAVState


EnergyTarget = RingRole | Literal["final"]


@dataclass(frozen=True, slots=True)
class EnergyReachabilityResult:
    """Energy budget for a candidate and its selected landing-zone fallback.

    ``required_energy`` and ``slack_energy`` use the UAV's configured energy
    units.  A result with ``selected_lz_id is None`` uses positive/negative
    infinity for the unavailable budget fields.
    """

    candidate_energy: float
    fallback_route_energy: float
    pad_wait_seconds: float
    required_energy: float
    slack_energy: float
    selected_lz_id: int | None


def _margin_energy(config: SimConfig) -> float:
    return float(config.emergency_energy_margin_seconds * config.power_reference)


def _target_phase(role: RingRole) -> FlightPhase:
    return {
        RingRole.OUTER: FlightPhase.RING_OUTER,
        RingRole.MIDDLE: FlightPhase.RING_MIDDLE,
        RingRole.INNER: FlightPhase.RING_INNER,
    }[role]


def _copy_state_at_ring(
    uav: UAVState,
    role: RingRole,
    phase_time: float,
    *,
    position: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    acceleration: np.ndarray | None = None,
    ring_progress: float | None = None,
) -> UAVState:
    """Return an independent ring-holder state for a generated route."""

    next_position = np.array(
        uav.position if position is None else position,
        dtype=float,
        copy=True,
    )
    next_velocity = np.array(
        uav.velocity if velocity is None else velocity,
        dtype=float,
        copy=True,
    )
    next_acceleration = np.array(
        uav.acceleration if acceleration is None else acceleration,
        dtype=float,
        copy=True,
    )
    if ring_progress is None:
        ring_progress = math.atan2(float(next_position[1]), float(next_position[0]))

    return replace(
        uav,
        phase=_target_phase(role),
        phase_entry_time=float(phase_time),
        ring_role=role,
        ring_progress=float(ring_progress),
        position=next_position,
        velocity=next_velocity,
        acceleration=next_acceleration,
        speed=float(np.linalg.norm(next_velocity[:2])),
        assigned_lz=None,
        merge_reservation_id=None,
        trajectory_plan_id=None,
        committed_plan=None,
        emergency_landing=False,
    )


def _future_ring_holder(
    uav: UAVState,
    candidate_plan: TrajectoryPlan,
    role: RingRole,
) -> UAVState:
    """Build the ring-holder state at the end of a transfer candidate."""

    end = candidate_plan.points[-1]
    return _copy_state_at_ring(
        uav,
        role,
        candidate_plan.end_time,
        position=end.position,
        velocity=end.velocity,
        acceleration=end.acceleration,
    )


def _advance_ring_holder(
    holder: UAVState,
    role: RingRole,
    until: float,
    airspace: Any,
    generator: TrajectoryGenerator,
) -> UAVState:
    """Advance a copied holder along its target ring to ``until``."""

    start = float(holder.phase_entry_time)
    duration = max(0.0, float(until) - start)
    if duration <= 0.0:
        return holder

    ring = airspace.actual_ring(role)
    target_speed = float(ring.nominal_speed + getattr(airspace, "speed_profile_bias", 0.0))
    target_speed = float(
        np.clip(target_speed, generator.config.speed_min, generator.config.speed_max)
    )
    points = generator.ring_hold_points(
        ring,
        float(holder.ring_progress),
        float(holder.speed),
        target_speed,
        start,
        duration,
    )
    end = points[-1]
    return _copy_state_at_ring(
        holder,
        role,
        until,
        position=end.position,
        velocity=end.velocity,
        acceleration=end.acceleration,
        ring_progress=math.atan2(float(end.position[1]), float(end.position[0])),
    )


def _infeasible_result(candidate_energy: float) -> EnergyReachabilityResult:
    return EnergyReachabilityResult(
        candidate_energy=float(candidate_energy),
        fallback_route_energy=math.inf,
        pad_wait_seconds=math.inf,
        required_energy=math.inf,
        slack_energy=-math.inf,
        selected_lz_id=None,
    )


def _matching_final_lz(
    candidate_plan: TrajectoryPlan,
    uav: UAVState,
    lz_manager: LandingZoneManager,
) -> LandingZone | None:
    """Find the LZ represented by a final candidate plan."""

    metadata_lz_id = candidate_plan.metadata.get("lz_id")
    if metadata_lz_id is not None:
        try:
            candidate_id = int(metadata_lz_id)
        except (TypeError, ValueError):
            candidate_id = None
        if candidate_id is not None:
            for zone in lz_manager.zones:
                if zone.lz_id == candidate_id:
                    return zone

    endpoint = np.asarray(candidate_plan.points[-1].position, dtype=float)
    for zone in lz_manager.zones:
        if np.allclose(endpoint, zone.position, atol=1e-6, rtol=0.0):
            return zone

    # Older callers may supply a plan without an endpoint that is exactly the
    # pad position.  In that compatibility case, use the UAV's assigned pad.
    if uav.assigned_lz is not None:
        for zone in lz_manager.zones:
            if zone.lz_id == uav.assigned_lz:
                return zone
    return None


def _usable_final_zone(zone: LandingZone, uav: UAVState) -> bool:
    if uav.mission_type not in zone.compatible_missions:
        return False
    if zone.status is LZStatus.FREE:
        return True
    return zone.status is LZStatus.RESERVED and zone.reserved_by == uav.uid


def compute_candidate_energy_budget(
    uav: UAVState,
    candidate_plan: TrajectoryPlan,
    target: EnergyTarget,
    now: float,
    airspace: Any,
    lz_manager: LandingZoneManager,
    trajectory_generator: TrajectoryGenerator,
    energy_model: EnergyModel,
    config: SimConfig,
    *,
    stop_at_first_feasible: bool = False,
) -> EnergyReachabilityResult:
    """Compute an executable energy lower bound for a candidate.

    For transfer targets, the fallback route is evaluated under the landing
    zone's reported release ETA and the ring-holder is advanced along the
    target ring during any pad wait.  The result is an executable energy lower
    bound under that reported pad release ETA; it is not a SafetyFilter
    certification or a guarantee of future feasibility.

    The function only scans the manager's landing zones (O(number of LZs)) and
    does not reserve, release, or otherwise mutate any object.

    ``stop_at_first_feasible`` is an opt-in guard-path shortcut.  It returns
    the first complete finite-ETA witness with non-negative slack; when no
    witness is feasible, the full scan still returns the minimum required
    budget.
    """

    candidate_energy = float(
        energy_model.estimate_plan_energy(
            candidate_plan,
            start=float(now),
            end=float(candidate_plan.end_time),
        )
    )
    reserve_energy = float(energy_model.min_landing_soc * uav.energy_capacity)
    margin_energy = _margin_energy(config)

    if target == "final":
        zone = _matching_final_lz(candidate_plan, uav, lz_manager)
        if zone is None or not _usable_final_zone(zone, uav):
            return _infeasible_result(candidate_energy)
        required = candidate_energy + reserve_energy + margin_energy
        return EnergyReachabilityResult(
            candidate_energy=candidate_energy,
            fallback_route_energy=0.0,
            pad_wait_seconds=0.0,
            required_energy=float(required),
            slack_energy=float(uav.energy_remaining - required),
            selected_lz_id=zone.lz_id,
        )

    if not isinstance(target, RingRole):
        raise TypeError("target must be a RingRole or 'final'")

    holder_at_end = _future_ring_holder(uav, candidate_plan, target)
    best: tuple[float, int, float, float] | None = None
    for zone in lz_manager.zones:
        if uav.mission_type not in zone.compatible_missions:
            continue

        release_at = max(
            float(candidate_plan.end_time),
            float(lz_manager.earliest_release_time(zone.lz_id, now=float(now))),
        )
        if not math.isfinite(release_at):
            continue

        wait_seconds = max(0.0, release_at - float(candidate_plan.end_time))
        holder_at_release = _advance_ring_holder(
            holder_at_end,
            target,
            release_at,
            airspace,
            trajectory_generator,
        )
        fallback_plan = trajectory_generator.emergency_landing(
            holder_at_release,
            zone,
            airspace,
            release_at,
        )
        fallback_energy = float(
            energy_model.estimate_plan_energy(
                fallback_plan,
                start=release_at,
                end=fallback_plan.end_time,
            )
        )
        wait_energy = float(
            energy_model.phase_power(holder_at_release.phase) * wait_seconds
        )
        required = (
            candidate_energy
            + wait_energy
            + fallback_energy
            + reserve_energy
            + margin_energy
        )
        item = (float(required), int(zone.lz_id), float(fallback_energy), float(wait_seconds))
        if best is None or item[:2] < best[:2]:
            best = item
        if stop_at_first_feasible and uav.energy_remaining - required >= -1e-12:
            return EnergyReachabilityResult(
                candidate_energy=candidate_energy,
                fallback_route_energy=fallback_energy,
                pad_wait_seconds=wait_seconds,
                required_energy=required,
                slack_energy=float(uav.energy_remaining - required),
                selected_lz_id=zone.lz_id,
            )

    if best is None:
        return _infeasible_result(candidate_energy)

    required, selected_lz_id, fallback_energy, wait_seconds = best
    return EnergyReachabilityResult(
        candidate_energy=candidate_energy,
        fallback_route_energy=fallback_energy,
        pad_wait_seconds=wait_seconds,
        required_energy=required,
        slack_energy=float(uav.energy_remaining - required),
        selected_lz_id=selected_lz_id,
    )


def minimum_immediate_emergency_soc(
    uav: UAVState,
    ring_role: RingRole,
    now: float,
    airspace: Any,
    lz_manager: LandingZoneManager,
    trajectory_generator: TrajectoryGenerator,
    energy_model: EnergyModel,
    config: SimConfig,
) -> float:
    """Return the minimum SOC required by an immediate solo emergency route.

    This helper evaluates each mission-compatible LZ from a copied ring-holder
    state at ``now``.  It is an isolated route-energy calculation and does not
    account for congestion, pad wait, or SafetyFilter certification.
    """

    holder = _copy_state_at_ring(uav, ring_role, now)
    required_socs: list[float] = []
    for zone in lz_manager.zones:
        if uav.mission_type not in zone.compatible_missions:
            continue
        route = trajectory_generator.emergency_landing(holder, zone, airspace, now)
        route_energy = float(
            energy_model.estimate_plan_energy(
                route,
                start=float(now),
                end=route.end_time,
            )
        )
        required_energy = (
            route_energy
            + energy_model.min_landing_soc * uav.energy_capacity
            + _margin_energy(config)
        )
        required_socs.append(float(required_energy / uav.energy_capacity))

    return min(required_socs, default=math.inf)


__all__ = [
    "EnergyReachabilityResult",
    "compute_candidate_energy_budget",
    "minimum_immediate_emergency_soc",
]
