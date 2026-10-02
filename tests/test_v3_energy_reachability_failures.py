"""Closed-loop V3 failure fixtures and conservative energy-query contracts."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.energy import EnergyModel
from terminal_marl.core.energy_reachability import (
    _advance_ring_holder, _copy_state_at_ring, compute_candidate_energy_budget,
    minimum_immediate_emergency_soc,
)
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.core.landing_zone import LandingZoneManager
from terminal_marl.core.trajectory_generator import TrajectoryGenerator, TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.trajectory_time_parameterization import (
    GeometricPath, TrajectoryConstraints, TrajectoryInfeasibleError,
)
from terminal_marl.core.uav import ArrivalRequest, MissionType, QueuedArrivalState, UAVState
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _components(speed=3.0):
    config = SimConfig(
        trajectory_schema_version="executable_trajectory_v3",
        final_approach_geometry_mode="continuous_inward_v1",
        arrival_model="finite_cohort", total_requested_uavs=0,
        arrival_rate_per_second=0.0, energy_model_mode="phase_motion_v2",
    )
    airspace = DynamicAirspace(config)
    manager, generator, energy = LandingZoneManager(), TrajectoryGenerator(config), EnergyModel(config)
    request = ArrivalRequest(7, 0.0, 0, MissionType.STANDARD, 1, config.energy_capacity, 0.8)
    queued = QueuedArrivalState(request, request.energy_capacity * request.initial_soc)
    uav = UAVState.from_request(7, request, queued, np.zeros(3), 0.0)
    ring = airspace.actual_ring(RingRole.MIDDLE)
    holder = _copy_state_at_ring(
        uav, RingRole.MIDDLE, 1.0,
        position=np.array((ring.radius, 0.0, ring.height)),
        velocity=np.array((0.0, speed, 0.0)),
        acceleration=np.array((-speed ** 2 / ring.radius, 0.0, 0.0)), ring_progress=0.0,
    )
    motion = generator.dynamics.ring_motion_step(ring, 0.0, speed, speed, 1.0)
    candidate = TrajectoryPlan(
        "budget-ring-fixture", holder.uid, "ring_transfer",
        (TrajectoryPoint(0.0, holder.position, holder.velocity, holder.acceleration),
         TrajectoryPoint(1.0, motion.position, motion.velocity, motion.acceleration)),
        0.0, 1.0, metadata={"target_phase": "ring_middle"},
    )
    return config, airspace, manager, generator, energy, holder, candidate


def _budget(parts, **kwargs):
    config, airspace, manager, generator, energy, holder, candidate = parts
    return compute_candidate_energy_budget(
        holder, candidate, RingRole.MIDDLE, 0.0, airspace, manager, generator, energy, config,
        **kwargs,
    )


def test_captured_departure_ceiling_overshoot_has_an_executable_bounded_curve():
    # Exact last failed parameterizer fixture from the first registered arm.
    config, _, _, generator, _, _, _ = _components()
    positions = np.array((
        (174.35504179966796, 360.000443609499, 70.0),
        (84.95421916306424, 373.6777257860062, 73.70370370370371),
        (26.620252544120717, 325.1368311075769, 82.96296296296298),
        (-9.844505369410037, 236.93818538198633, 95.0),
        (-33.63770188977542, 131.6422144170099, 107.03703703703704),
        (-53.95698432922291, 31.809344020422742, 116.29629629629628),
        (-80.0, -40.0, 120.0),
    ))
    velocity = np.array((-0.7259790029275796, 0.3516053984045133, 0.0))
    acceleration = np.array((-0.000709049505601575, -0.0014640135090039537, 0.0))
    points = generator.parameterizer.parameterize(
        GeometricPath(positions), start_time=71.7896144655951,
        initial_velocity=velocity, initial_acceleration=acceleration,
        target_horizontal_speed=3.0770963386623205,
        constraints=TrajectoryConstraints(
            end_horizontal_speed=0.0,
            end_direction=np.array((-254.35504179966796, -400.000443609499, 0.0)),
            end_curvature_vector=np.zeros(3),
        ),
    )
    plan = TrajectoryPlan(
        "captured-ceiling", 7, "emergency_landing", tuple(points), points[0].t, points[-1].t,
        metadata={"trajectory_schema_version": "executable_trajectory_v3"},
    )
    assert generator.validate_kinematics(plan) == []
    np.testing.assert_allclose(points[0].velocity, velocity, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(points[0].acceleration, acceleration, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(points[-1].position, positions[-1], atol=1e-9, rtol=0.0)
    assert max(segment.bounds()["max_altitude"] for segment in plan.polynomial_segments) <= (
        config.vehicle_profile.max_operating_altitude_agl_m + 1e-8
    )


def test_altitude_hull_never_replaces_an_infeasible_live_boundary():
    parameterizer = _components()[3].parameterizer
    positions = np.array(((0., 0., 120.), (1., 0., 119.)))
    velocities = np.array(((0., 0., 1.), (0., 0., 0.)))
    original = velocities.copy()
    assert parameterizer._v3_altitude_bounded_velocities(
        positions, np.array((0., 1.)), velocities, np.zeros((2, 3)),
    ) is None
    np.testing.assert_array_equal(velocities, original)


@pytest.mark.parametrize("duration", (0.21230663739135025, 1.05))
def test_short_pad_wait_prediction_matches_actual_partial_ring_execution(duration):
    config, airspace, _, generator, _, holder, _ = _components(speed=18.0)
    original = deepcopy(holder)
    projected = _advance_ring_holder(
        holder, RingRole.MIDDLE, holder.phase_entry_time + duration, airspace, generator,
    )
    env = TerminalParallelEnv(config)
    env.reset(23)
    executed = deepcopy(holder)
    env._advance_ring_hold(executed, LowLevelIntent(0.0, 0.0, 0.0), duration=duration)
    for name in ("position", "velocity", "acceleration"):
        np.testing.assert_allclose(getattr(projected, name), getattr(executed, name), atol=1e-9, rtol=0.0)
        np.testing.assert_array_equal(getattr(holder, name), getattr(original, name))
    assert 18.0 < projected.speed < 20.0
    assert projected.phase_entry_time == pytest.approx(holder.phase_entry_time + duration)


def test_one_infeasible_fallback_zone_does_not_hide_another_executable_zone(monkeypatch):
    parts = _components()
    generator = parts[3]
    original = generator.emergency_landing
    checked = []

    def route(holder, zone, airspace, now):
        checked.append(zone.lz_id)
        if zone.lz_id == 0:
            raise TrajectoryInfeasibleError("no executable zone-0 witness")
        return original(holder, zone, airspace, now)

    monkeypatch.setattr(generator, "emergency_landing", route)
    result = _budget(parts, stop_at_first_feasible=True)
    assert checked[:2] == [0, 1]
    assert result.selected_lz_id == 1
    assert math.isfinite(result.required_energy)
    assert result.slack_energy >= 0.0
    assert all(zone.reserved_by is None for zone in parts[2].zones)


def test_all_infeasible_fallbacks_return_the_existing_conservative_budget(monkeypatch):
    parts = _components()

    def unavailable(*args):
        raise TrajectoryInfeasibleError("no bounded candidate")

    monkeypatch.setattr(parts[3], "emergency_landing", unavailable)
    result = _budget(parts)
    assert result.selected_lz_id is None
    assert result.required_energy == math.inf
    assert result.slack_energy == -math.inf
    config, airspace, manager, generator, energy, holder, _ = parts
    assert minimum_immediate_emergency_soc(
        holder, RingRole.MIDDLE, 1.0, airspace, manager, generator, energy, config,
    ) == math.inf


@pytest.mark.parametrize("query", ("candidate", "immediate"))
def test_unrelated_runtime_errors_remain_visible(monkeypatch, query):
    parts = _components()

    def broken(*args):
        raise RuntimeError("unexpected implementation error")

    monkeypatch.setattr(parts[3], "emergency_landing", broken)
    with pytest.raises(RuntimeError, match="unexpected implementation error"):
        if query == "candidate":
            _budget(parts)
        else:
            config, airspace, manager, generator, energy, holder, _ = parts
            minimum_immediate_emergency_soc(
                holder, RingRole.MIDDLE, 1.0, airspace, manager, generator, energy, config,
            )
