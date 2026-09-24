from __future__ import annotations

import math

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.flow_trajectories import FlowTrajectoryGenerator
from terminal_marl.core.landing_zone import LandingZoneManager
from terminal_marl.core.trajectory_generator import TrajectoryGenerator, TrajectoryPoint
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    UAVState,
)
from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL


def _scalar_oracle(dynamics: VehicleDynamics, points) -> list[str]:
    rows = list(points)
    if len(rows) < 2:
        return ["invalid_time_parameterization"]
    errors: list[str] = []
    for previous, current in zip(rows, rows[1:]):
        errors.extend(dynamics.validate_transition(previous, current))
    return list(dict.fromkeys(errors))


def _points(*rows) -> tuple[TrajectoryPoint, ...]:
    return tuple(
        TrajectoryPoint(
            time,
            np.asarray(position, dtype=float),
            None if velocity is None else np.asarray(velocity, dtype=float),
            None if acceleration is None else np.asarray(acceleration, dtype=float),
            yaw,
            yaw_rate,
        )
        for time, position, velocity, acceleration, yaw, yaw_rate in rows
    )


@pytest.mark.parametrize(
    "points",
    [
        _points(
            (0.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
            (1.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
            (2.0, (2.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
        ),
        # Declared zero velocity must not hide a large sampled displacement.
        _points(
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
            (1.0, (100.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
        ),
        # Explicit speed limit violation.
        _points(
            (0.0, (0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
            (1.0, (0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
        ),
        # Horizontal acceleration limit violation.
        _points(
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
            (1.0, (0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
        ),
        # Climb-rate limit violation.
        _points(
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), None, None),
            (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), None, None),
        ),
        # Descent-rate limit violation.
        _points(
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, -7.0), (0.0, 0.0, 0.0), None, None),
            (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, -7.0), (0.0, 0.0, 0.0), None, None),
        ),
        # Yaw-rate limit violation.
        _points(
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, math.radians(20.0)),
            (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, math.radians(20.0)),
        ),
        # Invalid time parameterization.
        _points(
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
            (0.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
        ),
    ],
)
def test_validate_plan_matches_scalar_oracle_for_single_constraint_cases(points):
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    expected = _scalar_oracle(dynamics, points)
    assert dynamics.validate_plan(points) == expected


def test_validate_plan_preserves_mixed_error_order():
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    points = _points(
        (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, 0.0),
        (1.0, (30.0, 0.0, 0.0), (30.0, 0.0, 5.0), (0.0, 0.0, 0.0), None, math.radians(20.0)),
    )
    expected = _scalar_oracle(dynamics, points)
    assert expected == [
        "speed_limit",
        "climb_rate_limit",
        "horizontal_acceleration_limit",
        "vertical_acceleration_limit",
        "yaw_rate_limit",
    ]
    assert dynamics.validate_plan(points) == expected


def test_validate_plan_uses_vectorized_pass_path(monkeypatch):
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    points = _points(
        (0.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
        (1.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None),
    )
    calls = 0
    predicate = dynamics.samples_are_feasible

    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        return predicate(*args, **kwargs)

    monkeypatch.setattr(dynamics, "samples_are_feasible", wrapped)
    assert dynamics.validate_plan(points) == []
    assert calls == 1


def _flow_components(role: RingRole):
    config = SimConfig()
    airspace = DynamicAirspace(config)
    flow = FlowTrajectoryGenerator(config, TrajectoryGenerator(config))
    request = ArrivalRequest(1, 0.0, 0, MissionType.STANDARD, 1, config.energy_capacity, 1.0)
    queued = QueuedArrivalState(request, request.energy_capacity)
    uav = UAVState.from_request(1, request, queued, np.zeros(3), 0.0)
    uav.ring_role = role
    uav.phase = {
        RingRole.OUTER: FlightPhase.RING_OUTER,
        RingRole.MIDDLE: FlightPhase.RING_MIDDLE,
        RingRole.INNER: FlightPhase.RING_INNER,
    }[role]
    ring = airspace.actual_ring(role)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    uav.acceleration = np.asarray((-(ring.nominal_speed**2) / ring.radius, 0.0, 0.0), dtype=float)
    uav.ring_progress = 0.0
    return airspace, flow, uav


def test_validate_plan_matches_scalar_oracle_for_two_real_flow_plans():
    outer_airspace, flow, outer_uav = _flow_components(RingRole.OUTER)
    outer_plan = flow.ring_transfer(outer_uav, RingRole.MIDDLE, outer_airspace, 0.0, 50.0)

    inner_airspace, inner_flow, inner_uav = _flow_components(RingRole.INNER)
    final_plan = inner_flow.final_approach(
        inner_uav, LandingZoneManager().zones[0], inner_airspace, 0.0, 50.0,
    )

    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    for plan in (outer_plan, final_plan):
        expected = _scalar_oracle(dynamics, plan.points)
        assert expected == []
        assert dynamics.validate_plan(plan.points) == expected
