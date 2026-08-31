from __future__ import annotations

import numpy as np

from terminal_marl.core.trajectory_time_parameterization import GeometricPath, TrajectoryConstraints, TrajectoryTimeParameterizer
from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL


def test_forward_backward_profile_respects_speed_acceleration_and_vertical_limits():
    parameterizer = TrajectoryTimeParameterizer(VehicleDynamics(RESEARCH_EVTOL))
    path = GeometricPath.cubic_bezier(np.zeros(3), np.asarray((50., 0., 20.)), np.asarray((100., 50., 50.)), np.asarray((150., 50., 80.)))
    points = parameterizer.parameterize(path, start_time=0.0, initial_velocity=np.zeros(3), target_horizontal_speed=25.0,
                                        constraints=TrajectoryConstraints(end_horizontal_speed=0.0))
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    assert points[-1].t > 0.0
    assert not dynamics.validate_plan(points)
    assert max(np.linalg.norm(point.velocity[:2]) for point in points) <= RESEARCH_EVTOL.max_horizontal_speed + 1e-6
    assert max(point.velocity[2] for point in points) <= RESEARCH_EVTOL.max_climb_rate + 1e-6


def test_curvature_caps_speed_below_straight_line_operational_limit():
    parameterizer = TrajectoryTimeParameterizer(VehicleDynamics(RESEARCH_EVTOL))
    path = GeometricPath.cubic_bezier(np.asarray((20., 0., 0.)), np.asarray((20., 20., 0.)),
                                     np.asarray((-20., 20., 0.)), np.asarray((-20., 0., 0.)))
    points = parameterizer.parameterize(path, start_time=0.0, initial_velocity=np.zeros(3), target_horizontal_speed=25.0)
    assert max(np.linalg.norm(point.velocity[:2]) for point in points) < 25.0


def test_entry_speed_limit_preserves_nonzero_tangent_continuity_for_curved_transfer():
    parameterizer = TrajectoryTimeParameterizer(VehicleDynamics(RESEARCH_EVTOL))
    path = GeometricPath.cubic_bezier(
        np.asarray((400.0, 0.0, 70.0)), np.asarray((400.0, 160.0, 70.0)),
        np.asarray((200.0, -160.0, 40.0)), np.asarray((200.0, 0.0, 40.0)), samples=41,
    )
    limit = parameterizer.entry_speed_limit(path)
    assert 0.0 < limit < RESEARCH_EVTOL.max_horizontal_speed
    points = parameterizer.parameterize(
        path, start_time=0.0, initial_velocity=np.asarray((0.0, limit, 0.0)),
        target_horizontal_speed=max(limit, 16.0), constraints=TrajectoryConstraints(end_horizontal_speed=16.0),
    )
    assert not VehicleDynamics(RESEARCH_EVTOL).validate_plan(points)
    assert np.linalg.norm(points[0].velocity) > 0.0
