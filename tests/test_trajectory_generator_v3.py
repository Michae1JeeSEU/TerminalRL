from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.trajectory_generator import TrajectoryGenerator, TrajectoryPlan


def _config(**overrides) -> SimConfig:
    values = {
        "trajectory_schema_version": "executable_trajectory_v3",
        "final_approach_geometry_mode": "continuous_inward_v1",
    }
    values.update(overrides)
    if values.get("final_approach_execution_mode") == "speed_timing_v1":
        values["action_schema_version"] = "low_action_schema_v4_2_hierarchical"
    return SimConfig(**values)


def _uav(airspace, role: RingRole, uid: int = 1, speed: float | None = None):
    ring = airspace.actual_ring(role)
    live_speed = ring.nominal_speed if speed is None else float(speed)
    return SimpleNamespace(
        uid=uid,
        position=np.asarray((ring.radius, 0.0, ring.height), dtype=float),
        velocity=np.asarray((0.0, live_speed, 0.0), dtype=float),
        acceleration=np.asarray((-live_speed * live_speed / ring.radius, 0.0, 0.0), dtype=float),
        speed=live_speed,
        ring_progress=0.0,
        ring_role=role,
        committed_plan=None,
    )


def _lz():
    return SimpleNamespace(lz_id=0, position=np.asarray((0.0, -40.0, 0.0), dtype=float))


def _reservation(slot: float = 30.0, angle: float = 0.0):
    return SimpleNamespace(reservation_id=f"r-{slot}-{angle}", merge_time=float(slot), merge_angle=float(angle))


def test_v3_config_is_explicitly_opt_in_and_static_geometry_only():
    assert SimConfig().trajectory_schema_version == "executable_trajectory_v2"
    with pytest.raises(ValueError, match="performance_constrained"):
        SimConfig(trajectory_schema_version="executable_trajectory_v3", vehicle_execution_mode="legacy_kinematic")
    with pytest.raises(ValueError, match="dynamic geometry execution is not qualified"):
        SimConfig(trajectory_schema_version="executable_trajectory_v3", high_level_enabled=True)


def test_v3_ring_prefix_preserves_live_state_and_endpoint_helper():
    config = _config()
    airspace = DynamicAirspace(config)
    generator = TrajectoryGenerator(config)
    ring = airspace.actual_ring(RingRole.INNER)
    initial_acceleration = np.asarray((-16.0 * 16.0 / ring.radius, 0.0, 0.0))
    required = generator._v3_required_ring_braking_duration(
        ring, 0.0, 16.0, 10.0, initial_acceleration=initial_acceleration,
    )
    points = generator.ring_hold_points(
        ring, 0.0, 16.0, 10.0, 5.0, required + 1.0,
        initial_acceleration=initial_acceleration,
    )
    assert points[0].t == pytest.approx(5.0)
    np.testing.assert_allclose(points[0].position, np.asarray((ring.radius, 0.0, ring.height)))
    np.testing.assert_allclose(points[0].velocity, np.asarray((0.0, 16.0, 0.0)))
    np.testing.assert_allclose(points[0].acceleration, initial_acceleration)
    assert np.linalg.norm(points[-1].velocity[:2]) == pytest.approx(10.0, abs=1e-5)
    endpoint_position, endpoint_velocity = generator._ring_hold_endpoint(
        ring, 0.0, 16.0, 10.0, required + 1.0,
        initial_acceleration=initial_acceleration,
    )
    np.testing.assert_allclose(endpoint_position, points[-1].position, atol=1e-10)
    np.testing.assert_allclose(endpoint_velocity, points[-1].velocity, atol=1e-10)
    plan = TrajectoryPlan(
        "v3-hold", 1, "ring_hold", tuple(points), points[0].t, points[-1].t,
        metadata={"trajectory_schema_version": "executable_trajectory_v3"},
    )
    assert generator.validate_kinematics(plan) == []


def test_v3_ring_prefix_eases_nonzero_tangential_acceleration_continuously():
    config = _config()
    airspace = DynamicAirspace(config)
    generator = TrajectoryGenerator(config)
    ring = airspace.actual_ring(RingRole.INNER)
    alpha = 0.2
    initial_acceleration = np.asarray((-16.0 * 16.0 / ring.radius, alpha, 0.0))
    required = generator._v3_required_ring_braking_duration(
        ring, 0.0, 16.0, 10.0, initial_acceleration=initial_acceleration,
    )
    points = generator.ring_hold_points(
        ring, 0.0, 16.0, 10.0, 0.0, required + 1.0,
        initial_acceleration=initial_acceleration,
    )
    plan = TrajectoryPlan(
        "v3-alpha", 1, "ring_hold", tuple(points), points[0].t, points[-1].t,
        metadata={"trajectory_schema_version": "executable_trajectory_v3"},
    )
    before = plan.sample(0.4 - 1e-5)
    at_ease = plan.sample(0.4)
    after = plan.sample(0.4 + 1e-5)
    np.testing.assert_allclose((after.position - before.position) / (2e-5), at_ease.velocity, atol=2e-4)
    assert np.linalg.norm(after.velocity - before.velocity) < 1e-3
    assert abs(float(np.dot(at_ease.acceleration, np.asarray((-1.0, 0.0, 0.0))))) < 2.0


def test_v3_continuous_final_proposals_bind_speed_timing_and_validate():
    config = _config(final_approach_execution_mode="speed_timing_v1")
    airspace = DynamicAirspace(config)
    generator = TrajectoryGenerator(config)
    uav = _uav(airspace, RingRole.INNER, speed=16.0)
    plans = []
    for slot, residual in ((0.0, -2.0), (30.0, 0.0), (60.0, 2.0)):
        plan = generator.final_approach(
            uav, _lz(), _reservation(slot), 0.0,
            scheduled_delay=0.0, speed_residual_mps=residual, airspace=airspace,
        )
        plans.append(plan)
        assert plan.metadata["trajectory_schema_version"] == "executable_trajectory_v3"
        assert generator.validate_kinematics(plan) == []
        assert plan.metadata["accepted_reservation_departure"] >= slot - 1e-8
        np.testing.assert_allclose(plan.points[0].position, uav.position)
        np.testing.assert_allclose(plan.points[0].velocity, uav.velocity)
        np.testing.assert_allclose(plan.points[0].acceleration, uav.acceleration)
    assert len({plan.trajectory_hash for plan in plans}) == 3
    assert plans[1].metadata["accepted_reservation_departure"] == pytest.approx(30.0, abs=1e-8)
    assert plans[2].metadata["accepted_reservation_departure"] == pytest.approx(60.0, abs=1e-8)


def test_v3_legacy_continuous_mapping_keeps_residual_out_of_geometry():
    config = _config(final_approach_execution_mode="legacy")
    airspace = DynamicAirspace(config)
    generator = TrajectoryGenerator(config)
    uav = _uav(airspace, RingRole.INNER, speed=16.0)
    left = generator.final_approach(
        uav, _lz(), _reservation(30.0), 0.0,
        speed_residual_mps=-2.0, airspace=airspace,
    )
    right = generator.final_approach(
        uav, _lz(), _reservation(90.0), 0.0,
        speed_residual_mps=2.0, airspace=airspace,
    )
    assert left.trajectory_hash == right.trajectory_hash
    assert left.end_time == pytest.approx(right.end_time, abs=1e-9)
    assert left.metadata["accepted_target_speed_mps"] == pytest.approx(10.0)
    assert right.metadata["accepted_target_speed_mps"] == pytest.approx(10.0)


def test_v3_inbound_transfer_and_emergency_use_executable_schema():
    config = _config()
    airspace = DynamicAirspace(config)
    generator = TrajectoryGenerator(config)
    outer = airspace.actual_ring(RingRole.OUTER)
    inbound_uav = SimpleNamespace(
        uid=2,
        position=np.asarray((800.0, 0.0, 110.0)),
        velocity=np.asarray((-5.0, 0.0, 0.0)),
        acceleration=np.zeros(3),
        speed=5.0,
        ring_progress=None,
        ring_role=None,
        committed_plan=None,
    )
    inbound = generator.inbound(inbound_uav, SimpleNamespace(angle=0.0), airspace, 0.0)
    assert inbound.metadata["trajectory_schema_version"] == "executable_trajectory_v3"
    np.testing.assert_allclose(inbound.points[-1].position, np.asarray((outer.radius, 0.0, outer.height)), atol=1e-8)
    assert generator.validate_kinematics(inbound) == []

    holder = _uav(airspace, RingRole.OUTER, uid=3)
    transfer = generator.ring_transfer(
        holder, RingRole.OUTER, RingRole.MIDDLE, _reservation(120.0, 1.0), airspace, 0.0,
    )
    assert transfer.metadata["trajectory_schema_version"] == "executable_trajectory_v3"
    assert generator.validate_kinematics(transfer) == []
    np.testing.assert_allclose(
        transfer.points[-1].position,
        generator._ring_position(airspace.actual_ring(RingRole.MIDDLE), 1.0),
        atol=1e-7,
    )

    emergency = generator.emergency_landing(holder, _lz(), airspace, 0.0)
    assert emergency.metadata["trajectory_schema_version"] == "executable_trajectory_v3"
    assert generator.validate_kinematics(emergency) == []
    np.testing.assert_allclose(emergency.points[-1].position, _lz().position, atol=1e-7)


def test_v3_brake_bezier_mode_preserves_live_boundary():
    config = _config(final_approach_geometry_mode="brake_bezier_v1")
    airspace = DynamicAirspace(config)
    generator = TrajectoryGenerator(config)
    uav = _uav(airspace, RingRole.INNER, speed=10.0)
    plan = generator.final_approach(uav, _lz(), _reservation(0.0), 0.0, airspace=airspace)
    assert plan.metadata["trajectory_schema_version"] == "executable_trajectory_v3"
    np.testing.assert_allclose(plan.points[0].position, uav.position)
    np.testing.assert_allclose(plan.points[0].velocity, uav.velocity)
    np.testing.assert_allclose(plan.points[0].acceleration, uav.acceleration)
    assert generator.validate_kinematics(plan) == []
