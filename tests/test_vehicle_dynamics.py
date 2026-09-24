from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.vehicle_dynamics import DynamicsState, VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL, ScaleConfig, get_vehicle_profile
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.checkpoints import IncompatibleCheckpointError, checkpoint_metadata, validate_checkpoint_metadata
from terminal_marl.core.energy import EnergyModel


def test_research_evtol_profile_is_the_only_main_simulation_profile():
    profile = get_vehicle_profile("research_evtol")
    assert profile is RESEARCH_EVTOL
    assert (profile.vehicle_model_name, profile.profile_version) == ("EHang EH216-S", "eh216s_research_v2")
    assert profile.min_landing_soc == pytest.approx(0.10)
    assert (profile.max_horizontal_speed, profile.never_exceed_speed, profile.design_speed) == (25.0, 29.17, 36.11)
    with pytest.raises(ValueError, match="only research_evtol"):
        get_vehicle_profile("crazyflie")


def test_research_landing_floor_is_profile_config_yaml_consistent_and_keeps_queue_reserve():
    config = SimConfig(energy_model_mode="phase_motion_v2")
    assert config.min_landing_soc == pytest.approx(0.10)
    assert config.soc_reserve == pytest.approx(0.25)
    yaml_text = Path("terminal_marl/configs/vehicle_profiles/research_evtol.yaml").read_text(encoding="utf-8")
    assert "profile_version: eh216s_research_v2" in yaml_text
    assert "min_landing_soc: 0.10" in yaml_text


def test_energy_model_floor_boundaries_and_150_over_1500_reserve_arithmetic():
    config = SimConfig(energy_model_mode="phase_motion_v2")
    energy = EnergyModel(config)
    assert energy.min_landing_soc == pytest.approx(0.10)
    assert energy.min_landing_soc * config.energy_capacity == pytest.approx(150.0)
    state = SimpleNamespace(energy_capacity=1500.0, soc=0.0)
    for soc, expected in ((0.099999, False), (0.10, True), (0.100001, True)):
        state.soc = soc
        assert energy.has_landing_reserve(state) is expected
    request_state = SimpleNamespace(request=SimpleNamespace(energy_capacity=1500.0))
    assert energy.admission_required_energy(request_state) == pytest.approx(
        config.phase_motion_nominal_solo_energy + 150.0
    )


def test_checkpoint_metadata_binds_the_fresh_profile_version():
    config = SimConfig()
    fresh = checkpoint_metadata(config)
    validate_checkpoint_metadata(fresh, config)
    old = dict(fresh)
    old["vehicle_profile_version"] = "eh216s_research_v1"
    with pytest.raises(IncompatibleCheckpointError):
        validate_checkpoint_metadata(old, config)


def test_trajectory_plan_fallback_metadata_uses_fresh_profile_version():
    plan = TrajectoryPlan(
        "profile-version-test",
        0,
        "test",
        (TrajectoryPoint(0.0, np.zeros(3)), TrajectoryPoint(1.0, np.ones(3))),
        0.0,
        1.0,
    )
    assert plan.metadata["vehicle_profile_version"] == "eh216s_research_v2"


def test_config_migrates_terminal_geometry_and_performance_to_profile_views():
    config = SimConfig()
    assert config.vehicle_execution_mode == "performance_constrained"
    assert config.max_speed == 25.0 and config.max_acceleration == 1.5
    assert (config.max_climb_rate, config.max_descent_rate) == (4.0, 6.0)
    assert config.gate_height == 110.0
    assert tuple((config.rings[role].height, config.rings[role].nominal_speed) for role in RingRole) == ((100.0, 22.0), (70.0, 20.0), (40.0, 16.0))
    assert (config.soc_warning, config.soc_reserve, config.soc_emergency) == (.35, .25, .20)


def test_ring_and_state_dynamics_enforce_frozen_acceleration_envelopes():
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    assert dynamics.ring_speed_limit(200.0) == pytest.approx(math.sqrt(1.5 * 200.0))
    next_speed = dynamics.limited_tangential_speed(0.0, 25.0, 200.0, 1.0)
    assert 1.49 < next_speed <= 1.5
    state = DynamicsState(np.zeros(3), np.asarray((26.0, 0.0, 7.0)), np.asarray((1.6, 0.0, 1.1)))
    errors = dynamics.validate_state(state)
    assert {"speed_limit", "climb_rate_limit", "horizontal_acceleration_limit", "vertical_acceleration_limit"} <= set(errors)


def test_ring_motion_step_uses_the_exact_finite_timestep_acceleration_envelope():
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    ring = type("Ring", (), {"radius": 200.0, "height": 40.0})()
    first = dynamics.ring_motion_step(ring, 0.0, 0.0, 16.0, .2)
    second = dynamics.ring_motion_step(ring, first.theta, first.speed, 16.0, .2)
    acceleration = np.linalg.norm((second.velocity - first.velocity) / .2)
    assert first.speed <= .3 + 1e-8
    assert acceleration <= RESEARCH_EVTOL.max_lateral_accel + 1e-6


def test_inner_ring_speed_boundary_allows_target_directed_finite_step_corrections():
    dynamics = VehicleDynamics(RESEARCH_EVTOL)
    radius = 200.0
    dt = .2
    saturated = dynamics.ring_speed_limit(radius)

    braking = dynamics.limited_tangential_speed(saturated, 14.0, radius, dt)
    near_cap = dynamics.limited_tangential_speed(17.0256736324044, 18.0, radius, dt)

    assert 14.0 <= braking < saturated
    assert 17.0256736324044 < near_cap <= saturated
    assert saturated - braking <= RESEARCH_EVTOL.max_horizontal_decel * dt + 1e-9
    assert near_cap - 17.0256736324044 <= RESEARCH_EVTOL.max_horizontal_accel * dt + 1e-9

    for current, next_speed in ((saturated, braking), (17.0256736324044, near_cap)):
        angle = .5 * (current + next_speed) * dt / radius
        discrete_acceleration = math.hypot(
            -math.sin(angle) * next_speed,
            math.cos(angle) * next_speed - current,
        ) / dt
        continuous_acceleration = math.hypot(
            (next_speed - current) / dt,
            next_speed * next_speed / radius,
        )
        assert discrete_acceleration <= RESEARCH_EVTOL.max_lateral_accel + 1e-9
        assert continuous_acceleration <= RESEARCH_EVTOL.max_lateral_accel + 1e-9


def test_scale_config_is_a_future_interface_without_hardware_parameters():
    assert ScaleConfig(.1, .5).length_scale == .1
    with pytest.raises(ValueError):
        ScaleConfig(0., 1.)


def test_safety_filter_rejects_a_plan_whose_displacement_bypasses_declared_velocity():
    config = SimConfig()
    plan = TrajectoryPlan("forged", 1, "test", [
        TrajectoryPoint(0.0, np.zeros(3)),
        TrajectoryPoint(1.0, np.asarray((30.0, 0.0, 0.0))),
    ], 0.0, 1.0)
    snapshot = type("Snapshot", (), {"now": 0.0, "active": {}, "airspace": None})()
    verdict = SafetyFilter(config).filter_plan(plan, {}, snapshot)
    assert not verdict.approved
    assert verdict.reason == "dynamics_speed_limit"
