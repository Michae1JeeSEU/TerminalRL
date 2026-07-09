"""Integration tests for fixed/dynamic airspace and dry-run trajectories."""

import numpy as np
import pytest

from terminal_marl import ExperimentMode, SimConfig, TerminalParallelEnv
from terminal_marl.core.uav import FlightPhase


def test_fixed_mode_rejects_airspace_change() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.FIXED_RULE)
    old_state = env.airspace.state_vector()

    info = env.apply_high_level_action(np.ones(8, dtype=np.float32))

    assert np.array_equal(env.airspace.state_vector(), old_state)
    assert info["clipped"]
    assert "fixed_airspace_override" in info["violations"]


def test_dynamic_mode_updates_airspace_and_live_ring_specs() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.DYNAMIC_HRL)
    old_state = env.airspace.state_vector()

    info = env.apply_high_level_action(
        np.array([1.0, -1.0, 1.0, -1.0, 0.5, -0.5, 1.0, -1.0])
    )

    assert not np.array_equal(info["new_state"], old_state)
    rings = {ring.index: ring for ring in env.config.rings}
    assert rings[2].radius == pytest.approx(env.airspace.outer_ring.radius)
    assert rings[1].height == pytest.approx(env.airspace.inner_ring.height)
    assert rings[2].speed == pytest.approx(env.airspace.outer_ring.speed)
    assert env.config.ring_to_ring_min_time_gap_s == pytest.approx(env.airspace.ring_headway)
    assert env.config.final_min_time_gap_s == pytest.approx(env.airspace.final_headway)


def test_new_modes_append_normalized_airspace_observation() -> None:
    legacy = TerminalParallelEnv(SimConfig())
    legacy_obs, _ = legacy.reset(seed=3)
    dynamic = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.DYNAMIC_HRL)
    dynamic_obs, _ = dynamic.reset(seed=3)

    legacy_vector = legacy_obs[legacy.possible_agents[0]]
    dynamic_vector = dynamic_obs[dynamic.possible_agents[0]]
    assert dynamic_vector.shape[0] == legacy_vector.shape[0] + 8
    assert np.allclose(dynamic_vector[-8:], dynamic.airspace.normalized_state_vector())
    assert np.all((-1.0 <= dynamic_vector[-8:]) & (dynamic_vector[-8:] <= 1.0))


def test_reset_restores_initial_dynamic_airspace() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.DYNAMIC_HRL)
    initial = env.airspace.state_vector()
    env.apply_high_level_action(np.ones(8))
    assert not np.array_equal(env.airspace.state_vector(), initial)

    env.reset(seed=4)
    assert np.array_equal(env.airspace.state_vector(), initial)


def test_environment_trajectory_generator_uses_current_ring() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.DYNAMIC_RULE)
    env.reset(seed=5)
    aircraft = env.aircraft[0]
    outer = env.airspace.outer_ring
    aircraft.phase = FlightPhase.RING
    aircraft.assigned_ring = outer.index
    aircraft.position = np.array([outer.radius, 0.0, outer.height])
    aircraft.speed = outer.speed

    plan = env.trajectory_plan_for(aircraft, duration=5.0, dt=1.0)
    radii = [np.hypot(point.x, point.y) for point in plan.points]

    assert plan.mode == "ring_hold"
    assert np.allclose(radii, outer.radius)
    assert np.allclose([point.z for point in plan.points], outer.height)
    assert plan.metadata["safety_filter"]["approved"]


def test_step_accepts_optional_high_level_action() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.DYNAMIC_RULE)
    env.reset(seed=6)
    old_radius = env.airspace.outer_ring.radius

    actions = {agent: np.zeros(4, dtype=np.float32) for agent in env.agents}
    env.step(actions, high_level_action=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert env.airspace.outer_ring.radius > old_radius
