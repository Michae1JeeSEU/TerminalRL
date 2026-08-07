"""Regression tests for event-gated structural airspace reconfiguration."""

from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.config import load_experiment_config
from terminal_marl.control.geometry_event_gate import (
    GateConfig,
    GateMetrics,
    GateMode,
    GateState,
    GeometryEventGate,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def metrics(score_kind: str) -> GateMetrics:
    if score_kind == "high":
        return GateMetrics(
            normalized_congestion=1.0,
            approach_backlog_ratio=1.0,
            occupancy_pressure=1.0,
            backlog_growth=0.5,
            normalized_conflict_rate=0.1,
            recent_arrivals=2.0,
            recent_landings=0.0,
            arrival_service_imbalance=1.0,
            active_capacity_ratio=0.9,
        )
    return GateMetrics(
        normalized_congestion=0.0,
        approach_backlog_ratio=0.0,
        occupancy_pressure=0.0,
        backlog_growth=0.0,
        normalized_conflict_rate=0.0,
        recent_arrivals=0.0,
        recent_landings=1.0,
        arrival_service_imbalance=0.0,
        active_capacity_ratio=0.1,
    )


def gate() -> GeometryEventGate:
    return GeometryEventGate(
        config=GateConfig(
            metrics_sampling_interval_seconds=20.0,
            geometry_min_dwell_seconds=600.0,
            theta_on=0.55,
            theta_off=0.35,
            k_on=3,
            k_off=2,
            recovery_k=2,
        ),
        mode=GateMode.EVENT,
    )


def test_transient_one_sample_spike_does_not_trigger() -> None:
    g = gate()

    g.update(metrics("high"), time_seconds=0.0, force_sample=True)
    info = g.update(metrics("low"), time_seconds=20.0, force_sample=True)

    assert not info["geometry_gate_active"]
    assert g.state is GateState.NORMAL


def test_k_on_sustained_abnormal_samples_trigger_with_hysteresis() -> None:
    g = gate()

    for i in range(3):
        info = g.update(metrics("high"), time_seconds=20.0 * i, force_sample=True)

    assert info["geometry_gate_active"]
    assert g.state is GateState.TRIGGERED
    assert g.config.theta_off < g.config.theta_on


def test_k_off_closes_gate_after_sustained_recovery() -> None:
    g = gate()
    for i in range(3):
        g.update(metrics("high"), time_seconds=20.0 * i, force_sample=True)

    for i in range(2):
        info = g.update(metrics("low"), time_seconds=100.0 + 20.0 * i, force_sample=True)

    assert not info["geometry_gate_active"]


def test_geometry_dwell_blocks_open_gate_until_600s() -> None:
    g = gate()
    g.last_geometry_change_time = 100.0
    for i in range(3):
        g.update(metrics("high"), time_seconds=120.0 + 20.0 * i, force_sample=True)

    allowed, reason = g.geometry_allowed(time_seconds=200.0)

    assert not allowed
    assert reason == "dwell_locked"


def test_event_gate_closed_blocks_geometry_even_when_dwell_satisfied() -> None:
    g = gate()

    applied, info = g.transform_action(np.ones(8), time_seconds=1000.0)

    assert not info["geometry_action_allowed"]
    assert np.allclose(applied[:4], 0.0)
    assert np.allclose(applied[4:], 1.0)


def test_periodic_gate_allows_geometry_when_dwell_satisfied() -> None:
    g = GeometryEventGate(config=gate().config, mode=GateMode.PERIODIC)

    applied, info = g.transform_action(np.ones(8), time_seconds=1000.0)

    assert info["geometry_action_allowed"]
    assert np.allclose(applied, 1.0)


def test_periodic_gate_blocks_geometry_when_dwell_locked_but_keeps_operational() -> None:
    g = GeometryEventGate(config=gate().config, mode=GateMode.PERIODIC)
    g.last_geometry_change_time = 900.0

    applied, info = g.transform_action(np.ones(8), time_seconds=1000.0)

    assert not info["geometry_action_allowed"]
    assert np.allclose(applied[:4], 0.0)
    assert np.allclose(applied[4:], 1.0)


def test_nominal_recovery_can_become_eligible_after_stable_normal() -> None:
    g = gate()

    for i in range(2):
        info = g.update(
            metrics("low"),
            time_seconds=20.0 * i,
            geometry_deviation_l1=2.0,
            force_sample=True,
        )

    assert info["gate_state"] == GateState.RECOVERY_ELIGIBLE.value
    allowed, reason = g.geometry_allowed(time_seconds=1000.0, geometry_deviation_l1=2.0)
    assert allowed
    assert reason == "nominal_recovery_eligible"


def test_event_gated_env_masks_only_geometry_dimensions() -> None:
    env = TerminalParallelEnv(load_experiment_config("event_gated_hrl"))
    env.reset(seed=0)
    old = env.airspace.state_vector().copy()

    info = env.apply_high_level_action(np.ones(8, dtype=np.float32))
    new = env.airspace.state_vector()

    assert not info["geometry_gate"]["geometry_action_allowed"]
    assert np.allclose(new[:4], old[:4])
    assert np.any(np.abs(new[4:] - old[4:]) > 0.0)
    assert np.allclose(info["requested_action"][:4], 1.0)
    assert np.allclose(info["applied_action"][:4], 0.0)


def test_periodic_env_allows_geometry_initially() -> None:
    env = TerminalParallelEnv(load_experiment_config("periodic_hrl"))
    env.reset(seed=0)
    old = env.airspace.state_vector().copy()

    info = env.apply_high_level_action(np.ones(8, dtype=np.float32))
    new = env.airspace.state_vector()

    assert info["geometry_gate"]["geometry_action_allowed"]
    assert np.any(np.abs(new[:4] - old[:4]) > 0.0)


def test_gate_state_enters_high_level_observation_and_resets() -> None:
    env = TerminalParallelEnv(load_experiment_config("event_gated_hrl"))
    env.reset(seed=0)
    obs = env.get_high_level_observation()
    env.geometry_gate.trigger_active = True
    env.geometry_gate.trigger_persistence_count = 99

    env.reset(seed=1)
    reset_obs = env.get_high_level_observation()

    assert obs.shape == reset_obs.shape
    assert env.geometry_gate.trigger_persistence_count == 0
    assert np.isfinite(reset_obs).all()


def test_geometry_change_does_not_reset_vtac_or_phase_timers() -> None:
    env = TerminalParallelEnv(load_experiment_config("periodic_hrl"))
    env.reset(seed=0)
    aircraft = env.aircraft[0]
    aircraft.assigned_ring_step = 12
    aircraft.ring_established_step = 34

    env.apply_high_level_action(np.ones(8, dtype=np.float32))

    assert aircraft.assigned_ring_step == 12
    assert aircraft.ring_established_step == 34
