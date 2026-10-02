"""Tests for the opt-in warm-start solo energy precondition."""

from __future__ import annotations

import pytest

from terminal_marl.core.config import SimConfig
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _warm_config(**overrides) -> SimConfig:
    values = {
        "reset_mode": "warm_start_rings",
        "warm_start_uavs": 3,
        "warm_start_steady_motion": True,
        "fixed_arrival_schedule": (0.0, 0.0, 0.0),
        "total_requested_uavs": 3,
        "initial_soc_mode": "fixed",
        "initial_soc": 0.41,
        "energy_model_mode": "phase_motion_v2",
        "emergency_landing_enabled": True,
        "warm_start_energy_reachability_check_enabled": True,
    }
    values.update(overrides)
    return SimConfig(**values)


def test_low_soc_outer_warm_start_is_rejected_with_required_soc() -> None:
    # The unchanged V2 solo-route minimum is about 0.20608 in this fixture;
    # 0.35 is feasible and cannot exercise the rejection precondition.
    config = _warm_config(warm_start_uavs=1, fixed_arrival_schedule=(0.0,), total_requested_uavs=1, initial_soc=0.15)

    with pytest.raises(ValueError, match=r"uid=0, ring=outer.*initial_soc=0\.150000000.*required_soc="):
        TerminalParallelEnv(config)


def test_soc_point_four_one_passes_small_all_ring_warm_start() -> None:
    env = TerminalParallelEnv(_warm_config())

    assert len(env.active) == 3
    assert [uav.ring_role.value for uav in env.active.values()] == ["outer", "middle", "inner"]
    assert all(uav.soc == pytest.approx(0.41) for uav in env.active.values())


def test_default_flag_false_preserves_existing_warm_start_reset() -> None:
    config = _warm_config(
        warm_start_energy_reachability_check_enabled=False,
        initial_soc=0.35,
    )
    env = TerminalParallelEnv(config)

    assert config.warm_start_energy_reachability_check_enabled is False
    assert len(env.active) == 3


@pytest.mark.parametrize(
    "overrides",
    (
        {"reset_mode": "flow_from_empty"},
        {"energy_model_mode": "legacy_phase"},
        {"emergency_landing_enabled": False},
    ),
)
def test_enabled_flag_requires_compatible_reset_and_energy_modes(overrides: dict) -> None:
    with pytest.raises(ValueError, match="warm_start_energy_reachability_check_enabled requires"):
        _warm_config(**overrides)
