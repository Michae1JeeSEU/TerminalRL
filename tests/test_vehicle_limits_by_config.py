"""Vehicle-limit regression tests for full-scale vs Crazyflie presets."""

import pytest

from terminal_marl.config import load_experiment_config


def test_full_scale_configs_have_transition_acceleration_headroom() -> None:
    for name in (
        "fixed_rule",
        "fixed_residual",
        "dynamic_rule",
        "dynamic_hrl",
        "full_hrl_no_safety",
    ):
        config = load_experiment_config(name)
        inner = min(config.rings, key=lambda ring: ring.radius)
        assert config.max_acceleration >= 2.0 * inner.speed**2 / inner.radius
        assert config.max_acceleration == pytest.approx(4.0)


def test_crazyflie_lab_keeps_scaled_limits_and_simulation_only() -> None:
    config = load_experiment_config("crazyflie_lab")
    assert config.simulation_only
    assert config.max_acceleration == pytest.approx(0.8)
    assert config.speed_max < 1.0
