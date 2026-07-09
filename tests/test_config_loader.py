"""Tests for YAML preset loading into SimConfig."""

from pathlib import Path

import pytest

from terminal_marl import SimConfig, TerminalParallelEnv
from terminal_marl.config import load_experiment_config, load_sim_config


def test_crazyflie_yaml_generates_scaled_sim_config() -> None:
    config = load_experiment_config("crazyflie_lab")

    assert isinstance(config, SimConfig)
    assert config.config_name == "crazyflie_lab"
    assert config.experiment_mode == "crazyflie_lab"
    assert config.simulation_only
    assert config.dt == pytest.approx(0.2)
    assert config.high_level_interval_steps == 25
    assert [(ring.index, ring.radius) for ring in config.rings] == [(1, 0.7), (2, 1.2)]
    assert config.speed_max == pytest.approx(0.6)
    assert config.max_acceleration == pytest.approx(0.8)
    assert config.safe_distance == pytest.approx(0.5)
    assert config.world_boundary == pytest.approx((-1.8, 1.8, -1.8, 1.8, 0.1, 1.6))


@pytest.mark.parametrize(
    "name",
    ["fixed_rule", "fixed_residual", "dynamic_rule", "dynamic_hrl"],
)
def test_ablation_presets_load_and_construct_environment(name: str) -> None:
    config = load_experiment_config(name)
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(seed=1)

    assert config.config_name == name
    assert config.experiment_mode == name
    assert observations
    assert env.airspace.validate()


def test_loader_supports_all_requested_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        """
name: custom
experiment_mode: dynamic_hrl
simulation_only: true
dt: 0.5
high_level_interval: 5.0
rings:
  outer:
    index: 2
    radius: 300.0
    height: 300.0
    speed: 24.0
    capacity: 5
  inner:
    index: 1
    radius: 150.0
    height: 150.0
    speed: 16.0
    capacity: 2
vehicle_limits:
  speed_min: 5.0
  speed_max: 30.0
  max_acceleration: 1.5
safety:
  safe_distance: 60.0
  nmac_distance: 30.0
timing:
  final_min_time_gap_s: 9.0
arrival_flow:
  initial_aircraft_count: 4
  arrival_rate: 0.2
reward_weights:
  progress_weight: 0.125
""".strip(),
        encoding="utf-8",
    )

    config = load_sim_config(config_path)

    assert config.high_level_interval_steps == 10
    assert config.max_acceleration == pytest.approx(1.5)
    assert config.safe_distance == pytest.approx(60.0)
    assert config.final_min_time_gap_s == pytest.approx(9.0)
    assert config.initial_aircraft_count == 4
    assert config.arrival_rate == pytest.approx(0.2)
    assert config.progress_weight == pytest.approx(0.125)


def test_loader_overlays_without_mutating_python_default(tmp_path: Path) -> None:
    base = SimConfig(speed_max=31.0)
    path = tmp_path / "overlay.yaml"
    path.write_text(
        "name: overlay\nexperiment_mode: fixed_rule\nsimulation_only: true\n",
        encoding="utf-8",
    )

    loaded = load_sim_config(path, base=base)

    assert loaded is not base
    assert loaded.speed_max == pytest.approx(31.0)
    assert base.config_name == "python_default"


def test_non_simulation_preset_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "hardware.yaml"
    path.write_text(
        "name: forbidden\nexperiment_mode: crazyflie_lab\nsimulation_only: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="simulation_only"):
        load_sim_config(path)
