"""Tests for simulation-only full-HRL-without-safety ablation."""

from pathlib import Path

import pytest

from terminal_marl import ExperimentMode, TerminalParallelEnv
from terminal_marl.config import load_experiment_config, load_sim_config
from terminal_marl.experiments import run_rollout


def test_no_safety_preset_is_dynamic_and_simulation_only() -> None:
    config = load_experiment_config("full_hrl_no_safety")
    env = TerminalParallelEnv(config)
    assert config.simulation_only
    assert env.experiment_mode is ExperimentMode.FULL_HRL_NO_SAFETY
    assert not env.safety_enabled
    assert env.airspace.dynamic_enabled


def test_no_safety_metric_is_recorded() -> None:
    metrics = run_rollout("full_hrl_no_safety", max_steps=1)
    assert metrics["safety_enabled"] == 0.0


def test_hardware_like_config_cannot_disable_safety(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text(
        """
name: unsafe
experiment_mode: full_hrl_no_safety
simulation_only: false
safety:
  enabled: false
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="simulation_only"):
        load_sim_config(path)


def test_crazyflie_lab_cannot_disable_safety(tmp_path: Path) -> None:
    path = tmp_path / "lab_unsafe.yaml"
    path.write_text(
        """
name: lab_unsafe
experiment_mode: crazyflie_lab
simulation_only: true
safety:
  enabled: false
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="crazyflie_lab"):
        load_sim_config(path)
