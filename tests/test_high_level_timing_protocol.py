"""Tests for decoupled high-level timing protocol fields."""

from pathlib import Path

import pytest

from terminal_marl.config import load_sim_config
from terminal_marl.core.config import SimConfig
from terminal_marl.experiments.evaluation_protocol import (
    WindowedEvaluationProtocol,
    configure_high_level_timing,
)


def test_yaml_high_level_protocol_maps_decision_seconds_to_steps(tmp_path: Path) -> None:
    path = tmp_path / "timing.yaml"
    path.write_text(
        """
name: timing
experiment_mode: dynamic_hrl
simulation_only: true
dt: 1.0
high_level:
  metrics_sampling_interval_seconds: 20
  decision_interval_seconds: 60
  reconfiguration_dwell_seconds: 60
  geometry_transition_seconds: 20
""".strip(),
        encoding="utf-8",
    )

    config = load_sim_config(path)

    assert config.metrics_sampling_interval_seconds == pytest.approx(20.0)
    assert config.decision_interval_seconds == pytest.approx(60.0)
    assert config.reconfiguration_dwell_seconds == pytest.approx(60.0)
    assert config.geometry_transition_seconds == pytest.approx(20.0)
    assert config.high_level_interval_steps == 60


def test_legacy_high_level_interval_still_sets_decision_interval(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "name: legacy\nexperiment_mode: dynamic_hrl\nsimulation_only: true\ndt: 0.5\nhigh_level_interval: 5\n",
        encoding="utf-8",
    )

    config = load_sim_config(path)

    assert config.high_level_interval_steps == 10
    assert config.decision_interval_seconds == pytest.approx(5.0)
    assert config.reconfiguration_dwell_seconds == pytest.approx(5.0)


def test_configure_protocol_keeps_sampling_separate_from_decision_interval() -> None:
    base = SimConfig(dt=1.0, high_level_interval_steps=20)
    protocol = WindowedEvaluationProtocol(
        metrics_sampling_interval_seconds=20.0,
        decision_interval_seconds=100.0,
        reconfiguration_dwell_seconds=100.0,
        geometry_transition_seconds=20.0,
    )

    updated = configure_high_level_timing(base, protocol)

    assert updated.metrics_sampling_interval_seconds == pytest.approx(20.0)
    assert updated.decision_interval_seconds == pytest.approx(100.0)
    assert updated.high_level_interval_steps == 100
    assert base.high_level_interval_steps == 20
