"""Tests for traffic-demand profile overrides."""

import pytest

from terminal_marl.config import load_experiment_config
from terminal_marl.experiments import DEMAND_PROFILES, apply_demand_profile, run_rollout


@pytest.mark.parametrize("level", ["low", "medium", "high", "bursty"])
def test_demand_profile_changes_arrival_flow(level: str) -> None:
    base = load_experiment_config("fixed_rule")
    updated = apply_demand_profile(base, level)
    expected = DEMAND_PROFILES[level]
    assert updated.initial_aircraft_count == expected["initial_aircraft_count"]
    assert updated.formation_rows * updated.formation_cols == updated.initial_aircraft_count
    assert updated.arrival_rate == expected["arrival_rate"]
    assert updated.traffic_demand_level == level


def test_demand_override_does_not_mutate_source_config() -> None:
    base = load_experiment_config("fixed_rule")
    original = base.initial_aircraft_count
    apply_demand_profile(base, "bursty")
    assert base.initial_aircraft_count == original


def test_rollout_records_demand_level() -> None:
    metrics = run_rollout("dynamic_rule", traffic_demand_level="high", max_steps=1)
    assert metrics["traffic_demand_level"] == "high"


def test_medium_demand_uses_recalibrated_capacity_rate() -> None:
    updated = apply_demand_profile(load_experiment_config("dynamic_hrl"), "medium")
    assert updated.initial_aircraft_count == 6
    assert updated.arrival_rate == pytest.approx(0.00625)


def test_unknown_demand_is_rejected() -> None:
    with pytest.raises(ValueError, match="demand"):
        apply_demand_profile(load_experiment_config("fixed_rule"), "extreme")
