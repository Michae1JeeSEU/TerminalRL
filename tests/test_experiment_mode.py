"""Tests for unified experiment-mode routing."""

import pytest

from terminal_marl import ExperimentMode, SimConfig, TerminalParallelEnv


def test_required_experiment_modes_are_available() -> None:
    values = {mode.value for mode in ExperimentMode}
    assert {
        "fixed_rule",
        "fixed_residual",
        "dynamic_rule",
        "dynamic_hrl",
        "event_rule",
        "periodic_hrl",
        "event_gated_hrl",
        "crazyflie_lab",
    } <= values


@pytest.mark.parametrize(
    ("mode", "dynamic", "action_mode"),
    [
        (ExperimentMode.FIXED_RULE, False, "rule"),
        (ExperimentMode.FIXED_RESIDUAL, False, "intention_residual"),
        (ExperimentMode.DYNAMIC_RULE, True, "rule"),
        (ExperimentMode.DYNAMIC_HRL, True, "intention_residual"),
        (ExperimentMode.EVENT_RULE, True, "rule"),
        (ExperimentMode.PERIODIC_HRL, True, "intention_residual"),
        (ExperimentMode.EVENT_GATED_HRL, True, "intention_residual"),
        (ExperimentMode.CRAZYFLIE_LAB, True, "intention_residual"),
    ],
)
def test_mode_switches_are_consistent(
    mode: ExperimentMode,
    dynamic: bool,
    action_mode: str,
) -> None:
    assert mode.dynamic_airspace is dynamic
    assert mode.action_mode == action_mode
    assert mode.simulation_only


def test_environment_routes_from_string_mode() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode="dynamic_hrl")
    assert env.experiment_mode is ExperimentMode.DYNAMIC_HRL
    assert env.airspace.dynamic_enabled
    assert env.action_mode == "intention_residual"
    assert env.config.experiment_mode == "dynamic_hrl"


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported experiment mode"):
        ExperimentMode.parse("free_flight")


def test_legacy_default_preserves_three_dimensional_action() -> None:
    env = TerminalParallelEnv(SimConfig())
    assert env.experiment_mode is ExperimentMode.LEGACY_RESIDUAL
    assert env.action_spec(env.possible_agents[0]).shape == (3,)
