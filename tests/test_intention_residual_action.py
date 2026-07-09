"""Tests for the four-dimensional traffic-intention residual mode."""

import numpy as np
import pytest

from terminal_marl import ExperimentMode, SimConfig, TerminalParallelEnv


def test_intention_action_spec_has_four_named_semantics() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.FIXED_RESIDUAL)
    spec = env.action_spec(env.possible_agents[0])

    assert spec.shape == (4,)
    _, info = env.safety_filter.filter_intention_action(np.zeros(4))
    assert info["semantics"] == (
        "release_request",
        "speed_residual",
        "slot_residual",
        "timing_residual",
    )


def test_intention_residual_never_directly_changes_heading_or_altitude() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.FIXED_RESIDUAL)
    base = (1.25, 20.0, 180.0)

    command = env._apply_intention_residual(
        base,
        np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float32),
    )

    assert command[0] == base[0]
    assert command[2] == base[2]
    assert command[1] > base[1]


def test_zero_intention_keeps_rule_release_as_neutral_default() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.DYNAMIC_HRL)
    env.reset(seed=10)
    actions = {agent: np.zeros(4, dtype=np.float32) for agent in env.agents}

    _filtered, release_requests = env._prepare_intention_actions(actions)

    assert release_requests
    assert all(release_requests.values())


def test_negative_release_and_timing_can_defer_transfer() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.DYNAMIC_HRL)
    env.reset(seed=11)
    actions = {
        agent: np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32)
        for agent in env.agents
    }

    _filtered, release_requests = env._prepare_intention_actions(actions)

    assert not any(release_requests.values())


def test_step_reports_filtered_intention_fields() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.FIXED_RESIDUAL)
    env.reset(seed=12)
    actions = {
        agent: np.array([2.0, 0.5, -0.25, 0.75], dtype=np.float32)
        for agent in env.agents
    }

    _obs, _rewards, _terms, _truncs, infos = env.step(actions)
    first = infos[env.possible_agents[0]]["intention_action"]

    assert first["release_request"] == pytest.approx(1.0)
    assert first["speed_residual"] == pytest.approx(0.5)
    assert first["slot_residual"] == pytest.approx(-0.25)
    assert first["timing_residual"] == pytest.approx(0.75)
    assert "action_bounds" in first["violations"]


def test_rule_mode_ignores_residual_values_but_still_steps() -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.FIXED_RULE)
    env.reset(seed=13)
    actions = {agent: np.ones(4, dtype=np.float32) for agent in env.agents}

    _obs, rewards, _terms, _truncs, infos = env.step(actions)

    assert rewards
    assert all(info["action_mode"] == "rule" for info in infos.values())
    assert all(info["intention_action"] is None for info in infos.values())
