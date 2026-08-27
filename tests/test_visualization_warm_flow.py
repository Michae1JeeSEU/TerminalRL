"""Headless contract checks for the fixed-rule populated visualization scene."""
from __future__ import annotations

from terminal_marl.core.config import RingRole
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.visualization.rollout_viewer import _config_for_mode


def test_fixed_rule_warm_flow_populates_rings_by_capacity_ratio_and_keeps_arrivals():
    config = _config_for_mode("fixed_rule", 120.0, "fixed_rule_warm_flow")
    assert config.reset_mode == "warm_start_rings"
    assert config.warm_start_uavs == 12
    assert config.gate_probabilities == (0.25, 0.25, 0.25, 0.25)
    env = TerminalParallelEnv(config)
    env.reset(seed=7)
    assert {role: env.airspace.occupancy(role) for role in RingRole} == {
        RingRole.OUTER: 6, RingRole.MIDDLE: 4, RingRole.INNER: 2,
    }
    assert env.arrivals.next_time == 8.0


def test_fixed_rule_warm_flow_cannot_be_used_to_visualize_a_learned_mode():
    try:
        _config_for_mode("fixed_low_rl", 120.0, "fixed_rule_warm_flow")
    except ValueError as error:
        assert "fixed_rule" in str(error)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("fixed-rule-only visualization scenario accepted a learned mode")
