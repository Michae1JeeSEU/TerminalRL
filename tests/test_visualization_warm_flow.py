"""Headless contract checks for the fixed-rule populated visualization scene."""
from __future__ import annotations

from terminal_marl.core.config import RingRole
from terminal_marl.core.uav import FlightPhase
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
    for uav in env.active.values():
        ring = env.airspace.actual_ring(uav.ring_role)
        assert uav.speed == ring.nominal_speed
        assert abs(sum(value * value for value in uav.velocity) ** 0.5 - uav.speed) < 1e-9


def test_fixed_rule_warm_flow_cannot_be_used_to_visualize_a_learned_mode():
    try:
        _config_for_mode("fixed_low_rl", 120.0, "fixed_rule_warm_flow")
    except ValueError as error:
        assert "fixed_rule" in str(error)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("fixed-rule-only visualization scenario accepted a learned mode")


def test_fixed_rule_warm_flow_keeps_o2m_and_m2i_moving_under_dynamics_constraints():
    config = _config_for_mode("fixed_rule", 120.0, "fixed_rule_warm_flow")
    env = TerminalParallelEnv(config)
    env.reset(seed=17)

    saw_o2m = saw_m2i = False
    for _ in range(90):
        env.step()
        saw_o2m |= any(uav.phase is FlightPhase.TRANSFER_O2M for uav in env.active.values())
        saw_m2i |= any(uav.phase is FlightPhase.TRANSFER_M2I for uav in env.active.values())
        for uav in env.active.values():
            if uav.committed_plan is None or uav.committed_plan.mode != "ring_transfer":
                continue
            assert uav.committed_plan.metadata["transfer_entry_speed_mps"] > 0.0

    assert saw_o2m and saw_m2i
    assert env.telemetry["arrivals_admitted"] > 0
    assert env.telemetry["hard_separation_violations"] == 0
    assert env.telemetry["nmac"] == 0
    assert env.telemetry["collisions"] == 0
