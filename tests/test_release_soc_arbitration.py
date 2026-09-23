"""Focused contracts for the opt-in SOC-aware release arbitration mode."""

from __future__ import annotations

from dataclasses import replace

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _warm_config(**overrides) -> SimConfig:
    values = dict(
        reset_mode="warm_start_rings",
        warm_start_uavs=12,
        fixed_arrival_schedule=(0.0,) * 12,
        total_requested_uavs=12,
        release_arbitration_mode="downstream_soc_priority",
    )
    values.update(overrides)
    return replace(SimConfig(), **values)


def test_soc_priority_mode_is_opt_in_and_legacy_default_is_unchanged() -> None:
    assert SimConfig().release_arbitration_mode == "legacy_insertion"
    assert _warm_config().release_arbitration_mode == "downstream_soc_priority"


def test_soc_priority_orders_downstream_stages_then_soc_priority_wait_uid() -> None:
    env = TerminalParallelEnv(_warm_config())
    env.reset(seed=809006)

    grouped = {
        role: sorted(
            (uav for uav in env.active.values() if uav.ring_role is role),
            key=lambda item: item.uid,
        )
        for role in RingRole
    }
    assert {role: len(items) for role, items in grouped.items()} == {
        RingRole.OUTER: 6,
        RingRole.MIDDLE: 4,
        RingRole.INNER: 2,
    }

    expected_by_role: dict[RingRole, list[int]] = {}
    for role, items in grouped.items():
        # The first two aircraft share SOC, so priority decides their order.
        # Remaining aircraft share all arbitration fields, so UID decides.
        for index, item in enumerate(items):
            item.energy_remaining = item.energy_capacity * (0.30 if index < 2 else 0.60)
            item.dynamic_priority = 1.0 if index == 0 else 4.0 if index == 1 else 2.0
            item.external_wait_time = 50.0 if index == 0 else 10.0 if index == 1 else 20.0
            item.internal_wait_time = 0.0
        expected_by_role[role] = [items[1].uid, items[0].uid] + [item.uid for item in items[2:]]

    ordered = env._release_processing_order()
    ordered_uids = [uid for uid, _ in ordered]
    expected = (
        expected_by_role[RingRole.INNER]
        + expected_by_role[RingRole.MIDDLE]
        + expected_by_role[RingRole.OUTER]
    )
    assert ordered_uids == expected


def test_default_mode_keeps_insertion_order_even_when_soc_differs() -> None:
    env = TerminalParallelEnv(
        _warm_config(release_arbitration_mode="legacy_insertion")
    )
    env.reset(seed=809006)
    for index, uav in enumerate(env.active.values()):
        uav.energy_remaining = uav.energy_capacity * (0.20 + 0.01 * index)
        uav.dynamic_priority = 100.0 - index

    assert [uid for uid, _ in env._release_processing_order()] == list(env.active)
