"""Focused contracts for the opt-in dense landing-flow semantics.

The historical Phase-A defaults are deliberately covered alongside the dense
viewer profile.  This keeps the throughput changes opt-in and prevents the
legacy qualification protocols from silently changing their replay order or
single-final-resource assumption.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.landing_zone import LandingZoneManager
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.core.uav import FlightPhase
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.evaluate_p2_dense_throughput_v1 import config_for as dense_config_for
from terminal_marl.visualization.ppo_checkpoint_viewer import _p2_scene_config


def _dense_warm_config(**overrides) -> SimConfig:
    """Build the canonical 6/4/2 warm-start state without loading a checkpoint."""

    values = dict(
        reset_mode="warm_start_rings",
        warm_start_uavs=12,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,) * 12,
        total_requested_uavs=12,
        lz_assignment_mode="nearest_available",
        release_arbitration_mode="downstream_priority",
        final_queue_capacity=3,
    )
    values.update(overrides)
    return replace(SimConfig(), **values)


def test_legacy_defaults_remain_lowest_id_single_final_capacity() -> None:
    config = SimConfig()

    assert config.lz_assignment_mode == "lowest_id"
    assert config.release_arbitration_mode == "legacy_insertion"
    assert config.final_queue_capacity == 1


def test_p2_dense24_opts_into_throughput_resources_and_arbitration() -> None:
    config = _p2_scene_config("p2_dense24")

    assert config.lz_assignment_mode == "nearest_available"
    assert config.release_arbitration_mode == "downstream_priority"
    assert config.final_queue_capacity == 3


@pytest.mark.parametrize(
    ("semantics", "lz_mode", "arbitration_mode", "capacity"),
    (
        ("legacy", "lowest_id", "legacy_insertion", 1),
        ("throughput_v1", "nearest_available", "downstream_priority", 3),
    ),
)
def test_dense_throughput_diagnostic_config_selects_explicit_semantics(
    semantics: str,
    lz_mode: str,
    arbitration_mode: str,
    capacity: int,
) -> None:
    config = dense_config_for(semantics)

    assert config.lz_assignment_mode == lz_mode
    assert config.release_arbitration_mode == arbitration_mode
    assert config.final_queue_capacity == capacity


def test_dense_throughput_diagnostic_rejects_unknown_semantics() -> None:
    with pytest.raises(ValueError, match="unsupported dense semantics"):
        dense_config_for("not_a_dense_semantics")


def test_downstream_priority_orders_inner_middle_outer_then_priority_wait_uid() -> None:
    env = TerminalParallelEnv(_dense_warm_config())
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

    # Every phase contains a priority tie, a wait tie, and a UID tie-break.
    # The same pattern is truncated for the smaller middle/inner groups.
    ordering_fields = (
        (1.0, 10.0),
        (4.0, 10.0),
        (4.0, 20.0),
        (4.0, 20.0),
        (4.0, 5.0),
        (1.0, 50.0),
    )
    expected_by_role: dict[RingRole, list[int]] = {}
    for role, items in grouped.items():
        for item, (priority, total_wait) in zip(items, ordering_fields):
            item.dynamic_priority = priority
            item.external_wait_time = total_wait
            item.internal_wait_time = 0.0
        expected_by_role[role] = [
            item.uid
            for item in sorted(
                items,
                key=lambda item: (
                    -item.dynamic_priority,
                    -(item.external_wait_time + item.internal_wait_time),
                    item.uid,
                ),
            )
        ]

    ordered = env._release_processing_order()
    ordered_uids = [uid for uid, _ in ordered]
    expected = (
        expected_by_role[RingRole.INNER]
        + expected_by_role[RingRole.MIDDLE]
        + expected_by_role[RingRole.OUTER]
    )
    assert ordered_uids == expected


def test_three_inner_requests_use_distinct_lzs_after_safety_gate_and_capacity_stops_fourth() -> None:
    config = _dense_warm_config()
    env = TerminalParallelEnv(config)
    env.reset(seed=809006)
    inner = sorted(env.active.values(), key=lambda item: item.uid)[:3]
    ring = env.airspace.actual_ring(RingRole.INNER)
    for index, uav in enumerate(inner):
        # Make the selected holders an explicit three-request inner-ring probe;
        # only the resource/safety gate is under test here.
        theta = index * (np.pi / 2.0)
        uav.phase = FlightPhase.RING_INNER
        uav.ring_role = RingRole.INNER
        uav.ring_progress = theta
        uav.position = np.asarray(
            (ring.radius * np.cos(theta), ring.radius * np.sin(theta), ring.height),
            dtype=float,
        )
        uav.committed_plan = None

    manager = LandingZoneManager(config.lz_assignment_mode)
    assigned: list[int] = []
    for uav in inner:
        final_available = sum(item.phase is FlightPhase.FINAL for item in inner) < config.final_queue_capacity
        verdict = env.safety.filter_intent(
            uav,
            LowLevelIntent(1.0, 0.0, 0.0),
            final_available=final_available,
        )
        assert verdict.approved
        lz_id = manager.choose_rule_zone(uav, env.now)
        assert lz_id is not None
        reservation = manager.reserve(lz_id, uav.uid, env.now)
        assert reservation.approved
        assigned.append(lz_id)
        uav.phase = FlightPhase.FINAL

    assert len(assigned) == 3
    assert len(set(assigned)) == 3

    fourth = next(item for item in env.active.values() if item.uid not in {uav.uid for uav in inner})
    fourth.phase = FlightPhase.RING_INNER
    fourth.ring_role = RingRole.INNER
    blocked = env.safety.filter_intent(
        fourth,
        LowLevelIntent(1.0, 0.0, 0.0),
        final_available=sum(item.phase is FlightPhase.FINAL for item in inner) < config.final_queue_capacity,
    )
    assert not blocked.approved
    assert blocked.reason == "final_unavailable"
    assert len(manager.available_zones(fourth, env.now)) == 3
