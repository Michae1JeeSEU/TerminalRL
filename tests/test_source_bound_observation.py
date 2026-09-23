from __future__ import annotations

import math

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.landing_zone import LZStatus
from terminal_marl.core.observations import (
    LOW_OBS_V2_1,
    LOW_OBS_V2_3,
    _center_wait_norm,
    build_center_broadcast,
    build_low_observation_v2_1,
    observation_schema_for_version,
)
from terminal_marl.core.topology_neighbors import TopologyCache
from terminal_marl.envs.parallel_env import TerminalParallelEnv


V2_3 = "phaseA_low_obs_v2_3_source_bound_88d"


def _config(*, schema: str = V2_3, count: int = 2, comm_range_m: float | None = 320.0) -> SimConfig:
    return SimConfig(
        experiment_mode="fixed_low_student",
        observation_schema_version=schema,
        neighbor_comm_range_m=comm_range_m,
        emergency_landing_enabled=True,
        reset_mode="warm_start_rings",
        warm_start_uavs=count,
        fixed_arrival_schedule=tuple(0.0 for _ in range(count)),
        total_requested_uavs=count,
        max_active_uavs=max(4, count),
    )


def test_v2_3_is_source_named_88d_finite_and_hash_distinct() -> None:
    env = TerminalParallelEnv(_config(count=2))
    observations, _ = env.reset(1203)
    assert LOW_OBS_V2_3.dimension == 88
    assert LOW_OBS_V2_3.feature_hash != LOW_OBS_V2_1.feature_hash
    assert observation_schema_for_version(V2_3) is LOW_OBS_V2_3
    assert observations and all(value.shape == (88,) for value in observations.values())
    assert all(np.isfinite(value).all() for value in observations.values())


def test_finite_comm_range_keeps_near_slot_and_excludes_far_uav() -> None:
    env = TerminalParallelEnv(_config(count=3))
    env.reset(1204)
    ego, near, far = tuple(env.active.values())
    ring = env.airspace.actual_ring(RingRole.OUTER)
    for item in (ego, near, far):
        item.ring_role = RingRole.OUTER
        item.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    ego.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    near.position = np.asarray((ring.radius, 100.0, ring.height), dtype=float)
    far.position = np.asarray((ring.radius + 400.0, 0.0, ring.height), dtype=float)
    cache = TopologyCache(
        env.airspace,
        env.active.values(),
        now=env.now,
        cpa_horizon=env.config.shield_horizon_seconds,
        comm_range_m=320.0,
    )
    slots = cache.slots_for(ego.uid)
    slot_uids = {slot.uid for slot in slots if slot.present}
    assert near.uid in slot_uids
    assert far.uid not in slot_uids


def test_center_broadcast_contains_urgent_count_and_per_pad_eta() -> None:
    env = TerminalParallelEnv(_config(count=2))
    env.reset(1205)
    ego, urgent = tuple(env.active.values())
    urgent.energy_remaining = 0.30 * urgent.energy_capacity
    env.lz_manager.reserve(0, urgent.uid, env.now)
    center = build_center_broadcast(env)
    assert urgent.uid in center.urgent_unassigned_uids
    assert center.free_lz_fraction == pytest.approx(5.0 / 6.0)
    reserved = next(zone for zone in center.landing_zones if zone.lz_id == 0)
    free = next(zone for zone in center.landing_zones if zone.lz_id == 1)
    assert reserved.status == "reserved"
    assert math.isinf(reserved.eta_seconds)
    assert free.status == "free"
    assert free.eta_seconds == pytest.approx(0.0)
    observation = env._observations()[str(ego.uid)]
    assert np.isfinite(observation).all()
    assert observation[LOW_OBS_V2_3.slices["center_urgent_unassigned_count_norm"]][0] > 0.0


def test_v2_1_vector_is_unchanged_when_source_bound_config_is_disabled() -> None:
    env = TerminalParallelEnv(_config(schema="phaseA_low_obs_v2_1_retry_state", count=2))
    env.reset(1206)
    uav = next(iter(env.active.values()))
    before = build_low_observation_v2_1(uav, env)
    env.config.neighbor_comm_range_m = 320.0
    after = build_low_observation_v2_1(uav, env)
    assert np.array_equal(before, after)


@pytest.mark.parametrize("bad_range", (None, 0.0, -1.0, math.inf, math.nan))
def test_v2_3_requires_finite_positive_comm_range(bad_range: float | None) -> None:
    with pytest.raises(ValueError, match="neighbor_comm_range_m"):
        _config(comm_range_m=bad_range)


def test_reserved_only_center_eta_maps_to_bounded_unavailable_value() -> None:
    env = TerminalParallelEnv(_config(count=1))
    env.reset(1207)
    uav = next(iter(env.active.values()))
    for lz_id, zone in enumerate(env.lz_manager.zones):
        env.lz_manager.reserve(lz_id, uav.uid + lz_id + 1, env.now)
    center = build_center_broadcast(env)
    assert all(zone.status is LZStatus.RESERVED for zone in env.lz_manager.zones)
    assert all(math.isinf(zone.eta_seconds) for zone in center.landing_zones)
    observation = env._observations()[str(uav.uid)]
    index = LOW_OBS_V2_3.slices["center_earliest_compatible_lz_release_wait_norm"]
    assert observation[index][0] == pytest.approx(1.0)


def test_known_reserved_center_eta_is_broadcast_and_normalized() -> None:
    env = TerminalParallelEnv(_config(count=1))
    env.reset(1208)
    uav = next(iter(env.active.values()))
    owners = {}
    for lz_id in range(len(env.lz_manager.zones)):
        owner = uav.uid + lz_id + 1
        owners[lz_id] = owner
        env.lz_manager.reserve(lz_id, owner, env.now)
    assert env.lz_manager.note_committed_landing(0, owners[0], env.now + 15.0)
    center = build_center_broadcast(env)
    known = next(zone for zone in center.landing_zones if zone.lz_id == 0)
    unknown = next(zone for zone in center.landing_zones if zone.lz_id == 1)
    assert known.eta_seconds == pytest.approx(15.0)
    assert math.isinf(unknown.eta_seconds)
    observation = env._observations()[str(uav.uid)]
    index = LOW_OBS_V2_3.slices["center_earliest_compatible_lz_release_wait_norm"]
    assert observation[index][0] == pytest.approx(15.0 / (15.0 + env.config.headway_max))


def test_center_wait_norm_remains_monotone_beyond_headway_max() -> None:
    config = _config(count=1)
    at_headway = _center_wait_norm(30.0, config)
    long_wait = _center_wait_norm(200.0, config)
    assert at_headway == pytest.approx(30.0 / (30.0 + config.headway_max))
    assert long_wait == pytest.approx(200.0 / (200.0 + config.headway_max))
    assert 0.0 < at_headway < long_wait < 1.0
