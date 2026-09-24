from __future__ import annotations

import numpy as np

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.trajectory_generator import TrajectoryGenerator
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeScheduler
from terminal_marl.envs.parallel_env import WorldSnapshot


def _fixture():
    config = SimConfig(
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=0)
    uav = next(iter(env.active.values()))
    scheduler = MergeScheduler(config)
    proposal = scheduler.propose(
        uav,
        LowLevelIntent(1.0, 0.0, 0.0),
        WorldSnapshot(env.now, env.active, env.airspace),
        RingRole.MIDDLE,
    )
    reservation = scheduler.reserve(uav, proposal)
    return env, uav, reservation


def test_ring_hold_endpoint_matches_full_authoritative_hold_at_exact_endpoint():
    env, uav, _ = _fixture()
    generator = TrajectoryGenerator(env.config)
    ring = env.airspace.actual_ring(RingRole.OUTER)

    for duration in (0.2, 1.13, 8.0, 31.75):
        endpoint = generator._ring_hold_endpoint(
            ring, uav.ring_progress, uav.speed, ring.nominal_speed, duration,
        )
        reference = generator.ring_hold_points(
            ring, uav.ring_progress, uav.speed, ring.nominal_speed, env.now, duration,
        )[-1]
        np.testing.assert_array_equal(endpoint[0], reference.position)
        np.testing.assert_array_equal(endpoint[1], reference.velocity)


def test_repaired_transfer_materializes_hold_only_for_setup_and_selected_candidate(monkeypatch):
    env, uav, reservation = _fixture()
    generator = env.generator
    calls = []
    original = generator.ring_hold_points

    def counted(*args, **kwargs):
        calls.append((args[-1], kwargs.get("duration")))
        return original(*args, **kwargs)

    monkeypatch.setattr(generator, "ring_hold_points", counted)
    plan = generator.ring_transfer(
        uav, RingRole.OUTER, RingRole.MIDDLE, reservation, env.airspace, env.now,
    )

    assert plan.metadata["transfer_geometry"] == "forward_annular_v1"
    assert len(calls) <= 2
