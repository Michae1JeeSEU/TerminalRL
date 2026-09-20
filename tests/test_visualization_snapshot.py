from __future__ import annotations

import numpy as np

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.visualization.snapshot import build_world_render_snapshot


def _config(**overrides) -> SimConfig:
    values = {
        "arrival_model": "finite_cohort",
        "fixed_arrival_schedule": (0.0,),
        "total_requested_uavs": 1,
        "max_simulation_seconds": 100.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _signature(env: TerminalParallelEnv) -> dict:
    return {
        "now": env.now,
        "step_count": env.step_count,
        "active": {
            uid: (uav.phase.value, tuple(uav.position), uav.energy_remaining, uav.trajectory_plan_id)
            for uid, uav in env.active.items()
        },
        "airspace": tuple(
            (env.airspace.actual_ring(role).radius, env.airspace.actual_ring(role).height,
             env.airspace.target_ring(role).radius, env.airspace.target_ring(role).height)
            for role in RingRole
        ),
        "lz": tuple((zone.status.value, zone.reserved_by, zone.occupied_by, zone.service_remaining) for zone in env.lz_manager.zones),
        "telemetry": dict(env.telemetry),
        "queue": len(env.admission.queue.items),
    }


def test_snapshot_is_read_only() -> None:
    env = TerminalParallelEnv(_config())
    env.reset(0)
    env.step()
    before = _signature(env)
    snapshot = build_world_render_snapshot(env)
    assert _signature(env) == before
    assert snapshot.time == env.now
    assert isinstance(snapshot.uavs[0].position, tuple)
    assert snapshot.uavs[0].energy_remaining == env.active[snapshot.uavs[0].uid].energy_remaining
    assert snapshot.uavs[0].base_priority == env.active[snapshot.uavs[0].uid].base_priority


def test_snapshot_uses_actual_dynamic_geometry() -> None:
    env = TerminalParallelEnv(_config())
    env.reset(0)
    env.apply_high_level_action(np.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), geometry_gate=True)
    env.airspace.step_transition(10.0)
    env.now = 10.0
    snapshot = build_world_render_snapshot(env)
    outer = next(ring for ring in snapshot.rings if ring.role == "outer")
    assert outer.radius == env.airspace.actual_ring(RingRole.OUTER).radius
    assert outer.radius != env.config.rings[RingRole.OUTER].radius
    assert outer.target_radius == env.airspace.target_ring(RingRole.OUTER).radius


def test_snapshot_uses_only_remaining_canonical_plan_points() -> None:
    env = TerminalParallelEnv(_config())
    env.reset(0)
    env.step()
    uav = next(iter(env.active.values()))
    assert uav.committed_plan is not None
    snapshot_uav = build_world_render_snapshot(env).uavs[0]
    assert snapshot_uav.trajectory_plan_id == uav.trajectory_plan_id
    expected = (tuple(float(value) for value in uav.position),) + tuple(
        tuple(float(value) for value in point.position)
        for point in uav.committed_plan.points
        if point.t > env.now + 1e-9
    )
    assert snapshot_uav.trajectory_points == expected
    assert snapshot_uav.trajectory_points[0] == tuple(float(value) for value in uav.position)


def test_snapshot_trajectory_downsampling_honors_exact_point_limit() -> None:
    env = TerminalParallelEnv(_config())
    env.reset(0)
    env.step()
    snapshot_uav = build_world_render_snapshot(env, trajectory_max_points=7).uavs[0]
    assert 2 <= len(snapshot_uav.trajectory_points) <= 7
    assert snapshot_uav.trajectory_points[0] == tuple(float(value) for value in env.active[snapshot_uav.uid].position)
    assert snapshot_uav.trajectory_points[-1] == tuple(
        float(value) for value in env.active[snapshot_uav.uid].committed_plan.points[-1].position
    )


def test_external_queue_is_not_fake_airborne_uav() -> None:
    env = TerminalParallelEnv(_config(fixed_arrival_schedule=(0.0, 1.0), total_requested_uavs=2, max_active_uavs=1))
    env.reset(0)
    env.step()
    env.step()
    snapshot = build_world_render_snapshot(env)
    assert snapshot.external_queue_size == 1
    assert len(snapshot.uavs) == len(env.active) == 1
