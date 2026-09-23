from __future__ import annotations

import json

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_joint_index_v4_1
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MissionType,
    QueuedArrivalState,
    UAVState,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _config(**overrides) -> SimConfig:
    values = {
        "action_schema_version": "low_action_schema_v4_2_hierarchical",
        "observation_schema_version": "phaseA_low_obs_v2_1_retry_state",
        "experiment_mode": "fixed_low_student",
        "teacher_enabled": False,
        "scheduler_mode": "execution",
        "runtime_proposal_shortlist_k": 3,
        "energy_model_mode": "phase_motion_v2",
        "emergency_landing_enabled": True,
        "emergency_priority_guard_enabled": True,
        "emergency_trigger_soc": 0.40,
        "arrival_model": "finite_cohort",
        "total_requested_uavs": 0,
        "arrival_rate_per_second": 0.0,
        "max_simulation_seconds": 120.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _ring_uav(env: TerminalParallelEnv, role: RingRole, *, soc: float, uid: int) -> UAVState:
    request = ArrivalRequest(
        uid, 0.0, 0, MissionType.STANDARD, 1, env.config.energy_capacity, soc
    )
    queued = QueuedArrivalState(request, request.energy_capacity * soc)
    uav = UAVState.from_request(uid, request, queued, np.zeros(3), env.now)
    uav.uid = uid
    uav.phase = {
        RingRole.OUTER: FlightPhase.RING_OUTER,
        RingRole.MIDDLE: FlightPhase.RING_MIDDLE,
        RingRole.INNER: FlightPhase.RING_INNER,
    }[role]
    uav.ring_role = role
    uav.ring_progress = 0.0
    ring = env.airspace.actual_ring(role)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    uav.acceleration = np.asarray(
        (-(ring.nominal_speed**2) / ring.radius, 0.0, 0.0), dtype=float
    )
    env.active[uid] = uav
    env.ledger.request(request.arrival_id)
    return uav


def test_guard_defaults_off_and_rejects_incompatible_enablement() -> None:
    assert SimConfig().emergency_priority_guard_enabled is False
    with pytest.raises(ValueError, match="requires emergency_landing_enabled"):
        SimConfig(emergency_priority_guard_enabled=True)
    with pytest.raises(ValueError, match="requires emergency_landing_enabled"):
        SimConfig(
            emergency_priority_guard_enabled=True,
            emergency_landing_enabled=True,
        )
    with pytest.raises(ValueError, match="requires emergency_landing_enabled"):
        SimConfig(
            emergency_priority_guard_enabled=True,
            emergency_landing_enabled=True,
            energy_model_mode="phase_motion_v2",
        )


def test_low_soc_ring_transfer_guard_has_no_reservation_or_retry() -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.OUTER, soc=0.401, uid=1)
    intent = LowLevelIntent(1.0, 0.0, 0.0)

    decision = env._runtime_shortlist_release(uav, intent, (intent,))

    assert decision == "emergency_energy_guard"
    assert uav.committed_plan is None
    assert uav.safety_retry_after == 0.0
    assert uav.emergency_retry_after == 0.0
    assert not env.scheduler.reservations
    assert env.last_release_telemetry[uav.uid]["primary_blocker"] == "SOC"
    assert env.telemetry["emergency_energy_guard_rejections"] == 1


def test_healthy_final_waits_when_one_pad_is_needed_by_urgent_other() -> None:
    env = TerminalParallelEnv(_config())
    requester = _ring_uav(env, RingRole.INNER, soc=0.80, uid=1)
    urgent = _ring_uav(env, RingRole.OUTER, soc=0.39, uid=2)
    for lz_id in range(1, len(env.lz_manager.zones)):
        assert env.lz_manager.reserve(lz_id, 100 + lz_id, env.now).approved
    intent = LowLevelIntent(1.0, 0.0, 0.0)

    decision = env._runtime_shortlist_release(requester, intent, (intent,))

    assert decision == "emergency_lz_priority"
    assert requester.committed_plan is None
    assert env.lz_manager.zones[0].status.value == "free"
    assert not env.scheduler.reservations
    assert env.last_release_telemetry[requester.uid]["primary_blocker"] == "LZ_RESOURCE"
    assert urgent.assigned_lz is None
    assert env.telemetry["emergency_lz_priority_rejections"] == 1


def test_at_risk_requester_may_take_safe_final_approach() -> None:
    env = TerminalParallelEnv(_config())
    requester = _ring_uav(env, RingRole.INNER, soc=0.39, uid=1)
    intent = LowLevelIntent(1.0, 0.0, 0.0)

    decision = env._runtime_shortlist_release(requester, intent, (intent,))

    assert decision == "approved"
    assert requester.assigned_lz is not None
    assert requester.committed_plan is not None
    assert requester.safety_retry_after == 0.0


def test_emergency_retry_cooldown_is_independent_from_normal_release_retry() -> None:
    env = TerminalParallelEnv(_config())
    uav = _ring_uav(env, RingRole.INNER, soc=0.39, uid=1)
    uav.safety_retry_after = 100.0
    for lz_id in range(len(env.lz_manager.zones)):
        assert env.lz_manager.reserve(lz_id, 100 + lz_id, env.now).approved

    assert env._try_emergency_landing(uav) is False
    assert env.telemetry["emergency_attempts"] == 1
    assert uav.safety_retry_after == 100.0
    assert uav.emergency_retry_after == 5.0


def test_flagged_v4_2_action_tape_replays_exactly_with_guard_state(tmp_path) -> None:
    config = _config(
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
    )
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(901)
    actions = {
        uid: hierarchical_from_joint_index_v4_1(0).normalized
        for uid in observations
    }
    recorder = SafetyTraceRecorder(tmp_path, dt=config.dt)
    recorder.start_episode(env, episode_seed=901, policy_rng_state={})
    record = recorder.record_pre_step(env, observations, actions)
    _, _, _, _, info = env.step(actions)
    recorder.record_post_step(env, record, info)
    path = recorder.dump(reason="emergency_priority_guard_test")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "emergency_retry_after" in next(iter(payload["records"][0]["pre_state"]["uavs"]))
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), path)
    assert replay["exact"]
