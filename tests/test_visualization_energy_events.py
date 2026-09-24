from __future__ import annotations

import pickle

from terminal_marl.core.config import SimConfig
from terminal_marl.core.uav import TerminalOutcome
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.visualization.pyvista_viewer import _snapshot_for_process
from terminal_marl.visualization.schema import TerminalEventRenderState
from terminal_marl.visualization.snapshot import build_world_render_snapshot


def _config(**overrides) -> SimConfig:
    values = {
        "arrival_model": "finite_cohort",
        "fixed_arrival_schedule": (0.0, 0.0),
        "total_requested_uavs": 2,
        "max_active_uavs": 2,
        "max_simulation_seconds": 100.0,
        "emergency_landing_enabled": True,
        "emergency_trigger_soc": 0.40,
    }
    values.update(overrides)
    return SimConfig(**values)


def _active_env(**overrides) -> TerminalParallelEnv:
    env = TerminalParallelEnv(_config(**overrides))
    env.reset(0)
    env.step()
    return env


def test_snapshot_energy_state_and_event_log_are_read_only_and_pickleable() -> None:
    env = _active_env()
    before = [dict(row) for row in env.energy_outcomes]
    snapshot = build_world_render_snapshot(env)
    copied = _snapshot_for_process(snapshot)

    assert [dict(row) for row in env.energy_outcomes] == before
    assert pickle.loads(pickle.dumps(copied)) == copied
    assert all(isinstance(event, TerminalEventRenderState) for event in snapshot.terminal_events)


def test_energy_alert_marks_low_soc_and_emergency_with_emergency_priority() -> None:
    env = _active_env()
    uav = next(iter(env.active.values()))
    uav.energy_remaining = 0.30 * uav.energy_capacity

    low_soc = next(item for item in build_world_render_snapshot(env).uavs if item.uid == uav.uid)
    assert low_soc.energy_alert == "low_soc"
    assert low_soc.emergency_landing is False

    uav.emergency_landing = True
    emergency = next(item for item in build_world_render_snapshot(env).uavs if item.uid == uav.uid)
    assert emergency.energy_alert == "emergency"
    assert emergency.emergency_landing is True


def test_resolved_failure_is_retained_after_active_uav_is_removed() -> None:
    env = _active_env()
    uav = next(iter(env.active.values()))
    position = tuple(float(value) for value in uav.position)
    env._resolve(uav, TerminalOutcome.FAILED_ENERGY, "energy_depleted")

    snapshot = build_world_render_snapshot(env)
    assert uav.uid not in env.active
    assert len(snapshot.terminal_events) == 1
    event = snapshot.terminal_events[0]
    assert event.uid == uav.uid
    assert event.outcome == TerminalOutcome.FAILED_ENERGY.value
    assert event.position == position
    assert event.reason == "energy_depleted"


def test_terminal_event_snapshots_accumulate_without_dropping_prior_events() -> None:
    env = _active_env()
    first, second = tuple(env.active.values())
    env._resolve(first, TerminalOutcome.FAILED_ENERGY, "first_failure")
    first_snapshot = build_world_render_snapshot(env)

    env._resolve(second, TerminalOutcome.FAILED_ENERGY, "second_failure")
    second_snapshot = build_world_render_snapshot(env)

    assert [event.reason for event in first_snapshot.terminal_events] == ["first_failure"]
    assert [event.reason for event in second_snapshot.terminal_events] == [
        "first_failure",
        "second_failure",
    ]


def test_external_queue_outcome_with_no_uid_keeps_position_unknown() -> None:
    env = TerminalParallelEnv(_config(fixed_arrival_schedule=(0.0,), total_requested_uavs=1))
    env.reset(0)
    env.energy_outcomes.append(
        {
            "arrival_id": 91,
            "uid": None,
            "outcome": TerminalOutcome.DIVERTED_CAPACITY.value,
            "reason": "capacity",
            "time": 3.0,
            "soc": 0.20,
            "energy_remaining": 300.0,
            "phase": "external_queue",
        }
    )

    event = build_world_render_snapshot(env).terminal_events[0]
    assert event.uid is None
    assert event.position is None
