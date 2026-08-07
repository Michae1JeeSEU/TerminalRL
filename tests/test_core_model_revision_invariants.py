import numpy as np

from terminal_marl.core.config import SimConfig
from terminal_marl.core.lifecycle import UAVLifecycleState, UAVOutcomeRecord
from terminal_marl.core.ring_capacity import geometric_ring_capacity
from terminal_marl.core.uav import UAVState
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def test_acceleration_limit_uses_sim_config_value() -> None:
    uav = UAVState(
        uid=1,
        position=np.array([0.0, 0.0, 100.0]),
        speed=10.0,
        heading=0.0,
    )

    uav.update_dynamics(
        desired_heading=0.0,
        desired_speed=20.0,
        desired_altitude=100.0,
        dt=1.0,
        max_turn_rate_deg=30.0,
        max_climb_rate=5.0,
        max_descent_rate=5.0,
        speed_min=0.0,
        speed_max=50.0,
        max_acceleration=4.0,
    )

    assert uav.speed == 14.0


def test_fixed_total_request_count_is_exact_and_conserved_until_guard() -> None:
    config = SimConfig(
        experiment_mode="dynamic_rule",
        action_mode="rule",
        initial_aircraft_count=0,
        total_requested_uavs=3,
        arrival_rate=1.0,
        arrival_model="deterministic_interval",
        max_active_uavs=1,
        max_steps=2,
        battery_endurance_seconds=100.0,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=7)

    for _ in range(4):
        env.step({})

    summary = env.simulation_outcome_summary()
    assert env.arrival_request_count == 3
    assert summary["total_requested"] == 3
    assert (
        summary["landed"]
        + summary["crashed_collision"]
        + summary["crashed_battery"]
        + summary["unresolved"]
        == 3
    )
    assert summary["conservation_ok"]


def test_battery_depletion_in_external_queue_becomes_terminal_outcome() -> None:
    config = SimConfig(
        experiment_mode="dynamic_rule",
        action_mode="rule",
        initial_aircraft_count=0,
        total_requested_uavs=2,
        arrival_rate=10.0,
        arrival_model="deterministic_interval",
        max_active_uavs=0 + 1,
        max_steps=20,
        battery_endurance_seconds=0.5,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=0)

    for _ in range(3):
        env.step({})

    outcomes = {record.terminal_outcome for record in env.outcome_ledger.values()}
    assert UAVLifecycleState.CRASHED_BATTERY.value in outcomes
    assert env.metrics()["battery_crash_count"] >= 1


def test_collision_distance_marks_single_terminal_collision_event() -> None:
    config = SimConfig(
        initial_aircraft_count=0,
        total_requested_uavs=0,
        collision_distance=25.0,
        safe_distance=80.0,
        max_steps=10,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=0)
    a = UAVState(uid=1, position=np.array([0.0, 0.0, 100.0]), speed=0.0, heading=0.0)
    b = UAVState(uid=2, position=np.array([1.0, 0.0, 100.0]), speed=0.0, heading=0.0)
    for arrival_id, ac in enumerate((a, b)):
        ac.arrival_id = arrival_id
        ac.request_time = 0.0
        ac.admission_time = 0.0
    env.aircraft = [a, b]
    env.outcome_ledger = {}
    for arrival_id, ac in enumerate(env.aircraft):
        env.outcome_ledger[arrival_id] = env.outcome_ledger.get(arrival_id) or UAVOutcomeRecord(
            arrival_id=arrival_id,
            request_time=0.0,
            scenario="unit",
            admission_time=0.0,
            uid=ac.uid,
        )

    env._track_conflicts()
    env._track_conflicts()

    assert env.collision_count == 1
    assert all(ac.finished for ac in env.aircraft)
    assert {
        record.terminal_outcome for record in env.outcome_ledger.values()
    } == {UAVLifecycleState.CRASHED_COLLISION.value}


def test_geometry_transition_capacity_uses_actual_radius_not_target() -> None:
    config = SimConfig(
        experiment_mode="dynamic_hrl",
        action_mode="rule",
        initial_aircraft_count=0,
        total_requested_uavs=0,
        geometry_transition_seconds=20.0,
        dt=5.0,
        high_level_interval_steps=1,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=0)
    old_radius = env.airspace.outer_ring.radius

    info = env.apply_high_level_action(np.array([1.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=float))
    target_radius = float(info["target_state"][0])

    assert env.airspace.outer_ring.radius == old_radius
    assert target_radius > old_radius

    env.step({})

    actual_radius = env.airspace.outer_ring.radius
    assert old_radius < actual_radius < target_radius
    assert env.airspace.outer_ring.capacity == geometric_ring_capacity(
        actual_radius,
        config.safe_distance + config.ring_separation_margin_m,
    )
