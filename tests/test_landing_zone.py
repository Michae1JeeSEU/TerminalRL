import math
import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.landing_zone import LZStatus, LandingZoneManager
from terminal_marl.core.merge_scheduler import MergeReservation
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MISSION_SERVICE_SECONDS,
    MissionType,
    QueuedArrivalState,
    UAVState,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def make_uav(position, mission=MissionType.CARGO):
    return UAVState(
        uid=7,
        arrival_id=7,
        request_time=0.0,
        mission_type=mission,
        base_priority=1,
        assigned_gate=0,
        energy_capacity=100.0,
        phase=FlightPhase.RING_INNER,
        position=np.asarray(position, dtype=float),
        velocity=np.zeros(3),
        speed=10.0,
        heading=0.0,
        phase_entry_time=0.0,
        energy_remaining=100.0,
    )


def test_default_assignment_preserves_lowest_id_rule():
    manager = LandingZoneManager()
    uav = make_uav(manager.zones[-1].position)

    assert manager.assignment_mode == "lowest_id"
    assert manager.rank_available_zones(uav) == [0, 1, 2, 3, 4, 5]
    assert manager.choose_rule_zone(uav) == 0


def test_nearest_assignment_ranks_by_three_dimensional_distance():
    manager = LandingZoneManager("nearest_available")
    uav = make_uav((0.0, 39.0, 1.0))

    assert manager.choose_rule_zone(uav) == 4
    assert manager.rank_available_zones(uav)[0] == 4


def test_nearest_assignment_uses_lz_id_as_tie_breaker():
    manager = LandingZoneManager("nearest_available")
    manager.zones[0].position = np.array((1.0, 0.0, 0.0))
    manager.zones[1].position = np.array((-1.0, 0.0, 0.0))
    uav = make_uav((0.0, 0.0, 0.0))

    assert manager.choose_rule_zone(uav) == 0


def test_reserved_occupied_and_incompatible_zones_are_excluded():
    manager = LandingZoneManager("nearest_available")
    manager.zones[2].compatible_missions = frozenset((MissionType.STANDARD,))
    assert manager.reserve(0, uav_id=11, now=0.0).approved
    assert manager.reserve(1, uav_id=12, now=0.0).approved
    manager.touchdown(1, uav_id=12, mission=MissionType.CARGO)
    uav = make_uav(manager.zones[0].position)

    assert manager.zones[1].status is LZStatus.OCCUPIED
    assert manager.available_zones(uav) == [3, 4, 5]
    assert manager.choose_rule_zone(uav) == 3


def test_earliest_release_time_tracks_reserved_eta_and_service_lifecycle():
    manager = LandingZoneManager()
    assert manager.earliest_release_time(0, now=10.0) == pytest.approx(10.0)

    assert manager.reserve(0, uav_id=11, now=10.0).approved
    assert manager.zones[0].expected_release_at is None
    assert manager.earliest_release_time(0, now=11.0) == math.inf
    assert not manager.note_committed_landing(0, uav_id=12, expected_release_at=20.0)
    assert manager.zones[0].expected_release_at is None
    with pytest.raises(ValueError, match="finite"):
        manager.note_committed_landing(0, uav_id=11, expected_release_at=math.inf)
    with pytest.raises(ValueError, match="reservation_start"):
        manager.note_committed_landing(0, uav_id=11, expected_release_at=9.0)

    assert manager.note_committed_landing(0, uav_id=11, expected_release_at=25.0)
    assert manager.earliest_release_time(0, now=11.0) == pytest.approx(25.0)
    assert manager.earliest_release_time(0, now=30.0) == pytest.approx(30.0)

    manager.touchdown(0, uav_id=11, mission=MissionType.CARGO)
    service = manager.zones[0].service_remaining
    assert manager.zones[0].expected_release_at is None
    assert manager.earliest_release_time(0, now=30.0) == pytest.approx(30.0 + service)
    assert manager.step(service) == [11]
    assert manager.zones[0].status is LZStatus.FREE
    assert manager.zones[0].expected_release_at is None
    assert manager.earliest_release_time(0, now=30.0 + service) == pytest.approx(30.0 + service)


def _inner_env_for_commit() -> tuple[TerminalParallelEnv, UAVState, int]:
    config = SimConfig(
        arrival_model="finite_cohort",
        total_requested_uavs=0,
        arrival_rate_per_second=0.0,
        max_simulation_seconds=120.0,
    )
    env = TerminalParallelEnv(config)
    request = ArrivalRequest(1, 0.0, 0, MissionType.STANDARD, 1, config.energy_capacity, 1.0)
    queued = QueuedArrivalState(request, request.energy_capacity)
    uav = UAVState.from_request(0, request, queued, np.zeros(3), env.now)
    uav.phase = FlightPhase.RING_INNER
    uav.ring_role = RingRole.INNER
    uav.ring_progress = 0.0
    ring = env.airspace.actual_ring(RingRole.INNER)
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    uav.acceleration = np.asarray((-(ring.nominal_speed ** 2) / ring.radius, 0.0, 0.0), dtype=float)
    env.active[uav.uid] = uav
    env.ledger.request(request.arrival_id)
    lz_id = env.lz_manager.choose_rule_zone(uav, env.now)
    assert lz_id is not None
    assert env.lz_manager.reserve(lz_id, uav.uid, env.now).approved
    return env, uav, lz_id


def test_environment_commit_records_landing_eta_only_after_safety_approval():
    env, uav, lz_id = _inner_env_for_commit()
    reservation = MergeReservation(
        "test-final", uav.uid, "ring_inner", "final", env.now, None, None, None,
    )
    plan = env.generator.final_approach(
        uav, env.lz_manager.zones[lz_id], reservation, env.now, airspace=env.airspace,
    )
    assert env._commit(uav, plan)
    expected = plan.end_time + env.config.dt + MISSION_SERVICE_SECONDS[uav.mission_type]
    assert env.lz_manager.zones[lz_id].expected_release_at == pytest.approx(expected)


def test_environment_failed_commit_does_not_record_landing_eta():
    env, uav, lz_id = _inner_env_for_commit()
    bad_plan = TrajectoryPlan(
        "bad-final", uav.uid, "final_approach",
        (
            TrajectoryPoint(0.0, uav.position, uav.velocity),
            TrajectoryPoint(1.0, uav.position + np.asarray((1.0, 0.0, 0.0)), uav.velocity),
        ),
        0.0, 1.0,
        metadata={"target_phase": "landing", "vehicle_profile_id": "wrong"},
    )
    assert not env._commit(uav, bad_plan)
    assert env.lz_manager.zones[lz_id].expected_release_at is None


def test_invalid_assignment_mode_is_rejected():
    with pytest.raises(ValueError, match="assignment_mode"):
        LandingZoneManager("random")
