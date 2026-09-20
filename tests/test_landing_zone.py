import numpy as np
import pytest

from terminal_marl.core.landing_zone import LZStatus, LandingZoneManager
from terminal_marl.core.uav import FlightPhase, MissionType, UAVState


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


def test_invalid_assignment_mode_is_rejected():
    with pytest.raises(ValueError, match="assignment_mode"):
        LandingZoneManager("random")
