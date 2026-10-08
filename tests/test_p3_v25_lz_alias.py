from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from terminal_marl.core.landing_zone import LandingZoneManager,LZStatus
from terminal_marl.scripts.diagnose_p3_v25_lz_alias import swap_ground_service


def test_swap_preserves_pad_geometry_and_ground_service_duration():
    manager=LandingZoneManager();manager.zones[2].status=LZStatus.OCCUPIED
    manager.zones[2].occupied_by=42;manager.zones[2].service_remaining=17.
    env=SimpleNamespace(lz_manager=manager,active={},service_uavs={42:SimpleNamespace(
        assigned_lz=2,position=manager.zones[2].position.copy())})
    positions=[z.position.copy() for z in manager.zones]
    swap_ground_service(env,0,2)
    assert manager.zones[0].occupied_by==42 and manager.zones[0].service_remaining==17.
    assert manager.zones[2].status is LZStatus.FREE and env.service_uavs[42].assigned_lz==0
    for p,z in zip(positions,manager.zones):np.testing.assert_array_equal(p,z.position)
    np.testing.assert_array_equal(env.service_uavs[42].position,manager.zones[0].position)
    with pytest.raises(AssertionError):swap_ground_service(env,0,2)
