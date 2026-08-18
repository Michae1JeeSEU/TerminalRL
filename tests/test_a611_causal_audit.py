from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.ring_capacity import geometric_ring_capacity
from terminal_marl.training.a611_causal_audit import (
    _classify_ring,
    _decomposition,
    demand_profiles,
    static_expanded_rings,
)


def _row(completed, waiting):
    return {"metrics": {"completed": completed, "all_request_waiting_area": waiting}}


def test_a611_static_geometry_is_a_real_derived_capacity_plus_one():
    config = SimConfig()
    for role in RingRole:
        rings = static_expanded_rings(config, role)
        assert geometric_ring_capacity(rings[role].radius, config.safe_distance) == geometric_ring_capacity(config.rings[role].radius, config.safe_distance) + 1


def test_a611_profiles_cover_geometry_dwell_timescales():
    profiles = demand_profiles(.02)
    assert profiles["short_burst"].segments[1][1] - profiles["short_burst"].segments[1][0] < 600
    assert profiles["medium_burst"].segments[1][1] - profiles["medium_burst"].segments[1][0] == 600
    assert profiles["long_burst"].segments[1][1] - profiles["long_burst"].segments[1][0] > 1200


def test_a611_static_dynamic_decomposition_categories():
    nominal = _row(10, 1000.)
    assert _decomposition(nominal, _row(10, 995.), _row(10, 995.)) == "CAPACITY_NOT_BINDING"
    assert _decomposition(nominal, _row(11, 900.), _row(10, 1000.)) == "RECONFIGURATION_COST_OFFSETS_GAIN"
    assert _decomposition(nominal, _row(11, 900.), _row(11, 900.)) == "CAPACITY_VALUE_PROVEN"


def test_a611_ring_classification_reads_authoritative_interface_container():
    row = {"role": "outer", "decomposition": "CAPACITY_NOT_BINDING", "conservative_rejection_rate": 0.,
           "nominal": {"interfaces": {"interfaces": {"o2m": {"safety_blocked_count": 0, "merge_gap_blocked_count": 0, "capacity_blocked_release_count": 0}}}}}
    assert _classify_ring(row) == "CAPACITY_NOT_BINDING"
