"""SAFE fragment reuse retains the original continuous geometry decisions."""
from dataclasses import replace
import math

import pytest

from terminal_marl.core.rotating_slots import geometry_evidence as ge, safety as sf, trajectory as tr
from terminal_marl.core.rotating_slots.config import Section, load_config
from terminal_marl.core.rotating_slots.world import _Traffic


def orbit(name, radius, start=0., end=10., turn=-3.5):
    c = load_config()
    return tr.TrajectoryPlan(name, 'INBOUND', start, end,
        profile_hash=c.profile_hash, config_hash=c.config_hash,
        segments=(tr.Segment(start, end-start, (radius,), (0., turn), (20.,)),))


@pytest.fixture(autouse=True)
def cold_cache():
    sf.clear_geometry_evidence_cache()
    yield
    sf.clear_geometry_evidence_cache()


def test_same_immutable_fragment_reuses_safe_continuous_proof():
    a, b = orbit('a', 100.), orbit('b', 200.)
    original = sf.validate_separation_original(a, b, 0., 10.)
    first = sf.validate_separation(a, b, 0., 10.)
    before = sf.geometry_evidence_counts()
    # A serialized exact copy and a wrapper with a different enclosing horizon
    # describe the same fragment. Neither changes the encoded geometry.
    copy = tr.TrajectoryPlan.from_dict(b.to_dict())
    second = sf.validate_separation(a, _Traffic('b', copy, (), -5., 15.), 0., 10.)
    after = sf.geometry_evidence_counts()
    assert original.accepted and first.accepted and second.accepted
    assert before['fine'] > 0 and after['fine'] == before['fine']
    assert after['hit'] > before['hit']
    assert after['compute'] == before['compute']
    assert first.details['lower_bound_m'] == second.details['lower_bound_m']
    assert second.details['checked_until'] == 10.


def test_same_digest_with_different_payload_is_rejected_by_constructor():
    plan = orbit('a', 100.)
    data = plan.to_dict()
    data['segments'][0]['r'][0] = 200.
    with pytest.raises(ValueError, match='trajectory hash mismatch'):
        tr.TrajectoryPlan.from_dict(data)


def test_complete_config_contents_and_numeric_parameters_invalidate():
    a, b = orbit('a', 100.), orbit('b', 200.)
    c = load_config()
    assert sf.validate_separation(a, b, 0., 10., c).accepted
    before = sf.geometry_evidence_counts()
    # Deliberately retain all supplied digest strings to test full-content
    # identity, independently of the normal config loader's drift rejection.
    payload = c.to_dict()
    payload['safety']['hard_separation_m'] = 110.
    changed = replace(c, sections=Section(payload))
    assert not sf.validate_separation(a, b, 0., 10., changed).accepted
    after = sf.geometry_evidence_counts()
    assert after['hit'] == before['hit'] and after['miss'] > before['miss']
    before = after
    assert sf.validate_separation(a, b, 0., 10., c, max_depth=31, min_interval=2e-7).accepted
    after = sf.geometry_evidence_counts()
    assert after['hit'] == before['hit'] and after['compute'] > before['compute']


def test_absolute_time_interval_invalidate():
    a, b = orbit('a', 100.), orbit('b', 200.)
    assert sf.validate_separation(a, b, 0., 10.).accepted
    before = sf.geometry_evidence_counts()
    assert sf.validate_separation(a, b, .125, 9.875).accepted
    after = sf.geometry_evidence_counts()
    assert after['hit'] == before['hit'] and after['miss'] > before['miss']


def test_future_background_crossing_and_different_prefix_tail_are_checked():
    a, b = orbit('a', 100.), orbit('background', 200.)
    quiet = _Traffic('background', b, (), 0., 10.)
    assert sf.validate_separation(a, quiet, 0., 10.).accepted
    # The untouched background has a future commitment, rather than being a
    # newly selected candidate. The warm prefix cannot stand in for this tail.
    crossing = replace(orbit('background', 100., 5., 10.), trajectory_hash='',
        segments=(tr.Segment(5., 5., (100.,), (-1.75, -1.75), (20.,)),))
    future = _Traffic('background', b, (crossing,), 0., 10.)
    warm = sf.validate_separation(a, future, 0., 10.)
    sf.clear_geometry_evidence_cache()
    cold = sf.validate_separation(a, future, 0., 10.)
    original = sf.validate_separation_original(a, future, 0., 10.)
    assert not warm.accepted and not cold.accepted and not original.accepted
    assert warm.status == cold.status == original.status == 'UNSAFE'


def test_switch_at_fragment_right_endpoint_has_distinct_evidence():
    a, b = orbit('a', 100.), orbit('b', 200.)
    assert sf.validate_separation(a, _Traffic('b', b, (), 0., 10.), 0., 10.).accepted
    before = sf.geometry_evidence_counts()
    # It begins exactly at the horizon endpoint and remains safely separated.
    # Its identity must nevertheless invalidate the final closed fragment.
    tail = orbit('b', 300., 10., 20.)
    assert sf.validate_separation(a, _Traffic('b', b, (tail,), 0., 10.), 0., 10.).accepted
    after = sf.geometry_evidence_counts()
    assert after['hit'] - before['hit'] == 7
    assert after['miss'] - before['miss'] == 1


def test_only_entirely_canonical_stable_fragments_use_existing_analytic_proof():
    a = tr.make_stable_plan('a', 'outer', 0)
    b = tr.make_stable_plan('b', 'outer', 1)
    future = tr.make_stable_plan('b', 'outer', 2, 5., 20.)
    traffic = _Traffic('b', b, (future,), 0., 20.)
    result = sf.validate_separation(a, traffic, 0., 20.)
    assert result.accepted
    counts = sf.geometry_evidence_counts()
    assert counts['analytic'] > 0
    # The fragment ending at the switch cannot use a single-plan stable proof.
    assert counts['fine'] > 0


def test_far_endpoints_never_prove_safe_midinterval_crossing():
    c = load_config()
    a = tr.TrajectoryPlan('a', 'INBOUND', 0., 10., profile_hash=c.profile_hash,
        config_hash=c.config_hash, segments=(tr.Segment(0., 10., (200., -400.), (0.,), (20.,)),))
    b = orbit('b', 0., turn=0.)
    assert math.dist(a.state(0.).p, b.state(0.).p) > 80.
    assert math.dist(a.state(10.).p, b.state(10.).p) > 80.
    for _ in range(2):
        result = sf.validate_separation(a, b, 0., 10.)
        assert not result.accepted and result.status == 'UNSAFE'


def test_unknown_is_not_stored_as_safe():
    a, b = orbit('a', 100.), orbit('b', 180.)
    first = sf.validate_separation(a, b, 0., 10., max_depth=0)
    before = sf.geometry_evidence_counts()
    second = sf.validate_separation(a, b, 0., 10., max_depth=0)
    after = sf.geometry_evidence_counts()
    assert first.status == second.status == 'UNKNOWN'
    assert not first.accepted and not second.accepted
    assert before['store'] == after['store'] == 0
    assert after['fine'] > before['fine'] and after['hit'] == 0


def test_unknown_trajectory_implementation_bypasses_cache():
    class Proxy:
        def __init__(self, plan):
            self.plan = plan
        def state(self, t):
            return self.plan.state(t)
        def bounds(self, a, b, derivative=0):
            return self.plan.bounds(a, b, derivative)
        @property
        def segments(self):
            return self.plan.segments

    a, b = Proxy(orbit('a', 100.)), orbit('b', 200.)
    assert sf.validate_separation(a, b, 0., 10.).accepted
    assert sf.validate_separation(a, b, 0., 10.).accepted
    assert sf.geometry_evidence_counts()['bypass'] > 0
    assert sf.geometry_evidence_cache_stats()['entries'] == 0


def test_cache_is_bounded_and_clear_resets_evidence_and_counts(monkeypatch):
    monkeypatch.setattr(ge, '_MAX_ENTRIES', 2)
    a, b = orbit('a', 100.), orbit('b', 200.)
    assert sf.validate_separation(a, b, 0., 10.).accepted
    stats = sf.geometry_evidence_cache_stats()
    assert stats['entries'] <= 2 and stats['eviction'] > 0
    assert stats['estimated_bytes'] <= stats['max_bytes']
    sf.clear_geometry_evidence_cache()
    assert sf.geometry_evidence_cache_stats()['entries'] == 0
    assert sf.geometry_evidence_cache_stats()['estimated_bytes'] == 0
    assert not any(sf.geometry_evidence_counts().values())
