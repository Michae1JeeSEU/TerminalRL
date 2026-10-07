import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from terminal_marl.scripts import diagnose_p3_teacher_context as d


def label_at(rank):
    catalogue = d.STATE_ONLY_JOINT_CATALOGUE_V4_1
    release = catalogue[rank].index > 0
    return dict(teacher_audit=[dict(rank=i, joint_index=c.index, decoded=c.as_dict(),
        feasible=i == rank, reason=('certified' if release else 'wait_legal') if i == rank else 'predicted_separation')
        for i, c in enumerate(catalogue[:rank+1])], teacher_gate=int(release),
        teacher_joint_index=catalogue[rank].index, release_intents_checked=min(25, rank+1),
        candidate_plan_calls=min(25, rank+1),
        selected_certificate=dict(trajectory_hash='abc', checked_hash='abc') if release else None)


@pytest.mark.parametrize('rank', [0, 4, 25])
def test_prefix_accepts_release_and_exhaustive_wait(rank):
    assert d.audit_prefix(label_at(rank))['selected_rank'] == rank


@pytest.mark.parametrize('corruption', ['order', 'earlier_feasible', 'certificate', 'selected'])
def test_prefix_rejects_broken_label(corruption):
    label = label_at(4)
    if corruption == 'order': label['teacher_audit'][0]['joint_index'] = 0
    if corruption == 'earlier_feasible': label['teacher_audit'][0]['feasible'] = True
    if corruption == 'certificate': label['selected_certificate']['checked_hash'] = 'bad'
    if corruption == 'selected': label['teacher_joint_index'] = 1
    with pytest.raises(AssertionError): d.audit_prefix(label)


def test_trace_preserves_native_arrays_and_observes_blocker():
    point, swept = np.array([3., 2.]), np.array([1.])
    original = lambda *a: (point, swept)
    safety = NS(_track_distances_with_margin=original, _tube_distances_with_margin=lambda *a: np.array([4., 5., 3.]))
    trace = d.NativeWitness(safety)
    trace.install()
    def filter_plan():
        other_id = 7
        post_samples = np.array([2., 3.])
        return safety._track_distances_with_margin(None, None, post_samples, 'flight_v3', 'flight_v3')
    result = filter_plan()
    assert result[0] is point and result[1] is swept
    assert trace.events[0]['uid'] == 7 and trace.events[0]['stage'] == 'post'
    assert trace.events[0]['point']['time'] == 3.
    trace.restore()
    assert safety._track_distances_with_margin is original


def test_output_is_exclusive_and_json_finite(tmp_path):
    output = tmp_path/'result.json'
    d.write(output, dict(status='test', point=d.minimum([3., 1., 4.], [0., 1., 2.]), vector=np.array([1., 2.])))
    assert json.loads(output.read_text())['point']['distance'] == 1.
    with pytest.raises(FileExistsError): d.write(output, {})


def test_dense_grid_includes_endpoint():
    result = d.grid(2., 2.11, .05)
    assert result[0] == 2. and result[-1] == 2.11
    assert np.max(np.diff(result)) <= .05+1e-9


def test_state_check_allows_only_known_candidate_counters():
    before = dict(position=[1., 2., 3.], telemetry=dict(repair_candidates_checked=2))
    after = dict(position=[1., 2., 3.], telemetry=dict(repair_candidates_checked=3))
    assert d.state_delta(before, after) == {'repair_candidates_checked': 1}
    with pytest.raises(AssertionError): d.state_delta(before, dict(after, position=[2., 2., 3.]))
    with pytest.raises(AssertionError): d.state_delta(before, dict(after, telemetry=dict(collisions=1)))


def test_context_reports_relative_gap_without_changing_historical_wire_value():
    middle, inner = d.RingRole.MIDDLE, d.RingRole.INNER
    rings = {middle: NS(radius=400., height=70., nominal_speed=20.),
             inner: NS(radius=200., height=40., nominal_speed=16.)}
    ego = NS(uid=4, ring_role=middle, ring_progress=np.pi, position=np.array([-400., 0., 70.]),
             velocity=np.array([0., -20., 0.]), dynamic_priority=0., committed_plan=None, merge_reservation_id=None)
    other = NS(uid=5, ring_role=inner, ring_progress=0., position=np.array([200., 0., 40.]),
               velocity=np.array([0., 16., 0.]), dynamic_priority=0., committed_plan=None, merge_reservation_id=None)
    env = NS(airspace=NS(actual_ring=rings.__getitem__, speed_profile_bias=0.), now=0., active={4: ego, 5: other},
             config=NS(shield_horizon_seconds=30., safe_distance=80., max_speed=25., priority_feature_scale=5.))
    result = d.observation_context(env, ego)
    predecessor, successor = result['target_ring_predecessor'], result['target_ring_successor']
    assert predecessor['path_gap'] == pytest.approx(296.)
    assert predecessor['encoded'][1] == pytest.approx(.925)
    assert predecessor['relative_path_gap'] == pytest.approx(200.*np.pi)
    assert successor['relative_path_gap'] == pytest.approx(-200.*np.pi)
    assert predecessor['relative_gap_encoded'] == 1.
