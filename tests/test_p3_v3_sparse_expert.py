"""Sparse labels must not control the rollout or turn missing labels into WAIT."""
import json
import pickle
import subprocess

import numpy as np
import pytest

from terminal_marl.configs.p3_integrated_repair import build_config
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts import collect_p3_v3_sparse_expert as sparse


def environment():
    # Existing development fixture seed, outside the new smoke/pilot cases.
    return TerminalParallelEnv(build_config(requests=3, warm_start=3, high_soc=True), seed=892901)


def test_fixed_windows_select_first_eligible_numeric_uid_and_never_refill():
    assert sparse.selected_uid(0, [], set()) is None
    assert sparse.selected_uid(10, ['10','2','1'], set()) == '1'
    assert sparse.selected_uid(74, ['10','2','1'], {0}) is None
    assert sparse.selected_uid(80, ['10','2','1'], {0}) == '2'
    assert sparse.selected_uid(155, ['10','2','1'], {0,1}) == '10'
    assert sparse.selected_uid(1799, ['1'], set()) == '1'
    assert sparse.selected_uid(1800, ['1'], set()) is None


def test_raw_rows_keep_null_labels_and_pre_action_mask():
    env = environment()
    env.active[0].safety_retry_after = 10
    observations = env._observations()
    rows = sparse.raw_rows(env, observations, sparse.rule_tape(env), seed=892901, tick=0, pre_hash='x')
    assert len(rows) == 3 and all(r['teacher_label'] is None for r in rows)
    assert not rows[0]['actor_mask'] and rows[0]['behavior_request'] == [0.,0.,0.]
    observations['0'] = np.zeros(88)
    with pytest.raises(ValueError, match='invalid 90D'):
        sparse.raw_rows(env, observations, sparse.rule_tape(env), seed=892901, tick=0, pre_hash='x')


def test_real_rollout_is_unlabelled_and_query_matches_teacher_without_changing_files(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('teacher used during rollout')
    with monkeypatch.context() as context:
        context.setattr(sparse, 'teacher_joint_expert_v4_1', forbidden)
        flow, selected = sparse.rollout(environment(), tmp_path/'run', seed=892901,
                                        max_steps=2, wall_seconds=120, smoke=True)
    assert flow['status'] == 'SMOKE_OK', flow['error']
    assert flow['ticks'] == 2 and flow['rows'] == 6 and len(selected) == 1
    assert flow['first_tick_replay_equal'] and flow['censored']
    assert all(flow['telemetry'][key] == 0 for key in sparse.SAFETY_KEYS)
    rows_path = tmp_path/'run/rows.jsonl'
    before_rows = rows_path.read_bytes()
    rows = [json.loads(x) for x in before_rows.splitlines()]
    assert all(row['teacher_label'] is None for row in rows)
    assert sum(row['actor_mask'] for row in rows) == 3
    entry_path = tmp_path/'run/snapshots/window_00.json'
    data = (tmp_path/'run/snapshots/window_00.pkl').read_bytes()
    label = sparse.query_snapshot(entry_path)
    independent = pickle.loads(data)
    expected, audit = sparse.teacher_joint_expert_v4_1(independent, independent.active[0], state_only=True)
    assert label['teacher_joint_index'] == expected.index
    assert label['teacher_audit'] == sparse._json(audit)
    assert label['selected_certificate']['checked_hash'] == label['selected_certificate']['trajectory_hash']
    assert not label['executed_as_behavior'] and label['actor_mask']
    assert rows_path.read_bytes() == before_rows
    assert (tmp_path/'run/snapshots/window_00.pkl').read_bytes() == data
    # Exercise the real child CLI and its JSON result, not a mocked label.
    annotation = sparse.annotate(tmp_path/'run', selected, wall_seconds=120)
    assert annotation['status'] == 'COMPLETED' and annotation['labelled'] == 1
    disk = json.loads((tmp_path/'run/labels/window_00.json').read_text())
    assert disk['teacher_joint_index'] == label['teacher_joint_index']
    assert disk['teacher_audit'] == label['teacher_audit']
    assert disk['decision_id'] == rows[0]['decision_id']


def test_rollout_cap_keeps_business_censored_without_running_labels(tmp_path):
    flow, selected = sparse.rollout(environment(), tmp_path/'run', seed=892901, wall_seconds=0)
    assert flow['status'] == 'CENSORED' and flow['ticks'] == 0
    assert selected == [] and flow['unresolved'] > 0


def test_failed_execution_retains_raw_prestate_and_requests(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('injected step failure')
    monkeypatch.setattr(sparse, 'execute_tape', fail)
    flow, selected = sparse.rollout(environment(), tmp_path/'run', seed=892901, max_steps=2)
    assert flow['status'] == 'ERROR' and flow['ticks'] == 0
    assert flow['rows'] == 3 and selected
    assert (tmp_path/'run/failure_before.pkl').exists()
    rows = [json.loads(x) for x in (tmp_path/'run/rows.jsonl').read_text().splitlines()]
    assert all(r['teacher_label'] is None for r in rows)
    assert json.loads((tmp_path/'run/failure_request.json').read_text())['actions']


@pytest.mark.parametrize('timeout', [False, True])
def test_annotation_cap_retains_unlabelled_records(tmp_path, monkeypatch, timeout):
    (tmp_path/'labels').mkdir()
    selected = [dict(window=i, row={'decision_id':f'd{i}'}) for i in range(2)]
    calls = []
    def expire(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    monkeypatch.setattr(sparse.subprocess, 'run', expire)
    result = sparse.annotate(tmp_path, selected, wall_seconds=30 if timeout else 0)
    assert result['status'] == 'CENSORED' and result['labelled'] == 0
    assert len(calls) == int(timeout)
    assert [r['status'] for r in result['entries']] == (
        ['TIMEOUT','NOT_ATTEMPTED'] if timeout else ['NOT_ATTEMPTED','NOT_ATTEMPTED'])
    assert not list((tmp_path/'labels').glob('*.json'))


def test_untrusted_snapshot_rejected_before_unpickling(tmp_path):
    (tmp_path/'x.pkl').write_bytes(b'not a pickle')
    sparse.write_json(tmp_path/'entry.json', dict(snapshot='x.pkl', snapshot_sha256='wrong'))
    with pytest.raises(ValueError, match='snapshot hash differs'):
        sparse.query_snapshot(tmp_path/'entry.json')
