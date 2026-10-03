"""Fresh expert labels must follow V3 physics and remain separate from execution."""
import json
import pickle

import numpy as np
import pytest

from terminal_marl.configs.p3_integrated_repair import build_config
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts import collect_p3_v3_expert as collector


def environment():
    return TerminalParallelEnv(build_config(requests=3, warm_start=3, high_soc=True), seed=892901)


def test_real_labels_are_90d_direct_c2_and_query_does_not_pollute_live_env():
    env = environment()
    observations = env._observations()
    data = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
    rows, tape = collector.labels_from_snapshot(data, observations, seed=892901, tick=0, repeat=True)
    assert pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL) == data
    assert len(rows) == 3
    assert set(tape['actions']) == set(observations)
    assert {r['interface'] for r in rows} == {'o2m','m2i','i2f'}
    assert all(r['teacher_gate'] == 1 and r['teacher_repeat_checked'] for r in rows)
    for row in rows:
        assert len(row['observation']) == 90 and row['observation_dim'] == 90
        assert row['teacher_proposal'] == row['teacher_joint_index']-1
        cert = row['selected_certificate']
        assert cert['checked_hash'] == cert['trajectory_hash']
        assert cert['mode'] == ('final_approach' if row['interface']=='i2f' else 'ring_transfer')
        assert tape['shortlists'][row['uid']] == [row['requested_action']]
    assert env.config.v3_landing_path_mode == 'direct_c2'


def test_retry_is_retained_forced_wait_with_null_proposal():
    env = environment()
    env.active[0].safety_retry_after = 10
    obs = env._observations()
    rows, _ = collector.labels_from_snapshot(pickle.dumps(env), obs, seed=892901, tick=0)
    row = next(r for r in rows if r['uid']=='0')
    assert not row['actor_mask'] and row['retry_active']
    assert row['teacher_gate'] == 0 and row['teacher_proposal'] is None
    assert row['selected_certificate'] is None


def test_smoke_preserves_every_row_and_checks_replay(tmp_path):
    result = collector.collect(environment(), tmp_path/'run', seed=892901,
                               max_steps=2, wall_seconds=120, smoke=True)
    assert result['status'] == 'SMOKE_OK', result['error']
    rows = [json.loads(x) for x in (tmp_path/'run/rows.jsonl').read_text().splitlines()]
    steps = [json.loads(x) for x in (tmp_path/'run/steps.jsonl').read_text().splitlines()]
    assert result['ticks'] == len(steps) == 2
    assert result['rows'] == len(rows) == 6
    assert len({r['decision_id'] for r in rows}) == 6
    assert result['row_counts'] == {'RELEASE':3, 'forced_WAIT':3}
    assert result['first_tick_replay_equal']
    assert not result['terminal'] and result['censored']
    assert not result['I0_QUALIFIED'] and not result['PPO_ALLOWED']
    assert not result['trained'] and not result['checkpoint_loaded']
    assert all(result['telemetry'][k] == 0 for k in collector.SAFETY_KEYS)
    assert json.loads((tmp_path/'run/result.json').read_text()) == result


def test_failure_keeps_pre_step_rows_and_state(tmp_path, monkeypatch):
    def broken(*args):
        raise RuntimeError('injected execution error')
    monkeypatch.setattr(collector, 'execute_tape', broken)
    result = collector.collect(environment(), tmp_path/'run', seed=892901, max_steps=2)
    assert result['status'] == 'ERROR'
    assert result['ticks'] == 0 and result['rows'] == 3
    assert 'injected execution error' in result['error']['message']
    assert (tmp_path/'run/failure_before.pkl').exists()
    assert len((tmp_path/'run/rows.jsonl').read_text().splitlines()) == 3


def test_wall_cap_cannot_be_mistaken_for_success(tmp_path):
    result = collector.collect(environment(), tmp_path/'run', seed=892901, wall_seconds=0)
    assert result['status'] == 'CENSORED' and result['ticks'] == 0
    assert result['stop_reason'] == 'wall_limit' and result['censored']


def test_wrong_width_fails_before_labels_are_produced():
    env = environment()
    observations = env._observations()
    observations['0'] = np.zeros(88, dtype=np.float32)
    with pytest.raises(ValueError, match='invalid 90D'):
        collector.labels_from_snapshot(pickle.dumps(env), observations, seed=892901, tick=0)
