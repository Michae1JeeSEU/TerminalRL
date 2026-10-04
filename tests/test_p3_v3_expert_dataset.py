"""Data-only fixtures: no environment reset, teacher query, or optimizer."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from terminal_marl.training import p3_v3_expert_dataset as dataset


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


def fixture_episode(tmp_path, joints=(21, 0, 0, None, 13), *, partition='train', seed=101):
    case = tmp_path / 'case'
    episode = case / f'seed_{seed}'
    (episode / 'labels').mkdir(parents=True)
    schema = (dataset.DEVELOPMENT_SCHEMA if partition == dataset.DEVELOPMENT_PARTITION
              else dataset.DATA_SCHEMA)
    protocol = dict(schema=schema, partition=partition, seed=seed)
    write(case / 'protocol.json', protocol)
    write(episode / 'config.json', dict(observation_schema_version=dataset.OBSERVATION_SCHEMA,
        action_schema_version='low_action_schema_v4_2_hierarchical',
        trajectory_schema_version='executable_trajectory_v3'))
    rows, entries = [], []
    for index, joint in enumerate(joints):
        actor = index != 2
        row = dict(schema=schema, partition=partition, seed=seed, tick=index, uid=str(index),
            decision_id=f'{schema}:{seed}:{index}:{index}', observation_dim=90,
            observation_schema=dataset.OBSERVATION_SCHEMA, observation=(np.arange(90) / 100).tolist(),
            actor_mask=actor, teacher_label=None, pre_state_hash=f'pre_{index}',
            behavior_request=[0., 0., 0.], behavior_shortlist=[], actual_action=[0., 0., 0.])
        rows.append(row)
        if joint is None:
            continue
        proposal = joint - 1 if joint else None
        gate = int(joint > 0)
        # Independent catalogue arithmetic, not the loader's mapping helper.
        request = ([1., (-1., -.5, 0., .5, 1.)[proposal // 5],
                    (-1., -.5, 0., .5, 1.)[proposal % 5]] if gate else [0., 0., 0.])
        label = dict(status='OK', decision_id=row['decision_id'], tick=index, uid=str(index),
            window=index, actor_mask=actor, observation=row['observation'],
            pre_state_hash=row['pre_state_hash'], teacher_gate=gate, teacher_proposal=proposal,
            teacher_joint_index=joint, teacher_request=request, executed_as_behavior=False,
            selected_certificate=dict(trajectory_hash='checked', checked_hash='checked') if gate else None)
        write(episode / 'labels' / f'window_{index:02d}.json', label)
        entries.append(dict(status='OK', window=index, decision_id=row['decision_id'],
            label_file=f'labels/window_{index:02d}.json', teacher_gate=gate, teacher_proposal=proposal))
    save_rows(episode, rows)
    write(episode / 'annotation_result.json', dict(status='COMPLETED', entries=entries,
          selected=len(entries), labelled=len(entries)))
    return episode


def save_rows(episode, rows):
    (episode / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def raw_rows(episode):
    return [json.loads(line) for line in (episode / 'rows.jsonl').read_text().splitlines()]


def test_only_valid_actor_labels_enter_gate_and_release_enters_proposal(tmp_path):
    episode = fixture_episode(tmp_path)
    annotation = read(episode / 'annotation_result.json')
    annotation['entries'][-1]['status'] = 'TIMEOUT'  # Stale OK label is deliberately still on disk.
    annotation.update(status='CENSORED', labelled=3)
    write(episode / 'annotation_result.json', annotation)
    batch = dataset.load_sparse_training_episode(episode, allowed_train_seeds={101})
    assert batch.observations.shape == (2, 90) and batch.observations.dtype == np.float32
    assert batch.gate_target.tolist() == [1, 0]
    assert batch.proposal_target.tolist() == [20, -1]
    assert batch.joint_target.tolist() == [21, 0]
    assert batch.proposal_loss_mask.tolist() == [True, False]
    assert batch.gate_target.dtype == batch.proposal_target.dtype == np.int64
    assert batch.summary['raw_rows'] == 5
    assert batch.summary['actor_eligible_rows'] == 4
    assert batch.summary['valid_labels'] == 3
    assert batch.summary['unlabelled_rows'] == 2
    assert batch.summary['ignored_forced_labels'] == 1
    assert batch.summary['gate_supervision_rows'] == 2
    assert batch.summary['proposal_supervision_rows'] == 1
    # Neither the recorded behavior WAIT nor actual WAIT replaces teacher RELEASE.
    np.testing.assert_array_equal(batch.observations[0], np.arange(90, dtype=np.float32) / 100)


@pytest.mark.parametrize('status', ['TIMEOUT', 'ERROR', 'NOT_ATTEMPTED'])
def test_invalid_status_never_uses_stale_label_as_wait_or_release(tmp_path, status):
    episode = fixture_episode(tmp_path, joints=(13,))
    annotation = read(episode / 'annotation_result.json')
    annotation['entries'][0]['status'] = status
    annotation['labelled'] = 0
    write(episode / 'annotation_result.json', annotation)
    report = dataset.inspect_sparse_episode(episode, expected_seed=101, expected_partition='train')
    assert report['gate_supervision_rows'] == report['teacher_wait_rows'] == 0
    assert report['unlabelled_rows'] == 1
    with pytest.raises(ValueError, match='no labelled actor rows'):
        dataset.load_sparse_training_episode(episode, allowed_train_seeds={101})


@pytest.mark.parametrize('partition', [dataset.DEVELOPMENT_PARTITION, 'audit', 'Q'])
def test_training_rejects_other_partitions_even_with_allowed_seed(tmp_path, partition):
    episode = fixture_episode(tmp_path, partition=partition)
    with pytest.raises(ValueError, match='partition mismatch'):
        dataset.load_sparse_training_episode(episode, allowed_train_seeds={101})


def test_seed_and_schema_cannot_be_silently_reinterpreted(tmp_path):
    episode = fixture_episode(tmp_path)
    with pytest.raises(ValueError, match='outside explicitly allowed'):
        dataset.load_sparse_training_episode(episode, allowed_train_seeds={102})
    protocol = read(episode.parent / 'protocol.json')
    protocol['schema'] = dataset.DEVELOPMENT_SCHEMA
    write(episode.parent / 'protocol.json', protocol)
    with pytest.raises(ValueError, match='development schema/partition'):
        dataset.load_sparse_training_episode(episode, allowed_train_seeds={101})


@pytest.mark.parametrize('field,value,message', [
    ('observation', [0.] * 88, '90D'),
    ('observation', [float('nan')] * 90, 'finite'),
    ('observation', [1e40] * 90, 'finite'),
    ('observation_schema', 'phaseA_low_obs_v2_1_retry_state', 'schema/dimension'),
    ('actor_mask', 'False', 'actor_mask'),
    ('seed', 102, 'raw seed'),
    ('tick', .5, 'tick'),
    ('uid', '01', 'UID'),
    ('decision_id', 'wrong', 'decision_id'),
    ('partition', 'audit', 'raw schema/partition'),
    ('teacher_label', {'gate': 0}, 'must remain null'),
])
def test_raw_contract_errors_reject_instead_of_coercing(tmp_path, field, value, message):
    episode = fixture_episode(tmp_path)
    rows = raw_rows(episode)
    rows[0][field] = value
    save_rows(episode, rows)
    with pytest.raises(ValueError, match=message):
        dataset.load_sparse_training_episode(episode, allowed_train_seeds={101})


@pytest.mark.parametrize('field,value,message', [
    ('teacher_gate', 0, 'joint/gate/proposal'),
    ('teacher_proposal', 21, 'joint/gate/proposal'),
    ('teacher_joint_index', 26, 'execution index'),
    ('teacher_request', [0., 0., 0.], 'teacher request'),
    ('observation', [0.] * 90, 'observation differs'),
    ('pre_state_hash', 'another', 'pre-state'),
    ('actor_mask', False, 'actor_mask mismatch'),
    ('decision_id', 'another', 'decision_id mismatch'),
    ('selected_certificate', {'trajectory_hash': 'a', 'checked_hash': 'b'}, 'certificate'),
])
def test_bad_label_join_and_action_mapping_reject(tmp_path, field, value, message):
    episode = fixture_episode(tmp_path)
    path = episode / 'labels/window_00.json'
    label = read(path)
    label[field] = value
    write(path, label)
    with pytest.raises(ValueError, match=message):
        dataset.load_sparse_training_episode(episode, allowed_train_seeds={101})


@pytest.mark.parametrize('kind', ['raw_duplicate', 'label_duplicate', 'orphan'])
def test_duplicate_or_orphan_ids_do_not_weight_or_attach_wrong_sample(tmp_path, kind):
    episode = fixture_episode(tmp_path)
    if kind == 'raw_duplicate':
        rows = raw_rows(episode)
        save_rows(episode, rows + [copy.deepcopy(rows[0])])
    else:
        path = episode / 'annotation_result.json'
        annotation = read(path)
        if kind == 'label_duplicate':
            annotation['entries'].append(copy.deepcopy(annotation['entries'][0]))
            annotation['selected'] += 1
        else:
            annotation['entries'][0]['decision_id'] = 'orphan'
        write(path, annotation)
    with pytest.raises(ValueError, match='duplicate|orphan'):
        dataset.load_sparse_training_episode(episode, allowed_train_seeds={101})


def test_all_26_joint_actions_keep_zero_based_proposal_indices(tmp_path):
    episode = fixture_episode(tmp_path, joints=tuple(range(26)))
    rows = raw_rows(episode)
    rows[2]['actor_mask'] = True
    save_rows(episode, rows)
    path = episode / 'labels/window_02.json'
    label = read(path)
    label['actor_mask'] = True
    write(path, label)
    batch = dataset.load_sparse_training_episode(episode, allowed_train_seeds={101})
    assert batch.joint_target.tolist() == list(range(26))
    assert batch.gate_target.tolist() == [0] + [1] * 25
    assert batch.proposal_target.tolist() == [-1] + list(range(25))
    assert batch.proposal_loss_mask.sum() == 25


def test_counts_cli_is_serializable_and_does_not_write_inputs_or_reset(tmp_path, monkeypatch):
    from terminal_marl.envs.parallel_env import TerminalParallelEnv
    def forbidden(*args, **kwargs):
        raise AssertionError('data ingestion must not construct an environment')
    monkeypatch.setattr(TerminalParallelEnv, '__init__', forbidden)
    episode = fixture_episode(tmp_path, partition=dataset.DEVELOPMENT_PARTITION)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in episode.parent.rglob('*') if p.is_file()}
    report = dataset.inspect_sparse_episode(episode, expected_seed=101)
    result = subprocess.run([sys.executable, '-B', '-m', 'terminal_marl.training.p3_v3_expert_dataset',
        '--episode', str(episode), '--seed', '101'], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == report
    assert report['gate_supervision_rows'] == 3
    assert report['proposal_supervision_rows'] == 2
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in episode.parent.rglob('*') if p.is_file()} == before
