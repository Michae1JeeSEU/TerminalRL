"""Join V3/90D sparse teacher labels without importing a learner.

The CLI and inspect_sparse_episode return counts only. Development records
cannot be returned by the training entry point. The non-development wire
schema below is a data contract, not an I0 qualification or a seed protocol.
No environment, teacher, model, optimizer or checkpoint is created here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Collection

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import (
    ACTION_SCHEMA_V4_2, hierarchical_from_joint_index_v4_1,
)

DATA_SCHEMA = 'p3_v3_90d_sparse_expert_v1'
DEVELOPMENT_SCHEMA = 'p3_v3_90d_sparse_expert_development_v1'
DEVELOPMENT_PARTITION = 'development_pilot_never_train'
OBSERVATION_SCHEMA = 'phaseA_low_obs_v2_4_service_context_90d'
OBSERVATION_DIM = 90
PARTITIONS = {DEVELOPMENT_PARTITION, 'train', 'audit', 'Q'}
LABEL_STATUSES = {'OK', 'TIMEOUT', 'ERROR', 'NOT_ATTEMPTED'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f'invalid {name}')
    return value


def _boolean(value, name):
    _require(type(value) is bool, f'invalid {name}')
    return value


def _observation(value):
    values = np.asarray(value)
    _require(values.shape == (OBSERVATION_DIM,) and values.dtype.kind in 'fiu',
             'observation must be a numeric 90D vector')
    with np.errstate(over='ignore', invalid='ignore'):
        values = values.astype(np.float32)
    _require(np.isfinite(values).all(), 'observation must be finite in float32')
    return values


def _read(path):
    return json.loads(path.read_text(encoding='utf8'))


@dataclass(frozen=True)
class V3ExpertBatch:
    """Only labelled actor rows; WAIT has proposal_target=-1, never zero.

    Every returned row contributes to gate supervision. Only rows selected
    by proposal_loss_mask contribute to proposal supervision. Counts include
    excluded raw rows so a caller cannot confuse rollout size with sample size.
    """
    observations: np.ndarray
    gate_target: np.ndarray
    proposal_target: np.ndarray
    joint_target: np.ndarray
    decision_ids: tuple[str, ...]
    summary: dict

    @property
    def proposal_loss_mask(self):
        return self.gate_target == 1


def _join_episode(episode, *, expected_partition, allowed_seeds, training):
    episode = Path(episode)
    _require(expected_partition in PARTITIONS, 'unknown expected partition')
    seeds = {_integer(seed, 'allowed seed') for seed in allowed_seeds}
    _require(bool(seeds), 'explicit allowed seeds are required')
    protocol = _read(episode.parent / 'protocol.json')
    partition, schema = protocol['partition'], protocol['schema']
    _require(partition == expected_partition, 'protocol partition mismatch')
    _require(schema in {DATA_SCHEMA, DEVELOPMENT_SCHEMA}, 'unsupported dataset schema')
    _require((schema == DEVELOPMENT_SCHEMA) == (partition == DEVELOPMENT_PARTITION),
             'development schema/partition mismatch')
    if training:
        _require(partition == 'train' and schema == DATA_SCHEMA, 'training requires train partition')
    seed = _integer(protocol['seed'], 'protocol seed')
    _require(seed in seeds, 'seed outside explicitly allowed partition seeds')
    config = _read(episode / 'config.json')
    _require(config['observation_schema_version'] == OBSERVATION_SCHEMA,
             'config observation schema mismatch')
    _require(config['action_schema_version'] == ACTION_SCHEMA_V4_2,
             'config action schema mismatch')
    _require(config['trajectory_schema_version'] == 'executable_trajectory_v3',
             'config trajectory schema mismatch')

    rows, observations, by_id = [], [], {}
    with (episode / 'rows.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            _require(row['schema'] == schema and row['partition'] == partition,
                     'raw schema/partition mismatch')
            _require(_integer(row['seed'], 'row seed') == seed, 'raw seed mismatch')
            tick = _integer(row['tick'], 'tick')
            uid = row['uid']
            _require(isinstance(uid, str) and uid.isascii() and uid.isdecimal()
                     and str(int(uid)) == uid, 'UID must be a canonical numeric string')
            decision_id = f'{schema}:{seed}:{tick}:{uid}'
            _require(row['decision_id'] == decision_id, 'decision_id does not match row indices')
            _require(decision_id not in by_id, 'duplicate decision_id')
            _require(_integer(row['observation_dim'], 'observation_dim') == OBSERVATION_DIM
                     and row['observation_schema'] == OBSERVATION_SCHEMA,
                     'raw observation schema/dimension mismatch')
            _boolean(row['actor_mask'], 'actor_mask')
            _require(row['teacher_label'] is None, 'raw teacher_label must remain null')
            _require(isinstance(row['pre_state_hash'], str) and bool(row['pre_state_hash']),
                     'missing pre-state identity')
            by_id[decision_id] = len(rows)
            observations.append(_observation(row['observation']))
            rows.append(row)

    # Preserve the full raw row order while constructing masks. Missing labels
    # use -1; they are never assigned a WAIT gate or a proposal target.
    size = len(rows)
    gates = np.full(size, -1, dtype=np.int64)
    proposals = np.full(size, -1, dtype=np.int64)
    joint = np.full(size, -1, dtype=np.int64)
    valid = np.zeros(size, dtype=bool)
    actor = np.asarray([row['actor_mask'] for row in rows], dtype=bool)
    annotation = _read(episode / 'annotation_result.json')
    seen, statuses = set(), Counter()
    entries = annotation['entries']
    _require(_integer(annotation['selected'], 'selected count') == len(entries),
             'annotation selected count mismatch')
    for item in entries:
        decision_id, status = item['decision_id'], item['status']
        _require(decision_id in by_id, 'orphan annotation decision_id')
        _require(decision_id not in seen, 'duplicate annotation decision_id')
        _require(status in LABEL_STATUSES, 'unknown annotation status')
        seen.add(decision_id)
        statuses[status] += 1
        if status != 'OK':
            # Even a stale label file cannot upgrade TIMEOUT/ERROR to a target.
            continue
        index = by_id[decision_id]
        row = rows[index]
        window = _integer(item['window'], 'window')
        label_file = f'labels/window_{window:02d}.json'
        _require(item['label_file'] == label_file, 'unexpected label_file')
        label = _read(episode / label_file)
        _require(label['status'] == 'OK' and label['decision_id'] == decision_id,
                 'label status/decision_id mismatch')
        _require(_integer(label['tick'], 'label tick') == row['tick'] and label['uid'] == row['uid']
                 and _integer(label['window'], 'label window') == window,
                 'label indices mismatch')
        _require(_boolean(label['actor_mask'], 'label actor_mask') == row['actor_mask'],
                 'label actor_mask mismatch')
        _require(label['pre_state_hash'] == row['pre_state_hash'], 'label pre-state mismatch')
        _require(np.array_equal(_observation(label['observation']), observations[index]),
                 'label observation differs from raw observation')
        _require(label['executed_as_behavior'] is False, 'teacher label is not behavior execution')
        j = _integer(label['teacher_joint_index'], 'teacher joint index')
        action = hierarchical_from_joint_index_v4_1(j)
        gate = _integer(label['teacher_gate'], 'teacher gate')
        proposal = label['teacher_proposal']
        if proposal is not None:
            _integer(proposal, 'teacher proposal')
        _require(gate == action.gate and proposal == action.proposal, 'joint/gate/proposal mismatch')
        request = np.asarray(label['teacher_request'])
        _require(request.shape == (3,) and request.dtype.kind in 'fiu'
                 and np.array_equal(request, action.normalized), 'teacher request mismatch')
        _require(item['teacher_gate'] == gate and item['teacher_proposal'] == proposal,
                 'annotation target differs from label')
        certificate = label['selected_certificate']
        if gate == 1:
            _require(isinstance(certificate, dict) and bool(certificate.get('trajectory_hash'))
                     and certificate.get('checked_hash') == certificate['trajectory_hash'],
                     'RELEASE lacks matching checked certificate')
        else:
            _require(certificate is None, 'WAIT must not have a RELEASE certificate')
        valid[index], gates[index], joint[index] = True, gate, j
        proposals[index] = -1 if proposal is None else proposal
    _require(_integer(annotation['labelled'], 'labelled count') == int(valid.sum()),
             'annotation labelled count mismatch')
    gate_mask = valid & actor
    proposal_mask = gate_mask & (gates == 1)
    summary = dict(schema=schema, partition=partition, seed=seed,
        raw_rows=size, actor_eligible_rows=int(actor.sum()), forced_wait_rows=int((~actor).sum()),
        valid_labels=int(valid.sum()), unlabelled_rows=int((~valid).sum()),
        ignored_forced_labels=int((valid & ~actor).sum()),
        gate_supervision_rows=int(gate_mask.sum()), proposal_supervision_rows=int(proposal_mask.sum()),
        teacher_wait_rows=int((gate_mask & (gates == 0)).sum()),
        teacher_release_rows=int(proposal_mask.sum()), annotation_status_counts=dict(statuses),
        proposal_counts={str(k): v for k, v in sorted(Counter(proposals[proposal_mask].tolist()).items())})
    # A future BC caller gets only actual supervision, not forced or unknown
    # targets. Keep an empty batch shaped (0, 90) for incomplete data inspection.
    x = np.asarray(observations, dtype=np.float32).reshape(size, OBSERVATION_DIM)
    return V3ExpertBatch(x[gate_mask], gates[gate_mask], proposals[gate_mask], joint[gate_mask],
        tuple(row['decision_id'] for row, keep in zip(rows, gate_mask) if keep), summary)


def inspect_sparse_episode(episode, *, expected_seed, expected_partition=DEVELOPMENT_PARTITION):
    """Read-only counts; does not expose development arrays to a trainer."""
    return _join_episode(episode, expected_partition=expected_partition,
                         allowed_seeds=(expected_seed,), training=False).summary


def load_sparse_training_episode(episode, *, allowed_train_seeds: Collection[int]):
    """Require an explicitly frozen train seed set and an actual train partition.

    Passing this structural check does not establish I0 coverage/qualification.
    Audit and Q need separate, explicitly authorized evaluation entry points.
    """
    batch = _join_episode(episode, expected_partition='train',
                          allowed_seeds=allowed_train_seeds, training=True)
    _require(bool(batch.decision_ids), 'no labelled actor rows for training')
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--partition', choices=sorted(PARTITIONS), default=DEVELOPMENT_PARTITION)
    args = parser.parse_args()
    print(json.dumps(inspect_sparse_episode(args.episode, expected_seed=args.seed,
                     expected_partition=args.partition), allow_nan=False, sort_keys=True))


if __name__ == '__main__':
    main()
