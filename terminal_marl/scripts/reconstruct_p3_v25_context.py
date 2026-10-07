"""Rebuild observations from original train pre-states without changing labels."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import time

import numpy as np

from ..core.observations import LOW_OBS_V2_4, build_low_observation_v2_4
from ..core.safety_trace import _digest, _json
from ..core.trajectory_context_observation import (VERSION, EGO_SCHEMA, TRAFFIC_SCHEMA,
    FORECAST_SECONDS, TrajectoryContextCache, build_low_observation_v2_5)
from ..training.p3_v25_context_dataset import DATA_SCHEMA, decode_row, load_context_dataset
from ..training.p3_v3_expert_dataset import load_sparse_training_episode
from .diagnose_p3_teacher_context import CASES, read, sha
from .diagnose_p3_direct_contract import canonical, state_record

PROBE = {(0, '4'), (92, '0'), (101, '0'), (152, '6')}


def dump(path, value):
    path.write_text(json.dumps(_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def run(root, output, mode, wall_seconds):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counts, inputs, probe_rows = Counter(), [], []
    dump(output/'protocol.json', dict(schema=DATA_SCHEMA, observation_schema=VERSION,
        mode=mode, cases=CASES, forecast_seconds=FORECAST_SECONDS, wall_seconds=wall_seconds,
        ego_features=[(f.name, f.size) for f in EGO_SCHEMA.features],
        traffic_features=[(f.name, f.size) for f in TRAFFIC_SCHEMA.features],
        environment_steps=0, environment_resets=0, teacher_queries=0, parameter_updates=0))
    with (output/'observations.jsonl').open('x', encoding='utf8') as stream:
        for seed, relative, expected in (CASES[:1] if mode == 'probe' else CASES):
            episode = root/relative/f'seed_{seed}'
            batch = load_sparse_training_episode(episode, allowed_train_seeds=(seed,))
            assert len(batch.decision_ids) == expected
            label_by_id = {did: (int(gate), int(proposal)) for did, gate, proposal
                in zip(batch.decision_ids, batch.gate_target, batch.proposal_target)}
            entries = [read(episode/'snapshots'/f"window_{e['window']:02d}.json")
                       for e in read(episode/'annotation_result.json')['entries']]
            if mode == 'probe':
                entries = [e for e in entries if (e['row']['tick'], e['row']['uid']) in PROBE]
            # Original actor order is tick/UID; retain it and reuse each shared
            # immutable snapshot while the adjacent rows belong to that tick.
            assert [(e['row']['tick'], int(e['row']['uid'])) for e in entries] == sorted(
                (e['row']['tick'], int(e['row']['uid'])) for e in entries)
            snapshot_name, env, cache = None, None, None
            snapshot_hashes = {}
            for entry in entries:
                if time.monotonic()-started > wall_seconds:
                    raise TimeoutError('reconstruction wall budget exhausted')
                row = entry['row']
                if entry['snapshot'] != snapshot_name:
                    assert Path(entry['snapshot']).name == entry['snapshot']
                    path = episode/'snapshots'/entry['snapshot']
                    digest = sha(path)
                    assert digest == entry['snapshot_sha256']
                    env = pickle.loads(path.read_bytes())
                    before = canonical(state_record(env))
                    assert _digest(before) == row['pre_state_hash']
                    cache = TrajectoryContextCache(env)
                    snapshot_name = entry['snapshot']
                    snapshot_hashes[snapshot_name] = digest
                    counts['snapshots'] += 1
                assert entry['snapshot_sha256'] == snapshot_hashes[snapshot_name]
                assert _digest(before) == row['pre_state_hash']
                ego = env.active[int(row['uid'])]
                original_obs = np.asarray(row['observation'], dtype=np.float32)
                assert np.array_equal(build_low_observation_v2_4(ego, env), original_obs)
                value = cache.build(ego)
                preserved = np.ones(90, dtype=bool)
                for name in ('target_ring_predecessor', 'target_ring_successor'):
                    preserved[LOW_OBS_V2_4.slices[name]] = False
                assert np.array_equal(value.ego[:90][preserved], original_obs[preserved])
                assert set(value.traffic_uids) == set(env.active)-{ego.uid}
                assert canonical(state_record(env)) == before
                assert np.array_equal(build_low_observation_v2_4(ego, env), original_obs)
                gate, proposal = label_by_id[row['decision_id']]
                rebuilt = dict(schema=DATA_SCHEMA, observation_schema=VERSION, partition='train',
                    ego_feature_hash=EGO_SCHEMA.feature_hash, traffic_feature_hash=TRAFFIC_SCHEMA.feature_hash,
                    seed=seed, tick=row['tick'], uid=row['uid'], decision_id=row['decision_id'], actor_mask=True,
                    original_pre_state_hash=row['pre_state_hash'], original_snapshot_sha256=entry['snapshot_sha256'],
                    teacher_gate=gate, teacher_proposal=None if gate == 0 else proposal,
                    ego=value.ego.tolist(), traffic=value.traffic.tolist(), traffic_mask=value.traffic_mask.tolist(),
                    trace=dict(traffic_uids=value.traffic_uids, target_uids=value.target_uids,
                               target_reference_seconds=value.target_reference_seconds))
                decoded, decoded_gate, decoded_proposal = decode_row(rebuilt)
                assert (decoded_gate, decoded_proposal) == (gate, proposal)
                assert np.array_equal(decoded.traffic, value.traffic)
                stream.write(json.dumps(rebuilt, allow_nan=False)+'\n')
                counts['rows'] += 1; counts['RELEASE' if gate else 'WAIT'] += 1
                counts['traffic_rows'] += len(value.traffic)
                counts['max_neighbors'] = max(counts['max_neighbors'], len(value.traffic))
                if seed == 893001 and (row['tick'], row['uid']) in PROBE:
                    direct = build_low_observation_v2_5(ego, env)
                    assert np.array_equal(direct.ego, value.ego) and np.array_equal(direct.traffic, value.traffic)
                    expected_blocker = {(92, '0'): 4, (152, '6'): 1}.get((row['tick'], row['uid']))
                    if expected_blocker is not None:
                        assert expected_blocker in value.traffic_uids
                    probe_rows.append(rebuilt)
                if counts['rows'] % 50 == 0:
                    stream.flush()
                    dump(output/'progress.json', dict(counts=counts, elapsed_seconds=time.monotonic()-started))
            inputs.append(dict(seed=seed, snapshot_sha256=snapshot_hashes,
                annotation_sha256=sha(episode/'annotation_result.json')))
    assert counts['rows'] == (4 if mode == 'probe' else 1046)
    dump(output/'probe_rows.json', probe_rows)
    dump(output/'inputs.json', inputs)
    result = dict(status='RECONSTRUCTION_COMPLETE', mode=mode, rows=counts['rows'], counts=counts,
        elapsed_seconds=time.monotonic()-started, environment_steps=0, environment_resets=0,
        teacher_queries=0, parameter_updates=0, checkpoint_writes=0)
    dump(output/'result.json', result)
    if mode == 'full':
        loaded = load_context_dataset(output, allowed_train_seeds={s for s, _, _ in CASES})
        assert len(loaded.decision_ids) == 1046 and int(loaded.gate_target.sum()) == 229
        assert loaded.padded()[0].shape == (1046, 105)
        result['loader_verified'] = True
        dump(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['probe', 'full'], required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    try:
        result = run(args.root, args.output, args.mode, 180 if args.mode == 'probe' else 900)
    except Exception as error:
        if args.output.exists():
            dump(args.output/'result.json', dict(status='ERROR', mode=args.mode, error=repr(error)))
        raise
    print(json.dumps(result))


if __name__ == '__main__':
    main()
