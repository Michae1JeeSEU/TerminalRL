"""Fresh labelled train cycles for the committed-arrival planner, no learner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..configs.p3_integrated_repair import build_config, FORECAST_PROFILE
from ..envs.parallel_env import TerminalParallelEnv
from ..scripts import collect_p3_v3_sparse_expert as sparse
from ..scripts.collect_p3_v3_i0 import rollout_blockers, coverage_report
from .p3_v25_imitation_qualification import rebuild_context, sha
from .p3_v25_context_dataset import decode_row
from .p3_v25_set_learnability import PairedInputs
from .p3_v3_bc_learnability import Data, require, read, write
from .p3_v3_expert_dataset import DATA_SCHEMA, DEVELOPMENT_SCHEMA, DEVELOPMENT_PARTITION
from ..core.trajectory_context_observation import pad_trajectory_context

PROTOCOL = 'p3_v25_forecast_initial_train_20261008_v1'
CASES = ((893301, 12, 6, True), (893302, 24, 12, True), (893303, 24, 12, False))
SMOKE_SEED = 892807
GAP_MODE = 'committed_arrivals_v1'


def collect(output, index, *, smoke=False):
    require(type(index) is int and 1 <= index <= (1 if smoke else len(CASES)), 'unknown train case')
    seed, requests, warm, high = (SMOKE_SEED, 3, 3, True) if smoke else CASES[index-1]
    partition = DEVELOPMENT_PARTITION if smoke else 'train'
    schema = DEVELOPMENT_SCHEMA if smoke else DATA_SCHEMA
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    config = build_config(requests=requests, warm_start=warm, high_soc=high, forecast_merge_gap=True)
    protocol = dict(protocol=PROTOCOL, stage='initial_train_collection', profile=FORECAST_PROFILE,
        merge_gap_prediction_mode=GAP_MODE, seed=seed, index=index, partition=partition, schema=schema,
        requests=requests, warm_start=warm, high_soc=high, smoke=smoke,
        max_steps=2 if smoke else 1800, rollout_wall_seconds=180 if smoke else 3600,
        annotation_wall_seconds=180 if smoke else 7200, max_selected=2048,
        max_release_intents=51200, selection='all actor pre-states, tick/numeric UID order',
        behavior='rule_nominal_k3', teacher='state-only first feasible; not joint approval',
        checkpoint_loaded=False, I0_QUALIFIED=False, PPO_ALLOWED=False)
    write(output/'protocol.json', protocol)
    report = dict(protocol=PROTOCOL, seed=seed, partition=partition, status='ERROR', error=None,
                  I0_QUALIFIED=False, PPO_ALLOWED=False)
    try:
        env = TerminalParallelEnv(config, seed=seed)
        episode = output/f'seed_{seed}'
        flow, selected = sparse.rollout(env, episode, seed=seed, max_steps=protocol['max_steps'],
            wall_seconds=protocol['rollout_wall_seconds'], smoke=smoke, all_actor=True,
            max_selected=2048, schema=schema, partition=partition)
        blockers = rollout_blockers(flow, smoke=smoke)
        report.update(rollout=flow, blockers=blockers)
        write(output/'result.json', dict(report, status='INCOMPLETE' if blockers else 'ANNOTATING'))
        if not blockers:
            annotation = sparse.annotate(episode, selected, wall_seconds=protocol['annotation_wall_seconds'])
            report['annotation'] = annotation
            if annotation['status'] != 'COMPLETED' or annotation['selected'] != annotation['labelled']:
                blockers.append('incomplete_annotation')
            if not blockers:
                report['reconstruction'] = rebuild_context(episode, seed=seed, partition=partition,
                                                           expected_gap_mode=GAP_MODE)
                report['coverage'] = coverage_report(episode)
        report['status'] = ('SMOKE_OK' if smoke else 'COLLECTION_OK') if not blockers else 'INCOMPLETE'
    except Exception as error:
        report.update(status='ERROR', error=repr(error))
    write(output/'result.json', report)
    return report


def load_train(directories):
    """Only the three declared new train cycles; Q and legacy labels rejected."""
    require(len(directories) == len(CASES), 'all initial train cycles are required')
    observations, gates, proposals, ids, raw, contexts, inputs = [], [], [], [], [], [], []
    for directory, (seed, requests, warm, high) in zip(directories, CASES):
        directory = Path(directory)
        protocol, report = read(directory/'protocol.json'), read(directory/'result.json')
        require(protocol['protocol'] == PROTOCOL and report['protocol'] == PROTOCOL
                and protocol['partition'] == report['partition'] == 'train'
                and protocol['seed'] == report['seed'] == seed
                and protocol['merge_gap_prediction_mode'] == GAP_MODE and not protocol['smoke']
                and (protocol['requests'],protocol['warm_start'],protocol['high_soc']) == (requests,warm,high),
                'wrong train identity or planner')
        require(report['status'] == 'COLLECTION_OK' and report['error'] is None and not report['blockers'],
                'incomplete train cycle')
        path = directory/f'seed_{seed}'/'context_observations.jsonl'
        digest = sha(path)
        require(digest == report['reconstruction']['sha256'], 'train contexts changed')
        count = 0
        for line in path.read_text(encoding='utf8').splitlines():
            row = json.loads(line)
            require(row['seed'] == seed and row['decision_id'] == f"{DATA_SCHEMA}:{seed}:{row['tick']}:{row['uid']}"
                    and row['decision_id'] not in ids, 'wrong or duplicate train decision')
            obs, gate, proposal = decode_row(row)
            observations.append(obs); gates.append(gate); proposals.append(proposal); ids.append(row['decision_id'])
            raw.append(row['original_observation']); contexts.append(row['context']); count += 1
        require(count == report['reconstruction']['rows'] > 0, 'train row count differs')
        inputs.append(dict(seed=seed, rows=count, sha256=digest, result_sha256=sha(directory/'result.json')))
    raw_array = np.asarray(raw, dtype=np.float32)
    require(raw_array.shape == (len(ids),90) and np.isfinite(raw_array).all(), 'invalid original train observation')
    data = Data(torch.from_numpy(raw_array), torch.tensor(gates), torch.tensor(proposals), tuple(ids), contexts, inputs)
    return PairedInputs(data, tuple(torch.from_numpy(a) for a in pad_trajectory_context(observations)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    report = collect(args.output, args.index, smoke=args.smoke)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['status'] in {'SMOKE_OK','COLLECTION_OK'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
