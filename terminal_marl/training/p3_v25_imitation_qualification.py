"""Fresh V2.5 imitation candidates and one-use, explicitly separated Q data.

Collection uses the existing rule/expert contract. Q is never returned by a
training loader. No policy rollout, DAgger, PPO or automatic admission occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from ..algo.trajectory_set_actor import TrajectorySetActor
from ..configs.p3_integrated_repair import build_config
from ..core.observations import build_low_observation_v2_4
from ..core.safety_trace import _digest
from ..core.trajectory_context_observation import (
    VERSION, EGO_SCHEMA, TRAFFIC_SCHEMA, TrajectoryContextCache,
    TrajectoryContextObservation, pad_trajectory_context,
)
from ..envs.parallel_env import TerminalParallelEnv
from ..scripts import collect_p3_v3_sparse_expert as sparse
from ..scripts.collect_p3_v3_i0 import rollout_blockers, coverage_report
from ..scripts.diagnose_p3_direct_contract import canonical, state_record
from .p3_v25_context_dataset import DATA_SCHEMA as CONTEXT_SCHEMA
from .p3_v25_set_learnability import fit_arm, load_paired, synthetic, measure
from .p3_v3_bc_learnability import Data, require, read, write
from .p3_v3_expert_dataset import (
    DATA_SCHEMA, DEVELOPMENT_SCHEMA, DEVELOPMENT_PARTITION, _join_episode,
)

PROTOCOL = 'p3_v25_fresh_i1_20261008_v1'
SEEDS = (91801, 91802, 91803)
Q_CASES = ((895301, 12, 6, True), (895302, 24, 12, True), (895303, 24, 12, False))
SMOKE_SEED = 892805


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort_manifest(q_directories):
    """Freeze completed Q metadata and byte hashes, without decoding any row."""
    require(len(q_directories) == 3, 'all frozen Q cases are required')
    entries = []
    for directory, case in zip(q_directories, Q_CASES):
        directory, seed = Path(directory), case[0]
        report, protocol = read(directory/'result.json'), read(directory/'protocol.json')
        require(report['status'] == 'COLLECTION_OK' and report['error'] is None
                and not report['blockers'] and report['seed'] == seed
                and protocol['seed'] == seed and protocol['partition'] == 'Q'
                and protocol['protocol'] == PROTOCOL, 'incomplete or wrong Q cohort')
        path = directory/f'seed_{seed}'/'context_observations.jsonl'
        digest = sha(path)
        require(digest == report['reconstruction']['sha256'], 'Q context file changed')
        entries.append(dict(seed=seed, rows=report['reconstruction']['rows'], sha256=digest,
                            result_sha256=sha(directory/'result.json')))
    return entries


def fit(root, output, *, q_directories=(), smoke=False):
    """Fresh fixed-final fits; only frozen Q metadata/hashes, never row arrays."""
    cohort = [] if smoke else cohort_manifest(q_directories)
    markers = [] if smoke else [Path(path)/'training_consumed.json' for path in q_directories]
    require(not any(path.exists() for path in markers), 'Q already authorized a training attempt')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for path in markers:
        with path.open('x', encoding='utf8') as stream:
            json.dump(dict(protocol=PROTOCOL, seeds=SEEDS, output=str(output)), stream)
    paired = synthetic() if smoke else load_paired(root)
    seeds, updates = ((17,), 2) if smoke else (SEEDS, 5000)
    write(output/'protocol.json', dict(protocol=PROTOCOL, stage='fit', smoke=smoke,
        seeds=seeds, updates=updates, batch_size=128, learning_rate=.0003,
        wall_seconds_per_seed=300, gradient_clip=.5, selection='fixed final; all seeds',
        observation_schema=VERSION, Q_row_access=False, frozen_Q_metadata=cohort, PPO_ALLOWED=False))
    write(output/'inputs.json', dict(decision_ids=paired.original.ids, episodes=paired.original.inputs))
    results, checkpoints = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        initial = deepcopy(TrajectorySetActor().state_dict())
        case = output/f'seed_{seed}'
        result = fit_arm(paired, case, 'new_information', seed, initial,
                         updates=updates, wall_seconds=300, protocol=PROTOCOL)
        # Reuse the tested fitter's unqualified checkpoint without changing it.
        path = case/'diagnostic_final.pt'
        checkpoints.append(dict(seed=seed, path=path.relative_to(output).as_posix(), sha256=sha(path)))
        results.append(result)
    result = dict(status='FIT_FROZEN', protocol=PROTOCOL, smoke=smoke,
                  fits=results, checkpoints=checkpoints, Q_row_access=False, frozen_Q_metadata=cohort,
                  I1_QUALIFIED=False, STUDENT_QUALIFIED=False, PPO_ALLOWED=False)
    write(output/'result.json', result)
    return result


def frozen_models(fit_directory, *, smoke=False):
    directory = Path(fit_directory)
    report = read(directory/'result.json')
    require(report['status'] == 'FIT_FROZEN' and report['protocol'] == PROTOCOL
            and report['smoke'] is smoke, 'fresh fit manifest is not frozen')
    require([c['seed'] for c in report['checkpoints']] == ([17] if smoke else list(SEEDS)),
            'candidate seed set differs')
    for item in report['checkpoints']:
        require(item['path'] == f"seed_{item['seed']}/diagnostic_final.pt", 'unexpected checkpoint path')
        require(sha(directory/item['path']) == item['sha256'], 'frozen checkpoint changed')
    return report


def rebuild_q(episode, *, seed, partition, wall_seconds=900):
    """Evaluation-only wrapper; never admit train through the Q decoder."""
    require(partition in {'Q', DEVELOPMENT_PARTITION}, 'Q reconstruction partition differs')
    return rebuild_context(episode, seed=seed, partition=partition, wall_seconds=wall_seconds)


def rebuild_context(episode, *, seed, partition, wall_seconds=900, expected_gap_mode=None):
    """Rebuild labelled immutable states, with distinct train/evaluation decoders."""
    require(partition in {'train', 'Q', DEVELOPMENT_PARTITION}, 'unknown context partition')
    episode = Path(episode)
    batch = _join_episode(episode, expected_partition=partition, allowed_seeds={seed}, training=False)
    require(batch.summary['actor_eligible_rows'] == len(batch.decision_ids) > 0,
            'Q must have every actor labelled')
    require(batch.summary['ignored_forced_labels'] == 0, 'forced actor has a Q label')
    labels = dict(zip(batch.decision_ids, zip(batch.gate_target, batch.proposal_target)))
    raw_contexts, prior = {}, {}
    for line in (episode/'rows.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        old = prior.get(row['uid'])
        raw_contexts[row['decision_id']] = dict(seed=seed, interface=row['interface'],
            low_soc=row['soc'] < .55, fast_inner=row['interface'] == 'i2f' and row['speed'] >= 15,
            busy_lz=row['busy_lz'] > 0, after_retry=bool(old and old['retry_active']))
        prior[row['uid']] = row
    started, cache, previous_snapshot, before = time.monotonic(), None, None, None
    written = []
    with (episode/'context_observations.jsonl').open('x', encoding='utf8') as stream:
        for item in read(episode/'annotation_result.json')['entries']:
            require(time.monotonic()-started < wall_seconds, 'Q reconstruction budget exhausted')
            require(item['status'] == 'OK', 'incomplete Q annotation')
            entry = read(episode/'snapshots'/f"window_{item['window']:02d}.json")
            row, name = entry['row'], entry['snapshot']
            require(Path(name).name == name, 'invalid snapshot name')
            path = episode/'snapshots'/name
            require(sha(path) == entry['snapshot_sha256'], 'trusted snapshot hash differs')
            if name != previous_snapshot:
                env = pickle.loads(path.read_bytes())
                if expected_gap_mode is not None:
                    require(env.config.merge_gap_prediction_mode == expected_gap_mode,
                            'snapshot planner mode differs')
                before = canonical(state_record(env))
                cache = TrajectoryContextCache(env)
                previous_snapshot = name
            require(_digest(before) == row['pre_state_hash'], 'snapshot state differs')
            require(np.array_equal(build_low_observation_v2_4(env.active[int(row['uid'])], env),
                                   np.asarray(row['observation'], dtype=np.float32)), 'original Q observation differs')
            value = cache.build(env.active[int(row['uid'])])
            require(canonical(state_record(env)) == before, 'Q reconstruction mutated state')
            gate, proposal = labels[row['decision_id']]
            record = dict(schema=CONTEXT_SCHEMA, observation_schema=VERSION, partition=partition,
                ego_feature_hash=EGO_SCHEMA.feature_hash, traffic_feature_hash=TRAFFIC_SCHEMA.feature_hash,
                seed=seed, tick=row['tick'], uid=row['uid'], decision_id=row['decision_id'], actor_mask=True,
                teacher_gate=int(gate), teacher_proposal=None if gate == 0 else int(proposal),
                ego=value.ego.tolist(), traffic=value.traffic.tolist(), traffic_mask=value.traffic_mask.tolist(),
                context=raw_contexts[row['decision_id']], original_observation=row['observation'],
                original_pre_state_hash=row['pre_state_hash'], snapshot_sha256=entry['snapshot_sha256'])
            if partition == 'train':
                from .p3_v25_context_dataset import decode_row
                decode_row(record)
            else:
                decode_evaluation_row(record, seed=seed, partition=partition)
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            written.append(row['decision_id'])
    require(written == list(batch.decision_ids), 'Q reconstruction label order differs')
    return dict(rows=len(written), sha256=sha(episode/'context_observations.jsonl'),
                elapsed_seconds=time.monotonic()-started, dataset_check=batch.summary)


def decode_evaluation_row(row, *, seed, partition='Q'):
    """Evaluation-only decoder, never relabels Q as train to bypass a loader."""
    require(partition in {'Q', DEVELOPMENT_PARTITION}, 'evaluation does not accept train data')
    require(row['partition'] == partition and row['seed'] == seed, 'Q partition/seed mismatch')
    require(row['schema'] == CONTEXT_SCHEMA and row['observation_schema'] == VERSION
            and row['ego_feature_hash'] == EGO_SCHEMA.feature_hash
            and row['traffic_feature_hash'] == TRAFFIC_SCHEMA.feature_hash, 'Q feature schema differs')
    wire = DEVELOPMENT_SCHEMA if partition == DEVELOPMENT_PARTITION else DATA_SCHEMA
    require(row['decision_id'] == f"{wire}:{seed}:{row['tick']}:{row['uid']}"
            and row['actor_mask'] is True, 'invalid Q decision identity/mask')
    ego, traffic = np.asarray(row['ego'], dtype=np.float32), np.asarray(row['traffic'], dtype=np.float32)
    if traffic.shape == (0,):
        traffic = traffic.reshape(0, 50)
    mask = np.asarray(row['traffic_mask'])
    if not len(mask):
        mask = mask.astype(bool)
    require(ego.shape == (105,) and traffic.ndim == 2 and traffic.shape[1] == 50
            and mask.shape == (len(traffic),) and mask.dtype == np.bool_ and mask.all()
            and np.isfinite(ego).all() and np.isfinite(traffic).all(), 'invalid Q tensors')
    gate, proposal = row['teacher_gate'], row['teacher_proposal']
    require(type(gate) is int and gate in (0, 1), 'invalid Q gate')
    require((gate == 0 and proposal is None) or
            (gate == 1 and type(proposal) is int and 0 <= proposal < 25), 'invalid Q proposal')
    return TrajectoryContextObservation(ego, traffic, mask, (), (None, None), 0.), gate, -1 if gate == 0 else proposal


def collect(output, index, *, smoke=False):
    require(type(index) is int and 1 <= index <= (1 if smoke else 3), 'unknown Q case')
    seed, requests, warm, high = (SMOKE_SEED, 3, 3, True) if smoke else Q_CASES[index-1]
    partition = DEVELOPMENT_PARTITION if smoke else 'Q'
    schema = DEVELOPMENT_SCHEMA if smoke else DATA_SCHEMA
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(protocol=PROTOCOL, stage='collect_Q', seed=seed, partition=partition,
        schema=schema, requests=requests, warm_start=warm, high_soc=high, smoke=smoke,
        max_steps=2 if smoke else 1800, rollout_wall_seconds=180 if smoke else 3600,
        annotation_wall_seconds=180 if smoke else 7200, max_selected=2048,
        selection='all actor pre-states, tick/numeric UID order',
        behavior='rule_nominal_k3', teacher='state-only first feasible; not joint approval',
        candidate_seeds=SEEDS, checkpoint_loaded=False, PPO_ALLOWED=False)
    write(output/'protocol.json', protocol)
    report = dict(protocol=PROTOCOL, seed=seed, partition=partition, status='ERROR', error=None)
    try:
        env = TerminalParallelEnv(build_config(requests=requests, warm_start=warm, high_soc=high), seed=seed)
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
                report['reconstruction'] = rebuild_q(episode, seed=seed, partition=partition)
                report['coverage'] = coverage_report(episode)
        report['status'] = ('SMOKE_OK' if smoke else 'COLLECTION_OK') if not blockers else 'INCOMPLETE'
    except Exception as error:
        report.update(status='ERROR', error=repr(error))
    write(output/'result.json', report)
    return report


def gates(metrics, predictions, targets):
    """Existing I1 thresholds, reported independently of execution safety."""
    overall = metrics['groups']['overall']
    gate, proposal = overall['gate'], overall['proposal']
    confusion = gate['confusion']
    f1s = []
    for label in (0, 1):
        tp = confusion[label][label]
        denominator = 2*tp + confusion[label][1-label] + confusion[1-label][label]
        f1s.append(2*tp/denominator if denominator else 0.)
    decoded = Counter(0 if p['gate'] == 0 else p['proposal']+1 for p in predictions)
    release_proposals = Counter(p['proposal'] for p in predictions if p['gate'])
    n = len(predictions)
    checks = dict(WAIT_recall=gate['WAIT_recall'] is not None and gate['WAIT_recall'] >= .90,
        RELEASE_recall=gate['RELEASE_recall'] is not None and gate['RELEASE_recall'] >= .70,
        macro_F1=sum(f1s)/2 >= .80, proposal_top1=proposal is not None and proposal['top1'] >= .85,
        proposal_top3=proposal is not None and proposal['top3'] >= .95,
        decoded_accuracy=overall['joint_accuracy'] >= .85,
        stable_classes_present={11,12,13,14} <= set(targets),
        stable_classes_predicted={11,12,13,14} <= set(decoded),
        both_predicted_gates=all(gate[f'prediction_{name}']/n >= .05 for name in ('WAIT','RELEASE')),
        proposal_not_collapsed=bool(release_proposals) and max(release_proposals.values())/sum(release_proposals.values()) < .95)
    return dict(passed=all(checks.values()), checks=checks, macro_F1=sum(f1s)/2,
                decoded_prediction_counts=decoded)


def evaluate(fit_directory, q_directories, output):
    manifest = frozen_models(fit_directory)
    cohort = cohort_manifest(q_directories)
    require(cohort == manifest['frozen_Q_metadata'], 'Q differs from pre-training freeze')
    # One evaluation consumes this cohort even if later reading/scoring fails.
    # It is local experiment state, not a general security access mechanism.
    markers = [Path(path)/'evaluation_consumed.json' for path in q_directories]
    require(not any(path.exists() for path in markers), 'Q was already consumed')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for path in markers:
        with path.open('x', encoding='utf8') as stream:
            json.dump(dict(protocol=PROTOCOL, checkpoints=manifest['checkpoints'], output=str(output)), stream)
    observations, raw, gs, ps, ids, contexts, inputs = [], [], [], [], [], [], []
    for directory, case in zip(q_directories, Q_CASES):
        directory, seed = Path(directory), case[0]
        result, protocol = read(directory/'result.json'), read(directory/'protocol.json')
        require(result['status'] == 'COLLECTION_OK' and result['error'] is None
                and not result['blockers'] and protocol['partition'] == 'Q', 'Q collection identity differs')
        path = directory/f'seed_{seed}'/'context_observations.jsonl'
        require(sha(path) == result['reconstruction']['sha256'], 'Q context file changed')
        count = 0
        for line in path.read_text(encoding='utf8').splitlines():
            row = json.loads(line)
            obs, gate, proposal = decode_evaluation_row(row, seed=seed)
            require(row['decision_id'] not in ids, 'duplicate Q row')
            observations.append(obs); raw.append(row['original_observation']); gs.append(gate); ps.append(proposal)
            ids.append(row['decision_id']); contexts.append(row['context']); count += 1
        require(count == result['reconstruction']['rows'], 'incomplete Q context file')
        inputs.append(dict(seed=seed, rows=count, sha256=sha(path)))
    data = Data(torch.tensor(raw, dtype=torch.float32), torch.tensor(gs), torch.tensor(ps), tuple(ids), contexts, inputs)
    tensors = tuple(torch.from_numpy(a) for a in pad_trajectory_context(observations))
    results = []
    for item in manifest['checkpoints']:
        payload = torch.load(Path(fit_directory)/item['path'], map_location='cpu', weights_only=True)
        model = TrajectorySetActor(); model.load_state_dict(payload['model']); model.requires_grad_(False)
        metrics, predictions = measure(model, tensors, data)
        decision = gates(metrics, predictions, [0 if g == 0 else p+1 for g,p in zip(gs,ps)])
        write(output/f"predictions_{item['seed']}.json", predictions)
        results.append(dict(seed=item['seed'], metrics=metrics, decision=decision, checkpoint_sha256=item['sha256']))
    frozen_models(fit_directory)
    summaries = {}
    for name in ('WAIT_recall', 'RELEASE_recall', 'proposal_top1', 'proposal_top3', 'joint_accuracy', 'macro_F1'):
        values = []
        for result in results:
            overall = result['metrics']['groups']['overall']
            if name in {'WAIT_recall','RELEASE_recall'}:
                value = overall['gate'][name]
            elif name.startswith('proposal_'):
                value = overall['proposal'][name.removeprefix('proposal_')] if overall['proposal'] else None
            elif name == 'macro_F1':
                value = result['decision']['macro_F1']
            else:
                value = overall[name]
            values.append(value)
        summaries[name] = None if any(value is None for value in values) else dict(
            mean=float(np.mean(values)), std=float(np.std(values)), minimum=min(values), maximum=max(values))
    report = dict(status='EVALUATED', protocol=PROTOCOL, inputs=inputs, seeds=results,
        across_seeds=summaries,
        all_imitation_gates_passed=all(r['decision']['passed'] for r in results),
        STUDENT_QUALIFIED=False, PPO_ALLOWED=False, parameter_updates=0)
    write(output/'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('fit','collect','evaluate'), required=True)
    parser.add_argument('--root', type=Path, default=Path('/home/user/training_outputs/terminal'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit-directory', type=Path)
    parser.add_argument('--q-directories', type=Path, nargs=3)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'fit':
        require(args.smoke or args.q_directories, 'freeze the Q cohort before fresh training')
        result = fit(args.root, args.output, q_directories=args.q_directories or (), smoke=args.smoke)
    elif args.mode == 'collect':
        result = collect(args.output, args.index, smoke=args.smoke)
    else:
        require(not args.smoke and args.fit_directory and args.q_directories, 'missing evaluation inputs')
        result = evaluate(args.fit_directory, args.q_directories, args.output)
    print(json.dumps(dict(status=result['status'])))
    return 0 if result['status'] in {'FIT_FROZEN','SMOKE_OK','COLLECTION_OK','EVALUATED'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
