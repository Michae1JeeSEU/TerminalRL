"""Frozen V3 I0 collection: full rule cycles and every eligible actor decision.

One episode per invocation. No learner or qualification evaluation is run.
V1 remains unchanged; opt-in V2 measures the larger all-actor query budget
on fresh dense high/mixed-SOC episodes with the original wall-clock limits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import traceback

from terminal_marl.configs.p3_integrated_repair import PROFILE, build_config
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts import collect_p3_v3_sparse_expert as sparse
from terminal_marl.training.p3_v3_expert_dataset import (
    DATA_SCHEMA, DEVELOPMENT_SCHEMA, DEVELOPMENT_PARTITION, inspect_sparse_episode,
)

PROTOCOL = 'p3_v3_i0_all_actor_20261004_v1'
SMOKE_SEED = 892501
MAX_SELECTED = 256
ROLLOUT_SECONDS = 3600
ANNOTATION_SECONDS = 7200
MAX_TICKS = 1800
SCENARIOS = ((12, 6, True), (24, 12, True), (24, 12, False))
PROTOCOL_V2 = 'p3_v3_i0_all_actor_20261004_v2_budget'
MAX_SELECTED_V2 = 2048
SMOKE_SEED_V2 = 892601
SCENARIOS_V2 = ((24, 12, True), (24, 12, False))


def protocol_settings(version):
    sparse.require(version in {'v1', 'v2'}, 'unknown I0 protocol version')
    if version == 'v2':
        return dict(protocol=PROTOCOL_V2, max_selected=MAX_SELECTED_V2, smoke_seed=SMOKE_SEED_V2)
    return dict(protocol=PROTOCOL, max_selected=MAX_SELECTED, smoke_seed=SMOKE_SEED)


def case_definition(partition, index, *, protocol_version='v1'):
    protocol_settings(protocol_version)
    if protocol_version == 'v2':
        sparse.require(partition == 'train', 'V2 only admits the two fresh train collection cases')
        sparse.require(type(index) is int and 1 <= index <= 2, 'index outside frozen V2 cases')
        requests, warm, high = SCENARIOS_V2[index-1]
        return dict(partition=partition, index=index, seed=893100+index, requests=requests,
                    warm_start=warm, high_soc=high)
    limits = {'train': (893000, 24), 'audit': (894000, 6)}
    sparse.require(partition in limits, 'only train/audit collection is supported; Q is untouched')
    base, size = limits[partition]
    sparse.require(type(index) is int and 1 <= index <= size, 'index outside frozen partition')
    requests, warm, high = SCENARIOS[(index-1) % len(SCENARIOS)]
    return dict(partition=partition, index=index, seed=base+index, requests=requests,
                warm_start=warm, high_soc=high)


def rollout_blockers(flow, *, smoke=False):
    blockers = []
    if flow['error'] is not None or flow['status'] != ('SMOKE_OK' if smoke else 'COMPLETED'):
        blockers.append('rollout_incomplete_or_error')
    if not flow['first_tick_replay_equal']:
        blockers.append('first_tick_replay')
    if any(flow['telemetry'].get(key) != 0 for key in sparse.SAFETY_KEYS):
        blockers.append('actual_safety_or_hash')
    if flow['selection_overflow']:
        blockers.append('actor_query_budget_exceeded')
    if not smoke:
        outcomes = flow['outcome_counts']
        if (not flow['terminal'] or flow['censored'] or flow['unresolved']
                or flow['active'] or flow['queue'] or flow['service']):
            blockers.append('unresolved_cycle')
        normal_and_emergency = outcomes.get('completed', 0) + outcomes.get('emergency_landed', 0)
        if normal_and_emergency != flow['requested'] or sum(outcomes.values()) != normal_and_emergency:
            blockers.append('failed_deferred_or_missing_outcome')
    return blockers


def coverage_report(episode):
    """Descriptive pre-state coverage; never changes selection or labels."""
    annotation = json.loads((episode/'annotation_result.json').read_text())
    labels = {}
    for item in annotation['entries']:
        if item['status'] == 'OK':
            label = json.loads((episode/item['label_file']).read_text())
            labels[item['decision_id']] = label
    previous, previous_teacher = {}, {}
    counts = {kind: Counter() for kind in ('raw', 'actor', 'label')}
    interface_gates = defaultdict(Counter)
    interface_proposals = defaultdict(Counter)
    transitions = Counter()
    with (episode/'rows.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            old = previous.get(row['uid'])
            context = sparse.contexts(row)
            context['after_retry'] = bool(row['actor_mask'] and old and old['retry_active'])
            for kind, included in [('raw', True), ('actor', row['actor_mask']),
                                   ('label', row['decision_id'] in labels)]:
                if included:
                    counts[kind].update({key: int(value) for key, value in context.items()})
                    counts[kind][f"interface_{row['interface']}"] += 1
            label = labels.get(row['decision_id'])
            if label is not None:
                gate = label['teacher_gate']
                interface_gates[row['interface']][str(gate)] += 1
                if gate:
                    interface_proposals[row['interface']][str(label['teacher_proposal'])] += 1
                prior = previous_teacher.get(row['uid'])
                if prior is not None and prior['teacher_gate'] == 0 and gate == 1:
                    transitions['same_uid_WAIT_to_RELEASE'] += 1
                    if prior['interface'] == row['interface']:
                        transitions['same_uid_same_interface_WAIT_to_RELEASE'] += 1
                previous_teacher[row['uid']] = dict(teacher_gate=gate, interface=row['interface'])
            previous[row['uid']] = row
    return dict(contexts={key: dict(value) for key, value in counts.items()},
                interface_gates={key: dict(value) for key, value in interface_gates.items()},
                interface_proposals={key: dict(value) for key, value in interface_proposals.items()},
                teacher_transitions=dict(transitions))


def run(output, *, partition='train', index=1, smoke=False, protocol_version='v1'):
    settings = protocol_settings(protocol_version)
    case = (dict(partition=DEVELOPMENT_PARTITION, index=0, seed=settings['smoke_seed'],
                 requests=3, warm_start=3, high_soc=True) if smoke else
            case_definition(partition, index, protocol_version=protocol_version))
    schema = DEVELOPMENT_SCHEMA if smoke else DATA_SCHEMA
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(protocol=settings['protocol'], schema=schema, profile=PROFILE, **case,
        max_steps=2 if smoke else MAX_TICKS,
        rollout_wall_seconds=180 if smoke else ROLLOUT_SECONDS,
        annotation_wall_seconds=180 if smoke else ANNOTATION_SECONDS,
        max_selected=settings['max_selected'], max_release_intents=settings['max_selected']*25,
        selection='every actor_mask=true pre-state in tick/numeric UID order; overflow is incomplete',
        behavior='integrated_repair_rule_nominal_k3',
        label_join='decision_id; OK only; null is not WAIT',
        window_field='sequential query slot; not a time window',
        I0_QUALIFIED=False, PPO_ALLOWED=False)
    sparse.write_json(output/'protocol.json', protocol)  # Saved before the first reset.
    report = dict(protocol=settings['protocol'], schema=schema, **case, status='ERROR', error=None,
                  trained=False, optimizer_created=False, checkpoint_loaded=False,
                  heldout_used=False, I0_QUALIFIED=False, PPO_ALLOWED=False)
    try:
        env = TerminalParallelEnv(build_config(requests=case['requests'], warm_start=case['warm_start'],
                                  high_soc=case['high_soc']), seed=case['seed'])
        episode = output/f"seed_{case['seed']}"
        flow, selected = sparse.rollout(env, episode, seed=case['seed'],
            max_steps=protocol['max_steps'], wall_seconds=protocol['rollout_wall_seconds'], smoke=smoke,
            all_actor=True, max_selected=protocol['max_selected'], schema=schema, partition=case['partition'])
        blockers = rollout_blockers(flow, smoke=smoke)
        report.update(rollout=flow, blockers=blockers,
                      annotation=dict(status='NOT_RUN', selected=len(selected), labelled=0))
        sparse.write_json(output/'result.json', dict(report, status='ANNOTATING' if not blockers else 'INCOMPLETE'))
        if not blockers:
            annotation = sparse.annotate(episode, selected, wall_seconds=protocol['annotation_wall_seconds'])
            report['annotation'] = annotation
            summary = inspect_sparse_episode(episode, expected_seed=case['seed'],
                                             expected_partition=case['partition'])
            report.update(dataset_check=summary, coverage=coverage_report(episode))
            if annotation['status'] != 'COMPLETED':
                blockers.append('annotation_incomplete_or_error')
            if (not selected or summary['gate_supervision_rows'] != flow['row_counts'].get('actor', 0)
                    or summary['ignored_forced_labels'] != 0):
                blockers.append('not_all_actor_rows_labelled')
        report['status'] = ('SMOKE_OK' if smoke else 'COLLECTION_OK') if not blockers else 'INCOMPLETE'
    except Exception as exc:
        report.update(status='ERROR', error=dict(type=type(exc).__name__, message=str(exc),
                                               traceback=traceback.format_exc()))
    sparse.write_json(output/'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition', choices=('train', 'audit'), default='train')
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--protocol-version', choices=('v1', 'v2'), default='v1')
    args = parser.parse_args()
    if args.smoke and (args.partition != 'train' or args.index != 1):
        parser.error('smoke cannot specify another partition/index')
    report = run(args.output, partition=args.partition, index=args.index, smoke=args.smoke,
                 protocol_version=args.protocol_version)
    print(json.dumps(dict(status=report['status'], seed=report['seed'],
                          labels=report.get('annotation', {}).get('labelled', 0))))
    return 0 if report['status'] in {'SMOKE_OK', 'COLLECTION_OK'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
