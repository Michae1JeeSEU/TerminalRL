import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
import sys

index = int(sys.argv[1])
assert index == 2
seed = 893100 + index
root = Path('D:/7_codespace/training_outputs/terminal/p3_v3_i0_v2_20261004')/('dense_high_launch2' if index == 1 else 'dense_mixed')
episode = root/f'seed_{seed}'
read = lambda path: json.loads(path.read_text())
result = read(root/'result.json')
assert result['protocol'] == 'p3_v3_i0_all_actor_20261004_v2_budget'
flow = result['rollout']
assert result['status'] == 'INCOMPLETE' and result['error'] is None
assert set(result['blockers']) == {'annotation_incomplete_or_error', 'not_all_actor_rows_labelled'}
assert flow['status'] == 'COMPLETED' and flow['terminal'] and not flow['censored']
assert flow['error'] is None and flow['selection_overflow'] == 0 and flow['unresolved'] == 0
assert flow['active'] == flow['queue'] == flow['service'] == 0
assert flow['first_tick_replay_equal']
assert sum(flow['outcome_counts'].values()) == flow['requested']
assert set(flow['outcome_counts']) <= {'completed', 'emergency_landed'}
assert result['annotation']['status'] == 'CENSORED'
assert 7200 <= result['annotation']['elapsed_seconds'] < 7202
rows = [json.loads(x) for x in (episode/'rows.jsonl').read_text().splitlines()]
steps = [json.loads(x) for x in (episode/'steps.jsonl').read_text().splitlines()]
selected = read(episode/'selected.json')
annotation = read(episode/'annotation_result.json')
assert len(rows) == flow['rows'] and len(steps) == flow['ticks']
assert steps and [s['tick'] for s in steps] == list(range(len(steps)))
assert len({r['decision_id'] for r in rows}) == len(rows)
assert [(r['tick'], int(r['uid'])) for r in rows] == sorted((r['tick'], int(r['uid'])) for r in rows)
by_id = {}
contexts = {'raw': Counter(), 'actor': Counter(), 'label': Counter()}
rule_requests, executed, fallback, reasons = Counter(), Counter(), Counter(), Counter()
prior, after_retry = {}, {}
for row in rows:
    by_id[row['decision_id']] = row
    step = steps[row['tick']]
    assert len(row['observation']) == 90 and all(math.isfinite(x) for x in row['observation'])
    assert row['teacher_label'] is None and row['observation_dim'] == 90
    assert row['seed'] == seed and row['partition'] == 'train'
    assert row['schema'] == 'p3_v3_90d_sparse_expert_v1'
    assert row['behavior_request'] == step['tape']['actions'][row['uid']]
    assert row['behavior_shortlist'] == step['tape']['shortlists'].get(row['uid'], [])
    assert row['pre_state_hash'] == step['pre_state_hash']
    old = prior.get(row['uid'])
    recovery = bool(row['actor_mask'] and old and old['retry_active'])
    after_retry[row['decision_id']] = recovery
    context = dict(fast_inner=row['interface']=='i2f' and row['speed']>=15,
                   busy_lz=row['busy_lz']>0, retry=row['retry_active'], low_soc=row['soc']<.55,
                   after_retry=recovery)
    for kind, include in [('raw',True), ('actor', row['actor_mask'])]:
        if include:
            contexts[kind].update({k:int(v) for k,v in context.items()})
            contexts[kind][f"interface_{row['interface']}"] += 1
    if row['actor_mask']:
        assert not row['retry_active'] and row['soc']>0
        assert row['phase'] in {'ring_outer','ring_middle','ring_inner'}
    prior[row['uid']] = row
    req = bool(row['behavior_request'][0])
    rule_requests[str(int(req))] += 1
    trace = step['action_traces'][row['uid']]
    assert trace['requested_low_action'] == row['behavior_request'] and not trace['teacher_used']
    executed[str(int(bool(trace['execution_applied_action'][0])))] += 1
    rank = trace['runtime_shortlist_rank']
    if rank is not None:
        assert trace['execution_applied_action'] == row['behavior_shortlist'][rank-1]
        fallback[str(rank)] += 1
    elif req:
        reasons[str(step['release_telemetry'].get(row['uid'], {}).get('primary_blocker'))] += 1
eligible = [row for row in rows if row['actor_mask']]
assert [e['row']['decision_id'] for e in selected] == [row['decision_id'] for row in eligible]
assert [e['window'] for e in selected] == list(range(len(selected)))
assert len(selected) == annotation['selected'] == 482
assert annotation['labelled'] == result['dataset_check']['valid_labels'] == 300
assert [a['decision_id'] for a in annotation['entries']] == [e['row']['decision_id'] for e in selected]
assert [a['status'] for a in annotation['entries']] == ['OK']*300 + ['TIMEOUT'] + ['NOT_ATTEMPTED']*181
assert len(list((episode/'labels').glob('*.json'))) == 300
labels, proposals, pairs, snapshot_hashes = [], Counter(), Counter(), {}
interface_gates, interface_proposals = defaultdict(Counter), defaultdict(Counter)
prior_teacher, transitions = {}, Counter()
for entry in selected:
    row = by_id[entry['row']['decision_id']]
    assert row == entry['row']
    path = episode/'snapshots'/entry['snapshot']
    if path not in snapshot_hashes:
        snapshot_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert snapshot_hashes[path] == entry['snapshot_sha256']
    label_path = episode/'labels'/f"window_{entry['window']:02d}.json"
    if annotation['entries'][entry['window']]['status'] != 'OK':
        assert not label_path.exists()
        continue
    label = read(label_path)
    assert label['status'] == 'OK' and label['decision_id'] == row['decision_id']
    assert label['observation'] == row['observation'] and label['pre_state_hash'] == row['pre_state_hash']
    assert label['actor_mask'] and row['actor_mask'] and not label['executed_as_behavior']
    gate, proposal = label['teacher_gate'], label['teacher_proposal']
    assert int(label['teacher_joint_index']>0) == gate
    assert proposal == (label['teacher_joint_index']-1 if gate else None)
    assert 1 <= len(label['teacher_audit']) <= 26
    assert label['teacher_audit'][-1]['joint_index'] == label['teacher_joint_index']
    assert label['teacher_audit'][-1]['feasible'] and not any(a['feasible'] for a in label['teacher_audit'][:-1])
    assert sum(r['decoded']['release_request']>0 for r in label['teacher_audit']) == label['release_intents_checked']
    if gate:
        cert = label['selected_certificate']
        assert cert['trajectory_hash'] == cert['checked_hash']
        expected = [1., (-1.,-.5,0.,.5,1.)[proposal//5], (-1.,-.5,0.,.5,1.)[proposal%5]]
        assert label['teacher_request'] == expected
        proposals[str(proposal)] += 1
        interface_proposals[row['interface']][str(proposal)] += 1
    else:
        assert label['selected_certificate'] is None and label['teacher_request'] == [0.,0.,0.]
    contexts['label'].update({k:int(v) for k,v in label['contexts'].items()})
    contexts['label']['after_retry'] += int(after_retry[row['decision_id']])
    contexts['label'][f"interface_{row['interface']}"] += 1
    interface_gates[row['interface']][str(gate)] += 1
    previous = prior_teacher.get(row['uid'])
    if previous and previous['teacher_gate'] == 0 and gate:
        transitions['same_uid_WAIT_to_RELEASE'] += 1
        if previous['interface'] == row['interface']:
            transitions['same_uid_same_interface_WAIT_to_RELEASE'] += 1
    prior_teacher[row['uid']] = dict(teacher_gate=gate, interface=row['interface'])
    trace = steps[row['tick']]['action_traces'][row['uid']]
    labels.append(dict(slot=entry['window'], tick=row['tick'], uid=row['uid'], interface=row['interface'],
        soc=row['soc'], teacher_gate=gate, teacher_proposal=proposal,
        teacher_request=label['teacher_request'], behavior=row['behavior_request'], execution=trace['execution_applied_action'],
        intent_checks=label['release_intents_checked'], plan_calls=label['candidate_plan_calls'], seconds=label['elapsed_seconds']))
    pairs[f"teacher{gate}_actual{int(bool(trace['execution_applied_action'][0]))}"] += 1
assert {k:dict(v) for k,v in contexts.items()} == result['coverage']['contexts']
assert dict(transitions) == result['coverage']['teacher_transitions']
assert {k:dict(v) for k,v in interface_gates.items()} == result['coverage']['interface_gates']
assert {k:dict(v) for k,v in interface_proposals.items()} == result['coverage']['interface_proposals']
assert dict(proposals) == result['dataset_check']['proposal_counts']
assert read(episode/'first_live.json') == read(episode/'first_replay.json')
telemetry = flow['telemetry']
for key in ('collisions','nmac','hard_separation_violations','operational_buffer_entries',
            'shield_fail_closed','joint_safety_fail_closed','runtime_traffic_interventions',
            'checked_executed_plan_hash_mismatches'):
    assert telemetry[key] == 0, key
energy = flow['energy_outcomes']
from statistics import median
cost_groups = {}
for label in labels:
    key = label['interface'] + ('_R' if label['teacher_gate'] else '_W')
    group = cost_groups.setdefault(key, dict(n=0, seconds=0., intents=0, plans=0))
    group['n'] += 1; group['seconds'] += label['seconds']; group['intents'] += label['intent_checks']; group['plans'] += label['plan_calls']
slow = [label for label in labels if label['seconds'] >= 30]
label_seconds = sum(label['seconds'] for label in labels)
missing_contexts = {k:contexts['actor'][k]-contexts['label'][k] for k in contexts['actor']}
cost_summary = dict(groups=cost_groups, completed_query_seconds=label_seconds,
    annotation_minus_completed_query_seconds=annotation['elapsed_seconds']-label_seconds,
    median_seconds=median(label['seconds'] for label in labels), slow_count=len(slow),
    slow_seconds=sum(label['seconds'] for label in slow), earliest_slow=slow[0] if slow else None,
    last_ok=labels[-1], timeout_row=selected[300]['row'], missing_contexts=missing_contexts,
    statuses=dict(Counter(a['status'] for a in annotation['entries'])), snapshot_count=len(snapshot_hashes))

print(json.dumps(dict(case=index, status=result['status'], rows=len(rows), steps=len(steps),
    actor_rows=len(eligible), cost=cost_summary, contexts={k:dict(v) for k,v in contexts.items()},
    interface_gates={k:dict(v) for k,v in interface_gates.items()}, proposals=dict(proposals),
    selected_teacher_actual=dict(pairs), rule_requests=dict(rule_requests), executed=dict(executed),
    executed_rank=dict(fallback), behavior_release_rejected_reasons=dict(reasons), teacher_transitions=dict(transitions),
    teacher_actual_gate_differences=[label for label in labels if label['teacher_gate'] != int(bool(label['execution'][0]))],
    terminal_soc_min=min(x['soc'] for x in energy), terminal_soc_max=max(x['soc'] for x in energy),
    terminal_energy_min=min(x['energy_remaining'] for x in energy),
    emergency={k:v for k,v in telemetry.items() if k.startswith('emergency_')},
    min_actual=telemetry['minimum_actual_separation'], min_swept=telemetry['minimum_swept_separation'],
    label_intents=sum(x['intent_checks'] for x in labels), label_plan_calls=sum(x['plan_calls'] for x in labels),
    longest_queries=sorted(labels,key=lambda x:x['seconds'], reverse=True)[:3],
    rollout_seconds=flow['elapsed_seconds'], annotation_seconds=result['annotation']['elapsed_seconds']), indent=2))
