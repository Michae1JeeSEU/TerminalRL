import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
import sys

import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.training import p3_v25_forecast_data as new_data
from terminal_marl.training import p3_v25_imitation_qualification as q
p=argparse.ArgumentParser();p.add_argument('case',choices=['smoke','train_A','train_B','train_C']);p.add_argument('--output',required=True);args=p.parse_args()
output=Path(args.output);assert not output.exists()
smoke=args.case=='smoke'
index=0 if smoke else ['train_A','train_B','train_C'].index(args.case)+1
seed,requests,warm,high=(new_data.SMOKE_SEED,3,3,True) if smoke else new_data.CASES[index-1]
partition=new_data.DEVELOPMENT_PARTITION if smoke else 'train'
schema=new_data.DEVELOPMENT_SCHEMA if smoke else new_data.DATA_SCHEMA
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_forecast_train_20261008')/args.case

episode = root/f'seed_{seed}'
read = lambda path: json.loads(path.read_text())
result = read(root/'result.json')
assert result['protocol'] == new_data.PROTOCOL
protocol = read(root/'protocol.json')
assert protocol['seed'] == seed and protocol['max_selected'] == 2048
assert protocol['max_release_intents'] == 51200 and protocol['annotation_wall_seconds'] == (180 if smoke else 7200)
assert protocol['max_steps'] == (2 if smoke else 1800) and protocol['rollout_wall_seconds'] == (180 if smoke else 3600)
assert (protocol['requests'],protocol['warm_start'],protocol['high_soc']) == (requests,warm,high)
assert protocol['merge_gap_prediction_mode']==new_data.GAP_MODE
for k in ['I0_QUALIFIED','PPO_ALLOWED']:
    assert result[k] is False
flow = result['rollout']
assert result['status'] == ('SMOKE_OK' if smoke else 'COLLECTION_OK') and result['error'] is None and not result['blockers']
assert flow['status'] == ('SMOKE_OK' if smoke else 'COMPLETED')
assert flow['censored'] is smoke and flow['terminal'] is not smoke
assert flow['error'] is None and flow['selection_overflow'] == 0 and flow['unresolved'] == (requests if smoke else 0)
assert flow['active'] == (requests if smoke else 0) and flow['queue']==flow['service']==0
assert flow['first_tick_replay_equal']
assert sum(flow['outcome_counts'].values()) == (0 if smoke else flow['requested'])
assert set(flow['outcome_counts']) <= {'completed', 'emergency_landed'}
assert result['annotation']['status'] == 'COMPLETED'
rows = [json.loads(x) for x in (episode/'rows.jsonl').read_text().splitlines()]
steps = [json.loads(x) for x in (episode/'steps.jsonl').read_text().splitlines()]
selected = read(episode/'selected.json')
annotation = read(episode/'annotation_result.json')
assert read(episode/'rollout_result.json') == flow
assert annotation == result['annotation']
assert len(selected) <= 2048 and annotation['elapsed_seconds'] <= protocol['annotation_wall_seconds']
assert len(rows) == flow['rows'] and len(steps) == flow['ticks']
assert steps and [s['tick'] for s in steps] == list(range(len(steps)))
assert len({r['decision_id'] for r in rows}) == len(rows)
assert [(r['tick'], int(r['uid'])) for r in rows] == sorted((r['tick'], int(r['uid'])) for r in rows)
by_id = {}
contexts = {'raw': Counter(), 'actor': Counter(), 'label': Counter()}
rule_requests, executed, fallback, reasons = Counter(), Counter(), Counter(), Counter()
unranked_release_traces = []
prior, after_retry = {}, {}
for row in rows:
    by_id[row['decision_id']] = row
    step = steps[row['tick']]
    assert len(row['observation']) == 90 and all(math.isfinite(x) for x in row['observation'])
    assert row['teacher_label'] is None and row['observation_dim'] == 90
    assert row['seed'] == seed and row['partition'] == partition
    assert row['schema'] == schema
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
        reason = step['release_telemetry'].get(row['uid'], {}).get('primary_blocker')
        if reason is not None:
            reasons[str(reason)] += 1
        else:
            unranked_release_traces.append(dict(tick=row['tick'], uid=row['uid'], soc=row['soc'],
                trace_request=trace['requested_low_action'], trace_applied=trace['execution_applied_action']))
eligible = [row for row in rows if row['actor_mask']]
assert [e['row']['decision_id'] for e in selected] == [row['decision_id'] for row in eligible]
assert [e['window'] for e in selected] == list(range(len(selected)))
assert result['reconstruction']['dataset_check']['raw_rows'] == len(rows)
assert result['reconstruction']['dataset_check']['actor_eligible_rows'] == result['reconstruction']['dataset_check']['gate_supervision_rows'] == len(eligible)
assert result['reconstruction']['dataset_check']['forced_wait_rows'] == len(rows)-len(eligible)
assert len(selected) == annotation['selected'] == annotation['labelled'] == result['reconstruction']['dataset_check']['valid_labels']
assert [a['decision_id'] for a in annotation['entries']] == [e['row']['decision_id'] for e in selected]
assert all(a['status'] == 'OK' for a in annotation['entries'])
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
    label = read(episode/'labels'/f"window_{entry['window']:02d}.json")
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
assert dict(proposals) == result['reconstruction']['dataset_check']['proposal_counts']
assert sum(proposals.values()) == result['reconstruction']['dataset_check']['proposal_supervision_rows']
assert sum(x['intent_checks'] for x in labels) <= 51200
assert read(episode/'first_live.json') == read(episode/'first_replay.json')
context_path=episode/'context_observations.jsonl'
assert hashlib.sha256(context_path.read_bytes()).hexdigest()==result['reconstruction']['sha256']
context_rows=[json.loads(x) for x in context_path.read_text().splitlines()]
assert len(context_rows)==len(eligible)==result['reconstruction']['rows']
assert [r['decision_id'] for r in context_rows]==[r['decision_id'] for r in eligible]
label_by_id={e['row']['decision_id']:read(episode/'labels'/f"window_{e['window']:02d}.json") for e in selected}
for value in context_rows:
    original=by_id[value['decision_id']];label=label_by_id[value['decision_id']]
    assert value['seed']==seed and value['partition']==partition
    obs,gate,proposal=q.decode_evaluation_row(value,seed=seed,partition=partition) if smoke else new_data.decode_row(value)
    assert value['original_observation']==original['observation'] and value['original_pre_state_hash']==original['pre_state_hash']
    assert gate==label['teacher_gate'] and (None if proposal<0 else proposal)==label['teacher_proposal']
telemetry = flow['telemetry']
for key in ('collisions','nmac','hard_separation_violations','operational_buffer_entries',
            'shield_fail_closed','joint_safety_fail_closed','runtime_traffic_interventions',
            'checked_executed_plan_hash_mismatches'):
    assert telemetry[key] == 0, key
energy = flow['energy_outcomes']
summary=dict(case=args.case, status=result['status'], rows=len(rows), steps=len(steps),
    actor_rows=len(eligible), contexts={k:dict(v) for k,v in contexts.items()},
    interface_gates={k:dict(v) for k,v in interface_gates.items()}, proposals=dict(proposals),
    teacher_vs_reported_trace_gate=dict(pairs), rule_requests=dict(rule_requests), reported_trace_gates=dict(executed),
    normal_K3_approval_ranks=dict(fallback), unranked_release_traces=unranked_release_traces, behavior_release_rejected_reasons=dict(reasons), teacher_transitions=dict(transitions),
    teacher_trace_gate_differences=[label for label in labels if label['teacher_gate'] != int(bool(label['execution'][0]))],
    terminal_soc_min=min((x['soc'] for x in energy),default=None), terminal_soc_max=max((x['soc'] for x in energy),default=None),
    terminal_energy_min=min((x['energy_remaining'] for x in energy),default=None),
    emergency={k:v for k,v in telemetry.items() if k.startswith('emergency_')},
    min_actual=telemetry['minimum_actual_separation'], min_swept=telemetry['minimum_swept_separation'],
    label_intents=sum(x['intent_checks'] for x in labels), label_plan_calls=sum(x['plan_calls'] for x in labels),
    longest_queries=sorted(labels,key=lambda x:x['seconds'], reverse=True)[:3],
    rollout_seconds=flow['elapsed_seconds'], annotation_seconds=result['annotation']['elapsed_seconds'])
output.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status='FULL_ANALYSIS_COMPLETE',output=str(output),rows=len(rows),labels=len(labels))))
