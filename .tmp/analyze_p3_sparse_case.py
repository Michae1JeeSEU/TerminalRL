import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
import sys

case = sys.argv[1]
seed = {'high':892402, 'mixed':892403}[case]
root = Path('D:/7_codespace/training_outputs/terminal/p3_v3_sparse_pilot_20261004')/case
episode = root/f'seed_{seed}'
result = json.loads((root/'result.json').read_text())
rows = [json.loads(x) for x in (episode/'rows.jsonl').read_text().splitlines()]
steps = [json.loads(x) for x in (episode/'steps.jsonl').read_text().splitlines()]
selected = json.loads((episode/'selected.json').read_text())
assert len(rows) == result['rollout']['rows']
assert len(steps) == result['rollout']['ticks']
assert [s['tick'] for s in steps] == list(range(len(steps)))
assert len({r['decision_id'] for r in rows}) == len(rows)
by_tick = defaultdict(list)
by_id = {}
contexts, actor_contexts = Counter(), Counter()
rule_requests, executed, fallback, reasons, recoveries = Counter(), Counter(), Counter(), Counter(), Counter()
prior = {}
for row in rows:
    by_tick[row['tick']].append(row)
    by_id[row['decision_id']] = row
    step = steps[row['tick']]
    assert len(row['observation']) == 90 and all(math.isfinite(x) for x in row['observation'])
    assert row['teacher_label'] is None and row['observation_dim'] == 90
    assert row['seed'] == seed and row['partition'] == 'development_pilot_never_train'
    assert row['behavior_request'] == step['tape']['actions'][row['uid']]
    assert row['behavior_shortlist'] == step['tape']['shortlists'].get(row['uid'], [])
    assert row['pre_state_hash'] == step['pre_state_hash']
    context = dict(fast_inner=row['interface']=='i2f' and row['speed']>=15,
                   busy_lz=row['busy_lz']>0, retry=row['retry_active'], low_soc=row['soc']<.55)
    contexts.update({k:int(v) for k,v in context.items()})
    if row['actor_mask']:
        actor_contexts.update({k:int(v) for k,v in context.items()})
        assert not row['retry_active'] and row['soc']>0
        assert row['phase'] in {'ring_outer','ring_middle','ring_inner'}
    req = bool(row['behavior_request'][0])
    old = prior.get(row['uid'])
    if old and not old['behavior_request'][0] and req:
        recoveries['behavior_WAIT_to_RELEASE'] += 1
        recoveries['behavior_retry_to_RELEASE'] += int(old['retry_active'])
    prior[row['uid']] = row
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
assert dict(contexts) == result['rollout']['raw_contexts']
expected = {}
for tick, group in by_tick.items():
    eligible = sorted((r for r in group if r['actor_mask']), key=lambda r:int(r['uid']))
    window = tick//75
    if eligible and window not in expected:
        expected[window] = eligible[window % len(eligible)]['decision_id']
assert {e['window']:e['row']['decision_id'] for e in selected} == expected
labels, label_contexts, proposals, pair_counts = [], Counter(), Counter(), Counter()
for entry in selected:
    row = by_id[entry['row']['decision_id']]
    assert row == entry['row']
    data = (episode/'snapshots'/entry['snapshot']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry['snapshot_sha256']
    label = json.loads((episode/'labels'/f"window_{entry['window']:02d}.json").read_text())
    assert label['status'] == 'OK' and label['decision_id'] == row['decision_id']
    assert label['observation'] == row['observation'] and label['pre_state_hash'] == row['pre_state_hash']
    assert label['actor_mask'] and row['actor_mask'] and not label['executed_as_behavior']
    assert int(label['teacher_joint_index']>0) == label['teacher_gate']
    assert label['teacher_proposal'] == (label['teacher_joint_index']-1 if label['teacher_gate'] else None)
    assert len(label['teacher_audit']) <= 26
    assert label['teacher_audit'][-1]['joint_index'] == label['teacher_joint_index']
    assert label['teacher_audit'][-1]['feasible']
    assert sum(r['decoded']['release_request']>0 for r in label['teacher_audit']) == label['release_intents_checked']
    if label['teacher_gate']:
        cert = label['selected_certificate']
        assert cert['trajectory_hash'] == cert['checked_hash']
        proposals[str(label['teacher_proposal'])] += 1
    else:
        assert label['selected_certificate'] is None and label['teacher_request'] == [0.,0.,0.]
    label_contexts.update({k:int(v) for k,v in label['contexts'].items()})
    trace = steps[row['tick']]['action_traces'][row['uid']]
    labels.append(dict(window=entry['window'], tick=row['tick'], uid=row['uid'], interface=row['interface'],
        soc=row['soc'], teacher_gate=label['teacher_gate'], teacher_proposal=label['teacher_proposal'],
        behavior=row['behavior_request'], execution=trace['execution_applied_action'],
        intent_checks=label['release_intents_checked'], plan_calls=label['candidate_plan_calls'],
        reasons=dict(Counter(r['reason'] for r in label['teacher_audit'])), seconds=label['elapsed_seconds']))
    pair_counts[f"teacher{label['teacher_gate']}_actual{int(bool(trace['execution_applied_action'][0]))}"] += 1
assert json.loads((episode/'first_live.json').read_text()) == json.loads((episode/'first_replay.json').read_text())
energy = result['rollout']['energy_outcomes']
summary = dict(case=case, status=result['status'], rows=len(rows), steps=len(steps),
    actor_rows=sum(r['actor_mask'] for r in rows), contexts=dict(contexts), actor_contexts=dict(actor_contexts),
    selected_contexts=dict(label_contexts), proposals=dict(proposals), labels=labels,
    selected_teacher_actual=dict(pair_counts), rule_requests=dict(rule_requests), executed=dict(executed),
    executed_rank=dict(fallback), behavior_release_rejected_reasons=dict(reasons),
    recovery_events=dict(recoveries),
    terminal_soc_min=min(x['soc'] for x in energy), terminal_soc_max=max(x['soc'] for x in energy),
    terminal_energy_min=min(x['energy_remaining'] for x in energy),
    label_intents=sum(x['intent_checks'] for x in labels), label_plan_calls=sum(x['plan_calls'] for x in labels),
    rollout_seconds=result['rollout']['elapsed_seconds'], annotation_seconds=result['annotation']['elapsed_seconds'])
print(json.dumps(summary, indent=2))
