"""Read-only analysis of the completed, unannotated train02; no environment calls."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

root = Path('D:/7_codespace/training_outputs/terminal/p3_v3_i0_20261004/train_02')
ep = root/'seed_893002'
read = lambda p: json.loads(p.read_text())
result = read(root/'result.json')
flow = result['rollout']
assert result['status'] == 'INCOMPLETE' and result['error'] is None
assert result['blockers'] == ['actor_query_budget_exceeded']
assert result['annotation'] == dict(status='NOT_RUN', selected=256, labelled=0)
assert flow == read(ep/'rollout_result.json')
assert flow['terminal'] and not flow['censored'] and flow['error'] is None
assert flow['active'] == flow['queue'] == flow['service'] == flow['unresolved'] == 0
assert flow['outcome_counts'] == {'completed':23, 'emergency_landed':1}
assert read(ep/'first_live.json') == read(ep/'first_replay.json')
assert not list((ep/'labels').glob('*.json'))
rows = [json.loads(x) for x in (ep/'rows.jsonl').read_text().splitlines()]
steps = [json.loads(x) for x in (ep/'steps.jsonl').read_text().splitlines()]
selected = read(ep/'selected.json')
assert len(rows) == flow['rows'] == 10769 and len(steps) == flow['ticks'] == 943
assert [s['tick'] for s in steps] == list(range(len(steps)))
assert len({r['decision_id'] for r in rows}) == len(rows)
assert [(r['tick'], int(r['uid'])) for r in rows] == sorted((r['tick'], int(r['uid'])) for r in rows)
actors = [r for r in rows if r['actor_mask']]
assert len(actors) == flow['row_counts']['actor'] == 772
assert [e['row'] for e in selected] == actors[:256]
assert len(actors)-len(selected) == flow['selection_overflow'] == 516
assert [e['window'] for e in selected] == list(range(256))
hashes = {}
for entry in selected:
    path = ep/'snapshots'/entry['snapshot']
    if path not in hashes:
        hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert hashes[path] == entry['snapshot_sha256']
    assert read(ep/'snapshots'/f"window_{entry['window']:02d}.json") == entry

counts = {k: Counter() for k in ('raw','actor','selected','overflow')}
reasons = defaultdict(Counter)
per_uid = defaultdict(Counter)
bins = defaultdict(Counter)
prev = {}
selected_ids = {e['row']['decision_id'] for e in selected}
execution = Counter()
rank = Counter()
first_examples = {}
actor_rows = defaultdict(list)
pre_final = Counter(r['tick'] for r in rows if r['phase']=='final')
resource_pre_final = Counter()
resource_decisions = Counter()
for row in rows:
    step = steps[row['tick']]
    assert row['observation_dim'] == len(row['observation']) == 90
    assert all(math.isfinite(v) for v in row['observation'])
    assert row['teacher_label'] is None and row['partition'] == 'train' and row['seed'] == 893002
    assert row['schema'] == 'p3_v3_90d_sparse_expert_v1'
    assert row['pre_state_hash'] == step['pre_state_hash']
    assert row['behavior_request'] == step['tape']['actions'][row['uid']]
    assert row['behavior_shortlist'] == step['tape']['shortlists'].get(row['uid'], [])
    trace = step['action_traces'][row['uid']]
    assert trace['requested_low_action'] == row['behavior_request'] and not trace['teacher_used']
    if trace['runtime_shortlist_rank'] is not None:
        assert trace['execution_applied_action'] == row['behavior_shortlist'][trace['runtime_shortlist_rank']-1]
        rank[str(trace['runtime_shortlist_rank'])] += 1
    old = prev.get(row['uid'])
    context = dict(fast_inner=row['interface']=='i2f' and row['speed']>=15,
                   busy_lz=row['busy_lz']>0, retry=row['retry_active'], low_soc=row['soc']<.55,
                   after_retry=bool(row['actor_mask'] and old and old['retry_active']))
    groups = ['raw'] + (['actor', 'selected' if row['decision_id'] in selected_ids else 'overflow'] if row['actor_mask'] else [])
    for group in groups:
        counts[group].update({k:int(v) for k,v in context.items()})
        counts[group]['rows'] += 1
        counts[group][f"interface_{row['interface']}"] += 1
    request, applied = int(bool(row['behavior_request'][0])), int(bool(trace['execution_applied_action'][0]))
    execution[f"actor{int(row['actor_mask'])}_request{request}_actual{applied}"] += 1
    if row['actor_mask']:
        assert row['phase'] in {'ring_outer','ring_middle','ring_inner'} and row['soc']>0 and not row['retry_active']
        assert request == 1
        reason = 'APPROVED' if applied else str(step['release_telemetry'].get(row['uid'], {}).get('primary_blocker'))
        reasons[row['interface']][reason] += 1
        if reason == 'FINAL_RESOURCE':
            resource_pre_final[str(pre_final[row['tick']])] += 1
            resource_decisions[step['release_telemetry'][row['uid']]['decision_history'][-1]] += 1
        per_uid[row['uid']][row['interface']] += 1
        per_uid[row['uid']][reason] += 1
        bins[str(row['tick']//100*100)][reason] += 1
        actor_rows[row['uid']].append(row)
        if reason not in first_examples:
            first_examples[reason] = dict(tick=row['tick'],uid=row['uid'],soc=row['soc'],
                interface=row['interface'], telemetry=step['release_telemetry'].get(row['uid']), trace=trace)
    prev[row['uid']] = row
assert {k:counts['raw'][k] for k in flow['raw_contexts']} == flow['raw_contexts']
telemetry = flow['telemetry']
for key in ('collisions','nmac','hard_separation_violations','operational_buffer_entries',
            'shield_fail_closed','joint_safety_fail_closed','runtime_traffic_interventions',
            'checked_executed_plan_hash_mismatches'):
    assert telemetry[key] == 0
max_runs = []
for uid, rr in actor_rows.items():
    best = current = 1
    for old, new in zip(rr, rr[1:]):
        current = current+1 if new['tick'] == old['tick']+1 and new['interface']==old['interface'] else 1
        best = max(best,current)
    max_runs.append(dict(uid=uid,actor_rows=len(rr),longest_consecutive_actor_run=best,
        first_tick=rr[0]['tick'],last_tick=rr[-1]['tick'],unique_pre_hashes=len({r['pre_state_hash'] for r in rr}),
        unique_observations=len({tuple(r['observation']) for r in rr})))
summary = dict(status='OVERFLOW_TRACE_CHECK_OK',root_status=result['status'],flow_status=flow['status'],
    counts={k:dict(v) for k,v in counts.items()}, reasons={k:dict(v) for k,v in reasons.items()},
    execution=dict(execution),ranks=dict(rank),per_uid={k:dict(v) for k,v in per_uid.items()},
    hundred_tick_bins={k:dict(v) for k,v in bins.items()},
    longest_actor_runs=sorted(max_runs,key=lambda x:x['longest_consecutive_actor_run'],reverse=True),
    last_selected={k:actors[255][k] for k in ('decision_id','tick','uid','soc','interface')},
    first_overflow={k:actors[256][k] for k in ('decision_id','tick','uid','soc','interface')},
    first_examples=first_examples,
    selected_unique_snapshots=len(hashes),
    resource_pre_final_counts=dict(resource_pre_final), resource_decisions=dict(resource_decisions),
    terminal_soc_min=min(x['soc'] for x in flow['energy_outcomes']),
    terminal_soc_max=max(x['soc'] for x in flow['energy_outcomes']),
    terminal_energy_min=min(x['energy_remaining'] for x in flow['energy_outcomes']),
    source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (root/'result.json',ep/'rows.jsonl',ep/'steps.jsonl',ep/'selected.json')})
print(json.dumps(summary,indent=2))
