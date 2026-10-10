import hashlib
import json
from pathlib import Path

root = Path('D:/7_codespace/training_outputs/terminal/p3_v3_teacher_safety_audit_20261008/witness_v2')
results, summary = [], []
for index in range(6):
    path = root/f'candidate_{index}.json'
    r = json.loads(path.read_text())
    assert int((root/f'candidate_{index}.exit').read_text()) == 0
    assert r['status'] == 'CANDIDATE_WITNESS_COMPLETE' and r['original_reason_equal']
    assert r['physical_resource_state_unchanged'] and r['observation_equal']
    assert r['plan']['integrity'] and r['plan']['dynamics_errors'] == []
    events = []
    for e in r['events']:
        lower = min(e[key]['distance'] for key in ('point', 'swept', 'interval_bound') if e.get(key))
        dense = e['dense']
        events.append(dict(uid=e['uid'], kind=e['kind'], stage=e['stage'],
            raw_min=e['point']['distance'], lower_bound=lower,
            dense_min=dense['dense_minimum']['distance'], time=dense['dense_minimum']['time'],
            visible_slots=dense['visible_slots'], current_distance=dense['current_distance'],
            other_phase=dense['phase'], other_plan=dense.get('other_plan'),
            tube_width=e.get('max_angular_width')))
    events.sort(key=lambda e:e['lower_bound'])
    assert events
    if r['native_reason'] == 'certified':
        assert r['selected_certificate_equal'] and all(e['lower_bound'] >= 80 for e in events)
    else:
        assert events[0]['lower_bound'] < 80
    summary.append(dict(index=index, tick=r['tick'], uid=r['uid'], proposal=r['proposal_index'],
        request=r['request'], teacher_proposal=r['original_teacher_proposal'], reason=r['native_reason'],
        certificate_equal=r['selected_certificate_equal'], hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        duration=r['plan']['duration'], scheduled_departure=r['plan']['metadata']['scheduled_departure'],
        merge_angle=r['plan']['merge_angle'], closest=events[0],
        all_observed_pairs=len(events), slots={k:v['uid'] for k,v in r['slots'].items()},
        target_gaps={k:{key:v.get(key) for key in ('uid','path_gap','relative_path_gap','relative_gap_encoded','baseline_merge_horizon')}
            for k,v in r['slots'].items() if 'target_ring' in k},
        energy_counters=r['query_counter_delta'], elapsed=r['elapsed_seconds']))
    results.append(r)
differences={k:dict(tick92=results[0]['observation'][k],tick101=results[1]['observation'][k])
    for k in results[0]['observation'] if results[0]['observation'][k]!=results[1]['observation'][k]}
report=dict(summary=summary, uid0_observation_differences=differences,
    total_candidate_wall_seconds=sum(json.loads(p.read_text())['wall_seconds'] for p in root.glob('*.cost.json')),
    failed_first_wall_seconds=json.loads((root.parent/'candidate_0.cost.json').read_text())['wall_seconds'])
Path('.tmp/p3_teacher_context_analysis_20261008.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(report,indent=2))
