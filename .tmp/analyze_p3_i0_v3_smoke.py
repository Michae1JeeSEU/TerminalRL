from pathlib import Path
import hashlib
import json
import math

root = Path('D:/7_codespace/training_outputs/terminal/p3_v3_i0_v3_20261007/smoke')
p = root / 'seed_892701'
def read(path):
    return json.loads(path.read_text(encoding='utf8'))
result, protocol = read(root/'result.json'), read(root/'protocol.json')
assert hashlib.sha256((root/'result.json').read_bytes()).hexdigest() == '504c09bb768bea01988c5f6dd55d31ffdc756df7f13b3a0b83ca91cdd31e8d43'
assert hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest() == 'fbeec15c6e5c949ccbc9d663ca00eae39ded772d298fba64a038caabdd897779'
assert protocol['protocol'] == 'p3_v3_i0_all_actor_20261007_v3_numeric'
assert protocol['seed'] == 892701 and protocol['max_steps'] == 2
assert (protocol['max_selected'], protocol['max_release_intents']) == (2048,51200)
assert protocol['rollout_wall_seconds'] == protocol['annotation_wall_seconds'] == 180
assert result['status'] == 'SMOKE_OK' and result['error'] is None and not result['blockers']
for k in ['trained','optimizer_created','checkpoint_loaded','heldout_used','I0_QUALIFIED','PPO_ALLOWED']:
    assert result[k] is False
rows = [json.loads(s) for s in (p/'rows.jsonl').read_text().splitlines()]
steps = [json.loads(s) for s in (p/'steps.jsonl').read_text().splitlines()]
selected = read(p/'selected.json')
assert len(rows) == 6 and len(steps) == 2 and len(selected) == 3
assert len({r['decision_id'] for r in rows}) == 6
assert [(r['tick'],int(r['uid'])) for r in rows] == sorted((r['tick'],int(r['uid'])) for r in rows)
actor = [r for r in rows if r['actor_mask']]
assert [s['row'] for s in selected] == actor
for r in rows:
    assert len(r['observation']) == r['observation_dim'] == 90
    assert all(math.isfinite(x) for x in r['observation'])
    assert r['teacher_label'] is None and r['partition'] == 'development_pilot_never_train'
    step = steps[r['tick']]
    assert r['pre_state_hash'] == step['pre_state_hash']
    assert r['behavior_request'] == step['tape']['actions'][r['uid']]
    assert r['behavior_shortlist'] == step['tape']['shortlists'].get(r['uid'], [])
    assert not step['action_traces'][r['uid']]['teacher_used']
for n, (s, entry) in enumerate(zip(selected, result['annotation']['entries'], strict=True)):
    assert s['window'] == entry['window'] == n
    assert hashlib.sha256((p/'snapshots'/s['snapshot']).read_bytes()).hexdigest() == s['snapshot_sha256']
    assert read(p/'snapshots'/f'window_{n:02d}.json') == s
    label = read(p/entry['label_file'])
    assert entry['status'] == label['status'] == 'OK' and entry['exit_code'] == 0
    for k in ['decision_id','observation','pre_state_hash','actor_mask','tick','uid']:
        assert label[k] == s['row'][k]
    assert label['teacher_gate'] == 1 and label['teacher_proposal'] == 12
    assert label['teacher_joint_index'] == 13 and label['teacher_request'] == [1,0,0]
    assert label['executed_as_behavior'] is False
    assert len(label['teacher_audit']) == 1 and label['teacher_audit'][0]['feasible']
    assert label['selected_certificate']['checked_hash'] == label['selected_certificate']['trajectory_hash']
assert read(p/'first_live.json') == read(p/'first_replay.json')
assert read(p/'rollout_result.json') == result['rollout']
assert read(p/'annotation_result.json') == result['annotation']
for k in ['hard_separation_violations','nmac','collisions','checked_executed_plan_hash_mismatches','energy_reachability_negative_budgets']:
    assert result['rollout']['telemetry'][k] == 0
assert result['dataset_check']['raw_rows'] == 6
assert result['dataset_check']['gate_supervision_rows'] == result['dataset_check']['proposal_supervision_rows'] == 3
assert result['rollout']['censored'] and result['rollout']['unresolved'] == 3
print('SMOKE_ANALYSIS_OK: 6 raw / 3 actor / 3 OK labels; all joins, snapshots, replay, safety checked; intentional 2-tick cutoff')
