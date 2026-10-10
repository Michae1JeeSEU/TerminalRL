from pathlib import Path
import json
from collections import Counter

root = Path('D:/7_codespace/training_outputs/terminal/p3_v3_sparse_pilot_20261004/smoke')
episode = root/'seed_892401'
result = json.loads((root/'result.json').read_text())
rows = [json.loads(line) for line in (episode/'rows.jsonl').read_text().splitlines()]
steps = [json.loads(line) for line in (episode/'steps.jsonl').read_text().splitlines()]
selected = json.loads((episode/'selected.json').read_text())
label = json.loads((episode/'labels/window_00.json').read_text())
assert result['status'] == 'SMOKE_OK'
assert len(rows) == 6 and len(steps) == 2 and len(selected) == 1
assert len({r['decision_id'] for r in rows}) == 6
assert all(len(r['observation']) == 90 and r['teacher_label'] is None for r in rows)
assert all(r['behavior_request'] == steps[r['tick']]['tape']['actions'][r['uid']] for r in rows)
assert all(r['pre_state_hash'] == steps[r['tick']]['pre_state_hash'] for r in rows)
assert label['decision_id'] == selected[0]['row']['decision_id'] == rows[0]['decision_id']
assert label['observation'] == rows[0]['observation'] and label['actor_mask']
assert label['teacher_gate'] == 1 and not label['executed_as_behavior']
assert label['selected_certificate']['checked_hash'] == label['selected_certificate']['trajectory_hash']
assert json.loads((episode/'first_live.json').read_text()) == json.loads((episode/'first_replay.json').read_text())
print(json.dumps(dict(status=result['status'], rows=len(rows), steps=len(steps),
    actor_rows=sum(r['actor_mask'] for r in rows), selected=len(selected),
    label_uid=label['uid'], joint_index=label['teacher_joint_index'],
    intent_checks=label['release_intents_checked'], plan_calls=label['candidate_plan_calls'],
    behavior_decisions=Counter(t['decision'] for s in steps for t in s['action_traces'].values()
                               if 'decision' in t),
    rollout_seconds=result['rollout']['elapsed_seconds'],
    annotation_seconds=result['annotation']['elapsed_seconds']), ensure_ascii=False))
