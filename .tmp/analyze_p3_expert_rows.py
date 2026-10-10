import json
from pathlib import Path
from collections import Counter
import math

root=Path('D:/7_codespace/training_outputs/terminal/p3_v3_i0_pilot_20261003/pilot/seed_892201')
rows=[json.loads(x) for x in (root/'rows.jsonl').read_text().splitlines()]
steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()]
by_tick={s['tick']:s for s in steps}
checks=Counter(); masks=Counter(); releases=Counter(); reject=Counter(); unlabelled=Counter()
for r in rows:
    checks['valid90']+=len(r['observation'])==90 and all(math.isfinite(x) for x in r['observation'])
    checks['mask_label_valid']+=r['actor_mask'] or r['teacher_gate']==0
    checks['label_index_valid']+=(r['teacher_proposal'] is None and r['teacher_joint_index']==0) if r['teacher_gate']==0 else r['teacher_proposal']==r['teacher_joint_index']-1
    checks['request_exact']+=by_tick[r['tick']]['tape']['actions'][r['uid']]==r['requested_action']
    checks['state_hash_exact']+=by_tick[r['tick']]['pre_state_hash']==r['pre_state_hash']
    masks[('actor_' if r['actor_mask'] else 'forced_')+str(r['teacher_gate'])]+=1
    if r['teacher_gate']:
        t=by_tick[r['tick']]['action_traces'][r['uid']]
        releases['requested']+=1
        releases['approved']+=t['runtime_shortlist_rank'] is not None
        c=r['selected_certificate']; checks['release_cert_match']+=c['checked_hash']==c['trajectory_hash']
        if t['runtime_shortlist_rank'] is None:
            releases['rejected']+=1
            rel=by_tick[r['tick']]['release_telemetry'].get(r['uid'],{})
            reject[str(rel.get('primary_blocker'))]+=1
for s in steps:
    for uid,trace in s['action_traces'].items():
        if uid not in s['tape']['actions']:
            unlabelled['rows']+=1
            unlabelled['release']+=trace['requested_low_action'][0]>0
            unlabelled['executed_release']+=trace['execution_applied_action'][0]>0
targets=[dict(tick=r['tick'],uid=r['uid'],phase=r['phase'],gate=r['teacher_gate'],soc=r['soc'],audit_count=len(r['teacher_audit'])) for r in rows if r['tick']==150]
print(json.dumps(dict(rows=len(rows),steps=len(steps),unique_decisions=len({r['decision_id'] for r in rows}),checks=checks,masks=masks,releases=releases,reject=reject,new_admission_traces=unlabelled,tick150=targets),indent=2))
