"""Read existing failed rollout records and trusted final snapshots, no stepping."""
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.training.p3_v25_rollout import load_steps
from terminal_marl.core.observations import LOW_OBS_V2_4

root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_neutral_course_20261008/student_initial/dense24_fixed/episode')
out=Path('.tmp/p3_neutral_failed_student_inspection.json');assert not out.exists()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
result=read(root/'result.json');assert result['error']['message']=='failed_energy'
after=pickle.loads((root/'failure_after.pkl').read_bytes())
before=pickle.loads((root/'failure_before.pkl').read_bytes())
targets={str(x['uid']) for x in result['energy_outcomes'] if x['outcome']=='failed_energy'}|{str(uid) for uid in after.active}
steps=load_steps(root/'steps.pt')
requests=[json.loads(line) for line in (root/'requests.jsonl').read_text(encoding='utf8').splitlines()]
assert len(steps)==len(requests)==result['ticks']
timelines={uid:[] for uid in sorted(targets,key=int)}
for row,record in zip(steps,requests):
    uids=sorted(record['actor_mask'],key=int)
    for uid in targets.intersection(uids):
        i=uids.index(uid);ego=row.ego[i];trace=record['action_traces'][uid]
        import torch
        probability=float(torch.tensor(row.old_gate_logits[i]).softmax(-1)[1])
        timelines[uid].append(dict(tick=record['tick'],soc=float(ego[LOW_OBS_V2_4.slices['soc']][0]),
            phase=int(ego[:8].argmax()),actor=bool(row.actor_mask[i]),probability=probability,
            request=record['tape']['actions'][uid],execution=trace['execution_applied_action'],
            shortlist_rank=trace.get('runtime_shortlist_rank'),
            release_telemetry=record['release_telemetry'].get(uid),trace=trace))
summaries={}
for uid,rows in timelines.items():
    actors=[r for r in rows if r['actor']]
    requests_r=[r for r in actors if r['request'][0]]
    reasons=Counter(str((r['release_telemetry'] or {}).get('primary_blocker')) for r in requests_r)
    summaries[uid]=dict(first_tick=rows[0]['tick'],last_tick=rows[-1]['tick'],actors=len(actors),
        release=len(requests_r),wait=len(actors)-len(requests_r),release_reasons=dict(reasons),
        first_actor=actors[:5],first_releases=requests_r[:5],last_actors=actors[-5:],
        soc_crossings={str(c):next((r for r in actors if r['soc']<=c),None) for c in (.7,.6,.5,.4,.3,.2,.1)})
def state(env):
    return dict(now=env.now,active=[dict(uid=u.uid,phase=u.phase.value,soc=u.soc,speed=u.speed,
        position=list(u.position),energy_remaining=u.energy_remaining,
        committed=u.committed_plan is not None,emergency=u.emergency_landing)
        for u in env.active.values()],telemetry=env.telemetry)
report=dict(status='INSPECTION_COMPLETE',source_error=result['error'],targets=sorted(targets,key=int),
    summaries=summaries,before=state(before),after=state(after),timelines=timelines)
out.write_text(json.dumps(report,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n',encoding='utf8')
print(json.dumps(dict(status=report['status'],targets=report['targets'],output=str(out))))
