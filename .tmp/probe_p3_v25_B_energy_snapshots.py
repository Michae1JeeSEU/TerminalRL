import hashlib
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.core.safety_trace import _digest
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical,state_record
from terminal_marl.training.p3_v25_runtime import actor_eligible

root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_student_dev_20261008')
reports=[]
for case in ('B_det','B_sto'):
    hashes={r['tick']+1:r['post_hash'] for r in map(json.loads,(root/case/'steps.jsonl').read_text().splitlines())
            if r['tick']+1 in {400,600,850}}
    for tick in (400,600,850):
        path=root/case/f'snapshot_{tick}.pkl'
        data=path.read_bytes();env=pickle.loads(data)
        assert _digest(canonical(state_record(env)))==hashes[tick]
        item=dict(case=case,tick=tick,snapshot_sha256=hashlib.sha256(data).hexdigest(),targets=[])
        for arrival in (17,22):
            uavs=[u for u in env.active.values() if u.arrival_id==arrival]
            if not uavs:
                item['targets'].append(dict(arrival_id=arrival,active=False,
                    outcome=env.ledger.outcomes.get(arrival)))
                continue
            assert len(uavs)==1
            u=uavs[0];p=u.committed_plan
            item['targets'].append(dict(arrival_id=arrival,uid=u.uid,phase=u.phase.value,
                soc=u.soc,speed=u.speed,actor_eligible=actor_eligible(env,u),
                internal_wait=u.internal_wait_time,
                holding_by_ring={r.value:v for r,v in u.holding_time_by_ring.items()},
                safety_retry_remaining=max(0,u.safety_retry_after-env.now),
                plan=None if p is None else dict(mode=p.mode,start=p.start_time,end=p.end_time,
                    remaining=p.end_time-env.now,checked_hash=env.safety.checked_trajectory_hashes[p.plan_id],
                    trajectory_hash=p.trajectory_hash)))
        reports.append(item)
print(json.dumps(reports,indent=2))
