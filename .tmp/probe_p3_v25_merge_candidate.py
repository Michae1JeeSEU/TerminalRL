"""Two private nominal candidates at one already recorded development state."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys
import time
from types import MethodType
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.core.config import RingRole
from terminal_marl.core.merge_scheduler import LowLevelIntent, RepairCandidate
from terminal_marl.core.safety_trace import _digest
from terminal_marl.envs.parallel_env import WorldSnapshot
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical, state_record, _json

p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--tick',type=int,required=True)
p.add_argument('--uid',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);assert not out.exists()
query=next(x for x in json.loads(Path('.tmp/p3_v25_merge_context_full.json').read_text())['queries']
    if (x['case'],x['tick'],x['uid'])==(a.case,a.tick,a.uid))
assert query['forecast_gap_angle'] is not None
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_student_dev_20261008')/a.case
payload=(root/f'snapshot_{a.tick}.pkl').read_bytes()
expected=json.loads((root/'steps.jsonl').read_text().splitlines()[a.tick-1])['post_hash']
results=[]
for variant in ('original','forecast_gap'):
    env=pickle.loads(payload);assert _digest(canonical(state_record(env)))==expected
    ego=env.active[a.uid];target=RingRole(query['target'])
    snap=WorldSnapshot(env.now,env.active,env.airspace)
    initial=env.scheduler.propose(ego,LowLevelIntent(1.,0.,0.),snap,target)
    assert abs(initial.merge_time-query['merge_time'])<1e-9 and abs(initial.merge_angle-query['old_angle'])<1e-9
    if variant=='forecast_gap':
        def gap(self,uav,tgt,moment,snapshot,speed):
            assert uav.uid==a.uid and tgt is target and abs(moment-query['merge_time'])<1e-9
            return query['forecast_gap_angle']
        env.scheduler._gap_center=MethodType(gap,env.scheduler)
    start=time.monotonic()
    prop,plan,reason=env._candidate_plan(ego,RepairCandidate('requested',LowLevelIntent(1.,0.,0.)),target,snap,None)
    results.append(dict(variant=variant,reason=reason,elapsed_seconds=time.monotonic()-start,
        proposal=None if prop is None else asdict(prop),
        plan=None if plan is None else dict(start=plan.start_time,end=plan.end_time,duration=plan.end_time-plan.start_time,
            hash=plan.trajectory_hash,metadata=plan.metadata,
            checked_hash=env.safety.checked_trajectory_hashes.get(plan.plan_id)),
        audit=env.safety.last_predicted_track_audit))
    # Persist each returned candidate before evaluating the next one.
    out.write_text(json.dumps(_json(dict(status='PARTIAL' if len(results)==1 else 'COMPLETE',
        case=a.case,tick=a.tick,uid=a.uid,reset_calls=0,step_calls=0,teacher_calls=0,
        candidates=results)),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='COMPLETE',output=str(out),summary=[dict(variant=x['variant'],reason=x['reason'],
    elapsed_seconds=x['elapsed_seconds'],duration=None if x['plan'] is None else x['plan']['duration']) for x in results])))
