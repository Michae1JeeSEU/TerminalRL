"""Read-only target-ring membership/phase check on existing trusted snapshots."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terminal_marl.core.config import RingRole
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.safety_trace import _digest
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical, state_record
from terminal_marl.training.p3_v25_runtime import actor_eligible

p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);assert not out.exists()
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_student_dev_20261008')
chosen=[('B_sto',400),('C_det',400),('C_det',600),('C_det',850)]
if a.smoke:chosen=chosen[:1]
reports=[];counts=Counter()
for case,tick in chosen:
    env=pickle.loads((root/case/f'snapshot_{tick}.pkl').read_bytes())
    before=_digest(canonical(state_record(env)))
    row=json.loads((root/case/'steps.jsonl').read_text().splitlines()[tick-1])
    assert before==row['post_hash'] and env.now==tick
    snapshot=SimpleNamespace(now=env.now,active=env.active,airspace=env.airspace)
    predictor=SafetyFilter(env.config)
    for uid in sorted(env.active):
        ego=env.active[uid]
        target={RingRole.OUTER:RingRole.MIDDLE,RingRole.MIDDLE:RingRole.INNER}.get(ego.ring_role)
        if target is None or not actor_eligible(env,ego):continue
        prop=env.scheduler.propose(ego,LowLevelIntent(1.,0.,0.),snapshot,target)
        old_ids=[];predicted=[];details=[]
        for other in env.active.values():
            if other.uid==uid:continue
            plan=other.committed_plan
            old=other.ring_role is target
            include=(old if plan is None else plan.metadata.get('target_phase')=='ring_'+target.value and plan.end_time<=prop.merge_time+1e-9)
            if old:old_ids.append(other.uid)
            if not (old or include):continue
            times=np.asarray([prop.merge_time])
            pos=np.asarray(predictor._holder_positions(other,snapshot,times) if plan is None else predictor._committed_positions(plan,snapshot,times))[0]
            angle=math.atan2(pos[1],pos[0])%(2*math.pi)
            if include:predicted.append((other.uid,angle))
            details.append(dict(uid=other.uid,phase=other.phase.value,current_ring=getattr(other.ring_role,'value',None),
                plan_target=None if plan is None else plan.metadata.get('target_phase'),
                plan_end=None if plan is None else plan.end_time,old_member=old,future_member=bool(include),future_angle=angle))
        angles=sorted(angle for _,angle in predicted)
        if angles:
            gaps=[((angles[(i+1)%len(angles)]-x)%(2*math.pi),x) for i,x in enumerate(angles)]
            gap,start=max(gaps) if len(angles)>1 else (2*math.pi,angles[0])
            corrected=(start+gap/2)%(2*math.pi)
        else:corrected=None
        new_ids=[u for u,_ in predicted]
        missing=sorted(set(new_ids)-set(old_ids));stale=sorted(set(old_ids)-set(new_ids))
        counts['queries']+=1;counts['missing_incoming']+=bool(missing);counts['stale_departing']+=bool(stale)
        ring=env.airspace.actual_ring(target)
        def clearance(angle):
            if angle is None or not predicted:return None
            return min(2*ring.radius*abs(math.sin((angle-x)/2)) for _,x in predicted)
        reports.append(dict(case=case,tick=tick,uid=uid,arrival_id=ego.arrival_id,source=ego.ring_role.value,
            target=target.value,merge_time=prop.merge_time,old_angle=prop.merge_angle,forecast_gap_angle=corrected,
            old_endpoint_chord_clearance=clearance(prop.merge_angle),forecast_endpoint_chord_clearance=clearance(corrected),
            missing_incoming=missing,stale_departing=stale,neighbors=details))
    assert before==_digest(canonical(state_record(env)))
result=dict(status='READ_ONLY_COMPLETE',reset_calls=0,step_calls=0,teacher_calls=0,candidate_calls=0,counts=counts,queries=reports)
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],counts=counts,output=str(out))))
