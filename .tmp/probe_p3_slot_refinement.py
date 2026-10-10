"""Private bounded slot refinement on one previously saved, verified state."""
import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import pickle
import sys
import textwrap
import time
from types import MethodType
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import terminal_marl.core.trajectory_generator as generator_module
from terminal_marl.core.config import RingRole
from terminal_marl.core.merge_scheduler import LowLevelIntent,RepairCandidate
from terminal_marl.core.safety_trace import _digest
from terminal_marl.envs.parallel_env import WorldSnapshot
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical,state_record,_json

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--syntax-only',action='store_true');a=p.parse_args()
out=Path(a.output);assert not out.exists()
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_student_dev_20261008/B_sto')
payload=(root/'snapshot_400.pkl').read_bytes()
expected=json.loads((root/'steps.jsonl').read_text().splitlines()[399])['post_hash']
source=textwrap.dedent(inspect.getsource(generator_module.TrajectoryGenerator._ring_transfer_v3))
marker='        if feasible_results:\n            chosen = min('
assert source.count(marker)==1
refinement='''        if feasible_results:
            ordered = sorted(feasible_results, key=lambda item: item[0])
            brackets = [(left,right) for left,right in zip(ordered,ordered[1:])
                if (left[1][1][-1].t-merge_time)*(right[1][1][-1].t-merge_time) < 0]
            if brackets:
                left,right = min(brackets,key=lambda pair: max(abs(pair[0][1][1][-1].t-merge_time),abs(pair[1][1][1][-1].t-merge_time)))
                for refinement_index in range(12):
                    moment = (left[0]+right[0])/2
                    try:
                        result = build(moment,entry_cap)
                    except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError):
                        break
                    item = (moment,result)
                    feasible_results.append(item)
                    probe_log.append(dict(departure=moment,end=float(result[1][-1].t),refined=True))
                    if (result[1][-1].t-merge_time)*(left[1][1][-1].t-merge_time)>0:
                        left=item
                    else:
                        right=item
            chosen = min('''
refined=source.replace(marker,refinement)
old='''                    abs(float(item[1][1][-1].t) - merge_time),
                    0 if float(item[1][1][-1].t) <= merge_time + 1e-6 else 1,'''
new='''                    0 if float(item[1][1][-1].t) >= merge_time - 1e-6 else 1,
                    abs(float(item[1][1][-1].t) - merge_time),'''
assert refined.count(old)==1;refined=refined.replace(old,new)
compile(refined,'private_refinement','exec')
if a.syntax_only:
    out.write_text(json.dumps(dict(status='SYNTAX_SMOKE_OK',queries=0)))
    print(out);raise SystemExit(0)
results=[]
for variant in ('current_forecast','refined_slot'):
    env=pickle.loads(payload);assert _digest(canonical(state_record(env)))==expected
    env.config.merge_gap_prediction_mode='committed_arrivals_v1'
    log=[]
    if variant=='refined_slot':
        namespace=dict(generator_module.__dict__,probe_log=log)
        exec(refined,namespace)
        env.generator._ring_transfer_v3=MethodType(namespace['_ring_transfer_v3'],env.generator)
    ego=env.active[114];target=RingRole.MIDDLE if ego.ring_role is RingRole.OUTER else RingRole.INNER
    assert ego.committed_plan is None and env.now==400
    snapshot=WorldSnapshot(env.now,env.active,env.airspace)
    requested=env.scheduler.propose(ego,LowLevelIntent(1.,0.,0.),snapshot,target)
    start=time.monotonic()
    prop,plan,reason=env._candidate_plan(ego,RepairCandidate('requested',LowLevelIntent(1.,0.,0.)),target,snapshot,None)
    results.append(dict(variant=variant,reason=reason,elapsed=time.monotonic()-start,requested=asdict(requested),
        returned=None if prop is None else asdict(prop),refinement_candidates=log,
        plan=None if plan is None else dict(start=plan.start_time,end=plan.end_time,duration=plan.end_time-plan.start_time,
            metadata=plan.metadata,hash=plan.trajectory_hash,checked=env.safety.checked_trajectory_hashes.get(plan.plan_id)),
        last_release=env.scheduler.last_release))
    out.write_text(json.dumps(_json(dict(status='PARTIAL' if len(results)<2 else 'COMPLETE',case='B_sto',tick=400,uid=114,
        reset_calls=0,step_calls=0,teacher_calls=0,maximum_extra_builds=12,results=results)),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status='COMPLETE',output=str(out),results=[dict(variant=r['variant'],reason=r['reason'],elapsed=r['elapsed'],end=r['plan']['end'] if r['plan'] else None) for r in results])))
