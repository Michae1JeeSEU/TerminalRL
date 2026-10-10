"""Bounded old-state/new-contract query. Never modifies original artifacts."""
import argparse
from pathlib import Path
import pickle
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.scripts.diagnose_p3_teacher_context import NativeWitness,read,sha
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical,state_record,write_json
from terminal_marl.core.config import RingRole
from terminal_marl.core.merge_scheduler import LowLevelIntent,RepairCandidate
from terminal_marl.envs.parallel_env import WorldSnapshot
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--tick',type=int,required=True);p.add_argument('--uid',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--nominal-only',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
r=dict(status='ERROR',error=None,reset_count=0,step_count=0,training=False,tick=a.tick,uid=a.uid)
try:
    assert read(a.source/'result.json')['status']=='DIAGNOSIS_COMPLETE'
    path=a.source/f'state_{a.tick}.pkl';env=pickle.loads(path.read_bytes())
    old=read(a.source/f'query_{a.tick}_{a.uid}.json')
    assert env.now==a.tick and env.active[a.uid].soc==old['soc']
    assert env.config is env.safety.config and env.config is env.priority.config
    env.config.nominal_holder_separation_mode='swept_curve_v1';env.config.priority_wait_mode='total_wait_v1'
    for u in env.active.values():env.priority.update(u,env.now)
    before=canonical(state_record(env));certs=[];native=env.safety.filter_plan
    def filtered(plan,committed,snapshot):
        trace=NativeWitness(env.safety);trace.install()
        try:verdict=native(plan,committed,snapshot)
        finally:trace.restore()
        certs.append(dict(approved=verdict.approved,reason=verdict.reason,
            trajectory_hash=plan.trajectory_hash,checked_hash=verdict.checked_trajectory_hash,events=trace.events))
        return verdict
    env.safety.filter_plan=filtered
    if a.nominal_only:
        proposal,plan,reason=env._candidate_plan(env.active[a.uid],RepairCandidate('repaired_query',LowLevelIntent(1.,0.,0.),0.),RingRole.MIDDLE,WorldSnapshot(env.now,env.active,env.airspace),None)
        result=dict(nominal_reason=reason or 'certified',nominal_feasible=reason is None)
    else:
        label,audit=teacher_joint_expert_v4_1(env,env.active[a.uid],state_only=True)
        result=dict(joint_index=label.index,request=label.normalized,audit=audit)
    after=canonical(state_record(env))
    before.pop('telemetry');after.pop('telemetry');assert before==after,'physical state changed'
    assert all(not c['approved'] or c['checked_hash']==c['trajectory_hash'] for c in certs)
    r.update(status='QUERY_COMPLETE',nominal_only=a.nominal_only,snapshot_sha256=sha(path),
        old_joint_index=old['joint_index'],**result,certificates=certs)
except Exception as error:r['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
r['elapsed_seconds']=time.monotonic()-started;write_json(a.output/'result.json',r)
print(r['status'],r['error']);raise SystemExit(0 if r['error'] is None else 1)
