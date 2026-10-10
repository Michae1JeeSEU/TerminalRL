"""One native candidate witness; no reset/step or modified safety return."""
import argparse
from pathlib import Path
import pickle
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from terminal_marl.core.config import RingRole
from terminal_marl.core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from terminal_marl.core.merge_scheduler import LowLevelIntent,RepairCandidate
from terminal_marl.core.safety_trace import _digest
from terminal_marl.core.trajectory_context_observation import TrajectoryContextCache
from terminal_marl.envs.parallel_env import WorldSnapshot
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical,state_record,write_json
from terminal_marl.scripts.diagnose_p3_teacher_context import NativeWitness,read,sha,state_delta,dense_event,grid,minimum

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--tick',type=int,required=True);p.add_argument('--uid',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
r=dict(status='ERROR',error=None,reset_count=0,step_count=0,training=False,tick=a.tick,uid=a.uid)
started=time.monotonic()
try:
    complete=read(a.source/'result.json')
    assert complete['status']=='DIAGNOSIS_COMPLETE' and complete['exact_replay']
    original=read(a.source/f'query_{a.tick}_{a.uid}.json')
    path=a.source/f'state_{a.tick}.pkl';env=pickle.loads(path.read_bytes())
    assert env.now==a.tick and env.active[a.uid].soc==original['soc']
    before=canonical(state_record(env));obs=TrajectoryContextCache(env).build(env.active[a.uid])
    snap=WorldSnapshot(env.now,env.active,env.airspace);trace=NativeWitness(env.safety)
    action=JOINT_ACTION_CATALOG_V4_1[13]
    trace.install()
    try:
        proposal,plan,reason=env._candidate_plan(env.active[a.uid],RepairCandidate('native_witness',LowLevelIntent(*action.normalized),0.),RingRole.MIDDLE,snap,None)
    finally:trace.restore()
    delta=state_delta(before,canonical(state_record(env)))
    expected=next(x for x in original['audit'] if x['joint_index']==13)
    assert (reason or 'certified')==expected['reason'] and (reason is None)==expected['feasible']
    assert plan is not None and trace.events
    event=trace.events[-1];dense=dense_event(env,plan,snap,event,{})
    times=grid(event['start'],event['end'],env.config.physics_dt)
    left=np.asarray(plan.sample_positions(times) if event['stage']=='plan' else env.safety._post_plan_positions(plan,snap,times))
    other=env.active[event['uid']]
    right=np.asarray(env.safety._committed_positions(other.committed_plan,snap,times) if other.committed_plan is not None else env.safety._holder_positions(other,snap,times))
    ld=env.safety._track_descriptor(plan) if event['stage']=='plan' else 'holder_proxy'
    rd=env.safety._track_descriptor(other.committed_plan) if other.committed_plan is not None else 'holder_proxy'
    point,swept=env.safety._track_distances_with_margin(left,right,times,ld,rd)
    i=obs.traffic_uids.index(event['uid'])
    r.update(status='WITNESS_COMPLETE',snapshot_sha256=sha(path),pre_hash=_digest(before),
        candidate_calls=1,native_reason=reason or 'certified',native_events=trace.events,dense=dense,
        nominal_track_bound=dict(point=minimum(point,times),swept=minimum(swept,times[:-1])),
        observation=dict(shape=list(obs.ego.shape),traffic_shape=list(obs.traffic.shape),
            blocker_in_traffic=True,blocker_row=obs.traffic[i],target_uids=obs.target_uids),
        plan=dict(start=plan.start_time,end=plan.end_time,hash=plan.trajectory_hash,
            dynamics=env.safety.dynamics.validate_trajectory(plan),metadata=plan.metadata),
        candidate_state_counter_delta=delta)
except Exception as error:
    r['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
r['elapsed_seconds']=time.monotonic()-started
write_json(a.output/'result.json',r)
print(r['status'],r['error'])
raise SystemExit(0 if r['error'] is None else 1)
