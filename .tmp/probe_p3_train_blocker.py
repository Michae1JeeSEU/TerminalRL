"""One preserved train snapshot, one candidate, native pair checks; no rollout."""
import argparse, hashlib, json, pickle, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terminal_marl.scripts.diagnose_p3_teacher_context import NativeWitness, dense_event, observation_context, state_delta
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical, state_record
from terminal_marl.core.safety_trace import _digest, _json
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.config import RingRole
from terminal_marl.core.merge_scheduler import RepairCandidate, LowLevelIntent
from terminal_marl.core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from terminal_marl.core.resource_context_observation import build_low_observation_v2_6
from terminal_marl.core.trajectory_context_observation import TrajectoryContextCache, FORECAST_SECONDS
from terminal_marl.training.v4_1_joint_teacher import _release_context
from terminal_marl.envs.parallel_env import WorldSnapshot

def run(episode, window, joint):
    start=time.monotonic()
    entry=json.loads((episode/'snapshots'/f'window_{window:02d}.json').read_text())
    label=json.loads((episode/'labels'/f'window_{window:02d}.json').read_text())
    row=entry['row'];assert row['partition']=='train' and row['actor_mask']
    path=episode/'snapshots'/entry['snapshot'];assert path.parent==episode/'snapshots'
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['snapshot_sha256']
    env=pickle.loads(raw)
    before=canonical(state_record(env));assert _digest(before)==row['pre_state_hash']==label['pre_state_hash']
    uid=int(row['uid']);ego=env.active[uid];target,_=_release_context(env,ego)
    assert target!='final', 'first output smoke intentionally uses a ring transfer'
    obs=build_low_observation_v2_6(ego,env)
    action=JOINT_ACTION_CATALOG_V4_1[joint];assert joint>0
    snapshot=WorldSnapshot(env.now,env.active,env.airspace)
    trace=NativeWitness(env.safety);trace.install()
    try:
        proposal,plan,reason=env._candidate_plan(ego,RepairCandidate('native_pair_probe',LowLevelIntent(*action.normalized),0.),target,snapshot,None)
    finally:trace.restore()
    assert plan is not None
    expected=next(x for x in label['teacher_audit'] if x['joint_index']==joint)
    assert expected['reason']==(reason or 'certified')
    after_query=canonical(state_record(env))
    delta=state_delta(before,after_query)
    pairs=[];slots=observation_context(env,ego);cache=TrajectoryContextCache(env)
    for other_id,other in env.active.items():
        if other_id==uid:continue
        pair_snapshot=WorldSnapshot(env.now,{uid:ego,other_id:other},env.airspace)
        safety=SafetyFilter(env.config);witness=NativeWitness(safety);witness.install()
        try:
            verdict=safety.filter_plan(plan,{other_id:other.committed_plan} if other.committed_plan else {},pair_snapshot)
        finally:witness.restore()
        events=[]
        for e in witness.events:
            d=dense_event(env,plan,snapshot,e,slots)
            # Compare the seven stored forecast knots with the exact predictor
            # at the native pair's dense minimum; never treat interpolation as safety.
            t=d['dense_minimum']['time'];knots=env.now+np.asarray((0.,)+FORECAST_SECONDS)
            forecast=cache.positions(other,knots)
            interpolated=np.array([np.interp(t,knots,forecast[:,axis]) for axis in range(3)])
            d['forecast_interpolation_error_at_min']=float(np.linalg.norm(interpolated-np.asarray(d['other_at_min'])))
            d['minimum_within_forecast_horizon']=bool(knots[0]<=t<=knots[-1])
            events.append(dict(**e,dense=d))
        pairs.append(dict(uid=other_id,approved=verdict.approved,reason=verdict.reason,
            current_distance=float(np.linalg.norm(other.position-ego.position)),
            present_in_set=other_id in obs.traffic_uids,events=events))
    blockers=[p['uid'] for p in pairs if not p['approved']]
    # The original canonical candidate stays byte-identical. This is a diagnostic
    # world with blockers removed, not a runnable scenario or new training label.
    remaining={i:u for i,u in env.active.items() if i not in blockers}
    clear=SafetyFilter(env.config).filter_plan(plan,{i:u.committed_plan for i,u in remaining.items() if u.committed_plan},WorldSnapshot(env.now,remaining,env.airspace))
    assert after_query==canonical(state_record(env))
    return dict(status='COMPLETE',decision_id=row['decision_id'],joint=joint,teacher_joint=label['teacher_joint_index'],
        reason=reason or 'certified',snapshot_sha256=entry['snapshot_sha256'],plan_hash=plan.trajectory_hash,
        duration=plan.end_time-plan.start_time,safe_distance=env.config.safe_distance,original_state_unchanged=True,
        counter_delta=delta,traffic_uids=obs.traffic_uids,slots={k:v['uid'] for k,v in slots.items()},
        pairs=pairs,blockers=blockers,without_blockers=dict(approved=clear.approved,reason=clear.reason),
        resets=0,steps=0,Q_accessed=False,elapsed_seconds=time.monotonic()-start)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--window',type=int,required=True)
    p.add_argument('--joint',type=int,default=13);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();result=run(a.episode,a.window,a.joint)
    with a.output.open('x',encoding='utf8') as f:json.dump(_json(result),f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','decision_id','joint','reason','blockers','without_blockers','elapsed_seconds')}))
