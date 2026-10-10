import copy,hashlib,json,pickle,sys
from itertools import combinations
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.core.runtime_separation import swept_minimum_distance
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_neutral_course_v2_20261008/Q_dense24_fixed/seed_895502')
raw=(root/'failure_before.pkl').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='7ca4cd7069cd14e4101da1916f948f14da3f72d334ff8e1e4b96dc6d67d81c1f'
env=pickle.loads(raw);tape=json.loads((root/'failure_request.json').read_text())
assert hashlib.sha256((root/'failure_request.json').read_bytes()).hexdigest()=='150a9192a4dfb5ddcc8682ec8d8981b3e7af966742aaee4ab17006e56239ea5e'
captured={};original=env.runtime_traffic.protect
def protected(world,intents,traces):
    result=original(world,intents,traces)
    captured['now']=world.now
    captured['active']=copy.deepcopy(world.active)
    for dt in (.2,.005):
        times,tracks=world.runtime_traffic._joint_position_tracks(world,{},set(world.active),horizon=world.config.dt,sample_dt=dt)
        captured[dt]=(times,tracks)
    return result
env.runtime_traffic.protect=protected
original_observe=env.separation_monitor.observe
def observe(states,previous,**kwargs):
    captured['pairs']=[]
    for left,right in combinations(sorted(states),2):
        minimum,fraction=swept_minimum_distance(previous[left],states[left].position,previous[right],states[right].position)
        if minimum<80.:
            captured['pairs'].append(dict(pair=(left,right),swept=minimum,fraction=fraction,
                actual_execution_tracks=None if not kwargs else {uid:kwargs['executed_tracks'][uid] for uid in (left,right)},
                actual=float(np.linalg.norm(states[left].position-states[right].position)),
                previous_left=previous[left],previous_right=previous[right],
                end_left=states[left].position.copy(),end_right=states[right].position.copy()))
    return original_observe(states,previous,**kwargs)
env.separation_monitor.observe=observe
actions={uid:np.array(row,np.float32) for uid,row in tape['actions'].items()}
shortlists={uid:tuple(np.array(row,np.float32) for row in rows) for uid,rows in tape['shortlists'].items()}
env.step(actions,shortlists)
result=dict(now=env.now,pairs=captured['pairs'],monitor_telemetry=env.separation_monitor.telemetry,
    prediction_execution_error=env.runtime_trace[-1]['prediction_execution_position_error'],
    runtime_gate=env.runtime_traffic.last_recursive_gate)
for item in result['pairs']:
    l,r=item['pair'];item['curves']={}
    for dt in (.2,.005):
        times,tracks=captured[dt]
        point,swept=env.runtime_traffic._position_track_distances(tracks[l],tracks[r])
        item['curves'][str(dt)]=dict(point_min=point.min(),swept_min=swept.min(),time=times[int(np.argmin(point))],
            endpoint_error_l=float(np.linalg.norm(tracks[l][-1]-item['end_left'])),
            endpoint_error_r=float(np.linalg.norm(tracks[r][-1]-item['end_right'])))
    item['states']=[]
    for uid in (l,r):
        u=captured['active'][uid];p=u.committed_plan
        item['states'].append(dict(uid=uid,position=u.position,speed=u.speed,theta=u.ring_progress,phase=u.phase,ring=u.ring_role,
            plan=None if p is None else dict(start=p.start_time,end=p.end_time,hash=p.trajectory_hash,metadata=p.metadata)))
print(json.dumps(canonical(result),indent=2))
