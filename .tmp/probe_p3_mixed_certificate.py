import hashlib,json,pickle,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_neutral_course_v2_20261008/student_initial/dense24_mixed/episode')
raw=(root/'failure_after.pkl').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='e8280556cb012bfde56d1b540708bb68f62e42a5af623a92664015c517b1c517'
env=pickle.loads(raw); shield=env.runtime_traffic; safety=env.safety
u=env.active[2]; plan=env.active[31].committed_plan
snap=SimpleNamespace(now=env.now,active=env.active,airspace=env.airspace)
times,tracks=shield._joint_position_tracks(env,{}, {2,31})
info=dict(shield._joint_track_info)
point,swept=shield._position_track_distances(tracks[2],tracks[31])
correction=np.array([shield._pair_curve_error(info[2],info[31],float(dt)) for dt in np.diff(times)])
idx=int(np.argmin(swept-correction))
exact=np.asarray(safety._holder_positions(u,snap,times,0.0))
ep,es=safety._track_distances_with_margin(exact,tracks[31],times,'holder_proxy','flight_v3')
result=dict(now=env.now,physics_dt=env.config.physics_dt,safety_dt=env.config.safety_sampling_dt,
    residual_limit=env.config.max_speed_residual_mps,info=info,
    runtime=dict(point_min=point.min(),swept_min=swept.min(),bound_min=(swept-correction).min(),
    interval=times[idx:idx+2],interval_raw=swept[idx],interval_margin=correction[idx]),
    direct_holder=dict(max_sample_error=np.linalg.norm(exact-tracks[2],axis=1).max(),point_min=ep.min(),bound_min=es.min()))
for dt in (.2,.1,.02,.005):
    ts=np.arange(plan.start_time,plan.end_time+1e-9,dt)
    if ts[-1]<plan.end_time: ts=np.append(ts,plan.end_time)
    left=np.asarray(safety._holder_positions(u,snap,ts,0.0));right=plan.sample_positions(ts)
    p,s=safety._track_distances_with_margin(left,right,ts,'holder_proxy','flight_v3')
    result[str(dt)]=dict(point_min=p.min(),bound_min=s.min(),time=ts[int(np.argmin(p))])
other={uid:v.committed_plan for uid,v in env.active.items() if v.committed_plan is not None and uid!=31}
result['filter_plan']=safety.filter_plan(plan,other,snap)
print(json.dumps(canonical(result),indent=2))
