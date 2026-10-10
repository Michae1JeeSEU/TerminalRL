from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import math,pickle
import numpy as np
from terminal_marl.core.trajectory_generator import TrajectoryPoint, TrajectoryPlan
from terminal_marl.scripts.diagnose_p3_inner_exit import measure_candidate
from terminal_marl.scripts.diagnose_p3_integrated_repair import write_json
source=Path(sys.argv[1]); output=Path(sys.argv[2])
if output.exists():raise SystemExit('preserve existing output')
with source.open('rb') as stream:env=pickle.load(stream)
u=min(env.active.values(),key=lambda u:(u.soc,u.uid))
ring=env.airspace.actual_ring(u.ring_role)
assert u.uid==6 and abs(env.now-600.)<1e-9 and abs(u.speed-16.)<1e-9
lz_id=env.lz_manager.rank_available_zones(u,env.now)[0]
assert lz_id==1
T=60.; goal_radius=100.; dr=ring.radius-goal_radius; omega=u.speed/ring.radius
points=[]
for t in np.linspace(0.,T,int(round(T/env.config.physics_dt))+1):
 s=t/T; q=10*s**3-15*s**4+6*s**5
 q1=30*s**2-60*s**3+30*s**4; q2=60*s-180*s**2+120*s**3
 r=ring.radius-dr*q; r1=-dr*q1/T; r2=-dr*q2/T**2
 theta=u.ring_progress+omega*t
 er=np.array([math.cos(theta),math.sin(theta),0.]); et=np.array([-math.sin(theta),math.cos(theta),0.])
 p=r*er+np.array([0.,0.,ring.height]); v=r1*er+r*omega*et
 a=(r2-r*omega**2)*er+2*r1*omega*et
 points.append(TrajectoryPoint(env.now+t,p,v,a))
shadow=deepcopy(u);end=points[-1]
shadow.position=end.position.copy();shadow.velocity=end.velocity.copy();shadow.acceleration=end.acceleration.copy()
shadow.speed=float(np.linalg.norm(end.velocity[:2]));shadow.ring_progress=u.ring_progress+omega*T
inner=replace(ring,radius=goal_radius)
construction_airspace=SimpleNamespace(actual_ring=lambda role:inner)
reservation=SimpleNamespace(merge_time=end.t,reservation_id='static_spiral_probe')
tail=env.generator.final_approach(shadow,env.lz_manager.zones[lz_id],reservation,end.t,airspace=construction_airspace)
metadata=dict(tail.metadata);metadata.pop('trajectory_hash',None)
metadata.update(scheduled_departure=env.now,diagnostic_only=True,spiral_duration=T,spiral_target_radius=goal_radius)
plan=TrajectoryPlan('static_spiral_probe',u.uid,'final_approach',tuple(points[:-1])+tail.points,env.now,tail.end_time,metadata=metadata)
result=dict(snapshot=str(source),uid=u.uid,lz_id=lz_id,variant='diagnostic_only_constant_angular_rate_inward',production_modified=False,PPO_ALLOWED=False)
result.update(measure_candidate(env,u,plan));write_json(output,result);print(output)
