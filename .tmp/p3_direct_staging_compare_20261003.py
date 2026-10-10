from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pickle,math
import numpy as np
from terminal_marl.core.trajectory_generator import TrajectoryPoint,TrajectoryPlan
from terminal_marl.core.trajectory_time_parameterization import GeometricPath,TrajectoryConstraints
from terminal_marl.scripts.diagnose_p3_inner_exit import measure_candidate
from terminal_marl.scripts.diagnose_p3_integrated_repair import write_json
source=Path(sys.argv[1]);out=Path(sys.argv[2]);assert not out.exists()
with source.open('rb') as f:env=pickle.load(f)
rows=[]
for uid in sorted(env.active):
 u=env.active[uid]
 for lzid in env.lz_manager.rank_available_zones(u,env.now):
  z=env.lz_manager.zones[lzid];staging=np.array([z.position[0],z.position[1],u.position[2]]);distance=np.linalg.norm(staging-u.position)
  base=max(2*u.speed/1.5,1.875*distance/10.,1.)
  attempts=[];chosen=None
  for factor in (1.,1.1,1.21,1.331,1.4641,1.61051,1.771561,1.9487171,2.14358881,2.357947691):
   T=base*factor
   points=(TrajectoryPoint(env.now,u.position,u.velocity,u.acceleration),TrajectoryPoint(env.now+T,staging,np.zeros(3),np.zeros(3)))
   plan=TrajectoryPlan('direct_probe',uid,'final_approach',points,env.now,env.now+T,metadata=dict(trajectory_schema_version='executable_trajectory_v3',target_phase='landing',scheduled_departure=env.now,lz_id=lzid))
   errors=env.generator.dynamics.validate_trajectory(plan)
   ts=np.linspace(env.now,env.now+T,501);positions=plan.sample_positions(ts)
   attempts.append(dict(duration=T,errors=errors,max_radius=float(np.max(np.linalg.norm(positions[:,:2],axis=1)))))
   if not errors:
    leg=env.generator.parameterizer.parameterize(GeometricPath.line(staging,z.position,samples=17),start_time=env.now+T,initial_velocity=np.zeros(3),initial_acceleration=np.zeros(3),target_horizontal_speed=5.,constraints=TrajectoryConstraints(end_horizontal_speed=0.,tangent_yaw=False,end_direction=np.array([0.,0.,-1.]),end_curvature_vector=np.zeros(3)))
    plan=TrajectoryPlan('direct_staging_probe',uid,'final_approach',points[:-1]+tuple(leg),env.now,leg[-1].t,metadata=dict(trajectory_schema_version='executable_trajectory_v3',target_phase='landing',scheduled_departure=env.now,lz_id=lzid))
    chosen=measure_candidate(env,u,plan);break
  rows.append(dict(uid=uid,lz_id=lzid,attempts=attempts,chosen=chosen))
write_json(out,dict(rows=rows,PPO_ALLOWED=False,trained=False));print(out)
