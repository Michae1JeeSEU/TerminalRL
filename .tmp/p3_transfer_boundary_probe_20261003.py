from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from terminal_marl.configs.p3_integrated_repair import build_config
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from terminal_marl.core.config import RingRole
from terminal_marl.core.merge_scheduler import LowLevelIntent, RepairCandidate
from terminal_marl.scripts.diagnose_p3_integrated_repair import write_json
out=Path(sys.argv[1])
if out.exists(): raise SystemExit('output exists')
env=TerminalParallelEnv(build_config(requests=12,warm_start=6,high_soc=True),seed=890501)
rows=[]
for uid,target in [(0,RingRole.MIDDLE),(4,RingRole.INNER)]:
 for residual in (0.,-1.,1.):
  u=env.active[uid]
  _,p,reason=env._candidate_plan(u,RepairCandidate('boundary_probe',LowLevelIntent.from_array(np.array([1.,residual,0.]))),target,WorldSnapshot(env.now,env.active,env.airspace),None)
  row=dict(uid=uid,residual=residual,reason=reason)
  if p is not None:
   row['dynamics_errors']=env.generator.dynamics.validate_trajectory(p)
   diffs=[]
   for i,seg in enumerate(p.polynomial_segments):
    for name,t,point in [('start',seg.start_time,p.points[i]),('end',seg.end_time,p.points[i+1])]:
     observed=seg.sample(t)
     delta=[float(np.max(np.abs(a-b))) for a,b in zip(observed,(point.position,point.velocity,point.acceleration))]
     if max(delta)>1e-6:
      diffs.append(dict(segment=i,side=name,time=t,duration=seg.duration,pva_max_errors=delta,expected=[x.tolist() for x in (point.position,point.velocity,point.acceleration)],observed=[x.tolist() for x in observed]))
   row['endpoint_differences']=diffs
   row['minimum_segment_duration']=min(x.duration for x in p.polynomial_segments)
  rows.append(row)
write_json(out,dict(rows=rows,PPO_ALLOWED=False,trained=False))
print(out)
