"""Read-only action-tape diagnosis of radial interpolation excursion."""
import argparse,json
from pathlib import Path
import numpy as np
from terminal_marl.training.run_v4_2_ppo_p2 import p2_config
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.core.safety_trace import SafetyTraceRecorder


def run(tape,output):
    env=TerminalParallelEnv(p2_config(12));worst={'excursion_m':0.}
    def observe(world):
        for u in world.active.values():
            p=u.committed_plan
            if p is None or p.mode!='ring_transfer':continue
            radius=float(np.linalg.norm(p.points[-1].position[:2]));excursion=radius-float(np.linalg.norm(u.position[:2]))
            if excursion<=worst['excursion_m']:continue
            xy=np.asarray([q.position[:2] for q in p.points]);segments=np.diff(xy,axis=0)
            t=np.clip(-np.sum(xy[:-1]*segments,axis=1)/np.maximum(np.sum(segments**2,axis=1),1e-20),0.,1.)
            r=np.linalg.norm(xy[:-1]+t[:,None]*segments,axis=1);i=int(r.argmin())
            worst.update(excursion_m=excursion,uid=u.uid,time=world.now,target_radius=radius,geometry=p.metadata['transfer_geometry'],
                minimum_node_radius=float(np.linalg.norm(xy,axis=1).min()),minimum_segment_radius=float(r[i]),
                segment=xy[i:i+2].tolist(),segment_t=float(t[i]),path=[q.position.tolist() for q in p.points],
                metadata={k:v for k,v in p.metadata.items() if isinstance(v,(str,int,float,bool))})
    env.set_step_observer(observe);replay=SafetyTraceRecorder.replay_action_tape(env,tape)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(dict(exact=replay['exact'],worst=worst),indent=2))
    print(json.dumps(dict(exact=replay['exact'],worst={k:v for k,v in worst.items() if k not in ('path','metadata')})))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('tape','output'):p.add_argument('--'+n,type=Path,required=True)
    run(**vars(p.parse_args()))
