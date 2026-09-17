"""Replay the failed P2 action tape to isolate batch-shape likelihood rounding."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from terminal_marl.training.run_v4_2_ppo_p2 import p2_config
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.algo.joint_ppo_v4_2 import hierarchical_joint_logprob
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def run(checkpoint,tape,output):
    policy=StudentPPOV42(88);config=p2_config(12)
    load_checkpoint(checkpoint,policy,config,load_optimizer=True)
    data=json.loads(tape.read_text());env=TerminalParallelEnv(config);obs,_=env.reset(data['episode_seed'])
    maximum_full=maximum_subset=0.;worst=None
    for row in data['action_tape']:
        ids=sorted(row['a_requested'])
        if ids:
            world=np.asarray([obs[uid] for uid in ids],dtype=np.float32)
            mask=np.asarray([env.active[int(uid)].committed_plan is None and env.active[int(uid)].phase.value in {'ring_outer','ring_middle','ring_inner'} and int(uid) not in env.future_release_reservations for uid in ids])
            gate=torch.tensor([row['requested_gate'][uid] for uid in ids])
            prop=torch.tensor([-1 if row['requested_proposal'][uid] is None else row['requested_proposal'][uid] for uid in ids])
            old=np.asarray([row['joint_log_prob'][uid] for uid in ids])
            if mask.any():
                with torch.no_grad():
                    gd,pd=policy.model.distributions(torch.as_tensor(world))
                    _,_,lp=hierarchical_joint_logprob(gd,pd,gate,prop)
                    gd,pd=policy.model.distributions(torch.as_tensor(world[mask]))
                    _,_,sub=hierarchical_joint_logprob(gd,pd,gate[mask],prop[mask])
                full=float(np.max(np.abs(lp.numpy()[mask]-old[mask])))
                subset=float(np.max(np.abs(sub.numpy()-old[mask])))
                maximum_full=max(maximum_full,full)
                if subset>maximum_subset:
                    maximum_subset=subset;worst=dict(step=row['step'],world_size=len(ids),actors=int(mask.sum()),full_error=full,subset_error=subset)
        obs,*_=env.step({uid:np.asarray(a,dtype=np.float32) for uid,a in row['a_requested'].items()})
    result=dict(diagnostic_only=True,seed=data['episode_seed'],steps=len(data['action_tape']),full_world_error=maximum_full,actor_subset_error=maximum_subset,worst=worst)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('checkpoint','tape','output'):p.add_argument('--'+n,type=Path,required=True)
    torch.set_num_threads(1);run(**vars(p.parse_args()))
