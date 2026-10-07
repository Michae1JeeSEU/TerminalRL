"""One critic-only warmup from complete stochastic neutral-course episodes.

This entry is run only after the primary's source/episode review. It preserves
the actor and zero gate adapter, discards its critic optimizer, and grants no
PPO admission. PPO must collect fresh values/requests after this warmup.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..algo.trajectory_set_ppo import TrajectorySetActorCritic, actor_batch
from ..configs.p3_v25_course import PROFILE, SCENARIOS, build_config
from .p3_v25_runtime import load_actor
from .p3_v25_rollout import load_steps, critic_warmup
from .p3_v25_set_learnability import digest_state
from .p3_v25_task_imitation import frozen_models
from .p3_v25_task_student import PROTOCOL as STUDENT_PROTOCOL, blockers
from .p3_v25_imitation_qualification import sha
from .p3_v3_bc_learnability import require, read, write

PROTOCOL='p3_v25_neutral_critic_initialization_20261008_v1'


def run(fit_directory,development_directories,output):
    manifest=frozen_models(fit_directory)
    source=manifest['checkpoints'][0]
    require(len(development_directories)==3,'three complete course conditions required')
    actor,_=load_actor(Path(fit_directory)/source['path'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(93001)
        model=TrajectorySetActorCritic(actor)
    episodes,inputs=[],[]
    for directory,scenario in zip(development_directories,SCENARIOS):
        directory=Path(directory)
        protocol,report=read(directory/'protocol.json'),read(directory/'result.json')
        require(protocol['protocol']==report['protocol']==STUDENT_PROTOCOL and
            protocol['profile']==report['profile']==PROFILE and
            protocol['scenario']==report['scenario']==scenario and
            not protocol['smoke'] and not protocol['deterministic'] and
            protocol['source']==report['source']==source and
            report['status']=='DEVELOPMENT_COMPLETE' and not report['blockers'],
            'wrong source/profile or incomplete development episode')
        result=read(directory/'episode/result.json')
        require(result==report['rollout'] and not blockers(result),'invalid complete rollout')
        physics=read(directory/'episode/protocol.json')['physics_config']
        require(physics==json.loads(json.dumps(asdict(build_config(scenario)))),'course physics changed')
        rows=load_steps(directory/'episode/steps.pt')
        require(len(rows)==result['ticks'],'incomplete tensor episode')
        with torch.no_grad():
            for row in rows:
                batch=actor_batch([row])
                if batch is None:continue
                gate,proposal=model.distributions(*batch[:3])
                require(torch.allclose(gate.logits,batch[6],atol=1e-5,rtol=0) and
                    torch.allclose(proposal.logits,torch.from_numpy(row.old_proposal_logits[row.actor_mask]),atol=1e-5,rtol=0),
                    'development behavior differs from frozen source')
        episodes.append(rows)
        inputs.append(dict(directory=str(directory),scenario=scenario,seed=protocol['seed'],
            result_sha256=sha(directory/'result.json'),steps_sha256=sha(directory/'episode/steps.pt')))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(protocol=PROTOCOL,profile=PROFILE,source=source,inputs=inputs,
        updates=500,batch_size=32,learning_rate=.0003,gamma=.99,seed=93001,wall_seconds=600,
        optimizer='critic only; discarded after warmup',PPO_ALLOWED=False))
    before=digest_state(model.state_dict())
    report=critic_warmup(model,episodes,updates=500,batch_size=32,learning_rate=.0003,
                         gamma=.99,seed=93001,wall_seconds=600)
    require(all(torch.count_nonzero(p)==0 for p in model.gate_adapter.parameters()),'adapter changed')
    frozen_models(fit_directory)
    path=output/'critic_initialized.pt'
    torch.save(dict(protocol=PROTOCOL,profile=PROFILE,source=source,model=model.state_dict(),
                    optimizer_state_included=False),path)
    result=dict(status='CRITIC_WARMUP_COMPLETE',protocol=PROTOCOL,profile=PROFILE,source=source,
        inputs=inputs,warmup=report,initial_model_hash=before,final_model_hash=digest_state(model.state_dict()),
        checkpoint=path.name,checkpoint_sha256=sha(path),PPO_ALLOWED=False)
    write(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-directory',type=Path,required=True)
    parser.add_argument('--development-directories',type=Path,nargs=3,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    report=run(args.fit_directory,args.development_directories,args.output)
    print(json.dumps(dict(status=report['status'],PPO_ALLOWED=False)))


if __name__=='__main__':main()
