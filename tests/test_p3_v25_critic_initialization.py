from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from terminal_marl.algo.trajectory_set_actor import TrajectorySetActor
from terminal_marl.algo.trajectory_set_ppo import TrajectorySetActorCritic, SetStep
from terminal_marl.core.trajectory_context_observation import VERSION
from terminal_marl.training import p3_v25_critic_initialization as init
from terminal_marl.training.p3_v25_rollout import save_steps, critic_warmup
from terminal_marl.training.p3_v25_task_student import SAFETY_KEYS


def fixture(tmp_path,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(159)
    fit=tmp_path/'fit';fit.mkdir();actor=TrajectorySetActor()
    path=fit/'actor.pt'
    torch.save(dict(model=actor.state_dict(),observation_schema=VERSION,arm='new_information'),path)
    source=dict(seed=91901,path='actor.pt',sha256=init.sha(path))
    monkeypatch.setattr(init,'frozen_models',lambda _:dict(checkpoints=[source]))
    model=TrajectorySetActorCritic(actor)
    directories=[]
    for index,scenario in enumerate(init.SCENARIOS):
        directory=tmp_path/scenario;(directory/'episode').mkdir(parents=True)
        protocol=dict(protocol=init.STUDENT_PROTOCOL,profile=init.PROFILE,scenario=scenario,
                      source=source,smoke=False,deterministic=False,seed=892960+index)
        result=dict(status='COMPLETED',terminal=True,censored=False,error=None,replay_exact=True,
            telemetry={k:0 for k in SAFETY_KEYS},requested=1,unresolved=0,active=0,queue=0,service=0,
            counts=dict(actor=2),outcome_counts=dict(completed=1),ticks=2)
        init.write(directory/'protocol.json',protocol)
        init.write(directory/'result.json',dict(protocol,status='DEVELOPMENT_COMPLETE',blockers=[],rollout=result))
        init.write(directory/'episode/result.json',result)
        init.write(directory/'episode/protocol.json',dict(physics_config=asdict(init.build_config(scenario))))
        rows=[]
        for tick in range(2):
            ego=torch.randn(1,105);traffic=torch.zeros(1,0,50);mask=torch.zeros(1,0,dtype=torch.bool)
            with torch.no_grad():gd,pd=model.distributions(ego,traffic,mask)
            gate=np.array([1]);proposal=np.array([12])
            rows.append(SetStep(ego.numpy(),traffic.numpy(),mask.numpy(),np.zeros(26,np.float32),
                gate,proposal,gd.log_prob(torch.from_numpy(gate)).numpy(),pd.log_prob(torch.from_numpy(proposal)).numpy(),
                gd.logits.numpy(),pd.logits.numpy(),np.ones(1,bool),0.,0.,-1. if tick==0 else 3.,tick==1,tick==1))
        save_steps(directory/'episode/steps.pt',rows)
        directories.append(directory)
    return fit,directories,source


def test_warmup_artifact_preserves_actor_adapter_and_has_no_optimizer(tmp_path,monkeypatch):
    fit,dirs,source=fixture(tmp_path,monkeypatch)
    captured=[]
    def bounded(model,episodes,**kwargs):
        captured.append(kwargs.copy());kwargs['updates']=5
        return critic_warmup(model,episodes,**kwargs)
    monkeypatch.setattr(init,'critic_warmup',bounded)
    before=[init.sha(p/'episode/steps.pt') for p in dirs]
    result=init.run(fit,dirs,tmp_path/'warmup')
    assert result['status']=='CRITIC_WARMUP_COMPLETE' and not result['PPO_ALLOWED']
    assert result['warmup']['team_steps']==6 and captured[0]['updates']==500
    assert captured[0]['gamma']==.99 and captured[0]['wall_seconds']==600
    payload=torch.load(tmp_path/'warmup/critic_initialized.pt',weights_only=True)
    base=torch.load(fit/'actor.pt',weights_only=True)['model']
    assert all(torch.equal(v,payload['model']['actor.'+k]) for k,v in base.items())
    assert all(torch.count_nonzero(v)==0 for k,v in payload['model'].items() if k.startswith('gate_adapter.'))
    assert not payload['optimizer_state_included'] and 'optimizer' not in payload
    assert init.sha(fit/'actor.pt')==source['sha256']
    assert before==[init.sha(p/'episode/steps.pt') for p in dirs]


@pytest.mark.parametrize('change',('task_profile','deterministic','failed_energy'))
def test_wrong_task_or_incomplete_episode_cannot_initialize_critic(tmp_path,monkeypatch,change):
    fit,dirs,_=fixture(tmp_path,monkeypatch)
    path=dirs[0]/'protocol.json';value=init.read(path)
    if change=='task_profile':value['profile']='historical_priority_course'
    elif change=='deterministic':value['deterministic']=True
    else:
        report=init.read(dirs[0]/'result.json');report['rollout']['outcome_counts']={'failed_energy':1}
        init.write(dirs[0]/'result.json',report);init.write(dirs[0]/'episode/result.json',report['rollout'])
    init.write(path,value)
    monkeypatch.setattr(init,'critic_warmup',lambda *a,**k:pytest.fail('invalid episode reached optimizer'))
    with pytest.raises(ValueError):init.run(fit,dirs,tmp_path/'warmup')
    assert not (tmp_path/'warmup').exists()
