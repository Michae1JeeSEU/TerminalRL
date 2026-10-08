from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.algo.trajectory_set_actor import TrajectorySetActor
from terminal_marl.algo.trajectory_set_ppo import SetStep,TrajectorySetActorCritic
from terminal_marl.algo.joint_ppo_v4_2 import hierarchical_joint_logprob
from terminal_marl.core.trajectory_context_observation import VERSION
from terminal_marl.training import p3_v25_ppo_block as block
from terminal_marl.training.p3_v25_task_student import SAFETY_KEYS


def complete(count):
    rows=[dict(outcome='completed',energy_consumed=12.,wait_seconds=float(i),
        request_to_touchdown_seconds=float(i+20),request_to_completion_seconds=float(i+50)) for i in range(count)]
    return dict(status='COMPLETED',terminal=True,censored=False,error=None,replay_exact=True,
        telemetry={k:0 for k in SAFETY_KEYS},requested=count,unresolved=0,active=0,queue=0,service=0,
        counts=dict(actor=2),outcome_counts=dict(completed=count),ticks=2,reward=2.,
        task_metrics=dict(completed_rows=rows))


def fixture(tmp_path,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(719)
    fit=tmp_path/'fit';fit.mkdir();actor=TrajectorySetActor()
    path=fit/'actor.pt';torch.save(dict(model=actor.state_dict(),observation_schema=VERSION,arm='new_information'),path)
    source=dict(seed=91901,path='actor.pt',sha256=block.sha(path))
    model=TrajectorySetActorCritic(actor);warm=tmp_path/'critic.pt'
    torch.save(dict(protocol=block.CRITIC_PROTOCOL,profile=block.PROFILE,source=source,
        model=model.state_dict(),optimizer_state_included=False),warm)
    baseline=[dict(scenario=s,seed=892970+i,policy_seed=392970+i,result=complete(block.build_config(s).total_requested_uavs))
              for i,s in enumerate(block.SCENARIOS)]
    monkeypatch.setattr(block,'input_review',lambda *a:(source,warm,baseline))
    monkeypatch.setattr(block,'frozen_models',lambda *a:dict(checkpoints=[source]))
    reference=(torch.randn(3,105),torch.zeros(3,0,50),torch.zeros(3,0,dtype=torch.bool))
    def data(directories):
        assert directories==('train0','train1','train2')
        return SimpleNamespace(new=reference)
    monkeypatch.setattr(block,'load_train',data)
    calls=[]
    def rollout(model,config,*,seed,policy_seed,output,max_steps):
        calls.append(dict(seed=seed,policy_seed=policy_seed,model=block.digest_state(model.state_dict())))
        assert not config.task_priority_enabled and config.low_level_priority_time_weight==0
        assert max_steps==config.max_simulation_seconds/config.dt
        output.mkdir();rows=[]
        for tick in range(2):
            ego=torch.randn(1,105);traffic=torch.zeros(1,0,50);mask=torch.zeros(1,0,dtype=torch.bool)
            with torch.no_grad():
                gd,pd=model.distributions(ego,traffic,mask);g=gd.sample()
                p=pd.sample() if g.item() else torch.tensor([-1])
                gl,pl,_=hierarchical_joint_logprob(gd,pd,g,p)
            rows.append(SetStep(ego.numpy(),traffic.numpy(),mask.numpy(),np.zeros(26,np.float32),
                g.numpy(),p.numpy(),gl.numpy(),pl.numpy(),gd.logits.numpy(),pd.logits.numpy(),
                np.ones(1,bool),0.,0.,float(tick+1),tick==1,tick==1))
        return rows,complete(config.total_requested_uavs)
    monkeypatch.setattr(block,'collect_episode',rollout)
    return fit,source,actor,calls


def test_three_bounded_updates_then_paired_eval_never_promote(tmp_path,monkeypatch):
    fit,source,actor,calls=fixture(tmp_path,monkeypatch)
    result=block.run(fit,'q','warm',(),('train0','train1','train2'),tmp_path/'pilot')
    assert result['status']=='PILOT_COMPLETE' and not result['formal_allowed'] and not result['candidate_retained']
    assert [c['seed'] for c in calls[:3]]==list(block.TRAIN_SEEDS)
    assert [c['seed'] for c in calls[3:]]==[892970,892971,892972]
    assert result['accepted_updates']>0 and len(result['evaluation'])==3
    final=torch.load(tmp_path/'pilot/after_03.pt',weights_only=True)
    assert all(torch.equal(v,final['model']['actor.'+k]) for k,v in actor.state_dict().items())
    assert final['completed_cycles']==3 and final['optimizer']['state']
    assert block.sha(fit/'actor.pt')==source['sha256']
    assert result['baseline_aggregate']['normal']==result['candidate_aggregate']['normal']==54
    assert result['baseline_aggregate']['delay']['wait_seconds']['max']==23.
    before=block.sha(tmp_path/'pilot/result.json')
    with pytest.raises(ValueError,match='already exists'):
        block.run(fit,'q','warm',(),('train0','train1','train2'),tmp_path/'pilot')
    assert block.sha(tmp_path/'pilot/result.json')==before


def test_update_timeout_retains_checkpoint_and_stops_next_cycle(tmp_path,monkeypatch):
    fit,_,_,calls=fixture(tmp_path,monkeypatch)
    original=block.TrajectorySetPPO.update
    def expired(self,steps,**kwargs):
        kwargs['wall_seconds']=1e-12
        return original(self,steps,**kwargs)
    monkeypatch.setattr(block.TrajectorySetPPO,'update',expired)
    result=block.run(fit,'q','warm',(),('train0','train1','train2'),tmp_path/'pilot')
    assert result['status']=='BLOCKED' and result['blockers']==['train_1:update_budget_exhausted']
    assert len(calls)==1 and (tmp_path/'pilot/after_01.pt').exists()


def test_failed_qualification_stops_before_reading_critic_or_constructing_optimizer(tmp_path,monkeypatch):
    monkeypatch.setattr(block,'frozen_models',lambda _:dict(checkpoints=[dict(seed=91901)]))
    q=tmp_path/'Q';q.mkdir()
    block.write(q/'result.json',dict(protocol=block.FIT_PROTOCOL,profile=block.PROFILE,
        status='EVALUATED',parameter_updates=0,all_imitation_gates_passed=False))
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('invalid qualification reached optimizer'))
    with pytest.raises(ValueError,match='qualification missing or failed'):
        block.input_review('fit',q,'nonexistent',())
