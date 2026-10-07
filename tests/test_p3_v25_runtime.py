from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.algo.trajectory_set_actor import TrajectorySetActor
from terminal_marl.configs.p3_integrated_repair import build_config
from terminal_marl.core.trajectory_context_observation import VERSION
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical,state_record
from terminal_marl.training.p3_v25_runtime import actor_eligible,request,check_request,load_actor
from terminal_marl.scripts.collect_p3_v3_expert import actor_eligible as collection_mask


@pytest.fixture
def env():
    cfg=replace(build_config(requests=6,warm_start=6,high_soc=True),warm_start_energy_reachability_check_enabled=False)
    return TerminalParallelEnv(cfg,seed=112233)


@pytest.mark.parametrize('deterministic',[True,False])
def test_request_mask_likelihood_k3_and_state_unchanged(env,monkeypatch,deterministic):
    model=TrajectorySetActor()
    with torch.no_grad():
        model.gate_head.weight.zero_();model.gate_head.bias.copy_(torch.tensor([-30.,30.]))
    env.active[0].safety_retry_after=env.now+10
    before=canonical(state_record(env))
    monkeypatch.setattr(env,'_candidate_plan',lambda *a,**k:pytest.fail('policy queried candidate'))
    torch.manual_seed(41)
    row=request(model,env,deterministic=deterministic)
    check_request(row)
    assert canonical(state_record(env))==before
    assert np.array_equal(row.actor_mask,[collection_mask(env,env.active[int(u)]) for u in row.uids])
    assert row.gates[0]==0 and row.proposals[0]==-1 and row.likelihoods['0']==dict(gate=0.,proposal=None,joint=0.)
    assert row.gates[1:].sum()==5
    assert all(len(v)==3 for v in row.shortlists.values())
    torch.manual_seed(41)
    second=request(model,env,deterministic=deterministic)
    assert second.tape()==row.tape() and second.likelihoods==row.likelihoods


def test_checkpoint_explicit_schema_and_new_information(tmp_path):
    model=TrajectorySetActor();path=tmp_path/'actor.pt'
    torch.save(dict(model=model.state_dict(),observation_schema=VERSION,arm='new_information'),path)
    loaded,_=load_actor(path)
    assert all(not p.requires_grad for p in loaded.parameters())
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    torch.save(dict(model=model.state_dict(),observation_schema=VERSION,arm='old_information'),path)
    with pytest.raises(ValueError): load_actor(path)


def test_smoke_uses_scheduler_and_exact_request_replay(tmp_path):
    from terminal_marl.scripts.evaluate_p3_v25_student import run
    path=tmp_path/'actor.pt'
    torch.manual_seed(112233);model=TrajectorySetActor()
    torch.save(dict(model=model.state_dict(),observation_schema=VERSION,arm='new_information'),path)
    result=run(SimpleNamespace(output=tmp_path/'smoke',checkpoint=path,seed=112233,policy_seed=112233,
               requests=3,warm_start=3,high_soc=True,deterministic=True,smoke=True))
    assert result['status']=='SMOKE_OK' and result['ticks']==2 and result['replay_exact']
    assert result['PPO_ALLOWED'] is False and result['error'] is None
    assert result['telemetry']['checked_executed_plan_hash_mismatches']==0
