"""Complete-cycle V2.5 team rollout and critic-only initialization.

These functions expose no automatic admission or experiment entry point.
Failed/incomplete episodes remain artifacts and cannot initialize the critic.
"""
from dataclasses import fields
from collections import Counter
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import torch

from ..algo.trajectory_set_ppo import SetStep,tensors
from ..core.safety_trace import _digest,_json
from ..envs.parallel_env import TerminalParallelEnv
from ..scripts.diagnose_p3_direct_contract import canonical,execute_tape,state_record,require,write_json
from ..scripts.collect_p3_v3_sparse_expert import natural_terminal
from .p3_v25_runtime import request,check_request
from .p3_v25_training import central_context,step_record
from .p3_v25_set_learnability import digest_state


@torch.no_grad()
def world_value(model,env):
    from ..core.trajectory_context_observation import pad_trajectory_context
    observations=env.trajectory_context_observations()
    ordered=[observations[u] for u in sorted(observations,key=int)]
    if ordered:
        data=tuple(torch.from_numpy(a) for a in pad_trajectory_context(ordered))
    else:
        data=(torch.empty(0,105),torch.empty(0,0,50),torch.empty(0,0,dtype=torch.bool))
    return float(model.value(*data,torch.from_numpy(central_context(env))))


def save_steps(path,steps):
    # Only tensors and builtin scalars, readable with weights_only=True.
    payload=[]
    for row in steps:
        item={f.name:getattr(row,f.name) for f in fields(SetStep)}
        payload.append({k:torch.from_numpy(v.copy()) if isinstance(v,np.ndarray) else v for k,v in item.items()})
    torch.save(payload,path)


def load_steps(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    rows=[SetStep(**{k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in item.items()}) for item in payload]
    for row in rows: row.validate()
    return rows


def collect_episode(model,config,*,seed,policy_seed,output,max_steps=1800,wall_seconds=3600,
                    deterministic=False,verify_replay=True):
    import json
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    model.eval();before_weights=digest_state(model.state_dict())
    steps=[];records=[];counts=Counter();env=None;error=None;terminal=False;replayed=False
    start=time.monotonic()
    write_json(output/'protocol.json',dict(seed=seed,policy_seed=policy_seed,max_steps=max_steps,
        wall_seconds=wall_seconds,deterministic=deterministic,teacher_enabled=False,
        policy_initial_hash=before_weights,PPO_ALLOWED=False))
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(policy_seed)
            env=TerminalParallelEnv(config,seed=seed)
            initial=pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL)
            (output/'first_before.pkl').write_bytes(initial)
            with (output/'requests.jsonl').open('x',encoding='utf8') as stream:
                for tick in range(max_steps):
                    require(time.monotonic()-start<wall_seconds,'rollout wall budget exceeded')
                    state=canonical(state_record(env));context=central_context(env)
                    row=request(model,env,deterministic=deterministic);check_request(row)
                    with torch.no_grad():
                        value=float(model.value(*row.tensors,torch.from_numpy(context))) if row.uids else world_value(model,env)
                    require(state==canonical(state_record(env)),'inference changed environment')
                    before=pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL)
                    live=execute_tape(env,row.tape())
                    terminal=natural_terminal(env)
                    reward=float(live['outputs'][4]['team_reward'])
                    new=step_record(row,context,value=value,next_value=0. if terminal else world_value(model,env),
                        reward=reward,terminated=terminal,episode_end=terminal or tick+1==max_steps,
                        stochastic_behavior=not deterministic)
                    # Deterministic evaluation records are intentionally not
                    # PPO data; shape/probability checks still apply.
                    if not deterministic: new.validate()
                    steps.append(new)
                    counts['actor']+=int(row.actor_mask.sum());counts['RELEASE']+=int(row.gates.sum())
                    counts['shield_actor']+=sum(bool(live['outputs'][4]['action_traces'][u]['shield_used'])
                                              for u,m in zip(row.uids,row.actor_mask) if m)
                    record=dict(tick=tick,pre_hash=_digest(state),tape=row.tape(),post_hash=_digest(live['after']),reward=reward)
                    records.append(record);stream.write(json.dumps(_json(record),allow_nan=False)+'\n');stream.flush()
                    outcomes=Counter(v.value for v in env.ledger.outcomes.values())
                    if tick==0 or (tick+1)%25==0 or terminal:
                        write_json(output/'progress.json',dict(ticks=tick+1,outcomes=outcomes,counts=counts,elapsed=time.monotonic()-start))
                    require(not outcomes.get('failed_energy'),'failed_energy')
                    if terminal: break
            if verify_replay:
                replay=pickle.loads(initial);replay_start=time.monotonic()
                for record in records:
                    require(time.monotonic()-replay_start<wall_seconds,'replay wall budget exceeded')
                    require(_digest(canonical(state_record(replay)))==record['pre_hash'],'replay pre-state differs')
                    result=execute_tape(replay,record['tape'])
                    require(_digest(result['after'])==record['post_hash'],'replay post-state differs')
                    require(result['outputs'][4]['team_reward']==record['reward'],'replay reward differs')
                replayed=True
            require(digest_state(model.state_dict())==before_weights,'rollout changed policy/critic')
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        if 'before' in locals(): (output/'failure_before.pkl').write_bytes(before)
        if env is not None: (output/'failure_after.pkl').write_bytes(pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL))
    save_steps(output/'steps.pt',steps)
    result=dict(status='ERROR' if error else 'COMPLETED' if terminal else 'CENSORED',error=error,
        ticks=len(steps),terminal=terminal,censored=not terminal,replay_exact=replayed,counts=counts,
        reward=sum(r.reward for r in steps),elapsed_seconds=time.monotonic()-start,
        PPO_ALLOWED=False,deterministic=deterministic)
    if env is not None:
        result.update(outcomes=dict(env.ledger.outcomes),outcome_counts=dict(Counter(v.value for v in env.ledger.outcomes.values())),
            requested=len(env.ledger.requested),unresolved=len(env.ledger.unresolved),active=len(env.active),
            queue=len(env.admission.queue.items),service=len(env.service_uavs),telemetry=dict(env.telemetry),
            energy_outcomes=env.energy_outcomes,external_queue_uav_seconds=env.external_queue_uav_seconds,
            shield_actor_fraction=counts['shield_actor']/max(1,counts['actor']))
    write_json(output/'result.json',result)
    return steps,result


def critic_warmup(model,episodes,*,updates=500,batch_size=32,seed=1,learning_rate=3e-4,gamma=.99):
    if not episodes or updates<1 or batch_size<1: raise ValueError('positive warmup budget and complete episodes required')
    rows=[];targets=[]
    for episode in episodes:
        if not episode or not episode[-1].terminated or any(r.episode_end for r in episode[:-1]):
            raise ValueError('critic warmup requires whole natural-terminal episodes')
        for row in episode: row.validate()
        future=0.;episode_targets=[]
        for row in reversed(episode):
            future=row.reward+gamma*future;episode_targets.append(future)
        rows.extend(episode);targets.extend(reversed(episode_targets))
    target=torch.tensor(targets,dtype=torch.float32)
    params=[p for name,p in model.named_parameters() if name.startswith('critic_')]
    optimizer=torch.optim.Adam(params,lr=learning_rate)
    frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('critic_')}
    rng=np.random.default_rng(seed)
    with torch.no_grad():
        initial=torch.stack([model.value(*tensors(r),torch.from_numpy(r.context)) for r in rows])
        initial_mse=float((initial-target).square().mean())
    for _ in range(updates):
        indices=rng.choice(len(rows),size=min(batch_size,len(rows)),replace=False)
        predicted=torch.stack([model.value(*tensors(rows[i]),torch.from_numpy(rows[i].context)) for i in indices])
        loss=(predicted-target[indices]).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite critic loss')
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
    with torch.no_grad():
        final=torch.stack([model.value(*tensors(r),torch.from_numpy(r.context)) for r in rows])
    require(all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()),'warmup changed actor')
    return dict(updates=updates,team_steps=len(rows),initial_mse=initial_mse,
                final_mse=float((final-target).square().mean()),targets=targets)
