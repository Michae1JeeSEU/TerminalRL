"""P2 held-out natural-flow performance comparison with student and nominal rule."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from terminal_marl.training.run_v4_2_ppo_block2 import restore
from terminal_marl.training.run_v4_2_ppo_p2_resume_v3 import p2_config, p2_source_hashes
from terminal_marl.training.run_v4_2_curriculum_pilot import episode as rule_episode
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import _sha, reserve_once
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r3 import _write
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from terminal_marl.training.student_rollout_v4_2 import rollout
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv

BASELINE_SHA='6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f'


def evaluate(policy, config, seed, output, smoke):
    _, summary=rollout(policy,config,seed=seed,max_steps=160 if smoke else int(config.max_simulation_seconds/config.dt),
                       output=output,deterministic=True,collect_labels=False)
    env=TerminalParallelEnv(config)
    geometry=dict(samples=0,reverse_samples=0,max_inner_excursion_m=0.)
    peaks=dict(active=0,queue=0)
    def observe(world):
        peaks['active']=max(peaks['active'],len(world.active))
        peaks['queue']=max(peaks['queue'],len(world.admission.queue.items))
        for u in world.active.values():
            plan=u.committed_plan
            if plan is None or plan.mode!='ring_transfer':continue
            xy,v=u.position[:2],u.velocity[:2]
            r=float(np.linalg.norm(xy))
            geometry['samples']+=1
            geometry['reverse_samples']+=int((xy[0]*v[1]-xy[1]*v[0])/max(r,1e-9)<-.001)
            geometry['max_inner_excursion_m']=max(geometry['max_inner_excursion_m'],float(np.linalg.norm(plan.points[-1].position[:2]))-r)
    env.set_step_observer(observe)
    replay=SafetyTraceRecorder.replay_action_tape(env,summary['action_tape'])
    logprob=_policy_tape_logprob_exact(policy,config,summary['action_tape'])
    completed=env.completed_uavs
    result=dict(seed=seed,requests=config.total_requested_uavs,seconds=summary['seconds'],completed=summary['completed'],
        terminal=summary['terminal'],safety=summary['safety'],geometry=geometry,
        replay_exact=replay['exact'],requested_logprob_exact=logprob,active_peak=peaks['active'],queue_peak=peaks['queue'],
        mean_wait_seconds=float(np.mean([u.internal_wait_time+u.external_wait_time for u in completed])) if completed else None,
        mean_request_to_touchdown_seconds=float(np.mean([u.touchdown_time-u.request_time for u in completed])) if completed else None,
        shield=summary['counts']['shield'],shield_actor_fraction=summary['counts']['shield_actor']/max(1,summary['counts']['actor_decisions']),
        summary=summary)
    result['gates']=dict(complete=result['terminal'] and result['completed']==config.total_requested_uavs,
        safety=not any(result['safety'].values()),replay=replay['exact'] and logprob,
        shield=result['shield_actor_fraction']<=.05,
        geometry=geometry['samples']>0 and geometry['reverse_samples']==0 and geometry['max_inner_excursion_m']<=.05)
    _write(output/'evaluation.json',result)
    return result


def run(output,checkpoint,baseline,history_root,smoke=False):
    if _sha(baseline)!=BASELINE_SHA:raise ValueError('P2 baseline mismatch')
    if not smoke:
        training=json.loads((checkpoint.parent/'manifest.json').read_text())
        if not training['PASS'] or len(training['records'])!=8 or checkpoint.name!='checkpoint_8.pt':
            raise ValueError('requires completed P2 final checkpoint')
        if training['source_hashes']!=p2_source_hashes():raise ValueError('P2 implementation changed')
    actors={name:restore(path,dict(initial_checkpoint_sha256=_sha(path),learning_rate=.00003))
            for name,path in [('candidate',checkpoint),('baseline',baseline)]}
    seeds=[756000] if smoke else list(range(755001,755007))
    output.mkdir(parents=True,exist_ok=False)
    if set(seeds)&historical_registry(history_root,output)['seeds']:raise ValueError('held-out seeds consumed')
    if not smoke:reserve_once(checkpoint.parent/'p2_heldout_consumed.json',dict(seeds=seeds,checkpoint_sha256=_sha(checkpoint),output=str(output)))
    _write(output/'pre_reset_registry.json',dict(seeds=seeds,smoke=smoke,controllers=['candidate','baseline','rule']))
    records=[]
    for i,seed in enumerate(seeds):
        config=p2_config((6,8,12)[i%3])
        for name in ('candidate','baseline','rule'):
            folder=output/f'{name}_{seed}'
            result=rule_episode(config,seed,folder,max_steps=160 if smoke else None) if name=='rule' else evaluate(actors[name],config,seed,folder,smoke)
            records.append(dict(controller=name,result=result))
            _write(output/'progress.json',dict(completed=len(records),total=3*len(seeds),records=records))
    aggregates={}
    for name in ('candidate','baseline','rule'):
        rows=[r['result'] for r in records if r['controller']==name]
        done=sum(r['completed'] for r in rows)
        aggregates[name]=dict(completed=done,mean_wait_seconds=sum((r['mean_wait_seconds'] or 0)*r['completed'] for r in rows)/max(1,done),
            mean_request_to_touchdown_seconds=sum((r['mean_request_to_touchdown_seconds'] or 0)*r['completed'] for r in rows)/max(1,done),
            all_gates=all(all(r['gates'].values()) for r in rows))
    result=dict(stage='P2_HELDOUT_EVALUATION',PASS=not smoke and aggregates['candidate']['all_gates'],smoke=smoke,
        records=records,aggregates=aggregates,checkpoint_sha256=_sha(checkpoint),baseline_sha256=BASELINE_SHA,
        training_called=False,source_hashes={**p2_source_hashes(),'evaluation':_sha(Path(__file__))})
    _write(output/'manifest.json',result)
    print(json.dumps(dict(PASS=result['PASS'],aggregates=aggregates)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','checkpoint','baseline','history-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    torch.set_num_threads(1)
    run(**vars(p.parse_args()))
