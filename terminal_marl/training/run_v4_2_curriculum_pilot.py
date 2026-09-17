"""P2 natural-flow reachability pilot; rule and student share requests and physics."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from terminal_marl.training.run_v4_2_ppo_block2 import restore, replay_geometry, source_hashes
from terminal_marl.training.run_v4_2_v2_1_student_qualification import protocol, scenario_config
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r3 import _write
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import _sha
from terminal_marl.training.student_rollout_v4_2 import student_request
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.envs.parallel_env import TerminalParallelEnv

CHECKPOINT_SHA = '6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f'


def config_for(n):
    schedule=tuple(10.+60.*i for i in range(n))
    return replace(scenario_config('flow6',protocol()), total_requested_uavs=n,
        fixed_arrival_schedule=schedule, max_simulation_seconds=schedule[-1]+900.)


def episode(config, seed, output, policy=None, max_steps=None):
    output.mkdir(parents=True, exist_ok=False)
    env=TerminalParallelEnv(config)
    obs,_=env.reset(seed)
    tape=SafetyTraceRecorder(output/'action_tapes', dt=config.dt)
    tape.start_episode(env,episode_seed=seed,policy_rng_state=None)
    peak_active=peak_queue=shield=0
    exception=None
    terminal=False
    for tick in range(max_steps or int(config.max_simulation_seconds/config.dt)):
        actions={} if policy is None else student_request(policy,env,obs,deterministic=True)[3]
        row=tape.record_pre_step(env,obs,actions)
        try:
            obs,_,_,_,info=env.step(actions)
        except SafetyInvariantError as exc:
            tape.record_safety_invariant(row,exc)
            exception=str(exc)
            break
        tape.record_post_step(env,row,info)
        shield+=sum(bool(t['shield_used']) for t in info['action_traces'].values())
        if any(t['teacher_used'] for t in info['action_traces'].values()):
            raise RuntimeError('Teacher unexpectedly used')
        peak_active=max(peak_active,len(env.active))
        peak_queue=max(peak_queue,len(env.admission.queue.items))
        terminal=env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs
        if tick%25 == 0 or terminal:
            _write(output/'progress.json',dict(seed=seed,seconds=env.now,completed=len(env.completed_uavs),peak_active=peak_active,peak_queue=peak_queue))
        if terminal:break
    path=tape.dump(reason='P2_reachability_pilot')
    replay,geometry=replay_geometry(config,path)
    safety={k:int(env.telemetry.get(k,0)) for k in ('collisions','nmac','hard_separation_violations','uncertified_execution')}
    safety['SafetyInvariantError']=int(exception is not None)
    completed=env.completed_uavs
    result=dict(seed=seed,requests=config.total_requested_uavs,seconds=env.now,completed=len(completed),
        terminal=terminal,outcomes=dict(env.ledger.outcomes),active_peak=peak_active,queue_peak=peak_queue,
        remaining_active=len(env.active),remaining_queue=len(env.admission.queue.items),shield=shield,
        mean_wait_seconds=float(np.mean([u.external_wait_time+u.internal_wait_time for u in completed])) if completed else None,
        mean_request_to_touchdown_seconds=float(np.mean([u.touchdown_time-u.request_time for u in completed])) if completed else None,
        safety=safety,safety_error=exception,replay_exact=replay,geometry=geometry,action_tape=str(path))
    result['gates']=dict(complete=terminal and len(completed)==config.total_requested_uavs,
        safety=not any(safety.values()),replay=replay,
        geometry=geometry['samples']>0 and geometry['reverse_samples']==0 and geometry['max_inner_excursion_m']<=.05)
    _write(output/'summary.json',result)
    return result


def run(output,checkpoint,history_root,smoke=False):
    policy=restore(checkpoint,dict(initial_checkpoint_sha256=CHECKPOINT_SHA,learning_rate=.00003))
    output.mkdir(parents=True,exist_ok=False)
    cases=[(748000,6)] if smoke else [(748001,6),(748002,8),(748003,12),(748004,6),(748005,8),(748006,12)]
    if {s for s,n in cases} & historical_registry(history_root,output)['seeds']:
        raise ValueError('pilot seeds reused')
    _write(output/'pre_reset_registry.json',dict(seeds=[s for s,n in cases],cases=cases,smoke=smoke))
    records=[]
    for seed,n in cases:
        for name,actor in [('rule',None),('student',policy)]:
            result=episode(config_for(n),seed,output/f'{name}_{seed}',actor,32 if smoke else None)
            records.append(dict(controller=name,summary=result))
            _write(output/'progress.json',dict(completed=len(records),total=2*len(cases),records=records))
            if any(result['safety'].values()):break
        if any(records[-1]['summary']['safety'].values()):break
    result=dict(stage='P2_REACHABILITY_PILOT',smoke=smoke,records=records,training_called=False,
        PASS=not smoke and len(records)==12 and all(all(r['summary']['gates'].values()) for r in records),
        source_hashes={**source_hashes(),'pilot':_sha(Path(__file__))},checkpoint_sha256=CHECKPOINT_SHA)
    _write(output/'manifest.json',result)
    print(json.dumps(dict(PASS=result['PASS'],smoke=smoke,records=len(records))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','checkpoint','history-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    torch.set_num_threads(1)
    run(**vars(p.parse_args()))
