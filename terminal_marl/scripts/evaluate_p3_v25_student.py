"""Bounded V2.5 student development rollout with complete request-tape replay."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback

import torch

from ..configs.p3_integrated_repair import build_config
from ..core.safety_trace import _digest, _json
from ..core.trajectory_context_observation import VERSION
from ..envs.parallel_env import TerminalParallelEnv
from ..training.p3_v25_runtime import load_actor, request, check_request
from ..training.p3_v25_set_learnability import digest_state
from .diagnose_p3_direct_contract import execute_tape, state_record, canonical, require, write_json, SAFETY_KEYS
from .collect_p3_v3_sparse_expert import natural_terminal


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    config=build_config(requests=args.requests,warm_start=args.warm_start,high_soc=args.high_soc,
                        forecast_merge_gap=getattr(args,'forecast_merge_gap',False))
    write_json(args.output/'protocol.json',dict(mode='development_student_never_qualification',
        policy_observation_schema=VERSION, physics_config=asdict(config), checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        seed=args.seed,policy_seed=args.policy_seed,deterministic=args.deterministic,
        max_steps=2 if args.smoke else 1800,wall_seconds=3600,replay_wall_seconds=3600,
        PPO_ALLOWED=False,teacher_enabled=False))
    env=None; ticks=0; total_reward=0.; terminal=False; replay_ok=False; error=None
    counts=Counter(); started=time.monotonic()
    try:
        model,_=load_actor(args.checkpoint)
        weights=digest_state(model.state_dict())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.policy_seed)
            env=TerminalParallelEnv(config,seed=args.seed)
            initial=pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL)
            (args.output/'first_before.pkl').write_bytes(initial)
            with (args.output/'steps.jsonl').open('x',encoding='utf8') as stream:
                for tick in range(2 if args.smoke else 1800):
                    require(time.monotonic()-started<3600,'rollout wall budget exceeded')
                    pre=canonical(state_record(env))
                    row=request(model,env,deterministic=args.deterministic)
                    check_request(row)
                    require(canonical(state_record(env))==pre,'policy changed environment')
                    tape=row.tape()
                    before=pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL)
                    live=execute_tape(env,tape)
                    info=live['outputs'][4]
                    total_reward+=float(info['team_reward'])
                    counts['actor']+=int(row.actor_mask.sum())
                    counts['RELEASE']+=int(row.gates.sum())
                    counts['WAIT']+=int(row.actor_mask.sum()-row.gates.sum())
                    counts['shield_actor']+=sum(bool(info['action_traces'][uid].get('shield_used')) for uid,m in zip(row.uids,row.actor_mask) if m)
                    record=dict(tick=tick,pre_hash=_digest(pre),tape=tape,
                        actor_mask=dict(zip(row.uids,row.actor_mask.tolist())),likelihoods=row.likelihoods,
                        post_hash=_digest(live['after']),reward=float(info['team_reward']),
                        action_traces=info['action_traces'],release_telemetry=canonical(env.last_release_telemetry))
                    stream.write(json.dumps(_json(record),allow_nan=False)+'\n'); stream.flush()
                    ticks=tick+1; terminal=natural_terminal(env)
                    if ticks in (400,600,850):
                        (args.output/f'snapshot_{ticks}.pkl').write_bytes(pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL))
                    if ticks==1 or ticks%25==0 or terminal:
                        write_json(args.output/'progress.json',dict(stage='rollout',ticks=ticks,
                            outcomes=dict(env.ledger.outcomes),active=len(env.active),counts=counts,
                            elapsed_seconds=time.monotonic()-started))
                    require(not Counter(env.ledger.outcomes.values()).get('failed_energy'),'failed energy')
                    if terminal: break
            replay=pickle.loads(initial)
            replay_started=time.monotonic()
            for line in (args.output/'steps.jsonl').read_text().splitlines():
                require(time.monotonic()-replay_started<3600,'replay wall budget exceeded')
                record=json.loads(line)
                require(_digest(canonical(state_record(replay)))==record['pre_hash'],'replay pre-state mismatch')
                after=execute_tape(replay,record['tape'])
                require(_digest(after['after'])==record['post_hash'],'replay post-state mismatch')
                require(after['outputs'][4]['team_reward']==record['reward'],'replay reward mismatch')
            replay_ok=True
            require(digest_state(model.state_dict())==weights,'inference changed model')
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        if env is not None:
            (args.output/'failure_after.pkl').write_bytes(pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL))
        if 'before' in locals(): (args.output/'failure_before.pkl').write_bytes(before)
    result=dict(status='ERROR' if error else 'SMOKE_OK' if args.smoke else 'COMPLETED' if terminal else 'CENSORED',
        error=error,ticks=ticks,terminal=terminal,censored=not terminal,replay_exact=replay_ok,
        counts=counts,reward_total=total_reward,elapsed_seconds=time.monotonic()-started,
        checkpoint=str(args.checkpoint),seed=args.seed,policy_seed=args.policy_seed,deterministic=args.deterministic,
        PPO_ALLOWED=False,STUDENT_QUALIFIED=False)
    if env is not None:
        result.update(outcome_counts=dict(Counter(env.ledger.outcomes.values())),outcomes=dict(env.ledger.outcomes),
            requested=len(env.ledger.requested),unresolved=len(env.ledger.unresolved),active=len(env.active),
            queue=len(env.admission.queue.items),service=len(env.service_uavs),telemetry=dict(env.telemetry),
            shield_actor_fraction=counts['shield_actor']/max(1,counts['actor']),energy_outcomes=env.energy_outcomes,
            external_queue_uav_seconds=env.external_queue_uav_seconds)
    write_json(args.output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--policy-seed',type=int,required=True)
    p.add_argument('--requests',type=int,default=12);p.add_argument('--warm-start',type=int,default=6)
    p.add_argument('--high-soc',action='store_true');p.add_argument('--deterministic',action='store_true')
    p.add_argument('--forecast-merge-gap',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();r=run(a)
    print(json.dumps({k:r[k] for k in ('status','ticks','terminal','error')}))
    return 1 if r['error'] else 0


if __name__=='__main__': raise SystemExit(main())
