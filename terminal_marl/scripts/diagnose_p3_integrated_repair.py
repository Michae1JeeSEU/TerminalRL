"""Small sequential, no-training checks for the integrated P3 repair.

Use --check-only first, then --smoke, then an explicitly bounded --run.
Outputs are new development evidence; failures and censored runs are retained.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

import numpy as np

from terminal_marl.configs.p3_integrated_repair import PROFILE, build_config, initialize_legacy_source
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        if hasattr(obj, 'value'): return obj.value
        if isinstance(obj, Path): return str(obj)
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, default=convert, indent=2, ensure_ascii=False), encoding='utf-8')


def run(args):
    config = build_config(requests=args.requests, warm_start=args.warm_start, high_soc=args.high_soc)
    result = dict(profile=PROFILE, config=asdict(config), controller=args.controller,
                  env_seed=args.seed, policy_seed=args.policy_seed, PPO_ALLOWED=False,
                  trained=False, qualification='NOT_QUALIFIED')
    policy = None
    if args.controller == 'source':
        if args.checkpoint is None: raise ValueError('--checkpoint required for source initializer')
        policy, result['migration'] = initialize_legacy_source(args.checkpoint, config)
    if args.check_only:
        result.update(status='CHECK_ONLY_OK', environment_constructed=False)
        return result
    import torch
    from terminal_marl.training.student_rollout_v4_2 import _student_request_details
    env = None
    limit = 2 if args.smoke else args.max_steps
    started = time.monotonic()
    ticks, total_reward, terminal = 0, 0.0, False
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.policy_seed)
            env = TerminalParallelEnv(config, seed=args.seed)
            observations = env._observations()
            for tick in range(limit):
                if policy is None:
                    actions, shortlists = {}, {}
                    for uid, u in env.active.items():
                        eligible = u.committed_plan is None and u.ring_role is not None and env.now >= u.safety_retry_after
                        actions[str(uid)] = np.array([1., 0., 0.]) if eligible else np.zeros(3)
                        if eligible:
                            shortlists[str(uid)] = tuple(np.array(row) for row in ((1.,0.,0.),(1.,-1.,0.),(1.,1.,0.)))
                else:
                    details = _student_request_details(policy, env, observations, deterministic=False, shortlist_k=3)
                    actions, shortlists = details[3], details[9]
                observations, rewards, _, _, info = env.step(actions, shortlists)
                ticks = tick+1
                total_reward += float(info.get('team_reward', sum(rewards.values())))
                terminal = env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs
                if ticks == 1 or ticks % 25 == 0 or terminal:
                    write_json(args.output/'progress.json', dict(status='running',ticks=ticks,
                        elapsed_seconds=time.monotonic()-started, active=len(env.active),
                        queue=len(env.admission.queue.items), outcomes=dict(env.ledger.outcomes)))
                if terminal: break
        result.update(status='COMPLETED' if terminal else 'CENSORED', error=None)
    except Exception as exc:
        result.update(status='ERROR', error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    if env is None:
        result.update(ticks=0, terminal=False, censored=True)
        return result
    result.update(ticks=ticks, terminal=terminal, censored=not terminal,
        reward_total=total_reward, elapsed_seconds=time.monotonic()-started,
        outcomes=dict(env.ledger.outcomes), outcome_counts=dict(Counter(env.ledger.outcomes.values())),
        telemetry=dict(env.telemetry), energy_outcomes=env.energy_outcomes,
        active=len(env.active), queue=len(env.admission.queue.items), service=len(env.service_uavs),
        generated_requests=len(env.request_times), external_queue_uav_seconds=env.external_queue_uav_seconds)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only',action='store_true'); mode.add_argument('--smoke',action='store_true'); mode.add_argument('--run',action='store_true')
    p.add_argument('--controller',choices=('rule','source'),default='rule')
    p.add_argument('--checkpoint',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--requests',type=int,default=3); p.add_argument('--warm-start',type=int,default=0)
    p.add_argument('--high-soc',action='store_true'); p.add_argument('--seed',type=int,default=890301)
    p.add_argument('--policy-seed',type=int,default=890401); p.add_argument('--max-steps',type=int,default=1800)
    args=p.parse_args()
    if args.max_steps < 1: p.error('--max-steps must be positive')
    if args.output.exists(): p.error('output already exists; preserve earlier evidence')
    args.output.mkdir(parents=True)
    result=run(args); write_json(args.output/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','ticks','terminal','outcome_counts','error')}))
    return 1 if result['status']=='ERROR' else 0


if __name__=='__main__':
    raise SystemExit(main())
