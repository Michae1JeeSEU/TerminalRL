"""Small sequential, no-training checks for the integrated P3 repair.

Use --check-only first, then --smoke, then an explicitly bounded --run.
Outputs are new development evidence; failures and censored runs are retained.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
import pickle
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



def record_blockers(env, output, ticks, snapshot_ticks=()):
    """Observe the completed tick without querying candidates or changing state.

    Pickles are local development snapshots from this trusted run, for bounded
    same-state diagnostics. They are not checkpoints or evaluation results.
    """
    def uav_state(u):
        plan = u.committed_plan
        return dict(uid=u.uid, arrival_id=u.arrival_id, phase=u.phase.value,
            ring_role=getattr(u.ring_role, 'value', None), soc=float(u.soc),
            energy_remaining=float(u.energy_remaining), speed=float(u.speed),
            position=np.asarray(u.position).tolist(), retry_after=float(u.safety_retry_after),
            emergency_retry_after=float(u.emergency_retry_after), assigned_lz=u.assigned_lz,
            plan_mode=None if plan is None else plan.mode,
            plan_end=None if plan is None else float(plan.end_time))
    rows = []
    for uid, row in env.last_release_telemetry.items():
        rows.append(dict(uid=uid, arrival_id=row['arrival_id'], interface=row['interface'],
            decision=row['decision_history'][-1], primary_blocker=row['primary_blocker'],
            secondary_constraints=row['secondary_constraints'],
            final_lz_pairs_checked=row.get('final_lz_pairs_checked', [])))
    record = dict(ticks=ticks, time=env.now, releases=rows,
        last_safety_decisions=dict(env.last_safety_decisions),
        emergency={k:v for k,v in env.telemetry.items() if k.startswith('emergency_')},
        outcomes=dict(env.ledger.outcomes))
    if ticks == 1 or ticks % 25 == 0 or ticks in snapshot_ticks:
        record.update(active=[uav_state(u) for u in env.active.values()],
            energy_diagnostics=env.energy_reachability_diagnostics)
    with (output/'blockers.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    if ticks in snapshot_ticks:
        with (output/f'snapshot_{ticks}.pkl').open('xb') as stream:
            pickle.dump(env, stream, protocol=pickle.HIGHEST_PROTOCOL)


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
                if getattr(args, 'trace_blockers', False):
                    record_blockers(env, args.output, ticks, getattr(args, 'snapshot_ticks', ()))
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
    p.add_argument('--trace-blockers',action='store_true')
    p.add_argument('--snapshot-ticks',type=int,nargs='*',default=[])
    args=p.parse_args()
    if args.max_steps < 1: p.error('--max-steps must be positive')
    if args.snapshot_ticks and not args.trace_blockers: p.error('--snapshot-ticks requires --trace-blockers')
    if any(t < 1 or t > (2 if args.smoke else args.max_steps) for t in args.snapshot_ticks):
        p.error('snapshot ticks must be within the selected run limit')
    if args.output.exists(): p.error('output already exists; preserve earlier evidence')
    args.output.mkdir(parents=True)
    result=run(args); write_json(args.output/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','ticks','terminal','outcome_counts','error')}))
    return 1 if result['status']=='ERROR' else 0


if __name__=='__main__':
    raise SystemExit(main())
