"""Teacher-free neutral-course P3 development episodes from the frozen canonical fit.

An episode is evidence for the primary's qualification decision, not automatic
PPO admission. The requested-action tape and tensor records remain available
for exact replay and subsequent critic initialization with the same actor.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from ..algo.trajectory_set_ppo import TrajectorySetActorCritic
from ..configs.p3_v25_course import build_config, PROFILE, SCENARIOS
from ..scripts.diagnose_p3_direct_contract import SAFETY_KEYS
from .p3_v25_runtime import load_actor
from .p3_v25_rollout import collect_episode
from .p3_v25_task_imitation import frozen_models, SEEDS
from .p3_v3_bc_learnability import require, write

PROTOCOL = 'p3_v25_neutral_course_student_development_20261008_v1'


def blockers(result, *, smoke=False):
    issues=[]
    if result['error'] is not None:issues.append('technical_error')
    if not result['replay_exact']:issues.append('replay_incomplete')
    if any(result.get('telemetry',{}).get(key)!=0 for key in SAFETY_KEYS):
        issues.append('actual_safety_or_hash')
    if result.get('shield_actor_fraction',0.)>.05:issues.append('actor_takeover_fraction')
    if not smoke:
        outcomes=result.get('outcome_counts',{})
        count=outcomes.get('completed',0)+outcomes.get('emergency_landed',0)
        if result['status']!='COMPLETED' or not result['terminal'] or result['censored']:
            issues.append('cycle_incomplete')
        if any(result.get(key,1) for key in ('unresolved','active','queue','service')):
            issues.append('unresolved_cycle')
        if count!=result.get('requested') or sum(outcomes.values())!=count:
            issues.append('failed_or_deferred_outcome')
        if result.get('counts',{}).get('actor',0)==0:issues.append('no_actor_decisions')
    return issues


def run(fit_directory,output,*,scenario,seed,policy_seed,deterministic=False,smoke=False):
    manifest=frozen_models(fit_directory)
    canonical=manifest['checkpoints'][0]
    require(canonical['seed']==SEEDS[0],'canonical source is fixed by seed order')
    config=build_config(scenario)
    if smoke:
        require(scenario=='dense24_mixed','smoke requires three interface holders')
        config=replace(config,total_requested_uavs=3,warm_start_uavs=3,fixed_arrival_schedule=(0.,)*3)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(protocol=PROTOCOL,profile=PROFILE,scenario=scenario,
        seed=seed,policy_seed=policy_seed,smoke=smoke,deterministic=deterministic,
        source=canonical,fit_directory=str(fit_directory),teacher_enabled=False,
        parameter_updates=0,STUDENT_QUALIFIED=False,PPO_ALLOWED=False))
    actor,_=load_actor(Path(fit_directory)/canonical['path'])
    # The value network cannot affect the policy; no optimizer is created.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(92001)
        model=TrajectorySetActorCritic(actor).requires_grad_(False)
    _,result=collect_episode(model,config,seed=seed,policy_seed=policy_seed,
        output=output/'episode',deterministic=deterministic,
        max_steps=2 if smoke else int(config.max_simulation_seconds/config.dt))
    issues=blockers(result,smoke=smoke)
    frozen_models(fit_directory)
    report=dict(protocol=PROTOCOL,profile=PROFILE,scenario=scenario,seed=seed,
        source=canonical,rollout=result,blockers=issues,
        status='BLOCKED' if issues else 'SMOKE_OK' if smoke else 'DEVELOPMENT_COMPLETE',
        STUDENT_QUALIFIED=False,PPO_ALLOWED=False,parameter_updates=0)
    write(output/'result.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenario',choices=SCENARIOS,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--policy-seed',type=int,required=True)
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    report=run(args.fit_directory,args.output,scenario=args.scenario,seed=args.seed,
        policy_seed=args.policy_seed,deterministic=args.deterministic,smoke=args.smoke)
    print(json.dumps(dict(status=report['status'],blockers=report['blockers'])))
    return 1 if report['blockers'] else 0


if __name__=='__main__':raise SystemExit(main())
