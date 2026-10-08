"""A bounded three-cycle PPO block followed by paired development evaluation.

Requires a frozen fit, its once-only imitation result and same-source complete
student episodes/critic warmup. No Q observations or teacher enter PPO. This
first block does not launch formal training or promote its own candidate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..algo.trajectory_set_ppo import TrajectorySetPPO
from ..configs.p3_v25_course import PROFILE, SCENARIOS, build_config
from .p3_v25_course_data import load_train
from .p3_v25_critic_initialization import PROTOCOL as CRITIC_PROTOCOL
from .p3_v25_imitation_qualification import sha
from .p3_v25_runtime import load_actor
from .p3_v25_rollout import collect_episode
from .p3_v25_set_learnability import digest_state
from .p3_v25_task_imitation import PROTOCOL as FIT_PROTOCOL, frozen_models, SEEDS
from .p3_v25_task_student import PROTOCOL as STUDENT_PROTOCOL, blockers
from .p3_v3_bc_learnability import require, read, write

PROTOCOL='p3_v25_neutral_bounded_ppo_20261008_v1'
TRAIN_SEEDS=(896501,896502,896503)
POLICY_SEEDS=(936501,936502,936503)
OPTIONS=dict(epochs=2,minibatch_size=32,gamma=.99,gae_lambda=.95,
             clip_ratio=.1,value_coef=.5,entropy_coef=.001,target_kl=.01)


def input_review(fit_directory,qualification_directory,warmup_directory,development_directories):
    fit=frozen_models(fit_directory);source=fit['checkpoints'][0]
    qualification=read(Path(qualification_directory)/'result.json')
    require(qualification['protocol']==FIT_PROTOCOL and qualification['profile']==PROFILE
        and qualification['status']=='EVALUATED' and qualification['parameter_updates']==0
        and qualification['all_imitation_gates_passed'], 'imitation qualification missing or failed')
    require([v['seed'] for v in qualification['seeds']]==list(SEEDS)
        and all(q['checkpoint_sha256']==m['sha256'] and q['decision']['passed']
                for q,m in zip(qualification['seeds'],fit['checkpoints'])), 'qualification source differs')
    warmup_directory=Path(warmup_directory);warmup=read(warmup_directory/'result.json')
    require(warmup['protocol']==CRITIC_PROTOCOL and warmup['profile']==PROFILE
        and warmup['status']=='CRITIC_WARMUP_COMPLETE' and warmup['source']==source,
        'wrong critic initialization')
    checkpoint=warmup_directory/warmup['checkpoint']
    require(checkpoint.name=='critic_initialized.pt' and sha(checkpoint)==warmup['checkpoint_sha256'],
            'critic checkpoint changed')
    require(len(development_directories)==len(SCENARIOS),'all student conditions required')
    baselines=[]
    for directory,scenario,item in zip(development_directories,SCENARIOS,warmup['inputs']):
        directory=Path(directory);report=read(directory/'result.json');protocol=read(directory/'protocol.json')
        require(report['protocol']==protocol['protocol']==STUDENT_PROTOCOL
            and report['profile']==protocol['profile']==PROFILE
            and report['source']==protocol['source']==source
            and report['scenario']==protocol['scenario']==scenario
            and report['status']=='DEVELOPMENT_COMPLETE' and not report['blockers']
            and not protocol['deterministic'] and not protocol['smoke']
            and not blockers(report['rollout']), 'student cycle unqualified')
        require(item['scenario']==scenario and item['seed']==protocol['seed']
            and item['result_sha256']==sha(directory/'result.json')
            and item['steps_sha256']==sha(directory/'episode/steps.pt'), 'critic/student inputs differ')
        baselines.append(dict(scenario=scenario,seed=protocol['seed'],policy_seed=protocol['policy_seed'],
            result=report['rollout'],source_result_sha256=sha(directory/'result.json')))
    require(len(warmup['inputs'])==len(SCENARIOS),'incomplete critic input manifest')
    return source,checkpoint,baselines


def aggregate(results):
    rows=[row for result in results for row in result['task_metrics']['completed_rows']]
    def distribution(key):
        values=[r[key] for r in rows]
        return dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),max=max(values)) if values else None
    return dict(episodes=len(results),completed_requests=len(rows),
        normal=sum(r['outcome']=='completed' for r in rows),
        emergency=sum(r['outcome']=='emergency_landed' for r in rows),
        energy_consumed=sum(r['energy_consumed'] for r in rows),
        reward=sum(r['reward'] for r in results),
        delay={key:distribution(key) for key in
               ('wait_seconds','request_to_touchdown_seconds','request_to_completion_seconds')})


def _run(fit_directory,qualification_directory,warmup_directory,development_directories,train_directories,output):
    source,warm_path,baselines=input_review(fit_directory,qualification_directory,warmup_directory,development_directories)
    reference=load_train(train_directories).new  # Strict train partition; never load Q rows.
    actor,_=load_actor(Path(fit_directory)/source['path'])
    payload=torch.load(warm_path,map_location='cpu',weights_only=True)
    require(payload['protocol']==CRITIC_PROTOCOL and payload['profile']==PROFILE
        and payload['source']==source and not payload['optimizer_state_included'],'wrong warmup payload')
    for name,value in actor.state_dict().items():
        require(torch.equal(value,payload['model']['actor.'+name]),'warmup changed source actor')
    require(all(torch.count_nonzero(v)==0 for k,v in payload['model'].items() if k.startswith('gate_adapter.')),
            'warmup has nonzero gate adapter')
    torch.manual_seed(936500);np.random.seed(936500)
    policy=TrajectorySetPPO(actor,learning_rate=3e-5,reference=reference,
                            reference_limit=.02,reference_coefficient=1.)
    policy.model.load_state_dict(payload['model'])
    require(not policy.optimizer.state,'PPO optimizer must be fresh after critic warmup')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(protocol=PROTOCOL,profile=PROFILE,source=source,
        authorization_scope='bounded three-cycle block only',train_seeds=TRAIN_SEEDS,policy_seeds=POLICY_SEEDS,
        learning_rate=3e-5,reference_kl_limit=.02,reference_coefficient=1.,options=OPTIONS,
        rollout_and_replay_wall_seconds_each=3600,update_wall_seconds=600,
        evaluation_pairs=[{k:v for k,v in item.items() if k!='result'} for item in baselines],
        Q_observations_accessed=False,teacher_enabled=False,formal_allowed=False))
    original_actor=digest_state(policy.model.actor.state_dict())
    report=dict(protocol=PROTOCOL,profile=PROFILE,source=source,status='RUNNING',training=[],evaluation=[],
                blockers=[],formal_allowed=False,candidate_retained=False)
    write(output/'result.json',report)
    for index,(scenario,seed,policy_seed) in enumerate(zip(SCENARIOS,TRAIN_SEEDS,POLICY_SEEDS)):
        episode=output/f'train_{index+1:02d}'
        config=build_config(scenario)
        steps,result=collect_episode(policy.model,config,seed=seed,policy_seed=policy_seed,
            output=episode,max_steps=int(config.max_simulation_seconds/config.dt))
        issues=blockers(result)
        item=dict(scenario=scenario,seed=seed,rollout=result,blockers=issues)
        report['training'].append(item)
        if issues:
            report.update(status='BLOCKED',blockers=[f'train_{index+1}:{x}' for x in issues])
            write(output/'result.json',report);return report
        np.random.seed(936510+index)
        before=digest_state(policy.model.state_dict())
        item['update']=policy.update(steps,training_authorized=True,wall_seconds=600,**OPTIONS)
        require(digest_state(policy.model.actor.state_dict())==original_actor,'PPO changed frozen actor/proposal')
        item.update(before_model_hash=before,after_model_hash=digest_state(policy.model.state_dict()))
        path=output/f'after_{index+1:02d}.pt'
        torch.save(dict(protocol=PROTOCOL,profile=PROFILE,source=source,model=policy.model.state_dict(),
            optimizer=policy.optimizer.state_dict(),completed_cycles=index+1,formal_allowed=False),path)
        item.update(checkpoint=path.name,checkpoint_sha256=sha(path))
        if item['update']['budget_exhausted']:
            report.update(status='BLOCKED',blockers=[f'train_{index+1}:update_budget_exhausted'])
            write(output/'result.json',report);return report
        write(output/'result.json',report)
    for index,baseline in enumerate(baselines):
        config=build_config(baseline['scenario'])
        _,result=collect_episode(policy.model,config,seed=baseline['seed'],policy_seed=baseline['policy_seed'],
            output=output/f'evaluation_{index+1:02d}',max_steps=int(config.max_simulation_seconds/config.dt))
        issues=blockers(result)
        report['evaluation'].append(dict(scenario=baseline['scenario'],baseline=baseline['result'],candidate=result,blockers=issues))
        if issues:
            report.update(status='BLOCKED',blockers=[f'evaluation_{index+1}:{x}' for x in issues])
            write(output/'result.json',report);return report
        write(output/'result.json',report)
    report.update(status='PILOT_COMPLETE',baseline_aggregate=aggregate([v['result'] for v in baselines]),
        candidate_aggregate=aggregate([v['candidate'] for v in report['evaluation']]),
        accepted_updates=sum(v['update']['accepted_updates'] for v in report['training']),
        rejected_updates=sum(v['update']['rejected_updates'] for v in report['training']))
    frozen_models(fit_directory);write(output/'result.json',report)
    return report


def run(fit_directory,qualification_directory,warmup_directory,development_directories,train_directories,output):
    # Never turn an already-existing output into an error artifact for a new run.
    require(not Path(output).exists(),'bounded PPO output already exists')
    try:
        return _run(fit_directory,qualification_directory,warmup_directory,development_directories,train_directories,output)
    except Exception as error:
        path=Path(output)/'result.json'
        if not path.exists():raise
        report=read(path)
        report.update(status='ERROR',error=repr(error),blockers=['technical_error'])
        write(path,report)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fit-directory','qualification-directory','warmup-directory','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('development-directories','train-directories'):
        parser.add_argument('--'+name,type=Path,nargs=3,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    result=run(args.fit_directory,args.qualification_directory,args.warmup_directory,
               args.development_directories,args.train_directories,args.output)
    print(json.dumps(dict(status=result['status'],formal_allowed=False)))
    return 0 if result['status']=='PILOT_COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
