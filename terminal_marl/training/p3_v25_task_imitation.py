"""Fresh neutral-course P3 imitation fits and a single independent Q evaluation.

Fit only opens the declared train cycles. Q is collected after candidates are
frozen; evaluating it cannot update weights or grant student/PPO admission.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

import torch

from ..algo.trajectory_set_actor import TrajectorySetActor
from ..configs.p3_v25_course import PROFILE
from ..core.trajectory_context_observation import VERSION
from .p3_v25_course_data import load_train, load_q_for_evaluation, Q_CASES, validated_case
from .p3_v25_set_learnability import fit_arm, synthetic, measure
from .p3_v25_imitation_qualification import gates, sha
from .p3_v3_bc_learnability import require, read, write

PROTOCOL = 'p3_v25_neutral_course_imitation_20261008_v1'
SEEDS = (91901,91902,91903)


def fit(directories,output,*,smoke=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    paired=synthetic() if smoke else load_train(directories)
    seeds,updates=((17,),2) if smoke else (SEEDS,5000)
    write(output/'protocol.json',dict(protocol=PROTOCOL,profile=PROFILE,stage='fit',smoke=smoke,
        seeds=seeds,updates=updates,batch_size=128,learning_rate=.0003,gradient_clip=.5,
        wall_seconds_per_seed=300,selection='fixed final; all seeds; canonical first seed',
        observation_schema=VERSION,Q_row_access=False,PPO_ALLOWED=False))
    write(output/'inputs.json',dict(decision_ids=paired.original.ids,episodes=paired.original.inputs))
    results,checkpoints=[],[]
    for seed in seeds:
        torch.manual_seed(seed)
        initial=deepcopy(TrajectorySetActor().state_dict())
        case=output/f'seed_{seed}'
        result=fit_arm(paired,case,'new_information',seed,initial,updates=updates,
                       wall_seconds=300,protocol=PROTOCOL)
        path=case/'diagnostic_final.pt'
        checkpoints.append(dict(seed=seed,path=path.relative_to(output).as_posix(),sha256=sha(path)))
        results.append(result)
    report=dict(status='FIT_FROZEN',protocol=PROTOCOL,profile=PROFILE,smoke=smoke,
        fits=results,checkpoints=checkpoints,Q_row_access=False,
        I1_QUALIFIED=False,STUDENT_QUALIFIED=False,PPO_ALLOWED=False)
    write(output/'result.json',report)
    return report


def frozen_models(directory,*,smoke=False):
    directory=Path(directory);report=read(directory/'result.json')
    require(report['status']=='FIT_FROZEN' and report['protocol']==PROTOCOL and
        report['profile']==PROFILE and report['smoke'] is smoke,'wrong neutral-course frozen fit')
    require([x['seed'] for x in report['checkpoints']]==([17] if smoke else list(SEEDS)),
        'candidate seed set differs')
    for item in report['checkpoints']:
        require(item['path']==f"seed_{item['seed']}/diagnostic_final.pt" and
            sha(directory/item['path'])==item['sha256'],'frozen checkpoint changed')
    return report


def evaluate(fit_directory,q_directories,output):
    manifest=frozen_models(fit_directory)
    require(len(q_directories)==len(Q_CASES),'all neutral course Q cycles required')
    # Validate metadata and file bytes before recording one-use consumption.
    for directory,(seed,scenario) in zip(q_directories,Q_CASES):
        validated_case(directory,seed,scenario,'Q')
    markers=[Path(directory)/'evaluation_consumed.json' for directory in q_directories]
    require(not any(path.exists() for path in markers),'Q already consumed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for path in markers:
        with path.open('x',encoding='utf8') as stream:
            json.dump(dict(protocol=PROTOCOL,fit_directory=str(fit_directory),
                checkpoints=manifest['checkpoints'],output=str(output)),stream)
    paired=load_q_for_evaluation(q_directories)
    results=[]
    targets=[0 if g==0 else p+1 for g,p in zip(paired.original.gate.tolist(),paired.original.proposal.tolist())]
    for item in manifest['checkpoints']:
        payload=torch.load(Path(fit_directory)/item['path'],map_location='cpu',weights_only=True)
        require(payload['protocol']==PROTOCOL and payload['seed']==item['seed'], 'checkpoint provenance differs')
        model=TrajectorySetActor();model.load_state_dict(payload['model']);model.requires_grad_(False)
        metrics,predictions=measure(model,paired.new,paired.original)
        write(output/f"predictions_{item['seed']}.json",predictions)
        results.append(dict(seed=item['seed'],metrics=metrics,decision=gates(metrics,predictions,targets),
                            checkpoint_sha256=item['sha256']))
    frozen_models(fit_directory)
    report=dict(status='EVALUATED',protocol=PROTOCOL,profile=PROFILE,inputs=paired.original.inputs,
        seeds=results,all_imitation_gates_passed=all(r['decision']['passed'] for r in results),
        parameter_updates=0,STUDENT_QUALIFIED=False,PPO_ALLOWED=False)
    write(output/'result.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('fit','evaluate'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-directories',type=Path,nargs=3)
    parser.add_argument('--q-directories',type=Path,nargs=3)
    parser.add_argument('--fit-directory',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.mode=='fit':
        require(args.smoke or args.train_directories,'missing train data')
        result=fit(args.train_directories or (),args.output,smoke=args.smoke)
    else:
        require(not args.smoke and args.fit_directory and args.q_directories,'missing frozen evaluation inputs')
        result=evaluate(args.fit_directory,args.q_directories,args.output)
    print(json.dumps(dict(status=result['status'])))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
