"""Current neutral P3 course data; train and Q use separate complete cycles."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from ..configs.p3_v25_course import build_config, PROFILE, SCENARIOS
from ..envs.parallel_env import TerminalParallelEnv
from ..scripts import collect_p3_v3_sparse_expert as sparse
from ..scripts.collect_p3_v3_i0 import rollout_blockers, coverage_report
from .p3_v25_imitation_qualification import rebuild_context, sha, decode_evaluation_row
from .p3_v25_context_dataset import decode_row
from .p3_v25_set_learnability import PairedInputs
from .p3_v3_bc_learnability import Data, require, read, write
from .p3_v3_expert_dataset import DATA_SCHEMA, DEVELOPMENT_SCHEMA, DEVELOPMENT_PARTITION
from ..core.trajectory_context_observation import pad_trajectory_context

PROTOCOL = 'p3_v25_neutral_course_data_20261008_v2'
TRAIN_CASES = tuple(zip((893511,893512,893513),SCENARIOS))
Q_CASES = tuple(zip((895501,895502,895503),SCENARIOS))
SMOKE_SEED = 892863
GAP_MODE = 'committed_arrivals_v1'


def task_settings(config):
    keys = ('task_priority_enabled','priority_mode','initial_priority','initial_soc_mode',
        'priority_wait_mode','nominal_holder_separation_mode',
        'initial_soc','initial_soc_min','initial_soc_max','mission_probabilities',
        'release_arbitration_mode','low_level_priority_time_weight',
        'low_level_soc_urgency_time_weight','low_level_soc_urgency_threshold',
        'reset_mode','warm_start_uavs','fixed_arrival_schedule','max_simulation_seconds',
        'merge_gap_prediction_mode','v3_landing_path_mode')
    return json.loads(json.dumps({key:getattr(config,key) for key in keys}))


def collect(output, index, *, partition='train', smoke=False, annotation_wall_seconds=7200):
    require(partition in {'train','Q'}, 'unknown partition')
    require(type(index) is int and 1<=index<=(1 if smoke else 3), 'unknown task case')
    require(annotation_wall_seconds in (7200, 21600), 'undeclared annotation budget')
    cases = TRAIN_CASES if partition=='train' else Q_CASES
    seed,scenario = (SMOKE_SEED,'dense24_mixed') if smoke else cases[index-1]
    return collect_case(output,index=index,seed=seed,scenario=scenario,partition=partition,
        smoke=smoke,annotation_wall_seconds=annotation_wall_seconds,protocol_name=PROTOCOL)


def collect_case(output,*,index,seed,scenario,partition,smoke,annotation_wall_seconds,protocol_name,
                 learning_observation_schema='phaseA_low_obs_v2_5_trajectory_context_set'):
    """Shared mechanics; public protocol entry points own frozen seed/partition lists."""
    require(partition in {'train','Q'}, 'unknown partition')
    require(annotation_wall_seconds in (7200,21600), 'undeclared annotation budget')
    config = build_config(scenario)
    if smoke:
        config = replace(config,total_requested_uavs=3,warm_start_uavs=3,fixed_arrival_schedule=(0.,)*3)
    wire_partition = DEVELOPMENT_PARTITION if smoke else partition
    schema = DEVELOPMENT_SCHEMA if smoke else DATA_SCHEMA
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    protocol=dict(protocol=protocol_name,profile=PROFILE,index=index,seed=seed,scenario=scenario,
        learning_observation_schema=learning_observation_schema,
        partition=wire_partition,schema=schema,smoke=smoke,task=task_settings(config),
        max_steps=2 if smoke else int(config.max_simulation_seconds/config.dt),
        rollout_wall_seconds=180 if smoke else 3600,
        annotation_wall_seconds=180 if smoke else annotation_wall_seconds,
        max_selected=2048,max_release_intents=51200,selection='all actor pre-states; tick/numeric UID',
        behavior='rule_nominal_k3',teacher='state-only first feasible; not joint/global optimum',
        checkpoint_loaded=False,I0_QUALIFIED=False,PPO_ALLOWED=False)
    write(output/'protocol.json',protocol)
    report=dict(protocol=protocol_name,profile=PROFILE,seed=seed,scenario=scenario,partition=wire_partition,
        status='ERROR',error=None,I0_QUALIFIED=False,PPO_ALLOWED=False)
    try:
        env=TerminalParallelEnv(config,seed=seed)
        episode=output/f'seed_{seed}'
        flow,selected=sparse.rollout(env,episode,seed=seed,max_steps=protocol['max_steps'],
            wall_seconds=protocol['rollout_wall_seconds'],smoke=smoke,all_actor=True,
            max_selected=2048,schema=schema,partition=wire_partition)
        blockers=rollout_blockers(flow,smoke=smoke)
        report.update(rollout=flow,blockers=blockers)
        write(output/'result.json',dict(report,status='INCOMPLETE' if blockers else 'ANNOTATING'))
        if not blockers:
            annotation=sparse.annotate(episode,selected,wall_seconds=protocol['annotation_wall_seconds'])
            report['annotation']=annotation
            if annotation['status']!='COMPLETED' or annotation['selected']!=annotation['labelled']:
                blockers.append('incomplete_annotation')
            if not blockers:
                report['reconstruction']=rebuild_context(episode,seed=seed,partition=wire_partition,
                                                         expected_gap_mode=GAP_MODE)
                report['coverage']=coverage_report(episode)
        report['status']=('SMOKE_OK' if smoke else 'COLLECTION_OK') if not blockers else 'INCOMPLETE'
    except Exception as error:
        report.update(status='ERROR',error=repr(error))
    write(output/'result.json',report)
    return report


def validated_case(directory, seed, scenario, partition):
    directory=Path(directory)
    protocol,report=read(directory/'protocol.json'),read(directory/'result.json')
    require(protocol['protocol']==report['protocol']==PROTOCOL and
        protocol['profile']==report['profile']==PROFILE and
        protocol['partition']==report['partition']==partition and
        protocol['seed']==report['seed']==seed and
        protocol['scenario']==report['scenario']==scenario and not protocol['smoke'] and
        protocol['task']==task_settings(build_config(scenario)), 'wrong task/partition/identity')
    require(report['status']=='COLLECTION_OK' and report['error'] is None and not report['blockers'],
        'incomplete task cycle')
    path=directory/f'seed_{seed}'/'context_observations.jsonl'
    require(sha(path)==report['reconstruction']['sha256'], 'task context bytes changed')
    return path,report


def _load(directories, *, partition):
    require(partition in {'train','Q'}, 'unknown task partition')
    cases=TRAIN_CASES if partition=='train' else Q_CASES
    require(len(directories)==len(cases), 'all declared task cycles required')
    observations,raw,gates,proposals,ids,contexts,inputs=[],[],[],[],[],[],[]
    seen=set()
    for directory,(seed,scenario) in zip(directories,cases):
        path,report=validated_case(directory,seed,scenario,partition)
        count=0
        for line in path.read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            require(row['seed']==seed and row['decision_id']==f"{DATA_SCHEMA}:{seed}:{row['tick']}:{row['uid']}"
                and row['decision_id'] not in seen, 'wrong or duplicate task decision')
            obs,gate,proposal=(decode_row(row) if partition=='train' else decode_evaluation_row(row,seed=seed))
            observations.append(obs);raw.append(row['original_observation']);gates.append(gate)
            proposals.append(proposal);ids.append(row['decision_id']);seen.add(row['decision_id'])
            contexts.append(row['context']);count+=1
        require(count==report['reconstruction']['rows']>0, 'task row count differs')
        inputs.append(dict(seed=seed,scenario=scenario,rows=count,partition=partition,sha256=sha(path)))
    raw=np.asarray(raw,dtype=np.float32)
    require(raw.shape==(len(ids),90) and np.isfinite(raw).all(), 'invalid task raw observation')
    data=Data(torch.from_numpy(raw),torch.tensor(gates),torch.tensor(proposals),tuple(ids),contexts,inputs)
    return PairedInputs(data,tuple(torch.from_numpy(a) for a in pad_trajectory_context(observations)))


def load_train(directories):
    return _load(directories,partition='train')


def load_q_for_evaluation(directories):
    """Explicit evaluation-only path; never fed into fit_arm."""
    return _load(directories,partition='Q')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--partition',choices=('train','Q'),default='train')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--annotation-wall-seconds',type=int,choices=(7200,21600),default=7200)
    args=parser.parse_args()
    report=collect(args.output,args.index,partition=args.partition,smoke=args.smoke,
                   annotation_wall_seconds=args.annotation_wall_seconds)
    print(json.dumps(dict(status=report['status'],seed=report['seed'],error=report['error'])))
    return 0 if report['status'] in {'SMOKE_OK','COLLECTION_OK'} else 2


if __name__=='__main__':
    raise SystemExit(main())
