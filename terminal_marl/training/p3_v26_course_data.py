"""Bounded independent train cycles with explicit V2.6 resource observations.

No Q entry point or learning; old labels and 105D context remain immutable.
"""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from ..core.resource_context_observation import ResourceContextCache,VERSION,EGO_SCHEMA
from ..core.trajectory_context_observation import (VERSION as OLD_VERSION,
    EGO_SCHEMA as OLD_EGO_SCHEMA,TrajectoryContextObservation)
from .p3_v25_course_data import collect_case,PROFILE,task_settings,build_config
from .p3_v25_context_dataset import decode_row
from .p3_v25_imitation_qualification import sha
from .p3_v3_bc_learnability import require,read,write

PROTOCOL='p3_v26_spatial_resources_independent_train_20261009_v1'
TRAIN_CASES=((893601,'flow6_mixed'),(893602,'dense24_fixed'),(893603,'dense24_mixed'))
SMOKE_SEED=892871


def decode_resource_row(row):
    require(row['observation_schema']==VERSION and row['ego_feature_hash']==EGO_SCHEMA.feature_hash,
        'wrong V2.6 observation contract')
    ego=np.asarray(row['ego'],dtype=np.float32)
    require(ego.shape==(159,) and np.isfinite(ego).all(),'invalid resource tensor')
    old=dict(row,observation_schema=OLD_VERSION,ego_feature_hash=OLD_EGO_SCHEMA.feature_hash,ego=row['ego'][:105])
    obs,gate,proposal=decode_row(old)
    return TrajectoryContextObservation(ego,obs.traffic,obs.traffic_mask,(),(None,None),0.),gate,proposal


def rebuild_resources(episode,*,seed,smoke=False):
    episode=Path(episode)
    selected={e['row']['decision_id']:e for e in read(episode/'selected.json')}
    cached=None;count=0
    output=episode/'resource_context_observations.jsonl'
    with output.open('x',encoding='utf8') as stream:
        for line in (episode/'context_observations.jsonl').read_text(encoding='utf8').splitlines():
            row=json.loads(line);entry=selected.pop(row['decision_id'])
            require(row['seed']==seed,'resource seed differs')
            name=entry['snapshot'];require(Path(name).name==name,'invalid snapshot path')
            path=episode/'snapshots'/name
            require(sha(path)==entry['snapshot_sha256'],'resource snapshot changed')
            if name!=cached:
                env=pickle.loads(path.read_bytes());cache=ResourceContextCache(env);cached=name
            value=cache.build(env.active[int(row['uid'])])
            traffic=np.asarray(row['traffic'],dtype=np.float32).reshape(-1,50)
            require(np.allclose(value.ego[:105],row['ego'],atol=1e-7,rtol=1e-6) and
                np.allclose(value.traffic,traffic,atol=1e-7,rtol=1e-6),'original context differs')
            rebuilt=dict(row,ego=row['ego']+value.ego[105:].tolist(),
                observation_schema=VERSION,ego_feature_hash=EGO_SCHEMA.feature_hash)
            if not smoke:decode_resource_row(rebuilt)
            stream.write(json.dumps(rebuilt,allow_nan=False)+'\n');count+=1
    require(not selected and count>0,'resource row set differs')
    return dict(rows=count,sha256=sha(output),observation_schema=VERSION,ego_dimension=159,
        original_context_sha256=sha(episode/'context_observations.jsonl'),original_labels_preserved=True)


def collect(output,index,*,smoke=False):
    require(type(index) is int and 1<=index<=(1 if smoke else len(TRAIN_CASES)),'undeclared train index')
    seed,scenario=(SMOKE_SEED,'dense24_mixed') if smoke else TRAIN_CASES[index-1]
    return collect_resource_case(output,index=index,seed=seed,scenario=scenario,smoke=smoke,protocol_name=PROTOCOL)


def collect_resource_case(output,*,index,seed,scenario,smoke,protocol_name):
    """Each public entry declares its own immutable train seed list."""
    report=collect_case(output,index=index,seed=seed,scenario=scenario,partition='train',
        smoke=smoke,annotation_wall_seconds=21600,protocol_name=protocol_name,learning_observation_schema=VERSION)
    if report['status'] in ('SMOKE_OK','COLLECTION_OK'):
        try:
            report['resource_reconstruction']=rebuild_resources(Path(output)/f'seed_{seed}',seed=seed,smoke=smoke)
        except Exception as error:report.update(status='ERROR',error=repr(error))
        write(Path(output)/'result.json',report)
    return report


def validated_case(directory,index):
    seed,scenario=TRAIN_CASES[index-1]
    return validated_resource_case(directory,seed=seed,scenario=scenario,protocol_name=PROTOCOL)


def validated_resource_case(directory,*,seed,scenario,protocol_name):
    directory=Path(directory)
    protocol,report=read(directory/'protocol.json'),read(directory/'result.json')
    require(protocol['protocol']==report['protocol']==protocol_name and
        protocol['profile']==report['profile']==PROFILE and
        protocol['partition']==report['partition']=='train' and
        protocol['seed']==report['seed']==seed and protocol['scenario']==report['scenario']==scenario and
        not protocol['smoke'] and protocol['task']==task_settings(build_config(scenario)) and
        protocol['learning_observation_schema']==VERSION,
        'wrong V2.6 train identity')
    require(report['status']=='COLLECTION_OK' and report['error'] is None and not report['blockers'],
        'incomplete V2.6 cycle')
    path=directory/f'seed_{seed}'/'resource_context_observations.jsonl'
    require(sha(path)==report['resource_reconstruction']['sha256'],'resource bytes changed')
    return path,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--index',required=True,type=int)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    result=collect(args.output,args.index,smoke=args.smoke)
    print(json.dumps(dict(status=result['status'],seed=result['seed'],error=result['error'])))
    return 0 if result['status'] in ('SMOKE_OK','COLLECTION_OK') else 2


if __name__=='__main__':raise SystemExit(main())
