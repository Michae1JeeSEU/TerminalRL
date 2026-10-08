"""Bounded train-state counterfactual: pad identity versus aggregate resources."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from terminal_marl.core.landing_zone import LZStatus
from terminal_marl.core.trajectory_context_observation import TrajectoryContextCache
from terminal_marl.training.p3_v25_course_data import TRAIN_CASES, validated_case
from terminal_marl.training.p3_v3_bc_learnability import read, write
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1


def swap_ground_service(env,free_id,occupied_id):
    free,occupied=env.lz_manager.zones[free_id],env.lz_manager.zones[occupied_id]
    assert free.status is LZStatus.FREE and occupied.status is LZStatus.OCCUPIED
    assert free.compatible_missions==occupied.compatible_missions
    assert free.reserved_by is None and occupied.reserved_by is None
    uid=occupied.occupied_by
    assert uid in env.service_uavs and uid not in env.active
    assert env.service_uavs[uid].assigned_lz==occupied_id
    free.status=LZStatus.OCCUPIED;free.occupied_by=uid;free.service_remaining=occupied.service_remaining
    free.expected_release_at=None
    occupied.status=LZStatus.FREE;occupied.occupied_by=None;occupied.service_remaining=0.
    occupied.expected_release_at=None
    env.service_uavs[uid].assigned_lz=free_id
    env.service_uavs[uid].position=free.position.copy()


def equal_observation(a,b):
    return all(np.array_equal(getattr(a,k),getattr(b,k)) for k in ('ego','traffic','traffic_mask'))


def run(directories,output,*,limit=4):
    assert 1<=limit<=4
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();candidates=[]
    for directory,(seed,scenario) in zip(directories,TRAIN_CASES):
        validated_case(directory,seed,scenario,'train')
        episode=Path(directory)/f'seed_{seed}'
        for entry in read(episode/'selected.json'):
            row=entry['row']
            if row['interface']!='i2f':continue
            label=read(episode/'labels'/f"window_{entry['window']:02d}.json")
            if label['teacher_gate']!=1:continue
            candidates.append((label['teacher_proposal']==12,seed,row['tick'],int(row['uid']),episode,entry,label))
    report=dict(status='RUNNING',Q_rows_accessed=False,environment_steps=0,
        max_states=limit,max_teacher_queries=2*limit,wall_seconds=300,states=[])
    write(output/'result.json',report)
    used=set()
    for _,seed,tick,uid,episode,entry,label in sorted(candidates,key=lambda v:v[:4]):
        if len(report['states'])>=limit:break
        if time.monotonic()-started>=300:
            report['status']='BUDGET_EXHAUSTED';break
        if (seed,uid) in used:continue
        path=episode/'snapshots'/entry['snapshot'];data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==entry['snapshot_sha256']
        env=pickle.loads(data);ego=env.active[uid]
        free=[z for z in env.lz_manager.zones if z.status is LZStatus.FREE]
        occupied=[z for z in env.lz_manager.zones if z.status is LZStatus.OCCUPIED
                  and z.occupied_by in env.service_uavs]
        if not free or not occupied:continue
        # Remove the pad used by the preserved feasible witness, rather than
        # an unrelated nearest pad that the original teacher already rejected.
        selected_pad=label['teacher_audit'][-1].get('final_lz_selected_id')
        a=next((z for z in free if z.lz_id==selected_pad),None)
        if a is None:continue
        b=max(occupied,key=lambda z:np.linalg.norm(z.position-ego.position))
        variant=deepcopy(env);swap_ground_service(variant,a.lz_id,b.lz_id)
        before=TrajectoryContextCache(env).build(ego)
        after=TrajectoryContextCache(variant).build(variant.active[uid])
        exact=equal_observation(before,after)
        original,audit=teacher_joint_expert_v4_1(env,ego,state_only=True)
        assert original.index==label['teacher_joint_index'],'original train label did not reproduce'
        changed,changed_audit=teacher_joint_expert_v4_1(variant,variant.active[uid],state_only=True)
        used.add((seed,uid))
        report['states'].append(dict(seed=seed,tick=tick,uid=uid,free_to_occupied=a.lz_id,
            occupied_to_free=b.lz_id,snapshot_sha256=entry['snapshot_sha256'],
            observation_exact=exact,original_joint=original.index,counterfactual_joint=changed.index,
            original_queries=len(audit),counterfactual_queries=len(changed_audit),
            same_observation_different_label=exact and original.index!=changed.index))
        write(output/'result.json',report)
    if report['status']=='RUNNING':report['status']='COMPLETE'
    report.update(elapsed_seconds=time.monotonic()-started,
        witnesses=sum(s['same_observation_different_label'] for s in report['states']))
    write(output/'result.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-directories',nargs=3,required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--limit',type=int,default=4);args=parser.parse_args()
    report=run(args.train_directories,args.output,limit=args.limit)
    print(json.dumps(report))


if __name__=='__main__':main()
