"""Bounded train-only candidate/destination/neighbor supervision from saved states.

These are offline counterfactual pair checks of an unchanged canonical plan.
They neither replace original teacher labels nor enter the policy observation.
No environment reset/step or Q input is supported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import time

import numpy as np

from ..core.config import RingRole
from ..core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from ..core.merge_scheduler import LowLevelIntent, RepairCandidate
from ..core.resource_context_observation import build_low_observation_v2_6
from ..core.safety_filter import SafetyFilter
from ..core.safety_trace import _digest, _json
from ..core.uav import FlightPhase
from ..envs.parallel_env import WorldSnapshot
from ..scripts.diagnose_p3_direct_contract import canonical, state_record
from ..scripts.diagnose_p3_teacher_context import NativeWitness, state_delta
from . import p3_v25_course_data as old
from . import p3_v26_course_data as new
from .p3_v3_bc_learnability import read, write, require
from .p3_v25_imitation_qualification import sha
from .v4_1_joint_teacher import _release_context

PROTOCOL='p3_train_neighbor_witness_20261009_v1'
TRAFFIC_REASONS=('predicted_', 'nominal_')
MAX_STATES=144
MAX_PLAN_CALLS=1728  # 144 states x 2 intents x 6 physical destinations


def stratified(rows):
    """Four evenly spaced decisions per interface/label, separately per cycle."""
    groups=defaultdict(list)
    for row in rows:
        require(row['partition']=='train' and row['actor_mask'], 'only actor train rows')
        groups[(row['context']['interface'],int(row['teacher_gate']))].append(row)
    chosen=[]
    for key in sorted(groups):
        group=sorted(groups[key],key=lambda r:(r['tick'],int(r['uid'])))
        chosen.extend(group[i] for i in np.linspace(0,len(group)-1,min(4,len(group)),dtype=int))
    return sorted(chosen,key=lambda r:(r['tick'],int(r['uid'])))


def candidate_indices(label):
    # Positive states add their first feasible intent; all-WAIT states add a
    # fixed extreme intent already checked by the original teacher. Unknown
    # suffixes of RELEASE labels are never fabricated as negative examples.
    values=[13]
    alternate=int(label['teacher_joint_index']) if label['teacher_gate'] else 1
    if alternate!=13:values.append(alternate)
    checked={x['joint_index'] for x in label['teacher_audit']}
    require(set(values)<=checked,'candidate was not in original checked prefix')
    return values


def pair_check(env,ego,other,plan):
    """No rerouting after removal: the exact original plan is tested per pair."""
    safety=SafetyFilter(env.config)
    witness=NativeWitness(safety);witness.install()
    snapshot=WorldSnapshot(env.now,{ego.uid:ego,other.uid:other},env.airspace)
    try:
        verdict=safety.filter_plan(plan,{other.uid:other.committed_plan} if other.committed_plan else {},snapshot)
    finally:witness.restore()
    blocked=(False if verdict.approved else True if verdict.reason.startswith(TRAFFIC_REASONS) else None)
    return dict(uid=int(other.uid),blocked=blocked,approved=verdict.approved,reason=verdict.reason,events=witness.events)


def global_block_reason(env,ego,target,interface):
    if interface in env.airspace.affected_interfaces():return 'geometry_frozen'
    if ego.soc<=0:return 'soc'
    if target=='final':
        if sum(u.phase is FlightPhase.FINAL for u in env.active.values())>=env.config.final_queue_capacity:
            return 'final_capacity'
        if env._emergency_lz_priority_blocks(ego):return 'emergency_lz_priority'
        if not env.lz_manager.rank_available_zones(ego,env.now):return 'lz_unavailable'
    return None


def witness_state(episode,entry,row,label,*,remaining_calls):
    name=entry['snapshot'];require(Path(name).name==name,'invalid snapshot filename')
    path=episode/'snapshots'/name
    require(sha(path)==entry['snapshot_sha256'],'snapshot hash differs')
    env=pickle.loads(path.read_bytes())  # explicit completed train run + per-file SHA
    before=canonical(state_record(env));require(_digest(before)==row['pre_state_hash']==label['pre_state_hash'], 'pre-state differs')
    require(not env.config.task_priority_enabled,'random priority must remain disabled')
    ego=env.active[int(row['uid'])];target,interface=_release_context(env,ego)
    observation=build_low_observation_v2_6(ego,env)
    require(np.allclose(observation.ego[:105],row['ego'][:105],rtol=1e-6,atol=1e-7)
        and np.allclose(observation.traffic,np.asarray(row['traffic']).reshape(-1,50),rtol=1e-6,atol=1e-7),
        'saved observation differs')
    values=dict(decision_id=row['decision_id'],seed=row['seed'],tick=row['tick'],uid=row['uid'],
        interface=interface,teacher_gate=label['teacher_gate'],teacher_joint=label['teacher_joint_index'],
        snapshot_sha256=entry['snapshot_sha256'],ego=row['ego'][:105]+observation.ego[105:].tolist(),
        traffic=row['traffic'],traffic_mask=row['traffic_mask'],traffic_uids=observation.traffic_uids,
        candidates=[],plan_calls=0)
    blocked=global_block_reason(env,ego,target,interface)
    snapshot=WorldSnapshot(env.now,env.active,env.airspace)
    for joint in candidate_indices(label):
        expected=next(a for a in label['teacher_audit'] if a['joint_index']==joint)
        item=dict(joint=joint,original_feasible=expected['feasible'],original_reason=expected['reason'],
            global_reason=blocked,destinations=[])
        values['candidates'].append(item)
        if blocked:
            require(not expected['feasible'] and expected['reason']==blocked,'global rejection differs')
            continue
        destinations=list(env.lz_manager.rank_available_zones(ego,env.now)) if target=='final' else [None]
        first_ok=None;last_reason=None
        for destination in destinations:
            require(values['plan_calls']<remaining_calls,'native plan budget exhausted')
            action=JOINT_ACTION_CATALOG_V4_1[joint]
            proposal,plan,reason=env._candidate_plan(ego,
                RepairCandidate('offline_neighbor_witness',LowLevelIntent(*action.normalized),0.),
                target,snapshot,destination)
            values['plan_calls']+=1
            ok=proposal is not None and plan is not None and reason is None
            if ok and first_ok is None:first_ok=destination
            last_reason=reason or 'certified'
            pairs=[]
            if plan is not None:
                for uid in observation.traffic_uids:pairs.append(pair_check(env,ego,env.active[uid],plan))
            item['destinations'].append(dict(pad=destination,feasible=ok,reason=last_reason,
                plan_hash=None if plan is None else plan.trajectory_hash,
                duration=None if plan is None else plan.end_time-plan.start_time,pairs=pairs))
        feasible=any(d['feasible'] for d in item['destinations'])
        require(feasible==expected['feasible'],'original teacher feasibility differs')
        if not feasible:require(last_reason==expected['reason'],'original rejection differs')
        if target=='final' and feasible:
            require(first_ok==expected['final_lz_selected_id'],'first feasible pad differs')
        # Approval must imply every isolated pair passes. Rejection can also
        # come from resources/energy/dynamics, so unknown pairs stay null.
        for d in item['destinations']:
            if d['feasible']:require(all(p['approved'] for p in d['pairs']),'approved candidate has blocked pair')
    values['query_counter_delta']=state_delta(before,canonical(state_record(env)))
    return values


def selected_inputs(directories):
    require(len(directories)==6,'six complete train cycles required')
    selected=[];inputs=[]
    for index,directory in enumerate(map(Path,directories)):
        if index<3:
            seed,scenario=old.TRAIN_CASES[index]
            path,report=old.validated_case(directory,seed,scenario,'train')
        else:
            seed,scenario=new.TRAIN_CASES[index-3]
            path,report=new.validated_case(directory,index-2)
        episode=directory/f'seed_{seed}'
        entries={e['row']['decision_id']:e for e in read(episode/'selected.json')}
        rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        annotations={e['decision_id']:e for e in report['annotation']['entries']}
        for row in stratified(rows):
            did=row['decision_id'];entry=entries[did];label=read(episode/annotations[did]['label_file'])
            require(label['status']=='OK' and label['decision_id']==did and label['actor_mask'],'invalid original label')
            # All current context formats preserve this hash in the raw entry.
            row=dict(row,pre_state_hash=entry['row']['pre_state_hash'])
            selected.append((episode,entry,row,label))
        inputs.append(dict(seed=seed,context_sha256=sha(path),selected=sum(x[2]['seed']==seed for x in selected)))
    require(len(selected)<=MAX_STATES,'state budget exceeded')
    return selected,inputs


def collect(directories,output,*,smoke=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    selected,inputs=selected_inputs(directories)
    if smoke:
        # One original train state per interface, used only to validate output.
        selected=[next(x for x in selected if x[2]['context']['interface']==interface)
                  for interface in ('o2m','m2i','i2f')]
    protocol=dict(protocol=PROTOCOL,inputs=inputs,smoke=smoke,selected_ids=[x[2]['decision_id'] for x in selected],
        max_states=MAX_STATES,max_plan_calls=MAX_PLAN_CALLS,wall_seconds=120 if smoke else 3600,
        resets=0,steps=0,Q_accessed=False,original_labels_unchanged=True,
        pair_semantics='same_canonical_plan_isolated_native_safety_check',unknown_is_not_safe_or_blocked=True)
    write(output/'protocol.json',protocol)
    started=time.monotonic();result=dict(status='RUNNING',states=0,plan_calls=0,error=None)
    write(output/'result.json',result)
    try:
        with (output/'witnesses.jsonl').open('x',encoding='utf8') as stream:
            for episode,entry,row,label in selected:
                require(time.monotonic()-started<protocol['wall_seconds'],'wall budget exhausted')
                witness=witness_state(episode,entry,row,label,remaining_calls=MAX_PLAN_CALLS-result['plan_calls'])
                stream.write(json.dumps(_json(witness),allow_nan=False)+'\n');stream.flush()
                result['states']+=1;result['plan_calls']+=witness['plan_calls']
                result['elapsed_seconds']=time.monotonic()-started;write(output/'result.json',result)
        require(time.monotonic()-started<protocol['wall_seconds'],'wall budget exhausted')
        result.update(status='SMOKE_OK' if smoke else 'COMPLETE',witness_sha256=sha(output/'witnesses.jsonl'))
    except Exception as error:result.update(status='INCOMPLETE',error=repr(error))
    result['elapsed_seconds']=time.monotonic()-started;write(output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train-directories',nargs=6,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    result=collect(a.train_directories,a.output,smoke=a.smoke);print(json.dumps(result))
    return 0 if result['status'] in ('SMOKE_OK','COMPLETE') else 2


if __name__=='__main__':raise SystemExit(main())
