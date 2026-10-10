"""Whole-cycle train development: intent-neighbor actor with/without native supervision."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import multiprocessing as mp
import os
from pathlib import Path
import json
import time

import torch

from ..algo.intent_neighbor_actor import IntentNeighborActor
from ..algo.joint_ppo_v4_2 import hierarchical_bc_loss
from ..training.p3_neighbor_witness_dataset import (WitnessTargets,load_witness_targets,
    combine_witness_targets,auxiliary_loss,blocker_ranking_loss,SUPPLEMENTAL_PROTOCOL,PREFIX_COVERAGE_PROTOCOL)
from ..training.p3_v26_train_dataset import load_six_cycle_train,load_nine_cycle_train
from ..training.p3_v25_set_learnability import synthetic
from ..training.p3_v3_bc_learnability import classification,read,require,write


def split(paired,witness,held_seed):
    seeds=torch.tensor([c['seed'] for c in paired.original.contexts])
    held=(seeds==held_seed);wh=(witness.seeds==held_seed)
    require(torch.equal(seeds[witness.state_index],witness.seeds),'auxiliary split identity mismatch')
    return torch.where(~held)[0],torch.where(held)[0],torch.where(~wh)[0],torch.where(wh)[0]


@torch.no_grad()
def measure(model,paired,indices):
    gates=[];proposals=[]
    for batch in indices.split(32):
        gate,proposal=model(*(t[batch] for t in paired.new));gates.append(gate);proposals.append(proposal)
    gate=torch.cat(gates);proposal=torch.cat(proposals)
    y=paired.original
    loss,pieces=hierarchical_bc_loss(gate,proposal,y.gate[indices],y.proposal[indices])
    return dict(loss=float(loss),gate_loss=float(pieces['gate']),proposal_loss=float(pieces['proposal']),
        overall=classification(gate,proposal,y.gate[indices],y.proposal[indices]))


def _ranking_support(target):
    known=target.known.bool()
    blocked=known & target.blocked.bool()
    safe=known & ~target.blocked.bool()
    eligible=blocked.any(-1) & safe.any(-1)
    return dict(queries=int(len(target.state_index)),known_pairs=int(known.sum()),
        blocked_pairs=int(blocked.sum()),eligible_queries=int(eligible.sum()))


@torch.no_grad()
def measure_pairs(model,paired,target,*,blocker_ranking_weight=0.):
    pairs=[];scenes=[]
    for batch in torch.arange(len(target.state_index)).split(64):
        t=target.take(batch)
        pair,scene=model.queried_scores(*(x[t.state_index] for x in paired.new),t.proposal,t.pad)
        pairs.append(pair);scenes.append(scene)
    pairs=torch.cat(pairs);scenes=torch.cat(scenes)
    positive=target.known & target.blocked.bool();negative=target.known & ~target.blocked.bool()
    eligible=positive.any(-1)
    # Top-risk counts as an identified blocker only if that very neighbor was
    # explicitly checked and blocked. Unknown alternatives are not negatives.
    top=pairs.masked_fill(~paired.new[2][target.state_index],-torch.inf).argmax(-1)
    hit=positive[torch.arange(len(top)),top]
    def rate(values,mask):return float(values[mask].float().mean()) if mask.any() else None
    result=dict(queries=len(top),known_pairs=int(target.known.sum()),blocked_pairs=int(positive.sum()),
        blocked_recall=rate(pairs>=0,positive),safe_recall=rate(pairs<0,negative),
        blocker_top1=rate(hit,eligible),blocked_queries=int(eligible.sum()),
        auxiliary_loss=float(auxiliary_loss(pairs,scenes,target)))
    if blocker_ranking_weight:
        result.update(blocker_ranking_loss=float(blocker_ranking_loss(pairs,target)),
            blocker_ranking_queries=_ranking_support(target)['eligible_queries'])
    return result


def smoke_inputs(*,supplemental=False):
    paired=synthetic();n=len(paired.original.ids)
    ego=torch.cat((paired.new[0],torch.zeros(n,54)),-1);ego[:,8:11]=0;ego[:,8]=1
    paired.new=(ego,*paired.new[1:])
    if supplemental:
        seeds=tuple(range(1,10))
        for i,context in enumerate(paired.original.contexts):context['seed']=seeds[i%len(seeds)]
    else:
        for i,context in enumerate(paired.original.contexts):context['seed']=1 if i<n//2 else 2
    known=paired.new[2].clone();blocked=torch.zeros_like(known,dtype=torch.float)
    blocked[:n//2,0]=1
    witness=WitnessTargets(torch.arange(n),torch.full((n,),12),torch.zeros(n,dtype=torch.long),blocked,known,
        torch.zeros(n),torch.ones(n,dtype=torch.bool),torch.tensor([c['seed'] for c in paired.original.contexts]))
    return paired,witness


def _atomic_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'{path.name}.{os.getpid()}.tmp')
    write(temporary,value)
    for attempt in range(200):
        try:
            temporary.replace(path);return
        except PermissionError:
            if attempt==199:raise
            time.sleep(.01)


def _progress_path(output,seed,arm):
    return Path(output)/f'fold_{int(seed)}_{arm}.progress.json'


def _progress_payload(record,*,status,stage,error=None,progress_path=None):
    return dict(schema='p3_neighbor_fold_progress_20261010_v1',status=status,stage=stage,
        held_seed=record['held_seed'],arm=record['arm'],curve=record['curve'],
        elapsed_seconds=record.get('elapsed_seconds',0.),error=error,record=record,
        progress_path=None if progress_path is None else str(progress_path))


def _initial_fold(paired,witness,seed,arm,progress_path):
    train,held,wtrain,wheld=split(paired,witness,seed)
    record=dict(held_seed=seed,arm=arm,train_rows=len(train),held_rows=len(held),
        train_auxiliary_queries=len(wtrain),held_auxiliary_queries=len(wheld),curve=[],
        progress_path=str(progress_path))
    _atomic_write(progress_path,_progress_payload(record,status='RUNNING',stage='initial',
        progress_path=progress_path))
    return record


def train_fold(paired,witness,seed,arm,gate_mode,marks,wall_seconds,progress_path,
               reference_witness=None,blocker_ranking_weight=0.):
    """Train one deterministic seed/arm and persist independently readable progress."""
    progress_path=Path(progress_path)
    record=_initial_fold(paired,witness,seed,arm,progress_path)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(91911);model=IntentNeighborActor(gate_mode=gate_mode);optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
        generator=torch.Generator().manual_seed(91911);aux_generator=torch.Generator().manual_seed(91912)
        train,held,wtrain,wheld=split(paired,witness,seed)
        reference_wheld=None
        if reference_witness is not None:
            _,_,_,reference_wheld=split(paired,reference_witness,seed)
        permutation=torch.randperm(len(train),generator=generator);offset=0
        started=time.monotonic()
        for step in range(marks[-1]+1):
            require(time.monotonic()-started<wall_seconds,'fold wall budget exhausted')
            if step:
                if offset==len(train):permutation=torch.randperm(len(train),generator=generator);offset=0
                local=permutation[offset:offset+128];offset+=len(local);batch=train[local]
                optimizer.zero_grad(set_to_none=True)
                gate,proposal=model(*(t[batch] for t in paired.new))
                loss,_=hierarchical_bc_loss(gate,proposal,paired.original.gate[batch],paired.original.proposal[batch])
                if arm=='native_pair_auxiliary':
                    chosen=wtrain[torch.randint(len(wtrain),(64,),generator=aux_generator)]
                    target=witness.take(chosen)
                    pair,scene=model.queried_scores(*(t[target.state_index] for t in paired.new),target.proposal,target.pad)
                    loss=loss+auxiliary_loss(pair,scene,target,
                        blocker_ranking_weight=blocker_ranking_weight)
                require(bool(torch.isfinite(loss)),'nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True);optimizer.step()
            if step in marks:
                curve_item=dict(update=step,train=measure(model,paired,train),validation=measure(model,paired,held),
                    train_pairs=measure_pairs(model,paired,witness.take(wtrain),
                        blocker_ranking_weight=blocker_ranking_weight),
                    validation_pairs=measure_pairs(model,paired,witness.take(wheld),
                        blocker_ranking_weight=blocker_ranking_weight))
                if reference_witness is not None:
                    curve_item['validation_reference_pairs']=measure_pairs(
                        model,paired,reference_witness.take(reference_wheld),
                        blocker_ranking_weight=blocker_ranking_weight)
                record['curve'].append(curve_item)
                record['elapsed_seconds']=time.monotonic()-started
                _atomic_write(progress_path,_progress_payload(record,status='RUNNING',stage=f'mark_{step}',
                    progress_path=progress_path))
        record['elapsed_seconds']=time.monotonic()-started
        _atomic_write(progress_path,_progress_payload(record,status='COMPLETE',stage='terminal',
            progress_path=progress_path))
        return dict(status='COMPLETE',record=record,error=None,progress_path=str(progress_path))
    except Exception as error:
        record['elapsed_seconds']=time.monotonic()-locals().get('started',time.monotonic())
        _atomic_write(progress_path,_progress_payload(record,status='INCOMPLETE',stage='terminal',
            error=repr(error),progress_path=progress_path))
        return dict(status='INCOMPLETE',record=record,error=repr(error),progress_path=str(progress_path))


def _train_fold_job(paired,witness,seed,arm,gate_mode,marks,wall_seconds,progress_path,
                    reference_witness=None,blocker_ranking_weight=0.):
    torch.set_num_threads(1)
    return train_fold(paired,witness,seed,arm,gate_mode,marks,wall_seconds,progress_path,
        reference_witness=reference_witness,blocker_ranking_weight=blocker_ranking_weight)


def _read_progress(path):
    try:return read(path)
    except (FileNotFoundError,PermissionError,json.JSONDecodeError):return None


def _progress_signature(specs):
    signature=[]
    for spec in specs:
        path=Path(spec['progress_path'])
        try:
            stat=path.stat();signature.append((str(path),stat.st_mtime_ns,stat.st_size))
        except FileNotFoundError:
            signature.append((str(path),None,None))
    return tuple(signature)


def _ordered_folds(specs):
    folds=[]
    for spec in specs:
        progress=_read_progress(spec['progress_path'])
        if progress is None:continue
        record=dict(progress.get('record') or {})
        if record:folds.append(record)
    return folds


def _write_root_result(path,protocol,specs,status,error=None):
    payload=dict(status=status,protocol=protocol,folds=_ordered_folds(specs),error=error)
    _atomic_write(path,payload)
    return payload


def _mark_progress_failure(spec,error):
    progress=_read_progress(spec['progress_path'])
    if progress is None:return
    progress.update(status='INCOMPLETE',stage='terminal',error=error)
    _atomic_write(spec['progress_path'],progress)


def _execute_folds(paired,witness,specs,*,workers,root_path,protocol,reference_witness=None,
                   blocker_ranking_weight=0.):
    for spec in specs:
        _initial_fold(paired,witness,spec['seed'],spec['arm'],spec['progress_path'])
    _write_root_result(root_path,protocol,specs,'RUNNING')
    root_error=None
    if workers==1:
        for spec in specs:
            outcome=train_fold(paired,witness,spec['seed'],spec['arm'],spec['gate_mode'],
                spec['marks'],spec['wall_seconds'],spec['progress_path'],
                reference_witness=reference_witness,blocker_ranking_weight=blocker_ranking_weight)
            if outcome['status']!='COMPLETE' and root_error is None:root_error=outcome['error']
            _write_root_result(root_path,protocol,specs,'RUNNING',root_error)
            if root_error is not None:break
        return _write_root_result(root_path,protocol,specs,
            'INCOMPLETE' if root_error else 'COMPLETE',root_error)
    context=mp.get_context('spawn')
    futures={}
    next_index=0
    last_root_write=time.monotonic()
    last_signature=_progress_signature(specs)

    def write_root_if_due(status,*,force=False):
        nonlocal last_root_write,last_signature
        now=time.monotonic();signature=_progress_signature(specs)
        if force or now-last_root_write>=2. or signature!=last_signature:
            _write_root_result(root_path,protocol,specs,status,root_error)
            last_root_write=now;last_signature=signature

    with ProcessPoolExecutor(max_workers=min(workers,len(specs)),mp_context=context) as pool:
        pending=set()

        def submit_next():
            nonlocal next_index
            if root_error is not None or next_index>=len(specs):return False
            spec=specs[next_index];next_index+=1
            future=pool.submit(_train_fold_job,paired,witness,spec['seed'],spec['arm'],spec['gate_mode'],
                spec['marks'],spec['wall_seconds'],str(spec['progress_path']),reference_witness,
                blocker_ranking_weight)
            futures[future]=spec;pending.add(future)
            return True

        while len(pending)<workers and submit_next():
            pass
        while pending:
            done,pending=wait(pending,timeout=.2,return_when=FIRST_COMPLETED)
            if not done:
                write_root_if_due('INCOMPLETE' if root_error else 'RUNNING')
                continue
            for future in done:
                spec=futures[future]
                try:outcome=future.result()
                except Exception as error:
                    outcome=dict(status='INCOMPLETE',error=repr(error));_mark_progress_failure(spec,repr(error))
                if outcome['status']!='COMPLETE' and root_error is None:root_error=outcome.get('error')
            if root_error is None:
                while len(pending)<workers and submit_next():
                    pass
            else:
                # There are never more than ``workers`` submitted futures, so
                # no unbounded executor queue can survive a sibling failure.
                for future in pending:
                    future.cancel()
            write_root_if_due('INCOMPLETE' if root_error else
                ('RUNNING' if pending else 'COMPLETE'),force=True)
    return _write_root_result(root_path,protocol,specs,
        'INCOMPLETE' if root_error else 'COMPLETE',root_error)


def run(original,additional,witness_directory,output,*,smoke=False,max_candidate_gate=False,
        supplemental=None,supplemental_witness=None,prefix_witness=None,workers=1,
        blocker_ranking_weight=0.):
    supplemental_route=supplemental is not None or supplemental_witness is not None
    prefix_route=prefix_witness is not None
    require((supplemental is None)==(supplemental_witness is None),
        'supplemental train and witness inputs must be provided together')
    require(not (supplemental_route and max_candidate_gate),
        'supplemental route does not support max_candidate_gate')
    require(isinstance(workers,int) and not isinstance(workers,bool) and 1<=workers<=4,
        'workers must be an integer in 1..4')
    require(not prefix_route or supplemental_route and not smoke,
        'prefix witness requires the paired nine-cycle supplemental route')
    require(not prefix_route or not max_candidate_gate,
        'prefix witness does not support max_candidate_gate')
    require(blocker_ranking_weight in (0.,.25),'blocker ranking weight is fixed at 0.25')
    require(not blocker_ranking_weight or prefix_route and supplemental_route and not smoke,
        'blocker ranking requires the non-smoke prefix supplemental route')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    reference_witness=None
    if smoke:paired,witness=smoke_inputs(supplemental=supplemental_route)
    elif supplemental_route:
        paired=load_nine_cycle_train(original,additional,supplemental)
        parts=[load_witness_targets(witness_directory,paired),load_witness_targets(supplemental_witness,paired)]
        if prefix_route:reference_witness=combine_witness_targets(tuple(parts))
        if prefix_route:parts.append(load_witness_targets(prefix_witness,paired))
        witness=combine_witness_targets(tuple(parts))
    else:
        paired=load_six_cycle_train(original,additional)
        witness=load_witness_targets(witness_directory,paired)
    seeds=sorted({c['seed'] for c in paired.original.contexts})
    marks=(0,1,2) if smoke else (0,100,500,1000,2000,5000)
    gate_mode='max_candidate' if max_candidate_gate else 'first_feasible_union'
    arms=['native_pair_auxiliary'] if (max_candidate_gate or supplemental_route) else ['without_auxiliary','native_pair_auxiliary']
    protocol=dict(architecture=IntentNeighborActor.architecture,observation='existing V2.6 159+N50',
        architecture_variant=f'{IntentNeighborActor.architecture}__gate_{gate_mode}',gate_mode=gate_mode,
        smoke=smoke,input_seeds=seeds,model_seed=91911,arms=arms,
        marks=marks,learning_rate=.0003,batch_size=128,auxiliary_batch_size=64,gradient_clip=.5,
        auxiliary_weight=1.,scene_auxiliary_weight=.25,wall_seconds_per_fold_arm=1200,
        original_loss='unweighted_gate_CE_plus_RELEASE_proposal_CE',
        pair_loss='mean_of_blocked_and_safe_BCE_means_ignoring_unknown',
        Q_accessed=False,teacher_queries=0,environment_steps=0,checkpoint_written=False,
        all_auxiliary_rows_follow_original_whole_cycle_split=True,workers=workers)
    if supplemental_route:
        protocol.update(source_group='old6_plus_supplemental3',
            comparison_control_reference='p3_neighbor_relation_cv_20261009/six_cycle',
            comparison_note='old6 native_pair_auxiliary control reference; supplemental three train-only development cycles, not qualification',
            supplemental_train_seeds=(seeds if smoke else [893604,893605,893606]),
            supplemental_role='train_only_development_not_qualification',
            supplemental_witness_protocol=SUPPLEMENTAL_PROTOCOL,
            qualification=False)
    if prefix_route:
        protocol.update(source_group='old6_plus_supplemental3_plus_prefix',
            comparison_control_reference='p3_neighbor_nine_cycle_cv_20261009/nine_cycle',
            prefix_witness_protocol=PREFIX_COVERAGE_PROTOCOL,
            prefix_witness_directory=str(prefix_witness),
            prefix_role='new_train_decision_joint_candidates_only_not_qualification',
            prefix_train_seeds=seeds,
            qualification=False)
    if blocker_ranking_weight:
        require(reference_witness is not None,'blocker ranking requires a reference witness')
        protocol.update(loss_variant='native_pair_auxiliary_plus_blocker_ranking',
            comparison_control_reference='p3_neighbor_prefix_ring_reuse_20261010/nine_cycle',
            comparison_note=
                'same nine-fold 3149 actor rows; same 1270 expanded and 473 original-held '
                'reference supports; sole loss change is fixed 0.25 blocker ranking',
            blocker_ranking_loss_variant='known_blocked_mass_within_known_neighbors',
            blocker_ranking_weight=float(blocker_ranking_weight),
            blocker_ranking_formula=
                'mean_eligible[logsumexp(known_logits)-logsumexp(known_blocked_logits)], '
                'eligible requires at least one known blocked and one known safe; unknown/absent excluded',
            blocker_ranking_reference_support=_ranking_support(reference_witness),
            blocker_ranking_training_support=_ranking_support(witness))
    if max_candidate_gate:
        protocol.update(comparison_control_reference='p3_neighbor_relation_cv_20261009/six_cycle',
            comparison_note='max gate tests count-amplification; no promotion')
    write(output/'protocol.json',protocol)
    result_path=output/'result.json';result=dict(status='RUNNING',protocol=protocol,folds=[],error=None)
    _atomic_write(result_path,result)
    try:
        torch.set_num_threads(1)
        specs=[]
        for seed in seeds:
            for arm in protocol['arms']:
                specs.append(dict(seed=seed,arm=arm,gate_mode=gate_mode,marks=marks,wall_seconds=1200,
                    progress_path=_progress_path(output,seed,arm)))
        result=_execute_folds(paired,witness,specs,workers=workers,root_path=result_path,protocol=protocol,
            reference_witness=reference_witness,blocker_ranking_weight=blocker_ranking_weight)
    except Exception as error:result.update(status='INCOMPLETE',error=repr(error))
    _atomic_write(result_path,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train-directories',nargs=3,type=Path)
    p.add_argument('--additional-train-directories',nargs=3,type=Path);p.add_argument('--witness-directory',type=Path)
    p.add_argument('--supplemental-train-directories',nargs=3,type=Path)
    p.add_argument('--supplemental-witness-directory',type=Path)
    p.add_argument('--prefix-witness-directory',type=Path)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-candidate-gate',action='store_true',
        help='run the bounded max-candidate gate hypothesis only')
    p.add_argument('--blocker-ranking',action='store_true',
        help='enable the fixed 0.25 known-blocker ranking loss on the prefix route');a=p.parse_args()
    supplemental_args=(a.supplemental_train_directories,a.supplemental_witness_directory)
    if (supplemental_args[0] is None)!=(supplemental_args[1] is None):
        p.error('--supplemental-train-directories and --supplemental-witness-directory must be provided together')
    if not a.smoke and not all((a.train_directories,a.additional_train_directories,a.witness_directory)):p.error('all train and witness inputs required')
    if a.max_candidate_gate and supplemental_args[0] is not None:p.error('--max-candidate-gate cannot be combined with supplemental inputs')
    if a.prefix_witness_directory is not None and supplemental_args[0] is None:
        p.error('--prefix-witness-directory requires supplemental nine-cycle inputs')
    if a.prefix_witness_directory is not None and a.smoke:
        p.error('--prefix-witness-directory is not available in smoke mode')
    if a.blocker_ranking and (a.prefix_witness_directory is None or a.smoke or supplemental_args[0] is None):
        p.error('--blocker-ranking requires the non-smoke prefix supplemental route')
    if not 1<=a.workers<=4:p.error('--workers must be in 1..4')
    torch.set_num_threads(1)
    result=run(a.train_directories,a.additional_train_directories,a.witness_directory,a.output,smoke=a.smoke,
        max_candidate_gate=a.max_candidate_gate,supplemental=a.supplemental_train_directories,
        supplemental_witness=a.supplemental_witness_directory,prefix_witness=a.prefix_witness_directory,
        workers=a.workers,blocker_ranking_weight=.25 if a.blocker_ranking else 0.)
    print(json.dumps(dict(status=result['status'],folds=len(result['folds']),error=result['error'])))
    return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
