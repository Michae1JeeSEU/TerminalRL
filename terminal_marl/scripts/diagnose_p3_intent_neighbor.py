"""Whole-cycle train development: intent-neighbor actor with/without native supervision."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import torch

from ..algo.intent_neighbor_actor import IntentNeighborActor
from ..algo.joint_ppo_v4_2 import hierarchical_bc_loss
from ..training.p3_neighbor_witness_dataset import WitnessTargets,load_witness_targets,auxiliary_loss
from ..training.p3_v26_train_dataset import load_six_cycle_train
from ..training.p3_v25_set_learnability import synthetic
from ..training.p3_v3_bc_learnability import classification,require,write


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


@torch.no_grad()
def measure_pairs(model,paired,target):
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
    return dict(queries=len(top),known_pairs=int(target.known.sum()),blocked_pairs=int(positive.sum()),
        blocked_recall=rate(pairs>=0,positive),safe_recall=rate(pairs<0,negative),
        blocker_top1=rate(hit,eligible),blocked_queries=int(eligible.sum()),
        auxiliary_loss=float(auxiliary_loss(pairs,scenes,target)))


def smoke_inputs():
    paired=synthetic();n=len(paired.original.ids)
    ego=torch.cat((paired.new[0],torch.zeros(n,54)),-1);ego[:,8:11]=0;ego[:,8]=1
    paired.new=(ego,*paired.new[1:])
    for i,context in enumerate(paired.original.contexts):context['seed']=1 if i<n//2 else 2
    known=paired.new[2].clone();blocked=torch.zeros_like(known,dtype=torch.float)
    blocked[:n//2,0]=1
    witness=WitnessTargets(torch.arange(n),torch.full((n,),12),torch.zeros(n,dtype=torch.long),blocked,known,
        torch.zeros(n),torch.ones(n,dtype=torch.bool),torch.tensor([c['seed'] for c in paired.original.contexts]))
    return paired,witness


def run(original,additional,witness_directory,output,*,smoke=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if smoke:paired,witness=smoke_inputs()
    else:
        paired=load_six_cycle_train(original,additional)
        witness=load_witness_targets(witness_directory,paired)
    seeds=sorted({c['seed'] for c in paired.original.contexts})
    marks=(0,1,2) if smoke else (0,100,500,1000,2000,5000)
    protocol=dict(architecture=IntentNeighborActor.architecture,observation='existing V2.6 159+N50',
        smoke=smoke,input_seeds=seeds,model_seed=91911,arms=['without_auxiliary','native_pair_auxiliary'],
        marks=marks,learning_rate=.0003,batch_size=128,auxiliary_batch_size=64,gradient_clip=.5,
        auxiliary_weight=1.,scene_auxiliary_weight=.25,wall_seconds_per_fold_arm=1200,
        original_loss='unweighted_gate_CE_plus_RELEASE_proposal_CE',
        pair_loss='mean_of_blocked_and_safe_BCE_means_ignoring_unknown',
        Q_accessed=False,teacher_queries=0,environment_steps=0,checkpoint_written=False,
        all_auxiliary_rows_follow_original_whole_cycle_split=True)
    write(output/'protocol.json',protocol)
    result=dict(status='RUNNING',protocol=protocol,folds=[],error=None);write(output/'result.json',result)
    try:
        for seed in seeds:
            train,held,wtrain,wheld=split(paired,witness,seed)
            for arm in protocol['arms']:
                torch.manual_seed(91911);model=IntentNeighborActor();optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
                generator=torch.Generator().manual_seed(91911);aux_generator=torch.Generator().manual_seed(91912)
                permutation=torch.randperm(len(train),generator=generator);offset=0
                record=dict(held_seed=seed,arm=arm,train_rows=len(train),held_rows=len(held),
                    train_auxiliary_queries=len(wtrain),held_auxiliary_queries=len(wheld),curve=[])
                result['folds'].append(record);started=time.monotonic()
                for step in range(marks[-1]+1):
                    require(time.monotonic()-started<1200,'fold wall budget exhausted')
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
                            loss=loss+auxiliary_loss(pair,scene,target)
                        require(bool(torch.isfinite(loss)),'nonfinite loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True);optimizer.step()
                    if step in marks:
                        record['curve'].append(dict(update=step,train=measure(model,paired,train),validation=measure(model,paired,held),
                            train_pairs=measure_pairs(model,paired,witness.take(wtrain)),
                            validation_pairs=measure_pairs(model,paired,witness.take(wheld))))
                        record['elapsed_seconds']=time.monotonic()-started;write(output/'result.json',result)
                record['elapsed_seconds']=time.monotonic()-started
        result['status']='COMPLETE'
    except Exception as error:result.update(status='INCOMPLETE',error=repr(error))
    write(output/'result.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train-directories',nargs=3,type=Path)
    p.add_argument('--additional-train-directories',nargs=3,type=Path);p.add_argument('--witness-directory',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if not a.smoke and not all((a.train_directories,a.additional_train_directories,a.witness_directory)):p.error('all train and witness inputs required')
    torch.set_num_threads(1)
    result=run(a.train_directories,a.additional_train_directories,a.witness_directory,a.output,smoke=a.smoke)
    print(json.dumps(dict(status=result['status'],folds=len(result['folds']),error=result['error'])))
    return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
