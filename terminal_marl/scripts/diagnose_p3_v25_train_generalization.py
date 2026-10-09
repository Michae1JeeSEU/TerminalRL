"""Train-only leave-one-cycle-out learning curves; no deployable checkpoint."""
import argparse
import json
from pathlib import Path
import time

import torch

from terminal_marl.algo.joint_ppo_v4_2 import hierarchical_bc_loss
from terminal_marl.algo.trajectory_set_actor import TrajectorySetActor
from terminal_marl.training.p3_v25_course_data import load_train, TRAIN_CASES
from terminal_marl.training.p3_v25_set_learnability import measure, synthetic
from terminal_marl.training.p3_v3_bc_learnability import Data, write


def subset(paired, indices):
    data=paired.original
    result=Data(data.x[indices],data.gate[indices],data.proposal[indices],
        tuple(data.ids[i] for i in indices.tolist()),[data.contexts[i] for i in indices.tolist()],data.inputs)
    return result,tuple(t[indices] for t in paired.new)


def run(directories,output,*,smoke=False,resource_context=False,additional_directories=(),ordered_proposals=False,
        conditioned_neighbors=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if ordered_proposals and not resource_context:
        raise ValueError('ordered proposals require resource observations')
    if conditioned_neighbors and (not resource_context or ordered_proposals):
        raise ValueError('conditioned neighbors require resource observations and a single architecture')
    if additional_directories:
        if smoke or not resource_context:raise ValueError('independent cycles require real resource train mode')
        from terminal_marl.training.p3_v26_train_dataset import load_six_cycle_train
        paired=load_six_cycle_train(directories,additional_directories)
    elif resource_context and not smoke:
        from terminal_marl.training.p3_v26_resource_inputs import load_resource_train
        paired=load_resource_train(directories)
    else:
        paired=synthetic() if smoke else load_train(directories)
        if resource_context:
            paired.new=(torch.cat((paired.new[0],torch.zeros(len(paired.original.ids),54)),1),*paired.new[1:])
    seeds=sorted({c['seed'] for c in paired.original.contexts})
    groups=([torch.arange(8),torch.arange(8,16)] if smoke else
        [torch.tensor([i for i,c in enumerate(paired.original.contexts) if c['seed']==s]) for s in seeds])
    if not smoke:
        expected=[s for s,_ in TRAIN_CASES]
        if additional_directories:
            from terminal_marl.training.p3_v26_course_data import TRAIN_CASES as ADDITIONAL_CASES
            expected += [s for s,_ in ADDITIONAL_CASES]
        assert seeds==sorted(expected)
    marks=(1,2) if smoke else (0,100,500,1000,2000,5000)
    protocol=dict(mode='train_cycle_cross_validation',smoke=smoke,model_seed=91911,
        architecture=('ordered_candidate_set_v1' if ordered_proposals else
                      'ego_conditioned_neighbor_set_v1' if conditioned_neighbors else
                      'independent_gate_proposal_set_v1'),
        resource_context=resource_context,ego_dim=paired.new[0].shape[1],
        input_seeds=seeds,measurements=list(marks),updates=marks[-1],learning_rate=.0003,
        batch_size=128,gradient_clip=.5,wall_seconds_per_fold=300,
        Q_rows_accessed=False,environment_steps=0,teacher_queries=0,checkpoint_written=False)
    write(output/'protocol.json',protocol)
    result=dict(status='RUNNING',protocol=protocol,folds=[])
    write(output/'result.json',result)
    for fold,held in enumerate(groups):
        started=time.monotonic()
        mask=torch.ones(len(paired.original.ids),dtype=torch.bool);mask[held]=False
        indices=torch.arange(len(mask))[mask]
        train,tx=subset(paired,indices);validation,vx=subset(paired,held)
        torch.manual_seed(91911)
        if ordered_proposals:
            from terminal_marl.algo.ordered_proposal_actor import OrderedProposalActor
            model=OrderedProposalActor()
        elif conditioned_neighbors:
            from terminal_marl.algo.conditioned_set_actor import ConditionedSetActor
            model=ConditionedSetActor()
        else:
            model=TrajectorySetActor(ego_dim=paired.new[0].shape[1])
        optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
        generator=torch.Generator().manual_seed(91911)
        permutation=torch.randperm(len(indices),generator=generator);offset=0
        train_classes=sorted(set(train.proposal[train.gate.bool()].tolist()))
        test_classes=sorted(set(validation.proposal[validation.gate.bool()].tolist()))
        record=dict(held_seed=None if smoke else seeds[fold],train_rows=len(indices),
            validation_rows=len(held),train_release=int(train.gate.sum()),validation_release=int(validation.gate.sum()),
            unseen_proposals=sorted(set(test_classes)-set(train_classes)),curve=[])
        result['folds'].append(record)
        for update in range(marks[-1]+1):
            if time.monotonic()-started>=300:
                result['status']='BUDGET_EXHAUSTED';write(output/'result.json',result)
                return result
            if update:
                if offset==len(indices):
                    permutation=torch.randperm(len(indices),generator=generator);offset=0
                batch=permutation[offset:offset+128];offset+=len(batch)
                model.train();optimizer.zero_grad(set_to_none=True)
                gate,proposal=model(*(t[batch] for t in tx))
                loss,_=hierarchical_bc_loss(gate,proposal,train.gate[batch],train.proposal[batch])
                if not bool(torch.isfinite(loss)):
                    raise ValueError('nonfinite diagnostic loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
                optimizer.step()
            if update in marks:
                values={}
                for name,x,data in (('train',tx,train),('validation',vx,validation)):
                    metric,_=measure(model,x,data)
                    values[name]=dict(loss=metric['loss'],gate_loss=metric['gate_loss'],
                        proposal_loss=metric['proposal_loss'],overall=metric['groups']['overall'])
                record['curve'].append(dict(update=update,**values))
                write(output/'result.json',result)
        record['elapsed_seconds']=time.monotonic()-started
    result['status']='COMPLETE';write(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-directories',nargs=3,type=Path)
    parser.add_argument('--additional-train-directories',nargs=3,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--ordered-proposals',action='store_true')
    parser.add_argument('--conditioned-neighbors',action='store_true')
    parser.add_argument('--resource-context',action='store_true');args=parser.parse_args()
    if not args.smoke and not args.train_directories:parser.error('train directories required')
    torch.set_num_threads(1)
    result=run(args.train_directories or (),args.output,smoke=args.smoke,resource_context=args.resource_context,
        additional_directories=args.additional_train_directories or (),ordered_proposals=args.ordered_proposals,
        conditioned_neighbors=args.conditioned_neighbors)
    print(json.dumps(dict(status=result['status'],folds=len(result['folds']))))
    return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
