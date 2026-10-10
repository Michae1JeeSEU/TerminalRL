"""Fixed endpoint error decomposition. No training, reset, or teacher query."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from terminal_marl.training import p3_v3_bc_learnability as bc


def ledger(g,p,y,q):
    result=dict(correct_WAIT=int(((y==0)&(g==0)).sum()),
        false_RELEASE=int(((y==0)&(g==1)).sum()),
        false_WAIT=int(((y==1)&(g==0)).sum()),
        correct_RELEASE=int(((y==1)&(g==1)&(p==q)).sum()),
        wrong_proposal_after_correct_RELEASE_gate=int(((y==1)&(g==1)&(p!=q)).sum()))
    assert sum(result.values())==len(y)
    return result


def prefix_status(label,proposal):
    if proposal==label['teacher_proposal']:
        return 'teacher_target'
    candidates=[r for r in label['teacher_audit'] if r['joint_index']==proposal+1]
    assert len(candidates)<=1
    if not candidates:
        return 'unqueried'
    return 'checked_feasible' if candidates[0]['feasible'] else 'checked_rejected'


def self_test():
    value=ledger(torch.tensor([0,1,0,1,1]),torch.tensor([0,0,0,2,0]),
        torch.tensor([0,0,1,1,1]),torch.tensor([-1,-1,2,2,2]))
    assert set(value.values())=={1}
    label=dict(teacher_proposal=12,teacher_audit=[dict(joint_index=13,feasible=True),dict(joint_index=12,feasible=False)])
    assert prefix_status(label,12)=='teacher_target'
    assert prefix_status(label,11)=='checked_rejected'
    assert prefix_status(label,14)=='unqueried'
    assert json.loads(json.dumps(value,allow_nan=False))==value
    print(json.dumps(dict(status='ERROR_PROBE_SELF_TEST_OK',ledger=value)))


def run(root,output):
    assert not output.exists()
    torch.set_num_threads(1)
    data=bc.load_inputs(root)
    raw,labels={},{}
    for seed,relative,_ in bc.CASES:
        episode=root/relative/f'seed_{seed}'
        for line in (episode/'rows.jsonl').read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            if row['actor_mask']:raw[row['decision_id']]=row
        for entry in bc.read(episode/'annotation_result.json')['entries']:
            label=bc.read(episode/entry['label_file'])
            labels[label['decision_id']]=label
    assert set(raw)==set(labels)==set(data.ids)
    for i,did in enumerate(data.ids):
        assert labels[did]['teacher_gate']==int(data.gate[i])
        assert labels[did]['teacher_proposal']==(int(data.proposal[i]) if data.gate[i]==1 else None)
    cases={'original':'p3_v3_bc_learnability_20261007/fit',
           'separate':'p3_v3_bc_separate_encoders_20261007/fit'}
    result=dict(status='ERROR_PROBE_COMPLETE',parameter_updates=0,optimizer_created=False,
        environment_resets=0,teacher_queries=0,checkpoint_writes=0,checkpoints=[],groups={},examples=[])
    predictions={name:[] for name in cases}
    for name,relative in cases.items():
        for sid in bc.MODEL_SEEDS:
            case=root/relative/f'seed_{sid}'
            path=case/'final_diagnostic.pt'
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            cp=torch.load(path,map_location='cpu',weights_only=True)
            model=(bc.SeparateEncodersDiagnosticV42(90) if name=='separate' else bc.HierarchicalActorCentralCriticV42(90))
            assert cp['model_class']==type(model).__name__
            model.load_state_dict(cp['model_state'])
            model.eval()
            measured=bc.measure(model,data)
            assert json.loads(json.dumps(measured))==bc.read(case/'result.json')['final']
            with torch.no_grad():
                gate,proposal=model.distributions(data.x)
                g,p=gate.logits.argmax(-1),proposal.logits.argmax(-1)
                scores=gate.probs[:,1]
            predictions[name].append(dict(g=g,p=p,scores=scores))
            for group,mask in bc.context_masks(data).items():
                counts=ledger(g[mask],p[mask],data.gate[mask],data.proposal[mask])
                assert counts['correct_WAIT']+counts['correct_RELEASE']==round(measured['groups'][group]['joint_accuracy']*int(mask.sum()))
                conditional=Counter(); requested=Counter()
                for i in torch.where(mask & (data.gate==1))[0].tolist():
                    status=prefix_status(labels[data.ids[i]],int(p[i]))
                    conditional[status]+=1
                    if g[i]==1:requested[status]+=1
                result['groups'].setdefault(group,[]).append(dict(model=name,seed=sid,ledger=counts,
                    conditional_proposal_vs_teacher_prefix=dict(conditional),
                    predicted_RELEASE_vs_teacher_prefix=dict(requested)))
            assert digest==hashlib.sha256(path.read_bytes()).hexdigest()
            result['checkpoints'].append(dict(model=name,seed=sid,sha256=digest,final_metrics_equal=True,unchanged=True))
    m2i=bc.context_masks(data)['m2i']
    pred=predictions['separate']
    candidates={
        'persistent_m2i_false_RELEASE':m2i & (data.gate==0) & torch.stack([r['g']==1 for r in pred]).all(0),
        'persistent_m2i_false_WAIT':m2i & (data.gate==1) & torch.stack([r['g']==0 for r in pred]).all(0),
        'persistent_m2i_proposal_error':m2i & (data.gate==1) & torch.stack([r['p']!=data.proposal for r in pred]).all(0),
    }
    result['persistent_m2i_counts']={k:int(v.sum()) for k,v in candidates.items()}
    for category,mask in candidates.items():
        # First two in frozen input order, not highest confidence or most dramatic.
        for i in torch.where(mask)[0].tolist()[:2]:
            did=data.ids[i]; label=labels[did]; row=raw[did]
            result['examples'].append(dict(category=category,decision_id=did,
                state={k:row[k] for k in ('seed','tick','uid','interface','soc','speed','busy_lz','retry_active')},
                teacher_gate=label['teacher_gate'],teacher_proposal=label['teacher_proposal'],
                predictions={name:[dict(seed=sid,gate=int(r['g'][i]),proposal=int(r['p'][i]),
                    RELEASE_probability=float(r['scores'][i])) for sid,r in zip(bc.MODEL_SEEDS,rs)]
                    for name,rs in predictions.items()},
                audit=[{k:r[k] for k in ('rank','joint_index','feasible','reason')} for r in label['teacher_audit']]))
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(status=result['status'],checkpoints=len(result['checkpoints']),examples=len(result['examples']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        assert args.root and args.output
        run(args.root,args.output)
