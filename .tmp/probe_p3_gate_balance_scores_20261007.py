"""Read-only ranking/analytical-threshold comparison of six diagnostic endpoints."""
from pathlib import Path
import json
import hashlib
import torch
from terminal_marl.training import p3_v3_bc_learnability as bc

torch.set_num_threads(1)
def auc(scores, labels):
    pos,neg=scores[labels==1],scores[labels==0]
    if not len(pos) or not len(neg):
        return None
    delta=pos[:,None]-neg[None,:]
    return float(((delta>0).double()+.5*(delta==0).double()).mean())
assert auc(torch.tensor([0.,1.]),torch.tensor([0,1])) == 1.
assert auc(torch.tensor([.5,.5]),torch.tensor([0,1])) == .5

root=Path('/home/user/training_outputs/terminal')
data=bc.load_inputs(root)
cutoff=229/1046  # w_WAIT/(w_WAIT+w_RELEASE), fixed from dataset counts, no sweep.
cases={'unweighted':root/bc.REFERENCE_CASE,
       'weighted':root/'p3_v3_bc_gate_balance_20261007/fit_serialization_fix'}
masks={'overall':torch.ones(len(data.gate),dtype=torch.bool)}
for key in ('o2m','m2i','i2f'):
    masks[key]=torch.tensor([c['interface']==key for c in data.contexts])
for key in ('low_soc','fast_inner','busy_lz','after_retry'):
    masks[key]=torch.tensor([c[key] for c in data.contexts])
rows=[]
for seed in bc.MODEL_SEEDS:
    record={'seed':seed}
    for arm,case in cases.items():
        path=case/f'seed_{seed}/final_diagnostic.pt'
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        cp=torch.load(path,map_location='cpu',weights_only=True)
        assert cp['model_seed']==seed and cp['observation_dim']==90
        assert cp['diagnostic_only'] and not cp['qualified']
        model=bc.HierarchicalActorCentralCriticV42(90)
        model.load_state_dict(cp['model_state'],strict=True); model.eval()
        measured=bc.measure(model,data)
        assert json.loads(json.dumps(measured)) == bc.read(case/f'seed_{seed}/result.json')['final']
        with torch.no_grad():
            g,p=model.distributions(data.x)
            scores=g.logits[:,1]-g.logits[:,0]
            groups={key:{'auc':auc(scores[mask],data.gate[mask])} for key,mask in masks.items()}
            if arm=='unweighted':
                shifted=torch.stack((torch.zeros(len(data.gate)),
                    torch.where(g.probs[:,1]>cutoff,1.,-1.)),dim=1)
                for key,mask in masks.items():
                    groups[key]['analytical_threshold']=bc.classification(shifted[mask],p.logits[mask],
                        data.gate[mask],data.proposal[mask])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        record[arm]={'checkpoint_sha':sha,'saved_final_exact':True,'groups':groups}
    rows.append(record)
result=dict(status='SCORE_PROBE_COMPLETE',threshold=cutoff,threshold_source='229/1046; no search',
    scope='in_sample_only',optimizer_created=False,parameter_updates=0,environment_resets=0,
    teacher_queries=0,checkpoint_written=False,synthetic_checks=True,rows=rows)
print(json.dumps(result,indent=2,allow_nan=False))
