"""Read only the fixed paired endpoints; no model or optimizer creation."""
from pathlib import Path
from statistics import mean
import json
root=Path('D:/7_codespace/training_outputs/terminal')
old=root/'p3_v3_bc_learnability_20261007/fit'
new=root/'p3_v3_bc_proposal_detach_20261007/fit'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
a,b=read(old/'result.json'),read(new/'result.json')
score=read(root/'p3_v3_bc_gate_balance_20261007/score_probe.json')
assert score['status']=='SCORE_PROBE_COMPLETE'
score_by_seed={r['seed']:r for r in score['rows']}
assert a['status']==b['status']=='FIT_COMPLETE'
assert read(old/'inputs.json')==read(new/'inputs.json')
protocol=read(new/'protocol.json')
assert protocol['protocol']=='p3_v3_90d_bc_proposal_detach_20261007'
assert protocol['detach_proposal_encoder'] and protocol['gate_weights'] is None
assert protocol['model_seeds']==[91701,91702,91703] and protocol['updates']==1000
assert protocol['environment_resets']==protocol['teacher_queries']==0 and not protocol['audit_Q_accessed']
assert a['counts']==b['counts']==dict(rows=1046,WAIT=817,RELEASE=229)

def extract(group,auc):
    g=group['gate']; c=g['confusion']; nw,nr=map(sum,c)
    assert nw+nr==group['rows'] and g['WAIT_recall']==c[0][0]/nw and g['RELEASE_recall']==c[1][1]/nr
    return dict(rows=nw+nr,WAIT=nw,RELEASE=nr,confusion=c,accuracy=g['accuracy'],
        balanced=g['balanced_accuracy'],R_recall=g['RELEASE_recall'],W_recall=g['WAIT_recall'],
        false_RELEASE=c[0][1],false_RELEASE_rate=c[0][1]/nw,
        proposal_top1=group['proposal']['top1'],proposal_top3=group['proposal']['top3'],
        joint_accuracy=group['joint_accuracy'],gate_auc=auc)

groups,rows={},[]
assert [r['seed'] for r in b['seeds']]==[91701,91702,91703]
for before,after in zip(a['seeds'],b['seeds']):
    sid=after['seed']; assert sid==before['seed']
    assert after==read(new/f'seed_{sid}/result.json')
    assert after['reference_initial_equal'] and before['initial']==after['initial']
    assert after['updates']==1000 and after['elapsed_seconds']<=300
    assert after['detach_proposal_encoder'] and after['gate_weights'] is None
    assert after['critic_unchanged'] and all(after['actor_module_changed'].values())
    assert not any(after[k] for k in ('I0_QUALIFIED','student_qualified','PPO_ALLOWED'))
    assert (new/f'seed_{sid}/final_diagnostic.pt').is_file()
    row=dict(seed=sid,seconds=after['elapsed_seconds'],loss_before=before['final']['loss'],
        loss_after=after['final']['loss'],gate_loss_before=before['final']['gate_loss'],
        gate_loss_after=after['final']['gate_loss'],proposal_loss_before=before['final']['proposal_loss'],
        proposal_loss_after=after['final']['proposal_loss'])
    for name,values in after['final']['groups'].items():
        old_auc=score_by_seed[sid]['unweighted']['groups'].get(name,{}).get('auc')
        old_metrics=extract(before['final']['groups'][name],old_auc)
        new_metrics=extract(values,after['final_gate_auc'][name])
        assert all(old_metrics[k]==new_metrics[k] for k in ('rows','WAIT','RELEASE'))
        comparison=dict(seed=sid,before=old_metrics,after=new_metrics)
        groups.setdefault(name,[]).append(comparison)
        if name=='overall':row.update(comparison)
    rows.append(row)
keys=('accuracy','balanced','R_recall','W_recall','false_RELEASE','false_RELEASE_rate',
    'proposal_top1','proposal_top3','joint_accuracy','gate_auc')
averages={group:{side:{k:mean(r[side][k] for r in rs) if all(r[side][k] is not None for r in rs) else None
    for k in keys} for side in ('before','after')} for group,rs in groups.items()}
report=dict(status='PAIRED_DETACH_ANALYSIS_COMPLETE',rows=rows,groups=groups,averages=averages)
out=Path('.tmp/p3_proposal_detach_analysis_20261007.json'); assert not out.exists()
out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps(dict(status=report['status'],rows=rows,averages=averages),indent=2,allow_nan=False))
