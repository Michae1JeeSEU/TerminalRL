"""Read the three fixed endpoints only; no model, optimizer or environment."""
from pathlib import Path
from statistics import mean
import json

root=Path('D:/7_codespace/training_outputs/terminal')
paths={
    'original':root/'p3_v3_bc_learnability_20261007/fit',
    'detach':root/'p3_v3_bc_proposal_detach_20261007/fit',
    'separate':root/'p3_v3_bc_separate_encoders_20261007/fit',
}
read=lambda p:json.loads(p.read_text(encoding='utf8'))
results={name:read(path/'result.json') for name,path in paths.items()}
inputs={name:read(path/'inputs.json') for name,path in paths.items()}
assert inputs['original']==inputs['detach']==inputs['separate']
assert all(v['status']=='FIT_COMPLETE' for v in results.values())
protocol=read(paths['separate']/'protocol.json')
assert protocol['protocol']=='p3_v3_90d_bc_separate_encoders_20261007'
assert protocol['model']=='SeparateEncodersDiagnosticV42' and protocol['separate_encoders']
assert not protocol['detach_proposal_encoder'] and protocol['gate_weights'] is None
assert protocol['model_seeds']==[91701,91702,91703] and protocol['updates']==1000
assert protocol['environment_resets']==protocol['teacher_queries']==0 and not protocol['audit_Q_accessed']
score=read(root/'p3_v3_bc_gate_balance_20261007/score_probe.json')
assert score['status']=='SCORE_PROBE_COMPLETE'
old_scores={r['seed']:r['unweighted']['groups'] for r in score['rows']}
by_seed={name:{r['seed']:r for r in value['seeds']} for name,value in results.items()}
assert all(list(value)==[91701,91702,91703] for value in by_seed.values())

def extract(group,auc):
    gate=group['gate']; confusion=gate['confusion']; nw,nr=map(sum,confusion)
    assert nw+nr==group['rows']
    assert gate['WAIT_recall']==confusion[0][0]/nw and gate['RELEASE_recall']==confusion[1][1]/nr
    return dict(rows=nw+nr,WAIT=nw,RELEASE=nr,confusion=confusion,accuracy=gate['accuracy'],
        balanced=gate['balanced_accuracy'],R_recall=gate['RELEASE_recall'],W_recall=gate['WAIT_recall'],
        false_RELEASE_rate=confusion[0][1]/nw,proposal_top1=group['proposal']['top1'],
        proposal_top3=group['proposal']['top3'],joint_accuracy=group['joint_accuracy'],gate_auc=auc)

groups,rows={},[]
for sid in (91701,91702,91703):
    new=by_seed['separate'][sid]
    assert new==read(paths['separate']/f'seed_{sid}/result.json')
    assert new['initial']==by_seed['original'][sid]['initial']==by_seed['detach'][sid]['initial']
    assert new['reference_initial_equal'] and new['updates']==1000 and new['elapsed_seconds']<=300
    assert new['critic_unchanged'] and new['separate_encoders'] and not new['detach_proposal_encoder']
    assert new['gate_weights'] is None
    assert set(new['actor_module_changed'])=={'actor_encoder','gate_head','proposal_encoder','proposal_head'}
    assert all(new['actor_module_changed'].values())
    assert new['parameter_counts']==dict(actor=57339,added_proposal_encoder=27360,reference_actor=29979,critic=55297)
    assert not any(new[k] for k in ('I0_QUALIFIED','student_qualified','PPO_ALLOWED'))
    assert (paths['separate']/f'seed_{sid}/final_diagnostic.pt').is_file()
    row=dict(seed=sid,seconds=new['elapsed_seconds'],losses={},proposal_predictions={})
    for name in paths:
        value=by_seed[name][sid]
        row['losses'][name]={k:value['final'][k] for k in ('loss','gate_loss','proposal_loss')}
        row['proposal_predictions'][name]=value['final']['groups']['overall']['proposal']['prediction_counts']
    for group in new['final']['groups']:
        comparison=dict(seed=sid)
        for name in paths:
            value=by_seed[name][sid]
            auc=(old_scores[sid].get(group,{}).get('auc') if name=='original'
                 else value['final_gate_auc'][group])
            comparison[name]=extract(value['final']['groups'][group],auc)
        assert len({(comparison[name]['rows'],comparison[name]['WAIT'],comparison[name]['RELEASE']) for name in paths})==1
        groups.setdefault(group,[]).append(comparison)
        if group=='overall':row.update(comparison)
    rows.append(row)
keys=('accuracy','balanced','R_recall','W_recall','false_RELEASE_rate','proposal_top1',
      'proposal_top3','joint_accuracy','gate_auc')
averages={group:{name:{k:mean(r[name][k] for r in rs) if all(r[name][k] is not None for r in rs) else None
    for k in keys} for name in paths} for group,rs in groups.items()}
report=dict(status='SEPARATE_ENCODERS_ANALYSIS_COMPLETE',rows=rows,groups=groups,averages=averages)
out=Path('.tmp/p3_separate_encoders_analysis_20261007.json'); assert not out.exists()
out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps(dict(status=report['status'],rows=rows,averages=averages),indent=2,allow_nan=False))
