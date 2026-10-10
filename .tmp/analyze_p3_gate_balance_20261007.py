"""Primary's read-only, paired final-result extraction. No training/model loads."""
from pathlib import Path
from statistics import mean
import json
import math

root = Path('D:/7_codespace/training_outputs/terminal')
old = root/'p3_v3_bc_learnability_20261007/fit'
new = root/'p3_v3_bc_gate_balance_20261007/fit_serialization_fix'
read = lambda p: json.loads(p.read_text(encoding='utf8'))
a,b = read(old/'result.json'), read(new/'result.json')
assert a['status'] == b['status'] == 'FIT_COMPLETE'
assert read(old/'inputs.json') == read(new/'inputs.json')
p = read(new/'protocol.json')
assert p['model_seeds'] == [91701,91702,91703] and p['updates'] == 1000
assert p['gate_weights'] == [1046/(2*817),1046/(2*229)]
assert p['environment_resets'] == p['teacher_queries'] == 0 and not p['audit_Q_accessed']
assert a['counts'] == b['counts'] == dict(rows=1046,WAIT=817,RELEASE=229)

def extract(group):
    g=group['gate']; c=g['confusion']
    nw,nr=sum(c[0]),sum(c[1])
    assert nw+nr == group['rows'] and all(isinstance(v,int) and v>=0 for row in c for v in row)
    assert g['WAIT_recall'] == c[0][0]/nw and g['RELEASE_recall'] == c[1][1]/nr
    assert g['balanced_accuracy'] == (g['WAIT_recall']+g['RELEASE_recall'])/2
    return dict(rows=nw+nr,WAIT=nw,RELEASE=nr,confusion=c,accuracy=g['accuracy'],
        balanced=g['balanced_accuracy'],R_recall=g['RELEASE_recall'],W_recall=g['WAIT_recall'],
        false_RELEASE=c[0][1],false_RELEASE_rate=c[0][1]/nw,
        RELEASE_precision=c[1][1]/(c[0][1]+c[1][1]) if c[0][1]+c[1][1] else None,
        proposal_top1=group['proposal']['top1'],proposal_top3=group['proposal']['top3'],
        joint_accuracy=group['joint_accuracy'])

rows,groups=[],{}
assert [s['seed'] for s in b['seeds']] == [91701,91702,91703]
for before,after in zip(a['seeds'],b['seeds']):
    sid=after['seed']
    assert before['seed'] == sid and after == read(new/f'seed_{sid}/result.json')
    assert after['updates'] == 1000 and after['reference_initial_equal']
    assert before['initial'] == after['initial']
    assert after['critic_unchanged'] and after['actor_changed']
    assert not any(after[k] for k in ('I0_QUALIFIED','PPO_ALLOWED','student_qualified'))
    assert after['elapsed_seconds'] <= 300 and (new/f'seed_{sid}/final_diagnostic.pt').is_file()
    assert set(before['final']['groups']) == set(after['final']['groups'])
    row=dict(seed=sid,seconds=after['elapsed_seconds'],
        initial_weighted_objective=after['initial_training_objective'],
        final_weighted_objective=after['final_training_objective'],
        unweighted_loss_before=before['final']['loss'],unweighted_loss_after=after['final']['loss'])
    for group in after['final']['groups']:
        x,y=extract(before['final']['groups'][group]),extract(after['final']['groups'][group])
        assert all(x[k]==y[k] for k in ('rows','WAIT','RELEASE'))
        delta={k:y[k]-x[k] for k in ('balanced','R_recall','W_recall','false_RELEASE',
            'false_RELEASE_rate','proposal_top1','proposal_top3','joint_accuracy')}
        comparison=dict(seed=sid,before=x,after=y,delta=delta)
        groups.setdefault(group,[]).append(comparison)
        if group=='overall': row.update(comparison)
    rows.append(row)
metrics=('accuracy','balanced','R_recall','W_recall','false_RELEASE','false_RELEASE_rate',
    'RELEASE_precision','proposal_top1','proposal_top3','joint_accuracy')
averages={group:{side:{key:mean(r[side][key] for r in rs) for key in metrics}
    for side in ('before','after')} for group,rs in groups.items()}
report=dict(status='PAIRED_RESULT_CHECK_COMPLETE',rows=rows,groups=groups,averages=averages)
out=Path('.tmp/p3_gate_balance_analysis_20261007.json'); assert not out.exists()
out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps(dict(status=report['status'],rows=rows,averages=averages),indent=2,allow_nan=False))
