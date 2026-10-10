from pathlib import Path
import json
import math
from statistics import mean

root = Path('D:/7_codespace/training_outputs/terminal/p3_v3_bc_learnability_20261007/fit')
read = lambda p: json.loads(p.read_text(encoding='utf8'))
result, protocol, inputs = (read(root/f'{name}.json') for name in ('result','protocol','inputs'))
assert result['status'] == 'FIT_COMPLETE' and result['mode'] == 'fit'
assert protocol['model_seeds'] == [91701,91702,91703] and protocol['updates'] == 1000
assert protocol['batch_size'] == 128 and protocol['learning_rate'] == .0003
assert protocol['environment_resets'] == protocol['teacher_queries'] == 0
assert not protocol['audit_Q_accessed'] and not protocol['checkpoint_selection']
assert inputs['counts'] == result['counts'] == {'rows':1046,'WAIT':817,'RELEASE':229}
assert len(inputs['decision_ids']) == len(set(inputs['decision_ids'])) == 1046
assert [x['seed'] for x in inputs['episodes']] == [893001,893101,893201]
assert [x['seed'] for x in result['seeds']] == [91701,91702,91703]
rows, groups = [], {}
for seed in result['seeds']:
    sid = seed['seed']
    assert seed == read(root/f'seed_{sid}/result.json')
    assert seed['updates'] == 1000 and seed['elapsed_seconds'] <= 300
    assert seed['critic_unchanged'] and seed['actor_changed']
    assert not seed['student_qualified'] and not seed['I0_QUALIFIED'] and not seed['PPO_ALLOWED']
    assert (root/f'seed_{sid}/final_diagnostic.pt').is_file()
    row = dict(seed=sid, seconds=seed['elapsed_seconds'])
    for endpoint in ('initial','final'):
        m = seed[endpoint]; overall = m['groups']['overall']
        c = overall['gate']['confusion']
        assert sum(c[0]) == 817 and sum(c[1]) == 229
        assert overall['rows'] == 1046 and overall['proposal']['rows'] == 229
        assert all(math.isfinite(m[k]) for k in ('loss','gate_loss','proposal_loss'))
        row[endpoint] = dict(loss=m['loss'], gate_loss=m['gate_loss'], proposal_loss=m['proposal_loss'],
            accuracy=overall['gate']['accuracy'], balanced=overall['gate']['balanced_accuracy'],
            wait_recall=overall['gate']['WAIT_recall'], release_recall=overall['gate']['RELEASE_recall'],
            predicted_release=overall['gate']['prediction_RELEASE'],
            proposal_top1=overall['proposal']['top1'], proposal_top3=overall['proposal']['top3'],
            joint_accuracy=overall['joint_accuracy'])
    rows.append(row)
    for group, metrics in seed['final']['groups'].items():
        groups.setdefault(group,[]).append(dict(seed=sid,**metrics))
summary = dict(status='PRIMARY_ANALYSIS_OK', rows=rows, baselines=inputs['baselines'],
    final_mean_range={k:dict(mean=mean(r['final'][k] for r in rows),
        minimum=min(r['final'][k] for r in rows),maximum=max(r['final'][k] for r in rows))
        for k in rows[0]['final']}, groups=groups)
out = Path('.tmp/p3_v3_bc_analysis_20261007.json')
out.write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k!='groups'},indent=2))
