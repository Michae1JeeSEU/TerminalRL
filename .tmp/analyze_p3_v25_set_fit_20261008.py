from pathlib import Path
import json
import statistics
from collections import Counter

root = Path('D:/7_codespace/training_outputs/terminal')
out = root/'p3_v25_set_fit_20261008/fit'
result = json.loads((out/'result.json').read_text())
assert result['status'] == 'FIT_COMPLETE' and result['parameter_updates'] == 6000
rows = [json.loads(s) for s in (root/'p3_v3_observation_v25_20261008/full/observations.jsonl').read_text().splitlines()]
targets = {r['decision_id']: (r['teacher_gate'], r['teacher_proposal']) for r in rows}
fits = {(f['seed'], f['arm']): f for f in result['fits']}
names = ['overall','o2m','m2i','i2f','low_soc','fast_inner','busy_lz','after_retry',
         'episode_893001','episode_893101','episode_893201']
def compact(f, name):
    g = f['final']['groups'][name]
    return dict(n=g['rows'], R=g['proposal']['rows'] if g['proposal'] else 0,
        auc=f['final']['auc'][name], balanced=g['gate']['balanced_accuracy'],
        recall_R=g['gate']['RELEASE_recall'], false_R=1-g['gate']['WAIT_recall'] if g['gate']['WAIT_recall'] is not None else None,
        proposal1=g['proposal']['top1'] if g['proposal'] else None,
        proposal3=g['proposal']['top3'] if g['proposal'] else None, joint=g['joint_accuracy'])

all_predictions={}
for seed in (91701,91702,91703):
    a,b=[fits[seed,arm] for arm in ('old_information','new_information')]
    assert a['initial_state_hash']==b['initial_state_hash'] and a['batch_order_hash']==b['batch_order_hash']
    for f in (a,b):
        assert f['status']=='FIT_COMPLETE' and f['updates']==1000 and all(f['changed'].values())
        pred=json.loads((out/f"{seed}_{f['arm']}"/'predictions.json').read_text())
        assert len(pred)==len(targets)==1046 and {p['decision_id'] for p in pred}==set(targets)
        confusion=[[0,0],[0,0]]; proposal_right=top3=joint=0
        for p in pred:
            g,q=targets[p['decision_id']]
            confusion[g][p['gate']]+=1
            proposal_right += bool(g and p['proposal']==q)
            top3 += bool(g and q in p['proposal_top3'])
            joint += bool(p['gate']==g and (not g or p['proposal']==q))
        group=f['final']['groups']['overall']
        assert confusion==group['gate']['confusion']
        assert abs(proposal_right/229-group['proposal']['top1'])<1e-7
        assert abs(top3/229-group['proposal']['top3'])<1e-7
        assert abs(joint/1046-group['joint_accuracy'])<1e-7
        all_predictions[seed,f['arm']]={p['decision_id']:p for p in pred}
        print(json.dumps(dict(seed=seed,arm=f['arm'],seconds=f['elapsed_seconds'],
            initial_loss=f['initial']['loss'],final_loss=f['final']['loss'],
            confusion=confusion, proposal_counts=group['proposal']['prediction_counts'],
            **compact(f,'overall'))))

for name in names:
    summary={}
    for arm in ('old_information','new_information'):
        records=[compact(fits[seed,arm],name) for seed in (91701,91702,91703)]
        summary[arm]={k: statistics.mean(r[k] for r in records) if all(r[k] is not None for r in records) else None for k in records[0]}
    print(json.dumps(dict(group=name,**summary)))

for tick,uid in ((0,'4'),(92,'0'),(101,'0'),(152,'6')):
    did=f'p3_v3_90d_sparse_expert_v1:893001:{tick}:{uid}'
    print(json.dumps(dict(witness=did,target=targets[did],
        predictions={f'{seed}_{arm}':all_predictions[seed,arm][did] for seed,arm in all_predictions})))

# Match requests only to existing single-state teacher-prefix evidence. A
# candidate never queried by the original teacher remains unknown.
labels = {}
for seed, case in ((893001,'p3_v3_i0_20261004/train_01'),
                   (893101,'p3_v3_i0_v2_20261004/dense_high_launch2'),
                   (893201,'p3_v3_i0_v3_20261007/dense_mixed')):
    episode=root/case/f'seed_{seed}'
    for item in json.loads((episode/'annotation_result.json').read_text())['entries']:
        label=json.loads((episode/item['label_file']).read_text())
        did=label['decision_id']
        assert did not in labels and targets[did]==(label['teacher_gate'],label['teacher_proposal'])
        labels[did]=label
assert set(labels)==set(targets)
for (seed,arm),predictions in all_predictions.items():
    counts,reasons=Counter(),Counter()
    for did,p in predictions.items():
        g,q=targets[did]
        counts['FN'] += int(g and not p['gate'])
        counts['FP'] += int(not g and p['gate'])
        counts['gate_correct_wrong_proposal'] += int(g and p['gate'] and q!=p['proposal'])
        if not p['gate']:
            continue
        observed={a['joint_index']-1:a for a in labels[did]['teacher_audit'] if a['joint_index']>0}
        item=observed.get(p['proposal'])
        if item is None:
            assert g and p['proposal']!=q
            counts['requested_unqueried']+=1
        elif item['feasible']:
            assert g and p['proposal']==q
            counts['requested_selected_certified']+=1
        else:
            counts['requested_known_rejected']+=1
            counts['rejected_when_teacher_release' if g else 'rejected_when_teacher_wait']+=1
            reasons[item['reason']]+=1
    print(json.dumps(dict(prefix_crosscheck=True,seed=seed,arm=arm,counts=counts,reasons=reasons)))
