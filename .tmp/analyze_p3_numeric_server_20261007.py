from pathlib import Path
import hashlib
import json

root = Path('D:/7_codespace/training_outputs/terminal/p3_numeric_bounds_validation_20261007')
def read(path):
    return json.loads(path.read_text())
report = read(root/'full/result.json')
assert report['status'] == 'COMPARISON_OK' and len(report['cases']) == 8
assert hashlib.sha256((root/'full/result.json').read_bytes()).hexdigest() == '32faf7b9a9f56dcf72b5da93fb7b354348395f7f188b58f257db6c9e46bc4900'
original = Path('D:/7_codespace/training_outputs/terminal/p3_v3_i0_v2_20261004')
pairs = [('mixed', 0), ('mixed', 19), ('mixed', 299), ('high', 177)]
total = dict(baseline=0., candidate=0.)
intents, plans, steps = 0, 0, 0
rows = []
for index, (scene, slot) in enumerate(pairs):
    parent = root/('smoke' if index == 0 else 'full')
    a = parent/f'{index}_{scene}_{slot}_baseline'
    b = parent/f'{index}_{scene}_{slot}_candidate'
    qa, qb = read(a/'query.json'), read(b/'query.json')
    assert qa == qb
    episode = original/('dense_mixed/seed_893102' if scene == 'mixed' else 'dense_high_launch2/seed_893101')
    label = read(episode/'labels'/f'window_{slot:02d}.json')
    assert all(label[key] == value for key, value in qa.items())
    for mode, directory in [('baseline', a), ('candidate', b)]:
        item = read(directory/'result.json')
        assert read(directory/'exit.json') == {'exit_code': 0}
        assert item['status'] == 'QUERY_OK' and item['source_files_unchanged']
        total[mode] += item['seconds']
        steps += item['env_steps']
        intents += qa['release_intents_checked']
        plans += qa['candidate_plan_calls']
    if index in (2, 3):
        assert read(a/'step.json') == read(b/'step.json')
    ta, tb = read(a/'result.json')['seconds'], read(b/'result.json')['seconds']
    rows.append(dict(scene=scene, slot=slot, teacher_gate=qa['teacher_gate'],
                     proposal=qa['teacher_proposal'], baseline_seconds=ta, candidate_seconds=tb,
                     speedup=ta/tb, reduction_fraction=1-tb/ta,
                     intents_each=qa['release_intents_checked'], plans_each=qa['candidate_plan_calls']))
assert steps == 4 and intents <= 200
out = dict(status='ANALYSIS_OK', rows=rows, seconds=total, speedup=total['baseline']/total['candidate'],
           reduction_fraction=1-total['candidate']/total['baseline'], teacher_queries=8,
           release_intents=intents, candidate_plan_calls=plans, env_steps=steps, reset_count=0,
           full_outputs_exact=True, prior_local_failed_queries=1, trained=False)
Path('.tmp/p3_numeric_server_analysis_20261007.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
