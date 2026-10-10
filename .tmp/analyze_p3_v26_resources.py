"""Reuse complete immutable row/label checks, then verify added resource rows."""
from pathlib import Path
import sys
supplemental='--supplemental' in sys.argv
if supplemental:sys.argv.remove('--supplemental')
body=Path('.tmp/analyze_p3_v25_course_data.py').read_text(encoding='utf8')
body=body.replace('from terminal_marl.training import p3_v25_course_data as new_data',
'''from types import SimpleNamespace
from terminal_marl.training import p3_v25_course_data as old_data
from terminal_marl.training import p3_v26_course_data as resource_data
new_data=SimpleNamespace(**{k:v for k,v in vars(old_data).items() if not k.startswith('__')})
new_data.PROTOCOL=resource_data.PROTOCOL
new_data.TRAIN_CASES=resource_data.TRAIN_CASES
new_data.SMOKE_SEED=resource_data.SMOKE_SEED''')
body=body.replace("'train_dense24_mixed','train_dense24_fixed_annotation_v2'", "'train_dense24_mixed'")
body=body.replace("repaired=args.case=='train_dense24_fixed_annotation_v2'",'repaired=False')
body=body.replace('p3_v25_neutral_course_20261008',
                  'p3_v26_supplemental_20261009' if supplemental else 'p3_v26_resources_20261009')
if supplemental:
    body=body.replace('new_data.PROTOCOL=resource_data.PROTOCOL',
        'from terminal_marl.training import p3_v26_supplemental_data as supplement\nnew_data.PROTOCOL=supplement.PROTOCOL')
    body=body.replace('new_data.TRAIN_CASES=resource_data.TRAIN_CASES','new_data.TRAIN_CASES=supplement.TRAIN_CASES')
    body=body.replace('new_data.SMOKE_SEED=resource_data.SMOKE_SEED','new_data.SMOKE_SEED=supplement.SMOKE_SEED')
    body=body.replace('train_flow6_mixed','train_medium12_mixed')
    body=body.replace("'dense24_mixed') if smoke", "'medium12_mixed') if smoke")
body=body.replace('/args.case',"/('smoke' if smoke else f'train_{index:02d}')")
body=body.replace("180 if smoke else 21600 if repaired or args.case=='train_dense24_mixed' else 7200",'180 if smoke else 21600')
checks='''
resource_path=episode/'resource_context_observations.jsonl'
assert protocol['learning_observation_schema']==resource_data.VERSION
assert hashlib.sha256(resource_path.read_bytes()).hexdigest()==result['resource_reconstruction']['sha256']
assert result['resource_reconstruction']['original_context_sha256']==result['reconstruction']['sha256']
resource_rows=[json.loads(s) for s in resource_path.read_text().splitlines()]
assert len(resource_rows)==len(context_rows)==result['resource_reconstruction']['rows']
for original,value in zip(context_rows,resource_rows):
    assert value['observation_schema']==resource_data.VERSION
    assert value['ego_feature_hash']==resource_data.EGO_SCHEMA.feature_hash
    assert len(value['ego'])==159 and all(math.isfinite(v) for v in value['ego'])
    assert value['ego'][:105]==original['ego']
    assert {k:v for k,v in value.items() if k not in {'ego','observation_schema','ego_feature_hash'}}=={k:v for k,v in original.items() if k not in {'ego','observation_schema','ego_feature_hash'}}
    if not smoke:resource_data.decode_resource_row(value)
summary.update(resource_rows=len(resource_rows),resource_ego_dim=159,original_prefix_labels_traffic_exact=True,
    resource_sha256=result['resource_reconstruction']['sha256'],root_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest())
'''
body=body.replace('output.write_text(json.dumps(summary',checks+'\noutput.write_text(json.dumps(summary')
exec(compile(body,__file__,'exec'))
