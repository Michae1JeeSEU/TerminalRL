import hashlib,json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1

output=Path('.tmp/p3_six_train_teacher_prefix.json');assert not output.exists()
base=Path('D:/7_codespace/training_outputs/terminal')
old=base/'p3_v25_neutral_course_v2_20261008'
new=base/'p3_v26_resources_20261009'
cases=[old/name for name in ('train_flow6_mixed','train_dense24_fixed','train_dense24_mixed')]+[new/f'train_{i:02d}' for i in (1,2,3)]
order=[x.index for x in sorted(JOINT_ACTION_CATALOG_V4_1[1:],key=lambda x:(abs(x.speed_residual),abs(x.merge_time_residual),x.index))]
result={'status':'COMPLETE','Q_accessed':False,'new_queries':0,'cases':[]}
for root in cases:
 r=json.loads((root/'result.json').read_text());seed=r['seed'];assert r['status']=='COLLECTION_OK' and r['partition']=='train'
 episode=root/f'seed_{seed}';counts=Counter();reasons=Counter()
 for entry in r['annotation']['entries']:
  label=json.loads((episode/entry['label_file']).read_text());assert label['status']=='OK' and label['actor_mask']
  assert label['decision_id']==entry['decision_id']
  audit=label['teacher_audit'];releases=[x for x in audit if x['joint_index']]
  assert [x['joint_index'] for x in releases]==order[:len(releases)]
  assert len(releases)==label['release_intents_checked']
  if label['teacher_gate']:
   assert releases[-1]['joint_index']==label['teacher_joint_index']==label['teacher_proposal']+1
   assert releases[-1]['feasible'] and all(not x['feasible'] for x in releases[:-1])
   counts['RELEASE']+=1;counts['known_positive']+=1;counts['known_negative']+=len(releases)-1
   counts['unqueried']+=25-len(releases)
  else:
   assert len(releases)==25 and all(not x['feasible'] for x in releases)
   assert audit[-1]['joint_index']==0 and audit[-1]['feasible']
   counts['WAIT']+=1;counts['known_negative']+=25
  reasons.update(x['reason'] for x in releases if not x['feasible'])
 result['cases'].append({'seed':seed,'counts':dict(counts),'negative_reasons':dict(reasons)})
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));print('SHA',hashlib.sha256(output.read_bytes()).hexdigest())
