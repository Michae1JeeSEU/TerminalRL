"""Profile only the first recorded candidate; never repeat a full query."""
import cProfile
import json
from pathlib import Path
import pickle
import pstats
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.core.merge_scheduler import LowLevelIntent,RepairCandidate
from terminal_marl.core.config import RingRole
from terminal_marl.core.safety_trace import _digest
from terminal_marl.envs.parallel_env import WorldSnapshot
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical,sha256,state_record,write_json

source=Path('D:/7_codespace/training_outputs/terminal/p3_v25_neutral_course_20261008/train_dense24_fixed/seed_893502')
output=Path('.tmp/p3_v25_dense_cost_candidate');output.mkdir(exist_ok=False)
assert sha256(source/'selected.json')=='76f505d230601fc3037282aed841021f1caf421a8f0ae4e6085e4cb033531130'
entry=json.loads((source/'selected.json').read_text())[129]
row=entry['row'];assert row['tick']==218 and row['uid']=='49'
path=source/'snapshots'/entry['snapshot'];assert sha256(path)==entry['snapshot_sha256']
label_path=source/'labels/window_129.json';label_hash=sha256(label_path)
label=json.loads(label_path.read_text());expected=label['teacher_audit'][0]
assert label['status']=='OK' and expected['joint_index']==13 and not expected['feasible']
data=path.read_bytes();env=pickle.loads(data)
assert _digest(canonical(state_record(env)))==row['pre_state_hash']
uav=env.active[49];assert uav.ring_role is RingRole.OUTER
candidate=RepairCandidate('v4_1_joint_teacher',LowLevelIntent(1.,0.,0.),0.)
profiler=cProfile.Profile();started=time.monotonic()
proposal,plan,reason=profiler.runcall(env._candidate_plan,uav,candidate,RingRole.MIDDLE,
    WorldSnapshot(env.now,env.active,env.airspace),None)
seconds=time.monotonic()-started
profiler.dump_stats(str(output/'candidate.prof'))
top=[]
for (file,line,name),(primitive,calls,own,cumulative,_) in sorted(pstats.Stats(profiler).stats.items(),key=lambda item:item[1][3],reverse=True)[:50]:
    top.append(dict(file=file,line=line,function=name,calls=calls,own_seconds=own,cumulative_seconds=cumulative))
result=dict(status='PROFILE_OK',seconds=seconds,slot=129,tick=218,uid=49,joint_index=13,
    reason=reason,expected_reason=expected['reason'],profile_top=top,reset_count=0,env_steps=0,
    candidate_calls=1,teacher_queries=0,source_snapshot_sha256=entry['snapshot_sha256'])
write_json(output/'result.json',result)
assert proposal is None or plan is None or reason is not None
assert str(reason or 'candidate_rejected')==expected['reason']
assert sha256(path)==entry['snapshot_sha256'] and sha256(label_path)==label_hash
print(json.dumps({k:v for k,v in result.items() if k!='profile_top'}))
