"""Native-platform integrated optimization verification; reads trusted existing snapshots only."""
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.core.trajectory_generator import TrajectoryGenerator
from terminal_marl.core.trajectory_time_parameterization import TrajectoryTimeParameterizer
from terminal_marl.scripts.collect_p3_v3_sparse_expert import query_snapshot
from terminal_marl.scripts.diagnose_p3_direct_contract import sha256,write_json

assert sys.platform=='linux'
slot=int(sys.argv[1]);assert slot in (0,129)
root=Path('/home/user/training_outputs/terminal/p3_v25_neutral_course_20261008/train_dense24_fixed/seed_893502')
original=json.loads((root.parent/'result.json').read_text())
assert original['status'] in ('COLLECTION_OK','INCOMPLETE','ERROR'), 'original annotation must be terminal'
output=Path('/home/user/training_outputs/terminal/p3_v25_hull_production_query_20261008')/f'query_{slot}'
output.mkdir(parents=True,exist_ok=False)
assert sha256(root/'selected.json')=='76f505d230601fc3037282aed841021f1caf421a8f0ae4e6085e4cb033531130'
entry_path=root/'snapshots'/f'window_{slot:02d}.json';entry=json.loads(entry_path.read_text())
label_path=root/'labels'/f'window_{slot:02d}.json';label_sha=sha256(label_path)
label=json.loads(label_path.read_text());assert label['status']=='OK'
assert sha256(root/'snapshots'/entry['snapshot'])==entry['snapshot_sha256']
start=time.perf_counter();result=query_snapshot(entry_path);seconds=time.perf_counter()-start
write_json(output/'query.json',result)
mismatches=[key for key,value in result.items() if key not in label or value!=label[key]]
report=dict(status='MISMATCH' if mismatches else 'QUERY_EXACT',slot=slot,seconds=seconds,
    teacher_queries=1,release_intents_checked=result['release_intents_checked'],
    candidate_plan_calls=result['candidate_plan_calls'],teacher_gate=result['teacher_gate'],
    teacher_proposal=result['teacher_proposal'],mismatches=mismatches,reset_count=0,env_steps=0)
write_json(output/'result.json',report)
assert sha256(root/'snapshots'/entry['snapshot'])==entry['snapshot_sha256'] and sha256(label_path)==label_sha
print(json.dumps(report),flush=True)
assert not mismatches, mismatches
