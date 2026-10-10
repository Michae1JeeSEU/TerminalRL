"""Inactive candidate replay of existing complete expert labels only."""
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

spec=importlib.util.spec_from_file_location('hull_candidate',Path(__file__).with_name('p3_ring_hold_hull_candidate.py'))
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
TrajectoryGenerator._v3_ring_hold_points=candidate.ring_hold_points
TrajectoryTimeParameterizer._v3_segments_feasible=candidate.segments_feasible
slot=int(sys.argv[1]);assert slot in (0,129)
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_neutral_course_20261008/train_dense24_fixed/seed_893502')
output=Path(f'.tmp/p3_v25_hull_query_{slot}');output.mkdir(exist_ok=False)
assert sha256(root/'selected.json')=='76f505d230601fc3037282aed841021f1caf421a8f0ae4e6085e4cb033531130'
entry_path=root/'snapshots'/f'window_{slot:02d}.json';entry=json.loads(entry_path.read_text())
label_path=root/'labels'/f'window_{slot:02d}.json';label_sha=sha256(label_path)
label=json.loads(label_path.read_text());assert label['status']=='OK'
start=time.perf_counter();result=query_snapshot(entry_path);seconds=time.perf_counter()-start
write_json(output/'query.json',result)
assert all(key in label and value==label[key] for key,value in result.items())
assert sha256(root/'snapshots'/entry['snapshot'])==entry['snapshot_sha256'] and sha256(label_path)==label_sha
report=dict(status='QUERY_EXACT',slot=slot,seconds=seconds,teacher_queries=1,
    release_intents_checked=result['release_intents_checked'],candidate_plan_calls=result['candidate_plan_calls'],
    teacher_gate=result['teacher_gate'],teacher_proposal=result['teacher_proposal'],
    original_certificate_and_audit_exact=True,reset_count=0,env_steps=0)
write_json(output/'result.json',report);print(json.dumps(report))
