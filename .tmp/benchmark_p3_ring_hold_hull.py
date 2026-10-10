"""Four same-state candidate calls, unprofiled, inactive implementation only."""
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
from types import MethodType
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.core.merge_scheduler import LowLevelIntent,RepairCandidate
from terminal_marl.core.config import RingRole
from terminal_marl.envs.parallel_env import WorldSnapshot
from terminal_marl.scripts.diagnose_p3_direct_contract import sha256,write_json

spec=importlib.util.spec_from_file_location('hull_candidate',Path(__file__).with_name('p3_ring_hold_hull_candidate.py'))
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
source=Path('D:/7_codespace/training_outputs/terminal/p3_v25_neutral_course_20261008/train_dense24_fixed/seed_893502')
output=Path('.tmp/p3_v25_dense_cost_hull_combined');output.mkdir(exist_ok=False)
assert sha256(source/'selected.json')=='76f505d230601fc3037282aed841021f1caf421a8f0ae4e6085e4cb033531130'
entries=json.loads((source/'selected.json').read_text());results=[]
for slot in (0,129):
    entry=entries[slot];path=source/'snapshots'/entry['snapshot']
    assert sha256(path)==entry['snapshot_sha256'];data=path.read_bytes()
    expected=json.loads((source/'labels'/f'window_{slot:02d}.json').read_text())['teacher_audit'][0]
    assert expected['joint_index']==13
    for mode in ('baseline','candidate'):
        env=pickle.loads(data);uav=env.active[int(entry['row']['uid'])]
        assert uav.ring_role is RingRole.OUTER
        if mode=='candidate':
            env.generator._v3_ring_hold_points=MethodType(candidate.ring_hold_points,env.generator)
            env.generator.parameterizer._v3_segments_feasible=MethodType(candidate.segments_feasible,env.generator.parameterizer)
        generated=[];original=env.generator.ring_transfer
        def record(*args,**kwargs):
            plan=original(*args,**kwargs);generated.append(plan.trajectory_hash);return plan
        env.generator.ring_transfer=record
        started=time.perf_counter()
        proposal,plan,reason=env._candidate_plan(uav,RepairCandidate('v4_1_joint_teacher',LowLevelIntent(1.,0.,0.),0.),
            RingRole.MIDDLE,WorldSnapshot(env.now,env.active,env.airspace),None)
        seconds=time.perf_counter()-started
        feasible=proposal is not None and plan is not None and reason is None
        assert feasible==expected['feasible']
        assert ('certified' if feasible else str(reason or 'candidate_rejected'))==expected['reason']
        item=dict(slot=slot,mode=mode,seconds=seconds,reason=reason,feasible=feasible,generated_hashes=generated)
        results.append(item);write_json(output/'progress.json',results)
    assert results[-1]['generated_hashes']==results[-2]['generated_hashes']
    assert sha256(path)==entry['snapshot_sha256']
write_json(output/'result.json',dict(status='COMPARISON_OK',cases=results,reset_count=0,env_steps=0,candidate_calls=4,teacher_queries=0))
print(json.dumps(results))
