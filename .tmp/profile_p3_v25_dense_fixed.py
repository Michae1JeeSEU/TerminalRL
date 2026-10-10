"""Two local, immutable saved-state profiles; no collector restart or step."""
import json
from pathlib import Path
import pickle
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from terminal_marl.core.observations import build_low_observation_v2_4
from terminal_marl.core.safety_trace import _digest
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical, sha256, state_record, write_json
from terminal_marl.scripts.diagnose_p3_expert_cost import profile_query

slot = int(sys.argv[1])
assert slot in (0, 129)
source = Path('D:/7_codespace/training_outputs/terminal/p3_v25_neutral_course_20261008/train_dense24_fixed/seed_893502')
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=False)
report = dict(status='ERROR', slot=slot, partition='diagnostic_never_train',
              reset_count=0, env_steps=0, trained=False, error=None)
start = time.monotonic()
try:
    assert sha256(source/'selected.json') == '76f505d230601fc3037282aed841021f1caf421a8f0ae4e6085e4cb033531130'
    entry = json.loads((source/'selected.json').read_text())[slot]
    row = entry['row']
    data = source/'snapshots'/entry['snapshot']
    assert sha256(data) == entry['snapshot_sha256']
    original_label = source/'labels'/f'window_{slot:02d}.json'
    label_sha = sha256(original_label)
    label = json.loads(original_label.read_text())
    assert label['status'] == 'OK' and label['decision_id'] == row['decision_id']
    if slot == 129:
        assert row['tick'] == 218 and row['uid'] == '49'
    env = pickle.loads(data.read_bytes())
    assert _digest(canonical(state_record(env))) == row['pre_state_hash']
    assert np.array_equal(np.asarray(row['observation'], dtype=np.float32),
                          build_low_observation_v2_4(env.active[int(row['uid'])], env))
    before = pickle.dumps(env)
    report.update(tick=row['tick'], uid=row['uid'], recorded_seconds=label['elapsed_seconds'],
                  source_snapshot_sha256=entry['snapshot_sha256'])
    write_json(output/'progress.json', report)
    query = profile_query(env, row['uid'], output)
    write_json(output/'query.json', query)
    assert pickle.dumps(env) == before
    assert query['joint_index'] == label['teacher_joint_index'] and query['audit'] == label['teacher_audit']
    assert len(query['candidates']) == label['candidate_plan_calls']
    assert sha256(data) == entry['snapshot_sha256'] and sha256(original_label) == label_sha
    report.update(status='SMOKE_OK' if slot == 0 else 'PROFILE_OK', query=query,
                  recorded_label_audit_exact=True, original_state_unchanged=True,
                  source_files_unchanged=True, teacher_queries=1)
except Exception as exc:
    report['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
report['elapsed_seconds'] = time.monotonic()-start
write_json(output/'result.json', report)
print(json.dumps({k: report[k] for k in ('status','slot','error','elapsed_seconds')}))
raise SystemExit(0 if report['error'] is None else 1)
