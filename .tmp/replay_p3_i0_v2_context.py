"""Read-only-to-evidence, one original taped step; no reset, query or labels."""
import hashlib
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terminal_marl.core.safety_trace import _digest
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical, execute_tape, state_record

slot = int(sys.argv[1])
assert slot in (101, 432, 576)
out = Path(sys.argv[2])
assert not out.exists(), out
episode = Path(sys.argv[3])
entry = json.loads((episode/'selected.json').read_text())[slot]
row = entry['row']
data = (episode/'snapshots'/entry['snapshot']).read_bytes()
assert hashlib.sha256(data).hexdigest() == entry['snapshot_sha256']
env = pickle.loads(data)
assert _digest(canonical(state_record(env))) == row['pre_state_hash']
step = [json.loads(line) for line in (episode/'steps.jsonl').read_text().splitlines()][row['tick']]
assert step['pre_state_hash'] == row['pre_state_hash']

def context():
    uav = env.active[int(row['uid'])]
    return dict(now=env.now, available=env.lz_manager.rank_available_zones(uav, env.now),
        zones=[dict(id=z.lz_id, status=z.status.value, service=z.service_remaining,
                    reserved_by=z.reserved_by, occupied_by=z.occupied_by) for z in env.lz_manager.zones],
        active={str(uid): dict(phase=u.phase.value, lz=u.assigned_lz,
                plan=None if u.committed_plan is None else u.committed_plan.trajectory_hash)
                for uid,u in env.active.items()})

before = context()
original = env._release
history = []
at_release = []
def record_release(uav, intent, shortlist=None):
    if uav.uid == int(row['uid']):
        at_release.append(dict(context=context(), earlier_releases=list(history)))
    decision = original(uav, intent, shortlist)
    history.append(dict(uid=uav.uid, decision=decision))
    return decision
env._release = record_release
live = execute_tape(env, step['tape'])
assert len(at_release) == 1
assert _digest(live['after']) == step['post_state_hash']
assert live['outputs'][4]['action_traces'] == step['action_traces']
assert canonical(env.last_release_telemetry) == step['release_telemetry']
result = dict(status='EXACT_REPLAY_OK', slot=slot, tick=row['tick'], uid=row['uid'],
    source_snapshot_sha256=entry['snapshot_sha256'], before=before, at_release=at_release[0],
    telemetry=step['release_telemetry'][row['uid']], reset_count=0, steps=1,
    teacher_queries=0, partition='diagnostic_never_train', original_artifacts_changed=False)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(dict(status=result['status'],slot=slot,tick=row['tick'],path=str(out))))
