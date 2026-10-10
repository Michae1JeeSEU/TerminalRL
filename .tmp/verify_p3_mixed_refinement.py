import hashlib,json,pickle,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_neutral_course_v2_20261008/student_initial/dense24_mixed/episode')
raw=(root/'failure_after.pkl').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='e8280556cb012bfde56d1b540708bb68f62e42a5af623a92664015c517b1c517'
env=pickle.loads(raw);start=time.monotonic()
shield=env.runtime_traffic
pairs=shield._joint_pairs(env)
now=env.now
verdict=shield._recursive_safe(env,{},pairs)
assert verdict.approved,verdict
assert env.now==now
print(json.dumps(canonical(dict(verdict=verdict,gate=shield.last_recursive_gate,
    active=len(env.active),pairs=len(pairs),elapsed=time.monotonic()-start,
    state_time_unchanged=env.now==now,plan_hash=env.active[31].committed_plan.trajectory_hash)),indent=2))
