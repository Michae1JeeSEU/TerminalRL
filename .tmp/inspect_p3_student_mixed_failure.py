"""Read the two hash-verified snapshots from the failed new student episode."""
import hashlib,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_neutral_course_v2_20261008/student_initial/dense24_mixed/episode')
hashes={'failure_before.pkl':'f703f5ae408010c4fe62f74f5c2e70ad7657ebd974f4787f2c3c0fba0c504b04',
        'failure_after.pkl':'e8280556cb012bfde56d1b540708bb68f62e42a5af623a92664015c517b1c517'}
result={}
for name,digest in hashes.items():
    raw=(root/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
    env=pickle.loads(raw)
    rows=[]
    for uid,u in sorted(env.active.items()):
        plan=u.committed_plan
        rows.append(dict(uid=uid,phase=u.phase,ring=u.ring_role,position=u.position,velocity=u.velocity,
            speed=u.speed,soc=u.soc,ring_progress=u.ring_progress,
            plan=None if plan is None else dict(id=plan.plan_id,start=plan.start_time,end=plan.end_time,
            metadata=plan.metadata,first=plan.points[0],last=plan.points[-1])))
    result[name]=dict(now=env.now,diagnostic=env.runtime_traffic.last_certification_diagnostic,
        gate=env.runtime_traffic.last_recursive_gate,runtime_telemetry=env.runtime_traffic.telemetry,
        active=rows,release_telemetry=env.last_release_telemetry)
print(json.dumps(canonical(result),indent=2,ensure_ascii=False))
