"""Two fixed development states; zero resets/steps/learning, no Q consumption."""
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.core.safety_trace import _digest,_json
from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_joint_index_v4_1
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical,state_record
from terminal_marl.training.p3_v25_runtime import actor_eligible
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1

base=Path('D:/7_codespace/training_outputs/terminal/p3_v25_student_dev_20261008/B_sto')
tape={r['tick']:r for r in map(json.loads,(base/'steps.jsonl').read_text().splitlines())}
out=[]
for tick,uid in ((400,114),(600,38)):
    started=time.monotonic();path=base/f'snapshot_{tick}.pkl';data=path.read_bytes();env=pickle.loads(data)
    assert _digest(canonical(state_record(env)))==tape[tick]['pre_hash']==tape[tick-1]['post_hash']
    u=env.active[uid];assert actor_eligible(env,u) and tape[tick]['actor_mask'][str(uid)]
    assert tape[tick]['tape']['actions'][str(uid)][0]==0
    certificates=[];original=env._candidate_plan
    def recorded(*args,**kwargs):
        proposal,plan,reason=original(*args,**kwargs)
        if proposal is not None and plan is not None and reason is None:
            checked=env.safety.checked_trajectory_hashes.get(plan.plan_id)
            assert checked==plan.trajectory_hash
            certificates.append(dict(mode=plan.mode,start=plan.start_time,end=plan.end_time,
                duration=plan.end_time-plan.start_time,trajectory_hash=plan.trajectory_hash,checked_hash=checked))
        return proposal,plan,reason
    env._candidate_plan=recorded
    selected,audit=teacher_joint_expert_v4_1(env,u,state_only=True)
    action=hierarchical_from_joint_index_v4_1(selected.index)
    assert bool(action.gate)==bool(certificates)
    out.append(dict(tick=tick,uid=uid,arrival_id=u.arrival_id,phase=u.phase.value,soc=u.soc,
        snapshot_sha256=hashlib.sha256(data).hexdigest(),original_request=tape[tick]['tape']['actions'][str(uid)],
        original_likelihood=tape[tick]['likelihoods'][str(uid)],teacher_gate=action.gate,
        teacher_proposal=action.proposal,teacher_request=action.normalized.tolist(),
        selected_certificate=certificates[-1] if certificates else None,audit=audit,
        elapsed_seconds=time.monotonic()-started,executed=False))
print(json.dumps(_json(out),indent=2,allow_nan=False))
