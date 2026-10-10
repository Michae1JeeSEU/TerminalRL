"""Exact saved-tape reconstruction and bounded queries for the failed student."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.core.safety_trace import _digest,_json
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical,state_record,execute_tape,write_json
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
report=dict(status='ERROR',error=None,reset_count=0,training=False,smoke=a.smoke,queries=[])
started=time.monotonic()
try:
    for name,digest in {'first_before.pkl':'7dd0ff04561d40abfd7898c9b8d09cebaba8e3eb214010dadedcd4b6d69c7a0c',
        'requests.jsonl':'c13a80147f7158bd14facaaccfe5fea20d5ea8854561cc3aee8340648e29e948'}.items():
        assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==digest,name
    records=[json.loads(line) for line in (a.source/'requests.jsonl').read_text(encoding='utf8').splitlines()]
    env=pickle.loads((a.source/'first_before.pkl').read_bytes())
    targets={} if a.smoke else {125:(29,),164:(49,),365:(29,),665:(29,49)}
    end=2 if a.smoke else max(targets)
    for tick in range(end+1):
        assert time.monotonic()-started<1800,'diagnostic wall cap'
        record=records[tick]
        assert record['tick']==tick and _digest(canonical(state_record(env)))==record['pre_hash'],f'pre hash {tick}'
        if tick in targets:
            saved=pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL)
            (a.output/f'state_{tick}.pkl').write_bytes(saved)
            for uid in targets[tick]:
                probe=pickle.loads(saved);u=probe.active[uid]
                assert record['actor_mask'][str(uid)] and probe.now>=u.safety_retry_after
                safety=[];original=probe.safety.filter_plan
                def filtered(plan,committed,snapshot):
                    verdict=original(plan,committed,snapshot)
                    safety.append(dict(verdict=asdict(verdict),audit=deepcopy(probe.safety.last_predicted_track_audit)))
                    return verdict
                probe.safety.filter_plan=filtered
                begin=time.monotonic()
                label,audit=teacher_joint_expert_v4_1(probe,u,state_only=True)
                item=_json(dict(tick=tick,uid=uid,soc=u.soc,phase=u.phase.value,position=u.position,
                    policy_request=record['tape']['actions'][str(uid)],joint_index=label.index,
                    teacher_request=label.normalized,audit=audit,safety=safety,elapsed_seconds=time.monotonic()-begin,
                    active=[dict(uid=x.uid,soc=x.soc,phase=x.phase.value,position=x.position,
                                 committed=x.committed_plan is not None) for x in probe.active.values()]))
                write_json(a.output/f'query_{tick}_{uid}.json',item)
                report['queries'].append({key:item[key] for key in ('tick','uid','soc','joint_index','elapsed_seconds')})
                assert pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL)==saved,'query changed replay state'
        if tick==end:break
        live=execute_tape(env,record['tape'])
        assert _digest(live['after'])==record['post_hash'],f'post hash {tick}'
        assert live['outputs'][4]['team_reward']==record['reward'],f'reward {tick}'
        if tick==0 or (tick+1)%25==0:
            write_json(a.output/'progress.json',dict(replayed=tick+1,target=end,elapsed_seconds=time.monotonic()-started))
    report.update(status='SMOKE_OK' if a.smoke else 'DIAGNOSIS_COMPLETE',replayed=end,exact_replay=True)
except Exception as error:
    report.update(error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
report['elapsed_seconds']=time.monotonic()-started
write_json(a.output/'result.json',report)
print(json.dumps(report))
raise SystemExit(0 if report['error'] is None else 1)
