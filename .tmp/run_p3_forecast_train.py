"""One primary-approved collection phase; never automatically starts the next."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('case',choices=['smoke','train_A','train_B','train_C']);a=p.parse_args()
root='/home/user/training_outputs/terminal';base=root+'/p3_v25_forecast_train_20261008';out=base+'/'+a.case
module='terminal_marl.training.p3_v25_forecast_data';python='/home/user/envs/terminal_env/bin/python';q=shlex.quote
files=['terminal_marl/training/p3_v25_forecast_data.py','terminal_marl/training/p3_v25_imitation_qualification.py',
 'terminal_marl/training/p3_v25_context_dataset.py','terminal_marl/core/trajectory_context_observation.py',
 'terminal_marl/scripts/collect_p3_v3_sparse_expert.py','terminal_marl/scripts/collect_p3_v3_i0.py',
 'terminal_marl/core/config.py','terminal_marl/core/merge_scheduler.py','terminal_marl/core/merge_gap_forecast.py',
 'terminal_marl/core/trajectory_generator.py','terminal_marl/core/safety_filter.py','terminal_marl/envs/parallel_env.py',
 'terminal_marl/configs/p3_integrated_repair.py','docs/evidence/v4_2/p3_v25_forecast_train_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
index=1 if a.case=='smoke' else ['train_A','train_B','train_C'].index(a.case)+1
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u','-m',module,
 '--index',str(index),'--output',out]+(['--smoke'] if a.case=='smoke' else [])
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
 'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.(training.p3_v25_forecast_data|scripts.diagnose_p3_integrated_repair|scripts.evaluate_p3_v25_student|scripts.collect_p3_v3_)\' || true)"']
for suffix in ('','.pid','.log','.exit'): lines.append('test ! -e '+q(out+suffix))
check=('import json;from pathlib import Path;r=json.loads(Path('+repr(root+'/p3_future_merge_gap_20261008/C_rule/result.json')+').read_text());'
       'assert r["status"]=="COMPLETED" and r["error"] is None and not r["censored"];'
       'assert r["config"]["merge_gap_prediction_mode"]=="committed_arrivals_v1";'
       'assert len(r["outcomes"])==24 and all(v in {"completed","emergency_landed"} for v in r["outcomes"].values());'
       'assert all(r["telemetry"][k]==0 for k in ("hard_separation_violations","collisions","nmac","checked_executed_plan_hash_mismatches","shield_fail_closed"))')
lines.append(q(python)+' -c '+q(check))
for previous in ([] if a.case=='smoke' else ['smoke']+['train_A','train_B'][:index-1]):
    status='SMOKE_OK' if previous=='smoke' else 'COLLECTION_OK'
    check='import json;from pathlib import Path;r=json.loads(Path('+repr(base+'/'+previous+'/result.json')+').read_text());assert r["status"]=='+repr(status)+' and r["error"] is None and not r["blockers"]'
    lines.append(q(python)+' -c '+q(check))
seed=892807 if a.case=='smoke' else 893300+index
inventory="""import json
from pathlib import Path
seed=SEED
collisions=[]
for path in Path('/home/user/training_outputs/terminal').rglob('protocol.json'):
    try: value=json.loads(path.read_text())
    except (OSError,ValueError): continue
    if value.get('seed')==seed or value.get('env_seed')==seed: collisions.append(str(path))
print(json.dumps({'seed':seed,'prior_protocols':collisions}))
assert not collisions
""".replace('SEED',str(seed))
lines.append(q(python)+' -c '+q(inventory))
lines+=['mkdir -p '+q(base),'set +e','timeout '+('600' if a.case=='smoke' else '12000')+' '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
