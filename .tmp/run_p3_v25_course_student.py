"""One primary-approved collection phase; never automatically starts the next."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('case',choices=['smoke','flow6_mixed','dense24_fixed','dense24_mixed']);a=p.parse_args()
root='/home/user/training_outputs/terminal';base=root+'/p3_v25_neutral_course_20261008';fit=base+'/fit_initial';out=base+'/student_initial/'+a.case
module='terminal_marl.training.p3_v25_task_student';python='/home/user/envs/terminal_env/bin/python';q=shlex.quote
files=['terminal_marl/training/p3_v25_task_student.py','terminal_marl/training/p3_v25_task_imitation.py',
 'terminal_marl/training/p3_v25_runtime.py','terminal_marl/training/p3_v25_rollout.py',
 'terminal_marl/training/p3_v25_training.py','terminal_marl/training/p3_v25_task_metrics.py',
 'terminal_marl/algo/trajectory_set_actor.py','terminal_marl/algo/trajectory_set_ppo.py',
 'terminal_marl/core/trajectory_context_observation.py','terminal_marl/configs/p3_integrated_repair.py',
 'terminal_marl/configs/p3_v25_course.py','terminal_marl/core/config.py','terminal_marl/core/merge_scheduler.py',
 'terminal_marl/core/quintic_hull.py','terminal_marl/core/trajectory_time_parameterization.py','terminal_marl/core/trajectory_generator.py','terminal_marl/core/safety_filter.py','terminal_marl/envs/parallel_env.py',
 'docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
cases=['flow6_mixed','dense24_fixed','dense24_mixed']
index=0 if a.case=='smoke' else cases.index(a.case)+1
seed=892852+index;policy_seed=929852+index
scenario='dense24_mixed' if a.case=='smoke' else a.case
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u','-m',module,
 '--fit-directory',fit,'--scenario',scenario,'--seed',str(seed),'--policy-seed',str(policy_seed),
 '--output',out]+(['--smoke'] if a.case=='smoke' else [])
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
 'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.(training.p3_v25_annotation_repair|training.p3_v25_task_student|training.p3_v25_course_data|training.p3_v25_task_data|training.p3_v25_forecast_data|scripts.diagnose_p3_integrated_repair|scripts.evaluate_p3_v25_student|scripts.collect_p3_v3_)\' || true)"']
for suffix in ('','.pid','.log','.exit'): lines.append('test ! -e '+q(out+suffix))
for previous in ([] if a.case=='smoke' else ['smoke']+cases[:index-1]):
    status='SMOKE_OK' if previous=='smoke' else 'DEVELOPMENT_COMPLETE'
    check='import json;from pathlib import Path;r=json.loads(Path('+repr(base+'/student_initial/'+previous+'/result.json')+').read_text());assert r["status"]=='+repr(status)+' and not r["blockers"]'
    lines.append(q(python)+' -c '+q(check))
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
lines+=['mkdir -p '+q(base+'/student_initial'),'set +e','timeout '+('600' if a.case=='smoke' else '7600')+' '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
