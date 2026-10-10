"""One primary-approved collection phase; never automatically starts the next."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

root='/home/user/training_outputs/terminal';base=root+'/p3_v25_task_20261008';out=base+'/fit_initial'
module='terminal_marl.training.p3_v25_task_imitation';python='/home/user/envs/terminal_env/bin/python';q=shlex.quote
files=['terminal_marl/training/p3_v25_task_imitation.py','terminal_marl/training/p3_v25_task_data.py',
 'terminal_marl/training/p3_v25_imitation_qualification.py','terminal_marl/training/p3_v25_set_learnability.py',
 'terminal_marl/training/p3_v25_context_dataset.py','terminal_marl/training/p3_v3_bc_learnability.py',
 'terminal_marl/algo/trajectory_set_actor.py','terminal_marl/algo/joint_ppo_v4_2.py',
 'terminal_marl/configs/p3_v25_task.py','docs/evidence/v4_2/p3_v25_actual_task_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u','-m',module,
 '--mode','fit','--train-directories',*[base+'/'+c for c in ['train_flow6','train_dense24','train_flow12']],
 '--output',out]
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
 'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.(training.p3_v25_task_imitation|training.p3_v25_task_student|training.p3_v25_task_data|training.p3_v25_forecast_data|scripts.diagnose_p3_integrated_repair|scripts.evaluate_p3_v25_student|scripts.collect_p3_v3_)\' || true)"']
for suffix in ('','.pid','.log','.exit'): lines.append('test ! -e '+q(out+suffix))
for previous in ['train_flow6','train_dense24','train_flow12']:
    check='import json;from pathlib import Path;r=json.loads(Path('+repr(base+'/'+previous+'/result.json')+').read_text());assert r["status"]=="COLLECTION_OK" and r["error"] is None and not r["blockers"]'
    lines.append(q(python)+' -c '+q(check))
lines+=['mkdir -p '+q(base),'set +e','timeout '+'1050'+' '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
