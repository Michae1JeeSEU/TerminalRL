import hashlib
from pathlib import Path
import shlex
import subprocess

root='/home/user/training_outputs/terminal/p3_v25_student_dev_20261008'
out=root+'/C_rule_same_seed'
files=['terminal_marl/scripts/diagnose_p3_integrated_repair.py','terminal_marl/configs/p3_integrated_repair.py',
 'terminal_marl/envs/parallel_env.py','terminal_marl/core/safety_filter.py','terminal_marl/core/trajectory_generator.py',
 'docs/evidence/v4_2/p3_v25_formal_readiness_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
python='/home/user/envs/terminal_env/bin/python';q=shlex.quote
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u','-m',
 'terminal_marl.scripts.diagnose_p3_integrated_repair','--run','--controller','rule','--requests','24',
 '--warm-start','12','--seed','892804','--policy-seed','928804','--max-steps','1800',
 '--trace-blockers','--snapshot-ticks','1','400','600','850','--output',out]
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
 'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.scripts.(evaluate_p3_v25_student|diagnose_p3_integrated_repair)\' || true)"']
for suffix in ('','.pid','.log','.exit'):lines.append('test ! -e '+q(out+suffix))
for case in ('C_det','C_sto'):
    code='import json; from pathlib import Path; r=json.loads(Path('+repr(root+'/'+case+'/result.json')+').read_text()); assert r["status"]=="COMPLETED" and r["replay_exact"] and r["error"] is None'
    lines.append(q(python)+' -c '+q(code))
lines+=['set +e','timeout 3700 '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
