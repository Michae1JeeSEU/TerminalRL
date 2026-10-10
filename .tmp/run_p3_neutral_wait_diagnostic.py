import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess
p=argparse.ArgumentParser();p.add_argument('case',choices=('smoke','early_late'));a=p.parse_args()
python='/home/user/envs/terminal_env/bin/python';q=shlex.quote
base='/home/user/training_outputs/terminal/p3_neutral_student_fixed_blocker_20261008'
out=base+'/'+a.case
source='/home/user/training_outputs/terminal/p3_v25_neutral_course_20261008/student_initial/dense24_fixed/episode'
files=['.tmp/diagnose_p3_neutral_wait.py','terminal_marl/core/priority.py',
 'terminal_marl/core/trajectory_generator.py','terminal_marl/core/quintic_hull.py',
 'terminal_marl/core/trajectory_time_parameterization.py','terminal_marl/core/safety_filter.py',
 'terminal_marl/envs/parallel_env.py','terminal_marl/training/v4_1_joint_teacher.py',
 'docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -']
lines+=['test ! -e '+q(out+suffix) for suffix in ('','.pid','.log','.exit')]
lines.append('test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u (-m terminal_marl.training.p3_v25_|.tmp/diagnose_p3_neutral_wait.py)\' || true)"')
if a.case=='early_late':
    check='import json;from pathlib import Path;r=json.loads(Path('+repr(base+'/smoke/result.json')+').read_text());assert r["status"]=="SMOKE_OK" and r["exact_replay"] and r["replayed"]==2'
    lines.append(q(python)+' -c '+q(check))
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u',
 '.tmp/diagnose_p3_neutral_wait.py','--source',source,'--output',out]+(['--smoke'] if a.case=='smoke' else [])
lines+=['mkdir -p '+q(base),'set +e','timeout '+('120' if a.case=='smoke' else '1900')+' '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
