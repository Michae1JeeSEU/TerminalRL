import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser(); p.add_argument('mode',choices=['smoke','curve']); a=p.parse_args()
q=shlex.quote
root='/home/user/training_outputs/terminal'; outroot=root+'/p3_v25_set_curve_20261008'; out=outroot+'/'+a.mode
files=['terminal_marl/training/p3_v25_fit_curve.py','terminal_marl/training/p3_v25_set_learnability.py',
       'terminal_marl/algo/trajectory_set_actor.py','terminal_marl/training/p3_v25_context_dataset.py',
       'docs/evidence/v4_2/p3_v25_formal_readiness_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
 '/home/user/envs/terminal_env/bin/python','-u','-m','terminal_marl.training.p3_v25_fit_curve',
 '--root',root,'--output',out]+(['--smoke'] if a.mode=='smoke' else [])
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
 'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.training.p3_v25_fit_curve\' || true)"']
if a.mode=='curve':
    code="import json; from pathlib import Path; r=json.loads(Path("+repr(outroot+'/smoke/result.json')+").read_text()); assert r['status']=='CURVE_COMPLETE' and r['smoke']"
    lines.append('/home/user/envs/terminal_env/bin/python -c '+q(code))
for suffix in ('','.pid','.log','.exit'): lines.append('test ! -e '+q(out+suffix))
lines+=['mkdir -p '+q(outroot),'set +e','timeout '+('180' if a.mode=='smoke' else '930')+' '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
