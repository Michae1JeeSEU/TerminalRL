import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['prepare', 'smoke', 'fit'])
args = parser.parse_args()
q = shlex.quote
root = '/home/user/training_outputs/terminal'
outroot = root+'/p3_v25_set_fit_20261008'
out = outroot+'/'+args.mode
files = ['terminal_marl/algo/trajectory_set_actor.py', 'terminal_marl/training/p3_v25_set_learnability.py',
    'terminal_marl/training/p3_v25_context_dataset.py', 'terminal_marl/training/p3_v3_bc_learnability.py',
    'terminal_marl/core/trajectory_context_observation.py', 'terminal_marl/algo/joint_ppo_v4_2.py',
    'docs/evidence/v4_2/p3_v25_set_fit_20261008.md']
checks = ''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
command = shlex.join(['env', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
    '/home/user/envs/terminal_env/bin/python', '-u', '-m', 'terminal_marl.training.p3_v25_set_learnability',
    '--root', root, '--output', out, '--mode', args.mode])
lines = ['set -eu', 'cd /home/user/projects/terminal', 'printf %s '+q(checks)+' | sha256sum -c -',
    'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.training.p3_v25_set_learnability\' || true)"']
if args.mode == 'fit':
    check = "import json; from pathlib import Path; r=Path("+repr(outroot)+"); assert json.loads((r/'prepare/result.json').read_text())['status']=='PREPARED'; s=json.loads((r/'smoke/result.json').read_text()); assert s['status']=='FIT_COMPLETE' and s['parameter_updates']==4"
    lines.append('/home/user/envs/terminal_env/bin/python -c '+q(check))
for suffix in ('', '.log', '.exit', '.pid'):
    lines.append('test ! -e '+q(out+suffix))
lines += ['mkdir -p '+q(outroot), 'set +e',
    'timeout '+('1830' if args.mode == 'fit' else '180')+' '+command+' > '+q(out+'.log')+' 2>&1 &',
    'task_pid=$!', 'printf \'%s\n\' "$task_pid" > '+q(out+'.pid'), 'wait "$task_pid"', 'code=$?',
    'printf \'%s\n\' "$code" > '+q(out+'.exit'), 'cat '+q(out+'.log'), 'exit "$code"']
raise SystemExit(subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'lab-server',
    '\n'.join(lines)]).returncode)
