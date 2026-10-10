import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['probe', 'full'])
args = parser.parse_args()
q = shlex.quote
root = '/home/user/training_outputs/terminal'
out = root+'/p3_v3_observation_v25_20261008/'+args.mode
files = ['terminal_marl/core/trajectory_context_observation.py', 'terminal_marl/envs/parallel_env.py',
    'terminal_marl/training/p3_v25_context_dataset.py', 'terminal_marl/scripts/reconstruct_p3_v25_context.py',
    'terminal_marl/core/observations.py', 'terminal_marl/core/safety_filter.py',
    'terminal_marl/core/topology_neighbors.py', 'terminal_marl/core/vehicle_dynamics.py']
checks = ''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
command = shlex.join(['env', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
    '/home/user/envs/terminal_env/bin/python', '-u', '-m', 'terminal_marl.scripts.reconstruct_p3_v25_context',
    '--root', root, '--output', out, '--mode', args.mode])
lines = ['set -eu', 'cd /home/user/projects/terminal', 'printf %s '+q(checks)+' | sha256sum -c -',
    'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.scripts.reconstruct_p3_v25_context\' || true)"']
for suffix in ('', '.log', '.exit', '.pid'):
    lines.append('test ! -e '+q(out+suffix))
lines += ['mkdir -p '+q(str(Path(out).parent).replace('\\','/')), 'set +e',
    'timeout '+('210' if args.mode == 'probe' else '930')+' '+command+' > '+q(out+'.log')+' 2>&1 &',
    'task_pid=$!', 'printf \'%s\n\' "$task_pid" > '+q(out+'.pid'), 'wait "$task_pid"', 'code=$?',
    'printf \'%s\n\' "$code" > '+q(out+'.exit'), 'cat '+q(out+'.log'), 'exit "$code"']
raise SystemExit(subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'lab-server',
    '\n'.join(lines)]).returncode)
