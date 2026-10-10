"""One native-platform saved-state diagnostic, only after the collector exits."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('slot',type=int,choices=(0,129));a=p.parse_args()
q=shlex.quote
python='/home/user/envs/terminal_env/bin/python'
files=['.tmp/verify_p3_hull_query_server.py','.tmp/p3_ring_hold_hull_candidate.py',
 'terminal_marl/core/trajectory_generator.py','terminal_marl/core/trajectory_time_parameterization.py',
 'terminal_marl/core/polynomial_trajectory.py','terminal_marl/core/safety_filter.py',
 'terminal_marl/core/vehicle_dynamics.py','terminal_marl/envs/parallel_env.py']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
base='/home/user/training_outputs/terminal/p3_v25_hull_query_20261008'
out=base+f'/query_{a.slot}'
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
 'test ! -d /proc/2822837','test ! -d /proc/2822836',
 'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.(training.p3_v25_course_data|scripts.collect_p3_v3_sparse_expert)\' || true)"']
for suffix in ('','.log','.exit'):lines.append('test ! -e '+q(out+suffix))
if a.slot==129:
    check="import json;from pathlib import Path;assert json.loads(Path("+repr(base+'/query_0/result.json')+").read_text())['status']=='QUERY_EXACT'"
    lines.append(q(python)+' -c '+q(check))
lines+=['mkdir -p '+q(base),'set +e','timeout 120 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '+q(python)+' -u .tmp/verify_p3_hull_query_server.py '+str(a.slot)+' > '+q(out+'.log')+' 2>&1',
 'code=$?','printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
