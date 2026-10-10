"""Exact launcher; labels, first candidate, and remaining candidates are separate stages."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

SOURCE = 'terminal_marl/scripts/diagnose_p3_teacher_context.py'
HELPER = '.tmp/run_p3_teacher_audit_20261008.py'
ROOT = '/home/user/training_outputs/terminal'
OUT = ROOT+'/p3_v3_teacher_safety_audit_20261008'
PYTHON = '/home/user/envs/terminal_env/bin/python'

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['labels', 'first', 'remaining'])
parser.add_argument('--server', action='store_true')
args = parser.parse_args()
if not args.server:
    files = [SOURCE, HELPER, 'terminal_marl/core/safety_filter.py',
             'terminal_marl/core/topology_neighbors.py', 'terminal_marl/core/observations.py',
             'terminal_marl/training/v4_1_joint_teacher.py', 'terminal_marl/envs/parallel_env.py']
    checks = ''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
    remote = '\n'.join(['set -eu', 'cd /home/user/projects/terminal',
        'printf %s '+shlex.quote(checks)+' | sha256sum -c -',
        shlex.join(['env', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                    PYTHON, HELPER, args.stage, '--server'])])
    raise SystemExit(subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'lab-server', remote]).returncode)

assert Path.cwd() == Path('/home/user/projects/terminal')
out = Path(OUT)
if args.stage == 'labels':
    out.mkdir(exist_ok=False)
    indices = [None]
elif args.stage == 'first':
    assert json.loads((out/'labels.json').read_text())['status'] == 'LABEL_AUDIT_COMPLETE'
    assert int((out/'candidate_0.exit').read_text()) == 1
    out = out/'witness_v2'
    out.mkdir(exist_ok=False)
    indices = [0]
else:
    out = out/'witness_v2'
    first = json.loads((out/'candidate_0.json').read_text())
    assert first['status'] == 'CANDIDATE_WITNESS_COMPLETE'
    assert int((out/'candidate_0.exit').read_text()) == 0
    indices = [1, 2, 3, 4, 5]
failed_cost = 0. if args.stage == 'labels' else json.loads((Path(OUT)/'candidate_0.cost.json').read_text())['wall_seconds']
used = failed_cost if args.stage != 'remaining' else failed_cost+json.loads((out/'candidate_0.cost.json').read_text())['wall_seconds']
for index in indices:
    stem = 'labels' if index is None else f'candidate_{index}'
    for suffix in ('.json', '.log', '.exit', '.cost.json'):
        assert not (out/(stem+suffix)).exists()
    command = [PYTHON, '-u', '-m', 'terminal_marl.scripts.diagnose_p3_teacher_context',
        '--root', ROOT, '--output', str(out/(stem+'.json'))]
    if index is not None:
        command += ['--candidate-index', str(index)]
    started = time.monotonic()
    with (out/(stem+'.log')).open('x') as log:
        try:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                timeout=180 if index is None else min(180., max(0.01, 600.-used)), check=False)
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = 124
    cost = time.monotonic()-started
    used += cost
    (out/(stem+'.exit')).write_text(str(code)+'\n')
    (out/(stem+'.cost.json')).write_text(json.dumps(dict(wall_seconds=cost, candidate_budget_used=used)))
    print(json.dumps(dict(stem=stem, exit_code=code, wall_seconds=cost)), flush=True)
    if code != 0:
        print((out/(stem+'.log')).read_text())
        raise SystemExit(code)
