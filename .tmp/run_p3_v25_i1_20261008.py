"""Launch exactly one primary-approved phase, after Syncthing/hash checks."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser()
p.add_argument('case',choices=['smoke_collect','smoke_fit','Q_A','Q_B','Q_C','fit','evaluate'])
a = p.parse_args()
root = '/home/user/training_outputs/terminal'
base = root+'/p3_v25_i1_20261008'
smoke_base = root+'/p3_v25_i1_smoke_20261008'
out = (smoke_base+'/'+a.case.removeprefix('smoke_')) if a.case.startswith('smoke_') else base+'/'+a.case
module = 'terminal_marl.training.p3_v25_imitation_qualification'
files = ['terminal_marl/training/p3_v25_imitation_qualification.py',
 'terminal_marl/training/p3_v25_set_learnability.py','terminal_marl/training/p3_v25_context_dataset.py',
 'terminal_marl/scripts/collect_p3_v3_sparse_expert.py','terminal_marl/scripts/collect_p3_v3_i0.py',
 'terminal_marl/training/p3_v3_expert_dataset.py','terminal_marl/algo/trajectory_set_actor.py',
 'terminal_marl/core/trajectory_context_observation.py','terminal_marl/envs/parallel_env.py',
 'terminal_marl/core/safety_filter.py','terminal_marl/configs/p3_integrated_repair.py',
 'docs/evidence/v4_2/p3_v25_i1_qualification_20261008.md']
checks = ''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
args = ['--output',out]
if a.case.startswith('Q_') or a.case == 'smoke_collect':
    args += ['--mode','collect','--index',str(['Q_A','Q_B','Q_C'].index(a.case)+1) if a.case.startswith('Q_') else '1']
elif a.case in {'fit','smoke_fit'}:
    args += ['--mode','fit','--root',root]
else:
    args += ['--mode','evaluate','--fit-directory',base+'/fit']
if a.case.startswith('smoke_'):
    args += ['--smoke']
if a.case in {'fit','evaluate'}:
    args += ['--q-directories',*(base+'/'+k for k in ('Q_A','Q_B','Q_C'))]
python = '/home/user/envs/terminal_env/bin/python'
command = ['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u','-m',module,*args]
q = shlex.quote
lines = ['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
 'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.(scripts.evaluate_p3_v25_student|training.p3_v25_imitation_qualification)\' || true)"']
for suffix in ('','.pid','.log','.exit'):
    lines += ['test ! -e '+q(out+suffix)]
if a.case in {'Q_B','Q_C','fit','evaluate'}:
    required = {'Q_B':['Q_A'],'Q_C':['Q_A','Q_B'],'fit':['Q_A','Q_B','Q_C'],'evaluate':['fit']}[a.case]
    for item in required:
        expected = 'FIT_FROZEN' if item == 'fit' else 'COLLECTION_OK'
        script = 'import json; from pathlib import Path; r=json.loads(Path('+repr(base+'/'+item+'/result.json')+').read_text()); assert r["status"]=='+repr(expected)
        lines += [q(python)+' -c '+q(script)]
if a.case == 'Q_A':
    inventory = """import json
from pathlib import Path
root=Path('/home/user/training_outputs/terminal')
wanted={895301,895302,895303,892805}
collisions=[]
def matches(v):
    if isinstance(v,dict): return any(matches(x) for x in v.values())
    if isinstance(v,list): return any(matches(x) for x in v)
    return type(v) is int and v in wanted
for path in root.rglob('protocol.json'):
    if 'p3_v25_i1_smoke_20261008' in path.parts: continue
    try: value=json.loads(path.read_text())
    except (OSError,ValueError): continue
    if matches(value): collisions.append(str(path))
print(json.dumps({'historical_protocol_seed_collisions':collisions}))
assert not collisions
"""
    lines += [q(python)+' -c '+q(inventory)]
limit = 600 if a.case.startswith('smoke_') else 1200 if a.case in {'fit','evaluate'} else 12000
lines += ['mkdir -p '+q(str(Path(out).parent).replace('\\','/')),'set +e',
 'timeout '+str(limit)+' '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
