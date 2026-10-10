import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('case',choices=['smoke','A_det','A_sto','B_det','B_sto','C_det','C_sto']);a=p.parse_args()
q=shlex.quote
root='/home/user/training_outputs/terminal';outroot=root+'/p3_v25_student_dev_20261008';out=outroot+'/'+a.case
checkpoint='p3_v25_set_curve_20261008/curve/seed_91701/diagnostic_final.pt'
seed,policy_seed,requests,warm,high=(892801,928801,3,3,True) if a.case=='smoke' else {
    'A':(892802,928802,12,6,True),'B':(892803,928803,24,12,True),'C':(892804,928804,24,12,False)}[a.case[0]]
files=['terminal_marl/training/p3_v25_runtime.py','terminal_marl/scripts/evaluate_p3_v25_student.py',
 'terminal_marl/algo/trajectory_set_actor.py','terminal_marl/core/trajectory_context_observation.py',
 'terminal_marl/core/safety_filter.py','terminal_marl/envs/parallel_env.py','terminal_marl/configs/p3_integrated_repair.py',
 'terminal_marl/scripts/diagnose_p3_direct_contract.py','docs/evidence/v4_2/p3_v25_formal_readiness_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
checks+=hashlib.sha256((Path('D:/7_codespace/training_outputs/terminal')/checkpoint).read_bytes()).hexdigest()+'  '+root+'/'+checkpoint+'\n'
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
 '/home/user/envs/terminal_env/bin/python','-u','-m','terminal_marl.scripts.evaluate_p3_v25_student',
 '--checkpoint',root+'/'+checkpoint,'--output',out,'--seed',str(seed),'--policy-seed',str(policy_seed),
 '--requests',str(requests),'--warm-start',str(warm)]
if high:command+=['--high-soc']
if a.case=='smoke' or a.case.endswith('_det'):command+=['--deterministic']
if a.case=='smoke':command+=['--smoke']
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
 'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.scripts.evaluate_p3_v25_student\' || true)"']
for suffix in ('','.pid','.log','.exit'):lines.append('test ! -e '+q(out+suffix))
lines+=['mkdir -p '+q(outroot),'set +e','timeout '+('180' if a.case=='smoke' else '7300')+' '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
