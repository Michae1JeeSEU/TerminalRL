import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('case',choices=['smoke','C_rule']);a=p.parse_args()
root='/home/user/training_outputs/terminal/p3_future_merge_gap_20261008';out=root+'/'+a.case
q=shlex.quote;python='/home/user/envs/terminal_env/bin/python'
files=['terminal_marl/core/config.py','terminal_marl/core/merge_scheduler.py',
 'terminal_marl/core/merge_gap_forecast.py','terminal_marl/core/safety_filter.py',
 'terminal_marl/core/trajectory_generator.py','terminal_marl/envs/parallel_env.py',
 'terminal_marl/configs/p3_integrated_repair.py','terminal_marl/scripts/diagnose_p3_integrated_repair.py',
 'docs/evidence/v4_2/p3_v25_formal_readiness_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u','-m',
 'terminal_marl.scripts.diagnose_p3_integrated_repair','--smoke' if a.case=='smoke' else '--run',
 '--controller','rule','--forecast-merge-gap','--requests','3' if a.case=='smoke' else '24',
 '--warm-start','3' if a.case=='smoke' else '12','--seed','892806' if a.case=='smoke' else '892804',
 '--policy-seed','928806' if a.case=='smoke' else '928804','--max-steps','1800',
 '--trace-blockers','--snapshot-ticks','1']
if a.case=='smoke':command+=['--high-soc']
else:command+=['400','600','850']
command+=['--output',out]
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
 'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.scripts.(evaluate_p3_v25_student|diagnose_p3_integrated_repair)\' || true)"']
for suffix in ('','.pid','.log','.exit'):lines.append('test ! -e '+q(out+suffix))
if a.case=='C_rule':
    code=('import json;from pathlib import Path;r=json.loads(Path('+repr(root+'/smoke/result.json')+').read_text());'
          'assert r["error"] is None and r["ticks"]==2 and r["config"]["merge_gap_prediction_mode"]=="committed_arrivals_v1";'
          'assert all(r["telemetry"][k]==0 for k in ("hard_separation_violations","collisions","nmac","checked_executed_plan_hash_mismatches","shield_fail_closed"))')
    lines.append(q(python)+' -c '+q(code))
lines+=['mkdir -p '+q(root),'set +e','timeout '+('180' if a.case=='smoke' else '3700')+' '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
