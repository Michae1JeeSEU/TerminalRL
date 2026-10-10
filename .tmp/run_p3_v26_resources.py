"""Bounded local orchestrator; sources reach server exclusively by Syncthing."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('stage',choices=('check','smoke','train_01','train_02','train_03'))
a=p.parse_args();q=shlex.quote
base='/home/user/training_outputs/terminal/p3_v26_resources_20261009'
python='/home/user/envs/terminal_env/bin/python'
tests=['tests/test_p3_v26_course_data.py','tests/test_p3_v25_course_data.py',
 'tests/test_p3_v25_train_generalization.py','tests/test_resource_context_observation.py',
 'tests/test_p3_v25_runtime.py','tests/test_trajectory_set_actor.py','tests/test_trajectory_set_ppo.py',
 'tests/test_p3_v25_critic_initialization.py']
files=[
 'terminal_marl/training/p3_v26_course_data.py','terminal_marl/training/p3_v26_resource_inputs.py',
 'terminal_marl/training/p3_v25_course_data.py','terminal_marl/training/p3_v25_imitation_qualification.py',
 'terminal_marl/training/p3_v25_context_dataset.py','terminal_marl/core/resource_context_observation.py',
 'terminal_marl/core/trajectory_context_observation.py','terminal_marl/core/observations.py',
 'terminal_marl/training/p3_v25_runtime.py','terminal_marl/training/p3_v25_rollout.py',
 'terminal_marl/training/p3_v25_training.py','terminal_marl/algo/trajectory_set_actor.py',
 'terminal_marl/algo/trajectory_set_ppo.py','terminal_marl/scripts/diagnose_p3_v25_train_generalization.py',
 'terminal_marl/scripts/collect_p3_v3_sparse_expert.py','terminal_marl/scripts/collect_p3_v3_i0.py',
 'terminal_marl/core/config.py','terminal_marl/core/priority.py','terminal_marl/core/merge_scheduler.py',
 'terminal_marl/core/merge_gap_forecast.py','terminal_marl/core/quintic_hull.py',
 'terminal_marl/core/trajectory_time_parameterization.py','terminal_marl/core/trajectory_generator.py',
 'terminal_marl/core/safety_filter.py','terminal_marl/core/runtime_traffic_safety.py',
 'terminal_marl/core/runtime_separation.py','terminal_marl/envs/parallel_env.py',
 'terminal_marl/configs/p3_integrated_repair.py','terminal_marl/configs/p3_v25_course.py',
 'docs/evidence/v4_2/p3_v25_generalization_repair_20261009.md']+tests
hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
checks=''.join(d+'  '+f+'\n' for f,d in hashes.items())
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
 'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python.*terminal_marl\' || true)"',
 'mkdir -p '+q(base)]
if a.stage=='check':
    lines+=['test ! -e '+q(base+'/preflight.log'),'set +e',
      'env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '+q(python)+' -m pytest '+shlex.join(tests)+' -q > '+q(base+'/preflight.log')+' 2>&1',
      'code=$?','printf \'%s\\n\' "$code" > '+q(base+'/preflight.exit'),
      'cat '+q(base+'/preflight.log'),'exit "$code"']
else:
    index=1 if a.stage=='smoke' else int(a.stage[-2:]);seed=892871 if a.stage=='smoke' else 893600+index
    out=base+'/'+a.stage
    lines+=['test "$(cat '+q(base+'/preflight.exit')+')" = 0']
    lines+=['test ! -e '+q(out+s) for s in ('','.pid','.wrapper.pid','.log','.exit')]
    guard='''import json
from pathlib import Path
root=Path(BASE)
if STAGE!='smoke':
    r=json.loads((root/'smoke/result.json').read_text())
    assert r['status']=='SMOKE_OK' and r['resource_reconstruction']['rows']==3
for i in range(1,INDEX):
    from terminal_marl.training.p3_v26_course_data import validated_case
    validated_case(root/f'train_{i:02d}',i)
matches=[]
for p in Path('/home/user/training_outputs/terminal').rglob('protocol.json'):
    try:r=json.loads(p.read_text())
    except (OSError,ValueError):continue
    if r.get('seed')==SEED or r.get('env_seed')==SEED:matches.append(str(p))
assert not matches, matches
print({'preflight_ok':True,'seed':SEED,'duplicates':matches})
'''.replace('BASE',repr(base)).replace('STAGE',repr(a.stage)).replace('INDEX',str(index)).replace('SEED',str(seed))
    lines.append(q(python)+' -c '+q(guard))
    command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
      'timeout','600' if a.stage=='smoke' else '27000',python,'-u','-m',
      'terminal_marl.training.p3_v26_course_data','--index',str(index),'--output',out]
    if a.stage=='smoke':command+=['--smoke']
    body='\n'.join(['cd /home/user/projects/terminal',shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
      'task_pid=$!','printf \'%s\\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
      'printf \'%s\\n\' "$code" > '+q(out+'.exit'),'exit "$code"'])
    lines+=['nohup sh -c '+q(body)+' > /dev/null 2>&1 < /dev/null &',
      'wrapper_pid=$!','printf \'%s\\n\' "$wrapper_pid" > '+q(out+'.wrapper.pid'),
      'printf \'wrapper_pid=%s\\n\' "$wrapper_pid"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
