"""Prepared bounded learning launch; never usable before real Q qualification."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('stage',choices=('critic','ppo'));a=p.parse_args()
base='/home/user/training_outputs/terminal/p3_v25_neutral_course_v2_20261008'
out=base+('/critic_initialization' if a.stage=='critic' else '/ppo_pilot')
python='/home/user/envs/terminal_env/bin/python';q=shlex.quote
cases=('flow6_mixed','dense24_fixed','dense24_mixed')
development=[base+('/student_runtime_refined/' if s=='dense24_mixed' else '/student_initial/')+s for s in cases]
files=[
 'terminal_marl/training/p3_v25_critic_initialization.py','terminal_marl/training/p3_v25_ppo_block.py',
 'terminal_marl/training/p3_v25_task_imitation.py','terminal_marl/training/p3_v25_task_student.py',
 'terminal_marl/training/p3_v25_course_data.py','terminal_marl/training/p3_v25_rollout.py',
 'terminal_marl/training/p3_v25_runtime.py','terminal_marl/algo/trajectory_set_actor.py',
 'terminal_marl/algo/trajectory_set_ppo.py','terminal_marl/core/trajectory_context_observation.py',
 'terminal_marl/core/runtime_separation.py','terminal_marl/core/runtime_traffic_safety.py',
 'terminal_marl/core/safety_filter.py','terminal_marl/core/trajectory_generator.py',
 'terminal_marl/core/trajectory_time_parameterization.py','terminal_marl/core/quintic_hull.py',
 'terminal_marl/core/priority.py','terminal_marl/core/merge_gap_forecast.py',
 'terminal_marl/core/merge_scheduler.py','terminal_marl/core/config.py',
 'terminal_marl/envs/parallel_env.py','terminal_marl/configs/p3_v25_course.py',
 'terminal_marl/configs/p3_integrated_repair.py',
 'docs/evidence/v4_2/p3_v25_nominal_curve_fresh_training_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -']
lines+=['test ! -e '+q(out+suffix) for suffix in ('','.pid','.log','.exit')]
lines.append('test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.(training.p3_v25_|scripts.collect_p3_v3_)\' || true)"')
guard='''import json
from pathlib import Path
from terminal_marl.training.p3_v25_task_imitation import frozen_models,PROTOCOL,SEEDS
from terminal_marl.training.p3_v25_task_student import blockers
from terminal_marl.configs.p3_v25_course import PROFILE,SCENARIOS
base=Path(BASE)
fit=frozen_models(base/'fit_initial')
assert fit['checkpoints'][0]['sha256']=='16d3ee800263023386341cb92a6335bb63dd487a346181a0275bf4c3ecb63600'
qualification=json.loads((base/'qualification_initial'/'result.json').read_text())
assert qualification['status']=='EVALUATED' and qualification['protocol']==PROTOCOL and qualification['profile']==PROFILE
assert qualification['all_imitation_gates_passed'] and qualification['parameter_updates']==0
assert [r['seed'] for r in qualification['seeds']]==list(SEEDS)
assert all(r['decision']['passed'] and r['checkpoint_sha256']==c['sha256'] for r,c in zip(qualification['seeds'],fit['checkpoints']))
for directory,scenario in zip(__DEV_PATHS__,SCENARIOS):
    root=Path(directory);r=json.loads((root/'result.json').read_text());p=json.loads((root/'protocol.json').read_text())
    assert r['profile']==p['profile']==PROFILE and r['source']==p['source']==fit['checkpoints'][0]
    assert r['status']=='DEVELOPMENT_COMPLETE' and not r['blockers'] and not blockers(r['rollout'])
    assert not p['deterministic'] and not p['smoke'] and not p['teacher_enabled']
print({'qualification_passed':True,'bounded_stage':STAGE,'formal_launch':False})
'''.replace('BASE',repr(base)).replace('__DEV_PATHS__',repr(development)).replace('STAGE',repr(a.stage))
lines.append(q(python)+' -c '+q(guard))
if a.stage=='ppo':
    inventory='''import json
from pathlib import Path
from terminal_marl.training.p3_v25_ppo_block import input_review,TRAIN_SEEDS,POLICY_SEEDS
base=Path(BASE)
input_review(base/'fit_initial',base/'qualification_initial',base/'critic_initialization',__DEV_PATHS__)
assert TRAIN_SEEDS==(896501,896502,896503) and POLICY_SEEDS==(936501,936502,936503)
matches=[]
for path in Path('/home/user/training_outputs/terminal').rglob('protocol.json'):
    try:value=json.loads(path.read_text())
    except (OSError,ValueError):continue
    if value.get('seed') in TRAIN_SEEDS or value.get('env_seed') in TRAIN_SEEDS:matches.append(str(path))
assert not matches,matches
'''.replace('BASE',repr(base)).replace('__DEV_PATHS__',repr(development))
    lines.append(q(python)+' -c '+q(inventory))
module='terminal_marl.training.'+('p3_v25_critic_initialization' if a.stage=='critic' else 'p3_v25_ppo_block')
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u','-m',module,
 '--fit-directory',base+'/fit_initial','--development-directories',*development,'--output',out]
if a.stage=='ppo':
    command+=['--qualification-directory',base+'/qualification_initial','--warmup-directory',base+'/critic_initialization',
              '--train-directories',*[base+'/train_'+s for s in cases]]
budget=900 if a.stage=='critic' else 46800
lines+=['set +e','timeout '+str(budget)+' '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
