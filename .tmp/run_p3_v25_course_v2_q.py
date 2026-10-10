"""Launch one declared Q collection only after all new student cycles finish."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser()
p.add_argument('case',choices=('flow6_mixed','dense24_fixed','dense24_mixed'))
a=p.parse_args()
base='/home/user/training_outputs/terminal/p3_v25_neutral_course_v2_20261008'
cases=('flow6_mixed','dense24_fixed','dense24_mixed')
index=cases.index(a.case)+1;seed=895500+index
out=base+'/Q_'+a.case
if a.case=='dense24_fixed':
    out=base+'/Q_substep_monitor_dense24_fixed'
python='/home/user/envs/terminal_env/bin/python';q=shlex.quote
files=[
 'terminal_marl/training/p3_v25_course_data.py','terminal_marl/training/p3_v25_task_imitation.py',
 'terminal_marl/training/p3_v25_task_student.py','terminal_marl/training/p3_v25_imitation_qualification.py',
 'terminal_marl/training/p3_v25_context_dataset.py','terminal_marl/core/trajectory_context_observation.py',
 'terminal_marl/scripts/collect_p3_v3_sparse_expert.py','terminal_marl/scripts/collect_p3_v3_i0.py',
 'terminal_marl/core/config.py','terminal_marl/core/priority.py','terminal_marl/core/merge_scheduler.py',
 'terminal_marl/core/merge_gap_forecast.py','terminal_marl/core/quintic_hull.py',
 'terminal_marl/core/trajectory_time_parameterization.py','terminal_marl/core/trajectory_generator.py',
 'terminal_marl/core/safety_filter.py','terminal_marl/core/runtime_traffic_safety.py',
 'terminal_marl/core/runtime_separation.py','terminal_marl/envs/parallel_env.py',
 'terminal_marl/configs/p3_integrated_repair.py','terminal_marl/configs/p3_v25_course.py',
 'docs/evidence/v4_2/p3_v25_nominal_curve_fresh_training_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -']
lines+=['test ! -e '+q(out+suffix) for suffix in ('','.pid','.log','.exit')]
lines.append('test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.(training.p3_v25_|scripts.collect_p3_v3_)\' || true)"')
guard='''import json
from pathlib import Path
from terminal_marl.training.p3_v25_task_imitation import frozen_models
from terminal_marl.training.p3_v25_task_student import blockers
from terminal_marl.training.p3_v25_course_data import PROTOCOL,Q_CASES
from terminal_marl.configs.p3_v25_course import PROFILE,SCENARIOS
base=Path(BASE)
assert PROTOCOL=='p3_v25_neutral_course_data_20261008_v2'
assert [c[0] for c in Q_CASES]==[895501,895502,895503]
fit=frozen_models(base/'fit_initial')
assert fit['checkpoints'][0]['sha256']=='16d3ee800263023386341cb92a6335bb63dd487a346181a0275bf4c3ecb63600'
for scenario in SCENARIOS:
    root=base/('student_runtime_refined' if scenario=='dense24_mixed' else 'student_initial')/scenario
    r=json.loads((root/'result.json').read_text());p=json.loads((root/'protocol.json').read_text())
    assert r['profile']==p['profile']==PROFILE and r['source']==p['source']==fit['checkpoints'][0]
    assert r['status']=='DEVELOPMENT_COMPLETE' and not r['blockers'] and not blockers(r['rollout'])
    assert not p['deterministic'] and not p['smoke'] and not p['teacher_enabled']
print({'frozen_candidates':fit['checkpoints'],'Q_reset_yet':False})
'''.replace('BASE',repr(base))
lines.append(q(python)+' -c '+q(guard))
for previous in cases[:index-1]:
    previous_path=base+('/Q_substep_monitor_dense24_fixed' if previous=='dense24_fixed' else '/Q_'+previous)
    guard='import json;from pathlib import Path;r=json.loads(Path('+repr(previous_path+'/result.json')+').read_text());assert r["status"]=="COLLECTION_OK" and r["error"] is None and not r["blockers"] and r["partition"]=="Q"'
    lines.append(q(python)+' -c '+q(guard))
inventory='''import json,hashlib
from pathlib import Path
matches=[]
for path in Path('/home/user/training_outputs/terminal').rglob('protocol.json'):
    try:value=json.loads(path.read_text())
    except (OSError,ValueError):continue
    if value.get('seed')==SEED or value.get('env_seed')==SEED:matches.append(str(path))
print({'seed':SEED,'prior_protocols':matches})
if SEED==895502:
    original=Path(BASE)/'Q_dense24_fixed'
    assert matches==[str(original/'protocol.json')], matches
    assert hashlib.sha256((original/'result.json').read_bytes()).hexdigest()=='d86f3ec21a314a5777b144a1c3d6ba82da807fff887a115ce0c331de1e069b34'
    assert not (original/'seed_895502'/'annotation_result.json').exists()
    assert not (original/'evaluation_consumed.json').exists()
else:
    assert not matches
'''.replace('SEED',str(seed)).replace('BASE',repr(base))
lines.append(q(python)+' -c '+q(inventory))
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u','-m',
 'terminal_marl.training.p3_v25_course_data','--partition','Q','--index',str(index),
 '--annotation-wall-seconds','21600','--output',out]
lines+=['set +e','timeout 27000 '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
