"""Launch only the approved complete annotation of saved train893502."""
import hashlib
from pathlib import Path
import shlex
import subprocess

q=shlex.quote;python='/home/user/envs/terminal_env/bin/python'
base='/home/user/training_outputs/terminal/p3_v25_neutral_course_20261008'
source=base+'/train_dense24_fixed';out=base+'/train_dense24_fixed_annotation_v2'
files=['terminal_marl/training/p3_v25_annotation_repair.py','terminal_marl/training/p3_v25_course_data.py',
 'terminal_marl/training/p3_v25_imitation_qualification.py','terminal_marl/training/p3_v25_context_dataset.py',
 'terminal_marl/scripts/collect_p3_v3_sparse_expert.py','terminal_marl/core/quintic_hull.py',
 'terminal_marl/core/trajectory_generator.py','terminal_marl/core/trajectory_time_parameterization.py',
 'terminal_marl/core/polynomial_trajectory.py','terminal_marl/core/safety_filter.py','terminal_marl/envs/parallel_env.py',
 'terminal_marl/configs/p3_v25_course.py','docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u','-m',
 'terminal_marl.training.p3_v25_annotation_repair','--source-directory',source,'--output',out]
proof="""import hashlib,json
from pathlib import Path
root=Path(SOURCE)
assert hashlib.sha256((root/'result.json').read_bytes()).hexdigest()=='943e18142b78d2c5f64c4299334724e4b2b257f78f4c6eaeff4cf18de53c7a8b'
r=json.loads((root/'result.json').read_text())
assert r['status']=='INCOMPLETE' and r['blockers']==['incomplete_annotation']
assert r['annotation']['selected']==500 and r['annotation']['labelled']==201
assert r['annotation']['status']=='CENSORED'
for slot in (0,129):
 p=Path('/home/user/training_outputs/terminal/p3_v25_hull_production_query_20261008')/f'query_{slot}/result.json'
 assert json.loads(p.read_text())['status']=='QUERY_EXACT'
""".replace('SOURCE',repr(source))
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
 'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.(training.p3_v25_annotation_repair|training.p3_v25_course_data|scripts.collect_p3_v3_)\' || true)"',
 q(python)+' -c '+q(proof)]
for suffix in ('','.pid','.log','.exit'):lines.append('test ! -e '+q(out+suffix))
lines+=['set +e','timeout 24000 '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
