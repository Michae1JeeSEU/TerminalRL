"""Prepared once-only evaluation of three frozen finals on the three Q cycles."""
import hashlib
from pathlib import Path
import shlex
import subprocess

base='/home/user/training_outputs/terminal/p3_v25_neutral_course_v2_20261008'
out=base+'/qualification_initial'
python='/home/user/envs/terminal_env/bin/python';q=shlex.quote
cases=('flow6_mixed','dense24_fixed','dense24_mixed')
files=['terminal_marl/training/p3_v25_task_imitation.py','terminal_marl/training/p3_v25_course_data.py',
       'terminal_marl/training/p3_v25_imitation_qualification.py','terminal_marl/training/p3_v25_context_dataset.py',
       'terminal_marl/training/p3_v25_set_learnability.py','terminal_marl/algo/trajectory_set_actor.py',
       'terminal_marl/configs/p3_v25_course.py',
       'docs/evidence/v4_2/p3_v25_nominal_curve_fresh_training_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -']
lines+=['test ! -e '+q(out+suffix) for suffix in ('','.pid','.log','.exit')]
lines.append('test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u -m terminal_marl.(training.p3_v25_|scripts.collect_p3_v3_)\' || true)"')
guard='''from pathlib import Path
from terminal_marl.training.p3_v25_task_imitation import frozen_models
from terminal_marl.training.p3_v25_course_data import Q_CASES,validated_case
base=Path(BASE)
fit=frozen_models(base/'fit_initial')
assert fit['checkpoints'][0]['sha256']=='16d3ee800263023386341cb92a6335bb63dd487a346181a0275bf4c3ecb63600'
for seed,scenario in Q_CASES:
    directory=base/('Q_substep_monitor_dense24_fixed' if scenario=='dense24_fixed' else 'Q_'+scenario)
    validated_case(directory,seed,scenario,'Q')
    assert not (directory/'evaluation_consumed.json').exists()
print({'Q_complete_unscored':True,'frozen_candidates':fit['checkpoints']})
'''.replace('BASE',repr(base))
lines.append(q(python)+' -c '+q(guard))
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u','-m',
 'terminal_marl.training.p3_v25_task_imitation','--mode','evaluate','--fit-directory',base+'/fit_initial',
 '--q-directories',*[base+('/Q_substep_monitor_dense24_fixed' if s=='dense24_fixed' else '/Q_'+s) for s in cases],'--output',out]
lines+=['set +e','timeout 900 '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
