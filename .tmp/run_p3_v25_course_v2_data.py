"""Guarded fresh data round for the corrected nominal-curve contract."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('case',choices=('smoke','train_flow6_mixed','train_dense24_fixed','train_dense24_mixed'));a=p.parse_args()
base='/home/user/training_outputs/terminal/p3_v25_neutral_course_v2_20261008';out=base+'/'+a.case
python='/home/user/envs/terminal_env/bin/python';q=shlex.quote
files=['terminal_marl/training/p3_v25_course_data.py','terminal_marl/training/p3_v25_imitation_qualification.py',
 'terminal_marl/training/p3_v25_context_dataset.py','terminal_marl/core/trajectory_context_observation.py',
 'terminal_marl/scripts/collect_p3_v3_sparse_expert.py','terminal_marl/scripts/collect_p3_v3_i0.py',
 'terminal_marl/core/config.py','terminal_marl/core/priority.py','terminal_marl/core/merge_scheduler.py','terminal_marl/core/merge_gap_forecast.py',
 'terminal_marl/core/quintic_hull.py','terminal_marl/core/trajectory_time_parameterization.py','terminal_marl/core/trajectory_generator.py',
 'terminal_marl/core/safety_filter.py','terminal_marl/envs/parallel_env.py','terminal_marl/configs/p3_integrated_repair.py',
 'terminal_marl/configs/p3_v25_course.py','docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files)
cases=['train_flow6_mixed','train_dense24_fixed','train_dense24_mixed'];smoke=a.case=='smoke'
index=1 if smoke else cases.index(a.case)+1;seed=892863 if smoke else 893510+index
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -']
lines+=['test ! -e '+q(out+suffix) for suffix in ('','.pid','.log','.exit')]
lines.append('test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python -u (-m terminal_marl.(training.p3_v25_|scripts.collect_p3_v3_)|.tmp/probe_p3_nominal_curve_episode.py)\' || true)"')
guard="from terminal_marl.training.p3_v25_course_data import PROTOCOL,TRAIN_CASES,SMOKE_SEED;assert PROTOCOL=='p3_v25_neutral_course_data_20261008_v2' and SMOKE_SEED==892863 and [r[0] for r in TRAIN_CASES]==[893511,893512,893513]"
lines.append(q(python)+' -c '+q(guard))
prior='/home/user/training_outputs/terminal/p3_v25_nominal_curve_validation_20261008/source_dense_fixed/result.json'
guard='import json;from pathlib import Path;r=json.loads(Path('+repr(prior)+').read_text());assert r["status"]=="COUNTERFACTUAL_COMPLETE" and not r["blockers"]'
lines.append(q(python)+' -c '+q(guard))
for previous in ([] if smoke else ['smoke']+cases[:index-1]):
    status='SMOKE_OK' if previous=='smoke' else 'COLLECTION_OK'
    guard='import json;from pathlib import Path;r=json.loads(Path('+repr(base+'/'+previous+'/result.json')+').read_text());assert r["status"]=='+repr(status)+' and r["error"] is None and not r["blockers"]'
    lines.append(q(python)+' -c '+q(guard))
inventory='''import json
from pathlib import Path
matches=[]
for path in Path('/home/user/training_outputs/terminal').rglob('protocol.json'):
    try: value=json.loads(path.read_text())
    except (OSError,ValueError):continue
    if value.get('seed')==SEED or value.get('env_seed')==SEED:matches.append(str(path))
print({'seed':SEED,'prior_protocols':matches})
assert not matches
'''.replace('SEED',str(seed))
lines.append(q(python)+' -c '+q(inventory))
command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-u','-m',
 'terminal_marl.training.p3_v25_course_data','--index',str(index),'--output',out,
 '--annotation-wall-seconds','21600']+(['--smoke'] if smoke else [])
lines+=['mkdir -p '+q(base),'set +e','timeout '+('600' if smoke else '27000')+' '+shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
 'task_pid=$!','printf \'%s\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
 'printf \'%s\n\' "$code" > '+q(out+'.exit'),'cat '+q(out+'.log'),'exit "$code"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
