"""Bounded max-candidate-gate comparison launcher; sources arrive by Syncthing."""
import argparse,ast,hashlib,shlex,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('stage',choices=('check','smoke','launch'));a=p.parse_args();q=shlex.quote
base='/home/user/training_outputs/terminal/p3_neighbor_gate_max_20261009'
python='/home/user/envs/terminal_env/bin/python'
tests=['tests/test_intent_neighbor_actor.py','tests/test_p3_neighbor_witness_data.py',
       'tests/test_p3_neighbor_witness_dataset.py','tests/test_ordered_proposal_actor.py',
       'tests/test_p3_v26_train_dataset.py']
tree=ast.parse(Path('.tmp/run_p3_v26_resources.py').read_text())
node=next(x.value for x in tree.body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='files')
files=ast.literal_eval(node.left)+tests+[
 'terminal_marl/algo/intent_neighbor_actor.py','terminal_marl/algo/ordered_proposal_actor.py',
 'terminal_marl/algo/joint_ppo_v4_2.py','terminal_marl/scripts/diagnose_p3_intent_neighbor.py',
 'terminal_marl/training/p3_neighbor_witness_data.py','terminal_marl/training/p3_neighbor_witness_dataset.py',
 'terminal_marl/training/p3_v26_train_dataset.py','terminal_marl/training/p3_v25_set_learnability.py',
 'terminal_marl/training/p3_v3_bc_learnability.py',
 'docs/evidence/v4_2/p3_neighbor_observability_repair_20261009.md',
 'docs/evidence/v4_2/p3_neighbor_gate_calibration_20261009.md']
files+=['.tmp/p3_neighbor_witness_train_20261009/'+x for x in ('protocol.json','result.json','witnesses.jsonl')]
checks=''.join(hashlib.sha256(Path(f).read_bytes()).hexdigest()+'  '+f+'\n' for f in dict.fromkeys(files))
lines=['set -eu','cd /home/user/projects/terminal','printf %s '+q(checks)+' | sha256sum -c -',
       'test -z "$(pgrep -f \'^/home/user/envs/terminal_env/bin/python.*terminal_marl\' || true)"']

if a.stage=='check':
    lines+=['test ! -e '+q(base),'mkdir -p '+q(base),'set +e',
      'env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '+q(python)+' -m pytest '+shlex.join(tests)+' -q > '+q(base+'/preflight.log')+' 2>&1',
      'code=$?','printf \'%s\\n\' "$code" > '+q(base+'/preflight.exit'),
      'cat '+q(base+'/preflight.log'),'exit "$code"']
else:
    smoke=base+'/smoke_synthetic'
    lines+=['test -d '+q(base),'test "$(cat '+q(base+'/preflight.exit')+')" = 0']
    if a.stage=='smoke':
        out=smoke
        lines+=['test ! -e '+q(out+s) for s in ('','.pid','.wrapper.pid','.log','.exit')]
        command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
            'timeout','1200',python,'-u','-m','terminal_marl.scripts.diagnose_p3_intent_neighbor',
            '--smoke','--max-candidate-gate','--output',out]
    else:
        lines+=['test "$(cat '+q(smoke+'.exit')+')" = 0']
        smoke_check='''import json
from pathlib import Path
r=json.loads(Path(%r).read_text())
assert r['status']=='COMPLETE'
assert r['protocol']['gate_mode']=='max_candidate'
assert r['protocol']['arms']==['native_pair_auxiliary']
assert len(r['folds'])==2
assert r['protocol']['marks']==[0,1,2]
assert all(len(f['curve'])==3 and f['curve'][-1]['update']==2 for f in r['folds'])
print('SMOKE_OK')
''' % (smoke+'/result.json')
        lines+=['env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '+q(python)+' -c '+q(smoke_check)]
        out=base+'/six_cycle'
        lines+=['test ! -e '+q(out+s) for s in ('','.pid','.wrapper.pid','.log','.exit')]
        root='/home/user/training_outputs/terminal/'
        original=root+'p3_v25_neutral_course_v2_20261008/'
        extra=root+'p3_v26_resources_20261009/'
        command=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
            'timeout','7800',python,'-u','-m','terminal_marl.scripts.diagnose_p3_intent_neighbor',
            '--train-directories',*[original+s for s in ('train_flow6_mixed','train_dense24_fixed','train_dense24_mixed')],
            '--additional-train-directories',*[extra+f'train_{i:02d}' for i in (1,2,3)],
            '--witness-directory','/home/user/projects/terminal/.tmp/p3_neighbor_witness_train_20261009',
            '--output',out,'--max-candidate-gate']
    body='\n'.join(['cd /home/user/projects/terminal',shlex.join(command)+' > '+q(out+'.log')+' 2>&1 &',
       'task_pid=$!','printf \'%s\\n\' "$task_pid" > '+q(out+'.pid'),'wait "$task_pid"','code=$?',
       'printf \'%s\\n\' "$code" > '+q(out+'.exit'),'exit "$code"'])
    lines+=['nohup sh -c '+q(body)+' > /dev/null 2>&1 < /dev/null &',
      'wrapper_pid=$!','printf \'%s\\n\' "$wrapper_pid" > '+q(out+'.wrapper.pid'),
      'printf \'wrapper_pid=%s\\n\' "$wrapper_pid"']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)]).returncode)
