import json,shlex,subprocess
script=r'''import json,os,signal,time
from pathlib import Path
base=Path('/home/user/training_outputs/terminal/p3_v25_task_20261008/train_dense24')
report=json.loads((base/'result.json').read_text())
assert report['status']=='ANNOTATING' and report['rollout']['status']=='COMPLETED'
assert report['seed']==893402 and report['rollout']['terminal']
parent=Path('/proc/2819670')
args=(parent/'cmdline').read_bytes().decode().strip('\0').split('\0')
assert 'terminal_marl.training.p3_v25_task_data' in args and str(base) in args
assert os.readlink(parent/'cwd')=='/home/user/projects/terminal'
children=[]
for path in Path('/proc').iterdir():
    if not path.name.isdecimal():continue
    try:
        argv=(path/'cmdline').read_bytes().decode().strip('\0').split('\0')
        if '--query-entry' in argv and any(str(base/'seed_893402') in a for a in argv):
            assert os.readlink(path/'cwd')=='/home/user/projects/terminal'
            ppid=int(next(s.split()[1] for s in (path/'status').read_text().splitlines() if s.startswith('PPid:')))
            assert ppid==2819670
            children.append((int(path.name),argv))
    except (FileNotFoundError,ProcessLookupError):pass
stop=dict(reason='Primary stopped superseded priority-enabled annotation; current research course is task-neutral. Original incomplete files retained; not a collection pass.',
    seed=893402,root_status_before=report['status'],rollout_outcomes=report['rollout']['outcome_counts'],
    completed_label_files=len(list((base/'seed_893402/labels').glob('window_*.json'))),
    parent_pid=2819670,parent_argv=args,children=children,signal='SIGTERM',time_unix=time.time())
with Path(str(base)+'.stopped.json').open('x') as stream:json.dump(stop,stream,indent=2)
for pid,_ in children:
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
os.kill(2819670,signal.SIGTERM)
print(json.dumps(stop))
'''
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server',shlex.join(['/home/user/envs/terminal_env/bin/python','-c',script])]).returncode)
