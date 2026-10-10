"""One read-only snapshot of a named neutral-course learning/development output."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser()
p.add_argument('case',choices=['fit_initial','student_initial/smoke',
    'student_initial/flow6_mixed','student_initial/dense24_fixed','student_initial/dense24_mixed'])
p.add_argument('--save',required=True)
a=p.parse_args()
out=Path(a.save)
assert not out.exists()
base='/home/user/training_outputs/terminal/p3_v25_neutral_course_20261008/'+a.case
script='''import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
base=Path(BASE)
report={'case':str(base),'utc':datetime.now(timezone.utc).isoformat(),'processes':[],'files':{}}
for directory in Path('/proc').iterdir():
    if not directory.name.isdecimal():continue
    try:
        argv=(directory/'cmdline').read_bytes().decode().strip('\\0').split('\\0')
        if str(base) in argv:
            report['processes'].append({'pid':int(directory.name),'argv':argv,'cwd':os.readlink(directory/'cwd')})
    except (OSError,UnicodeError):pass
names=['protocol.json','result.json','inputs.json']
if base.name=='fit_initial':
    names += [f'seed_{seed}/{name}' for seed in (91901,91902,91903)
              for name in ('progress.json','result.json','diagnostic_final.pt','predictions.json')]
else:
    names += ['episode/'+name for name in ('protocol.json','progress.json','result.json','steps.pt','requests.jsonl','first_before.pkl')]
paths=[base/name for name in names]+[Path(str(base)+suffix) for suffix in ('.pid','.log','.exit')]
for path in paths:
    item={'path':str(path),'exists':path.exists()}
    if path.exists():
        data=path.read_bytes();item.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        if path.suffix=='.json' and path.name not in ('inputs.json','predictions.json'):
            item['json']=json.loads(data)
        elif path.suffix in ('.log','.pid','.exit'):
            item['text']=data[-5000:].decode(errors='replace')
        elif path.suffix=='.jsonl':item['lines']=len(data.splitlines())
    report['files'][str(path.relative_to(base)) if path.is_relative_to(base) else path.name]=item
print(json.dumps(report))
'''.replace('BASE',repr(base))
run=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server',
    shlex.join(['/home/user/envs/terminal_env/bin/python','-c',script])],capture_output=True)
if run.returncode:raise RuntimeError(run.stderr.decode(errors='replace'))
data=json.loads(run.stdout)
for item in data['files'].values():
    local=Path(item['path'].replace('/home/user/training_outputs/terminal','D:/7_codespace/training_outputs/terminal'))
    if item['exists'] and local.exists():
        item['local_sha_equal']=hashlib.sha256(local.read_bytes()).hexdigest()==item['sha256']
out.write_text(json.dumps(data,indent=2),encoding='utf8')
print(json.dumps({'saved':str(out),'utc':data['utc'],'processes':data['processes'],
                 'files':{k:{x:y for x,y in v.items() if x not in ('json','text')} for k,v in data['files'].items()}}))
