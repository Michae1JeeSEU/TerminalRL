import argparse
import json
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('case',choices=['smoke','train_A','train_B','train_C']);p.add_argument('--save',required=True);a=p.parse_args()
seed=892807 if a.case=='smoke' else 893301+['train_A','train_B','train_C'].index(a.case)
script='''import hashlib,json,os
from pathlib import Path
base=Path(BASE);seed=SEED
episode=base/f'seed_{seed}'
report={'case':base.name,'processes':[],'files':{}}
for directory in Path('/proc').iterdir():
    if not directory.name.isdecimal(): continue
    try:
        argv=(directory/'cmdline').read_bytes().decode().strip('\\0').split('\\0')
        if str(base) in argv or ('--query-entry' in argv and any(str(episode) in part for part in argv)):
            report['processes'].append({'pid':int(directory.name),'argv':argv,'cwd':os.readlink(directory/'cwd')})
    except (OSError,UnicodeError): pass
paths=[base/'protocol.json',base/'result.json',*[episode/name for name in ('progress.json','rollout_result.json','annotation_progress.json','annotation_result.json','rows.jsonl','steps.jsonl','selected.json','context_observations.jsonl','first_before.pkl')],*[Path(str(base)+name) for name in ('.pid','.log','.exit')]]
for path in paths:
    item={'path':str(path),'exists':path.exists()}
    if path.exists():
        data=path.read_bytes();item.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        if path.suffix=='.json':item['json']=json.loads(data)
        elif path.suffix in ('.log','.pid','.exit'):item['text']=data[-3000:].decode(errors='replace')
        elif path.suffix=='.jsonl':item['lines']=len(data.splitlines())
    report['files'][str(path.relative_to(base)) if path.is_relative_to(base) else path.name]=item
print(json.dumps(report,ensure_ascii=False))
'''.replace('BASE',repr('/home/user/training_outputs/terminal/p3_v25_forecast_train_20261008/'+a.case)).replace('SEED',str(seed))
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server',shlex.join(['/home/user/envs/terminal_env/bin/python','-c',script])],capture_output=True)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
data=json.loads(r.stdout)
import hashlib
for item in data['files'].values():
    local=Path(item['path'].replace('/home/user/training_outputs/terminal','D:/7_codespace/training_outputs/terminal'))
    if item['exists'] and local.exists():item['local_sha_equal']=hashlib.sha256(local.read_bytes()).hexdigest()==item['sha256']
out=Path(a.save);assert not out.exists();out.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(dict(saved=str(out),processes=data['processes'],files={k:{x:y for x,y in v.items() if x not in ('json','text')} for k,v in data['files'].items()})))
