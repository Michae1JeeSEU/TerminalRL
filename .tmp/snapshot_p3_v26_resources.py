"""One read-only remote observation; does not poll, start, stop or alter jobs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('case',choices=('smoke','train_01','train_02','train_03'))
p.add_argument('--output',type=Path,required=True)
p.add_argument('--supplemental',action='store_true');args=p.parse_args()
assert not args.output.exists()
remote='''import datetime,hashlib,json,os,subprocess
from pathlib import Path
base=Path('/home/user/training_outputs/terminal/p3_v26_resources_20261009')
case=CASE;root=base/case
report={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case':case,'adjacent':{},'files':{},'processes':[]}
def file(path,content=False):
    if not path.exists():return {'exists':False}
    data=path.read_bytes();result={'exists':True,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mtime':path.stat().st_mtime}
    if content:
        try:result['content']=json.loads(data)
        except ValueError:result['content']=data.decode(errors='replace')[-5000:]
    return result
for suffix in ('.pid','.wrapper.pid','.exit','.log'):
    report['adjacent'][suffix]=file(base/(case+suffix),True)
for name in ('protocol.json','result.json'):
    report['files'][name]=file(root/name,True)
protocol=report['files']['protocol.json'].get('content')
if isinstance(protocol,dict):
    episode=root/f"seed_{protocol['seed']}"
    for name in ('progress.json','rollout_result.json','annotation_progress.json','annotation_result.json'):
        value=file(episode/name,True)
        if name=='annotation_result.json' and isinstance(value.get('content'),dict):
            value['content']={k:v for k,v in value['content'].items() if k!='entries'}
        report['files'][name]=value
    for name in ('rows.jsonl','steps.jsonl','context_observations.jsonl','resource_context_observations.jsonl'):
        value=file(episode/name)
        if value['exists']:
            with (episode/name).open('rb') as stream:value['lines']=sum(1 for _ in stream)
        report['files'][name]=value
    selected=file(episode/'selected.json')
    if selected['exists']:selected['entries']=len(json.loads((episode/'selected.json').read_text()))
    report['files']['selected.json']=selected
    report['label_files']=len(list((episode/'labels').glob('window_*.json')))
for line in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]:
    values=line.strip().split(None,2)
    if len(values)!=3:continue
    pid,ppid,command=values
    if command.startswith('/home/user/envs/terminal_env/bin/python -u -m terminal_marl.'):
        try:cwd=os.readlink('/proc/'+pid+'/cwd')
        except OSError:cwd=None
        report['processes'].append({'pid':int(pid),'ppid':int(ppid),'argv':command,'cwd':cwd})
print(json.dumps(report,allow_nan=False))
'''.replace('CASE',repr(args.case))
if args.supplemental:remote=remote.replace('p3_v26_resources_20261009','p3_v26_supplemental_20261009')
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server',
 '/home/user/envs/terminal_env/bin/python -c '+shlex.quote(remote)],capture_output=True,text=True)
assert result.returncode==0,result.stderr
record=json.loads(result.stdout)
args.output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps({'output':str(args.output),'time_utc':record['time_utc'],
 'status':record['files']['result.json'].get('content',{}).get('status'),
 'processes':record['processes']}))
