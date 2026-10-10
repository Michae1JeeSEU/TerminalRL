import argparse
import json
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('case',choices=['smoke','A_det','A_sto','B_det','B_sto','C_det','C_sto'])
p.add_argument('--save',type=Path,required=True);a=p.parse_args()
script='''import hashlib,json,os
from pathlib import Path
case=CASE
base=Path('/home/user/training_outputs/terminal/p3_v25_student_dev_20261008')/case
report={'case':case,'processes':[],'files':{}}
for directory in Path('/proc').iterdir():
    if not directory.name.isdecimal(): continue
    try:
        argv=(directory/'cmdline').read_bytes().decode().strip('\\0').split('\\0')
        if 'terminal_marl.scripts.evaluate_p3_v25_student' in argv and str(base) in argv:
            report['processes'].append({'pid':int(directory.name),'argv':argv,'cwd':os.readlink(directory/'cwd')})
    except (OSError,UnicodeError): pass
for name in ('protocol.json','progress.json','result.json','steps.jsonl','first_before.pkl','.pid','.log','.exit'):
    path=Path(str(base)+name) if name.startswith('.') else base/name
    item={'path':str(path),'exists':path.exists()}
    if path.exists():
        data=path.read_bytes();item.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        if name.endswith('.json'): item['json']=json.loads(data)
        elif name.startswith('.'): item['text']=data[-3000:].decode(errors='replace')
        elif name=='steps.jsonl': item['lines']=len(data.splitlines())
    report['files'][name]=item
print(json.dumps(report,ensure_ascii=False))
'''.replace('case=CASE','case='+repr(a.case))
command=shlex.join(['/home/user/envs/terminal_env/bin/python','-c',script])
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server',command],capture_output=True)
if result.returncode:
    raise RuntimeError(result.stderr.decode(errors='replace'))
data=json.loads(result.stdout)
for item in data['files'].values():
    local=Path(item['path'].replace('/home/user/training_outputs/terminal','D:/7_codespace/training_outputs/terminal'))
    if item['exists'] and local.exists():
        import hashlib
        item['local_sha_equal']=hashlib.sha256(local.read_bytes()).hexdigest()==item['sha256']
a.save.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
print(a.save)
print(json.dumps({'processes':data['processes'],'files':{k:{x:y for x,y in v.items() if x not in ('json','text')} for k,v in data['files'].items()}},ensure_ascii=False))
