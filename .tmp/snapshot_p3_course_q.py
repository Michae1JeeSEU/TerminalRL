"""Read one bounded Q snapshot with the known server interpreter, no mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('case',choices=('Q_dense24_mixed','Q_substep_monitor_dense24_fixed'))
p.add_argument('--output',required=True);a=p.parse_args()
output=Path(a.output);assert not output.exists()
relative='p3_v25_neutral_course_v2_20261008/'+a.case
root='/home/user/training_outputs/terminal/'+relative
seed=895503 if a.case=='Q_dense24_mixed' else 895502
code='''import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__ROOT__);episode=root/('seed_'+str(__SEED__))
summary={'root':str(root),'files':{},'adjacent':{},'records':{}}
for suffix in ('.pid','.log','.exit'):
    path=Path(str(root)+suffix)
    summary['adjacent'][suffix]={'exists':path.exists(),'text':path.read_text()[-1500:] if path.exists() else None}
for path in [root/'protocol.json',root/'result.json',*[episode/name for name in
    ('progress.json','rollout_result.json','annotation_progress.json','annotation_result.json','rows.jsonl','steps.jsonl','selected.json','context_observations.jsonl')]]:
    relative=path.relative_to(root).as_posix()
    if not path.exists():continue
    data=path.read_bytes()
    summary['files'][relative]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    if path.suffix=='.json' and path.name!='selected.json':
        value=json.loads(data)
        summary['records'][relative]={key:value[key] for key in
            ('status','stage','error','blockers','ticks','rows','row_counts','selected','labelled','elapsed_seconds','outcome_counts','censored','terminal','unresolved') if key in value}
processes=[]
for line in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines():
    if str(root) in line and (' -m terminal_marl.' in line or 'timeout 27000 ' in line) and not ' -c ' in line:
        pid=line.strip().split()[0]
        try:cwd=os.readlink('/proc/'+pid+'/cwd')
        except OSError:cwd=None
        processes.append({'line':line.strip(),'cwd':cwd})
summary['processes']=processes
print(json.dumps(summary))
'''.replace('__ROOT__',repr(root)).replace('__SEED__',str(seed))
cmd='/home/user/envs/terminal_env/bin/python -c '+shlex.quote(code)
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server',cmd],capture_output=True,text=True)
assert result.returncode==0,(result.returncode,result.stderr)
report=json.loads(result.stdout)
local=Path('D:/7_codespace/training_outputs/terminal')/relative
for name,item in report['files'].items():
    path=local/name
    item['local_sha256']=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    item['synchronized']=item['sha256']==item['local_sha256']
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'output':str(output),'processes':report['processes'],'adjacent':report['adjacent'],
    'records':report['records'],'unsynchronized':[k for k,v in report['files'].items() if not v['synchronized']]}))
