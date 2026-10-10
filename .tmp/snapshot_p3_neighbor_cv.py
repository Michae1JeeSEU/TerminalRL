import argparse,json,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
script='''import json,os,hashlib,time
from pathlib import Path
root=Path('/home/user/training_outputs/terminal/p3_neighbor_relation_cv_20261009')
r={'time':time.time(),'files':{},'processes':[]}
for name in ['six_cycle.pid','six_cycle.wrapper.pid','six_cycle.log','six_cycle.exit','preflight.exit','six_cycle/protocol.json','six_cycle/result.json']:
 p=root/name
 if p.exists():
  data=p.read_bytes();r['files'][name]={'sha256':hashlib.sha256(data).hexdigest(),'size':len(data),'mtime':p.stat().st_mtime,'text':data.decode(errors='replace')[-200000:]}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode().strip('\\0').split('\\0');argv=' '.join(args)
  if args[:3]==['/home/user/envs/terminal_env/bin/python','-u','-m'] and args[3:4]==['terminal_marl.scripts.diagnose_p3_intent_neighbor']:
   r['processes'].append({'pid':int(p.name),'argv':argv,'cwd':os.readlink(p/'cwd'),'stat':(p/'stat').read_text()})
 except (OSError,UnicodeError):pass
print(json.dumps(r))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server',
 '/home/user/envs/terminal_env/bin/python -c '+shlex.quote(script)],capture_output=True,text=True)
assert r.returncode==0,r.stderr
value=json.loads(r.stdout);a.output.write_text(json.dumps(value,indent=2)+'\n')
report=value['files'].get('six_cycle/result.json',{}).get('text')
result=json.loads(report) if report else None
print(json.dumps({'processes':value['processes'],'files':list(value['files']),
 'status':None if result is None else result['status'],'error':None if result is None else result['error'],
 'folds':None if result is None else [{'seed':f['held_seed'],'arm':f['arm'],'last_update':f['curve'][-1]['update'] if f['curve'] else None,'seconds':f.get('elapsed_seconds')} for f in result['folds']]},indent=2))
