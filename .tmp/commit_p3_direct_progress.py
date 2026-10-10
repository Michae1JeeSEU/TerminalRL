from pathlib import Path
import subprocess

def git(*args,input=None):
 r=subprocess.run(['git',*args],input=input,capture_output=True)
 if r.returncode: raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 return r.stdout

assert not git('diff','--cached','--name-only').strip(),'STOP: nonempty index'
paths=['docs/evidence/v4_2/p3_direct_c2_repair_validation_20261003.md']
prefix=Path('.tmp/p3_validation_status_prefix_20261003.txt').read_text(encoding='utf8').strip()+'\n\n'
base=git('show','HEAD:docs/CURRENT_STATUS.md').decode('utf8')
a=base.index('## 2026-10-03 P3修复效果逐步验证（最新）')
b=base.index('## 2026-10-03 P3 统一实现修复（最新）',a)
body=base[:a]+prefix+base[b:]
print(git('status','--short').decode('utf8',errors='replace'))
git('add','--',*paths)
blob=git('hash-object','-w','--stdin',input=body.encode('utf8')).decode().strip()
git('update-index','--cacheinfo',f'100644,{blob},docs/CURRENT_STATUS.md')
assert set(git('diff','--cached','--name-only').decode().splitlines())==set(paths+['docs/CURRENT_STATUS.md'])
print(git('diff','--cached','--check').decode())
print(git('diff','--cached','--stat').decode())
print(git('commit','-m','docs: conclude P3 direct landing validation and qualification boundary').decode())
print('COMMIT='+git('rev-parse','HEAD').decode().strip())
print('REMOTES='+repr(git('remote','-v').decode()))
print('INDEX='+repr(git('diff','--cached','--name-only').decode()))
