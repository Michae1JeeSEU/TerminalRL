from pathlib import Path
import subprocess

def git(*args,input=None):
 r=subprocess.run(['git',*args],input=input,capture_output=True)
 if r.returncode: raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 return r.stdout
assert not git('diff','--cached','--name-only').strip(),'STOP: nonemptyindex'
prefix=Path('.tmp/p3_validation_status_prefix_20261003.txt').read_text(encoding='utf8').strip()+'\n\n'
base=git('show','HEAD:docs/CURRENT_STATUS.md').decode('utf8')
a=base.index('## 2026-10-03 P3修复效果逐步验证（最新）')
b=base.index('## 2026-10-03 P3 统一实现修复（最新）',a)
body=base[:a]+prefix+base[b:]
git('add','--','docs/evidence/v4_2/p3_integrated_validation_20261003.md','docs/evidence/v4_2/p3_inner_exit_blocker_findings_20261003.md','terminal_marl/scripts/diagnose_p3_inner_exit.py','tests/test_p3_integrated_repair.py')
blob=git('hash-object','-w','--stdin',input=body.encode('utf8')).decode().strip()
git('update-index','--cacheinfo',f'100644,{blob},docs/CURRENT_STATUS.md')
assert set(git('diff','--cached','--name-only').decode().splitlines())=={'docs/CURRENT_STATUS.md','docs/evidence/v4_2/p3_integrated_validation_20261003.md','docs/evidence/v4_2/p3_inner_exit_blocker_findings_20261003.md','terminal_marl/scripts/diagnose_p3_inner_exit.py','tests/test_p3_integrated_repair.py'}
print(git('diff','--cached','--check').decode());print(git('diff','--cached','--stat').decode())
print(git('commit','-m','feat: diagnose P3 inner-exit conflict and viable route alternative').decode())
print('COMMIT='+git('rev-parse','HEAD').decode().strip());print('REMOTES='+repr(git('remote','-v').decode()));print('INDEX='+repr(git('diff','--cached','--name-only').decode()))
