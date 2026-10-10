from pathlib import Path
import subprocess

def git(*args,input=None):
    r=subprocess.run(['git',*args],input=input,capture_output=True)
    if r.returncode: raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
    return r.stdout
assert not git('diff','--cached','--name-only').strip(), 'STOP: nonempty index'
paths=['terminal_marl/envs/parallel_env.py','tests/test_p3_integrated_repair.py','docs/evidence/v4_2/p3_integrated_validation_20261003.md']
prefix=Path('.tmp/p3_validation_status_prefix_20261003.txt').read_text(encoding='utf8').strip()+'\n\n'
base=git('show','HEAD:docs/CURRENT_STATUS.md').decode('utf8');first,rest=base.split('\n',1)
assert prefix.splitlines()[0] not in base
body=first+'\n\n'+prefix+rest.lstrip('\r\n')
git('add','--',*paths)
blob=git('hash-object','-w','--stdin',input=body.encode('utf8')).decode().strip()
git('update-index','--cacheinfo',f'100644,{blob},docs/CURRENT_STATUS.md')
assert set(git('diff','--cached','--name-only').decode().splitlines())==set(paths)|{'docs/CURRENT_STATUS.md'}
print(git('diff','--cached','--check').decode())
print(git('diff','--cached','--stat').decode())
Path('.tmp/p3_validation_staged_tree.txt').write_bytes(git('write-tree').strip())
