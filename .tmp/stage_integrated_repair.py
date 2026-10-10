from pathlib import Path
import subprocess

def git(*args,input=None):
    result=subprocess.run(['git',*args],input=input,capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8',errors='replace'))
    return result.stdout
if git('diff','--cached','--name-only').strip():
    raise SystemExit('STOP: index already contains changes; primary must review')
files=Path('.tmp/p3_integrated_commit_files.txt').read_text(encoding='utf8').splitlines()
assert len(files)==20 and len(set(files))==20
prefix=Path('.tmp/p3_integrated_status_prefix_20261003.txt').read_text(encoding='utf8').strip()+'\n\n'
current=git('show','HEAD:docs/CURRENT_STATUS.md').decode('utf8')
if '## 2026-10-03 P3 统一实现修复（最新）' in current:
    raise SystemExit('STOP: latest block already in HEAD')
first,rest=current.split('\n',1)
content=first+'\n\n'+prefix+rest.lstrip('\r\n')
# README and technical_spec also received independent visualization edits
# during this task. Stage only the primary-owned P3 appended sections.
partials={'docs/CURRENT_STATUS.md':content}
for path,heading in [('docs/README.md','## 2026-10-03 当前修复入口'),
                     ('docs/technical_spec.md','## 2026-10-03 P3 repair interface')]:
    working=Path(path).read_text(encoding='utf8')
    base=git('show','HEAD:'+path).decode('utf8')
    assert heading not in base
    suffix=working[working.index(heading):]
    if path.endswith('technical_spec.md'):
        suffix=suffix.split('\n### 33.1 Local visualization console',1)[0]
    partials[path]=base.rstrip()+'\n\n\n'+suffix.rstrip()+'\n'
git('add','--',*[f for f in files if f not in partials])
for path,body in partials.items():
    blob=git('hash-object','-w','--stdin',input=body.encode('utf8')).decode().strip()
    git('update-index','--cacheinfo',f'100644,{blob},{path}')
actual=set(git('diff','--cached','--name-only').decode().splitlines())
assert actual==set(files)|{'docs/CURRENT_STATUS.md'},actual
print(git('diff','--cached','--stat').decode())
print(git('diff','--cached','--check').decode())
Path('.tmp/p3_staged_tree.txt').write_bytes(git('write-tree').strip())
print('STAGED_21_OWNED_FILES; no commit created')
