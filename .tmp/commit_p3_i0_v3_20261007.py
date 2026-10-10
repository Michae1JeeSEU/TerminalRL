from pathlib import Path
import hashlib
import subprocess
import sys
sys.stdout.reconfigure(encoding='utf-8')
def git(*args, input=None):
    p = subprocess.run(['git', *args], input=input, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode('utf8', errors='replace'))
    return p.stdout
assert git('rev-parse', 'HEAD').decode().strip() == '0804d77f7b53958eb7cd29f5f99d3aa4db4920ac'
assert not git('diff', '--cached', '--name-only').strip()
files = ['terminal_marl/scripts/collect_p3_v3_i0.py', 'tests/test_p3_v3_i0.py', 'docs/evidence/v4_2/p3_v3_i0_v3_numeric_20261007.md']
assert hashlib.sha256(Path(files[0]).read_bytes()).hexdigest() == 'd267c13910582312634caadbd9bcab52db9fe5b8883f986b1a0034fb6f9e9247'
git('diff', '--check', '--', *files, 'docs/CURRENT_STATUS.md')
start, end = '## 2026-10-03 P3修复效果逐步验证（最新）', '## 2026-10-03 P3 统一实现修复（最新）'
local = Path('docs/CURRENT_STATUS.md').read_text(encoding='utf8')
base = git('show', 'HEAD:docs/CURRENT_STATUS.md').decode('utf8')
assert local.count(start) == base.count(start) == local.count(end) == base.count(end) == 1
body = base[:base.index(start)] + local[local.index(start):local.index(end)] + base[base.index(end):]
git('add', '--', *files)
blob = git('hash-object', '-w', '--stdin', input=body.encode()).decode().strip()
git('update-index', '--cacheinfo', f'100644,{blob},docs/CURRENT_STATUS.md')
assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == {*files, 'docs/CURRENT_STATUS.md'}
git('diff', '--cached', '--check')
print(git('diff', '--cached', '--stat').decode())
print(git('commit', '-m', 'feat: freeze one fresh mixed I0 collection after numeric speedup').decode())
print('COMMIT='+git('rev-parse', 'HEAD').decode().strip())
print('INDEX='+repr(git('diff', '--cached', '--name-only').decode()))
print('REMOTES='+repr(git('remote', '-v').decode()))
