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
assert git('rev-parse', 'HEAD').decode().strip() == 'c0aa1c33c650bbfd7cabe58c15cff3cbae1b65fd'
assert not git('diff', '--cached', '--name-only').strip()
files = ['terminal_marl/core/polynomial_trajectory.py',
         'docs/evidence/v4_2/p3_numeric_bounds_server_validation_20261007.md',
         'docs/evidence/v4_2/p3_numeric_bounds_optimization_20261005.md']
assert hashlib.sha256(Path(files[0]).read_bytes()).hexdigest() == '603a2997df35bac03771bc9d69d99f1fed48c899cce0ff1e93acf75908fea1e6'
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
print(git('commit', '-m', 'perf: reduce exact quintic bounds overhead without changing trajectories').decode())
print('COMMIT='+git('rev-parse', 'HEAD').decode().strip())
print('INDEX='+repr(git('diff', '--cached', '--name-only').decode()))
print('REMOTES='+repr(git('remote', '-v').decode()))
