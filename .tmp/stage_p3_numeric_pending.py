from pathlib import Path
import subprocess


def git(*args, input=None):
    result = subprocess.run(['git', *args], input=input, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8', errors='replace'))
    return result.stdout


assert git('rev-parse', 'HEAD').decode().strip() == 'f01477a4d2ac4502ef1e8cb059735237948e18c9'
assert not git('diff', '--cached', '--name-only').strip()
assert not git('diff', '--', 'terminal_marl/core/polynomial_trajectory.py').strip()
patch = 'docs/evidence/v4_2/p3_numeric_bounds_candidate_20261005.patch'
git('apply', '--check', '--unidiff-zero', patch)
files = [patch, 'docs/evidence/v4_2/p3_numeric_bounds_optimization_20261005.md',
         'tests/test_polynomial_trajectory_numeric_equivalence.py']
start = '## 2026-10-03 P3修复效果逐步验证（最新）'
end = '## 2026-10-03 P3 统一实现修复（最新）'
local = Path('docs/CURRENT_STATUS.md').read_text(encoding='utf8')
base = git('show', 'HEAD:docs/CURRENT_STATUS.md').decode('utf8')
assert local.count(start) == base.count(start) == 1
assert local.count(end) == base.count(end) == 1
body = base[:base.index(start)] + local[local.index(start):local.index(end)] + base[base.index(end):]
git('add', '--', *files)
blob = git('hash-object', '-w', '--stdin', input=body.encode('utf8')).decode().strip()
git('update-index', '--cacheinfo', f'100644,{blob},docs/CURRENT_STATUS.md')
assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == {*files, 'docs/CURRENT_STATUS.md'}
print(git('diff', '--cached', '--check').decode())
print(git('diff', '--cached', '--stat').decode())
print(git('diff', '--cached', '--', 'docs/CURRENT_STATUS.md').decode())
