from pathlib import Path
import subprocess
import sys
sys.stdout.reconfigure(encoding='utf-8')


def git(*args, input=None):
    result = subprocess.run(['git', *args], input=input, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8', errors='replace'))
    return result.stdout


assert git('rev-parse', 'HEAD').decode().strip() == '0edc99111eeab4b7c99acfaf98c1b5a9629f8483'
assert not git('diff', '--cached', '--name-only').strip()
assert not git('diff', '--', 'terminal_marl/core/polynomial_trajectory.py').strip()
report = 'docs/evidence/v4_2/p3_numeric_bounds_optimization_20261005.md'
start = '## 2026-10-03 P3修复效果逐步验证（最新）'
end = '## 2026-10-03 P3 统一实现修复（最新）'
local = Path('docs/CURRENT_STATUS.md').read_text(encoding='utf8')
base = git('show', 'HEAD:docs/CURRENT_STATUS.md').decode('utf8')
assert local.count(start) == base.count(start) == 1
assert local.count(end) == base.count(end) == 1
body = base[:base.index(start)] + local[local.index(start):local.index(end)] + base[base.index(end):]
git('diff', '--check', '--', report, 'docs/CURRENT_STATUS.md')
git('add', '--', report)
blob = git('hash-object', '-w', '--stdin', input=body.encode()).decode().strip()
git('update-index', '--cacheinfo', f'100644,{blob},docs/CURRENT_STATUS.md')
assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == {report, 'docs/CURRENT_STATUS.md'}
git('diff', '--cached', '--check')
print(git('diff', '--cached', '--stat').decode())
print(git('commit', '-m', 'docs: record SSH handshake blocker on validation resume').decode())
print('COMMIT=' + git('rev-parse', 'HEAD').decode().strip())
print('INDEX=' + repr(git('diff', '--cached', '--name-only').decode()))
print('REMOTES=' + repr(git('remote', '-v').decode()))
