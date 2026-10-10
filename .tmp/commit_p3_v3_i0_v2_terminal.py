from pathlib import Path
import subprocess


def git(*args, input=None):
    result = subprocess.run(['git', *args], input=input, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8', errors='replace'))
    return result.stdout


assert git('rev-parse', 'HEAD').decode().strip() == '23c4f4d71f61d6c48aff82401b2af089b10145ed'
assert not git('diff', '--cached', '--name-only').strip()
path = 'docs/evidence/v4_2/p3_v3_i0_v2_budget_20261004.md'
start = '## 2026-10-03 P3修复效果逐步验证（最新）'
end = '## 2026-10-03 P3 统一实现修复（最新）'
local = Path('docs/CURRENT_STATUS.md').read_text(encoding='utf8')
base = git('show', 'HEAD:docs/CURRENT_STATUS.md').decode('utf8')
assert local.count(start) == base.count(start) == 1
assert local.count(end) == base.count(end) == 1
body = base[:base.index(start)] + local[local.index(start):local.index(end)] + base[base.index(end):]
git('add', '--', path)
blob = git('hash-object', '-w', '--stdin', input=body.encode('utf8')).decode().strip()
git('update-index', '--cacheinfo', f'100644,{blob},docs/CURRENT_STATUS.md')
assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == {path, 'docs/CURRENT_STATUS.md'}
print(git('diff', '--cached', '--check').decode())
print(git('diff', '--cached', '--stat').decode())
print(git('commit', '-m', 'docs: conclude V2 I0 budget validation and annotation diagnosis').decode())
print('COMMIT=' + git('rev-parse', 'HEAD').decode().strip())
print('REMOTES=' + repr(git('remote', '-v').decode()))
print('INDEX=' + repr(git('diff', '--cached', '--name-only').decode()))
