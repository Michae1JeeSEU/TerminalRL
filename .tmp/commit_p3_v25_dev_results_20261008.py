from pathlib import Path
import subprocess
import sys
sys.stdout.reconfigure(encoding='utf8')


def git(*args, input=None):
    result = subprocess.run(['git', *args], input=input, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stdout+result.stderr).decode('utf8', errors='replace'))
    return result.stdout


assert git('rev-parse', 'HEAD').decode().strip() == '4edee3af0a06c25683c16874c0a71b8e5901fad2'
assert not git('diff', '--cached', '--name-only').strip()
files = ['docs/evidence/v4_2/p3_v25_formal_readiness_20261008.md']
start, end = '## 2026-10-03 P3修复效果逐步验证（最新）', '## 2026-10-03 P3 统一实现修复（最新）'
local = Path('docs/CURRENT_STATUS.md').read_text(encoding='utf8')
base = git('show', 'HEAD:docs/CURRENT_STATUS.md').decode('utf8')
assert local.count(start) == base.count(start) == local.count(end) == base.count(end) == 1
base_block = base[base.index(start):base.index(end)]
local_block = local[local.index(start):local.index(end)]
assert local_block.startswith(base_block.rstrip()), 'unexpected preexisting latest-block edits'
body = base[:base.index(start)]+local_block+base[base.index(end):]
git('diff', '--check', '--', *files, 'docs/CURRENT_STATUS.md')
git('add', '--', *files)
blob = git('hash-object', '-w', '--stdin', input=body.encode()).decode().strip()
git('update-index', '--cacheinfo', f'100644,{blob},docs/CURRENT_STATUS.md')
assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == {*files, 'docs/CURRENT_STATUS.md'}
git('diff', '--cached', '--check')
print(git('diff', '--cached', '--stat').decode())
print(git('diff', '--cached', '--', 'docs/CURRENT_STATUS.md').decode())
print(git('commit', '-m', 'docs: record complete V2.5 dense deterministic validation').decode())
print('COMMIT='+git('rev-parse', 'HEAD').decode().strip())
print('INDEX='+repr(git('diff', '--cached', '--name-only').decode()))
print('REMOTES='+repr(git('remote', '-v').decode()))

