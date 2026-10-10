from pathlib import Path
import hashlib
import subprocess


def git(*args, input=None):
    result = subprocess.run(['git', *args], input=input, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8', errors='replace'))
    return result.stdout


assert git('rev-parse', 'HEAD').decode().strip() == '23672a30d311394b39b9d8536ec0809ffe41f4af'
assert not git('diff', '--cached', '--name-only').strip()
for path, expected in {
    'terminal_marl/scripts/collect_p3_v3_i0.py': '66956ea3ee0284c7c7186e43d3b916963bec63f2e9c6f50597270908884d4243',
    'terminal_marl/scripts/collect_p3_v3_sparse_expert.py': '851b1de4bcaec6919b24a50e37881003b1b4898aecbb3ca84bf2c88343726fbe',
    'tests/test_p3_v3_i0.py': '7ffb0cd4981d80a8725db337d8e60504504bb12973e595a5949b1499f38fd9a1',
}.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
paths = [
    'terminal_marl/scripts/collect_p3_v3_i0.py',
    'terminal_marl/scripts/collect_p3_v3_sparse_expert.py',
    'tests/test_p3_v3_i0.py',
    'docs/evidence/v4_2/p3_v3_i0_protocol_20261004.md',
]
start = '## 2026-10-03 P3修复效果逐步验证（最新）'
end = '## 2026-10-03 P3 统一实现修复（最新）'
local = Path('docs/CURRENT_STATUS.md').read_text(encoding='utf8')
base = git('show', 'HEAD:docs/CURRENT_STATUS.md').decode('utf8')
assert local.count(start) == base.count(start) == 1
assert local.count(end) == base.count(end) == 1
body = base[:base.index(start)] + local[local.index(start):local.index(end)] + base[base.index(end):]
git('add', '--', *paths)
blob = git('hash-object', '-w', '--stdin', input=body.encode('utf8')).decode().strip()
git('update-index', '--cacheinfo', f'100644,{blob},docs/CURRENT_STATUS.md')
assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == {*paths, 'docs/CURRENT_STATUS.md'}
print(git('diff', '--cached', '--check').decode())
print(git('diff', '--cached', '--stat').decode())
print(git('commit', '-m', 'feat: collect all eligible V3 actor states under frozen I0 protocol').decode())
print('COMMIT=' + git('rev-parse', 'HEAD').decode().strip())
print('REMOTES=' + repr(git('remote', '-v').decode()))
print('INDEX=' + repr(git('diff', '--cached', '--name-only').decode()))
