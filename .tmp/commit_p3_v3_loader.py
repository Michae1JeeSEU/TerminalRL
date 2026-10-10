from pathlib import Path
import hashlib
import subprocess


def git(*args, input=None):
    result = subprocess.run(['git', *args], input=input, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8', errors='replace'))
    return result.stdout


assert git('rev-parse', 'HEAD').decode().strip() == '8e86b66b915ec953b6bf7c0a9ab4a9c736a34859'
assert not git('diff', '--cached', '--name-only').strip()
for path, expected in {
    'terminal_marl/training/p3_v3_expert_dataset.py': '109df36b42576d87a2137c96b71febfd6e11876191d1be60ec44eaf3b7122d6f',
    'tests/test_p3_v3_expert_dataset.py': '854f39cf11771076eaa4ea2d2c240f29f48372007dfee4c7bd022a3c83bbe98c',
}.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
paths = [
    'terminal_marl/training/p3_v3_expert_dataset.py',
    'tests/test_p3_v3_expert_dataset.py',
    'docs/technical_spec.md',
    'docs/evidence/v4_2/p3_v3_loader_validation_20261004.md',
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
print(git('commit', '-m', 'feat: add masked V3 expert dataset loader with split isolation').decode())
print('COMMIT=' + git('rev-parse', 'HEAD').decode().strip())
print('REMOTES=' + repr(git('remote', '-v').decode()))
print('INDEX=' + repr(git('diff', '--cached', '--name-only').decode()))
