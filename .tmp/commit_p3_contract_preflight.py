from pathlib import Path
import subprocess


def git(*args, input=None):
    result = subprocess.run(['git', *args], input=input, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8', errors='replace'))
    return result.stdout


assert git('rev-parse', 'HEAD').decode().strip() == 'b20d1effe3cd488a0403101ae031da4a19804a33', 'HEAD changed; return to primary'
assert not git('diff', '--cached', '--name-only').strip(), 'nonempty index; return to primary'
paths = [
    'terminal_marl/configs/p3_integrated_repair.py',
    'terminal_marl/scripts/diagnose_p3_direct_contract.py',
    'tests/test_p3_direct_contract.py',
    'docs/evidence/v4_2/p3_direct_c2_contract_preflight_20261003.md',
    'docs/evidence/v4_2/p3_direct_c2_repair_validation_20261003.md',
]
start = '## 2026-10-03 P3修复效果逐步验证（最新）'
end = '## 2026-10-03 P3 统一实现修复（最新）'
local = Path('docs/CURRENT_STATUS.md').read_text(encoding='utf8')
prefix = local[local.index(start):local.index(end)]
base = git('show', 'HEAD:docs/CURRENT_STATUS.md').decode('utf8')
body = base[:base.index(start)] + prefix + base[base.index(end):]
git('add', '--', *paths)
blob = git('hash-object', '-w', '--stdin', input=body.encode('utf8')).decode().strip()
git('update-index', '--cacheinfo', f'100644,{blob},docs/CURRENT_STATUS.md')
assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == set(paths + ['docs/CURRENT_STATUS.md'])
print(git('diff', '--cached', '--check').decode())
print(git('diff', '--cached', '--stat').decode())
print(git('commit', '-m', 'fix: keep P3 source inference optimizer-free and verify snapshot contracts').decode())
print('COMMIT=' + git('rev-parse', 'HEAD').decode().strip())
print('REMOTES=' + repr(git('remote', '-v').decode()))
print('INDEX=' + repr(git('diff', '--cached', '--name-only').decode()))
