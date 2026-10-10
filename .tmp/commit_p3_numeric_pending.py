from pathlib import Path
import hashlib
import subprocess
import sys
sys.stdout.reconfigure(encoding='utf-8')


def git(*args):
    result = subprocess.run(['git', *args], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8', errors='replace'))
    return result.stdout


expected = {'docs/CURRENT_STATUS.md',
            'docs/evidence/v4_2/p3_numeric_bounds_candidate_20261005.patch',
            'docs/evidence/v4_2/p3_numeric_bounds_optimization_20261005.md',
            'tests/test_polynomial_trajectory_numeric_equivalence.py'}
assert git('rev-parse', 'HEAD').decode().strip() == 'f01477a4d2ac4502ef1e8cb059735237948e18c9'
assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == expected
assert hashlib.sha256(Path('terminal_marl/core/polynomial_trajectory.py').read_bytes()).hexdigest() == '832a88f432f083010cbc34a9dbc2c14b8b9fe6cbe016292bc43c0e12695d0a9f'
git('diff', '--cached', '--check')
git('apply', '--check', '--unidiff-zero', 'docs/evidence/v4_2/p3_numeric_bounds_candidate_20261005.patch')
print(git('diff', '--cached', '--stat').decode())
print(git('commit', '-m', 'test: prepare exact bounds optimization pending server validation').decode())
print('COMMIT=' + git('rev-parse', 'HEAD').decode().strip())
print('INDEX=' + repr(git('diff', '--cached', '--name-only').decode()))
print('REMOTES=' + repr(git('remote', '-v').decode()))
print(git('status', '--short', '--untracked-files=no').decode())
