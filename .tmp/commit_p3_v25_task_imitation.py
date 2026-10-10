from pathlib import Path
import subprocess
import sys
sys.stdout.reconfigure(encoding='utf8')


def git(*args, input=None):
    result = subprocess.run(['git', *args], input=input, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stdout+result.stderr).decode('utf8', errors='replace'))
    return result.stdout


assert git('rev-parse', 'HEAD').decode().strip() == 'a8c821be8d2af6efa8b48e38da9af0185e311145'
assert not git('diff', '--cached', '--name-only').strip()
files = ['terminal_marl/training/p3_v25_task_imitation.py','tests/test_p3_v25_task_imitation.py','docs/evidence/v4_2/p3_v25_actual_task_20261008.md']
git('diff', '--check', '--', *files)
git('add', '--', *files)
assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == set(files)
git('diff', '--cached', '--check')
print(git('diff', '--cached', '--stat').decode())
print(git('diff', '--cached', '--').decode())
print(git('commit', '-m', 'feat: add isolated actual-task imitation qualification entry').decode())
print('COMMIT='+git('rev-parse', 'HEAD').decode().strip())
print('INDEX='+repr(git('diff', '--cached', '--name-only').decode()))
print('REMOTES='+repr(git('remote', '-v').decode()))

