from pathlib import Path
import subprocess
import sys
sys.stdout.reconfigure(encoding='utf8')


def git(*args, input=None):
    result = subprocess.run(['git', *args], input=input, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stdout+result.stderr).decode('utf8', errors='replace'))
    return result.stdout


assert git('rev-parse', 'HEAD').decode().strip() == 'e635efb23cfe8ad5cd833e08b2694c7500943111'
assert not git('diff', '--cached', '--name-only').strip()
files = ['terminal_marl/training/p3_v25_task_student.py','tests/test_p3_v25_task_student.py','docs/evidence/v4_2/p3_v25_actual_task_20261008.md']
git('diff', '--check', '--', *files)
git('add', '--', *files)
assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == set(files)
git('diff', '--cached', '--check')
print(git('diff', '--cached', '--stat').decode())
print(git('diff', '--cached', '--').decode())
print(git('commit', '-m', 'feat: add teacher-free P3 task student development runner').decode())
print('COMMIT='+git('rev-parse', 'HEAD').decode().strip())
print('INDEX='+repr(git('diff', '--cached', '--name-only').decode()))
print('REMOTES='+repr(git('remote', '-v').decode()))

