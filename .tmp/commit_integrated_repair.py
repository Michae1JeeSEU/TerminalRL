from pathlib import Path
import subprocess

def git(*args):
    result=subprocess.run(['git',*args],capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr.decode('utf8',errors='replace'))
    return result.stdout
expected=Path('.tmp/p3_staged_tree.txt').read_bytes().strip()
assert git('write-tree').strip()==expected, 'STOP: staged tree changed after primary review'
print(git('diff','--cached','--check').decode())
print(git('commit','-m','fix: repair P3 dynamics and execution contracts').decode())
print('COMMIT='+git('rev-parse','HEAD').decode().strip())
print(git('show','--stat','--oneline','HEAD').decode())
remotes=git('remote','-v').decode()
print('REMOTES='+repr(remotes))
print('INDEX='+repr(git('diff','--cached','--name-only').decode()))
