"""Exact authorized launcher; sends shell commands, never uploads source files."""
import argparse
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('case', choices=('smoke', 'dense_mixed'))
parser.add_argument('--print-only', action='store_true')
args = parser.parse_args()
cwd = '/home/user/projects/terminal'
python = '/home/user/envs/terminal_env/bin/python'
base = '/home/user/training_outputs/terminal/p3_v3_i0_v3_20261007'
output = f'{base}/{args.case}'
module = 'terminal_marl.scripts.collect_p3_v3_i0'
command = ['env', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
           python, '-u', '-m', module, '--protocol-version', 'v3']
command += ['--smoke'] if args.case == 'smoke' else ['--partition', 'train', '--index', '1']
command += ['--output', output]
hashes = {
    'terminal_marl/core/polynomial_trajectory.py': '603a2997df35bac03771bc9d69d99f1fed48c899cce0ff1e93acf75908fea1e6',
    'terminal_marl/scripts/collect_p3_v3_i0.py': 'd267c13910582312634caadbd9bcab52db9fe5b8883f986b1a0034fb6f9e9247',
    'terminal_marl/scripts/collect_p3_v3_sparse_expert.py': '851b1de4bcaec6919b24a50e37881003b1b4898aecbb3ca84bf2c88343726fbe',
    'terminal_marl/training/p3_v3_expert_dataset.py': '109df36b42576d87a2137c96b71febfd6e11876191d1be60ec44eaf3b7122d6f',
    'terminal_marl/configs/p3_integrated_repair.py': 'eee7eb0a2419e2990334edeed73ffb28f94699b4157d54c7815002ec792e2f23',
}
wrapper = f'''set +e
cd {shlex.quote(cwd)} || exit 101
{' '.join(map(shlex.quote, command))} &
child=$!
printf '%s\\n' "$child" > {shlex.quote(output+'.pid')}
wait "$child"
code=$?
printf '%s\\n' "$code" > {shlex.quote(output+'.exit')}
exit "$code"
'''
lines = ['set -eu', f'cd {shlex.quote(cwd)}', f'test "$(pwd -P)" = {shlex.quote(cwd)}']
for suffix in ('', '.pid', '.wrapper.pid', '.exit', '.log'):
    lines.append(f'test ! -e {shlex.quote(output+suffix)}')
lines.append("if ps -C python -o args= | grep -E 'terminal_marl|/home/user/projects/terminal/' >/dev/null; then echo 'project Python already active'; exit 93; fi")
checks = '\n'.join(f'{digest}  {name}' for name, digest in hashes.items())+'\n'
lines.append(f'printf %s {shlex.quote(checks)} | sha256sum -c -')
if args.case == 'dense_mixed':
    check = f"import json; from pathlib import Path; p=Path('{base}/smoke'); assert json.loads((p/'result.json').read_text())['status']=='SMOKE_OK'; assert Path(str(p)+'.exit').read_text().strip()=='0'"
    lines.append(f'{python} -c {shlex.quote(check)}')
lines += [f'mkdir -p {shlex.quote(base)}',
          f'nohup bash -c {shlex.quote(wrapper)} > {shlex.quote(output+".log")} 2>&1 < /dev/null &',
          'wrapper=$!', f'printf \'%s\\n\' "$wrapper" > {shlex.quote(output+".wrapper.pid")}',
          'sleep 2', f'cat {shlex.quote(output+".pid")}',
          f'child=$(cat {shlex.quote(output+".pid")})',
          'ps -p "$wrapper,$child" -o pid,ppid,etime,args || true',
          'readlink /proc/"$child"/cwd || true',
          f'test -f {shlex.quote(output+"/protocol.json")} && cat {shlex.quote(output+"/protocol.json")} || true']
remote = '\n'.join(lines)+'\n'
if args.print_only:
    print(remote)
else:
    raise SystemExit(subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'lab-server', remote]).returncode)
