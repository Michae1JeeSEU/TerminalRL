"""Launch the one frozen offline diagnostic; SSH commands only, no source upload."""
import shlex
import subprocess

cwd = '/home/user/projects/terminal'
python = '/home/user/envs/terminal_env/bin/python'
root = '/home/user/training_outputs/terminal/p3_v3_bc_learnability_20261007'
out = root+'/fit'
hashes = {
 'terminal_marl/training/p3_v3_bc_learnability.py': '260797096285cdfbba5eb463cfa74b0a58497ebdf24ee56fb6e27c937339dfe4',
 'terminal_marl/training/p3_v3_expert_dataset.py': '109df36b42576d87a2137c96b71febfd6e11876191d1be60ec44eaf3b7122d6f',
 'terminal_marl/algo/joint_ppo_v4_2.py': 'abc2f3678b68de7808328ffdeccc5fa7f18d6c21527a38627fdcfc38fd173490',
}
command = ['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python,
 '-u','-m','terminal_marl.training.p3_v3_bc_learnability',
 '--root','/home/user/training_outputs/terminal','--output',out,'--mode','fit']
wrapper = f'''set +e
cd {shlex.quote(cwd)} || exit 101
{' '.join(map(shlex.quote,command))} &
child=$!
printf '%s\n' "$child" > {shlex.quote(out+'.pid')}
wait "$child"
code=$?
printf '%s\n' "$code" > {shlex.quote(out+'.exit')}
exit "$code"
'''
lines=['set -eu',f'cd {shlex.quote(cwd)}',f'test "$(pwd -P)" = {shlex.quote(cwd)}']
for suffix in ('','.pid','.wrapper.pid','.log','.exit'):
    lines.append(f'test ! -e {shlex.quote(out+suffix)}')
lines.append("if ps -C python -o args= | grep -E 'terminal_marl|/home/user/projects/terminal/' >/dev/null; then echo 'project Python already active'; exit 93; fi")
checks='\n'.join(f'{value}  {name}' for name,value in hashes.items())+'\n'
lines.append(f'printf %s {shlex.quote(checks)} | sha256sum -c -')
check=f"import json; from pathlib import Path; p=Path('{root}'); assert json.loads((p/'prepare/result.json').read_text())['status']=='PREPARED'; assert json.loads((p/'smoke/result.json').read_text())['status']=='SMOKE_OK'; assert (p/'prepare.exit').read_text().strip() == (p/'smoke.exit').read_text().strip() == '0'"
lines.append(f'{python} -c {shlex.quote(check)}')
lines += [f'nohup bash -c {shlex.quote(wrapper)} > {shlex.quote(out+".log")} 2>&1 < /dev/null &',
          'wrapper=$!',f'printf \'%s\\n\' "$wrapper" > {shlex.quote(out+".wrapper.pid")}',
          'sleep 2',f'child=$(cat {shlex.quote(out+".pid")})',
          'ps -p "$wrapper,$child" -o pid,ppid,etime,args || true',
          'readlink /proc/"$child"/cwd || true']
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','lab-server','\n'.join(lines)+'\n']).returncode)
