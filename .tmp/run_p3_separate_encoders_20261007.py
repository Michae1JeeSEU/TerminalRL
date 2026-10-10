"""Run the frozen diagnostic via SSH; source reaches server through Syncthing."""
import shlex
import subprocess
import sys
phase=sys.argv[1]
assert phase in ('validate','fit')
cwd='/home/user/projects/terminal'
python='/home/user/envs/terminal_env/bin/python'
root='/home/user/training_outputs/terminal/p3_v3_bc_separate_encoders_20261007'
hashes={
 'terminal_marl/training/p3_v3_bc_learnability.py':'9c62f5eb74e7afe72813601462e4ed3ea960e1ae1651585feb4bada7345b84e8',
 'tests/test_p3_v3_bc_learnability.py':'558b49b600a4a7ce9703a1e37dea4e908ae7045fec450570eb4ea8dfadebff31',
 'terminal_marl/training/p3_v3_expert_dataset.py':'109df36b42576d87a2137c96b71febfd6e11876191d1be60ec44eaf3b7122d6f',
 'terminal_marl/algo/joint_ppo_v4_2.py':'abc2f3678b68de7808328ffdeccc5fa7f18d6c21527a38627fdcfc38fd173490',
}
q=shlex.quote
env=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python]
def command(mode):
    return shlex.join(env+['-u','-m','terminal_marl.training.p3_v3_bc_learnability',
        '--root','/home/user/training_outputs/terminal','--output',root+'/'+mode,
        '--mode',mode,'--separate-encoders'])
lines=['set -eu','cd '+q(cwd),'test "$(pwd -P)" = '+q(cwd)]
checks=''.join(f'{sha}  {file}\n' for file,sha in hashes.items())
lines.append('printf %s '+q(checks)+' | sha256sum -c -')
lines.append("if ps -C python -o args= | grep -E 'terminal_marl|/home/user/projects/terminal/' >/dev/null; then echo 'project Python already active'; exit 93; fi")
if phase=='validate':
    lines.extend(['test ! -e '+q(root),'mkdir '+q(root)])
    jobs=[('tests',shlex.join(env+['-m','pytest','tests/test_p3_v3_bc_learnability.py',
        'tests/test_p3_v3_expert_dataset.py','-q']))]
    jobs.extend((mode,command(mode)) for mode in ('prepare','smoke'))
    for name,cmd in jobs:
        lines.extend(['set +e',cmd+' > '+q(root+'/'+name+'.log')+' 2>&1','code=$?',
            'printf \'%s\n\' "$code" > '+q(root+'/'+name+'.exit'),'set -e','test "$code" -eq 0'])
    lines.append('cat '+shlex.join([root+'/tests.log',root+'/prepare.log',root+'/smoke.log']))
else:
    out=root+'/fit'
    for suffix in ('','.pid','.wrapper.pid','.log','.exit'):
        lines.append('test ! -e '+q(out+suffix))
    check=(f"import json; from pathlib import Path; p=Path('{root}'); "
        "assert json.loads((p/'prepare/result.json').read_text())['status']=='PREPARED'; "
        "assert json.loads((p/'smoke/result.json').read_text())['status']=='SMOKE_OK'; "
        "assert all((p/(s+'.exit')).read_text().strip()=='0' for s in ('tests','prepare','smoke'))")
    lines.append(python+' -c '+q(check))
    wrapper=f'''set +e
cd {q(cwd)} || exit 101
{command('fit')} &
child=$!
printf '%s\n' "$child" > {q(out+'.pid')}
wait "$child"
code=$?
printf '%s\n' "$code" > {q(out+'.exit')}
exit "$code"
'''
    lines.extend(['nohup bash -c '+q(wrapper)+' > '+q(out+'.log')+' 2>&1 < /dev/null &',
        'wrapper=$!','printf \'%s\n\' "$wrapper" > '+q(out+'.wrapper.pid'),
        'sleep 1','child=$(cat '+q(out+'.pid')+')',
        'ps -p "$wrapper,$child" -o pid,ppid,etime,args || true','readlink /proc/"$child"/cwd || true'])
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
    'lab-server','\n'.join(lines)+'\n']).returncode)
