"""Execute frozen SSH commands; no source transfer or server Git."""
import shlex
import subprocess
import sys

phase = sys.argv[1]
assert phase in ('validate', 'fit', 'fit_serialization_fix')
cwd = '/home/user/projects/terminal'
python = '/home/user/envs/terminal_env/bin/python'
root = '/home/user/training_outputs/terminal/p3_v3_bc_gate_balance_20261007'
hashes = {
 'terminal_marl/training/p3_v3_bc_learnability.py': '933b5696cbb0b818eabb789e0473df48a4e199b90ece7a4f1d698e8c1804e37d',
 'tests/test_p3_v3_bc_learnability.py': 'a0d650f8bdcd68f30fdc230bd2ae6849f14f03ca67d9a5ebc27badd8ff94a6a7',
 'terminal_marl/training/p3_v3_expert_dataset.py': '109df36b42576d87a2137c96b71febfd6e11876191d1be60ec44eaf3b7122d6f',
 'terminal_marl/algo/joint_ppo_v4_2.py': 'abc2f3678b68de7808328ffdeccc5fa7f18d6c21527a38627fdcfc38fd173490',
}
quote = shlex.quote
env = ['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',python]
def command(mode, output_name=None):
    return shlex.join(env + ['-u','-m','terminal_marl.training.p3_v3_bc_learnability',
        '--root','/home/user/training_outputs/terminal','--output',root+'/'+(output_name or mode),
        '--mode',mode,'--gate-balance'])
lines = ['set -eu', 'cd '+quote(cwd), 'test "$(pwd -P)" = '+quote(cwd)]
checks = ''.join(f'{sha}  {file}\n' for file,sha in hashes.items())
lines.append('printf %s '+quote(checks)+' | sha256sum -c -')
lines.append("if ps -C python -o args= | grep -E 'terminal_marl|/home/user/projects/terminal/' >/dev/null; then echo 'project Python already active'; exit 93; fi")
if phase == 'validate':
    lines.extend(['test ! -e '+quote(root), 'mkdir '+quote(root)])
    jobs = [('tests', shlex.join(env+['-m','pytest','tests/test_p3_v3_bc_learnability.py',
        'tests/test_p3_v3_expert_dataset.py','-q']))]
    jobs.extend((mode,command(mode)) for mode in ('prepare','smoke'))
    for name, cmd in jobs:
        lines.extend(['set +e',cmd+' > '+quote(root+'/'+name+'.log')+' 2>&1', 'result_code=$?',
            'printf \'%s\n\' "$result_code" > '+quote(root+'/'+name+'.exit'),
            'set -e','test "$result_code" -eq 0'])
    lines.append('cat '+shlex.join([root+'/tests.log',root+'/prepare.log',root+'/smoke.log']))
else:
    out = root+'/'+phase
    for suffix in ('','.pid','.wrapper.pid','.log','.exit'):
        lines.append('test ! -e '+quote(out+suffix))
    check = (f"import json; from pathlib import Path; p=Path('{root}'); "
        "assert json.loads((p/'prepare/result.json').read_text())['status']=='PREPARED'; "
        "assert json.loads((p/'smoke/result.json').read_text())['status']=='SMOKE_OK'; "
        "assert all((p/(s+'.exit')).read_text().strip()=='0' for s in ('tests','prepare','smoke'))")
    lines.append(python+' -c '+quote(check))
    if phase == 'fit_serialization_fix':
        check = (f"import json; from pathlib import Path; p=Path('{root}'); "
            "assert (p/'serialization_tests.exit').read_text().strip()=='0'; "
            "proof=json.loads((p/'initial_check.json').read_text()); "
            "assert proof['json_form_equal'] and not proof['optimizer_created']; "
            "old=json.loads((p/'fit/result.json').read_text()); "
            "assert old['status']=='ERROR' and old['seeds']==[] and "
            "old['error']['message']=='paired initialization metrics differ'")
        lines.append(python+' -c '+quote(check))
    wrapper = f'''set +e
cd {quote(cwd)} || exit 101
{command('fit',phase)} &
child=$!
printf '%s\n' "$child" > {quote(out+'.pid')}
wait "$child"
code=$?
printf '%s\n' "$code" > {quote(out+'.exit')}
exit "$code"
'''
    lines.extend(['nohup bash -c '+quote(wrapper)+' > '+quote(out+'.log')+' 2>&1 < /dev/null &',
        'wrapper=$!', 'printf \'%s\n\' "$wrapper" > '+quote(out+'.wrapper.pid'),
        'sleep 1', 'child=$(cat '+quote(out+'.pid')+')',
        'ps -p "$wrapper,$child" -o pid,ppid,etime,args || true',
        'readlink /proc/"$child"/cwd || true'])
raise SystemExit(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
    'lab-server','\n'.join(lines)+'\n']).returncode)
