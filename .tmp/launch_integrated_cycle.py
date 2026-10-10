import subprocess, shlex
root='/home/user/training_outputs/terminal/p3_integrated_repair_20261003'
name='cycle_rule_3'
command='PYTHONDONTWRITEBYTECODE=1 /home/user/envs/terminal_env/bin/python -u -m terminal_marl.scripts.diagnose_p3_integrated_repair --run --high-soc --controller rule --requests 3 --warm-start 0 --seed 890301 --policy-seed 890401 --max-steps 1800 --output '+root+'/'+name
body='cd /home/user/projects/terminal || exit 99\n'+command+'\nrc=$?\nprintf "%s\\n" "$rc" > '+root+'/'+name+'.exit\nexit "$rc"'
script='set -eu\n'+''.join('test ! -e '+shlex.quote(root+'/'+name+suffix)+'\n' for suffix in ('','.log','.exit'))+'nohup bash -c '+shlex.quote(body)+' > '+shlex.quote(root+'/'+name+'.log')+' 2>&1 < /dev/null &\npid=$!\nprintf "LAUNCH_PID=%s\\n" "$pid"\n'
p=subprocess.run(['ssh','lab-server','bash -s'],input=script,text=True,capture_output=True)
print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
