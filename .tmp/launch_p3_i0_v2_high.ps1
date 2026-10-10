$ErrorActionPreference = 'Stop'
$remoteCommand = @'
set -eu
cd /home/user/projects/terminal
/home/user/envs/terminal_env/bin/python - <<'PY'
from pathlib import Path
import hashlib, importlib.util, json
root = Path('/home/user/projects/terminal')
assert Path.cwd().resolve() == root.resolve(), 'wrong launch cwd'
spec = importlib.util.find_spec('terminal_marl')
assert spec and str(root/'terminal_marl') in list(spec.submodule_search_locations), 'wrong module location'
expected = {
'terminal_marl/scripts/collect_p3_v3_i0.py':'cf4b45043a39538326906fafcc4822611b26c7c385df8ee7b3d80fa6eed3e44c',
'terminal_marl/scripts/collect_p3_v3_sparse_expert.py':'851b1de4bcaec6919b24a50e37881003b1b4898aecbb3ca84bf2c88343726fbe',
'terminal_marl/training/p3_v3_expert_dataset.py':'109df36b42576d87a2137c96b71febfd6e11876191d1be60ec44eaf3b7122d6f',
'terminal_marl/configs/p3_integrated_repair.py':'eee7eb0a2419e2990334edeed73ffb28f94699b4157d54c7815002ec792e2f23',
'terminal_marl/core/polynomial_trajectory.py':'832a88f432f083010cbc34a9dbc2c14b8b9fe6cbe016292bc43c0e12695d0a9f'}
for name, digest in expected.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
base = Path('/home/user/training_outputs/terminal/p3_v3_i0_v2_20261004')
old = base/'dense_high'
assert (base/'dense_high.exit').read_text().strip() == '1'
assert "No module named 'terminal_marl'" in (base/'dense_high.log').read_text()
assert not (old/'protocol.json').exists(), 'old collector may have started'
assert not old.exists() or not any(old.iterdir()), 'old output is not empty'
new = base/'dense_high_launch2'
for path in (new, Path(str(new)+'.pid'), Path(str(new)+'.log'), Path(str(new)+'.exit')):
    assert not path.exists(), f'output already exists: {path}'
active = []
modules = {'terminal_marl.scripts.collect_p3_v3_i0', 'terminal_marl.scripts.collect_p3_v3_sparse_expert'}
for process in Path('/proc').iterdir():
    if not process.name.isdecimal():
        continue
    try:
        argv = (process/'cmdline').read_bytes().split(b'\0')
    except (OSError, PermissionError):
        continue
    words = [a.decode(errors='replace') for a in argv if a]
    if words and Path(words[0]).name.startswith('python') and any(m in words for m in modules):
        active.append((process.name, words))
assert not active, active
print(json.dumps({'preflight':'OK','cwd':str(Path.cwd()),'output':str(new),'old_failure_preserved':True}))
PY
out=/home/user/training_outputs/terminal/p3_v3_i0_v2_20261004/dense_high_launch2
nohup bash -c '
cd /home/user/projects/terminal || exit 97
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/user/envs/terminal_env/bin/python -u -m terminal_marl.scripts.collect_p3_v3_i0 --protocol-version v2 --partition train --index 1 --output "$1" &
child=$!
printf "%s\n" "$child" > "$1.pid"
wait "$child"
rc=$?
printf "%s\n" "$rc" > "$1.exit"
exit "$rc"
' _ "$out" </dev/null >"$out.log" 2>&1 &
printf 'wrapper_pid=%s\n' "$!"
'@
$remoteCommand = $remoteCommand.Replace("`r`n", "`n")
$remoteCommand | ssh -T -o BatchMode=yes -o ConnectTimeout=15 lab-server 'bash -s'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
