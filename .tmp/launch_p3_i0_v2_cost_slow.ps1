$ErrorActionPreference = 'Stop'
$remoteCommand = @'
set -eu
cd /home/user/projects/terminal
/home/user/envs/terminal_env/bin/python - <<'PY'
from pathlib import Path
import hashlib
assert Path.cwd() == Path('/home/user/projects/terminal')
for file, digest in {
    '.tmp/profile_p3_i0_v2_mixed.py': '7eb101e7210dc67f23b6cc4a567a4a7c313ab6b3d877c529265974f477314329',
    'terminal_marl/scripts/diagnose_p3_expert_cost.py': '5ec3468891c69b0bdf411e201984135e4b9ecb6b06da1ec6e648472ea316275c',
}.items():
    assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest, file
base = Path('/home/user/training_outputs/terminal/p3_v3_i0_v2_20261004/dense_mixed_cost_diagnostic')
old = base/'earliest_slow'
assert old.is_dir() and not any(old.iterdir())
assert (base/'earliest_slow.exit').read_text().strip() == '1'
assert 'FileExistsError' in (base/'earliest_slow.log').read_text()
out = base/'earliest_slow_launch2'
for p in (out,Path(str(out)+'.log'),Path(str(out)+'.exit')):
    assert not p.exists(), p
for proc in Path('/proc').iterdir():
    if not proc.name.isdecimal(): continue
    try: argv = [a.decode(errors='replace') for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
    except OSError: continue
    if argv and Path(argv[0]).name.startswith('python'):
        assert not any('collect_p3_v3_i0' in a or 'collect_p3_v3_sparse_expert' in a or 'profile_p3_i0_v2_mixed.py' in a for a in argv), argv
print('SLOW_DIAGNOSTIC_PREFLIGHT_OK',flush=True)
PY
out=/home/user/training_outputs/terminal/p3_v3_i0_v2_20261004/dense_mixed_cost_diagnostic/earliest_slow_launch2
set +e
timeout --signal=TERM --kill-after=5s 120s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/user/envs/terminal_env/bin/python .tmp/profile_p3_i0_v2_mixed.py 19 /home/user/training_outputs/terminal/p3_v3_i0_v2_20261004/dense_mixed/seed_893102 "$out" >"$out.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$out.exit"
cat "$out.log"
exit "$rc"
'@
$remoteCommand = $remoteCommand.Replace("`r`n", "`n")
$remoteCommand | ssh -T -o BatchMode=yes -o ConnectTimeout=15 lab-server 'bash -s'
exit $LASTEXITCODE
