from pathlib import Path
import difflib
import hashlib

root = Path(__file__).resolve().parents[1]
baseline = (root/'.tmp/p3_polynomial_baseline_20261005.py').read_text()
candidate = (root/'.tmp/p3_polynomial_candidate_20261005.py').read_text()
name = 'terminal_marl/core/polynomial_trajectory.py'
assert (root/name).read_text() == baseline
patch = ''.join(difflib.unified_diff(baseline.splitlines(keepends=True), candidate.splitlines(keepends=True),
                                   fromfile='a/'+name, tofile='b/'+name, n=0))
out = root/'docs/evidence/v4_2/p3_numeric_bounds_candidate_20261005.patch'
assert hashlib.sha256(out.read_bytes()).hexdigest() == 'ce610c4b067678d2516a925e0f9d65b11970d56ab092f910203f90a83fb5f902'
with out.open('w', encoding='utf-8', newline='\n') as f:
    f.write(patch)
print(out, hashlib.sha256(out.read_bytes()).hexdigest())
