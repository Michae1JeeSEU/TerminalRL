from pathlib import Path
import hashlib
source = Path('terminal_marl/core/polynomial_trajectory.py')
candidate = Path('.tmp/p3_polynomial_candidate_20261005.py').read_bytes()
assert hashlib.sha256(source.read_bytes()).hexdigest() == '832a88f432f083010cbc34a9dbc2c14b8b9fe6cbe016292bc43c0e12695d0a9f'
assert hashlib.sha256(candidate).hexdigest() == '603a2997df35bac03771bc9d69d99f1fed48c899cce0ff1e93acf75908fea1e6'
source.write_bytes(candidate)
assert hashlib.sha256(source.read_bytes()).hexdigest() == '603a2997df35bac03771bc9d69d99f1fed48c899cce0ff1e93acf75908fea1e6'
print('Applied exact validated candidate locally.')
