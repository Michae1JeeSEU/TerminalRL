"""Repeat two registered failure prefixes after the bounded V3 repair."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from terminal_marl.scripts import evaluate_p3_foundation_repair_20261002 as evaluator

root = Path(__file__).resolve().parent.parent
output = Path('/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/energy_numeric_repair_smoke')
names = ('terminal_marl/core/trajectory_generator.py',
    'terminal_marl/core/trajectory_time_parameterization.py',
    'terminal_marl/core/energy_reachability.py',
    'terminal_marl/scripts/evaluate_p3_foundation_repair_20261002.py')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data, exclusive=False):
    encoded = json.dumps(evaluator._jsonable(data), indent=2, sort_keys=True, allow_nan=False) + '\n'
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        stream.write(encoded)

output.mkdir(parents=True, exist_ok=True)
if (output / 'result.json').exists() or (output / 'progress.json').exists():
    raise RuntimeError('refusing to overwrite an existing smoke run')
before = {name: sha(root / name) for name in names}
if sha(evaluator.SOURCE_CHECKPOINT) != evaluator.SOURCE_CHECKPOINT_SHA256:
    raise RuntimeError('fixed source checkpoint hash mismatch')
# Exercise output serialization before creating an environment.
example = {'status': 'OUTPUT_PREFLIGHT_OK', 'rows': [{'ticks': 14, 'error': None}]}
write(output / 'output_preflight.json', example, exclusive=True)
if json.loads((output / 'output_preflight.json').read_text()) != example:
    raise RuntimeError('serialization roundtrip mismatch')
base = evaluator._base_config()
cases = (('legacy88', 14), ('nominal_rule_repaired90', 10))
arms = []
for name, limit in cases:
    spec = next(row for row in evaluator._arm_specs() if row['cell_id'] == 'cell_1' and row['arm_id'] == name)
    def progress(row):
        write(output / 'progress.json', {'status': 'running', 'arm_id': name,
            'tick': row['tick'] + 1, 'progress': row,
            'observed_at_utc': datetime.now(timezone.utc).isoformat()})
    arm = evaluator._run_arm(spec, base, max_steps=limit, progress_callback=progress)
    write(output / (name + '.json'), arm, exclusive=True)
    arms.append(arm)
    if arm['error'] is not None or evaluator._hard_safety_issue(arm):
        break
after = {name: sha(root / name) for name in names}
valid = before == after and len(arms) == 2 and all(
    arm['status'] == 'censored' and len(arm['progress']) == limit
    and arm['error'] is None and not evaluator._hard_safety_issue(arm)
    for arm, (_, limit) in zip(arms, cases))
payload = {'status': 'PREFIX_SMOKE_OK' if valid else 'PREFIX_SMOKE_FAILED',
    'purpose': 'known exception prefixes only; not full-cycle qualification',
    'source_hashes_before': before, 'source_hashes_after': after,
    'checkpoint_sha256': sha(evaluator.SOURCE_CHECKPOINT), 'arms': arms,
    'frozen_flags': {'PPO_ALLOWED': False, 'heldout_used': False, 'teacher_used': False,
        'optimizer_used': False, 'checkpoint_writes': False}}
write(output / 'result.json', payload, exclusive=True)
write(output / 'progress.json', {'status': payload['status'], 'completed_arms': len(arms)})
print(json.dumps({'status': payload['status'], 'artifact': str(output / 'result.json'),
    'sha256': sha(output / 'result.json'), 'arms': [{
        'arm_id': arm['arm_id'], 'status': arm['status'], 'ticks': len(arm['progress']),
        'error': arm['error'], 'safety': arm['safety'], 'validity': arm['validity']}
        for arm in arms]}, indent=2))
raise SystemExit(0 if valid else 2)
