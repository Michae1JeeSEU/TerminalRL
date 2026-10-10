"""Read-only exact-input collision check; no model, fit, or environment."""
from collections import defaultdict, Counter
from pathlib import Path
import json
import numpy as np
from terminal_marl.training.p3_v3_bc_learnability import load_inputs


def collisions(x, gate, proposal, ids):
    assert np.isfinite(x).all()
    groups = defaultdict(list)
    for i, row in enumerate(x):
        groups[tuple(float(v) for v in row)].append(i)
    repeated = [indices for indices in groups.values() if len(indices) > 1]
    gate_conflicts = [g for g in repeated if len({int(gate[i]) for i in g}) > 1]
    proposal_conflicts = [g for g in repeated
        if len({int(proposal[i]) for i in g if gate[i] == 1}) > 1]
    return dict(rows=len(x), unique_inputs=len(groups), repeated_groups=len(repeated),
        repeated_rows=sum(map(len, repeated)), gate_conflicting_groups=len(gate_conflicts),
        release_proposal_conflicting_groups=len(proposal_conflicts),
        gate_conflicting_ids=[[ids[i] for i in g] for g in gate_conflicts],
        proposal_conflicting_ids=[[ids[i] for i in g] for g in proposal_conflicts])


test = collisions(np.array([[0.,1.],[-0.,1.],[2.,3.],[2.,3.],[4.,5.]]),
    [0,1,1,1,0], [-1,2,3,4,-1], list('abcde'))
assert test['unique_inputs'] == 3 and test['repeated_groups'] == 2
assert test['gate_conflicting_ids'] == [['a','b']]
assert test['proposal_conflicting_ids'] == [['c','d']]
data = load_inputs(Path('D:/7_codespace/training_outputs/terminal'))
assert data.x.shape == (1046,90) and len(set(data.ids)) == 1046
result = collisions(data.x.numpy(), data.gate.tolist(), data.proposal.tolist(), data.ids)
result.update(status='EXACT_INPUT_CHECK_COMPLETE', synthetic_self_check=True,
    observation_dtype=str(data.x.dtype), observation_dim=90, model_created=False,
    optimizer_created=False, environment_resets=0, teacher_queries=0,
    gate_counts=dict(Counter(data.gate.tolist())),
    limit='Exact equality only; absence of collisions does not establish observability or rule out nearby label boundaries.')
out = Path('.tmp/p3_v3_exact_inputs_20261007.json')
assert not out.exists()
out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps(result, indent=2))
