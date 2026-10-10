from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from terminal_marl.core.resource_context_observation import ResourceContextCache
from terminal_marl.core.trajectory_context_observation import TrajectoryContextCache
from terminal_marl.scripts.diagnose_p3_v25_lz_alias import swap_ground_service,equal_observation
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_neutral_course_v2_20261008/train_dense24_fixed/seed_893512')
output=Path('.tmp/p3_resource_context_witness.json');assert not output.exists()
entry=next(e for e in json.loads((root/'selected.json').read_text()) if e['row']['tick']==255 and int(e['row']['uid'])==0)
data=(root/'snapshots'/entry['snapshot']).read_bytes()
assert hashlib.sha256(data).hexdigest()==entry['snapshot_sha256']=='87cdf2821eae25381d7d08a26210c259694b5ef953be2d4113b7034b103cf548'
env=pickle.loads(data);changed=deepcopy(env);swap_ground_service(changed,4,5)
old=[TrajectoryContextCache(e).build(e.active[0]) for e in (env,changed)]
new=[ResourceContextCache(e).build(e.active[0]) for e in (env,changed)]
assert equal_observation(*old) and not equal_observation(*new)
for a,b in zip(old,new):
    np.testing.assert_array_equal(a.ego,b.ego[:105]);np.testing.assert_array_equal(a.traffic,b.traffic)
differences=np.flatnonzero(new[0].ego!=new[1].ego).tolist()
result=dict(status='VERIFIED',old_dimensions=105,new_dimensions=159,differing_ego_indices=differences,
    old_same=True,new_same=False,old_prefix_and_traffic_exact=True,teacher_queries=0,environment_steps=0)
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
