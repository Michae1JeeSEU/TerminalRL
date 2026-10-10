import json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_marl.training.p3_v26_train_dataset import load_six_cycle_train
from terminal_marl.training.p3_neighbor_witness_dataset import load_witness_targets
root=Path('D:/7_codespace/training_outputs/terminal')
old=root/'p3_v25_neutral_course_v2_20261008';new=root/'p3_v26_resources_20261009'
paired=load_six_cycle_train([old/x for x in ('train_flow6_mixed','train_dense24_fixed','train_dense24_mixed')],
    [new/f'train_{i:02d}' for i in (1,2,3)])
witness=load_witness_targets('.tmp/p3_neighbor_witness_train_20261009',paired)
result=dict(train_rows=len(paired.original.ids),max_neighbors=paired.new[1].shape[1],queries=len(witness.state_index),
    known_pairs=int(witness.known.sum()),blocked_pairs=int((witness.blocked.bool()&witness.known).sum()),
    known_scene=int(witness.scene_known.sum()),blocked_scene=int((witness.scene_blocked.bool()&witness.scene_known).sum()),
    by_seed=dict(Counter(witness.seeds.tolist())),original_labels_and_inputs_consistent=True)
path=Path('.tmp/p3_neighbor_witness_analysis.json');assert not path.exists()
path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
