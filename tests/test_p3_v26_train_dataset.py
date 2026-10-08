import json

import pytest
import torch

from terminal_marl.training import p3_v26_train_dataset as data
from terminal_marl.training import p3_v26_course_data as course
from terminal_marl.training import p3_v25_imitation_qualification as q


def cases(tmp_path):
    directories=[]
    for i,(seed,scenario) in enumerate(course.TRAIN_CASES,1):
        root=tmp_path/f'train_{i:02d}';episode=root/f'seed_{seed}';episode.mkdir(parents=True)
        row=dict(schema=q.CONTEXT_SCHEMA,observation_schema=course.VERSION,partition='train',seed=seed,
            tick=0,uid='0',decision_id=f'{data.DATA_SCHEMA}:{seed}:0:0',actor_mask=True,
            ego_feature_hash=course.EGO_SCHEMA.feature_hash,traffic_feature_hash=q.TRAFFIC_SCHEMA.feature_hash,
            ego=[0.]*159,traffic=[[float(i)]*50]*(i-1),traffic_mask=[True]*(i-1),
            teacher_gate=1 if i==2 else 0,teacher_proposal=12 if i==2 else None,
            original_observation=[0.]*90,context=dict(seed=seed,interface='m2i'))
        path=episode/'resource_context_observations.jsonl';path.write_text(json.dumps(row)+'\n')
        identity=dict(protocol=course.PROTOCOL,profile=course.PROFILE,partition='train',seed=seed,scenario=scenario)
        q.write(root/'protocol.json',dict(identity,smoke=False,task=course.task_settings(course.build_config(scenario)),
            learning_observation_schema=course.VERSION))
        q.write(root/'result.json',dict(identity,status='COLLECTION_OK',error=None,blockers=[],
            reconstruction=dict(rows=1),resource_reconstruction=dict(rows=1,sha256=q.sha(path))))
        directories.append(root)
    return directories


def test_complete_cycles_preserve_labels_and_ragged_traffic_and_reject_q(tmp_path):
    paths=cases(tmp_path);value=data.load_independent_train(paths)
    assert value.original.gate.tolist()==[0,1,0] and value.original.proposal.tolist()==[-1,12,-1]
    assert value.new[0].shape==(3,159) and value.new[2].sum(1).tolist()==[0,1,2]
    assert value.new[1][1,0,0]==2 and value.new[1][1,1,0]==0
    combined=data.combine_train([value]);torch.testing.assert_close(combined.new[1],value.new[1])
    with pytest.raises(ValueError,match='duplicate'):data.combine_train([value,value])
    with pytest.raises(ValueError,match='all three'):data.load_independent_train(paths[:2])
    protocol=q.read(paths[0]/'protocol.json');q.write(paths[0]/'protocol.json',dict(protocol,partition='Q'))
    with pytest.raises(ValueError,match='identity'):data.load_independent_train(paths)


def test_changed_bytes_and_incomplete_cycle_are_rejected(tmp_path):
    paths=cases(tmp_path)
    report=q.read(paths[0]/'result.json');q.write(paths[0]/'result.json',dict(report,status='INCOMPLETE'))
    with pytest.raises(ValueError,match='incomplete'):data.load_independent_train(paths)
    q.write(paths[0]/'result.json',report)
    path=paths[0]/f'seed_{course.TRAIN_CASES[0][0]}'/'resource_context_observations.jsonl'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='bytes changed'):data.load_independent_train(paths)
