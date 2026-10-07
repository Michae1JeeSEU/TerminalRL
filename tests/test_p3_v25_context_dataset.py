import copy
import json

import numpy as np
import pytest

from terminal_marl.core.trajectory_context_observation import VERSION, EGO_SCHEMA, TRAFFIC_SCHEMA
from terminal_marl.training import p3_v25_context_dataset as dataset


def row(gate=1):
    return dict(schema=dataset.DATA_SCHEMA, observation_schema=VERSION, partition='train',
        ego_feature_hash=EGO_SCHEMA.feature_hash, traffic_feature_hash=TRAFFIC_SCHEMA.feature_hash,
        actor_mask=True, ego=[0.]*105, traffic=[[0.]*50], traffic_mask=[True],
        teacher_gate=gate, teacher_proposal=12 if gate else None,
        seed=893001, tick=0, uid='4', decision_id='p3_v3_90d_sparse_expert_v1:893001:0:4',
        trace=dict(traffic_uids=[98765], target_uids=[98765,98765]))


def test_numeric_loader_keeps_labels_and_excludes_trace(tmp_path):
    r = row()
    (tmp_path/'result.json').write_text(json.dumps(dict(status='RECONSTRUCTION_COMPLETE',mode='full',rows=1)))
    (tmp_path/'observations.jsonl').write_text(json.dumps(r)+'\n')
    batch = dataset.load_context_dataset(tmp_path, allowed_train_seeds={893001})
    assert batch.gate_target.tolist() == [1] and batch.proposal_target.tolist() == [12]
    assert batch.observations[0].traffic_uids == ()
    assert batch.padded()[1].shape == (1, 1, 50)
    assert not batch.padded()[1].any()


@pytest.mark.parametrize('field,value', [('schema','old'), ('partition','audit'), ('actor_mask',False),
    ('ego',[0.]*90), ('traffic_mask',[1]), ('ego_feature_hash','wrong'), ('teacher_proposal',None)])
def test_loader_rejects_wrong_contract(field, value):
    r = row(); r[field] = value
    with pytest.raises(ValueError): dataset.decode_row(r)


def test_wait_is_not_a_proposal_and_empty_set_is_valid():
    r = row(0); r.update(traffic=[], traffic_mask=[])
    obs, gate, proposal = dataset.decode_row(r)
    assert (gate,proposal) == (0,-1) and obs.traffic.shape == (0,50)


@pytest.mark.parametrize('mode,status', [('probe','RECONSTRUCTION_COMPLETE'),('full','ERROR')])
def test_incomplete_or_probe_not_training_input(tmp_path, mode, status):
    (tmp_path/'result.json').write_text(json.dumps(dict(mode=mode,status=status)))
    with pytest.raises(ValueError): dataset.load_context_dataset(tmp_path,allowed_train_seeds={893001})
