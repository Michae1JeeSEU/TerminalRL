import json

import numpy as np
import pytest

from terminal_marl.training import p3_v26_course_data as data
from terminal_marl.training import p3_v25_course_data as old


def test_fresh_protocol_exists_before_reset_and_no_q_entry(tmp_path,monkeypatch):
    def stop(config,*,seed):
        protocol=data.read(tmp_path/'case/protocol.json')
        assert seed==893602 and protocol['partition']=='train'
        assert protocol['learning_observation_schema']==data.VERSION
        assert protocol['annotation_wall_seconds']==21600 and protocol['max_selected']==2048
        assert not config.task_priority_enabled and config.initial_soc==.8
        raise RuntimeError('before reset')
    monkeypatch.setattr(old,'TerminalParallelEnv',stop)
    result=data.collect(tmp_path/'case',2)
    assert result['status']=='ERROR' and 'before reset' in result['error']
    with pytest.raises(ValueError):data.collect(tmp_path/'unknown',4)
    with pytest.raises(FileExistsError):data.collect(tmp_path/'case',2)


def test_resource_decode_rejects_q_and_invalid_width():
    from terminal_marl.training.p3_v25_imitation_qualification import CONTEXT_SCHEMA,TRAFFIC_SCHEMA
    row=dict(schema=CONTEXT_SCHEMA,observation_schema=data.VERSION,partition='train',
        ego_feature_hash=data.EGO_SCHEMA.feature_hash,traffic_feature_hash=TRAFFIC_SCHEMA.feature_hash,
        actor_mask=True,ego=[0.]*159,traffic=[],traffic_mask=[],teacher_gate=0,teacher_proposal=None)
    obs,gate,proposal=data.decode_resource_row(row)
    assert obs.ego.shape==(159,) and gate==0 and proposal==-1
    with pytest.raises(ValueError):data.decode_resource_row(dict(row,partition='Q'))
    with pytest.raises(ValueError):data.decode_resource_row(dict(row,ego=[0.]*105))


def test_small_collection_keeps_original_labels_and_resource_output(tmp_path):
    result=data.collect(tmp_path/'smoke',1,smoke=True)
    assert result['status']=='SMOKE_OK' and result['resource_reconstruction']['rows']==3
    episode=tmp_path/'smoke'/f'seed_{data.SMOKE_SEED}'
    old_rows=[json.loads(s) for s in (episode/'context_observations.jsonl').read_text().splitlines()]
    new_rows=[json.loads(s) for s in (episode/'resource_context_observations.jsonl').read_text().splitlines()]
    for old_row,new in zip(old_rows,new_rows):
        assert old_row['decision_id']==new['decision_id']
        assert old_row['ego']==new['ego'][:105] and old_row['traffic']==new['traffic']
        assert old_row['teacher_gate']==new['teacher_gate'] and old_row['teacher_proposal']==new['teacher_proposal']
        assert new['partition']==old.DEVELOPMENT_PARTITION and len(new['ego'])==159
        assert np.isfinite(new['ego']).all()
    with pytest.raises(FileExistsError):data.rebuild_resources(episode,seed=data.SMOKE_SEED,smoke=True)
