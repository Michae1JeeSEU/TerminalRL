import json
import pytest

from terminal_marl.training import p3_v25_forecast_data as data
from terminal_marl.training import p3_v25_imitation_qualification as q


def fake_train(tmp_path):
    directories = []
    for index, (seed, requests, warm, high) in enumerate(data.CASES, 1):
        directory = tmp_path/f'train{index}'; episode = directory/f'seed_{seed}'
        episode.mkdir(parents=True)
        row = dict(schema=q.CONTEXT_SCHEMA, observation_schema=q.VERSION, partition='train', seed=seed,
            tick=0, uid='4', decision_id=f'{q.DATA_SCHEMA}:{seed}:0:4', actor_mask=True,
            ego_feature_hash=q.EGO_SCHEMA.feature_hash, traffic_feature_hash=q.TRAFFIC_SCHEMA.feature_hash,
            ego=[0.]*105, traffic=[], traffic_mask=[], teacher_gate=0, teacher_proposal=None,
            original_observation=[0.]*90, context=dict(seed=seed,interface='m2i'))
        path = episode/'context_observations.jsonl'
        path.write_text(json.dumps(row)+'\n',encoding='utf8')
        q.write(directory/'protocol.json',dict(protocol=data.PROTOCOL,partition='train',seed=seed,
            merge_gap_prediction_mode=data.GAP_MODE,smoke=False,requests=requests,warm_start=warm,high_soc=high))
        q.write(directory/'result.json',dict(protocol=data.PROTOCOL,partition='train',seed=seed,
            status='COLLECTION_OK',error=None,blockers=[],reconstruction=dict(rows=1,sha256=q.sha(path))))
        directories.append(directory)
    return directories


def test_loader_rejects_q_old_planner_and_incomplete_cycles(tmp_path):
    directories = fake_train(tmp_path)
    value = data.load_train(directories)
    assert value.original.gate.tolist() == [0,0,0]
    assert value.original.proposal.tolist() == [-1,-1,-1]
    assert value.new[0].shape == (3,105) and len(value.original.ids) == 3
    path = directories[0]/'protocol.json'; original=q.read(path)
    for change in (dict(partition='Q'),dict(merge_gap_prediction_mode='current_holders_legacy'),dict(smoke=True)):
        q.write(path,dict(original,**change))
        with pytest.raises(ValueError): data.load_train(directories)
    q.write(path,original)
    report_path=directories[0]/'result.json'; report=q.read(report_path)
    q.write(report_path,dict(report,status='INCOMPLETE'))
    with pytest.raises(ValueError): data.load_train(directories)


def test_loader_preserves_partition_and_checks_bytes_even_with_matching_report(tmp_path):
    directories=fake_train(tmp_path)
    episode=directories[0]/f'seed_{data.CASES[0][0]}'
    path=episode/'context_observations.jsonl'; row=json.loads(path.read_text())
    row['partition']='Q';path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='contexts changed'): data.load_train(directories)
    report_path=directories[0]/'result.json'; report=q.read(report_path)
    report['reconstruction']['sha256']=q.sha(path);q.write(report_path,report)
    with pytest.raises(ValueError,match='train data'): data.load_train(directories)


def test_forecast_protocol_saved_before_first_reset(tmp_path,monkeypatch):
    def stop(config,*,seed):
        protocol=q.read(tmp_path/'case/protocol.json')
        assert config.merge_gap_prediction_mode==data.GAP_MODE==protocol['merge_gap_prediction_mode']
        assert seed==data.SMOKE_SEED==protocol['seed']
        assert protocol['partition']==data.DEVELOPMENT_PARTITION
        raise RuntimeError('stop before reset')
    monkeypatch.setattr(data,'TerminalParallelEnv',stop)
    result=data.collect(tmp_path/'case',1,smoke=True)
    assert result['status']=='ERROR' and 'before reset' in result['error']
    with pytest.raises(FileExistsError): data.collect(tmp_path/'case',1,smoke=True)
    with pytest.raises(ValueError): data.collect(tmp_path/'wrong',4)


def test_train_reconstruction_uses_new_snapshot_planner_and_preserves_raw_labels(tmp_path):
    seed=892904
    case=tmp_path/'case';case.mkdir()
    q.write(case/'protocol.json',dict(seed=seed,partition='train',schema=data.DATA_SCHEMA))
    env=data.TerminalParallelEnv(data.build_config(requests=3,warm_start=3,high_soc=True,
        forecast_merge_gap=True),seed=seed)
    episode=case/f'seed_{seed}'
    flow,selected=data.sparse.rollout(env,episode,seed=seed,max_steps=2,wall_seconds=120,
        smoke=True,all_actor=True,schema=data.DATA_SCHEMA,partition='train')
    assert flow['status']=='SMOKE_OK'
    annotation=data.sparse.annotate(episode,selected,wall_seconds=120)
    assert annotation['status']=='COMPLETED'
    original=(episode/'rows.jsonl').read_bytes()
    with pytest.raises(ValueError,match='partition differs'):
        q.rebuild_q(episode,seed=seed,partition='train')
    result=q.rebuild_context(episode,seed=seed,partition='train',expected_gap_mode=data.GAP_MODE)
    assert result['rows']==len(selected)==3
    assert (episode/'rows.jsonl').read_bytes()==original
    labels={entry['decision_id']:q.read(episode/entry['label_file']) for entry in annotation['entries']}
    for line in (episode/'context_observations.jsonl').read_text().splitlines():
        row=json.loads(line);obs,gate,proposal=data.decode_row(row)
        assert gate==labels[row['decision_id']]['teacher_gate']
        assert (None if proposal<0 else proposal)==labels[row['decision_id']]['teacher_proposal']
        assert row['partition']=='train' and obs.ego.shape==(105,)
