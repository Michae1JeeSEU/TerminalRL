import json
from dataclasses import replace

import pytest

from terminal_marl.training import p3_v25_course_data as data
from terminal_marl.training import p3_v25_imitation_qualification as q


def fake_cases(tmp_path,partition):
    directories=[]
    for seed,scenario in (data.TRAIN_CASES if partition=='train' else data.Q_CASES):
        directory=tmp_path/f'{partition}_{scenario}';episode=directory/f'seed_{seed}'
        episode.mkdir(parents=True)
        row=dict(schema=q.CONTEXT_SCHEMA,observation_schema=q.VERSION,partition=partition,seed=seed,
            tick=0,uid='4',decision_id=f'{data.DATA_SCHEMA}:{seed}:0:4',actor_mask=True,
            ego_feature_hash=q.EGO_SCHEMA.feature_hash,traffic_feature_hash=q.TRAFFIC_SCHEMA.feature_hash,
            ego=[0.]*105,traffic=[],traffic_mask=[],teacher_gate=0,teacher_proposal=None,
            original_observation=[0.]*90,context=dict(seed=seed,interface='m2i'))
        path=episode/'context_observations.jsonl';path.write_text(json.dumps(row)+'\n')
        identity=dict(protocol=data.PROTOCOL,profile=data.PROFILE,partition=partition,seed=seed,scenario=scenario)
        q.write(directory/'protocol.json',dict(identity,smoke=False,task=data.task_settings(data.build_config(scenario))))
        q.write(directory/'result.json',dict(identity,status='COLLECTION_OK',error=None,blockers=[],
            reconstruction=dict(rows=1,sha256=q.sha(path))))
        directories.append(directory)
    return directories


def test_neutral_course_loader_requires_full_declared_train_and_never_q(tmp_path):
    train=fake_cases(tmp_path,'train');qualification=fake_cases(tmp_path,'Q')
    value=data.load_train(train)
    assert value.original.gate.tolist()==[0,0,0] and value.original.proposal.tolist()==[-1,-1,-1]
    assert value.new[0].shape==(3,105)
    assert all(i['partition']=='train' for i in value.original.inputs)
    with pytest.raises(ValueError): data.load_train(qualification)
    with pytest.raises(ValueError): data.load_train(train[:2])
    assert all(i['partition']=='Q' for i in data.load_q_for_evaluation(qualification).original.inputs)
    path=train[0]/'protocol.json';original=q.read(path)
    for key,value in (('task_priority_enabled',True),('release_arbitration_mode','downstream_priority'),
                      ('fixed_arrival_schedule',[0,0,0,0,0,0])):
        changed=json.loads(json.dumps(original));changed['task'][key]=value;q.write(path,changed)
        with pytest.raises(ValueError,match='wrong task'): data.load_train(train)
    q.write(path,original)
    report_path=train[0]/'result.json';report=q.read(report_path)
    q.write(report_path,dict(report,status='INCOMPLETE'))
    with pytest.raises(ValueError,match='incomplete'): data.load_train(train)


def test_task_protocol_saved_before_reset(tmp_path,monkeypatch):
    def stop(config,*,seed):
        protocol=q.read(tmp_path/'case/protocol.json')
        assert protocol['seed']==seed==data.SMOKE_SEED
        assert protocol['task']==data.task_settings(config)
        assert not config.task_priority_enabled and config.initial_soc_min==.55
        assert protocol['partition']==data.DEVELOPMENT_PARTITION
        raise RuntimeError('before reset')
    monkeypatch.setattr(data,'TerminalParallelEnv',stop)
    result=data.collect(tmp_path/'case',1,smoke=True)
    assert result['status']=='ERROR' and 'before reset' in result['error']
    with pytest.raises(FileExistsError): data.collect(tmp_path/'case',1,smoke=True)


def test_revised_annotation_budget_is_written_before_reset(tmp_path,monkeypatch):
    def stop(config, *, seed):
        protocol=q.read(tmp_path/'case/protocol.json')
        assert protocol['seed']==seed==893503
        assert protocol['annotation_wall_seconds']==21600
        assert protocol['rollout_wall_seconds']==3600 and protocol['max_steps']==1800
        raise RuntimeError('before reset')
    monkeypatch.setattr(data,'TerminalParallelEnv',stop)
    result=data.collect(tmp_path/'case',3,annotation_wall_seconds=21600)
    assert result['status']=='ERROR' and 'before reset' in result['error']
    with pytest.raises(ValueError,match='undeclared annotation budget'):
        data.collect(tmp_path/'undeclared',3,annotation_wall_seconds=999999)


def test_flow_empty_ticks_and_arrival_are_in_tape(tmp_path):
    seed=892951
    env=data.TerminalParallelEnv(data.build_config('flow6_mixed'),seed=seed)
    episode=tmp_path/'episode'
    flow,_=data.sparse.rollout(env,episode,seed=seed,max_steps=12,wall_seconds=120,
        smoke=True,all_actor=True,schema=data.DEVELOPMENT_SCHEMA,partition=data.DEVELOPMENT_PARTITION)
    assert flow['status']=='SMOKE_OK' and flow['ticks']==12
    assert flow['generated_requests']==1 and flow['first_tick_replay_equal']
    steps=[json.loads(s) for s in (episode/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==12 and not steps[0]['tape']['actions']


def test_neutral_reconstruction_keeps_service_context_and_masks_task_preferences(tmp_path):
    seed=892952
    config=replace(data.build_config('dense24_mixed'),total_requested_uavs=3,warm_start_uavs=3,
                   fixed_arrival_schedule=(0.,)*3)
    case=tmp_path/'case';case.mkdir()
    q.write(case/'protocol.json',dict(seed=seed,partition='train',schema=data.DATA_SCHEMA))
    env=data.TerminalParallelEnv(config,seed=seed);episode=case/f'seed_{seed}'
    flow,selected=data.sparse.rollout(env,episode,seed=seed,max_steps=2,wall_seconds=120,
        smoke=True,all_actor=True,schema=data.DATA_SCHEMA,partition='train')
    assert flow['status']=='SMOKE_OK'
    annotation=data.sparse.annotate(episode,selected,wall_seconds=180)
    assert annotation['status']=='COMPLETED'
    original=(episode/'rows.jsonl').read_bytes()
    report=data.rebuild_context(episode,seed=seed,partition='train',expected_gap_mode=data.GAP_MODE)
    assert report['rows']==len(selected)==3
    assert (episode/'rows.jsonl').read_bytes()==original
    for line in (episode/'context_observations.jsonl').read_text().splitlines():
        row=json.loads(line);obs,gate,proposal=data.decode_row(row)
        assert obs.ego[16]==row['original_observation'][16]
        assert obs.ego[18:22].tolist()==row['original_observation'][18:22]
        assert obs.ego[18:22].tolist()==[1.,0.,0.,0.]
        assert not env.config.task_priority_enabled
        assert all(u.base_priority==1 for u in env.active.values())
        assert gate in (0,1) and (proposal<0)==(gate==0)
