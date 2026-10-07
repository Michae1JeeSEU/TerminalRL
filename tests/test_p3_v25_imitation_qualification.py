import copy
import json

import pytest
import torch

from terminal_marl.training import p3_v25_imitation_qualification as q
from terminal_marl.training.p3_v25_context_dataset import decode_row


def row(gate=1):
    return dict(schema=q.CONTEXT_SCHEMA, observation_schema=q.VERSION, partition='Q',
        seed=895301, tick=0, uid='4', decision_id=f'{q.DATA_SCHEMA}:895301:0:4',
        ego_feature_hash=q.EGO_SCHEMA.feature_hash, traffic_feature_hash=q.TRAFFIC_SCHEMA.feature_hash,
        actor_mask=True, ego=[0.]*105, traffic=[[0.]*50], traffic_mask=[True],
        teacher_gate=gate, teacher_proposal=12 if gate else None)


def test_q_decoder_stays_separate_from_training_and_preserves_wait():
    r = row(0)
    value, gate, proposal = q.decode_evaluation_row(r, seed=895301)
    assert (gate, proposal) == (0,-1) and value.traffic_uids == ()
    with pytest.raises(ValueError):
        decode_row(r)
    with pytest.raises(ValueError):
        q.decode_evaluation_row(dict(r, partition='train'), seed=895301, partition='train')
    with pytest.raises(ValueError):
        q.decode_evaluation_row(r, seed=895302)


@pytest.mark.parametrize('key,value', [('traffic_mask',[1]), ('actor_mask',False),
    ('teacher_proposal',None), ('ego',[0.]*90), ('observation_schema','old'),
    ('decision_id','wrong'), ('teacher_gate',True)])
def test_q_bad_contract_fails(key,value):
    r = row(); r[key] = value
    with pytest.raises(ValueError):
        q.decode_evaluation_row(r, seed=895301)


def test_fixed_fit_smoke_has_fresh_checkpoints_and_no_q_access(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    def forbidden(*args, **kwargs):
        raise AssertionError('synthetic fit must not open Q or real training data')
    monkeypatch.setattr(q, 'cohort_manifest', forbidden)
    monkeypatch.setattr(q, 'load_paired', forbidden)
    result = q.fit(tmp_path/'unused', tmp_path/'fit', smoke=True)
    assert result['status'] == 'FIT_FROZEN' and result['fits'][0]['updates'] == 2
    assert result['Q_row_access'] is False and not result['PPO_ALLOWED']
    assert q.frozen_models(tmp_path/'fit', smoke=True) == json.loads(json.dumps(result))
    with pytest.raises(ValueError):
        q.frozen_models(tmp_path/'fit')
    path = tmp_path/'fit'/result['checkpoints'][0]['path']
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        q.frozen_models(tmp_path/'fit', smoke=True)


def test_collection_protocol_precedes_reset_and_does_not_load_model(tmp_path, monkeypatch):
    def forbidden(config, *, seed):
        protocol = json.loads((tmp_path/'case/protocol.json').read_text())
        assert seed == protocol['seed'] == q.SMOKE_SEED
        assert protocol['checkpoint_loaded'] is False
        assert protocol['partition'] == q.DEVELOPMENT_PARTITION
        raise RuntimeError('injected, no reset')
    monkeypatch.setattr(q, 'TerminalParallelEnv', forbidden)
    report = q.collect(tmp_path/'case', 1, smoke=True)
    assert report['status'] == 'ERROR' and 'injected' in report['error']
    with pytest.raises(FileExistsError):
        q.collect(tmp_path/'case', 1, smoke=True)


def test_reconstruction_round_trip_without_relabelling_q(tmp_path):
    # Development-only fixture; production qualification seeds are never reset.
    seed = 892901
    case = tmp_path/'case'; case.mkdir()
    q.write(case/'protocol.json', dict(seed=seed, partition=q.DEVELOPMENT_PARTITION,
                                      schema=q.DEVELOPMENT_SCHEMA))
    env = q.TerminalParallelEnv(q.build_config(requests=3,warm_start=3,high_soc=True),seed=seed)
    episode = case/f'seed_{seed}'
    flow, selected = q.sparse.rollout(env,episode,seed=seed,max_steps=2,wall_seconds=120,
        smoke=True,all_actor=True,schema=q.DEVELOPMENT_SCHEMA,partition=q.DEVELOPMENT_PARTITION)
    assert flow['status'] == 'SMOKE_OK'
    annotation = q.sparse.annotate(episode,selected,wall_seconds=120)
    assert annotation['status'] == 'COMPLETED'
    before = (episode/'rows.jsonl').read_bytes()
    result = q.rebuild_q(episode,seed=seed,partition=q.DEVELOPMENT_PARTITION)
    assert result['rows'] == 3
    rows = [json.loads(line) for line in (episode/'context_observations.jsonl').read_text().splitlines()]
    assert all(r['partition'] == q.DEVELOPMENT_PARTITION for r in rows)
    assert [r['decision_id'] for r in rows] == [r['row']['decision_id'] for r in selected]
    assert (episode/'rows.jsonl').read_bytes() == before
    for r in rows:
        with pytest.raises(ValueError):
            q.decode_evaluation_row(r,seed=seed)
    assert all(r['context']['interface'] in {'o2m','m2i','i2f'} for r in rows)


def test_imitation_thresholds_distinguish_gate_and_conditional_proposal():
    predictions = [dict(gate=0,proposal=0)]*50 + [dict(gate=1,proposal=10+i%4) for i in range(50)]
    metrics = dict(groups=dict(overall=dict(gate=dict(confusion=[[50,0],[0,50]],WAIT_recall=1.,
        RELEASE_recall=1.,prediction_WAIT=50,prediction_RELEASE=50),proposal=dict(top1=1.,top3=1.),joint_accuracy=1.)))
    assert q.gates(metrics,predictions,[0,11,12,13,14])['passed']
    modified = copy.deepcopy(metrics)
    modified['groups']['overall']['proposal']['top3'] = .94
    result = q.gates(modified,predictions,[0,11,12,13,14])
    assert not result['passed'] and not result['checks']['proposal_top3']
    assert not q.gates(metrics,predictions,[0,11,12,13])['checks']['stable_classes_present']


def fake_cohort(tmp_path):
    directories = []
    for i, (seed, *_rest) in enumerate(q.Q_CASES):
        directory = tmp_path/f'q{i}'; directory.mkdir()
        episode = directory/f'seed_{seed}'; episode.mkdir()
        path = episode/'context_observations.jsonl'; path.write_text('not parsed for metadata\n')
        q.write(directory/'protocol.json',dict(protocol=q.PROTOCOL,partition='Q',seed=seed))
        q.write(directory/'result.json',dict(status='COLLECTION_OK',error=None,blockers=[],seed=seed,
            reconstruction=dict(rows=1,sha256=q.sha(path))))
        directories.append(directory)
    return directories


def test_failed_training_attempt_cannot_consume_same_q_again(tmp_path, monkeypatch):
    directories = fake_cohort(tmp_path)
    def failed(*args):
        raise RuntimeError('injected before any updates')
    monkeypatch.setattr(q,'load_paired',failed)
    with pytest.raises(RuntimeError):
        q.fit(tmp_path,tmp_path/'fit',q_directories=directories)
    assert all((directory/'training_consumed.json').exists() for directory in directories)
    with pytest.raises(ValueError,match='already authorized'):
        q.fit(tmp_path,tmp_path/'fit_again',q_directories=directories)


def test_q_byte_change_is_detected_without_parsing_rows(tmp_path):
    directories = fake_cohort(tmp_path)
    assert len(q.cohort_manifest(directories)) == 3
    path = directories[0]/f'seed_{q.Q_CASES[0][0]}'/'context_observations.jsonl'
    path.write_text('changed\n')
    with pytest.raises(ValueError,match='context file changed'):
        q.cohort_manifest(directories)


def test_complete_evaluation_emits_all_seed_metrics_and_consumes_q_once(tmp_path):
    # Synthetic arrays/checkpoints only: no production seed reset or training.
    torch.set_num_threads(1)
    directories = fake_cohort(tmp_path)
    for directory, (seed, *_rest) in zip(directories,q.Q_CASES):
        records = []
        for i in range(10):
            r = row(i % 2)
            r.update(seed=seed,uid=str(i),decision_id=f'{q.DATA_SCHEMA}:{seed}:0:{i}',
                original_observation=[0.]*90,
                context=dict(seed=seed,interface=('o2m','m2i','i2f')[i%3],
                             low_soc=False,fast_inner=False,busy_lz=False,after_retry=False))
            if r['teacher_gate']:
                r['teacher_proposal'] = 10+(i//2)%4
            records.append(r)
        path = directory/f'seed_{seed}'/'context_observations.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        q.write(directory/'result.json',dict(status='COLLECTION_OK',error=None,blockers=[],seed=seed,
            reconstruction=dict(rows=10,sha256=q.sha(path))))
    fit_directory = tmp_path/'fit'; fit_directory.mkdir()
    checkpoints = []
    torch.manual_seed(123)
    for seed in q.SEEDS:
        case = fit_directory/f'seed_{seed}'; case.mkdir()
        path = case/'diagnostic_final.pt'
        torch.save(dict(model=q.TrajectorySetActor().state_dict()),path)
        checkpoints.append(dict(seed=seed,path=path.relative_to(fit_directory).as_posix(),sha256=q.sha(path)))
    q.write(fit_directory/'result.json',dict(status='FIT_FROZEN',protocol=q.PROTOCOL,smoke=False,
        checkpoints=checkpoints,frozen_Q_metadata=q.cohort_manifest(directories)))
    result = q.evaluate(fit_directory,directories,tmp_path/'evaluation')
    assert result['status'] == 'EVALUATED' and result['parameter_updates'] == 0
    assert result['STUDENT_QUALIFIED'] is result['PPO_ALLOWED'] is False
    assert len(result['seeds']) == 3 and result['across_seeds']['joint_accuracy'] is not None
    for seed in q.SEEDS:
        assert len(q.read(tmp_path/'evaluation'/f'predictions_{seed}.json')) == 30
    assert all((directory/'evaluation_consumed.json').exists() for directory in directories)
    with pytest.raises(ValueError,match='already consumed'):
        q.evaluate(fit_directory,directories,tmp_path/'evaluation_again')
