import json

import pytest
import torch

from terminal_marl.training import p3_v25_task_imitation as learning


def test_synthetic_smoke_never_opens_real_train_or_q(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    def forbidden(*args,**kwargs): raise AssertionError('real data accessed by smoke')
    monkeypatch.setattr(learning,'load_train',forbidden)
    monkeypatch.setattr(learning,'load_q_for_evaluation',forbidden)
    report=learning.fit((),tmp_path/'fit',smoke=True)
    assert report['status']=='FIT_FROZEN' and not report['Q_row_access']
    assert report['fits'][0]['updates']==2
    assert learning.frozen_models(tmp_path/'fit',smoke=True)==json.loads(json.dumps(report))
    with pytest.raises(ValueError): learning.frozen_models(tmp_path/'fit')
    path=tmp_path/'fit'/report['checkpoints'][0]['path'];path.write_bytes(path.read_bytes()+b'change')
    with pytest.raises(ValueError,match='checkpoint changed'): learning.frozen_models(tmp_path/'fit',smoke=True)


def test_non_smoke_fit_calls_only_strict_task_training_loader(tmp_path,monkeypatch):
    called=[]
    def stop(directories):
        called.append(directories);raise RuntimeError('stop before optimizer')
    monkeypatch.setattr(learning,'load_train',stop)
    monkeypatch.setattr(learning,'load_q_for_evaluation',lambda *a:pytest.fail('fit opened Q'))
    with pytest.raises(RuntimeError,match='before optimizer'):
        learning.fit(('a','b','c'),tmp_path/'fit')
    assert called==[('a','b','c')]


def test_q_scoring_is_once_and_never_updates_candidates(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    directory=tmp_path/'fit';directory.mkdir();checkpoints=[]
    for seed in learning.SEEDS:
        path=directory/f'seed_{seed}'/'diagnostic_final.pt';path.parent.mkdir()
        torch.save(dict(model=learning.TrajectorySetActor().state_dict(),protocol=learning.PROTOCOL,seed=seed),path)
        checkpoints.append(dict(seed=seed,path=path.relative_to(directory).as_posix(),sha256=learning.sha(path)))
    learning.write(directory/'result.json',dict(status='FIT_FROZEN',protocol=learning.PROTOCOL,
        profile=learning.PROFILE,smoke=False,checkpoints=checkpoints))
    qs=[tmp_path/f'q{i}' for i in range(3)]
    for path in qs:path.mkdir()
    calls=[]
    monkeypatch.setattr(learning,'validated_case',lambda *args:calls.append(args))
    monkeypatch.setattr(learning,'load_q_for_evaluation',lambda _:learning.synthetic())
    monkeypatch.setattr(learning,'load_train',lambda _:pytest.fail('evaluation opened train'))
    report=learning.evaluate(directory,qs,tmp_path/'evaluation')
    assert len(calls)==3 and report['status']=='EVALUATED'
    assert report['parameter_updates']==0 and not report['STUDENT_QUALIFIED'] and not report['PPO_ALLOWED']
    assert len(report['seeds'])==3
    for item in checkpoints:assert learning.sha(directory/item['path'])==item['sha256']
    assert all((path/'evaluation_consumed.json').exists() for path in qs)
    with pytest.raises(ValueError,match='already consumed'):
        learning.evaluate(directory,qs,tmp_path/'evaluation_again')
